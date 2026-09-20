"""Bounded shared policy context for yield forecasts, with original source spans.

These excerpts describe market-wide policy, not an entity's observed target.
They supplement entity-bound evidence without changing forecasts or adding facts.
"""

from __future__ import annotations

import re

from .indexer import Chunk, IndexedCorpus, dated_on_or_before
from .quantities import TargetSpec
from .reasoner import _alias_hit, _entity_aliases

_POLICY = re.compile(
    r"\b(?:federal funds rate|policy[- ]rate|monetary policy|central bank)\b", re.I
)
_CURVE = re.compile(
    r"\b(?:front[- ]end|long[- ]end|yield curve|curve shape|term premium)\b", re.I
)
_EXPLANATORY = re.compile(
    r"\b(?:expected|expectations|sensitive|anchored|projected|pricing|priced|inversion)\b",
    re.I,
)
_ISSUER = re.compile(
    r"\b(?:company|corporation|inc\.|our |we |revenue|earnings|revolving credit)\b",
    re.I,
)
_POLICY_ROW = re.compile(r"(?im)^[ \t]*(?:Federal funds rate|Policy[- ]rate)[ \t]*\n")
_TABLE_HEADER = re.compile(r"(?im)^[ \t]*(?:Table\s+\d+\.|Variable\b)")
_SECTION_BREAK = re.compile(
    r"(?im)^[ \t]*(?:June projection|Previous projection|Note:|Figure\s+\d|Table\s+\d)"
)


def shared_macro_context(
    task: dict,
    entity: dict,
    corpus: IndexedCorpus,
    *,
    max_chunks: int = 4,
    max_chars: int = 4_000,
    max_span_chars: int = 1_400,
) -> list[Chunk]:
    """Select dated policy/curve excerpts without assuming a forecast direction.

    Vertical policy projection tables require their column header and row together.
    A group which cannot fit the budgets is omitted, never silently truncated.
    """
    if min(max_chunks, max_chars, max_span_chars) < 1:
        raise ValueError("shared-context budgets must be positive")
    spec = TargetSpec.from_task(task, entity)
    if spec.mode != "change_bps" or entity.get("cik"):
        return []
    own_aliases = _entity_aliases(entity)
    other_aliases = {
        alias
        for row in task.get("entities", [])
        if row.get("entity_id") != entity.get("entity_id")
        for alias in _entity_aliases(row)
    } - set(own_aliases)
    groups: list[tuple[int, list[Chunk]]] = []
    for doc_id, text in sorted(corpus.doc_texts.items()):
        date = corpus.doc_dates.get(doc_id)
        if not dated_on_or_before(date, spec.cutoff):
            continue
        # Corporate filings are not market-wide policy documents merely because
        # an interest-rate risk paragraph mentions the central bank.
        if "EDGAR" in doc_id.upper() or _ISSUER.search(text[:500]):
            continue
        for match in _POLICY_ROW.finditer(text):
            prefix = text[: match.start()]
            headers = list(_TABLE_HEADER.finditer(prefix))
            if not headers:
                continue
            start = headers[-1].start()
            # The first variable row ends the column header. Require years and
            # an explicit statistical heading before selecting numeric rows.
            header_end = re.search(
                r"(?im)^[ \t]*(?:Change in real GDP|Real GDP|Unemployment rate|"
                r"PCE inflation|Inflation|Federal funds rate|Policy[- ]rate)[ \t]*\n",
                text[start:],
            )
            if header_end is None:
                continue
            header = (start, start + header_end.start())
            header_text = text[header[0] : header[1]]
            if not (
                re.search(r"\b20\d{2}\b", header_text)
                and re.search(r"\b(?:Median|Projection|Forecast)\b", header_text, re.I)
            ):
                continue
            section_end = _SECTION_BREAK.search(text, match.end())
            end = section_end.start() if section_end else len(text)
            row = (match.start(), end)
            if re.search(r"\d", text[row[0] : row[1]]):
                spans = [header, row]
                if all(0 < b - a <= max_span_chars for a, b in spans):
                    groups.append(
                        (4, [Chunk(doc_id, date, a, b, text[a:b]) for a, b in spans])
                    )
        for paragraph in re.finditer(r"[^\n]+", text):
            # Sentences keep market-wide policy separate from unrelated issuer
            # facts in the same paragraph. Keep original offsets and whitespace.
            boundaries = [paragraph.start()]
            boundaries.extend(
                paragraph.start() + m.end()
                for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", paragraph.group())
            )
            boundaries.append(paragraph.end())
            for start, end in zip(boundaries, boundaries[1:]):
                snippet = text[start:end]
                if not 45 <= len(snippet) <= max_span_chars or _ISSUER.search(snippet):
                    continue
                if _alias_hit(snippet, other_aliases):
                    continue
                policy = bool(_POLICY.search(snippet))
                curve = bool(_CURVE.search(snippet) and _EXPLANATORY.search(snippet))
                if not (policy or curve):
                    continue
                # A title alone does not explain policy or the maturity curve.
                if not re.search(
                    r"\b(?:raise[ds]?|lower(?:ed|s)?|increase[ds]?|reduce[ds]?|"
                    r"remain[s]?|target|expected|expectations|sensitive|anchored|"
                    r"project(?:ed|ion|ions)|price[ds]?|balance|assess)\b",
                    snippet,
                    re.I,
                ):
                    continue
                operative = bool(
                    re.search(
                        r"\b(?:raise[ds]?|lower(?:ed|s)?|increase[ds]?|reduce[ds]?|"
                        r"maintain(?:ed|s)?|kept|hold(?:s)?)\b",
                        snippet,
                        re.I,
                    )
                )
                priority = 3 if policy and operative else 2 if curve else 1
                if re.search(r"\b(?:preferred|voting|voted)\b", snippet, re.I):
                    priority = 1
                groups.append((priority, [Chunk(doc_id, date, start, end, snippet)]))
    selected: list[Chunk] = []
    used = 0
    for _, group in sorted(
        groups,
        key=lambda item: (-item[0], item[1][0].doc_id, item[1][0].span_start),
    ):
        if len(selected) + len(group) > max_chunks:
            continue
        size = sum(len(chunk.text) for chunk in group)
        if used + size > max_chars:
            continue
        if any(
            a.doc_id == b.doc_id
            and max(a.span_start, b.span_start) < min(a.span_end, b.span_end)
            for a in group
            for b in selected
        ):
            continue
        selected.extend(group)
        used += size
    return selected
