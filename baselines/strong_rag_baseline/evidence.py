"""Optional entity-bound retrieval with exact, inspectable source annotations.

Annotations are lexical mentions, not a table parser or inferred relationships.
In particular, a quantity never inherits the requested metric, fiscal period, or
unit. The caller can compare this retrieval strategy with the existing baseline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .indexer import Chunk, IndexedCorpus, calendar_date
from .quantities import finite_number
from .reasoner import _alias_hit, _entity_aliases, collect_windows
from .retriever import BM25Index
from .schema import target_name, target_type

_PERIOD = re.compile(
    r"\b(?:Q[1-4]\s+(?:FY\s*)?20\d{2}|(?:FY\s*)?20\d{2}\s+Q[1-4]|"
    r"FY\s*20\d{2}|(?:three|six|nine|twelve) months ended "
    r"(?:20\d{2}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},?\s+20\d{2}))\b",
    re.I,
)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_METRIC = re.compile(
    r"\b(?:diluted EPS|EPS|earnings per (?:diluted )?share|revenue|"
    r"operating income|net income|operating margin|gross margin|"
    r"yield|bid.to.cover(?: ratio)?|open interest|default probability)\b",
    re.I,
)
_NUMBER = re.compile(
    r"(?<![\w.])(?P<currency>USD\s*|EUR\s*|GBP\s*|\$\s*)?"
    r"(?P<value>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![\w]|\.\d)"
    r"(?:\s*(?P<scale>thousand|million|billion)\b)?"
    r"(?:\s*(?P<unit>%|percent\b|bps\b|basis points?\b|"
    r"USD\b|EUR\b|GBP\b|dollars?\b))?",
    re.I,
)


@dataclass(frozen=True)
class EvidencePacket:
    chunks: list[Chunk]
    ledger: dict[str, Any]


def evidence_queries(task: dict, entity: dict) -> list[str]:
    """Search the requested quantity from observation and expectation angles."""
    identity = " ".join(dict.fromkeys(_entity_aliases(entity)))
    metric = target_name(task)
    base = " ".join(part for part in (identity, metric) if part).strip()
    return [
        base,
        f"{base} historical prior previous reported",
        f"{base} consensus estimates expectations",
        f"{base} guidance forecast outlook risks",
    ]


def _mentions(pattern: re.Pattern, chunk: Chunk) -> list[dict[str, Any]]:
    return [
        {
            "text": match.group(),
            "span_start": chunk.span_start + match.start(),
            "span_end": chunk.span_start + match.end(),
        }
        for match in pattern.finditer(chunk.text)
    ]


def _quantities(chunk: Chunk) -> list[dict[str, Any]]:
    excluded = [
        match.span()
        for pattern in (_PERIOD, _DATE)
        for match in pattern.finditer(chunk.text)
    ]
    result = []
    for match in _NUMBER.finditer(chunk.text):
        if any(start <= match.start("value") < end for start, end in excluded):
            continue
        raw = match.group("value")
        value = float(raw.replace(",", ""))
        currency = (match.group("currency") or "").strip() or None
        unit = match.group("unit")
        # Parentheses denote a negative accounting value only when they directly
        # surround this token; other punctuation does not change the sign.
        if chunk.text[: match.start()].endswith("(") and chunk.text[
            match.end() :
        ].startswith(")"):
            value = -abs(value)
        result.append(
            {
                "text": match.group(),
                "span_start": chunk.span_start + match.start(),
                "span_end": chunk.span_start + match.end(),
                "value": value if finite_number(value) else None,
                "unit": unit or currency,
                "currency": currency,
                "scale": match.group("scale"),
                "metric": None,
                "period": None,
            }
        )
    return result


def _header_start(chunk: Chunk, text: str, other_aliases: set[str]) -> int:
    """Keep one adjacent table/period heading, without another entity's row."""
    before = text[: chunk.span_start].rstrip("\r\n")
    start = before.rfind("\n") + 1
    heading = before[start:]
    if (
        not heading
        or len(heading) > 240
        or _alias_hit(heading, other_aliases)
        or not (_PERIOD.search(heading) or _METRIC.search(heading))
        or _quantities(Chunk(chunk.doc_id, chunk.doc_date, start, len(before), heading))
    ):
        return chunk.span_start
    return start


def prepare_evidence(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    *,
    top_k: int = 8,
    max_chars: int = 12_000,
    max_span_chars: int = 1_400,
) -> EvidencePacket:
    """Return bounded excerpts and a JSON-serializable provenance ledger.

    Reciprocal rank fusion gives each query an equal vote. Unknown relationships
    remain null; document dates are never substituted for financial periods.
    """
    if top_k < 1 or max_chars < 1 or max_span_chars < 1:
        raise ValueError("evidence budgets must be positive")
    cutoff = task.get("cutoff_date")
    calendar_date(cutoff)
    aliases = _entity_aliases(entity)
    roster = task.get("entities") or []
    other_aliases = {
        alias
        for row in roster
        if row.get("entity_id") != entity.get("entity_id")
        for alias in _entity_aliases(row)
    } - set(aliases)
    windows = collect_windows(
        entity,
        corpus,
        index,
        cutoff=cutoff,
        top_k=top_k * 4,
        extra_needles=[target_name(task)],
        roster=roster,
    )
    candidates: dict[tuple[str, int, int], Chunk] = {}
    for window in windows:
        if window.entity_ambiguous:
            continue
        text = corpus.doc_texts[window.doc_id]
        chunk = Chunk(
            window.doc_id,
            window.doc_date,
            window.span_start,
            window.span_end,
            window.text,
        )
        start = _header_start(chunk, text, other_aliases)
        end = min(window.span_end, start + max_span_chars)
        if end < window.span_end:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        snippet = text[start:end]
        if not snippet.strip():
            continue
        # Cropping cannot turn a text-bound excerpt into an unbound excerpt.
        if _alias_hit(window.text, aliases) and not _alias_hit(snippet, aliases):
            continue
        candidates[(window.doc_id, start, end)] = Chunk(
            window.doc_id, window.doc_date, start, end, snippet
        )
    queries = evidence_queries(task, entity)
    scoped_index = BM25Index(list(candidates.values()), cutoff)
    scores: dict[tuple[str, int, int], float] = {}
    query_hits: dict[tuple[str, int, int], list[int]] = {}
    for query_id, query in enumerate(queries):
        for rank, hit in enumerate(scoped_index.search(query, top_k * 4), 1):
            key = (hit.chunk.doc_id, hit.chunk.span_start, hit.chunk.span_end)
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
            query_hits.setdefault(key, []).append(query_id)
    selected: list[Chunk] = []
    records = []
    used_chars = 0
    cik = str(entity.get("cik") or "").zfill(10) if entity.get("cik") else ""
    for key in sorted(scores, key=lambda key: (-scores[key], *key)):
        chunk = candidates[key]
        if used_chars + len(chunk.text) > max_chars:
            continue
        if any(
            prior.doc_id == chunk.doc_id
            and max(
                0,
                min(prior.span_end, chunk.span_end)
                - max(prior.span_start, chunk.span_start),
            )
            >= 0.8 * min(len(prior.text), len(chunk.text))
            for prior in selected
        ):
            continue
        selected.append(chunk)
        used_chars += len(chunk.text)
        matched = [alias for alias in aliases if _alias_hit(chunk.text, [alias])]
        provenance = (
            "text_alias"
            if matched
            else "document_cik"
            if cik and cik in chunk.doc_id
            else "document_alias"
        )
        records.append(
            {
                "doc_id": chunk.doc_id,
                "doc_date": chunk.doc_date,
                "span_start": chunk.span_start,
                "span_end": chunk.span_end,
                "text": chunk.text,
                "binding": {"source": provenance, "aliases": matched},
                "query_ids": query_hits[key],
                "metrics": _mentions(_METRIC, chunk),
                "periods": _mentions(_PERIOD, chunk),
                "quantities": _quantities(chunk),
            }
        )
        if len(selected) == top_k:
            break
    target = task.get("target") or {}
    ledger = {
        "strategy": "evidence",
        "entity": dict(entity),
        "target": {**target, "type": target_type(task)},
        "cutoff": cutoff,
        "resolution": task.get("resolution_date")
        or entity.get("resolving_release_date")
        or entity.get("expected_report_date"),
        "unit": target.get("unit") or target.get("units") or entity.get("unit"),
        "queries": queries,
        "annotation_semantics": "unlinked lexical mentions; null means unknown",
        "characters": used_chars,
        "records": records,
    }
    return EvidencePacket(chunks=selected, ledger=ledger)
