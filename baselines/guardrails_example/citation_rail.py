"""Pre-submission citation rail for Track 4 answers.

Two local checks an agent can run on its OWN draft answer before submitting:

1. **Date rail** — every cited ``doc_id`` must resolve to a corpus document whose
   ``doc_date`` is on or before the task ``cutoff_date``. Citing a document that
   is missing from the frozen corpus, or dated after the cutoff, is flagged.
2. **Shape rail** — every claim must carry a well-formed ``(doc_id, span_start,
   span_end)`` triple whose offsets resolve inside the cited document's text.

These are the two mistakes the adversarial variants (stale-filing traps,
Family 5) are designed to elicit. Running the rail locally lets an agent drop or
repair a bad claim before it ever reaches the organizer's scoring pipeline.

The rail is advisory and participant-side only: it reduces YOUR gate failures.
The competition's embargo and faithfulness gates are deterministic organizer
code and are the authority on every submission.

Text/offset convention mirrors the scorer (and ``baseline_agent.indexer``): a
document's text is its flat ``text`` field if present, else its ``spans[].text``
values joined with a single space; span offsets are global character offsets
into that string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_MANIFEST_NAME = "manifest.json"


@dataclass
class CorpusDoc:
    doc_id: str
    text: str
    doc_date: str | None


@dataclass
class RailFinding:
    """One problem the rail found in a draft answer."""

    entity_id: str
    claim_index: int
    code: str  # unknown_doc | stale_doc | missing_field | bad_span | empty_claim
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] entity={self.entity_id} claim#{self.claim_index}: {self.message}"


def _doc_text(doc: dict) -> str:
    """Concatenate spans (or use a flat ``text`` field) — mirrors the scorer."""
    if isinstance(doc.get("text"), str):
        return doc["text"]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return " ".join(sp.get("text", "") for sp in spans if isinstance(sp, dict))
    return ""


def load_corpus(corpus_dir: str | Path) -> dict[str, CorpusDoc]:
    """Load every corpus document keyed by ``doc_id`` (skips the manifest)."""
    corpus_dir = Path(corpus_dir)
    docs: dict[str, CorpusDoc] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc_id = raw.get("doc_id", path.stem)
        docs[doc_id] = CorpusDoc(
            doc_id=doc_id, text=_doc_text(raw), doc_date=raw.get("doc_date")
        )
    return docs


def filter_retrieved(
    docs: list[CorpusDoc], cutoff_date: str
) -> tuple[list[CorpusDoc], list[CorpusDoc]]:
    """Split a retrieval pool into (usable, stale) by ``doc_date <= cutoff_date``.

    Use this rail at retrieval time so post-cutoff material never reaches the
    reasoning step. Documents with no ``doc_date`` are treated as stale — an
    undatable document cannot be shown to be embargo-safe.
    """
    usable: list[CorpusDoc] = []
    stale: list[CorpusDoc] = []
    for doc in docs:
        if doc.doc_date is not None and doc.doc_date <= cutoff_date:
            usable.append(doc)
        else:
            stale.append(doc)
    return usable, stale


def check_answer(
    answer: dict, corpus: dict[str, CorpusDoc], cutoff_date: str
) -> list[RailFinding]:
    """Run both rails over a draft answer; return every finding (empty = clean)."""
    findings: list[RailFinding] = []
    for entity in answer.get("entity_predictions", []):
        entity_id = str(entity.get("entity_id", "?"))
        for i, claim in enumerate(entity.get("claims", [])):
            findings.extend(_check_claim(entity_id, i, claim, corpus, cutoff_date))
    return findings


def _check_claim(
    entity_id: str,
    index: int,
    claim: dict,
    corpus: dict[str, CorpusDoc],
    cutoff_date: str,
) -> list[RailFinding]:
    findings: list[RailFinding] = []

    def flag(code: str, message: str) -> None:
        findings.append(RailFinding(entity_id, index, code, message))

    # Shape rail: required fields present and well-typed.
    missing = [k for k in ("doc_id", "span_start", "span_end", "claim") if k not in claim]
    if missing:
        flag("missing_field", f"claim is missing field(s): {', '.join(missing)}")
        return findings  # nothing further is checkable

    if not str(claim["claim"]).strip():
        flag("empty_claim", "claim text is empty")

    start, end = claim["span_start"], claim["span_end"]
    if not isinstance(start, int) or not isinstance(end, int):
        flag("bad_span", f"span offsets must be integers (got {start!r}, {end!r})")
        return findings
    if start < 0 or end <= start:
        flag("bad_span", f"span [{start}, {end}) is not a valid half-open range")
        return findings

    # Date rail: the cited document must exist in the frozen corpus and pre-date
    # the cutoff. A doc_id the corpus does not contain usually means the agent
    # cited something it fetched live — exactly the stale-evidence mistake.
    doc = corpus.get(claim["doc_id"])
    if doc is None:
        flag(
            "unknown_doc",
            f"cited doc_id {claim['doc_id']!r} is not in the frozen corpus",
        )
        return findings
    if doc.doc_date is None or doc.doc_date > cutoff_date:
        flag(
            "stale_doc",
            f"cited doc {doc.doc_id!r} has doc_date={doc.doc_date!r}, "
            f"after cutoff {cutoff_date!r}",
        )

    # Shape rail, continued: offsets must resolve inside the document text.
    if end > len(doc.text):
        flag(
            "bad_span",
            f"span [{start}, {end}) exceeds document length {len(doc.text)}",
        )

    return findings
