"""Corpus indexer: one retrievable chunk per corpus span, with global offsets.

Corpus documents carry their text as a ``spans`` array (or a flat ``text``
field). The scorer's offset convention concatenates span texts with a single
space, so a span that starts at position ``p`` in that concatenation can be
cited directly as ``(doc_id, p, p + len(span_text))``. Indexing at span
granularity therefore gives every chunk an exact, citation-ready offset pair
for free — no separate span search needed for chunk-level citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_MANIFEST_NAME = "manifest.json"
_CHUNK_CHARS = 2400
_CHUNK_OVERLAP = 240


def calendar_date(value: Any) -> date:
    if not isinstance(value, str):
        raise ValueError("citation date must be an ISO calendar date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("citation date must be an ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError("citation date must use YYYY-MM-DD")
    return parsed


def dated_on_or_before(value: Any, cutoff: str) -> bool:
    cutoff_date = calendar_date(cutoff)
    try:
        return calendar_date(value) <= cutoff_date
    except ValueError:
        return False


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_date: str | None
    span_start: int  # global char offset into the document's joined text
    span_end: int
    text: str


@dataclass(frozen=True)
class IndexedCorpus:
    chunks: list[Chunk]
    doc_texts: dict[str, str]  # doc_id -> full joined text (for span finding)
    doc_dates: dict[str, str | None]


def _iter_span_texts(doc: dict) -> list[str]:
    if isinstance(doc.get("text"), str):
        return [doc["text"]]
    spans = doc.get("spans")
    if isinstance(spans, list):
        return [sp.get("text", "") for sp in spans if isinstance(sp, dict)]
    return []


def build_index(corpus_dir: str | Path) -> IndexedCorpus:
    """Index every corpus document under ``corpus_dir`` (skips the manifest)."""
    chunks: list[Chunk] = []
    doc_texts: dict[str, str] = {}
    doc_dates: dict[str, str | None] = {}
    for path in sorted(Path(corpus_dir).glob("*.json")):
        if path.name == _MANIFEST_NAME:
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc_id = path.stem
        doc_date = doc.get("doc_date")
        offset = 0
        parts: list[str] = []
        for text in _iter_span_texts(doc):
            if text:
                start = 0
                while start < len(text):
                    end = min(start + _CHUNK_CHARS, len(text))
                    if end < len(text):
                        boundary = text.rfind("\n", start + _CHUNK_CHARS // 2, end)
                        if boundary >= 0:
                            end = boundary + 1
                    chunks.append(
                        Chunk(
                            doc_id=doc_id,
                            doc_date=doc_date,
                            span_start=offset + start,
                            span_end=offset + end,
                            text=text[start:end],
                        )
                    )
                    if end == len(text):
                        break
                    start = end - _CHUNK_OVERLAP
            parts.append(text)
            offset += len(text) + 1  # +1 for the joining space
        doc_texts[doc_id] = " ".join(parts)
        doc_dates[doc_id] = doc_date
    return IndexedCorpus(chunks=chunks, doc_texts=doc_texts, doc_dates=doc_dates)
