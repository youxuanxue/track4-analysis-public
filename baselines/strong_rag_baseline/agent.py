"""Per-entity orchestration: retrieve → reason → verify spans → assemble claims.

Model predictions and citations are accepted together. If any part is invalid,
the entire entity falls back to the grounded reasoner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .client import ModelClient
from .indexer import Chunk, IndexedCorpus
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .reasoner import ground_entity, prediction_from_grounded
from .retriever import BM25Index
from .schema import interval_level, legal_labels, target_name, target_type
from .validation import validate_prediction

#: Query terms appended to every entity query — steer retrieval toward result
#: and outlook language. Target-name / label tokens from the task are appended
#: per entity so an unpublished family still retrieves its own vocabulary.
_QUERY_SUFFIX = "results revenue earnings guidance outlook growth"


@dataclass
class EntityResult:
    prediction: dict  # entity_predictions[] element
    dropped_claims: int
    model_raw: str
    rationale: str = ""


def _parse_model_json(raw: str) -> dict:
    """Accept one JSON object, with optional complete Markdown fences."""
    if not isinstance(raw, str):
        raise ValueError("model reply is not text")
    if len(raw) > 1_048_576:
        raise ValueError("model reply exceeds 1 MiB")
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model reply is not an object")
    return parsed


def _entity_query(entity: dict, task: dict | None = None) -> str:
    parts = [
        str(entity.get(key, ""))
        for key in ("name", "entity_id", "sector", "series_id", "description")
    ]
    if task:
        tname = target_name(task)
        if tname and tname != "outcome":
            parts.append(tname)
        parts.extend(legal_labels(task))
    return " ".join(p for p in parts if p) + " " + _QUERY_SUFFIX


def _ground_claims(
    evidence: list[dict],
    corpus: IndexedCorpus,
    retrieved: list[Chunk],
) -> tuple[list[dict], int]:
    """Resolve exact quotes inside retrieved chunks, preserving their offsets."""
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
        return [], 1
    claims: list[dict] = []
    dropped = 0
    retrieved_by_doc: dict[str, list[Chunk]] = {}
    for chunk in retrieved:
        retrieved_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    for item in evidence:
        if not isinstance(item, dict):
            dropped += 1
            continue
        doc_id = item.get("doc_id", "")
        quote = item.get("quote")
        claim_text = item.get("claim")
        if (
            not isinstance(doc_id, str)
            or not isinstance(quote, str)
            or not quote.strip()
            or not isinstance(claim_text, str)
            or not claim_text.strip()
            or len(claim_text) > 4000
        ):
            dropped += 1
            continue
        span = None
        for chunk in retrieved_by_doc.get(doc_id, []):
            offset = chunk.text.find(quote)
            if offset >= 0:
                start = chunk.span_start + offset
                end = start + len(quote)
                if corpus.doc_texts.get(doc_id, "")[start:end] == quote:
                    span = start, end
                    break
        if span is None:
            dropped += 1
            continue
        claims.append(
            {
                "doc_id": doc_id,
                "span_start": span[0],
                "span_end": span[1],
                "claim": claim_text.strip(),
            }
        )
    return claims, dropped


def run_entity_grounded(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    top_k: int,
) -> EntityResult:
    """Build a quantity-aware fallback with separate observations and assumptions."""
    grounded = ground_entity(task, entity, corpus, index, top_k=top_k)
    prediction = prediction_from_grounded(
        entity,
        grounded,
        level=interval_level(task),
        kind=target_type(task),
    )
    return EntityResult(
        prediction=prediction,
        dropped_claims=0,
        model_raw="",
        rationale=grounded.rationale,
    )


def run_entity(
    task: dict,
    entity: dict,
    index: BM25Index,
    corpus: IndexedCorpus,
    client: ModelClient,
    top_k: int,
) -> EntityResult:
    retrieved = [s.chunk for s in index.search(_entity_query(entity, task), top_k)]
    try:
        raw = client.complete(SYSTEM_PROMPT, build_user_prompt(task, entity, retrieved))
        parsed = _parse_model_json(raw)
        kind = target_type(task)
        point = parsed.get("point_forecast")
        band = parsed.get("interval")
        claims, dropped = _ground_claims(parsed.get("evidence"), corpus, retrieved)
        if not claims or dropped:
            raise ValueError(
                "model evidence does not exactly resolve in retrieved text"
            )
        prediction: dict = {
            "entity_id": entity.get("entity_id", ""),
            "interval": band,
            "claims": claims,
        }
        if kind == "classification":
            prediction["label"] = parsed.get("label")
        if point is not None:
            prediction["point_forecast"] = point
        validate_prediction(task, prediction, corpus)
        prediction["interval"] = {key: band[key] for key in ("level", "lo", "hi")}
    except (
        OSError,
        ValueError,
        RuntimeError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        RecursionError,
    ):
        return run_entity_grounded(task, entity, index, corpus, top_k)
    return EntityResult(prediction=prediction, dropped_claims=dropped, model_raw=raw)
