"""Submission checks shared by model acceptance and final answer assembly."""

from __future__ import annotations

from .indexer import IndexedCorpus, calendar_date
from .quantities import TargetSpec
from .schema import target_type


def validate_prediction(task: dict, prediction: dict, corpus: IndexedCorpus) -> None:
    """Raise ValueError for an unusable prediction; never silently repair evidence."""
    kind = target_type(task)
    if kind not in {"classification", "regression", "ranking"}:
        raise ValueError("task must declare a supported target type")
    entity = next(
        (
            row
            for row in task.get("entities", [])
            if row.get("entity_id") == prediction.get("entity_id")
        ),
        {},
    )
    TargetSpec.from_task(task, entity).validate_prediction(prediction)
    interval = prediction.get("interval")
    if not isinstance(interval, dict) or interval.get("level") != 0.9:
        raise ValueError("interval level must be 0.9")
    claims = prediction.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("prediction must contain at least one citation")
    cutoff = calendar_date(task.get("cutoff_date"))
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("citation must be an object")
        doc_id = claim.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in corpus.doc_texts:
            raise ValueError("citation does not resolve to a corpus document")
        if calendar_date(corpus.doc_dates.get(doc_id)) > cutoff:
            raise ValueError("citation date is after the task cutoff")
        start, end = claim.get("span_start"), claim.get("span_end")
        text = corpus.doc_texts[doc_id]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
            or not text[start:end].strip()
        ):
            raise ValueError("citation offsets must resolve to nonempty source text")
        if not isinstance(claim.get("claim"), str) or not claim["claim"].strip():
            raise ValueError("citation claim must be nonempty text")


def validate_answer(task: dict, answer: dict, corpus: IndexedCorpus) -> None:
    entities = task.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("task must contain a nonempty entity roster")
    roster = [entity.get("entity_id") for entity in entities]
    predictions = answer.get("entity_predictions", [])
    ids = [prediction.get("entity_id") for prediction in predictions]
    if (
        any(not isinstance(eid, str) or not eid for eid in roster + ids)
        or len(set(roster)) != len(roster)
        or len(set(ids)) != len(ids)
        or set(ids) != set(roster)
    ):
        raise ValueError(
            "prediction entity IDs must match the exact unique task roster"
        )
    if answer.get("task_id") != task.get("task_id"):
        raise ValueError("answer task_id does not match the task")
    if answer.get("target_type") != target_type(task):
        raise ValueError("answer target_type does not match the task")
    ranks = [prediction.get("rank") for prediction in predictions]
    if any(rank is not None for rank in ranks):
        if any(
            not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks
        ) or sorted(ranks) != list(range(1, len(ids) + 1)):
            raise ValueError("ranks must form a complete permutation of the roster")
    for prediction in predictions:
        validate_prediction(task, prediction, corpus)
