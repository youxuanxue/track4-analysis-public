"""Assemble and self-check the final answer JSON.

The self-check enforces the mistakes that are cheap to catch locally and fatal
at scoring time: claims whose spans do not resolve in the corpus, missing
interval bounds, and a wrong top-level shape (``notes`` must be an object).
It is a last-line assertion — grounding already happened in the agent.
"""

from __future__ import annotations

from statistics import median

from .agent import EntityResult
from .indexer import IndexedCorpus
from .schema import target_type
from .validation import validate_answer


def _shrink_regression_predictions(
    predictions: list[dict], kind: str | None, results: list[EntityResult]
) -> list[dict]:
    """Shrink extreme cross-sectional outliers towards the median for ungrounded/model estimates."""
    if kind != "regression" or len(predictions) < 3:
        return predictions
    # ``rationale`` belongs to the grounded reasoner and is intentionally empty
    # for accepted model responses.  Looking for the word "model" there made
    # this safety net dead code on every successful API prediction.  The source
    # marker is the authoritative signal and also keeps this independent of
    # human-readable rationale wording.
    shrunk_entities = {
        r.prediction["entity_id"] for r in results if r.source == "model"
    }
    if len(shrunk_entities) < 3:
        return predictions
    raw_points = [
        p.get("point_forecast")
        for p in predictions
        if p.get("entity_id") in shrunk_entities
    ]
    if not all(
        isinstance(p, (int, float)) and not isinstance(p, bool)
        for p in raw_points
    ):
        return predictions
    pts = [float(p) for p in raw_points]
    med = median(pts)
    deviations = [abs(p - med) for p in pts]
    mad = median(deviations)
    if mad < 1e-9:
        return predictions
    max_dev = 2.5 * mad
    updated = []
    for pred in predictions:
        new_p = dict(pred)
        eid = new_p.get("entity_id")
        if eid in shrunk_entities:
            pt = float(new_p["point_forecast"])
            clamped = max(med - max_dev, min(med + max_dev, pt))
            shrunk = 0.85 * clamped + 0.15 * med
            new_p["point_forecast"] = round(shrunk, 4)
            if "interval" in new_p and isinstance(new_p["interval"], dict):
                lo = new_p["interval"].get("lo")
                hi = new_p["interval"].get("hi")
                if lo is not None and new_p["point_forecast"] < lo:
                    new_p["interval"]["lo"] = new_p["point_forecast"]
                if hi is not None and new_p["point_forecast"] > hi:
                    new_p["interval"]["hi"] = new_p["point_forecast"]
        updated.append(new_p)
    return updated


def _normalize_rank_outputs(
    predictions: list[dict], kind: str | None
) -> list[dict]:
    """Repair the common ``rank 1 = best`` response on ranking tasks.

    The scorer orders ``point_forecast`` with larger values first.  A model can
    still occasionally return a permutation of one-based rank integers despite
    the prompt asking for a continuous score.  Such a permutation is
    unambiguous and can be inverted without changing the evidence or the
    roster.  Genuine metric forecasts and tied values pass through unchanged.
    """
    if kind != "ranking" or len(predictions) < 2:
        return predictions
    raw = [p.get("point_forecast") for p in predictions]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        for value in raw
    ):
        return predictions
    values = [int(value) for value in raw]
    count = len(values)
    if set(values) != set(range(1, count + 1)):
        return predictions
    updated: list[dict] = []
    for prediction, rank in zip(predictions, values, strict=True):
        new_prediction = dict(prediction)
        point = count - rank + 1
        new_prediction["point_forecast"] = point
        interval = new_prediction.get("interval")
        if isinstance(interval, dict):
            interval = dict(interval)
            if isinstance(interval.get("lo"), (int, float)):
                interval["lo"] = min(interval["lo"], point)
            if isinstance(interval.get("hi"), (int, float)):
                interval["hi"] = max(interval["hi"], point)
            new_prediction["interval"] = interval
        updated.append(new_prediction)
    return updated


def build_answer(
    task: dict, results: list[EntityResult], corpus: IndexedCorpus
) -> dict:
    total_dropped = sum(r.dropped_claims for r in results)
    total_claims = sum(len(r.prediction["claims"]) for r in results)
    kind = target_type(task)
    predictions = [dict(r.prediction) for r in results]
    predictions = _shrink_regression_predictions(predictions, kind, results)
    predictions = _normalize_rank_outputs(predictions, kind)
    answer: dict = {
        "task_id": task.get("task_id", ""),
        "schema_version": task.get("schema_version", "3"),
    }
    if kind is not None:
        answer["target_type"] = kind
    answer["entity_predictions"] = predictions
    answer["evidence_trace"] = (
        f"strong_rag_baseline: BM25 retrieval and evidence-grounded prediction. "
        f"{total_claims} citations kept, {total_dropped} evidence items dropped. "
        f"Final citation offsets and dates checked against the current corpus. "
        f"This validation does not measure predictive quality or NLI faithfulness."
    )
    answer["notes"] = {
        "agent": "strong_rag_baseline",
        "retrieval": "bm25-span-chunks",
        "reasoner": "validated-model-or-deterministic-fallback",
        "dropped_evidence_items": total_dropped,
        "fallback_rationale": {
            r.prediction["entity_id"]: r.rationale for r in results if r.rationale
        },
    }
    validate_answer(task, answer, corpus)
    return answer
