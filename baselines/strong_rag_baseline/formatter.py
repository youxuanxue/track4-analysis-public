"""Assemble and self-check the final answer JSON.

The self-check enforces the mistakes that are cheap to catch locally and fatal
at scoring time: claims whose spans do not resolve in the corpus, missing
interval bounds, and a wrong top-level shape (``notes`` must be an object).
It is a last-line assertion — grounding already happened in the agent.
"""

from __future__ import annotations

from .agent import EntityResult
from .indexer import IndexedCorpus
from .schema import target_type
from .validation import validate_answer


def build_answer(
    task: dict, results: list[EntityResult], corpus: IndexedCorpus
) -> dict:
    total_dropped = sum(r.dropped_claims for r in results)
    total_claims = sum(len(r.prediction["claims"]) for r in results)
    kind = target_type(task)
    predictions = [dict(r.prediction) for r in results]
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
