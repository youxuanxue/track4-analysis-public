"""Prediction-bound evidence: the hypothesis is derived from the submission, never authored by it.

The measured defect: the entailment hypothesis was ``claim["text"]`` — a string the participant
wrote. A submission whose label and point forecast were both wrong scored **faithfulness 1.0** by
citing a verbatim, entirely unrelated passage of the corpus and describing it accurately. Quoting
the corpus back at itself was a perfect score on the gate that exists to check the *prediction* is
grounded.

This module derives a **canonical hypothesis** from the submitted prediction fields
(``label`` / ``point_forecast`` / ``rank`` / ``interval``) plus the trusted task schema (target
name, entity display name, interval level), and hands the shared
``qfbench2_common.scoring.faithfulness.citation_faithfulness`` a claim list whose ``text`` is that
sentence and whose ``citations`` are exactly the citations the participant attached to that entity.
The shared entailment primitive is therefore reused verbatim — Track 4 owns the *semantics*
(what the hypothesis is), the Hub owns the *primitive* (how entailment is computed), which is the
ownership split in global rule 3.

Two consequences worth stating plainly:

* **Evidence-to-entity relevance is structural.** A citation can only support the entity whose
  prediction it was attached to, because the claim list is built per entity. Evidence pooled across
  entities can no longer support a prediction it was never offered for.
* **The threshold needs recalibrating.** ``faithfulness_threshold`` (0.80) was calibrated against
  entailment of participant-authored prose, which is a far easier target than entailment of a
  prediction. The threshold stays a trusted plan/card parameter so recalibration is a data change;
  it is recorded as an open decision, not silently absorbed by lowering the bar in code.

The participant's own claim text is still parsed — it is kept as an operator-side diagnostic — but
it is never the hypothesis and never reaches the judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .alignment import AlignedPredictions

__all__ = [
    "HypothesisSpec",
    "canonical_hypothesis",
    "prediction_claims",
]


def _fmt(value: float) -> str:
    """Stable decimal rendering, so the same prediction always yields the same hypothesis string."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """The trusted half of the hypothesis: everything that is NOT the participant's prediction."""

    target_type: str
    interval_level: float
    #: Human-readable name of the quantity being predicted, from the trusted task target.
    target_name: str = "outcome"
    #: Optional trusted display names, entity_id -> name. Absent ids fall back to the id.
    entity_names: Mapping[str, str] | None = None

    def name_for(self, entity_id: str) -> str:
        if self.entity_names and entity_id in self.entity_names:
            return f"{self.entity_names[entity_id]} ({entity_id})"
        return entity_id

    @classmethod
    def from_task(
        cls, task: Mapping[str, Any], *, target_type: str, interval_level: float
    ) -> HypothesisSpec:
        target = task.get("target")
        target_name = "outcome"
        if (
            isinstance(target, Mapping)
            and isinstance(target.get("name"), str)
            and target["name"]
        ):
            target_name = target["name"].replace("_", " ")
        names: dict[str, str] = {}
        entities = task.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if (
                    isinstance(entity, Mapping)
                    and isinstance(entity.get("entity_id"), str)
                    and isinstance(entity.get("name"), str)
                    and entity["name"]
                ):
                    names[entity["entity_id"]] = entity["name"]
        return cls(
            target_type=target_type,
            interval_level=float(interval_level),
            target_name=target_name,
            entity_names=names or None,
        )


def canonical_hypothesis(
    spec: HypothesisSpec,
    *,
    entity_id: str,
    label: str,
    point_forecast: float,
    rank: int | None,
    lo: float,
    hi: float,
) -> str:
    """The one sentence the judge is asked about, built only from trusted schema + submitted values.

    Deterministic by construction: the same submitted numbers always produce the same string, so a
    re-run of the same submission asks the judge the same question. Nothing the participant wrote
    in prose appears anywhere in it.
    """
    subject = spec.name_for(entity_id)
    level_pct = _fmt(spec.interval_level * 100.0)
    interval_clause = (
        f" The {level_pct}% prediction interval for the {spec.target_name} of {subject} "
        f"is {_fmt(lo)} to {_fmt(hi)}."
    )
    if spec.target_type == "classification":
        head = f"The {spec.target_name} of {subject} is {label}."
    elif spec.target_type == "regression":
        head = f"The {spec.target_name} of {subject} is {_fmt(point_forecast)}."
    else:  # ranking
        position = f"ranked {rank}" if rank is not None else "ranked"
        head = (
            f"By {spec.target_name}, {subject} is {position} among the entities in this task, "
            f"with a score of {_fmt(point_forecast)}."
        )
    return head + interval_clause


def prediction_claims(
    aligned: AlignedPredictions, spec: HypothesisSpec
) -> list[dict[str, Any]]:
    """One normalized claim per roster entity: hypothesis = the prediction, citations = its own.

    The returned list is in trusted roster order and has exactly one element per roster entity, so
    the faithfulness denominator is the roster and not the number of sentences the participant
    chose to write. Padding a submission with extra prose can no longer move the score.
    """
    claims: list[dict[str, Any]] = []
    for index, entity_id in enumerate(aligned.entity_ids):
        rank = aligned.ranks[index]
        claims.append(
            {
                "text": canonical_hypothesis(
                    spec,
                    entity_id=entity_id,
                    label=aligned.pred_labels[index],
                    point_forecast=aligned.pred_values[index],
                    rank=rank,
                    lo=aligned.lo[index],
                    hi=aligned.hi[index],
                ),
                "citations": list(aligned.citations_by_entity[index]),
            }
        )
    return claims
