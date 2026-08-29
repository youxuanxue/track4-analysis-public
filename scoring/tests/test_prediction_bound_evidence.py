"""T4-2: the judge is asked about the SUBMITTED PREDICTION, never about participant prose.

Measured on the pre-fix path: a submission whose label and point forecast were both wrong, citing a
verbatim but entirely unrelated passage of the corpus and describing that passage accurately,
scored **faithfulness 1.0** and passed the admissibility gate. The hypothesis handed to the judge
was ``claim["text"]`` — a string the participant wrote — so quoting the corpus back at itself was a
perfect score on the gate that exists to check the prediction is grounded.

The two tests the brief names are here as a pair:

* *supported prediction passes* — the positive control;
* *supported trivia paired with an unsupported prediction fails* — the exploit, with a judge that
  genuinely entails the trivia premise and still refuses, because the hypothesis it is asked about
  is the prediction and not the prose.

The recorded ``(premise, hypothesis)`` pairs are asserted directly: a gate that passes for the
wrong reason is indistinguishable from one that passes for the right reason unless you look at the
question it asked.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from qfbench2_track_analysis.alignment import EntityRoster, align_predictions
from qfbench2_track_analysis.codes import T4Reason
from qfbench2_track_analysis.hypothesis import (
    HypothesisSpec,
    canonical_hypothesis,
    prediction_claims,
)
from qfbench2_track_analysis.judge_factory import build_smoke_judge
from qfbench2_track_analysis.scoring import DOMAIN_MIN, UnitOutcome, score_unit

from .synthetic import (
    SUPPORTING_TEXT,
    TRIVIA_DOC,
    TRIVIA_TEXT,
    HypothesisAwareJudge,
    StubJudge,
    answer_for,
    build_unit,
)

ROSTER = ("SYN-A",)


def _run(tmp_path: pathlib.Path, answer: dict[str, Any], judge: object) -> UnitOutcome:
    unit = build_unit(tmp_path, entities=ROSTER, with_outcome=True)
    out = tmp_path / "res"
    out.mkdir(exist_ok=True)
    (out / "answer.json").write_text(json.dumps(answer), encoding="utf-8")
    _, provenance = build_smoke_judge()
    return score_unit(
        {"unit_dir": unit, "output_dir": out}, judge=judge, judge_provenance=provenance
    )


# --- positive control -------------------------------------------------------------------------
def test_a_supported_prediction_passes(tmp_path: pathlib.Path) -> None:
    judge = HypothesisAwareJudge(SUPPORTING_TEXT, required_in_hypothesis="is beat")
    outcome = _run(tmp_path, answer_for(entities=ROSTER, label="beat"), judge)
    assert outcome.state == "participant_success"
    assert outcome.diagnostics["faithfulness"] == 1.0


# --- the exploit ------------------------------------------------------------------------------
def test_supported_trivia_with_an_unsupported_prediction_fails(
    tmp_path: pathlib.Path,
) -> None:
    """The judge genuinely entails the trivia premise. The PREDICTION is still not supported."""
    judge = HypothesisAwareJudge(TRIVIA_TEXT, required_in_hypothesis="public holiday")
    answer = answer_for(
        entities=ROSTER,
        doc_id=TRIVIA_DOC,
        label="miss",
        claim_text="The synthetic exchange observes a public holiday in February.",
    )
    outcome = _run(tmp_path, answer, judge)
    assert outcome.state == "participant_failure"
    assert outcome.score == DOMAIN_MIN
    assert outcome.diagnostics["reason"] == T4Reason.EVIDENCE_UNSUPPORTED.value
    # And the reason it failed: the judge was asked about the prediction, not the prose.
    assert judge.calls, "the judge must be consulted at all"
    premise, hypothesis = judge.calls[0]
    assert premise == TRIVIA_TEXT
    assert "public holiday" not in hypothesis
    assert "is miss" in hypothesis


def test_the_hypothesis_never_contains_participant_authored_text(
    tmp_path: pathlib.Path,
) -> None:
    marker = "PARTICIPANT-AUTHORED-MARKER-STRING"
    judge = StubJudge((SUPPORTING_TEXT,))
    answer = answer_for(entities=ROSTER, claim_text=marker)
    _run(tmp_path, answer, judge)
    assert judge.calls
    assert all(marker not in hypothesis for _, hypothesis in judge.calls)


def test_changing_only_the_prediction_changes_the_question_asked(
    tmp_path: pathlib.Path,
) -> None:
    """If the hypothesis did not depend on the prediction, these two would be identical."""
    beat_judge = StubJudge((SUPPORTING_TEXT,))
    _run(tmp_path / "beat", answer_for(entities=ROSTER, label="beat"), beat_judge)
    miss_judge = StubJudge((SUPPORTING_TEXT,))
    _run(tmp_path / "miss", answer_for(entities=ROSTER, label="miss"), miss_judge)
    assert beat_judge.calls[0][1] != miss_judge.calls[0][1]


# --- structure ---------------------------------------------------------------------------------
def test_one_claim_per_roster_entity_so_padding_prose_cannot_move_the_score() -> None:
    roster = EntityRoster(entity_ids=("SYN-A", "SYN-B"))
    answer = answer_for(entities=("SYN-A", "SYN-B"))
    # Pad one entity with nine extra easily-entailed sentences.
    padding = dict(answer["entity_predictions"][0]["claims"][0])
    answer["entity_predictions"][0]["claims"].extend(dict(padding) for _ in range(9))
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.90
    )
    spec = HypothesisSpec(target_type="classification", interval_level=0.90)
    claims = prediction_claims(aligned, spec)
    assert (
        len(claims) == 2
    ), "the denominator is the roster, not the number of sentences written"


def test_evidence_is_bound_to_the_entity_it_was_offered_for() -> None:
    roster = EntityRoster(entity_ids=("SYN-A", "SYN-B"))
    answer = answer_for(entities=("SYN-A", "SYN-B"))
    answer["entity_predictions"][1]["claims"][0]["doc_id"] = "SYNTHDOC_OTHER"
    aligned = align_predictions(
        answer, roster, target_type="classification", interval_level=0.90
    )
    spec = HypothesisSpec(target_type="classification", interval_level=0.90)
    claims = prediction_claims(aligned, spec)
    assert [c["citations"][0]["doc_id"] for c in claims] == [
        answer["entity_predictions"][0]["claims"][0]["doc_id"],
        "SYNTHDOC_OTHER",
    ]


def test_the_canonical_hypothesis_is_deterministic() -> None:
    spec = HypothesisSpec(
        target_type="regression", interval_level=0.90, target_name="eps"
    )
    first = canonical_hypothesis(
        spec,
        entity_id="SYN-A",
        label="",
        point_forecast=1.25,
        rank=None,
        lo=1.0,
        hi=1.5,
    )
    second = canonical_hypothesis(
        spec,
        entity_id="SYN-A",
        label="",
        point_forecast=1.25,
        rank=None,
        lo=1.0,
        hi=1.5,
    )
    assert first == second
    assert "1.25" in first
