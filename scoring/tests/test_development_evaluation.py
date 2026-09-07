"""Local evaluation delegates to official gates without publishing reference data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from baselines.evaluation.assess import assess_unit
from qfbench2_track_analysis.codes import T4OrganizerFault
from qfbench2_track_analysis.hypothesis import prediction_claims
from qfbench2_track_analysis.judge_factory import ENV_JUDGE_SPEC
from qfbench2_track_analysis.scoring import (
    build_smoke_verifier,
    score_unit,
)

from .synthetic import answer_for, build_unit, outcome_for


def _case(
    tmp_path: Path,
    *,
    target_type: str = "classification",
    with_outcome: bool = False,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    unit = build_unit(tmp_path, target_type=target_type, with_outcome=with_outcome)
    manifest_path = unit / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_bytes = (unit / "task.json").read_bytes()
    manifest["files"].append(
        {
            "path": "task.json",
            "role": "task",
            "sha256": hashlib.sha256(task_bytes).hexdigest(),
            "bytes": len(task_bytes),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    answer = answer_for()
    answer["target_type"] = target_type
    truth = outcome_for()
    truth["target_type"] = target_type
    for index, (prediction, actual) in enumerate(
        zip(answer["entity_predictions"], truth["outcomes"])
    ):
        prediction["point_forecast"] = actual["y"]
        prediction["label"] = actual["true_label"]
        if target_type == "ranking":
            prediction["rank"] = index + 1
    _write_answer(output, answer)
    return unit, output, answer, truth


def _write_answer(output: Path, answer: dict[str, Any]) -> None:
    (output / "answer.json").write_text(json.dumps(answer), encoding="utf-8")


@pytest.mark.parametrize("target_type", ["classification", "regression", "ranking"])
def test_development_score_matches_canonical_scorer(
    tmp_path: Path, target_type: str
) -> None:
    unit, output, _, truth = _case(tmp_path, target_type=target_type)
    result = assess_unit(unit, output, realized=truth)
    ctx: dict[str, Any] = {
        "unit_dir": unit,
        "output_dir": output,
        "realized": truth,
    }
    build_smoke_verifier(ctx).run(ctx)
    expected = score_unit(
        ctx, judge=ctx["judge"], judge_provenance=ctx["judge_provenance"]
    )
    assert result["development_score"] == expected.score
    assert result["predictive_quality"] == 1.0
    assert result["interval_coverage"] == 1.0
    assert result["official_score"] is None
    assert result["rankable"] is False
    assert result["nli_faithfulness"] is None
    assert result["faithfulness_gate_applied"] is False
    assert [record["hypothesis"] for record in result["hypothesis_records"]] == [
        claim["text"]
        for claim in prediction_claims(ctx["_aligned"], ctx["_hypothesis_spec"])
    ]
    assert all(
        citation["span_valid"] and citation["embargo_clean"]
        for record in result["hypothesis_records"]
        for citation in record["citations"]
    )
    json.dumps(result, allow_nan=False)


def test_pure_label_development_unit_has_no_interval_coverage(tmp_path: Path) -> None:
    unit, output, answer, truth = _case(tmp_path)
    for actual in truth["outcomes"]:
        del actual["y"]
    for prediction in answer["entity_predictions"]:
        del prediction["point_forecast"]
    _write_answer(output, answer)
    result = assess_unit(unit, output, realized=truth)
    assert result["admissible"] is True
    assert result["development_score"] == pytest.approx(0.7)
    assert result["interval_coverage"] is None


def test_failed_submission_retains_canonical_worst_case(tmp_path: Path) -> None:
    unit, output, answer, truth = _case(tmp_path)
    answer["entity_predictions"].pop()
    _write_answer(output, answer)
    result = assess_unit(unit, output, realized=truth)
    assert result["admissible"] is False
    assert result["development_score"] == pytest.approx(-0.27)
    assert result["gates"]["g3_domain_semantics"]["passed"] is False
    assert result["predictive_quality"] is None
    assert result["labels"]


@pytest.mark.parametrize("with_outcome", [False, True])
def test_no_explicit_truth_never_scores_or_discovers_reference(
    tmp_path: Path, with_outcome: bool
) -> None:
    unit, output, _, _ = _case(tmp_path, with_outcome=with_outcome)
    if with_outcome:
        # The reference would abort if hydrate() discovered it.
        (unit / "reference" / "outcome.json").write_text("not JSON", encoding="utf-8")
    result = assess_unit(unit, output)
    assert result["admissible"] is True
    assert result["development_score"] is None
    assert result["predictive_quality"] is None
    assert result["interval_coverage"] is None


@pytest.mark.parametrize("defect", ["extra", "missing", "duplicate", "nan", "mixed"])
def test_invalid_external_truth_aborts_even_when_submission_is_bad(
    tmp_path: Path, defect: str
) -> None:
    unit, output, answer, truth = _case(tmp_path)
    if defect == "extra":
        truth["outcomes"].append(
            {"entity_id": "SYN-EXTRA", "true_label": "beat", "y": 4.0}
        )
    elif defect == "missing":
        truth["outcomes"].pop()
    elif defect == "duplicate":
        truth["outcomes"][1] = dict(truth["outcomes"][0])
    elif defect == "nan":
        truth["outcomes"][0]["y"] = float("nan")
    else:
        del truth["outcomes"][0]["y"]
    answer["entity_predictions"].pop()
    _write_answer(output, answer)
    with pytest.raises(T4OrganizerFault):
        assess_unit(unit, output, realized=truth)


def test_out_of_range_span_is_a_diagnostic_not_formal_nli_support(
    tmp_path: Path,
) -> None:
    unit, output, answer, _ = _case(tmp_path)
    answer["entity_predictions"][0]["claims"][0]["span_end"] = 100_000
    _write_answer(output, answer)
    result = assess_unit(unit, output)
    assert result["admissible"] is True
    citation = result["hypothesis_records"][0]["citations"][0]
    assert citation["span_valid"] is False
    assert citation["embargo_clean"] is True
    assert citation["doc_date"] == "2026-02-01"
    assert result["nli_faithfulness"] is None
    assert result["faithfulness_gate_applied"] is False


def test_production_profile_refuses_missing_pinned_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit, output, _, truth = _case(tmp_path)
    monkeypatch.delenv(ENV_JUDGE_SPEC, raising=False)
    with pytest.raises(T4OrganizerFault, match="no judge artifact"):
        assess_unit(unit, output, realized=truth, profile="production")


@pytest.mark.parametrize("metadata", ["card", "manifest"])
def test_invalid_unit_metadata_aborts_before_assessment(
    tmp_path: Path, metadata: str
) -> None:
    unit, output, _, _ = _case(tmp_path)
    if metadata == "card":
        card = unit / "card.toml"
        card.write_text(
            card.read_text().replace(
                'track           = "analysis"', 'track           = "invalid"'
            )
        )
    else:
        corpus_path = next((unit / "corpus").glob("*.json"))
        corpus_path.write_text(corpus_path.read_text() + " ")
    with pytest.raises(T4OrganizerFault, match="validation failed"):
        assess_unit(unit, output)
