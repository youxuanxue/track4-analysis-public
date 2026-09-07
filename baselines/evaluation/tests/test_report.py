import pytest

from baselines.evaluation.report import markdown, summarize


def row(case_id="a", score=None, admitted=True, returncode=0):
    return {
        "case_id": case_id,
        "seed": 7,
        "profile": "smoke",
        "status": "completed",
        "elapsed_s": 2.5,
        "execution": {"returncode": returncode},
        "assessment": {
            "development_score": score,
            "admissible": admitted,
            "nli_faithfulness": None,
        },
        "diagnostics": {
            "entities": [{"source": "grounded", "fallback_reason": "forced_grounded"}]
        },
    }


def test_public_preview_has_no_score_or_nli_measurement():
    summary = summarize([row()])
    assert summary["development_mean"] is None
    assert summary["official_score"] is None
    assert summary["rankable"] is False
    assert summary["nli_measured_runs"] == 0
    assert summary["fallback_reasons"] == {"forced_grounded": 1}
    report = {
        "mode": "grounded",
        "profile": "smoke",
        "runs": [row()],
        "summary": summary,
    }
    assert "Development composite mean: unmeasured" in markdown(report)


def test_failed_cases_stay_in_development_denominator():
    summary = summarize(
        [row(score=0.7), row("b", score=-0.27, admitted=False, returncode=1)]
    )
    assert summary["development_mean"] == pytest.approx(0.215)
    assert summary["runs"] == 2
    assert summary["admissible_runs"] == 1


def test_partial_scores_never_become_a_smaller_denominator():
    summary = summarize([row(score=0.7), row("b")])
    assert summary["development_scored_runs"] == 1
    assert summary["development_mean"] is None


def test_aborted_assessment_suppresses_aggregate():
    failed = row("b", score=-0.27)
    failed["status"] = "assessment_error"
    assert summarize([row(score=0.7), failed])["development_mean"] is None
