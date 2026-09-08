"""Compare whole matched rosters, with event-level rather than entity-level uncertainty."""

import copy

import pytest

from baselines.evaluation.compare import compare


def report():
    return {
        "provenance": {"toolkit_source_digest": "same"},
        "runs": [
            {
                "case_id": f"case-{i}",
                "seed": 1,
                "status": "completed",
                "profile": "smoke",
                "split": "test",
                "group": f"event-{i // 3}",
                "target_type": "regression",
                "input_digest": f"input-{i}",
                "truth_digest": f"truth-{i}",
                "assessment": {"development_score": 0.1, "admissible": True},
            }
            for i in range(6)
        ],
    }


def test_equal_event_weights_and_failed_units_retained():
    before = report()
    after = copy.deepcopy(before)
    after["runs"][0]["assessment"].update(development_score=-0.27, admissible=False)
    result = compare(before, after, samples=1000)
    assert result["overall"]["independent_groups"] == 2
    assert result["overall"]["event_mean_delta"] == pytest.approx(-0.37 / 6)
    assert result["after_failures"] == 1
    assert result == compare(before, after, samples=1000)
    assert result["rankable"] is False


@pytest.mark.parametrize(
    "field",
    ["input_digest", "truth_digest", "profile", "group", "split", "target_type"],
)
def test_incomparable_rows_refused(field):
    before, after = report(), report()
    after["runs"][0][field] = "different"
    with pytest.raises(ValueError, match="differs"):
        compare(before, after)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "unscored", "incomplete", "toolkit"]
)
def test_partial_or_mixed_scoreboards_refused(defect):
    before, after = report(), report()
    if defect == "missing":
        after["runs"].pop()
    elif defect == "duplicate":
        after["runs"].append(after["runs"][0])
    elif defect == "unscored":
        after["runs"][0]["assessment"]["development_score"] = None
    elif defect == "incomplete":
        after["runs"][0]["status"] = "running"
    else:
        after["provenance"]["toolkit_source_digest"] = "different"
    with pytest.raises(ValueError):
        compare(before, after)
