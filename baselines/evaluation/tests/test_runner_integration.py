"""Exercise the development runner across prediction and assessment boundaries."""

import json
import hashlib

import pytest

pytest.importorskip("qfbench2_common")

from baselines.evaluation import __main__ as runner  # noqa: E402
from scoring.tests.synthetic import answer_for, outcome_for  # noqa: E402
from scoring.tests.test_development_evaluation import _case  # noqa: E402


def build_unit(root):
    unit = _case(root)[0]
    manifest_path = unit / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for suffix in ("A", "B", "C"):
        doc = {
            "doc_date": "2026-02-01",
            "text": f"Synthetic Issuer {suffix} reported diluted EPS of 1.20 for Q4 2025.",
        }
        relative = f"corpus/SYN-{suffix}.json"
        payload = json.dumps(doc).encode()
        (unit / relative).write_bytes(payload)
        manifest["files"].append(
            {
                "path": relative,
                "role": "corpus",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    manifest_path.write_text(json.dumps(manifest))
    return unit


def test_public_runner_executes_real_baseline_without_scores(tmp_path):
    units = tmp_path / "inputs"
    build_unit(units)
    out = tmp_path / "report"
    assert runner.main(["--units", str(units), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["summary"]["execution_successes"] == 1
    assert report["summary"]["development_mean"] is None
    assert report["summary"]["nli_measured_runs"] == 0
    assert report["summary"]["entity_sources"] == {"grounded": 3}
    assert report["runs"][0]["assessment"]["hypothesis_records"]
    assert report["by_split"]["public-dev"]["runs"] == 1


@pytest.mark.parametrize("exit_code", [0, 1, 124])
def test_truth_read_after_prediction_and_failed_execution_cannot_score_answer(
    tmp_path, monkeypatch, exit_code
):
    unit = build_unit(tmp_path / "inputs")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps(outcome_for()))
    manifest = tmp_path / "roster.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "independent",
                        "unit_dir": str(unit),
                        "truth_path": str(truth),
                        "split": "test",
                        "group": "independent-event",
                    }
                ],
            }
        )
    )
    finished = False
    original_read = type(truth).read_text

    def checked_read(path, *args, **kwargs):
        if path == truth:
            assert finished, "truth was read before the predictor exited"
        return original_read(path, *args, **kwargs)

    def predict(stage, output, **kwargs):
        nonlocal finished
        assert set(path.name for path in stage.iterdir()) == {"task.json", "corpus"}
        assert not list(stage.rglob("truth.json"))
        runner.write_json(output / "answer.json", answer_for())
        finished = True
        return {
            "returncode": exit_code,
            "timed_out": exit_code == 124,
            "isolation": "test",
        }

    monkeypatch.setattr(type(truth), "read_text", checked_read)
    monkeypatch.setattr(runner, "run_local", predict)
    out = tmp_path / "report"
    assert runner.main(["--manifest", str(manifest), "--out", str(out)]) == (
        0 if exit_code == 0 else 1
    )
    report = json.loads((out / "report.json").read_text())
    score = report["summary"]["development_mean"]
    if exit_code:
        assert score == pytest.approx(-0.27)
        assert report["summary"]["admissible_runs"] == 0
    else:
        assert score > -0.27
    assert report["summary"]["runs"] == 1


def test_invalid_reference_aborts_without_a_partial_scoreboard(tmp_path, monkeypatch):
    unit = build_unit(tmp_path / "inputs")
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"outcomes": []}))
    manifest = tmp_path / "roster.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {
                        "id": "a",
                        "unit_dir": str(unit),
                        "truth_path": str(truth),
                        "split": "test",
                        "group": "a",
                    }
                ],
            }
        )
    )
    out = tmp_path / "report"
    assert runner.main(["--manifest", str(manifest), "--out", str(out)]) == 2
    assert not (out / "report.json").exists()


def test_existing_output_is_not_reused(tmp_path):
    units = tmp_path / "inputs"
    build_unit(units)
    out = tmp_path / "report"
    out.mkdir()
    (out / "report.json").write_text('{"preserved":"user artifact"}')
    assert runner.main(["--units", str(units), "--out", str(out)]) == 2
    assert json.loads((out / "report.json").read_text()) == {
        "preserved": "user artifact"
    }
