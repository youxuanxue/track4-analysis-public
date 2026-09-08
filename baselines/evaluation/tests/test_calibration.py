"""Synthetic controls for grouped quantiles, temporal isolation and replay."""

import copy
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from baselines.evaluation import calibration as cal
from baselines.evaluation.dataset import Case


def test_event_maxima_not_entities_determine_sample_size_and_radius():
    errors = {f"event-{i}": [float(i), -float(i + 1)] * 8 for i in range(9)}
    fitted = cal.block_radius(errors, 0.9)
    assert fitted["radius"] == 9
    assert fitted["event_count"] == fitted["order_statistic"] == 9
    assert (
        cal.block_radius({key: values * 10 for key, values in errors.items()}, 0.9)
        == fitted
    )
    with pytest.raises(ValueError, match="insufficient"):
        cal.block_radius({"one-month": list(range(1000))}, 0.9)
    with pytest.raises(ValueError, match="insufficient"):
        cal.block_radius(dict(list(errors.items())[:8]), 0.9)


def test_finite_sample_rank_and_ties_are_deterministic():
    result = cal.block_radius({str(i): [float(i)] for i in range(20)}, 0.9)
    assert result["order_statistic"] == 19
    assert result["radius"] == 18
    assert cal.block_radius({str(i): [0.0] for i in range(9)}, 0.9)["radius"] == 0


@pytest.mark.parametrize("level", [0, 1, True, float("nan"), float("inf")])
def test_invalid_levels_refused(level):
    with pytest.raises(ValueError):
        cal.block_radius({"a": [1]}, level)


@pytest.mark.parametrize(
    "errors",
    [{}, {"a": []}, {"a": [True]}, {"a": [float("nan")]}, {"a": [float("inf")]}],
)
def test_invalid_residuals_refused(errors):
    with pytest.raises(ValueError):
        cal.block_radius(errors, 0.9)


def case(tmp_path, name, cutoff, resolution, split="train", group=None):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "task.json").write_text(
        json.dumps({"cutoff_date": cutoff, "resolution_date": resolution})
    )
    return Case(name, folder, split, group or name)


def test_views_share_an_event_but_other_groups_cannot_duplicate_its_window(tmp_path):
    a = case(tmp_path, "a", "2024-01-01", "2024-01-20", group="event")
    view = case(tmp_path, "view", "2024-01-01", "2024-01-20", group="event")
    b = case(tmp_path, "b", "2024-02-01", "2024-02-20", split="calibration")
    cal.validate_partitions([a, view], [b])
    copied = Case(view.case_id, view.unit_dir, "train", "fake-independent-event")
    with pytest.raises(ValueError, match="overlap|duplicated"):
        cal.validate_partitions([a, copied], [b])


@pytest.mark.parametrize("defect", ["test", "future", "same_group", "overlap_views"])
def test_partition_leaks_are_rejected(tmp_path, defect):
    a = case(tmp_path, "a", "2024-01-01", "2024-01-20")
    b = case(tmp_path, "b", "2024-02-01", "2024-02-20", split="calibration")
    if defect == "test":
        b = Case(b.case_id, b.unit_dir, "test", b.group)
    elif defect == "future":
        a, b = (
            Case(b.case_id, b.unit_dir, "train", b.group),
            Case(a.case_id, a.unit_dir, "calibration", a.group),
        )
    elif defect == "same_group":
        b = Case(b.case_id, b.unit_dir, b.split, a.group)
    else:
        extra = case(tmp_path, "extra", "2024-01-02", "2024-01-21", group=a.group)
        with pytest.raises(ValueError, match="identical"):
            cal.validate_partitions([a, extra], [b])
        return
    with pytest.raises(ValueError):
        cal.validate_partitions([a], [b])


def task_and_answer():
    entity = {"entity_id": "SYN"}
    task = {
        "family": "invented",
        "entities": [entity],
        "cutoff_date": "2024-02-01",
        "resolution_date": "2024-02-21",
        "target": {"name": "ratio", "unit": "ratio", "type": "regression"},
    }
    answer = {
        "entity_predictions": [
            {
                "entity_id": "SYN",
                "point_forecast": 1.0,
                "interval": {"level": 0.9, "lo": 0, "hi": 100},
                "claims": [{"claim": "unchanged"}],
                "rank": 1,
            }
        ]
    }
    key, contract = cal._contract(task, entity)
    return task, answer, {key: {"radius": 2, "contract": contract}}


def test_apply_preserves_point_order_and_evidence_clipping_only_to_task_domain():
    task, answer, bins = task_and_answer()
    snapshot = copy.deepcopy(answer)
    result = cal._intervals(answer, task, bins)
    assert answer == snapshot
    prediction = result["entity_predictions"][0]
    assert prediction["interval"] == {"level": 0.9, "lo": 0, "hi": 3}
    assert {k: v for k, v in prediction.items() if k != "interval"} == {
        k: v for k, v in answer["entity_predictions"][0].items() if k != "interval"
    }
    assert "supersede" in result["notes"]["interval_calibration"]


def test_inferred_unit_does_not_authorize_calibration_without_a_declaration():
    task, answer, bins = task_and_answer()
    del task["target"]["unit"]
    with pytest.raises(ValueError, match="explicit"):
        cal._intervals(answer, task, bins)


@pytest.mark.parametrize(
    "defect", ["horizon", "unit", "kind", "family", "roster", "entity_unit", "currency"]
)
def test_calibration_cannot_be_borrowed_for_a_different_contract(defect):
    task, answer, bins = task_and_answer()
    if defect == "horizon":
        task["resolution_date"] = "2024-02-22"
    elif defect == "family":
        task["family"] = "different"
    elif defect == "roster":
        task["entities"].append({"entity_id": "OTHER"})
    elif defect in ("entity_unit", "currency"):
        task["entities"][0]["unit" if defect == "entity_unit" else "currency"] = (
            "different"
        )
    else:
        task["target"]["type" if defect == "kind" else "unit"] = (
            "ranking" if defect == "kind" else "percent"
        )
    with pytest.raises(ValueError, match="no fitted bin"):
        cal._intervals(answer, task, bins)


@pytest.mark.parametrize("mode", ["grounded", "model"])
def test_integration_replays_real_scorer_without_opening_application_truth_early(
    tmp_path, monkeypatch, mode
):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation import historical, __main__ as runner
    from baselines.evaluation.tests.test_runtime import write_runtime

    original_local = runner.run_local

    def synthetic_model(stage, output, **kwargs):
        result = original_local(stage, output, **(kwargs | {"mode": "grounded"}))
        path = output / "diagnostics.json"
        diagnostics = json.loads(path.read_text())
        diagnostics["mode"] = "model"
        for entity in diagnostics["entities"]:
            entity.update(source="model", fallback_reason=None)
        path.write_text(json.dumps(diagnostics))
        return result

    if mode == "model":
        monkeypatch.setattr(runner, "run_local", synthetic_model)
        monkeypatch.setenv("MODEL_ENDPOINT", "http://127.0.0.1:17171/v1")
        monkeypatch.setenv("MODEL_NAME", "synthetic")
        actual_provenance = runner.provenance
        monkeypatch.setattr(
            runner, "provenance", lambda: actual_provenance() | {"git_dirty": False}
        )

    spec_path = tmp_path / "events.json"
    events = []
    for month in range(1, 11):
        cutoff = date(2023, month, 1)
        events.append(
            {
                "id": f"synthetic-{month}",
                "cutoff": str(cutoff),
                "resolution": str(cutoff + timedelta(days=20)),
                "split": "train" if month <= 9 else "calibration",
            }
        )
    spec_path.write_text(json.dumps({"version": 1, "events": events}))

    def invented(cache, series, vintage, **kwargs):
        base = Decimal(historical.SERIES[series]) / 10
        rows = {
            vintage - timedelta(days=i): base + Decimal(vintage.day) / 100
            for i in range(1, 30)
        }
        return rows, {"url": "https://example.invalid/synthetic", "sha256": "synthetic"}

    monkeypatch.setattr(historical, "snapshot", invented)
    manifest = historical.build(spec_path, tmp_path / "cache", tmp_path / "benchmark")
    source = json.loads(manifest.read_text())
    manifests, reports, runtimes = {}, {}, {}
    for split in ("train", "calibration"):
        selected = [
            dict(
                c,
                unit_dir=str(manifest.parent / c["unit_dir"]),
                truth_path=str(manifest.parent / c["truth_path"]),
            )
            for c in source["cases"]
            if c["split"] == split
        ]
        manifests[split] = tmp_path / f"{split}.json"
        manifests[split].write_text(json.dumps({"version": 1, "cases": selected}))
        reports[split] = tmp_path / f"{split}-run" / "report.json"
        assert (
            runner.main(
                [
                    "--manifest",
                    str(manifests[split]),
                    "--out",
                    str(reports[split].parent),
                    "--mode",
                    mode,
                ]
            )
            == 0
        )
        if mode == "model":
            runtimes[split] = write_runtime(tmp_path, reports[split], manifests[split])
    original = json.loads(reports["calibration"].read_text())
    out = tmp_path / "adjusted"
    application_truth = {
        Path(c["truth_path"])
        for c in json.loads(manifests["calibration"].read_text())["cases"]
    }
    read_text = Path.read_text

    def guarded(path, *args, **kwargs):
        if path in application_truth:
            assert len(list(out.glob("*/seed-*/answer.json"))) == 3
            assert json.loads((out / "calibration.json").read_text())["bins"]
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    result = cal.calibrate(
        manifests["train"],
        reports["train"],
        manifests["calibration"],
        reports["calibration"],
        out,
        fit_runtime=runtimes.get("train"),
        apply_runtime=runtimes.get("calibration"),
    )
    assert result["summary"]["admissible_runs"] == 3
    assert result["summary"]["official_score"] is None
    artifact = json.loads((out / "calibration.json").read_text())
    assert {b["event_count"] for b in artifact["bins"].values()} == {9}
    assert {b["radius"] for b in artifact["bins"].values()} == {20}
    for before, after in zip(original["runs"], result["runs"]):
        assert (
            before["assessment"]["numeric_errors"]
            == after["assessment"]["numeric_errors"]
        )
        assert after["interval_diagnostics"]["after_mean_width"] == 40
        assert before["answer_sha256"] != after["answer_sha256"]
    if mode == "model":
        assert artifact["prediction_provenance"]["model_runtime_identity"]
        assert set(artifact["runtime_record_sha256"]) == {"fit", "apply"}
        # A different generation context must not borrow the earlier residuals.
        runtime = json.loads(runtimes["calibration"].read_text())
        runtime["command"][-1] = "8192"
        runtimes["calibration"].write_text(json.dumps(runtime))
        with pytest.raises(ValueError, match="runtime identities differ"):
            cal.calibrate(
                manifests["train"],
                reports["train"],
                manifests["calibration"],
                reports["calibration"],
                tmp_path / "changed-runtime",
                fit_runtime=runtimes["train"],
                apply_runtime=runtimes["calibration"],
            )
        runtime["command"][-1] = "16384"
        runtimes["calibration"].write_text(json.dumps(runtime))
        report = json.loads(reports["calibration"].read_text())
        report["runs"][0]["diagnostics"]["entities"][0]["source"] = "grounded"
        reports["calibration"].write_text(json.dumps(report))
        write_runtime(tmp_path, reports["calibration"], manifests["calibration"])
        with pytest.raises(ValueError, match="fallback entity"):
            cal.calibrate(
                manifests["train"],
                reports["train"],
                manifests["calibration"],
                reports["calibration"],
                tmp_path / "fallback-model",
                fit_runtime=runtimes["train"],
                apply_runtime=runtimes["calibration"],
            )
        return
    with pytest.raises(ValueError, match="must be new"):
        cal.calibrate(
            manifests["train"],
            reports["train"],
            manifests["calibration"],
            reports["calibration"],
            out,
        )
    saved_report = reports["train"].read_bytes()
    for field in ("source_digest", "toolkit_source_digest", "prediction_settings"):
        altered = json.loads(saved_report)
        altered["provenance"][field] = (
            {"mode": "grounded", "top_k": 1}
            if field == "prediction_settings"
            else "different"
        )
        reports["train"].write_text(json.dumps(altered))
        with pytest.raises(ValueError, match=field):
            cal.calibrate(
                manifests["train"],
                reports["train"],
                manifests["calibration"],
                reports["calibration"],
                tmp_path / field,
            )
        assert not (tmp_path / field).exists()
    reports["train"].write_bytes(saved_report)
    # An altered answer cannot acquire a fresh calibrated report.
    row = original["runs"][0]
    answer_path = (
        reports["calibration"].parent
        / row["case_id"]
        / f"seed-{row['seed']}"
        / "answer.json"
    )
    answer_path.write_text(answer_path.read_text() + " ")
    with pytest.raises(ValueError, match="answer digest"):
        cal.calibrate(
            manifests["train"],
            reports["train"],
            manifests["calibration"],
            reports["calibration"],
            tmp_path / "tampered",
        )


def test_model_or_incomplete_reports_cannot_supply_residuals(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"mode": "model", "profile": "smoke"}))
    with pytest.raises(ValueError, match="grounded"):
        cal._load_report(path, [])
    path.write_text(
        json.dumps(
            {
                "mode": "grounded",
                "profile": "smoke",
                "provenance": {"prediction_settings": {"mode": "grounded", "top_k": 8}},
                "runs": [],
            }
        )
    )
    with pytest.raises(ValueError, match="complete"):
        cal._load_report(path, [])


def test_public_output_is_refused_before_opening_any_reports(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    monkeypatch.setattr(cal, "public_roots", lambda: [public])
    missing = tmp_path / "not-read.json"
    with pytest.raises(ValueError, match="outside public worktrees"):
        cal.calibrate(missing, missing, missing, missing, public / "calibrated")
