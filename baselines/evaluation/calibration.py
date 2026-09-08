"""External event-block interval calibration for frozen local predictions.

Fit on earlier training events, apply to later calibration events, and rescore
through the official toolkit. Exchangeability is an assumption, not a property
established by chronological financial data or by this local smoke workflow.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from statistics import mean

from baselines.strong_rag_baseline.indexer import calendar_date
from baselines.strong_rag_baseline.quantities import TargetSpec, finite_number

from .assess import assess_unit
from .dataset import Case, load_cases, public_roots, require_external, stage_inputs
from .historical import write_json
from .report import markdown, summarize


def block_radius(errors: dict[str, list[float]], level: float) -> dict:
    """The finite-sample order statistic over event maxima, including ties."""
    if not finite_number(level) or not 0 < level < 1 or not errors:
        raise ValueError("calibration needs events and a level in (0, 1)")
    if any(
        not values or any(not finite_number(v) for v in values)
        for values in errors.values()
    ):
        raise ValueError("every event needs finite residuals")
    maxima = sorted(max(abs(v) for v in values) for values in errors.values())
    rank = int(
        (Decimal(len(maxima) + 1) * Decimal(str(level))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if rank > len(maxima):
        raise ValueError(
            "insufficient independent events for a finite calibrated radius"
        )
    return {
        "radius": maxima[rank - 1],
        "level": level,
        "event_count": len(maxima),
        "order_statistic": rank,
        "method": "absolute residual maximum per event; ceil((n+1)*level) order statistic",
    }


def _contract(task: dict, entity: dict) -> tuple[str, dict]:
    spec = TargetSpec.from_task(task, entity)
    cutoff, resolution = calendar_date(spec.cutoff), calendar_date(spec.resolution)
    target = task["target"]
    declared_unit = target.get("unit") or target.get("units") or entity.get("unit")
    if (
        not isinstance(task.get("family"), str)
        or not task["family"].strip()
        or not isinstance(declared_unit, str)
        or not declared_unit.strip()
        or not spec.unit
        or resolution <= cutoff
    ):
        raise ValueError(
            "calibration needs an explicit family, quantity unit and future horizon"
        )
    contract = {
        "family": task["family"],
        "declared_target": copy.deepcopy(task["target"]),
        "declared_entity_unit": entity.get("unit"),
        "declared_currency": entity.get("currency"),
        "name": spec.name,
        "kind": spec.kind,
        "unit": spec.unit,
        "mode": spec.mode,
        "labels": list(spec.labels),
        "level": spec.level,
        "minimum": spec.lower,
        "maximum": spec.upper,
        "horizon_days": (resolution - cutoff).days,
        "roster_size": len(task["entities"]),
    }
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return key, contract


def _event_windows(cases: list[Case]) -> dict[str, tuple]:
    windows = {}
    for case in cases:
        task = json.loads((case.unit_dir / "task.json").read_text())
        window = (
            calendar_date(task["cutoff_date"]),
            calendar_date(task["resolution_date"]),
        )
        if window[1] <= window[0]:
            raise ValueError("events need positive future horizons")
        if case.group in windows and windows[case.group] != window:
            raise ValueError("views of an event must have identical windows")
        windows[case.group] = window
    ordered = sorted(windows.values())
    if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("independent event windows overlap or are duplicated")
    return windows


def validate_partitions(fit: list[Case], apply: list[Case]) -> None:
    if (
        not fit
        or not apply
        or any(c.split != "train" for c in fit)
        or any(c.split != "calibration" for c in apply)
    ):
        raise ValueError(
            "fit must use train and apply must use calibration; test is forbidden"
        )
    left, right = _event_windows(fit), _event_windows(apply)
    if set(left) & set(right) or max(w[1] for w in left.values()) >= min(
        w[0] for w in right.values()
    ):
        raise ValueError("all fitting outcomes must precede every application cutoff")


def _load_report(
    path: Path, cases: list[Case], *, allow_model: bool = False
) -> tuple[dict, dict]:
    report = json.loads(path.read_text())
    if (
        report.get("mode")
        not in (("grounded", "model") if allow_model else ("grounded",))
        or report.get("profile") != "smoke"
    ):
        raise ValueError(
            "calibration needs grounded local smoke reports or verified model runtime records"
        )
    settings = report["provenance"].get("prediction_settings")
    if (
        not isinstance(settings, dict)
        or settings.get("mode") != report["mode"]
        or not isinstance(settings.get("top_k"), int)
        or isinstance(settings["top_k"], bool)
        or not 1 <= settings["top_k"] <= 30
    ):
        raise ValueError("report lacks frozen prediction settings; regenerate it")
    rows = {}
    for row in report["runs"]:
        key = row["case_id"], row["seed"]
        if (
            key in rows
            or row["status"] != "completed"
            or row["execution"]["returncode"] != 0
            or row["execution"]["isolation"] != "local-process"
            or row["profile"] != "smoke"
            or not row["assessment"]["admissible"]
            or not isinstance(row["seed"], int)
            or isinstance(row["seed"], bool)
            or not 0 <= row["seed"] < 2**32
        ):
            raise ValueError("calibration refuses duplicate, failed or nonlocal runs")
        rows[key] = row
    seeds = {seed for _, seed in rows}
    if not seeds or set(rows) != {(c.case_id, seed) for c in cases for seed in seeds}:
        raise ValueError("report does not cover the complete manifest/seed roster")
    return report, rows


def _checked_answer(case: Case, row: dict, report_path: Path) -> tuple[dict, Path]:
    from baselines.strong_rag_baseline.indexer import build_index
    from baselines.strong_rag_baseline.validation import validate_answer

    task = json.loads((case.unit_dir / "task.json").read_text())
    if (row["split"], row["group"], row["target_type"]) != (
        case.split,
        case.group,
        task["target"]["type"],
    ):
        raise ValueError("report partition or target disagrees with its manifest")
    with tempfile.TemporaryDirectory(prefix="t4-calibration-check-") as temp:
        digest = stage_inputs(case.unit_dir, Path(temp))
    if digest != row["input_digest"]:
        raise ValueError("prediction input digest changed")
    folder = report_path.parent / case.case_id / f"seed-{row['seed']}"
    payload = (folder / "answer.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["answer_sha256"]:
        raise ValueError("original answer digest changed")
    answer = json.loads(payload)
    validate_answer(task, answer, build_index(case.unit_dir / "corpus"))
    diagnostics = row.get("diagnostics", {})
    if diagnostics.get("mode") == "model":
        entities = diagnostics.get("entities", [])
        ids = [entity.get("entity_id") for entity in entities]
        if (
            len(ids) != len(task["entities"])
            or set(ids) != {entity["entity_id"] for entity in task["entities"]}
            or any(
                entity.get("source") != "model"
                or entity.get("fallback_reason") is not None
                for entity in entities
            )
        ):
            raise ValueError(
                "model calibration refuses missing or fallback entity predictions"
            )
    return answer, folder


def _truth(case: Case, row: dict) -> dict:
    if case.truth_path is None:
        raise ValueError("calibration assessment requires separate realized outcomes")
    payload = case.truth_path.read_text()
    if hashlib.sha256(payload.encode()).hexdigest() != row["truth_digest"]:
        raise ValueError("realized outcome digest changed")
    return json.loads(payload)


def _intervals(answer: dict, task: dict, bins: dict) -> dict:
    result = copy.deepcopy(answer)
    entities = {e["entity_id"]: e for e in task["entities"]}
    for prediction in result["entity_predictions"]:
        entity = entities[prediction["entity_id"]]
        spec = TargetSpec.from_task(task, entity)
        key, _ = _contract(task, entity)
        if key not in bins:
            raise ValueError("no fitted bin for the application quantity/horizon")
        point = prediction.get("point_forecast")
        if not finite_number(point):
            raise ValueError("calibration requires a finite point for every entity")
        radius = bins[key]["radius"]
        lo, hi = point - radius, point + radius
        if not finite_number(lo) or not finite_number(hi):
            raise ValueError("calibrated interval overflow")
        if spec.lower is not None:
            lo = max(lo, spec.lower)
        if spec.upper is not None:
            hi = min(hi, spec.upper)
        prediction["interval"] = {"level": spec.level, "lo": lo, "hi": hi}
        spec.validate_prediction(prediction)
    notes = result.setdefault("notes", {})
    notes["interval_calibration"] = (
        "Intervals supersede the original interval assumptions. "
        "They use earlier training-event residual maxima; see the external calibration artifact. "
        "Points, labels, ranks and source claims are unchanged. Historical citations do not "
        "establish these future bounds; production NLI has not been measured."
    )
    return result


def calibrate(
    fit_manifest: Path,
    fit_report: Path,
    apply_manifest: Path,
    apply_report: Path,
    out: Path,
    *,
    fit_runtime: Path | None = None,
    apply_runtime: Path | None = None,
) -> dict:
    for path in (fit_manifest, fit_report, apply_manifest, apply_report, out):
        require_external(path, public_roots())
    if out.exists():
        raise ValueError("calibration output must be new")
    fit = load_cases(units=None, manifest=fit_manifest)
    apply = load_cases(units=None, manifest=apply_manifest)
    validate_partitions(fit, apply)
    before, fit_rows = _load_report(
        fit_report, fit, allow_model=fit_runtime is not None
    )
    original, apply_rows = _load_report(
        apply_report, apply, allow_model=apply_runtime is not None
    )
    if before["mode"] != original["mode"]:
        raise ValueError("fitting and application prediction modes differ")
    if before["mode"] == "model":
        from .runtime import checked_runtime

        if fit_runtime is None or apply_runtime is None:
            raise ValueError("model calibration requires both local runtime records")
        for report, path, manifest, runtime in (
            (before, fit_report, fit_manifest, fit_runtime),
            (original, apply_report, apply_manifest, apply_runtime),
        ):
            report["provenance"]["model_runtime_identity"] = checked_runtime(
                runtime, path, manifest, report
            )
            if any(
                row.get("diagnostics", {}).get("mode") != "model"
                for row in report["runs"]
            ):
                raise ValueError(
                    "model calibration requires per-entity model diagnostics"
                )
        if (
            before["provenance"]["model_runtime_identity"]
            != original["provenance"]["model_runtime_identity"]
        ):
            raise ValueError("fitting and application model runtime identities differ")
    elif fit_runtime is not None or apply_runtime is not None:
        raise ValueError("grounded calibration does not accept model runtime records")
    from .__main__ import provenance

    current = provenance()
    for field in ("source_digest", "toolkit_source_digest", "prediction_settings"):
        if before["provenance"].get(field) != original["provenance"].get(
            field
        ) or not before["provenance"].get(field):
            raise ValueError(f"fitting and application {field} differ or are absent")
        if (
            field != "prediction_settings"
            and current[field] != before["provenance"][field]
        ):
            raise ValueError(f"current {field} differs from the recorded run")
    seeds = {seed for _, seed in fit_rows}
    if seeds != {seed for _, seed in apply_rows}:
        raise ValueError("fitting and application seeds differ")
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    contracts = {}
    for case in fit:
        task = json.loads((case.unit_dir / "task.json").read_text())
        for seed in sorted(seeds):
            row = fit_rows[(case.case_id, seed)]
            _, folder = _checked_answer(case, row, fit_report)
            assessment = assess_unit(case.unit_dir, folder, realized=_truth(case, row))
            if not assessment["admissible"] or assessment["numeric_errors"] is None:
                raise ValueError("fitting run lacks admissible numeric residuals")
            residuals = assessment["numeric_errors"]["entity_errors"]
            for entity in task["entities"]:
                key, contract = _contract(task, entity)
                contracts[key] = contract
                errors[key][case.group].append(residuals[entity["entity_id"]])
    bins = {
        key: {
            **block_radius(groups, contracts[key]["level"]),
            "contract": contracts[key],
        }
        for key, groups in sorted(errors.items())
    }
    artifact = {
        "version": 1,
        "bins": bins,
        "fit_report_sha256": hashlib.sha256(fit_report.read_bytes()).hexdigest(),
        "fit_manifest_sha256": hashlib.sha256(fit_manifest.read_bytes()).hexdigest(),
        "fit_groups": sorted({c.group for c in fit}),
        "prediction_provenance": before["provenance"],
        "runtime_record_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("fit", fit_runtime), ("apply", apply_runtime))
            if path is not None
        },
        "calibrator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Event-block coverage relies on exchangeable event residuals within each declared family/quantity/horizon. Temporal ordering prevents outcome leakage but does not prove exchangeability. Not an official NLI or leaderboard result.",
    }
    pending = []
    for case in apply:
        task = json.loads((case.unit_dir / "task.json").read_text())
        for seed in sorted(seeds):
            row = apply_rows[(case.case_id, seed)]
            answer, _ = _checked_answer(case, row, apply_report)
            pending.append((case, row, answer, _intervals(answer, task, bins)))
    # Freeze every interval before opening any application outcome.
    out.mkdir(parents=True)
    write_json(out / "calibration.json", artifact)
    for case, row, _, answer in pending:
        folder = out / case.case_id / f"seed-{row['seed']}"
        folder.mkdir(parents=True)
        write_json(folder / "answer.json", answer)
    rows = []
    for case, row, before_answer, answer in pending:
        folder = out / case.case_id / f"seed-{row['seed']}"
        assessment = assess_unit(case.unit_dir, folder, realized=_truth(case, row))
        updated = copy.deepcopy(row)
        updated.update(
            assessment=assessment,
            answer_sha256=hashlib.sha256(
                (folder / "answer.json").read_bytes()
            ).hexdigest(),
            interval_diagnostics={
                "before_mean_width": mean(
                    p["interval"]["hi"] - p["interval"]["lo"]
                    for p in before_answer["entity_predictions"]
                ),
                "after_mean_width": mean(
                    p["interval"]["hi"] - p["interval"]["lo"]
                    for p in answer["entity_predictions"]
                ),
                "original_answer_sha256": row["answer_sha256"],
            },
        )
        write_json(folder / "result.json", updated)
        rows.append(updated)
    report = copy.deepcopy(original)
    report.update(
        runs=rows, summary=summarize(rows), by_split={"calibration": summarize(rows)}
    )
    report["provenance"]["interval_calibration_sha256"] = hashlib.sha256(
        (out / "calibration.json").read_bytes()
    ).hexdigest()
    write_json(out / "report.json", report)
    (out / "report.md").write_text(markdown(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fit-manifest", "fit-report", "apply-manifest", "apply-report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fit-runtime", type=Path)
    parser.add_argument("--apply-runtime", type=Path)
    args = parser.parse_args(argv)
    try:
        report = calibrate(
            args.fit_manifest,
            args.fit_report,
            args.apply_manifest,
            args.apply_report,
            args.out,
            fit_runtime=args.fit_runtime,
            apply_runtime=args.apply_runtime,
        )
        print(json.dumps(report["summary"], indent=2))
        return (
            0
            if report["summary"]["admissible_runs"] == report["summary"]["runs"]
            else 1
        )
    except Exception as exc:
        print(f"Calibration aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
