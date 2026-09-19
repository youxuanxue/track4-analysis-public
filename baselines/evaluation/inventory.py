"""Audit candidate evaluation inventory for disjointness and coverage."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from .dataset import load_cases, stage_inputs

DEFAULT_MINIMUMS = {
    "min_domains": 3,
    "min_groups_per_domain": 2,
    "min_groups_per_target_type": 2,
}


def dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _roster_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("expected_runs"), list):
        return [dict(row) for row in data["expected_runs"]]
    cases = load_cases(units=None, manifest=path)
    rows = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="t4-inventory-input-") as directory:
            input_digest = stage_inputs(case.unit_dir, Path(directory) / "input")
        task = json.loads((case.unit_dir / "task.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "case_id": case.case_id,
                "group": case.group,
                "domain": case.domain,
                "target_type": task["target"]["type"],
                "input_digest": input_digest,
            }
        )
    return rows


def _validate_rows(rows: list[dict], source: Path) -> list[dict]:
    required = {"case_id", "group", "target_type", "input_digest"}
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"{source}: inventory row lacks required fields")
        if not isinstance(row["group"], str) or not row["group"]:
            raise ValueError(f"{source}: group must be a nonempty string")
        if not isinstance(row["target_type"], str) or not row["target_type"]:
            raise ValueError(f"{source}: target_type must be a nonempty string")
        if not isinstance(row["input_digest"], str) or not row["input_digest"]:
            raise ValueError(f"{source}: input_digest must be a nonempty string")
        normalized.append(
            {
                "case_id": str(row["case_id"]),
                "group": row["group"],
                "domain": row.get("domain"),
                "target_type": row["target_type"],
                "input_digest": row["input_digest"],
            }
        )
    if not normalized:
        raise ValueError(f"{source}: inventory is empty")
    return normalized


def audit(candidate: Path, consumed: list[Path], *, minimums: dict | None = None) -> dict:
    minimums = {**DEFAULT_MINIMUMS, **(minimums or {})}
    candidate_rows = _validate_rows(_roster_rows(candidate), candidate)
    consumed_rows = []
    for path in consumed:
        consumed_rows.extend(_validate_rows(_roster_rows(path), path))

    candidate_groups = {row["group"] for row in candidate_rows}
    consumed_groups = {row["group"] for row in consumed_rows}
    candidate_inputs = {row["input_digest"] for row in candidate_rows}
    consumed_inputs = {row["input_digest"] for row in consumed_rows}
    domains: dict[str, set[str]] = defaultdict(set)
    target_types: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        if row["domain"] is not None:
            domains[row["domain"]].add(row["group"])
        target_types[row["target_type"]].add(row["group"])

    failures = []
    if len(domains) < minimums["min_domains"]:
        failures.append("domains")
    if any(len(groups) < minimums["min_groups_per_domain"] for groups in domains.values()):
        failures.append("groups_per_domain")
    if any(len(groups) < minimums["min_groups_per_target_type"] for groups in target_types.values()):
        failures.append("groups_per_target_type")
    overlap = {
        "groups": sorted(candidate_groups & consumed_groups),
        "input_digests": sorted(candidate_inputs & consumed_inputs),
    }
    eligible = not failures and not overlap["groups"] and not overlap["input_digests"]
    return {
        "schema_version": 1,
        "candidate": str(candidate.resolve()),
        "consumed": sorted(str(path.resolve()) for path in consumed),
        "overlap": overlap,
        "counts": {
            "cases": len(candidate_rows),
            "groups": len(candidate_groups),
            "domains": {key: len(domains[key]) for key in sorted(domains)},
            "target_types": {key: len(target_types[key]) for key in sorted(target_types)},
        },
        "policy": minimums,
        "policy_failures": sorted(set(failures)),
        "eligible": eligible,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--consumed", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--minimum-domains", type=int, default=DEFAULT_MINIMUMS["min_domains"])
    parser.add_argument("--minimum-groups-per-domain", type=int, default=DEFAULT_MINIMUMS["min_groups_per_domain"])
    parser.add_argument("--minimum-groups-per-target-type", type=int, default=DEFAULT_MINIMUMS["min_groups_per_target_type"])
    args = parser.parse_args(argv)
    try:
        minimums = {
            "min_domains": args.minimum_domains,
            "min_groups_per_domain": args.minimum_groups_per_domain,
            "min_groups_per_target_type": args.minimum_groups_per_target_type,
        }
        if any(value < 1 for value in minimums.values()):
            raise ValueError("inventory minimums must be positive")
        result = audit(args.candidate.resolve(), [path.resolve() for path in args.consumed], minimums=minimums)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = {"schema_version": 1, "eligible": False, "error": f"{type(exc).__name__}: {exc}"}
        code = 2
    else:
        code = 0 if result["eligible"] else 1
    payload = dumps(result)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
