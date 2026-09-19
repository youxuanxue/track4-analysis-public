"""Audit candidate evaluation inventory for disjointness and coverage."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from .dataset import load_cases, stage_inputs

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO / "baselines" / "evaluation" / "acceptance-policy.json"
def load_policy(path: Path = POLICY_PATH) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    target_types = data.get("target_types")
    if (
        data.get("min_event_groups") != 60
        or data.get("min_domains") != 3
        or data.get("min_groups_per_stratum") != 20
        or target_types != ["classification", "regression", "ranking"]
    ):
        raise ValueError("acceptance policy does not define the required inventory minima")
    return {
        "min_event_groups": data["min_event_groups"],
        "min_domains": data["min_domains"],
        "min_groups_per_domain": data["min_groups_per_stratum"],
        "min_groups_per_target_type": 20,
        "target_types": list(target_types),
    }


def dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _roster_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "expected_runs" in data:
        raise ValueError(f"{path}: expected_runs is not an accepted inventory input")
    cases = load_cases(units=None, manifest=path)
    rows = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="t4-analysis-inventory-input-") as directory:
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


def _validate_rows(
    rows: list[dict], source: Path, policy: dict, *, reject_duplicate_input_digest: bool = True
) -> list[dict]:
    required = {"case_id", "group", "domain", "target_type", "input_digest"}
    normalized = []
    identities: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    input_digests: set[str] = set()
    groups: dict[str, tuple[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError(f"{source}: inventory row lacks required fields")
        if not isinstance(row["case_id"], str) or not row["case_id"]:
            raise ValueError(f"{source}: case_id must be a nonempty string")
        if not isinstance(row["group"], str) or not row["group"]:
            raise ValueError(f"{source}: group must be a nonempty string")
        if not isinstance(row["domain"], str) or not row["domain"]:
            raise ValueError(f"{source}: domain must be a nonempty string")
        if row["target_type"] not in policy["target_types"]:
            raise ValueError(f"{source}: unknown target_type")
        if not isinstance(row["input_digest"], str) or not row["input_digest"]:
            raise ValueError(f"{source}: input_digest must be a nonempty string")
        identity = (row["case_id"], row["group"])
        if identity in identities or row["case_id"] in case_ids:
            raise ValueError(f"{source}: duplicate case/group identity")
        identities.add(identity)
        case_ids.add(row["case_id"])
        if reject_duplicate_input_digest and row["input_digest"] in input_digests:
            raise ValueError(f"{source}: duplicate input_digest")
        input_digests.add(row["input_digest"])
        mapping = (row["domain"], row["target_type"])
        if row["group"] in groups and groups[row["group"]] != mapping:
            raise ValueError(f"{source}: group maps inconsistently")
        groups[row["group"]] = mapping
        normalized.append(
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "domain": row["domain"],
                "target_type": row["target_type"],
                "input_digest": row["input_digest"],
            }
        )
    if not normalized:
        raise ValueError(f"{source}: inventory is empty")
    return normalized


def audit(candidate: Path, consumed: list[Path], *, policy: dict | None = None) -> dict:
    policy = load_policy() if policy is None else policy
    candidate_rows = _validate_rows(_roster_rows(candidate), candidate, policy)
    consumed_sets = []
    for path in consumed:
        consumed_sets.append(_validate_rows(_roster_rows(path), path, policy))
    consumed_rows = [row for rows in consumed_sets for row in rows]
    _validate_rows(
        candidate_rows + consumed_rows,
        Path("candidate+consumed"),
        policy,
        reject_duplicate_input_digest=False,
    )

    manifest_rows = [candidate_rows, *consumed_sets]
    group_manifests: dict[str, set[int]] = defaultdict(set)
    input_manifests: dict[str, set[int]] = defaultdict(set)
    for index, rows in enumerate(manifest_rows):
        for row in rows:
            group_manifests[row["group"]].add(index)
            input_manifests[row["input_digest"]].add(index)
    overlap = {
        "groups": sorted(key for key, owners in group_manifests.items() if len(owners) > 1),
        "input_digests": sorted(key for key, owners in input_manifests.items() if len(owners) > 1),
    }
    candidate_groups = {row["group"] for row in candidate_rows}
    domains: dict[str, set[str]] = defaultdict(set)
    target_types: dict[str, set[str]] = defaultdict(set)
    for row in candidate_rows:
        if row["domain"] is not None:
            domains[row["domain"]].add(row["group"])
        target_types[row["target_type"]].add(row["group"])

    failures = []
    if len(candidate_groups) < policy["min_event_groups"]:
        failures.append("event_groups")
    if len(domains) < policy["min_domains"]:
        failures.append("domains")
    if any(len(groups) < policy["min_groups_per_domain"] for groups in domains.values()):
        failures.append("groups_per_domain")
    if set(target_types) != set(policy["target_types"]):
        failures.append("target_types")
    if any(len(target_types.get(kind, set())) < policy["min_groups_per_target_type"] for kind in policy["target_types"]):
        failures.append("groups_per_target_type")
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
        "policy": policy,
        "policy_failures": sorted(set(failures)),
        "eligible": eligible,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--consumed", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(args.candidate.resolve(), [path.resolve() for path in args.consumed])
        payload = dumps(result)
        if args.out:
            args.out.write_text(payload, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = {"schema_version": 1, "eligible": False, "error": f"{type(exc).__name__}: {exc}"}
        payload = dumps(result)
        code = 2
    else:
        code = 0 if result["eligible"] else 1
    print(payload, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
