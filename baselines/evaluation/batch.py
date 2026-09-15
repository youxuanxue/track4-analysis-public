"""Pre-register an external evaluation batch and consume its events before running.

The registry is a local evaluator-owned audit trail, not an anti-tamper service.
It prevents accidental reuse and post-hoc report substitution. It cannot certify
that a human has never inspected data or that event identifiers are truthful.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .acceptance import digest, load_policy, load_report
from .dataset import load_cases, public_roots, require_external, stage_inputs


@contextmanager
def locked(root: Path):
    require_external(root, public_roots())
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def write_new(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def snapshot(spec: dict) -> dict:
    if set(spec) != {"repo", "python", "image"}:
        raise ValueError("version spec requires exactly repo, python and image")
    repo = Path(spec["repo"]).resolve()
    if not repo.is_dir() or not Path(spec["python"]).is_file():
        raise ValueError("version repo and Python executable must exist")
    # This executes only evaluator-owned local source from the declared checkout.
    code = (
        "from baselines.evaluation.__main__ import provenance; "
        "from baselines.evaluation.engineering import image_identity; "
        "import json, sys; identity = provenance(); "
        "identity.update(image_identity(sys.argv[1]) if len(sys.argv) > 1 else {}); "
        "print(json.dumps(identity))"
    )
    result = subprocess.run(
        [spec["python"], "-c", code, *([spec["image"]] if spec["image"] else [])],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(result.stdout)


def verify_faults(record: dict) -> dict[str, Path]:
    """Recheck the originally registered recovery artifacts, including their runtime."""
    from .faults import load_suite

    frozen = record.get("faults", {})
    if frozen and set(frozen) != {"before", "after"}:
        raise ValueError("register recovery evidence for both versions")
    paths = {}
    for role, artifact in frozen.items():
        path = Path(artifact["path"])
        require_external(path, public_roots())
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError("fault report changed since registration")
        load_suite(path, record["versions"][role]["identity"])
        paths[role] = path
    return paths


def register(
    root: Path,
    manifest: Path,
    versions: dict,
    seeds: list[int],
    *,
    faults: dict[str, Path] | None = None,
) -> Path:
    require_external(manifest, public_roots())
    if set(versions) != {"before", "after"}:
        raise ValueError("freeze exactly before and after versions")
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(
            (not isinstance(s, int) or isinstance(s, bool)) or not 0 <= s < 2**32
            for s in seeds
        )
    ):
        raise ValueError("distinct unsigned integer seeds required")
    cases = load_cases(units=None, manifest=manifest)
    if any(
        case.split != "test" or case.truth_path is None or not case.domain
        for case in cases
    ):
        raise ValueError(
            "sealed batches require test-only cases with truth and domains"
        )
    expected = []
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="t4-seal-input-") as directory:
            input_digest = stage_inputs(case.unit_dir, Path(directory) / "inputs")
        truth_digest = hashlib.sha256(case.truth_path.read_bytes()).hexdigest()
        kind = json.loads((case.unit_dir / "task.json").read_text())["target"]["type"]
        for seed in seeds:
            expected.append(
                {
                    "case_id": case.case_id,
                    "seed": seed,
                    "group": case.group,
                    "domain": case.domain,
                    "split": "test",
                    "target_type": kind,
                    "input_digest": input_digest,
                    "truth_digest": truth_digest,
                }
            )
    frozen = {
        key: {"spec": spec, "identity": snapshot(spec)}
        for key, spec in versions.items()
    }
    if (
        frozen["before"]["identity"]["toolkit_source_digest"]
        != frozen["after"]["identity"]["toolkit_source_digest"]
    ):
        raise ValueError("both versions need the same toolkit bytes")
    record = {
        "version": 1,
        "policy_digest": digest(load_policy()),
        "manifest": str(manifest.resolve()),
        "manifest_digest": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "versions": frozen,
        "seeds": seeds,
        "expected_runs": expected,
        "mode": "grounded",
        "profile": "smoke",
        "budget": {
            "model_request_attempts": 0,
            "timeout_s": 480,
            "runs": 2 * len(expected),
        },
        "scope": "offline preregistration; does not certify unseen data, production runtime or artifact eligibility",
    }
    if faults is not None:
        if set(faults) != {"before", "after"}:
            raise ValueError("register recovery evidence for both versions")
        record["faults"] = {}
        for role, path in faults.items():
            require_external(path, public_roots())
            record["faults"][role] = {
                "path": str(path.absolute()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        verify_faults(record)
    batch_id = digest(record)
    with locked(root):
        events = root / "events"
        events.mkdir(exist_ok=True)
        for row in expected:
            for key in ("group", "input_digest"):
                marker = events / digest({key: row[key]})
                if marker.exists():
                    raise ValueError(
                        "event or input has already been reserved/consumed by a sealed batch"
                    )
        directory = root / "batches" / batch_id
        directory.mkdir(parents=True, exist_ok=False)
        write_new(directory / "registration.json", record)
        for key in ("group", "input_digest"):
            for value in {row[key] for row in expected}:
                write_new(events / digest({key: value}), {"batch_id": batch_id})
    return directory


def read_registration(directory: Path) -> dict:
    require_external(directory, public_roots())
    record = json.loads((directory / "registration.json").read_text())
    if digest(record) != directory.name:
        raise ValueError("registration was modified after freezing")
    if record["policy_digest"] != digest(load_policy()):
        raise ValueError("acceptance policy changed since registration")
    verify_faults(record)
    events = directory.parent.parent / "events"
    for key in ("group", "input_digest"):
        for value in {row[key] for row in record["expected_runs"]}:
            marker = json.loads((events / digest({key: value})).read_text())
            if marker.get("batch_id") != directory.name:
                raise ValueError(
                    "event reservation is missing or belongs to another batch"
                )
    return record


def validate_report(report: dict, record: dict, role: str) -> None:
    actual = {(r["case_id"], r["seed"]): r for r in report["runs"]}
    expected = {(r["case_id"], r["seed"]): r for r in record["expected_runs"]}
    if len(actual) != len(report["runs"]) or set(actual) != set(expected):
        raise ValueError("report does not cover the complete registered roster")
    for key, frozen in expected.items():
        if any(actual[key].get(field) != value for field, value in frozen.items()):
            raise ValueError(
                "report inputs, truth or grouping differ from registration"
            )
    identity = record["versions"][role]["identity"]
    for field in (
        "source_digest",
        "git_commit",
        "toolkit_source_digest",
        "image_id",
        *(key for key in ("runtime_digest", "runtime_manifest") if key in identity),
    ):
        if report["provenance"].get(field) != identity.get(field):
            raise ValueError(f"report {field} differs from frozen version")
    if (
        report.get("mode") != record["mode"]
        or report.get("profile") != record["profile"]
    ):
        raise ValueError("report execution mode differs from registration")


def run(directory: Path, role: str) -> Path:
    if role not in {"before", "after"}:
        raise ValueError("role must be before or after")
    root = directory.parent.parent
    with locked(root):
        record = read_registration(directory)
        frozen = record["versions"][role]
        current = snapshot(frozen["spec"])
        if current != frozen["identity"]:
            raise ValueError("version changed since registration")
        manifest = Path(record["manifest"])
        if (
            hashlib.sha256(manifest.read_bytes()).hexdigest()
            != record["manifest_digest"]
        ):
            raise ValueError("manifest changed since registration")
        # Starting burns this role, even if the process crashes. Never retry a
        # partially observed acceptance run under another seed or the same batch.
        write_new(
            directory / f"{role}.started.json", {"registration_digest": directory.name}
        )
    spec = frozen["spec"]
    out = directory / role
    argv = [
        spec["python"],
        "-m",
        "baselines.evaluation",
        "--manifest",
        record["manifest"],
        "--out",
        str(out),
        "--seeds",
        *map(str, record["seeds"]),
        "--mode",
        record["mode"],
        "--profile",
        record["profile"],
        "--timeout",
        str(record["budget"]["timeout_s"]),
    ]
    if spec["image"]:
        argv.extend(["--image", frozen["identity"]["image_id"]])
    log = directory / f"{role}.log"
    with log.open("x") as output:
        completed = subprocess.run(
            argv,
            cwd=spec["repo"],
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=record["budget"]["timeout_s"] * len(record["expected_runs"]) + 120,
        )
    if completed.returncode not in (0, 1):
        raise ValueError(
            "evaluation aborted; retain log and burned batch without publishing a partial verdict"
        )
    report_path = out / "report.json"
    report = load_report(report_path)
    validate_report(report, record, role)
    if snapshot(spec) != frozen["identity"]:
        raise ValueError("version changed during evaluation")
    with locked(root):
        read_registration(directory)
        write_new(
            directory / f"{role}.receipt.json",
            {
                "report_digest": digest(report),
                "registration_digest": directory.name,
                "report_path": str(report_path.resolve()),
                "returncode": completed.returncode,
            },
        )
    return report_path


def verified_reports(directory: Path) -> tuple[dict, dict, dict]:
    record = read_registration(directory)
    reports = []
    for role in ("before", "after"):
        started = json.loads((directory / f"{role}.started.json").read_text())
        receipt = json.loads((directory / f"{role}.receipt.json").read_text())
        report_path = directory / role / "report.json"
        if (
            started.get("registration_digest") != directory.name
            or receipt.get("registration_digest") != directory.name
        ):
            raise ValueError("run is not bound to this registration")
        if receipt["report_path"] != str(report_path.resolve()):
            raise ValueError("receipt points outside its registered run")
        report = load_report(report_path)
        if receipt["report_digest"] != digest(report):
            raise ValueError("report changed after evaluation receipt")
        validate_report(report, record, role)
        reports.append(report)
    return reports[0], reports[1], record


def decide(directory: Path) -> dict:
    from .acceptance import audit, check, stage

    with locked(directory.parent.parent):
        if (directory / "decision.json").exists():
            raise ValueError(
                "batch already consumed; inspect existing decision instead of re-deciding"
            )
        before, after, record = verified_reports(directory)
        faults = verify_faults(record)
        decision = audit(
            before,
            after,
            policy=load_policy(),
            before_faults=faults.get("before"),
            after_faults=faults.get("after"),
        )
        if "input_error" not in decision:
            # Replace only proof obligations actually established by this runner.
            for goal, name, detail in (
                (
                    "G1",
                    "planned_roster",
                    "both reports exactly match the frozen roster and version identities",
                ),
                (
                    "G2",
                    "fresh_holdout",
                    "one-time registered event batch; prior human exposure remains a provenance obligation",
                ),
            ):
                checks = decision["goals"][goal]["checks"]
                checks = [item for item in checks if item["name"] != name]
                checks.append(check(name, True, detail))
                decision["goals"][goal] = stage(checks)
        decision["registration_digest"] = directory.name
        decision["policy_digest"] = record["policy_digest"]
        write_new(directory / "decision.json", decision)
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("register")
    create.add_argument("--registry", type=Path, required=True)
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--versions", type=Path, required=True)
    create.add_argument("--seeds", nargs="+", type=int, required=True)
    create.add_argument("--before-faults", type=Path)
    create.add_argument("--after-faults", type=Path)
    execute = commands.add_parser("run")
    execute.add_argument("--batch", type=Path, required=True)
    execute.add_argument("--role", choices=("before", "after"), required=True)
    decision = commands.add_parser("decide")
    decision.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "register":
            require_external(args.versions, public_roots())
            faults = {
                role: path
                for role, path in (
                    ("before", args.before_faults),
                    ("after", args.after_faults),
                )
                if path is not None
            }
            print(
                register(
                    args.registry.resolve(),
                    args.manifest.resolve(),
                    json.loads(args.versions.read_text()),
                    args.seeds,
                    faults=faults or None,
                )
            )
        elif args.command == "run":
            print(run(args.batch.resolve(), args.role))
        else:
            result = decide(args.batch.resolve())
            print(
                json.dumps(
                    {key: value["status"] for key, value in result["goals"].items()}
                )
            )
            return (
                1 if any(g["status"] == "FAIL" for g in result["goals"].values()) else 2
            )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Batch aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
