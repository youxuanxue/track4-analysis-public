"""Apply verified G2 decisions to a local development incumbent and rollback chain.

The append-only journal lives beside the batch registry, outside public worktrees.
It does not grant production qualification, deploy images or submit to the contest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import batch
from .acceptance import digest, load_policy
from .artifacts import read_json_inside
from .dataset import public_roots, require_external


def _journal(root: Path) -> Path:
    require_external(root, public_roots())
    directory = root / "lifecycle"
    require_external(directory, public_roots())
    if directory.is_symlink():
        raise ValueError("lifecycle journal cannot be a symlink")
    return directory


def _apply(state: dict, action: str, data: dict) -> None:
    if action == "initialize":
        if state:
            raise ValueError("lifecycle already initialized")
        state.update(
            incumbent={
                **data["version"],
                "qualification": "bootstrap",
                "batch_id": None,
            },
            rollback=[],
            pending=None,
            consumed=[],
            production_candidate=None,
        )
    elif not state:
        raise ValueError("initialize the lifecycle first")
    elif action == "attach":
        if state["pending"] is not None or data["batch_id"] in state["consumed"]:
            raise ValueError("a batch is pending or has already been resolved")
        before = data["versions"]["before"]
        if before["identity"] != state["incumbent"]["identity"]:
            raise ValueError("batch baseline is not the current incumbent")
        if before["identity"] == data["versions"]["after"]["identity"]:
            raise ValueError("candidate must differ from incumbent")
        state["pending"] = data
    elif action == "resolve":
        pending = state["pending"]
        if pending is None or pending["batch_id"] != data["batch_id"]:
            raise ValueError("decision does not belong to the pending batch")
        promote = all(data["goals"][key] == "PASS" for key in ("G1", "G2"))
        if data["decision"] != ("PROMOTE_DEVELOPMENT" if promote else "KEEP_INCUMBENT"):
            raise ValueError("transition disagrees with acceptance gates")
        if promote:
            state["rollback"].append(state["incumbent"])
            state["incumbent"] = {
                **pending["versions"]["after"],
                "qualification": "G2",
                "batch_id": data["batch_id"],
            }
        state["consumed"].append(data["batch_id"])
        state["pending"] = None
    elif action == "rollback":
        if state["pending"] is not None:
            raise ValueError("resolve the pending batch before rollback")
        if not state["rollback"]:
            raise ValueError("no retained incumbent to restore")
        if state["rollback"][-1] != data["restore"]:
            raise ValueError("rollback must restore the previous retained incumbent")
        state["incumbent"] = state["rollback"].pop()
    elif action == "abandon":
        if state["pending"] is None or state["pending"]["batch_id"] != data["batch_id"]:
            raise ValueError("no matching pending batch")
        state["consumed"].append(data["batch_id"])
        state["pending"] = None
    else:
        raise ValueError("unknown lifecycle action")


def _read(root: Path) -> tuple[dict, list[dict]]:
    directory = _journal(root)
    state: dict = {}
    events = []
    previous = None
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        if path.name != f"{index:08d}.json":
            raise ValueError("lifecycle journal has a missing or unexpected record")
        envelope = read_json_inside(directory, path.name)
        event = envelope["event"]
        if envelope["digest"] != digest(event) or event["previous"] != previous:
            raise ValueError("lifecycle journal hash chain changed")
        if event["version"] != 1 or event["sequence"] != index:
            raise ValueError("unsupported lifecycle record")
        _apply(state, event["action"], event["data"])
        previous = envelope["digest"]
        events.append(envelope)
    return state, events


def _append(root: Path, action: str, data: dict) -> dict:
    state, events = _read(root)
    _apply(state, action, data)
    event = {
        "version": 1,
        "sequence": len(events),
        "previous": events[-1]["digest"] if events else None,
        "action": action,
        "data": data,
    }
    directory = _journal(root)
    directory.mkdir(exist_ok=True)
    batch.write_new(
        directory / f"{len(events):08d}.json", {"event": event, "digest": digest(event)}
    )
    return state


def status(root: Path) -> dict:
    with batch.locked(root):
        state, events = _read(root)
        if not events:
            raise ValueError("initialize the lifecycle first")
        return {**state, "journal_head": events[-1]["digest"], "events": len(events)}


def _frozen(spec: dict) -> dict:
    identity = batch.snapshot(spec)
    if identity.get("git_dirty") is not False:
        raise ValueError("lifecycle versions require clean immutable checkouts")
    return {"spec": spec, "identity": identity}


def initialize(root: Path, spec: dict) -> dict:
    with batch.locked(root):
        return _append(root, "initialize", {"version": _frozen(spec)})


def attach(root: Path, directory: Path) -> dict:
    """Bind selection before either side exposes acceptance results."""
    if directory.resolve().parent.parent != root.resolve():
        raise ValueError("batch must belong to this registry")
    with batch.locked(root):
        record = batch.read_registration(directory)
        if any(
            (directory / name).exists()
            for name in ("before.started.json", "after.started.json", "decision.json")
        ):
            raise ValueError("select the candidate before either acceptance run starts")
        for frozen in record["versions"].values():
            if _frozen(frozen["spec"]) != frozen:
                raise ValueError("version changed since registration")
        return _append(
            root,
            "attach",
            {
                "batch_id": directory.name,
                "versions": record["versions"],
                "policy_digest": record["policy_digest"],
            },
        )


def resolve(root: Path) -> dict:
    with batch.locked(root):
        state, _ = _read(root)
        pending = state.get("pending")
        if pending is None:
            raise ValueError("no pending batch")
        if pending["policy_digest"] != digest(load_policy()):
            raise ValueError("policy changed since selection")
        directory = root / "batches" / pending["batch_id"]
        record = batch.read_registration(directory)
        if record["versions"] != pending["versions"]:
            raise ValueError("registered pair changed since selection")
        decision = batch.verified_decision(directory)
        for frozen in pending["versions"].values():
            if _frozen(frozen["spec"]) != frozen:
                raise ValueError("version changed before lifecycle transition")
        goals = {key: value["status"] for key, value in decision["goals"].items()}
        promote = all(goals[key] == "PASS" for key in ("G1", "G2"))
        return _append(
            root,
            "resolve",
            {
                "batch_id": directory.name,
                "decision_digest": digest(decision),
                "goals": goals,
                "decision": "PROMOTE_DEVELOPMENT" if promote else "KEEP_INCUMBENT",
            },
        )


def rollback(root: Path, reason: str) -> dict:
    if not reason.strip():
        raise ValueError("record the rollback reason")
    with batch.locked(root):
        state, _ = _read(root)
        if not state.get("rollback"):
            raise ValueError("no retained incumbent to restore")
        restore = state["rollback"][-1]
        if _frozen(restore["spec"])["identity"] != restore["identity"]:
            raise ValueError("retained rollback version is no longer reproducible")
        return _append(root, "rollback", {"restore": restore, "reason": reason})


def abandon(root: Path, reason: str) -> dict:
    """Close an interrupted batch without retrying it or releasing its events."""
    if not reason.strip():
        raise ValueError("record the abandonment reason")
    with batch.locked(root):
        state, _ = _read(root)
        if not state.get("pending"):
            raise ValueError("no pending batch")
        return _append(
            root,
            "abandon",
            {"batch_id": state["pending"]["batch_id"], "reason": reason},
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    init.add_argument("--version", type=Path, required=True)
    select = commands.add_parser("attach")
    select.add_argument("--batch", type=Path, required=True)
    commands.add_parser("resolve")
    commands.add_parser("status")
    for name in ("rollback", "abandon"):
        commands.add_parser(name).add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.registry.resolve()
        if args.command == "initialize":
            require_external(args.version, public_roots())
            result = initialize(root, json.loads(args.version.read_text()))
        elif args.command == "attach":
            result = attach(root, args.batch.resolve())
        elif args.command == "resolve":
            result = resolve(root)
        elif args.command == "rollback":
            result = rollback(root, args.reason)
        elif args.command == "abandon":
            result = abandon(root, args.reason)
        else:
            result = status(root)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Lifecycle refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
