"""Verify local launcher records before borrowing model residuals across runs.

This binds local artifacts and configuration; it is not remote-service attestation.
The launcher owns the server and records report hashes only after evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .dataset import public_roots, require_external


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checked_runtime(
    path: Path, report_path: Path, manifest: Path, report: dict
) -> dict:
    require_external(path, public_roots())
    runtime = json.loads(path.read_text())
    relative = report_path.resolve().relative_to(path.resolve().parent).as_posix()
    if (
        runtime.get("complete") is not True
        or runtime.get("report_sha256", {}).get(relative) != file_digest(report_path)
        or runtime.get("manifest_sha256") != file_digest(manifest)
        or report["provenance"].get("git_dirty") is not False
        or runtime.get("repo_head") != report["provenance"].get("git_commit")
        or runtime.get("repo_diff_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("local runtime does not bind this clean report and manifest")
    codes = runtime.get("evaluation_exit_codes")
    if (
        not isinstance(codes, list)
        or not codes
        or any(
            not isinstance(code, int) or isinstance(code, bool) or code != 0
            for code in codes
        )
    ):
        raise ValueError("local runtime has incomplete or failed evaluations")
    command = runtime.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(arg, str) for arg in command)
    ):
        raise ValueError("local runtime lacks its launch command")
    # Only an ephemeral loopback port may differ. All model and runtime flags stay pinned.
    command = list(command)
    if command.count("--port") != 1 or command.count("--host") != 1:
        raise ValueError("local runtime requires one explicit loopback host and port")
    try:
        host = command[command.index("--host") + 1]
        port_index = command.index("--port") + 1
        if host != "127.0.0.1" or not 1 <= int(command[port_index]) <= 65535:
            raise ValueError("invalid loopback binding")
        command[port_index] = "<ephemeral-port>"
    except (IndexError, ValueError) as exc:
        raise ValueError("invalid local runtime binding") from exc
    binary_hash = runtime.get("runtime_binary_sha256")
    if not _sha256(binary_hash) or file_digest(Path(command[0])) != binary_hash:
        raise ValueError("local runtime binary digest changed or is absent")
    weights = runtime.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ValueError("local runtime lacks weight artifacts")
    paths = []
    for weight in weights:
        if (
            not isinstance(weight, dict)
            or not isinstance(weight.get("path"), str)
            or not _sha256(weight.get("sha256"))
        ):
            raise ValueError("invalid local weight identity")
        weight_path = Path(weight["path"])
        if (
            not isinstance(weight.get("bytes"), int)
            or isinstance(weight["bytes"], bool)
            or weight["bytes"] <= 0
            or weight_path.stat().st_size != weight["bytes"]
            or file_digest(weight_path) != weight["sha256"]
        ):
            raise ValueError("local model weight digest/size changed")
        paths.append(str(weight_path.resolve()))
    if len(set(paths)) != len(paths) or command.count("-m") != 1:
        raise ValueError("local runtime needs unique shards and one model file")
    try:
        if str(Path(command[command.index("-m") + 1]).resolve()) != paths[0]:
            raise ValueError("launch model differs from the recorded first shard")
    except IndexError as exc:
        raise ValueError("local runtime lacks its model argument") from exc
    settings = runtime.get("evaluation_settings")
    required = {
        "T4_MAX_TOKENS",
        "T4_MODEL_TIMEOUT_S",
        "T4_MODEL_RETRIES",
        "T4_UNIT_TIMEOUT_S",
        "T4_TOP_K",
        "T4_TEMPERATURE",
    }
    if (
        not isinstance(settings, dict)
        or set(settings) != required
        or not all(isinstance(v, str) and v for v in settings.values())
    ):
        raise ValueError("local runtime lacks complete prediction settings")
    if settings["T4_TOP_K"] != str(
        report["provenance"]["prediction_settings"]["top_k"]
    ):
        raise ValueError("local runtime retrieval settings disagree with report")
    for key in ("version", "backend"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            raise ValueError(f"local runtime lacks {key}")
    if (
        not _sha256(runtime.get("runner_sha256"))
        or not isinstance(runtime.get("host"), dict)
        or not runtime["host"]
    ):
        raise ValueError("local runtime lacks launcher/host identity")
    return {
        key: command if key == "command" else runtime[key]
        for key in (
            "command",
            "version",
            "runtime_binary_sha256",
            "weights",
            "runner_sha256",
            "host",
            "backend",
            "evaluation_settings",
        )
    }
