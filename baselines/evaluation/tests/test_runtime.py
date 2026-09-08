"""Synthetic local receipts bind reports, model bytes and launch settings."""

import hashlib
import json

import pytest

from baselines.evaluation.runtime import checked_runtime, file_digest


def write_runtime(root, report_path, manifest):
    binary = root / "synthetic-binary"
    weights = root / "synthetic-weights"
    binary.write_bytes(b"invented executable")
    weights.write_bytes(b"invented weight bytes")
    report = json.loads(report_path.read_text())
    value = {
        "complete": True,
        "report_sha256": {"report.json": file_digest(report_path)},
        "manifest_sha256": file_digest(manifest),
        "repo_head": report["provenance"]["git_commit"],
        "repo_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "evaluation_exit_codes": [0],
        "command": [
            str(binary),
            "-m",
            str(weights),
            "--host",
            "127.0.0.1",
            "--port",
            "17171",
            "--ctx-size",
            "16384",
        ],
        "runtime_binary_sha256": file_digest(binary),
        "weights": [
            {
                "path": str(weights),
                "sha256": file_digest(weights),
                "bytes": weights.stat().st_size,
            }
        ],
        "evaluation_settings": {
            "T4_MAX_TOKENS": "1024",
            "T4_MODEL_TIMEOUT_S": "30",
            "T4_MODEL_RETRIES": "2",
            "T4_UNIT_TIMEOUT_S": "480",
            "T4_TOP_K": str(report["provenance"]["prediction_settings"]["top_k"]),
            "T4_TEMPERATURE": "0",
        },
        "version": "synthetic",
        "backend": "synthetic CPU",
        "runner_sha256": hashlib.sha256(b"invented launcher").hexdigest(),
        "host": {"machine": "synthetic"},
    }
    path = report_path.parent / "runtime.json"
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def bound_runtime(tmp_path):
    report = {
        "provenance": {
            "git_commit": "synthetic",
            "git_dirty": False,
            "prediction_settings": {"mode": "model", "top_k": 4},
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    path = write_runtime(tmp_path, report_path, manifest)
    return path, report_path, manifest, report


def test_only_ephemeral_port_is_ignored_in_identity(bound_runtime):
    path, report_path, manifest, report = bound_runtime
    before = checked_runtime(path, report_path, manifest, report)
    runtime = json.loads(path.read_text())
    runtime["command"][runtime["command"].index("--port") + 1] = "17172"
    path.write_text(json.dumps(runtime))
    assert checked_runtime(path, report_path, manifest, report) == before
    runtime["command"][-1] = "8192"
    path.write_text(json.dumps(runtime))
    assert checked_runtime(path, report_path, manifest, report) != before


@pytest.mark.parametrize(
    "defect",
    [
        "report",
        "manifest",
        "binary",
        "weights",
        "incomplete",
        "failed",
        "dirty",
        "remote",
        "port",
        "wrong_model",
        "settings",
        "no_binary_hash",
    ],
)
def test_changed_or_unbound_model_records_are_refused(bound_runtime, defect):
    path, report_path, manifest, report = bound_runtime
    runtime = json.loads(path.read_text())
    if defect in ("report", "manifest"):
        target = report_path if defect == "report" else manifest
        target.write_text(target.read_text() + " ")
    elif defect in ("binary", "weights"):
        from pathlib import Path

        target = (
            runtime["command"][0]
            if defect == "binary"
            else runtime["weights"][0]["path"]
        )
        Path(target).write_bytes(b"tampered")
    elif defect == "incomplete":
        runtime["complete"] = False
    elif defect == "failed":
        runtime["evaluation_exit_codes"] = [1]
    elif defect == "dirty":
        report["provenance"]["git_dirty"] = True
    elif defect in ("remote", "port"):
        flag = "--host" if defect == "remote" else "--port"
        runtime["command"][runtime["command"].index(flag) + 1] = (
            "0.0.0.0" if defect == "remote" else "0"
        )
    elif defect == "wrong_model":
        runtime["command"][2] = runtime["command"][0]
    elif defect == "settings":
        runtime["evaluation_settings"]["T4_TOP_K"] = "2"
    else:
        del runtime["runtime_binary_sha256"]
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        checked_runtime(path, report_path, manifest, report)
