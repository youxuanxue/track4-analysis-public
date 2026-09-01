"""Submission image contract: Dockerfile + analyze.py entrypoint, no Docker required.

The Cloud VM this suite runs on does not have Docker. These tests pin the
recipe and run the same process boundary the ENTRYPOINT crosses.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_BASELINES = Path(__file__).resolve().parents[1]
_REPO = _BASELINES.parent
_DOCKERFILE = _BASELINES / "Dockerfile"
_ENTRYPOINT = _BASELINES / "analyze.py"
_UNIT = _REPO / "units" / "t4-EXAMPLE-eps-beat"
_LOCK = (
    _BASELINES
    / "strong_rag_baseline"
    / "tests"
    / "locks"
    / "official_gate_d4d0584.json"
)


def test_dockerfile_is_python_313_interface_2() -> None:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.13\b", text, flags=re.M), text.splitlines()[0]
    assert 'LABEL qfbench2.interface_version="2.0"' in text
    assert 'ENTRYPOINT ["python", "analyze.py"]' in text
    assert "CMD" in text and "analyze" in text
    assert not re.search(r"\b(torch|transformers|tensorflow|cuda)\b", text, flags=re.I)


def test_image_recipe_copies_the_reasoner_and_the_public_lock() -> None:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY strong_rag_baseline" in text
    assert "COPY analyze.py" in text
    assert _LOCK.is_file()
    # .dockerignore must not drop the lock JSON the reasoner loads at runtime.
    dockerignore = (_BASELINES / ".dockerignore").read_text(encoding="utf-8")
    assert "official_gate" not in dockerignore
    assert "locks" not in dockerignore or "Keep tests/locks" in dockerignore


def test_analyze_entrypoint_accepts_the_harness_argv(tmp_path: Path) -> None:
    out = tmp_path / "answer.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_ENTRYPOINT),
            "analyze",
            "--task",
            str(_UNIT / "task.json"),
            "--corpus",
            str(_UNIT / "corpus"),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    answer = json.loads(out.read_text(encoding="utf-8"))
    assert answer["task_id"] == "t4-EXAMPLE-eps-beat"
    pred = answer["entity_predictions"][0]
    lock = json.loads(_LOCK.read_text(encoding="utf-8"))
    expected = lock["units"]["t4-EXAMPLE-eps-beat"][0]
    assert pred["label"] == expected["label"]
    assert pred["point_forecast"] == expected["point_forecast"]
    assert pred["interval"]["lo"] == expected["interval"]["lo"]
    assert pred["interval"]["hi"] == expected["interval"]["hi"]


def test_wrong_verb_is_rejected(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_ENTRYPOINT),
            "simulate",
            "--task",
            str(_UNIT / "task.json"),
            "--corpus",
            str(_UNIT / "corpus"),
            "--out",
            str(tmp_path / "answer.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert not (tmp_path / "answer.json").exists()


def test_smoke_script_local_path_writes_a_schema_valid_answer(tmp_path: Path) -> None:
    script = _BASELINES / "smoke_image.sh"
    assert script.is_file()
    proc = subprocess.run(
        ["bash", str(script), str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "docker is not available" in proc.stdout or "docker is available" in proc.stdout
    assert (tmp_path / "answer.json").is_file()
