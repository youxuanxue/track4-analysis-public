"""Image recipe, packaged entrypoint and container-script failure contracts.

These tests do not build a Docker image. Run smoke_image.sh on a Docker host
to validate the actual container; local process checks remain distinct.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_BASELINES = Path(__file__).resolve().parents[1]
_REPO = _BASELINES.parent
_DOCKERFILE = _BASELINES / "Dockerfile"
_ENTRYPOINT = _BASELINES / "analyze.py"
_UNIT = _REPO / "units" / "t4-EXAMPLE-eps-beat"
_SMOKE = _BASELINES / "smoke_image.sh"


def test_dockerfile_is_python_313_interface_2() -> None:
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.13\b", text, flags=re.M)
    assert 'LABEL qfbench2.interface_version="2.0"' in text
    assert 'ENTRYPOINT ["python", "analyze.py"]' in text
    assert "CMD" in text and "analyze" in text
    assert "llama-server" not in text
    assert "ensure_gguf" not in text
    assert not re.search(r"\b(torch|transformers|tensorflow|cuda)\b", text, flags=re.I)


def _stage_runtime(destination: Path) -> Path:
    """Apply the recipe's COPY instructions without installing or executing the image."""
    destination.mkdir()
    for line in _DOCKERFILE.read_text(encoding="utf-8").splitlines():
        parts = shlex.split(line, comments=True)
        if not parts or parts[0] != "COPY":
            continue
        assert len(parts) == 3, f"COPY contract needs updating: {line}"
        _, source, target = parts
        files = list(_BASELINES.glob(source))
        assert files, f"empty image source: {source}"
        target_dir = destination / target
        target_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            assert file.is_file(), f"COPY must select runtime files, not trees: {file}"
            assert "tests" not in file.relative_to(_BASELINES).parts
            assert file.suffix != ".json", f"fixture copied into the image: {file}"
            shutil.copyfile(file, target_dir / file.name)
    return destination / "analyze.py"


def test_packaged_entrypoint_runs_without_test_fixtures(tmp_path: Path) -> None:
    entrypoint = _stage_runtime(tmp_path / "app")
    assert entrypoint.is_file()
    assert (entrypoint.parent / "strong_rag_baseline" / "reasoner.py").is_file()
    assert not list(entrypoint.parent.rglob("*.json"))
    assert not list(entrypoint.parent.rglob("tests"))
    out = tmp_path / "answer.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(entrypoint),
            "analyze",
            "--task",
            str(_UNIT / "task.json"),
            "--corpus",
            str(_UNIT / "corpus"),
            "--out",
            str(out),
            "--mock",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    answer = json.loads(out.read_text(encoding="utf-8"))
    task = json.loads((_UNIT / "task.json").read_text(encoding="utf-8"))
    assert answer["task_id"] == task["task_id"]
    predictions = answer["entity_predictions"]
    assert sorted(row["entity_id"] for row in predictions) == sorted(
        row["entity_id"] for row in task["entities"]
    )
    for prediction in predictions:
        interval = prediction["interval"]
        assert interval["level"] == 0.9
        assert all(math.isfinite(interval[key]) for key in ("lo", "hi"))
        assert interval["lo"] <= interval["hi"]
        assert prediction["claims"]
        for claim in prediction["claims"]:
            document = json.loads(
                (_UNIT / "corpus" / f"{claim['doc_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert document["doc_date"] <= task["cutoff_date"]
            text = document.get("text")
            if text is None:
                text = " ".join(span["text"] for span in document["spans"])
            assert 0 <= claim["span_start"] < claim["span_end"] <= len(text)
            assert text[claim["span_start"] : claim["span_end"]].strip()


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
        timeout=30,
    )
    assert proc.returncode == 2
    assert not (tmp_path / "answer.json").exists()


def _script_env(tmp_path: Path, docker_script: str | None) -> dict[str, str]:
    # Isolate PATH so the negative controls cannot accidentally use a real Docker daemon.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("dirname", "grep", "mkdir", "rm"):
        path = shutil.which(name)
        assert path, name
        (bin_dir / name).symlink_to(path)
    if docker_script is not None:
        docker = bin_dir / "docker"
        docker.write_text("#!/bin/sh\n" + docker_script, encoding="utf-8")
        docker.chmod(0o755)
    return {**os.environ, "PATH": str(bin_dir)}


@pytest.mark.parametrize("docker_script", [None, "exit 1\n"])
def test_smoke_script_fails_when_docker_cannot_run(
    tmp_path: Path,
    docker_script: str | None,
) -> None:
    output = tmp_path / "output"
    proc = subprocess.run(
        ["/bin/bash", str(_SMOKE), str(output)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=_script_env(tmp_path, docker_script),
        timeout=30,
    )
    assert proc.returncode != 0
    assert "container validation not performed" in proc.stderr
    assert "local CLI check" in proc.stderr
    assert "verified" not in proc.stdout
    assert not output.exists()


def test_smoke_script_propagates_image_build_failure(tmp_path: Path) -> None:
    output = tmp_path / "output"
    docker_script = (
        'case "$1" in\ninfo) exit 0;;\nbuild) exit 29;;\n*) exit 99;;\nesac\n'
    )
    proc = subprocess.run(
        ["/bin/bash", str(_SMOKE), str(output)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=_script_env(tmp_path, docker_script),
        timeout=30,
    )
    assert proc.returncode == 29, proc.stdout + proc.stderr
    assert "verified" not in proc.stdout
    assert not (output / "answer.json").exists()


@pytest.mark.parametrize("failed_command", ["start", "cp"])
def test_smoke_script_propagates_execution_or_copy_failure(
    tmp_path: Path, failed_command: str
) -> None:
    output = tmp_path / "output"
    docker_script = (
        f'if [ "$1" = "{failed_command}" ]; then exit 29; fi\n'
        'case "$1" in\ninfo|build|start|rm) exit 0;;\n'
        "create) echo test-container-id;;\n*) exit 99;;\nesac\n"
    )
    proc = subprocess.run(
        ["/bin/bash", str(_SMOKE), str(output)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=_script_env(tmp_path, docker_script),
        timeout=30,
    )
    assert proc.returncode == 29, proc.stdout + proc.stderr
    assert "verified" not in proc.stdout
    assert not (output / "answer.json").exists()
