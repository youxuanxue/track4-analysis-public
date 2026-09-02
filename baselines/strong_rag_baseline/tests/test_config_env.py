"""The reference agent still reads ``MODEL_NAME`` (finding H6, prose-drift sweep
2026-08-27) as the ``model`` string posted to the *local* llama.cpp server.

SUBMISSION_CLI.md's container-environment contract injects ``MODEL_NAME``. This
agent never sends that id to ``$MODEL_ENDPOINT`` — it only labels the loopback
``/v1/chat/completions`` body. ``MODEL_ID`` survives as a local-dev fallback;
``MODEL_NAME`` wins when both are set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strong_rag_baseline.config import Config  # noqa: E402

_ENV_VARS = ("MODEL_NAME", "MODEL_ID", "MODEL_ENDPOINT", "MODEL_TOKEN")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_harness_injected_model_name_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "house-model-pin-2026-07")
    assert Config.from_env().model_id == "house-model-pin-2026-07"


def test_model_name_beats_local_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "house-model-pin-2026-07")
    monkeypatch.setenv("MODEL_ID", "qwen2.5:7b")
    assert Config.from_env().model_id == "house-model-pin-2026-07"


def test_model_id_survives_as_local_dev_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_ID", "qwen2.5:7b")
    assert Config.from_env().model_id == "qwen2.5:7b"
