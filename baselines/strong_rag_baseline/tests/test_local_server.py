"""Local llama.cpp launcher: find weights, refuse non-loopback, fall back."""

from __future__ import annotations

from pathlib import Path

from baselines.strong_rag_baseline.client import HTTPModelClient
from baselines.strong_rag_baseline.config import Config
from baselines.strong_rag_baseline.local_server import (
    GGUF_FILENAME,
    find_gguf,
    find_llama_server,
    try_start_local_server,
)


def test_find_gguf_returns_none_when_weights_are_absent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("T4_GGUF_PATH", str(tmp_path / "missing.gguf"))
    assert find_gguf() is None


def test_find_gguf_reads_t4_gguf_path(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / GGUF_FILENAME
    gguf.write_bytes(b"not-a-real-gguf-but-present")
    monkeypatch.setenv("T4_GGUF_PATH", str(gguf))
    assert find_gguf() == gguf


def test_find_llama_server_missing_is_none(monkeypatch) -> None:
    monkeypatch.setenv("T4_LLAMA_SERVER", "/no/such/llama-server")
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.local_server.shutil.which",
        lambda _name: None,
    )
    assert find_llama_server() is None


def test_try_start_returns_none_without_binary_or_weights(monkeypatch) -> None:
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.local_server.find_gguf",
        lambda: None,
    )
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.local_server.find_llama_server",
        lambda: None,
    )
    assert try_start_local_server(startup_s=1.0) is None


def test_try_start_returns_none_when_process_exits(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / GGUF_FILENAME
    gguf.write_bytes(b"x")
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.local_server.find_gguf",
        lambda: gguf,
    )
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.local_server.find_llama_server",
        lambda: str(binary),
    )
    assert try_start_local_server(startup_s=2.0) is None


def test_http_client_refuses_a_non_loopback_url() -> None:
    client = HTTPModelClient(Config.from_env(), base_url="http://example.invalid/v1")
    try:
        client.complete("sys", "user")
    except RuntimeError as exc:
        assert "127.0.0.1" in str(exc)
    else:
        raise AssertionError("non-loopback URL must be refused")


def test_config_keeps_official_and_local_endpoints_separate(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8000/v1")
    cfg = Config.from_env()
    assert cfg.model_endpoint == "http://model:8000/v1"
    assert cfg.local_base_url.startswith("http://127.0.0.1")
    assert "model:8000" not in cfg.local_base_url


def test_mock_analyze_bypasses_model_endpoint_and_does_not_boot_llama(
    tmp_path, monkeypatch
) -> None:
    """The explicit offline mode performs no network calls or model startup."""
    from baselines.strong_rag_baseline.cli import main

    unit = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"
    opened: list[str] = []
    started = {"n": 0}

    def _boom(req, timeout=None):  # noqa: ARG001
        url = getattr(req, "full_url", str(req))
        opened.append(url)
        raise AssertionError(f"agent opened {url}")

    def _no_start(**_kwargs):
        started["n"] += 1
        raise AssertionError("official analyze must not start llama.cpp")

    monkeypatch.setenv("MODEL_ENDPOINT", "http://model:8000/v1")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    monkeypatch.setattr(
        "baselines.strong_rag_baseline.cli.try_start_local_server", _no_start
    )
    out = tmp_path / "answer.json"
    code = main(
        [
            "analyze",
            "--mock",
            "--task",
            str(unit / "task.json"),
            "--corpus",
            str(unit / "corpus"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert opened == []
    assert started["n"] == 0
    assert out.is_file()


def test_local_llama_flag_attempts_the_opt_in_server(tmp_path, monkeypatch) -> None:
    from baselines.strong_rag_baseline.cli import main

    unit = Path(__file__).resolve().parents[3] / "units" / "t4-EXAMPLE-eps-beat"
    started = {"n": 0}

    def _none(**_kwargs):
        started["n"] += 1
        return None

    monkeypatch.setattr(
        "baselines.strong_rag_baseline.cli.try_start_local_server", _none
    )
    out = tmp_path / "answer.json"
    code = main(
        [
            "analyze",
            "--local-llama",
            "--task",
            str(unit / "task.json"),
            "--corpus",
            str(unit / "corpus"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert started["n"] == 1
    assert out.is_file()


def test_config_local_llama_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("T4_LOCAL_LLAMA", raising=False)
    assert Config.from_env().local_llama is False
    monkeypatch.setenv("T4_LOCAL_LLAMA", "1")
    assert Config.from_env().local_llama is True
