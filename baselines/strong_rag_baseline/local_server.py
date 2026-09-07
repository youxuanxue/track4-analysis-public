"""Opt-in llama.cpp launcher on 127.0.0.1 (developer machine only).

Official ``analyze`` does not call this module. ``--local-llama`` /
``T4_LOCAL_LLAMA=1`` may start ``llama-server`` against a gitignored GGUF and
POST to loopback ``/v1/chat/completions``. It never reads ``$MODEL_ENDPOINT``.
Missing weights, a missing binary, or a failed health check return ``None``.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

GGUF_FILENAME = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 4096
DEFAULT_STARTUP_S = 90.0
DEFAULT_ALIAS = "Qwen2.5-7B-Instruct-Q4_K_M"


def _search_roots() -> list[Path]:
    here = Path(__file__).resolve()
    env = os.environ.get("T4_GGUF_PATH", "").strip()
    roots: list[Path] = []
    if env:
        roots.append(Path(env))
    roots.extend(
        [
            Path("/opt/models") / GGUF_FILENAME,
            Path("/models") / GGUF_FILENAME,
            here.parents[1] / "models" / GGUF_FILENAME,
            here.parents[2] / "models" / GGUF_FILENAME,
        ]
    )
    return roots


def find_gguf() -> Path | None:
    """Return the first readable baked/copied GGUF, or None."""
    for path in _search_roots():
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def find_llama_server() -> str | None:
    env = os.environ.get("T4_LLAMA_SERVER", "").strip()
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    return shutil.which("llama-server")


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _pick_port(host: str, preferred: int) -> int:
    if _port_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _health_ok(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _thread_count() -> int:
    n = os.cpu_count() or 4
    return max(1, n - 1)


@dataclass
class LocalLlamaServer:
    """A running llama.cpp process bound to loopback."""

    base_url: str
    host: str
    port: int
    process: subprocess.Popen[bytes]
    gguf: Path

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def try_start_local_server(
    *,
    host: str | None = None,
    port: int | None = None,
    ctx: int | None = None,
    startup_s: float | None = None,
) -> LocalLlamaServer | None:
    """Boot llama-server or return None so analyze can fall back."""
    gguf = find_gguf()
    binary = find_llama_server()
    if gguf is None or binary is None:
        return None

    bind_host = host or os.environ.get("T4_LOCAL_HOST", DEFAULT_HOST)
    # Loopback only — official scoring may be --network=none.
    if bind_host not in {"127.0.0.1", "localhost", "::1"}:
        bind_host = DEFAULT_HOST
    preferred = port if port is not None else int(os.environ.get("T4_LOCAL_PORT", str(DEFAULT_PORT)))
    bind_port = _pick_port(bind_host, preferred)
    n_ctx = ctx if ctx is not None else int(os.environ.get("T4_LOCAL_CTX", str(DEFAULT_CTX)))
    timeout = (
        startup_s
        if startup_s is not None
        else float(os.environ.get("T4_LOCAL_STARTUP_S", str(DEFAULT_STARTUP_S)))
    )

    cmd = [
        binary,
        "-m",
        str(gguf),
        "--host",
        bind_host,
        "--port",
        str(bind_port),
        "--ctx-size",
        str(max(512, n_ctx)),
        "--threads",
        str(_thread_count()),
        "--parallel",
        "1",
        "--alias",
        DEFAULT_ALIAS,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None

    health = f"http://{bind_host}:{bind_port}/health"
    deadline = time.monotonic() + max(5.0, timeout)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None
        if _health_ok(health):
            return LocalLlamaServer(
                base_url=f"http://{bind_host}:{bind_port}/v1",
                host=bind_host,
                port=bind_port,
                process=proc,
                gguf=gguf,
            )
        time.sleep(0.5)

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    return None
