"""Runtime configuration for the strong RAG baseline (env-driven, no files).

The submission talks to a llama.cpp server on 127.0.0.1. It does not read
``$MODEL_ENDPOINT`` and does not call a vendor API. ``MODEL_NAME`` / ``MODEL_ID``
are only the ``model`` string posted to the *local* ``/v1/chat/completions``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .local_server import DEFAULT_ALIAS, DEFAULT_CTX, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_STARTUP_S


@dataclass(frozen=True)
class Config:
    model_id: str
    seed: int  # forwarded to the model AND used for any local tie-breaking
    top_k: int  # retrieved chunks per entity
    timeout_s: float  # per model call
    max_retries: int
    temperature: float  # fixed at 0 for determinism; env override for experiments
    max_tokens: int
    local_host: str
    local_port: int
    local_ctx: int
    local_startup_s: float

    @property
    def local_base_url(self) -> str:
        return f"http://{self.local_host}:{self.local_port}/v1"

    @staticmethod
    def from_env() -> "Config":
        host = os.environ.get("T4_LOCAL_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
        if host not in {"127.0.0.1", "localhost", "::1"}:
            host = DEFAULT_HOST
        return Config(
            # MODEL_NAME is what the harness injects (SUBMISSION_CLI.md container
            # contract); MODEL_ID is a local-dev fallback only. Neither selects a
            # remote endpoint — the request always goes to 127.0.0.1.
            model_id=os.environ.get("MODEL_NAME")
            or os.environ.get("MODEL_ID", "")
            or DEFAULT_ALIAS,
            seed=int(os.environ.get("T4_SEED", "20260731")),
            top_k=int(os.environ.get("T4_TOP_K", "10")),
            timeout_s=float(os.environ.get("T4_MODEL_TIMEOUT_S", "45")),
            max_retries=int(os.environ.get("T4_MODEL_RETRIES", "2")),
            temperature=float(os.environ.get("T4_TEMPERATURE", "0")),
            max_tokens=int(os.environ.get("T4_MAX_TOKENS", "384")),
            local_host=host,
            local_port=int(os.environ.get("T4_LOCAL_PORT", str(DEFAULT_PORT))),
            local_ctx=int(os.environ.get("T4_LOCAL_CTX", str(DEFAULT_CTX))),
            local_startup_s=float(
                os.environ.get("T4_LOCAL_STARTUP_S", str(DEFAULT_STARTUP_S))
            ),
        )
