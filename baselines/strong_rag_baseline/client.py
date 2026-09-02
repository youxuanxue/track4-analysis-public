"""OpenAI-compatible chat client for the *local* llama.cpp server (stdlib only).

``analyze`` starts llama.cpp on 127.0.0.1 and posts to that loopback URL.
This module never reads ``$MODEL_ENDPOINT`` and never opens a vendor host.
Tests inject :class:`MockModelClient` — same interface, canned replies, no
network.

Determinism: temperature 0 and a fixed ``seed`` are sent on every request.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import Config


class ModelClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant message text for one chat exchange."""
        ...


@dataclass
class HTTPModelClient:
    config: Config
    base_url: str = ""

    def complete(self, system: str, user: str) -> str:
        endpoint = (self.base_url or self.config.local_base_url).rstrip("/")
        if not endpoint.startswith("http://127.0.0.1") and not endpoint.startswith(
            "http://localhost"
        ):
            raise RuntimeError(
                "refusing a non-loopback model URL; this agent only calls "
                "llama.cpp on 127.0.0.1"
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout_s
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"local model call failed after {self.config.max_retries} attempts"
        ) from last_error


@dataclass
class MockModelClient:
    """Test double: returns canned text, or delegates to a callable."""

    reply: str | Callable[[str, str], str]

    def complete(self, system: str, user: str) -> str:
        if callable(self.reply):
            return self.reply(system, user)
        return self.reply
