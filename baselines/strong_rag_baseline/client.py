"""OpenAI-compatible chat client for ``$MODEL_ENDPOINT`` (stdlib only).

The eval sandbox's only egress is the organizer-hosted model endpoint, reached
over an OpenAI-compatible ``/chat/completions`` protocol. Locally, any server
speaking that protocol works (ollama, llama.cpp, vLLM), and tests inject
:class:`MockModelClient` — same interface, canned replies, no network.

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

    def complete(self, system: str, user: str) -> str:
        if not self.config.model_endpoint:
            raise RuntimeError(
                "MODEL_ENDPOINT is not set. In the eval sandbox it is injected "
                "by the harness; locally, point it at an OpenAI-compatible "
                "server or use --mock."
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "seed": self.config.seed,
        }
        headers = {"Content-Type": "application/json"}
        if self.config.model_token:
            headers["Authorization"] = f"Bearer {self.config.model_token}"
        request = urllib.request.Request(
            f"{self.config.model_endpoint}/chat/completions",
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
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"model call failed after {self.config.max_retries} attempts"
        ) from last_error


@dataclass
class MockModelClient:
    """Test double: returns canned text, or delegates to a callable."""

    reply: str | Callable[[str, str], str]

    def complete(self, system: str, user: str) -> str:
        if callable(self.reply):
            return self.reply(system, user)
        return self.reply
