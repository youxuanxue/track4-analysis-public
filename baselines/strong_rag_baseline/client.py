"""OpenAI-compatible house client with a deadline shared by every entity."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from http.client import HTTPException
from typing import Callable, Protocol
from urllib.parse import urlsplit

from .config import Config


class ModelClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the assistant message text for one chat exchange."""
        ...


@dataclass
class HTTPModelClient:
    config: Config
    base_url: str = ""
    deadline: float | None = None

    def __post_init__(self) -> None:
        if self.deadline is None:
            self.deadline = time.monotonic() + self.config.unit_timeout_s

    def complete(self, system: str, user: str) -> str:
        return self._complete(system, user)

    def complete_json(self, system: str, user: str, schema: dict) -> str:
        return self._complete(system, user, schema)

    def _complete(self, system: str, user: str, schema: dict | None = None) -> str:
        endpoint = (self.base_url or self.config.model_endpoint).rstrip("/")
        url = urlsplit(endpoint)
        official = (
            bool(self.config.model_endpoint) and endpoint == self.config.model_endpoint
        )
        loopback = url.hostname in {"127.0.0.1", "localhost", "::1"}
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or not (official or loopback)
        ):
            raise RuntimeError(
                "model URL must be MODEL_ENDPOINT or an explicit loopback URL (127.0.0.1)"
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
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "entity_prediction",
                    "strict": True,
                    "schema": schema,
                },
            }
        headers = {"Content-Type": "application/json"}
        if official and self.config.model_token:
            headers["Authorization"] = f"Bearer {self.config.model_token}"
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            remaining = float(self.deadline) - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("unit model deadline exhausted") from last_error
            try:
                with urllib.request.urlopen(
                    request, timeout=min(self.config.timeout_s, remaining)
                ) as response:
                    parts: list[bytes] = []
                    size = 0
                    while True:
                        if time.monotonic() >= float(self.deadline):
                            raise TimeoutError(
                                "unit model deadline exhausted during response"
                            )
                        part = response.read1(min(65536, 1_048_577 - size))
                        size += len(part)
                        if size > 1_048_576:
                            raise ValueError("model response exceeds 1 MiB")
                        if not part:
                            break
                        parts.append(part)
                    data = b"".join(parts)
                body = json.loads(data.decode("utf-8"))
                if body["choices"][0].get("finish_reason") in {
                    "length",
                    "content_filter",
                }:
                    raise ValueError("model response did not complete")
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model response has no text content")
                return content
            except (
                OSError,
                HTTPException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RecursionError,
            ) as exc:
                last_error = exc
                remaining = float(self.deadline) - time.monotonic()
                if attempt + 1 < self.config.max_retries and remaining > 0:
                    time.sleep(min(2**attempt, remaining))
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
