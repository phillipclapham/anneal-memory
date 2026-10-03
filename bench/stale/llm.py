"""Minimal OpenAI chat-completions client for the STALE harness (stdlib only).

The key is read at runtime from ``OPENAI_API_KEY`` in the environment, or else from
the ``.env`` file named by ``--env-file`` (default: flow's ``.env.flow``). It is
never printed, written, or put in argv.

Every response's ``usage`` goes into a :class:`Ledger`, priced from :data:`PRICES`.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ENV_FILE = Path.home() / "Briefcase" / "flow" / ".env.flow"

# USD per 1M tokens: (input, cached input, output). Standard tier, read from the
# pricing table embedded in https://platform.openai.com/docs/pricing on 2026-10-03.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
}


class BudgetExceeded(RuntimeError):
    pass


def load_key(env_file: Path = DEFAULT_ENV_FILE) -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"OPENAI_API_KEY not found in the environment or {env_file}")


def _price_key(model: str) -> str:
    for name in sorted(PRICES, key=len, reverse=True):
        if model == name or model.startswith(name + "-"):
            return name
    raise KeyError(f"no price for {model!r}; add it to PRICES with its source")


class Ledger:
    """Thread-safe token and cost accounting, appended to a JSONL file."""

    def __init__(self, path: Path, max_usd: float) -> None:
        self.path = path
        self.max_usd = max_usd
        self._lock = threading.Lock()
        self.usd = 0.0
        self.tokens = {"prompt": 0, "cached": 0, "completion": 0}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self._add(row)

    def _add(self, row: dict) -> None:
        self.usd += row["usd"]
        self.tokens["prompt"] += row["prompt_tokens"]
        self.tokens["cached"] += row["cached_tokens"]
        self.tokens["completion"] += row["completion_tokens"]

    def check(self) -> None:
        if self.usd >= self.max_usd:
            raise BudgetExceeded(f"spent ${self.usd:.4f} >= cap ${self.max_usd:.2f}")

    def record(self, model: str, purpose: str, usage: dict) -> dict:
        p_in, p_cached, p_out = PRICES[_price_key(model)]
        prompt = int(usage.get("prompt_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        usd = ((prompt - cached) * p_in + cached * p_cached + completion * p_out) / 1e6
        row = {"model": model, "purpose": purpose, "prompt_tokens": prompt,
               "cached_tokens": cached, "completion_tokens": completion, "usd": usd,
               "t": time.time()}
        with self._lock:
            self._add(row)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        return row


class Client:
    def __init__(self, key: str, ledger: Ledger, timeout: float = 300.0) -> None:
        self._key = key
        self.ledger = ledger
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict], *, purpose: str,
             retries: int = 5, **params) -> tuple[str, dict]:
        """One chat completion. Returns (content, usage). Retries 429/5xx/timeouts."""
        self.ledger.check()
        body = json.dumps({"model": model, "messages": messages, **params}).encode()
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(API_URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.load(r)
                usage = data.get("usage") or {}
                self.ledger.record(data.get("model", model), purpose, usage)
                return data["choices"][0]["message"].get("content") or "", usage
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise last from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
            time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError(f"failed after {retries} attempts: {last}")
