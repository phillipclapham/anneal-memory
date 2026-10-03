"""Minimal OpenAI Chat Completions client (stdlib only) with a token/cost ledger.

The key comes from $OPENAI_API_KEY, or else from the dotenv-style file named by
$ANNEAL_BENCH_ENV_FILE (an OPENAI_API_KEY=... line). It lives only in this
process's memory: never printed, never written to disk, never put in argv.
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
# Ollama's OpenAI-compatible endpoint on the local daemon; ``:cloud`` models are served
# remotely through it. No key, no per-token price: every number from it is TUNING-ONLY.
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

# USD per 1M tokens. Source: OpenAI's published API pricing for gpt-5-mini
# (see PRICES_SOURCE); re-check before quoting a cost outside this bench.
PRICES = {"gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00}}
PRICES_SOURCE = "https://developers.openai.com/api/docs/pricing, Standard tier, gpt-5-mini row (read 2026-10-03)"


def is_ollama(model: str) -> bool:
    """An Ollama model name (``name:tag``, e.g. ``gpt-oss:120b-cloud``); OpenAI names have no colon."""
    return ":" in model


def _load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_file = os.environ.get("ANNEAL_BENCH_ENV_FILE")
    if not env_file:
        raise RuntimeError("set OPENAI_API_KEY or ANNEAL_BENCH_ENV_FILE")
    for line in Path(env_file).expanduser().read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError("OPENAI_API_KEY not found in $ANNEAL_BENCH_ENV_FILE")


class Ledger:
    """Thread-safe running totals of tokens and USD, per call role."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_role: dict[str, dict[str, float]] = {}

    def add(self, role: str, model: str, usage: dict) -> None:
        p = PRICES.get(model, {"input": 0.0, "cached_input": 0.0, "output": 0.0})
        prompt = usage.get("prompt_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        completion = usage.get("completion_tokens", 0)
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        usd = ((prompt - cached) * p["input"] + cached * p["cached_input"]
               + completion * p["output"]) / 1e6
        with self._lock:
            r = self.by_role.setdefault(role, {"calls": 0, "prompt": 0, "cached": 0,
                                               "completion": 0, "reasoning": 0, "usd": 0.0})
            r["calls"] += 1
            r["prompt"] += prompt
            r["cached"] += cached
            r["completion"] += completion
            r["reasoning"] += reasoning
            r["usd"] += usd

    def total_usd(self) -> float:
        with self._lock:
            return sum(r["usd"] for r in self.by_role.values())

    def summary(self) -> dict:
        with self._lock:
            tot = {k: sum(r[k] for r in self.by_role.values())
                   for k in ("calls", "prompt", "cached", "completion", "reasoning", "usd")}
            return {"by_role": json.loads(json.dumps(self.by_role)), "total": tot}


class Client:
    def __init__(self, model: str, ledger: Ledger, budget_usd: float) -> None:
        self.ollama = is_ollama(model)
        if not self.ollama and model not in PRICES:
            raise ValueError(f"no price on record for {model}")
        self.model = model
        self.ledger = ledger
        self.budget_usd = budget_usd
        self._key = None if self.ollama else _load_key()
        self._url = OLLAMA_URL if self.ollama else API_URL

    def chat(self, messages: list[dict], *, role: str, max_completion_tokens: int,
             retries: int = 5) -> tuple[str, str]:
        """One completion -> (content, finish_reason). No temperature/top-p
        override (paper protocol)."""
        msg, finish = self.chat_message(messages, role=role,
                                        max_completion_tokens=max_completion_tokens,
                                        retries=retries)
        return msg.get("content") or "", finish

    def chat_message(self, messages: list[dict], *, role: str, max_completion_tokens: int,
                     tools: list[dict] | None = None, retries: int = 5) -> tuple[dict, str]:
        """One completion -> (the assistant message dict, finish_reason). With ``tools``
        the message may carry ``tool_calls`` (OpenAI function-calling shape)."""
        if not self.ollama and self.ledger.total_usd() >= self.budget_usd:
            raise BudgetExceeded(f"budget ${self.budget_usd:.2f} reached")
        payload: dict = {"model": self.model, "messages": messages}
        # Ollama's compat layer reads max_tokens; OpenAI's reasoning models want
        # max_completion_tokens.
        payload["max_tokens" if self.ollama else "max_completion_tokens"] = max_completion_tokens
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        delay = 2.0
        for attempt in range(retries):
            req = urllib.request.Request(self._url, data=body, method="POST", headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if self.ollama and (e.code == 429 or _looks_like_usage_limit(detail)):
                    # Phill's rule: an Ollama usage/limit error stops the run at once.
                    # No retry loop, no paid fallback.
                    raise UsageLimit(f"Ollama usage limit (HTTP {e.code}): {detail}") from None
                if "insufficient_quota" in detail:  # no credits: retrying cannot help
                    raise BudgetExceeded(f"OpenAI account quota exhausted: {detail}") from None
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"HTTP {e.code} from {self._url}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"network error from {self._url}: {e}") from None
        if "error" in data and self.ollama and _looks_like_usage_limit(json.dumps(data["error"])):
            raise UsageLimit(f"Ollama usage limit: {data['error']}")
        self.ledger.add(role, self.model, data.get("usage", {}))
        choice = data["choices"][0]
        return choice["message"], choice.get("finish_reason") or ""


def _looks_like_usage_limit(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ("usage limit", "weekly limit", "rate limit", "quota",
                                "too many requests", "upgrade your plan"))


class BudgetExceeded(RuntimeError):
    pass


class UsageLimit(RuntimeError):
    """An Ollama usage/limit error. The run stops; it is never retried."""
