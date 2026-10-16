"""Minimal Anthropic Messages client (stdlib only) with a token/cost ledger.

Drop-in for oai.py's Client/Ledger/BudgetExceeded. The key comes ONLY from
$ANTHROPIC_API_KEY in this process's environment (set it on the harness command's own
line, never in a shell profile, settings file or .env). It is never printed or written.
"""

from __future__ import annotations

import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
# USD per 1M tokens, claude-haiku-5-5, prompts up to 100K tokens (Anthropic model table,
# read 2026-10-11 via the claude-api skill). Cache reads priced at 0.1x input (documented ratio).
PRICES = {"claude-haiku-5-5": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50},
          "claude-sonnet-5-5": {"input": 2.00, "cached_input": 0.20, "cache_write": 2.50, "output": 10.00}}
PRICES_SOURCE = "claude-api skill model table cached 2026-10-06: Haiku 5.5 (<=100K prompt) and Sonnet 5.5 rows; cache write priced at the documented 1.25x"
EFFORT = {"wrap": "medium", "probe-update": "medium", "answer": "low", "judge": "low", "gen": "medium"}


class BudgetExceeded(RuntimeError):
    pass


# The org limits CONCURRENT requests across all models (HTTP 429 "Number of concurrent
# requests ..."). Two processes at 30 + 15 workers (45 requests) exceeded it on 2026-10-11 and
# each hit killed a 38-session unit mid-way. The true limit was not measured: this cap is
# PER PROCESS, so run ONE process, or split it across processes with ANT_MAX_INFLIGHT. A 429
# is waited out instead of failing the unit, so an overshoot slows a run and does not kill it.
MAX_INFLIGHT = max(1, int(os.environ.get("ANT_MAX_INFLIGHT", "24")))
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT)
RATE_LIMIT_PATIENCE_S = 1800.0  # total time one call waits on 429s before it fails


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.by_role: dict[str, dict[str, float]] = {}

    def add(self, role: str, model: str, usage: dict) -> None:
        p = PRICES[model]
        fresh = usage.get("input_tokens", 0)
        cached = usage.get("cache_read_input_tokens", 0) or 0
        written = usage.get("cache_creation_input_tokens", 0) or 0
        out = usage.get("output_tokens", 0)
        usd = (fresh * p["input"] + cached * p["cached_input"]
               + written * p["cache_write"] + out * p["output"]) / 1e6
        with self._lock:
            r = self.by_role.setdefault(role, {"calls": 0, "prompt": 0, "cached": 0, "written": 0,
                                               "completion": 0, "reasoning": 0, "usd": 0.0})
            r["calls"] += 1
            r["prompt"] += fresh + cached + written
            r["cached"] += cached
            r["written"] += written
            r["completion"] += out
            r["usd"] += usd

    def total_usd(self) -> float:
        with self._lock:
            return sum(r["usd"] for r in self.by_role.values())

    def summary(self) -> dict:
        with self._lock:
            tot = {k: sum(r[k] for r in self.by_role.values())
                   for k in ("calls", "prompt", "cached", "written", "completion", "reasoning", "usd")}
            return {"by_role": json.loads(json.dumps(self.by_role)), "total": tot}


def _blocks(message: dict) -> list[dict]:
    """A message's content as API text blocks. A message may carry ``cache_prefix: N``: its
    first N characters become their own block with a cache breakpoint at the end. The two
    blocks concatenate to exactly the original string, so the prompt text is unchanged."""
    text, n = message["content"], message.get("cache_prefix")
    if not n:
        return [{"type": "text", "text": text}]
    if not 0 < n < len(text):
        raise ValueError(f"cache_prefix {n} outside the message ({len(text)} chars)")
    return [{"type": "text", "text": text[:n], "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": text[n:]}]


def build_payload(model: str, messages: list[dict], role: str, max_completion_tokens: int) -> dict:
    """The Messages request body. Caching is opt-in per message: ``cache`` on a system
    message puts a breakpoint at the end of the system block; ``cache_prefix`` on a user
    message caches that many leading characters. Minimum cacheable prefix on claude-haiku-5-5
    is 512 tokens (platform.claude.com prompt-caching table, read 2026-10-11); a shorter
    marked prefix is silently processed uncached."""
    system = [m for m in messages if m["role"] == "system"]
    turns = [{"role": m["role"], "content": _blocks(m) if m.get("cache_prefix") else m["content"]}
             for m in messages if m["role"] != "system"]
    payload = {"model": model, "max_tokens": max_completion_tokens, "messages": turns,
               "output_config": {"effort": EFFORT.get(role, "low")}}
    if system:
        text = "\n\n".join(m["content"] for m in system)
        if any(m.get("cache") for m in system):
            payload["system"] = [{"type": "text", "text": text,
                                  "cache_control": {"type": "ephemeral"}}]
        else:
            payload["system"] = text
    return payload


class Client:
    def __init__(self, model: str, ledger: Ledger, budget_usd: float) -> None:
        if model not in PRICES:
            raise ValueError(f"no price on record for {model}")
        self.model, self.ledger, self.budget_usd = model, ledger, budget_usd
        self._key = os.environ["ANTHROPIC_API_KEY"]

    def chat(self, messages: list[dict], *, role: str, max_completion_tokens: int,
             retries: int = 6) -> tuple[str, str]:
        """One completion -> (text, finish_reason mapped to 'stop' on end_turn)."""
        if self.ledger.total_usd() >= self.budget_usd:
            raise BudgetExceeded(f"budget ${self.budget_usd:.2f} reached")
        body = json.dumps(build_payload(self.model, messages, role, max_completion_tokens)).encode()
        delay = 2.0
        waited = 0.0
        attempt = -1
        while True:
            attempt += 1
            req = urllib.request.Request(API_URL, data=body, method="POST", headers={
                "x-api-key": self._key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
            try:
                with _INFLIGHT, urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                low = detail.lower()
                if "credit balance" in low or e.code in (401, 402, 403):
                    raise BudgetExceeded(f"Anthropic account refused (HTTP {e.code}): {detail}") from None
                if e.code == 429 and waited < RATE_LIMIT_PATIENCE_S:
                    attempt -= 1  # a 429 does not use up the retry budget
                    nap = random.uniform(5, 20)
                    waited += nap
                    time.sleep(nap)
                    continue
                if e.code in (500, 502, 503, 504, 529) and attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"Anthropic HTTP {e.code}: {detail}") from None
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                # a reset or truncated body mid-read is not a URLError, and a lost 80-session unit is ~40 paid calls
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"Anthropic network error: {e}") from None
        self.ledger.add(role, self.model, data.get("usage", {}))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        sr = data.get("stop_reason") or ""
        return text, ("stop" if sr == "end_turn" else sr)
