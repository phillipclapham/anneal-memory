"""OpenRouter chat client (stdlib only) with the same surface as ant.Client: ``chat(messages, role=,
max_completion_tokens=) -> (text, finish_reason)``, so a reader can be swapped without touching the rows.

The key comes ONLY from $OPENROUTER_API_KEY in this process's environment; it is never printed or
written. Every request carries ``provider: {"data_collection": "deny"}`` (OpenRouter provider
routing: only providers that do not collect user data) and ``usage: {"include": true}`` so the
response reports its own cost, which the ledger records.
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

from .ant import BudgetExceeded, Ledger

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
RATE_LIMIT_PATIENCE_S = 600.0
_INFLIGHT = threading.BoundedSemaphore(max(1, int(os.environ.get("OR_MAX_INFLIGHT", "12"))))


class _Transient(Exception):
    """A 200 response that reports an error and carries no completion; retried like a 5xx."""


class ORLedger(Ledger):
    """Same shape as ant.Ledger; the cost is the one OpenRouter reports for the request."""

    def add(self, role: str, model: str, usage: dict) -> None:  # type: ignore[override]
        usd = float(usage.get("cost") or 0.0)
        with self._lock:
            r = self.by_role.setdefault(role, {"calls": 0, "prompt": 0, "cached": 0, "written": 0,
                                               "completion": 0, "reasoning": 0, "usd": 0.0})
            r["calls"] += 1
            r["prompt"] += usage.get("prompt_tokens", 0)
            r["completion"] += usage.get("completion_tokens", 0)
            r["usd"] += usd


def live_prices() -> dict:
    """{model id: (usd per 1M prompt tokens, usd per 1M completion tokens)} from the live list."""
    with urllib.request.urlopen(urllib.request.Request(MODELS_URL), timeout=60) as resp:
        data = json.loads(resp.read())["data"]
    out = {}
    for m in data:
        try:
            out[m["id"]] = (float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def build_payload(model: str, messages: list[dict], max_completion_tokens: int) -> dict:
    return {"model": model, "max_tokens": max_completion_tokens,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "provider": {"data_collection": "deny"}, "usage": {"include": True}}


class Client:
    def __init__(self, model: str, ledger: Ledger, budget_usd: float) -> None:
        self.model, self.ledger, self.budget_usd = model, ledger, budget_usd
        self._key = os.environ["OPENROUTER_API_KEY"]

    def chat(self, messages: list[dict], *, role: str, max_completion_tokens: int,
             retries: int = 5) -> tuple[str, str]:
        if self.ledger.total_usd() >= self.budget_usd:
            raise BudgetExceeded(f"budget ${self.budget_usd:.2f} reached")
        body = json.dumps(build_payload(self.model, messages, max_completion_tokens)).encode()
        delay, waited, attempt = 2.0, 0.0, -1
        while True:
            attempt += 1
            req = urllib.request.Request(API_URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {self._key}", "content-type": "application/json"})
            try:
                with _INFLIGHT, urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read())
                # a response is billed whether or not it is usable: record it before judging it
                self.ledger.add(role, self.model, data.get("usage") or {})
                choice = (data.get("choices") or [{}])[0]
                err = data.get("error")
                if err and not data.get("choices"):
                    code = err.get("code") if isinstance(err, dict) else None
                    if code in (401, 402, 403):
                        raise BudgetExceeded(f"OpenRouter refused (in-body {code}): {str(err)[:200]}")
                    raise _Transient(f"in-body error: {str(err)[:200]}")
                if choice.get("finish_reason") == "error":
                    raise _Transient("errored completion")
                break
            except _Transient as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"OpenRouter returned no usable completion: {e}") from None
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (401, 402, 403):
                    raise BudgetExceeded(f"OpenRouter refused (HTTP {e.code}): {detail}") from None
                if e.code == 429 and waited < RATE_LIMIT_PATIENCE_S:
                    attempt -= 1
                    nap = random.uniform(3, 12)
                    waited += nap
                    time.sleep(nap)
                    continue
                if e.code in (408, 500, 502, 503, 504, 529) and attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from None
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                raise RuntimeError(f"OpenRouter network error: {e}") from None
        text = (choice.get("message") or {}).get("content") or ""
        fr = choice.get("finish_reason") or ""
        return text, ("stop" if fr in ("stop", "end_turn") else fr)
