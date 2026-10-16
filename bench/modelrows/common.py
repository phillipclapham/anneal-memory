"""Shared pieces for the model-call rows: session replay, the real wrap loop, the paper's
rolling-memory baseline, judges, and a scenario-clustered bootstrap.

Everything here runs on scratch stores in a caller-supplied work directory. The wrap loop is the
one the KL-08 in-state benchmark used, copied so this package stands alone.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from anneal_memory import EpisodeType, Store, prepare_wrap, validated_save_continuity
from anneal_memory.continuity import format_wrap_package_text

from .ant import BudgetExceeded, Client, Ledger  # noqa: F401

PROJECT_NAME = "Assistant"
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
WRAP_MAX = 32768
ANSWER_MAX = 4096
JUDGE_MAX = 1024
PROBE_UPDATE_MAX = 8192

# The paper's always-in-state probe (arXiv 2607.24368 Appendix 17), as KL-08 re-implemented it.
PROBE_UPDATER_SYSTEM = (
    "You maintain a memory file for a personal assistant. Store a fact ONLY if it meets at "
    "least one criterion:\n"
    "1. It would change what advice you give (constraints, risks, needs).\n"
    "2. The user would be upset or harmed if you forgot it.\n"
    "Do not store: assistant responses, general knowledge, instructions, opinions on media, "
    "idle questions. One fact per line. Max 200 lines. Output the updated memory file. "
    "Nothing else."
)
PROBE_UPDATER_USER = (
    "## CURRENT MEMORY FILE:\n{current_memory}\n## CONVERSATION:\n{conversation}\n"
    "## OUTPUT:\nWrite the complete updated memory file below:"
)
ANSWER_SYSTEM = (
    "You are a helpful personal assistant. Below is what you remember about this user from "
    "earlier conversations. Use it where it is relevant; do not mention that you have a memory "
    "file.\n--- MEMORY ---\n{memory}\n--- END OF MEMORY ---"
)
NO_MEMORY = "(you have no memory of earlier conversations with this user)"

CACHE_HEAD_ANCHOR = "Existing Proven to scan against:"


def exchanges(session: dict) -> list[str]:
    """One user turn plus the assistant turn(s) after it = one exchange = one episode."""
    out: list[list[str]] = []
    for turn in session["turns"]:
        label = "User" if turn["role"] == "user" else "Assistant"
        line = f"{label}: {turn['content']}"
        if turn["role"] == "user" or not out:
            out.append([line])
        else:
            out[-1].append(line)
    return ["\n".join(x) for x in out]


def transcript(session: dict) -> str:
    return "\n".join(exchanges(session))


def record_session(store: Store, session: dict) -> list[str]:
    return [store.record(c, EpisodeType.CONTEXT, source="conversation").id
            for c in exchanges(session)]


def new_wrap_stats() -> dict:
    return {"compose_calls": 0, "saved": 0, "cancelled": 0, "refusals": [], "not_ready": [],
            "finish_not_stop": [], "graduations_demoted": 0, "chars": []}


def wrap_session(store: Store, client: Client, stats: dict, where: dict | None = None) -> None:
    """One real anneal wrap: prepare_wrap -> LLM compose -> validated_save_continuity. A refused
    save is fed back once; a second refusal cancels the wrap (the episodes roll into the next)."""
    wrap = prepare_wrap(store)
    if wrap["status"] != "ready":
        stats["not_ready"].append(wrap["status"])
        return
    prompt = format_wrap_package_text(wrap)
    instructions = wrap["package"]["instructions"]
    if not prompt.startswith(instructions):
        raise RuntimeError("wrap prompt no longer opens with package['instructions']")
    msg = {"role": "user", "content": prompt}
    head = max(instructions.find(CACHE_HEAD_ANCHOR), 0)
    if head:
        msg["cache_prefix"] = head
    messages = [msg]
    for attempt in range(2):
        text, finish = client.chat(messages, role="wrap", max_completion_tokens=WRAP_MAX)
        stats["compose_calls"] += 1
        if finish != "stop":
            stats["finish_not_stop"].append(finish)
        try:
            if finish != "stop":
                raise ValueError(f"the output was cut off (finish reason {finish!r}); nothing was saved")
            res = validated_save_continuity(store, text, wrap_token=wrap["wrap_token"])
        except ValueError as e:  # includes SaveAuthorityError and the shrink gate
            m = str(e)
            stats["refusals"].append(m[:300])
            stats.setdefault("refusal_log", []).append(
                {**(where or {}), "attempt": attempt, "finish": finish, "message": m[:300],
                 "recovered": attempt == 0})
            if attempt == 0:
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "The save was refused:\n" + m
                              + "\n\nReturn the corrected full continuity file. Return ONLY the markdown."}]
                continue
            store.wrap_cancelled(expect_token=wrap["wrap_token"])
            stats["cancelled"] += 1
            return
        stats["saved"] += 1
        stats["graduations_demoted"] += res.get("graduations_demoted", 0)
        stats["chars"].append(res.get("chars", 0))
        return


def probe_update(memory: str, session: dict, client: Client, stats: dict) -> str:
    conv = "\n".join(f"[{'USER' if t['role'] == 'user' else 'ASSISTANT'}]: {t['content']}"
                     for t in session["turns"])
    text, finish = client.chat(
        [{"role": "system", "content": PROBE_UPDATER_SYSTEM},
         {"role": "user", "content": PROBE_UPDATER_USER.format(current_memory=memory, conversation=conv)}],
        role="probe-update", max_completion_tokens=PROBE_UPDATE_MAX)
    stats["compose_calls"] += 1
    if finish != "stop":
        stats["finish_not_stop"].append(finish)
    lines = text.strip("\n").split("\n")[:200]
    out = "\n".join(lines)
    raw = out.encode()
    if len(raw) > 25000:
        out = raw[:25000].decode(errors="ignore")
    return out


def answer(client: Client, memory: str, query: str) -> str:
    text, _ = client.chat([{"role": "system", "content": ANSWER_SYSTEM.format(memory=memory or NO_MEMORY)},
                           {"role": "user", "content": query}],
                          role="answer", max_completion_tokens=ANSWER_MAX)
    return text


def parse_json_obj(text: str) -> dict | None:
    """The LAST JSON object in a judge's text (a judge that self-corrects emits two)."""
    dec, last, i = json.JSONDecoder(), None, 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        i = end
        if isinstance(obj, dict):
            last = obj
    return last


def judge_json(client: Client, system: str, user: str) -> dict:
    text, _ = client.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                          role="judge", max_completion_tokens=JUDGE_MAX)
    parsed = parse_json_obj(text) or {}
    if "score" not in parsed:   # the judge sometimes closes its JSON badly; read the stated fields
        m = re.search(r'"score"\s*:\s*([01])', text)
        if m:
            parsed["score"] = int(m.group(1))
            r = re.search(r'"raises_evidence"\s*:\s*([01])', text)
            if r:
                parsed["raises_evidence"] = int(r.group(1))
    return {"parsed": parsed or None, "raw": text}


# --- statistics ------------------------------------------------------------------------------

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def cluster_bootstrap(rows: list[dict], key_fn, cluster_fn, reps: int = 4000, seed: int = 7):
    """95% CI of the mean of key_fn(row) resampling whole clusters (scenarios, not units)."""
    by: dict = {}
    for r in rows:
        v = key_fn(r)
        if v is not None:
            by.setdefault(cluster_fn(r), []).append(v)
    clusters = list(by.values())
    if not clusters:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        pick = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        tot = sum(sum(c) for c in pick)
        cnt = sum(len(c) for c in pick)
        stats.append(tot / cnt)
    stats.sort()
    allv = [v for c in clusters for v in c]
    return {"mean": sum(allv) / len(allv), "lo": stats[int(0.025 * reps)], "hi": stats[int(0.975 * reps) - 1],
            "n": len(allv), "clusters": len(clusters)}


def paired_diff(rows: list[dict], a_fn, b_fn, cluster_fn, reps: int = 4000, seed: int = 7):
    """Cluster bootstrap of mean(a - b) where both are defined on the same row."""
    return cluster_bootstrap(
        rows, lambda r: (None if a_fn(r) is None or b_fn(r) is None else a_fn(r) - b_fn(r)),
        cluster_fn, reps, seed)


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]
