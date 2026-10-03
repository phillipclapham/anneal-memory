#!/usr/bin/env python3
"""InMind eval harness for anneal-memory (arXiv 2607.24368, github.com/imlrz/InMind).

Conditions, all on the same tasks with the same reader and judge:
  none               reader gets no memory (floor)
  oracle             target user/assistant pair is the whole context (paper's backbone control)
  anneal-recall      every session recorded as episodes; context = retrieve_relevant(query)
  anneal-agentic     every session recorded as episodes; NO context is preloaded. The reader gets
                     anneal's real MCP `recall` tool (the server's own handler and schema) and may
                     call it up to MAX_TOOL_ROUNDS times before answering, the way an MCP client
                     uses it. The judge's context = every tool result the reader saw.
  anneal-continuity  prepare_wrap -> LLM compose -> validated_save_continuity after every
                     session; context = the continuity file, whole
  paper-probe        the paper's always-in-state probe (Appendix 17), re-implemented from its
                     printed prompts: one markdown file rewritten by the updater after every
                     session, truncated to 200 lines / 25,000 bytes, prepended whole

Run from the repo root, e.g.:
  ANNEAL_BENCH_ENV_FILE=path/to/.env PYTHONPATH=. python bench/inmind/run.py \
      --n 10 --seed 0 --budget 2.0 --out RUN_DIR
(or export OPENAI_API_KEY instead of ANNEAL_BENCH_ENV_FILE; see oai.py).
Tuning on Ollama (no key, no cost, TUNING-ONLY numbers): add --model gpt-oss:120b-cloud.
An Ollama usage/limit error stops the run at once (exit 3); it is never retried.

The InMind repo is expected at ~/.cache/anneal-bench/inmind/InMind (git clone
https://github.com/imlrz/InMind). Nothing from it is copied into this repo.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Which anneal-memory is measured: $ANNEAL_SRC names a source tree to import; set it EMPTY to
# use the installed package (e.g. a release in a clean venv); unset = this repo's tree. The run
# records anneal_memory.__file__ and __version__ in summary.json either way.
_src = os.environ.get("ANNEAL_SRC")
if _src is None:
    sys.path.insert(0, str(HERE.parents[1]))
elif _src:
    sys.path.insert(0, _src)

import anneal_memory  # noqa: E402
from anneal_memory import EpisodeType, Store, prepare_wrap, validated_save_continuity  # noqa: E402
from anneal_memory.continuity import format_wrap_package_text  # noqa: E402
from anneal_memory.retrieval import retrieve_relevant  # noqa: E402
from anneal_memory.server import TOOLS as MCP_TOOLS, Server as McpServer  # noqa: E402

from oai import (PRICES, PRICES_SOURCE, BudgetExceeded, Client, Ledger, UsageLimit,  # noqa: E402
                 is_agy, is_free_tier, is_ollama)

DEFAULT_INMIND = Path.home() / ".cache/anneal-bench/inmind/InMind"
MODEL = "gpt-5-mini"
ANSWER_MAX = 16384   # paper / repo protocol
JUDGE_MAX = 4096     # paper / repo protocol
WRAP_MAX = 16384     # the paper's updater uses 8,192; anneal's 20,000-char budget plus reasoning needs more
CONDITIONS = ("none", "oracle", "anneal-recall", "anneal-agentic", "anneal-continuity",
              "paper-probe")
STATEFUL = ("anneal-recall", "anneal-agentic", "anneal-continuity", "paper-probe")
MAX_TOOL_ROUNDS = 6      # agentic: tool-calling turns before the reader must answer
TOOL_RESULT_MAX = 20000  # agentic: chars of one recall result passed back (MCP recall's
                         # default limit is 100 episodes, which can exceed a reader's window)
AGENTIC_CONTEXT = (
    "No memory is preloaded. You have a `recall` tool that searches the user's past "
    "conversations with you; call it as many times as you need before answering."
)
PROBE_UPDATE_MAX = 8192  # paper Appendix 17

# Paper Appendix 17, verbatim from the arXiv HTML (2607.24368v1).
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
PROBE_ANSWER_SYSTEM = (
    "You are a helpful personal assistant with access to the user\u2019s personal memory. Use "
    "this memory to personalize your responses. If the memory contains relevant information, "
    "incorporate it naturally without explicitly referencing \u201cmy memory file.\u201d\n"
    "--- USER\u2019S PERSONAL MEMORY ---\n{memory}\n--- END OF MEMORY ---"
)
PROJECT_NAME = "Assistant"
TUNING_ONLY = (
    "TUNING-ONLY, NOT COMPARABLE TO THE PAPER: reader, judge and composer are {model} "
    "(Ollama or agy, quota-billed at $0), not gpt-5-mini. Use these numbers to tune and to "
    "compare conditions within one run, never against the paper's figures."
)

COMPARABILITY = (
    "Comparability: reader and judge are gpt-5-mini with the repo's answer and judge prompts, "
    "token limits and no sampling override, and the canonical middle-injection timeline, as in "
    "the paper; but these are fresh API samples (no seed control), a subset of the 125 tasks, and "
    "anneal-continuity's composer is anneal's own wrap prompt, not the paper's updater prompt. "
    "paper-probe is our re-implementation of the paper's probe from its printed prompts (session "
    "passed whole to the updater; the probe's own answerer prompt), not the authors' code."
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inmind(root: Path):
    """Load tasks, background and manifest through the repo's own scripts; verify hashes."""
    tl = _load_module("inmind_build_timeline", root / "evaluation/scripts/build_timeline.py")
    jp = _load_module("inmind_build_judge", root / "evaluation/scripts/build_judge_payloads.py")
    tasks_path = root / "benchmark/dataset/inmind.jsonl"
    bg_path = root / "evaluation/background/lme_s_background.jsonl"
    manifest = json.loads((root / "evaluation/background/manifest.json").read_text())
    want_tasks = (root / "benchmark/dataset/SHA256SUMS").read_text().split()[0]
    if sha256(tasks_path) != want_tasks:
        raise SystemExit("inmind.jsonl hash does not match SHA256SUMS")
    if sha256(bg_path) != manifest["sha256"]:
        raise SystemExit("background hash does not match manifest.json")
    tasks = {int(t["task_id"]): t for t in tl.read_jsonl(tasks_path)}
    background = tl.read_jsonl(bg_path)
    prompts = {
        "answer": (root / "evaluation/prompts/answer_system.txt").read_text(),
        **{m: p.read_text().rstrip() for m, p in jp.PROMPTS.items()},
    }
    return tl, jp, tasks, background, manifest, prompts, want_tasks


# --- memory-side helpers -------------------------------------------------------

def exchanges(session: dict) -> list[str]:
    """Group a session's turns into exchanges: one user turn plus the assistant turn(s)
    that follow it. One exchange = one anneal episode. (Per-turn episodes would drop
    short targets: retrieve_relevant skips episodes under MIN_EPISODE_LEN=80 chars.)"""
    out: list[list[str]] = []
    for turn in session["turns"]:
        label = "User" if turn["role"] == "user" else "Assistant"
        line = f"{label}: {turn['content']}"
        if turn["role"] == "user" or not out:
            out.append([line])
        else:
            out[-1].append(line)
    return ["\n".join(x) for x in out]


def record_session(store: Store, session: dict) -> list[str]:
    ids = []
    for content in exchanges(session):
        ep = store.record(content, EpisodeType.CONTEXT, source="conversation")
        ids.append(ep.id)
    return ids


def target_text(task: dict) -> str:
    return f"User: {task['user_message']}\nAssistant: {task['assistant_message']}"


def wrap_session(store: Store, client: Client, stats: dict) -> None:
    """One real anneal wrap: prepare_wrap -> LLM compose -> validated_save_continuity.
    A refused save is fed back to the composer once (what an MCP/CLI agent sees);
    a second refusal cancels the wrap, so the episodes roll into the next wrap."""
    wrap = prepare_wrap(store)
    if wrap["status"] != "ready":
        stats["not_ready"].append(wrap["status"])
        return
    prompt = format_wrap_package_text(wrap)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        text, finish = client.chat(messages, role="wrap", max_completion_tokens=WRAP_MAX)
        stats["compose_calls"] += 1
        if finish != "stop":
            stats["finish_not_stop"].append(finish)
        try:
            res = validated_save_continuity(store, text, wrap_token=wrap["wrap_token"])
        except ValueError as e:  # includes SaveAuthorityError and the shrink gate
            msg = str(e)
            stats["refusals"].append(msg[:300])
            if attempt == 0:
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "The save was refused:\n" + msg
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
    """Update(M, C_i) then Truncate, per the paper's Algorithm 1. The printed template
    shows one [USER]/[ASSISTANT] pair; a session is passed as all of its turns in order."""
    conv = "\n".join(f"[{'USER' if t['role'] == 'user' else 'ASSISTANT'}]: {t['content']}"
                     for t in session["turns"])
    text, finish = client.chat(
        [{"role": "system", "content": PROBE_UPDATER_SYSTEM},
         {"role": "user", "content": PROBE_UPDATER_USER.format(current_memory=memory,
                                                               conversation=conv)}],
        role="probe-update", max_completion_tokens=PROBE_UPDATE_MAX)
    stats["compose_calls"] += 1
    if finish != "stop":
        stats["finish_not_stop"].append(finish)
    lines = text.strip("\n").split("\n")
    if len(lines) > 200:
        stats["truncated"] = stats.get("truncated", 0) + 1
    out = "\n".join(lines[:200])
    raw = out.encode()
    if len(raw) > 25000:
        out = raw[:25000].decode(errors="ignore")
        stats["truncated"] = stats.get("truncated", 0) + 1
    return out


def build_prefix(sessions_prefix: list[dict], work: Path, client: Client | None,
                 conds: list[str]) -> dict:
    """Shared immutable phase-A bank (sessions 1-8), allowed by the protocol."""
    out = {}
    for cond in [c for c in STATEFUL if c in conds]:
        d = work / "prefix" / cond
        if (d / "done.json").exists():
            out[cond] = json.loads((d / "done.json").read_text())
            continue
        if cond in ("anneal-continuity", "paper-probe") and client is None:
            continue
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        stats = new_wrap_stats()
        if cond == "paper-probe":
            memory = ""
            for s in sessions_prefix:
                memory = probe_update(memory, s, client, stats)
            (d / "memory.md").write_text(memory)
            (d / "done.json").write_text(json.dumps(stats))
            out[cond] = stats
            continue
        store = Store(d / "m.db", project_name=PROJECT_NAME, audit=False)
        try:
            for s in sessions_prefix:
                record_session(store, s)
                if cond == "anneal-continuity":
                    wrap_session(store, client, stats)
        finally:
            store.close()
        (d / "done.json").write_text(json.dumps(stats))
        out[cond] = stats
    return out


def new_wrap_stats() -> dict:
    return {"compose_calls": 0, "saved": 0, "cancelled": 0, "refusals": [], "not_ready": [],
            "finish_not_stop": [], "graduations_demoted": 0, "chars": []}


def run_memory_task(cond: str, timeline: dict, task: dict, inj: int, work: Path,
                    client: Client) -> dict:
    """Replay sessions inj..end into a per-task copy of the prefix store, then read
    the context each query would see. Returns the context(s) plus diagnostics."""
    tdir = work / "tasks" / cond / str(task["task_id"])
    shutil.rmtree(tdir, ignore_errors=True)
    shutil.copytree(work / "prefix" / cond, tdir)
    (tdir / "done.json").unlink(missing_ok=True)
    stats = new_wrap_stats()
    if cond == "paper-probe":
        memory = (tdir / "memory.md").read_text()
        post_inject = ""
        for i, s in enumerate(timeline["sessions"][inj:], start=inj):
            memory = probe_update(memory, s, client, stats)
            if i == inj:
                post_inject = memory
        (tdir / "final_memory.md").write_text(memory)
        return {"contexts": {"naive_query": memory, "query": memory},
                "diag": {"wrap": stats, "continuity_chars": len(memory),
                         "post_injection_state": post_inject,
                         "target_substring_in_continuity":
                             task["user_message"][:40].lower() in memory.lower()}}
    store = Store(tdir / "m.db", project_name=PROJECT_NAME, audit=False)
    target_ids: list[str] = []
    post_inject = ""
    try:
        for i, s in enumerate(timeline["sessions"][inj:], start=inj):
            ids = record_session(store, s)
            if i == inj:
                target_ids = ids[-1:]  # the injected pair is the session's last exchange
            if cond == "anneal-continuity":
                wrap_session(store, client, stats)
                if i == inj:
                    post_inject = store.load_continuity() or ""
        if cond == "anneal-agentic":
            # The reader queries the store itself at answer time (see answer_agentic).
            return {"contexts": None, "db": tdir / "m.db", "target_ids": target_ids}
        if cond == "anneal-recall":
            ctx, diag = {}, {}
            for key in ("naive_query", "query"):
                r = retrieve_relevant(store, None, task[key])
                ctx[key] = "\n\n".join(e.content for e in r.episodes)
                diag[key] = {"keywords": r.query_keywords,
                             "n_episodes": len(r.episodes),
                             "target_retrieved": any(e.id in target_ids for e in r.episodes)}
            return {"contexts": ctx, "diag": diag}
        text = store.load_continuity() or ""
        (tdir / "final_continuity.md").write_text(text)
        return {"contexts": {"naive_query": text, "query": text},
                "diag": {"wrap": stats, "continuity_chars": len(text),
                         "post_injection_state": post_inject,
                         "target_substring_in_continuity":
                             task["user_message"][:40].lower() in text.lower()}}
    finally:
        store.close()


# --- answer + judge --------------------------------------------------------------

def answer(client: Client, prompts: dict, context: str, query: str, cond: str = "") -> str:
    if cond == "paper-probe":  # Appendix 17 gives the probe its own answerer prompt
        system = PROBE_ANSWER_SYSTEM.replace("{memory}", context)
    else:
        system = prompts["answer"].replace("{context}", context)
    text, _ = client.chat([{"role": "system", "content": system},
                           {"role": "user", "content": query}],
                          role="answer", max_completion_tokens=ANSWER_MAX)
    return text


AGY_MCP_SHIM = HERE / "agy_anneal_mcp.py"
AGY_AGENTIC_CONTEXT = (
    "No memory is preloaded. You have a `recall` tool (MCP server `anneal_bench`) that "
    "searches the user's past conversations with you; call it as many times as you need "
    "before answering. Use no other tool."
)


def answer_agentic_agy(client: Client, prompts: dict, db: Path, query: str,
                       target_ids: list[str]) -> tuple[str, str, dict]:
    """The agentic reader under agy: one `agy -p` turn whose only tool is anneal's real MCP
    `recall`, served by the recall-only shim on this task's store (agy's global MCP entry
    `anneal_bench` runs it only when ANNEAL_BENCH_DB is set). The judge's context is every
    recall result the shim logged, in order."""
    from oai import _flatten  # noqa: PLC0415
    system = prompts["answer"].replace("{context}", AGY_AGENTIC_CONTEXT)
    log = db.parent / f"recall_log.{abs(hash(query))}.jsonl"
    log.unlink(missing_ok=True)
    src = os.environ.get("ANNEAL_SRC")
    env = {"ANNEAL_BENCH_DB": str(db), "ANNEAL_BENCH_PY": sys.executable,
           "ANNEAL_BENCH_LOG": str(log)}
    if src:
        env["PYTHONPATH"] = src
    text, usage = client.agy_run(_flatten([{"role": "system", "content": system},
                                           {"role": "user", "content": query}]),
                                 role="answer", env_extra=env, allow_tools=True)
    calls: list[dict] = []
    seen: list[str] = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out = rec.get("text", "")
            if len(out) > TOOL_RESULT_MAX:
                out = out[:TOOL_RESULT_MAX] + "\n[truncated]"
            calls.append({"args": rec.get("args", {}), "chars": len(out),
                          "no_match": out.startswith("No matching")})
            seen.append(f"[recall {json.dumps(rec.get('args', {}), ensure_ascii=False)}]\n{out}")
    context = "\n\n".join(seen)
    diag = {"tool_calls": calls, "n_calls": len(calls),
            "target_retrieved": any(t in context for t in target_ids),
            "agy_usage": usage}
    return text, context, diag


def answer_agentic(client: Client, prompts: dict, db: Path, query: str,
                   target_ids: list[str]) -> tuple[str, str, dict]:
    """The reader answers with anneal's MCP ``recall`` as a tool. Returns (answer, context,
    diag), where context is every tool result the reader saw, in order (what the judge
    grades for target-recall), and diag records each call."""
    if client.agy:
        return answer_agentic_agy(client, prompts, db, query, target_ids)
    spec = next(t for t in MCP_TOOLS if t["name"] == "recall")
    tools = [{"type": "function", "function": {"name": "recall",
                                               "description": spec["description"],
                                               "parameters": spec["inputSchema"]}}]
    system = prompts["answer"].replace("{context}", AGENTIC_CONTEXT)
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": query}]
    calls: list[dict] = []
    seen: list[str] = []
    text = ""
    store = Store(db, project_name=PROJECT_NAME, audit=False)
    try:
        server = McpServer(store)
        for rnd in range(MAX_TOOL_ROUNDS + 1):
            offer = rnd < MAX_TOOL_ROUNDS
            msg, _finish = client.chat_message(messages, role="answer",
                                               max_completion_tokens=ANSWER_MAX,
                                               tools=tools if offer else None)
            tcs = msg.get("tool_calls") or []
            if not tcs or not offer:
                text = msg.get("content") or ""
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": tcs})
            for tc in tcs:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                if fn.get("name") == "recall":
                    # Through the server's tools/call dispatch, so a bad argument comes
                    # back as the MCP error result a real client sees, not an exception.
                    res = server._handle_tools_call({"name": "recall", "arguments": args})
                    out = "".join(c.get("text", "") for c in res.get("content", []))
                else:
                    out = f"Unknown tool: {fn.get('name')!r}. The only tool is `recall`."
                if len(out) > TOOL_RESULT_MAX:
                    out = out[:TOOL_RESULT_MAX] + "\n[truncated]"
                calls.append({"args": args, "chars": len(out),
                              "no_match": out.startswith("No matching")})
                seen.append(f"[recall {json.dumps(args, ensure_ascii=False)}]\n{out}")
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": out})
    finally:
        store.close()
    context = "\n\n".join(seen)
    diag = {"tool_calls": calls, "n_calls": len(calls),
            "target_retrieved": any(t in context for t in target_ids)}
    return text, context, diag


_SCORE_RE = re.compile(r'"score"\s*:\s*([01])')


def judge(client: Client, jp, prompts: dict, metric: str, task: dict, row: dict) -> dict:
    payload = jp.user_payload(metric, task, row)
    text, _ = client.chat([{"role": "system", "content": prompts[metric]},
                           {"role": "user", "content": payload}],
                          role="judge", max_completion_tokens=JUDGE_MAX)
    score = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            score = int(json.loads(m.group(0)).get("score"))
        except (ValueError, TypeError, AttributeError):
            pass
    if score is None:
        m2 = _SCORE_RE.search(text)
        score = int(m2.group(1)) if m2 else None
    return {"score": score, "raw": text}


def metrics_for(cond: str) -> list[str]:
    base = ["target-recall", "application", "answer-only"]
    return (["naive"] + base) if cond in STATEFUL else base


def evaluate(cond: str, task: dict, contexts: dict, client: Client, jp, prompts: dict,
             post_inject: str | None = None) -> dict:
    row = {"task_id": task["task_id"], "system": f"anneal-memory/{cond}",
           "config": {"answer_model": client.model, "judge_model": client.model,
                      "anneal_memory": anneal_memory.__version__}}
    if cond == "anneal-agentic":
        agentic = contexts["agentic"]
        for key, q in (("query", task["query"]), ("naive", task["naive_query"])):
            ans, ctx, d = answer_agentic(client, prompts, agentic["db"], q, agentic["target_ids"])
            row[key] = {"context": ctx, "answer": ans}
            agentic.setdefault("diag", {})["naive_query" if key == "naive" else "query"] = d
        row["judgements"] = {m: judge(client, jp, prompts, m, task, row) for m in metrics_for(cond)}
        return row
    q_ctx = contexts["query"]
    row["query"] = {"context": q_ctx, "answer": answer(client, prompts, q_ctx, task["query"], cond)}
    if "naive" in metrics_for(cond):
        n_ctx = contexts["naive_query"]
        row["naive"] = {"context": n_ctx,
                        "answer": answer(client, prompts, n_ctx, task["naive_query"], cond)}
    else:
        row["naive"] = {"context": "", "answer": "", "not_evaluated": True}
    row["judgements"] = {m: judge(client, jp, prompts, m, task, row) for m in metrics_for(cond)}
    if post_inject is not None:
        # Write-vs-retention split: was the fact in the state right after the
        # injection session's update, before the 38 later sessions?
        probe_row = {"query": {"context": post_inject, "answer": ""}}
        row["judgements"]["target-recall@inject"] = judge(
            client, jp, prompts, "target-recall", task, probe_row)
    return row


# --- driver ------------------------------------------------------------------------

def pick_tasks(tasks: dict, args) -> list[int]:
    if args.tasks:
        ids = [int(x) for x in args.tasks.split(",")]
        missing = [i for i in ids if i not in tasks]
        if missing:
            raise SystemExit(f"unknown task ids: {missing}")
        return ids
    ids = sorted(tasks)
    random.Random(args.seed).shuffle(ids)
    return sorted(ids[: args.n])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inmind", type=Path, default=DEFAULT_INMIND)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", help="comma-separated task_ids (overrides --n/--seed)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--budget", type=float, required=True, help="hard USD ceiling for this run")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=MODEL,
                    help="reader/judge/composer model; an Ollama name (e.g. gpt-oss:120b-cloud) "
                         "runs free and labels every number TUNING-ONLY")
    args = ap.parse_args()

    conds = [c for c in args.conditions.split(",") if c]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions: {bad}")
    tl, jp, tasks, background, manifest, prompts, tasks_sha = load_inmind(args.inmind)
    ids = pick_tasks(tasks, args)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    ledger = Ledger()
    client = Client(args.model, ledger, args.budget)
    tuning = is_free_tier(args.model)
    inj = int(manifest["injection_session_index"])
    print(f"anneal_memory from {anneal_memory.__file__} ({anneal_memory.__version__})")
    print(f"tasks ({len(ids)}): {ids}", flush=True)
    t0 = time.time()

    sessions_prefix = tl.build_timeline(tasks[ids[0]], background, manifest)["sessions"][:inj]
    llm_prefix = any(c in conds for c in ("anneal-continuity", "paper-probe"))
    prefix_stats = build_prefix(sessions_prefix, work, client if llm_prefix else None, conds)
    print(f"prefix built ({time.time() - t0:.0f}s, ${ledger.total_usd():.3f})", flush=True)

    rows: dict[str, list[dict]] = {c: [] for c in conds}
    errors: list[dict] = []

    def one(cond: str, tid: int) -> dict:
        task = tasks[tid]
        diag: dict = {}
        if cond == "none":
            contexts = {"naive_query": "", "query": ""}
        elif cond == "oracle":
            contexts = {"naive_query": target_text(task), "query": target_text(task)}
        else:
            timeline = tl.build_timeline(task, background, manifest)
            mem = run_memory_task(cond, timeline, task, inj, work, client)
            if cond == "anneal-agentic":
                agentic = {"db": mem["db"], "target_ids": mem["target_ids"]}
                row = evaluate(cond, task, {"agentic": agentic}, client, jp, prompts)
                row["diag"] = agentic.get("diag", {})
                return row
            contexts, diag = mem["contexts"], mem["diag"]
        row = evaluate(cond, task, {"naive_query": contexts["naive_query"],
                                    "query": contexts["query"]}, client, jp, prompts,
                       diag.get("post_injection_state"))
        row["diag"] = diag
        return row

    jobs = [(c, t) for c in conds for t in ids]
    stopped: str | None = None
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, c, t): (c, t) for c, t in jobs}
        for f in cf.as_completed(futs):
            c, t = futs[f]
            try:
                rows[c].append(f.result())
                print(f"  done {c} task {t} (${ledger.total_usd():.3f})", flush=True)
            except UsageLimit as e:
                # Stop the whole run: cancel everything not yet started.
                stopped = str(e)
                errors.append({"condition": c, "task_id": t, "error": f"usage-limit: {e}"})
                print(f"  OLLAMA USAGE LIMIT, STOPPING THE RUN: {e}", flush=True)
                for other in futs:
                    other.cancel()
            except cf.CancelledError:
                errors.append({"condition": c, "task_id": t, "error": "cancelled after usage limit"})
            except BudgetExceeded as e:
                errors.append({"condition": c, "task_id": t, "error": f"budget: {e}"})
            except Exception as e:  # recorded, never silently dropped
                errors.append({"condition": c, "task_id": t, "error": repr(e),
                               "trace": traceback.format_exc()[-2000:]})
                print(f"  ERROR {c} task {t}: {e!r}", flush=True)

    for c in conds:
        rows[c].sort(key=lambda r: r["task_id"])
        with (out / f"results.{c}.jsonl").open("w") as fh:
            for r in rows[c]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(rows, conds, ids, tasks)
    summary.update({
        "anneal_memory_file": anneal_memory.__file__,
        "anneal_memory_version": anneal_memory.__version__,
        "inmind_tasks_sha256": tasks_sha,
        "model": args.model, "prices_usd_per_1M": PRICES.get(args.model, "Ollama: $0, quota-limited"),
        "prices_source": PRICES_SOURCE if not tuning else (
            "agy (Antigravity CLI), Google AI Pro quota" if is_agy(args.model)
            else "Ollama cloud via the local daemon"),
        "tuning_only": tuning, "stopped_on_usage_limit": stopped,
        "agentic_condition": (
            f"agentic, capped {os.environ.get('ANNEAL_BENCH_MAX_CALLS', '6')}x"
            f"{os.environ.get('ANNEAL_BENCH_MAX_LIMIT', '10')} (recall calls per answer x "
            "episodes per call, no paging; agy MCP shim)" if is_agy(args.model)
            else f"agentic, up to {MAX_TOOL_ROUNDS} tool rounds (Ollama tool calls)"),
        "judge_output": "agy --json-schema {score: 0|1, reason}" if is_agy(args.model) else "free text",
        "cost": ledger.summary(), "prefix_wrap_stats": prefix_stats,
        "errors": errors, "elapsed_s": round(time.time() - t0, 1), "task_ids": ids,
        "comparability": TUNING_ONLY.format(model=args.model) if tuning else COMPARABILITY,
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print_table(summary)
    if stopped:
        raise SystemExit(3)
    if errors:  # a run with failed jobs must not exit in the shape of success
        raise SystemExit(2)


def summarize(rows: dict, conds: list[str], ids: list[int], tasks: dict) -> dict:
    table = {}
    for c in conds:
        rs = rows[c]
        entry: dict = {"n": len(rs)}
        for m in ("naive", "target-recall", "application", "answer-only", "target-recall@inject"):
            scores = [r["judgements"][m]["score"] for r in rs if m in r["judgements"]]
            if not scores:
                continue
            ok = [s for s in scores if s is not None]
            entry[m] = {"k": sum(ok), "n": len(ok), "unparsed": len(scores) - len(ok)}
        if c == "anneal-recall":
            entry["mechanical_target_retrieved"] = {
                q: sum(r["diag"][q]["target_retrieved"] for r in rs) for q in ("naive_query", "query")}
            entry["empty_context"] = {
                q: sum(r["diag"][q]["n_episodes"] == 0 for r in rs) for q in ("naive_query", "query")}
        if c == "anneal-agentic":
            entry["mechanical_target_retrieved"] = {
                q: sum(r["diag"][q]["target_retrieved"] for r in rs) for q in ("naive_query", "query")}
            entry["tool_calls"] = {
                q: sum(r["diag"][q]["n_calls"] for r in rs) for q in ("naive_query", "query")}
            entry["no_call_answers"] = {
                q: sum(r["diag"][q]["n_calls"] == 0 for r in rs) for q in ("naive_query", "query")}
            entry["no_match_calls"] = {
                q: sum(c["no_match"] for r in rs for c in r["diag"][q]["tool_calls"])
                for q in ("naive_query", "query")}
        if c in ("anneal-continuity", "paper-probe"):
            w = [r["diag"]["wrap"] for r in rs]
            entry["wraps"] = {k: sum(x[k] for x in w) for k in ("compose_calls", "saved", "cancelled")}
            entry["wraps"]["refusals"] = sum(len(x["refusals"]) for x in w)
            entry["wraps"]["finish_not_stop"] = sum(len(x["finish_not_stop"]) for x in w)
            entry["wraps"]["truncated"] = sum(x.get("truncated", 0) for x in w)
            entry["target_substring_in_continuity"] = sum(
                r["diag"]["target_substring_in_continuity"] for r in rs)
            entry["final_continuity_chars"] = [r["diag"]["continuity_chars"] for r in rs]
        table[c] = entry
    return {"table": table}


def print_table(s: dict) -> None:
    print()
    print("InMind (arXiv 2607.24368) x anneal-memory -- dataset: released repo, NOT reconstructed")
    print(f"model {s['model']} (reader + judge); tasks {s['task_ids']}")
    if s.get("tuning_only"):
        print("*** TUNING-ONLY: NOT COMPARABLE TO THE PAPER ***")
    print(f"{'condition':<19}{'n':>3}  {'naive':>9}  {'target-recall':>13}  {'application':>11}  "
          f"{'answer-only':>11}  {'recall@inject':>13}")
    for c, e in s["table"].items():
        def cell(m):
            if m not in e:
                return "-"
            v = e[m]
            return f"{v['k']}/{v['n']}" + (f" ({v['unparsed']}?)" if v["unparsed"] else "")
        print(f"{c:<19}{e['n']:>3}  {cell('naive'):>9}  {cell('target-recall'):>13}  "
              f"{cell('application'):>11}  {cell('answer-only'):>11}  "
              f"{cell('target-recall@inject'):>13}")
    tot = s["cost"]["total"]
    print(f"\ntokens: prompt {tot['prompt']:,} (cached {tot['cached']:,}), completion "
          f"{tot['completion']:,} (reasoning {tot['reasoning']:,}); cost ${tot['usd']:.3f} "
          f"at {s['prices_usd_per_1M']} per 1M ({s['prices_source']})")
    if s["errors"]:
        print(f"ERRORS: {len(s['errors'])} (see summary.json)")
    print(s["comparability"])


if __name__ == "__main__":
    main()
