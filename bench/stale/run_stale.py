"""STALE (arXiv 2605.06527) with anneal-memory as the memory. INTERNAL numbers.

Same items, same reader (GPT-4o-mini, the paper's memory-framework backbone), same
answer and judge prompts (copied verbatim in ``prompts.py``) across conditions:

``none``            the reader gets an empty history (floor: what the judge passes
                    with no memory at all).
``evidence``        only the old and new evidence sessions, with their timestamps
                    (sanity row: no distractors).
``full``            the whole haystack, trimmed to GPT-4o-mini's 128K window with the
                    paper's evidence-preserving trim (calibration against the paper's
                    GPT-4o-mini* row).
``anneal@K``        every haystack turn is one episode in a fresh temp Store,
                    timestamped from its session; the history is
                    ``retrieve_relevant(store, None, query, max_episodes=K)``.
``anneal+sup@K``    the same, after a wrap-style pass per session in which the reader
                    model proposes ``[supersedes: OLD by NEW]`` links that anneal's own
                    wrap path (``_record_wrap_supersessions``) validates and records.

Data: https://huggingface.co/datasets/STALEproj/STALE, ``T1_T2_400_FULL.json``
(CC BY 4.0), cached under ``~/.cache/anneal-bench/stale/``; never committed.

Run from the repo root so the repo's ``anneal_memory`` is imported::

    PYTHONPATH=. python bench/stale/run_stale.py --run pilot --n-per-type 5 --max-usd 2
    PYTHONPATH=. python bench/stale/run_stale.py --diag-oracle

Everything is cached per run directory (reader answers, judge verdicts, wrap
outputs), so a re-run with a larger ``--n-per-type`` only pays for the new items.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anneal_memory  # noqa: E402
from anneal_memory import continuity as _continuity  # noqa: E402
from anneal_memory.retrieval import retrieve_relevant  # noqa: E402
from anneal_memory.store import Store, _supersession_grounds  # noqa: E402

import prompts  # noqa: E402
from llm import Client, Ledger, load_key  # noqa: E402

CACHE = Path.home() / ".cache" / "anneal-bench" / "stale"
DATA_URL = "https://huggingface.co/datasets/STALEproj/STALE/resolve/main/T1_T2_400_FULL.json"
READER = "gpt-4o-mini"
JUDGE = "gpt-5.4-mini"  # with --judge-effort low; the pilot compared three judges
JUDGE_EFFORT = "low"
DIMS = ("dim1", "dim2", "dim3")
DIM_NAME = {"dim1": "SR", "dim2": "PR", "dim3": "IPA"}
TODAY = date(2026, 10, 3)  # fixed for determinism; no crystals exist, so unused in effect

# The paper's trim budget (run_target_model.py): 128000 - 2048 - 512 input tokens.
TRIM_INPUT_LIMIT = 128000 - 2048 - 512
# No tokenizer in the venv, so tokens are estimated from characters. Measured on this
# data (gpt-4o-mini usage.prompt_tokens): 4.25-4.8 chars per token. 4.0 sits below that,
# so a trimmed request never exceeds the window; the cost is that it over-trims (pilot:
# ~104K actual tokens of the 125,440 budget, 13-17 of 50 sessions removed).
CHARS_PER_TOKEN = 4.0

# Wrap-pass candidate retrieval: older user-turn episodes recalled per new user turn.
SUP_CANDIDATES_PER_TURN = 3


# --------------------------------------------------------------------------- data

def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"dataset missing: download {DATA_URL} to {path}")
    return json.loads(path.read_text())


def sample(items: list[dict], n_per_type: int, seed: int) -> list[dict]:
    """Prefix-stable stratified sample: a larger n contains every smaller n's items."""
    out = []
    for t in ("T1", "T2"):
        pool = sorted((i for i in items if i["type"] == t), key=lambda i: i["uid"])
        random.Random(f"{seed}:{t}").shuffle(pool)
        out += pool[:n_per_type]
    return out


def iso(ts: str, turn: int) -> str:
    """Session time ``YYYY-MM-DD HH:MM`` plus the turn index as seconds, in the
    Store's own timestamp format, so turns keep their order inside a session."""
    d, hm = ts.split(" ")
    return f"{d}T{hm}:{turn:02d}.000000Z"


# ------------------------------------------------------------------ anneal memory

class AnnealMemory:
    """One fresh temp Store per scenario. Episode i = one haystack turn."""

    def __init__(self, item: dict) -> None:
        self.item = item
        self.dir = Path(tempfile.mkdtemp(prefix="stale-anneal-"))
        self.store = Store(self.dir / "memory.db", audit=False, project_name="stale")
        self.meta: dict[str, tuple[int, int, str]] = {}  # id -> (session, turn, role)
        self.session_ids: list[list[str]] = []

    def close(self) -> None:
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def record_session(self, s: int) -> list[str]:
        ts = self.item["timestamps"][s]
        ids = []
        for j, turn in enumerate(self.item["haystack_session"][s]):
            if not turn["content"].strip():
                continue
            ep = self.store.record(turn["content"], "observation", source="stale-haystack",
                                   metadata={"session": s, "turn": j, "role": turn["role"]},
                                   timestamp=iso(ts, j))
            self.meta[ep.id] = (s, j, turn["role"])
            ids.append(ep.id)
        self.session_ids.append(ids)
        return ids

    def recall(self, query: str, k: int) -> tuple[str, dict]:
        res = retrieve_relevant(self.store, None, query, max_episodes=k,
                                associative=False, today=TODAY)
        old_s, new_s = self.item["relevant_session_index"]
        by_session: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        sessions_hit = []
        for ep in res.episodes:
            s, j, role = self.meta[ep.id]
            by_session[s].append((j, role, ep.content))
            sessions_hit.append(s)
        hist = ""
        for s in sorted(by_session):
            hist += f"\n=== Session {s + 1} [Time: {self.item['timestamps'][s]}] ===\n"
            for j, role, content in sorted(by_session[s]):
                hist += f"{'User' if role == 'user' else 'Assistant'}: {content}\n"
        diag = {
            "n": len(res.episodes),
            "keywords": res.query_keywords,
            "old_hit": old_s in sessions_hit,
            "new_hit": new_s in sessions_hit,
            "old_rank": sessions_hit.index(old_s) + 1 if old_s in sessions_hit else None,
            "new_rank": sessions_hit.index(new_s) + 1 if new_s in sessions_hit else None,
        }
        return hist, diag


def wrap_instruction() -> str:
    """anneal's own wrap instruction for supersession, read from the installed
    source so the prompt cannot drift from what the library tells a composer."""
    src = Path(_continuity.__file__).read_text()
    i = src.index("### Superseded facts")
    j = src.index("### Decisions", i)
    return src[i:j].strip()


SUP_SYSTEM = """You are composing a memory wrap for one conversation session with a user.
The memory stores each message as an episode with an 8-hex id. You are shown the
user's messages from THIS session and, under each, OLDER episodes the memory recalled
as possibly related. Your only task here is the supersession markers. The memory's
own instruction for them follows.

{instruction}

Output only marker lines of the form [supersedes: <old_id> by <new_id>], one per
line, where <new_id> is an episode of THIS session and <old_id> is one of the older
episodes shown. If nothing in this session replaces an older fact, output NONE."""


def supersede_pass(mem: AnnealMemory, client: Client, cache: "JsonlCache",
                   uid: str) -> dict:
    """Record every session; after each, one wrap call proposes links over that
    session's user turns and their recalled older user turns. Returns stats."""
    old_s, new_s = mem.item["relevant_session_index"]
    instruction = SUP_SYSTEM.format(instruction=wrap_instruction())
    stats = {"calls": 0, "proposed": 0, "recorded": 0, "rejected": 0,
             "reject_reasons": defaultdict(int), "oracle_links": 0,
             "links_into_evidence_old": 0, "candidate_pair_present": False}
    for s in range(len(mem.item["haystack_session"])):
        ids = mem.record_session(s)
        start = iso(mem.item["timestamps"][s], 0)
        blocks = []
        for eid in ids:
            if mem.meta[eid][2] != "user":
                continue
            content = mem.store.get(eid).content
            res = retrieve_relevant(mem.store, None, content, max_episodes=12,
                                    associative=False, today=TODAY)
            cands = [e for e in res.episodes
                     if e.timestamp < start and mem.meta[e.id][2] == "user"][:SUP_CANDIDATES_PER_TURN]
            if not cands:
                continue
            if s == new_s and any(mem.meta[c.id][0] == old_s for c in cands):
                stats["candidate_pair_present"] = True
            block = f"NEW [{eid}] (session {s + 1}): {content}\n"
            for c in cands:
                cs = mem.meta[c.id][0]
                block += f"  older [{c.id}] (session {cs + 1}, {mem.item['timestamps'][cs]}): {c.content}\n"
            blocks.append(block)
        if not blocks:
            continue
        key = f"{uid}:{s}"
        text = cache.get(key)
        if text is None:
            text, _ = client.chat(READER, [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "\n".join(blocks)},
            ], purpose="wrap", temperature=0)
            cache.put(key, text)
        stats["calls"] += 1
        markers = _continuity._SUPERSEDES_RE.findall(text)
        stats["proposed"] += len(markers)
        recorded, rejected = _continuity._record_wrap_supersessions(mem.store, text, set(ids))
        stats["recorded"] += recorded
        stats["rejected"] += len(rejected)
        for r in rejected:
            reason = re.sub(r"[0-9a-f]{8}", "<id>", r["reason"])
            reason = re.sub(r"\(\d{4}-[^)]*\)", "(<ts>)", reason)
            stats["reject_reasons"][reason[:120]] += 1
        for o, n in markers:
            o, n = o.lower(), n.lower()
            if not mem.store.supersession_exists(old_id=o, new_id=n):
                continue
            if o in mem.meta and mem.meta[o][0] == old_s:
                stats["links_into_evidence_old"] += 1
                if mem.meta.get(n, (None,))[0] == new_s:
                    stats["oracle_links"] += 1
    stats["reject_reasons"] = dict(stats["reject_reasons"])
    return stats


# ------------------------------------------------------------- other conditions

def full_trimmed(item: dict, query: str, dim_key: str) -> tuple[str, int]:
    """The paper's trim (run_target_model.trim_middle_noise_to_token_limit) with a
    character-based token estimate and a per-uid seeded RNG for phase 3."""
    sessions = list(item["haystack_session"])
    stamps = list(item["timestamps"])
    idx_old, idx_new = item["relevant_session_index"]
    rng = random.Random(f"{item['uid']}:{dim_key}")

    def est() -> int:
        sp, up = prompts.build_prompts(prompts.format_haystack(sessions, stamps), query, dim_key)
        return int((len(sp) + len(up)) / CHARS_PER_TOKEN) + 10

    removed = 0
    for _ in range(min(2, idx_old)):
        if est() <= TRIM_INPUT_LIMIT:
            break
        sessions.pop(0); stamps.pop(0); removed += 1; idx_old -= 1; idx_new -= 1
    for _ in range(min(2, len(sessions) - 1 - idx_new)):
        if est() <= TRIM_INPUT_LIMIT:
            break
        sessions.pop(); stamps.pop(); removed += 1
    middle = list(range(idx_old + 1, idx_new))
    while est() > TRIM_INPUT_LIMIT and middle:
        k = rng.choice(middle)
        sessions.pop(k); stamps.pop(k); removed += 1; idx_new -= 1
        middle = list(range(idx_old + 1, idx_new))
    return prompts.format_haystack(sessions, stamps), removed


def evidence_only(item: dict) -> str:
    o, n = item["relevant_session_index"]
    return prompts.format_haystack([item["haystack_session"][o], item["haystack_session"][n]],
                                   [item["timestamps"][o], item["timestamps"][n]])


# ----------------------------------------------------------------- caching/judge

class JsonlCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.data: dict[str, object] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                self.data[row["k"]] = row["v"]

    def get(self, k: str):
        return self.data.get(k)

    def put(self, k: str, v) -> None:
        with self._lock:
            self.data[k] = v
            with self.path.open("a") as f:
                f.write(json.dumps({"k": k, "v": v}) + "\n")


def judge(client: Client, item: dict, answers: dict[str, str], model: str,
          effort: str | None = None) -> dict:
    q = item["probing_queries"]
    user = prompts.judge_user_prompt(
        item.get("old_info", item.get("M_old", "")), item.get("M_new", ""),
        item.get("explanation", ""),
        q["dim1_query"], answers["dim1"], q["dim2_query"], answers["dim2"],
        q["dim3_query"], answers["dim3"])
    last = None
    for _ in range(3):
        content, _u = client.chat(model, [
            {"role": "system", "content": prompts.SYSTEM_PROMPT_ALL_IN_ONE_JUDGE},
            {"role": "user", "content": user},
        ], purpose="judge", response_format={"type": "json_object"},
            # Their call uses temperature=0; a reasoning model only takes the default.
            **({"reasoning_effort": effort} if effort and effort != "none" else {"temperature": 0}))
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
        if m:
            content = m.group(1)
        try:
            res = json.loads(content)
            return {d: bool(res.get(f"{d}_eval", {}).get("pass", False)) for d in DIMS} | {
                "raw": res}
        except json.JSONDecodeError as e:
            last = e
    return {d: False for d in DIMS} | {"error": str(last)}


# ------------------------------------------------------------------------ main

def run(args) -> None:
    print("anneal_memory imported from", anneal_memory.__file__, anneal_memory.__version__)
    items = sample(load_dataset(args.data), args.n_per_type, args.seed)
    rundir = CACHE / "runs" / args.run
    rundir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(rundir / "ledger.jsonl", args.max_usd)
    client = Client(load_key(), ledger)
    answers = JsonlCache(rundir / "reader.jsonl")
    jtag = args.judge + (f"@{args.judge_effort}" if args.judge_effort else "")
    verdicts = JsonlCache(rundir / f"judge-{jtag}.jsonl")
    wraps = JsonlCache(rundir / "wraps.jsonl")
    diags = JsonlCache(rundir / "diag.jsonl")
    conds = args.conditions.split(",")
    print(f"run={args.run} items={len(items)} conditions={conds} spent so far ${ledger.usd:.4f}")

    def answer(cond: str, item: dict, dim: str, history: str) -> None:
        key = f"{cond}|{item['uid']}|{dim}"
        if answers.get(key) is not None:
            return
        sp, up = prompts.build_prompts(history, item["probing_queries"][f"{dim}_query"],
                                       f"{dim}_query")
        text, usage = client.chat(READER, [{"role": "system", "content": sp},
                                           {"role": "user", "content": up}], purpose=cond)
        answers.put(key, {"text": text, "prompt_tokens": usage.get("prompt_tokens"),
                          "history_chars": len(history)})

    def prepare(item: dict) -> list[tuple[str, str, str]]:
        """The (condition, dim, history) jobs for one item. Builds its anneal
        stores (and runs its wrap pass) in the calling thread."""
        uid = item["uid"]
        jobs = []
        for cond in conds:
            if cond == "none":
                jobs += [(cond, d, "") for d in DIMS]
            elif cond == "evidence":
                h = evidence_only(item)
                jobs += [(cond, d, h) for d in DIMS]
            elif cond == "full":
                for d in DIMS:
                    if answers.get(f"{cond}|{uid}|{d}") is None:
                        h, removed = full_trimmed(item, item["probing_queries"][f"{d}_query"],
                                                  f"{d}_query")
                        diags.put(f"{cond}|{uid}|{d}", {"removed_sessions": removed})
                        jobs.append((cond, d, h))
        plain = [c for c in conds if c.startswith("anneal@")]
        sup = [c for c in conds if c.startswith("anneal+sup@")]
        for group in (plain, sup):
            if not group:
                continue
            mem = AnnealMemory(item)
            try:
                if group is sup:
                    st = supersede_pass(mem, client, wraps, uid)
                    diags.put(f"supstats|{uid}", st)
                    print(f"  sup {uid[:8]} {item['type']}: {json.dumps(st)}", flush=True)
                else:
                    for s in range(len(item["haystack_session"])):
                        mem.record_session(s)
                for c in group:
                    k = int(c.split("@")[1])
                    for d in DIMS:
                        h, dg = mem.recall(item["probing_queries"][f"{d}_query"], k)
                        diags.put(f"{c}|{uid}|{d}", dg)
                        jobs.append((c, d, h))
            finally:
                mem.close()
        return jobs

    with ThreadPoolExecutor(args.workers) as pool, ThreadPoolExecutor(args.item_workers) as ipool:
        futures = []
        for jobs, item in zip(ipool.map(prepare, items), items):
            futures += [pool.submit(answer, c, item, d, h) for c, d, h in jobs]
        for f in futures:
            f.result()

        def do_judge(cond: str, item: dict) -> None:
            key = f"{cond}|{item['uid']}"
            if verdicts.get(key) is not None:
                return
            a = {d: answers.get(f"{cond}|{item['uid']}|{d}")["text"] for d in DIMS}
            verdicts.put(key, judge(client, item, a, args.judge, args.judge_effort))

        for f in [pool.submit(do_judge, c, it) for it in items for c in conds]:
            f.result()

    summarize(items, conds, verdicts, diags, ledger, rundir, jtag)


def summarize(items, conds, verdicts, diags, ledger, rundir, judge_model) -> None:
    rows = {}
    print(f"\nreader={READER} judge={judge_model} items={len(items)} "
          f"(T1={sum(i['type'] == 'T1' for i in items)}, T2={sum(i['type'] == 'T2' for i in items)})")
    hdr = f"{'condition':<16}" + "".join(f"{t}-{DIM_NAME[d]:<4}" for t in ('T1', 'T2') for d in DIMS) + "  Overall  errs"
    print(hdr)
    for c in conds:
        cells = {}
        errs = 0
        for t in ("T1", "T2"):
            for d in DIMS:
                vs = [verdicts.get(f"{c}|{i['uid']}") for i in items if i["type"] == t]
                vs = [v for v in vs if v is not None]
                errs += sum("error" in v for v in vs) if d == "dim1" else 0
                cells[f"{t}-{DIM_NAME[d]}"] = (sum(v[d] for v in vs), len(vs))
        overall = sum(a / n for a, n in cells.values() if n) / max(1, sum(1 for a, n in cells.values() if n))
        rows[c] = {"cells": cells, "overall": overall, "judge_errors": errs}
        print(f"{c:<16}" + "".join(f"{a:>3}/{n:<4}" for a, n in cells.values())
              + f"  {overall * 100:5.1f}%  {errs}")
    # retrieval visibility for the anneal conditions (cf. the paper's Table 3)
    vis = {}
    for c in conds:
        if not c.startswith("anneal"):
            continue
        for d in DIMS:
            ds = [diags.get(f"{c}|{i['uid']}|{d}") for i in items]
            ds = [x for x in ds if x]
            if not ds:
                continue
            vis[f"{c}|{DIM_NAME[d]}"] = {
                "n": len(ds),
                "empty": sum(x["n"] == 0 for x in ds),
                "mean_returned": sum(x["n"] for x in ds) / len(ds),
                "new_hit": sum(x["new_hit"] for x in ds),
                "old_hit": sum(x["old_hit"] for x in ds),
                "both_hit": sum(x["new_hit"] and x["old_hit"] for x in ds),
                "old_top1": sum(x["old_rank"] == 1 for x in ds),
                "new_top1": sum(x["new_rank"] == 1 for x in ds),
            }
    if vis:
        print("\nretrieval visibility (counts over items):")
        for k, v in vis.items():
            print(f"  {k:<22} {json.dumps(v)}")
    sup = [diags.get(f"supstats|{i['uid']}") for i in items]
    sup = [s for s in sup if s]
    if sup:
        agg = defaultdict(int)
        reasons = defaultdict(int)
        for s in sup:
            for k, v in s.items():
                if isinstance(v, (int, bool)) and not isinstance(v, dict):
                    agg[k] += int(v)
            for r, n in s["reject_reasons"].items():
                reasons[r] += n
        print(f"\nsupersede pass over {len(sup)} items: {json.dumps(dict(agg))}")
        for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  rejected x{n}: {r}")
    print(f"\ntokens: prompt={ledger.tokens['prompt']:,} (cached {ledger.tokens['cached']:,}) "
          f"completion={ledger.tokens['completion']:,}  cost=${ledger.usd:.4f}")
    (rundir / f"summary-{judge_model}.json").write_text(json.dumps({
        "reader": READER, "judge": judge_model, "anneal_version": anneal_memory.__version__,
        "items": [i["uid"] for i in items], "rows": rows, "visibility": vis,
        "tokens": ledger.tokens, "usd": ledger.usd}, indent=1, default=str))


def diag_oracle(args) -> None:
    """No API. Would anneal's grounding floor admit the TRUE link? For every item:
    the user turn of the old session most like M_old against the user turn of the
    new session most like M_new, and also whether ANY user-turn pair grounds."""
    from anneal_memory.graduation import _meaningful_words

    items = load_dataset(args.data)
    by = defaultdict(lambda: [0, 0, 0])
    for it in items:
        o, n = it["relevant_session_index"]
        uo = [t["content"] for t in it["haystack_session"][o] if t["role"] == "user"]
        un = [t["content"] for t in it["haystack_session"][n] if t["role"] == "user"]

        def best(turns, ref):
            rw = _meaningful_words(ref)
            return max(turns, key=lambda t: len(_meaningful_words(t) & rw))

        b = by[it["type"]]
        b[0] += 1
        b[1] += _supersession_grounds(best(un, it["M_new"]), best(uo, it["M_old"]))
        b[2] += any(_supersession_grounds(x, y) for x in un for y in uo)
    for t, (tot, best_ok, any_ok) in sorted(by.items()):
        print(f"{t}: n={tot}  best-pair grounds={best_ok} ({best_ok / tot:.1%})  "
              f"any user-turn pair grounds={any_ok} ({any_ok / tot:.1%})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=CACHE / "T1_T2_400_FULL.json")
    p.add_argument("--run", default="pilot")
    p.add_argument("--n-per-type", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--conditions", default="none,evidence,full,anneal@10,anneal@3,anneal+sup@10")
    p.add_argument("--judge", default=JUDGE)
    p.add_argument("--judge-effort", default=JUDGE_EFFORT,
                   help="reasoning_effort for the judge; \"none\" sends temperature=0 instead")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--item-workers", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=2.0)
    p.add_argument("--diag-oracle", action="store_true")
    args = p.parse_args()
    if args.diag_oracle:
        diag_oracle(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
