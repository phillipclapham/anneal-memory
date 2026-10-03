#!/usr/bin/env python3
"""Mechanical, API-free check of anneal's on-cue recall on all 125 InMind tasks.

For each task: replay the canonical timeline into a fresh Store (one episode per
exchange, as run.py does), call retrieve_relevant for the direct and indirect
queries, and count whether the injected target episode came back. No LLM, no
judge: "retrieved" means the target episode's id is among the returned episodes.

  PYTHONPATH=. python bench/inmind/recall_diag.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from anneal_memory import Store  # noqa: E402
from anneal_memory.retrieval import retrieve_relevant  # noqa: E402

from run import DEFAULT_INMIND, PROJECT_NAME, load_inmind, record_session  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inmind", type=Path, default=DEFAULT_INMIND)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tl, _jp, tasks, background, manifest, _prompts, _sha = load_inmind(args.inmind)
    inj = int(manifest["injection_session_index"])
    rows = []
    for tid in sorted(tasks):
        task = tasks[tid]
        timeline = tl.build_timeline(task, background, manifest)
        with tempfile.TemporaryDirectory() as d:
            store = Store(Path(d) / "m.db", project_name=PROJECT_NAME, audit=False)
            try:
                target_id = None
                for i, s in enumerate(timeline["sessions"]):
                    ids = record_session(store, s)
                    if i == inj:
                        target_id = ids[-1]
                row = {"task_id": tid}
                for key in ("naive_query", "query"):
                    r = retrieve_relevant(store, None, task[key])
                    row[key] = {"keywords": r.query_keywords, "n": len(r.episodes),
                                "target": any(e.id == target_id for e in r.episodes)}
                rows.append(row)
            finally:
                store.close()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "recall_diag.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    for key in ("naive_query", "query"):
        hit = sum(r[key]["target"] for r in rows)
        empty = sum(r[key]["n"] == 0 for r in rows)
        short = sum(len(r[key]["keywords"]) < 2 for r in rows)
        print(f"{key:<12} target retrieved {hit}/{n}  empty context {empty}/{n}  "
              f"(<2 keywords: {short})")


if __name__ == "__main__":
    main()
