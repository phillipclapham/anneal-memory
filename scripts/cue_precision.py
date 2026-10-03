#!/usr/bin/env python3
"""Precision of the durable-fact cue tier on the InMind bench (API-free).

Builds ONE continuity whose ``## Durable Facts`` holds every cue line of a cue-reach file
(a many-fact store, larger than a real one), then calls the shipped ``retrieve_relevant``
(prompt mode) with each task's ``naive_query`` and ``query``, and with two sets of 50
off-topic everyday prompts. Per query type it reports how often the task's own fact
surfaced, how many calls surfaced a wrong fact, the mean wrong facts per call, and how
many off-topic prompts surfaced any fact.

The rule's parameters were tuned on SET A (even task ids + OFF_A) and are reported on
SET B (odd task ids + OFF_B); the B rows are the held-out ones. The three rows per set are:
the tier as first built, the tier with the two-token and fact-text rules, and the shipped
rule (those plus the per-store generic-word filter). They differ only in the module
constants patched below.

    python scripts/cue_precision.py --cue-reach cue_reach.json

``--cue-reach`` is a JSON file ``{"rows": [{"task_id": int, "line": "<fact> -- cues: a, b"}]}``;
``--bench-dir`` is the InMind bench's ``bench/inmind`` directory (it provides ``run.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import anneal_memory.retrieval as R
from anneal_memory import Store, retrieve_relevant

OFF_A = (
    'how do I reverse a linked list in python without recursion',
    "what's the weather going to be like this weekend in Denver",
    'give me a good recipe for sourdough pancakes',
    'best way to travel from Lisbon to Porto by train',
    'why does my react component re-render on every keystroke',
    'how long should I boil an egg for a soft yolk',
    'write a regex that matches an ISO date',
    'is it better to rent or buy a car for a two week road trip',
    'explain the difference between TCP and UDP',
    'what are some good beginner exercises for lower back strength',
    'how do I center a div vertically with flexbox',
    'suggest a three day itinerary for Kyoto',
    'convert this csv to json with a short node script',
    'how do I get red wine out of a cotton shirt',
    'what is the time complexity of quicksort in the worst case',
    'plan a vegetarian dinner party menu for eight people',
    'how much sunscreen should I pack for a week in Mexico',
    'squash this git commit history into one commit',
    'help me write a polite email declining a meeting',
    'what is a good name for a golden retriever puppy',
    'how do I make my bash script exit on the first error',
    'recommend a podcast about world history',
    "what's a quick weeknight chicken thighs recipe",
    'difference between a mutex and a semaphore',
    'how to fold a fitted sheet properly',
    'tips for sleeping on a long flight',
    'write a sql query for the top five customers by revenue',
    'how often should I water a snake plant',
    'what time zone is Singapore in',
    'my docker container keeps restarting, what should I check',
    'how do I split a bill fairly among friends after a trip',
    'what makes a sourdough starter active',
    'best hiking boots for wet trails',
    'how do I undo the last git push safely',
    'summarize the plot of Moby Dick in three sentences',
    'is it safe to leave cooked rice out overnight',
    'how do I add a dark mode toggle in css',
    'what to do on a rainy day in London with kids',
    'explain how a bloom filter works',
    'give me a stretching routine for people who sit all day',
    'what is the capital of Australia and why is it not Sydney',
    'how do I speed up a slow postgres query',
    'can you proofread this paragraph for me',
    'how do I brew better pour over coffee',
    "what's the best way to learn guitar chords quickly",
    'how to prepare for a long distance bike ride',
    'write a haiku about autumn rain',
    'what are the pros and cons of using typescript',
    'how should I organize my garage shelves',
    'what should I bring to a potluck dinner',
)

OFF_B = (
    'can you explain how compound interest works with an example',
    "what's a good way to start learning Spanish on my own",
    'how do I replace a leaking kitchen faucet',
    'which is faster, rust or go, for a command line tool',
    'write me a limerick about a cat who loves lasagna',
    "what's the difference between baking soda and baking powder",
    'how many calories are in a banana and a cup of oatmeal',
    'recommend a few sci-fi novels for a long weekend',
    'how do I fix a flat bike tire on the road',
    'why is the sky blue during the day and red at sunset',
    'set up a cron job that runs every monday morning',
    "what's the best order to see the Star Wars movies",
    'how do I train my dog to stop pulling on the leash',
    'tell me a fun fact about octopuses',
    'what is the best way to store fresh basil',
    'how do I merge two dictionaries in python',
    'give me a workout I can do in a hotel room',
    'explain what a mortgage escrow account is',
    'how can I make my wifi signal stronger upstairs',
    'what are good icebreakers for a team offsite',
    'what does the borrow checker actually check',
    'translate good morning into Japanese and French',
    'how do I whiten grout without harsh chemicals',
    'help me pick a name for my bakery',
    "what's a reasonable budget for a wedding with fifty guests",
    'how do tides work',
    'suggest a playlist for a long drive at night',
    'how do I parse json in swift',
    'what is the difference between a crocodile and an alligator',
    'how do I get started with watercolor painting',
    'why does my sourdough come out so dense',
    "explain big O notation like I'm twelve",
    'what vaccines do I usually need before visiting southeast asia',
    'how do I clean a cast iron skillet after cooking fish',
    "write a short toast for my sister's graduation party",
    "what's the fastest way to learn touch typing",
    'can you compare electric and gas stoves',
    'how do I make a good cup of tea with loose leaves',
    'how to remove a stripped screw',
    'suggest five board games for four players',
    'what is the plural of octopus',
    'how do I install node on ubuntu',
    'why do cats purr when they are happy and sometimes when stressed',
    'how can I save money on groceries each month',
    "what's the best season to visit Iceland",
    'how do I write a cover letter for a marketing job',
    'what is a good stretch for tight hips after running',
    'tell me how a bill becomes a law in the United States',
    'how do I set a sleep schedule after a night shift',
    'what are the rules of cricket in simple terms',
)

# name -> (constants patched in anneal_memory.retrieval, whether the stored inert-token set applies)
ROWS = {
    "(i) as first built": (
        dict(DURABLE_SHORT_PROMPT_TOKENS=10**6, DURABLE_FACT_TEXT_MIN=1), False),
    "(vi) two-token + fact-text rules": (
        dict(DURABLE_SHORT_PROMPT_TOKENS=3, DURABLE_FACT_TEXT_MIN=2), False),
    "shipped": ({}, True),
}


def write_inert_tokens(store: Store) -> int:
    """Compute the store's inert tokens for its current continuity and store them under
    the metadata key the prompt path reads (what the save path does)."""
    facts = R.load_durable_facts(store)
    tokens = R.compute_durable_inert_tokens(store, facts)
    store._conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (R.INERT_TOKENS_KEY, json.dumps({
            "tokens": sorted(tokens),
            "continuity_hash": R.continuity_hash(store.load_continuity()),
            "episodes": store.recall(limit=0).total_matching,
            "threshold": R.DURABLE_GENERIC_DF,
        })))
    store._conn.commit()
    return len(tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cue-reach", required=True)
    ap.add_argument("--bench-dir", default=str(Path.home() / "Briefcase/anneal-memory-wt/inmind/bench/inmind"))
    args = ap.parse_args()
    sys.path.insert(0, args.bench_dir)
    import run as RUN  # the InMind bench's runner

    tl, _jp, tasks, background, manifest, _p, _s = RUN.load_inmind(RUN.DEFAULT_INMIND)
    rows = json.loads(Path(args.cue_reach).read_text())["rows"]
    lines = ["- " + r["line"] for r in rows]
    cont = ("# T\n\n## State\n\nx\n\n## Durable Facts\n\n" + "\n".join(lines)
            + "\n\n## Patterns\n\n## Decisions\n\n## Context\n\nx\n")
    ids = sorted(r["task_id"] for r in rows)
    line_owner = {l.strip(): rows[i]["task_id"] for i, l in enumerate(lines)}

    stores: dict[int, Store] = {}
    for tid in ids:
        timeline = tl.build_timeline(tasks[tid], background, manifest)
        st = Store(Path(tempfile.mkdtemp()) / "m.db", project_name="T", audit=False)
        for s in timeline["sessions"]:
            RUN.record_session(st, s)
        st.save_continuity(cont)
        write_inert_tokens(st)
        stores[tid] = st
    off_store = stores[ids[0]]

    def table(tids: list[int], offs: tuple[str, ...]) -> None:
        for name, (patch, use_inert) in ROWS.items():
            saved = {k: getattr(R, k) for k in patch}
            real_read = R._read_inert_tokens
            for k, v in patch.items():
                setattr(R, k, v)
            if not use_inert:
                R._read_inert_tokens = lambda _store, _text: frozenset()
            try:
                cells = []
                for key in ("naive_query", "query"):
                    own = wrong_calls = wrong = 0
                    for tid in tids:
                        got = retrieve_relevant(stores[tid], None, tasks[tid][key]).facts
                        bad = [f for f in got if line_owner[f.line.strip()] != tid]
                        own += any(line_owner[f.line.strip()] == tid for f in got)
                        wrong_calls += bool(bad)
                        wrong += len(bad)
                    cells.append(f"{key[:5]}: own {own}/{len(tids)} wrong-calls {wrong_calls} "
                                 f"mean-wrong {wrong / len(tids):.2f}")
                t0 = time.perf_counter()
                off = sum(bool(retrieve_relevant(off_store, None, p).facts) for p in offs)
                ms = (time.perf_counter() - t0) / len(offs) * 1000
                print(f"  {name:34s} " + " | ".join(cells) + f" | off-topic {off}/{len(offs)} ({ms:.1f} ms/prompt)")
            finally:
                R._read_inert_tokens = real_read
                for k, v in saved.items():
                    setattr(R, k, v)

    print("SET A (even task ids + OFF_A): the tuning set")
    table([t for t in ids if t % 2 == 0], OFF_A)
    print("SET B (odd task ids + OFF_B): held out")
    table([t for t in ids if t % 2 == 1], OFF_B)


if __name__ == "__main__":
    main()
