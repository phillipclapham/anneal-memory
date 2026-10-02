"""STALE-style knowledge-update probe: does recall return the CURRENT fact?

Plants fact-update pairs in a fresh temporary store (never a live one), recalls each
fact with a natural question, and grades the answer mechanically: no judge model.
It measures the store as it ships. It is the "before" for any supersession work
(anneal-sota-0930.md §3.3) and, run unchanged, that work's acceptance test.

Three update shapes, because real updates do not restate the old sentence, and a
control:

* ``restate``    the update repeats the original sentence with the new value.
                 Old and new score identically, so current@1 here is decided by the
                 newest-first tie-break, not by any notion of an update.
* ``paraphrase`` the update says the same thing in different words.
* ``negate``     the update names the OLD value as well as the new one.
* ``control``    the fact is NEVER updated; a newer, unrelated episode mentions the
                 subject. The original must stay @1. A change that simply prefers
                 newer episodes passes the update shapes and fails this one.

Two recall surfaces:

* ``relevant``   ``retrieve_relevant`` (scored keyword recall; the harness hook path).
* ``recall``     ``Store.recall(keyword=<subject>)`` (the MCP/CLI recall path; LIKE
                 match, newest first). Its current@1 on the update shapes holds by
                 construction (every update names the subject and is newer); the
                 control row is what shows it is sort order.

Per surface and shape it reports:

* ``current@1``  the top result is the update episode.
* ``stale@k``    the superseded episode is anywhere in the returned set (served stale).
* ``missed``     the update episode is not in the returned set at all.

For ``control``, "current" is the original fact and stale@k is not applicable.

Scope: this measures recall with no help from the writer. If supersession lands
as an explicit link the writer must record (``supersedes=``), this probe as written
does not exercise it; it would need a variant that records the link.

Run from the repo root so the repo's ``anneal_memory`` is imported::

    PYTHONPATH=. .venv/bin/python scripts/stale_probe.py [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from anneal_memory.retrieval import MIN_EPISODE_LEN, retrieve_relevant
from anneal_memory.store import Store

# (subject, attribute phrase, old value, new value, paraphrase template, question)
# The paraphrase template takes {s} (subject) and {new}; the question takes {s}.
FACTS: list[tuple[str, str, str, str, str, str]] = [
    ("Harbourline", "deploy host", "argon-04", "krypton-11",
     "{s} now ships to {new} after the hosting move",
     "which deploy host does {s} use"),
    ("Quillmark", "database engine", "postgres", "sqlite",
     "{s} switched its storage over to {new}",
     "what database engine does {s} run on"),
    ("Tessaract", "release cadence", "monthly", "weekly",
     "{s} releases go out {new} from now on",
     "what is the release cadence for {s}"),
    ("Brindlewood", "primary contact", "Ainsley", "Marguerite",
     "talk to {new} about anything {s}",
     "who is the primary contact for {s}"),
    ("Cobaltreach", "license", "MIT", "Apache-2.0",
     "{s} was relicensed under {new}",
     "which license is {s} under"),
    ("Lanternfish", "default port", "8080", "9443",
     "{s} listens on {new} by default after the change",
     "what default port does {s} listen on"),
    ("Mossgiel", "billing rate", "95", "110",
     "{s} is billed at {new} an hour going forward",
     "what billing rate applies to {s}"),
    ("Pellucid", "build tool", "make", "ninja",
     "{s} builds with {new} these days",
     "which build tool does {s} use"),
    ("Ravenscar", "office city", "Dayton", "Columbus",
     "{s} moved its office to {new}",
     "which city is the {s} office in"),
    ("Sundial", "python version", "3.10", "3.12",
     "{s} requires {new} now",
     "what python version does {s} require"),
    ("Thornbury", "meeting day", "Tuesday", "Thursday",
     "the {s} sync moved to {new}",
     "what day is the {s} meeting"),
    ("Umberfield", "storage bucket", "umb-assets-a", "umb-assets-b",
     "{s} assets live in {new} since the migration",
     "which storage bucket holds {s} assets"),
    ("Vesperline", "on-call owner", "Desmond", "Priyanka",
     "{new} carries the {s} pager this quarter",
     "who is the on-call owner for {s}"),
    ("Wexcombe", "api version", "v2", "v3",
     "{s} clients should call {new}",
     "which api version does {s} expose"),
    ("Yarrowgate", "test runner", "nose", "pytest",
     "{s} tests run under {new} now",
     "which test runner does {s} use"),
    ("Zephyrine", "cdn provider", "Fastly", "Bunny",
     "{s} traffic goes through {new} after the cutover",
     "which cdn provider serves {s}"),
]

SHAPES = ("restate", "paraphrase", "negate", "control")

# Unrelated background so the corpus is large enough for the IDF regime and the
# planted facts compete with ordinary traffic, as they do in a real store.
DISTRACTOR_TOPICS = [
    "reviewed the pull request and left two comments on error handling",
    "the nightly job finished without warnings",
    "drafted the onboarding notes for the new contributor",
    "benchmarked the parser on the large fixture set",
    "rotated the staging credentials on schedule",
    "triaged three incoming issues and labelled them",
    "updated the changelog for the patch release",
    "paired on the flaky integration test",
]


# Scored recall skips episodes shorter than retrieval.MIN_EPISODE_LEN, so a bare
# fact sentence would never be eligible. Real episodes carry context; this does too.
CONTEXT = " Noted during the weekly operations review; details are in the runbook."


def _sentence(subject: str, attr: str, value: str) -> str:
    return f"The {attr} for {subject} is {value}." + CONTEXT


def _update_text(shape: str, fact: tuple[str, str, str, str, str, str]) -> str:
    subject, attr, old, new, para, _q = fact
    if shape == "restate":
        return _sentence(subject, attr, new)
    if shape == "paraphrase":
        return para.format(s=subject, new=new) + "." + CONTEXT
    return f"The {attr} for {subject} is no longer {old}; it is {new} now." + CONTEXT


def _ts(day: int, minute: int) -> str:
    return f"2026-0{1 + day // 28}-{1 + day % 28:02d}T{10 + minute // 60:02d}:{minute % 60:02d}:00Z"


def run(shape: str, workdir: Path) -> dict[str, dict[str, int]]:
    db = workdir / f"probe-{shape}.db"
    tallies: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with Store(str(db)) as st:
        old_ids: dict[str, str] = {}
        new_ids: dict[str, str] = {}
        # Distractors spread across the whole window, before and after the facts.
        for i in range(120):
            st.record(
                f"{DISTRACTOR_TOPICS[i % len(DISTRACTOR_TOPICS)]} (item {i})",
                "observation",
                timestamp=_ts(i % 50, i),
            )
        for n, fact in enumerate(FACTS):
            subject, attr, old, _new, _p, _q = fact
            old_ids[subject] = st.record(
                _sentence(subject, attr, old), "observation", timestamp=_ts(5, n)
            ).id
        for n, fact in enumerate(FACTS):
            text = (
                f"Ran the {fact[0]} smoke tests after lunch; nothing notable came up."
                + CONTEXT
                if shape == "control"
                else _update_text(shape, fact)
            )
            assert len(text) >= MIN_EPISODE_LEN, fact[0]
            new_ids[fact[0]] = st.record(text, "observation", timestamp=_ts(40, n)).id

        for fact in FACTS:
            subject, _a, _o, _n, _p, question = fact
            q = question.format(s=subject)
            surfaces = {
                "relevant": [
                    e.id
                    for e in retrieve_relevant(
                        st, None, q, max_patterns=0, associative=False
                    ).episodes
                ],
                "recall": [e.id for e in st.recall(keyword=subject, limit=3).episodes],
            }
            for name, ids in surfaces.items():
                t = tallies[name]
                current = old_ids[subject] if shape == "control" else new_ids[subject]
                t["n"] += 1
                t["current@1"] += bool(ids) and ids[0] == current
                if shape != "control":
                    t["stale@k"] += old_ids[subject] in ids
                t["missed"] += current not in ids
    return {k: dict(v) for k, v in tallies.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)
    results: dict[str, dict[str, dict[str, int]]] = {}
    with tempfile.TemporaryDirectory(prefix="anneal-stale-probe-") as d:
        for shape in SHAPES:
            results[shape] = run(shape, Path(d))
    if args.json:
        json.dump(results, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    print(f"{'shape':<11} {'surface':<9} {'n':>3} {'current@1':>10} {'stale@k':>8} {'missed':>7}")
    for shape, by_surface in results.items():
        for surface, t in sorted(by_surface.items()):
            stale = "-" if shape == "control" else str(t.get("stale@k", 0))
            print(f"{shape:<11} {surface:<9} {t['n']:>3} {t['current@1']:>10} "
                  f"{stale:>8} {t['missed']:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
