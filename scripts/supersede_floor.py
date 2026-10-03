"""Derive the supersession grounding floor: how much lexical overlap separates a
real update from an unrelated episode.

Measures, per pair, the meaningful words shared (the same tokenizer as
``graduation.check_explanation_overlap``) as a ratio of the SHORTER episode's
meaningful words, for three populations:

* true update pairs: stale_probe.py's 16 facts x 3 update shapes, with and
  without the probe's shared CONTEXT sentence (the padding inflates overlap);
* random pairs: N random episode pairs from a store;
* same-topic pairs: pairs sharing at least one rare term (2..5 episodes), the
  hard negatives: same subject, not an update.

Run it against a COPY of a store, never a live one::

    PYTHONPATH=. python scripts/supersede_floor.py /path/to/copy/memory.db

The shipped floor (``store.SUPERSEDE_MIN_OVERLAP_RATIO``) was chosen from this
output on 2026-10-02; the measured band is in CHANGELOG [0.9.23]. The sampled
counts move with the store (same seed, a few dozen more episodes, a different
sample), so read them as a band, never a constant.
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stale_probe as sp  # noqa: E402

from anneal_memory.graduation import _meaningful_words as _words  # noqa: E402
from anneal_memory.store import SUPERSEDE_MIN_OVERLAP_RATIO  # noqa: E402


def _ratio(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, min(len(wa), len(wb)))


def main(db: str, n: int = 500, seed: int = 7) -> None:
    # Deterministic for a given store and seed (the word sets are walked in
    # sorted order); a different seed or a grown store gives a different sample.
    eps = [r[0] for r in sqlite3.connect(f"file:{db}?mode=ro", uri=True)
           .execute("SELECT content FROM episodes ORDER BY id")]
    rnd = random.Random(seed)
    pops: dict[str, list[float]] = {}
    for ctx in (True, False):
        rows = []
        for f in sp.FACTS:
            s, attr, old, _new, _p, _q = f
            o = sp._sentence(s, attr, old) if ctx else f"The {attr} for {s} is {old}."
            for shape in ("restate", "paraphrase", "negate"):
                u = sp._update_text(shape, f)
                rows.append(_ratio(o, u if ctx else u.replace(sp.CONTEXT, "")))
        pops["true (probe, with CONTEXT)" if ctx else "true (probe, no CONTEXT)"] = rows
    pops["random"] = [_ratio(eps[i], eps[j])
                      for i, j in (rnd.sample(range(len(eps)), 2) for _ in range(n))]
    sets = [_words(e) for e in eps]
    df: dict[str, list[int]] = {}
    for k, s in enumerate(sets):
        for w in sorted(s):  # a set's order follows the hash seed
            df.setdefault(w, []).append(k)
    hard = sorted({tuple(sorted(rnd.sample(ks, 2))) for ks in df.values() if 2 <= len(ks) <= 5})
    pops["same-topic"] = [_ratio(eps[i], eps[j]) for i, j in rnd.sample(hard, min(n, len(hard)))]

    print(f"store: {len(eps)} episodes; floor = ratio >= {SUPERSEDE_MIN_OVERLAP_RATIO}")
    for name, xs in pops.items():
        xs = sorted(xs)
        qs = [round(xs[int(p * (len(xs) - 1))], 2) for p in (0, .1, .25, .5, .75, .9, 1)]
        passed = sum(x >= SUPERSEDE_MIN_OVERLAP_RATIO for x in xs)
        print(f"  {name:28} n={len(xs):4} pass={passed:4}  q0/10/25/50/75/90/100={qs}")


if __name__ == "__main__":
    main(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
