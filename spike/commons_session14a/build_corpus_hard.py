#!/usr/bin/env python3
"""
Session 14a.2 — HARD CORPUS builder.

Deliberately heterogeneous corpus spanning 8 distinct topical arcs from the
last 7 days (minus the 4h contamination window). Each arc contributes 3-4
episodes. Dedup across arcs: if an episode carries multiple arc tags, it
belongs to its highest-priority arc and cannot appear again.

This is the corpus designed to DISTINGUISH independent-priors-finding-
hidden-structure from agree-on-obvious-sequentials. Session 14a.1 proved
the mechanism works on homogeneous corpora; this tests whether it earns
its keep on heterogeneous ones.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

EPISODIC_DB = Path.home() / "Documents" / "flow" / "state" / "episodic.db"
OUT_PATH = Path(__file__).parent / "corpus_hard.json"

# (arc_name, tag_set, target_count) in priority order. Earlier arcs get
# first claim on episodes that carry tags from multiple arcs.
ARCS: list[tuple[str, tuple[str, ...], int]] = [
    ("anvil",               ("anvil",),                                      4),
    ("paper",               ("paper", "harness_thesis"),                     4),
    ("career",              ("career", "labor-economics"),                   4),
    ("anneal_engineering",  ("anneal-memory", "anneal_memory", "code_review"), 4),
    ("bilateral",           ("bilateral", "constellation"),                  4),
    ("narrative",           ("narrative", "discourse", "anansi"),            3),
    ("partnership_meta",    ("partnership", "rlhf", "flow-meta"),            4),
    ("strategic",           ("sovereignty", "positioning",
                             "shadow-productivity", "optimization-lock"),    3),
]

QUERY = """
SELECT DISTINCT e.id, e.timestamp, e.agent, e.type, e.source, e.content
FROM episodes e
JOIN episode_tags t ON t.episode_id = e.id
WHERE t.tag IN ({placeholders})
  AND datetime(e.timestamp) > datetime('now', '-7 days')
  AND datetime(e.timestamp) < datetime('now', '-4 hours')
ORDER BY e.timestamp DESC
"""


def main() -> None:
    conn = sqlite3.connect(EPISODIC_DB)
    used_ids: set[str] = set()
    arc_episodes: dict[str, list] = {}
    arc_assignments: dict[str, str] = {}  # episode_id → arc

    for arc_name, tags, target_count in ARCS:
        placeholders = ",".join("?" for _ in tags)
        rows = conn.execute(QUERY.format(placeholders=placeholders), tags).fetchall()
        picked = []
        for row in rows:
            if len(picked) >= target_count:
                break
            if row[0] in used_ids:
                continue
            picked.append(row)
            used_ids.add(row[0])
            arc_assignments[row[0]] = arc_name
        arc_episodes[arc_name] = picked
        print(f"arc {arc_name}: picked {len(picked)}/{target_count} "
              f"(available: {len(rows)})")

    conn.close()

    # Interleave episodes so the corpus isn't arc-blocked. Takes one from
    # each arc in round-robin order, then repeats — produces a mixed stream
    # so agents can't just group by consecutive indices.
    all_rows: list[tuple] = []
    max_per_arc = max(len(v) for v in arc_episodes.values())
    for i in range(max_per_arc):
        for arc_name, _, _ in ARCS:
            episodes = arc_episodes[arc_name]
            if i < len(episodes):
                all_rows.append(episodes[i])

    corpus = {
        "meta": {
            "window": "7d → 4h ago",
            "arcs": [arc for arc, _, _ in ARCS],
            "arc_targets": {arc: tgt for arc, _, tgt in ARCS},
            "arc_actual": {arc: len(rows) for arc, rows in arc_episodes.items()},
            "episode_count": len(all_rows),
            "note": (
                "Heterogeneous corpus designed to test whether cross-agent "
                "co-citation finds non-obvious cross-arc connections or "
                "collapses to near-zero yield on topically-distant episodes. "
                "Episodes interleaved across arcs so agents can't group by "
                "consecutive indices."
            ),
        },
        "episodes": [],
    }
    for idx, (eid, ts, agent, etype, source, content) in enumerate(all_rows, start=1):
        corpus["episodes"].append(
            {
                "idx": idx,
                "flow_id": eid,
                "timestamp": ts,
                "agent": agent,
                "type": etype,
                "source": source,
                "content": content,
                "arc": arc_assignments[eid],  # for post-hoc analysis only
            }
        )

    OUT_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"\nWrote {len(all_rows)} episodes to {OUT_PATH}")
    print(f"Arc distribution: {corpus['meta']['arc_actual']}")


if __name__ == "__main__":
    main()
