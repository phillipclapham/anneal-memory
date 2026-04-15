#!/usr/bin/env python3
"""
Session 14a Commons feasibility spike — corpus builder.

Pulls ~30 episodes from flow's state/episodic.db in the 48h→4h window (excludes
the Commons scoping session that just happened, preventing meta-contamination).
Filters to architecture-adjacent tags. Writes corpus.json with content indexed
1..N so the agents see numbered blocks, not flow-internal IDs.

The orchestrator (this script) does NOT print content — corpus.json is the only
surface where content lives until the runner dispatches it to agents.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

EPISODIC_DB = Path.home() / "Documents" / "flow" / "state" / "episodic.db"
OUT_PATH = Path(__file__).parent / "corpus.json"

THEME_TAGS = (
    "anneal-memory",
    "anneal_memory",
    "architecture",
    "harness_thesis",
    "multi-agent",
)

# 48h→4h window: captures the triple-arc day PLUS preceding 2 days of
# architecture work, but excludes the ~3h Commons scoping session that
# just concluded (would otherwise preview the spike's thesis).
WINDOW_SQL = """
SELECT DISTINCT e.id, e.timestamp, e.agent, e.type, e.source, e.content
FROM episodes e
JOIN episode_tags t ON t.episode_id = e.id
WHERE t.tag IN ({placeholders})
  AND datetime(e.timestamp) > datetime('now', '-48 hours')
  AND datetime(e.timestamp) < datetime('now', '-4 hours')
ORDER BY e.timestamp DESC
LIMIT 30
"""


def main() -> None:
    placeholders = ",".join("?" for _ in THEME_TAGS)
    query = WINDOW_SQL.format(placeholders=placeholders)
    with sqlite3.connect(EPISODIC_DB) as conn:
        rows = conn.execute(query, THEME_TAGS).fetchall()

    # Reverse so index 1 = oldest, index N = newest. Gives a natural arrow
    # of time inside the corpus that agents can reason about.
    rows = list(reversed(rows))

    corpus = {
        "meta": {
            "window": "48h → 4h ago",
            "theme_tags": list(THEME_TAGS),
            "episode_count": len(rows),
            "note": (
                "Corpus extracted without orchestrator reading content. "
                "Agents see only numbered blocks; original flow IDs are "
                "preserved in this file for post-hoc tracing but MUST NOT "
                "appear in prompts dispatched to agents."
            ),
        },
        "episodes": [],
    }
    for idx, (eid, ts, agent, etype, source, content) in enumerate(rows, start=1):
        corpus["episodes"].append(
            {
                "idx": idx,
                "flow_id": eid,  # for post-hoc tracing only
                "timestamp": ts,
                "agent": agent,
                "type": etype,
                "source": source,
                "content": content,
            }
        )

    OUT_PATH.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(rows)} episodes to {OUT_PATH}")
    print(f"Agents visible in corpus: {sorted({r[2] for r in rows})}")
    print(f"Types visible in corpus:  {sorted({r[3] for r in rows})}")


if __name__ == "__main__":
    main()
