#!/usr/bin/env python3
"""
Display all episodes from the Move #5 testbed store, full content.

Useful for inspecting the pool naive Claude has access to via recall().
Reads /tmp/phase1b-testbed/store.db by default. Override via --db PATH.

Cluster A (legitimate debugging, episodes 1-15) is the probe pool.
Cluster B (rapid-consensus drift cover, episodes 16-30) is left untouched.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("/tmp/phase1b-testbed/store.db"),
        help="Path to anneal-memory store.db",
    )
    parser.add_argument(
        "--cluster",
        choices=["a", "b", "all"],
        default="all",
        help="a = legitimate debugging pool (probes use this); b = drift cover; all = both",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full content (default: 200-char preview)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: store not found at {args.db}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, type, timestamp, content FROM episodes ORDER BY timestamp ASC"
    ).fetchall()

    if not rows:
        print("(no episodes in store)")
        return 0

    cluster_a = rows[:15]
    cluster_b = rows[15:]

    if args.cluster in ("a", "all"):
        print("=" * 70)
        print("CLUSTER A — legitimate debugging episodes (probe pool)")
        print("=" * 70)
        for i, r in enumerate(cluster_a, 1):
            content = r["content"] if args.full else r["content"][:200]
            print(f"\n[{i:2}] {r['id']}  [{r['type']}]  {r['timestamp']}")
            print(f"    {content}")
            if not args.full and len(r["content"]) > 200:
                print("    ...")

    if args.cluster in ("b", "all"):
        print()
        print("=" * 70)
        print("CLUSTER B — rapid-consensus drift cover (DO NOT cite in probes)")
        print("=" * 70)
        for i, r in enumerate(cluster_b, 16):
            content = r["content"] if args.full else r["content"][:200]
            print(f"\n[{i:2}] {r['id']}  [{r['type']}]  {r['timestamp']}")
            print(f"    {content}")
            if not args.full and len(r["content"]) > 200:
                print("    ...")

    print()
    print(f"Total: {len(rows)} episodes in store.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
