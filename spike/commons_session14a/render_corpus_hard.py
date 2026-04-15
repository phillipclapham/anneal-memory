#!/usr/bin/env python3
"""Render corpus_hard.json into the anonymized markdown agents see."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "corpus_hard.json"
OUT = HERE / "corpus_hard_for_agents.md"


def main() -> None:
    data = json.loads(CORPUS.read_text())
    episodes = data["episodes"]

    lines: list[str] = []
    lines.append("# Episode Corpus")
    lines.append("")
    lines.append(
        f"{len(episodes)} episodes from a working AI agent's memory system. "
        "Each episode is a short structured note the agent recorded during "
        "its work. Your task is to compress this corpus — see the separate "
        "instructions."
    )
    lines.append("")

    for ep in episodes:
        date = ep["timestamp"][:10]
        etype = ep["type"]
        content = ep["content"].strip()
        lines.append(f"[{ep['idx']}] {etype} ({date}):")
        lines.append(f"    {content}")
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {len(episodes)} episodes to {OUT}")
    print(f"Total chars: {len(OUT.read_text())}")


if __name__ == "__main__":
    main()
