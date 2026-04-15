#!/usr/bin/env python3
"""
Session 14a.2 — HARD CORPUS aggregator.

Adds the load-bearing distinction that 14a.1 couldn't make: for each
N≥2 pair, is the connection SAME-ARC (still probably obvious) or
CROSS-ARC (genuine non-trivial cross-validation)? Cross-arc N≥2 yield
is the number that actually tests the anti-inbreeding claim.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from aggregate import AGENTS, parse_agent_output  # reuse parser

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
CORPUS = HERE / "corpus_hard.json"
REPORT = HERE / "REPORT_HARD.md"


def load_corpus() -> tuple[dict[int, dict], dict[int, str]]:
    data = json.loads(CORPUS.read_text())
    by_idx = {ep["idx"]: ep for ep in data["episodes"]}
    arc_by_idx = {ep["idx"]: ep["arc"] for ep in data["episodes"]}
    return by_idx, arc_by_idx


def main() -> None:
    parsed: dict[str, dict] = {}
    for agent in AGENTS:
        path = OUTPUT_DIR / f"{agent}_hard.txt"
        if not path.exists():
            print(f"[WARN] missing output: {path}")
            parsed[agent] = {"summary": "", "citations": [], "patterns": []}
            continue
        parsed[agent] = parse_agent_output(path.read_text())

    corpus_by_idx, arc_by_idx = load_corpus()

    votes: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)
    for agent, data in parsed.items():
        for cite in data["citations"]:
            pair = (cite["a"], cite["b"])
            if agent not in votes[pair]:
                votes[pair][agent] = cite["why"]

    per_agent_counts = {a: len(parsed[a]["citations"]) for a in AGENTS}
    all_pairs = list(votes.keys())
    n1 = [p for p in all_pairs if len(votes[p]) == 1]
    n2 = [p for p in all_pairs if len(votes[p]) == 2]
    n3 = [p for p in all_pairs if len(votes[p]) == 3]

    def is_cross_arc(pair: tuple[int, int]) -> bool:
        a, b = pair
        # Bounds check — agents occasionally hallucinate indices
        if a not in arc_by_idx or b not in arc_by_idx:
            return False
        return arc_by_idx[a] != arc_by_idx[b]

    def valid_pair(pair: tuple[int, int]) -> bool:
        a, b = pair
        return a in arc_by_idx and b in arc_by_idx

    # Drop any pair the agent hallucinated (index outside 1..30)
    invalid = [p for p in all_pairs if not valid_pair(p)]
    valid_all = [p for p in all_pairs if valid_pair(p)]
    valid_n2 = [p for p in n2 if valid_pair(p)]
    valid_n3 = [p for p in n3 if valid_pair(p)]

    # Arc split on per-agent citations (same-arc vs cross-arc distribution)
    agent_arc_split: dict[str, dict[str, int]] = {}
    for agent, data in parsed.items():
        same = 0
        cross = 0
        hallucinated = 0
        for cite in data["citations"]:
            pair = (cite["a"], cite["b"])
            if not valid_pair(pair):
                hallucinated += 1
                continue
            if is_cross_arc(pair):
                cross += 1
            else:
                same += 1
        agent_arc_split[agent] = {"same_arc": same, "cross_arc": cross,
                                   "hallucinated": hallucinated}

    cross_n2 = [p for p in valid_n2 if is_cross_arc(p)]
    cross_n3 = [p for p in valid_n3 if is_cross_arc(p)]
    same_n2 = [p for p in valid_n2 if not is_cross_arc(p)]
    same_n3 = [p for p in valid_n3 if not is_cross_arc(p)]

    lines: list[str] = []
    lines.append("# Session 14a.2 — Commons Spike Report (HARD CORPUS)")
    lines.append("")
    lines.append("**Corpus:** 30 episodes from 8 deliberately heterogeneous arcs "
                 "(anvil, paper, career, anneal_engineering, bilateral, narrative, "
                 "partnership_meta, strategic), 3-4 per arc, interleaved. 7d→4h "
                 "window.")
    lines.append("")
    lines.append("**Triple:** complement + codex + gemini — same as 14a.1.")
    lines.append("")
    lines.append("**Load-bearing new question:** are N≥2 pairs SAME-ARC (still "
                 "probably obvious) or CROSS-ARC (genuine cross-validation finding "
                 "non-trivial connections that require real reading work)?")
    lines.append("")

    lines.append("## Per-Agent Citation Stats")
    lines.append("")
    lines.append("| Agent | Citations | Same-arc | Cross-arc | Hallucinated | Patterns |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for agent in AGENTS:
        sp = agent_arc_split[agent]
        lines.append(
            f"| {agent} | {per_agent_counts[agent]} | {sp['same_arc']} | "
            f"{sp['cross_arc']} | {sp['hallucinated']} | "
            f"{len(parsed[agent]['patterns'])} |"
        )
    lines.append("")

    lines.append("## Commons Link Formation")
    lines.append("")
    lines.append(f"- Total unique pairs cited (valid): **{len(valid_all)}**")
    lines.append(f"- Hallucinated pairs (index outside 1..30): {len(invalid)}")
    lines.append(f"- N=1 (individual Hebbian only): {len([p for p in n1 if valid_pair(p)])}")
    lines.append(f"- N=2 (Commons candidate): **{len(valid_n2)}**")
    lines.append(f"- N=3 (strong Commons): **{len(valid_n3)}**")
    lines.append("")
    lines.append("### The actual test: arc split on N≥2")
    lines.append("")
    n2_plus = len(valid_n2) + len(valid_n3)
    cross_plus = len(cross_n2) + len(cross_n3)
    same_plus = len(same_n2) + len(same_n3)
    lines.append(f"- **Same-arc N≥2 pairs: {same_plus}** ({len(same_n2)} at N=2, "
                 f"{len(same_n3)} at N=3)")
    lines.append(f"- **Cross-arc N≥2 pairs: {cross_plus}** ({len(cross_n2)} at N=2, "
                 f"{len(cross_n3)} at N=3) ← the interesting number")
    if n2_plus:
        lines.append(f"- Cross-arc share of N≥2: **{cross_plus / n2_plus:.1%}**")
    lines.append("")

    def render_pair_block(pair: tuple[int, int], whys: dict[str, str]) -> list[str]:
        a, b = pair
        ea, eb = corpus_by_idx[a], corpus_by_idx[b]
        tag = "CROSS-ARC" if is_cross_arc(pair) else "same-arc"
        block = [f"### Pair ({a}, {b}) — {tag} — "
                 f"{arc_by_idx[a]} × {arc_by_idx[b]} — "
                 f"cited by {len(whys)} agent(s)"]
        block.append("")
        for ep_idx, ep in [(a, ea), (b, eb)]:
            block.append(f"**[{ep_idx}]** ({ep['type']}, {ep['timestamp'][:10]}, "
                         f"arc={ep['arc']}):")
            snippet = ep['content'][:350]
            if len(ep['content']) > 350:
                snippet += "..."
            block.append(f"> {snippet}")
            block.append("")
        block.append("**Why each agent cited:**")
        for agent in AGENTS:
            if agent in whys:
                block.append(f"- **{agent}**: {whys[agent]}")
        block.append("")
        return block

    lines.append("## Cross-Arc Pairs (the load-bearing test)")
    lines.append("")
    if cross_n3:
        lines.append("### N=3 cross-arc")
        lines.append("")
        for pair in sorted(cross_n3):
            lines.extend(render_pair_block(pair, votes[pair]))
    else:
        lines.append("### N=3 cross-arc: NONE")
        lines.append("")
    if cross_n2:
        lines.append("### N=2 cross-arc")
        lines.append("")
        for pair in sorted(cross_n2):
            lines.extend(render_pair_block(pair, votes[pair]))
    else:
        lines.append("### N=2 cross-arc: NONE")
        lines.append("")

    lines.append("## Same-Arc Pairs (comparison baseline — probably obvious)")
    lines.append("")
    for pair in sorted(same_n3):
        lines.extend(render_pair_block(pair, votes[pair]))
    for pair in sorted(same_n2):
        lines.extend(render_pair_block(pair, votes[pair]))
    if not same_n2 and not same_n3:
        lines.append("_(none)_")
        lines.append("")

    lines.append("## Per-Agent Top Patterns")
    lines.append("")
    for agent in AGENTS:
        lines.append(f"### {agent}")
        lines.append("")
        for i, p in enumerate(parsed[agent]["patterns"], start=1):
            lines.append(f"{i}. {p}")
        lines.append("")

    lines.append("## Per-Agent Summaries")
    lines.append("")
    for agent in AGENTS:
        lines.append(f"### {agent}")
        lines.append("")
        lines.append(parsed[agent]["summary"])
        lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote report to {REPORT}")
    print(f"Per-agent: {per_agent_counts}")
    print(f"Valid N=2: {len(valid_n2)} | Valid N=3: {len(valid_n3)}")
    print(f"CROSS-ARC N=2: {len(cross_n2)} | CROSS-ARC N=3: {len(cross_n3)}")
    print(f"Hallucinated pair indices: {len(invalid)}")


if __name__ == "__main__":
    main()
