#!/usr/bin/env python3
"""
Session 14a Commons spike — aggregator.

Reads outputs/{complement,codex,gemini}.txt, parses each into
(summary, cited_connections, patterns), and computes the cross-agent
co-citation structure that a real Commons Hebbian layer would form.

The load-bearing question: at N=2 and N=3 thresholds, does multi-agent
co-citation produce structure that looks meaningful, or noise?

Outputs a markdown report at REPORT.md with:
  - per-agent citation stats
  - N=2 and N=3 intersection pair lists with each agent's "why"
  - qualitative inspection prompts (filled in by the orchestrator after
    reading the raw outputs)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "outputs"
CORPUS = HERE / "corpus.json"
REPORT = HERE / "REPORT.md"

AGENTS = ("complement", "codex", "gemini")

# Citation line: "(3, 17): brief why" — optional whitespace, optional colon.
CITE_RE = re.compile(r"^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[:\-]?\s*(.+?)\s*$")

# Section header detection — matches "### SUMMARY", "## SUMMARY", "SUMMARY", etc.
def section_header(line: str) -> str | None:
    stripped = line.strip().lstrip("#").strip()
    if stripped.upper() in {"SUMMARY", "CITED CONNECTIONS", "TOP PATTERNS"}:
        return stripped.upper()
    return None


def parse_agent_output(text: str) -> dict:
    """Split response into SUMMARY, CITED CONNECTIONS, TOP PATTERNS sections."""
    sections: dict[str, list[str]] = {
        "SUMMARY": [],
        "CITED CONNECTIONS": [],
        "TOP PATTERNS": [],
    }
    current: str | None = None
    for raw_line in text.splitlines():
        header = section_header(raw_line)
        if header is not None:
            current = header
            continue
        if current is not None:
            sections[current].append(raw_line)

    summary = "\n".join(sections["SUMMARY"]).strip()

    citations: list[dict] = []
    for line in sections["CITED CONNECTIONS"]:
        m = CITE_RE.match(line)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a == b:
            continue
        if a > b:
            a, b = b, a
        why = m.group(3).strip()
        if not why:
            continue
        citations.append({"a": a, "b": b, "why": why})

    patterns: list[str] = []
    for line in sections["TOP PATTERNS"]:
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\d+[.)]\s*", "", stripped)
        stripped = re.sub(r"^[-*]\s*", "", stripped)
        if stripped:
            patterns.append(stripped)

    return {
        "summary": summary,
        "citations": citations,
        "patterns": patterns,
    }


def load_corpus_indices() -> dict[int, dict]:
    data = json.loads(CORPUS.read_text())
    return {ep["idx"]: ep for ep in data["episodes"]}


def main() -> None:
    parsed: dict[str, dict] = {}
    for agent in AGENTS:
        path = OUTPUT_DIR / f"{agent}.txt"
        if not path.exists():
            print(f"[WARN] missing output: {path}")
            parsed[agent] = {"summary": "", "citations": [], "patterns": []}
            continue
        parsed[agent] = parse_agent_output(path.read_text())

    corpus_by_idx = load_corpus_indices()

    # Accumulate votes: {(a,b): {agent: why}}
    votes: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)
    for agent, data in parsed.items():
        for cite in data["citations"]:
            pair = (cite["a"], cite["b"])
            # Only first citation per agent per pair counts.
            if agent not in votes[pair]:
                votes[pair][agent] = cite["why"]

    per_agent_counts = {a: len(parsed[a]["citations"]) for a in AGENTS}
    n1_pairs = [p for p, v in votes.items() if len(v) == 1]
    n2_pairs = [p for p, v in votes.items() if len(v) == 2]
    n3_pairs = [p for p, v in votes.items() if len(v) == 3]

    total_unique_pairs = len(votes)
    n2_plus = len(n2_pairs) + len(n3_pairs)
    n2_plus_frac = n2_plus / total_unique_pairs if total_unique_pairs else 0.0

    # Build report
    lines: list[str] = []
    lines.append("# Session 14a — Commons Feasibility Spike Report")
    lines.append("")
    lines.append("**Corpus:** 30 episodes from flow's state/episodic.db, 48h→4h window, "
                 "theme tags {anneal-memory, architecture, harness_thesis, multi-agent}. "
                 "Contaminating Commons scoping session (within 4h) explicitly excluded.")
    lines.append("")
    lines.append("**Triple:** complement (Anthropic/Claude) + codex (OpenAI/GPT-5.3) + "
                 "gemini (Google). Three model families, three independent priors. "
                 "Flow-Claude (orchestrator) not in the voting triple — avoids priming "
                 "contamination from this session's Commons Foundation scoping work.")
    lines.append("")
    lines.append("**Process per agent:** same prompt + same corpus. Anonymized "
                 "numbered blocks [1..30]. Each agent produced a SUMMARY, a list "
                 "of CITED CONNECTIONS, and 3-5 TOP PATTERNS, independently.")
    lines.append("")

    lines.append("## Per-Agent Citation Stats")
    lines.append("")
    lines.append("| Agent | Citations | Patterns |")
    lines.append("|---|---:|---:|")
    for agent in AGENTS:
        lines.append(
            f"| {agent} | {per_agent_counts[agent]} | {len(parsed[agent]['patterns'])} |"
        )
    lines.append("")

    lines.append("## Commons Link Formation")
    lines.append("")
    lines.append(f"- Total unique pairs cited (across all agents): **{total_unique_pairs}**")
    lines.append(f"- Cited by exactly one agent (N=1, STAY IN INDIVIDUAL HEBBIAN): {len(n1_pairs)}")
    lines.append(
        f"- Cited by exactly two agents (N=2, Commons link candidate): **{len(n2_pairs)}**"
    )
    lines.append(
        f"- Cited by all three agents (N=3, strong Commons link): **{len(n3_pairs)}**"
    )
    lines.append(
        f"- Cross-validation yield (N≥2 as fraction of total unique pairs): "
        f"**{n2_plus_frac:.1%}**"
    )
    lines.append("")

    def render_pair_block(pair: tuple[int, int], whys: dict[str, str]) -> list[str]:
        a, b = pair
        block = [f"### Pair ({a}, {b}) — cited by {len(whys)} agent(s)"]
        block.append("")
        block.append(f"**Episode [{a}]** ({corpus_by_idx[a]['type']}, "
                     f"{corpus_by_idx[a]['timestamp'][:10]}):")
        block.append(f"> {corpus_by_idx[a]['content'][:350]}"
                     f"{'...' if len(corpus_by_idx[a]['content']) > 350 else ''}")
        block.append("")
        block.append(f"**Episode [{b}]** ({corpus_by_idx[b]['type']}, "
                     f"{corpus_by_idx[b]['timestamp'][:10]}):")
        block.append(f"> {corpus_by_idx[b]['content'][:350]}"
                     f"{'...' if len(corpus_by_idx[b]['content']) > 350 else ''}")
        block.append("")
        block.append("**Why each agent cited:**")
        for agent in AGENTS:
            if agent in whys:
                block.append(f"- **{agent}**: {whys[agent]}")
        block.append("")
        return block

    # N=3 pairs first (strongest signal)
    if n3_pairs:
        lines.append("## N=3 Pairs (all three agents converged)")
        lines.append("")
        lines.append("_These are pairs where all three heterogeneous generators "
                     "independently reached for the same connection. The load-bearing "
                     "qualitative question: did they cite for the same reason "
                     "(weak/convergence-bias) or different-but-compatible reasons "
                     "(strong/independent-structural)?_")
        lines.append("")
        for pair in sorted(n3_pairs):
            lines.extend(render_pair_block(pair, votes[pair]))
    else:
        lines.append("## N=3 Pairs: (none)")
        lines.append("")

    lines.append("## N=2 Pairs (two agents converged)")
    lines.append("")
    if n2_pairs:
        lines.append(f"_{len(n2_pairs)} pair(s). Same qualitative question as N=3 "
                     "applies — are the \"why\"s compatible?_")
        lines.append("")
        for pair in sorted(n2_pairs):
            lines.extend(render_pair_block(pair, votes[pair]))
    else:
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

    lines.append("## Per-Agent Summaries (for qualitative inspection)")
    lines.append("")
    for agent in AGENTS:
        lines.append(f"### {agent}")
        lines.append("")
        lines.append(parsed[agent]["summary"])
        lines.append("")

    lines.append("## Verdict — TO BE FILLED IN BY ORCHESTRATOR")
    lines.append("")
    lines.append("_The mechanical aggregation above is the raw data. The qualitative "
                 "read below is where the spike's actual answer lives._")
    lines.append("")
    lines.append("**Question 1: Did N≥2 pairs form at all?**")
    lines.append("")
    lines.append("**Question 2: Are the `why`s compatible (real convergence) or "
                 "different-but-forced (surface-level agreement)?**")
    lines.append("")
    lines.append("**Question 3: What's the signal-to-noise ratio? Are N≥2 pairs "
                 "structurally meaningful or just the most obviously linked episodes "
                 "anyone would cite?**")
    lines.append("")
    lines.append("**Verdict: Worth pursuing Commons at scale?**")
    lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote report to {REPORT}")
    print(f"Per-agent citations: {per_agent_counts}")
    print(f"N=2 pairs: {len(n2_pairs)} | N=3 pairs: {len(n3_pairs)} | "
          f"Total unique: {total_unique_pairs}")


if __name__ == "__main__":
    main()
