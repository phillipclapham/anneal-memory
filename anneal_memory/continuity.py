"""Continuity validation and wrap package preparation for anneal-memory.

Handles:
- Structural validation (4 required sections)
- Wrap package preparation (episodes + continuity + instructions for the agent)
- Section measurement

The continuity file is a 4-section markdown document:
  - State: Current focus, replaced each session
  - Patterns: Temporal graduation with FlowScript markers
  - Decisions: Committed decisions with lifecycle
  - Context: Compressed narrative, rewritten each session

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from .graduation import detect_stale_patterns
from .types import Episode

# The 4 required sections
_REQUIRED_SECTIONS = frozenset({"state", "patterns", "decisions", "context"})


def validate_structure(text: str) -> bool:
    """Validate that continuity text contains all 4 required sections.

    Checks case-insensitively for ## headers containing each section name.

    Args:
        text: The continuity file text.

    Returns:
        True if all 4 sections found, False otherwise.
    """
    text_lower = text.lower()
    found: set[str] = set()
    for line in text_lower.split("\n"):
        if line.startswith("## "):
            for section in _REQUIRED_SECTIONS:
                if re.search(rf"\b{section}\b", line):
                    found.add(section)
    return found == _REQUIRED_SECTIONS


def measure_sections(text: str) -> dict[str, int]:
    """Measure character count per section.

    Args:
        text: The continuity file text.

    Returns:
        Dict mapping section name -> character count.
    """
    sections: dict[str, int] = {}
    current_section = "_header"
    current_chars = 0

    for line in text.split("\n"):
        if line.startswith("## "):
            if current_chars > 0:
                sections[current_section] = current_chars
            current_section = line[3:].strip()
            current_chars = len(line) + 1
        else:
            current_chars += len(line) + 1

    if current_chars > 0:
        sections[current_section] = current_chars

    return sections


def format_episodes_for_wrap(episodes: list[Episode]) -> str:
    """Format episodes into a readable summary for the compression prompt.

    Groups episodes by type for clearer presentation.
    Includes 8-char IDs for citation references.

    Args:
        episodes: List of episodes to format.

    Returns:
        Formatted string suitable for inclusion in the wrap package.
    """
    if not episodes:
        return "(No episodes in this session)"

    by_type: dict[str, list[Episode]] = {}
    for ep in episodes:
        type_name = ep.type.value
        by_type.setdefault(type_name, []).append(ep)

    lines: list[str] = []
    for type_name, type_eps in sorted(by_type.items()):
        lines.append(f"\n### {type_name.title()}s ({len(type_eps)})")
        for ep in type_eps:
            source_info = f" [{ep.source}]" if ep.source != "agent" else ""
            lines.append(f"- ({ep.id}) {ep.content}{source_info}")

    return "\n".join(lines)


def prepare_wrap_package(
    episodes: list[Episode],
    existing_continuity: str | None,
    project_name: str,
    max_chars: int = 20000,
    today: str | None = None,
    staleness_days: int = 7,
) -> dict[str, Any]:
    """Prepare the full wrap package for the agent.

    This is what the agent receives when calling prepare_wrap.
    It contains everything needed to produce a compressed continuity file.

    Args:
        episodes: Episodes since last wrap (the compression window).
        existing_continuity: Current continuity text, or None for first session.
        project_name: Name for the continuity file header.
        max_chars: Maximum size of the continuity file.
        today: Override for today's date (YYYY-MM-DD). Defaults to actual today.
        staleness_days: Days before flagging stale patterns.

    Returns:
        Dict with keys: episodes, continuity, stale_patterns, instructions, today, max_chars
    """
    if today is None:
        today = date.today().isoformat()

    # Format episodes for the agent
    formatted_episodes = format_episodes_for_wrap(episodes)

    # Detect stale patterns in existing continuity
    stale_patterns: list[dict[str, Any]] = []
    if existing_continuity:
        stale = detect_stale_patterns(existing_continuity, today, staleness_days)
        stale_patterns = [
            {
                "line": s.line_number,
                "content": s.content,
                "level": s.level,
                "last_date": s.last_date,
                "days_stale": s.days_stale,
            }
            for s in stale
        ]

    # Build instructions
    instructions = _build_wrap_instructions(project_name, max_chars, today)

    return {
        "episodes": formatted_episodes,
        "episode_count": len(episodes),
        "continuity": existing_continuity,
        "stale_patterns": stale_patterns,
        "instructions": instructions,
        "today": today,
        "max_chars": max_chars,
    }


def build_engine_prompt(
    session_summary: str,
    existing_continuity: str | None,
    project_name: str,
    max_chars: int,
    today: str,
) -> str:
    """Build the full LLM prompt for automated compression (engine mode).

    This is used by the Engine class for programmatic compression.
    In MCP mode, the agent gets instructions via prepare_wrap_package instead.

    Args:
        session_summary: Formatted episode summary.
        existing_continuity: Current continuity text, or None.
        project_name: Name for the continuity file header.
        max_chars: Maximum continuity size in characters.
        today: Today's date as YYYY-MM-DD.

    Returns:
        The full prompt string for the compression LLM.
    """
    existing_section = ""
    if existing_continuity and existing_continuity.strip():
        existing_section = f"""
## Existing Continuity File
(This is the current continuity file from previous sessions. Update it with the new session data.)

<existing_continuity>
{existing_continuity}
</existing_continuity>
"""
    else:
        existing_section = """
## Existing Continuity File
(No existing continuity — this is the first session. Create a fresh continuity file.)
"""

    marker_ref = _marker_reference(today)

    return f"""You are a memory compression engine for an AI agent's reasoning memory.
Your job is to produce a compressed continuity file that captures the PRINCIPLES and
PATTERNS from this session, not just a summary. The act of compression is itself a
form of thinking — extract what matters, discard what doesn't.

## Session Data
(These are the typed episodes from this session's work.)

<session_data>
{session_summary}
</session_data>
{existing_section}
## Output Format

Produce a markdown file with EXACTLY these 4 section headers (use these headers VERBATIM):
`## State`, `## Patterns`, `## Decisions`, `## Context`
Stay within {max_chars} characters total.

{marker_ref}

## Quality Rules
- PRINCIPLES over facts. "We keep hitting X because Y" > "X happened again"
- JUDGMENT over mechanical rules. You decide what matters.
- DENSITY over length. One insightful line > three vague ones.
- When uncertain whether to keep something, ask: "Would the agent make a worse
  decision next session without this?" If no, cut it.
- Do NOT narrate markers in prose ("we noted a thought about..."). Use markers directly.
- Do NOT include the session data verbatim. Compress it.

## Output
Return ONLY the markdown continuity file. No explanation, no preamble, no code fences.
Start directly with `# {project_name} — Memory (v1)`.
Do NOT add extra headers like `## Summary` or `## Overview`.
Do NOT wrap the output in markdown code fences.
Do NOT number the sections."""


def _build_wrap_instructions(project_name: str, max_chars: int, today: str) -> str:
    """Build the compression instructions the agent receives via prepare_wrap.

    These are agent-facing instructions — conversational, direct.
    """
    marker_ref = _marker_reference(today)

    return f"""Compress your session episodes into your continuity file.

**Output:** A markdown file starting with `# {project_name} — Memory (v1)` containing
EXACTLY these 4 sections: `## State`, `## Patterns`, `## Decisions`, `## Context`.
Stay within {max_chars} characters.

{marker_ref}

**How to compress:**
- State: Replace with your current focus, active work, status. 2-5 lines.
- Patterns: Extract principles, not facts. Group in `{{topic: ...}}` blocks.
- Decisions: Keep committed decisions with rationale. Archive old ones.
- Context: Compressed narrative of recent work. Shape, not transcript. 5-15 lines.

**Quality:** One insightful line > three vague ones. If removing something wouldn't
change your next decision, cut it. Compress principles, not events.

**Return ONLY the markdown.** No explanation, no code fences."""


def _marker_reference(today: str) -> str:
    """The marker reference section shared between agent instructions and engine prompt."""
    return f"""### Density Markers (use in ## Patterns)
- `? question` — open question needing decision
- `thought: insight` — observation or principle worth preserving
- `✓ item` — completed/resolved
- `A -> B` — A causes or leads to B
- `A ><[axis] B` — tension between A and B on the named axis
- `[decided(rationale: "why", on: "date")] choice` — committed decision
- `[blocked(reason: "what", since: "date")] item` — waiting on dependency

### Temporal Graduation (CRITICAL — this is what makes the system learn)
- New pattern from THIS session → `| 1x ({today})`
- Validates existing 1x → `| 2x ({today}) [evidence: <episode_id> "how episode validates pattern"]`
- Validates existing 2x → `| 3x ({today}) [evidence: <episode_id> "explanation"]`
- Evidence citations are REQUIRED for graduations (2x and 3x). Cite the episode's 8-char ID.
- Patterns marked `(ungrounded)` need FRESH evidence from THIS session to re-graduate.
- Patterns at 3x: extract the PRINCIPLE, not the surface observations.
- Patterns older than 7 days with no new validation → remove (stale).
- Group related patterns: `{{topic: ...}}`

**Example:**
```
{{database_architecture:
  thought: ACID compliance outweighs raw speed | 2x ({today}) [evidence: 4931b6a8 "PostgreSQL chosen for ACID"]
  thought: connection pooling is the real bottleneck | 1x ({today})
  ? horizontal scaling strategy ><[single-writer vs multi-writer] | 1x ({today})
}}
```

### Decisions (use in ## Decisions)
Use `[decided(rationale: "why", on: "date")] choice` markers.
- Existing decisions still referenced by active State/Patterns → keep
- 3+ related decisions pointing same direction → extract principle to Patterns, archive individuals
- Decisions >30 days old referencing nothing active → remove"""
