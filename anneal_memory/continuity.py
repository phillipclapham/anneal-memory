"""Continuity validation and wrap package preparation for anneal-memory.

Handles:
- Structural validation (4 required sections)
- Wrap package preparation (episodes + continuity + instructions for the agent)
- Validated save (full pipeline: structure + graduation + associations + decay)
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

from .graduation import detect_stale_patterns, validate_graduations
from .types import AffectiveState, Episode

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

**Affective state:** When saving with save_continuity, optionally include your
functional state during this compression as `affective_state: {{"tag": "...", "intensity": 0.0-1.0}}`.
Reflect honestly — were you engaged, curious, uncertain, calm? How strongly (0-1)?
This creates persistent emotional associations between co-cited episodes.

**Return ONLY the markdown.** No explanation, no code fences."""


def _marker_reference(today: str) -> str:
    """The marker reference section used in agent compression instructions."""
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


def validated_save_continuity(
    store: Any,
    text: str,
    affective_state: AffectiveState | None = None,
) -> dict[str, Any]:
    """Save continuity with the full validation pipeline.

    This is the library equivalent of what the MCP server and CLI do:
    structure validation, graduation citation checking, Hebbian association
    formation, decay, metadata update, and wrap completion recording.

    Use this instead of bare ``store.save_continuity()`` to get the immune
    system, associations, and decay — the features that make anneal-memory
    different from a flat file.

    Args:
        store: A Store instance.
        text: The agent-compressed continuity text.
        affective_state: Optional agent functional state during this wrap.

    Returns:
        Dict with wrap results: path, episodes_compressed, graduations_validated,
        graduations_demoted, associations_formed, associations_strengthened,
        associations_decayed, skipped_prepare.

    Raises:
        ValueError: If text is empty or missing required sections.
    """
    from .associations import process_wrap_associations

    if not text or not text.strip():
        raise ValueError("Continuity text cannot be empty")

    # Validate structure (4 required sections)
    if not validate_structure(text):
        raise ValueError(
            "Continuity must contain all 4 sections: "
            "## State, ## Patterns, ## Decisions, ## Context"
        )

    # Check if prepare_wrap was called first
    skipped_prepare = not store.status().wrap_in_progress

    # Get current session's episodes for citation validation
    episodes = store.episodes_since_wrap()
    valid_ids = {ep.id[:8].lower() for ep in episodes}
    node_content_map = {ep.id[:8].lower(): ep.content for ep in episodes}

    # Check citation history
    meta = store.load_meta()
    citations_seen = meta.get("citations_seen", False)

    # Validate graduations (demotes bad citations in-place)
    today = date.today().isoformat()
    grad_result = validate_graduations(
        text=text,
        valid_ids=valid_ids,
        today=today,
        node_content_map=node_content_map,
        citations_seen=citations_seen,
    )

    # Save the (possibly modified) continuity text
    path = store.save_continuity(grad_result.text)

    # Record Hebbian associations from validated co-citations + decay
    assoc_formed, assoc_strengthened, assoc_decayed = \
        process_wrap_associations(store, grad_result, affective_state)

    # Update metadata
    if grad_result.validated > 0 or grad_result.citation_counts:
        meta["citations_seen"] = True
    meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
    store.save_meta(meta)

    # Record wrap completion
    sections = measure_sections(grad_result.text)
    patterns = len(re.findall(r"\|\s*\d+x", grad_result.text))
    wrap_result = store.wrap_completed(
        episodes_compressed=len(episodes),
        continuity_chars=len(grad_result.text),
        graduations_validated=grad_result.validated,
        graduations_demoted=grad_result.demoted,
        citation_reuse_max=max(grad_result.citation_counts.values())
        if grad_result.citation_counts else 0,
        patterns_extracted=patterns,
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
    )

    return {
        "path": path,
        "episodes_compressed": len(episodes),
        "graduations_validated": grad_result.validated,
        "graduations_demoted": grad_result.demoted,
        "associations_formed": assoc_formed,
        "associations_strengthened": assoc_strengthened,
        "associations_decayed": assoc_decayed,
        "skipped_prepare": skipped_prepare,
        "wrap_result": wrap_result,
    }
