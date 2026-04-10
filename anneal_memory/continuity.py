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
from typing import TYPE_CHECKING, Any

from .graduation import detect_stale_patterns, validate_graduations
from .types import AffectiveState, Episode

if TYPE_CHECKING:
    from .store import Store

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
    """Pure helper — build an agent-facing compression package from pre-fetched inputs.

    .. warning::
        **Most callers should use :func:`prepare_wrap` instead.** That
        function is the canonical store-aware pipeline: it fetches
        episodes, loads the current continuity, builds this package,
        marks the wrap in progress (``store.wrap_started()``), and
        attaches Hebbian association context — the full lifecycle.

        This function does NONE of those things. It is a pure
        helper that takes episodes and continuity text already in hand
        and returns a package dict. It does not touch a store.
        Consequently, calling this function and then calling
        :func:`validated_save_continuity` on the same store will leave
        ``skipped_prepare=True`` in the save result, because no wrap
        lifecycle was ever started.

        This helper is kept public for (a) unit tests of package
        construction in isolation, and (b) advanced library users
        who manage their own episode fetching and wrap lifecycle.
        For the standard session-boundary wrap flow, use
        :func:`prepare_wrap`.

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


def prepare_wrap(
    store: Store,
    *,
    max_chars: int = 20000,
    staleness_days: int = 7,
) -> dict[str, Any]:
    """Run the full store-aware prepare_wrap pipeline.

    **This is the canonical prepare_wrap entry point.** The MCP server
    and the ``prepare-wrap`` CLI subcommand both call this function —
    they are thin transport adapters that delegate the domain work here
    and format the returned dict for their output surface.

    Handles the full lifecycle: fetches episodes, detects the empty
    case (and clears any stale wrap-in-progress flag), builds the
    agent-facing compression package via :func:`prepare_wrap_package`,
    marks the wrap as in progress, and attaches Hebbian association
    context for the episodes being compressed.

    Contrast with :func:`prepare_wrap_package`, which is a pure helper
    that takes pre-fetched episodes and continuity text and returns the
    package dict without touching the store. Most callers want this
    function; ``prepare_wrap_package`` is for tests and advanced users
    managing their own store state.

    .. warning::
        **The prepare/save window is not frozen.** The set of episodes
        returned here is a snapshot taken at call time. If the caller
        records additional episodes after ``prepare_wrap`` returns but
        before :func:`validated_save_continuity` is called,
        ``validated_save_continuity`` will re-fetch and compress the
        larger set, even though the agent's compression was based on
        the smaller set shown here. Graduation validation and the
        ``episodes_compressed`` metric will reflect the expanded set.

        For single-threaded agent workflows (the normal MCP and CLI
        case), this is not an issue — the agent reasons in one pass
        between prepare and save and does not record new episodes.
        For framework integrations where episode recording can
        interleave with session wrapping, you must serialize
        ``prepare_wrap`` → compression → ``validated_save_continuity``
        as a critical section with no intervening ``store.record()``
        calls.

        A session-handshake token to frozen the window is tracked as
        a v0.3.0 hardening follow-up.

    Args:
        store: A Store instance.
        max_chars: Maximum target size for the continuity file.
        staleness_days: Days before flagging stale patterns.

    Returns:
        Dict with keys:
          - ``status`` (str): ``"empty"`` (no episodes to wrap) or
            ``"ready"`` (package built, wrap marked in progress)
          - ``message`` (str): short human-readable status summary
          - ``episode_count`` (int): number of episodes in the wrap window
          - ``package`` (dict | None): the result of
            :func:`prepare_wrap_package`, or ``None`` if empty
          - ``assoc_context`` (str | None): Hebbian association context
            for the episodes being compressed, or ``None`` if empty or
            no associations exist

    Note:
        On ``status == "empty"`` the function calls ``wrap_cancelled()``
        on the store to clear any stale in-progress flag. On
        ``status == "ready"`` it calls ``wrap_started()``. Either way,
        the store's wrap lifecycle state is consistent after the call.
    """
    episodes = store.episodes_since_wrap()

    if not episodes:
        store.wrap_cancelled()
        return {
            "status": "empty",
            "message": "No episodes since last wrap. Nothing to compress.",
            "episode_count": 0,
            "package": None,
            "assoc_context": None,
        }

    # All store reads and package construction happen BEFORE wrap_started().
    # If any of them raises, the store is left with no stale wrap-in-progress
    # flag — symmetric with wrap_cancelled() on the empty path.
    existing = store.load_continuity()
    package = prepare_wrap_package(
        episodes=episodes,
        existing_continuity=existing,
        project_name=store.project_name,
        max_chars=max_chars,
        staleness_days=staleness_days,
    )
    episode_ids = [ep.id for ep in episodes]
    assoc_context = store.get_association_context(episode_ids) or None

    # Mark wrap in progress only after every upstream operation succeeded.
    store.wrap_started()

    return {
        "status": "ready",
        "message": f"Ready to compress {len(episodes)} episode(s).",
        "episode_count": len(episodes),
        "package": package,
        "assoc_context": assoc_context,
    }


def format_wrap_package_text(result: dict[str, Any]) -> str:
    """Render a :func:`prepare_wrap` result as agent-facing display text.

    This is the canonical text representation used by both MCP and CLI
    transports. It assembles the compression instructions, episode
    listing, existing continuity, stale patterns, and Hebbian context
    into a single markdown-formatted string ready to be handed to the
    agent doing the compression.

    Transports that want the canonical presentation call this; library
    users who want to format the package differently can build their
    own text from the structured dict instead.

    Args:
        result: The return value of :func:`prepare_wrap`.

    Returns:
        The formatted text. For an empty result, returns the status
        message unchanged.
    """
    # Any status other than "ready" (empty, or some future state) is
    # rendered as its bare message. Only "ready" has a package to format.
    if result["status"] != "ready":
        return result["message"]

    package = result["package"]
    parts: list[str] = [package["instructions"], "\n---\n"]
    parts.append(f"## Episodes This Session ({package['episode_count']})")
    parts.append(package["episodes"])

    if package["continuity"]:
        parts.append("\n---\n## Current Continuity File")
        parts.append(package["continuity"])
    else:
        parts.append(
            "\n---\n(No existing continuity file — this is the first wrap.)"
        )

    if package["stale_patterns"]:
        parts.append("\n---\n## Stale Patterns (consider removing)")
        for sp in package["stale_patterns"]:
            parts.append(
                f"- Line {sp['line']}: {sp['content']}"
                f" ({sp['days_stale']}d stale)"
            )

    if result["assoc_context"]:
        parts.append("\n---\n" + result["assoc_context"])

    return "\n".join(parts)


def validated_save_continuity(
    store: Store,
    text: str,
    affective_state: AffectiveState | None = None,
) -> dict[str, Any]:
    """Save continuity with the full validation pipeline.

    **This is the canonical save_continuity pipeline.** The MCP server and
    the ``save-continuity`` CLI subcommand both call this function — they
    are thin transport adapters that parse their inputs, delegate the
    domain work here, and format their outputs. Library users calling this
    function get the exact same pipeline as MCP and CLI users.

    The pipeline: structure validation → citation-based graduation
    validation → save → Hebbian association formation → decay → metadata
    update → wrap completion. Every stage of the immune system runs.

    Use this instead of bare ``store.save_continuity()`` — the raw store
    method is a file write that bypasses graduation, associations, and
    decay. ``validated_save_continuity`` is what you want whenever an
    agent has finished compressing its session.

    .. warning::
        **The episode set is re-fetched at save time.** This function
        calls ``store.episodes_since_wrap()`` internally. If the caller
        recorded additional episodes between :func:`prepare_wrap` and
        this call, the new episodes are included in validation and
        metrics even though the agent's compression text was produced
        from the smaller set that ``prepare_wrap`` showed. For
        single-threaded agent workflows this is not an issue. For
        framework integrations that interleave episode recording with
        session wrapping, you must treat the ``prepare_wrap`` →
        compression → ``validated_save_continuity`` sequence as a
        critical section. See :func:`prepare_wrap` for the full
        discussion.

    Args:
        store: A Store instance.
        text: The agent-compressed continuity text.
        affective_state: Optional agent functional state during this wrap.

    Returns:
        Dict with keys:
          - ``path`` (str): path to the saved continuity file
          - ``chars`` (int): byte count of the saved continuity text
            (top-level convenience for transports; same as
            ``wrap_result.chars``)
          - ``episodes_compressed`` (int): count of episodes in this wrap
          - ``graduations_validated`` (int): citations that validated
          - ``graduations_demoted`` (int): citations demoted due to
            bad/missing evidence, *including* bare graduations
          - ``demoted`` (int): citations demoted due to bad evidence only
          - ``bare_demoted`` (int): bare (evidence-free) 2x/3x
            graduations demoted for missing citations
          - ``citation_reuse_max`` (int): max times any single episode
            was cited in this wrap
          - ``gaming_suspects`` (list[str]): episode IDs flagged for
            suspicious citation reuse
          - ``associations_formed`` (int)
          - ``associations_strengthened`` (int)
          - ``associations_decayed`` (int)
          - ``sections`` (dict[str, int]): char count per continuity section
          - ``skipped_prepare`` (bool): True if ``prepare_wrap`` was not
            called first
          - ``wrap_result`` (WrapResult): the store-level wrap record
            (contains the same metrics — primarily for library users
            who want a typed object rather than a dict)

    Raises:
        ValueError: If text is empty or missing required sections.
        OSError: If the filesystem write of the continuity sidecar
            fails. Not caught by this function; propagates to the
            caller. MCP and CLI transports currently let it propagate
            as well — consider handling at your transport boundary.
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
    total_demoted = grad_result.demoted + grad_result.bare_demoted
    wrap_result = store.wrap_completed(
        episodes_compressed=len(episodes),
        continuity_chars=len(grad_result.text),
        graduations_validated=grad_result.validated,
        graduations_demoted=total_demoted,
        citation_reuse_max=grad_result.citation_reuse_max,
        patterns_extracted=patterns,
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
    )

    return {
        "path": path,
        "chars": len(grad_result.text),
        "episodes_compressed": len(episodes),
        "graduations_validated": grad_result.validated,
        "graduations_demoted": total_demoted,
        "demoted": grad_result.demoted,
        "bare_demoted": grad_result.bare_demoted,
        "citation_reuse_max": grad_result.citation_reuse_max,
        "gaming_suspects": list(grad_result.gaming_suspects),
        "associations_formed": assoc_formed,
        "associations_strengthened": assoc_strengthened,
        "associations_decayed": assoc_decayed,
        "sections": sections,
        "skipped_prepare": skipped_prepare,
        "wrap_result": wrap_result,
    }
