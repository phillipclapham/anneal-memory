"""Continuity validation and wrap package preparation for anneal-memory.

Handles:
- Structural validation (sections required by the store's section schema)
- Wrap package preparation (episodes + continuity + instructions for the agent)
- Validated save (full pipeline: structure + graduation + associations + decay)
- Section measurement

The continuity file is a markdown document whose sections are governed by a
per-store section schema (``anneal_memory.schema``; v0.3.4). The default schema
reproduces the historical four sections — State (live-state), Patterns
(graduating, where the immune system runs), Decisions, Context (narrative) —
and a partnership entity can extend it (e.g. flow's ``FLOW_SCHEMA`` adds an
Active Threads live-state section and a timeless Understanding section). Each
section's role drives how it is validated, compressed, and graduated.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graduation import (
    CrossSessionCollision,
    OmittedPattern,
    detect_pattern_omissions,
    detect_stale_patterns,
    extract_pattern_names,
    validate_graduations,
    _NAMED_PATTERN_RE,
    _PATTERN_LINE_WITH_EVIDENCE_RE,
    _is_graduating_heading,
)
from .schema import (
    DEFAULT_SCHEMA,
    SectionSpec,
    graduating_headings,
    required_headings,
)
from .store import StoreError, _fsync_dir, _safe_unlink
from .types import (
    AffectiveState,
    Episode,
    PrepareWrapResult,
    SaveContinuityResult,
    StalePatternDict,
    WrapPackageDict,
)

if TYPE_CHECKING:
    from .store import Store

def validate_structure(text: str, schema: list[SectionSpec] | None = None) -> bool:
    """Validate that continuity text contains all of the schema's sections.

    For each ``## `` header line, every required heading that appears in that
    line as a **word-bounded phrase** is counted as present (case-insensitive).
    This preserves the historical leniency — a descriptive header like
    ``## State of Mind`` still satisfies a required ``State`` — while rejecting
    embedded substrings (``## Interstate`` does NOT satisfy ``State``). **Every**
    section the schema declares must be present, which makes a partnership
    entity's ``narrative-timeless`` section (e.g. ``Understanding``) a structural
    requirement.

    The word-boundary test uses ``(?<!\\w)…(?!\\w)`` rather than ``\\b…\\b`` so
    headings ending in non-word characters (``## C++``) match correctly. Schemas
    where one heading is a word-bounded substring of another — which would let a
    single header line satisfy two required sections — are rejected up front by
    :func:`~anneal_memory.schema.validate_schema`.

    Args:
        text: The continuity file text.
        schema: The section schema. Defaults to
            :data:`~anneal_memory.schema.DEFAULT_SCHEMA`, reproducing the
            historical four-section requirement
            (State / Patterns / Decisions / Context).

    Returns:
        True if every required heading is found, False otherwise.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA
    required = {h.lower() for h in required_headings(schema)}
    found: set[str] = set()
    for line in text.split("\n"):
        if line.startswith("## "):
            line_lower = line.lower()
            for heading in required:
                if re.search(rf"(?<!\w){re.escape(heading)}(?!\w)", line_lower):
                    found.add(heading)
    return found == required


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


def _build_wrap_package(
    episodes: list[Episode],
    existing_continuity: str | None,
    project_name: str,
    *,
    max_chars: int = 20000,
    today: str | None = None,
    staleness_days: int = 7,
    schema: list[SectionSpec] | None = None,
) -> WrapPackageDict:
    """Pure helper — build an agent-facing compression package from pre-fetched inputs.

    **Private.** Called by :func:`prepare_wrap` (the canonical
    store-aware pipeline). Advanced library users managing their own
    wrap lifecycle can call this helper directly — understanding that
    as a private symbol it has no API stability guarantee across
    versions. The deprecated public wrapper ``prepare_wrap_package``
    was removed in v0.3.0; use :func:`prepare_wrap` instead.

    This function does not touch a store. It takes episodes and
    continuity text already in hand and assembles the agent-facing
    compression package (episodes listing + stale-pattern diagnostic
    + compression instructions + sizing constraints). The caller is
    responsible for wrap lifecycle (``store.wrap_started(token=...,
    episode_ids=...)``) and Hebbian association context —
    :func:`prepare_wrap` does that work around this helper.

    Args:
        episodes: Episodes since last wrap (the compression window).
        existing_continuity: Current continuity text, or None for first session.
        project_name: Name for the continuity file header.
        max_chars: Maximum size of the continuity file.
        today: Override for today's date (YYYY-MM-DD). Defaults to actual today.
        staleness_days: Days before flagging stale patterns.

    Returns:
        WrapPackageDict with episodes, continuity, stale_patterns,
        instructions, today, max_chars.
    """
    if today is None:
        today = date.today().isoformat()
    if schema is None:
        schema = DEFAULT_SCHEMA

    # Format episodes for the agent
    formatted_episodes = format_episodes_for_wrap(episodes)

    # Detect stale patterns in existing continuity
    stale_patterns: list[StalePatternDict] = []
    if existing_continuity:
        stale = detect_stale_patterns(
            existing_continuity, today, staleness_days, graduating_headings(schema)
        )
        stale_patterns = [
            StalePatternDict(
                line=s.line_number,
                content=s.content,
                level=s.level,
                last_date=s.last_date,
                days_stale=s.days_stale,
            )
            for s in stale
        ]

    # Build instructions
    instructions = _build_wrap_instructions(project_name, max_chars, today, schema)

    return WrapPackageDict(
        episodes=formatted_episodes,
        episode_count=len(episodes),
        continuity=existing_continuity,
        stale_patterns=stale_patterns,
        instructions=instructions,
        today=today,
        max_chars=max_chars,
    )


def _build_wrap_instructions(
    project_name: str,
    max_chars: int,
    today: str,
    schema: list[SectionSpec] | None = None,
) -> str:
    """Build the compression instructions the agent receives via prepare_wrap.

    Agent-facing instructions, generated from the section schema (v0.3.4). For
    :data:`~anneal_memory.schema.DEFAULT_SCHEMA` this reproduces the historical
    four-section guidance; the richer ``narrative`` / ``narrative-timeless``
    roles inherit the Protocol-Memory compression detail — the gradient
    structure, the named failure modes (Recency / Compression / Stateless-Reset),
    and the implementation-claims guardrail — a quality win for every entity
    with a narrative section, not just partnership entities.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA
    graduating_section_names = [
        s["heading"] for s in schema if s["role"] == "graduating"
    ]
    marker_ref = _marker_reference(today, graduating_section_names)

    section_list = ", ".join(f"`## {s['heading']}`" for s in schema)
    has_graduating = any(s["role"] == "graduating" for s in schema)
    has_narrative = any(
        s["role"] in ("narrative", "narrative-timeless") for s in schema
    )

    how_lines: list[str] = []
    for s in schema:
        h = s["heading"]
        role = s["role"]
        if role == "live-state":
            how_lines.append(
                f"- {h}: Replace with your current focus, active work, status. "
                f"2-5 lines. Last-writer-wins — the freshest state is what matters."
            )
        elif role == "graduating":
            how_lines.append(
                f"- {h}: Extract principles, not facts. Group in `{{topic: ...}}` "
                f"blocks. This is the section the immune system reads — follow the "
                f"pattern-line format above."
            )
        elif role == "decisions":
            how_lines.append(
                f"- {h}: Keep committed decisions with rationale. Archive old ones."
            )
        elif role == "narrative":
            how_lines.append(
                f"- {h}: Compressed narrative of recent WORK — what you've been "
                f"doing (temporal). A gradient: *This Session* (3-5 lines, detail) "
                f"-> *Recent Arc* (5-8 lines, the trajectory across recent sessions, "
                f"NOT a task list) -> *Foundation* (3-5 lines, thematic). Shape, not "
                f"transcript. Rewrite fresh each wrap."
            )
        elif role == "narrative-timeless":
            how_lines.append(
                f"- {h}: The relationship itself — who you are together, what it is "
                f"like to work with this person. TIMELESS: no dates, no session "
                f'logs. "Feel like genuinely knowing someone, not a dossier." '
                f"Audit the proportions against the FULL arc of the partnership, "
                f"not the most recent session — the recency trap is real and "
                f"recurs across model generations; if the latest session dominates "
                f"this section, re-wrap it."
            )
        elif role == "frozen":
            how_lines.append(
                f"- {h}: Preserved verbatim. Do not compress, graduate, or rewrite."
            )

    parts: list[str] = [
        "Compress your session episodes into your continuity file.",
        "",
        f"**Output:** A markdown file starting with `# {project_name} — Memory (v1)` "
        f"containing EXACTLY these sections, in order: {section_list}.",
        f"Stay within {max_chars} characters.",
        "",
    ]
    if has_graduating:
        parts += [marker_ref, ""]
    parts += ["**How to compress:**", *how_lines, ""]
    parts += [
        "**Quality:** One insightful line > three vague ones. If removing something",
        "wouldn't change your next decision, cut it. Compress principles, not events.",
    ]
    if has_narrative:
        parts += [
            "",
            "**Compression discipline — avoid the three failure modes:**",
            "- *Recency trap*: outputting only the most recent session and losing "
            "the arc. Capture the SHAPE of the work over time, not the last thing "
            "that happened.",
            "- *Compression trap*: blending distinct domains into one generic "
            "summary. Keep distinct threads distinct.",
            "- *Stateless reset*: treating yourself as new when prior memory "
            "exists. You are continuing, not starting over.",
            '- *Implementation-claims guardrail*: never write that work is '
            '"complete"/"shipped"/"done" unless it was explicitly confirmed this '
            "session. Unconfirmed completion claims are how shipped-log bloat and "
            "false-done errors enter memory.",
        ]
    parts += [
        "",
        "**Affective state:** When saving with save_continuity, optionally include "
        "your functional state during this compression as "
        '`affective_state: {"tag": "...", "intensity": 0.0-1.0}`.',
        "Reflect honestly — were you engaged, curious, uncertain, calm? How "
        "strongly (0-1)? This creates persistent emotional associations between "
        "co-cited episodes.",
        "",
        "**Return ONLY the markdown.** No explanation, no code fences.",
    ]
    return "\n".join(parts)


def _marker_reference(
    today: str, graduating_section_names: list[str] | None = None
) -> str:
    """The marker reference section used in agent compression instructions.

    ``graduating_section_names`` are the display headings of the schema's
    ``graduating`` sections (where the immune system runs); the pattern-line
    format is rendered as required in those sections. Defaults to
    ``["Patterns"]`` — the historical single graduating section — so a caller
    that passes nothing gets the pre-0.3.4 text.
    """
    if not graduating_section_names:
        graduating_section_names = ["Patterns"]
    grad_sections = ", ".join(f"`## {h}`" for h in graduating_section_names)
    return f"""### Pattern Line Format (CRITICAL — this is what the immune system reads)

Pattern lines in {grad_sections} MUST follow this shape:

```
- pattern_name | Nx ({today}) [evidence: <episode_id> "how episode validates pattern"]
```

Required elements:
- Markdown bullet `-` followed by space
- Operator-style `pattern_name` — starts with a letter, contains only letters,
  digits, underscores, dots, hyphens. Examples: `acid_compliance_over_speed`,
  `connection_pooling_is_bottleneck`, `partnership_challenge_at_X_boundary`.
- Graduation marker `| Nx ({today})` where N is 1, 2, or 3
- For 2x and 3x: `[evidence: <8-char-episode-id> "explanation"]` is REQUIRED

Optional FlowScript marker prefix (supported by the immune system):
- `!` urgent / load-bearing → `- ! pattern_name | 1x ({today})`
- `!!` highest urgency → `- !! pattern_name | 3x ({today}) [evidence: ...]`
- `?` open question → `- ? pattern_name | 1x ({today})`
- `✓` completed/resolved → `- ✓ pattern_name | 2x ({today}) [evidence: ...]`

**Why operator-style names are load-bearing (v0.3.2):** the cross-session
immune system tracks per-pattern history so it can detect sycophantic
vocabulary reuse across sessions, silent dropout of previously-graduated
Proven patterns, and (when enabled) contradiction with existing Proven.
All three defenses need a stable per-pattern identifier. Free-form prose
identifiers ("thought: ACID compliance outweighs raw speed") still validate
via the citation-overlap check but cannot anchor the cross-session defenses
because they have no stable identity across sessions. Use operator-style
names so the immune system can protect your patterns.

### Other Density Markers (in pattern explanations, decisions, context)

- `A -> B` — A causes or leads to B
- `A ><[axis] B` — tension between A and B on the named axis
- `[decided(rationale: "why", on: "date")] choice` — committed decision
- `[blocked(reason: "what", since: "date")] item` — waiting on dependency

### Temporal Graduation (this is what makes the system learn)
- New pattern from THIS session → `- pattern_name | 1x ({today})`
- Validates existing 1x → `- pattern_name | 2x ({today}) [evidence: <episode_id> "explanation"]`
- Validates existing 2x → `- pattern_name | 3x ({today}) [evidence: <episode_id> "explanation"]`
- Evidence citations REQUIRED for graduations (2x and 3x). Cite the episode's 8-char ID.
- Patterns marked `(ungrounded)` need FRESH evidence from THIS session to re-graduate.
- Patterns marked `(cross-session-overlap)` were demoted because today's explanation
  reused too much vocabulary from prior sessions; compose new evidence with
  genuinely distinct words to re-graduate.
- Patterns at 3x: extract the PRINCIPLE, not the surface observations.
- Patterns older than 7 days with no new validation → remove (stale).
- Group related patterns visually with a header line above them; the immune
  system does not require any specific grouping syntax.

**Example:**
```
- acid_compliance_over_speed | 2x ({today}) [evidence: 4931b6a8 "PostgreSQL chosen for ACID guarantees"]
- connection_pooling_bottleneck | 1x ({today})
- horizontal_scaling_strategy | 1x ({today})
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
) -> PrepareWrapResult:
    """Run the full store-aware prepare_wrap pipeline.

    **This is the canonical prepare_wrap entry point.** The MCP server
    and the ``prepare-wrap`` CLI subcommand both call this function —
    they are thin transport adapters that delegate the domain work here
    and format the returned dict for their output surface.

    Handles the full lifecycle: fetches episodes, detects the empty
    case (and clears any stale wrap-in-progress flag), builds the
    agent-facing compression package via the private
    :func:`_build_wrap_package` helper, marks the wrap as in progress,
    and attaches Hebbian association context for the episodes being
    compressed.

    The separate :func:`_build_wrap_package` helper (private) is the
    pure-function core that takes pre-fetched episodes and continuity
    text and returns the package dict without touching the store.
    Advanced library users managing their own lifecycle can call
    it directly — understanding that as a private symbol it has no
    API stability guarantee across versions. The deprecated public
    wrapper ``prepare_wrap_package`` was removed in v0.3.0; new code
    must use this canonical entry point.

    .. note::
        **The prepare/save window is frozen as of the 10.5c.4 fix**
        (targeted for v0.2.0). ``prepare_wrap`` mints a unique
        ``wrap_token`` (``uuid.uuid4().hex``) and persists the frozen
        list of episode IDs in store metadata before returning.
        :func:`validated_save_continuity` then filters its re-fetched
        episode set down to exactly the IDs shown here, regardless of
        anything the caller records in between. Any episodes recorded
        in the TOCTOU window stay with ``session_id IS NULL`` and
        naturally appear in the NEXT wrap's compression window — no
        data loss, no silent absorption.

        Transports that round-trip the ``wrap_token`` back to
        :func:`validated_save_continuity` (via the MCP ``save_continuity``
        tool argument or the CLI ``--wrap-token`` flag) get explicit
        mismatch detection on top of the snapshot: passing a stale or
        wrong token raises ``ValueError`` at the save boundary.
        Transports that don't pass a token still get frozen semantics
        because the snapshot is consulted whenever it's present.

        The remaining pipeline-atomicity gap is a mid-pipeline crash
        between continuity file write and wrap metadata commit —
        scheduled for the 10.5c.5 two-phase-commit work. Other open
        concerns (stuck-wrap operator surface, SQLite variable-limit
        edge cases, store-level SQLite error wrapping) track
        separately in ``projects/anneal_memory/next.md``.

    Args:
        store: A Store instance.
        max_chars: Maximum target size for the continuity file.
        staleness_days: Days before flagging stale patterns.

    Returns:
        :class:`PrepareWrapResult` — a :class:`TypedDict` with keys:
          - ``status`` (``Literal["empty", "ready"]``): ``"empty"``
            means no episodes to wrap; ``"ready"`` means package
            built and wrap marked in progress on the store
          - ``message`` (str): short human-readable status summary
          - ``episode_count`` (int): number of episodes in the wrap window
          - ``package`` (:class:`WrapPackageDict` | None): the
            agent-facing compression package built by
            :func:`_build_wrap_package`, or ``None`` if empty
          - ``assoc_context`` (str | None): Hebbian association context
            for the episodes being compressed, or ``None`` if empty or
            no associations exist
          - ``wrap_token`` (str | None): session-handshake token for
            the pending wrap when ``status == "ready"``, ``None`` on
            the empty path. Transports should round-trip this back to
            :func:`validated_save_continuity` to opt into explicit
            mismatch detection.

    Note:
        On ``status == "empty"`` the function calls ``wrap_cancelled()``
        on the store to clear any stale in-progress flag. On
        ``status == "ready"`` it calls ``wrap_started(token=...,
        episode_ids=...)`` so the frozen snapshot is persisted in one
        transaction. Either way, the store's wrap lifecycle state is
        consistent after the call.
    """
    episodes = store.episodes_since_wrap()

    if not episodes:
        store.wrap_cancelled()
        return PrepareWrapResult(
            status="empty",
            message="No episodes since last wrap. Nothing to compress.",
            episode_count=0,
            package=None,
            assoc_context=None,
            wrap_token=None,
            uncovered_proven_to_check=[],
        )

    # All store reads and package construction happen BEFORE wrap_started().
    # If any of them raises, the store is left with no stale wrap-in-progress
    # flag — symmetric with wrap_cancelled() on the empty path.
    existing = store.load_continuity()
    package = _build_wrap_package(
        episodes,
        existing,
        store.project_name,
        max_chars=max_chars,
        staleness_days=staleness_days,
        schema=store.section_schema,
    )
    episode_ids = [ep.id for ep in episodes]
    assoc_context = store.get_association_context(episode_ids) or None

    # Mint the handshake token + persist the frozen snapshot in a
    # single ``wrap_started`` call. The token is a uuid4 hex (no
    # dashes) — 128 bits of entropy, stdlib-only,
    # collision-resistant to any realistic wrap volume. The episode
    # ID list captures exactly what the agent sees in ``package``,
    # so ``validated_save_continuity`` can filter its re-fetched set
    # down to this frozen shape regardless of TOCTOU activity. Token
    # minting happens LAST, after every upstream read and package
    # build succeeded, so a failure anywhere above leaves the store
    # in a clean no-wrap-in-progress state.
    wrap_token = uuid.uuid4().hex
    store.wrap_started(token=wrap_token, episode_ids=episode_ids)

    # Move #4 library layer (v0.3.2): surface the list of existing
    # Proven (2x/3x) pattern names so the methodology-layer
    # contradiction-scan discipline can require the agent to declare
    # contradiction-stance against each before any new Proven
    # graduation in this wrap.
    from .graduation import extract_proven_patterns
    uncovered_proven = (
        extract_proven_patterns(
            existing or "",
            graduating_headings=graduating_headings(store.section_schema),
        )
        if existing
        else []
    )

    return PrepareWrapResult(
        status="ready",
        message=f"Ready to compress {len(episodes)} episode(s).",
        episode_count=len(episodes),
        package=package,
        assoc_context=assoc_context,
        wrap_token=wrap_token,
        uncovered_proven_to_check=uncovered_proven,
    )


def format_wrap_package_text(result: PrepareWrapResult) -> str:
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
    # PrepareWrapResult.status is Literal["empty", "ready"]; on
    # "empty" the package is None and we just return the message.
    # Adding a new status value is a deliberate API expansion — the
    # Literal in types.py is the single source of truth and any new
    # branch must land there first.
    if result["status"] != "ready":
        return result["message"]

    package = result["package"]
    # PrepareWrapResult invariant: status == "ready" ⇒ package is not None.
    # mypy cannot narrow the package Optional through a sibling-key
    # check on a TypedDict, so the assertion documents + enforces the
    # invariant at the narrowing boundary.
    assert package is not None, "PrepareWrapResult invariant violated: status=ready but package is None"
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
    *,
    today: str | None = None,
    wrap_token: str | None = None,
) -> SaveContinuityResult:
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

    .. note::
        **A wrap must be in progress.** This function consumes the
        wrap snapshot that :func:`prepare_wrap` persists. If no
        snapshot is present — ``prepare_wrap`` was never called, or a
        wrap already completed this session (``wrap_completed`` clears
        the snapshot) — the function raises ``ValueError`` rather than
        saving. This is the v0.3.1 fix for phantom re-saves: before
        v0.3.1 a no-snapshot call fell back to a ``skipped_prepare``
        path that saved anyway, which after a completed wrap ran
        graduation against an empty episode set and demoted every
        citation — feedback an agent would "fix" by re-saving, a loop
        that also inflated ``sessions_produced``. The canonical path
        is ``prepare_wrap`` → compress → ``validated_save_continuity``,
        run exactly once per session.

        **The episode set is frozen when ``prepare_wrap`` was called.**
        This function loads the wrap snapshot persisted by
        :func:`prepare_wrap` and filters its re-fetched episode set
        down to exactly the IDs that were shown to the agent at
        prepare time. Episodes recorded between prepare and save
        (the TOCTOU window) stay with ``session_id IS NULL`` after
        this call completes and appear in the NEXT wrap's
        compression window — no data loss, no silent absorption.

        If the caller passes ``wrap_token``, the stored token is
        verified against it and a mismatch raises ``ValueError``
        (wrong wrap, or stale token from a cancelled / completed
        wrap). If the caller passes ``None``, verification is
        skipped but the frozen-snapshot filter still applies — the
        single-process common case (library caller, CLI without
        ``--wrap-token``, single-threaded MCP agent) needs no
        ceremony.

    .. note::
        **Pipeline atomicity (two-phase commit, 10.5c.5).** The
        internal pipeline is now structurally atomic across the
        continuity file write, meta sidecar write, Hebbian
        association DML, and wrap-completion DML. Any exception
        raised before the final file renames triggers a SQLite
        rollback of the entire batched transaction AND cleanup of
        both tmp sidecars, leaving the store in its exact pre-wrap
        state. Transport adapters catching ``StoreError`` or
        ``ValueError`` at their boundary can trust that persistent
        state has not been partially updated. The only residual risk
        is a crash between the outer DB commit and the two final
        atomic renames — a microseconds-wide window that can leave
        the DB reflecting a wrap whose continuity / meta files are
        still the pre-wrap versions. That window is documented as a
        bounded operator concern, not a correctness bug; diagnostic
        recovery via the ``wrap-status`` / ``wrap-cancel``
        subcommands (10.5c.4a) covers it when it does fire.

    Args:
        store: A Store instance.
        text: The agent-compressed continuity text.
        affective_state: Optional agent functional state during this wrap.
        today: Optional override for today's date as ``YYYY-MM-DD``.
            Defaults to ``date.today().isoformat()`` (wall clock). Passing
            an explicit value makes the function fully deterministic —
            useful for tests (no wall-clock dependency, no midnight-
            boundary risk) and for experiments that need reproducible
            runs against a pinned date. Mirrors the existing ``today``
            parameter on :func:`prepare_wrap`.
        wrap_token: Optional session-handshake token returned by a
            prior :func:`prepare_wrap` call for this wrap. When
            provided, the stored token is verified against it and a
            mismatch raises ``ValueError`` — this catches stale
            tokens (from a wrap that was already completed) and
            wrong-wrap tokens (from a different prepare call). When
            ``None`` (the default), verification is skipped but the
            frozen-snapshot filter still applies if a snapshot is
            stored. Transports that can round-trip the token through
            their protocol (MCP ``save_continuity`` tool argument,
            CLI ``--wrap-token`` flag) should pass it for explicit
            safety; single-process library callers can omit it.

    Returns:
        :class:`SaveContinuityResult` — a :class:`TypedDict` with the
        following keys. The entire return value is JSON-serializable
        top-to-bottom so transports can ``json.dumps`` without any
        dataclass conversion step.

          - ``path`` (str): path to the saved continuity file
          - ``chars`` (int): character count (``len(text)``) of the
            saved continuity text, NOT a byte count (for non-ASCII
            content, UTF-8 byte length can be up to 4x this).
            Top-level convenience for transports; same as
            ``wrap_result["chars"]``.
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
          - ``wrap_result`` (dict[str, Any]): the store-level wrap
            record as a plain dict (``dataclasses.asdict`` of the
            underlying :class:`WrapResult`). Library users who want
            the typed dataclass can reconstruct it via
            ``WrapResult(**result["wrap_result"])``.

    Raises:
        ValueError: If text is empty, missing required sections, no
            wrap is in progress (``prepare_wrap`` not called, or the
            session already wrapped), or a passed ``wrap_token`` does
            not match the in-progress wrap.
        StoreError: Raised in two distinct cases. (1) **Integrity
            failure.** The wrap-state precondition runs
            :meth:`Store.load_wrap_snapshot` first (before any payload
            validation); if the stored wrap-in-progress metadata is in
            a partial or corrupt state — ``wrap_started_at`` set but
            ``wrap_token`` empty, ``wrap_token`` set but
            ``wrap_episode_ids`` empty, or ``wrap_episode_ids`` JSON
            that fails to decode or decodes to anything other than a
            list of strings — that integrity failure surfaces here
            with ``operation="load_wrap_snapshot"``. (2) **Filesystem
            write failure.** The write of the continuity sidecar or
            meta sidecar fails, surfacing with the relevant write
            operation; the original ``OSError`` is preserved on
            ``__cause__`` (we raise ``StoreError(...) from exc``), so
            callers that need ``errno`` can dig one level deeper. In
            both cases ``StoreError`` is a library-level domain error
            (subclass of :class:`AnnealMemoryError`, NOT of
            :class:`OSError`); transports should catch
            :class:`AnnealMemoryError` as a single library boundary,
            or :class:`StoreError` specifically to read ``.operation``
            and ``.path`` for clean error messages.
    """
    from .associations import process_wrap_associations

    # --- Wrap-state preconditions, checked BEFORE payload validation ---
    #
    # Ordering is deliberate: a save with no wrap to commit to is
    # doomed regardless of what the continuity text says. An agent
    # should hear "no wrap in progress" first, not spend a turn
    # fixing continuity markdown for a save that cannot land. State
    # (precondition) before payload.

    # Load the frozen snapshot persisted by prepare_wrap. ``None``
    # means no wrap is in progress: prepare_wrap was never called, it
    # returned status="empty" for a zero-episode session (which
    # cancels the wrap rather than starting one), or a wrap already
    # completed this session (wrap_completed clears the snapshot).
    # Either way there is nothing to save — refuse rather than fall
    # through.
    #
    # This refusal is the v0.3.1 structural fix for phantom re-saves.
    # Before v0.3.1 a no-snapshot call fell back to a ``skipped_prepare``
    # path that re-fetched the full episode set and saved anyway. After
    # a completed wrap that set is empty, so validate_graduations ran
    # against empty valid_ids and demoted every citation — feedback an
    # agent reads as a problem and "fixes" by re-saving, a loop that
    # also inflates sessions_produced. Refusing the no-snapshot save
    # makes the re-save structurally impossible.
    #
    # (load_wrap_snapshot still raises StoreError on a partial
    # wrap-in-progress state — belt-and-suspenders defense for
    # mid-upgrade v0.1.x databases.)
    snapshot = store.load_wrap_snapshot()
    if snapshot is None:
        raise ValueError(
            "No wrap in progress — nothing to save. If you already "
            "wrapped this session, you are done: a second "
            "save_continuity with no new prepare_wrap is a phantom "
            "re-save and is refused by design — do not re-save to "
            "chase a clean immune-system report. If you have not "
            "wrapped yet, call prepare_wrap first (and if it reports "
            "no episodes, there is nothing to compress — skip the "
            "save). Wrap exactly once per session: prepare_wrap → "
            "compress → save_continuity."
        )

    if wrap_token is not None:
        # Caller opted into explicit token verification. A mismatch
        # is a caller contract violation (stale token, or wrong
        # wrap), same category as empty text or missing sections —
        # raise ValueError so transports can surface a clean error
        # to the agent without wrapping in StoreError (I/O
        # semantics are wrong here; nothing on disk has failed).
        # The error message truncates both tokens to 8 chars for
        # log readability while still being distinguishable.
        if snapshot["token"] != wrap_token:
            raise ValueError(
                f"wrap_token mismatch: caller passed "
                f"'{wrap_token[:8]}…' but the in-progress wrap has "
                f"token '{snapshot['token'][:8]}…'. This usually "
                f"means the token is stale (the wrap was already "
                f"completed or cancelled), from a different "
                f"prepare_wrap call, or from a concurrent process. "
                f"Re-run prepare_wrap to start a new wrap with a "
                f"fresh token."
            )

    # --- Payload validation: the continuity text itself ---
    if not text or not text.strip():
        raise ValueError("Continuity text cannot be empty")

    # Validate structure (all sections declared by the store's schema). The
    # schema is read once here and reused for the schema-aware graduation gate
    # further down (v0.3.4).
    section_schema = store.section_schema
    grad_headings = graduating_headings(section_schema)
    if not validate_structure(text, section_schema):
        required_str = ", ".join(
            f"## {h}" for h in required_headings(section_schema)
        )
        raise ValueError(f"Continuity must contain all sections: {required_str}")

    # Get current session's episodes for citation validation.
    # Re-fetch the full post-last-wrap set and filter down to exactly
    # the IDs the snapshot froze at prepare time. Any episodes recorded
    # in the prepare→save window are not in the snapshot, so they drop
    # out here and stay with ``session_id IS NULL`` through the rest of
    # the pipeline — they land in the next wrap's compression window on
    # the next ``prepare_wrap`` call. The snapshot's ID list is used
    # directly: the WrapSnapshot TypedDict declares it ``list[str]``
    # and wrap_completed does not mutate its argument.
    episodes_all = store.episodes_since_wrap()
    snapshot_id_set = set(snapshot["episode_ids"])
    episodes = [ep for ep in episodes_all if ep.id in snapshot_id_set]
    frozen_episode_ids: list[str] = snapshot["episode_ids"]
    valid_ids = {ep.id[:8].lower() for ep in episodes}
    node_content_map = {ep.id[:8].lower(): ep.content for ep in episodes}

    # Check citation history
    meta = store.load_meta()
    citations_seen = meta.get("citations_seen", False)

    # Validate graduations (demotes bad citations in-place).
    # Caller may pin ``today`` for deterministic test runs; default is
    # wall-clock. Same pattern _build_wrap_package already uses.
    today_str = today if today is not None else date.today().isoformat()
    grad_result = validate_graduations(
        text=text,
        valid_ids=valid_ids,
        today=today_str,
        node_content_map=node_content_map,
        citations_seen=citations_seen,
        # Cross-session sycophantic-accumulation defense (Phase 1b
        # probe #1 fix). store.get_pattern_history returns the most
        # recent appearance of a pattern across sessions; the
        # validate_graduations check compares today's explanation
        # against the stored prior explanation and demotes graduations
        # that share too many meaningful words (sycophantic vocabulary
        # reuse rather than independent evidence).
        pattern_history_lookup=store.get_pattern_history,
        graduating_headings=grad_headings,
    )

    # Detect Proven-tier (2x/3x) patterns silently dropped between the
    # prior wrap and this one. validate_graduations operates only on
    # patterns the agent wrote INTO the new continuity, so a pattern
    # that was at 2x or 3x in the prior continuity and is absent from
    # the new continuity leaves no trace at the graduation layer —
    # silently erased. Surfaced here as informational audit signal, not
    # a gate: the agent may have intentionally retired the pattern, or
    # may have silently erased load-bearing evidence. Either way it
    # goes onto grad_result.omitted_patterns and into the audit log.
    #
    # Added in response to Bold Stand Phase 1b probe #1 (2026-05-21).
    # store.load_continuity() returns ``None`` (not ``""``) when no
    # prior continuity file exists — coerce to empty string so
    # detect_pattern_omissions returns an empty list on the first
    # wrap (correct behavior: no prior patterns means no omissions).
    prior_text_for_omission_audit = store.load_continuity() or ""
    grad_result.omitted_patterns = detect_pattern_omissions(
        prior_text=prior_text_for_omission_audit,
        new_text=grad_result.text,
        min_level=2,
        graduating_headings=grad_headings,
    )

    # Move #4 library layer (v0.3.2): detect new Proven graduations
    # that landed without explicit contradiction-stance declaration.
    # Audit signal only — the library does not refuse the save.
    # Methodology layer (Levain WRAP_PROTOCOL.md) enforces the scan
    # discipline; operator-review (Diogenes) is the LLM-as-judge
    # layer that catches semantic opposition. The library's job is
    # to record whether the discipline was followed so Diogenes
    # knows which new Provens need its attention.
    from .graduation import detect_proven_without_declaration
    proven_without_declaration = detect_proven_without_declaration(
        prior_text=prior_text_for_omission_audit,
        new_text=grad_result.text,
        today=today_str,  # v0.3.3 MEDIUM #4 fix — today-aware
        min_level=2,
        graduating_headings=grad_headings,
    )

    # 10.5c.5 TWO-PHASE COMMIT PIPELINE
    #
    # Phase 1: write continuity.md.tmp (no rename yet) so we know the
    #          content is durable on disk before committing DB state.
    # Phase 2: inside ``store._batch()``, accumulate DB DML (associations
    #          + meta + wrap_completed) without intermediate commits and
    #          write meta.json.tmp (no rename yet). The batch context
    #          manager commits the single outer SQLite transaction on
    #          successful exit and flushes queued audit events.
    # Phase 3: after the DB commit succeeds, atomically rename both
    #          tmp sidecars to their final paths.
    # Phase 4: fire the continuity_saved audit event.
    #
    # Crash windows:
    #   - Before the batch commit: DB rolls back + tmp files cleaned up.
    #     Store is in exact pre-wrap state.
    #   - Between batch commit and final renames: DB reflects the new
    #     wrap, both .tmp files PERSIST on disk (deliberately NOT
    #     cleaned up — they hold the new content and are required
    #     for operator recovery). continuity.md and meta.json are
    #     still the pre-wrap versions. Operator recovery path: the
    #     .md.tmp and .json.tmp files can be manually ``mv``'d to
    #     their final paths; the wrap-status subcommand still reports
    #     the pre-wrap state since wrap_completed cleared the
    #     in-progress metadata during Phase 2.
    #   - Between the two renames: continuity is the new version;
    #     meta.json.tmp still holds the new meta. Same operator
    #     recovery applies — ``mv x.json.tmp x.json`` finishes the
    #     externalization.
    #
    # The load-bearing invariant: once the batch has committed the
    # DB, the outer ``except`` MUST NOT unlink the tmp files. They
    # are the committed state awaiting externalization. Pre-commit
    # cleanup is unchanged (pre-wrap state is restored cleanly).
    sections = measure_sections(grad_result.text)
    patterns = len(re.findall(r"\|\s*\d+x", grad_result.text))
    total_demoted = grad_result.demoted + grad_result.bare_demoted
    # Hoisted out of the try block so ``path`` is unambiguously bound
    # on every code path (including the residual-window recovery
    # comment above where mypy would otherwise flag a possibly-unbound
    # reference). ``continuity_path`` is a property that reads
    # ``self._path`` — it's stable across the pipeline.
    path = str(store.continuity_path)

    # Phase 1: continuity tmp write.
    # Derive the tmp filename suffix from the wrap_token so
    # continuity.tmp and meta.tmp (written later in Phase 2) share a
    # recoverability identity — operators recovering from multiple
    # crashed wraps can pair the two tmp files by token prefix.
    # 10.5c.5 L3 Fix #19 (codex HIGH).
    tmp_token_prefix: str = snapshot["token"][:12]
    cont_tmp: Path | None = store._prepare_continuity_write(
        grad_result.text, token_hex=tmp_token_prefix
    )
    meta_tmp: Path | None = None
    wrap_result = None
    # Once ``db_committed`` flips True, the outer ``except`` preserves
    # tmp files instead of cleaning them up — they represent committed
    # state awaiting externalization. Cleaning them up would destroy
    # the new content permanently (L1 HIGH + L2 M2 data-loss path).
    db_committed = False

    try:
        # Phase 2: batched DB DML.
        with store._batch():
            assoc_formed, assoc_strengthened, assoc_decayed = \
                process_wrap_associations(store, grad_result, affective_state)

            if grad_result.validated > 0 or grad_result.citation_counts:
                meta["citations_seen"] = True
            meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1

            # Write meta tmp inside the batch window — not a DB op,
            # but scoped here so a failure rolls back the DB alongside
            # cleaning up both tmp files. Order matters only for the
            # cleanup path: if this raises, the outer ``try`` below
            # cleans up cont_tmp (and meta_tmp stays None) and the
            # batch's ``except`` rolls back the DB. The shared
            # ``tmp_token_prefix`` pairs this tmp with cont_tmp so
            # operator recovery can match them by prefix.
            meta_tmp = store._prepare_meta_write(
                meta, token_hex=tmp_token_prefix
            )

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
                section_sizes=sections,
                episode_ids=frozen_episode_ids,
                # Pass the token from the snapshot in hand rather than
                # having wrap_completed re-read metadata. Removes a
                # within-method SELECT-before-clear sequence that
                # Layer 1 L3 flagged as a TOCTOU-within-TOCTOU-fix
                # pattern.
                wrap_token=snapshot["token"],
            )

            # Update cross-session pattern history. Scan the
            # post-validation continuity text for every named pattern
            # line with an [evidence: ...] explanation that was AUTHORED
            # TODAY. That includes today's 1x mentions (preserves their
            # explanation for the next session's cross-session check)
            # and today's surviving 2x/3x graduations (the demoted
            # ones already lost their evidence tag via _demote_line so
            # they won't match here, which is correct).
            #
            # Today-only gate (v0.3.2 fix for Codex MEDIUM from the
            # 4-layer review): without this, carried-forward pattern
            # lines whose explanation prose differs from the prior
            # session's stored version would silently overwrite the
            # canonical prior explanation in pattern_history — letting
            # unvalidated carry-forward edits pollute the corpus the
            # cross-session check relies on. validate_graduations
            # intentionally skips non-today graduation lines for the
            # same family of reasons; the upsert path must match.
            #
            # Patterns demoted to 1x with `(cross-session-overlap)`
            # marker specifically don't have an evidence tag anymore
            # so they're skipped — the prior session's history
            # remains authoritative, exactly the desired behavior.
            #
            # wrap_id stays None for now: WrapResult is a dataclass
            # without the wraps.id field, and adding it would be a
            # caller-visible return-shape change unrelated to the
            # cross-session check itself. The audit log captures the
            # wrap timing and pattern_history's last_seen_at field
            # gives the per-pattern timestamp — the marginal forensic
            # value of an explicit wrap_id pointer is low.
            # v0.3.3 HIGH #1 fix: gate upsert loop to `## Patterns`
            # section only. Codex L3 caught that v0.3.2 fixed the
            # Anti-Patterns parsing leak at the graduation.py side
            # (validate_graduations / extract_pattern_names /
            # detect_stale_patterns all use _is_patterns_heading) but
            # did NOT propagate the fix to the upsert path here.
            # Result: `## Anti-Patterns` bullets matching the widened
            # _NAMED_PATTERN_RE still polluted the pattern_history DB
            # via the upsert loop. The section guard closes that gap.
            in_patterns_section = False
            for line in grad_result.text.split("\n"):
                if line.startswith("## "):
                    in_patterns_section = _is_graduating_heading(line, grad_headings)
                    continue
                if not in_patterns_section:
                    continue
                name_match = _NAMED_PATTERN_RE.match(line)
                if name_match is None:
                    continue
                # Use the any-level evidence regex (not the 2x/3x-only
                # GRADUATION_RE) so 1x mentions with explanations also
                # anchor cross-session history. Without this, the
                # first graduation step (1x → 2x) would have no prior
                # explanation to compare against and the cross-session
                # defense would always skip on the FIRST graduation —
                # exactly the step Phase 1b probe #1 exploits.
                evidence_match = _PATTERN_LINE_WITH_EVIDENCE_RE.search(line)
                if evidence_match is None:
                    # No [evidence: ...] tag (1x without explanation,
                    # or a demoted line) — nothing to anchor the
                    # cross-session check against.
                    continue
                # Today-only gate (Codex MEDIUM v0.3.2): only upsert
                # for lines authored this wrap. Carried-forward lines
                # with non-today dates are skipped to keep the
                # cross-session corpus authoritative.
                line_date = evidence_match.group(2)
                if line_date != today_str:
                    continue
                explanation = evidence_match.group(4)
                if not explanation:
                    continue
                try:
                    pattern_level = int(name_match.group(2))
                except ValueError:
                    continue
                store.upsert_pattern_history(
                    pattern_name=name_match.group(1),
                    level=pattern_level,
                    explanation=explanation,
                    wrap_id=None,
                )
            # Batch context manager commits here on successful exit.

        # Batch exited without raising → DB is committed. From this
        # point forward, failure must NOT unlink the tmp files.
        db_committed = True

        # Phase 3: DB commit succeeded — externalize files.
        # At this point cont_tmp is still the Path returned from
        # _prepare_continuity_write (we only clear it to None after the
        # successful rename below). The type is Path | None only because
        # of the consumed-handle pattern further down.
        assert cont_tmp is not None
        try:
            cont_tmp.replace(store.continuity_path)
        except OSError as exc:
            raise StoreError(
                f"Failed to rename continuity tmp to "
                f"{store.continuity_path}: {exc}. "
                f"The DB has committed the wrap but externalization "
                f"is incomplete; the new continuity content is "
                f"preserved at {cont_tmp}. Manually move it to "
                f"{store.continuity_path} to finish recovery.",
                operation="save_continuity",
                path=str(store.continuity_path),
            ) from exc
        # cont_tmp is now the final file, not a tmp file. Clear the
        # handle so the except clause below doesn't try to preserve
        # a path that no longer refers to a tmp sidecar.
        cont_tmp = None

        # Explicit None check instead of ``assert`` so the guard
        # survives ``python -O`` (which strips assertions). Under
        # ``-O`` the old assert vanished and ``meta_tmp.replace(...)``
        # would raise ``AttributeError`` on None — wrong exception
        # type for the transport layer. L3 complement F3 + contrarian F6.
        if meta_tmp is None:
            raise StoreError(
                "internal pipeline invariant violated: meta_tmp is "
                "None after the batch committed — this indicates a "
                "bug in validated_save_continuity's control flow. "
                "The DB has committed but the meta sidecar was "
                "never staged; the store is in a partial-commit "
                "state. Manual recovery required.",
                operation="save_meta",
                path=str(store.meta_path),
            )
        try:
            meta_tmp.replace(store.meta_path)
        except OSError as exc:
            raise StoreError(
                f"Failed to rename meta tmp to {store.meta_path}: "
                f"{exc}. The DB has committed the wrap and the "
                f"continuity file was externalized; only the meta "
                f"sidecar rename failed. The new meta content is "
                f"preserved at {meta_tmp}. Manually move it to "
                f"{store.meta_path} to finish recovery.",
                operation="save_meta",
                path=str(store.meta_path),
            ) from exc
        meta_tmp = None
        # Directory fsync after both renames so the rename syscalls
        # themselves are durable. Without this, a crash immediately
        # after a successful rename can revert to the pre-rename
        # directory entry on some POSIX filesystems. Best-effort.
        _fsync_dir(store.continuity_path.parent)

    except BaseException:
        # Cleanup policy depends on whether the DB has committed.
        #
        # ``BaseException`` scope chosen deliberately: we want
        # cleanup to run on ^C (KeyboardInterrupt) and sys.exit()
        # too, since a half-completed wrap with orphan tmp files is
        # worse than a clean pre-wrap state. SystemExit and
        # GeneratorExit are rare enough at this layer that the
        # consistent "always clean up pre-commit" policy is fine.
        #
        # Post-commit failures (rename OSError, Phase 4 audit error,
        # anything raised after the ``with store._batch()`` block
        # exits successfully): PRESERVE the tmp files. They hold
        # the new content and the operator needs them for recovery
        # via ``mv``. Cleaning them up here would destroy committed
        # state permanently (L1 HIGH + L2 M2).
        if not db_committed:
            if cont_tmp is not None:
                _safe_unlink(cont_tmp)
            if meta_tmp is not None:
                _safe_unlink(meta_tmp)
        raise

    # Phase 4: fire the continuity_saved audit event (after the file
    # has been externalized — matches the pre-10.5c.5 "audit after
    # rename" invariant). The earlier DB-side audit events
    # (associations_updated, associations_decayed, wrap_completed)
    # were already flushed by the batch context manager at its
    # successful exit.
    #
    # Audit exceptions are swallowed here — same pattern and
    # rationale as the batch's deferred-audit flush. At this point
    # the wrap is fully committed and externalized; an audit log
    # failure must not cause the pipeline to report failure to the
    # caller. L3 complement F4.
    if store._audit is not None:
        try:
            audit_payload: dict[str, Any] = {
                "chars": len(grad_result.text),
                "content_hash": hashlib.sha256(
                    grad_result.text.encode("utf-8")
                ).hexdigest(),
            }
            # Capture Proven-tier pattern omissions in the audit chain.
            # detect_pattern_omissions returns an empty list for the
            # common case (first wrap, or all prior 2x/3x patterns
            # carried forward at some level), in which case we omit
            # the key entirely to keep the routine-case audit entry
            # lean. When omissions DID happen, the audit chain records
            # exactly which graduated patterns disappeared at this
            # wrap — operators and downstream review (Diogenes,
            # consultation, audit-chain queries) can see what was
            # dropped without re-reading prior continuity files.
            if grad_result.omitted_patterns:
                audit_payload["omitted_patterns"] = [
                    {"name": op.name, "prior_level": op.prior_level}
                    for op in grad_result.omitted_patterns
                ]
            # Cross-session collisions ride into the audit chain on
            # the same "only when they fired" basis as omissions —
            # routine wraps stay lean, but any drift attempt that
            # tripped the cross-session check leaves a hash-chained
            # record naming the pattern, the level it tried to reach,
            # the meaningful words that overlapped with the prior
            # session, and the prior session's explanation text. Full
            # forensic trail for operator review.
            if grad_result.cross_session_collisions:
                audit_payload["cross_session_collisions"] = [
                    {
                        "name": coll.name,
                        "today_level": coll.today_level,
                        "overlap_words": list(coll.overlap_words),
                        "prior_explanation": coll.prior_explanation,
                    }
                    for coll in grad_result.cross_session_collisions
                ]
            # Move #4 library layer audit signal (v0.3.2): new Proven
            # graduations that landed without explicit contradiction-
            # stance declaration ride into the hash-chained audit log
            # so operator-review (Diogenes weekly sweep) can find them.
            # Lean omission when no new Provens skipped declaration.
            if proven_without_declaration:
                audit_payload["proven_without_contradicts_declaration"] = [
                    {"name": p.name, "level": p.level}
                    for p in proven_without_declaration
                ]
            store._audit.log("continuity_saved", audit_payload)
        except Exception:
            # Audit is best-effort. Silently drop a failing
            # continuity_saved event rather than reporting a false
            # failure after a fully committed wrap.
            pass

    # Phase 5: auto-prune if retention is configured. In the pre-10.5c.5
    # pipeline this ran inside wrap_completed via ``self.prune()``;
    # the batched pipeline explicitly suppresses prune inside the
    # batch (it's a separate DML burst with its own commit semantics
    # that do NOT belong inside the wrap transaction), so the pipeline
    # caller must invoke it after the batch exits. Without this call
    # the canonical pipeline silently stops honoring retention_days —
    # a data-lifecycle regression caught by Layer 1 review.
    if store._retention_days is not None:
        # Explicit None check rather than relying on flow-narrowing
        # for ``-O`` safety (L3 complement F5). By this point
        # wrap_result must be set — the batch completed successfully
        # and wrap_completed returned a value. If it IS None here,
        # something is deeply wrong and raising is correct.
        if wrap_result is None:
            raise StoreError(
                "internal pipeline invariant violated: wrap_result "
                "is None after a successful batch commit. This "
                "indicates a bug in validated_save_continuity's "
                "control flow.",
                operation="save_continuity",
                path=path,
            )
        pruned = store.prune()
        # Attach pruned count to the wrap_result so the return value
        # reflects the actual post-wrap store state. WrapResult is a
        # regular (non-frozen) dataclass so direct mutation is safe.
        wrap_result.pruned_count = pruned

    # Render omitted_patterns to plain dicts so the entire return
    # value stays JSON-serializable (mirrors the asdict() treatment of
    # wrap_result below). OmittedPattern is a small dataclass; asdict
    # is unnecessary overhead for two fields, so we render explicitly.
    omitted_patterns_payload: list[dict[str, Any]] = [
        {"name": op.name, "prior_level": op.prior_level}
        for op in grad_result.omitted_patterns
    ]
    cross_session_collisions_payload: list[dict[str, Any]] = [
        {
            "name": coll.name,
            "today_level": coll.today_level,
            "overlap_words": list(coll.overlap_words),
            "prior_explanation": coll.prior_explanation,
        }
        for coll in grad_result.cross_session_collisions
    ]
    proven_without_declaration_payload: list[dict[str, Any]] = [
        {"name": p.name, "level": p.level}
        for p in proven_without_declaration
    ]

    return SaveContinuityResult(
        path=path,
        chars=len(grad_result.text),
        episodes_compressed=len(episodes),
        graduations_validated=grad_result.validated,
        graduations_demoted=total_demoted,
        demoted=grad_result.demoted,
        bare_demoted=grad_result.bare_demoted,
        citation_reuse_max=grad_result.citation_reuse_max,
        skipped_non_today=grad_result.skipped_non_today,
        gaming_suspects=list(grad_result.gaming_suspects),
        omitted_patterns=omitted_patterns_payload,
        cross_session_collisions=cross_session_collisions_payload,
        proven_without_contradicts_declaration=proven_without_declaration_payload,
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
        sections=sections,
        # asdict() makes the full return value JSON-serializable
        # top-to-bottom. Library users who want the typed object can
        # do ``WrapResult(**result["wrap_result"])``; everyone else
        # can ``json.dumps(result)`` with no ceremony.
        wrap_result=asdict(wrap_result),
    )
