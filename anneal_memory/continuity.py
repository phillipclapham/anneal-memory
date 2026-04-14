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

import hashlib
import re
import uuid
import warnings
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graduation import detect_stale_patterns, validate_graduations
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


def _build_wrap_package(
    episodes: list[Episode],
    existing_continuity: str | None,
    project_name: str,
    *,
    max_chars: int = 20000,
    today: str | None = None,
    staleness_days: int = 7,
) -> WrapPackageDict:
    """Pure helper — build an agent-facing compression package from pre-fetched inputs.

    **Private.** Called by :func:`prepare_wrap` (the canonical
    store-aware pipeline) and by :func:`prepare_wrap_package` (the
    deprecated public wrapper kept for one release cycle to give
    existing callers a ``DeprecationWarning`` before removal).

    This function does not touch a store. It takes episodes and
    continuity text already in hand and assembles the agent-facing
    compression package (episodes listing + stale-pattern diagnostic
    + compression instructions + sizing constraints). The caller is
    responsible for wrap lifecycle (``store.wrap_started()``) and
    Hebbian association context — :func:`prepare_wrap` does that
    work around this helper.

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

    # Format episodes for the agent
    formatted_episodes = format_episodes_for_wrap(episodes)

    # Detect stale patterns in existing continuity
    stale_patterns: list[StalePatternDict] = []
    if existing_continuity:
        stale = detect_stale_patterns(existing_continuity, today, staleness_days)
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
    instructions = _build_wrap_instructions(project_name, max_chars, today)

    return WrapPackageDict(
        episodes=formatted_episodes,
        episode_count=len(episodes),
        continuity=existing_continuity,
        stale_patterns=stale_patterns,
        instructions=instructions,
        today=today,
        max_chars=max_chars,
    )


def prepare_wrap_package(
    episodes: list[Episode],
    existing_continuity: str | None,
    project_name: str,
    max_chars: int = 20000,
    today: str | None = None,
    staleness_days: int = 7,
) -> WrapPackageDict:
    """Deprecated thin wrapper over :func:`_build_wrap_package`.

    .. deprecated:: 0.2.0
        ``prepare_wrap_package`` is deprecated since 0.2.0 and will
        be removed in 0.3.0. Use :func:`prepare_wrap` (the canonical
        store-aware pipeline) instead. This wrapper exists only to
        give existing callers a :class:`DeprecationWarning` for one
        release cycle before the public name is removed.

        If you are an advanced library user who genuinely needs to
        construct a package from pre-fetched episodes without
        touching a store (for unit testing package construction in
        isolation, or for a custom wrap lifecycle), use the private
        :func:`_build_wrap_package` helper directly — understanding
        that as a private symbol it has no API stability guarantee.

    The wrapper emits a :class:`DeprecationWarning` with
    ``stacklevel=2`` so the warning surfaces at the caller's source
    line, then delegates to the canonical helper without mutating
    any arguments.

    Args:
        episodes: Episodes since last wrap (the compression window).
        existing_continuity: Current continuity text, or None for first session.
        project_name: Name for the continuity file header.
        max_chars: Maximum size of the continuity file.
        today: Override for today's date (YYYY-MM-DD). Defaults to actual today.
        staleness_days: Days before flagging stale patterns.

    Returns:
        WrapPackageDict — see :func:`_build_wrap_package`.
    """
    # NOTE: this warning intentionally names ``_build_wrap_package``
    # (a private symbol). The only remaining audience for this
    # deprecated public wrapper is advanced users managing their
    # own wrap lifecycle — normal users are already on
    # ``prepare_wrap``. Naming the private migration target here is
    # the same pattern Python stdlib uses (e.g. ``logging.warn``
    # directing advanced callers at ``logging.warning``,
    # ``asyncio.get_event_loop`` naming ``get_running_loop``). Do not
    # "clean up" the private symbol out of this message without
    # reading the 10.5c.3 fix-pass review (Layer 2 N3).
    warnings.warn(
        "prepare_wrap_package is deprecated since 0.2.0 and will be "
        "removed in 0.3.0. Use prepare_wrap(store, ...) for the "
        "canonical store-aware pipeline, or the private "
        "_build_wrap_package helper for advanced custom lifecycles. "
        "See the 'Canonical entry points' section of the library "
        "quickstart for migration guidance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_wrap_package(
        episodes,
        existing_continuity,
        project_name,
        max_chars=max_chars,
        today=today,
        staleness_days=staleness_days,
    )


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
    wrapper :func:`prepare_wrap_package` exists for one release cycle
    to give v0.1.x callers a warning before removal in v0.3.0; do not
    reach for it in new code.

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

    return PrepareWrapResult(
        status="ready",
        message=f"Ready to compress {len(episodes)} episode(s).",
        episode_count=len(episodes),
        package=package,
        assoc_context=assoc_context,
        wrap_token=wrap_token,
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
        **The episode set is frozen when ``prepare_wrap`` was called.**
        As of 10.5c.4 this function loads the wrap snapshot persisted
        by :func:`prepare_wrap` and filters its re-fetched episode set
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

        The legacy ``skipped_prepare`` path — calling this function
        without any prior ``prepare_wrap`` call — is unchanged. With
        no snapshot present, the function falls back to its
        pre-10.5c.4 behavior of re-fetching the full episode set
        since the last completed wrap. Note that this path keeps
        the TOCTOU window OPEN by design; callers that bypass
        ``prepare_wrap`` opt out of the fix, and the
        ``skipped_prepare=True`` flag on the return value surfaces
        that to the caller. This is a documented foot-gun, not an
        accidental escape hatch — the canonical path is
        ``prepare_wrap`` → ``validated_save_continuity``, and the
        skipped-prepare path exists only for advanced library users
        who are managing the wrap lifecycle themselves.

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
          - ``skipped_prepare`` (bool): True if ``prepare_wrap`` was not
            called first
          - ``wrap_result`` (dict[str, Any]): the store-level wrap
            record as a plain dict (``dataclasses.asdict`` of the
            underlying :class:`WrapResult`). Library users who want
            the typed dataclass can reconstruct it via
            ``WrapResult(**result["wrap_result"])``.

    Raises:
        ValueError: If text is empty or missing required sections.
        StoreError: If the filesystem write of the continuity sidecar
            or meta sidecar fails. ``StoreError`` is a library-level
            domain error (subclass of :class:`AnnealMemoryError`, NOT
            of :class:`OSError`). Transports should catch
            :class:`AnnealMemoryError` as a single library boundary,
            or :class:`StoreError` specifically to read ``.operation``
            and ``.path`` for clean error messages. The original
            ``OSError`` is preserved on ``__cause__`` (we raise
            ``StoreError(...) from exc``), so callers that need
            ``errno`` can dig one level deeper.
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

    # Load the frozen snapshot persisted by prepare_wrap. None iff
    # no wrap is currently in progress — either the caller is on
    # the legacy ``skipped_prepare`` path (calling this function
    # without having run prepare_wrap first) or the store is idle.
    # Derive ``skipped_prepare`` from snapshot presence so the
    # return-value semantics collapse onto a single source of truth:
    # the persisted snapshot. Eliminates the 10.5c.4 Layer 1 finding
    # where a caller could reach "wrap_in_progress=True but snapshot
    # absent" via a direct no-arg ``store.wrap_started()`` call —
    # load_wrap_snapshot now raises StoreError on that state before
    # we'd even get here, so the only way ``snapshot is None`` is
    # the legitimate legacy path.
    snapshot = store.load_wrap_snapshot()
    skipped_prepare = snapshot is None

    if snapshot is not None and wrap_token is not None:
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

    # Get current session's episodes for citation validation.
    #
    # With a snapshot: re-fetch the full post-last-wrap set and
    # filter down to exactly the IDs the snapshot froze at prepare
    # time. Any episodes recorded in the TOCTOU window drop out
    # here and stay with ``session_id IS NULL`` through the rest
    # of the pipeline — they land in the next wrap's compression
    # window on the next ``prepare_wrap`` call.
    #
    # Without a snapshot (legitimate ``skipped_prepare`` path,
    # library caller bypassing prepare_wrap entirely): fall back
    # to the pre-10.5c.4 behavior of using the full re-fetched
    # set. Preserves backward compatibility for that path. Note
    # that this path keeps the TOCTOU window OPEN by design —
    # callers who bypass prepare_wrap opt out of the fix, and the
    # ``skipped_prepare=True`` flag in the return surfaces that
    # to the caller.
    episodes_all = store.episodes_since_wrap()
    frozen_episode_ids: list[str] | None
    if snapshot is not None:
        snapshot_id_set = set(snapshot["episode_ids"])
        episodes = [ep for ep in episodes_all if ep.id in snapshot_id_set]
        # Use the snapshot's ID list directly — the TypedDict
        # already declares it as list[str] and wrap_completed does
        # not mutate its argument. The outer ``if snapshot is not
        # None`` guard means the field access is safe.
        frozen_episode_ids = snapshot["episode_ids"]
    else:
        episodes = episodes_all
        frozen_episode_ids = None
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
    # 10.5c.5 L3 Fix #19 (codex HIGH). When the snapshot is absent
    # (legacy skipped_prepare path), fall back to a per-call random
    # uuid — those saves don't leave orphans anyway because they
    # don't use the two-phase pipeline.
    tmp_token_prefix: str | None = (
        snapshot["token"][:12] if snapshot is not None else None
    )
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
                wrap_token=snapshot["token"] if snapshot is not None else None,
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
            store._audit.log("continuity_saved", {
                "chars": len(grad_result.text),
                "content_hash": hashlib.sha256(
                    grad_result.text.encode("utf-8")
                ).hexdigest(),
            })
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
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
        sections=sections,
        skipped_prepare=skipped_prepare,
        # asdict() makes the full return value JSON-serializable
        # top-to-bottom. Library users who want the typed object can
        # do ``WrapResult(**result["wrap_result"])``; everyone else
        # can ``json.dumps(result)`` with no ceremony.
        wrap_result=asdict(wrap_result),
    )
