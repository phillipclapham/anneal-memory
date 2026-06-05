"""Graduation and demotion logic for anneal-memory.

Handles temporal graduation validation:
- Citation checking (do graduated patterns cite real episodes?)
- Explanation overlap (does the evidence actually support the pattern?)
- Citation decay detection (which patterns haven't been reinforced?)
- Citation gaming detection (suspicious reuse of single episodes)
- Demotion (demote ungrounded or stale graduations)

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import re
from datetime import datetime as _datetime, date as _date
from dataclasses import dataclass, field
from typing import Any, Callable

from .schema import DEFAULT_GRADUATING


# Matches graduated patterns (2x or 3x) WITH [evidence: <id> "explanation"] citations.
# Captures: (1) level, (2) date, (3) cited IDs, (4) optional explanation in quotes.
_GRADUATION_RE = re.compile(
    r"\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*\[evidence:\s*"
    r"([a-fA-F0-9][a-fA-F0-9, ]*)"  # one or more hex IDs
    r'(?:\s+"([^"]*)")?\s*\]'  # optional quoted explanation
)

# Matches bare graduations (2x or 3x) WITHOUT [evidence:] tags.
_BARE_GRADUATION_RE = re.compile(
    r"\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*(?!\[evidence:)"
)

# Matches any pattern with temporal marker (Nx)
_PATTERN_RE = re.compile(
    r"\|\s*(\d+)x\s*\((\d{4}-\d{2}-\d{2})\)"
)

# Matches a pattern line at ANY level (1x/2x/3x) with an
# ``[evidence: HEXID "explanation"]`` tag. Distinct from
# ``_GRADUATION_RE`` (which is scoped to 2x/3x for validation
# purposes) and from ``_PATTERN_RE`` (which ignores evidence).
# Used by the cross-session history-upsert path so that 1x mentions
# with explanations also anchor the per-pattern history — without
# this, the first graduation (1x → 2x) wouldn't see any prior
# explanation to compare against, defeating the cross-session
# defense on initially-developing patterns.
_PATTERN_LINE_WITH_EVIDENCE_RE = re.compile(
    r"\|\s*(\d+)x\s*\((\d{4}-\d{2}-\d{2})\)\s*\[evidence:\s*"
    r"([a-fA-F0-9][a-fA-F0-9, ]*)"
    r'(?:\s+"([^"]*)")?\s*\]'
)

# Matches a full pattern line in ## Patterns, capturing the pattern name
# (operator-style identifier following any optional FlowScript marker prefix)
# and the graduation level. Used by extract_pattern_names() /
# detect_pattern_omissions() / extract_contradiction_declarations() /
# detect_proven_without_declaration() / the cross-session history upsert.
#
# Grammar:
#     [explicit pattern signal][optional FlowScript markers] operator_style_name | Nx
#
# The "explicit pattern signal" is ONE of: a ``- `` dash bullet (canonical), a
# FlowScript marker (``!`` ``!!`` ``?`` ``✓`` ``*``), OR leading INDENTATION
# (an indented member under a group header). Operator-style names are
# intentionally narrow — alphanumeric + underscore + period + hyphen, must
# start with a letter.
#
# AM-HISTUPSERT-BULLET (v0.4.5): the leading ``- `` bullet is OPTIONAL. The
# per-name immune system (history upsert, cross-session sycophancy gate,
# omission/contradiction/level extraction) previously required a dash bullet,
# so an entity that grouped its patterns under header lines with *indented,
# bullet-less* operator-named members (flow's ``{{topic: ...}}`` convention)
# matched _GRADUATION_RE (the demoter, bullet-agnostic) but NOT this regex —
# its patterns were demoted but never protected, and pattern_history stayed
# empty (0 rows / 20+ wraps; an ``invisible_infrastructure_failure``). Making
# the dash optional re-symmetrises the two: the same lines the demoter sees
# now also anchor the per-name defenses.
#
# WHY a bullet-less line still needs an explicit signal (dash/marker/indent):
# without one, a single capitalized prose word at column 0 followed by
# ``| Nx`` (``Throughput | 3x when batched``, ``Total | 3x``) would be read as
# a phantom pattern — the operator-name token alone is NOT a sufficient
# false-positive guard for column-0 prose. Requiring indentation (or a
# dash/marker) when the dash is absent cleanly separates flow's real grouped
# members (always indented under a ``{{header}}``) from top-level prose. The
# residual ambiguity — an INDENTED single-word line that is prose, not a
# pattern — is irreducible (indentation is exactly what makes a grouped member
# a member); inside ``## Patterns`` an indented ``name | Nx`` is taken to BE a
# pattern by convention. (Verified against a live grouped continuity: 15 named
# members matched, 0 spurious matches on group headers or prose.)
#
# FlowScript marker prefixes may appear between the (optional) bullet and the
# name; this is the convention flow's continuity uses heavily. Examples match:
#     - pattern_name | 2x (2026-05-21) [evidence: abc12345 "..."]
#     - !! pattern_name | 3x (2026-05-21) [evidence: abc12345 "..."]
#       pattern_name | 1x (2026-05-21)            # INDENTED, bullet-less (grouped)
#       ✓ pattern_name | 2x (2026-05-21) [evidence: abc12345 "..."]   # indent + marker
#     ! pattern_name | 1x (2026-05-21)            # column-0 marker (explicit signal)
#
# Lines that do NOT match (intentional):
#     thought: ACID compliance outweighs raw speed | 2x ...
#         ↑ ``thought`` is followed by ``:`` not ``| Nx`` — no operator name
#     {topic: ...}                                 # a group HEADER line
#         ↑ starts with ``{``, not a letter — header lines never carry the
#           ``name | Nx`` shape, so they are skipped; only members count
#     some prose sentence with | 2x buried in it
#         ↑ the multi-word phrase has a space before ``|`` — no single-token name
#     Throughput | 3x when batched                 # column-0 single-word prose
#         ↑ no dash, no marker, no indentation — rejected as prose (the
#           indent/dash/marker signal is what distinguishes it from a member)
# NOTE: whitespace is matched as HORIZONTAL only (``[ \t]``, not ``\s``). All
# callers split on "\n" before matching, so a line never contains a newline —
# but using ``\s`` for the indentation branch (b) would let ``\n`` count as
# "indentation" if this helper were ever reused on raw multi-line text
# (``_NAMED_PATTERN_RE.match("\nFoo | 2x")`` would match), crossing logical
# lines. ``[ \t]`` keeps "indentation" meaning indentation. (codex L3, v0.4.5.)
_NAMED_PATTERN_RE = re.compile(
    r"^(?:"
    r"[ \t]*-[ \t]+"                            # (a) "- " dash bullet (canonical), any indent
    r"|[ \t]+"                                  # (b) leading indentation (grouped member)
    r"|(?=(?:!!|!|\?|✓|\*)[ \t])"               # (c) FlowScript marker at column 0 (zero-width)
    r")"
    r"(?:(?:!!|!|\?|✓|\*)[ \t]+)?"              # optional FlowScript marker prefix
    r"([A-Za-z][A-Za-z0-9_.\-]*)"               # operator-style identifier (ASCII)
    r"[ \t]*\|[ \t]*(\d+)x"                     # graduation marker
)

# AM-PERNAME-LINEBIND (v0.4.6): the name AND its evidence tag captured in ONE
# anchored match, so the level/date/ids/explanation are guaranteed to belong to
# the SAME marker as the name. The pre-0.4.6 upsert path matched the name with
# ``_NAMED_PATTERN_RE.match`` (anchored, grabs the FIRST marker) and the
# evidence with ``_PATTERN_LINE_WITH_EVIDENCE_RE.search`` (UNANCHORED, grabs the
# first WELL-FORMED ``| Nx (date) [evidence:]`` anywhere on the line). On a
# malformed line carrying TWO ``name | Nx [evidence:]`` markers, if the first
# marker had been demoted (``_demote_line`` strips its evidence tag), the
# unanchored search skipped past the now-tagless first marker and bound the
# SECOND marker's level/date/explanation to the FIRST marker's name — polluting
# pattern_history with a mismatched (name, level, explanation) tuple. Capturing
# both in one anchored regex makes the binding structural (codex L3, 0.4.5 — a
# pre-existing per-name edge proven on clean main, deferred into this pass).
# Groups: (1) name, (2) level, (3) date, (4) ids, (5) explanation (optional).
# The name prefix mirrors _NAMED_PATTERN_RE exactly; the tail mirrors
# _PATTERN_LINE_WITH_EVIDENCE_RE exactly. A line with no ``[evidence:]`` simply
# does not match (the upsert path already skips such lines), preserving behavior
# on every well-formed single-marker line.
_NAMED_PATTERN_WITH_EVIDENCE_RE = re.compile(
    r"^(?:"
    r"[ \t]*-[ \t]+"
    r"|[ \t]+"
    r"|(?=(?:!!|!|\?|✓|\*)[ \t])"
    r")"
    r"(?:(?:!!|!|\?|✓|\*)[ \t]+)?"
    r"([A-Za-z][A-Za-z0-9_.\-]*)"               # (1) operator-style identifier
    r"[ \t]*\|[ \t]*(\d+)x"                     # (2) level
    r"[ \t]*\((\d{4}-\d{2}-\d{2})\)"            # (3) date
    r"[ \t]*\[evidence:[ \t]*"
    r"([a-fA-F0-9][a-fA-F0-9, ]*)"              # (4) cited ids
    r'(?:[ \t]+"([^"]*)")?[ \t]*\]'             # (5) explanation (optional)
)

# Matches `[contradicts: name_a, name_b, ...]` annotation on a pattern
# line. Move #4 library layer (v0.3.2): agents declare which existing
# Proven patterns a new pattern explicitly contradicts. Comma-separated
# operator-style names inside the brackets. Library treats this as audit
# data — does NOT verify the named patterns exist or perform semantic
# judgment; methodology layer (Levain) + operator-review (Diogenes)
# carry the enforcement and detection responsibilities.
_CONTRADICTS_RE = re.compile(
    r"\[contradicts:\s*([A-Za-z][\w\.\-]*(?:\s*,\s*[A-Za-z][\w\.\-]*)*)\s*\]"
)

# Matches `[no-contradicts]` declaration on a pattern line — the agent
# explicitly considered contradiction with existing Proven and declares
# none. Together with `_CONTRADICTS_RE`, satisfies the methodology-layer
# scan requirement; absence of EITHER on a new Proven graduation
# surfaces as audit signal `proven_without_contradicts_declaration`.
_NO_CONTRADICTS_RE = re.compile(r"\[no-contradicts\]")


@dataclass
class ProvenWithoutDeclaration:
    """A new Proven graduation that landed without explicit contradiction
    stance (neither ``[contradicts: ...]`` nor ``[no-contradicts]``).

    Surfaced by :func:`detect_proven_without_declaration` as an audit
    signal. The library does NOT refuse the save — methodology-layer
    discipline (Levain WRAP_PROTOCOL.md mandatory contradiction-scan)
    enforces the requirement at agent-prompt time, and operator-review
    (Diogenes weekly sweep) is the LLM-as-judge layer that actually
    detects semantic opposition between patterns. The library is the
    audit substrate: records whether the discipline was followed,
    leaves enforcement and detection to layers above and beside.

    Added in v0.3.2 as the library-layer component of the Move #4
    contradiction-detection architecture (reframed from the
    locked-design Move #4 after the 4-layer review surfaced that
    token-similarity triggers cannot close Phase 1b probe #1's
    divergent-vocabulary variant).
    """

    name: str  # Pattern identifier (operator-style name)
    level: int  # Level at which the pattern graduated (2 or 3)


def _is_graduating_heading(
    line: str, graduating_headings: frozenset[str] = DEFAULT_GRADUATING
) -> bool:
    """Return True iff ``line`` is a ``## ``-heading whose role is ``graduating``.

    ``graduating_headings`` is a set of normalized ``## heading`` markers
    (lowercased — e.g. ``{"## patterns"}``), produced by
    :func:`anneal_memory.schema.graduating_headings` from a store's section
    schema and threaded in by the continuity pipeline (v0.3.4). It defaults to
    :data:`~anneal_memory.schema.DEFAULT_GRADUATING` (``{"## patterns"}``) so a
    caller that passes no schema gets exactly the historical behavior — the
    immune system / graduation scan runs only in ``## Patterns``.

    The exact-match-against-a-set design preserves the v0.3.2 Anti-Patterns fix:
    ``## Anti-Patterns`` / ``## Other patterns`` / ``## Design Patterns`` do not
    match ``## patterns`` and so are never parsed as the graduated-patterns
    section.
    """
    return line.strip().lower() in graduating_headings


def _is_patterns_heading(line: str) -> bool:
    """Backward-compatible alias: the default (``## Patterns``) graduating gate.

    Equivalent to ``_is_graduating_heading(line)`` with the default
    :data:`~anneal_memory.schema.DEFAULT_GRADUATING`. Retained because the
    historical name is referenced across the codebase and reads clearly at the
    DEFAULT_SCHEMA sites; schema-aware call sites use
    :func:`_is_graduating_heading` with the threaded set. Retained for external /
    legacy callers; after the v0.3.4 schema refactor there are no internal call
    sites — the canonical pipeline calls :func:`_is_graduating_heading` with the
    schema-derived graduating set.
    """
    return _is_graduating_heading(line)


# Stop words for explanation overlap checking
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could this that these those "
    "it its he she they we you i me my our his her their in on at to "
    "for of by with from and or but not no nor so if as".split()
)


@dataclass
class GraduationResult:
    """Result of graduation validation."""

    text: str  # Possibly modified text (demoted patterns)
    validated: int  # Citations that checked out
    demoted: int  # Citations that failed -> demoted
    citation_reuse_max: int  # Max times any single node was cited
    bare_demoted: int  # Bare graduations demoted (no evidence tag)
    # Graduation-format lines whose date != today (carried-forward
    # patterns from prior sessions, OR a test-authoring bug where a
    # hardcoded date drifted from wall-clock). Callers that intend
    # to exercise validation (tests, deterministic experiments) can
    # assert this is zero as a structural invariant — catches the
    # Finding #3 class of test drift without any log/warning noise
    # in production where non-today skips are the normal case.
    skipped_non_today: int = 0
    citation_counts: dict[str, int] = field(default_factory=dict)  # ep_id -> times cited
    gaming_suspects: list[str] = field(default_factory=list)  # IDs cited >= threshold
    direct_co_citations: list[tuple[str, str]] = field(default_factory=list)  # Pairs from same line
    all_validated_ids: list[set[str]] = field(default_factory=list)  # Per-line co-cited real-episode ID sets — NOT graduation-validated since AM-QUOTEFOOTGUN (v0.4.1): populated on demoted-grounding lines too, feeds session co-citation. (Rename to co_cited_id_sets is a v0.5 candidate.)
    # AM-WARN (v0.4.2) support: did ANY citation-bearing graduation this wrap
    # cite at least one id that resolves to a real store episode? Tracked
    # INDEPENDENT of the cross-session immune gate — unlike all_validated_ids,
    # which is suppressed on a cross-session-overlap demote (graduation.py
    # gate at the co-citation block). AM-WARN Signal A ("citations resolved to
    # ZERO episodes — wrong id namespace") must read THIS, not all_validated_ids:
    # a cross-session-demoted line has resolving ids (the immune gate fired on
    # the explanation, not the ids), so reading all_validated_ids would
    # misdiagnose a healthy immune-gate demotion as a dead-namespace graph.
    any_citation_resolved: bool = False
    # Patterns at 2x/3x in the prior continuity that are absent at any
    # level in the new continuity. Surfaced as audit signal — see
    # detect_pattern_omissions() and OmittedPattern docstring.
    omitted_patterns: list["OmittedPattern"] = field(default_factory=list)
    # Graduations refused because today's explanation overlaps too
    # heavily with the pattern's prior-session explanation. Closes
    # the slow-drift sycophantic-accumulation gap surfaced by Phase
    # 1b probe #1. The graduation in the text was demoted (same
    # mechanic as the ungrounded path) when this list is non-empty.
    cross_session_collisions: list["CrossSessionCollision"] = field(default_factory=list)
    # AM-CARRYFORWARD (v0.4.6): graduated lines HELD at their level instead of
    # demoted, because they are at/below their earned high-water mark and were
    # grounded recently (warm). The ungrounded-citation demotion path only —
    # the (cross-session-overlap) immune path is never carried forward. Each
    # entry is a CarriedForward. Empty when nothing was held (no history, cold,
    # or never earned the level). See CarriedForward for the full mechanism.
    carried_forward: list["CarriedForward"] = field(default_factory=list)


@dataclass
class StalenessInfo:
    """Information about a potentially stale pattern."""

    line_number: int
    content: str  # The pattern line
    level: int  # Current graduation level
    last_date: str  # Date of last validation
    days_stale: int  # Days since last validation


@dataclass
class CrossSessionCollision:
    """A graduation event blocked by cross-session explanation overlap.

    Surfaced when today's graduation explanation for a pattern shares
    too many meaningful words with the pattern's most recent prior
    appearance (tracked in the pattern_history table). The shape of
    the failure: legitimate multi-faceted evidence uses different
    vocabulary per session because each citation is a genuinely
    different angle on the pattern; sycophantic accumulation rephrases
    the same claim and reuses trope vocabulary.

    Detected at validate_graduations time. The graduation gets demoted
    same way an ungrounded citation does (3x→2x or 2x→1x, marked
    ``(cross-session-overlap)``). Pattern-history row is NOT updated
    when this fires — the prior session's explanation remains
    authoritative until the agent composes a new explanation with
    actually-different vocabulary.

    Added in response to Bold Stand Phase 1b probe #1 (2026-05-21),
    which demonstrated that the citation-validation pipeline could not
    distinguish a sycophantic pattern accumulating fresh vocabulary
    from a legitimate one accumulating distinct evidence. Closes the
    slow-drift attack class structurally.
    """

    name: str  # Pattern identifier
    today_level: int  # Level the agent attempted to graduate to (2 or 3)
    overlap_words: list[str]  # Meaningful words shared with prior explanation
    prior_explanation: str  # The prior session's explanation text


@dataclass
class CarriedForward:
    """A graduated pattern HELD at its level instead of demoted, because it
    is at/below its earned high-water mark AND was grounded recently (warm).

    AM-CARRYFORWARD (v0.4.6). The pre-0.4.6 demoter ratcheted a 2x/3x line
    down by one level whenever THIS wrap's citation failed to resolve — but a
    failed citation means "this session's domain did not cleanly re-ground the
    pattern," NOT "the pattern is fading." A load-bearing Proven decayed by
    session-domain rather than by importance (``decays_by_session_domain_not_importance``).

    Carryforward consults the pattern's ``pattern_history``: if the line is at
    or below its ``max_level_reached`` (it genuinely earned this level before)
    AND its last successful grounding (``last_seen_at``) is within
    ``carryforward_cold_days``, the line HOLDS — its level is kept and its
    ``[evidence:]`` tag is replaced with ``(carried-forward)``. Because a held
    line loses its evidence tag, it does NOT upsert pattern_history this wrap,
    so ``last_seen_at`` does not advance: a pattern that keeps failing to ground
    decays toward cold on its own and eventually ages out (the recency signal
    IS the failing-streak signal — no separate tracking). Scope: the
    ungrounded-citation path ONLY. The ``(cross-session-overlap)`` immune
    demotion is never carried forward — that path is the anti-sycophancy
    defense, and protecting it would blunt it.

    A carried-forward pattern at the TOP graduation tier (max_level_reached
    >= 3) that keeps needing the hold is a candidate to GRADUATE OUT to a
    stable home (e.g. partnership.md) or RETIRE — surfaced as an assisted
    notice on the save result, never silently lost.
    """

    name: str  # Pattern identifier (operator-style name)
    held_level: int  # The level the line was held at (NOT demoted from)
    max_level_reached: int  # The pattern's recorded high-water mark
    days_since_grounded: int  # Whole days since last_seen_at (recency)


@dataclass
class OmittedPattern:
    """A pattern that was at Proven-tier (2x/3x) in the prior continuity
    but is absent at any level in the new continuity.

    Silent omission of a graduated pattern is a threat the rest of the
    citation-layer pipeline does not catch: validate_graduations only
    sees patterns the agent *wrote into* the new continuity, so a
    pattern the agent simply *did not carry forward* leaves no trace at
    the graduation layer. detect_pattern_omissions surfaces these as an
    informational audit signal — the agent may have intentionally
    retired the pattern, or may have silently erased load-bearing
    evidence. Either way, operators and downstream review (Diogenes,
    consultation, audit-chain queries) can see what was dropped.

    Discovered as a reachable gap during Bold Stand Phase 1b probe #1
    (2026-05-21) — an adversarial agent dropped two legitimate Proven
    patterns across three drift sessions with no library detection.

    Known false-positive shape (documented as audit-signal-not-gate
    behavior, v0.3.2 4-layer review WARNING #3): renaming a Proven
    pattern across wraps surfaces an OmittedPattern entry for the old
    name. The library cannot distinguish "renamed pattern" from
    "silently dropped pattern" without semantic comparison, which the
    no-LLM-as-judge axiom prohibits at the library layer. Operators
    who rename patterns can safely ignore omission entries whose name
    is structurally similar to a freshly-introduced pattern in the
    same wrap; the audit chain has both records for post-hoc review.
    """

    name: str  # Pattern identifier (operator-style name before the | marker)
    prior_level: int  # Level at which it appeared in the previous wrap (2 or 3)


def validate_graduations(
    text: str,
    valid_ids: set[str],
    today: str,
    node_content_map: dict[str, str] | None = None,
    citations_seen: bool = False,
    pattern_history_lookup: "Callable[[str], dict[str, Any] | None] | None" = None,
    cross_session_overlap_threshold: int = 3,
    graduating_headings: frozenset[str] = DEFAULT_GRADUATING,
    carryforward_cold_days: int | None = 7,
) -> GraduationResult:
    """Validate evidence citations on graduated patterns.

    Scans the ## Patterns section for 2x/3x lines with [evidence: <id> "explanation"].
    Only validates citations whose date matches today (newly graduated this session).
    Carried-forward patterns from previous sessions pass through unchanged.

    Validation checks (all must pass):
    1. At least one cited ID exists in the current session's episode set
    2. If an explanation is provided and node_content_map is available,
       the explanation must reference actual content from the cited episode
    3. If pattern_history_lookup is provided AND the pattern has been seen
       in a prior session, today's explanation must NOT share
       ``cross_session_overlap_threshold`` or more meaningful words with
       the prior session's explanation. This catches sycophantic
       accumulation where each session rephrases the same claim with
       overlapping vocabulary — closes the slow-drift attack from
       Bold Stand Phase 1b probe #1 (2026-05-21). Evaluated whenever
       check 1 passes (a cited ID resolves), INDEPENDENT of check 2's
       explanation-grounding result — see AM-XSESSION-LINKGATE below.

    If check 1 fails (no cited ID resolves) the line demotes and marks
    ``(ungrounded)``. If check 2 fails (explanation doesn't reference any
    cited episode's content) the line demotes and marks ``(ungrounded)``.
    If check 3 fails the line demotes and marks ``(cross-session-overlap)``.
    Check 3 takes precedence on the marker: a line that is BOTH ungrounded
    (check 2) and cross-session-overlapping (check 3) is marked
    ``(cross-session-overlap)`` — the more specific immune signal — and its
    Hebbian co-citation link is suppressed (AM-XSESSION-LINKGATE, v0.4.3).
    Before AM-QUOTEFOOTGUN (v0.4.1) check 3 ran only on the validated
    path, so gating it on check 2 was harmless; once 0.4.1 decoupled link
    formation from grounding, the gate had to follow the link onto the
    demoted path or a sycophantic re-graduation with a fresh-but-ungrounded
    explanation would form an unsuppressed link.

    Args:
        text: The continuity file text (full markdown).
        valid_ids: Set of 8-char hex IDs from current session's episodes.
        today: Today's date as YYYY-MM-DD string.
        node_content_map: Optional mapping of episode ID -> content for overlap checking.
        citations_seen: If True, bare graduations (no evidence tag) are also demoted.
        pattern_history_lookup: Optional callable that returns the
            cross-session history dict for a given pattern name (keys:
            ``max_level_reached``, ``last_explanation``, ``last_seen_at``,
            ``last_wrap_id``) or None if the pattern has no history.
            When None, the cross-session overlap check is skipped (preserves
            the pre-Move-#3 behavior for library callers that don't wire
            history through). The canonical save pipeline supplies
            ``store.get_pattern_history`` here.
        cross_session_overlap_threshold: Meaningful-word overlap count
            at which the cross-session check refuses today's graduation.
            Default 3 — calibrated against Phase 1b probe #1 such that
            sycophantic vocabulary reuse (5+ shared words) trips while
            normal cross-session graduations with distinct evidence
            (≤2 shared trope words) pass cleanly.

    Returns:
        GraduationResult with possibly modified text and validation counts.
    """
    lines = text.split("\n")
    in_patterns = False
    validated = 0
    demoted = 0
    bare_demoted = 0
    skipped_non_today = 0
    citation_counts: dict[str, int] = {}
    direct_co_citations: list[tuple[str, str]] = []
    all_validated_ids: list[set[str]] = []
    # AM-WARN (v0.4.2): tracked independent of the cross-session immune gate
    # (see the field docstring on GraduationResult).
    any_citation_resolved = False
    cross_session_collisions: list[CrossSessionCollision] = []
    carried_forward: list[CarriedForward] = []

    for i, line in enumerate(lines):
        # Track section boundaries
        if line.startswith("## "):
            in_patterns = _is_graduating_heading(line, graduating_headings)
            continue
        if not in_patterns:
            continue

        # Check for citations with evidence tags
        match = _GRADUATION_RE.search(line)
        if match:
            level = int(match.group(1))
            date_str = match.group(2)
            cited_raw = match.group(3)
            explanation = match.group(4)

            # Only validate today's citations. Non-today graduations
            # are either legitimately carried-forward from a prior
            # session (normal) or a test-authoring bug where a
            # hardcoded date drifted from wall-clock (Diogenes
            # Finding #3 class — recurred 3x before the counter
            # was added). Increment ``skipped_non_today`` so callers
            # that care can assert on it; production ignores it.
            if date_str != today:
                skipped_non_today += 1
                continue

            # Normalize cited IDs
            cited_ids = {
                cid.strip().lower()[:8]
                for cid in re.split(r"[,\s]+", cited_raw)
                if cid.strip()
            }

            # Track citation frequency
            for cid in cited_ids & valid_ids:
                citation_counts[cid] = citation_counts.get(cid, 0) + 1

            # Check 1: at least one cited ID exists
            ids_valid = bool(cited_ids & valid_ids)

            # Check 2: explanation references ANY cited episode's content.
            # ``grounding_checked`` records whether grounding was ACTUALLY
            # evaluated (node_content_map present) — distinct from
            # ``explanation_valid``'s optimistic default. The AM-PRESERVE-VS-
            # SYCOPHANCY exemption (Site 2) requires grounding_checked, so a direct
            # caller that passes valid_ids + history but OMITS node_content_map
            # cannot earn the exemption by vacuous default and re-open the
            # exact-copy graph-poison (codex L3 convergence MEDIUM).
            grounding_checked = bool(ids_valid and explanation and node_content_map)
            explanation_valid = True
            # ``and node_content_map is not None`` is implied by grounding_checked
            # (which is bool(... and node_content_map)); the explicit clause narrows
            # the type for mypy WITHOUT an assert (-O-safe, matching the codebase's
            # invariant-check style).
            if grounding_checked and node_content_map is not None:
                # Check all valid cited IDs — pass if ANY has content overlap
                explanation_valid = False
                for cid in cited_ids & valid_ids:
                    node_content = node_content_map.get(cid, "")
                    if node_content and check_explanation_overlap(explanation, node_content):
                        explanation_valid = True
                        break

            # Cross-session sycophantic-accumulation check (Phase 1b probe #1).
            # AM-XSESSION-LINKGATE (v0.4.3): computed whenever the citation
            # resolves (``ids_valid``) and the pattern has prior history —
            # INDEPENDENT of ``explanation_valid`` (the within-session
            # grounding result). Pre-AM-QUOTEFOOTGUN (v0.4.1) this gate only
            # had to cover the validated path, so requiring explanation_valid
            # here was harmless. But 0.4.1 decoupled Hebbian link formation
            # from grounding — the co-citation block below fires on the
            # demoted-grounding path too — so the cross-session gate must run
            # there as well, or a sycophantic re-graduation carrying a
            # fresh-but-ungrounded explanation would form an unsuppressed
            # link. Gating on ``ids_valid`` (not ``explanation_valid``) aligns
            # this check with the link-formation precondition at the
            # co-citation block (``if ids_valid and not
            # cross_session_overlap_words``) for operator-NAMED patterns. If
            # pattern_history_lookup is None (library caller not wiring history
            # through), the check is skipped and pre-Move-#3 behavior holds.
            #
            # Two paths can still form a co-citation link this gate cannot
            # reach — both PRE-EXISTING blind spots of the whole per-name
            # immune layer (Move #2/#3/#4), NOT introduced by this change:
            #   (a) ``and explanation`` — a bare ``[evidence: id]`` citation
            #       has no explanation text to compare across sessions.
            #   (b) unnamed/freeform lines (``thought:`` / ``{topic:}``-grouped)
            #       match _GRADUATION_RE but not _NAMED_PATTERN_RE, so
            #       cross_session_overlap_words can never be set for them (the
            #       name_match block below is its only writer) — and per-name
            #       history was never anchored to them, so the lookup returns
            #       None regardless.
            # AM-XSESSION-LINKGATE closes the hole for the NAMED patterns the
            # gate was ever able to cover; it does not widen coverage to (a)/(b).
            cross_session_overlap_words: list[str] = []
            prior_explanation_for_check = ""
            if ids_valid and pattern_history_lookup is not None and explanation:
                # AM-PERNAME-LINEBIND (codex L3, v0.4.6): bind the name to its
                # OWN evidence marker (combined regex, anchored), not a
                # line-anchored _NAMED_PATTERN_RE.match that always returns the
                # FIRST name. On a malformed two-marker line whose first marker
                # was demoted (stripped) and a LATER marker carries the evidence
                # validation matched, the old binding attributed the later
                # marker's explanation to the first name's history. The combined
                # regex matches only when the line-start marker carries
                # evidence (= the marker validation matched), so name↔marker
                # stay aligned; otherwise it returns None and the cross-session
                # check correctly skips this malformed line.
                # The trailing level guard (``name_match.group(2) == str(level)``)
                # is the same level-alignment fix as in _carryforward_decision:
                # the combined regex is any-level + anchored while _GRADUATION_RE
                # is 2x/3x-only, so on a malformed line with a leading
                # ``1x [evidence:]`` and a later 2x/3x marker, the name must bind
                # to the marker validation actually matched (codex L3 re-verify).
                name_match = _NAMED_PATTERN_WITH_EVIDENCE_RE.match(line)
                if name_match is not None and name_match.group(2) == str(level):
                    pattern_name = name_match.group(1)
                    history = pattern_history_lookup(pattern_name)
                    if history:
                        # Compare today's explanation against each prior
                        # explanation INDIVIDUALLY and take the maximum
                        # overlap. Demote iff the worst-case (highest)
                        # per-prior overlap meets the threshold.
                        #
                        # v0.3.2 reframe (4-layer review convergence —
                        # Complement F2, Gemini observation 1, Codex
                        # HIGH, Contrarian Finding 4): the previous
                        # implementation compared today's explanation
                        # against the UNION of all prior explanation
                        # words (whole corpus). That structurally
                        # false-positives on long-lived legitimate
                        # patterns whose domain vocabulary naturally
                        # recurs across multi-faceted evidence: the
                        # union word-set grows monotonically with
                        # sessions, so the chance any new explanation
                        # shares ≥threshold words with the union
                        # approaches 1 over time. Per-prior comparison
                        # avoids this — legitimate distinct evidence
                        # passes against each prior individually even
                        # when accumulated vocabulary is large.
                        #
                        # Closure scope: this catches the lexical-
                        # rephrasing variant of slow-drift sycophantic
                        # accumulation (Phase 1b probe #1 S5 — today's
                        # explanation reuses ≥threshold words from a
                        # specific prior explanation). It does NOT
                        # catch the deliberately-divergent-vocabulary
                        # variant (S6 — today's explanation uses
                        # entirely new words against every prior).
                        # Divergent-vocab is Move #4's domain.
                        corpus = history.get("explanation_corpus") or ""
                        if not corpus:
                            # Legacy stores that only have last_explanation
                            corpus = history.get("last_explanation") or ""
                        prior_explanations = [
                            e for e in corpus.split("\n") if e.strip()
                        ]
                        # AM-PRESERVE-VS-SYCOPHANCY: exempt VERBATIM preservation
                        # (byte-identical explanation) — but ONLY when the line is
                        # held at/below its earned high-water mark, i.e. NOT
                        # inflating. A byte-identical LEVEL-UP (claiming a new high
                        # with the exact prior words + a fresh episode) is still
                        # suspect and stays caught; preservation never inflates.
                        # (Carryforward applies the same level gate via its own
                        # ``level > max_level`` check.) A re-worded high-overlap
                        # explanation also still trips the gate.
                        max_level_hist = history.get("max_level_reached")
                        # codex L3 (HIGH): require CURRENT grounding for the
                        # exemption. A byte-identical explanation whose cited LIVE
                        # episodes do NOT ground it (``explanation_valid`` False)
                        # must still demote — otherwise the co-citation block below
                        # (gated on ``not cross_session_overlap_words``) forges a
                        # Hebbian link between the unrelated cited episodes from an
                        # exact-copy accumulation (graph poisoning). The core
                        # preservation case (a dead citation carried forward) never
                        # reaches Site 2 — ``ids_valid`` is False there — and is held
                        # by carryforward, which forms no link.
                        is_preservation = (
                            grounding_checked
                            and explanation_valid
                            and isinstance(max_level_hist, int)
                            and level <= max_level_hist
                            and _is_verbatim_preservation(
                                explanation, prior_explanations
                            )
                        )
                        if prior_explanations and not is_preservation:
                            best_overlap: set[str] = set()
                            best_prior = ""
                            for prior in prior_explanations:
                                shared = _meaningful_word_overlap(explanation, prior)
                                if len(shared) > len(best_overlap):
                                    best_overlap = shared
                                    best_prior = prior
                            if len(best_overlap) >= cross_session_overlap_threshold:
                                cross_session_overlap_words = sorted(best_overlap)
                                prior_explanation_for_check = best_prior

            # AM-QUOTEFOOTGUN (v0.4.1): compute the co-cited episode set up
            # front so Hebbian link formation can be DECOUPLED from the
            # explanation-overlap immune gate below. ``explanation_valid``
            # governs graduation GROUNDING (validated vs demoted) — it must
            # NOT govern whether two episodes that were cited together get a
            # co-occurrence link. (Pre-0.4.1 the extraction was bolted inside
            # the validated branch, so anneal's own documented quoted
            # ``[evidence: id "why"]`` format silently formed 0 links whenever
            # the paraphrased explanation missed the ≥2-meaningful-word
            # lexical overlap, while bare ``[evidence: id, id]`` linked fine.)
            valid_cited = sorted(cited_ids & valid_ids)
            # AM-WARN (v0.4.2): record that this wrap had at least one
            # citation resolving to a real store episode — BEFORE any gate.
            # This is the un-gated truth AM-WARN Signal A needs (a
            # cross-session-demoted line still has resolving ids).
            if valid_cited:
                any_citation_resolved = True

            if ids_valid and explanation_valid and not cross_session_overlap_words:
                validated += 1
            elif cross_session_overlap_words:
                # Cross-session check fired: today's explanation reuses
                # vocabulary from the pattern's prior-session
                # explanation. Demote with a distinct marker so
                # operators can tell this apart from the ungrounded
                # path. The pattern_history row is NOT updated when
                # this fires — the prior session's explanation remains
                # authoritative until the agent composes a new
                # explanation with actually-different vocabulary.
                demoted += 1
                lines[i] = _demote_line(line, match, level, marker="(cross-session-overlap)")
                # `pattern_name` is in scope from the upper-block match
                # — this branch only fires when cross_session_overlap_words
                # is non-empty, which itself only happens after the
                # `_NAMED_PATTERN_RE.match(line)` above succeeded (line 348),
                # so pattern_name is guaranteed to be set. NOTE #1 v0.3.2
                # cleanup — dropped the redundant re-match + dead
                # `"<unnamed>"` fallback that the 4-layer review flagged.
                cross_session_collisions.append(CrossSessionCollision(
                    name=pattern_name,
                    today_level=level,
                    overlap_words=cross_session_overlap_words,
                    prior_explanation=prior_explanation_for_check,
                ))
            else:
                # AM-CARRYFORWARD (v0.4.6): the ungrounded-citation demotion
                # path. Before ratcheting the level down, check whether this is
                # a load-bearing pattern whose grounding merely failed THIS
                # session — at/below its earned high-water mark AND grounded
                # recently (warm). If so, HOLD it instead of demoting. Cold or
                # never-earned-this-level → demote as before. This consults the
                # SAME pattern_history the cross-session gate uses, but on the
                # ungrounded path (ids may not resolve), so it does its own
                # name-bound lookup; the (cross-session-overlap) immune branch
                # above is deliberately NOT carried forward.
                held = _carryforward_decision(
                    line=line,
                    level=level,
                    today=today,
                    pattern_history_lookup=pattern_history_lookup,
                    carryforward_cold_days=carryforward_cold_days,
                    cross_session_overlap_threshold=cross_session_overlap_threshold,
                )
                if held is not None:
                    lines[i] = _carryforward_line(line, match, level)
                    carried_forward.append(held)
                else:
                    demoted += 1
                    lines[i] = _demote_line(line, match, level)

            # Decoupled co-citation extraction (AM-QUOTEFOOTGUN). Record the
            # Hebbian co-occurrence for any line citing real episodes,
            # independent of explanation grounding — EXCEPT when the
            # cross-session immune defense flagged suspected sycophantic
            # re-graduation (we don't strengthen the graph from a gamed
            # accumulation). The graph records what fired together; the
            # immune gate above still decides what graduates. Note this
            # block runs even on the demoted explanation path, so a real
            # but paraphrased co-citation keeps its link.
            if ids_valid and not cross_session_overlap_words:
                if len(valid_cited) >= 2:
                    for idx_a in range(len(valid_cited)):
                        for idx_b in range(idx_a + 1, len(valid_cited)):
                            direct_co_citations.append(
                                (valid_cited[idx_a], valid_cited[idx_b])
                            )
                # Per-line co-cited IDs feed session-level co-citation
                all_validated_ids.append(set(valid_cited))
            continue

        # Fail-safe sunset: bare graduations without evidence
        if not citations_seen:
            continue

        bare_match = _BARE_GRADUATION_RE.search(line)
        if not bare_match:
            continue

        bare_level = int(bare_match.group(1))
        bare_date = bare_match.group(2)
        if bare_date != today:
            skipped_non_today += 1
            continue

        bare_demoted += 1
        old_marker = bare_match.group(0)
        # Diogenes 2026-05-22 LOW fix: parallel to the v0.3.3 _demote_line
        # treatment. `_BARE_GRADUATION_RE` accepts `\|\s*Nx` (space-optional),
        # so on `|2x` input the literal `.replace(f"| {N}x", ...)` no-op'd
        # and `bare_demoted` incremented while the text retained the old
        # level. State corruption mirroring the original `_demote_line` bug.
        # Use the same regex substitution against `\|\s*Nx` so all spacing
        # variants demote correctly. count=1 keeps the rewrite scoped.
        new_marker = re.sub(
            rf"\|\s*{bare_level}x",
            f"| {bare_level - 1}x",
            old_marker,
            count=1,
        )
        lines[i] = line.replace(old_marker, new_marker + " (needs-evidence)")

    reuse_max = max(citation_counts.values()) if citation_counts else 0
    gaming_suspects = detect_citation_gaming(citation_counts)

    return GraduationResult(
        text="\n".join(lines),
        validated=validated,
        demoted=demoted,
        citation_reuse_max=reuse_max,
        bare_demoted=bare_demoted,
        skipped_non_today=skipped_non_today,
        citation_counts=dict(citation_counts),
        gaming_suspects=gaming_suspects,
        direct_co_citations=direct_co_citations,
        all_validated_ids=all_validated_ids,
        any_citation_resolved=any_citation_resolved,
        cross_session_collisions=cross_session_collisions,
        carried_forward=carried_forward,
    )


def _meaningful_word_overlap(text_a: str, text_b: str) -> set[str]:
    """Return the set of meaningful words shared by two explanation texts.

    Uses the same tokenization rule as :func:`check_explanation_overlap`:
    split on non-alphanumeric, lowercase, drop stop words and tokens of
    length ≤2. Returning the actual set (not just a count) lets callers
    surface which specific words triggered a cross-session collision —
    valuable for the audit log and for operator review.
    """
    def words(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-zA-Z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP_WORDS
        }
    return words(text_a) & words(text_b)


def _is_verbatim_preservation(explanation: str, priors: list[str]) -> bool:
    """True if ``explanation`` is byte-identical (whitespace-normalized) to any
    of ``priors`` — the pattern line was PRESERVED verbatim, not re-worded.

    AM-PRESERVE-VS-SYCOPHANCY: both sycophancy sites — the cross-session-overlap
    gate and :func:`_carryforward_decision`'s internal guard — treat high
    vocabulary overlap with a prior explanation as suspected sycophantic
    re-graduation. But a verbatim-PRESERVED explanation is 100% overlap, so the
    very mechanism built to protect a warm, at-peak pattern (carryforward)
    refuses to protect the ones carried forward unchanged — the gate erodes the
    most-stable, highest-value patterns *because* they are stable. A sycophant
    RE-WORDS (overlap high, text different); preservation is text-IDENTICAL.
    Byte-identity is the un-fakeable discriminator, and it cannot be gamed for
    gain: a verbatim copy can only HOLD a level (carryforward holds at/below
    ``max_level_reached``; the cross-session exemption only avoids a demotion),
    never inflate past the earned high-water mark. So a byte-identical
    explanation is exempt from the overlap demotion at both sites, while a
    merely-overlapping (re-worded) explanation still trips the gate.

    Normalization collapses internal whitespace and strips the ends, so a
    preserved line that was only re-wrapped still reads as preservation —
    re-wrapping changes no words, and a sycophantic re-graduation changes
    words, not just whitespace.
    """
    norm = " ".join(explanation.split())
    if not norm:
        return False
    return any(norm == " ".join(p.split()) for p in priors)


def check_explanation_overlap(explanation: str, episode_content: str) -> bool:
    """Check if an explanation references actual content from the cited episode.

    Uses word overlap (excluding stop words). At least 2 meaningful words
    from the explanation must appear in the episode content. Single-word
    overlap is too easy to game (e.g., "database" matching any DB-related
    episode). Two words provide meaningful grounding while still allowing
    paraphrasing.

    Args:
        explanation: The quoted explanation from the evidence tag.
        episode_content: The full content of the cited episode.

    Returns:
        True if the explanation sufficiently references the episode content.
    """
    def meaningful_words(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-zA-Z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP_WORDS
        }

    explanation_words = meaningful_words(explanation)
    episode_words = meaningful_words(episode_content)
    return len(explanation_words & episode_words) >= 2


def extract_session_co_citations(
    all_validated_ids: list[set[str]],
) -> set[tuple[str, str]]:
    """Extract session-level co-citation pairs from per-line co-cited IDs.

    Session co-citations are pairs of episodes cited in DIFFERENT pattern
    lines during the same wrap. These represent a weaker association signal
    than direct co-citations (same line).

    Args:
        all_validated_ids: List of sets, each containing the real (existing)
            co-cited episode IDs from a single pattern line. Since
            AM-QUOTEFOOTGUN (v0.4.1) these are NOT necessarily
            graduation-validated — a demoted-grounding line still contributes
            its real co-cited IDs (co-occurrence is independent of grounding).

    Returns:
        Set of canonical (id_a, id_b) pairs where id_a < id_b.
    """
    # Pre-build index: episode_id -> set of line indices it appears in
    # O(n*k) where n = number of lines, k = avg IDs per line
    id_to_lines: dict[str, set[int]] = {}
    all_ids: set[str] = set()
    for idx, id_set in enumerate(all_validated_ids):
        for ep_id in id_set:
            all_ids.add(ep_id)
            if ep_id not in id_to_lines:
                id_to_lines[ep_id] = set()
            id_to_lines[ep_id].add(idx)

    # Generate pairs between IDs that appear in DIFFERENT lines
    # O(n^2) over unique IDs with O(1) line-set lookups
    pairs: set[tuple[str, str]] = set()
    id_list = sorted(all_ids)
    for i in range(len(id_list)):
        a_lines = id_to_lines[id_list[i]]
        for j in range(i + 1, len(id_list)):
            b_lines = id_to_lines[id_list[j]]
            if a_lines != b_lines or len(a_lines) > 1:
                # They appear in at least one different line context
                pairs.add((id_list[i], id_list[j]))

    return pairs


def detect_stale_patterns(
    text: str,
    today: str,
    staleness_days: int = 7,
    graduating_headings: frozenset[str] = DEFAULT_GRADUATING,
) -> list[StalenessInfo]:
    """Find patterns that haven't been validated recently.

    Scans the ## Patterns section for patterns with dates older than
    staleness_days. These are candidates for removal or demotion.

    Args:
        text: The continuity file text.
        today: Today's date as YYYY-MM-DD string.
        staleness_days: Days without validation before flagging. Default 7.

    Returns:
        List of StalenessInfo for patterns exceeding the staleness threshold.
    """
    today_dt = _datetime.strptime(today, "%Y-%m-%d")
    stale: list[StalenessInfo] = []

    lines = text.split("\n")
    in_patterns = False

    for i, line in enumerate(lines):
        if line.startswith("## "):
            in_patterns = _is_graduating_heading(line, graduating_headings)
            continue
        if not in_patterns:
            continue

        match = _PATTERN_RE.search(line)
        if not match:
            continue

        level = int(match.group(1))
        date_str = match.group(2)

        try:
            pattern_dt = _datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        days_old = (today_dt - pattern_dt).days
        if days_old >= staleness_days:
            stale.append(StalenessInfo(
                line_number=i + 1,  # 1-indexed for human readability
                content=line.strip(),
                level=level,
                last_date=date_str,
                days_stale=days_old,
            ))

    return stale


def extract_pattern_names(
    text: str, graduating_headings: frozenset[str] = DEFAULT_GRADUATING
) -> dict[str, int]:
    """Extract the max graduation level per pattern name in the ## Patterns section.

    Scans only the ## Patterns section (consistent with validate_graduations
    and detect_stale_patterns) and returns a mapping of pattern name →
    highest level (Nx) seen. If a pattern name appears on multiple lines
    in the section (which would be a malformed continuity but is not
    enforced anywhere), the highest level wins.

    Pattern names are matched against _NAMED_PATTERN_RE — operator-style
    identifiers (starts with a letter; contains letters, digits, underscore,
    dot, hyphen). Patterns that don't match this grammar (e.g. free-form
    prose bullets that happen to contain ``| Nx``) are silently skipped.
    This is intentional: continuity-internal documentation that resembles
    a graduation marker should not be mistaken for one.

    Args:
        text: The continuity file text.

    Returns:
        Mapping ``{pattern_name: max_level_seen}`` for every named pattern
        found in the ``## Patterns`` section.
    """
    result: dict[str, int] = {}
    in_patterns = False
    for line in text.split("\n"):
        if line.startswith("## "):
            in_patterns = _is_graduating_heading(line, graduating_headings)
            continue
        if not in_patterns:
            continue
        match = _NAMED_PATTERN_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        try:
            level = int(match.group(2))
        except ValueError:
            continue
        prior = result.get(name, 0)
        if level > prior:
            result[name] = level
    return result


def detect_pattern_omissions(
    prior_text: str,
    new_text: str,
    min_level: int = 2,
    graduating_headings: frozenset[str] = DEFAULT_GRADUATING,
) -> list[OmittedPattern]:
    """Detect Proven-tier patterns silently dropped between two wraps.

    Compares the ``## Patterns`` section of the prior continuity against
    the new continuity and surfaces any pattern at ``min_level`` or
    higher in the prior text that is absent at any level in the new
    text. Default ``min_level=2`` so 1x ("Developing") drops are NOT
    surfaced — those are normal lifecycle; only 2x/3x ("Proven-tier")
    omissions are tracked.

    This is informational, not a gate. The library does not refuse the
    save when patterns are omitted — the agent may have intentionally
    retired the pattern, or may have silently erased load-bearing
    evidence. The returned list goes onto :class:`GraduationResult` and
    into the wrap audit event so operators and downstream review can
    see what was dropped between sessions.

    Added in response to Bold Stand Phase 1b probe #1 (2026-05-21),
    which surfaced silent pattern omission as a reachable adversarial-
    agent gap not caught by validate_graduations.

    Args:
        prior_text: The continuity text BEFORE this wrap (the
            persisted file state at the start of the session).
        new_text: The continuity text the agent is about to save.
        min_level: Minimum level in prior_text to count as an omission
            when missing from new_text. Default 2.

    Returns:
        List of :class:`OmittedPattern` for every pattern at or above
        ``min_level`` in ``prior_text`` that is not present at any
        level in ``new_text``.
    """
    prior_levels = extract_pattern_names(prior_text, graduating_headings)
    new_names = set(extract_pattern_names(new_text, graduating_headings).keys())
    omissions: list[OmittedPattern] = []
    for name, prior_level in prior_levels.items():
        if prior_level < min_level:
            continue
        if name in new_names:
            continue
        omissions.append(OmittedPattern(name=name, prior_level=prior_level))
    # Stable order so callers / tests / audit logs see consistent output
    omissions.sort(key=lambda op: (-op.prior_level, op.name))
    return omissions


def extract_proven_patterns(
    text: str,
    min_level: int = 2,
    graduating_headings: frozenset[str] = DEFAULT_GRADUATING,
) -> list[str]:
    """Return the operator-style names of every Proven-tier pattern in
    the continuity's ``## Patterns`` section.

    Move #4 library layer (v0.3.2): this is the list of names that the
    methodology-layer contradiction-scan discipline (Levain
    WRAP_PROTOCOL.md) requires the agent to consider before any new
    Proven graduation. ``prepare_wrap`` surfaces this as the
    ``uncovered_proven_to_check`` field so the agent has the list in
    front of them during compression.

    The default ``min_level=2`` means 1x ("Developing") patterns are
    NOT returned — only Proven-tier (2x and 3x) patterns are
    contradiction-scan targets. Adjust ``min_level`` if a stricter or
    looser definition of "Proven" applies in a calling context.

    Args:
        text: The continuity file text.
        min_level: Minimum level to include. Default 2.

    Returns:
        Sorted list of operator-style pattern names. Sorted so
        downstream methodology-layer prompts and audit-log entries see
        a deterministic order.
    """
    level_map = extract_pattern_names(text, graduating_headings)
    names = [name for name, lvl in level_map.items() if lvl >= min_level]
    return sorted(names)


def extract_contradiction_declarations(
    text: str, graduating_headings: frozenset[str] = DEFAULT_GRADUATING
) -> dict[str, list[str]]:
    """Extract every pattern line's contradiction-stance declaration.

    Returns a mapping from pattern name → list of contradicted names
    declared on that line. A pattern with ``[no-contradicts]`` returns
    an empty list (explicit no-contradiction declaration); a pattern
    with no annotation at all is absent from the returned dict.

    Move #4 library layer (v0.3.2). Used by
    :func:`detect_proven_without_declaration` to identify new Proven
    graduations that skipped the methodology-layer contradiction-scan
    discipline.

    Args:
        text: The continuity file text.

    Returns:
        Mapping ``{pattern_name: [contradicted_name, ...]}``. Empty
        list means ``[no-contradicts]`` declared. Absence from the
        dict means neither declaration was present (caught by
        :func:`detect_proven_without_declaration` for new Provens).
    """
    declarations: dict[str, list[str]] = {}
    in_patterns = False
    for line in text.split("\n"):
        if line.startswith("## "):
            in_patterns = _is_graduating_heading(line, graduating_headings)
            continue
        if not in_patterns:
            continue
        name_match = _NAMED_PATTERN_RE.match(line)
        if not name_match:
            continue
        name = name_match.group(1)
        # v0.3.3 MEDIUM #3 fix: strip any `[evidence: ... "..."]` block
        # from the line before searching for declarations. Without this,
        # an evidence explanation containing literal `[no-contradicts]`
        # or `[contradicts: X]` text would spoof the declaration —
        # `[evidence: abc12345 "we wrote [no-contradicts] in a log"]`
        # would be treated as an explicit no-contradicts declaration
        # for the pattern. Codex L3 v0.3.2 review caught this.
        evidence_stripped_line = _PATTERN_LINE_WITH_EVIDENCE_RE.sub("", line)
        if _NO_CONTRADICTS_RE.search(evidence_stripped_line):
            declarations[name] = []
            continue
        contradicts_match = _CONTRADICTS_RE.search(evidence_stripped_line)
        if contradicts_match:
            raw_names = contradicts_match.group(1)
            contradicted = [
                n.strip() for n in raw_names.split(",") if n.strip()
            ]
            declarations[name] = contradicted
    return declarations


def detect_proven_without_declaration(
    prior_text: str,
    new_text: str,
    today: str | None = None,
    min_level: int = 2,
    graduating_headings: frozenset[str] = DEFAULT_GRADUATING,
) -> list[ProvenWithoutDeclaration]:
    """Detect NEW Proven graduations that landed without contradiction
    stance declaration.

    A "new Proven" is a pattern at ``min_level`` or higher in the new
    continuity that did NOT appear at the same level in the prior
    continuity AND whose graduation line carries today's date — i.e.,
    the agent graduated it to Proven-tier in *this* wrap, not a
    restored/imported older line that happens to be absent from the
    prior text. The methodology layer (Levain WRAP_PROTOCOL.md)
    requires every such graduation to carry either
    ``[contradicts: name_a, ...]`` or ``[no-contradicts]``; this
    function surfaces the ones that didn't.

    The library does NOT refuse the save on these — audit signal,
    not gate. The signal flows through ``SaveContinuityResult``'s
    ``proven_without_contradicts_declaration`` field and into the
    hash-chained audit log under the ``continuity_saved`` event.
    Operator-review (Diogenes weekly sweep) is the LLM-as-judge layer
    that actually inspects pairs for semantic opposition; this
    audit-signal lets Diogenes know which graduations need its
    attention.

    Added in v0.3.2 as the library-layer surface for the Move #4
    contradiction-detection architecture. ``today`` parameter added
    in v0.3.3 to close the today-awareness gap Codex L3 caught (a
    restored/imported old-date line was being flagged as a new
    Proven graduation under v0.3.2's level-only comparison).

    Args:
        prior_text: The continuity text BEFORE this wrap.
        new_text: The continuity text the agent is about to save.
        today: Today's date as YYYY-MM-DD. When provided, only
            graduation lines whose date == today are eligible to
            be flagged as "new Proven without declaration." When
            ``None``, falls back to v0.3.2's level-only comparison
            (preserves backward compatibility for callers that
            don't pass today; not recommended — pass today).
        min_level: Minimum level to count as "Proven". Default 2.

    Returns:
        List of :class:`ProvenWithoutDeclaration` for every NEW Proven
        graduation in the new continuity that lacks either declaration.
        Sorted by (level descending, name ascending) for stable
        downstream output.
    """
    prior_levels = extract_pattern_names(prior_text, graduating_headings)

    # AM-PERNAME-LINEBIND (v0.4.6): bind level + date + contradiction-stance
    # to the SAME physical line a name graduated on, instead of three
    # independent name-keyed maps. Pre-0.4.6 combined extract_pattern_names
    # (max-level per name), a separate last-wins name->date map, and
    # extract_contradiction_declarations (name->stance, also last-wins). On a
    # malformed line set carrying the SAME name twice — e.g. a 3x line with NO
    # stance plus a 1x dup carrying [no-contradicts] — the name-keyed
    # declaration map let the 1x dup's stance satisfy the 3x line's MISSING
    # declaration (a false-negative audit), and the name-keyed date map could
    # bind the wrong line's date. Scanning per-line and keeping the record of
    # the HIGHEST-level line per name (first such line on a tie, deterministic)
    # keeps the (level, date, stance) tuple coherent. On well-formed
    # unique-name input the recorded level equals extract_pattern_names'
    # max-level and the date/stance come from that same line, so this is
    # behavior-identical there. (codex L3, 0.4.5 — a pre-existing per-name
    # edge proven on clean main, deferred into this pass.)
    #
    # v0.3.3 MEDIUM #4 (today-awareness) preserved: when ``today`` is given,
    # only graduation lines dated today are eligible — a restored/imported
    # old-date Proven line absent from prior is NOT flagged as a new
    # graduation needing a declaration.
    new_records: dict[str, tuple[int, str | None, bool]] = {}
    in_patterns = False
    for line in new_text.split("\n"):
        if line.startswith("## "):
            in_patterns = _is_graduating_heading(line, graduating_headings)
            continue
        if not in_patterns:
            continue
        name_match = _NAMED_PATTERN_RE.match(line)
        if name_match is None:
            continue
        try:
            line_level = int(name_match.group(2))
        except ValueError:
            continue
        name = name_match.group(1)
        existing = new_records.get(name)
        if existing is not None and existing[0] >= line_level:
            # Already recorded a line at this level or higher (first-wins
            # on ties) — keep the graduating line's coherent record.
            continue
        date_match = _PATTERN_RE.search(line)
        line_date = date_match.group(2) if date_match is not None else None
        # Contradiction stance on THIS line. Evidence-stripped so an
        # explanation that quotes "[no-contradicts]"/"[contradicts: X]"
        # cannot spoof the stance (same guard as
        # extract_contradiction_declarations, v0.3.3 MEDIUM #3).
        evidence_stripped = _PATTERN_LINE_WITH_EVIDENCE_RE.sub("", line)
        has_declaration = bool(
            _NO_CONTRADICTS_RE.search(evidence_stripped)
            or _CONTRADICTS_RE.search(evidence_stripped)
        )
        new_records[name] = (line_level, line_date, has_declaration)

    new_provens: list[ProvenWithoutDeclaration] = []
    for name, (new_level, line_date, has_declaration) in new_records.items():
        if new_level < min_level:
            continue
        prior_level = prior_levels.get(name, 0)
        if prior_level >= new_level:
            # Carried forward at same or higher level — not a new graduation
            continue
        if today is not None and line_date != today:
            # Not authored this wrap — restored/imported old-date line
            # that the discipline cannot require to carry a contradiction
            # declaration for "today's graduation."
            continue
        if has_declaration:
            # The graduating line itself carried an explicit stance.
            continue
        new_provens.append(ProvenWithoutDeclaration(name=name, level=new_level))

    new_provens.sort(key=lambda p: (-p.level, p.name))
    return new_provens


def detect_citation_gaming(citation_counts: dict[str, int], threshold: int = 3) -> list[str]:
    """Detect episodes cited suspiciously many times.

    If a single episode is cited as evidence for many different patterns,
    it may indicate the LLM is gaming the citation system rather than
    finding genuine independent evidence.

    Args:
        citation_counts: Mapping of episode ID -> citation count.
        threshold: Number of citations before flagging. Default 3.

    Returns:
        List of episode IDs that exceed the threshold.
    """
    return [
        ep_id for ep_id, count in citation_counts.items()
        if count >= threshold
    ]


# -- Internal helpers --


def _demote_line(
    line: str,
    match: re.Match,
    level: int,
    marker: str = "(ungrounded)",
) -> str:
    """Demote a graduated pattern line (3x->2x or 2x->1x) and mark it.

    Uses positional replacement via match span to avoid fragility
    from str.replace on LLM-generated text that might contain
    duplicate marker-like substrings.

    Args:
        line: The full line being demoted.
        match: The ``_GRADUATION_RE`` match object that captured the
            graduation marker.
        level: The current level (2 or 3) to demote from.
        marker: The text to replace the ``[evidence: ...]`` tag with.
            Default ``(ungrounded)`` (the citation-validation failure
            path); cross-session-overlap demotions pass
            ``(cross-session-overlap)`` so operators can distinguish
            the failure mode at a glance.
    """
    old_marker = match.group(0)
    # v0.3.3 HIGH #2 fix: use regex substitution against the actual
    # graduation pattern (`\|\s*Nx`) rather than a literal `f"| {N}x"`
    # string replace. The widened `_NAMED_PATTERN_RE` in v0.3.2
    # accepts `|2x` (no space after pipe) AND `| 2x` (with space) AND
    # `|  2x` (multiple spaces) — the literal replace only matched the
    # single-space form, so demotion silently no-op'd on the
    # no-space/multi-space variants. Counter said `demoted == 1` but
    # text retained the old level. State corruption. Codex L3 caught
    # this end-to-end against a `|2x` test input. The regex replaces
    # only the FIRST match (count=1) so we don't accidentally rewrite
    # graduation markers elsewhere in the captured span.
    new_marker = re.sub(
        rf"\|\s*{level}x",
        f"| {level - 1}x",
        old_marker,
        count=1,
    )
    # Replace evidence tag with the demotion marker
    new_marker = re.sub(
        r'\[evidence:\s*[a-fA-F0-9][a-fA-F0-9, ]*(?:\s+"[^"]*")?\s*\]',
        marker, new_marker
    )
    # Positional replacement — immune to duplicate marker text elsewhere in line
    start, end = match.span()
    return line[:start] + new_marker + line[end:]


def _days_between(last_seen_at: Any, today: str) -> int | None:
    """Whole days from ``last_seen_at`` to ``today``.

    Both are ISO; only the leading ``YYYY-MM-DD`` is read, so a full
    timestamp like ``2026-06-04T14:00:00Z`` (the format
    ``upsert_pattern_history`` stores) parses fine. Returns ``None`` if
    either value is missing or unparseable — callers treat ``None`` as
    "no recency signal" and fall through to demotion (conservative: no
    recency → no carryforward protection). A negative result
    (``last_seen_at`` in the future relative to ``today``) is returned
    as-is — this is an honest date-diff util; the carryforward POLICY
    (:func:`_carryforward_decision`) decides how to treat negatives (it
    rejects them as untrustworthy rather than maximally warm).
    """
    if not isinstance(last_seen_at, str) or len(last_seen_at) < 10:
        return None
    if not isinstance(today, str) or len(today) < 10:
        return None
    try:
        prior = _date.fromisoformat(last_seen_at[:10])
        now = _date.fromisoformat(today[:10])
    except ValueError:
        return None
    return (now - prior).days


def _carryforward_decision(
    line: str,
    level: int,
    today: str,
    pattern_history_lookup: Callable[[str], dict[str, Any] | None] | None,
    carryforward_cold_days: int | None,
    cross_session_overlap_threshold: int = 3,
) -> CarriedForward | None:
    """Decide whether an ungrounded 2x/3x line should be HELD instead of
    demoted (AM-CARRYFORWARD, v0.4.6). Returns a :class:`CarriedForward`
    to hold the line, or ``None`` to fall through to demotion.

    Holds iff ALL of:
      * carryforward is enabled (``carryforward_cold_days`` is not None);
      * a ``pattern_history_lookup`` is wired;
      * the line's NAME binds to its OWN evidence marker via
        :data:`_NAMED_PATTERN_WITH_EVIDENCE_RE` (anchored at line start) —
        anonymous/freeform lines, and malformed multi-marker lines whose
        line-start marker does NOT carry the evidence (so a LATER marker is
        what validation matched), return None and demote. Binding the name to
        its own marker is what stops a held marker from using a DIFFERENT
        pattern's history on a malformed line (AM-PERNAME-LINEBIND; codex L3).
      * the line's explanation does NOT sycophantically overlap the pattern's
        prior history (see below);
      * the pattern has history with an integer ``max_level_reached``;
      * ``level <= max_level_reached`` (it genuinely earned this level);
      * ``last_seen_at`` is within ``carryforward_cold_days`` of today
        (warm — grounded recently in SOME prior session, independent of
        this session's failed citation).

    See :class:`CarriedForward` for the full rationale (domain-blind
    erosion fix; warmth decays on its own because a held line does not
    upsert pattern_history).
    """
    if carryforward_cold_days is None or pattern_history_lookup is None:
        return None
    # AM-PERNAME-LINEBIND (codex L3, v0.4.6): bind the name to its OWN evidence
    # marker. The combined regex is anchored at line start and requires the
    # line-start marker to carry well-formed evidence — its evidence syntax is
    # equivalent to _GRADUATION_RE's, so when it matches, its marker IS the
    # first evidence marker (the one _GRADUATION_RE.search found), guaranteeing
    # name↔marker alignment. When it returns None, the marker validation
    # matched is a LATER marker the line-start name does not own (e.g. the
    # first marker was demoted and stripped) → decline, demote. This replaces
    # a line-anchored _NAMED_PATTERN_RE.match that always returned the FIRST
    # name and could cross-bind it to a later marker's level on a malformed
    # two-marker line.
    name_match = _NAMED_PATTERN_WITH_EVIDENCE_RE.match(line)
    if name_match is None:
        return None
    # Level-alignment guard (codex L3 re-verify, v0.4.6): the combined regex is
    # ANY-level + anchored, but validation matched a 2x/3x marker
    # (_GRADUATION_RE is 2x/3x-only + unanchored). On a malformed line whose
    # LEADING marker is e.g. ``1x [evidence:]`` and a LATER marker is the 3x
    # validation matched, the combined regex binds the leading 1x name while
    # `level` is the later marker's — a cross-bind. Requiring the bound marker's
    # level to equal the validation `level` closes this: when they match, the
    # combined regex's (first, line-start) evidence marker IS a 2x/3x marker, so
    # nothing with evidence precedes it and it is necessarily the same marker
    # _GRADUATION_RE.search found. Mismatch → the line-start name does not own
    # the validated marker → decline.
    if name_match.group(2) != str(level):
        return None
    name = name_match.group(1)
    explanation = name_match.group(5)
    history = pattern_history_lookup(name)
    if not history:
        return None
    # AM-CARRYFORWARD scope-airtightness (codex L3, v0.4.6): NEVER carry forward
    # a line whose explanation sycophantically overlaps the pattern's prior
    # history. The cross-session immune demotion only runs on the ids-RESOLVING
    # path (AM-XSESSION-LINKGATE keys it on ``ids_valid``), so a warm at-peak
    # pattern with a DEAD citation and a vocabulary-reused explanation would
    # otherwise skip the overlap check entirely and be HELD — shielding a
    # suspected sycophantic re-citation. Run the same per-prior overlap test
    # here, independent of id resolution: if it trips, decline (the line falls
    # through to plain ``(ungrounded)`` demotion). This refuses PROTECTION to a
    # suspicious line; it does not alter the cross-session collision REPORTING
    # or the immune trigger (no link to suppress on the dead-id path).
    if explanation:
        corpus = history.get("explanation_corpus") or history.get("last_explanation") or ""
        priors = [e for e in corpus.split("\n") if e.strip()]
        # AM-PRESERVE-VS-SYCOPHANCY: a byte-identical explanation is verbatim
        # PRESERVATION, not sycophantic re-wording — hold it (a sycophant
        # re-words; preservation is identical). Only a re-worded high-overlap
        # explanation refuses the carryforward protection.
        if not _is_verbatim_preservation(explanation, priors):
            for prior in priors:
                if len(_meaningful_word_overlap(explanation, prior)) >= cross_session_overlap_threshold:
                    return None
    max_level = history.get("max_level_reached")
    if not isinstance(max_level, int):
        return None
    if level > max_level:
        # Never earned this level — don't protect an un-earned rung.
        return None
    days = _days_between(history.get("last_seen_at"), today)
    if days is None or days < -1 or days > carryforward_cold_days:
        # No recency signal, a clearly-FUTURE last_seen_at (more than one day
        # ahead = clock corruption / leaked future date — not trustworthy, do
        # NOT treat as maximally warm), or grounded too long ago (cold) → age
        # out. The -1 tolerance absorbs the legitimate ≤1-day skew between the
        # store's UTC ``last_seen_at`` and a local-calendar ``today`` (a real
        # grounding in US evening hours is stamped "tomorrow UTC"); rejecting
        # it would false-cold a genuinely-warm pattern (codex L3). Conservative-
        # demotion otherwise: absent/untrustworthy recency → no protection.
        return None
    return CarriedForward(
        name=name,
        held_level=level,
        max_level_reached=max_level,
        # Clamp the reported recency to >= 0: the decision tolerates a -1 UTC/
        # local skew, but a negative "days since grounded" would read oddly in
        # the audit surface.
        days_since_grounded=max(0, days),
    )


def _carryforward_line(line: str, match: re.Match, level: int) -> str:
    """Hold a graduated line at its level (AM-CARRYFORWARD), replacing the
    ``[evidence: ...]`` tag with a ``(carried-forward)`` marker.

    Mirrors :func:`_demote_line`'s positional rewrite but does NOT
    decrement the level. Stripping the evidence tag is intentional: a
    carried-forward line does not match the upsert path's
    evidence-bearing regex, so it does not upsert pattern_history and
    ``last_seen_at`` does not advance — the warmth that protected it
    decays naturally, and a pattern that keeps failing to ground ages
    out on its own (the recency signal IS the failing-streak signal).
    """
    old_marker = match.group(0)
    new_marker = re.sub(
        r'\[evidence:\s*[a-fA-F0-9][a-fA-F0-9, ]*(?:\s+"[^"]*")?\s*\]',
        "(carried-forward)",
        old_marker,
    )
    start, end = match.span()
    return line[:start] + new_marker + line[end:]
