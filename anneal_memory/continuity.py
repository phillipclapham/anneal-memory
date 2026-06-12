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
import warnings
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graduation import (
    CrossSessionCollision,
    OmittedPattern,
    PatternSummary,
    detect_pattern_omissions,
    detect_stale_patterns,
    extract_pattern_names,
    extract_pattern_summaries,
    validate_graduations,
    _NAMED_PATTERN_RE,
    _NAMED_PATTERN_WITH_EVIDENCE_RE,
    _is_graduating_heading,
)
from .schema import (
    DEFAULT_SCHEMA,
    SectionSpec,
    default_max_chars,
    graduating_headings,
    required_headings,
    schema_role_warning,
)
from .crystal import CrystalError, CrystalStore
from .store import StoreError, WrapInProgressError, _fsync_dir, _safe_unlink
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


def _matching_required_headings(line_lower: str, required: set[str]) -> list[str]:
    """Required schema headings (lowercased) satisfied by one ``## `` header
    line, via the lenient word-bounded match.

    More than one match means the header is AMBIGUOUS — a single line satisfying
    two required sections, e.g. ``## Patterns and Understanding``. That would let
    a merged header route one body into two protected roles and defeat the
    shrink gate (v0.3.5), and it also lets one line silently satisfy two
    ``validate_structure`` requirements. Callers treat ``len > 1`` as malformed.
    """
    return [
        h
        for h in required
        if re.search(rf"(?<!\w){re.escape(h)}(?!\w)", line_lower)
    ]


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
            matched = _matching_required_headings(line.lower(), required)
            # An ambiguous header (one line satisfying multiple required
            # sections, e.g. "## Patterns and Understanding") is malformed: it
            # merges two protected roles into one body and would defeat the
            # shrink gate. Reject the whole structure so each section keeps its
            # own header line.
            if len(matched) > 1:
                return False
            found.update(matched)
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


# --- Catastrophic-shrink gate (v0.3.5) ---------------------------------------
#
# A wrap that collapses a protected-role section — the timeless felt layer
# (``narrative-timeless``) or the graduated-identity layer (``graduating``) —
# or the whole neocortex is almost always a recency-trap / stateless-reset
# failure, not a deliberate edit. (flow dual-write wrap #4, 2026-05-31: a
# single-session wrap compressed a 19,702-char neocortex down to 1,608 —
# ``## Understanding`` became one paragraph, ``## Patterns`` one line — and
# nothing structural caught it; ``validate_structure`` passed because every
# heading was still present, just gutted.)
#
# ``structural_invariants_beat_discipline``: the proportion-check that's meant
# to prevent this is discipline, and discipline drifts across wrap drivers (a
# different model, the same model under load, a single-task session). This gate
# refuses the collapse at the save boundary instead of trusting every driver to
# hold the check.
#
# SCOPE — partnership entities only. The gate fires ONLY when the store's schema
# declares a ``narrative-timeless`` section (the felt layer): flow's
# ``FLOW_SCHEMA`` today, and future Levain-seeded partnership entities. Pure ops
# entities (``DEFAULT_SCHEMA`` — daemon, anansi, Argus, diogenes, nexus, prism)
# are NOT gated and keep their exact pre-0.3.5 audit-not-gate behavior
# (byte-identical; the schema feature's backward-compat invariant holds). Three
# reasons: they wrap autonomously overnight with no human to pass an override;
# they legitimately consolidate many graduated patterns into a few dense
# meta-patterns (a large Patterns shrink that is correct, not a collapse — the
# graduation philosophy *encourages* it); and they have no felt layer to lose.
# The gate guards exactly the entities that (a) declared a felt / identity layer
# worth protecting and (b) wrap with a human present who can pass
# ``allow_shrink`` for a deliberate diet. Decomposition along the existing
# partnership-vs-ops schema seam, not a new distinction.
#
# Retain floors (partnership entities only): the ``narrative-timeless`` (felt)
# and ``graduating`` (identity) sections must each retain >=50% of prior mass;
# the whole document >=25%. Sections / documents below the char floor are never
# gated (thin sections cannot meaningfully "collapse"). A deliberate diet
# (one-time migration recompression) passes ``allow_shrink=True``.
# AM-CRYSTAL-MIGRATE: a DELIBERATELY-LOOSE scan for "is this pattern still present
# in the new working set" — matches a ``name | Nx`` graduation marker ANYWHERE on a
# line (column-0, bullet, indented, OR a second marker merged onto another line),
# unlike the anchored first-marker ``_NAMED_PATTERN_RE``. It is used ONLY to CANCEL
# credit (decide a pattern survived), so over-detection is the SAFE bias: a survivor
# that dodges the anchored regex (a column-0 bare line, a merged second marker) is
# still caught here → not credited as departed → the gate stays strict. The earn-
# credit side keeps the strict anchored regex (under-detection there is also safe).
_ANY_GRADUATION_MARKER_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.\-]*)[ \t]*\|[ \t]*\d+x")

_SHRINK_GATE_MIN_PRIOR_CHARS = 500
_SHRINK_RETAIN_FRACTION: dict[str, float] = {
    "narrative-timeless": 0.5,
    "graduating": 0.5,
}
_DOC_SHRINK_RETAIN_FRACTION = 0.25


def _schema_section_masses(text: str, schema: list[SectionSpec]) -> dict[str, int]:
    """Character count per schema section, keyed by lowercased schema heading.

    Walks the ``## `` header lines and credits each section's character span to
    the schema heading(s) it satisfies, using the SAME word-bounded match
    :func:`validate_structure` uses — so a descriptive header like
    ``## State of Mind`` is credited to a required ``State`` section. A header
    matching no schema heading is ignored; a header matching several (which
    :func:`~anneal_memory.schema.validate_schema` forbids among the schema's own
    headings) credits each, which is the conservative choice for a gate.
    """
    required = {h.lower() for h in required_headings(schema)}
    masses: dict[str, int] = {h: 0 for h in required}
    current: list[str] = []
    current_chars = 0

    def _flush() -> None:
        for h in current:
            masses[h] += current_chars

    for line in text.split("\n"):
        if line.startswith("## "):
            _flush()
            matched = _matching_required_headings(line.lower(), required)
            # Single-credit only: an ambiguous header (matches >1 required
            # section) credits its span to NONE — conservative for a shrink
            # gate, since a merged body must not fake mass for two protected
            # roles. validated_save_continuity rejects such headers up front,
            # so this is defense in depth.
            current = matched if len(matched) == 1 else []
            current_chars = len(line) + 1
        else:
            current_chars += len(line) + 1
    _flush()
    return masses


def _role_section_body(text: str, schema: list[SectionSpec], role: str) -> list[str]:
    """Body lines (header lines excluded) of every section whose schema role is
    ``role``. Used by the crystallization-credit accounting to find a departed
    pattern's prior line mass. Mirrors :func:`_schema_section_masses`'s
    single-credit walk: an ambiguous header credits NONE."""
    target = {s["heading"].lower() for s in schema if s["role"] == role}
    if not target:
        return []
    required = {h.lower() for h in required_headings(schema)}
    out: list[str] = []
    in_target = False
    for line in text.split("\n"):
        if line.startswith("## "):
            matched = _matching_required_headings(line.lower(), required)
            in_target = len(matched) == 1 and matched[0] in target
            continue  # header line excluded — patterns live in the body
        if in_target:
            out.append(line)
    return out


def _crystallization_credit(
    prior_text: str | None,
    new_text: str,
    schema: list[SectionSpec],
    crystal_store: CrystalStore | None,
) -> dict[str, int]:
    """Char credit, per protected role, for graduated patterns that DEPARTED the
    ``## Patterns`` working set into the crystallized store this wrap.

    The structural distinction the shrink gate needs (``structural_invariants_
    beat_discipline``): a pattern that vanished from ``## Patterns`` because it was
    *crystallized out* is NOT lost — it's recoverable from the crystal store (via
    recall / re-warm), so its departure can never be catastrophic identity loss; a
    pattern that vanished because the wrap *recency-trapped* the section IS lost. The
    crystal store is the un-fakeable anchor: a recency-trapped pattern is NOT in the
    store, so it earns zero credit and (via the ``(prior - credit)`` gate formula)
    is still gated independently. **Provenance by recoverability, not by date** — we
    deliberately do NOT gate on ``crystallized_on == today``: a date is a coarse,
    spoofable proxy (a stale same-day row, or a re-crystallized-after-re-warm pattern
    whose origin date is old) and the recoverability invariant is the real safety
    property. Any pattern currently LIVE in the store whose defining line left
    ``## Patterns`` is recoverable, full stop.

    Returns a ``{role: credited_chars}`` map (only the ``graduating`` role can earn
    credit — crystallization is a graduating-section concept). Empty (gate behaves
    exactly as pre-AM-CRYSTAL) when there's no crystal store, no prior text, or
    nothing departed. A corrupt crystal store earns no credit (gate stays strict —
    a crystal-store fault must never WEAKEN the gate; ``active()`` also filters out a
    drifting ``status != "crystallized"`` row).

    ACCOUNTING (the safety-critical part — credit must never EXCEED genuinely
    departed mass): credit is keyed on each prior line's OWNER pattern — the
    ``name | Nx`` the immune system parses via :data:`_NAMED_PATTERN_RE` (the same
    records graduation validation accepts) — NOT any name merely MENTIONED in the
    line. So a line counts at most ONCE (one owner per line), a ``[[sibling]]``
    cross-reference to a departed pattern does NOT credit the referencing line (its
    owner stayed), and a substring name can't leak credit (the regex binds the owner
    token exactly). A line is credited iff its owner is a live crystallized pattern
    AND its name no longer appears as a graduation marker anywhere in the new working
    set — a deliberately-LOOSE all-markers scan (:data:`_ANY_GRADUATION_MARKER_RE`),
    so a survivor that dodges the anchored first-marker regex (a column-0 bare line
    or a merged second marker) is still seen as present and NOT credited;
    over-detecting survivors under-credits, the safe bias."""
    if crystal_store is None or not prior_text:
        return {}
    # Route active() through _crystal_active_safe so the credit path inherits the SAME
    # (CrystalError, OSError) fault barrier + degrade-but-warn as the package build —
    # a crystal fault yields no-credit (the gate stays strict), never breaks the save.
    # Was a CrystalError-only catch that let an OSError escape and abort the wrap
    # (codex L3, 2026-06-06).
    crystallized_names = {
        c.get("name")
        for c in _crystal_active_safe(crystal_store)  # status == "crystallized" only
        if isinstance(c.get("name"), str)
    }
    if not crystallized_names:
        return {}

    prior_body = _role_section_body(prior_text, schema, "graduating")
    new_body_text = "\n".join(_role_section_body(new_text, schema, "graduating"))
    # Names still PRESENT as a marker anywhere in the new working set. GENEROUS
    # (loose all-markers) scan, not the anchored first-marker regex: a survivor that
    # dodges anchoring (a column-0 bare line, OR a second marker merged onto another
    # pattern's line) is still detected here → NOT credited as departed. Over-detect
    # = under-credit = the gate stays strict. This is the cancel-credit side; the
    # earn-credit side below stays strict-anchored.
    new_present = {m.group(1) for m in _ANY_GRADUATION_MARKER_RE.finditer(new_body_text)}

    credit = 0
    for line in prior_body:
        m = _NAMED_PATTERN_RE.match(line)
        if m is None:
            continue  # not a pattern-definition line → no owner → never credited
        # A line carrying MORE THAN ONE graduation marker is ambiguous/malformed:
        # crediting its full length on the anchored (first) owner would also credit a
        # SECOND pattern's mass that hitchhiked onto the line — and that second
        # pattern may NOT be recoverable (not in the store), masking a recency-trap of
        # its mass. A well-formed pattern line has exactly one marker; a multi-marker
        # line earns ZERO credit (its mass stays in the protected baseline). Safe bias.
        if len(_ANY_GRADUATION_MARKER_RE.findall(line)) != 1:
            continue
        owner = m.group(1)
        if owner in crystallized_names and owner not in new_present:
            credit += len(line) + 1  # this owner's defining line genuinely departed
    return {"graduating": credit} if credit else {}


def _check_no_catastrophic_shrink(
    prior_text: str | None,
    new_text: str,
    schema: list[SectionSpec],
    *,
    allow_shrink: bool,
    crystallized_credit: dict[str, int] | None = None,
) -> None:
    """Refuse a wrap that collapses a protected-role section or the whole
    neocortex, unless ``allow_shrink`` is set.

    The structural backstop for the felt / identity layers: a recency-trapped
    or stateless-reset wrap silently guts the timeless ``narrative-timeless``
    section and/or the ``graduating`` identity section. This refuses at the save
    boundary (raising :class:`ValueError`, leaving the wrap in progress so the
    agent can re-wrap — identical handling to a structure-validation failure).
    Deliberate diets pass ``allow_shrink=True``.

    ``crystallized_credit`` (AM-CRYSTAL-MIGRATE) maps a protected role → chars that
    DEPARTED to the crystallized store this wrap (computed, and crystal-store-
    grounded by recoverability, by :func:`_crystallization_credit`). The credited
    mass is SUBTRACTED from PRIOR (the protected baseline) before the retain check —
    NOT added to new — because crystallized-out patterns are recoverable from the
    store, so they leave the "must still be here" baseline. The retain floor then
    applies to the non-crystallized remainder, so a wrap crystallizing patterns OUT
    is recognized as a recoverable MOVE while the UN-credited (recency-trapped) loss
    is gated on its own: a near-total collapse can't slip through just because half
    of it was legitimate crystallization. The gate stays ON (no blanket
    ``allow_shrink``). ``None`` ⇒ no credit ⇒ byte-identical to pre-AM-CRYSTAL.

    No-ops when there is no prior continuity (first wrap) or the relevant prior
    side is below :data:`_SHRINK_GATE_MIN_PRIOR_CHARS` (nothing meaningful to
    collapse).
    """
    # Strict override: a safety gate must be fail-closed. Only a literal True
    # bypasses — a stray ``allow_shrink="false"`` / ``1`` from a loosely-typed
    # caller (bridge, script, JSON wrapper) must NOT disable the gate. Mirrors
    # the MCP adapter's ``is True`` coercion at the core boundary.
    if allow_shrink is True:
        return
    if not prior_text or not prior_text.strip():
        return

    credit = crystallized_credit or {}

    # Partnership entities only (see module comment). An entity that declared a
    # narrative-timeless felt section is opting into felt/identity protection;
    # ops entities (no such section) consolidate aggressively + autonomously by
    # design and keep their pre-0.3.5 behavior.
    if not any(s["role"] == "narrative-timeless" for s in schema):
        return

    display_by_lower = {s["heading"].lower(): s["heading"] for s in schema}
    prior_masses = _schema_section_masses(prior_text, schema)
    new_masses = _schema_section_masses(new_text, schema)

    # Group the protected (retain-floored) headings by ROLE. The gate protects
    # the LAYER — a role: the felt layer (narrative-timeless), the identity
    # layer (graduating) — not an individual heading. A schema may split a
    # protected layer across several headings; each could sit under the
    # per-section char floor while the LAYER as a whole is large and
    # collapsing. Summing the role's headings catches that split-collapse. For
    # a single-heading-per-role schema (flow's FLOW_SCHEMA) the aggregate equals
    # the lone heading's mass, so behavior — and the offender message naming
    # that heading — is byte-identical to the pre-aggregate check.
    headings_by_role: dict[str, list[str]] = {}
    for spec in schema:
        spec_role = spec["role"]
        if spec_role in _SHRINK_RETAIN_FRACTION:
            headings_by_role.setdefault(spec_role, []).append(
                spec["heading"].lower()
            )

    offenders: list[str] = []
    for role, heading_lowers in headings_by_role.items():
        retain = _SHRINK_RETAIN_FRACTION[role]
        prior_mass = sum(prior_masses.get(h, 0) for h in heading_lowers)
        # Crystallized-out patterns are RECOVERABLE (in the store), so they leave the
        # PROTECTED baseline: the retain floor applies to what should still be HERE
        # (the non-crystallized mass), not the original total. Subtract credit from
        # prior — NOT add to new — so the UN-credited (recency-trapped) loss is gated
        # on its own and a near-total collapse can't slip through merely because half
        # of it was legitimate crystallization. Credit is bounded to [0, prior_mass].
        role_credit = min(max(credit.get(role, 0), 0), prior_mass)
        effective_prior = prior_mass - role_credit
        if effective_prior < _SHRINK_GATE_MIN_PRIOR_CHARS:
            continue  # the non-crystallized remainder is sub-floor — nothing meaningful to collapse
        new_mass = sum(new_masses.get(h, 0) for h in heading_lowers)
        if new_mass < effective_prior * retain:
            pct = round(100 * (1 - new_mass / effective_prior))
            label = " + ".join(f"'{display_by_lower[h]}'" for h in heading_lowers)
            credited = (
                f" ({role_credit} chars crystallized out → recoverable; "
                f"{effective_prior} should remain)"
                if role_credit else ""
            )
            offenders.append(
                f"  - {label} ({role}): "
                f"{prior_mass} -> {new_mass} chars{credited} "
                f"({pct}% smaller vs the non-crystallized baseline; "
                f"must retain >={int(retain * 100)}%)"
            )

    # Whole-document backstop — computed EXCLUDING the graduating section. The
    # graduating layer is already per-role checked at the stronger 50% floor, and it
    # is the crystallization site; removing it from both sides makes crystallization
    # NEUTRAL to this backstop (no fungible "credit" that could offset a recency-trap
    # of an unprotected section like Decisions/Context elsewhere in the doc). So this
    # backstop protects the non-graduating remainder at its design-chosen 25% — the
    # same with or without crystallization. (No credit term here: graduating is gone.)
    grad_lowers = headings_by_role.get("graduating", [])
    grad_prior = sum(prior_masses.get(h, 0) for h in grad_lowers)
    grad_new = sum(new_masses.get(h, 0) for h in grad_lowers)
    nongrad_prior = len(prior_text) - grad_prior
    nongrad_new = len(new_text) - grad_new
    if (
        nongrad_prior >= _SHRINK_GATE_MIN_PRIOR_CHARS
        and nongrad_new < nongrad_prior * _DOC_SHRINK_RETAIN_FRACTION
    ):
        pct = round(100 * (1 - nongrad_new / nongrad_prior))
        offenders.append(
            f"  - whole continuity (excl. graduating): {nongrad_prior} -> "
            f"{nongrad_new} chars ({pct}% smaller; must retain "
            f">={int(_DOC_SHRINK_RETAIN_FRACTION * 100)}%)"
        )

    if not offenders:
        return

    raise ValueError(
        "Refusing to save: this wrap collapses protected memory layer(s) — "
        "almost always a recency-trap or stateless-reset failure (the latest "
        "session compressed over the accumulated identity), not a deliberate "
        "edit:\n"
        + "\n".join(offenders)
        + "\n\nThe narrative-timeless (felt) and graduating (identity) layers "
        "carry the continuity that makes you yourself across sessions — they "
        "evolve, they do not reset. Re-wrap preserving them: carry the prior "
        "content forward and update it, auditing proportions against the FULL "
        "arc of the work, not just this session. If this shrink is genuinely "
        "intended (a deliberate diet / migration recompression), pass "
        'allow_shrink=True (CLI: --allow-shrink; MCP: "allow_shrink": true).'
    )


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


def _crystal_active_safe(crystal_store: CrystalStore | None) -> list:
    """The live crystallized corpus, or ``[]`` — a crystal-store fault must NEVER
    break a wrap (the wrap pipeline degrades to no-crystal behavior, not failure).

    Catches BOTH ``CrystalError`` (corrupt/invalid store, schema-too-new) AND raw
    ``OSError`` (PermissionError, disk I/O, IsADirectoryError). ``CrystalStore._load``
    re-raises malformed JSON as ``CrystalError`` and returns the empty shape on
    ``FileNotFoundError``, but lets an ordinary ``OSError`` escape — which would
    otherwise abort the wrap through every read site and violate the invariant above
    (codex L3, 2026-06-06). Degrade-but-SURFACE: the wrap drops the crystallized tier
    yet emits a ``UserWarning`` so the fault keeps a diagnostic (a guard must not
    silently mask the root cause it protects against). The crystal tier is ADDITIVE —
    losing it for one wrap is safe; breaking the wrap is not."""
    if crystal_store is None:
        return []
    try:
        return crystal_store.active()
    except (CrystalError, OSError) as exc:
        warnings.warn(
            f"crystal store unreadable ({type(exc).__name__}: {exc}); this wrap "
            f"proceeds WITHOUT the crystallized tier (contradiction/dedup scans and "
            f"shrink-credit degrade to no-crystal). Inspect the store by hand.",
            UserWarning,
            stacklevel=2,
        )
        return []


def _build_wrap_package(
    episodes: list[Episode],
    existing_continuity: str | None,
    project_name: str,
    *,
    max_chars: int | None = None,
    today: str | None = None,
    staleness_days: int = 7,
    schema: list[SectionSpec] | None = None,
    crystal_store: CrystalStore | None = None,
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
        max_chars: Maximum size of the continuity file. ``None`` derives a
            schema-aware default (see :func:`~anneal_memory.schema.default_max_chars`).
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
    if max_chars is None:
        # AM-SCHEMA-BUDGET: schema-aware default. DEFAULT_SCHEMA -> 20000
        # (byte-compatible); a richer schema (FLOW_SCHEMA) gets headroom for its
        # incompressible felt/structural sections. Single resolution point —
        # feeds both _build_wrap_instructions and the returned package.
        max_chars = default_max_chars(schema)

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

    # AM-CONTRASCAN-EMIT (v0.4.3): compute the existing-Proven list ONCE here
    # (single source of truth) — it feeds BOTH the contradiction-scan
    # instruction emitted inside _build_wrap_instructions AND prepare_wrap's
    # uncovered_proven_to_check (read back from the returned package). One
    # computation site means the discipline and its data cannot drift.
    from .graduation import extract_proven_patterns
    grad_headings = graduating_headings(schema)
    uncovered_proven = (
        extract_proven_patterns(
            existing_continuity,
            graduating_headings=grad_headings,
        )
        if existing_continuity
        else []
    )

    # AM-SEMDUP (v0.5.0): the existing graduated corpus (name + level + a
    # one-line meaning) over ALL named levels (min_level=1) — a fresh-vocab
    # duplicate most dangerously enters as a NEW 1x under a new name. Computed
    # once here so the dedup-scan block in _build_wrap_instructions and any
    # downstream inspection derive from one extraction.
    pattern_summaries = (
        extract_pattern_summaries(
            existing_continuity,
            graduating_headings=grad_headings,
        )
        if existing_continuity
        else []
    )

    # AM-CRYSTAL-MIGRATE: the crystallized tier is the 2nd surfacing point. The
    # bulk of Proven wisdom lives in the crystal store (OUT of ## Patterns), so the
    # contradiction + dedup scans must ALSO scan it — else a wrap silently re-forks
    # or contradicts a pattern it can no longer see. Extend both corpora with the
    # live crystal patterns (dedup by name; the crystal level wins a tie since a
    # crystallized pattern is the canonical home once it has left the working set).
    crystallization_candidates: list[StalePatternDict] = []
    rewarm_candidates: list[str] = []
    crystal_active = _crystal_active_safe(crystal_store)
    if crystal_active:
        # Route level coercion through CrystalStore._safe_level so a hand-edited /
        # migrated non-numeric row level can't crash the wrap (the crystal-fault-
        # never-breaks-a-wrap invariant; bool-safe — _safe_level rejects nothing but
        # never raises).
        _proven_seen = set(uncovered_proven)
        for c in crystal_active:
            name = c.get("name")
            if (isinstance(name, str) and CrystalStore._safe_level(c.get("level")) >= 2
                    and name not in _proven_seen):
                uncovered_proven.append(name)
                _proven_seen.add(name)
        _summary_names = {s.name for s in pattern_summaries}
        for c in crystal_active:
            name = c.get("name")
            if isinstance(name, str) and name not in _summary_names:
                pattern_summaries.append(
                    PatternSummary(name, CrystalStore._safe_level(c.get("level")),
                                   str(c.get("explanation", ""))[:120])
                )
                _summary_names.add(name)
        pattern_summaries.sort(key=lambda r: (-r.level, r.name))
        # Re-warm candidates: hot crystallized patterns the working set should
        # re-cache (propose-not-auto — the composer decides what returns to ## Patterns).
        if crystal_store is not None:
            try:
                today_date = date.fromisoformat(today)
                rewarm_candidates = [
                    str(c["name"])
                    for c in crystal_store.surface_rewarm_candidates(today=today_date)
                ]
            except (CrystalError, ValueError, OSError):
                # OSError added (codex L3, 2026-06-06): same crystal-fault-never-breaks
                # -a-wrap invariant as _crystal_active_safe. rewarm is a cosmetic propose
                # -not-auto hint, so it degrades silently (the load-bearing active()/credit
                # paths warn via _crystal_active_safe; a lost hint needs no alarm).
                rewarm_candidates = []

    # Crystallization candidates: cold-Proven patterns in ## Patterns ready to route
    # OUT (constitution / crystallize / compost — composer-judged). The cold signal
    # is the staleness flag; the Proven (2x/3x) filter is the high-water proxy. Only
    # surfaced when a crystal store is present (no store ⇒ no crystallization tier ⇒
    # byte-identical pre-AM-CRYSTAL package, consistent with the gate's None path).
    if crystal_store is not None:
        crystallization_candidates = [s for s in stale_patterns if s["level"] >= 2]

    # Build instructions (the contradiction-scan + dedup-scan blocks are emitted
    # when there is a graduating section + the relevant existing patterns).
    instructions = _build_wrap_instructions(
        project_name, max_chars, today, schema, uncovered_proven, pattern_summaries,
        crystallization_candidates=crystallization_candidates,
        rewarm_candidates=rewarm_candidates,
    )

    return WrapPackageDict(
        episodes=formatted_episodes,
        episode_count=len(episodes),
        continuity=existing_continuity,
        stale_patterns=stale_patterns,
        uncovered_proven=uncovered_proven,
        crystallization_candidates=crystallization_candidates,
        rewarm_candidates=rewarm_candidates,
        instructions=instructions,
        today=today,
        max_chars=max_chars,
    )


def _build_wrap_instructions(
    project_name: str,
    max_chars: int | None,
    today: str,
    schema: list[SectionSpec] | None = None,
    uncovered_proven: list[str] | None = None,
    pattern_summaries: list[PatternSummary] | None = None,
    *,
    crystallization_candidates: list[StalePatternDict] | None = None,
    rewarm_candidates: list[str] | None = None,
) -> str:
    """Build the compression instructions the agent receives via prepare_wrap.

    Agent-facing instructions, generated from the section schema (v0.3.4). For
    :data:`~anneal_memory.schema.DEFAULT_SCHEMA` this reproduces the historical
    four-section guidance; the richer ``narrative`` / ``narrative-timeless``
    roles inherit the Protocol-Memory compression detail — the gradient
    structure, the named failure modes (Recency / Compression / Stateless-Reset),
    and the implementation-claims guardrail — a quality win for every entity
    with a narrative section, not just partnership entities.

    ``uncovered_proven`` (AM-CONTRASCAN-EMIT, v0.4.3): the existing Proven
    pattern names to scan against. When non-empty AND the schema has a
    graduating section, a contradiction-scan instruction block is emitted
    inline with the list so the methodology-layer discipline travels WITH the
    package rather than living in a separate protocol doc an entity can retire.
    Defaults to ``None`` (no block) so direct callers stay backward-compatible.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA
    if max_chars is None:
        # AM-SCHEMA-BUDGET: resolve here too for direct callers (the prepare_wrap
        # path resolves in _build_wrap_package and passes a concrete int).
        max_chars = default_max_chars(schema)
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
    # AM-CONTRASCAN-EMIT (v0.4.3): emit the methodology-layer contradiction-
    # scan instruction inline with the existing-Proven list, so the discipline
    # travels WITH the package (not in a separate doc an entity can retire).
    # Only when there's a graduating section AND prior Proven to scan against —
    # a first wrap (or a store with no Proven yet) has nothing to declare a
    # stance against, so no block is emitted.
    if has_graduating and uncovered_proven:
        parts += [_contradiction_scan_block(uncovered_proven), ""]
    # AM-SEMDUP (v0.5.0): the merge-don't-fork dedup scan — sibling of the
    # contradiction scan. Surfaces the existing graduated corpus (name + a
    # one-line meaning) so the agent can catch a pattern it is about to
    # re-graduate under fresh vocabulary + a new name (which the lexical
    # cross-session immune gate structurally cannot see). Only when there's a
    # graduating section AND existing named patterns to scan against.
    if has_graduating and pattern_summaries:
        parts += [_semantic_dedup_block(pattern_summaries), ""]
    # AM-CRYSTAL-MIGRATE: the crystallization routing block — only when there's a
    # graduating section AND something to route (cold-Proven OUT or hot patterns IN).
    if has_graduating and (crystallization_candidates or rewarm_candidates):
        parts += [
            _crystallization_block(crystallization_candidates, rewarm_candidates), ""
        ]
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
- pattern_name | Nx ({today}) [evidence: <episode_id1>, <episode_id2> "how BOTH episodes validate the pattern"]
```

Required elements:
- Markdown bullet `-` followed by space
- Operator-style `pattern_name` — starts with a letter, contains only letters,
  digits, underscores, dots, hyphens. Examples: `acid_compliance_over_speed`,
  `connection_pooling_is_bottleneck`, `partnership_challenge_at_X_boundary`.
- Graduation marker `| Nx ({today})` where N is 1, 2, or 3
- For 2x and 3x: an `[evidence: ... "explanation"]` tag is REQUIRED (the TAG is
  required, not a particular id count). Include 2+ episode ids when more than one
  genuinely supports the pattern — co-citation is what FORMS the Hebbian link (see
  "Wiring the associative graph" below). A single id is fine when only one episode
  truly applies; do not pad to reach two.

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
- Validates existing 1x → `- pattern_name | 2x ({today}) [evidence: <id1>, <id2> "how both episodes validate"]`
- Validates existing 2x → `- pattern_name | 3x ({today}) [evidence: <id1>, <id2> "how both episodes validate"]`
- Evidence citations REQUIRED for graduations (2x and 3x). Cite 2+ episode 8-char
  IDs that co-support the pattern — co-citation forms the Hebbian link. A single id
  is acceptable ONLY when just one episode genuinely supports the pattern (do not
  pad with unrelated ids to hit two).
- **Preserved (still true, but NOT re-exercised this session): KEEP the pattern's
  EXISTING date and evidence — do NOT re-stamp `({today})`.** The date marks when the
  pattern was last genuinely grounded. Re-stamping today's date on a pattern you only
  carried forward unchanged falsely claims you re-grounded it today, and the immune
  system validates ONLY today-dated lines — so a today-stamped-but-not-re-grounded
  pattern is needlessly demoted (`(ungrounded)` / `(cross-session-overlap)`). Use
  `({today})` ONLY when you genuinely re-exercised the pattern with NEW evidence this
  session; otherwise carry it forward verbatim with its prior date, and it ages out
  naturally at 7 days (below) if never re-grounded.
- Patterns marked `(ungrounded)` need FRESH evidence from THIS session to re-graduate.
- Patterns marked `(cross-session-overlap)` were demoted because today's explanation
  reused too much vocabulary from prior sessions; compose new evidence with
  genuinely distinct words to re-graduate.
- Patterns at 3x: extract the PRINCIPLE, not the surface observations.
- Patterns older than 7 days with no new validation → remove (stale).
- Group related patterns visually with a header line above them if you like —
  grouping is fine and the `- ` bullet is OPTIONAL. Each pattern LINE needs an
  explicit signal — a `- ` bullet, a FlowScript marker, OR indentation under a
  group header — followed by the `operator_name | Nx` shape, to be protected. A
  free-form prose line with no `name | Nx` marker is invisible to the per-name
  immune system (it can be demoted by the citation check yet is never protected
  by the cross-session sycophancy gate or the high-water mark). Header lines
  (e.g. `{{topic: ...}}`) are skipped because they don't carry the `name | Nx`
  shape. One caution: do NOT write a non-pattern note in the `operator_name | Nx`
  shape as an INDENTED line under a pattern — any such line in this section is
  read as a pattern.

### Wiring the associative graph (CO-CITATION — required to form Hebbian links)
A lone single-id graduation wires no DIRECT link: a direct Hebbian link forms between
episodes CO-CITED in ONE evidence tag (same line, 2+ ids). (Two separate single-id
graduation lines CAN still form a weaker SESSION-level pair across the wrap — but
rely on same-line co-citation, not that incidental path, and a single-graduation wrap
forms nothing at all.) Otherwise the graph only decays — every wrap erodes it and
nothing replenishes it, until associative recall (the on-demand pattern surfacing the
whole system depends on) goes dark. So, every wrap:
- **FORM:** co-cite 2+ THIS-session episodes in a graduating pattern's evidence —
  `[evidence: <id1>, <id2> "how BOTH episodes validate the pattern"]`. Cite episodes
  that genuinely co-support the pattern; do not pad with unrelated ids. A single id
  is fine when only one episode truly applies — co-cite when more than one does.
- A wrap that graduates patterns but forms ZERO links (every graduation cited a lone
  id when pairs were available) is under-wired — the immune system flags it
  (AM-LINKGATE). NOTE: strengthening EXISTING links against decay is NOT done by
  re-citing in a wrap — wrapped episodes leave the current-wrap window, so a
  re-citation dead-ids. A use-driven strengthening counterforce (strengthen the
  pairs that get co-retrieved) is a separate arc; do not attempt it from the wrap.

**Example (note the co-citation — two ids per graduation, not one):**
```
- acid_compliance_over_speed | 2x ({today}) [evidence: 4931b6a8, 7c2d1e90 "both the migration post-mortem AND the load-test trace chose PostgreSQL for ACID guarantees"]
- connection_pooling_bottleneck | 1x ({today})
- horizontal_scaling_strategy | 1x ({today})
```

### Decisions (use in ## Decisions)
Use `[decided(rationale: "why", on: "date")] choice` markers.
- Existing decisions still referenced by active State/Patterns → keep
- 3+ related decisions pointing same direction → extract principle to Patterns, archive individuals
- Decisions >30 days old referencing nothing active → remove"""


def _contradiction_scan_block(uncovered_proven: list[str]) -> str:
    """The contradiction-scan instruction emitted into the wrap package.

    AM-CONTRASCAN-EMIT (v0.4.3): the methodology-layer contradiction-scan
    discipline used to live only in an external protocol doc (Levain
    ``WRAP_PROTOCOL.md``). When an entity retired its local copy of that doc
    the discipline silently vanished — even though ``prepare_wrap`` still
    surfaced the ``uncovered_proven_to_check`` DATA. The data shipped without
    the instruction that consumes it: a downstream-consumer-lost-its-input
    ``invisible_infrastructure_failure``. Emitting the instruction here, inline
    with the list, makes the discipline travel WITH the package as agent-facing
    text that every transport renders (``format_wrap_package_text`` puts
    ``instructions`` first); no entity can drop it by editing a doc it no
    longer reads. ``structural_invariants_beat_discipline``.

    The marker contract matches the save-side detector EXACTLY
    (:func:`~anneal_memory.graduation.extract_contradiction_declarations` /
    :func:`~anneal_memory.graduation.detect_proven_without_declaration`):
    ``[contradicts: name_a, name_b]`` or ``[no-contradicts]``. The signal is
    audit-only — a new Proven without a stance is logged for operator review
    (the Diogenes contradiction sweep), not refused at save.
    """
    proven_list = "\n".join(f"- {name}" for name in uncovered_proven)
    return f"""### Contradiction Scan (REQUIRED before graduating any new Proven)

Before you graduate a pattern to 2x or 3x (Proven tier) in this wrap, scan it
against your existing Proven patterns listed below. On the line of EACH pattern
you newly graduate to Proven tier, declare a contradiction-stance:

- `[contradicts: name_a, name_b]` — the new pattern opposes or supersedes one
  or more existing Proven (name them), OR
- `[no-contradicts]` — the new pattern is genuinely orthogonal to all of them.

A new Proven graduation carrying NEITHER marker is logged for operator review
(it is not refused — the immune system flags it for the contradiction sweep to
inspect for semantic opposition).

Existing Proven to scan against:
{proven_list}"""


# AM-SEMDUP (v0.5.0): cap the rendered dedup list. The full set always lives in
# the in-package continuity's graduating section; the cap bounds the agent-facing
# block for a pathologically large pattern set (graduation is ruthless by design,
# so a real store rarely approaches this) and the overflow is announced — never
# silently truncated.
_SEMDUP_CAP = 50


def _semantic_dedup_block(summaries: list[PatternSummary]) -> str:
    """The merge-don't-fork dedup-scan instruction emitted into the wrap package.

    AM-SEMDUP (v0.5.0): the cross-session immune system catches a pattern
    re-cited with overlapping VOCABULARY (citation overlap) and a pattern that
    CONTRADICTS an existing one (the contradiction scan), but NOT the same
    PRINCIPLE re-graduated under FRESH words and a NEW name — a silent duplicate
    that forks the pattern graph (two names for one principle, each accruing half
    the evidence). Fresh vocabulary is, by definition, LOW lexical overlap, so a
    lexical/citation check structurally cannot see it; only the agent's
    (vocabulary-invariant) semantic judgment can. So — like AM-CONTRASCAN-EMIT —
    the library SURFACES the existing graduated corpus (name + level + a one-line
    meaning, so semantic not just name overlap is judgeable) plus the merge
    instruction; the agent decides (the no-LLM-as-judge axiom: the library cannot
    decide semantic identity). ``structural_invariants_beat_discipline``: the
    discipline travels WITH the package — flow carried it as a retired hand rule
    (the WRAP_PROTOCOL pre-wrap pattern recall) that an entity could silently
    drop by editing a doc it no longer reads.

    ``summaries`` are sorted ``(level desc, name)``; capped at
    :data:`_SEMDUP_CAP` with an explicit overflow note.
    """
    shown = summaries[:_SEMDUP_CAP]
    overflow = len(summaries) - len(shown)
    listing = "\n".join(
        f"- {name} ({level}x)" + (f": {summary}" if summary else "")
        for name, level, summary in shown
    )
    overflow_note = (
        f"\n- …plus {overflow} more graduated pattern(s) in your continuity's "
        f"graduating section — scan those too."
        if overflow > 0
        else ""
    )
    return f"""### Pattern Dedup Scan (merge, don't fork — before composing ANY new pattern)

The immune system catches a pattern re-cited with overlapping VOCABULARY and a
pattern that CONTRADICTS an existing one — but it CANNOT catch the same
PRINCIPLE re-graduated under FRESH words and a NEW name. That silent duplicate
forks your pattern graph: two names for one principle, each accruing half the
evidence. Before composing ANY new pattern (any level), scan it against your
existing graduated patterns below:

{listing}{overflow_note}

If your new pattern is the SAME principle as one of these under different words,
MERGE it — re-graduate the EXISTING name with your new evidence — instead of
forking a new name. Fork only when the principle is genuinely distinct."""


def _crystallization_block(
    crystallization_candidates: list[StalePatternDict] | None,
    rewarm_candidates: list[str] | None,
) -> str:
    """The crystallization routing instruction emitted into the wrap package
    (AM-CRYSTAL-MIGRATE). The ``## Patterns`` working set had no OUT path — a
    one-way 3x ratchet that monotonically bloats until an always-loaded list stops
    *working* (attention doesn't scale). This surfaces the two movements across the
    working⇄crystallized membrane so the composer can keep the working set bounded:

      - cold-Proven patterns ready to route OUT — composer-judged 3 ways:
        → CONSTITUTION (a miss CORRUPTS the substrate — keep always-loaded, in the
          harness's bedrock, NOT the on-demand store)
        → CRYSTALLIZE (timeless + just-in-time — the bulk; ``anneal-memory crystal
          crystallize``, retrieved on cue, off the always-loaded budget)
        → COMPOST (phase-specific + cold — drop it; its episodes remain as the
          re-graduation safety net)
      - hot crystallized patterns to consider pulling back IN (the work re-warmed
        their domain — re-add to ``## Patterns`` if currently load-bearing).

    Propose-not-auto: the library SURFACES; the composer (or operator) decides + acts
    (the no-LLM-as-judge axiom — the library cannot judge permanence vs activation-
    mode). The risk gate is non-negotiable: only ever COMPOST a phase-specific
    pattern, NEVER a timeless one (episodic recall is the backstop, but forgetting is
    the dangerous direction)."""
    out_list = ""
    if crystallization_candidates:
        out_list = "\n".join(
            f"- {c['content'].strip()}  (cold {c['days_stale']}d)"
            for c in crystallization_candidates
        )
    in_list = ""
    if rewarm_candidates:
        in_list = "\n".join(f"- {name}" for name in rewarm_candidates)

    parts = ["### Crystallization Routing (keep the working set bounded)", ""]
    if out_list:
        parts += [
            "These Proven patterns have gone COLD in your working set — route each "
            "(propose, don't auto-apply): → CONSTITUTION (catastrophic-if-missed → "
            "the harness's always-loaded bedrock), → CRYSTALLIZE (timeless + "
            "just-in-time → the on-demand crystal store, off the always-loaded "
            "budget), or → COMPOST (phase-specific + cold → drop; episodes remain "
            "for re-graduation). Risk gate: only ever COMPOST a phase-specific "
            "pattern, NEVER a timeless one.",
            "",
            out_list,
            "",
            # AM-CRYSTAL-DECISION-CHANNEL: ask for the decision back in a machine-
            # parseable form so a consumer executes it (parse_crystal_decisions),
            # rather than re-reading free prose. The enum spellings MUST match
            # VALID_ROUTES / VALID_PERMANENCE / VALID_ACTIVATION_MODES exactly.
            "After you decide, record your routing as a fenced `crystal-decisions` "
            "block — one pipe-delimited row per pattern you are routing OUT, so the "
            "decision is executed structurally (not re-parsed from prose):",
            "",
            "```crystal-decisions",
            "name | route | permanence | activation_mode",
            "```",
            "",
            "where `route` ∈ {constitution, crystallize, compost}, `permanence` ∈ "
            "{timeless, phase-specific}, and `activation_mode` ∈ {just-in-time, "
            "catastrophic}. Write each pattern's name PLAINLY — exactly as it reads "
            "in `## Patterns`, with no markdown emphasis — so the row grounds back to "
            "its graduation line. Prerequisite: only `crystallize` a pattern when a "
            "retrieval surface exists (a recall hook or the crystallized index), "
            "else it leaves the always-loaded set with no way back. A `compost` + "
            "`timeless` row is REFUSED (the forget-path is gated structurally) — "
            "re-route it or mark it phase-specific. Omit a pattern to leave it in "
            "the working set; a malformed row is skipped, never fatal.",
            "",
        ]
    if in_list:
        parts += [
            "These crystallized patterns are HOT again (their domain re-warmed) — "
            "consider pulling them back INTO `## Patterns` if currently load-bearing:",
            "",
            in_list,
            "",
        ]
    return "\n".join(parts).rstrip()


def prepare_wrap(
    store: Store,
    *,
    max_chars: int | None = None,
    staleness_days: int = 7,
    crystal_store: CrystalStore | None = None,
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
        max_chars: Maximum target size for the continuity file. ``None``
            (default) derives a schema-aware budget via
            :func:`~anneal_memory.schema.default_max_chars` — 20000 for the
            ops DEFAULT_SCHEMA (byte-compatible), larger for a richer schema
            (e.g. FLOW_SCHEMA's felt/structural sections). An explicit int
            always overrides.
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
          - ``crystallization_candidates`` (list[StalePatternDict]):
            cold-Proven patterns ready to route OUT (constitution /
            crystallize / compost). ``[]`` when no ``crystal_store`` is
            passed, none qualify, or on the empty path.
          - ``rewarm_candidates`` (list[str]): names of HOT crystallized
            patterns to consider re-caching into ``## Patterns``. ``[]``
            without a ``crystal_store`` or on the empty path.

    Args (crystal):
        crystal_store: optional :class:`CrystalStore` (the on-demand
            crystallized tier). When passed, the wrap surfaces the two
            routing lists above + extends the dedup/contradiction scans
            to read the crystal corpus. ``None`` ⇒ byte-identical to the
            pre-AM-CRYSTAL behavior.

    Raises:
        WrapInProgressError: If a wrap is already in progress
            (``wrap_started_at`` set) AND there are real episodes to
            compress. The single-writer guard (AM-PREPARE-GUARD, 0.4.2)
            refuses to clobber the in-flight wrap's token + snapshot.
            Finish the open wrap with :func:`validated_save_continuity`
            or abandon it with :meth:`Store.wrap_cancelled`, then call
            ``prepare_wrap`` again. The empty path (no episodes) does NOT
            raise — it clears a stale/degenerate flag and returns
            ``status == "empty"``, preserving stuck-wrap auto-recovery.

    Note:
        On ``status == "empty"`` the function calls ``wrap_cancelled()``
        on the store to clear any stale in-progress flag. On
        ``status == "ready"`` it calls ``wrap_started(token=...,
        episode_ids=...)`` so the frozen snapshot is persisted in one
        transaction. Either way, the store's wrap lifecycle state is
        consistent after the call. A refused call (WrapInProgressError)
        leaves the in-flight wrap untouched — it never reaches
        ``wrap_started``.
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
            schema_warning=None,
            crystallization_candidates=[],
            rewarm_candidates=[],
        )

    # AM-PREPARE-GUARD (0.4.2): real episodes to compress AND a wrap
    # already in progress = a clobber. The consolidate is single-writer
    # by design; a second prepare_wrap would overwrite the in-flight
    # wrap's token + frozen episode snapshot, stranding the first wrap's
    # compression (saveable only with a token the store no longer holds —
    # the old behavior, where only the save-side CAS caught it after the
    # agent had spent the compression). Refuse structurally so EVERY
    # adapter inherits single-writer safety (this guard used to live only
    # in flow's CLI wrapper; Levain/MCP/CLI callers were unprotected). The
    # check sits AFTER the empty-path above on purpose: an empty wrap
    # window can never strand real episodes, so the empty path keeps its
    # stale-flag auto-recovery (a degenerate empty-snapshot in-progress
    # wrap is cleared, not refused). Recovery from a genuinely stuck wrap
    # with real episodes is validated_save_continuity (finish) or
    # store.wrap_cancelled() (abandon), then prepare_wrap again. The
    # wrap_started() write-point carries the same guard as a structural
    # backstop (and closes the check→write window below for an unlocked
    # concurrent library caller).
    started = store.get_wrap_started_at()
    if started:
        raise WrapInProgressError(started_at=started)

    # All store reads and package construction happen BEFORE wrap_started().
    # If any of them raises, the store is left with no stale wrap-in-progress
    # flag — symmetric with wrap_cancelled() on the empty path.
    existing = store.load_continuity()
    # Read the section schema fail-closed (v0.3.5): a corrupt persisted schema
    # must REFUSE the wrap, not silently degrade a partnership store to ops
    # behavior (which would disable the catastrophic-shrink gate). Read once
    # here; reuse for the package build + graduating-heading extraction below.
    schema = store.section_schema_for_wrap()
    # AM-ROLECHECK (v0.5.0): a VALID-but-mis-roled schema yields a silently
    # thinner package (the immune/pattern format, contradiction scan, felt
    # proportion-check all emit by ROLE) — the v0.3.5 shrink gate only refuses
    # CORRUPT schemas. Warn loudly (UserWarning, mirroring AM-WARN) + surface
    # structurally on the result, so an entity that trusts the generator and
    # dropped its static reference still notices the generator under-delivered.
    schema_warning = schema_role_warning(schema)
    if schema_warning is not None:
        warnings.warn(schema_warning, UserWarning, stacklevel=2)
    package = _build_wrap_package(
        episodes,
        existing,
        store.project_name,
        max_chars=max_chars,
        staleness_days=staleness_days,
        schema=schema,
        crystal_store=crystal_store,
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
    # AM-SCHEMASNAPSHOT: freeze the EXACT schema we read above (line ~884) into
    # the wrap snapshot, so validated_save_continuity reads back this same schema
    # rather than re-reading a possibly-concurrently-changed live schema. Passing
    # the already-read `schema` (not letting wrap_started re-read live) closes the
    # read→wrap_started micro-window airtight.
    store.wrap_started(
        token=wrap_token, episode_ids=episode_ids, section_schema=schema
    )

    # Move #4 library layer (v0.3.2): surface the list of existing
    # Proven (2x/3x) pattern names so the methodology-layer
    # contradiction-scan discipline can require the agent to declare
    # contradiction-stance against each before any new Proven
    # graduation in this wrap. AM-CONTRASCAN-EMIT (v0.4.3): the list is
    # computed once inside _build_wrap_package (which also emits the scan
    # INSTRUCTION from it) — read it back from the package so the data
    # the caller inspects and the instruction the agent reads can never
    # drift apart.
    return PrepareWrapResult(
        status="ready",
        message=f"Ready to compress {len(episodes)} episode(s).",
        episode_count=len(episodes),
        package=package,
        assoc_context=assoc_context,
        wrap_token=wrap_token,
        uncovered_proven_to_check=package["uncovered_proven"],
        schema_warning=schema_warning,
        crystallization_candidates=package["crystallization_candidates"],
        rewarm_candidates=package["rewarm_candidates"],
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
    allow_shrink: bool = False,
    carryforward_cold_days: int | None = 7,
    crystal_store: CrystalStore | None = None,
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
        allow_shrink: Override for the catastrophic-shrink gate
            (v0.3.5). The gate applies only to PARTNERSHIP entities —
            stores whose schema declares a ``narrative-timeless``
            section (e.g. flow's ``FLOW_SCHEMA``); ops entities on the
            default schema are never gated and ignore this flag. For a
            gated entity, a wrap that collapses a protected memory
            layer — the ``narrative-timeless`` (felt) or ``graduating``
            (identity) section below 50% of its prior mass, or the
            whole continuity below 25% — raises ``ValueError`` (a
            recency-trap / stateless-reset wrap silently gutting the
            felt / identity layers). Pass ``True`` only for a
            deliberate diet (a one-time migration recompression that
            intentionally shrinks the neocortex); the override is
            surfaced on the CLI as ``--allow-shrink`` and on the MCP
            ``save_continuity`` tool as ``"allow_shrink": true``.

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
            session already wrapped), a passed ``wrap_token`` does
            not match the in-progress wrap, or the wrap catastrophically
            collapses a protected memory layer and ``allow_shrink`` is
            not set.
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
    # further down (v0.3.4). Read fail-closed (v0.3.5): a corrupt persisted
    # schema must refuse the save rather than silently fall back to the ops
    # DEFAULT_SCHEMA and disable the catastrophic-shrink gate below.
    section_schema = store.section_schema_for_wrap()
    grad_headings = graduating_headings(section_schema)
    # Reject ambiguous merged headings (e.g. "## Patterns and Understanding")
    # with a clear message before the generic all-sections check: one header
    # satisfying two required sections would route a single body into two
    # protected roles and defeat the shrink gate (v0.3.5). Each section needs
    # its own '## ' header line.
    _required_lower = {h.lower() for h in required_headings(section_schema)}
    for _line in text.split("\n"):
        if _line.startswith("## "):
            _matched = _matching_required_headings(_line.lower(), _required_lower)
            if len(_matched) > 1:
                raise ValueError(
                    f"Ambiguous section heading {_line.strip()!r} satisfies "
                    f"multiple schema sections ({', '.join(sorted(_matched))}). "
                    "Give each section its own '## ' header so the felt / "
                    "identity layers stay distinct."
                )
    if not validate_structure(text, section_schema):
        required_str = ", ".join(
            f"## {h}" for h in required_headings(section_schema)
        )
        raise ValueError(f"Continuity must contain all sections: {required_str}")

    # Catastrophic-shrink gate (v0.3.5). Load the prior continuity ONCE here
    # and reuse it for the silent-omission audit further down. The gate runs
    # before the episode fetch + graduation so a collapsing wrap fails fast;
    # raising ValueError leaves the wrap in progress (same as the
    # structure-validation failure above), so the agent re-wraps with the
    # felt/identity layers preserved (or passes allow_shrink for a deliberate
    # diet) without losing the prepared wrap.
    prior_continuity = store.load_continuity()
    # AM-CRYSTAL-MIGRATE: credit chars that crystallized OUT of the graduating
    # section this wrap (crystal-store-grounded, by recoverability not date), so the
    # gate reads a crystallization as a recoverable MOVE — the (prior - credit) gate
    # formula then gates the UN-credited (recency-trapped) loss independently.
    crystallized_credit = _crystallization_credit(
        prior_continuity, text, section_schema, crystal_store
    )
    _check_no_catastrophic_shrink(
        prior_continuity, text, section_schema, allow_shrink=allow_shrink,
        crystallized_credit=crystallized_credit,
    )

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
        # AM-CARRYFORWARD (v0.4.6): hold a load-bearing pattern at its level
        # instead of ratcheting it down when THIS wrap's citation fails to
        # resolve, IF it is at/below its earned high-water mark and was grounded
        # within carryforward_cold_days (warm). Ungrounded path only; the
        # cross-session immune demotion is untouched. None disables it.
        carryforward_cold_days=carryforward_cold_days,
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
    prior_text_for_omission_audit = prior_continuity or ""
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
                # AM-PERNAME-LINEBIND (v0.4.6): capture the name AND its
                # evidence tag in ONE anchored match, so the level/date/
                # explanation are guaranteed to belong to the SAME marker as
                # the name. The pre-0.4.6 path matched the name with
                # _NAMED_PATTERN_RE.match (anchored, first marker) and the
                # evidence separately with _PATTERN_LINE_WITH_EVIDENCE_RE.search
                # (UNANCHORED) — on a malformed line carrying two
                # ``name | Nx [evidence:]`` markers whose first marker had been
                # demoted (evidence tag stripped), the unanchored search bound
                # the SECOND marker's evidence to the FIRST marker's name,
                # polluting pattern_history. The combined regex matches the
                # any-level evidence form (not the 2x/3x-only GRADUATION_RE) so
                # 1x mentions with explanations still anchor cross-session
                # history (the 1x → 2x first-graduation step Phase 1b probe #1
                # exploits). A line with no ``[evidence:]`` simply doesn't match
                # (1x without explanation, or a demoted line) — nothing to
                # anchor against — preserving the prior skip.
                ev_match = _NAMED_PATTERN_WITH_EVIDENCE_RE.match(line)
                if ev_match is None:
                    continue
                # Today-only gate (Codex MEDIUM v0.3.2): only upsert
                # for lines authored this wrap. Carried-forward lines
                # with non-today dates are skipped to keep the
                # cross-session corpus authoritative.
                line_date = ev_match.group(3)
                if line_date != today_str:
                    continue
                explanation = ev_match.group(5)
                if not explanation:
                    continue
                try:
                    pattern_level = int(ev_match.group(2))
                except ValueError:
                    continue
                store.upsert_pattern_history(
                    pattern_name=ev_match.group(1),
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
    # AM-CARRYFORWARD (v0.4.6): patterns HELD at their level this wrap instead
    # of demoted (at/below their earned high-water mark and warm). Surfaced as
    # an audit signal so operators/flow can see what the domain-blind demoter
    # would otherwise have eroded.
    carried_forward_payload: list[dict[str, Any]] = [
        {
            "name": cf.name,
            "held_level": cf.held_level,
            "max_level_reached": cf.max_level_reached,
            "days_since_grounded": cf.days_since_grounded,
            # AM-PRESERVE-BARE-PATH (v0.5.0): True = a cited line whose citation
            # failed to resolve (v0.4.6 path); False = a bare preservation with
            # no citation (v0.5.0 path). Lets operators reconcile AM-WARN's
            # cited_graduations count against the held set.
            "cited": cf.cited,
        }
        for cf in grad_result.carried_forward
    ]

    # AM-WARN (v0.4.2) + AM-LINKGATE (v0.8.3): detect the dead-Hebbian-graph
    # mis-wire. THREE signals — but they are NOT all the same kind. (A) and (B)
    # are STRUCTURAL and false-positive-free: they fire only on a genuine
    # mis-wire (citations that resolve to nothing / a write path that drops
    # available pairs). (C) is a DISCIPLINE REMINDER, not a structural alarm — it
    # has a benign case (a wrap whose graduations each had only ONE genuinely
    # relevant episode), so it is worded as a nudge, never as a proven defect, and
    # must not push toward padding. A wrap with NO graduations at all (a pure
    # state/narrative wrap) stays silent on all three.
    #   (A) graduated patterns carried evidence citations but NONE resolved to an
    #       episode in this store (e.g. ids minted in another namespace) -> the
    #       graph cannot form and stays dead. This is the
    #       invisible_infrastructure_failure that ran silent for ~10 wraps.
    #   (B) co-citation pairs WERE available but nothing formed or strengthened
    #       -> the association write path itself is mis-wired.
    #   (C) AM-LINKGATE: graduations validated and their citations resolved, but
    #       NO graduation offered a co-citation pair -> 0 links form and the graph
    #       only decays. v0.4.2 deliberately excused this as "nothing to co-cite =
    #       healthy"; that excusal HID the dominant under-wiring habit (single-id
    #       graduation is the minimum that passes the format yet wires nothing).
    #       But single-id is ALSO correct when only one episode truly supports the
    #       pattern — so (C) REMINDS the agent to co-cite 2+ when more than one
    #       episode genuinely applies; it must NOT push toward padding with
    #       unrelated ids (the anti-pattern the wrap instructions forbid).
    #       NB (AM-LINKGATE-DECAY, separate arc): (C) catches under-wiring DURING
    #       graduations; it does NOT address decay BETWEEN them. Links form only on
    #       graduation, but decay runs every wrap, so a sparse-graduation stretch
    #       erodes the graph even with perfect co-citation. The use-driven
    #       strengthening counterforce is owned by that arc, not by this signal.
    association_warning: str | None = None
    # AM-CARRYFORWARD (v0.4.6) interaction: a CITED carried-forward line is a
    # graduation that carried a citation which failed to resolve this wrap —
    # held instead of demoted. It MUST count toward cited_graduations or
    # carryforward would silently MASK AM-WARN Signal A: flow's real
    # wrong-namespace bug (all citations resolve to zero episodes) demotes
    # pre-0.4.6 → cited_graduations > 0 → the namespace alarm fires; with
    # carryforward those same warm at-peak lines are HELD → demoted drops to 0,
    # and without this term the alarm would go silent (re-creating the very
    # invisible_infrastructure_failure AM-WARN exists to catch). Carryforward
    # protects the LEVEL; AM-WARN must still surface the root-cause namespace
    # mis-wire (AM-IDALIAS territory). any_citation_resolved already accounts
    # for held lines whose ids DID resolve, so a healthy held line stays silent.
    #
    # AM-PRESERVE-BARE-PATH (v0.5.0) interaction — the MIRROR-IMAGE hazard: a
    # BARE carried-forward line carried NO citation at all (a preserved Proven
    # re-stamped to today without re-grounding). Counting it would FABRICATE a
    # "citation resolved to zero episodes" alarm on a wrap that has no citations
    # to diagnose — protection-creating-a-false-diagnostic, the inverse of the
    # masking above. So count ONLY ``cited`` carries here. This does NOT re-open
    # the masking hole: a real namespace bug still demotes/holds its CITED lines,
    # whose count drives the alarm; the bare carries were never part of that
    # signal (no citation = nothing to resolve to zero).
    cited_carried = sum(1 for cf in grad_result.carried_forward if cf.cited)
    cited_graduations = (
        grad_result.validated + grad_result.demoted + cited_carried
    )
    # Read the GATE-INDEPENDENT resolution signal, NOT any(all_validated_ids):
    # all_validated_ids is suppressed on a cross-session-overlap demote (the
    # immune gate firing on the EXPLANATION, not the ids), so a healthy
    # immune-gate demotion would otherwise misfire Signal A as a dead-namespace
    # graph. any_citation_resolved is True iff some graduation cited a real
    # store episode this wrap, regardless of grounding or cross-session status.
    resolved_any = grad_result.any_citation_resolved
    # Signal B must consider the SAME co-citation set the association
    # pipeline actually attempts to record (direct pairs + cross-line
    # SESSION pairs) — see process_wrap_associations, which forms both via
    # extract_session_co_citations(all_validated_ids). The pre-0.4.2 check
    # `any(len(s) >= 2 ...)` only saw same-line multi-id sets, so a mis-wire
    # that manifested ONLY in cross-line session pairs (two lines each
    # citing one different real episode) was invisible to Signal B. Mirror
    # the pipeline exactly so "available but 0 formed/strengthened" catches
    # that path too. (codex L3 MEDIUM, 0.4.2.)
    from .graduation import extract_session_co_citations
    session_pairs = extract_session_co_citations(grad_result.all_validated_ids)
    cocitation_available = bool(grad_result.direct_co_citations) or bool(session_pairs)
    if cited_graduations > 0 and not resolved_any:
        association_warning = (
            f"{cited_graduations} graduated-pattern citation(s) this wrap resolved "
            f"to ZERO episodes in this store — the Hebbian association graph cannot "
            f"form and will stay dead. Check the citation id namespace (cite this "
            f"store's episode ids, not ids minted elsewhere)."
        )
    elif cocitation_available and assoc_formed == 0 and assoc_strengthened == 0:
        association_warning = (
            "Co-citation pairs were available this wrap but 0 associations formed "
            "or strengthened — the association write path appears mis-wired."
        )
    elif (
        len(episodes) >= 2
        and grad_result.validated > 0
        and resolved_any
        and not cocitation_available
        and assoc_formed == 0
        and assoc_strengthened == 0
    ):
        # Signal C (AM-LINKGATE): graduations validated + citations resolved, but
        # NO graduation offered a co-citation pair, so 0 links formed. Gated on
        # >=2 episodes so a legitimately tiny session (nothing to co-cite WITH) is
        # not nagged. This is USUALLY single-id under-wiring — but a multi-episode
        # wrap whose graduations each had only one genuinely relevant episode is a
        # BENIGN exception, so this is a discipline reminder, not a proven defect.
        # (assoc_formed/assoc_strengthened == 0 in the guard are DEFENSIVE
        # INVARIANTS: with `not cocitation_available` the write path has no pair to
        # form or strengthen, so both are necessarily 0 — kept explicit so the
        # branch reads as "no links happened" even if a future path could feed
        # associations without going through the co-citation set.)
        association_warning = (
            f"AM-LINKGATE: {grad_result.validated} graduation(s) validated this wrap "
            f"but no graduation offered a co-citation pair, so 0 Hebbian links formed. "
            f"A single-id citation validates the pattern yet wires NOTHING; the graph "
            f"then only decays, wrap after wrap, until associative recall goes dark. "
            f"Where more than one this-session episode genuinely supports a graduating "
            f"pattern, co-cite 2+ in its evidence to FORM a link — but do NOT pad with "
            f"unrelated ids; a graduation with a single genuinely-relevant episode is "
            f"fine."
        )
    if association_warning is not None:
        warnings.warn(association_warning, UserWarning, stacklevel=2)

    # AM-CARRYFORWARD (v0.4.6): assisted "graduate OUT or retire" surface.
    # A TOP-tier pattern (max_level_reached >= 3) that needed the hold this
    # wrap is a candidate to either graduate OUT to a stable home (e.g.
    # partnership.md, where a permanent truth lives without the per-wrap
    # citation treadmill) or RETIRE — never silently lost. Emitted as a
    # UserWarning (mirroring AM-WARN) so the signal is loud, not buried in a
    # return field. Lower-tier carries (2x) are held silently — they are the
    # normal domain-blind-erosion-fix case, not a graduate-out decision.
    graduate_out = sorted(
        {cf.name for cf in grad_result.carried_forward if cf.max_level_reached >= 3}
    )
    if graduate_out:
        warnings.warn(
            f"{len(graduate_out)} top-tier (3x) pattern(s) were carried forward "
            f"this wrap (held at level despite an ungrounded citation): "
            f"{', '.join(graduate_out)}. A permanent truth that keeps needing the "
            f"hold is a candidate to graduate OUT to a stable home (e.g. "
            f"partnership.md) or retire — review, don't leave it on the citation "
            f"treadmill.",
            UserWarning,
            stacklevel=2,
        )

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
        carried_forward=carried_forward_payload,
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
        association_warning=association_warning,
        sections=sections,
        # asdict() makes the full return value JSON-serializable
        # top-to-bottom. Library users who want the typed object can
        # do ``WrapResult(**result["wrap_result"])``; everyone else
        # can ``json.dumps(result)`` with no ceremony.
        wrap_result=asdict(wrap_result),
    )
