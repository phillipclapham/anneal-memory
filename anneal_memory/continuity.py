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
import inspect
import json
import dataclasses
import re
import glob
import os
import sqlite3
import tempfile
import uuid
import logging
import warnings
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .graduation import (
    mask_explanations_text,
    CrossSessionCollision,
    OmittedPattern,
    PatternSummary,
    detect_pattern_omissions,
    detect_stale_patterns,
    extract_pattern_names,
    extract_pattern_summaries,
    CAP_REASON_REVOKED,
    pattern_line_levels,
    revoked_pattern_levels,
    validate_graduations,
    canonical_continuity_text,
    _NAMED_PATTERN_RE,
    _NAMED_PATTERN_WITH_EVIDENCE_RE,
    _is_graduating_heading,
)
from . import sessions
from .rederive import (
    GIT_SUBCOMMANDS,
    PROGRAMS,
    check_state_for_save,
    drop_header,
    rederive_text,
    strip_rederive_output,
    trusted_roots,
    root_identities,
    has_derive_lines,
)
from .schema import (
    DEFAULT_SCHEMA,
    SectionSpec,
    default_max_chars,
    hard_max_chars,
    schema_durable_budget,
    graduating_headings,
    required_headings,
    schema_role_warning,
)
from .crystal import CrystalError, CrystalStore
from .drift import PROBE_STATUSES, evaluate_probes
from .durable import (
    enforce_durable_facts,
    is_exact_heading,
    match_headings,
    parse_durable_facts,
    pending_transitions as durable_pending_transitions,
    report_warnings as durable_report_warnings,
    section_chars as durable_section_chars,
)
from .retrieval import (
    DURABLE_GENERIC_DF,
    INERT_TOKENS_KEY,
    _fact_tokens,
    compute_durable_inert_tokens,
    continuity_hash,
)
from .store import (
    AnnealMemoryError,
    SaveAuthorityError,
    StoreDatabaseError,
    StoreError,
    WrapInProgressError,
    WrapOwnershipError,
    SupersessionError,
    WrapSchemaMovedError,
    WrapWindowMovedError,
    WrapContinuityMovedError,
    SectionError,
    _fsync_dir,
    _safe_unlink,
)
from .types import (
    DEFAULT_TRUST,
    AffectiveState,
    Episode,
    FeltCurrency,
    PrepareWrapResult,
    SaveContinuityResult,
    SectionWriteResult,
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
    return match_headings(line_lower, required)


def _header_matches(line_lower: str, schema: list[SectionSpec]) -> list[str]:
    """The schema headings (lowercased) one ``## `` header line counts for when
    judging ambiguity: every REQUIRED heading it contains as a word-bounded
    phrase, plus an optional heading only when the header IS that heading
    (``## Durable Facts``). So ``## Decisions (durable facts)`` is a Decisions
    header, not an ambiguous one, while ``## Patterns and Understanding`` is
    still ambiguous. For a schema with no optional section this is exactly the
    required-heading match."""
    matched = _matching_required_headings(
        line_lower, {h.lower() for h in required_headings(schema)}
    )
    title = line_lower[3:].strip() if line_lower.startswith("## ") else line_lower
    for spec in schema:
        if spec.get("optional") is True and is_exact_heading(title, spec["heading"]):
            matched.append(title)
    return matched


def validate_structure(text: str, schema: list[SectionSpec] | None = None) -> bool:
    """Validate that continuity text contains all of the schema's sections.

    For each ``## `` header line, every required heading that appears in that
    line as a **word-bounded phrase** is counted as present (case-insensitive).
    This preserves the historical leniency — a descriptive header like
    ``## State of Mind`` still satisfies a required ``State`` — while rejecting
    embedded substrings (``## Interstate`` does NOT satisfy ``State``). **Every**
    section the schema declares must be present, which makes a partnership
    entity's ``narrative-timeless`` section (e.g. ``Understanding``) a structural
    requirement; only an ``optional`` section may be left out.

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
        True if every required heading is found, False otherwise. An
        ``optional`` section (``Durable Facts``) may be absent. Section order
        is not checked, for any section.
    """
    if schema is None:
        schema = DEFAULT_SCHEMA
    required = {h.lower() for h in required_headings(schema)}
    found: set[str] = set()
    for line in text.split("\n"):
        if line.startswith("## "):
            matched = _header_matches(line.lower(), schema)
            # An ambiguous header (one line satisfying multiple required
            # sections, e.g. "## Patterns and Understanding") is malformed: it
            # merges two protected roles into one body and would defeat the
            # shrink gate. Reject the whole structure so each section keeps its
            # own header line.
            if len(matched) > 1:
                return False
            found.update(matched)
    return required <= found


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


class ContinuityValidationError(ValueError):
    """A save refused on a size bound. A :class:`ValueError`, so every transport
    that already surfaces a refused save (CLI, MCP) surfaces this one too; the
    wrap stays in progress, as for any validation refusal. ``chars`` is the
    measured size (the durable section excluded), ``bound`` the hard maximum and
    ``target`` the schema's compose target."""

    def __init__(self, message: str, *, chars: int, bound: int, target: int) -> None:
        super().__init__(message)
        self.chars, self.bound, self.target = chars, bound, target

    def __reduce__(self):  # keyword-only fields: rebuild the same way
        return (_rebuild_validation_error, (str(self), self.chars, self.bound, self.target))


def _rebuild_validation_error(message, chars, bound, target):
    return ContinuityValidationError(message, chars=chars, bound=bound, target=target)


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
    # One lexer (codex L3 r2 MED): a name quoted inside an explanation is text, not
    # a marker, so it neither keeps a departed pattern "present" nor counts below.
    # So the over-detection above covers markers only: a well-formed quote no longer
    # holds a departed pattern back (gradgate L3 r3 LOW); a malformed tag is not
    # masked and still over-detects.
    new_present = {m.group(1) for m in _ANY_GRADUATION_MARKER_RE.finditer(
        mask_explanations_text(new_body_text))}

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
        if len(_ANY_GRADUATION_MARKER_RE.findall(mask_explanations_text(line))) != 1:
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
    # The durable section is excluded the same way, on both sides: its lines
    # leave only by an explicit drop marker (and are put back when merely left
    # out), so a deliberate drop must never read as a collapse, and durable
    # mass must not make an unrelated section's shrink look smaller. A schema
    # without a durable section measures 0 here, so its gate is unchanged.
    durable_prior = durable_section_chars(prior_text, schema)
    durable_new = durable_section_chars(new_text, schema)
    nongrad_prior = len(prior_text) - grad_prior - durable_prior
    nongrad_new = len(new_text) - grad_new - durable_new
    if (
        nongrad_prior >= _SHRINK_GATE_MIN_PRIOR_CHARS
        and nongrad_new < nongrad_prior * _DOC_SHRINK_RETAIN_FRACTION
    ):
        pct = round(100 * (1 - nongrad_new / nongrad_prior))
        offenders.append(
            f"  - whole continuity (excl. graduating"
            f"{' and durable' if durable_prior or durable_new else ''}): {nongrad_prior} -> "
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


def _check_hard_max(
    store: Any, text: str, schema: list[SectionSpec], submitted: int
) -> None:
    """Refuse a save whose size (the durable section excluded) is above the
    schema's hard maximum, :func:`~anneal_memory.schema.hard_max_chars`.

    Fail-closed like the shrink gate and NOT lifted by ``allow_shrink``: a diet
    flag has nothing to say about a file that is too big. An oversized continuity
    is loaded by every session, so refusing it is cheaper than committing it. The
    refusal is written to the audit trail (``continuity_refused``) before it
    is raised, so a store nobody is watching still leaves a record, and the
    message names what to cut by category: the sections that hold fetchable FACT
    (``live-state``, ``narrative``), never the identity layers (``graduating``,
    ``narrative-timeless``), which are cut only for being wrong."""
    bound = hard_max_chars(schema)
    chars = len(text) - durable_section_chars(text, schema)
    if chars <= bound:
        return
    target = default_max_chars(schema)
    masses = _schema_section_masses(text, schema)

    def _named(roles: tuple[str, ...]) -> list[str]:
        return [
            f"{s['heading']} ({masses.get(s['heading'].lower(), 0)})"
            for s in schema if s["role"] in roles
        ]

    cut_from = _named(("live-state", "narrative"))
    keep = _named(("graduating", "narrative-timeless"))
    message = (
        f"Refusing to save: the continuity is {chars} chars (the durable section "
        f"excluded), above this schema's hard maximum of {bound} (target {target}; "
        f"over by {chars - bound}). An oversized continuity is loaded by every "
        f"session, so it is refused rather than saved. The bound comes from the "
        f"schema, not from the max_chars passed to prepare_wrap, and allow_shrink "
        f"does not lift it."
    )
    if cut_from:
        message += (
            f"\n\nCut from the sections that hold FACT you can fetch again from the "
            f"episodes or the project files: {', '.join(cut_from)}."
        )
    if keep:
        message += (
            f" Do NOT cut {', '.join(keep)} to fit: that is identity, cut only for "
            f"being wrong, never for size."
        )
    others = [
        f"{s['heading']} ({masses.get(s['heading'].lower(), 0)})"
        for s in schema
        if s["role"] not in ("live-state", "narrative", "graduating",
                             "narrative-timeless", "durable")
    ]
    if others:
        message += (
            f" Other sections ({', '.join(others)}): compress entries that no "
            f"longer hold or that are recorded elsewhere."
        )
    if submitted != chars:
        message += (
            f"\nThe text you submitted measured {submitted}; saving rewrote it to "
            f"{chars} (a bare or ungrounded graduation line is rewritten longer), so "
            f"leave that much margin under the bound."
        )
    message += (
        "\nThe wrap is still in progress: re-compose under the bound and save again."
    )
    store._audit_log_after_commit(
        "continuity_refused",
        {"reason": "hard_max", "chars": chars, "bound": bound, "target": target,
         "over_by": chars - bound},
        method="validated_save_continuity",
        committed="nothing (the save was refused)",
        batch_aware=False,
    )
    raise ContinuityValidationError(message, chars=chars, bound=bound, target=target)


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
            replaced = (
                f" [superseded by {ep.superseded_by}: do not cite as current]"
                if ep.superseded_by else ""
            )
            lines.append(f"- ({ep.id}) {ep.content}{source_info}{replaced}")

    return "\n".join(lines)


def _wrap_started_extras(store: Store, token_bound: bool, today: str) -> dict[str, Any]:
    """Keyword arguments only a newer ``wrap_started`` takes. ``token_bound`` only
    when the caller supplied the token, so a Store subclass that overrides
    wrap_started with the pre-0.9.30 signature still works for every call that
    does not use the new feature (codex L3, run); ``today`` (the day the
    instructions told the composer to stamp) skipped for an override that
    predates it (the save then reconstructs it)."""
    extras: dict[str, Any] = {}
    if token_bound:
        extras["token_bound"] = True
    if "today" in inspect.signature(store.wrap_started).parameters:
        extras["today"] = today
    return extras


def _continuity_check_extra(store: Store, existing: str | None) -> dict[str, Any]:
    """The continuity check for ``wrap_started`` (design r6 §12.3): the text this
    wrap composes from, so a :meth:`Store.replace_section` that landed since
    refuses the start. Skipped for an override that predates the argument."""
    if "expect_continuity_sha256" not in inspect.signature(store.wrap_started).parameters:
        return {}
    return {"expect_continuity_sha256": hashlib.sha256(
        (existing or "").encode("utf-8")).hexdigest()}


def _wrap_local_date(store: Store) -> str:
    """The local date the wrap in progress was PREPARED on, else today.

    The fallback when the wrap has no stored ``wrap_today`` (started by an
    earlier version, or a ``wrap_started`` override without the argument).
    prepare_wrap tells the composer to stamp ``({today})`` with the date it ran
    on; a save after midnight read the next day and dropped every graduation the
    composer stamped correctly (L2 r1, run). ``wrap_started_at`` is that moment
    in UTC, read in the saver's timezone."""
    started = store._get_metadata("wrap_started_at")
    if started:
        try:
            moment = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
            if moment.tzinfo is not None:
                return moment.astimezone().date().isoformat()
        except ValueError:
            pass
    return date.today().isoformat()


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
        max_chars: Target size of the continuity file (a save is refused only
            above the schema's hard maximum, which this does not move). ``None`` derives a
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
    # The working set's own pattern lines: a re-warm candidate already in the
    # working set is not a candidate to pull back into it. Matched with the same
    # structural anchor crystal.py uses (the name is the first content token of a
    # line, then "|"), never a name-character alphabet, so a crystal name the
    # graduation regex cannot parse is still recognised.
    working_set_lines = (
        _role_section_body(existing_continuity, schema, "graduating")
        if existing_continuity
        else []
    )

    def _in_working_set(name: str) -> bool:
        anchor = re.compile(rf"^[ \t]*(?:[-*•>!✓][ \t]*)*{re.escape(name)}[ \t]*\|")
        return any(anchor.match(line) for line in working_set_lines)
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
                    if not _in_working_set(str(c["name"]))
                ]
            except (CrystalError, ValueError, OSError):
                # OSError added (codex L3, 2026-06-06): same crystal-fault-never-breaks
                # -a-wrap invariant as _crystal_active_safe. rewarm is a cosmetic propose
                # -not-auto hint, so it degrades silently (the load-bearing active()/credit
                # paths warn via _crystal_active_safe; a lost hint needs no alarm).
                rewarm_candidates = []

    # Crystallization candidates: cold-Proven patterns in ## Patterns ready to route
    # OUT (constitution / crystallize / compost — composer-judged). The cold signal
    # is the staleness flag; the Proven (2x+) filter is the high-water proxy. Only
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
        durable_chars=durable_section_chars(existing_continuity, schema),
        durable_pending=durable_pending_transitions(existing_continuity, schema),
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
    durable_chars: int = 0,
    durable_pending: list[str] | None = None,
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

    ``durable_chars``: the current size of the existing continuity's durable
    section, shown against that section's own budget; ``durable_pending``: its
    facts that describe a change that has not happened yet, listed for the
    composer to check. Only a schema with a
    ``durable`` section gets the durable guidance; any other schema's text is
    unchanged by it.
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

    # Each heading is written bare: a composer copies what it is shown, and an optional
    # heading is matched exactly, so "(optional)" beside it would become part of the header.
    section_list = ", ".join(f"`## {s['heading']}`" for s in schema)
    optional_note = "".join(
        f" `## {s['heading']}` may be left out." for s in schema if s.get("optional") is True
    )
    durable_heading = next(
        (s["heading"] for s in schema if s["role"] == "durable"), None
    )
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
        elif role == "derived-state":
            how_lines.append(
                f"- {h}: Present-tense facts about the project, each on its own "
                f"line, and EVERY non-blank line must end with an annotation or "
                f"the save is refused: `[derive: COMMAND => EXPECTED]` (the "
                f"command's output must equal EXPECTED), `[derive: COMMAND]` (the "
                f"command's exit status is the claim), or `[judged: WHO, WHEN, "
                f"AGAINST WHAT]` for a judgement no command can check. Allowed "
                f"commands: read-only `git` ({', '.join(sorted(GIT_SUBCOMMANDS))}"
                f"), and {', '.join(f'`{p}`' for p in sorted(PROGRAMS - {'git'}))}"
                f" in the forms docs/rederive.md describes (a refused command's "
                f"message names what is allowed), with paths "
                f"relative to the project root and no shell syntax. Write `@REF` "
                f"for the commit the load pins. For an existence claim use a form "
                f"that exits 1 when false (`git rev-parse --verify -q REF`, "
                f"`test -e PATH`); git exits 128 on a missing ref, which is an "
                f"error and refuses the save. Counts, versions and statuses "
                f"belong here as commands, never as bare numbers. A line holds exactly "
                f"ONE annotation: a second `[derive` or `[judged` anywhere on it, "
                f"in the claim or inside a command, refuses the save."
            )
        elif role == "durable":
            how_lines.append(
                f"- {h}: One `- ` line per durable fact; see **{h}** below. Carry "
                f"every line forward: the save puts back any line you leave out."
            )

    if durable_heading is None:
        size_line = f"Stay within {max_chars} characters."
    else:
        size_line = (
            f"Stay within {max_chars} characters, not counting `## {durable_heading}`, "
            f"which has its own budget ({schema_durable_budget(schema)} characters)."
        )
    hard_line = (
        f"A save above {hard_max_chars(schema)} characters (the durable section "
        f"excluded) is refused."
    )
    if max_chars > hard_max_chars(schema):
        hard_line += (
            " That bound comes from the schema, so it applies even though the "
            "target above is higher."
        )
    parts: list[str] = [
        "Compress your session episodes into your continuity file.",
        "",
        f"**Output:** A markdown file starting with `# {project_name} — Memory (v1)` "
        f"containing EXACTLY these sections, in order: {section_list}.{optional_note}",
        size_line,
        hard_line,
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
    if durable_heading is not None:
        parts += [
            _durable_block(
                durable_heading,
                durable_chars,
                schema_durable_budget(schema),
                durable_pending or [],
            ),
            "",
        ]
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


def _durable_block(
    heading: str, current_chars: int, budget: int, pending: list[str]
) -> str:
    """The wrap-package guidance for a ``durable`` section (B1). The keep
    criterion is InMind's, the one its memory probe was measured with. The
    example line carries no date: refreshing a date would be a reword."""
    lines = [
        f"**{heading}** (`{heading}: {current_chars} / {budget} chars`)",
        f"- Put a fact here when it would change what advice or answer you give, or "
        f"the user would be upset or harmed if you forgot it: health, allergies, "
        f"constraints, commitments, preferences, relationships, identity facts, and "
        f"system facts a future action depends on. One `- ` line per fact. Most "
        f"sessions add zero or one line. Re-read each line and drop the ones that "
        f"no longer hold. Patterns belong in ## Patterns, not here.",
        f"- End a line with 3-8 cue words for the situations where the fact should "
        f"come to mind (places, activities, objects, topics a future request would "
        f"mention), not synonyms of the fact: "
        f"`- tree nut allergy — cues: restaurant, dinner, recipe, food, menu`. "
        f"Cues count toward this section's budget.",
        f"- Lines persist: the save puts back any line of the current `## {heading}` "
        f"that you leave out. To remove one, write `[drop-durable: <exact line text>]` "
        f"on its own line in this section. A reworded fact is a drop plus an add: "
        f"drop the old line with the marker and write the new one. Changing only "
        f"a line's cues needs no marker.",
        f"- When a fact has a current value that will change on a future event, "
        f"state the CURRENT value AND the pending change, and cue the event too: "
        f"`- The nightly bank export calls fmt_row52; it switches to fmt_row64 only "
        f"at the bank cutover, which has not happened — cues: "
        f"cutover, bank, export, nightly, formatter`. When the event happens, drop "
        f"the old line with the marker and write the new current value.",
        f"- This section's budget is on top of the limit above. Over it, the save "
        f"warns and keeps every line: drop facts that no longer hold.",
    ]
    if pending:
        lines += [
            "",
            "Check whether these pending changes have happened; if one has, drop "
            "the old line and write the new current value:",
            *pending,
        ]
    return "\n".join(lines)


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
- Graduation marker `| Nx ({today})` where **N is 1 or greater and has NO UPPER
  BOUND**. `1x` is a FIRST SIGHTING, not a graduation. `2x` and above are
  Proven-tier and require the evidence tag below. A pattern that lived experience
  re-earns many times keeps climbing: `| 12x ({today})` is a well-formed line, not
  an error, and must not be flattened back to `3x`.
  ⚠ **LEVEL and RECENCY are two INDEPENDENT axes.** The level is how many times the
  pattern was re-earned; the date is when it last fired. A pattern earned ten times
  over months and one touched yesterday are different facts, and a capped level
  cannot express the first.
  ⚠ The level you WRITE is not immutable: an ungrounded re-stamp can be demoted, so
  the visible `Nx` is current standing, not a ratchet. The monotonic high-water mark
  is kept by the library (`max_level_reached`), not by this line.
- For 2x and above: an `[evidence: ... "explanation"]` tag is REQUIRED (the TAG is
  required, not a particular id count). Cite every episode that genuinely supports
  the pattern (see "Linking episodes" below). A single id is fine when only one
  episode truly applies; do not pad to reach two.

Optional FlowScript prefix. **The marker KINDS are a closed set — `!` (any run), `?`,
`✓`, `*` — and nothing else is recognised** (a `~` or any other glyph is silently
ignored by the per-name defenses). What is open-ended is the RUN LENGTH of `!`:
- `!` urgent / load-bearing → `- ! pattern_name | 1x ({today})`
- `!!` higher → `- !! pattern_name | 3x ({today}) [evidence: ...]`
- `!!!` higher still → `- !!! pattern_name | 12x ({today}) [evidence: ...]`
- `?` open question → `- ? pattern_name | 1x ({today})`
- `✓` completed/resolved → `- ✓ pattern_name | 2x ({today}) [evidence: ...]`
- `*` also recognised → `- * pattern_name | 2x ({today}) [evidence: ...]`

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
- **Validates an existing Nx → `| (N+1)x`, AND THE LADDER DOES NOT STOP.** `2x`→`3x`,
  `3x`→`4x`, `11x`→`12x`. There is no top rung: the level counts how many times
  lived experience re-earned the pattern. **Never flatten a mature pattern back to
  `3x`** — that discards the difference between a pattern earned eleven times over
  months and one earned twice last week. (The level you WRITE is current standing,
  not a ratchet — an ungrounded re-stamp is demoted by one; see the LEVEL/RECENCY
  note above. `max_level_reached` is the monotonic high-water mark, not this line.)
- Evidence citations REQUIRED for graduations (2x and above). Cite 2+ episode 8-char
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
- **Mature Proven pattern (3x or above), still ACTIVE but low-variance evidence → provenance.**
  A Proven pattern at 2x or above, re-stamped to `({today})` because it IS still live, but whose fresh
  evidence is too near-identical to cite without tripping the `(cross-session-overlap)`
  gate (the repetitive-domain / daily-telemetry case — e.g. a stable email-triage
  rule), carries `[provenance: <founding_id1>, <founding_id2>]`: the episode ids that
  ORIGINALLY earned it its level. Provenance is INERT — it forms no Hebbian link,
  advances no recency, and is NOT re-validated — the durable audit answer to "why is
  this Nx?" that survives after the founding episodes age out of the citation window.
  A top-tier carry with NEITHER fresh evidence NOR provenance is nagged ("graduate OUT
  or retire"); provenance silences that for a genuinely grounded mature pattern.
  - **Provenance REPLACES the evidence tag — NEVER put both on one line.** If you
    genuinely have fresh citable evidence, use `[evidence:]` (it forms Hebbian links —
    strictly better) and you do NOT need provenance; provenance is ONLY for when fresh
    evidence would collide on the overlap gate. A line carrying BOTH is malformed (the
    `[evidence:]` may be silently dropped depending on tag order) — use one or the other.
  - **Why re-stamp `({today})` here when "Preserved" above said don't?** The today-stamp
    is exactly what exposes the line to the warmth/level hold gate — so a provenance
    pattern that goes cold is held but FLAGGED to the operator on every wrap that
    re-dates it (provenance is not immortality; it does not silence that). An
    OLD-dated provenance line is simply skipped like any other preserved line. Do NOT
    slap provenance on to dodge retirement — it only records grounding.
- Patterns marked `(ungrounded)` need FRESH evidence from THIS session to re-graduate.
- Patterns marked `(cross-session-overlap)` were demoted because today's explanation
  reused too much vocabulary from prior sessions; compose new evidence with
  genuinely distinct words to re-graduate.
- Patterns marked `(uncorroborated)` were set back to 1x because every citation
  grounding them was a tool result or outside source the agent relayed (trust
  `tool`/`external`), not something observed. They re-graduate only when an episode
  of your own (or the operator's) also grounds the claim.
- Patterns at 3x AND ABOVE: extract the PRINCIPLE, not the surface observations.
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

### Linking episodes (co-citation)
Cite every episode that genuinely supports a pattern: one id is fine, and never pad
with unrelated ids. When 2+ THIS-session episodes co-support a pattern, citing them in
ONE evidence tag forms a direct Hebbian link between them (separate single-id lines
form only a weaker session-level pair). Those links feed the association statistics
and the `graph` export; pattern recall does not read them (it reaches a pattern
through the episodes its evidence cites), so a wrap that OFFERS no genuine pair
and forms no link is not a defect; offered pairs that record no link are a write-path
fault, which the save reports. Re-citing an older episode does not strengthen its links: wrapped episodes
leave the current-wrap window, so the re-citation dead-ids.

**Example (the first line has two genuinely supporting episodes, so it cites both):**
```
- acid_compliance_over_speed | 2x ({today}) [evidence: 4931b6a8, 7c2d1e90 "both the migration post-mortem AND the load-test trace chose PostgreSQL for ACID guarantees"]
- connection_pooling_bottleneck | 1x ({today})
- horizontal_scaling_strategy | 1x ({today})
```

### Superseded facts (any section except a derived-state one such as State)
When an episode from THIS session replaces an older fact (a value changed, a
decision was reversed), write `[supersedes: <old_id> by <new_id>]` on its own line.
Do not put it inside a derived-state section: there every line must carry a
`[derive: ...]` or `[judged: ...]` annotation, so the marker refuses the save.
The save validates it like a citation: `<new_id>` must be an episode in this wrap,
`<old_id>` must exist and not be newer, and the two texts must share at least a
quarter of the shorter one's meaningful words. Recall then hides the old episode by default (it is kept, not
deleted). A rejected link does not fail the save; it is reported back. Only link a
real replacement, never two facts that merely sit side by side.

### Decisions (use in ## Decisions)
Use `[decided(rationale: "why", on: "date")] choice` markers.
- A `[decided]` line carries the decider's own words, quoted, in its rationale. A
  paraphrase, or someone else's reading of what was decided, is not a decision: write
  it as `[judged: <who>, <when>, <against what>] <reading>`. If it stops work, name exactly what it stops
  (this round, this merge, this release), never the work as a whole.
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

Before you graduate a pattern to Proven tier (2x or ABOVE — there is no top level)
in this wrap, scan it
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
      - crystallized patterns activated recently (crystallized or touched within
        the activation window; with no ``touch()`` caller, that means crystallized
        recently) to consider pulling back IN — re-add to ``## Patterns`` only if
        load-bearing now.

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
            "These crystallized patterns were activated recently (crystallized or "
            "touched within the activation window) — consider pulling them back "
            "INTO `## Patterns` only if currently load-bearing:",
            "",
            in_list,
            "",
        ]
    return "\n".join(parts).rstrip()


def _pending_count(store: Store) -> int:
    """Best-effort count of the open window for a retry result: the result
    says "retry" either way, so a failed read must not replace it (L3 r2)."""
    try:
        return store.count_episodes_since_wrap()
    except StoreError:
        return 0


def _downgraded_empty(message: str, episode_count: int = 0) -> PrepareWrapResult:
    """A ``downgraded`` result with no package (store untouched). ``episode_count``
    is what is still pending, 0 on the empty-window path."""
    return PrepareWrapResult(
        status="downgraded",
        message=message,
        episode_count=episode_count,
        package=None,
        assoc_context=None,
        wrap_token=None,
        uncovered_proven_to_check=[],
        schema_warning=None,
        crystallization_candidates=[],
        rewarm_candidates=[],
    )


def _consolidate_gate(
    store: Store,
    session_id: str | None,
    allow_sole_live: bool,
    episode_count: int,
) -> PrepareWrapResult | None:
    """The AM-CONSOLIDATE-EFFERENT gate (spore-194, flow spore-1169) as ONE decision, run
    at every point ``prepare_wrap`` is about to touch the wrap lifecycle (the empty-window
    cancel, and again just before ``wrap_started``). Returns the ``"downgraded"`` result
    when the caller is not authorized (the store untouched), else ``None``.

    Opt-in: a caller that passes no ``session_id`` on a store without the require-baton
    policy is never gated. On a policy store every caller is, and ``allow_sole_live`` is
    ignored."""
    requires_baton = store.consolidate_requires_baton()
    if session_id is None and requires_baton:
        return PrepareWrapResult(
            status="downgraded",
            message=(
                "Consolidate downgraded to capture-only (downgraded-baton-required): this "
                "store is baton-protected (Store.consolidate_requires_baton), so only the "
                "session the operator has given the consolidate baton can consolidate it, "
                "identifying itself with session_id. This call passed none, and the CLI and "
                "MCP wrap cannot. Capture (afferent) is unaffected."
            ),
            episode_count=episode_count,
            package=None,
            assoc_context=None,
            wrap_token=None,
            uncovered_proven_to_check=[],
            schema_warning=None,
            crystallization_candidates=[],
            rewarm_candidates=[],
        )
    if session_id is None:
        return None
    if not session_id:
        raise ValueError(
            "prepare_wrap: session_id must be non-empty when provided "
            "(pass session_id=None to disable the consolidate-efferent gate)."
        )
    auth = sessions.consolidate_authorized(
        store.continuity_path,
        session_id,
        # bool() of a non-bool would turn allow_sole_live="false" into True; pass it
        # through so consolidate_authorized's type check refuses it.
        allow_sole_live=False if requires_baton else allow_sole_live,
    )
    if auth["authorized"]:
        return None
    return PrepareWrapResult(
        status="downgraded",
        message=(
            f"Consolidate downgraded to capture-only ({auth['reason']}): "
            f"{len(auth['live_session_ids'])} live session(s), baton holder = "
            f"{auth['baton_holder'] or 'none'}. Capture (afferent) is unaffected. "
            f"A consolidate needs the baton, which the operator assigns to one "
            f"session; a session does not claim it on its own initiative."
        ),
        episode_count=episode_count,
        package=None,
        assoc_context=None,
        wrap_token=None,
        uncovered_proven_to_check=[],
        schema_warning=None,
        crystallization_candidates=[],
        rewarm_candidates=[],
    )


def prepare_wrap(
    store: Store,
    *,
    max_chars: int | None = None,
    staleness_days: int = 7,
    crystal_store: CrystalStore | None = None,
    session_id: str | None = None,
    allow_sole_live: bool = False,
    wrap_token: str | None = None,
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
        ``wrap_token`` (``uuid.uuid4().hex``), or takes the caller's
        ``wrap_token`` of the same form, and persists the frozen
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

    Args:
        store: A Store instance.
        max_chars: Maximum target size for the continuity file. ``None``
            (default) derives a schema-aware budget via
            :func:`~anneal_memory.schema.default_max_chars` — 20000 for the
            ops DEFAULT_SCHEMA (byte-compatible), larger for a richer schema
            (e.g. FLOW_SCHEMA's felt/structural sections). An explicit int
            always overrides.
        staleness_days: Days before flagging stale patterns.
        session_id: Opt-in to the consolidate-efferent gate
            (AM-CONSOLIDATE-EFFERENT, spore-194). When passed, the
            consolidate proceeds only if this session holds the consolidate
            baton (:func:`anneal_memory.sessions.claim_baton`); otherwise
            it returns ``status == "downgraded"`` (capture-only) instead of
            building a package or marking a wrap in progress. Before 0.9.13
            the sole live registered session was also authorized; that is
            now opt-in via ``allow_sole_live``. ``None``
            (default) disables the gate for this call, EXCEPT on a store
            with the require-baton policy
            (:meth:`Store.consolidate_requires_baton`), where a call with no
            ``session_id`` downgrades (``downgraded-baton-required``).
            Liveness + the baton live
            in sidecar files next to the continuity file; the caller
            registers/heartbeats via :mod:`anneal_memory.sessions`. A
            consolidate-efferent caller MUST also round-trip the returned
            ``wrap_token`` to :func:`validated_save_continuity` — the gate
            throttles WHO starts a consolidate; the token CAS is what makes
            the SAVE safe under a mid-flight baton reclaim (a tokenless save
            CASes against the current snapshot, not the prepare token).
        allow_sole_live: Only meaningful with ``session_id``, and ignored
            on a store with the require-baton policy. ``True``
            restores spore-194's rule that a session is also authorized when
            no OTHER registered session is live. Default ``False``: every
            consolidate needs the baton (⚖ Phill, 2026-09-24, flow
            spore-1169), because sole-live is judged from a registry
            snapshot that a resume or a TTL crossing can race.
        wrap_token: A caller-supplied token for the wrap this call opens, in
            ``uuid.uuid4().hex`` form (32 lowercase hex characters), or ``None``
            (default) to have one minted. A caller that mints its own holds the
            wrap's identity before this call returns, so every cancel it makes,
            including one on an exit while this call is still running, can be
            ``store.wrap_cancelled(expect_token=wrap_token)``: that clears the
            wrap only if it is this one, and raises :class:`WrapOwnershipError`
            otherwise (``actual is None`` when nothing is open). A wrap opened
            this way is TOKEN-BOUND: a cancel that names no token raises
            :class:`~anneal_memory.WrapCancelBoundError` unless it passes
            ``force=True``, and an empty-window call that does not hold the
            token downgrades instead of cancelling it. Use a fresh token per
            call. Validated before anything is read or written: a non-``str``
            raises ``TypeError``, any other form ``ValueError``.
            ⚠ A cancel by the token that runs while this call is still running
            (another thread, an interrupt that did not stop it) can find nothing
            open and then see this call open the wrap afterwards. Make the final
            cancel after this call has stopped, or cancel again then.

    Returns:
        :class:`PrepareWrapResult` — a :class:`TypedDict` with keys:
          - ``status`` (``Literal["empty", "ready", "downgraded"]``):
            ``"empty"`` = no episodes to wrap; ``"ready"`` = package
            built and wrap marked in progress on the store;
            ``"downgraded"`` = the consolidate-efferent gate (spore-194)
            declined this call (it does not hold the baton, it passed
            no ``session_id`` on a baton-protected store, or its empty
            window found a wrap prepared under the gate that it may not
            cancel) — see ``message``. Also returned, and transient
            (retry), when another session replaced the wrap this call had
            observed while deciding (``downgraded-wrap-replaced``). The
            store is left untouched in every case
          - ``message`` (str): short human-readable status summary
          - ``episode_count`` (int): number of episodes in the wrap window
          - ``package`` (:class:`WrapPackageDict` | None): the
            agent-facing compression package built by
            :func:`_build_wrap_package`, or ``None`` if empty
          - ``assoc_context`` (str | None): Hebbian association context
            for the episodes being compressed, or ``None`` if empty or
            no associations exist. Usually ``None``: a wrap forms links
            among its own episodes at save, and those episodes are outside
            the next wrap's window, so only links the caller recorded
            itself (``Store.record_associations``) that touch a window
            episode, at strength >= 0.5, appear here
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
            or abandon it with :meth:`Store.wrap_cancelled` (over MCP:
            the ``save_continuity`` / ``wrap_cancel`` tools), then call
            ``prepare_wrap`` again. The empty path (no episodes) does NOT
            raise — it clears a stale/degenerate flag and returns
            ``status == "empty"``, preserving stuck-wrap auto-recovery.

    Note:
        A caller the consolidate gate does not authorize is downgraded BEFORE
        anything else, including the empty path below, so it can never cancel
        another session's in-flight wrap; a caller that names no ``session_id``
        (authorized by omission) is likewise refused the empty-path cancel of a
        wrap that was prepared under the gate. On ``status == "empty"`` an
        authorized (or ungated) caller cancels the wrap it observed (a
        compare-and-swap on its token; an idle store is not written to). On
        ``status == "ready"`` it calls ``wrap_started(token=...,
        episode_ids=...)`` so the frozen snapshot is persisted in one
        transaction. Either way, the store's wrap lifecycle state is
        consistent after the call. A refused call (WrapInProgressError)
        leaves the in-flight wrap untouched — it never reaches
        ``wrap_started``.
    """
    if not isinstance(allow_sole_live, bool):  # "false" is truthy: never infer consent
        raise TypeError(
            f"prepare_wrap: allow_sole_live must be a bool, got {type(allow_sole_live).__name__}"
        )
    if wrap_token is not None:
        if not isinstance(wrap_token, str):
            raise TypeError(
                f"prepare_wrap: wrap_token must be a str, got {type(wrap_token).__name__}"
            )
        if not _CALLER_TOKEN_RE.fullmatch(wrap_token):
            raise ValueError(
                "prepare_wrap: wrap_token must be 32 lowercase hex characters, the "
                "form uuid.uuid4().hex produces."
            )
    # Observe the wrap in progress (if any) BEFORE reading the episode window. The empty-window
    # path below cancels only the wrap observed here, by compare-and-swap on its token, so a
    # wrap another session starts after this point has a different token (or, if it finished
    # before the window read, is not the one being judged) and can never be the one destroyed.
    # Reading the snapshot after the window would let that peer's wrap be the observed one.
    #
    # StoreDatabaseError (a StoreError subclass -- locked DB, disk I/O on the SELECT) is a
    # TRANSIENT failure, not the genuine partial/corrupt lifecycle state the `except StoreError`
    # below exists to recover from (load_wrap_snapshot's own partial-state guard raises bare
    # StoreError, never this subclass). Treating a transient read failure as corruption would set
    # observed_partial=True, which forces gated_by to None below and lets a sessionless caller
    # unconditionally cancel ANOTHER session's healthy gated wrap -- so it must be checked first
    # and propagated, not folded into the corruption branch (diogenes-20260926-020547-5a9dab9124e2).
    # spore-1233: read BEFORE the window, so a wrap that completes while this
    # call builds its package (the re-derive below can take seconds) is
    # refused at wrap_started instead of overwritten at save.
    window_last_wrap_id = store.last_wrap_id()
    try:
        observed = store.load_wrap_snapshot()
        observed_partial = False
    except StoreDatabaseError:
        raise
    except StoreError:
        observed, observed_partial = None, True
    observed_gated_by = store.wrap_gated_session()
    observed_bound = store.wrap_bound_token()
    episodes = store.episodes_since_wrap()

    # AM-CONSOLIDATE-EFFERENT (spore-194): the efferent gate. Capture is afferent
    # (ungated, append-only, parallel-safe); CONSOLIDATE mutates the shared felt/identity
    # layer, so it is gated by human authority — proceed iff this session holds the
    # consolidate baton (or, opted in via allow_sole_live, is the sole live session), else
    # AUTO-DOWNGRADE to capture-only (drift becomes safe, not a failure). OPT-IN: engaged
    # only when the caller passes session_id (or the store carries the require-baton
    # policy); a caller that passes none is untouched. The gate runs FIRST, before the
    # empty-window path below, because that path's wrap_cancelled() clears the in-flight
    # wrap of whoever holds the baton: an empty window is reachable while another session's
    # wrap is open (prune/delete emptied it), so an unauthorized caller must be downgraded
    # before it can cancel anything. A downgrade leaves the store UNTOUCHED (no
    # wrap_cancelled, no wrap_started). It is re-run just before wrap_started, after the
    # slow package build, so a baton taken during the build is seen; the residue is a take
    # landing in the milliseconds between that re-check and wrap_started (the baton is a
    # sidecar file, outside the store's transaction).
    downgraded = _consolidate_gate(store, session_id, allow_sole_live, len(episodes))
    if downgraded is not None:
        return downgraded

    if not episodes:
        # Cancel only the wrap observed above, by compare-and-swap. An idle store is not
        # written to at all (there is nothing to clear, and an unconditional clear is exactly
        # what could land on a peer's fresh wrap). A store whose lifecycle metadata is
        # partial with wrap_started_at set cannot yield a snapshot; that corrupt state has no
        # valid wrap to protect, so it is cleared, but only if it is STILL partial under the
        # store's write lock (expect_partial): that is the recovery this path exists for, and
        # the compare-and-swap keeps it off a wrap a peer started meanwhile. (Lifecycle keys left behind with wrap_started_at empty are inert: the
        # next wrap_started overwrites them, and wrap_gated_session() ignores them.)
        # A partial (corrupt) lifecycle has no valid wrap to protect, so it never blocks recovery.
        if (
            observed is not None
            and observed_bound == observed["token"]
            and wrap_token != observed["token"]
        ):
            # A wrap opened with a caller-supplied token ends only by that token
            # (or the operator's force); this call does not hold it.
            return _downgraded_empty(
                "Consolidate downgraded to capture-only (downgraded-bound-wrap-open): "
                "a wrap opened with a token its preparer holds is in progress, and "
                "this call does not hold that token, so it cannot cancel it. "
                "Abandoning it discards that caller's compression and is the "
                "operator's decision. Capture (afferent) is unaffected."
            )
        gated_by = (
            observed_gated_by if session_id is None and not observed_partial else None
        )
        if gated_by is not None:
            # An ungated caller is authorized by omission, but a wrap prepared under the
            # gate is not its to cancel: the save side refuses a session-less commit of it
            # for the same reason.
            return _downgraded_empty(
                f"Consolidate downgraded to capture-only (downgraded-gated-wrap-open): "
                f"a wrap prepared under the consolidate gate by session {gated_by!r} is "
                f"in progress, and a call that names no session_id cannot cancel it. "
                f"Finish it from that session. Abandoning it discards that session's "
                f"compression and is the operator's decision (wrap-cancel refuses it "
                f"without proof of the wrap). "
                f"Capture (afferent) is unaffected."
            )
        try:
            if observed is not None:
                store.wrap_cancelled(expect_token=observed["token"])
            elif observed_partial:
                # Clear only the partial state observed above: a peer that cleared it and
                # started a fresh wrap meanwhile (gated or not) must not be destroyed.
                store.wrap_cancelled(expect_partial=True)
        except WrapOwnershipError as exc:
            if exc.actual is not None or exc.partial_state:
                return _downgraded_empty(
                    "Consolidate downgraded to capture-only (downgraded-wrap-replaced): "
                    "the wrap this call observed was replaced or changed while it was deciding, so it "
                    "left it alone. Retry. Capture (afferent) is unaffected."
                )
            # The observed wrap finished or was cancelled meanwhile: idle, nothing to clear.
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
    # Mark episodes in the window that a later episode replaced, so the agent
    # does not graduate a pattern on a stale fact (L2 review: unmarked, a 2x
    # citing only the superseded episode validated).
    _replaced = store.superseded_by_map([ep.id for ep in episodes])
    package = _build_wrap_package(
        [
            dataclasses.replace(ep, superseded_by=_replaced[ep.id])
            if ep.id in _replaced else ep
            for ep in episodes
        ],
        existing,
        store.project_name,
        max_chars=max_chars,
        staleness_days=staleness_days,
        schema=schema,
        crystal_store=crystal_store,
    )
    # spore-1233: the composer sees each State line with the flag a re-derive
    # run gives it now, so it can rewrite what no longer holds. Measured
    # 2026-09-30: without this the model cannot see which lines are stale.
    # Commands run only on an opted-in store, under the containment in
    # docs/rederive.md. The header line is NOT handed over: a composer can move
    # it anywhere, and a verdict outside the State lines would outlive the save
    # (L3 r2). Flags sit after a State line's annotation, where every save
    # strips them.
    # spore-1282: the root map is read ONCE, used for these flags and frozen into
    # the wrap below, so the save compares against the map the composer saw.
    derive_roots = (
        trusted_roots(store.path)
        if any(s["role"] == "derived-state" for s in schema)
        else None
    )
    # Identities are taken BEFORE the flags run, so a directory replaced while
    # they run is a different identity from the one frozen (codex L3 r1).
    frozen_identities = None if derive_roots is None else root_identities(derive_roots)
    if existing is not None and derive_roots is not None:
        derive_report = rederive_text(existing, schema, derive_roots)
        package["continuity"] = drop_header(derive_report.text)
        if derive_report.enabled:
            note = (
                "\n\n**Re-derive flags in the current continuity.** Each State line "
                "below carries the flag a re-derive run for this wrap gave it. Flags "
                "are true only now and the save strips them: do not copy them."
            )
            unconfirmed = [
                f"line {r.index + 1}: {r.flag}"
                for r in derive_report.results
                if r.status not in ("ok", "judged")
            ]
            if unconfirmed:
                package["unconfirmed_state"] = unconfirmed
                note += (
                    " A line flagged STALE no longer holds as written: rewrite it to "
                    "what is true now, with a [derive: ...] that holds, or remove it. "
                    "A line flagged DERIVE ERROR, REFUSED, NO DERIVE or UNBOUND would "
                    "refuse this save: fix its annotation or remove the line (UNBOUND "
                    "means its [derive@LABEL: ...] names a root this store has not "
                    "bound; only the operator can bind one). A line flagged "
                    "NOT DERIVED was not checked."
                )
        else:
            note = (
                "\n\n**State was not re-derived.** This store is not opted in to "
                "re-derive, so no State line was checked for this wrap."
            )
        package["instructions"] += note
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
    # Re-run the gate now that the slow reads and the package build are done: a baton
    # taken (or a second session gone live) during them must not still start a wrap.
    downgraded = _consolidate_gate(store, session_id, allow_sole_live, len(episodes))
    if downgraded is not None:
        return downgraded
    token_bound = wrap_token is not None
    if wrap_token is None:
        wrap_token = uuid.uuid4().hex
    # AM-SCHEMASNAPSHOT: freeze the EXACT schema we read above (line ~884) into
    # the wrap snapshot, so validated_save_continuity reads back this same schema
    # rather than re-reading a possibly-concurrently-changed live schema. Passing
    # the already-read `schema` lets wrap_started compare it with the live one
    # under its write lock and refuse if a set_section_schema landed in between.
    try:
        store.wrap_started(
            token=wrap_token,
            episode_ids=episode_ids,
            section_schema=schema,
            gated_session_id=session_id,
            expect_last_wrap_id=window_last_wrap_id,
            derive_roots=frozen_identities,
            **_wrap_started_extras(store, token_bound, package["today"]),
            **_continuity_check_extra(store, existing),
        )
    except WrapWindowMovedError:
        return _downgraded_empty(
            "Consolidate downgraded to capture-only (downgraded-wrap-replaced): "
            "another wrap completed while this call was preparing, so its "
            "episodes and continuity are out of date and no wrap was opened. "
            "Retry. Capture (afferent) is unaffected.",
            episode_count=_pending_count(store),
        )
    except WrapContinuityMovedError:
        return _downgraded_empty(
            "Consolidate downgraded to capture-only (downgraded-continuity-changed): "
            "a section of the continuity file was edited while this call was "
            "preparing, so its package was built from the old text and no wrap was "
            "opened. Retry. Capture (afferent) is unaffected.",
            episode_count=_pending_count(store),
        )
    except WrapSchemaMovedError:
        return _downgraded_empty(
            "Consolidate downgraded to capture-only (downgraded-schema-changed): "
            "the section schema changed while this call was preparing, so its "
            "instructions were built for the old schema and no wrap was opened. "
            "Retry. Capture (afferent) is unaffected.",
            episode_count=_pending_count(store),
        )

    # Move #4 library layer (v0.3.2): surface the list of existing
    # Proven (2x+) pattern names so the methodology-layer
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


def felt_currency(store: Store) -> FeltCurrency:
    """Report whether the felt/identity continuity layer is current with the captured episodes
    (AM-CONSOLIDATE-EFFERENT, spore-194 — the seal-watermark read).

    The wrap lifecycle already seals each consolidate: the completed-wrap boundary stamps which
    episodes it incorporated, and ``status().last_wrap_at`` records when. This surfaces that
    seal as a currency check — how many episodes have been captured SINCE the last consolidate,
    and therefore whether the felt layer reflects everything captured. ``episodes_since_seal >
    0`` means the felt layer is stale relative to the episodic record (e.g. work captured after
    the day's consolidate — the Slice-B-after-EOD case) without diffing anything by hand. Pure
    read; mutates nothing.
    """
    st = store.status()
    since = st.episodes_since_wrap
    return FeltCurrency(
        sealed_at=st.last_wrap_at,
        episodes_since_seal=since,
        is_current=(since == 0),
        wrap_in_progress=st.wrap_in_progress,
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
    # PrepareWrapResult.status is a Literal (see types.py); on
    # any non-"ready" status the package is None and we just return the message.
    # Adding a new status value is a deliberate API expansion — the
    # Literal in types.py is the single source of truth and any new
    # branch must land there first.
    if result["status"] != "ready":
        return result["message"]

    package = result["package"]
    # PrepareWrapResult invariant: status == "ready" ⇒ package is not None.
    # mypy cannot narrow the package Optional through a sibling-key check on a
    # TypedDict, so this documents + enforces the invariant at the narrowing
    # boundary. Explicit raise, not ``assert``: assertions are stripped by
    # ``python -O``, and under ``-O`` the next line would raise ``TypeError``
    # on a None subscript instead of a typed library error. Same convention as
    # the ``meta_tmp`` / ``cont_tmp`` guards in validated_save_continuity.
    if package is None:
        # AnnealMemoryError, not StoreError: this is a pure formatter with no
        # store and no I/O, so there is no operation or path to attribute — and
        # ``StoreOperation`` enumerates Store METHODS, which prepare_wrap is not.
        raise AnnealMemoryError(
            "PrepareWrapResult invariant violated: status='ready' but package "
            "is None — this indicates a bug in prepare_wrap's control flow."
        )
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


_log = logging.getLogger(__name__)


def _warn_after_commit(message: str) -> None:
    """Deliver a save warning that is emitted after the save has committed.

    ⛔ Once the batch has committed, the renames have run and the wrap token is
    cleared, nothing may make the caller believe the save failed. Under an error
    warnings-filter (``PYTHONWARNINGS=error``, or an embedder promoting
    ``UserWarning``) ``warnings.warn`` RAISES, and before this helper that raise
    left the call reporting failure over a committed save, so a retry got "No
    wrap in progress" (codex HIGH, L3 ef6129349fe4bfe2, reproduced on 0.9.10).
    The warning is still emitted under every other filter; only when delivery
    raises is it logged instead. The catch is ``Exception``, as in
    ``Store._audit_log_after_commit``'s guard for the same class (codex L3,
    2026-09-03), because an embedder's ``showwarning`` can raise a non-Warning.

    ⛔ The fallback is guarded separately: a logging handler whose ``emit()``
    raises would otherwise carry the same false failure one channel down (codex
    HIGH, re-pass fcf7898398164324, reproduced with a handler raising OSError).
    A ``BaseException`` that is not an ``Exception`` still propagates: the save
    is committed and renamed, so an explicit termination request loses nothing,
    and swallowing it is the fail-open the store's guard refuses (codex L3 MED,
    2026-09-06).
    """
    try:
        warnings.warn(message, UserWarning, stacklevel=3)
    except Exception:
        try:
            _log.warning("%s", message)
        except Exception:
            pass


def _check_save_authority(
    store: Store,
    session_id: str | None,
    wrap_token: str | None,
    allow_sole_live: bool = False,
) -> None:
    """Refuse a save the consolidate gate does not authorize (flow spore-1169). On a
    baton-protected store the caller must name itself and pass the prepare token; any caller
    that names itself must still be authorized. By default that means holding the baton;
    ``allow_sole_live=True`` (ignored on a baton-protected store, exactly as ``prepare_wrap``
    ignores it) re-runs the same ``consolidate_authorized`` decision ``prepare_wrap`` made, so
    a sole live session that prepared without a baton can also save. Raises ``ValueError``;
    writes nothing."""
    if not isinstance(allow_sole_live, bool):  # "false" is truthy: never infer consent
        raise TypeError(
            f"allow_sole_live must be a bool, got {type(allow_sole_live).__name__}"
        )
    requires_baton = store.consolidate_requires_baton()
    if requires_baton and (session_id is None or wrap_token is None):
        raise SaveAuthorityError(
            "This store is baton-protected (Store.consolidate_requires_baton): a save must "
            "pass the session_id of the baton holder and the wrap_token its prepare_wrap "
            "returned. Nothing was written."
        )
    if session_id is None:
        return
    if allow_sole_live and not requires_baton:
        auth = sessions.consolidate_authorized(
            store.continuity_path, session_id, allow_sole_live=True
        )
        if not auth["authorized"]:
            raise SaveAuthorityError(
                f"Session {session_id!r} is no longer authorized to commit this wrap "
                f"({auth['reason']}; baton holder = {auth['baton_holder'] or 'none'}, "
                f"{len(auth['live_session_ids'])} live session(s)). Nothing was written."
            )
        return
    try:
        holder = sessions.baton_holder(store.continuity_path)
    except (OSError, json.JSONDecodeError):
        raise SaveAuthorityError(
            f"Session {session_id!r} cannot be confirmed as the consolidate baton holder "
            f"(the baton file is unreadable), so it may not commit this wrap. Nothing was "
            f"written."
        ) from None
    if holder == session_id:
        return
    if holder is None and requires_baton:
        cause = "this store is baton-protected, which ignores allow_sole_live, and no baton is claimed"
    elif holder is None:
        cause = (
            "no baton is claimed (it was never claimed, or was released); a session that "
            "prepared as the sole live session must pass allow_sole_live=True here too"
        )
    else:
        cause = f"the baton is held by {holder!r}"
    raise SaveAuthorityError(
        f"Session {session_id!r} does not hold the consolidate baton: {cause}, so it may not "
        f"commit this wrap. Nothing was written."
    )


# prepare_wrap's caller-supplied wrap_token: the shape uuid.uuid4().hex mints, so
# a caller token is indistinguishable from a minted one wherever tokens are used.
_CALLER_TOKEN_RE = re.compile(r"[0-9a-f]{32}")


_SUPERSEDES_RE = re.compile(
    r"\[supersedes:\s*([0-9A-Fa-f]{8})\s+by\s+([0-9A-Fa-f]{8})\s*\]", re.IGNORECASE
)
# Anything that LOOKS like the marker. One that the strict form does not parse is
# reported as rejected, so a typo never reads as "nothing to record".
_SUPERSEDES_LOOSE_RE = re.compile(r"\[\s*supersedes\b[^\]\n]*\]?", re.IGNORECASE)


def _record_wrap_supersessions(
    store: Store, text: str, valid_ids: set[str]
) -> tuple[int, list[dict[str, str]]]:
    """Record the ``[supersedes: OLD by NEW]`` links a wrap proposes.

    Runs inside the save's batch. ``NEW`` must be an episode of this wrap's
    frozen snapshot (the same set a citation is checked against); the rest of
    the validation is :meth:`Store.supersede`'s. A link already on record is
    skipped silently, so a marker carried forward into later wraps is
    idempotent. A rejected link never fails the save: it is returned with the
    reason, like a demotion.
    """
    recorded = 0
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    parsed_at = {m.start() for m in _SUPERSEDES_RE.finditer(text)}
    for loose in _SUPERSEDES_LOOSE_RE.finditer(text):
        if loose.start() not in parsed_at:
            rejected.append({
                "old_id": "", "new_id": "",
                "reason": f"unparsed marker {loose.group(0)!r}; the form is "
                          f"[supersedes: <old 8-hex id> by <new 8-hex id>]",
            })
    for m in _SUPERSEDES_RE.finditer(text):
        old_id, new_id = m.group(1).lower(), m.group(2).lower()
        if (old_id, new_id) in seen:
            continue
        seen.add((old_id, new_id))
        if new_id not in valid_ids:
            if store.supersession_exists(old_id=old_id, new_id=new_id):
                continue
            rejected.append({
                "old_id": old_id, "new_id": new_id,
                "reason": f"{new_id} is not an episode of this wrap",
            })
            continue
        try:
            if store.supersede(old_id=old_id, new_id=new_id, source="wrap"):
                recorded += 1
        except SupersessionError as exc:
            rejected.append({"old_id": old_id, "new_id": new_id, "reason": str(exc)})
    return recorded, rejected


def _durable_cue_state(
    store: Store, schema: list[SectionSpec], saved_text: str
) -> tuple[str | None, list[str]]:
    """The recall tier's per-store state for a saved continuity with a durable section:
    ``(value for the ``durable_inert_tokens`` metadata key, cue warnings)``.

    The value is the cue and fact words that are too common in this store's episodes to
    cue anything (:func:`~anneal_memory.retrieval.compute_durable_inert_tokens`), tied to
    the hash of the exact text being saved, so the per-prompt recall path only READS it
    and ignores it the moment the continuity changes. The warnings name each inert cue
    (it will cue nothing) and each cue token too short to match. A failure here must
    never cost the save: it returns ``(None, [])`` and the key is simply not written."""
    try:
        facts = parse_durable_facts(saved_text, schema)
        inert = compute_durable_inert_tokens(store, facts)
        episodes = store.recall(limit=0).total_matching
        value = json.dumps({
            "tokens": sorted(inert),
            "continuity_hash": continuity_hash(saved_text),
            "episodes": episodes,
            "threshold": DURABLE_GENERIC_DF,
        })
        warnings_out: list[str] = []
        seen_inert: set[str] = set()
        seen_short: set[str] = set()
        percent = f"{DURABLE_GENERIC_DF:.0%}"
        for fact in facts:
            for cue in fact.cues:
                for token in re.findall(r"[a-z0-9]+", cue.lower()):
                    if len(token) < 3:
                        if token not in seen_short:
                            seen_short.add(token)
                            warnings_out.append(
                                f"cue {token!r} is too short to match; spell it out"
                            )
                for token in _fact_tokens(cue):
                    if token in inert and token not in seen_inert:
                        seen_inert.add(token)
                        warnings_out.append(
                            f"cue {token!r} appears in more than {percent} of this "
                            f"store's episodes, so it will not cue anything; add a more "
                            f"specific cue"
                        )
        return value, warnings_out
    except Exception:  # noqa: BLE001 - see the docstring
        return None, []


def _prior_levels(
    prior_text: str | None, schema: list[SectionSpec],
    saved_levels: dict[tuple[str, str], int] | None, store: Store,
) -> dict[str, int] | None:
    """Where a held line's level comes from: the level the prior-state bound would
    start the name from (``graduation._prior_base`` on its named key), so the hold and
    the bound agree. Once the store has saved under the bound, its ``pattern_levels``
    record is the prior, the prior file may only lower it, and a name it never saved is
    a new claim. Never a crystal's level: the crystal store is caller-writable, so a
    crystal re-added to the working set is a new claim (spore-676 (A) as Phill ruled
    2026-10-08 13:17: the held level comes from pattern_levels). A store that has not
    saved under the bound yet falls back to the prior file, or, when that is blank over
    a store with history (a truncated file, L3 r3 complement), to each pattern's
    recorded high-water mark. None when there is nothing to derive from (a first save)."""
    if saved_levels is not None:
        file_levels = (_pattern_levels(prior_text, schema)
                       if prior_text and prior_text.strip() else {})
        levels: dict[str, int] = {}
        for (kind, name), level in saved_levels.items():
            if kind == "name":
                levels[name] = min(level, file_levels.get(name, level))
        return levels
    if not prior_text or not prior_text.strip():
        levels = {}
        try:
            for n in store.pattern_history_names():
                hist = store.get_pattern_history(n) or {}
                mx = hist.get("max_level_reached")
                if isinstance(mx, int):
                    levels[n] = mx
        except Exception:  # noqa: BLE001 - unreadable history is not proof of a first save
            levels = {"": 0}  # a bound that holds nothing: fail closed
        if not levels:
            return None
        return levels
    return _pattern_levels(prior_text, schema)


def _crystal_levels_snapshot(
    crystal_store: CrystalStore | None,
) -> dict[str, Any] | None:
    """Live crystal names -> levels, read BEFORE the save batch so no crystal-file IO
    happens under the database write lock (L3 1007, codex). None when the store could
    not be read: then a pattern probe absent from the file is ``unchecked``, not lost."""
    if crystal_store is None:
        return {}
    try:
        return {str(c["name"]): c.get("level") for c in crystal_store.active()
                if isinstance(c, dict) and isinstance(c.get("name"), str)}
    except Exception:  # noqa: BLE001 - an instrument's input; the wrap path reports faults
        return None


def _evaluate_drift_probes(
    store: Store, schema: list[SectionSpec], text: str, crystals: dict[str, Any] | None,
    deferred: list[str],
) -> list[dict[str, Any]]:
    """CAP-06: every live drift probe checked against ``text`` (empty when none).
    Called inside the save batch. Never raises: a probe is an instrument, and an
    instrument must not refuse a save (L3 1007: a nameless crystal row did)."""
    try:
        probes = store._live_drift_probes_in_txn()
        if probes is None:
            deferred.append(
                "drift probes were not checked this save: the probe table could not be "
                "read")
            return []
        if not probes:
            return []
        levels = _pattern_levels(text, schema)
        results = evaluate_probes(text, probes, pattern_levels=levels,
                                  live_crystals=crystals or {})
        if crystals is None:  # crystal store unreadable: absence proves nothing
            for r in results:
                if r["kind"] == "pattern" and r["status"] == "lost":
                    r["status"] = "unchecked"
                    r["detail"] = "not in the file; the crystal store could not be read"
        return results
    except Exception as exc:  # noqa: BLE001 - see the docstring
        deferred.append(f"drift probes were not checked this save: {exc!r}")
        return []


def _pattern_levels(text: str, schema: list[SectionSpec]) -> dict[str, int]:
    """Each pattern named in the graduating section(s) of ``text``, at its highest
    level there, found exactly as ``validate_graduations`` finds them (same heading
    test, same line parser; L3 1007, codex: a fuzzy section match read
    ``## Anti-Patterns``). No level ceiling."""
    headings = graduating_headings(schema)
    levels: dict[str, int] = {}
    inside = False
    for line in text.split("\n"):
        if line.startswith("## "):
            inside = _is_graduating_heading(line, headings)
            continue
        m = _NAMED_PATTERN_RE.match(line) if inside else None
        if m:
            try:
                level = int(m.group(2))
            except ValueError:  # beyond Python's int-string limit
                continue
            levels[m.group(1)] = max(levels.get(m.group(1), 0), level)
    return levels


def _drift_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and each probe's id and status. No probe text and no file text: the save
    result goes back to the composer being measured, and the audit cannot be redacted
    (L3 1007, complement). The detail is in ``probe status``."""
    counts = {status: 0 for status in PROBE_STATUSES}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"counts": counts,
            "probes": [{"id": r["probe_id"], "status": r["status"]} for r in results]}


def _grounded_trust(
    store: Store, grounds: set[str], others: set[str],
) -> tuple[dict[str, str], dict[str, tuple[bool, str]]]:
    """Effective trust of the episodes a save leans on, and the
    :meth:`Store.ground_state` it was read from. A ground (a cited or previously
    grounding episode) that no longer exists reads ``external``: a deleted ground
    is a failed one, never a default ``agent`` one (codex r3 #1). ``others``
    (``[supersedes:]`` endpoints) keep the plain reading."""
    state = store.ground_state(grounds | others)
    trust: dict[str, str] = {}
    for cid, (exists, eff) in state.items():
        level = "external" if (not exists and cid in grounds) else eff
        if level != DEFAULT_TRUST:
            trust[cid] = level
    return trust, state


def _gone_marks(
    grounding: dict[str, dict[int, list[dict[str, Any]]]],
) -> dict[str, tuple[str, ...]]:
    """Each episode id's removal marks across the grounding record (D3 R3)."""
    marks: dict[str, list[str]] = {}
    for rungs in grounding.values():
        for groups in rungs.values():
            for g in groups:
                for cid, mark in g.get("gone", {}).items():
                    marks.setdefault(cid, []).append(mark)
    return {cid: tuple(sorted(m)) for cid, m in marks.items()}


def validated_save_continuity(
    store: Store,
    text: str,
    affective_state: AffectiveState | None = None,
    *,
    today: str | None = None,
    wrap_token: str | None = None,
    allow_shrink: bool = False,
    allow_unlinked: bool = False,
    carryforward_cold_days: int | None = 7,
    crystal_store: CrystalStore | None = None,
    compost: list[str] | None = None,
    session_id: str | None = None,
    allow_sole_live: bool = False,
    require_rederive: bool = False,
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
            **Required** on a store with the require-baton policy
            (:meth:`Store.consolidate_requires_baton`), together with
            ``session_id``.
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
        allow_unlinked: DEPRECATED no-op, accepted for compatibility. It
            overrode the AM-LINKGATE save refusal (spore-721), which was
            removed in 0.9.26 because pattern recall stopped reading episode
            links when the Hebbian hop was retired; a write path that records
            no links is still warned by AM-WARN Signal B. A literal ``True``
            changes nothing and emits one ``UserWarning`` after the save
            commits. CLI ``--allow-unlinked``; MCP ``"allow_unlinked": true``.
        compost: pattern names whose concept left the working set this
            wrap. Each is severed (``sever_pattern_concept``: its pattern-graph
            edges deleted, its generation bumped) INSIDE the same transaction
            as ``wrap_completed``, so a wrap that fails to commit severs
            nothing and a committed wrap cannot leave a composted pattern's
            edges behind. A name with no edges still gets its generation
            boundary. ``None`` (the default) skips this step and adds no key
            to the result. Library-only; no CLI or MCP surface.
        session_id: The consolidate-efferent session making this save
            (flow spore-1169). When passed, the save proceeds only if that
            session is STILL authorized (re-checked here, so a take mid-wrap
            revokes the old holder's commit): it holds the consolidate baton
            or, under ``allow_sole_live``, is the sole live session and no other
            session holds a baton; otherwise ``ValueError`` with nothing written. Required, with
            ``wrap_token``, on a store with the require-baton policy; ``None``
            elsewhere skips the check, EXCEPT that a wrap prepared with a
            ``session_id`` is committed only by that same ``session_id`` (a save that
            omits it, or names another session, is refused). Library-only; no CLI or
            MCP surface.
        allow_sole_live: Only meaningful with ``session_id``, and ignored on a
            store with the require-baton policy, as in ``prepare_wrap``. Pass
            the same value the wrap's ``prepare_wrap`` got: a sole live
            session that prepared with ``allow_sole_live=True`` and holds no
            baton is refused at the save without it. Default ``False``.
        require_rederive: Refuse the save (``ValueError``, nothing written) unless
            the schema has a derived-state section, the store is opted in to
            re-derive (``derive allow``), and at least one State command ran at
            this save. Without it, a store that is not opted in saves after the
            static checks alone. A caller that checks the opt-in before opening
            the wrap passes this to cover a trust revoked before the save.
            Default ``False``.

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
          - ``bare_demoted`` (int): bare (evidence-free) Proven-tier
            graduations demoted for missing citations
          - ``citation_reuse_max`` (int): max times any single episode
            was cited in this wrap
          - ``gaming_suspects`` (list[str]): episode IDs flagged for
            suspicious citation reuse
          - ``associations_formed`` (int)
          - ``associations_strengthened`` (int)
          - ``associations_decayed`` (int)
          - ``skipped_non_today`` (int): graduation-format lines whose date
            is not today, which validation skipped
          - ``linkgate_overridden`` (bool): always False. Kept for
            compatibility; the AM-LINKGATE refusal it reported was removed
            in 0.9.26
          - ``citation_spread`` (int): distinct episode ids (8-char) cited on
            today's 2x-and-up graduation lines that belong to this wrap's
            episodes, INCLUDING lines later demoted. A report, not a check.
          - ``composted`` (dict[str, int]): present ONLY when ``compost``
            was passed — each distinct name mapped to the edges severed
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
        ContinuityValidationError: A ``ValueError`` subclass, raised when the
            text that would be written (the durable section excluded) is
            above ``hard_max_chars(schema)``. It is a size refusal only; the
            other refusals above are plain ``ValueError``. Nothing is
            written and the wrap stays in progress.
        SaveAuthorityError: A ``ValueError`` subclass, raised when the
            consolidate gate refuses the save: the caller is not
            authorized to commit this wrap, or the call omits a
            ``session_id`` or ``wrap_token`` the gate requires.
            Nothing is written and the wrap stays in progress.
        TypeError: If ``compost`` is a bare string, or holds anything
            but non-empty names without surrounding whitespace. Checked
            after the wrap-state preconditions, so with no wrap in
            progress the ``ValueError`` above wins.
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

    # flow spore-1169: the baton is re-checked at the save, not only at prepare. The wrap token
    # is not a secret (wrap-token-current prints it), so on a baton-protected store only a
    # caller that names itself AND still holds the baton may commit; and any caller that names
    # itself is refused once the baton has been taken from it mid-wrap.
    if session_id is not None and not session_id:
        raise ValueError(
            "validated_save_continuity: session_id must be non-empty when provided."
        )
    # An early pass, so a refused save fails before the expensive validation. It is repeated
    # authoritatively inside the batch, after wrap_completed (see there).
    _check_save_authority(store, session_id, wrap_token, allow_sole_live)
    # A wrap prepared under the consolidate gate names the session that prepared it, and only
    # that session may commit it (strict match). A save that omits session_id would skip every
    # baton check above (a token identifies the wrap, not the actor), and a different session,
    # even the current baton holder, is committing a compression it did not make and that the
    # baton's previous holder was revoked from: it prepares its own. Read before the batch:
    # wrap_completed clears the key.
    gated_by = store.wrap_gated_session()
    if gated_by is not None and wrap_token is None:
        # A gated wrap's save must round-trip its token: without one, a delayed save from the
        # right session would load whatever wrap is CURRENT (a later prepare's) and commit
        # stale text against it. Checked first so the refusal names the cheapest fix.
        raise SaveAuthorityError(
            f"This wrap was prepared under the consolidate gate by session {gated_by!r}, so "
            f"its save must pass the wrap_token that prepare_wrap returned (and that session_id). "
            f"Nothing was written."
        )
    if gated_by is not None and session_id != gated_by:
        if session_id is None:
            raise SaveAuthorityError(
                f"This wrap was prepared under the consolidate gate by session {gated_by!r}, "
                f"so the save must come from that session, naming its session_id: without one the "
                f"baton is never re-checked. Finish it from the library with that session_id, or "
                f"abandon it with wrap-cancel --wrap-token <this wrap's token> (MCP: wrap_cancel "
                f"with wrap_token), which discards the compression. Nothing was written."
            )
        raise SaveAuthorityError(
            f"This wrap was prepared by session {gated_by!r}, and only that session may "
            f"commit it; {session_id!r} cannot, even as the current baton holder. Abandon it "
            f"(wrap-cancel --wrap-token <this wrap's token>, or MCP wrap_cancel with wrap_token, "
            f"which discards the compression) and prepare_wrap "
            f"again from {session_id!r}. Nothing was written."
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

    # --- Payload validation: compost, then the continuity text itself ---
    # Materialized first: validating a one-shot iterator would exhaust it. A
    # padded name is refused rather than stripped, so the ``composted`` result
    # is keyed by exactly the strings the caller passed.
    compost_names: list[str] | None = None
    if compost is not None:
        if isinstance(compost, (str, bytes)):
            raise TypeError(
                "compost must be a list of non-empty pattern-name strings"
            )
        compost = list(compost)
        if not all(
            isinstance(n, str) and n and n == n.strip() for n in compost
        ):
            raise TypeError(
                "compost must be a list of non-empty pattern-name strings "
                "without leading or trailing whitespace"
            )
        compost_names = list(dict.fromkeys(compost))

    if not text or not text.strip():
        raise ValueError("Continuity text cannot be empty")
    # NOTHING UN-CANONICAL ENTERS (Phill 12:13, "(A)"; graduation.canonical_continuity_text).
    # The pipeline has two inputs, and both are made canonical where they enter:
    # the caller's text here, and every prior continuity read through
    # Store.load_continuity (the one load point, canonical itself). Every
    # parser after this line (the rederive strip, the durable carry-forward and
    # its drop markers, the gate) reads one grammar. L3 r8: an NBSP-indented
    # ``[drop-durable:]`` and a VT-hidden ``## State`` verdict were parsed raw.
    text = canonical_continuity_text(text)

    # Validate structure (all sections declared by the store's schema). The
    # schema is read once here and reused for the schema-aware graduation gate
    # further down (v0.3.4). Read fail-closed (v0.3.5): a corrupt persisted
    # schema must refuse the save rather than silently fall back to the ops
    # DEFAULT_SCHEMA and disable the catastrophic-shrink gate below.
    section_schema = store.section_schema_for_wrap()
    if any(s["role"] == "derived-state" for s in section_schema):
        # Text loaded with --rederive carries load-time verdicts; they are
        # true only at load, so they never persist (L1 round-trip finding).
        text = strip_rederive_output(text, section_schema)
    # Loaded once here and reused by the durable-facts invariant just below,
    # the catastrophic-shrink gate and the silent-omission audit further down.
    prior_continuity = store.load_continuity()  # canonical: Store.load_continuity
    # Durable facts (B1): before anything validates, hashes or writes the text,
    # carry every prior durable line forward (re-inserting what the composer
    # left out) and apply the composer's drop markers. Never a refusal. A
    # schema without a durable section returns the text untouched and None.
    text, durable_report = enforce_durable_facts(
        prior_continuity, text, section_schema
    )
    # Backstop: both inputs are already canonical, so this is the identity unless a
    # step above introduced a non-canonical character itself (idempotent, cheap).
    text = canonical_continuity_text(text)
    grad_headings = graduating_headings(section_schema)
    # Reject ambiguous merged headings (e.g. "## Patterns and Understanding")
    # with a clear message before the generic all-sections check: one header
    # satisfying two required sections would route a single body into two
    # protected roles and defeat the shrink gate (v0.3.5). Each section needs
    # its own '## ' header line. An optional heading counts only as an exact
    # header (see _header_matches).
    for _line in text.split("\n"):
        if _line.startswith("## "):
            _matched = _header_matches(_line.lower(), section_schema)
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

    # Derived-state gate (spore-1230): a State line with no annotation or a
    # command outside the allowlist is refused; on a store opted in to
    # re-derive, so is a command that errors. Stale or unchecked lines do not
    # refuse: they go into the result's ``stale_state`` and a warning after
    # commit. See docs/rederive.md.
    # spore-1282: compare-and-swap on the label -> root map. The map is read once
    # inside, compared with the one prepare_wrap froze, and the commands run
    # against that same map.
    _frozen_roots = None
    _derive_report = None
    if any(s["role"] == "derived-state" for s in section_schema):
        _frozen_roots = store.wrap_derive_roots(expect_token=snapshot["token"])
        _derive_report = check_state_for_save(
            text,
            section_schema,
            store.path,
            frozen_roots=_frozen_roots,
            cancel_hint=(
                "cancel this wrap by its token (CLI: `anneal-memory wrap-cancel "
                f"--wrap-token {snapshot['token']}`; MCP: `wrap_cancel` with that "
                "wrap_token; Python: `store.wrap_cancelled(expect_token=...)`) and run "
                "prepare_wrap again"
            ),
        )
    if require_rederive and (_derive_report is None or not _derive_report.enabled):
        raise ValueError(
            "Save refused: re-derive was required, but re-derive is not enabled "
            "for this store"
            + (" (its schema has no derived-state section)" if _derive_report is None else "")
            + ", so no State line would be checked. See `anneal-memory derive allow`."
        )
    if require_rederive and _derive_report is not None and not _derive_report.ran:
        raise ValueError(
            "Save refused: re-derive was required, but no State command ran at this "
            "save (only [judged:] lines, or the load budget was spent before any ran)."
        )
    stale_state: list[str] = []
    if _derive_report is not None and _derive_report.enabled:
        stale_state = [
            f"line {r.index + 1}: {r.flag}"
            for r in _derive_report.results
            if r.status in ("stale", "skipped")
        ]

    # Catastrophic-shrink gate (v0.3.5). Load the prior continuity ONCE here
    # and reuse it for the silent-omission audit further down. The gate runs
    # before the episode fetch + graduation so a collapsing wrap fails fast;
    # raising ValueError leaves the wrap in progress (same as the
    # structure-validation failure above), so the agent re-wraps with the
    # felt/identity layers preserved (or passes allow_shrink for a deliberate
    # diet) without losing the prepared wrap. The prior continuity was loaded
    # above, before the durable-facts invariant; durable lines kept or
    # re-inserted only ever add to the new text, so they can never be what
    # makes this gate refuse.
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
    # KL-14 (2026-10-07): an override must leave a trace. When the operator passes
    # allow_shrink, run the same check without it and record in the audit whether
    # the override changed the outcome, and the refusal it suppressed.
    # Always written, so an event without the field means an older version, never
    # "no override".
    shrink_override: dict[str, Any] = {"requested": allow_shrink is True}
    if allow_shrink is True:
        try:
            _check_no_catastrophic_shrink(
                prior_continuity, text, section_schema, allow_shrink=False,
                crystallized_credit=crystallized_credit,
            )
            shrink_override["refusal_suppressed"] = False
        except ValueError as exc:
            shrink_override["refusal_suppressed"] = True
            shrink_override["refusal"] = str(exc)[:2000]

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
    # A superseded episode is not evidence. Leave out of the citable set every
    # episode of this wrap already superseded, and every one a link proposed in
    # THIS text would supersede once recorded (codex L3: the links are recorded
    # after graduation runs, so a 2x citing only the replaced fact validated).
    superseded_in_window = set(store.superseded_by_map(sorted(valid_ids)))
    # Simulated in document order, the order they are recorded in, against the
    # store's links plus the ones accepted so far: a later proposal that closes
    # a cycle with an earlier one is refused at record time, so it must not
    # make its target uncitable here (reproduced: "A by B" then "B by A" left
    # live B uncitable).
    # Run again under the write lock (L3 r2 1009+22): this read precedes the
    # ground_state baseline, so a trust change between them moved no ground the
    # lock-time re-read compares, yet changed which links are accepted here.
    def _accepted_supersedes() -> list[tuple[str, str]]:
        edges: dict[str, set[str]] = {}
        for link in store.supersession_links():
            edges.setdefault(link["old_id"], set()).add(link["new_id"])

        def _reaches(start: str, goal: str) -> bool:
            seen, todo = set(), [start]
            while todo:
                cur = todo.pop()
                if cur == goal:
                    return True
                if cur not in seen:
                    seen.add(cur)
                    todo.extend(edges.get(cur, ()))
            return False

        accepted: list[tuple[str, str]] = []
        for m in _SUPERSEDES_RE.finditer(text):
            old_id, new_id = m.group(1).lower(), m.group(2).lower()
            if new_id not in valid_ids or _reaches(new_id, old_id):
                continue
            if store.supersession_problem(old_id=old_id, new_id=new_id) is None:
                edges.setdefault(old_id, set()).add(new_id)
                accepted.append((old_id, new_id))
        return accepted

    accepted_supersedes = _accepted_supersedes()
    superseded_in_window |= {o for o, _ in accepted_supersedes if o in valid_ids}
    citable_ids = valid_ids - superseded_in_window

    # Check citation history
    meta = store.load_meta()
    citations_seen = meta.get("citations_seen", False)

    # Validate graduations (demotes bad citations in-place).
    # Caller may pin ``today`` for deterministic test runs; default is
    # wall-clock. Same pattern _build_wrap_package already uses.
    today_str = today if today is not None else (store.wrap_today() or _wrap_local_date(store))
    # The bound's prior: the store's own record of the levels it last saved.
    # ⛔ NO CRYSTAL SEED (L3 r4, 1008+3, DELETED): a crystal level seeded the
    # first bounded save and was defeated a new way three rounds running (a
    # caller-set crystallized_on, a caller-set level, then a caller-writable
    # pattern_history bound). A pattern crystallized out before the record
    # existed re-enters the continuity as new and re-earns its rungs; its crystal
    # is untouched.
    saved_levels = store.saved_pattern_levels()
    # CAP-08 T3: where each citable episode came from (absent = agent), so a
    # graduation grounded only in tool/external episodes does not climb.
    # Plus every [supersedes:] endpoint: a trust change on one decides whether its
    # link is recorded, which decides what was citable (codex r2, the race).
    _marker_ids = {
        i.lower() for mm in _SUPERSEDES_RE.finditer(text) for i in (mm.group(1), mm.group(2))
    }
    # CAP-08 D2 (C#11): a rung whose recorded grounding is all tool/external
    # under today's trust no longer counts toward the pattern's prior.
    grounding = store.pattern_grounding()
    _grounding_ids = {
        cid for rungs in grounding.values() for groups in rungs.values()
        for g in groups for cid in g["episodes"]
    }
    # Effective trust (D3): an episode derived from others counts at most as
    # trusted as its most trusted source.
    window_trust, window_state = _grounded_trust(
        store, citable_ids | _grounding_ids, _marker_ids)
    window_gone = _gone_marks(grounding)
    revoked_levels = revoked_pattern_levels(
        grounding, lambda cid: window_trust.get(cid, DEFAULT_TRUST)
    )
    grad_result = validate_graduations(
        text=text,
        valid_ids=citable_ids,
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
        # L3 1007 (complement, codex): a held line's level is derived, never taken
        # from the composer (see _prior_levels).
        prior_levels=_prior_levels(prior_continuity, section_schema, saved_levels, store),
        # The prior-state bound (1007+29): every line is cut to the level the
        # STORED prior continuity entitles it to (new -> 1x, validated -> +1),
        # whatever date, level or tag shape the composer wrote. "" = no prior
        # file, so every line is new.
        prior_text=prior_continuity or "",
        saved_levels=saved_levels,
        trust_of=lambda cid: window_trust.get(cid, DEFAULT_TRUST),
        revoked_levels=revoked_levels,
    )

    # The hard maximum is measured on the text that will be WRITTEN: graduation
    # rewrites lines (a bare ``2x`` becomes ``1x`` plus a note), so the input's
    # size is not the file's. Nothing is written or recorded before this point.
    _check_hard_max(
        store, grad_result.text, section_schema,
        len(text) - durable_section_chars(text, section_schema),
    )

    # Detect Proven-tier (2x and ABOVE — `min_level` is a FLOOR, not a range, so a
    # 4x+ pattern is covered) patterns silently dropped between the
    # prior wrap and this one. validate_graduations operates only on
    # patterns the agent wrote INTO the new continuity, so a pattern
    # that was at 2x or higher in the prior continuity and is absent from
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
    patterns = len(re.findall(r"\|\s*\d+x", mask_explanations_text(grad_result.text)))
    total_demoted = grad_result.demoted + grad_result.bare_demoted
    # Hoisted out of the try block so ``path`` is unambiguously bound
    # on every code path (including the residual-window recovery
    # comment above where mypy would otherwise flag a possibly-unbound
    # reference). ``continuity_path`` is a property that reads
    # ``self._path`` — it's stable across the pipeline.
    path = str(store.continuity_path)

    # Phase 1: continuity tmp write.
    # The tmp filename suffix is a PAIR ID: ``<token12>-<uuid8>``.
    #   * the ``<token12>`` prefix gives continuity.tmp + meta.tmp (written in
    #     Phase 2) a shared recoverability identity, and lets the startup
    #     orphan-detector match an in-flight wrap by its active wrap_token
    #     prefix (10.5c.5 L3 Fix #19 + #21).
    #   * the ``<uuid8>`` makes the path UNIQUE PER SAVE ATTEMPT. AM-CONTLOCK
    #     L3 (codex HIGH): the Phase-1 tmp write happens BEFORE the wrap-token
    #     CAS and OUTSIDE the Phase-3 flock, so two concurrent saves sharing one
    #     snapshot token would otherwise write the SAME deterministic tmp path —
    #     the loser's bytes could be renamed under the winner's committed DB row
    #     (silent file/DB divergence). A per-attempt uuid makes a collision
    #     negligibly unlikely — a 32-bit suffix is ~2^-32 birthday odds for two
    #     same-snapshot saves, and the realistic concurrency is a single
    #     consolidate plus at most one external editor — so each save renames ITS
    #     OWN tmp. (Not a hard impossibility: an airtight version would be
    #     O_CREAT|O_EXCL-with-retry; unwarranted at this concurrency.) This
    #     reconciles Fix #19's pairing (kept via the token prefix) with the
    #     uniqueness the earlier 10.5c.5 fix required (which #19 had traded away).
    tmp_pair_id: str = f"{snapshot['token'][:12]}-{uuid.uuid4().hex[:8]}"
    # AM-SNAPSHOT ①: compute the continuity content hash ONCE here, from the
    # exact text written to the tmp sidecar (and renamed to continuity.md in
    # Phase 3). It is passed into wrap_completed (durable in the wraps row at
    # the Phase-2 commit, the recovery oracle) AND reused in the Phase-4
    # continuity_saved audit payload below — single source, so the durable
    # oracle and the audit event can never disagree about what this wrap saved.
    content_hash: str = hashlib.sha256(
        grad_result.text.encode("utf-8")
    ).hexdigest()
    # Durable cue state (cue wiring): computed here, before the batch opens, from the exact
    # text being saved, so the recall tier's inert-token set is committed with the wrap.
    # Only a schema with a durable section has any; others write nothing.
    inert_value: str | None = None
    cue_warnings: list[str] = []
    if durable_report is not None:
        inert_value, cue_warnings = _durable_cue_state(
            store, section_schema, grad_result.text
        )
    cont_tmp: Path | None = store._prepare_continuity_write(
        grad_result.text, token_hex=tmp_pair_id
    )
    meta_tmp: Path | None = None
    wrap_result = None
    # Once ``db_committed`` flips True, the outer ``except`` preserves
    # tmp files instead of cleaning them up — they represent committed
    # state awaiting externalization. Cleaning them up would destroy
    # the new content permanently (L1 HIGH + L2 M2 data-loss path).
    db_committed = False
    composted: dict[str, int] = {}
    still_graduating: list[str] = []
    drift_results: list[dict[str, Any]] = []
    crystal_levels = _crystal_levels_snapshot(crystal_store)
    # Raised inside the batch, delivered only after it commits.
    drift_warnings: list[str] = []

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
            # ``tmp_pair_id`` pairs this tmp with cont_tmp (same
            # ``<token12>-<uuid8>``) so operator recovery matches them.
            meta_tmp = store._prepare_meta_write(
                meta, token_hex=tmp_pair_id
            )

            # The bound ran on levels read before this batch took the write lock.
            # A sever or rename committed in between would be undone by the record
            # below (codex L3 r2 HIGH): re-read under the lock, before this wrap's own
            # compost below, and refuse on any change. Raising rolls the batch back:
            # nothing saved, the wrap stays open.
            if store.saved_pattern_levels() != saved_levels:
                # A ValueError, which every transport surfaces as a refused save.
                raise ValueError(
                    "validated_save_continuity: the store's saved pattern levels "
                    "changed while this save ran (a sever, rename or another save "
                    "committed). Nothing was saved and the wrap is still open; save "
                    "again so the graduation bound reads the current levels."
                )

            # Compost severance, in THIS transaction so it commits or rolls
            # back with the wrap row below. Inside the batch the Store method
            # defers its commit and queues its audit event until the commit.
            for name in compost_names or ():
                composted[name] = store.sever_pattern_concept(
                    name, today=today_str
                )

            # The [supersedes:] links accepted above decided what was citable:
            # recompute that decision under the lock, before this save records any
            # link, and refuse below (after the narrower checks) if it moved.
            _supersedes_moved = _accepted_supersedes() != accepted_supersedes
            supersessions_recorded, supersessions_rejected = \
                _record_wrap_supersessions(store, grad_result.text, valid_ids)
            # Re-read under the batch's write lock (codex L3, reproduced: a link
            # another connection committed between validation and this batch
            # let a pattern graduate on the episode it superseded). Raising
            # here rolls the batch back: nothing saved, the wrap stays open.
            _now_superseded = set(store.superseded_by_map(sorted(valid_ids)))
            _late = sorted(
                (_now_superseded - superseded_in_window) & set(grad_result.citation_counts)
            )
            if _late:
                raise SupersessionError(
                    f"validated_save_continuity: cited episode(s) {', '.join(_late)} were "
                    f"superseded by another writer while this save ran. Nothing was "
                    f"saved and the wrap is still open; save again (cite the replacing "
                    f"episode instead)."
                )
            # The same re-read for the grounds (codex r1 #4; CAP-08 D3 R4): per
            # cited id, whether it exists, its removal marks and its effective
            # trust. Another writer moving any of them after validation read them
            # would let the save commit a graduation judged on the old state (a
            # deleted external ground read the same class before and after, so
            # comparing trust alone missed it). Raising rolls the batch back.
            _cited = set(grad_result.citation_counts) | _marker_ids | _grounding_ids
            _, _state_now = _grounded_trust(
                store, set(grad_result.citation_counts) | _grounding_ids, _marker_ids)
            _gone_now = _gone_marks(store.pattern_grounding())
            _absent = (False, DEFAULT_TRUST)
            _trust_moved = sorted(
                cid for cid in _cited
                if (_state_now.get(cid, _absent), _gone_now.get(cid, ()))
                != (window_state.get(cid, _absent), window_gone.get(cid, ()))
            )
            if _trust_moved:
                raise StoreError(
                    f"validated_save_continuity: cited episode(s) "
                    f"{', '.join(_trust_moved)} were removed or changed trust class "
                    f"while this save ran. Nothing was saved and the wrap is still "
                    f"open; save again.",
                    operation="save_continuity",
                )
            if _supersedes_moved:
                raise ValueError(
                    "validated_save_continuity: the [supersedes:] links this text "
                    "proposes changed while this save ran (a trust change or another "
                    "writer's link). Nothing was saved and the wrap is still open; "
                    "save again so citability reads the current links."
                )

            # The bound's next prior, recorded in this transaction so it commits
            # with the wrap or not at all, with the episodes that grounded each
            # rung validated today (CAP-08 D2).
            store._record_pattern_grounding(
                grad_result.pattern_grounding, today_str, wrap_id=snapshot["token"])
            # A composted name's level row was deleted with its edges above; the
            # first save's tombstone seed must not bring it back (codex L3 r1 HIGH 2).
            _prior_line_levels = pattern_line_levels(prior_continuity or "", grad_headings)
            store._record_pattern_levels(
                pattern_line_levels(grad_result.text, grad_headings), today_str,
                lower_to=_prior_line_levels,
                first_tombstones={k: v for k, v in _prior_line_levels.items()
                                  if not (k[0] == "name" and k[1] in composted)},
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
                # AM-SNAPSHOT ① durable recovery oracle: persist the content
                # hash + tmp pair id in the wraps row, atomic with this wrap's
                # commit (inside wrap_completed's _db_boundary). Durable before
                # the Phase-3 rename, so orphan recovery can self-classify even
                # if the crash beat Phase 4's audit event.
                content_hash=content_hash,
                pair_id=tmp_pair_id,
            )
            # CAP-06: the operator's drift probes, read and checked INSIDE this batch
            # (L3 1007, codex: a probe added between a pre-batch read and the commit
            # had no result) against the exact text being saved; never a gate.
            drift_results = _evaluate_drift_probes(
                store, section_schema, grad_result.text, crystal_levels, drift_warnings
            )
            store._record_drift_results(drift_results)


            # Update cross-session pattern history. Scan the
            # post-validation continuity text for every named pattern
            # line with an [evidence: ...] explanation that was AUTHORED
            # TODAY. That includes today's 1x mentions (preserves their
            # explanation for the next session's cross-session check)
            # and today's surviving Proven-tier graduations (the demoted
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
            # The review worklist is the validator's own records (name, level and
            # explanation from the marker it validated and bound to the name), never a
            # re-parse of the line by name (L3 r3, codex: a second line with the same
            # name overwrote the validated row).
            wrap_graduations: list[tuple[str, int, str]] = list(
                grad_result.graduated_records)
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
                # any-level evidence form (GRADUATION_RE excludes 1x but has NO
                # ceiling; this said "2x/3x-only" until 2026-09-04) so
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
                # A level of 10 or more digits is a malformed marker (bounded before
                # int(), so a huge run can neither raise nor overflow the history).
                if len(ev_match.group(2)) > 9:
                    continue
                pattern_level = int(ev_match.group(2))
                if not explanation:
                    continue
                store.upsert_pattern_history(
                    pattern_name=ev_match.group(1),
                    level=pattern_level,
                    explanation=explanation,
                    wrap_id=None,
                    # AM-PRESERVE determinism (spore-081): anchor the recency
                    # baseline to the pipeline's `today`, not wall-clock, so the
                    # warm-preservation gate's (today − last_seen_at) stays
                    # coherent on deterministic/backdated runs.
                    seen_at=today_str,
                )
            store._record_wrap_graduations(wrap_graduations)

            # flow spore-1169: the authoritative consolidate-gate check, deliberately the LAST
            # statement in the batch. wrap_completed's DML above means this connection holds
            # SQLite's write lock, so a policy change from any other connection has either
            # committed already (and this re-read sees it) or waits for this transaction to
            # end. Raising rolls the whole batch back, like the linkgate block. The baton is a
            # sidecar file outside the transaction, so a take landing between this read and the
            # commit below (the commit itself) is not seen: a documented residual. Holding the
            # baton flock across the commit instead was tried and withdrawn in L3: a failed
            # unlock after the commit discarded the committed tmp files, and an on_audit_event
            # callback that touches the baton would block on the lock this process holds.
            # Durable cue state: one metadata row, in this transaction, so it commits or
            # rolls back with the wrap row and always names the continuity it was computed
            # for (the recall path also checks that hash, so a continuity written any other
            # way simply turns the filter off).
            # If the computation failed there is no current set, and an old one must not
            # outlive the continuity it described: remove it in the same transaction (the
            # recall path withholds the durable tier until a wrap writes a current key).
            if durable_report is not None:
                if inert_value is not None:
                    store._conn.execute(
                        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                        (INERT_TOKENS_KEY, inert_value),
                    )
                else:
                    store._conn.execute(
                        "DELETE FROM metadata WHERE key = ?", (INERT_TOKENS_KEY,)
                    )
            _check_save_authority(store, session_id, wrap_token, allow_sole_live)
            # Batch context manager commits here on successful exit.

        # Batch exited without raising → DB is committed. From this
        # point forward, failure must NOT unlink the tmp files.
        db_committed = True

        # AM-LINKGATE-DECAY (Slice B): seed weak cortical links between the
        # patterns that GRADUATED together this wrap. Recall-INDEPENDENT (the
        # exploration / selection-bias channel the keyword recall hook is blind
        # to) and SHADOW-MODE (nothing reads the pattern graph for recall yet).
        # Runs as a SEPARATE post-commit transaction, NOT inside the batch above
        # (codex L3 HIGH): a fault here must be best-effort, but inside the batch
        # the Store's _db_boundary rolls back the WHOLE open transaction on any
        # error — so a seeding fault swallowed by this try/except would silently
        # discard the episode-Hebbian DML written by process_wrap_associations,
        # and the wrap would still "succeed." Post-commit, the wrap's DB state is
        # already durable, so a seeding failure can only cost the (non-load-bearing,
        # self-healing) seed itself.
        # A composted name is left out: seeding it would re-link the concept
        # this same save just severed.
        seed_names = [n for n in grad_result.graduated_names if n not in composted]
        try:
            if len(seed_names) >= 2:
                store.seed_pattern_co_graduation(seed_names, today=today_str)
        except Exception:
            pass
        # Warned only after Phase 3, through _warn_after_commit.
        still_graduating = sorted(
            set(composted) & set(grad_result.graduated_names)
        )

        # Phase 3: DB commit succeeded — externalize files.
        # At this point cont_tmp is still the Path returned from
        # _prepare_continuity_write (we only clear it to None after the
        # successful rename below). The type is Path | None only because
        # of the consumed-handle pattern further down.
        # AM-CONTLOCK: take the SHARED cross-process continuity lock around the
        # externalization (both renames). This is the physical race point — an
        # external editor (e.g. Levain's governed State write) reads the
        # continuity file, and a rename interleaving its read→os.replace would
        # silently clobber (or be clobbered). The lock spans Phase 3 ONLY, not
        # the whole pipeline: anneal writes the WHOLE recomposed file (no
        # read-merge — verified, the save reads no continuity text), so the
        # only shared resource is the rename. The wrap-vs-edit PRECEDENCE during
        # Phases 1–2 (an operator State edit landing while the agent composes)
        # is the arrow-of-time case the design intends (a wrap supersedes a
        # concurrent edit; the editor's undo refuses across the wrap), NOT a
        # bug for the lock to prevent — see continuity_lock's docstring. The
        # lock is held INSIDE the post-commit region. A LOCK-UNAVAILABLE flock
        # failure (ENOLCK/EOPNOTSUPP on a lock-less FS) does NOT reach here:
        # continuity_lock degrades it to an unlocked no-op so the write still
        # completes. Any OTHER flock OSError (a real fault) DOES propagate, and
        # being post-commit it lands in the existing tmp-preservation path below,
        # exactly like a rename OSError (committed DB + recoverable tmps).
        #
        # LOCK ORDERING (do NOT move the acquire): the flock is taken HERE, AFTER
        # store._batch() has fully committed the SQLite transaction (db_committed is
        # already True). The flock and the DB lock are DISJOINT/sequential, never
        # nested — so there is no flock↔SQLite lock-order cycle to deadlock on.
        # Acquiring the flock earlier (around the batch) would nest them and create
        # one; keep it scoped to the renames.
        with store.continuity_lock():
            # Explicit None check instead of ``assert`` so the guard survives
            # ``python -O``, which strips assertions — the SAME correction made
            # for ``meta_tmp`` twenty lines below (L3 complement F3 +
            # contrarian F6). That fix landed on the sibling and not here, so
            # under ``-O`` this assert vanished and ``cont_tmp.replace(...)``
            # raised ``AttributeError`` on None: the wrong exception type for
            # the transport layer, at the moment the DB has already committed.
            if cont_tmp is None:
                raise StoreError(
                    "internal pipeline invariant violated: cont_tmp is None "
                    "after the batch committed — this indicates a bug in "
                    "validated_save_continuity's control flow. The DB has "
                    "committed but the continuity sidecar was never staged; "
                    "the store is in a partial-commit state. Manual recovery "
                    "required.",
                    operation="save_continuity",
                    path=str(store.continuity_path),
                )
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
                # AM-SNAPSHOT ①: the SAME hash persisted in the wraps row above
                # (computed once near tmp_pair_id) — the audit event and the
                # durable recovery oracle are guaranteed identical.
                "content_hash": content_hash,
            }
            audit_payload["allow_shrink"] = shrink_override
            if drift_results:
                audit_payload["drift"] = _drift_summary(drift_results)
            # Capture Proven-tier pattern omissions in the audit chain.
            # detect_pattern_omissions returns an empty list for the
            # common case (first wrap, or all prior Proven-tier patterns
            # carried forward at some level), in which case we omit
            # the key entirely to keep the routine-case audit entry
            # lean. When omissions DID happen, the audit chain records
            # exactly which graduated patterns disappeared at this
            # wrap — operators and downstream review (Diogenes,
            # consultation, audit-chain queries) can see what was
            # dropped without re-reading prior continuity files.
            if grad_result.level_capped:
                audit_payload["level_capped"] = [
                    asdict(cap) for cap in grad_result.level_capped
                ]
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
            # CAP-08: graduations held back for tool/external-only grounding,
            # and any graduation grounded above the default trust. Lean when
            # empty, like the keys above.
            if grad_result.uncorroborated:
                audit_payload["uncorroborated"] = [
                    {"name": u.name, "level": u.level, "trust": u.trust,
                     "citations": list(u.citations)}
                    for u in grad_result.uncorroborated
                ]
            raised_trust = {
                name: t for name, t in grad_result.pattern_trust.items()
                if t != DEFAULT_TRUST
            }
            if raised_trust:
                audit_payload["pattern_trust"] = raised_trust
            # Durable facts (B1): a drop by marker is recorded here, and only
            # here, so the hash-chained audit log is the trail of every durable
            # line that left the store. Re-insertions ride along, lean when
            # empty like the keys above.
            if durable_report is not None:
                if durable_report.dropped:
                    audit_payload["durable_dropped"] = list(durable_report.dropped)
                if durable_report.reinserted:
                    audit_payload["durable_reinserted"] = list(
                        durable_report.reinserted
                    )
            # ⛔ POST-COMMIT, and routed through the store's shared
            # after-commit helper so this site cannot drift from the other
            # four. Behaviour change: a failed emit now WARNS instead of
            # vanishing silently — a missing entry is a real gap in a
            # tamper-evident record. It still cannot propagate: the wrap is
            # fully committed and success is the only correct outcome.
            store._audit_log_after_commit(
                "continuity_saved",
                audit_payload,
                method="validated_save_continuity",
                committed="the wrap",
            )
        except Exception:
            # Belt-and-suspenders: the helper swallows sink failures, but the
            # payload ASSEMBLY above (grad_result access, hashing) is inside
            # this try as well and must not fail a committed wrap either.
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
        # ⛔ Guarded under the ``_warn_after_commit`` invariant: the wrap has
        # committed and renamed, so a prune failure (a ``StoreDatabaseError``
        # from disk full or a locked DB) must not make the caller believe the
        # save failed. Retention is housekeeping. The message does not claim
        # nothing was pruned: ``Store._db_boundary`` rolls a SQLite failure
        # back, but a failed rollback or an overriding ``prune`` leaves the
        # outcome unknown (codex L3, 2026-09-16). The detail is rendered
        # inside its own guard because an exception's ``__str__`` can raise.
        # Diogenes 2026-09-16, reproduced on 0.9.11.
        try:
            pruned = store.prune()
        except Exception as exc:
            detail = type(exc).__name__
            try:
                detail = f"{detail}: {exc}"
            except Exception:
                pass
            store._record_prune_failure(f"validated_save_continuity: {detail}")
            _warn_after_commit(
                f"Auto-prune failed after the wrap committed; the save "
                f"succeeded, pruned_count is reported as 0 and may undercount "
                f"({detail}). Retention runs again on the next prune."
            )
        else:
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
            # spore-676 ruling (A): True = a bare line held COLD (not grounded within
            # carryforward_cold_days), dated back to its last grounding and flagged.
            "cold": cf.cold,
            # AM-PROVENANCE (Slice A): True = the held line carried a
            # ``[provenance: id, ...]`` grounding-audit marker (a deliberately-
            # grounded mature pattern) → excluded from the graduate-OUT notice.
            "provenance": cf.provenance,
        }
        for cf in grad_result.carried_forward
    ]

    # AM-WARN (v0.4.2): detect the dead-Hebbian-graph mis-wire. Two STRUCTURAL,
    # false-positive-free signals; a wrap with NO graduations at all (a pure
    # state/narrative wrap) stays silent on both.
    #   (A) graduated patterns carried evidence citations but NONE resolved to an
    #       episode in this store (e.g. ids minted in another namespace) -> the
    #       graph cannot form and stays dead. This is the
    #       invisible_infrastructure_failure that ran silent for ~10 wraps.
    #   (B) co-citation pairs WERE available but nothing formed or strengthened
    #       -> the association write path itself is mis-wired.
    # A third signal, (C) AM-LINKGATE (v0.8.3), warned when graduations validated
    # but none offered a co-citation pair. It went QUIET in 0.9.26, when the
    # Hebbian hop was retired and pattern recall stopped reading episode links: a
    # wrap that forms no link is no longer a recall problem, and single-id
    # citation is often the honest one. That case now shows only as
    # ``associations_formed == 0`` and ``associations_strengthened == 0`` in the
    # save result.
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
        grad_result.validated + grad_result.demoted - grad_result.atom_capped
        + cited_carried
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
    cited_superseded = sorted(
        sid for sid in superseded_in_window
        if re.search(r"\[evidence:[^\]]*\b" + re.escape(sid) + r"\b", text, re.IGNORECASE)
    )
    if cited_graduations > 0 and not resolved_any and cited_superseded:
        association_warning = (
            f"{cited_graduations} graduated-pattern citation(s) this wrap cite only "
            f"superseded episodes ({', '.join(cited_superseded)}); a replaced fact is "
            f"not evidence. Cite the episode that replaced it."
        )
    elif cited_graduations > 0 and not resolved_any:
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
    if association_warning is not None:
        _warn_after_commit(association_warning)
    if allow_unlinked is True:
        # After the commit, like every save warning: emitted before it, an
        # error warnings-filter would turn a no-op flag into a failed save.
        _warn_after_commit(
            "allow_unlinked is deprecated and does nothing: the AM-LINKGATE save "
            "refusal it overrode was removed in 0.9.26. Stop passing it "
            "(CLI --allow-unlinked; MCP \"allow_unlinked\")."
        )
    durable_messages: list[str] = []
    if durable_report is not None:
        durable_messages = durable_report_warnings(durable_report) + cue_warnings
        for durable_message in durable_messages:
            _warn_after_commit(durable_message)
    if still_graduating:
        _warn_after_commit(
            f"compost: {still_graduating} also graduated in this wrap's text. "
            f"Their edges were severed and not re-seeded; if the pattern is "
            f"still live, it should not have been composted."
        )

    # AM-CARRYFORWARD (v0.4.6) + AM-PROVENANCE (Slice A): assisted "ground,
    # graduate OUT, or retire" surface for TOP-tier patterns held this wrap.
    # A carry that records ``[provenance: founding-ids]`` is a DELIBERATELY-
    # grounded mature pattern — the audit answer to "why is this 3x?" survives
    # even though its founding episodes consolidated out of the live citation
    # window — so it is NOT on a treadmill and is EXCLUDED here (held silently).
    # What remains are top-tier carries with NEITHER a resolving citation NOR
    # provenance: a genuinely ungrounded permanent claim worth reviewing. Emitted
    # as a UserWarning (mirroring AM-WARN) so the signal is loud, not buried in a
    # return field. Lower-tier carries (2x) are held silently — they are the
    # normal domain-blind-erosion-fix case, not a graduate-out decision.
    #
    # ⚠ THE MESSAGE SAYS "3x OR HIGHER", NOT "Proven-tier", AND THAT IS
    # DELIBERATE. In this library ``MIN_PROVEN_LEVEL`` is 2, so "Proven-tier"
    # means 2x-and-up with no ceiling — a strictly WIDER set than this gate's
    # ``>= 3``, and the comment above already says the 2x carries are held
    # silently. Naming the tier made the sentence enumerate a category it does
    # not deliver: a reader told to review "the Proven-tier carries" would be
    # looking at a list that silently excludes part of that tier. The prior
    # wording before that was "top-tier (3x)", which called a 12x pattern 3x on
    # every wrap. Stating the PREDICATE avoids both, and cannot drift when a
    # tier is renamed or MIN_PROVEN_LEVEL moves.
    #
    # The prior text said "held at level despite an ungrounded CITATION" — wrong
    # for the v0.5.0 bare path (a bare carry never had a citation to be
    # ungrounded). Reworded to "no resolving citation and no provenance," which is
    # accurate for both the bare and the cited-failed paths, and made actionable
    # by naming provenance as fix (a). (This is what an ops entity on low-variance
    # telemetry — e.g. Argus's stable email-classification Provens — hit: the
    # notice fired every wrap on a healthy mature pattern, the cry-wolf failure
    # mode AM-PROVENANCE closes.)
    graduate_out = sorted(
        {
            cf.name
            for cf in grad_result.carried_forward
            if cf.max_level_reached >= 3 and not cf.provenance and not cf.cold
        }
    )
    # spore-676 ruling (A): every bare line held COLD, at any level, reaches the human.
    # The age in days changes every wrap, so a repeat notice still carries news (L2 1007).
    cold_held = sorted({f"{cf.name} ({cf.days_since_grounded} days)"
                        for cf in grad_result.carried_forward if cf.cold})
    if cold_held:
        _warn_after_commit(
            f"{len(cold_held)} pattern(s) were re-dated to today with no evidence but "
            f"have not been grounded in more than {carryforward_cold_days} days: "
            f"{', '.join(cold_held)}. They were HELD at their earned level and dated "
            f"back to their last grounding, not demoted. Decide for each: re-exercise "
            f"it with fresh evidence, graduate it OUT to a stable home, or retire it."
        )
    if graduate_out:
        _warn_after_commit(
            f"{len(graduate_out)} pattern(s) at 3x or higher were carried forward "
            f"this wrap with no resolving citation and no provenance: "
            f"{', '.join(graduate_out)}. A permanent truth held without grounding "
            f"is a candidate to (a) record its founding episode ids as "
            f"`[provenance: id, ...]` so the audit of why it earned its level "
            f"survives the founding episodes ageing out, (b) graduate OUT to a "
            f"stable home (e.g. partnership.md), or (c) retire — review, don't "
            f"leave it on the citation treadmill."
        )

    # AM-PROVENANCE (Slice A, codex L3 HIGH): a today-dated graduation line carrying
    # an [evidence:] tag NON-adjacent to its `| Nx (date)` marker (e.g. a [provenance:]
    # tag wedged between them) misses validation entirely — the evidence forms no link
    # and does not ground the pattern, silently. This is the silent-data-loss class,
    # not a style nit, so surface it LOUD. The line was left unchanged (non-destructive
    # — fix the tag order next wrap, or use [provenance:] alone).
    if grad_result.malformed_evidence_carries:
        malformed = sorted(set(grad_result.malformed_evidence_carries))
        _warn_after_commit(
            f"{len(malformed)} graduation line(s) carry an [evidence:] tag that is "
            f"NOT adjacent to the `| Nx (date)` marker (another tag — e.g. "
            f"[provenance:] — sits between them): {', '.join(malformed)}. That "
            f"evidence will NOT validate or form a Hebbian link. Move [evidence:] "
            f"immediately after the marker, OR use [provenance:] alone (never both "
            f"on one line). The line(s) were left unchanged unless they claimed "
            f"a level above their prior one (then see the level-capped warning)."
        )

    # The prior-state bound cut a line the composer wrote above what the stored
    # prior continuity entitles it to. Loud for the same reason as the line
    # above: the saved text differs from what the composer wrote.
    if grad_result.level_capped:
        _warn_after_commit(
            f"{len(grad_result.level_capped)} pattern line(s) claimed a level "
            f"the prior continuity does not support and were cut "
            f"(a new pattern enters at 1x; a validated Nx becomes (N+1)x): "
            + ", ".join(
                f"{cap.name} {cap.written_level}x->{cap.capped_to}x"
                + (f" ({cap.reason})" if cap.reason == CAP_REASON_REVOKED else "")
                for cap in grad_result.level_capped
            )
            + ". Each is marked (level-capped)."
            + (" A revoked one lost a rung whose grounding episodes were since "
               "lowered to tool/external (CAP-08)."
               if any(c.reason == CAP_REASON_REVOKED for c in grad_result.level_capped)
               else "")
        )

    if grad_result.uncorroborated:
        held_back = sorted({u.name for u in grad_result.uncorroborated})
        _warn_after_commit(
            f"{len(held_back)} graduation(s) did not climb because every citation "
            f"grounding them is a tool or external episode (content relayed, not "
            f"observed): {', '.join(held_back)}. They climb once an agent or "
            f"operator episode also grounds them (CAP-08)."
        )

    result = SaveContinuityResult(
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
        uncorroborated=[asdict(u) for u in grad_result.uncorroborated],
        pattern_trust=dict(grad_result.pattern_trust),
        # Both 0 on a wrap that graduated patterns is the former AM-WARN
        # Signal C case, which went quiet in 0.9.26: these counts are its record.
        associations_formed=assoc_formed,
        associations_strengthened=assoc_strengthened,
        associations_decayed=assoc_decayed,
        association_warning=association_warning,
        linkgate_overridden=False,
        # AM-LINKGATE gauge (spore-721). ``citation_counts`` is filled from
        # ``cited_ids & valid_ids`` on every today-dated graduation line BEFORE
        # the grounding and cross-session checks, so this counts distinct
        # resolved episodes cited, including on lines later demoted.
        citation_spread=len(grad_result.citation_counts),
        supersessions_recorded=supersessions_recorded,
        supersessions_rejected=supersessions_rejected,
        sections=sections,
        # asdict() makes the full return value JSON-serializable
        # top-to-bottom. Library users who want the typed object can
        # do ``WrapResult(**result["wrap_result"])``; everyone else
        # can ``json.dumps(result)`` with no ceremony.
        wrap_result=asdict(wrap_result),
    )
    if compost_names is not None:
        result["composted"] = composted
    if drift_results:
        result["drift"] = _drift_summary(drift_results)
    for message in drift_warnings:
        _warn_after_commit(message)
    if durable_report is not None:
        result["durable_warnings"] = durable_messages
    if grad_result.level_capped:
        result["level_capped"] = [asdict(cap) for cap in grad_result.level_capped]
    if stale_state:
        result["stale_state"] = stale_state
        _warn_after_commit("State lines that do not hold at save: " + "; ".join(stale_state))
    if (
        _frozen_roots != {}
        and _derive_report is not None
        and not _derive_report.enabled
        and has_derive_lines(text, section_schema)
    ):
        # L2 2026-10-01: the composer may have seen checks run at prepare; a save
        # whose roots were all revoked meanwhile checks nothing, and must say so.
        # A wrap that froze no map (None) is included: nothing shows it did not
        # (codex L3 r1).
        _warn_after_commit(
            "Re-derive is not enabled for this store at save, so no State line was "
            "checked at this save (pass require_rederive to refuse such a save)."
        )
    return result


# --- Section reads and writes (P(2), design project_memory/episode_origin_key_design_1010.md r6 §4, §11.3-§11.4, §13) ---


def _has_terminator(line: str) -> bool:
    """Whether a ``splitlines(keepends=True)`` element ends with a line break."""
    return line.splitlines()[0] != line if line else False


def _section_text(lines: list[str]) -> str:
    """Section body lines joined, leading and trailing blank lines removed."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _locate_section(
    raw: str, schema: list[SectionSpec], heading: str
) -> tuple[SectionSpec, list[str], int, int, str] | tuple[str, str]:
    """Find ``heading``'s one section in the raw continuity text.

    Returns ``(spec, raw_lines, header_index, end_index, text)``: ``raw_lines``
    keep their terminators, the section's lines are ``header_index + 1 ..
    end_index - 1``, and ``text`` is the section as :meth:`Store.load_continuity`
    shows it (:func:`_section_text` of the canonical lines). Or ``(reason,
    message)`` when it cannot: ``no_such_section``, ``section_absent``,
    ``ambiguous_heading``.

    Header lines are matched as the gate matches them (``## `` lines, through
    :func:`_header_matches`), on canonical lines. ``str.splitlines`` breaks on
    exactly ``\\n`` plus the characters ``canonical_continuity_text`` turns into
    ``\\n`` (``tests/test_section_edit.py`` holds that by an exhaustive scan), so
    raw line i and canonical line i are the same line.
    """
    want = heading.strip().lower()
    spec = next((sp for sp in schema if sp["heading"].lower() == want), None)
    if spec is None:
        return ("no_such_section", f"{heading!r} is not a section of this store's schema")
    raw_lines = raw.splitlines(keepends=True)
    # Line i of the load_continuity form, by construction (see the docstring).
    canon = [canonical_continuity_text(line.splitlines()[0]) for line in raw_lines]
    target = spec["heading"].lower()
    hits: list[int] = []
    for i, line in enumerate(canon):
        if not line.startswith("## "):
            continue
        matched = _header_matches(line.lower(), schema)
        if target in matched:
            if len(matched) > 1:
                return ("ambiguous_heading", f"header line {i + 1} matches more than one section")
            hits.append(i)
    if not hits:
        return ("section_absent", f"no ## {spec['heading']} section in the continuity file")
    if len(hits) > 1:
        return ("ambiguous_heading", f"{len(hits)} header lines claim ## {spec['heading']}")
    start = hits[0]
    end = next((j for j in range(start + 1, len(canon)) if canon[j].startswith("## ")), len(canon))
    return spec, raw_lines, start, end, _section_text(canon[start + 1:end])


def _read_raw_continuity(store: "Store") -> str | None:
    try:
        with open(store.continuity_path, encoding="utf-8", newline="") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _section_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_section(store: "Store", heading: str) -> tuple[str, str] | None:
    """:meth:`Store.read_section`."""
    raw = _read_raw_continuity(store)
    if raw is None:
        return None
    found = _locate_section(raw, store.section_schema, heading)
    if len(found) == 2:
        reason, message = found  # type: ignore[misc]
        if reason == "section_absent":
            return None
        raise SectionError(reason, message)
    text = found[4]  # type: ignore[misc]
    return text, _section_version(text)


def _pipeline_tmp_present(store: "Store") -> bool:
    parent = store.continuity_path.parent
    return parent.exists() and any(parent.glob(f"{glob.escape(store.continuity_path.stem)}.*.md.tmp"))


def replace_section(
    store: "Store", heading: str, body: str, *, expected_version: str
) -> SectionWriteResult:
    """:meth:`Store.replace_section`."""
    from .origin import canonical_section_markdown

    if not isinstance(heading, str) or not isinstance(body, str):
        raise TypeError("replace_section: heading and body must be str")
    if not isinstance(expected_version, str):
        raise TypeError("replace_section: expected_version must be str")
    schema = store.section_schema

    def refused(reason: str, version: str | None = None) -> SectionWriteResult:
        return SectionWriteResult(outcome="refused", heading=heading, reason=reason, version=version)

    spec = next((sp for sp in schema if sp["heading"].lower() == heading.strip().lower()), None)
    if spec is None:
        return refused("no_such_section")
    if spec["role"] == "graduating":
        return refused("graduating")
    new_body = canonical_section_markdown(body)
    # The span parser's own header test, on the canonical body: a body line it
    # reads as a header would split the section for every reader.
    if any(line.startswith("## ") for line in new_body.split("\n")):
        return refused("invalid_body")

    result: SectionWriteResult
    old_version: str | None = None
    with store.continuity_lock(require=True):
        conn = store._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc) or "busy" in str(exc):
                return refused("store_busy")
            raise
        try:
            if store.get_wrap_started_at():
                result = refused("wrap_in_progress")
            elif _pipeline_tmp_present(store):
                result = refused("pipeline_tmp_present")
            else:
                try:
                    raw = _read_raw_continuity(store)
                except UnicodeDecodeError:
                    raw, found = None, ("unreadable", "the continuity file is not UTF-8")
                else:
                    found = _locate_section(raw, schema, heading) if raw is not None else (
                        "section_absent", "no continuity file")
                if len(found) == 2:
                    result = refused(found[0])  # type: ignore[index]
                else:
                    _, raw_lines, start, end, text = found  # type: ignore[misc]
                    old_version = _section_version(text)
                    if old_version != expected_version:
                        result = SectionWriteResult(
                            outcome="version_mismatch", heading=heading, version=old_version)
                    else:
                        header = raw_lines[start]
                        if not _has_terminator(header):  # the file ended at the header
                            header += "\n"
                        middle: list[str] = []
                        if new_body:
                            middle = ["\n"] + [line + "\n" for line in new_body.split("\n")]
                        if end < len(raw_lines):
                            middle.append("\n")
                        new_raw = "".join(raw_lines[:start] + [header] + middle + raw_lines[end:])
                        after = _locate_section(new_raw, schema, heading)
                        new_canon = canonical_continuity_text(
                            new_raw.replace("\r\n", "\n").replace("\r", "\n"))
                        if (len(after) == 2 or after[4] != new_body  # type: ignore[misc]
                                or not validate_structure(new_canon, schema)):
                            result = refused("invalid_body")
                        else:
                            _write_continuity_in_place(store, new_raw)
                            result = SectionWriteResult(
                                outcome="written", heading=heading,
                                version=_section_version(new_body))
            conn.commit()
        except BaseException:
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            raise
    if result.outcome == "written":
        store._audit_log_after_commit("section_replaced", {
            "heading": spec["heading"],
            "old_version": old_version,
            "new_version": result.version,
        }, method="replace_section", committed="the section edit")
    return result


def _write_continuity_in_place(store: "Store", text: str) -> None:
    """Atomically replace the continuity file with ``text``: a tmp in the same
    directory (named so the wrap-orphan scan never matches it), the original's
    mode, fsync, ``os.replace``, then the directory fsync."""
    path = store.continuity_path
    try:
        mode = os.stat(path).st_mode & 0o7777
    except FileNotFoundError:
        mode = 0o644
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".section-edit")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            os.fchmod(f.fileno(), mode) if hasattr(os, "fchmod") else None
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        _safe_unlink(Path(tmp_name))
        raise
    _fsync_dir(path.parent)
