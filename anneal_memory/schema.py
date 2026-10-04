"""anneal_memory.schema — pluggable continuity section schema (v0.3.4).

A continuity file is organized into ``## ``-headed sections. Through v0.3.3
anneal hardcoded exactly four — State / Patterns / Decisions / Context — in
three coupled places:

  - :func:`anneal_memory.continuity.validate_structure` (which sections are
    *required*),
  - :func:`anneal_memory.continuity._build_wrap_instructions` (what the agent
    is *told to write*),
  - the immune-system graduation gate (``_is_patterns_heading``, which section
    the citation/graduation scan *runs in*).

v0.3.4 makes that mapping a per-:class:`~anneal_memory.store.Store` config: an
ordered list of ``(heading, role)`` pairs. Roles:

  ``live-state``          Volatile current focus. Last-writer-wins; never graduated.
  ``graduating``          The immune system / graduation scan runs here. More than
                          one graduating section is allowed.
  ``decisions``           Committed decisions + rationale.
  ``narrative``           Context-equivalent: compressed *work* narrative (temporal —
                          "what we've been doing").
  ``narrative-timeless``  Relationship-shape: the felt layer (dateless — "who we
                          are together"). Partnership entities only; absent for
                          ops entities, so they carry zero extra weight.
  ``frozen``              Preserved verbatim; never compressed or graduated.
  ``derived-state``       Present-tense claims that prove themselves when read:
                          every line ends with ``[derive: …]`` or ``[judged: …]``
                          (see :mod:`anneal_memory.rederive` and
                          ``docs/rederive.md``). Used by :data:`PROJECT_SCHEMA`.
  ``durable``             Facts that survive by mechanism, one ``- `` line each,
                          carried from wrap to wrap until dropped by marker (see
                          :mod:`anneal_memory.durable`). The section has its own
                          char budget (:func:`durable_budget`) on top of
                          ``max_chars``.

A section may carry ``"optional": True`` (only the ``durable`` role may): it is
not required by ``validate_structure`` and is ignored when matching a schema to
its registered name, so a store that persisted a schema before the section
existed keeps its name and its exact behavior.

**Backward compatibility is the load-bearing invariant.** The PERSISTED schema
is the authority: a store that persisted the default or the partnership schema
before ``Durable Facts`` existed keeps exactly that schema (no durable section)
and behaves as before. :data:`DEFAULT_SCHEMA` adds only the optional
``Durable Facts`` section to the historical four, so a text without it
validates exactly as before; a store with no persisted schema falls back to
:data:`DEFAULT_SCHEMA` and so gets the optional section.

This module has **no internal dependencies** (imports only ``typing``) so it can
be imported by ``store.py``, ``continuity.py``, and ``graduation.py`` without a
circular-import risk.
"""

from __future__ import annotations

import math
import re
from typing import Literal, TypedDict

__all__ = [
    "SectionRole",
    "SectionSpec",
    "DEFAULT_SCHEMA",
    "FLOW_SCHEMA",
    "PROJECT_SCHEMA",
    "SCHEMA_NAMES",
    "schema_by_name",
    "name_for_schema",
    "DEFAULT_GRADUATING",
    "validate_schema",
    "heading_marker",
    "graduating_headings",
    "required_headings",
    "sections_by_role",
    "default_max_chars",
    "HARD_MAX_FACTOR",
    "hard_max_chars",
    "durable_budget",
    "DURABLE_BUDGET_FRACTION",
    "schema_role_warning",
]

SectionRole = Literal[
    "live-state",
    "graduating",
    "decisions",
    "narrative",
    "narrative-timeless",
    "frozen",
    "derived-state",
    "durable",
]

# The roles validate_schema accepts. Kept in sync with SectionRole by the
# test_schema_roles_match_literal regression test.
_VALID_ROLES: frozenset[str] = frozenset(
    {
        "live-state",
        "graduating",
        "decisions",
        "narrative",
        "narrative-timeless",
        "frozen",
        "derived-state",
        "durable",
    }
)


class _SectionSpecRequired(TypedDict):
    heading: str
    role: SectionRole


class SectionSpec(_SectionSpecRequired, total=False):
    """One continuity section: its markdown heading (without the ``## `` prefix)
    and the role that governs how the wrap pipeline treats it.

    ``optional`` (absent = ``False``): the section may be left out of a
    continuity text. Only the ``durable`` role may be optional."""

    optional: bool


# The canonical pre-0.3.4 State/Patterns/Decisions/Context model, plus the
# optional Durable Facts section after State. Any store without an explicit
# persisted schema uses this; a text without Durable Facts validates exactly
# as under <= 0.3.3.
DEFAULT_SCHEMA: list[SectionSpec] = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Durable Facts", "role": "durable", "optional": True},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
]

# flow's partnership schema (the dogfood that drove this feature). It carries
# TWO narrative roles: Context (work-shape, temporal) and Understanding
# (relationship-shape, timeless). Understanding is partnership-entity-only and
# is structurally required by validate_structure when this schema is in force,
# making the felt layer a guarantee rather than a discipline. Durable Facts
# sits after the live-state sections and is optional; a store that persisted
# this schema before that section existed keeps the schema it persisted.
FLOW_SCHEMA: list[SectionSpec] = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Active Threads", "role": "live-state"},
    {"heading": "Durable Facts", "role": "durable", "optional": True},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
    {"heading": "Understanding", "role": "narrative-timeless"},
]

# A project's memory (spore-1230, the perception bundle's Project Memory v2).
# Past claims cite their evidence (Lessons graduate by citation, like
# Patterns); present claims prove themselves when read (State is
# ``derived-state``, re-derived at load by :mod:`anneal_memory.rederive`).
# Plan carries the ``decisions`` role because intent and milestones are
# committed direction; Open is the volatile next-action list.
PROJECT_SCHEMA: list[SectionSpec] = [
    {"heading": "Plan", "role": "decisions"},
    {"heading": "State", "role": "derived-state"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Open", "role": "live-state"},
    {"heading": "Lessons", "role": "graduating"},
    {"heading": "History", "role": "narrative"},
]


# AM-INITSCHEMA: named schemas for CLI / adapter selection. "default" and
# "partnership" ARE the ops-vs-partnership fork (see the entity-architecture
# thesis); "project" is a project's own memory (PROJECT_SCHEMA above):
# "default" = the ops shape (byte-compatible with <= 0.3.3 for a text without
# the optional Durable Facts); the selectable named schema "partnership" =
# :data:`FLOW_SCHEMA` with
# the timeless ``Understanding`` (``narrative-timeless``) felt layer + the
# ``Active Threads`` live-awareness layer. The selection is load-bearing, not
# cosmetic: the felt-layer proportion-gate fires ONLY for a ``narrative-timeless``
# role and :func:`default_max_chars` only grants the felt headroom for the richer
# schema, so a partnership entity whose store is left on ``default`` silently
# runs the ops schema — no felt-gate, under-budgeted. ``init --schema partnership``
# persists the right schema at creation; ``set-schema`` migrates an existing store.
# Module-private so a caller cannot corrupt the shared schema constants through
# the registry — selection goes through schema_by_name (copy-on-return) and
# read-back through name_for_schema; SCHEMA_NAMES is the public name list.
_SCHEMAS_BY_NAME: dict[str, list[SectionSpec]] = {
    "default": DEFAULT_SCHEMA,
    "partnership": FLOW_SCHEMA,
    "project": PROJECT_SCHEMA,
}
# Sorted so the surfaced CLI ``choices=`` order is stable across runs.
SCHEMA_NAMES: tuple[str, ...] = tuple(sorted(_SCHEMAS_BY_NAME))


def schema_by_name(name: str) -> list[SectionSpec]:
    """Resolve a named schema (one of :data:`SCHEMA_NAMES`) to its
    :class:`SectionSpec` list.

    Returns a fresh copy (new list, new section dicts) — this is a public
    adapter entry point, so a caller mutating the result must not corrupt the
    shared module constants (:data:`DEFAULT_SCHEMA` / :data:`FLOW_SCHEMA`).

    Args:
        name: A schema name from :data:`SCHEMA_NAMES`.

    Returns:
        A fresh ``list[SectionSpec]`` for the named schema.

    Raises:
        ValueError: if ``name`` is not a known schema name.
    """
    try:
        schema = _SCHEMAS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown schema name {name!r} (valid: {', '.join(SCHEMA_NAMES)})"
        ) from None
    return [dict(spec) for spec in schema]  # type: ignore[misc]


def _core_pairs(schema: list[SectionSpec]) -> list[tuple[str, str]]:
    """Ordered ``(heading, role)`` pairs of the non-optional sections."""
    return [(s["heading"], s["role"]) for s in schema if s.get("optional") is not True]


def name_for_schema(schema: list[SectionSpec]) -> str | None:
    """Reverse of :func:`schema_by_name`: the registered name whose schema
    matches ``schema`` (by ordered ``(heading, role)`` pairs), or ``None`` for a
    custom/hand-built schema that matches no named one.

    Optional sections are ignored on both sides, so a store that persisted a
    named schema before an optional section was added to it (``default`` or
    ``partnership`` persisted before ``Durable Facts``) still reads back under
    its name.

    Lets a caller answer "is this store on the partnership schema?" without
    importing the constants — the read-back complement to selection.
    """
    target = _core_pairs(schema)
    for nm, sch in _SCHEMAS_BY_NAME.items():
        if _core_pairs(sch) == target:
            return nm
    return None


def validate_schema(schema: object) -> list[SectionSpec]:
    """Validate and normalize a section schema; return a normalized copy.

    Normalization strips surrounding whitespace from headings. Raises
    :class:`ValueError` on:

      - a non-list/tuple or empty schema,
      - an entry that is not a dict or is missing ``heading`` / ``role``,
      - an empty heading,
      - a duplicate heading (case-insensitive — headings index sections),
      - an unknown role,
      - an ``optional`` value that is not a bool, or ``optional: True`` on a
        role other than ``durable``,
      - more than one ``durable`` section,
      - **no ``graduating`` section** (the immune system needs somewhere to run).

    An ``optional`` key is kept in the normalized copy only when the entry
    carries one, so a schema without it normalizes exactly as before.

    Args:
        schema: Candidate schema — a list of ``{"heading": str, "role": str}``,
            optionally with ``"optional": bool``.

    Returns:
        A normalized ``list[SectionSpec]``.
    """
    if not isinstance(schema, (list, tuple)) or len(schema) == 0:
        raise ValueError(
            "section schema must be a non-empty list (or tuple) of sections"
        )

    normalized: list[SectionSpec] = []
    seen: set[str] = set()
    has_graduating = False
    durable_count = 0

    for i, entry in enumerate(schema):
        if not isinstance(entry, dict) or "heading" not in entry or "role" not in entry:
            raise ValueError(
                f"section schema entry {i} must be a dict with 'heading' and "
                f"'role' keys"
            )
        heading = str(entry["heading"]).strip()
        role = str(entry["role"])
        if not heading:
            raise ValueError(f"section schema entry {i}: heading must be non-empty")
        key = heading.lower()
        if key in seen:
            raise ValueError(f"section schema: duplicate heading {heading!r}")
        if role not in _VALID_ROLES:
            raise ValueError(
                f"section schema entry {i}: unknown role {role!r} "
                f"(valid: {', '.join(sorted(_VALID_ROLES))})"
            )
        if "optional" in entry:
            optional = entry["optional"]
            if not isinstance(optional, bool):
                raise ValueError(
                    f"section schema entry {i}: 'optional' must be a bool, got "
                    f"{type(optional).__name__}"
                )
            if optional and role != "durable":
                raise ValueError(
                    f"section schema entry {i}: only a 'durable' section may be "
                    f"optional (got role {role!r})"
                )
        seen.add(key)
        if role == "graduating":
            has_graduating = True
        if role == "durable":
            durable_count += 1
            if durable_count > 1:
                raise ValueError(
                    "section schema: at most one 'durable' section is allowed"
                )
        # role is a plain str at runtime but already validated against
        # _VALID_ROLES above; the Literal only narrows for type-checkers.
        # _VALID_ROLES is kept in sync with the SectionRole Literal by
        # test_section_role_literal_matches_valid_roles.
        spec = SectionSpec(heading=heading, role=role)  # type: ignore[typeddict-item]
        if "optional" in entry:
            spec["optional"] = entry["optional"]
        normalized.append(spec)

    if not has_graduating:
        raise ValueError(
            "section schema must declare at least one 'graduating' section "
            "(the immune system / graduation scan needs a section to run in)"
        )

    # Reject ambiguous schemas: if one heading appears as a word-bounded phrase
    # inside another, a single ``## `` header line could satisfy BOTH required
    # sections in validate_structure (which matches a required heading as a
    # word-bounded substring of the header line). Forbidding it keeps each
    # section satisfiable only by its own header — e.g. "State" + "State Machine",
    # or "Patterns" + "Anti-Patterns", are rejected.
    lowered = [s["heading"].lower() for s in normalized]
    for i, a in enumerate(lowered):
        for j, b in enumerate(lowered):
            if i != j and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", b):
                raise ValueError(
                    f"section schema: heading {normalized[i]['heading']!r} is a "
                    f"word-bounded substring of {normalized[j]['heading']!r}; one "
                    f"header line could satisfy both required sections. Use "
                    f"headings where none contains another."
                )
    return normalized


def heading_marker(heading: str) -> str:
    """The normalized ``## heading`` marker (lowercased) used for matching.

    Matches the comparison :func:`_is_graduating_heading` performs:
    ``line.strip().lower()``. So ``heading_marker("Patterns") == "## patterns"``.
    """
    return f"## {heading.strip().lower()}"


def graduating_headings(schema: list[SectionSpec]) -> frozenset[str]:
    """The set of normalized ``## heading`` markers whose role is ``graduating``.

    This is what the graduation gate (:func:`graduation._is_graduating_heading`)
    is threaded with. For :data:`DEFAULT_SCHEMA` it is ``{"## patterns"}`` —
    identical to the historical hardcoded gate.
    """
    return frozenset(
        heading_marker(s["heading"]) for s in schema if s["role"] == "graduating"
    )


def required_headings(schema: list[SectionSpec]) -> list[str]:
    """The heading texts (without ``## ``) that ``validate_structure`` requires.

    Every non-optional section in the schema is required to be present — which
    is what makes a ``narrative-timeless`` section like ``Understanding`` a
    structural guarantee for partnership entities. An ``optional`` section
    (``Durable Facts``) is not required.
    """
    return [s["heading"] for s in schema if s.get("optional") is not True]


def sections_by_role(schema: list[SectionSpec], role: str) -> list[SectionSpec]:
    """All sections in ``schema`` with the given ``role``, in schema order."""
    return [s for s in schema if s["role"] == role]


# AM-SCHEMA-BUDGET (v0.4.2): per-role budget weights for default_max_chars.
# The historical flat default (20000) was calibrated for DEFAULT_SCHEMA's exact
# section composition. A richer schema (a partnership entity's FLOW_SCHEMA, with
# an extra ``live-state`` "Active Threads" + a ``narrative-timeless``
# "Understanding") carries *incompressible structural* content a flat ceiling
# never budgeted for, so a flat default forces the felt layer to compress to
# fit — backwards. Each section BEYOND the DEFAULT_SCHEMA baseline adds budget
# by role.
_BUDGET_BASE = 20000
# One section of each of these roles is already covered by _BUDGET_BASE (they
# are exactly DEFAULT_SCHEMA), so the first of each adds nothing.
_BUDGET_BASELINE_FREE: dict[str, int] = {
    "live-state": 1,
    "graduating": 1,
    "decisions": 1,
    "narrative": 1,
}
# Budget added per section beyond the baseline, by role.
_BUDGET_EXTRA: dict[str, int] = {
    "live-state": 1500,          # e.g. flow's Active Threads (beyond State)
    "graduating": 2500,          # additional graduating sections
    "decisions": 1500,
    "narrative": 2500,           # additional work-narrative sections
    "narrative-timeless": 4000,  # the felt floor (Understanding) — incompressible
    "frozen": 1000,              # preserved verbatim
    "derived-state": 1500,       # annotated State (the trailers cost chars)
}


# The durable section's own budget, as a fraction of the store's max_chars. At
# the default 20000 that is 3000 chars: room for roughly 25-35 one-line facts
# without crowding the narrative sections, which keep all of max_chars because
# the durable budget sits on top of it rather than inside it.
DURABLE_BUDGET_FRACTION = 0.15


def durable_budget(max_chars: int) -> int:
    """The ``durable`` section's own char budget for a continuity whose other
    sections are budgeted ``max_chars``: :data:`DURABLE_BUDGET_FRACTION` of it,
    on top of it. Over this budget a save warns and keeps every line."""
    return int(max_chars * DURABLE_BUDGET_FRACTION)


def schema_durable_budget(schema: list[SectionSpec]) -> int:
    """The ``durable`` section's budget for ``schema``: :func:`durable_budget` of
    the schema's :func:`default_max_chars`, never of a caller's ``max_chars``.
    One function for the budget the wrap guidance shows and the budget the save's
    durable warning checks, so a composer is warned against the number it was
    shown (Diogenes 2026-10-04: the guidance used the caller's max_chars)."""
    return durable_budget(default_max_chars(schema))


def default_max_chars(schema: list[SectionSpec]) -> int:
    """Derive a default continuity-size budget (chars) from a schema's roles.

    The historical flat 20000 default was calibrated for :data:`DEFAULT_SCHEMA`
    (ops: State / Patterns / Decisions / Context). A richer schema needs more
    room or a flat ceiling forces the felt layer to compress: :data:`FLOW_SCHEMA`
    adds ``Active Threads`` (an extra ``live-state``) + ``Understanding``
    (``narrative-timeless`` — the incompressible felt floor, proportion-checked
    against the full arc). Each section *beyond* the DEFAULT_SCHEMA baseline
    composition adds budget by role.

    :data:`DEFAULT_SCHEMA` -> exactly ``20000`` (the byte-compatible invariant);
    :data:`FLOW_SCHEMA` -> larger. An explicit ``max_chars`` passed to
    ``prepare_wrap`` always overrides this default.

    The ``durable`` section is not counted here: it has its own budget,
    :func:`durable_budget` of this value, added on top, so its presence never
    squeezes the other sections and a schema gains or loses it without
    changing this number.

    Args:
        schema: A normalized section schema (list of :class:`SectionSpec`).

    Returns:
        The derived character budget.
    """
    budget = _BUDGET_BASE
    free = dict(_BUDGET_BASELINE_FREE)
    for section in schema:
        role = section["role"]
        if role == "durable":
            continue  # budgeted separately by durable_budget()
        if free.get(role, 0) > 0:
            free[role] -= 1  # covered by the base budget
        else:
            budget += _BUDGET_EXTRA.get(role, 0)
    return budget


# The save-time ceiling, as a multiple of the schema's own target. The target
# (:func:`default_max_chars`) is what a composer is ASKED to stay within; this is
# where a save is REFUSED. The factor is the smallest round one that clears the
# largest save measured across the stores (rationale and the measurement:
# ``project_memory/continuity_max_design_1003.md``, CHANGELOG 0.9.33).
HARD_MAX_FACTOR = 1.25


def hard_max_chars(schema: list[SectionSpec]) -> int:
    """The size above which a save is refused: ``ceil(1.25 * default_max_chars)``.

    Derived from the SCHEMA only, never from a ``max_chars`` passed to
    ``prepare_wrap`` (that argument moves the compose target, not this bound), so a
    reader such as ``levain doctor`` can compute the same number from the same
    schema. The ``durable`` section is outside it, as it is outside the target (it
    has its own budget, :func:`durable_budget`).
    """
    return math.ceil(HARD_MAX_FACTOR * default_max_chars(schema))


def schema_role_warning(schema: list[SectionSpec]) -> str | None:
    """AM-ROLECHECK (v0.5.0): warn when a schema's ROLE assignment would
    silently thin the ``prepare_wrap`` package.

    The wrap package is built conditionally on section ROLES, not headings
    (``continuity._build_wrap_instructions``): the pattern-line / immune format
    and the contradiction scan emit only for a ``graduating`` role; the felt
    proportion-check only for ``narrative-timeless``; the failure-mode
    discipline only for a ``narrative`` role. A schema that VALIDATES
    (:func:`validate_schema` passes) but assigns the WRONG role to a section
    produces a silently thinner package — and the v0.3.5 catastrophic-shrink
    gate only refuses CORRUPT schemas, never a valid-but-mis-roled one. Once an
    entity trusts the generator and drops its own static reference, nothing
    lets it notice the generator under-delivered (spore-013, anansi L3).

    Returns a single human-readable warning string, or ``None`` when the schema
    is a known-good named schema OR a benign custom schema. Two checks,
    most-specific first:

    1. **Role drift off a named schema** — the schema's heading SET exactly
       matches a named schema (:data:`SCHEMA_NAMES`) but a heading is assigned
       a DIFFERENT role than that named schema gives it. The precise "I
       hand-built a partnership-shaped schema and mis-roled a section" case
       (e.g. ``Understanding`` as ``narrative`` not ``narrative-timeless`` →
       loses the felt proportion-check AND the shrink-gate protection; or
       ``Patterns`` as ``narrative`` not ``graduating`` → loses the immune
       system). A mere REORDERING of a named schema's sections is NOT flagged
       (roles unchanged) — that's a legitimate custom profile.
    2. **No graduating section** — zero ``graduating`` roles means the immune
       system, the pattern-line format, and the contradiction scan are never
       emitted and patterns can never graduate. This is a BACKSTOP only: the
       canonical store path already HARD-REFUSES a 0-graduating schema at
       persist time (:func:`validate_schema` raises), so this check fires only
       for a caller passing a raw/unvalidated schema directly to this function.
       It never false-fires on a validated schema (which always has graduating).

    Note the division of labour: :func:`validate_schema` refuses the CORRUPT
    and graduating-LESS cases at persist; the v0.3.5 shrink gate refuses
    catastrophic felt collapse at save. AM-ROLECHECK covers the gap between
    them — a schema that VALIDATES and KEEPS a graduating section but is
    mis-roled elsewhere (the prepare-reachable, load-bearing case is
    ``narrative-timeless`` → ``narrative``, which silently drops the felt
    proportion-check and the shrink-gate's felt protection while passing every
    other guard).

    Like AM-WARN (``association_warning``), this WARNS — it does not refuse; the
    signal is loud (a ``UserWarning`` at ``prepare_wrap``) and structured (the
    ``schema_warning`` field on ``PrepareWrapResult``).

    **Known limitation (deliberate):** check 1 only catches a mis-role on a
    schema whose heading SET matches a NAMED schema. A fully CUSTOM partnership
    schema — novel headings AND a felt section mis-roled ``narrative`` instead
    of ``narrative-timeless`` — is NOT caught (there is no named reference to
    drift from). A role-shape heuristic was considered and rejected: it
    false-fires on legitimate ops-plus-extra schemas. This boundary should be
    revisited when a trusted-single-operator / Understanding-at-primacy profile
    (AM-CHIPSCHEMA, deferred to v2) ships, since that lands more custom
    partnership-shaped schemas.
    """
    # A known-good named schema (exact ordered heading+role match) is correct
    # by construction — never warn (no false positives for default/partnership).
    if name_for_schema(schema) is not None:
        return None

    # Check 1: role drift off a named schema with the SAME heading set. The
    # named schemas have unique headings, so a heading->role dict is lossless
    # for them; the drift scan iterates the candidate's sections in order so a
    # candidate with a duplicate heading is still checked entry-by-entry.
    target_headings = [s["heading"].strip().lower() for s in schema]
    target_heading_set = set(target_headings)
    for nm, named in _SCHEMAS_BY_NAME.items():
        named_role_by_heading = {
            s["heading"].strip().lower(): s["role"] for s in named
        }
        # The named schema's sections with and without its optional ones: a
        # schema persisted before an optional section existed still has the
        # same sections as its named schema.
        named_core = {
            s["heading"].strip().lower()
            for s in named
            if s.get("optional") is not True
        }
        if named_core <= target_heading_set <= set(named_role_by_heading):
            drifts = [
                (s["heading"], s["role"], named_role_by_heading[h])
                for s, h in zip(schema, target_headings)
                if s["role"] != named_role_by_heading[h]
            ]
            if drifts:
                detail = "; ".join(
                    f"section {heading!r} is role {actual!r} where the "
                    f"{nm!r} schema assigns {expected!r}"
                    for heading, actual, expected in drifts
                )
                return (
                    f"Section schema has the same sections as the {nm!r} schema "
                    f"but different role assignments: {detail}. This changes what "
                    f"prepare_wrap emits (the immune/pattern-line format keys off "
                    f"the 'graduating' role; the felt proportion-check off "
                    f"'narrative-timeless') — the package may be silently thinner "
                    f"than the {nm!r} schema intends. Re-check the role on the "
                    f"section(s) above, or ignore if the divergence is deliberate."
                )

    # Check 2 (universal floor): a schema with no graduating section silently
    # loses the immune system, the pattern-line format, and the contradiction
    # scan, and no pattern can graduate.
    if not any(s["role"] == "graduating" for s in schema):
        return (
            "Section schema has no 'graduating' section — the immune system, the "
            "pattern-line format, and the contradiction scan will NOT be emitted "
            "into the wrap package, and patterns cannot graduate (1x->2x->3x->..., "
            "no top rung). If "
            "this store is meant to develop graduated principles, mark a section "
            "with role 'graduating' (e.g. 'Patterns')."
        )

    return None


# Derived once at import: the default graduating-heading set. graduation.py
# imports this as the backward-compatible default for its scan gate, so a
# library caller that doesn't pass a schema gets exactly the v0.3.3 behavior.
DEFAULT_GRADUATING: frozenset[str] = graduating_headings(DEFAULT_SCHEMA)
