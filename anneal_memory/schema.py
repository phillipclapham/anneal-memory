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

**Backward compatibility is the load-bearing invariant.** :data:`DEFAULT_SCHEMA`
reproduces the exact pre-0.3.4 four-section behavior, and a store with no
persisted schema falls back to it — so every existing entity (Argus, daemon,
anansi, diogenes, nexus, prism) is byte-for-byte unaffected and needs no
migration.

This module has **no internal dependencies** (imports only ``typing``) so it can
be imported by ``store.py``, ``continuity.py``, and ``graduation.py`` without a
circular-import risk.
"""

from __future__ import annotations

import re
from typing import Literal, TypedDict

__all__ = [
    "SectionRole",
    "SectionSpec",
    "DEFAULT_SCHEMA",
    "FLOW_SCHEMA",
    "DEFAULT_GRADUATING",
    "validate_schema",
    "heading_marker",
    "graduating_headings",
    "required_headings",
    "sections_by_role",
    "default_max_chars",
]

SectionRole = Literal[
    "live-state",
    "graduating",
    "decisions",
    "narrative",
    "narrative-timeless",
    "frozen",
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
    }
)


class SectionSpec(TypedDict):
    """One continuity section: its markdown heading (without the ``## `` prefix)
    and the role that governs how the wrap pipeline treats it."""

    heading: str
    role: SectionRole


# The canonical pre-0.3.4 four-section model. Any store without an explicit
# persisted schema uses this -> identical behavior to <= 0.3.3.
DEFAULT_SCHEMA: list[SectionSpec] = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
]

# flow's partnership schema (the dogfood that drove this feature). It carries
# TWO narrative roles: Context (work-shape, temporal) and Understanding
# (relationship-shape, timeless). Understanding is partnership-entity-only and
# is structurally required by validate_structure when this schema is in force,
# making the felt layer a guarantee rather than a discipline.
FLOW_SCHEMA: list[SectionSpec] = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Active Threads", "role": "live-state"},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
    {"heading": "Understanding", "role": "narrative-timeless"},
]


def validate_schema(schema: object) -> list[SectionSpec]:
    """Validate and normalize a section schema; return a normalized copy.

    Normalization strips surrounding whitespace from headings. Raises
    :class:`ValueError` on:

      - a non-list/tuple or empty schema,
      - an entry that is not a dict or is missing ``heading`` / ``role``,
      - an empty heading,
      - a duplicate heading (case-insensitive — headings index sections),
      - an unknown role,
      - **no ``graduating`` section** (the immune system needs somewhere to run).

    Args:
        schema: Candidate schema — a list of ``{"heading": str, "role": str}``.

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
        seen.add(key)
        if role == "graduating":
            has_graduating = True
        # role is a plain str at runtime but already validated against
        # _VALID_ROLES above; the Literal only narrows for type-checkers.
        # _VALID_ROLES is kept in sync with the SectionRole Literal by
        # test_section_role_literal_matches_valid_roles.
        normalized.append(SectionSpec(heading=heading, role=role))  # type: ignore[typeddict-item]

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

    Every section in the schema is required to be present, in order — which is
    what makes a ``narrative-timeless`` section like ``Understanding`` a
    structural guarantee for partnership entities.
    """
    return [s["heading"] for s in schema]


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
}


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

    Args:
        schema: A normalized section schema (list of :class:`SectionSpec`).

    Returns:
        The derived character budget.
    """
    budget = _BUDGET_BASE
    free = dict(_BUDGET_BASELINE_FREE)
    for section in schema:
        role = section["role"]
        if free.get(role, 0) > 0:
            free[role] -= 1  # covered by the base budget
        else:
            budget += _BUDGET_EXTRA.get(role, 0)
    return budget


# Derived once at import: the default graduating-heading set. graduation.py
# imports this as the backward-compatible default for its scan gate, so a
# library caller that doesn't pass a schema gets exactly the v0.3.3 behavior.
DEFAULT_GRADUATING: frozenset[str] = graduating_headings(DEFAULT_SCHEMA)
