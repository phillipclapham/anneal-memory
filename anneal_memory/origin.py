"""Origin keys (P(2), design
``project_memory/episode_origin_key_design_1010.md`` r6).

An origin key is an opaque, immutable id a row carries from creation and never
reuses. One grammar holds for every origin key anneal stores or accepts as an
effect id: ASCII letters, digits and ``. _ : -``, 1 to 128 characters.
:func:`origin_key_usable` is the Python rule; :data:`ORIGIN_KEY_CHECK_SQL` is the
same rule as a SQLite column CHECK. ``tests/test_origin.py`` holds the two to one
fixture.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "ORIGIN_KEY_CHECK_SQL",
    "ORIGIN_KEY_MAX_LEN",
    "origin_key_usable",
    "strip_hidden_controls",
    "validate_origin_key",
]

ORIGIN_KEY_MAX_LEN = 128
_ORIGIN_KEY_RE = re.compile(r"[0-9A-Za-z._:-]{1,%d}" % ORIGIN_KEY_MAX_LEN, re.ASCII)

# The column CHECK, in SQLite's spelling of the rule above. ``typeof`` refuses a
# BLOB, ``instr`` a NUL (``length`` stops counting at one), the GLOB every other
# character outside the class.
ORIGIN_KEY_CHECK_SQL = (
    "origin_key IS NULL OR ("
    "typeof(origin_key) = 'text' AND instr(origin_key, char(0)) = 0 "
    f"AND length(origin_key) BETWEEN 1 AND {ORIGIN_KEY_MAX_LEN} "
    "AND origin_key NOT GLOB '*[^-0-9A-Za-z._:]*')"
)


# Format characters that reorder displayed text (the "Trojan Source" class): the
# embeddings, overrides and isolates, and the implicit marks.
_BIDI_CONTROLS = frozenset(
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c"
)


def strip_hidden_controls(value: str) -> str:
    """Remove bidi controls, lone surrogates, the tag block (U+E0000-E007F) and
    every control character but ``\\n`` and ``\\t``. The one filter the spore
    fields (:func:`~anneal_memory.spores.normalize_spore_field`) and the remote
    capture path share."""
    return "".join(
        ch for ch in value
        if ch in "\n\t" or (ch not in _BIDI_CONTROLS and not "\U000e0000" <= ch <= "\U000e007f" and unicodedata.category(ch) not in ("Cc", "Cs"))
    )


def origin_key_usable(key: object) -> bool:
    """True when ``key`` is in the origin-key grammar."""
    return isinstance(key, str) and _ORIGIN_KEY_RE.fullmatch(key) is not None


def validate_origin_key(key: object, *, what: str = "origin_key") -> str:
    """Return ``key`` when it is in the grammar; raise ``ValueError`` otherwise."""
    if not origin_key_usable(key):
        raise ValueError(
            f"{what} must be 1-{ORIGIN_KEY_MAX_LEN} ASCII letters, digits or '._:-' (got {key!r})."
        )
    return key  # type: ignore[return-value]
