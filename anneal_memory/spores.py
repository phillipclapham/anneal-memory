"""Prospective-intention layer for anneal-memory — typed open cognitive loops.

anneal's episodic + continuity + Hebbian + limbic layers are RETROSPECTIVE
memory: they accrete, compress, and graduate, and never complete. This module
adds the PROSPECTIVE layer — a parallel store of *open cognitive loops* that
MUST resolve. The discriminator from memory is **lifecycle**: memory never
completes; a spore completes.

A spore is one of three types, naming WHAT KIND of openness it is:

  - ``task``     — open *doing*        (descend done/dropped    | ascend project/thread)
  - ``question`` — open *not-knowing*  (descend answered/mooted | ascend episode/pattern)
  - ``thought``  — open *idea*         (descend explored/dropped| ascend essay/pattern/project)

All three share ONE lifecycle::

    plant ──▶ grow (germination tiers, COMPUTED) ──▶ resolve
                                                       │
                            ┌──────────────────────────┴──────────────────────┐
                            ▼                                                   ▼
                      DESCEND (compost)                                   ASCEND (transmute)
                      done/answered/explored/dropped                      → project/pattern/episode/…
                      = the self-clean                                    = the membrane INTO
                                                                            retrospective memory

``ascend`` is the membrane between the prospective and retrospective halves: a
question answered becomes a finding, a thought graduated becomes a pattern. In
**v1 ``ascend`` RECORDS A POINTER** to what the spore became (the ``ref``) — the
actual episode write stays the host's act. (v2 candidate: ``ascend`` auto-writes
the episode into the episodic store.)

Germination (computed at read-time, NEVER stored — parallel to "top of mind"):

  - ``growing``  seen < 3 days ago               — momentum, don't interrupt
  - ``resting``  seen 3–7 days ago               — mention gently
  - ``dormant``  seen > 7 days ago OR past next: — "still alive, or ready to compost?"
  - ``parked``   tier == parked                  — deliberate dormancy, not neglect

Storage: a single JSON document, written atomically (unique-tmp + fsync + rename +
dir-fsync — the same durability idiom as the SQLite store's ``_fsync_dir``). The
prospective set is small and mutable (open loops, frequently re-tiered and
resolved); a JSON document fits that shape where the append-heavy episodic corpus
fits SQLite. Zero dependencies beyond the Python stdlib. Unlike the episodic
store's single-process invariant, this layer is multi-writer-safe: every mutation
serializes under an exclusive ``fcntl`` lock spanning the whole load→mutate→save
(see :meth:`SporeStore._transaction`), since the use-case is parallel sessions +
overnight wrappers planting/resolving against one store. POSIX-only locking.

Dates vs timestamps: ``seen`` / ``next`` / ``created`` / ``resolution.on`` are
``YYYY-MM-DD`` **logical garden dates** in the operator's frame — injectable via
``today`` for deterministic runs, defaulting to ``date.today()``. ``resolution.at``
is an ISO-8601 **UTC** instant (anneal's machine-timestamp convention) — the
precise event time a wrap reads to consume "what ascended *this session*".

Lineage: the Protocol Memory "Seeds" model, lexeme changed (seed → spore) to
avoid colliding with the identity-*seed* that boots an entity. Full philosophy +
the Levain-generalization notes:
``projects/anneal_memory/spores_prospective_layer.md`` (flow repo).
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Literal, Mapping, TypedDict, cast

try:  # POSIX advisory locking; absent on Windows (see SporeStore._transaction).
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

from .origin import _BIDI_CONTROLS, strip_hidden_controls  # noqa: F401 (re-exported)
from .origin import validate_origin_key as _validate_narrow_origin_key
from .store import AnnealMemoryError

SPORE_SCHEMA_VERSION = 1

SporeType = Literal["task", "question", "thought"]
Tier = Literal["hot", "warm", "cold", "parked"]
Germination = Literal["growing", "resting", "dormant", "parked"]
Direction = Literal["descend", "ascend"]
Status = Literal["open", "resolved"]

VALID_TYPES: tuple[SporeType, ...] = ("task", "question", "thought")
VALID_TIERS: tuple[Tier, ...] = ("hot", "warm", "cold", "parked")
VALID_GERMINATIONS: tuple[Germination, ...] = ("growing", "resting", "dormant", "parked")

# The lifecycle is shared across types; the TERMINAL kinds are type-specific —
# descending a ``task`` as "answered" is a nonsensical terminal state. The
# universal neglect-descent ``composted`` is valid for every type.
# Public lookup tables: consumers query these by a runtime spore-``type`` string
# (loaded from JSON, so statically ``str`` — see e.g. a dashboard rendering the
# verbs available for a spore), so the key type is ``str``, not ``SporeType``.
# ``.get(type, frozenset())`` returns the empty set for an unknown type.
DESCEND_BY_TYPE: dict[str, frozenset[str]] = {
    "task": frozenset({"done", "dropped", "composted"}),
    "question": frozenset({"answered", "mooted", "composted"}),
    "thought": frozenset({"explored", "dropped", "composted"}),
}
ASCEND_BY_TYPE: dict[str, frozenset[str]] = {
    "task": frozenset({"project", "thread"}),
    "question": frozenset({"episode", "pattern"}),
    "thought": frozenset({"essay", "pattern", "project"}),
}
ALL_DESCEND_KINDS: tuple[str, ...] = tuple(
    sorted(set().union(*DESCEND_BY_TYPE.values()))
)
ALL_ASCEND_KINDS: tuple[str, ...] = tuple(
    sorted(set().union(*ASCEND_BY_TYPE.values()))
)

# Ranking weights for ``list_open`` / ``surface`` (intent-priority then salience
# then how-alive). Unknown values sort last.
_TIER_ORDER = {"hot": 0, "warm": 1, "cold": 2, "parked": 3}
_GERM_ORDER = {"growing": 0, "resting": 1, "dormant": 2, "parked": 3}


class SporeError(AnnealMemoryError):
    """Prospective-store operational error — corrupt store, unknown id, an id
    that's already resolved, or ambiguous-id drift.

    Inherits :class:`AnnealMemoryError` so a caller catching the library base
    catches spore-store failures in the same boundary as episodic-store ones.

    .. note::
        :class:`AnnealMemoryError` currently lives in ``store.py`` with a
        docstring noting it should move to a dedicated ``exceptions.py`` once a
        non-store-family error exists. :class:`SporeError` IS that first
        non-store-family error — the relocation is a low-risk follow-on, left
        out of this module's introduction to keep its blast radius to one new
        module + the ``__init__`` re-export.
    """


class _Unset:
    """Sentinel for ``update``: distinguishes "argument omitted" (leave
    unchanged) from "set to None/empty" (clear). A bare ``None`` default can't
    express that difference for the clearable fields."""


_UNSET = _Unset()


class ResolutionDict(TypedDict):
    """How a spore closed. ``ref`` is None for descends, the pointer-to-what-it-
    became for ascends. ``on`` is the logical date; ``at`` is the UTC instant."""

    direction: Direction
    kind: str
    ref: str | None
    on: str
    at: str


class SporeDict(TypedDict):
    """The stored shape of a spore. ``germination`` is deliberately ABSENT — it
    is computed from ``seen``/``next`` at read-time via :func:`germination_tier`,
    never persisted (a stored tier would drift the moment the clock moved)."""

    id: str
    type: SporeType
    text: str
    domain: str
    tier: Tier
    salience: int
    seen: str
    next: str | None
    created: str
    status: Status
    resolution: ResolutionDict | None
    pointer: str | None
    notes: list[str]
    # Immutable identity, assigned at creation and never reused (a fresh UUID unless
    # the creator names one). A spore stored before 0.9.42-era stores carries none
    # until the next write transaction backfills it (see SporeStore._transaction).
    origin_key: str
    # NOTE — ``disposition`` is DELIBERATELY NOT a field here. It is an opaque
    # operator-I/O routing tag (the Levain/flow Tray layer's ``seed``/``handoff``/
    # ``agenda`` vs the default ``loop``) that anneal carries verbatim but never
    # interprets — so it stays OUTSIDE the modeled schema (a plain loop is key-free;
    # additive, no schema bump). ``add``/``update`` set/clear it via a loosely-typed
    # dict view, keeping anneal blind to the taxonomy. (Also: ``NotRequired`` is
    # 3.11+, and this library targets 3.10 zero-dep — another reason it isn't a key.)


# ---------------------------------------------------------------------------
# date / value helpers
# ---------------------------------------------------------------------------

def _parse_date(value: object) -> date | None:
    """Parse a ``YYYY-MM-DD`` prefix to a date, else None. Lenient on the read
    path (parses ``value[:10]``) so a hand-edited ``'YYYY-MM-DD <note>'`` still
    reads as its date."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _validate_date(value: str | None, field: str = "next") -> str | None:
    """A write-path date must be exactly ``YYYY-MM-DD``, or ``None``/``''``
    (= unset / clear). Fail loud (``ValueError``) so a typo can't silently
    strand a spore dormant by being stored verbatim then read as no-date."""
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) != 10 or _parse_date(value) is None:
        raise ValueError(f"{field} must be YYYY-MM-DD (got {value!r}).")
    return value


def _safe_int(value: object, default: int = 0) -> int:
    """Coerce a possibly hand-edited / migrated salience to int — never crash a
    read path (``list_open`` / ``surface``) on a bad field."""
    try:
        return int(cast("int", value))
    except (ValueError, TypeError):
        return default


# The errnos that mean "this filesystem has no such flush", as opposed to a flush
# that failed: only these fall back (L3 r1, both lineages).
_FLUSH_UNSUPPORTED = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOTTY})


def _full_fsync(fd: int) -> None:
    """Flush ``fd`` to stable storage. On macOS ``os.fsync`` leaves the data in the
    drive's cache, so ``F_FULLFSYNC`` is used there (measured 10-10: a directory
    ``fsync`` after a rename took no measurable time, ``F_FULLFSYNC`` about 4 ms). A
    filesystem without the call falls back to ``os.fsync``; any other failure raises."""
    full = getattr(fcntl, "F_FULLFSYNC", None) if fcntl is not None else None
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError as exc:
            if exc.errno not in _FLUSH_UNSUPPORTED:
                raise
    os.fsync(fd)


def _open_dir(dir_path: Path) -> int | None:
    """An fd on ``dir_path`` for :func:`_flush_dir`, opened BEFORE the rename so a
    directory that cannot be opened fails the write before it commits (L3 r2: a
    failure after the rename made a committed keyless add look failed, and its
    retry duplicated the spore). None on Windows, which cannot open a directory."""
    if os.name == "nt":
        return None
    return os.open(dir_path, os.O_RDONLY)


def _flush_dir(fd: int | None) -> None:
    """Flush the directory so the rename itself is durable (:func:`_full_fsync`), then
    close ``fd``. A filesystem that cannot flush a directory is skipped; any other
    failure raises, after the rename: the write is in place, so a keyed retry finds it."""
    if fd is None:
        return
    try:
        _full_fsync(fd)
    except OSError as exc:
        if exc.errno not in _FLUSH_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def germination_tier(spore: SporeDict, today: date | None = None) -> Germination:
    """Compute ``growing | resting | dormant | parked`` from ``seen`` / ``next``.

    ``parked`` (``tier == "parked"``) is *deliberate* dormancy, distinct from
    dormant-by-neglect. A ``next:`` on or before ``today`` forces ``dormant``
    regardless of ``seen`` (it asked to be re-surfaced and the day arrived). No
    parseable ``seen`` → ``dormant`` (unknown age).
    """
    if spore.get("tier") == "parked":
        return "parked"
    today = today or date.today()
    nxt = _parse_date(spore.get("next"))
    if nxt and today >= nxt:
        return "dormant"
    seen = _parse_date(spore.get("seen"))
    if seen is None:
        return "dormant"
    age = (today - seen).days
    if age < 3:
        return "growing"
    if age <= 7:
        return "resting"
    return "dormant"


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

# Stored fields a spore's version leaves out: ``seen`` records engagement, not
# content (a touch that clears an elapsed ``next`` still changes the version), and
# ``origin_key`` never changes once set, so its backfill must not stale a read.
SPORE_VERSION_EXCLUDED: frozenset[str] = frozenset({"seen", "origin_key"})



def normalize_spore_field(value: str) -> str:
    """The text a spore stores for ``text``, ``domain`` and ``disposition``, as
    :meth:`SporeStore.add` and :meth:`SporeStore.update` write it: NFC, ``\\r\\n``
    and ``\\r`` as ``\\n``, bidi controls, lone surrogates and every other control character but
    ``\\n`` and ``\\t`` removed, and trailing whitespace stripped from each line and
    from the end. The tag block (U+E0000-E007F, which can spell hidden ASCII; a
    subdivision flag becomes a plain black flag) is removed. Other invisible
    characters (zero-width, variation selectors) are stored as given: no finite
    list removes every one, so showing them is the display's job. Exported so a caller
    can compute the stored value before writing."""
    v = strip_hidden_controls(value.replace("\r\n", "\n").replace("\r", "\n"))
    v = "\n".join(line.rstrip() for line in v.split("\n")).rstrip()
    # Last: a removed character can leave a base and a combining mark adjacent.
    return unicodedata.normalize("NFC", v)


def _is_valid_origin_key(origin_key: object) -> bool:
    return (
        isinstance(origin_key, str)
        and bool(origin_key)
        and origin_key == origin_key.strip()
        and origin_key.isprintable()
    )


def _validate_origin_key(origin_key: object) -> None:
    """A key is compared exactly, so one a copy could alter unseen (padding,
    control or format characters) is refused rather than stored. Spore text keeps
    format characters (:func:`normalize_spore_field`); a key refuses them."""
    if not _is_valid_origin_key(origin_key):
        raise ValueError(
            f"origin_key must be a non-empty printable string without surrounding spaces (got {origin_key!r})."
        )


def spore_version(spore: SporeDict) -> str:
    """The default version of a stored spore: a SHA-256 over every stored field
    except :data:`SPORE_VERSION_EXCLUDED`, as sorted-key compact JSON. Any edit to
    a versioned field changes it; a caller that read a spore passes the version it
    saw as ``expected_version`` to a mutator, which refuses if it no longer matches."""
    payload = {k: v for k, v in spore.items() if k not in SPORE_VERSION_EXCLUDED}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


VersionOf = Callable[[SporeDict], str]


def _check_expected_version(
    item: SporeDict, expected_version: str | None, version_of: VersionOf | None
) -> None:
    """Raise :class:`SporeError` unless the spore's current version equals
    ``expected_version``. Called inside :meth:`SporeStore._transaction`, after the
    spore is loaded under the lock and before anything is changed, so a write that
    lands between a caller's read and this one is refused, never overwritten."""
    if expected_version is None:
        return
    # A copy: the callback must not be able to change what this transaction saves.
    fn = version_of if version_of is not None else spore_version
    found = fn(copy.deepcopy(item))
    if not isinstance(found, str):
        raise TypeError(f"version_of must return a str (got {type(found).__name__}).")
    if found != expected_version:
        raise SporeError(
            f"spore '{item.get('id')}' changed since read (expected version "
            f"{expected_version!r}, found {found!r}); re-read the spore and retry."
        )


class PostconditionFailed(AnnealMemoryError):
    """:meth:`SporeStore.apply` mutated a spore and the result did not hold the
    effect's postcondition; nothing was saved. Not a :class:`SporeError`: a caller
    that retries a store failure must not retry this."""


class ApplyRefused(AnnealMemoryError, ValueError):
    """:meth:`SporeStore.apply` refused the effect itself (a malformed effect, an
    argument the spore cannot take); nothing was written and a retry fails the
    same way. Not a :class:`SporeError`, which is a store that could not be read."""


SporeOp = Literal["add", "update", "descend", "ascend", "delete"]

# The mutator args :meth:`SporeStore.apply` passes through, per op, and the ones it
# requires. ``add_note`` is left out of update on purpose: a date-stamped append no
# leaf can name, which a retried apply would append twice.
_APPLY_ARGS: dict[str, frozenset[str]] = {
    "add": frozenset({"type", "text", "domain", "tier", "salience", "next", "pointer",
                      "disposition", "today"}),
    "update": frozenset({"type", "tier", "next", "text", "salience", "domain", "pointer",
                         "disposition", "expect_disposition"}),
    "descend": frozenset({"kind", "expect_disposition", "today", "now"}),
    "ascend": frozenset({"kind", "ref", "expect_disposition", "today", "now"}),
    "delete": frozenset({"today", "now"}),
}
_APPLY_REQUIRED: dict[str, frozenset[str]] = {
    "add": frozenset({"type", "text"}),
    "descend": frozenset({"kind"}),
    "ascend": frozenset({"kind", "ref"}),
}
# Args that steer the write but are not stored fields, so no leaf names them.
_APPLY_NOT_FIELDS = frozenset({"today", "now", "expect_disposition"})
# The leaves a resolve must carry (and may carry), beside the implied direction.
_RESOLVE_LEAVES: dict[str, frozenset[tuple[str, ...]]] = {
    "descend": frozenset({("status",), ("resolution", "kind")}),
    "ascend": frozenset({("status",), ("resolution", "kind"), ("resolution", "ref")}),
}


@dataclass(frozen=True)
class SporeApply:
    """One effect for :meth:`SporeStore.apply`. ``origin_key`` names the spore for
    every op (its resource identity); ``spore_id``, when given, must be that spore's
    id. ``args`` are the matching public method's keyword args. ``postcondition``
    maps a field name, or a tuple path such as ``("resolution", "kind")``, to the
    value the effect leaves there: exactly the fields the op writes (see
    :meth:`SporeStore.apply`)."""

    op: SporeOp
    origin_key: str
    spore_id: str | None = None
    args: Mapping[str, object] = field(default_factory=dict)
    expected_version: str | None = None
    version_of: VersionOf | None = None
    postcondition: Mapping[str | tuple[str, ...], object] = field(default_factory=dict)


@dataclass(frozen=True)
class SporeApplyResult:
    """What :meth:`SporeStore.apply` did. ``spore`` is the stored spore after the
    call (None when it is deleted); ``version`` is its version as found, which on
    ``precondition_lost`` is the version it changed to (None when there is none)."""

    outcome: Literal["applied", "already", "precondition_lost"]
    spore_id: str | None
    spore: SporeDict | None
    version: str | None


_MISSING = object()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stored_form(path: tuple[str, ...], value: object) -> object:
    """A leaf value as the mutators store it, so a leaf written like the args
    compares equal to the stored field. A cleared ``disposition`` is an absent key."""
    if len(path) != 1:
        return value
    name = path[0]
    if name == "text" and isinstance(value, str):
        return normalize_spore_field(value)
    if name == "domain" and (value is None or isinstance(value, str)):
        return normalize_spore_field(value) if value else ""
    if name == "disposition" and (value is None or isinstance(value, str)):
        return (normalize_spore_field(value) if value else "") or _MISSING
    if name in ("pointer", "next") and (value is None or isinstance(value, str)):
        return value or None
    return value


def _normalize_leaves(
    postcondition: Mapping[str | tuple[str, ...], object],
) -> dict[tuple[str, ...], object]:
    if not isinstance(postcondition, Mapping):
        raise ValueError("postcondition must be a mapping of path to value.")
    leaves: dict[tuple[str, ...], object] = {}
    for path, value in postcondition.items():
        parts = (path,) if isinstance(path, str) else path
        if (not isinstance(parts, tuple) or not parts
                or not all(isinstance(p, str) and p for p in parts)):
            raise ValueError(f"a postcondition path must be a field name or a tuple of them (got {path!r}).")
        try:
            _canonical(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"postcondition value for {parts!r} is not JSON ({exc}).") from exc
        leaves[parts] = _stored_form(parts, value)
    return leaves


def _leaf_value(item: Mapping[str, object], path: tuple[str, ...]) -> object:
    node: object = item
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _same(a: object, b: object) -> bool:
    if a is _MISSING or b is _MISSING:
        return a is b
    return _canonical(a) == _canonical(b)


def _failed_leaves(item: Mapping[str, object], leaves: dict[tuple[str, ...], object]) -> list[str]:
    failed = []
    for path, want in leaves.items():
        got = _leaf_value(item, path)
        if not _same(got, want):
            shown = lambda v: "<absent>" if v is _MISSING else repr(v)  # noqa: E731
            failed.append(f"{'.'.join(path)}: want {shown(want)}, stored {shown(got)}")
    return failed


def _changed_fields(before: Mapping[str, object], after: Mapping[str, object]) -> set[str]:
    return {k for k in set(before) | set(after)
            if not _same(before.get(k, _MISSING), after.get(k, _MISSING))}


def _validate_guards(
    expected_version: object, version_of: object, expect_disposition: object = _UNSET
) -> None:
    """Refuse a malformed or unholdable compare before the lock is taken. Where
    there is no file lock (no ``fcntl``: Windows), neither compare (``expected_version``
    or ``expect_disposition``) can be made atomic with the write, so a guarded write
    is refused rather than checked against a state another process may replace."""
    if version_of is not None and not callable(version_of):
        raise ValueError(f"version_of must be callable (got {version_of!r}).")
    if expected_version is None:
        if version_of is not None:
            raise ValueError("version_of was passed without expected_version; nothing would be compared.")
    elif not isinstance(expected_version, str) or not expected_version:
        raise ValueError(
            f"expected_version must be a non-empty string or None (got {expected_version!r})."
        )
    guarded = expected_version is not None or not isinstance(expect_disposition, _Unset)
    if guarded and fcntl is None:
        raise SporeError(
            "a guarded write (expected_version or expect_disposition) needs a file lock, "
            "which this platform does not have; the compare could not be held through the write."
        )


class SporeStore:
    """A JSON-backed store of open cognitive loops (the prospective layer).

    Mirrors :class:`Store`'s constructor shape — an explicit path, no magic
    default — and its atomic-write durability (unique-tmp + fsync + rename +
    dir-fsync). UNLIKE the episodic :class:`Store` (a documented single-process
    invariant), the prospective layer is inherently MULTI-writer — its use-case
    is several parallel sessions plus overnight wrappers each planting / touching
    / resolving loops against one store. Every mutation therefore runs under an
    exclusive advisory lock spanning the whole load→mutate→save (see
    :meth:`_transaction`), so concurrent writers serialize safely with no lost
    updates or id collisions; reads stay lock-free (atomic ``os.replace`` means a
    reader always sees a complete committed document). POSIX-only locking — see
    :meth:`_transaction` for the non-POSIX degradation.

    Errors: operational failures (corrupt store, unknown id, already-resolved id,
    ambiguous-id drift) raise :class:`SporeError`; malformed caller arguments
    (bad date, unknown type/tier/kind, out-of-range salience) raise
    ``ValueError``. Neither is silently swallowed — a corrupt store NEVER
    silently re-inits (that would overwrite recoverable open loops on next save).
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    # --- io -----------------------------------------------------------------

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {
                "spores": [],
                "resolved": [],
                "schema_version": SPORE_SCHEMA_VERSION,
            }
        except UnicodeDecodeError as e:
            raise SporeError(
                f"{self.path} is not UTF-8 ({e}); refusing to proceed so recoverable "
                f"open loops aren't overwritten — inspect it by hand.") from e
        except json.JSONDecodeError as e:
            raise SporeError(
                f"{self.path} is not valid JSON ({e}); refusing to proceed so "
                f"recoverable open loops aren't overwritten on the next save — "
                f"inspect it (and any .tmp sidecar) by hand."
            ) from e
        if not isinstance(data, dict):
            raise SporeError(
                f"{self.path} must contain a JSON object, got "
                f"{type(data).__name__}."
            )
        data.setdefault("spores", [])
        data.setdefault("resolved", [])
        data.setdefault("schema_version", SPORE_SCHEMA_VERSION)
        if not isinstance(data["spores"], list) or not isinstance(
            data["resolved"], list
        ):
            raise SporeError(
                f"{self.path} is structurally invalid ('spores' and 'resolved' "
                f"must be lists); refusing to proceed so recoverable open loops "
                f"aren't lost — inspect it by hand."
            )
        if not all(isinstance(s, dict) for s in data["spores"]) or not all(
            isinstance(s, dict) for s in data["resolved"]
        ):
            raise SporeError(
                f"{self.path} has non-object rows in 'spores'/'resolved'; "
                f"refusing to proceed — inspect it by hand."
            )
        version = data["schema_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise SporeError(
                f"{self.path} has a non-integer schema_version ({version!r}); "
                f"refusing to proceed — inspect it by hand."
            )
        if version > SPORE_SCHEMA_VERSION:
            raise SporeError(
                f"{self.path} was written by a newer spore schema "
                f"(v{version} > v{SPORE_SCHEMA_VERSION}); refusing to read it so "
                f"fields this version doesn't understand aren't silently dropped "
                f"on the next save."
            )
        return data

    def _save(self, data: dict) -> None:
        target_dir = self.path.parent
        os.makedirs(target_dir, exist_ok=True)
        dir_fd = _open_dir(target_dir)
        # A UNIQUE tmp sibling, never a fixed ``<name>.tmp``: two writers must not
        # collide on one tmp path (the bug that made a fixed tmp's ``os.replace``
        # raise FileNotFoundError under concurrency). Mirrors ``store.py``'s
        # unique-suffix atomic-write idiom. The lock in :meth:`_transaction`
        # already serializes our own writers; the unique tmp also protects against
        # a leftover sidecar and any non-cooperating writer.
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=target_dir, prefix=self.path.name + ".", suffix=".tmp"
            )
        except BaseException:
            if dir_fd is not None:
                os.close(dir_fd)
            raise
        tmp_path = Path(tmp_name)
        try:
            try:
                f = os.fdopen(fd, "w", encoding="utf-8")
            except BaseException:
                os.close(fd)  # fdopen didn't take ownership — close the raw fd
                raise
            with f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                _full_fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            # Never leak the tmp sidecar (or the directory fd) if the write or
            # replace failed.
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if dir_fd is not None:
                os.close(dir_fd)
            raise
        _flush_dir(dir_fd)

    @contextmanager
    def _transaction(self, backfilled: list[int] | None = None) -> Iterator[dict]:
        """Serialize a full load→mutate→save against concurrent processes.

        The prospective layer is inherently MULTI-writer (parallel sessions +
        overnight wrappers each planting / touching / resolving against one
        store), so a mutation takes an **exclusive** advisory lock on a sibling
        ``<store>.lock`` and (re)loads the document INSIDE the lock — so
        :meth:`_next_id` and every field update see the latest committed state,
        with no lost updates and no id collisions. The lock is released when the
        fd closes or the process dies, so a crashed holder can never strand it.
        NOT reentrant — a mutator must never call another mutator (``flock`` is
        per-fd, so the same process re-acquiring would self-deadlock); today no
        mutator does. :meth:`_save` runs only on a clean exit that changed the document; an exception in the
        body (a bad ``kind``, an unknown id) skips the save and releases the lock.

        Reads (:meth:`get` / :meth:`list_open` / :meth:`surface`) stay lock-free:
        :meth:`_save` commits via an atomic ``os.replace``, so a reader always
        sees a complete committed document, never a torn one.

        Platform: ``fcntl`` is POSIX-only and reliable on a LOCAL filesystem. On
        a non-POSIX platform the lock degrades to a no-op (mirroring
        ``store._fsync_dir``'s best-effort Windows behavior); over NFS / a network
        filesystem ``flock`` may also silently no-op. In either case the
        unique-tmp + atomic-replace write still applies, so a *single-process*
        writer is unaffected, but cross-process serialization is guaranteed only
        for a local POSIX filesystem.
        """
        os.makedirs(self.path.parent, exist_ok=True)
        # os.open + flock go INSIDE the try so a flock failure (e.g. ENOLCK on a
        # lock-less FS) can't leak the lock fd: the finally's None-guard covers
        # both "os.open failed (fd still None)" and "flock failed (fd open)".
        lock_fd: int | None = None
        try:
            if fcntl is not None:
                lock_path = self.path.with_name(self.path.name + ".lock")
                lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            data = self._load()
            loaded = json.dumps(data, sort_keys=True)
            n = self._backfill_origin_keys(data)
            if backfilled is not None:
                backfilled.append(n)
            yield data
            # A body that changed nothing (a retried create, a no-op) writes nothing.
            if json.dumps(data, sort_keys=True) != loaded:
                self._save(data)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)

    @staticmethod
    def _needs_key(item: object) -> bool:
        key = item.get("origin_key") if isinstance(item, dict) else None
        return isinstance(item, dict) and (not isinstance(key, str) or not key)

    def _load_keyed(self) -> dict:
        """The document as the lock-free readers return it, with every spore keyed.

        A spore an older anneal appended (0.9.42 edits the document as raw dicts,
        so it keeps every other row's key and only its own new rows lack one) is
        keyed here: the read takes the write transaction, which backfills and
        saves, then reloads. It returns the rows as stored, the unkeyed ones with
        no ``origin_key``, when the file belongs to another user (a read must not
        hand this user the file: ``_save`` writes a new file as the writer) or when
        the write fails with any ``OSError`` (a read-only or full disk, a
        filesystem without locks), so a read never fails where it used to work.
        A caller treats a missing key as "no label resource yet" and never
        invents one. A ``SporeError`` (an unreadable document) propagates
        (design r6 §11.2, §12.4)."""
        data = self._load()
        if not any(self._needs_key(i) for i in data.get("spores", []) + data.get("resolved", [])):
            return data
        try:
            if hasattr(os, "geteuid") and os.stat(self.path).st_uid != os.geteuid():
                return data
            # Return the document the transaction keyed and saved, not a reload:
            # an older writer waiting on the lock could append a keyless row
            # between the release and a second read (L3 r1, codex).
            with self._transaction() as keyed:
                pass
        except OSError:
            return data
        return keyed

    @staticmethod
    def _backfill_origin_keys(data: dict) -> int:
        """Give every stored spore (open and resolved) without an ``origin_key`` a
        fresh one. Runs inside each write transaction, so the first write after an
        upgrade persists them all; returns how many were assigned."""
        n = 0
        for item in list(data.get("spores", [])) + list(data.get("resolved", [])):
            if SporeStore._needs_key(item):
                item["origin_key"] = uuid.uuid4().hex
                n += 1
        return n

    def backfill_origin_keys(self) -> int:
        """Assign an ``origin_key`` to every spore that has none, now, and return
        how many were assigned (0 on a store that already has them all)."""
        backfilled: list[int] = []
        with self._transaction(backfilled):
            pass
        return backfilled[0]

    # --- internal lookups ---------------------------------------------------

    @staticmethod
    def _next_id(data: dict) -> str:
        """Next ``spore-NNN`` id, counted across live, resolved and deleted spores
        so an id is never reused after a spore resolves or is deleted."""
        max_num = 0
        rows = list(data.get("spores", [])) + list(data.get("resolved", []))
        for item in rows + SporeStore._deleted(data):
            try:
                num = int(str(item["id"]).split("-")[1])
            except (IndexError, ValueError, KeyError, TypeError):
                continue
            max_num = max(max_num, num)
        return f"spore-{max_num + 1:03d}"

    @staticmethod
    def _find_open(data: dict, spore_id: str) -> SporeDict | None:
        """The one open spore with ``spore_id``. Two open spores sharing an id is
        store drift: a write would land on one copy while a caller's
        ``expected_version`` may describe the other, so it is refused here, the
        lookup every id-addressed mutator goes through. An id that is open and
        also resolved is refused the same way."""
        matches = [item for item in data.get("spores", []) if item.get("id") == spore_id]
        if len(matches) > 1:
            raise SporeError(
                f"{len(matches)} open spores share id {spore_id!r}; refusing to "
                f"write to an ambiguous id (store drift — repair by hand)."
            )
        if matches and any(r.get("id") == spore_id for r in data.get("resolved", [])):
            raise SporeError(
                f"spore {spore_id!r} is both open and resolved; refusing to write to "
                f"an ambiguous id (store drift — repair by hand)."
            )
        return cast("SporeDict", matches[0]) if matches else None

    @staticmethod
    def _find_by_origin_key(data: dict, origin_key: str) -> SporeDict | None:
        for item in list(data.get("spores", [])) + list(data.get("resolved", [])):
            key = item.get("origin_key") if isinstance(item, dict) else None
            if _is_valid_origin_key(key) and key == origin_key:
                return cast("SporeDict", item)
        return None

    def _require_open(self, data: dict, spore_id: str) -> SporeDict:
        """Return the open spore, or raise — distinguishing already-resolved
        (ids are never reused, so "not found" would mislead for a resolved id)
        from never-existed."""
        item = self._find_open(data, spore_id)
        if item is not None:
            if item.get("status") == "resolved":
                raise SporeError(
                    f"spore '{spore_id}' is in the open set but carries "
                    f"status='resolved' (store drift — repair by hand)."
                )
            return item
        for r in data.get("resolved", []):
            if r.get("id") == spore_id:
                res = r.get("resolution") or {}
                raise SporeError(
                    f"spore '{spore_id}' is already resolved "
                    f"({res.get('direction')}/{res.get('kind')} on "
                    f"{res.get('on')})."
                )
        raise SporeError(f"spore '{spore_id}' not found.")

    @staticmethod
    def _deleted(data: dict) -> list[dict]:
        """The ``deleted`` registry (:meth:`delete`): one ``{id, origin_key, on, at}``
        row per deleted spore, absent from a store that never deleted one."""
        rows = data.get("deleted", [])
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise SporeError(
                "the store's 'deleted' registry is not a list of objects; refusing to "
                "proceed — inspect it by hand.")
        return rows

    def _refuse_deleted_key(self, data: dict, origin_key: str) -> None:
        if any(r.get("origin_key") == origin_key for r in self._deleted(data)):
            raise SporeError(
                f"origin_key {origin_key!r} belonged to a deleted spore; a key is never reused.")

    @staticmethod
    def _find_any(data: dict, spore_id: str) -> SporeDict | None:
        """The one spore with ``spore_id``, open or resolved (the drift refusals of
        :meth:`_find_open` apply), or None."""
        item = SporeStore._find_open(data, spore_id)
        if item is not None:
            return item
        matches = [r for r in data.get("resolved", []) if r.get("id") == spore_id]
        if len(matches) > 1:
            raise SporeError(
                f"{len(matches)} resolved spores share id {spore_id!r}; refusing to "
                f"write to an ambiguous id (store drift — repair by hand).")
        return cast("SporeDict", matches[0]) if matches else None

    def _delete_locked(
        self, data: dict, item: SporeDict, today: date | None, now: datetime | None
    ) -> None:
        """Remove ``item`` from its set and record it in the ``deleted`` registry,
        inside the caller's transaction. The text is not kept."""
        if now is not None and now.tzinfo is None:
            raise ValueError("now must be timezone-aware (got a naive datetime).")
        spore_id = item["id"]
        registry = self._deleted(data)
        # By identity, never by id: under id drift another spore can share the id
        # (L3 r1 codex HIGH).
        data["spores"] = [r for r in data["spores"] if r is not item]
        data["resolved"] = [r for r in data["resolved"] if r is not item]
        registry.append({
            "id": spore_id,
            "origin_key": item.get("origin_key"),
            "on": (today or date.today()).isoformat(),
            "at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(
                timespec="seconds"),
        })
        data["deleted"] = registry

    # --- public API: delete -------------------------------------------------

    def delete(
        self,
        spore_id: str,
        *,
        expected_version: str,
        version_of: VersionOf | None = None,
        origin_key: str | None = None,
        today: date | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Delete a spore, open or resolved, if its version is still
        ``expected_version`` (else :class:`SporeError`, nothing written). For the undo
        of a create: a resolve is the way a loop ends. The spore leaves every set and
        its id and ``origin_key`` go to the store's ``deleted`` registry, so neither is
        used again by this version. ``origin_key``, when given, must be the spore's
        (a stored id can name a different spore than the caller read: see
        :meth:`apply`, which addresses by key). Returns True when this call deleted
        it, False when no stored spore has the id and the registry does; an id that
        never existed raises :class:`SporeError`."""
        if expected_version is None:
            raise ValueError("delete requires expected_version (the version the caller read).")
        _validate_guards(expected_version, version_of)
        if origin_key is not None:
            _validate_origin_key(origin_key)
        with self._transaction() as data:
            item = self._find_any(data, spore_id)
            if item is None:
                if any(r.get("id") == spore_id for r in self._deleted(data)):
                    return False
                raise SporeError(f"spore '{spore_id}' not found.")
            if origin_key is not None and item.get("origin_key") != origin_key:
                raise SporeError(
                    f"spore '{spore_id}' does not carry origin_key {origin_key!r}; "
                    f"re-read the spore and retry.")
            _check_expected_version(item, expected_version, version_of)
            self._delete_locked(data, item, today, now)
            return True

    # --- public API: apply (one effect, typed outcome) ------------------------

    @staticmethod
    def _apply_leaves(effect: SporeApply, args: dict) -> dict[tuple[str, ...], object]:
        """Check the effect's shape and return its leaves, before any lock. The
        leaves must name exactly what the op writes, so an effect cannot be found
        ``already`` on a field it does not write (a vacuous postcondition)."""
        op = effect.op
        if op not in _APPLY_ARGS:
            raise ValueError(f"op must be one of {sorted(_APPLY_ARGS)} (got {op!r}).")
        unknown = sorted(set(args) - _APPLY_ARGS[op])
        if unknown:
            raise ValueError(f"apply {op}: unknown args {unknown}.")
        missing = sorted(_APPLY_REQUIRED.get(op, frozenset()) - set(args))
        if missing:
            raise ValueError(f"apply {op}: missing args {missing}.")
        if args.get("today") is not None and (
                not isinstance(args["today"], date) or isinstance(args["today"], datetime)):
            raise ValueError(f"today must be a date (got {args['today']!r}).")
        if args.get("now") is not None and not isinstance(args["now"], datetime):
            raise ValueError(f"now must be a datetime (got {args['now']!r}).")
        _validate_origin_key(effect.origin_key)
        if effect.spore_id is not None and (not isinstance(effect.spore_id, str) or not effect.spore_id):
            raise ValueError("spore_id must be a non-empty string or None.")
        if op == "add":
            if effect.expected_version is not None:
                raise ValueError("apply add takes no expected_version (its precondition is the key's absence).")
        elif effect.expected_version is None:
            raise ValueError(f"apply {op} requires expected_version.")
        leaves = _normalize_leaves(effect.postcondition)
        paths = set(leaves)
        if op == "delete":
            if paths:
                raise ValueError("apply delete: the postcondition is absence; pass no leaves.")
        elif op in _RESOLVE_LEAVES:
            need = _RESOLVE_LEAVES[op]
            if not need <= paths <= need | {("resolution", "direction")}:
                raise ValueError(
                    f"apply {op}: the postcondition must name exactly {sorted(need)} "
                    f"(and may name resolution.direction); got {sorted(paths)}.")
            leaves[("resolution", "direction")] = op
        else:
            written = {(name,) for name in set(args) - _APPLY_NOT_FIELDS}
            if not written:
                raise ValueError(f"apply {op}: no field is written.")
            if paths != written:
                raise ValueError(
                    f"apply {op}: the postcondition must name exactly the fields written "
                    f"{sorted(written)}; got {sorted(paths)}.")
        return leaves

    def apply(self, effect: SporeApply) -> SporeApplyResult:
        """Apply one effect in one transaction, at most once, with a typed outcome:
        for a caller that decided on a spore's state and must land exactly that
        state (a consent broker's apply and its recovery call this the same way).
        The spore is the one carrying ``effect.origin_key``. Under the lock:

        1. ``already``: the effect's postcondition holds and nothing is written (a
           create: a spore carries the key, open or resolved, or the key belongs to a
           spore since deleted; a delete: the key's spore is deleted; otherwise the
           spore exists and every leaf holds);
        2. ``precondition_lost``: no spore carries the key, it is deleted, an
           update/descend/ascend finds it resolved, or its version is not
           ``expected_version``; nothing is written;
        3. otherwise the mutation runs as the matching public method runs it, and
           the result is checked BEFORE the save: every leaf holds, and no field
           outside the ones the op writes changed (a delete: the spore is in no set
           and exactly one ``deleted`` row records it). It holds → ``applied``; it
           fails → :class:`PostconditionFailed`, nothing saved.

        The postcondition names exactly what the op writes: for add and update the
        fields in ``args``; for descend ``status`` and ``("resolution", "kind")``;
        for ascend also ``("resolution", "ref")``; for delete nothing. The resolve
        direction is implied. Leaves are compared as the fields are stored (text,
        domain and disposition normalised; a cleared domain is ``""``, a cleared
        pointer ``None``, a cleared disposition an absent key) and as canonical JSON
        (``True`` never equals ``1``).

        Raises :class:`ApplyRefused` for an effect the store will not take (a
        malformed effect, a ``spore_id`` that is not the key's spore, an argument
        the spore cannot take), :class:`SporeError` for a store that cannot be read
        (and, without ``fcntl``, as on Windows, for every call), and ``OSError`` for
        a failed lock or save. None of them writes anything, except that an
        ``OSError`` raised by the flush after the rename leaves the write in place
        (a retry finds it ``already``)."""
        try:
            if not isinstance(effect, SporeApply):
                raise TypeError(f"apply takes a SporeApply (got {type(effect).__name__}).")
            args = dict(effect.args)
            leaves = self._apply_leaves(effect, args)
            op = effect.op
            fields = self._prepare_add(**args) if op == "add" else None  # type: ignore[arg-type]
            if op == "ascend":
                ref = args.get("ref")
                if not isinstance(ref, str) or not ref.strip():
                    raise ValueError("ascend requires a ref (what the spore became).")
            _validate_guards(effect.expected_version, effect.version_of,
                             args.get("expect_disposition", _UNSET))
            if fcntl is None:
                raise SporeError("apply needs a file lock, which this platform does not have.")
            with self._transaction() as data:
                return self._apply_locked(data, effect, args, leaves, fields)
        except (ApplyRefused, PostconditionFailed, SporeError):
            raise
        except (ValueError, TypeError) as exc:
            raise ApplyRefused(str(exc)) from exc

    def _apply_locked(
        self,
        data: dict,
        effect: SporeApply,
        args: dict,
        leaves: dict[tuple[str, ...], object],
        fields: dict | None,
    ) -> SporeApplyResult:
        op, key = effect.op, effect.origin_key
        version_of = effect.version_of if effect.version_of is not None else spore_version

        def version(item: SporeDict) -> str:
            found = version_of(copy.deepcopy(item))
            if not isinstance(found, str):
                raise TypeError(f"version_of must return a str (got {type(found).__name__}).")
            return found

        row = self._find_by_origin_key(data, key)
        gone = [r for r in self._deleted(data) if r.get("origin_key") == key]
        named = row if row is not None else (gone[0] if gone else None)
        if named is not None and effect.spore_id is not None and named.get("id") != effect.spore_id:
            raise ValueError(
                f"spore_id {effect.spore_id!r} is not the id of the spore carrying "
                f"origin_key {key!r} ({named.get('id')!r}).")
        if row is not None and self._find_any(data, row["id"]) is not row:
            # _find_any refuses a shared open id; this catches an id shared across sets.
            raise SporeError(
                f"spore id {row['id']!r} is shared by another spore (store drift — repair by hand).")

        if op == "add":
            if row is not None:
                return SporeApplyResult("already", row["id"], row, version(row))
            if gone:  # created, then deleted: the create happened
                return SporeApplyResult("already", gone[0].get("id"), None, None)
            created = self._add_locked(data, cast(dict, fields), key)
            failed = _failed_leaves(created, leaves)
            if created.get("origin_key") != key or created.get("status") != "open":
                failed.append("the created spore does not carry the key, open")
            self._raise_failed(created, failed)
            return SporeApplyResult("applied", created["id"], created, version(created))

        if row is None:
            if op == "delete" and gone:
                return SporeApplyResult("already", gone[0].get("id"), None, None)
            return SporeApplyResult(
                "precondition_lost", gone[0].get("id") if gone else effect.spore_id, None, None)
        spore_id = row["id"]
        if op != "delete" and not _failed_leaves(row, leaves):
            return SporeApplyResult("already", spore_id, row, version(row))
        found = version(row)
        expect_disposition = args.pop("expect_disposition", _UNSET)
        if found != effect.expected_version or (
                op != "delete" and row.get("status") == "resolved") or (
                not isinstance(expect_disposition, _Unset)
                and row.get("disposition") != expect_disposition):
            return SporeApplyResult("precondition_lost", spore_id, row, found)

        before = copy.deepcopy(row)
        if op == "delete":
            self._delete_locked(data, row, args.get("today"), args.get("now"))
            failed = []
            if self._find_by_origin_key(data, key) is not None:
                failed.append("the spore is still stored")
            rows = [r for r in self._deleted(data) if r.get("origin_key") == key]
            if len(rows) != 1 or rows[0].get("id") != spore_id:
                failed.append("the deleted registry does not hold exactly one row for it")
            self._raise_failed(before, failed)
            return SporeApplyResult("applied", spore_id, None, None)

        if op == "update":
            self._update_fields(row, **args)
        elif op == "descend":
            self._descend_locked(data, row, **args)
        else:
            self._ascend_locked(data, row, **args)
        after = self._find_by_origin_key(data, key)
        if after is None:
            self._raise_failed(before, ["the spore is gone after the write"])
        assert after is not None
        failed = _failed_leaves(after, leaves)
        allowed = {path[0] for path in leaves}
        stray = sorted(_changed_fields(before, after) - allowed)
        if stray:
            failed.append(f"fields the effect does not write changed: {stray}")
        if op in _RESOLVE_LEAVES:
            res = after.get("resolution")
            extra = set(res) - {"direction", "kind", "ref", "on", "at"} if isinstance(res, dict) else {"<not an object>"}
            if extra:
                failed.append(f"resolution carries unexpected keys {sorted(extra)}")
            if any(r.get("id") == spore_id for r in data.get("spores", [])):
                failed.append("the resolved spore is still in the open set")
        self._raise_failed(after, failed)
        return SporeApplyResult("applied", spore_id, after, version(after))

    @staticmethod
    def _raise_failed(item: Mapping[str, object], failed: list[str]) -> None:
        if failed:
            raise PostconditionFailed(
                f"spore '{item.get('id')}': the postcondition does not hold "
                f"({'; '.join(failed)}); nothing was saved.")

    # --- public API: plant --------------------------------------------------

    def add(
        self,
        *,
        type: SporeType,
        text: str,
        domain: str = "",
        tier: Tier = "warm",
        salience: int = 0,
        next: str | None = None,
        pointer: str | None = None,
        disposition: str | None = None,
        today: date | None = None,
        origin_key: str | None = None,
    ) -> SporeDict:
        """Plant a new spore. ``pointer`` is an optional link to fuller context for
        the open loop *as it stands* (a project path, a file, a doc) — distinct from
        ``ascend``'s ``ref``, which records what the spore *became* at resolution.
        Returns the created record.

        ``disposition`` is an OPAQUE operator-I/O routing tag (the Levain/flow Tray
        layer's ``seed``/``handoff``/``agenda`` vs the default ``loop``) — anneal
        stores a truthy value verbatim and NEVER interprets it (the disposition-aware
        layer owns the taxonomy + value-validation, mirroring how germination is
        computed outside the store). Pass ``None`` for a normal loop, which stays
        key-free (store-minimal, backward-identical).

        ``origin_key`` is the spore's immutable identity; omitted, a fresh UUID is
        assigned. Planting with a key some stored spore (open or resolved) already
        carries writes nothing and returns that spore, so a retried create lands
        once: the key alone decides, so the earlier spore is returned even if this
        call's fields differ. A key for a NEW spore must meet the origin-key grammar
        (:mod:`anneal_memory.origin`); a retry of a key stored before the grammar
        still returns its spore. ``text``, ``domain`` and ``disposition`` are
        stored as :func:`normalize_spore_field` returns them."""
        fields = self._prepare_add(
            type=type, text=text, domain=domain, tier=tier, salience=salience, next=next,
            pointer=pointer, disposition=disposition, today=today)
        if origin_key is not None:
            _validate_origin_key(origin_key)
        with self._transaction() as data:
            if origin_key is not None:
                existing = self._find_by_origin_key(data, origin_key)
                if existing is not None:
                    return existing
            return self._add_locked(data, fields, origin_key)

    @staticmethod
    def _prepare_add(
        *,
        type: SporeType,
        text: str,
        domain: str = "",
        tier: Tier = "warm",
        salience: int = 0,
        next: str | None = None,
        pointer: str | None = None,
        disposition: str | None = None,
        today: date | None = None,
    ) -> dict:
        """:meth:`add`'s argument checks and normalisation, before any lock: the
        stored values of a new spore, without its id or key."""
        if type not in VALID_TYPES:
            raise ValueError(f"type must be one of {VALID_TYPES} (got {type!r}).")
        if tier not in VALID_TIERS:
            raise ValueError(f"tier must be one of {VALID_TIERS} (got {tier!r}).")
        # bool is an int subclass — reject it explicitly so a stray True/False
        # can't be stored as salience 1/0.
        if not isinstance(salience, int) or isinstance(salience, bool) or not 0 <= salience <= 3:
            raise ValueError(f"salience must be an int 0–3 (got {salience!r}).")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required and must be a non-empty string (the open loop).")
        if not isinstance(domain, str):
            raise ValueError(f"domain must be a string (got {domain!r}).")
        if pointer is not None and not isinstance(pointer, str):
            raise ValueError(f"pointer must be a string or None (got {pointer!r}).")
        # Type-check only — anneal never validates the disposition *value* (it stays
        # blind to the Tray taxonomy); the disposition-aware layer gates the vocabulary.
        if disposition is not None and not isinstance(disposition, str):
            raise ValueError(f"disposition must be a string or None (got {disposition!r}).")
        text = normalize_spore_field(text)
        if not text:
            raise ValueError("text is required and must be a non-empty string (the open loop).")
        domain = normalize_spore_field(domain)
        if disposition is not None:
            disposition = normalize_spore_field(disposition)
        next_validated = _validate_date(next, "next")
        now = (today or date.today()).isoformat()
        return {
            "type": type, "text": text, "domain": domain or "", "tier": tier,
            "salience": salience, "next": next_validated, "now": now,
            "pointer": pointer or None, "disposition": disposition,
        }

    def _add_locked(self, data: dict, fields: dict, origin_key: str | None) -> SporeDict:
        """Create the spore ``fields`` describe, inside the caller's transaction. The
        caller has already returned any spore carrying ``origin_key``."""
        if origin_key is not None:
            self._refuse_deleted_key(data, origin_key)
            # A stored key the wider rule accepted still answers its retry (the
            # caller's lookup); a NEW spore's key must meet the grammar (design r6 §2).
            _validate_narrow_origin_key(origin_key)
        now = fields["now"]
        item: SporeDict = {
            "id": self._next_id(data),
            "type": fields["type"],
            "text": fields["text"],
            "domain": fields["domain"],
            "tier": fields["tier"],
            "salience": fields["salience"],
            "seen": now,
            "next": fields["next"],
            "created": now,
            "status": "open",
            "resolution": None,
            "pointer": fields["pointer"],
            "notes": [],
            "origin_key": origin_key if origin_key is not None else uuid.uuid4().hex,
        }
        # A truthy disposition is stored verbatim; a normal loop stays key-free.
        # Set via a loosely-typed view: disposition is an opaque extra key, not a
        # modeled SporeDict field (see the SporeDict note).
        if fields["disposition"]:
            cast("dict[str, object]", item)["disposition"] = fields["disposition"]
        data["spores"].append(item)
        return item

    # --- public API: read ---------------------------------------------------

    def get(self, spore_id: str) -> SporeDict | None:
        """Fetch a spore by id, searching the open set first then the resolved
        set (open takes precedence if an id somehow appears in both), or None."""
        data = self._load_keyed()
        for item in data.get("spores", []) + data.get("resolved", []):
            if item.get("id") == spore_id:
                return cast("SporeDict", item)
        return None

    def get_by_origin_key(self, origin_key: str) -> SporeDict | None:
        """The spore (open or resolved) carrying ``origin_key``, or None. Takes the
        stored-key rule, wider than the grammar new keys must meet: a key stored
        under the wider rule is found, though it cannot serve where the grammar is
        required (:func:`anneal_memory.origin.origin_key_usable` says which)."""
        _validate_origin_key(origin_key)
        return self._find_by_origin_key(self._load_keyed(), origin_key)

    def list_open(
        self,
        *,
        type: SporeType | None = None,
        tier: Tier | None = None,
        domain: str | None = None,
        germination: Germination | None = None,
        today: date | None = None,
    ) -> list[SporeDict]:
        """Open spores, filtered then ranked (tier → salience desc → germination).

        Germination is NOT written onto the returned dicts (it's computed, never
        stored); call :func:`germination_tier` on a row to annotate it.
        """
        today = today or date.today()
        items = list(self._load_keyed().get("spores", []))
        if type is not None:
            items = [s for s in items if s.get("type") == type]
        if tier is not None:
            items = [s for s in items if s.get("tier") == tier]
        if domain is not None:
            items = [s for s in items if s.get("domain") == domain]
        if germination is not None:
            items = [
                s for s in items
                if germination_tier(cast("SporeDict", s), today) == germination
            ]
        return self._rank(items, today)

    @staticmethod
    def _rank(items: list, today: date) -> list[SporeDict]:
        return sorted(
            (cast("SporeDict", s) for s in items),
            key=lambda s: (
                _TIER_ORDER.get(s.get("tier", ""), 9),
                -_safe_int(s.get("salience")),
                _GERM_ORDER.get(germination_tier(s, today), 9),
            ),
        )

    def surface(
        self, *, top_of_mind: bool = False, today: date | None = None
    ) -> list[SporeDict]:
        """The seed-side surface a salience generator consumes. With
        ``top_of_mind=True``, only the ToM contribution — spores that are ``hot``
        OR ``growing``, ranked, across all three types. Otherwise all open
        spores, ranked. The downstream consumer composes the full Top of Mind
        from this × Active Threads × recent ships."""
        today = today or date.today()
        open_items = list(self._load_keyed().get("spores", []))
        if top_of_mind:
            pool = [
                s for s in open_items
                if s.get("tier") == "hot"
                or germination_tier(cast("SporeDict", s), today) == "growing"
            ]
        else:
            pool = open_items
        return self._rank(pool, today)

    # --- public API: grow ---------------------------------------------------

    def touch(
        self,
        spore_id: str,
        *,
        today: date | None = None,
        expected_version: str | None = None,
        version_of: VersionOf | None = None,
    ) -> SporeDict:
        """Engage a spore: ``seen`` → today, AND clear an elapsed ``next:`` alarm
        (we're looking at it now, so it has fired) — returning the spore to
        ``growing`` rather than leaving a past ``next:`` forcing dormant. (A
        ``parked`` spore stays parked: parked is *deliberate* dormancy, changed via
        ``update(tier=...)``, not by touching.)
        """
        _validate_guards(expected_version, version_of, _UNSET)
        today = today or date.today()
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            _check_expected_version(item, expected_version, version_of)
            item["seen"] = today.isoformat()
            nxt = _parse_date(item.get("next"))
            if nxt and today >= nxt:
                item["next"] = None
            return item

    def update(
        self,
        spore_id: str,
        *,
        type: SporeType | _Unset = _UNSET,
        tier: Tier | _Unset = _UNSET,
        next: str | None | _Unset = _UNSET,
        text: str | _Unset = _UNSET,
        salience: int | _Unset = _UNSET,
        domain: str | _Unset = _UNSET,
        pointer: str | None | _Unset = _UNSET,
        disposition: str | None | _Unset = _UNSET,
        expect_disposition: str | None | _Unset = _UNSET,
        add_note: str | None = None,
        today: date | None = None,
        expected_version: str | None = None,
        version_of: VersionOf | None = None,
    ) -> SporeDict:
        """Metadata surgery on an open spore. Omitted arguments are left
        unchanged; passing ``None``/``''`` to ``next``/``pointer``/``domain``/
        ``disposition`` clears them. Deliberately does NOT bump ``seen`` —
        engagement is signalled explicitly via :meth:`touch`, which keeps
        germination honest.

        ``disposition`` is the opaque operator-I/O routing tag (see :meth:`add`):
        a truthy value re-routes the spore (e.g. the Tray layer's ``seed`` →
        ``handoff``), and ``None``/``''`` clears the key (metabolize back to a
        plain key-free loop). anneal stores/clears it without interpreting the
        value — the disposition-aware layer owns the taxonomy.

        ``type`` retypes the spore (validated against ``VALID_TYPES``). anneal allows
        it mechanically — types ARE anneal's taxonomy — but does NOT enforce the
        "only while forming" policy (a forming Tray item is plastic; a committed loop
        locks its type): that boundary is disposition-aware, so it lives in the
        disposition-aware caller (the Levain/flow write seam gates it), not here.

        ``expect_disposition`` is an OPTIMISTIC compare-and-set on the disposition
        field, checked INSIDE this transaction's lock so a caller that read the
        disposition in a separate (lock-free) step can guard against a concurrent
        change between its read and this write (the cross-process TOCTOU a
        read-then-write guard otherwise has). When provided, the spore's CURRENT
        raw disposition must equal it (``None`` = expect key-absent / a plain loop;
        a string = expect that exact value) or :class:`SporeError` is raised and
        nothing is written. Blind like everything else here — a raw value compare,
        never an interpretation of the tag. (Mirrors the continuity write's
        ``expected``-body stale-check.)

        ``expected_version`` is the same compare over the WHOLE spore: the version
        (``version_of``, default :func:`spore_version`) the caller read must still
        be current, checked under this transaction's lock, or :class:`SporeError`
        is raised and nothing is written. ``touch``, ``descend`` and ``ascend``
        take it too. A caller whose version is its own hash of the record passes
        that hash function as ``version_of``; it gets a copy of the record, runs
        while the lock is held, and must not touch the store. Without ``fcntl``
        (Windows) a guarded write (this or ``expect_disposition``) is refused; on a network filesystem whose
        ``flock`` silently does nothing the compare is not atomic (see
        :meth:`_transaction`).
        """
        _validate_guards(expected_version, version_of, expect_disposition)
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            _check_expected_version(item, expected_version, version_of)

            self._update_fields(
                item, type=type, tier=tier, next=next, text=text, salience=salience,
                domain=domain, pointer=pointer, disposition=disposition,
                expect_disposition=expect_disposition, add_note=add_note, today=today)
            return item

    @staticmethod
    def _update_fields(
        item: SporeDict,
        *,
        type: SporeType | _Unset = _UNSET,
        tier: Tier | _Unset = _UNSET,
        next: str | None | _Unset = _UNSET,
        text: str | _Unset = _UNSET,
        salience: int | _Unset = _UNSET,
        domain: str | _Unset = _UNSET,
        pointer: str | None | _Unset = _UNSET,
        disposition: str | None | _Unset = _UNSET,
        expect_disposition: str | None | _Unset = _UNSET,
        add_note: str | None = None,
        today: date | None = None,
    ) -> None:
        """:meth:`update`'s field changes on a spore already loaded and version-checked
        inside the caller's transaction."""
        if not isinstance(expect_disposition, _Unset):
            # Atomic optimistic-lock: the disposition the caller saw must still be
            # current (raw value compare — None ≡ key-absent). Guards the caller's
            # read-then-write window against a concurrent (cross-process) writer.
            found = item.get("disposition")
            if found != expect_disposition:
                raise SporeError(
                    f"disposition changed since read (expected {expect_disposition!r}, "
                    f"found {found!r}); re-read the spore and retry."
                )

        if not isinstance(type, _Unset):
            if type not in VALID_TYPES:
                raise ValueError(f"type must be one of {VALID_TYPES} (got {type!r}).")
            item["type"] = type
        if not isinstance(tier, _Unset):
            if tier not in VALID_TIERS:
                raise ValueError(f"tier must be one of {VALID_TIERS} (got {tier!r}).")
            item["tier"] = tier
        if not isinstance(next, _Unset):
            item["next"] = _validate_date(next, "next")
        if not isinstance(text, _Unset):
            if not isinstance(text, str) or not normalize_spore_field(text):
                raise ValueError("text must be a non-empty string (cannot clear to empty).")
            item["text"] = normalize_spore_field(text)
        if not isinstance(salience, _Unset):
            if not isinstance(salience, int) or isinstance(salience, bool) or not 0 <= salience <= 3:
                raise ValueError(f"salience must be an int 0–3 (got {salience!r}).")
            item["salience"] = salience
        if not isinstance(domain, _Unset):
            if domain is not None and not isinstance(domain, str):
                raise ValueError(f"domain must be a string or None (got {domain!r}).")
            item["domain"] = normalize_spore_field(domain) if domain else ""
        if not isinstance(pointer, _Unset):
            if pointer is not None and not isinstance(pointer, str):
                raise ValueError(f"pointer must be a string or None (got {pointer!r}).")
            item["pointer"] = pointer or None
        if not isinstance(disposition, _Unset):
            # Type-check only — the value stays uninterpreted by anneal. Set/clear
            # via a loosely-typed view (an opaque extra key, not a SporeDict field).
            if disposition is not None and not isinstance(disposition, str):
                raise ValueError(f"disposition must be a string or None (got {disposition!r}).")
            _view = cast("dict[str, object]", item)
            stored = normalize_spore_field(disposition) if disposition else ""
            if stored:
                _view["disposition"] = stored
            else:  # None / "" → metabolize back to a plain (key-free) loop
                _view.pop("disposition", None)
        if add_note:
            stamp = (today or date.today()).isoformat()
            if not isinstance(item.get("notes"), list):
                item["notes"] = []
            item["notes"].append(f"[{stamp}] {add_note}")


    # --- public API: resolve ------------------------------------------------

    def descend(
        self,
        spore_id: str,
        *,
        kind: str,
        expect_disposition: str | None | _Unset = _UNSET,
        today: date | None = None,
        now: datetime | None = None,
        expected_version: str | None = None,
        version_of: VersionOf | None = None,
    ) -> SporeDict:
        """Resolve a spore downward (compost / self-clean). ``kind`` must fit the
        spore's type (e.g. a ``task`` descends done/dropped/composted, never
        ``answered``).

        ``expect_disposition`` is an OPTIMISTIC compare-and-set on the disposition,
        checked INSIDE this transaction's lock (the SAME primitive as :meth:`ascend`
        and :meth:`update`). A caller whose RESOLVE KIND was chosen against a disposition
        read in a SEPARATE, lock-free step (e.g. a control surface picking "remove" for a
        Keep note vs "compost" for a loop off a rendered SNAPSHOT) passes the raw value it
        saw; if a concurrent writer re-routed the spore between that snapshot and this
        resolve, :class:`SporeError` is raised and nothing is resolved — closing the
        snapshot-then-resolve TOCTOU a separate-transaction guard otherwise has (``None`` =
        expect key-absent / a plain loop; a string = expect that exact value). Blind: a raw
        value compare, never an interpretation of the tag. ``expected_version``
        compares the whole spore the same way (see :meth:`update`)."""
        _validate_guards(expected_version, version_of, expect_disposition)
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            _check_expected_version(item, expected_version, version_of)
            self._descend_locked(data, item, kind, expect_disposition, today, now)
            return item

    def ascend(
        self,
        spore_id: str,
        *,
        kind: str,
        ref: str,
        expect_disposition: str | None | _Unset = _UNSET,
        today: date | None = None,
        now: datetime | None = None,
        expected_version: str | None = None,
        version_of: VersionOf | None = None,
    ) -> SporeDict:
        """Resolve a spore upward (transmute into memory/project — the membrane).
        ``kind`` must fit the spore's type. ``ref`` records WHAT the spore became
        (a project path / episode id / pattern name) — distinct from the spore's own
        ``pointer`` (context for the open loop). v1 records the ref; the actual
        episode write stays the host's act. For fully deterministic tests, pass both
        ``today`` (the logical date) and ``now`` (the UTC instant on ``resolution.at``).

        ``expect_disposition`` is an OPTIMISTIC compare-and-set on the disposition,
        checked INSIDE this transaction's lock (same primitive as :meth:`update`). A
        caller that read the disposition in a SEPARATE, lock-free step (e.g. a host
        enforcing a disposition-aware policy like "a Keep note can't ascend") passes the
        raw value it saw; if a concurrent writer changed it between that read and this
        resolve, :class:`SporeError` is raised and nothing is resolved — closing the
        read-then-resolve TOCTOU a separate-transaction guard otherwise has (``None`` =
        expect key-absent / a plain loop; a string = expect that exact value). Blind: a
        raw value compare, never an interpretation of the tag. ``expected_version``
        compares the whole spore the same way (see :meth:`update`)."""
        if not ref or not ref.strip():
            raise ValueError("ascend requires a ref (what the spore became).")
        _validate_guards(expected_version, version_of, expect_disposition)
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            _check_expected_version(item, expected_version, version_of)
            self._ascend_locked(data, item, kind, ref, expect_disposition, today, now)
            return item

    def _descend_locked(
        self,
        data: dict,
        item: SporeDict,
        kind: str,
        expect_disposition: str | None | _Unset = _UNSET,
        today: date | None = None,
        now: datetime | None = None,
    ) -> None:
        """:meth:`descend` on a spore already loaded and version-checked inside the
        caller's transaction."""
        if not isinstance(expect_disposition, _Unset):
            found = item.get("disposition")
            if found != expect_disposition:
                raise SporeError(
                    f"disposition changed since read (expected {expect_disposition!r}, "
                    f"found {found!r}); re-read the spore and retry."
                )
        valid = DESCEND_BY_TYPE.get(item["type"], frozenset())
        if kind not in valid:
            raise ValueError(
                f"descend kind {kind!r} is invalid for a {item.get('type')!r} "
                f"spore. Valid: {sorted(valid)}."
            )
        self._resolve(data, item, "descend", kind, None, today, now)

    def _ascend_locked(
        self,
        data: dict,
        item: SporeDict,
        kind: str,
        ref: str,
        expect_disposition: str | None | _Unset = _UNSET,
        today: date | None = None,
        now: datetime | None = None,
    ) -> None:
        """:meth:`ascend` on a spore already loaded and version-checked inside the
        caller's transaction."""
        if not isinstance(expect_disposition, _Unset):
            found = item.get("disposition")
            if found != expect_disposition:
                raise SporeError(
                    f"disposition changed since read (expected {expect_disposition!r}, "
                    f"found {found!r}); re-read the spore and retry."
                )
        valid = ASCEND_BY_TYPE.get(item["type"], frozenset())
        if kind not in valid:
            raise ValueError(
                f"ascend kind {kind!r} is invalid for a {item.get('type')!r} "
                f"spore. Valid: {sorted(valid)}."
            )
        self._resolve(data, item, "ascend", kind, ref, today, now)

    @staticmethod
    def _resolve(
        data: dict,
        item: SporeDict,
        direction: Direction,
        kind: str,
        ref: str | None,
        today: date | None,
        now: datetime | None = None,
    ) -> None:
        """Move an open spore to ``resolved`` with a resolution record. Fail loud
        on a duplicate open id rather than dropping both rows (id-equality removal
        would silently nuke the dup). ``at`` is a precise UTC instant so a wrap can
        later consume "what ascended THIS session", not merely this date."""
        spore_id = item["id"]
        # Defence in depth: every public caller already passed _find_open, which
        # refuses both drift shapes below; kept for any future direct caller.
        matches = [s for s in data["spores"] if s.get("id") == spore_id]
        if len(matches) != 1:
            raise SporeError(
                f"{len(matches)} open spores share id {spore_id!r}; refusing to "
                f"resolve an ambiguous id (store drift — repair by hand)."
            )
        if any(r.get("id") == spore_id for r in data["resolved"]):
            raise SporeError(
                f"spore '{spore_id}' already exists in the resolved set (store "
                f"drift — refusing to create a duplicate resolved id)."
            )
        if now is not None:
            if now.tzinfo is None:
                raise ValueError("now must be timezone-aware (got a naive datetime).")
            now = now.astimezone(timezone.utc)
        item["status"] = "resolved"
        item["resolution"] = {
            "direction": direction,
            "kind": kind,
            "ref": ref,
            "on": (today or date.today()).isoformat(),
            "at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        }
        data["spores"] = [s for s in data["spores"] if s.get("id") != spore_id]
        data["resolved"].append(item)
