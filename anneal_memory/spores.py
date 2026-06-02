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

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, TypedDict, cast

try:  # POSIX advisory locking; absent on Windows (see SporeStore._transaction).
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

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
DESCEND_BY_TYPE: dict[SporeType, frozenset[str]] = {
    "task": frozenset({"done", "dropped", "composted"}),
    "question": frozenset({"answered", "mooted", "composted"}),
    "thought": frozenset({"explored", "dropped", "composted"}),
}
ASCEND_BY_TYPE: dict[SporeType, frozenset[str]] = {
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


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort POSIX directory fsync so the rename itself is durable. Mirrors
    the episodic store's ``_fsync_dir``; on macOS ``os.fsync`` is weaker than
    ``F_FULLFSYNC`` — a documented platform limit, not a bug here."""
    try:
        fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


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
        # A UNIQUE tmp sibling, never a fixed ``<name>.tmp``: two writers must not
        # collide on one tmp path (the bug that made a fixed tmp's ``os.replace``
        # raise FileNotFoundError under concurrency). Mirrors ``store.py``'s
        # unique-suffix atomic-write idiom. The lock in :meth:`_transaction`
        # already serializes our own writers; the unique tmp also protects against
        # a leftover sidecar and any non-cooperating writer.
        fd, tmp_name = tempfile.mkstemp(
            dir=target_dir, prefix=self.path.name + ".", suffix=".tmp"
        )
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
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            # Never leak the tmp sidecar if the write or replace failed.
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        _fsync_dir(target_dir)

    @contextmanager
    def _transaction(self) -> Iterator[dict]:
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
        mutator does. :meth:`_save` runs only on a clean exit; an exception in the
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
            yield data
            self._save(data)
        finally:
            if lock_fd is not None:
                os.close(lock_fd)

    # --- internal lookups ---------------------------------------------------

    @staticmethod
    def _next_id(data: dict) -> str:
        """Next ``spore-NNN`` id, counted across BOTH live and resolved items so
        an id is never reused after a spore resolves."""
        max_num = 0
        for item in list(data.get("spores", [])) + list(data.get("resolved", [])):
            try:
                num = int(str(item["id"]).split("-")[1])
            except (IndexError, ValueError, KeyError, TypeError):
                continue
            max_num = max(max_num, num)
        return f"spore-{max_num + 1:03d}"

    @staticmethod
    def _find_open(data: dict, spore_id: str) -> SporeDict | None:
        for item in data.get("spores", []):
            if item.get("id") == spore_id:
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
        today: date | None = None,
    ) -> SporeDict:
        """Plant a new spore. ``pointer`` is an optional link to fuller context for
        the open loop *as it stands* (a project path, a file, a doc) — distinct from
        ``ascend``'s ``ref``, which records what the spore *became* at resolution.
        Returns the created record."""
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
        next_validated = _validate_date(next, "next")
        now = (today or date.today()).isoformat()
        with self._transaction() as data:
            item: SporeDict = {
                "id": self._next_id(data),
                "type": type,
                "text": text,
                "domain": domain or "",
                "tier": tier,
                "salience": salience,
                "seen": now,
                "next": next_validated,
                "created": now,
                "status": "open",
                "resolution": None,
                "pointer": pointer or None,
                "notes": [],
            }
            data["spores"].append(item)
            return item

    # --- public API: read ---------------------------------------------------

    def get(self, spore_id: str) -> SporeDict | None:
        """Fetch a spore by id, searching the open set first then the resolved
        set (open takes precedence if an id somehow appears in both), or None."""
        data = self._load()
        for item in data.get("spores", []) + data.get("resolved", []):
            if item.get("id") == spore_id:
                return cast("SporeDict", item)
        return None

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
        items = list(self._load().get("spores", []))
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
        open_items = list(self._load().get("spores", []))
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

    def touch(self, spore_id: str, *, today: date | None = None) -> SporeDict:
        """Engage a spore: ``seen`` → today, AND clear an elapsed ``next:`` alarm
        (we're looking at it now, so it has fired) — returning the spore to
        ``growing`` rather than leaving a past ``next:`` forcing dormant. (A
        ``parked`` spore stays parked: parked is *deliberate* dormancy, changed via
        ``update(tier=...)``, not by touching.)
        """
        today = today or date.today()
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            item["seen"] = today.isoformat()
            nxt = _parse_date(item.get("next"))
            if nxt and today >= nxt:
                item["next"] = None
            return item

    def update(
        self,
        spore_id: str,
        *,
        tier: Tier | _Unset = _UNSET,
        next: str | None | _Unset = _UNSET,
        text: str | _Unset = _UNSET,
        salience: int | _Unset = _UNSET,
        domain: str | _Unset = _UNSET,
        pointer: str | None | _Unset = _UNSET,
        add_note: str | None = None,
        today: date | None = None,
    ) -> SporeDict:
        """Metadata surgery on an open spore. Omitted arguments are left
        unchanged; passing ``None``/``''`` to ``next``/``pointer``/``domain``
        clears them. Deliberately does NOT bump ``seen`` — engagement is signalled
        explicitly via :meth:`touch`, which keeps germination honest.
        """
        with self._transaction() as data:
            item = self._require_open(data, spore_id)

            if not isinstance(tier, _Unset):
                if tier not in VALID_TIERS:
                    raise ValueError(f"tier must be one of {VALID_TIERS} (got {tier!r}).")
                item["tier"] = tier
            if not isinstance(next, _Unset):
                item["next"] = _validate_date(next, "next")
            if not isinstance(text, _Unset):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("text must be a non-empty string (cannot clear to empty).")
                item["text"] = text
            if not isinstance(salience, _Unset):
                if not isinstance(salience, int) or isinstance(salience, bool) or not 0 <= salience <= 3:
                    raise ValueError(f"salience must be an int 0–3 (got {salience!r}).")
                item["salience"] = salience
            if not isinstance(domain, _Unset):
                if domain is not None and not isinstance(domain, str):
                    raise ValueError(f"domain must be a string or None (got {domain!r}).")
                item["domain"] = domain or ""
            if not isinstance(pointer, _Unset):
                if pointer is not None and not isinstance(pointer, str):
                    raise ValueError(f"pointer must be a string or None (got {pointer!r}).")
                item["pointer"] = pointer or None
            if add_note:
                stamp = (today or date.today()).isoformat()
                if not isinstance(item.get("notes"), list):
                    item["notes"] = []
                item["notes"].append(f"[{stamp}] {add_note}")

            return item

    # --- public API: resolve ------------------------------------------------

    def descend(
        self,
        spore_id: str,
        *,
        kind: str,
        today: date | None = None,
        now: datetime | None = None,
    ) -> SporeDict:
        """Resolve a spore downward (compost / self-clean). ``kind`` must fit the
        spore's type (e.g. a ``task`` descends done/dropped/composted, never
        ``answered``)."""
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            valid = DESCEND_BY_TYPE.get(item["type"], frozenset())
            if kind not in valid:
                raise ValueError(
                    f"descend kind {kind!r} is invalid for a {item.get('type')!r} "
                    f"spore. Valid: {sorted(valid)}."
                )
            self._resolve(data, item, "descend", kind, None, today, now)
            return item

    def ascend(
        self,
        spore_id: str,
        *,
        kind: str,
        ref: str,
        today: date | None = None,
        now: datetime | None = None,
    ) -> SporeDict:
        """Resolve a spore upward (transmute into memory/project — the membrane).
        ``kind`` must fit the spore's type. ``ref`` records WHAT the spore became
        (a project path / episode id / pattern name) — distinct from the spore's own
        ``pointer`` (context for the open loop). v1 records the ref; the actual
        episode write stays the host's act. For fully deterministic tests, pass both
        ``today`` (the logical date) and ``now`` (the UTC instant on ``resolution.at``)."""
        if not ref or not ref.strip():
            raise ValueError("ascend requires a ref (what the spore became).")
        with self._transaction() as data:
            item = self._require_open(data, spore_id)
            valid = ASCEND_BY_TYPE.get(item["type"], frozenset())
            if kind not in valid:
                raise ValueError(
                    f"ascend kind {kind!r} is invalid for a {item.get('type')!r} "
                    f"spore. Valid: {sorted(valid)}."
                )
            self._resolve(data, item, "ascend", kind, ref, today, now)
            return item

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
