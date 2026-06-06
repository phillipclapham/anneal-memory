"""Crystallized-pattern layer for anneal-memory — the on-demand graduated tier.

anneal's continuity ``## Patterns`` section is a WORKING SET: developing (1x/2x)
plus recently-leaned-on patterns, always loaded, bounded by nature. But a pattern
graduates IN at 3x and the section had no OUT path — a one-way ratchet, so the
working set monotonically bloats and, past a point, an always-loaded list stops
*working* (attention doesn't scale — 50 always-on patterns drown each other out).

This module adds the missing top tier — the CRYSTALLIZED store. It holds every
proven-and-stable pattern *retrievably*, surfaced **on cue** by the recall hook
rather than always loaded, so a large body of graduated wisdom stays *effective*
without clogging context. The discriminator from the episodic layer is the same
event-vs-distilled seam anneal already keeps clean: episodes are EVENTS (what
happened), a crystallized pattern is DISTILLED (what the events taught). It is
deliberately a **spore-sibling** store — the same JSON-document shape, atomic-write
durability, and multi-writer locking as :mod:`anneal_memory.spores` — because
crystallized patterns are to graduated-wisdom what spores are to prospective-tasks:
a lifecycle'd store, surfaced on-cue by the SAME recall hook.

The four-tier memory architecture this completes::

    episodic        hippocampus            episodic.db        on cue   the event trail
    working set     working memory         ## Patterns        always   developing + hot
    CRYSTALLIZED ◀  cortical semantic      <stem>.crystal.json on cue   every proven pattern
    constitution    core identity/priors   the harness's      always   catastrophic-if-missed
                                            always-load layer

A crystallized pattern is routed on **two axes** (the seed's single timeless/phase
tag was too thin):

  - ``permanence``      ``timeless`` | ``phase-specific``
  - ``activation_mode`` ``just-in-time`` | ``catastrophic``

The store holds the ``timeless`` × ``just-in-time`` bulk. The other corners route
ELSEWHERE at migration time (composer-judged, propose-not-auto): a
``catastrophic``-if-missed pattern belongs in the always-loaded *constitution*
(never on-demand — a miss corrupts the substrate); a ``phase-specific`` + cold
pattern is COMPOSTED (its episodic trail kept as the re-graduation safety net).
The two axes are recorded on every row anyway, so a later re-route is auditable.

Lifecycle (shared shape with spores: a store you enter, are activated within, and
leave)::

    crystallize ──▶ activate (touch; activation tier COMPUTED) ──▶ retire
       │                                                              │
       the membrane OUT of the working set            falsified / superseded / merged /
       (graduated wisdom escapes the                  obsolete — kept in the retired set
       always-loaded budget)                          for audit, never silently dropped

Bidirectional working⇄crystallized via activation: a re-heating crystallized
pattern (its domain goes hot, so it keeps being leaned on) becomes a **re-warm
candidate** (:meth:`CrystalStore.surface_rewarm_candidates`) — the working set is a
*cache* over this backing store, activation-driven both ways. Crystallizing is NOT
forgetting: ``activation_aware_forgetting`` keys eviction on activation-recency, and
episodic recall re-graduates a wrongly-composted pattern from its evidence.

Activation tier (computed at read-time, NEVER stored — parallel to spores'
germination and to "top of mind"):

  - ``hot``      activated < 7 days ago        — re-warm candidate; pull into working set
  - ``warm``     activated 7–30 days ago       — live in the corpus, recall-reachable
  - ``cold``     activated 30–90 days ago      — settled; on-demand only
  - ``dormant``  activated > 90 days ago / unknown — deep store; recall or retire

Storage: a single JSON document beside the episodic db (``<stem>.crystal.json``),
written atomically (unique-tmp + fsync + rename + dir-fsync — the same durability
idiom as the SQLite store's ``_fsync_dir`` and :mod:`spores`). Like spores and
UNLIKE the episodic :class:`Store` (a documented single-process invariant), this
layer is **multi-writer-safe**: crystallization happens at WRAP time (single-writer
consolidate) but recall reads fire on EVERY turn across parallel sessions, so every
mutation serializes under an exclusive ``fcntl`` lock spanning the whole
load→mutate→save, while reads stay lock-free (atomic ``os.replace`` means a reader
always sees a complete committed document). POSIX-only locking. Zero dependencies
beyond the Python stdlib.

Dates: ``crystallized_on`` / ``last_activated_on`` / ``retirement.on`` are
``YYYY-MM-DD`` **logical dates** in the operator's frame — injectable via ``today``
for deterministic runs, defaulting to ``date.today()``. ``retirement.at`` is an
ISO-8601 **UTC** instant (anneal's machine-timestamp convention).

Full design + the AM-WORKINGSET architecture: ``projects/anneal_memory/next.md``
(🔴 AM-WORKINGSET) + ``contexts/master_plan.md`` (flow repo). The harness-side
per-turn surfacing hook is a Levain/flow adapter, NOT this module — anneal owns the
store + the retrieval API (:func:`anneal_memory.retrieval.retrieve_relevant`); the
harness owns the firing. "anneal only fully works inside a complementary harness."
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, TypedDict, cast

try:  # POSIX advisory locking; absent on Windows (see CrystalStore._transaction).
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

from .store import AnnealMemoryError

CRYSTAL_SCHEMA_VERSION = 1

Permanence = Literal["timeless", "phase-specific"]
ActivationMode = Literal["just-in-time", "catastrophic"]
Activation = Literal["hot", "warm", "cold", "dormant"]
CrystalStatus = Literal["crystallized", "retired"]

VALID_PERMANENCE: tuple[Permanence, ...] = ("timeless", "phase-specific")
VALID_ACTIVATION_MODES: tuple[ActivationMode, ...] = ("just-in-time", "catastrophic")
VALID_ACTIVATIONS: tuple[Activation, ...] = ("hot", "warm", "cold", "dormant")

# Only Proven-tier wisdom crystallizes — a pattern graduates IN to the working set
# at 2x/3x; crystallization is the stage PAST 3x (or a stable 2x leaving the hot set).
# A 1x developing pattern belongs in the working set, not the deep store.
VALID_LEVELS: tuple[int, ...] = (2, 3)

# How a crystallized pattern leaves the store. ``falsified`` = the contradiction
# scan pulled it (it was wrong); ``superseded`` = a sharper pattern replaced it;
# ``merged`` = folded into another (dedup); ``obsolete`` = its world changed. All
# four keep the row in the ``retired`` set — crystallized ≠ immortal, but the audit
# trail (and the re-graduation safety net) is never silently dropped.
RETIRE_KINDS: tuple[str, ...] = ("falsified", "superseded", "merged", "obsolete")

# Activation-tier thresholds (days). Pattern-scale, not task-scale (spores'
# germination is 3/7) — graduated wisdom re-heats over weeks, not days. The recall
# layer / a Levain adapter would expose these as config.
ACTIVATION_HOT_DAYS = 7
ACTIVATION_WARM_DAYS = 30
ACTIVATION_COLD_DAYS = 90

# Ranking weights for ``list_crystal`` / ``surface_rewarm_candidates`` (how-hot then
# how-graduated). Unknown values sort last.
_ACTIVATION_ORDER = {"hot": 0, "warm": 1, "cold": 2, "dormant": 3}


class CrystalError(AnnealMemoryError):
    """Crystallized-store operational error — corrupt store, unknown name, a name
    that's already retired, or ambiguous-name drift.

    Inherits :class:`AnnealMemoryError` so a caller catching the library base
    catches crystal-store failures in the same boundary as episodic- and
    spore-store ones.
    """


class _Unset:
    """Sentinel for ``update``: distinguishes "argument omitted" (leave unchanged)
    from "set to None/empty" (clear). A bare ``None`` default can't express that
    difference for the clearable fields."""


_UNSET = _Unset()


class RetirementDict(TypedDict):
    """How a crystallized pattern left the store. ``reason`` is optional free text;
    ``on`` is the logical date; ``at`` is the UTC instant."""

    kind: str
    reason: str | None
    on: str
    at: str


class CrystalDict(TypedDict):
    """The stored shape of a crystallized pattern. ``activation`` is deliberately
    ABSENT — it is computed from ``last_activated_on`` at read-time via
    :func:`activation_tier`, never persisted (a stored tier would drift the moment
    the clock moved, exactly like spores' germination)."""

    name: str
    level: int
    explanation: str
    evidence: list[str]
    permanence: Permanence
    activation_mode: ActivationMode
    tags: list[str]
    crystallized_on: str
    last_activated_on: str
    status: CrystalStatus
    retirement: RetirementDict | None
    source: str | None
    notes: list[str]


# ---------------------------------------------------------------------------
# date / value helpers (mirror spores.py — duplicated by design, like the
# _fsync_dir idiom shared store.py↔spores.py; keeps this module's blast radius
# to one new file rather than refactoring a shared util into the others)
# ---------------------------------------------------------------------------

def _parse_date(value: object) -> date | None:
    """Parse a ``YYYY-MM-DD`` prefix to a date, else None. Lenient on the read path
    (parses ``value[:10]``) so a hand-edited ``'YYYY-MM-DD <note>'`` still reads."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _validate_date(value: str | None, field: str) -> str | None:
    """A write-path date must be exactly ``YYYY-MM-DD``, or ``None``/``''`` (= unset).
    Fail loud (``ValueError``) so a typo can't silently strand a pattern dormant by
    being stored verbatim then read as no-date."""
    if value in (None, ""):
        return None
    if not isinstance(value, str) or len(value) != 10 or _parse_date(value) is None:
        raise ValueError(f"{field} must be YYYY-MM-DD (got {value!r}).")
    return value


def _clean_str_list(value: object, field: str) -> list[str]:
    """Coerce-and-validate a list of non-empty strings (evidence ids / tags),
    deduped order-preserving. Rejects a bare string (a common caller slip that would
    otherwise store each CHARACTER as an item). Empty/None → []."""
    if value in (None, ""):
        return []
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of strings (got {value!r}).")
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} entries must be non-empty strings (got {item!r}).")
        s = item.strip()
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort POSIX directory fsync so the rename itself is durable. Mirrors the
    episodic store's and spores' ``_fsync_dir``; on macOS ``os.fsync`` is weaker than
    ``F_FULLFSYNC`` — a documented platform limit, not a bug here."""
    try:
        fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def activation_tier(crystal: CrystalDict, today: date | None = None) -> Activation:
    """Compute ``hot | warm | cold | dormant`` from ``last_activated_on``.

    The re-heat signal driving working⇄crystallized: ``hot`` patterns are re-warm
    candidates (pull back into the working set); ``dormant`` ones are deep-store
    (recall-only, or retirement candidates). No parseable ``last_activated_on`` →
    ``dormant`` (unknown age — treat as cold-most, never spuriously hot)."""
    today = today or date.today()
    last = _parse_date(crystal.get("last_activated_on"))
    if last is None:
        return "dormant"
    age = (today - last).days
    if age < ACTIVATION_HOT_DAYS:
        return "hot"
    if age <= ACTIVATION_WARM_DAYS:
        return "warm"
    if age <= ACTIVATION_COLD_DAYS:
        return "cold"
    return "dormant"


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------

class CrystalStore:
    """A JSON-backed store of crystallized patterns (the on-demand graduated tier).

    Mirrors :class:`SporeStore` exactly — an explicit path (no magic default), the
    atomic-write durability (unique-tmp + fsync + rename + dir-fsync), and an
    exclusive advisory lock spanning every mutation's load→mutate→save so concurrent
    writers serialize with no lost updates and reads stay lock-free. The keyed
    identity is the pattern ``name`` (a semantic snake_case slug), not an auto-id:
    crystallizing the same name twice is an UPSERT (a pattern that re-warmed and
    re-cooled, or whose level/explanation sharpened), never a duplicate.

    Errors: operational failures (corrupt store, unknown name, already-retired name,
    ambiguous-name drift) raise :class:`CrystalError`; malformed caller arguments
    (bad date/level/permanence/kind) raise ``ValueError``. A corrupt store NEVER
    silently re-inits (that would overwrite recoverable crystallized wisdom on the
    next save).
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
                "crystal": [],
                "retired": [],
                "schema_version": CRYSTAL_SCHEMA_VERSION,
            }
        except json.JSONDecodeError as e:
            raise CrystalError(
                f"{self.path} is not valid JSON ({e}); refusing to proceed so "
                f"recoverable crystallized patterns aren't overwritten on the next "
                f"save — inspect it (and any .tmp sidecar) by hand."
            ) from e
        if not isinstance(data, dict):
            raise CrystalError(
                f"{self.path} must contain a JSON object, got {type(data).__name__}."
            )
        data.setdefault("crystal", [])
        data.setdefault("retired", [])
        data.setdefault("schema_version", CRYSTAL_SCHEMA_VERSION)
        if not isinstance(data["crystal"], list) or not isinstance(data["retired"], list):
            raise CrystalError(
                f"{self.path} is structurally invalid ('crystal' and 'retired' must "
                f"be lists); refusing to proceed so recoverable patterns aren't lost "
                f"— inspect it by hand."
            )
        if not all(isinstance(s, dict) for s in data["crystal"]) or not all(
            isinstance(s, dict) for s in data["retired"]
        ):
            raise CrystalError(
                f"{self.path} has non-object rows in 'crystal'/'retired'; refusing "
                f"to proceed — inspect it by hand."
            )
        version = data["schema_version"]
        if not isinstance(version, int) or isinstance(version, bool):
            raise CrystalError(
                f"{self.path} has a non-integer schema_version ({version!r}); "
                f"refusing to proceed — inspect it by hand."
            )
        if version > CRYSTAL_SCHEMA_VERSION:
            raise CrystalError(
                f"{self.path} was written by a newer crystal schema "
                f"(v{version} > v{CRYSTAL_SCHEMA_VERSION}); refusing to read it so "
                f"fields this version doesn't understand aren't silently dropped on "
                f"the next save."
            )
        return data

    def _save(self, data: dict) -> None:
        target_dir = self.path.parent
        os.makedirs(target_dir, exist_ok=True)
        # A UNIQUE tmp sibling, never a fixed ``<name>.tmp``: two writers must not
        # collide on one tmp path. Mirrors store.py / spores.py. The lock in
        # :meth:`_transaction` already serializes our own writers; the unique tmp
        # also protects against a leftover sidecar and any non-cooperating writer.
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
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        _fsync_dir(target_dir)

    @contextmanager
    def _transaction(self) -> Iterator[dict]:
        """Serialize a full load→mutate→save against concurrent processes.

        Identical contract to :meth:`SporeStore._transaction`: crystallization runs
        at wrap time but recall reads fire every turn across parallel sessions, so a
        mutation takes an **exclusive** advisory lock on a sibling ``<store>.lock``
        and (re)loads the document INSIDE the lock — no lost updates, no name
        collisions. The lock releases when the fd closes or the process dies, so a
        crashed holder can never strand it. NOT reentrant (``flock`` is per-fd; a
        mutator must never call another mutator). :meth:`_save` runs only on a clean
        exit; an exception in the body skips the save and releases the lock.

        Reads (:meth:`get` / :meth:`list_crystal` / :meth:`surface_rewarm_candidates`
        / :meth:`active`) stay lock-free: :meth:`_save` commits via atomic
        ``os.replace``, so a reader always sees a complete committed document.

        Platform: ``fcntl`` is POSIX-only and reliable on a LOCAL filesystem; on a
        non-POSIX platform the lock degrades to a no-op (the unique-tmp +
        atomic-replace write still protects a single-process writer; cross-process
        serialization is guaranteed only on a local POSIX filesystem)."""
        os.makedirs(self.path.parent, exist_ok=True)
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
    def _find_live(data: dict, name: str) -> CrystalDict | None:
        for item in data.get("crystal", []):
            if item.get("name") == name:
                return cast("CrystalDict", item)
        return None

    def _require_live(self, data: dict, name: str) -> CrystalDict:
        """Return the crystallized pattern, or raise — distinguishing already-retired
        from never-existed (a retired name shouldn't read as "not found")."""
        item = self._find_live(data, name)
        if item is not None:
            # "Live" means exactly status == "crystallized" — the SAME definition the
            # read paths (active / list_crystal / surface_rewarm_candidates) use, so a
            # drifted non-crystallized row in the live list (status "retired" or any
            # other corrupt value) is rejected here too rather than being touch/update-
            # mutable while invisible to reads.
            status = item.get("status")
            if status != "crystallized":
                raise CrystalError(
                    f"crystallized pattern {name!r} is in the live list but carries "
                    f"status={status!r} (store drift — repair by hand)."
                )
            return item
        for r in data.get("retired", []):
            if r.get("name") == name:
                ret = r.get("retirement") or {}
                raise CrystalError(
                    f"crystallized pattern {name!r} is already retired "
                    f"({ret.get('kind')} on {ret.get('on')})."
                )
        raise CrystalError(f"crystallized pattern {name!r} not found.")

    # --- public API: crystallize (the membrane OUT of the working set) ------

    def crystallize(
        self,
        *,
        name: str,
        level: int,
        explanation: str,
        evidence: list[str] | None = None,
        permanence: Permanence = "timeless",
        activation_mode: ActivationMode = "just-in-time",
        tags: list[str] | None = None,
        source: str | None = None,
        today: date | None = None,
    ) -> CrystalDict:
        """Crystallize a graduated pattern OUT of the working set into the on-demand
        store. UPSERT by ``name``: re-crystallizing an existing live pattern updates
        its content and raises its ``level`` monotonically (never lowers it — a
        pattern's earned high-water mark holds), preserving the original
        ``crystallized_on``. Re-crystallizing a RETIRED name un-retires it (a
        falsified pattern re-proven) with a fresh ``crystallized_on`` and a note.

        ``level`` must be 2 or 3 (only Proven-tier wisdom crystallizes). ``evidence``
        is the list of episode ids that grounded the pattern — the substrate for
        associative retrieval (pattern → evidence-episode → Hebbian co-pattern).
        ``permanence`` × ``activation_mode`` is the 2-axis routing record; the store
        holds the ``timeless`` × ``just-in-time`` bulk, but the tags are kept on every
        row so a later re-route is auditable.

        Crystallizing IS an activation event — both the new and upsert paths set
        ``last_activated_on`` to ``today`` (you just engaged the pattern to file/refile
        it). So a freshly-crystallized pattern starts ``hot`` and settles over the
        activation window; :meth:`surface_rewarm_candidates` will surface it until it
        cools, which is benign (re-warm is propose-not-auto — the composer that just
        crystallized it won't re-add it). Returns the stored record."""
        name = self._validate_name(name)
        level = self._validate_level(level)
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("explanation is required and must be a non-empty string.")
        if permanence not in VALID_PERMANENCE:
            raise ValueError(f"permanence must be one of {VALID_PERMANENCE} (got {permanence!r}).")
        if activation_mode not in VALID_ACTIVATION_MODES:
            raise ValueError(
                f"activation_mode must be one of {VALID_ACTIVATION_MODES} (got {activation_mode!r})."
            )
        if source is not None and not isinstance(source, str):
            raise ValueError(f"source must be a string or None (got {source!r}).")
        evidence_clean = _clean_str_list(evidence, "evidence")
        tags_clean = _clean_str_list(tags, "tags")
        now = (today or date.today()).isoformat()

        with self._transaction() as data:
            existing = self._find_live(data, name)
            if existing is not None:
                # Upsert a live pattern: monotonic level, fresh content, keep origin.
                # last_activated_on bumps (crystallizing is an activation event —
                # symmetric with the new-row path below). Re-crystallizing always
                # means the pattern is live now, so REPAIR any drifted status: a row
                # found in the live list carrying status != "crystallized" (store
                # drift) is reset to crystallized + retirement cleared.
                existing["level"] = max(self._safe_level(existing.get("level")), level)
                existing["explanation"] = explanation
                existing["evidence"] = evidence_clean
                existing["permanence"] = permanence
                existing["activation_mode"] = activation_mode
                existing["tags"] = tags_clean
                existing["last_activated_on"] = now
                existing["status"] = "crystallized"
                existing["retirement"] = None
                if source is not None:
                    existing["source"] = source or None
                return existing

            revived = self._pop_retired(data, name)
            item: CrystalDict = {
                "name": name,
                "level": level,
                "explanation": explanation,
                "evidence": evidence_clean,
                "permanence": permanence,
                "activation_mode": activation_mode,
                "tags": tags_clean,
                "crystallized_on": now,
                "last_activated_on": now,
                "status": "crystallized",
                "retirement": None,
                "source": source or None,
                "notes": [],
            }
            if revived is not None:
                prior: RetirementDict | dict = revived.get("retirement") or {}
                item["notes"] = [
                    f"[{now}] re-crystallized after retirement "
                    f"({prior.get('kind')} on {prior.get('on')})."
                ]
            data["crystal"].append(item)
            return item

    @staticmethod
    def _pop_retired(data: dict, name: str) -> CrystalDict | None:
        """Remove and return a retired row by name (for un-retire on re-crystallize),
        or None. Fail loud on a duplicate retired name rather than dropping both."""
        matches = [r for r in data.get("retired", []) if r.get("name") == name]
        if not matches:
            return None
        if len(matches) != 1:
            raise CrystalError(
                f"{len(matches)} retired patterns share name {name!r}; refusing to "
                f"revive an ambiguous name (store drift — repair by hand)."
            )
        data["retired"] = [r for r in data["retired"] if r.get("name") != name]
        return cast("CrystalDict", matches[0])

    # --- public API: read ---------------------------------------------------

    def get(self, name: str) -> CrystalDict | None:
        """Fetch a pattern by name, searching the live set first then the retired set
        (live takes precedence if a name somehow appears in both), or None."""
        data = self._load()
        for item in data.get("crystal", []) + data.get("retired", []):
            if item.get("name") == name:
                return cast("CrystalDict", item)
        return None

    def active(self) -> list[CrystalDict]:
        """Every live crystallized pattern, unranked. The raw corpus the retrieval
        layer (:func:`anneal_memory.retrieval.retrieve_relevant`) scores against a
        query, the wrap dedup/contradiction scans read in full, and the shrink-gate
        credit grounds on. Filters to ``status == "crystallized"`` rows: a corrupt
        ``status == "retired"`` row drifting in the live list is excluded, so it can
        never earn shrink credit nor surface as live (the "a crystal-store fault must
        never weaken the gate" invariant — a read-path filter, not a raise, because
        this feeds the wrap path which must not break)."""
        return [
            cast("CrystalDict", s)
            for s in self._load().get("crystal", [])
            if s.get("status") == "crystallized"
        ]

    def list_crystal(
        self,
        *,
        permanence: Permanence | None = None,
        activation_mode: ActivationMode | None = None,
        tag: str | None = None,
        activation: Activation | None = None,
        today: date | None = None,
    ) -> list[CrystalDict]:
        """Live patterns, filtered then ranked (activation → level desc → recency).

        Activation tier is NOT written onto the returned dicts (it's computed, never
        stored); call :func:`activation_tier` on a row to annotate it."""
        today = today or date.today()
        # Status-filtered live view (consistent with active()): a drifted
        # status != "crystallized" row never surfaces as live.
        items = [
            s for s in self._load().get("crystal", [])
            if s.get("status") == "crystallized"
        ]
        if permanence is not None:
            items = [s for s in items if s.get("permanence") == permanence]
        if activation_mode is not None:
            items = [s for s in items if s.get("activation_mode") == activation_mode]
        if tag is not None:
            items = [s for s in items if tag in (s.get("tags") or [])]
        if activation is not None:
            items = [
                s for s in items
                if activation_tier(cast("CrystalDict", s), today) == activation
            ]
        return self._rank(items, today)

    @staticmethod
    def _rank(items: list, today: date) -> list[CrystalDict]:
        return sorted(
            (cast("CrystalDict", s) for s in items),
            key=lambda s: (
                _ACTIVATION_ORDER.get(activation_tier(s, today), 9),
                -CrystalStore._safe_level(s.get("level")),
                _recency_key(s.get("last_activated_on")),
            ),
        )

    def surface_rewarm_candidates(
        self, *, today: date | None = None
    ) -> list[CrystalDict]:
        """The re-warm surface: live patterns whose activation tier is ``hot`` — they
        keep being leaned on, so the working set should cache them. The wrap composer
        consumes this (propose-not-auto) to decide which crystallized patterns return
        to the always-loaded ``## Patterns`` working set, deduping against what's
        already there. The working set is a cache over this backing store; this is the
        prefetch half of activation-driven working⇄crystallized movement."""
        today = today or date.today()
        hot = [
            s for s in self._load().get("crystal", [])
            if s.get("status") == "crystallized"
            and activation_tier(cast("CrystalDict", s), today) == "hot"
        ]
        return self._rank(hot, today)

    # --- public API: activate / surgery -------------------------------------

    def touch(self, name: str, *, today: date | None = None) -> CrystalDict:
        """Record that a live pattern was activated (leaned on / re-surfaced):
        ``last_activated_on`` → today, re-heating its activation tier. Called when a
        crystallized pattern is cited in a wrap or pulled back by the composer — NOT
        by the every-turn read hook (which stays lock-free; activation is a
        wrap-time, single-writer signal). The re-heat is what makes a dormant pattern
        a re-warm candidate again."""
        now = (today or date.today()).isoformat()
        with self._transaction() as data:
            item = self._require_live(data, name)
            item["last_activated_on"] = now
            return item

    def update(
        self,
        name: str,
        *,
        explanation: str | _Unset = _UNSET,
        level: int | _Unset = _UNSET,
        evidence: list[str] | _Unset = _UNSET,
        permanence: Permanence | _Unset = _UNSET,
        activation_mode: ActivationMode | _Unset = _UNSET,
        tags: list[str] | _Unset = _UNSET,
        source: str | None | _Unset = _UNSET,
        add_note: str | None = None,
        today: date | None = None,
    ) -> CrystalDict:
        """Metadata surgery on a live pattern (re-route the 2 axes, re-tag, sharpen
        the explanation) without re-crystallizing. Omitted arguments are left
        unchanged; passing ``None``/``''`` to ``source`` clears it. ``level`` here is
        set as given (the explicit-correction path — unlike :meth:`crystallize`'s
        monotonic upsert), but still must be 2 or 3. Deliberately does NOT bump
        ``last_activated_on`` — activation is signalled explicitly via :meth:`touch`,
        which keeps the tier honest."""
        with self._transaction() as data:
            item = self._require_live(data, name)
            if not isinstance(explanation, _Unset):
                if not isinstance(explanation, str) or not explanation.strip():
                    raise ValueError("explanation must be a non-empty string (cannot clear).")
                item["explanation"] = explanation
            if not isinstance(level, _Unset):
                item["level"] = self._validate_level(level)
            if not isinstance(evidence, _Unset):
                item["evidence"] = _clean_str_list(evidence, "evidence")
            if not isinstance(permanence, _Unset):
                if permanence not in VALID_PERMANENCE:
                    raise ValueError(f"permanence must be one of {VALID_PERMANENCE} (got {permanence!r}).")
                item["permanence"] = permanence
            if not isinstance(activation_mode, _Unset):
                if activation_mode not in VALID_ACTIVATION_MODES:
                    raise ValueError(
                        f"activation_mode must be one of {VALID_ACTIVATION_MODES} (got {activation_mode!r})."
                    )
                item["activation_mode"] = activation_mode
            if not isinstance(tags, _Unset):
                item["tags"] = _clean_str_list(tags, "tags")
            if not isinstance(source, _Unset):
                if source is not None and not isinstance(source, str):
                    raise ValueError(f"source must be a string or None (got {source!r}).")
                item["source"] = source or None
            if add_note:
                stamp = (today or date.today()).isoformat()
                if not isinstance(item.get("notes"), list):
                    item["notes"] = []
                item["notes"].append(f"[{stamp}] {add_note}")
            return item

    # --- public API: retire (the membrane out — crystallized ≠ immortal) ----

    def retire(
        self,
        name: str,
        *,
        kind: str,
        reason: str | None = None,
        today: date | None = None,
        now: datetime | None = None,
    ) -> CrystalDict:
        """Retire a crystallized pattern (it was falsified / superseded / merged /
        obsolete). Moves it to the ``retired`` set with a retirement record — kept for
        audit AND as the re-graduation trail (its evidence episodes still exist). NOT
        a silent delete. ``kind`` must be one of :data:`RETIRE_KINDS`. For fully
        deterministic tests pass both ``today`` (logical date) and ``now`` (the UTC
        instant on ``retirement.at``)."""
        if kind not in RETIRE_KINDS:
            raise ValueError(f"retire kind must be one of {RETIRE_KINDS} (got {kind!r}).")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"reason must be a string or None (got {reason!r}).")
        if now is not None:
            if now.tzinfo is None:
                raise ValueError("now must be timezone-aware (got a naive datetime).")
            now = now.astimezone(timezone.utc)
        with self._transaction() as data:
            item = self._require_live(data, name)
            # Fail loud on a duplicate live name rather than dropping both rows
            # (name-equality removal would silently nuke the dup).
            matches = [s for s in data["crystal"] if s.get("name") == name]
            if len(matches) != 1:
                raise CrystalError(
                    f"{len(matches)} live patterns share name {name!r}; refusing to "
                    f"retire an ambiguous name (store drift — repair by hand)."
                )
            if any(r.get("name") == name for r in data["retired"]):
                raise CrystalError(
                    f"pattern {name!r} already exists in the retired set (store drift "
                    f"— refusing to create a duplicate retired name)."
                )
            item["status"] = "retired"
            item["retirement"] = {
                "kind": kind,
                "reason": reason or None,
                "on": (today or date.today()).isoformat(),
                "at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
            }
            data["crystal"] = [s for s in data["crystal"] if s.get("name") != name]
            data["retired"].append(item)
            return item

    # --- validation helpers -------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name is required and must be a non-empty string (the pattern slug).")
        return name.strip()

    @staticmethod
    def _validate_level(level: int) -> int:
        # bool is an int subclass — reject it so a stray True/False can't read as 1/0.
        if not isinstance(level, int) or isinstance(level, bool) or level not in VALID_LEVELS:
            raise ValueError(f"level must be one of {VALID_LEVELS} (Proven-tier) (got {level!r}).")
        return level

    @staticmethod
    def _safe_level(value: object) -> int:
        """Coerce a possibly hand-edited / migrated level to int — never crash a read
        path (ranking) on a bad field; an unparseable level sorts as 0 (lowest)."""
        try:
            return int(cast("int", value))
        except (ValueError, TypeError):
            return 0


def _recency_key(value: object) -> float:
    """Ascending-sort key that orders dates MOST-RECENT-FIRST: a later date yields a
    smaller (more-negative) key. An unparseable/absent date sorts last (``inf``).
    Used as the final tiebreak in :meth:`CrystalStore._rank`."""
    d = _parse_date(value)
    if d is None:
        return float("inf")
    return -float(d.toordinal())
