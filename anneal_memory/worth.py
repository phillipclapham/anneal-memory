"""Outcome write-back, surfaced counts, and Memory-Worth counters (report-only).

anneal can say a pattern was honestly earned (citation-validated graduation). This
module is the start of saying whether a recalled memory ever helped. Three pieces:

1. **Outcome write-back** — :class:`OutcomeLog`. A harness that surfaced memory
   for an *exposure* (one recall event, identified by the harness's own id, e.g.
   a retrieval receipt's ``event_id``) appends what happened: for each surfaced
   item whether it was ``followed``, ``ignored`` or ``not_applicable``, and
   optionally whether the turn's outcome was a ``success`` or ``failure``. The log
   is append-only JSONL beside the episodic db (``<stem>.outcomes.jsonl``).
   Records for the same exposure id MERGE when read, in file order: a later label
   for an item replaces its earlier label, and a later non-null outcome replaces
   the earlier outcome. So labels and the outcome can be written at different
   times, and a label can be corrected; nothing is rewritten in place.

2. **Recall-surfaced counts** — :func:`fold_surfaced`. A harness's recall hook
   writes retrieval receipts without taking any anneal lock; :func:`fold_surfaced`
   is intended to run once per wrap, by the wrap's single writer, and folds them
   into ``surfaced_count`` / ``last_surfaced_on`` on each live crystal. It counts
   only what recall surfaced, not what an always-loaded file shows every turn.
   Exposure is NOT activation: this never touches ``last_activated_on``, so the
   activation tier (and therefore re-warm) is unaffected. Counting exposure as
   activation would make recall self-reinforcing.

3. **Worth** — :func:`compute_worth`. Two counters per crystal and per episode:
   retrieved-with-success and retrieved-with-failure (every labelled exposure
   whose outcome is known, whatever its label), plus the full label × outcome
   table, so ``followed``-with-outcome is one cell of it and a labeller that marks
   failures "ignored" shows up as a lopsided table. A crystal's exposure also
   credits the episodes it cites as evidence (the citation edge), at most once per
   episode per exposure. The result is a REPORT. Nothing in anneal reads it to
   rank, decay, re-heat or retire anything; a downstream failure is not evidence
   that a retrieved memory caused it.

Design background: ``Memory Worth`` (arXiv 2604.12007) for the two counters,
PMB's followed / ignored / not-applicable labels, MemQ for credit along the
provenance edges.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .crystal import CrystalError, CrystalStore

try:  # POSIX advisory lock; degrades to a no-op elsewhere (mirrors crystal.py)
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

OUTCOME_LOG_VERSION = 1

FOLLOWED_VALUES: tuple[str, ...] = ("followed", "ignored", "not_applicable")
OUTCOME_VALUES: tuple[str, ...] = ("success", "failure")
ITEM_KINDS: tuple[str, ...] = ("crystal", "episode")

# Bounds on caller-supplied identifiers: a line of the log must stay one line.
_MAX_ID_LEN = 256

# The document key in the crystal store that holds the fold's high-water mark.
FOLD_STATE_KEY = "surfaced_fold"

# A receipt is folded only once it is older than this. The hook stamps a receipt
# before it appends it, so a fold's cutoff is taken BEFORE its scan and lags the
# clock: a receipt stamped earlier but appended after a fold would otherwise fall
# behind the mark and never be counted. A hook that takes longer than this between
# stamping and appending is still missed.
DEFAULT_FOLD_SKEW_SECONDS = 60


def outcome_log_path(db_path: str | os.PathLike[str]) -> Path:
    """The outcome log beside an episodic db: ``<stem>.outcomes.jsonl``."""
    p = Path(db_path)
    return p.parent / f"{p.stem}.outcomes.jsonl"


def _check_id(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string (got {value!r}).")
    if len(value) > _MAX_ID_LEN:
        raise ValueError(f"{what} is longer than {_MAX_ID_LEN} characters.")
    if any(ch in value for ch in "\r\n\x00"):
        raise ValueError(f"{what} must not contain a line break or NUL.")
    return value


@dataclass(frozen=True)
class ExposureLabel:
    """One surfaced item in an exposure and whether it was used.

    ``kind`` is ``crystal`` (``ref`` = the pattern name) or ``episode``
    (``ref`` = the episode id). ``followed`` is one of :data:`FOLLOWED_VALUES`.
    """

    kind: str
    ref: str
    followed: str

    def __post_init__(self) -> None:
        if self.kind not in ITEM_KINDS:
            raise ValueError(f"kind must be one of {ITEM_KINDS} (got {self.kind!r}).")
        _check_id(self.ref, "ref")
        if self.followed not in FOLLOWED_VALUES:
            raise ValueError(
                f"followed must be one of {FOLLOWED_VALUES} (got {self.followed!r})."
            )


@dataclass(frozen=True)
class ExposedRef:
    """One item an exposure surfaced, with no judgement about whether it was
    used. The harness's receipt is the proof of exposure; an item listed here
    and also labelled in the same exposure counts as labelled."""

    kind: str
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in ITEM_KINDS:
            raise ValueError(f"kind must be one of {ITEM_KINDS} (got {self.kind!r}).")
        _check_id(self.ref, "ref")


class OutcomeLog:
    """Append-only JSONL of outcome records, keyed by exposure id.

    Appends are serialized with an exclusive advisory lock on the log file itself
    and fsynced, so concurrent harness processes do not interleave lines. Reads are
    lenient: a malformed or torn line is skipped and counted, never fatal, because
    the log is a measurement and a bad line must not take the report down.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def record(
        self,
        exposure_id: str,
        items: Sequence[ExposureLabel],
        *,
        outcome: str | None = None,
        exposed: Sequence[ExposedRef] = (),
        ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Append the outcome for one exposure and return the stored record.

        ``items`` may be empty only when ``outcome`` is given (an outcome written
        after the labels); a repeated (kind, ref) keeps its last label.
        ``outcome`` is ``success``, ``failure`` or ``None`` (not known yet).
        ``exposed`` lists what the exposure surfaced, labelled or not (from the
        harness's receipt); it is stored only when non-empty and never makes a
        record valid on its own, so a reader older than this field still accepts
        every record. Records for one ``exposure_id`` merge when read (see
        :meth:`latest`).
        """
        rec = _build_record(exposure_id, items, outcome, exposed, ts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            _append(fd, rec)
        finally:
            os.close(fd)
        return rec

    def record_if_missing(
        self,
        exposure_id: str,
        items: Sequence[ExposureLabel],
        *,
        outcome: str | None = None,
        exposed: Sequence[ExposedRef] = (),
        ts: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Append only what the log does not already hold for this exposure, and
        return the appended record, or ``None`` when nothing was missing.

        "Missing" is judged per EXPOSURE: ``items`` are written only when the
        exposure has no label at all yet (a judge who labelled any item judged the
        exposure, and a second judge's labels must not be mixed into it); ``outcome``
        only when the exposure has none. ``exposed`` rides along with whatever is
        written and is unioned on read, but it is never written on its own: a
        record with no label and no outcome is the shape every released reader
        skips as a bad line, so when only ``exposed`` would be new this returns
        ``None`` and writes nothing. Arguments are validated exactly as
        :meth:`record` validates them, before the log is touched.

        The read and the append are ONE span of the same exclusive lock
        :meth:`record` takes, so a :meth:`record` from another process (a human
        correction) lands either wholly before this call's read, where it is seen
        and kept, or wholly after its append, where it wins the merge. A
        :meth:`latest` followed by :meth:`record` is two spans and can overwrite
        it. Each call reads the whole log.
        """
        rec = _build_record(exposure_id, items, outcome, exposed, ts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            chunks: list[bytes] = []
            os.lseek(fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8", errors="replace")
            records = [
                r for r in (_parse_record(line) for line in text.split("\n") if line.strip())
                if r is not None
            ]
            cur = _merge_records(records).get(exposure_id)
            if cur and cur["items"]:
                rec["items"] = []
            if cur and cur["outcome"] is not None:
                rec["outcome"] = None
            if not rec["items"] and rec["outcome"] is None:
                return None
            _append(fd, rec)
        finally:
            os.close(fd)
        return rec

    def read(self) -> tuple[list[dict[str, Any]], int]:
        """All well-formed records in file order, and the count of skipped lines."""
        records: list[dict[str, Any]] = []
        bad = 0
        try:
            f = open(self.path, "r", encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return records, 0
        with f:
            for line in f:
                if not line.strip():
                    continue
                rec = _parse_record(line)
                if rec is None:
                    bad += 1
                else:
                    records.append(rec)
        return records, bad

    def latest(self) -> tuple[dict[str, dict[str, Any]], int]:
        """The merged state per exposure id, and the count of skipped lines.

        Records merge in file order: each item's last label wins, the last
        non-null outcome wins, and ``exposed`` is the union. The merged record
        carries ``items``, ``exposed`` and ``outcome``.
        """
        records, bad = self.read()
        return _merge_records(records), bad


def _build_record(
    exposure_id: str,
    items: Sequence[ExposureLabel],
    outcome: str | None,
    exposed: Sequence[ExposedRef],
    ts: datetime | None,
) -> dict[str, Any]:
    """Validate one record's arguments and return the record :meth:`OutcomeLog.record`
    would store. Raises ``ValueError`` before anything is written."""
    _check_id(exposure_id, "exposure_id")
    if outcome is not None and outcome not in OUTCOME_VALUES:
        raise ValueError(
            f"outcome must be one of {OUTCOME_VALUES} or None (got {outcome!r})."
        )
    merged: dict[tuple[str, str], ExposureLabel] = {}
    for item in items:
        if not isinstance(item, ExposureLabel):
            raise ValueError(f"items must be ExposureLabel values (got {item!r}).")
        merged[(item.kind, item.ref)] = item
    if not merged and outcome is None:
        raise ValueError("a record needs at least one labelled item or an outcome.")
    seen: dict[tuple[str, str], ExposedRef] = {}
    for ex in exposed:
        if not isinstance(ex, ExposedRef):
            raise ValueError(f"exposed must be ExposedRef values (got {ex!r}).")
        seen[(ex.kind, ex.ref)] = ex
    when = (ts or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rec: dict[str, Any] = {
        "v": OUTCOME_LOG_VERSION,
        "exposure_id": exposure_id,
        "ts": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outcome": outcome,
        "items": [
            {"kind": i.kind, "ref": i.ref, "followed": i.followed}
            for i in merged.values()
        ],
    }
    if seen:
        rec["exposed"] = [{"kind": e.kind, "ref": e.ref} for e in seen.values()]
    return rec


def _append(fd: int, rec: dict[str, Any]) -> None:
    """Write one record as one line to ``fd`` (opened ``O_APPEND``, already locked)
    and fsync it."""
    line = (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    view = memoryview(line)
    while view:
        written = os.write(fd, view)
        view = view[written:]
    os.fsync(fd)


def _merge_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge parsed records per exposure id, in order (see :meth:`OutcomeLog.latest`)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        cur = out.setdefault(
            rec["exposure_id"],
            {"exposure_id": rec["exposure_id"], "outcome": None, "_items": {},
             "_exposed": {}},
        )
        for i in rec["items"]:
            cur["_items"][(i["kind"], i["ref"])] = i
        for e in rec.get("exposed") or ():
            cur["_exposed"][(e["kind"], e["ref"])] = e
        if rec.get("outcome") is not None:
            cur["outcome"] = rec["outcome"]
    for cur in out.values():
        cur["items"] = list(cur.pop("_items").values())
        cur["exposed"] = list(cur.pop("_exposed").values())
    return out


def _parse_record(line: str) -> dict[str, Any] | None:
    try:
        rec = json.loads(line)
    except ValueError:
        return None
    if not isinstance(rec, dict) or rec.get("v") != OUTCOME_LOG_VERSION:
        return None
    try:
        _check_id(rec.get("exposure_id"), "exposure_id")
        outcome = rec.get("outcome")
        if outcome is not None and outcome not in OUTCOME_VALUES:
            return None
        items = rec.get("items")
        if not isinstance(items, list) or (not items and outcome is None):
            return None
        for i in items:
            if not isinstance(i, dict):
                return None
            ExposureLabel(i.get("kind"), i.get("ref"), i.get("followed"))  # type: ignore[arg-type]
    except ValueError:
        return None
    # ``exposed`` is advisory: an entry this version cannot read (a future kind,
    # a malformed ref) is dropped, never the record with its labels (L1).
    raw = rec.get("exposed")
    kept: list[dict[str, Any]] = []
    for e in raw if isinstance(raw, list) else ():
        try:
            if isinstance(e, dict):
                ExposedRef(e.get("kind"), e.get("ref"))  # type: ignore[arg-type]
                kept.append({"kind": e["kind"], "ref": e["ref"]})
        except ValueError:
            continue
    rec["exposed"] = kept
    return rec


# ---------------------------------------------------------------------------
# Surfaced counts: fold harness receipts into the crystal store at wrap time
# ---------------------------------------------------------------------------


def receipt_crystal_names(receipt: Mapping[str, Any]) -> list[str]:
    """The crystal names a retrieval receipt exposed, in rank order, deduplicated.

    Reads ``exposed[].pattern`` (the flow receipt shape, ``receipt_version`` 1-3)
    and ``exposed[].name`` as an alias. Items that are not dicts or carry no name
    are skipped.
    """
    exposed = receipt.get("exposed")
    if not isinstance(exposed, list):
        return []
    names: list[str] = []
    for item in exposed:
        if not isinstance(item, dict):
            continue
        name = item.get("pattern", item.get("name"))
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _receipt_day(query_date: object, ts: datetime) -> str:
    """The receipt's local date when it is a real ``YYYY-MM-DD`` within one day of
    ``ts`` (a local date differs from UTC by less than a day), else the UTC date of
    ``ts``."""
    if isinstance(query_date, str) and len(query_date) == 10:
        try:
            d = date.fromisoformat(query_date)
        except ValueError:
            d = None
        if d is not None and abs(d - ts.date()) <= timedelta(days=1):
            return d.isoformat()
    return ts.date().isoformat()


def _fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FoldResult:
    """What one :func:`fold_surfaced` call did."""

    previous_mark: str | None
    mark: str
    receipts_folded: int = 0
    exposures_counted: int = 0
    names_unknown: dict[str, int] = field(default_factory=dict)
    lines_skipped: int = 0
    event_id_missing: int = 0
    duplicates_skipped: int = 0
    paths_missing: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def fold_surfaced(
    crystal_store: CrystalStore,
    receipt_paths: Iterable[str | os.PathLike[str]],
    *,
    now: datetime | None = None,
    skew_seconds: int = DEFAULT_FOLD_SKEW_SECONDS,
) -> FoldResult:
    """Fold retrieval receipts into each live crystal's ``surfaced_count`` and
    ``last_surfaced_on``. Meant to run once per wrap, by the wrap's single writer.

    A receipt is counted when its ``ts`` is after the store's previous fold mark
    and at or before the cutoff ``now - skew_seconds``; the cutoff then becomes the
    new mark. Running it twice therefore counts nothing twice, and a receipt newer
    than the cutoff waits for the next fold. ``last_surfaced_on`` takes the
    receipt's ``query_date`` (the harness's local date) when present, else the UTC
    date of ``ts``. Names that are not a live crystal (retired, renamed, never
    crystallized) are reported in ``names_unknown`` and not stored.

    The mark is ONE value for the store, so every receipt source must be passed on
    every fold: a source left out of a fold loses the window that fold moved past.
    If none of the paths exists the fold refuses (``FileNotFoundError``) and the
    mark does not move, because a wrong path must not read as "nothing surfaced".
    A path that is missing while others exist (a rotated backup not yet created)
    is reported in ``paths_missing``.

    A receipt without an ``event_id`` is skipped and counted in
    ``event_id_missing``; the mark moves past it, so it is never counted later.
    Receipts are de-duplicated by ``event_id`` within a fold, so a log rotated
    while the fold reads it is not counted twice. Pass the live log FIRST and its
    rotated backup after it: a rotation during the read then moves already-read
    lines into the backup, where the de-duplication drops them, instead of moving
    unread lines past a file already read. Two rotations between folds still
    discard receipts the fold never saw.

    The whole scan runs inside the crystal store's exclusive lock, so two folds
    cannot both count the same window (crystallize / touch / update wait for it;
    reads do not). ``last_activated_on`` is never written.
    """
    if skew_seconds < 0:
        raise ValueError("skew_seconds must be >= 0.")
    paths = [Path(p) for p in receipt_paths]
    if not paths:
        raise ValueError("fold_surfaced needs at least one receipt path.")
    cutoff = (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - timedelta(
        seconds=skew_seconds
    )
    cutoff = cutoff.replace(microsecond=0)
    with crystal_store._transaction() as data:
        state = data.get(FOLD_STATE_KEY)
        prev_mark_str: str | None = None
        prev_mark: datetime | None = None
        if FOLD_STATE_KEY in data:  # present, even as null: it must parse
            # A present but unreadable mark must not read as "never folded": that
            # would count the whole history a second time on top of the stored counts.
            raw = state.get("through") if isinstance(state, dict) else None
            prev_mark = _parse_ts(raw)
            if prev_mark is None:
                raise CrystalError(
                    f"{crystal_store.path} holds an unreadable {FOLD_STATE_KEY!r} mark "
                    f"({state!r}); refusing to fold so no receipt is counted twice. "
                    f"Inspect it by hand."
                )
            prev_mark_str = str(raw)
        mark = cutoff if prev_mark is None or cutoff > prev_mark else prev_mark
        result = FoldResult(previous_mark=prev_mark_str, mark=_fmt_ts(mark))
        missing = [str(p) for p in paths if not p.is_file()]
        if len(missing) == len(paths):
            raise FileNotFoundError(
                f"none of the receipt paths exists ({', '.join(missing)}); the fold "
                f"mark was not moved."
            )
        result.paths_missing = missing
        if prev_mark is not None and cutoff <= prev_mark:
            return result
        live = {
            str(c.get("name")): c
            for c in data.get("crystal", [])
            if isinstance(c, dict) and c.get("status") == "crystallized"
        }
        counts: dict[str, int] = {}
        last_on: dict[str, str] = {}
        seen_events: set[str] = set()
        for path in paths:
            try:
                f = open(path, "r", encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            with f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        receipt = json.loads(line)
                    except ValueError:
                        result.lines_skipped += 1
                        continue
                    if not isinstance(receipt, dict):
                        result.lines_skipped += 1
                        continue
                    ts = _parse_ts(receipt.get("ts"))
                    if ts is None:
                        result.lines_skipped += 1
                        continue
                    if (prev_mark is not None and ts <= prev_mark) or ts > cutoff:
                        continue
                    names = receipt_crystal_names(receipt)
                    if not names:
                        continue
                    # No event_id, no way to tell a rotated copy from a new receipt,
                    # so it is not counted.
                    event_id = receipt.get("event_id")
                    if not isinstance(event_id, str) or not event_id:
                        result.event_id_missing += 1
                        continue
                    if event_id in seen_events:
                        result.duplicates_skipped += 1
                        continue
                    seen_events.add(event_id)
                    result.receipts_folded += 1
                    day = _receipt_day(receipt.get("query_date"), ts)
                    for name in names:
                        if name not in live:
                            result.names_unknown[name] = result.names_unknown.get(name, 0) + 1
                            continue
                        result.exposures_counted += 1
                        counts[name] = counts.get(name, 0) + 1
                        if day > last_on.get(name, ""):
                            last_on[name] = day
        for name, n in counts.items():
            row = live[name]
            prior = row.get("surfaced_count")
            base = prior if isinstance(prior, int) and not isinstance(prior, bool) and prior >= 0 else 0
            row["surfaced_count"] = base + n
            old_day = row.get("last_surfaced_on")
            if not isinstance(old_day, str) or last_on[name] > old_day:
                row["last_surfaced_on"] = last_on[name]
        data[FOLD_STATE_KEY] = {"through": result.mark}
        result.counts = counts
        return result


# ---------------------------------------------------------------------------
# Worth: report-only counters from the outcome log
# ---------------------------------------------------------------------------


@dataclass
class WorthRow:
    """Counters for one crystal or episode.

    ``success`` / ``failure`` are the two Worth counters: labelled exposures of
    this item whose outcome was known, whatever the label (retrieved-with-success
    and retrieved-with-failure). ``table`` splits them by label:
    ``table[label][outcome]`` with outcome ``success``, ``failure`` or ``unknown``,
    so ``table["followed"]["success"]`` is followed-with-success. ``followed`` /
    ``ignored`` / ``not_applicable`` count labels regardless of outcome.

    For an episode, an exposure counts at most once, whether it was labelled
    directly, cited by one or more crystals in the exposure, or both.
    ``credited_success`` / ``credited_failure`` are the part of ``success`` /
    ``failure`` from exposures that reached the episode ONLY through a crystal's
    evidence (the citation edge); those exposures carry no label of their own and
    so are not in ``table``. Evidence is read from the crystal store when the
    report runs, not as it was at exposure time.

    ``unlabelled_success`` / ``unlabelled_failure`` / ``unlabelled_unknown`` count
    exposures that listed this item in ``exposed`` without labelling it, by
    outcome. They are a separate column on purpose and are never in ``success`` /
    ``failure``: a label means something judged the item, an unlabelled exposure
    only that it was surfaced. An unlabelled crystal credits nothing to the
    episodes it cites.

    ``exposed_unrecorded`` (crystals only) counts the receipts passed to
    :func:`compute_worth` that exposed this crystal under an ``event_id`` with NO
    record in the outcome log at all. ``None`` when no receipts were passed, and
    always ``None`` for an episode, because receipts list crystals only.
    """

    kind: str
    ref: str
    success: int = 0
    failure: int = 0
    followed: int = 0
    ignored: int = 0
    not_applicable: int = 0
    credited_success: int = 0
    credited_failure: int = 0
    unlabelled_success: int = 0
    unlabelled_failure: int = 0
    unlabelled_unknown: int = 0
    exposed_unrecorded: int | None = None
    table: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            label: {"success": 0, "failure": 0, "unknown": 0} for label in FOLLOWED_VALUES
        }
    )
    surfaced_count: int | None = None
    last_surfaced_on: str | None = None
    last_activated_on: str | None = None
    live: bool = True

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["table"] = {k: dict(v) for k, v in self.table.items()}
        if d["exposed_unrecorded"] is None:  # no receipts passed: the output is unchanged
            del d["exposed_unrecorded"]
        return d

    def _count_unlabelled(self, outcome: str | None) -> None:
        if outcome == "success":
            self.unlabelled_success += 1
        elif outcome == "failure":
            self.unlabelled_failure += 1
        else:
            self.unlabelled_unknown += 1

    def _count(self, label: str | None, outcome: str | None, *, credited: bool = False) -> None:
        if label is not None:
            setattr(self, label, getattr(self, label) + 1)
            self.table[label][outcome or "unknown"] += 1
        if outcome == "success":
            self.success += 1
            self.credited_success += credited
        elif outcome == "failure":
            self.failure += 1
            self.credited_failure += credited


@dataclass
class WorthReport:
    """The full report. ``crystals`` covers every live crystal (zero rows
    included) plus any labelled or exposed name that is no longer live
    (``live`` False).
    ``exposures`` counts distinct exposure ids after merging. When receipts were
    passed, ``receipts_read`` counts the distinct exposing receipts read and
    ``receipts_skipped`` the ones that were not a dict or carried no ``event_id``
    (they cannot be matched to the log, so they are in no row); both are ``None``
    otherwise."""

    crystals: list[WorthRow]
    episodes: list[WorthRow]
    exposures: int
    lines_skipped: int
    receipts_read: int | None = None
    receipts_skipped: int | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "exposures": self.exposures,
            "lines_skipped": self.lines_skipped,
            "crystals": [r.as_dict() for r in self.crystals],
            "episodes": [r.as_dict() for r in self.episodes],
        }
        if self.receipts_read is not None:
            d["receipts_read"] = self.receipts_read
            d["receipts_skipped"] = self.receipts_skipped
        return d


def load_receipts(
    paths: Iterable[str | os.PathLike[str]],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Read retrieval receipts (JSONL) for :func:`compute_worth`: the receipts, the
    count of lines that are not a JSON object, and the paths that do not exist.
    Raises ``FileNotFoundError`` when none of the paths exists, so a wrong path
    cannot read as "nothing exposed". Pass the live log first and its rotated
    backup after it, as for :func:`fold_surfaced`; duplicates are dropped by
    ``event_id`` in :func:`compute_worth`."""
    ps = [Path(p) for p in paths]
    if not ps:
        raise ValueError("load_receipts needs at least one receipt path.")
    receipts: list[dict[str, Any]] = []
    bad = 0
    missing: list[str] = []
    for path in ps:
        try:
            f = open(path, "r", encoding="utf-8", errors="replace")
        except FileNotFoundError:
            missing.append(str(path))
            continue
        with f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                if isinstance(r, dict):
                    receipts.append(r)
                else:
                    bad += 1
    if len(missing) == len(ps):
        raise FileNotFoundError(f"none of the receipt paths exists ({', '.join(missing)}).")
    return receipts, bad, missing


def compute_worth(
    log: OutcomeLog,
    crystal_store: CrystalStore | None = None,
    receipts: Iterable[Mapping[str, Any]] | None = None,
) -> WorthReport:
    """Build the report-only Worth counters from the outcome log.

    Uses the merged record per exposure id (:meth:`OutcomeLog.latest`). Every
    labelled item counts toward its label; it counts toward ``success`` /
    ``failure`` when the exposure has an outcome. Each crystal in an exposure
    also reaches the episodes in its ``evidence``, whatever the crystal's own label
    (retrieved semantics; the crystal's ``table`` shows how it was labelled), live
    crystals only (a crystal no longer live has no evidence to walk), and each
    episode is counted once per exposure. An item in the record's ``exposed``
    list that this exposure did not label (and, for an episode, did not credit
    through a labelled crystal) counts in the ``unlabelled_*`` columns only.

    ``receipts`` (optional; the harness's retrieval receipts, e.g. from
    :func:`load_receipts`) adds ``exposed_unrecorded`` per crystal: each distinct
    ``event_id`` that exposed the crystal (:func:`receipt_crystal_names`) and has no
    record in the log. It is keyed by event id only: an exposure WITH a record that
    neither labels nor lists the crystal is not counted there. The window is
    whatever receipts are passed, so receipts older than the log count too. A
    receipt name that is not a live crystal gets a row with ``live`` False.
    Nothing here writes anywhere.
    """
    latest, bad = log.latest()
    live: dict[str, dict[str, Any]] = {}
    folded = False
    if crystal_store is not None:
        # ONE read of the document, so the rows and the fold mark come from the same
        # committed state (a fold between two reads would report 0 for a fresh count).
        doc = crystal_store._load()
        for c in doc.get("crystal", []):
            if isinstance(c, dict) and c.get("status") == "crystallized":
                live[str(c.get("name"))] = dict(c)
        # After a fold, a live crystal with no surfaced_count was surfaced zero
        # times. Before the first fold, or with a mark that does not parse, its
        # count is unknown (None), never a false zero.
        state = doc.get(FOLD_STATE_KEY)
        folded = isinstance(state, dict) and _parse_ts(state.get("through")) is not None

    crystals: dict[str, WorthRow] = {}
    episodes: dict[str, WorthRow] = {}
    with_receipts = receipts is not None

    def crow(name: str) -> WorthRow:
        if name not in crystals:
            c = live.get(name)
            crystals[name] = WorthRow(
                "crystal",
                name,
                surfaced_count=(c.get("surfaced_count", 0 if folded else None) if c else None),
                last_surfaced_on=c.get("last_surfaced_on") if c else None,
                last_activated_on=c.get("last_activated_on") if c else None,
                live=c is not None,
                exposed_unrecorded=0 if with_receipts else None,
            )
        return crystals[name]

    def erow(eid: str) -> WorthRow:
        if eid not in episodes:
            episodes[eid] = WorthRow("episode", eid)
        return episodes[eid]

    for name in live:
        crow(name)

    for rec in latest.values():
        outcome = rec.get("outcome")
        direct: dict[str, str] = {}
        cited: set[str] = set()
        for item in rec["items"]:
            if item["kind"] == "crystal":
                crow(item["ref"])._count(item["followed"], outcome)
                for e in live.get(item["ref"], {}).get("evidence") or []:
                    if isinstance(e, str) and e:
                        cited.add(e)
            else:
                direct[item["ref"]] = item["followed"]
        for eid, label in direct.items():
            erow(eid)._count(label, outcome)
        for eid in cited - direct.keys():
            erow(eid)._count(None, outcome, credited=True)
        # Counted already in this exposure: labelled directly, or (an episode)
        # credited through a labelled crystal's evidence. Once per exposure (L1).
        counted = {(i["kind"], i["ref"]) for i in rec["items"]}
        counted |= {("episode", eid) for eid in cited}
        for e in rec["exposed"]:
            if (e["kind"], e["ref"]) in counted:
                continue
            row = crow(e["ref"]) if e["kind"] == "crystal" else erow(e["ref"])
            row._count_unlabelled(outcome)

    receipts_read: int | None = None
    receipts_skipped: int | None = None
    if receipts is not None:
        receipts_read = receipts_skipped = 0
        seen_events: set[str] = set()
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                receipts_skipped += 1
                continue
            names = receipt_crystal_names(receipt)
            if not names:
                continue  # not an exposure
            event_id = receipt.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                receipts_skipped += 1
                continue
            if event_id in seen_events:  # a rotated copy of a receipt already read
                continue
            seen_events.add(event_id)
            receipts_read += 1
            if event_id in latest:
                continue
            for name in names:
                row = crow(name)
                row.exposed_unrecorded = (row.exposed_unrecorded or 0) + 1

    return WorthReport(
        crystals=sorted(crystals.values(), key=lambda r: r.ref),
        episodes=sorted(episodes.values(), key=lambda r: r.ref),
        exposures=len(latest),
        lines_skipped=bad,
        receipts_read=receipts_read,
        receipts_skipped=receipts_skipped,
    )
