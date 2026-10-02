"""Outcome write-back, surfaced counts, and Memory-Worth counters (report-only).

anneal can say a pattern was honestly earned (citation-validated graduation). This
module is the start of saying whether a recalled memory ever helped. Three pieces:

1. **Outcome write-back** — :class:`OutcomeLog`. A harness that surfaced memory
   for an *exposure* (one recall event, identified by the harness's own id, e.g.
   a retrieval receipt's ``event_id``) appends what happened: for each surfaced
   item whether it was ``followed``, ``ignored`` or ``not_applicable``, and
   optionally whether the turn's outcome was a ``success`` or ``failure``. The log
   is append-only JSONL beside the episodic db (``<stem>.outcomes.jsonl``). A
   later record for the same exposure id supersedes an earlier one, so a label can
   be corrected by writing it again; nothing is rewritten in place.

2. **Surfaced counts** — :func:`fold_surfaced`. The every-prompt recall hook
   appends receipts lock-free; at wrap time the single writer folds them into
   ``surfaced_count`` / ``last_surfaced_on`` on each live crystal. Exposure is
   NOT activation: this never touches ``last_activated_on``, so the activation
   tier (and therefore re-warm) is unaffected. Counting exposure as activation
   would make recall self-reinforcing.

3. **Worth** — :func:`compute_worth`. Two counters per crystal and per episode:
   retrieved-with-success and retrieved-with-failure, counted only for items the
   harness marked ``followed`` (surfaced AND used). A crystal's counts are also
   credited to the episodes it cites as evidence (the citation edge). The result
   is a REPORT. Nothing in anneal reads it to rank, decay, re-heat or retire
   anything; a downstream failure is not evidence that a followed memory caused it.

Design background: ``Memory Worth`` (arXiv 2604.12007) for the two counters,
PMB's followed / ignored / not-applicable labels, MemQ for credit along the
provenance edges.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .crystal import CrystalStore

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
        ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Append the outcome for one exposure and return the stored record.

        ``items`` must name at least one surfaced item; a repeated (kind, ref)
        keeps its last label. ``outcome`` is ``success``, ``failure`` or ``None``
        (not known yet). Writing the same ``exposure_id`` again supersedes the
        earlier record when the log is read.
        """
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
        if not merged:
            raise ValueError("items must name at least one surfaced item.")
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
        line = (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
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
        """The last record per exposure id, and the count of skipped lines."""
        records, bad = self.read()
        out: dict[str, dict[str, Any]] = {}
        for rec in records:
            out[rec["exposure_id"]] = rec
        return out, bad


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
        if not isinstance(items, list) or not items:
            return None
        for i in items:
            if not isinstance(i, dict):
                return None
            ExposureLabel(i.get("kind"), i.get("ref"), i.get("followed"))  # type: ignore[arg-type]
    except ValueError:
        return None
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

    The whole scan runs inside the crystal store's exclusive lock, so two folds
    cannot both count the same window. ``last_activated_on`` is never written.
    """
    if skew_seconds < 0:
        raise ValueError("skew_seconds must be >= 0.")
    paths = [Path(p) for p in receipt_paths]
    cutoff = (now or datetime.now(timezone.utc)).astimezone(timezone.utc) - timedelta(
        seconds=skew_seconds
    )
    cutoff = cutoff.replace(microsecond=0)
    with crystal_store._transaction() as data:
        state = data.get(FOLD_STATE_KEY)
        prev_mark_str = state.get("through") if isinstance(state, dict) else None
        prev_mark = _parse_ts(prev_mark_str)
        mark = cutoff if prev_mark is None or cutoff > prev_mark else prev_mark
        result = FoldResult(previous_mark=prev_mark_str, mark=_fmt_ts(mark))
        if prev_mark is not None and cutoff <= prev_mark:
            return result
        live = {
            str(c.get("name")): c
            for c in data.get("crystal", [])
            if isinstance(c, dict) and c.get("status") == "crystallized"
        }
        counts: dict[str, int] = {}
        last_on: dict[str, str] = {}
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
                    result.receipts_folded += 1
                    qd = receipt.get("query_date")
                    day = qd if isinstance(qd, str) and len(qd) == 10 else ts.date().isoformat()
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

    ``success`` / ``failure`` are the two Worth counters: exposures where this
    item was ``followed`` and the outcome was known. ``followed`` / ``ignored`` /
    ``not_applicable`` count labels regardless of outcome. For an episode,
    ``credited_success`` / ``credited_failure`` are the share of ``success`` /
    ``failure`` that came from a followed crystal citing it (the citation edge);
    the rest came from the episode being surfaced and followed directly.
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
    surfaced_count: int | None = None
    last_surfaced_on: str | None = None
    last_activated_on: str | None = None
    live: bool = True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class WorthReport:
    """The full report. ``crystals`` covers every live crystal (zero rows
    included) plus any labelled name that is no longer live (``live`` False)."""

    crystals: list[WorthRow]
    episodes: list[WorthRow]
    exposures: int
    lines_skipped: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "exposures": self.exposures,
            "lines_skipped": self.lines_skipped,
            "crystals": [r.as_dict() for r in self.crystals],
            "episodes": [r.as_dict() for r in self.episodes],
        }


def compute_worth(log: OutcomeLog, crystal_store: CrystalStore | None = None) -> WorthReport:
    """Build the report-only Worth counters from the outcome log.

    Uses the latest record per exposure id. An item counts toward ``success`` /
    ``failure`` only when it was ``followed`` and the record carries an outcome.
    A followed crystal's outcome is also credited to every episode in its
    ``evidence`` (live crystals only; a crystal no longer live has no evidence to
    walk). Nothing here writes anywhere.
    """
    latest, bad = log.latest()
    live: dict[str, dict[str, Any]] = {}
    folded = False
    if crystal_store is not None:
        for c in crystal_store.active():
            live[str(c.get("name"))] = dict(c)
        # After any fold, a live crystal with no surfaced_count was surfaced zero
        # times; before the first fold its count is unknown (None).
        folded = isinstance(crystal_store._load().get(FOLD_STATE_KEY), dict)

    crystals: dict[str, WorthRow] = {}
    episodes: dict[str, WorthRow] = {}

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
        for item in rec["items"]:
            row = crow(item["ref"]) if item["kind"] == "crystal" else erow(item["ref"])
            label = item["followed"]
            setattr(row, label, getattr(row, label) + 1)
            if label != "followed" or outcome is None:
                continue
            if outcome == "success":
                row.success += 1
            else:
                row.failure += 1
            if item["kind"] != "crystal":
                continue
            evidence = live.get(item["ref"], {}).get("evidence") or []
            for eid in dict.fromkeys(e for e in evidence if isinstance(e, str) and e):
                er = erow(eid)
                if outcome == "success":
                    er.success += 1
                    er.credited_success += 1
                else:
                    er.failure += 1
                    er.credited_failure += 1

    return WorthReport(
        crystals=sorted(crystals.values(), key=lambda r: r.ref),
        episodes=sorted(episodes.values(), key=lambda r: r.ref),
        exposures=len(latest),
        lines_skipped=bad,
    )
