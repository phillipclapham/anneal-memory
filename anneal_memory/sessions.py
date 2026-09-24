"""anneal_memory/sessions.py — the consolidate-efferent coordination layer (AM-CONSOLIDATE-EFFERENT).

WHY THIS EXISTS. anneal's wrap pipeline already protects the physical write so two consolidates
can never BOTH commit: the save-side ``wrap_token`` compare-and-swap (taken under
``continuity_lock``) is the hard backstop, with ``AM-PREPARE-GUARD`` an in-process guard on top
— soft across PROCESSES, since ``prepare_wrap`` does not hold the flock, so the cross-process
N-conversations model relies on the save-side CAS for the hard guarantee — and even that CAS is
hard only when the caller ROUND-TRIPS the prepare ``wrap_token`` to the save (a tokenless save
CASes against the CURRENT snapshot, so a mid-flight baton reclaim could land one session's text
under another's snapshot; consolidate-efferent callers MUST pass the returned ``wrap_token`` back
to ``validated_save_continuity``). What NONE of that prevents is N *sequential* consolidates from
N different live sessions on the same store: each
is a structurally-valid single wrap that passes the CAS, so the second proceeds the moment the
first clears the in-progress flag. When several agent sessions run in parallel over one store —
the real operating mode of a multi-conversation operator — each can independently recompose the
shared felt/identity continuity layer from its own narrow context. That is the *recency trap*:
no corruption, but the identity memory thrashes (the felt layer recomposed N times from N
partial viewpoints).

THE DISCIPLINE, MADE STRUCTURAL. Capture (episodic append) is AFFERENT — ungated, append-only,
parallel-safe; every session does it. CONSOLIDATE (recomposing the felt continuity layer) is
EFFERENT — it mutates shared identity state, so it is gated by human authority, structurally,
default-absent (the same membrane an autonomic system puts on its efferent edge). The invariant:

    a consolidate proceeds IFF  this session holds the consolidate baton;  otherwise it
    AUTO-DOWNGRADES to capture-only + a flag.

So drift becomes SAFE (a downgrade + a visible flag) rather than a silent identity-thrash.
The baton is claimed by the human designating ONE session as the consolidate seat, and taking
it from another session needs an explicit ``take=True``.

⚖ HISTORY (0.9.13, flow spore-1169). spore-194 also authorized a session when no OTHER session
was live (``sole-live-session``), so a single-session operator never saw the gate. Phill ruled
on 2026-09-24 that every consolidate requires the baton: sole-live is judged from a registry
snapshot that a resume or a TTL crossing can race, and the guard exists so growing automation
cannot recompose the felt layer unbidden. The old rule survives only as an explicit
``allow_sole_live=True`` on :func:`consolidate_authorized` / ``prepare_wrap``. The same day,
last-claim-wins on the baton was replaced by refuse-unless-``take``.

WHERE IT LIVES — a SIDECAR layer, not the DB. This extends anneal's lock-sidecar idiom (the
``.lock`` file ``continuity_lock`` uses), NOT the store's SQLite schema. Session liveness and the
baton are *coordination* state, not memory content; keeping them out of the DB preserves the
store's documented single-process / single-writer design invariant — this layer ENFORCES one
consolidator, it does NOT make concurrent DB writes safe. Sidecars also give crash-safety for
free: a TTL'd heartbeat reaps a session whose process died without closing.

OPT-IN. The layer engages when a caller passes a ``session_id`` to ``prepare_wrap`` (or to
``validated_save_continuity``), or when the store carries the require-baton policy
(``Store.set_consolidate_requires_baton``), which gates every caller. Otherwise a caller that
does not participate is unaffected — no registry is touched and the gate is inert. Identity is the CALLER's concern (an
agent / conversation id, stable for the conversation's lifetime); anneal supplies the registry +
baton + gate, the caller supplies the id and the heartbeat cadence.

PATHS. Everything is keyed off the *resolved* continuity path (the same anchor
``continuity_lock`` uses, ``Path(...).resolve()`` — so symlinked spellings collapse to one
identity; the spore-091 lesson). Sidecars sit next to the continuity file:
``<continuity>.sessions/`` (a dir of per-session files) and ``<continuity>.baton`` (the single
baton file).

⚠ ANCHOR CONSISTENCY (load-bearing — or the gate silently means nothing). All ``sessions.*``
calls for one store AND the ``prepare_wrap`` gate MUST pass the SAME path: use
``store.continuity_path`` everywhere. A caller that registers under a DIFFERENT spelling that
``.resolve()``s elsewhere lands its session files under a different ``.sessions`` dir → the gate
reads an empty registry and a baton claimed under one spelling is invisible under the other.
Under the default rule that costs a spurious downgrade; under ``allow_sole_live`` every
opted-in caller is wrongly ``sole-live-session`` → SILENT no-gating. (``.resolve()`` makes symlink/relative spellings of the SAME file safe; it cannot
rescue two genuinely different paths.) The integration (flow's ``anneal_dualwrite``) registers /
heartbeats / claims against ``store.continuity_path`` unconditionally on every
consolidate-capable conversation — without that discipline the registry is blind and the gate
inert. This layer is an OPT-IN COOPERATIVE protocol; it can only see sessions that participate.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import tempfile
import logging
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypedDict

from .store import AnnealMemoryError

# 90 minutes. Generous enough that an idle-but-still-open conversation is not falsely reaped
# between sparse anneal calls; bounded enough that yesterday's closed conversation is long gone
# by the next day's first wrap. Erring LONG is the safe direction: a falsely-"still-live" stale
# session can only cost an *unnecessary* downgrade (recoverable — claim the baton and re-run),
# never a silent unbidden consolidate, which is the harm the whole layer exists to prevent.
#
# SINGLE-MACHINE / LOCAL FS assumption: liveness is wall-clock TTL over sidecar mtimes and the
# atomicity rests on os.replace / os.link — both assume one machine, a coherent local filesystem, and a
# shared clock. On a networked mount (NFS/SMB) os.replace may not be atomic and each machine
# gets its OWN .sessions namespace (every session reads as sole). The store is laptop-sovereign
# by design, so this holds. The one wall-clock edge is a forward clock jump > TTL (it could
# falsely reap a live peer into a sole-grant under ``allow_sole_live``); by default no
# sole-grant exists, and a claimed baton overrides the TTL inference regardless of the clock.
DEFAULT_TTL_SECONDS = 5400

_SESSIONS_SUFFIX = ".sessions"
_BATON_SUFFIX = ".baton"
_BATON_LOCK_SUFFIX = ".baton.lock"
_SESSION_FILE_SUFFIX = ".session"

PathLike = str | os.PathLike[str]


class CorruptSidecarError(json.JSONDecodeError, AnnealMemoryError):
    """A baton or session sidecar was read but does not hold a valid payload: it decoded to
    the wrong JSON shape (``[]``, ``null``, a string, a dict with no usable ``session_id``), its
    bytes are not UTF-8, or the parser itself gave up on it (an integer past Python's digit
    limit, nesting deep enough to exhaust the recursion limit).

    Subclasses :class:`json.JSONDecodeError` on purpose. The documented contract of
    :func:`baton_holder` and :func:`live_sessions` is that an unreadable sidecar raises
    ``OSError`` or ``JSONDecodeError``, and callers catch exactly those two to fail closed.
    Before this class existed a wrong-shape baton raised ``AttributeError`` (``data.get`` on a
    list), which no caller caught, so ``claim_baton`` crashed instead of recovering and
    ``prepare_wrap`` crashed instead of downgrading (flow spore-1169). It is also an
    :class:`~anneal_memory.AnnealMemoryError`, like every other library error."""

    def __init__(self, path: PathLike, why: str) -> None:
        self.path = Path(path)
        self.why = why
        json.JSONDecodeError.__init__(self, f"{self.path.name}: {why}", "", 0)

    def __str__(self) -> str:
        return self.msg  # no "line 1 column 1": there is no parse position to report

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.path, self.why))


class BatonHeldError(AnnealMemoryError):
    """:func:`claim_baton` refused: another session holds the baton, or the baton file is
    unreadable, and the caller did not pass ``take=True``.

    ``holder`` is the other session's id, or ``None`` when ``unreadable`` is true. Taking the
    baton from another session is a deliberate act (⚖ Phill, 2026-09-24, flow spore-1169), so
    automation can never take it by accident."""

    def __init__(self, session_id: str, holder: str | None, unreadable: bool = False) -> None:
        self.session_id = session_id
        self.holder = holder
        self.unreadable = unreadable
        if unreadable:
            msg = ("claim_baton: the baton file is unreadable, so whether another session "
                   "holds it is unknown; pass take=True only if you know no session does")
        else:
            msg = (f"claim_baton: the baton is held by {holder!r}; pass take=True to take it "
                   "from that session deliberately")
        super().__init__(msg)

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.session_id, self.holder, self.unreadable))


def _finite_float(value: object) -> float:
    """A JSON number as a finite float, or ``0.0``. Untrusted sidecar JSON can hold a bool, an
    integer too large for a float (``OverflowError``), or ``Infinity``/``NaN``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        f = float(value)
    except OverflowError:
        return 0.0
    return f if math.isfinite(f) else 0.0


class SessionInfo(TypedDict):
    """A live (or recently-live) registered session. ``last_heartbeat`` is the session file's
    mtime — heartbeats touch mtime rather than rewrite the file, so it is the authoritative
    liveness timestamp (the stored payload carries only the immutable registration facts)."""

    session_id: str
    label: str | None
    pid: int | None
    registered_at: float
    last_heartbeat: float


class BatonClaim(TypedDict):
    """The result of :func:`claim_baton` — who now holds the baton, when, and who held it
    before (``None`` if it was unheld). The previous holder is surfaced so a re-designation is
    visible rather than silent."""

    session_id: str
    claimed_at: float
    previous_holder: str | None


class ConsolidateAuth(TypedDict):
    """The efferent-gate decision (see :func:`consolidate_authorized`). ``reason`` is one of:
    ``"holds-baton"`` (authorized) · ``"sole-live-session"`` (authorized, only when the caller
    passed ``allow_sole_live=True``) · ``"downgraded-no-baton"`` (no baton is claimed; claim it
    to consolidate) · ``"downgraded-not-baton-holder"`` (a LIVE baton-holder is someone else,
    or, under ``allow_sole_live``, other sessions are live and no baton is claimed) ·
    ``"downgraded-stale-baton-holder"`` (a baton was claimed by a session no longer live — the
    designated head went quiet; release or re-claim the baton) · ``"downgraded-registry-error"``
    (the registry or baton was unreadable — fail closed). Every ``downgraded-*`` reason →
    auto-downgrade to capture-only. ``live_session_ids`` and ``baton_holder`` are carried so the caller can
    surface an actionable flag on a downgrade."""

    authorized: bool
    reason: str
    session_id: str
    live_session_ids: list[str]
    baton_holder: str | None


# -- path derivation (resolved-anchor, the continuity_lock idiom) --


def _anchor(continuity_path: PathLike) -> Path:
    # resolve(strict=False): follow directory symlinks but tolerate a not-yet-existing
    # continuity leaf (first wrap), so sidecars are the file's resolved identity regardless of
    # how the caller spelled the path — the spore-091 non-serialization lesson.
    return Path(continuity_path).resolve()


def _registry_dir(continuity_path: PathLike) -> Path:
    cp = _anchor(continuity_path)
    return cp.with_name(cp.name + _SESSIONS_SUFFIX)


def _baton_path(continuity_path: PathLike) -> Path:
    cp = _anchor(continuity_path)
    return cp.with_name(cp.name + _BATON_SUFFIX)


def _session_file(continuity_path: PathLike, session_id: str) -> Path:
    # The caller-supplied session_id can be any string (a uuid, a path-ish id); hash it to a
    # fixed safe filename component. The true id round-trips via the file's JSON payload, so the
    # filename never has to be reversible.
    # usedforsecurity=False: this is a filename derivation, not a security hash — and it keeps
    # _session_file working in FIPS-strict environments where a bare sha1() call raises.
    digest = hashlib.sha1(session_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return _registry_dir(continuity_path) / f"{digest}{_SESSION_FILE_SUFFIX}"


def _atomic_write_json(
    path: Path, payload: dict[str, object], *, exclusive: bool = False
) -> None:
    """Create-or-replace ``path`` with ``payload`` atomically (tmp in the same dir → fsync →
    ``os.replace``), so a concurrent reader never sees a half-written file and a crash leaves
    either the old file or the new, never a torn one.

    ``exclusive=True`` only creates: the tmp is published with ``os.link``, which raises
    ``FileExistsError`` if ``path`` already exists, so of two racing creators exactly one wins
    and the other gets the error instead of silently overwriting. It needs hard links: on a
    filesystem without them ``os.link`` raises and the write fails closed, never falling back
    to an overwrite (an O_EXCL fallback was tried and withdrawn in L3, rounds 2-4: every
    version of its cleanup could delete a racer's baton or leave a wedged one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        fh = os.fdopen(fd, "w", encoding="utf-8")  # transfers fd ownership to fh
    except BaseException:
        os.close(fd)  # fdopen failed before taking ownership → close the raw fd ourselves
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        with fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        if exclusive:
            os.link(tmp, path)  # FileExistsError when another creator already won
        else:
            os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # a cleanup failure must not mask the original write/replace error
            pass
        raise
    if exclusive:
        try:
            os.unlink(tmp)  # the link published it; the tmp name is now only debris
        except OSError:
            pass


# -- session registry --


def register_session(
    continuity_path: PathLike, session_id: str, *, label: str | None = None
) -> None:
    """Register (or re-register) a live session. Idempotent — re-registering the same
    ``session_id`` refreshes its registration. Call once at conversation start; keep it live
    with :func:`heartbeat`."""
    if not session_id:
        raise ValueError("register_session: session_id must be non-empty")
    payload: dict[str, object] = {
        "session_id": session_id,
        "label": label,
        "pid": os.getpid(),
        "registered_at": time.time(),
    }
    _atomic_write_json(_session_file(continuity_path, session_id), payload)


def heartbeat(continuity_path: PathLike, session_id: str) -> None:
    """Mark ``session_id`` alive *now* (touch its file's mtime — the liveness timestamp). If the
    session was never registered (or was reaped/closed), this re-registers it, so a single
    ``heartbeat`` call on every store op is sufficient to keep a session live without a separate
    register step."""
    if not session_id:
        raise ValueError("heartbeat: session_id must be non-empty")
    path = _session_file(continuity_path, session_id)
    try:
        now = time.time()
        os.utime(path, (now, now))
    except FileNotFoundError:
        register_session(continuity_path, session_id)


def close_session(continuity_path: PathLike, session_id: str) -> None:
    """Unregister a session (conversation end). Best-effort — a never-closed session is reaped
    by TTL anyway. Also releases the baton if this session held it: a claimed baton OUTRANKS TTL
    liveness (it never expires on its own), so a held baton outliving its session leaves a
    phantom designation that downgrades every parallel non-holder (the
    ``downgraded-stale-baton-holder`` path) until a human re-claims. Releasing on graceful exit
    avoids that; an unclean death (no close) leaves the phantom, recovered by a deliberate
    ``claim_baton(..., take=True)``."""
    try:
        _session_file(continuity_path, session_id).unlink()
    except FileNotFoundError:
        pass
    finally:
        # The release runs even if the unlink failed: a held baton outliving its session is
        # the worse outcome.
        try:
            release_baton(continuity_path, session_id)
        except OSError as exc:
            # Best-effort teardown must not fail closing a session, but a baton this session
            # may still hold is exactly the phantom designation described above: say so. The
            # warning itself must not raise (-W error), so fall back to the logger.
            msg = (
                f"close_session: could not release the baton for {session_id!r} "
                f"({type(exc).__name__}: {exc}); if this session held it, it is still held."
            )
            try:
                warnings.warn(msg, stacklevel=2)
            except Exception:
                try:
                    logging.getLogger(__name__).warning(msg)
                except Exception:
                    pass  # the teardown contract outranks the report


def live_sessions(
    continuity_path: PathLike, *, ttl: int = DEFAULT_TTL_SECONDS
) -> list[SessionInfo]:
    """All sessions whose last heartbeat is within ``ttl`` seconds. A missing registry dir means
    no one ever registered → empty. Stale (TTL-expired) files and files reaped mid-scan are
    skipped. A FRESH file that cannot be read, cannot be parsed, or names no session RAISES
    (``OSError`` or ``JSONDecodeError``, the latter including :class:`CorruptSidecarError`):
    it is an unknown live peer, and skipping it would under-count live others into an
    over-authorization under ``allow_sole_live``. :func:`consolidate_authorized` catches both
    and fails CLOSED. Our own writes are atomic (``_atomic_write_json``), so an unparseable
    fresh file does not arise from normal operation."""
    rd = _registry_dir(continuity_path)
    now = time.time()
    out: list[SessionInfo] = []
    try:
        entries = list(rd.iterdir())
    except FileNotFoundError:
        return out
    for f in entries:
        if f.suffix != _SESSION_FILE_SUFFIX:
            continue
        try:
            mtime = f.stat().st_mtime
        except FileNotFoundError:
            continue  # reaped between iterdir() and stat()
        if now - mtime > ttl:
            continue  # stale → treat as dead (left on disk; reaped lazily, not raced-GC'd)
        # FRESH file → a possibly-live peer. An unreadable/corrupt fresh file is
        # authorization-relevant UNKNOWN, not absent: do NOT swallow it (swallowing under-counts
        # a live peer into a false sole-grant). Let OSError / JSONDecodeError propagate →
        # consolidate_authorized fails CLOSED. Only FileNotFoundError (reaped mid-scan = gone)
        # skips. A STALE corrupt file never reaches here (the TTL check above skipped it — a
        # dead session's corruption is irrelevant). Atomic writes mean a fresh corrupt file
        # does not arise from our own writes.
        try:
            data = _read_json_object(f)  # JSONDecodeError + non-ENOENT OSError propagate
        except FileNotFoundError:
            continue
        sid = data.get("session_id")
        if not isinstance(sid, str) or not sid:
            # A fresh file with no usable id is an unknown peer, the same as a corrupt one:
            # skipping it would under-count live others into a false sole-grant.
            raise CorruptSidecarError(f, "no usable session_id")
        # Coerce the non-key fields from untrusted JSON to their declared types — a malformed
        # file must not hand a downstream consumer a wrong-typed pid/label/registered_at.
        label = data.get("label")
        pid = data.get("pid")
        registered_at = data.get("registered_at")
        out.append(
            SessionInfo(
                session_id=sid,
                label=label if isinstance(label, str) else None,
                pid=pid if isinstance(pid, int) and not isinstance(pid, bool) else None,
                registered_at=_finite_float(registered_at),
                last_heartbeat=mtime,
            )
        )
    return out


# -- the consolidate baton --


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read ``path`` as a JSON object. ``FileNotFoundError`` and other ``OSError`` propagate;
    malformed JSON raises ``JSONDecodeError``; anything else the parse can raise (non-UTF-8
    bytes, an integer past the digit limit, nesting past the recursion limit) and a value that
    is not an object raise :class:`CorruptSidecarError`, itself a ``JSONDecodeError``. So a
    caller that fails closed on ``(OSError, JSONDecodeError)`` covers every unreadable file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorruptSidecarError(path, "not UTF-8") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CorruptSidecarError(path, f"unparseable ({type(exc).__name__})") from exc
    if not isinstance(data, dict):
        raise CorruptSidecarError(path, f"decoded to {type(data).__name__}, not an object")
    return data


def _read_baton(continuity_path: PathLike) -> dict[str, Any] | None:
    """The validated baton payload, or ``None`` if no baton file exists. Raises ``OSError`` /
    ``JSONDecodeError`` (including :class:`CorruptSidecarError`) when the file is unreadable or
    does not name a holder: a baton file with no usable ``session_id`` is corrupt, not absent."""
    path = _baton_path(continuity_path)
    try:
        data = _read_json_object(path)
    except FileNotFoundError:
        return None
    sid = data.get("session_id")
    if not isinstance(sid, str) or not sid:
        raise CorruptSidecarError(path, "no usable session_id")
    return data


@contextmanager
def _baton_lock(continuity_path: PathLike) -> Iterator[bool]:
    """Serialize baton claims and releases across processes with an exclusive advisory
    ``flock`` on ``<continuity>.baton.lock``, so a release's check-then-unlink cannot delete a
    baton that a concurrent take just wrote, and two takes cannot both report success.

    Yields whether the lock is held. Like ``continuity_lock``, it degrades to a no-op
    (yielding ``False``) where ``fcntl`` is missing (Windows) or the filesystem refuses
    ``flock``; :func:`claim_baton` then falls back to an exclusive create for an unheld baton.
    Readers never take it: every write is an atomic replace, so a read sees the old file or the
    new one."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX CI; the Windows job covers it
        yield False
        return
    cp = _anchor(continuity_path)
    lock_path = cp.with_name(cp.name + _BATON_LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in (errno.ENOLCK, errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
            acquired = False
        else:
            acquired = True
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def baton_holder(continuity_path: PathLike) -> str | None:
    """The session id currently holding the baton, or ``None`` if no baton is claimed. An
    ABSENT baton (``FileNotFoundError``) reads as ``None``; an UNREADABLE or corrupt baton
    file RAISES (``OSError`` / ``JSONDecodeError``; a wrong-shape or unparseable payload raises
    :class:`CorruptSidecarError`, a ``JSONDecodeError`` subclass) so
    :func:`consolidate_authorized` fails CLOSED rather than silently erasing a designation into
    a false grant. Absent and unreadable are different (the no-data≠no-event distinction, at
    the collector)."""
    data = _read_baton(continuity_path)
    return None if data is None else str(data["session_id"])


def holds_baton(continuity_path: PathLike, session_id: str) -> bool:
    """Does ``session_id`` currently hold the baton?"""
    return baton_holder(continuity_path) == session_id


def claim_baton(
    continuity_path: PathLike, session_id: str, *, take: bool = False
) -> BatonClaim:
    """Claim the consolidate baton for ``session_id`` (the human designating the integrator /
    consolidate seat — govern-not-trust: the human assigns authority).

    - No baton claimed: claim it.
    - Already held by ``session_id``: a no-op success. Nothing is written, and the original
      ``claimed_at`` is returned with ``previous_holder == session_id``.
    - Held by ANOTHER session, or the baton file is unreadable: refused with
      :class:`BatonHeldError` unless ``take=True``. ⚖ Phill, 2026-09-24 (flow spore-1169):
      taking the baton from another session must be deliberate, so automation cannot take it
      by accident. This replaces spore-194's last-claim-wins. With ``take=True`` the baton is
      replaced atomically; an unreadable one is replaced too, which is the recovery path for
      a corrupt baton file, and ``previous_holder`` is then ``None``.

    The decision and the write happen under :func:`_baton_lock`, so a concurrent claim or
    release cannot interleave. Where the lock is unavailable, an unheld baton is created
    exclusively instead (of two racing claimers exactly one wins), retried a bounded number of
    times if the baton keeps changing underneath. That create needs hard links: on a filesystem
    with neither ``flock`` nor hard links it raises the underlying ``OSError`` with nothing
    written, and ``take=True`` claims by deliberate overwrite instead.

    ⚠ This function cannot tell a human designation from an agent calling it: anyone who calls
    it while the baton is unheld gets it. Keeping the baton away from automation is the
    operator's part of the protocol (⚖ Phill: "this is also on human operator to manage").
    """
    if not session_id:
        raise ValueError("claim_baton: session_id must be non-empty")
    if not isinstance(take, bool):  # take="false" is truthy: never infer an authority transfer
        raise TypeError(f"claim_baton: take must be a bool, got {type(take).__name__}")
    path = _baton_path(continuity_path)
    with _baton_lock(continuity_path) as locked:
        for _attempt in range(3):
            try:
                current = _read_baton(continuity_path)
                unreadable = False
            except (OSError, json.JSONDecodeError):
                current, unreadable = None, True
            previous = None if current is None else str(current["session_id"])
            if current is not None and previous == session_id:
                return BatonClaim(
                    session_id=session_id,
                    claimed_at=_finite_float(current.get("claimed_at")),
                    previous_holder=session_id,
                )
            if (unreadable or previous is not None) and not take:
                raise BatonHeldError(session_id, previous, unreadable)
            claimed_at = time.time()
            payload: dict[str, object] = {
                "session_id": session_id,
                "claimed_at": claimed_at,
                "previous_holder": previous,
            }
            try:
                # An unheld baton without the lock: create-only, so a racing claimer loses
                # loudly instead of being silently overwritten.
                _atomic_write_json(path, payload, exclusive=not (locked or take))
            except FileExistsError:
                continue  # someone claimed it since our read: decide again against theirs
            return BatonClaim(
                session_id=session_id, claimed_at=claimed_at, previous_holder=previous
            )
    raise BatonHeldError(session_id, None, True)  # it kept changing: treat it as unknown


def release_baton(continuity_path: PathLike, session_id: str) -> bool:
    """Release the baton iff ``session_id`` holds it. Returns whether it was released. A
    session never steals another's release — only the holder (or :func:`close_session`) drops
    it. An unreadable or wrong-shape baton returns ``False`` without raising: ownership cannot
    be confirmed, so it is left for a deliberate ``claim_baton(..., take=True)``.

    The check and the unlink run under :func:`_baton_lock`, so a concurrent take cannot land
    between them and be deleted, which would leave the baton unheld and claimable by anyone.
    Where the lock is unavailable that window remains (see :func:`_baton_lock`).
    """
    try:
        if not _baton_path(continuity_path).exists():
            return False  # nothing to release; no lock, and no lock file created
    except OSError:
        return False  # cannot even stat it: ownership unconfirmable
    with _baton_lock(continuity_path):
        try:
            held = holds_baton(continuity_path, session_id)
        except (OSError, json.JSONDecodeError):
            return False  # unreadable: can't confirm we hold it → don't unlink another's
        if not held:
            return False
        try:
            _baton_path(continuity_path).unlink()
        except FileNotFoundError:
            pass
        return True


# -- the efferent gate (the one entry the wrap pipeline calls) --


def consolidate_authorized(
    continuity_path: PathLike,
    session_id: str,
    *,
    ttl: int = DEFAULT_TTL_SECONDS,
    allow_sole_live: bool = False,
) -> ConsolidateAuth:
    """The efferent-gate decision for ``session_id``: may it CONSOLIDATE (recompose the felt
    layer), or must it auto-downgrade to capture-only?

    Authorized iff this session HOLDS the baton. Otherwise a downgrade. An unreadable baton
    fails CLOSED (downgrade) — never perform the efferent act under authorization uncertainty.
    By default the session registry is read only to describe the result (``live_session_ids``,
    and the stale-holder reason); an unreadable registry leaves ``live_session_ids`` empty and
    decides nothing. Under ``allow_sole_live=True`` liveness can authorize, so an unreadable
    registry fails closed there too.

    ⚖ Phill, 2026-09-24 (flow spore-1105 (b), spore-1169): every consolidate requires the
    baton. Before 0.9.13 a session was also authorized when no OTHER session was live
    (``"sole-live-session"``). That judgement rests on a registry snapshot a resume or a TTL
    crossing can race, and the guard exists so that growing automation cannot recompose the
    felt layer unbidden. ``allow_sole_live=True`` restores the old rule; it is an explicit
    opt-in for a caller that knows it is the only writer, never a default.
    """
    if not session_id:
        raise ValueError("consolidate_authorized: session_id must be non-empty")
    if not isinstance(allow_sole_live, bool):
        raise TypeError(
            "consolidate_authorized: allow_sole_live must be a bool, "
            f"got {type(allow_sole_live).__name__}"
        )
    if not allow_sole_live:
        # The default rule needs only the baton. The registry is read for the message and the
        # stale-holder distinction, and a bad peer file must not block the holder: liveness
        # plays no part in this decision.
        try:
            holder = baton_holder(continuity_path)
        except (OSError, json.JSONDecodeError):
            return ConsolidateAuth(
                authorized=False,
                reason="downgraded-registry-error",
                session_id=session_id,
                live_session_ids=[],
                baton_holder=None,
            )
        try:
            live_ids: list[str] | None = [
                s["session_id"] for s in live_sessions(continuity_path, ttl=ttl)
            ]
        except (OSError, json.JSONDecodeError):
            live_ids = None  # unknown: never used to authorize, only to describe
        if holder == session_id:
            reason, authorized = "holds-baton", True
        elif holder is None:
            reason, authorized = "downgraded-no-baton", False
        elif live_ids is not None and holder not in live_ids:
            reason, authorized = "downgraded-stale-baton-holder", False
        else:
            reason, authorized = "downgraded-not-baton-holder", False
        return ConsolidateAuth(
            authorized=authorized,
            reason=reason,
            session_id=session_id,
            live_session_ids=live_ids or [],
            baton_holder=holder,
        )
    # allow_sole_live=True: spore-194's rule, where liveness can authorize, so an unreadable
    # registry fails CLOSED.
    try:
        live = live_sessions(continuity_path, ttl=ttl)
        holder = baton_holder(continuity_path)
    except (OSError, json.JSONDecodeError):
        return ConsolidateAuth(
            authorized=False,
            reason="downgraded-registry-error",
            session_id=session_id,
            live_session_ids=[],
            baton_holder=None,
        )
    live_ids_ = [s["session_id"] for s in live]
    others = [sid for sid in live_ids_ if sid != session_id]
    if holder == session_id:
        reason = "holds-baton"
        authorized = True
    elif holder is not None:
        # A baton has been CLAIMED by someone else → "designated mode" is active. A claimed
        # baton is a deliberate human designation that OUTRANKS the TTL liveness inference, so
        # the sole grant below does NOT apply while any baton exists. This closes the hole
        # (L1+L2 MED) where an idle-but-alive baton-holder is TTL-reaped and a parallel lane
        # then reads ITSELF as sole and consolidates unbidden. A TTL-stale holder still
        # downgrades (we cannot distinguish an idle-alive head from a crashed one), flagged
        # distinctly so the operator re-designates deliberately.
        reason = (
            "downgraded-not-baton-holder"
            if holder in live_ids_
            else "downgraded-stale-baton-holder"
        )
        authorized = False
    elif not others:
        # No baton designated anywhere, and I am the sole live session → authorized
        # (spore-194's single-session rule; never reached once any baton is claimed).
        reason = "sole-live-session"
        authorized = True
    else:
        reason = "downgraded-not-baton-holder"
        authorized = False
    return ConsolidateAuth(
        authorized=authorized,
        reason=reason,
        session_id=session_id,
        live_session_ids=live_ids_,
        baton_holder=holder,
    )
