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

OPT-IN. The whole layer engages only when a caller passes a ``session_id`` to ``prepare_wrap``.
A caller that does not participate (every existing caller; every single-session adopter) is
unaffected — no registry is touched and the gate is inert. Identity is the CALLER's concern (an
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

import hashlib
import json
import os
import tempfile
import time
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
_SESSION_FILE_SUFFIX = ".session"

PathLike = str | os.PathLike[str]


class CorruptSidecarError(json.JSONDecodeError):
    """A baton or session sidecar was read but does not hold a valid payload: it decoded to
    the wrong JSON shape (``[]``, ``null``, a string, a dict with no usable ``session_id``) or
    its bytes are not UTF-8.

    Subclasses :class:`json.JSONDecodeError` on purpose. The documented contract of
    :func:`baton_holder` and :func:`live_sessions` is that an unreadable sidecar raises
    ``OSError`` or ``JSONDecodeError``, and callers already catch exactly those two to fail
    closed. Before this class existed a wrong-shape baton raised ``AttributeError``
    (``data.get`` on a list), which no caller caught, so ``claim_baton`` crashed instead of
    recovering and ``prepare_wrap`` crashed instead of downgrading (flow spore-1169)."""

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(f"{path.name}: {why}", "", 0)
        self.path = path


class BatonHeldError(AnnealMemoryError):
    """:func:`claim_baton` refused: another session holds the baton, or the baton file is
    unreadable, and the caller did not pass ``take=True``.

    ``holder`` is the other session's id, or ``None`` when ``unreadable`` is true. Taking the
    baton from another session is a deliberate act (⚖ Phill, 2026-09-24, flow spore-1169), so
    automation can never take it by accident."""

    def __init__(self, session_id: str, holder: str | None, *, unreadable: bool) -> None:
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
    and the other gets the error instead of silently overwriting."""
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
    release_baton(continuity_path, session_id)


def live_sessions(
    continuity_path: PathLike, *, ttl: int = DEFAULT_TTL_SECONDS
) -> list[SessionInfo]:
    """All sessions whose last heartbeat is within ``ttl`` seconds. A missing registry dir means
    no one ever registered → empty. Stale (TTL-expired), reaped-mid-scan, or unparseable session
    files are skipped. ONE asymmetry, by design: an individual file's read/parse error skips
    just that file, but a non-``FileNotFoundError`` ``stat`` error (e.g. EACCES on one file)
    propagates and is caught by :func:`consolidate_authorized` as a fail-CLOSED downgrade — the
    safe direction (never silently under-count live others into an over-authorization). Our own
    writes are atomic (``_atomic_write_json``), so an unparseable fresh file does not arise from
    normal operation."""
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
                pid=pid if isinstance(pid, int) else None,
                registered_at=(
                    float(registered_at)
                    if isinstance(registered_at, (int, float))
                    else 0.0
                ),
                last_heartbeat=mtime,
            )
        )
    return out


# -- the consolidate baton --


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read ``path`` as a JSON object. ``FileNotFoundError`` and other ``OSError`` propagate;
    malformed JSON raises ``JSONDecodeError``; non-UTF-8 bytes or a non-object value raise
    :class:`CorruptSidecarError` (itself a ``JSONDecodeError``), so every caller that already
    fails closed on ``(OSError, JSONDecodeError)`` covers the wrong-shape case too."""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CorruptSidecarError(path, "not UTF-8") from exc
    data = json.loads(raw)
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


def baton_holder(continuity_path: PathLike) -> str | None:
    """The session id currently holding the baton, or ``None`` if no baton is claimed. An
    ABSENT baton (``FileNotFoundError``) reads as ``None``; an UNREADABLE or corrupt baton
    file RAISES (``OSError`` / ``JSONDecodeError``; a wrong-shape payload raises
    :class:`CorruptSidecarError`, a ``JSONDecodeError`` subclass) so :func:`consolidate_authorized`
    fails CLOSED rather than silently erasing a designation into a false sole-grant. Absent
    and unreadable are different (the no-data≠no-event distinction, at the collector)."""
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

    - No baton claimed: claim it. The file is created exclusively, so of two sessions racing
      for an unheld baton exactly one wins; the other gets :class:`BatonHeldError`.
    - Already held by ``session_id``: a no-op success. Nothing is written, and the original
      ``claimed_at`` is returned with ``previous_holder == session_id``.
    - Held by ANOTHER session, or the baton file is unreadable: refused with
      :class:`BatonHeldError` unless ``take=True``. ⚖ Phill, 2026-09-24 (flow spore-1169):
      taking the baton from another session must be deliberate, so automation cannot take it
      by accident. This replaces spore-194's last-claim-wins. With ``take=True`` the baton is
      replaced atomically; an unreadable one is replaced too, which is the recovery path for
      a corrupt baton file, and ``previous_holder`` is then ``None``.
    """
    if not session_id:
        raise ValueError("claim_baton: session_id must be non-empty")
    try:
        current = _read_baton(continuity_path)
        unreadable = False
    except (OSError, json.JSONDecodeError):
        current, unreadable = None, True
    previous = None if current is None else str(current["session_id"])
    if previous == session_id:
        claimed_at = current.get("claimed_at") if current is not None else None
        return BatonClaim(
            session_id=session_id,
            claimed_at=(
                float(claimed_at)
                if isinstance(claimed_at, (int, float)) and not isinstance(claimed_at, bool)
                else 0.0
            ),
            previous_holder=session_id,
        )
    if (unreadable or previous is not None) and not take:
        raise BatonHeldError(session_id, previous, unreadable=unreadable)
    claimed_at = time.time()
    payload: dict[str, object] = {
        "session_id": session_id,
        "claimed_at": claimed_at,
        "previous_holder": previous,
    }
    if take:
        _atomic_write_json(_baton_path(continuity_path), payload)
    else:
        try:
            _atomic_write_json(_baton_path(continuity_path), payload, exclusive=True)
        except FileExistsError:
            # Another session claimed between our read and our create. Report whoever holds
            # it now; if that happens to be us (a concurrent claim for the same id), succeed.
            try:
                winner = baton_holder(continuity_path)
            except (OSError, json.JSONDecodeError):
                raise BatonHeldError(session_id, None, unreadable=True) from None
            if winner == session_id:
                return claim_baton(continuity_path, session_id)
            raise BatonHeldError(session_id, winner, unreadable=False) from None
    return BatonClaim(
        session_id=session_id, claimed_at=claimed_at, previous_holder=previous
    )


def release_baton(continuity_path: PathLike, session_id: str) -> bool:
    """Release the baton iff ``session_id`` holds it. Returns whether it was released. A
    session never steals another's release — only the holder (or :func:`close_session`) drops
    it. An unreadable or wrong-shape baton returns ``False`` without raising: ownership cannot
    be confirmed, so it is left for a deliberate ``claim_baton(..., take=True)``.

    Narrow read-then-unlink window: if a concurrent :func:`claim_baton` lands a NEW holder
    between the ``holds_baton`` check and the ``unlink``, this deletes the new holder's baton
    file. The blast radius is bounded + fail-SAFE — the worst outcome is a spurious downgrade of
    the just-designated session (recoverable by re-claim); it can NEVER authorize a second
    consolidator (an unheld baton grants nothing unless the caller opted into
    ``allow_sole_live``, and then only via real session liveness, not the baton). Not worth
    lock machinery for advisory coordination state.
    """
    try:
        held = holds_baton(continuity_path, session_id)
    except (OSError, json.JSONDecodeError):
        return False  # an unreadable baton: can't confirm we hold it → don't unlink another's
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

    Authorized iff this session HOLDS the baton. Otherwise a downgrade. A registry read error
    fails CLOSED (downgrade) — never perform the efferent act under authorization uncertainty.

    ⚖ Phill, 2026-09-24 (flow spore-1105 (b), spore-1169): every consolidate requires the
    baton. Before 0.9.13 a session was also authorized when no OTHER session was live
    (``"sole-live-session"``). That judgement rests on a registry snapshot a resume or a TTL
    crossing can race, and the guard exists so that growing automation cannot recompose the
    felt layer unbidden. ``allow_sole_live=True`` restores the old rule; it is an explicit
    opt-in for a caller that knows it is the only writer, never a default.
    """
    if not session_id:
        raise ValueError("consolidate_authorized: session_id must be non-empty")
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
    live_ids = [s["session_id"] for s in live]
    others = [sid for sid in live_ids if sid != session_id]
    if holder == session_id:
        # I hold the baton → authorized even when others are live (the point of the baton).
        reason = "holds-baton"
        authorized = True
    elif holder is not None:
        # A baton has been CLAIMED by someone else → "designated mode" is active. A claimed
        # baton is a deliberate human designation that OUTRANKS the TTL liveness inference, so
        # the auto-sole-grant below does NOT apply while any baton exists. This closes the hole
        # (L1+L2 MED) where an idle-but-alive baton-holder is TTL-reaped and a parallel lane
        # then reads ITSELF as sole and consolidates unbidden — the exact harm this layer
        # exists to prevent. If the holder is itself TTL-stale we STILL downgrade (the safe,
        # efferent-default direction: we cannot distinguish an idle-alive head from a crashed
        # one), but flag it distinctly so the operator release/re-claims rather than silently
        # losing the human designation. A genuinely-dead holder is recovered by a deliberate
        # claim_baton(..., take=True) or close_session (releases on graceful exit).
        holder_live = holder in live_ids
        reason = (
            "downgraded-not-baton-holder"
            if holder_live
            else "downgraded-stale-baton-holder"
        )
        authorized = False
    elif not allow_sole_live:
        # No baton claimed, and the caller did not opt into the sole-live rule → downgrade,
        # however many sessions are live. Claiming the baton is the way in.
        reason = "downgraded-no-baton"
        authorized = False
    elif not others:
        # Opted in, no baton designated anywhere, and I am the sole live session → authorized
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
        live_session_ids=live_ids,
        baton_holder=holder,
    )
