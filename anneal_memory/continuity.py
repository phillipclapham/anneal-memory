"""The two calls of a wrap: ``prepare_wrap`` and ``validated_save_continuity``.

A wrap is a rolling rewrite (see :mod:`anneal_memory.rolling`). ``prepare_wrap`` opens
one step: it freezes one day's episodes, mints a token, and returns the prompt to run
(the measured updater prompt, filled with the current memory file and that day's log).
``validated_save_continuity`` takes the model's whole reply and keeps it, together with
every earlier version. A catch-up of several days is several steps, one after the other,
each on the file the step before it saved.

Around those two calls sits the store's wrap lifecycle, unchanged: one wrap open at a
time, a token that must come back, the consolidate baton (``session_id``), the
consolidation lease, idempotency keys, and the append-only version history.

Zero dependencies beyond the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import uuid
import warnings
from datetime import tzinfo
from typing import TYPE_CHECKING, Any, cast

from . import rolling, sessions
from .store import (
    IdempotencyConflictError,
    LeaseFencedError,
    _IDEMPOTENCY_SCOPE,
    _IdempotentReplay,
    _idempotency_request_hash,
    _validate_idempotency_key,
    SaveAuthorityError,
    StoreDatabaseError,
    StoreError,
    WrapContinuityMovedError,
    WrapInProgressError,
    WrapOwnershipError,
    WrapWindowMovedError,
)
from .types import (
    Episode,
    FeltCurrency,
    PrepareWrapResult,
    SaveContinuityReplayResult,
    SaveContinuityResult,
    WrapStepDict,
)

if TYPE_CHECKING:
    from .store import Store

_log = logging.getLogger(__name__)

# prepare_wrap's caller-supplied wrap_token: the shape uuid.uuid4().hex mints, so
# a caller token is indistinguishable from a minted one wherever tokens are used.
_CALLER_TOKEN_RE = re.compile(r"[0-9a-f]{32}")

# The save key row's fields a replay returns: an allowlist, so a field the row ever
# gains never reaches the caller unnamed here.
_SAVE_REPLAY_FIELDS = ("wrap_id", "chars", "path", "content_hash", "episodes_compressed")


def _pending_count(store: Store) -> int:
    """The episodes waiting since the last wrap. On a failed read the value is 1, a floor
    and not a count: a caller treats a non-zero count as "prepare again", which answers
    "empty" when nothing is left, while 0 would hide a day that is still owed. A save
    that committed must still return its result, so the read never raises here."""
    try:
        return store.count_episodes_since_wrap()
    except StoreError:
        return 1


def _no_step(status: str, message: str, pending: int) -> PrepareWrapResult:
    """A result that opens no step (the store untouched, or an idle one)."""
    return PrepareWrapResult(
        status=cast(Any, status),
        message=message,
        episode_count=0,
        pending_count=pending,
        wrap_token=None,
        step=None,
    )


def _claim_recipe(store: Store, session_id: str | None, episode_count: int) -> str:
    """The cadence line a downgrade carries: how much is waiting, and the one command
    that lets a named session consolidate. Consolidation is the operator's act, so the
    command is addressed to the operator: an agent shows it, never runs it. The baton
    is read here so the command carries ``--take`` exactly when a plain claim would be
    refused (held by another session, or unreadable)."""
    if episode_count == 0:
        return "Nothing is waiting to be consolidated yet."
    try:
        holder = sessions.baton_holder(store.continuity_path)
        take = " --take" if holder is not None and holder != session_id else ""
    except (OSError, json.JSONDecodeError):
        take = " --take"
    db = shlex.quote(str(store.path))
    waiting = (
        f"{episode_count} episode(s) since the last wrap are waiting to be consolidated. "
        "Show your human this; do not run it yourself: "
    )
    if session_id is None:
        return waiting + (
            "to consolidate, the operator gives the wrap a session id (CLI: --session-id "
            "NAME; MCP: start the server with --session-id NAME) and claims the baton for "
            f"it once: anneal-memory --db {db} baton claim NAME{take}"
        )
    return waiting + (
        "to let this session consolidate, the operator runs: "
        f"anneal-memory --db {db} baton claim {shlex.quote(session_id)}{take}"
    )


def _consolidate_gate(
    store: Store,
    session_id: str | None,
    allow_sole_live: bool,
    episode_count: int,
) -> PrepareWrapResult | None:
    """The consolidate gate as ONE decision, run at every point ``prepare_wrap`` is about
    to touch the wrap lifecycle (the empty-window cancel, and again just before
    ``wrap_started``). Returns the ``"downgraded"`` result when the caller is not
    authorized (the store untouched), else ``None``.

    Opt-in: a caller that passes no ``session_id`` on a store without the require-baton
    policy is never gated. On a policy store every caller is, and ``allow_sole_live`` is
    ignored."""
    requires_baton = store.consolidate_requires_baton()
    if session_id is None and requires_baton:
        return _no_step(
            "downgraded",
            "Consolidate downgraded to capture-only (downgraded-baton-required): this "
            "store is baton-protected (Store.consolidate_requires_baton), so only the "
            "session the operator has given the consolidate baton can consolidate it, "
            "identifying itself with session_id. This call passed none. Capture "
            "(afferent) is unaffected. " + _claim_recipe(store, None, episode_count),
            episode_count,
        )
    if session_id is None:
        return None
    if not session_id:
        raise ValueError(
            "prepare_wrap: session_id must be non-empty when provided "
            "(pass session_id=None to disable the consolidate gate)."
        )
    auth = sessions.consolidate_authorized(
        store.continuity_path,
        session_id,
        # pass it through as given: consolidate_authorized refuses a non-bool
        allow_sole_live=False if requires_baton else allow_sole_live,
    )
    if auth["authorized"]:
        return None
    return _no_step(
        "downgraded",
        f"Consolidate downgraded to capture-only ({auth['reason']}): "
        f"{len(auth['live_session_ids'])} live session(s), baton holder = "
        f"{auth['baton_holder'] or 'none'}. Capture (afferent) is unaffected. "
        f"A consolidate needs the baton, which the operator assigns to one "
        f"session; a session does not claim it on its own initiative. "
        + _claim_recipe(store, session_id, episode_count),
        episode_count,
    )


def _rows(episodes: list[Episode]) -> list[tuple[str, str, str, str]]:
    # The prompt's log line reads ``(observation)``. ``str()`` of the enum member is
    # ``EpisodeType.OBSERVATION`` on current Pythons, so the string value is taken.
    return [
        (ep.id, ep.timestamp, getattr(ep.type, "value", str(ep.type)), ep.content)
        for ep in episodes
    ]


def _fingerprint(episode_ids: list[str], memory: str | None, max_chars: int) -> str:
    """sha256 of what a keyed prepare's prompt is built FROM, never of the prompt (which
    carries episode text that would outlive an episode's erasure): the frozen episode
    ids, the memory file's hash and the size limit."""
    return hashlib.sha256(json.dumps({
        "episode_ids": sorted(episode_ids),
        "memory_sha256": hashlib.sha256((memory or "").encode("utf-8")).hexdigest(),
        "max_chars": max_chars,
    }, sort_keys=True).encode("utf-8")).hexdigest()


def _ready(
    step: rolling.Step,
    steps_after: int,
    pending: int,
    memory: str | None,
    max_chars: int,
    wrap_token: str,
) -> PrepareWrapResult:
    system, user = rolling.build_prompt(memory, step, max_chars=max_chars)
    view = WrapStepDict(
        day=step.day,
        chunk=step.chunk,
        chunks=step.chunks,
        episodes=len(step.episode_ids),
        steps_remaining=steps_after,
        max_chars=max_chars,
        memory_chars=len(memory) if memory else 0,
        system=system,
        user=user,
        prompt=rolling.render_prompt_text(system, user),
    )
    more = f" {steps_after} more step(s) follow." if steps_after else ""
    odd = (
        f" {step.unreadable} episode(s) have a timestamp that could not be read; "
        f"they are filed by its date prefix, else under {rolling.UNREADABLE_DAY}."
        if step.unreadable else ""
    )
    return PrepareWrapResult(
        status="ready",
        message=f"Ready: {len(step.episode_ids)} episode(s) of {step.day}.{more}{odd}",
        episode_count=len(step.episode_ids),
        pending_count=pending,
        wrap_token=wrap_token,
        step=view,
    )


def _replay_prepare(
    store: Store, result_json: str | None, *, max_chars: int, tz: tzinfo | None,
) -> PrepareWrapResult:
    """A keyed ``prepare_wrap`` retry: its key row holds the wrap token and the sha256
    of what the prompt was built from. While that wrap is the open one, the prompt is
    rebuilt from its frozen snapshot and returned only when its fingerprint is the
    stored one; otherwise ``IdempotencyConflictError`` with ``reason="sources_moved"``.
    Once the wrap is saved or cancelled, ``reason="wrap_closed"``."""
    stored = json.loads(result_json or "{}")
    wrap_token = stored.get("wrap_token") if isinstance(stored, dict) else None
    snapshot = store.load_wrap_snapshot()
    if snapshot is None or wrap_token is None or snapshot["token"] != wrap_token:
        raise IdempotencyConflictError(
            f"prepare_wrap: this idempotency key opened wrap {wrap_token!r}, which "
            "is no longer open (saved or cancelled), so there is no prompt to "
            "return. Nothing was written. Prepare again with a new key.",
            operation="wrap_started", path=str(store.path), reason="wrap_closed")
    frozen = set(snapshot["episode_ids"])
    window = store.episodes_since_wrap()
    step_eps = [ep for ep in window if ep.id in frozen]
    # The step is bound to its episode IDS: one deleted or pruned since would be missing
    # from the rebuilt prompt, and the save would still mark it wrapped.
    if len(step_eps) != len(frozen):
        raise IdempotencyConflictError(
            "prepare_wrap: this idempotency key's step can no longer be rebuilt: "
            f"{len(frozen) - len(step_eps)} of the episode(s) frozen into wrap "
            f"{wrap_token!r} were deleted or pruned since it was opened, so the step's "
            "episodes changed and no prompt is returned. Nothing was written. Cancel "
            "the wrap and prepare again with a new key.",
            operation="wrap_started", path=str(store.path), reason="sources_moved")
    memory = store.load_continuity()
    steps = rolling.plan_steps(_rows(step_eps), tz=tz)
    if len(steps) != 1 or _fingerprint(
            snapshot["episode_ids"], memory, max_chars) != stored.get("package_sha256"):
        raise IdempotencyConflictError(
            "prepare_wrap: this idempotency key's prompt can no longer be rebuilt "
            f"as it was returned (its sources moved since wrap {wrap_token!r} was "
            "opened), so no prompt is returned. Nothing was written; the wrap stays "
            "open. Save it from the first prompt, or cancel it and prepare again "
            "with a new key.",
            operation="wrap_started", path=str(store.path), reason="sources_moved")
    rest = rolling.plan_steps(_rows([ep for ep in window if ep.id not in frozen]), tz=tz)
    return _ready(steps[0], len(rest), len(window), memory, max_chars, wrap_token)


def prepare_wrap(
    store: Store,
    *,
    max_chars: int | None = None,
    tz: str | tzinfo | None = None,
    session_id: str | None = None,
    allow_sole_live: bool = False,
    wrap_token: str | None = None,
    idempotency_key: str | None = None,
    lease_token: str | None = None,
) -> PrepareWrapResult:
    """Open one step of the rolling wrap and return the prompt to run.

    **This is the canonical prepare entry point.** The MCP server and the CLI call this
    function; they only parse their transport's input and print the result.

    The step is the oldest day of episodes since the last wrap (a day is a calendar day
    in ``tz``, the machine's own zone by default; a day whose log passes
    :data:`anneal_memory.rolling.CHUNK_MAX` characters is cut into slices). The result's
    ``step`` holds ``system`` and ``user``, the two messages of the measured updater
    prompt filled with the current memory file and that day's log, and ``prompt``, the
    two as one text. **Run them in a fresh context, with nothing else,** and pass the
    model's whole reply to :func:`validated_save_continuity`. anneal calls no model.

    ``step["steps_remaining"]`` is how many steps follow this one. A catch-up of N days
    is N prepares and N saves in order, never one merged log.

    The step freezes its episode ids and a token in the store. Episodes recorded after
    this call belong to a later step. One wrap is open at a time: a second prepare while
    one is open raises :class:`WrapInProgressError`, and the open one is finished by
    saving it or abandoned with ``Store.wrap_cancelled``.

    Args:
        max_chars: The hard limit on the memory file the prompt states and ``save``
            enforces. ``None`` is :data:`anneal_memory.rolling.DEFAULT_MAX_CHARS`.
        tz: The zone days are cut in: an IANA name or a ``tzinfo``; ``None`` is the
            machine's own.
        session_id: Opt in to the consolidate baton gate. When passed, the step opens
            only if this session holds the baton (:func:`anneal_memory.sessions.claim_baton`);
            otherwise the result is ``"downgraded"`` (capture-only) and the store is
            untouched. ``None`` disables the gate, except on a store with the
            require-baton policy, where a call with no ``session_id`` downgrades. A
            gated caller must also round-trip ``wrap_token`` to the save.
        allow_sole_live: Only with ``session_id``; ignored on a require-baton store.
            ``True`` also authorizes a session when no other registered session is live.
        wrap_token: A caller-supplied token for the wrap (32 lowercase hex characters, the
            form ``uuid.uuid4().hex`` makes), or ``None`` to have one minted. A wrap opened
            this way is token-bound: a cancel that names no token needs ``force=True``.
        idempotency_key: A request key. A retry with the same key and request returns the
            same token and rebuilds the same prompt without opening another wrap; the same
            key with another request raises :class:`IdempotencyConflictError`.
        lease_token: Prepare under the consolidation lease this fencing token names
            (``Store.lease_acquire(...).token``); the save must present it too.

    Returns:
        A :class:`PrepareWrapResult`. ``status`` is ``"ready"`` (step open, ``step`` set),
        ``"empty"`` (no episodes since the last wrap) or ``"downgraded"`` (the baton gate
        declined, or a race made this call back off: ``message`` says which; retry).

    Raises:
        WrapInProgressError: A wrap is already open and there are episodes to wrap.
    """
    if not isinstance(allow_sole_live, bool):  # "false" is truthy: never infer consent
        raise TypeError(
            f"prepare_wrap: allow_sole_live must be a bool, got {type(allow_sole_live).__name__}"
        )
    if wrap_token is not None:
        if not isinstance(wrap_token, str):
            raise TypeError(
                f"prepare_wrap: wrap_token must be a str, got {type(wrap_token).__name__}"
            )
        if not _CALLER_TOKEN_RE.fullmatch(wrap_token):
            raise ValueError(
                "prepare_wrap: wrap_token must be 32 lowercase hex characters, the "
                "form uuid.uuid4().hex produces."
            )
    if lease_token is not None and (not isinstance(lease_token, str) or not lease_token):
        raise ValueError("prepare_wrap: lease_token must be a non-empty str")
    limit = rolling.validate_max_chars(max_chars)
    zone = rolling.resolve_tz(tz)
    idem_hash = ""
    if idempotency_key is not None:
        _validate_idempotency_key(idempotency_key)
        idem_hash = _idempotency_request_hash("wrap_started", _IDEMPOTENCY_SCOPE, store.store_id, {
            # the RESOLVED zone, so "utc" and "UTC" are one request
            "max_chars": limit,
            "tz": None if zone is None else (getattr(zone, "key", None) or str(zone)),
            "session_id": session_id, "allow_sole_live": allow_sole_live,
            "wrap_token": wrap_token, "lease_token": lease_token,
            "project_name": store.project_name,
        })
        replayed, stored = store._idempotency_replay(idempotency_key, idem_hash, "wrap_started")
        if replayed:
            # A replay is subject to the refusals the first call was: a newer meta, and
            # the baton as it is now.
            store.load_meta()
            downgraded = _consolidate_gate(
                store, session_id, allow_sole_live, _pending_count(store))
            if downgraded is not None:
                return downgraded
            return _replay_prepare(store, stored, max_chars=limit, tz=zone)
    # A meta written by a newer anneal-memory refuses HERE, before any wrap is opened.
    store.load_meta()
    # Observe the wrap in progress (if any) BEFORE reading the episode window. The
    # empty-window path below cancels only the wrap observed here, by compare-and-swap on
    # its token, so a wrap another session starts after this point has a different token
    # and can never be the one destroyed.
    #
    # StoreDatabaseError (a locked DB, disk I/O on the SELECT) is a TRANSIENT failure,
    # not the partial lifecycle state the ``except StoreError`` below recovers from.
    # Treating it as corruption would let a sessionless caller cancel ANOTHER session's
    # healthy gated wrap, so it is checked first and propagated.
    window_last_wrap_id = store.last_wrap_id()
    try:
        observed = store.load_wrap_snapshot()
        observed_partial = False
    except StoreDatabaseError:
        raise
    except StoreError:
        observed, observed_partial = None, True
    observed_gated_by = store.wrap_gated_session()
    observed_bound = store.wrap_bound_token()
    episodes = store.episodes_since_wrap()

    # The gate runs FIRST, before the empty-window path, because that path's wrap_cancelled()
    # clears the in-flight wrap of whoever holds the baton: an unauthorized caller must be
    # downgraded before it can cancel anything. A downgrade leaves the store untouched.
    downgraded = _consolidate_gate(store, session_id, allow_sole_live, len(episodes))
    if downgraded is not None:
        return downgraded

    if not episodes:
        if (
            observed is not None
            and observed_bound == observed["token"]
            and wrap_token != observed["token"]
        ):
            # A wrap opened with a caller-supplied token ends only by that token
            # (or the operator's force); this call does not hold it.
            return _no_step(
                "downgraded",
                "Consolidate downgraded to capture-only (downgraded-bound-wrap-open): "
                "a wrap opened with a token its preparer holds is in progress, and "
                "this call does not hold that token, so it cannot cancel it. "
                "Abandoning it discards that caller's compression and is the "
                "operator's decision. Capture (afferent) is unaffected.",
                0,
            )
        gated_by = observed_gated_by if session_id is None and not observed_partial else None
        if gated_by is not None:
            # An ungated caller is authorized by omission, but a wrap prepared under the
            # gate is not its to cancel: the save side refuses a session-less commit of it
            # for the same reason.
            return _no_step(
                "downgraded",
                f"Consolidate downgraded to capture-only (downgraded-gated-wrap-open): "
                f"a wrap prepared under the consolidate gate by session {gated_by!r} is "
                f"in progress, and a call that names no session_id cannot cancel it. "
                f"Finish it from that session. Abandoning it discards that session's "
                f"compression and is the operator's decision (wrap-cancel refuses it "
                f"without proof of the wrap). Capture (afferent) is unaffected.",
                0,
            )
        try:
            if observed is not None:
                store.wrap_cancelled(expect_token=observed["token"], lease_token=lease_token)
            elif observed_partial:
                # Clear only the partial state observed above.
                store.wrap_cancelled(expect_partial=True, lease_token=lease_token)
        except WrapOwnershipError as exc:
            if exc.actual is not None or exc.partial_state:
                return _no_step(
                    "downgraded",
                    "Consolidate downgraded to capture-only (downgraded-wrap-replaced): "
                    "the wrap this call observed was replaced or changed while it was deciding, "
                    "so it left it alone. Retry. Capture (afferent) is unaffected.",
                    0,
                )
            # The observed wrap finished or was cancelled meanwhile: idle, nothing to clear.
        return _no_step("empty", "No episodes since last wrap. Nothing to compress.", 0)

    # Real episodes to compress AND a wrap already in progress = a clobber. The consolidate
    # is single-writer by design; refuse structurally so every adapter inherits it.
    started = store.get_wrap_started_at()
    if started:
        if idempotency_key is not None:
            # A keyed retry whose first call opened the wrap after the early check
            # above is a replay, not a refusal.
            replayed, stored = store._idempotency_replay(idempotency_key, idem_hash, "wrap_started")
            if replayed:
                return _replay_prepare(store, stored, max_chars=limit, tz=zone)
        raise WrapInProgressError(started_at=started)

    # All reads and the prompt build happen BEFORE wrap_started(): if any of them raises,
    # the store is left with no stale wrap-in-progress flag.
    memory = store.load_continuity()
    steps = rolling.plan_steps(_rows(episodes), tz=zone)
    step = steps[0]
    # Re-run the gate now that the reads are done: a baton taken during them must not
    # still start a wrap.
    downgraded = _consolidate_gate(store, session_id, allow_sole_live, len(episodes))
    if downgraded is not None:
        return downgraded
    token_bound = wrap_token is not None
    if wrap_token is None:
        wrap_token = uuid.uuid4().hex
    result = _ready(step, len(steps) - 1, len(episodes), memory, limit, wrap_token)
    extras: dict[str, Any] = {}
    if token_bound:
        extras["token_bound"] = True
    if lease_token is not None:
        extras["lease_token"] = lease_token
    if idempotency_key is not None:
        extras["idempotency"] = (
            idempotency_key, idem_hash, json.dumps({
                "wrap_token": wrap_token,
                "package_sha256": _fingerprint(step.episode_ids, memory, limit)}))
    try:
        store.wrap_started(
            token=wrap_token,
            episode_ids=list(step.episode_ids),
            gated_session_id=session_id,
            expect_last_wrap_id=window_last_wrap_id,
            max_chars=limit,
            expect_continuity_sha256=hashlib.sha256((memory or "").encode("utf-8")).hexdigest(),
            **extras,
        )
    except _IdempotentReplay as replay:
        # A concurrent call with the same key and request opened the wrap first.
        return _replay_prepare(store, replay.result_json, max_chars=limit, tz=zone)
    except WrapWindowMovedError:
        return _no_step(
            "downgraded",
            "Consolidate downgraded to capture-only (downgraded-wrap-replaced): "
            "another wrap completed while this call was preparing, so its "
            "episodes and memory file are out of date and no wrap was opened. "
            "Retry. Capture (afferent) is unaffected.",
            _pending_count(store),
        )
    except WrapContinuityMovedError:
        return _no_step(
            "downgraded",
            "Consolidate downgraded to capture-only (downgraded-continuity-changed): "
            "the memory file was edited while this call was preparing, so its prompt "
            "was built from the old text and no wrap was opened. "
            "Retry. Capture (afferent) is unaffected.",
            _pending_count(store),
        )
    return result


def felt_currency(store: Store) -> FeltCurrency:
    """Report whether the memory file is current with the captured episodes.

    The wrap lifecycle seals each consolidate: ``status().last_wrap_at`` records when.
    This surfaces that seal as a currency check, how many episodes were captured since
    the last wrap, without diffing anything by hand. Pure read; mutates nothing.
    """
    st = store.status()
    since = st.episodes_since_wrap
    return FeltCurrency(
        sealed_at=st.last_wrap_at,
        episodes_since_seal=since,
        is_current=(since == 0),
        wrap_in_progress=st.wrap_in_progress,
    )


def format_wrap_package_text(result: PrepareWrapResult) -> str:
    """The prepare result as one text for a transport with a single output slot: the step's
    facts, the token to bring back, then the prompt to run in a fresh context."""
    step = result["step"]
    if result["status"] != "ready" or step is None:
        return result["message"]
    lines = [
        result["message"],
        f"Wrap token: {result['wrap_token']}",
        f"Step: {step['day']} ({step['episodes']} episode(s)); "
        f"{step['steps_remaining']} more step(s) after this one.",
        f"Memory limit: {step['max_chars']:,} characters.",
        "",
        "The prompt is everything from the SYSTEM banner below to the end: run only that, "
        "in a FRESH context, with nothing else, and save the model's whole reply (the "
        "complete memory file) with the token above. (`--json` gives it clean as `step.prompt`.)",
        "",
        step["prompt"],
    ]
    return "\n".join(lines)


def lease_refusal_text(
    exc: LeaseFencedError, *, saving: bool, wrap_token: str | None = None, mcp: bool = False,
) -> str:
    """One line for a transport that caught a :class:`LeaseFencedError` at a save or a
    cancel: what happened, that nothing was written, and the next step. Shared by the
    CLI and the MCP handlers so they say the same thing."""
    if saving:
        if exc.reason == "version_moved":
            what = ("another identity version was written since this step was prepared, "
                    "so the memory file it answers is no longer the current one")
        else:
            what = ("a consolidation lease (a 3.x feature) holds this step and no matching "
                    f"lease token was given (reason={exc.reason})")
        tok = wrap_token or "<token>"
        redo = (
            f"call wrap_cancel with wrap_token {tok}, then prepare_wrap again" if mcp
            else f"run `anneal-memory wrap-cancel --wrap-token {tok}`, then prepare-wrap again"
        )
        return (f"Error: this reply was not saved: {what}. Nothing was written. "
                f"Cancel the step and prepare it again: {redo}.")
    return (
        "Error: the wrap was not cancelled: a consolidation lease (a 3.x feature) holds it "
        f"and a cancel must present the lease token (reason={exc.reason}). Nothing was "
        "changed. Present it with the library call Store.wrap_cancelled(lease_token=...), "
        "or wait for the lease to expire and cancel again."
    )


def _warn_after_commit(message: str) -> None:
    """Deliver a save warning that is emitted after the save has committed.

    Once the batch has committed and the wrap token is cleared, nothing may make the
    caller believe the save failed. Under an error warnings-filter ``warnings.warn``
    RAISES, and that raise would report failure over a committed save, so a retry got
    "No wrap in progress". The warning is still emitted under every other filter; only
    when delivery raises is it logged instead, and a logging handler that raises is
    swallowed too. A ``BaseException`` that is not an ``Exception`` still propagates.
    """
    try:
        warnings.warn(message, UserWarning, stacklevel=3)
    except Exception:
        try:
            _log.warning("%s", message)
        except Exception:
            pass


def _check_save_authority(
    store: Store,
    session_id: str | None,
    wrap_token: str | None,
    allow_sole_live: bool = False,
) -> None:
    """Refuse a save the consolidate gate does not authorize. On a baton-protected store
    the caller must name itself and pass the prepare token; any caller that names itself
    must still be authorized. By default that means holding the baton;
    ``allow_sole_live=True`` (ignored on a baton-protected store, as ``prepare_wrap``
    ignores it) re-runs the same decision ``prepare_wrap`` made. Raises; writes nothing."""
    if not isinstance(allow_sole_live, bool):  # "false" is truthy: never infer consent
        raise TypeError(
            f"allow_sole_live must be a bool, got {type(allow_sole_live).__name__}"
        )
    requires_baton = store.consolidate_requires_baton()
    if requires_baton and (session_id is None or wrap_token is None):
        raise SaveAuthorityError(
            "This store is baton-protected (Store.consolidate_requires_baton): a save must "
            "name the baton holder's session id (CLI: --session-id; MCP: the server's "
            "--session-id; library: session_id=) and carry the token its prepare returned "
            "(the 'Wrap token:' line; CLI: --wrap-token; MCP and library: wrap_token). "
            "Nothing was written."
        )
    if session_id is None:
        return
    if allow_sole_live and not requires_baton:
        auth = sessions.consolidate_authorized(
            store.continuity_path, session_id, allow_sole_live=True
        )
        if not auth["authorized"]:
            raise SaveAuthorityError(
                f"Session {session_id!r} is no longer authorized to commit this wrap "
                f"({auth['reason']}; baton holder = {auth['baton_holder'] or 'none'}, "
                f"{len(auth['live_session_ids'])} live session(s)). Nothing was written."
            )
        return
    try:
        holder = sessions.baton_holder(store.continuity_path)
    except (OSError, json.JSONDecodeError):
        raise SaveAuthorityError(
            f"Session {session_id!r} cannot be confirmed as the consolidate baton holder "
            f"(the baton file is unreadable), so it may not commit this wrap. Nothing was "
            f"written."
        ) from None
    if holder == session_id:
        return
    if holder is None and requires_baton:
        cause = "this store is baton-protected, which ignores allow_sole_live, and no baton is claimed"
    elif holder is None:
        cause = (
            "no baton is claimed (it was never claimed, or was released); a session that "
            "prepared as the sole live session must pass allow_sole_live=True here too"
        )
    else:
        cause = f"the baton is held by {holder!r}"
    raise SaveAuthorityError(
        f"Session {session_id!r} does not hold the consolidate baton: {cause}, so it may not "
        f"commit this wrap. Nothing was written."
    )


def _save_replay(store: Store, result_json: str | None) -> SaveContinuityReplayResult:
    """A keyed save's replay: the key row's non-content subset, projected by
    ``_SAVE_REPLAY_FIELDS`` and marked ``replayed``, never the full result. A row that
    does not parse or lacks a field raises ``IdempotencyConflictError``
    (``reason="malformed_row"``), never a ``None`` standing in for a value. The one
    field not from the row is ``pending_count``: the window's size NOW (never the
    first call's, and never an unknown value read as 0), so a caller that lost the
    first response still learns whether later days are owed."""
    try:
        stored = json.loads(result_json or "")
    except ValueError:
        stored = None
    if not isinstance(stored, dict) or any(name not in stored for name in _SAVE_REPLAY_FIELDS):
        raise IdempotencyConflictError(
            "validated_save_continuity: this idempotency key's stored result is "
            "malformed, so the first call's result cannot be returned. Nothing "
            "was written.", operation="save_continuity", path=str(store.path),
            reason="malformed_row")
    # The save committed on the first call: a failed read here must not turn its replay
    # into an error (see _pending_count: on a failed read the value is the floor 1).
    return cast(SaveContinuityReplayResult, {
        **{name: stored.get(name) for name in _SAVE_REPLAY_FIELDS},
        "pending_count": _pending_count(store), "replayed": True})


def validated_save_continuity(
    store: Store,
    text: str,
    *,
    wrap_token: str | None = None,
    max_chars: int | None = None,
    session_id: str | None = None,
    allow_sole_live: bool = False,
    idempotency_key: str | None = None,
    lease_token: str | None = None,
    unattended: bool = False,
    model: str | None = None,
) -> SaveContinuityResult | SaveContinuityReplayResult:
    """Keep the model's reply to the step :func:`prepare_wrap` opened.

    ``text`` is the model's whole memory file. The checks are these and no others: a wrap
    is open and is the one ``wrap_token`` names, the caller may commit it, and the text is
    a string, not empty once surrounding whitespace is gone, and at most the step's
    ``max_chars`` characters. The file is stored exactly as checked (surrounding
    whitespace trimmed, nothing added). A refused save writes nothing and leaves the step open: run the step again and save
    again, or abandon it with ``Store.wrap_cancelled``.

    On success the text is the store's current memory file, the step's episodes are
    marked wrapped, and the text is appended to the version history, which keeps every
    earlier file: ``Store.get_wrap_history`` lists the wraps, the ``history`` command
    lists them with their sizes, and ``diff`` prints the text each wrap changed. If the result's ``pending_count`` is not
    zero, the next day's step is owed: prepare again.

    Args:
        wrap_token: The token ``prepare_wrap`` returned. Required: without it a save cannot
            tell which step the reply answers, so a stale reply could be kept for a later
            step. The keyword default is ``None`` only so the refusal can say this.
        max_chars: Leave it ``None``: the save uses the limit the step was prepared with
            (kept with the wrap). A value is accepted only when it equals that limit.
        session_id, allow_sole_live: The baton gate, as in ``prepare_wrap``; the baton is
            re-checked at the save, and a wrap prepared under it is saved only by it.
        idempotency_key: A retry of a committed save with the same key and text gets its
            first result back, marked ``replayed``, and writes nothing.
        lease_token: The consolidation lease's fencing token, when the wrap was prepared
            under one.
        unattended, model: Recorded on the wraps row: a wrap no operator watched, and the
            model that wrote the file.

    Raises:
        ValueError: No wrap open, no token or one that is not the open wrap's, a
            ``max_chars`` that is not the step's, or a text that is empty or over the limit.
        SaveAuthorityError: The caller may not commit this wrap.
    """
    if lease_token is not None and (not isinstance(lease_token, str) or not lease_token):
        raise ValueError("validated_save_continuity: lease_token must be a non-empty str")
    explicit_limit = None if max_chars is None else rolling.validate_max_chars(max_chars)
    idem_hash = ""
    if idempotency_key is not None:
        _validate_idempotency_key(idempotency_key)
        idem_hash = _idempotency_request_hash("save_continuity", _IDEMPOTENCY_SCOPE, store.store_id, {
            "text": text, "wrap_token": wrap_token, "max_chars": explicit_limit,
            "session_id": session_id, "allow_sole_live": allow_sole_live,
            "unattended": unattended, "model": model, "lease_token": lease_token,
        })
        replayed, stored = store._idempotency_replay(idempotency_key, idem_hash, "save_continuity")
        if replayed:
            return _save_replay(store, stored)
    # State (precondition) before payload: a save with no wrap to commit to is doomed
    # whatever the text says, and the agent should hear that first.
    snapshot = store.load_wrap_snapshot()
    if snapshot is None:
        raise ValueError(
            "No wrap in progress, nothing to save. If you already saved this step, you are "
            "done with it: a second save_continuity with no new prepare_wrap is refused by "
            "design. If you have not prepared yet, call prepare_wrap first (and if it "
            "reports no episodes, there is nothing to compress: skip the save)."
        )
    if session_id is not None and not session_id:
        raise ValueError(
            "validated_save_continuity: session_id must be non-empty when provided."
        )
    # An early pass, so a refused save fails before any work. It is repeated
    # authoritatively inside the batch, after wrap_completed.
    _check_save_authority(store, session_id, wrap_token, allow_sole_live)
    # A wrap prepared under the gate names the session that prepared it, and only that
    # session may commit it. A save that omits session_id would skip every baton check (a
    # token identifies the wrap, not the actor).
    gated_by = store.wrap_gated_session()
    if gated_by is not None and wrap_token is None:
        raise SaveAuthorityError(
            f"This wrap was prepared under the consolidate gate by session {gated_by!r}, so "
            f"its save must pass the wrap_token that prepare_wrap returned (and that session_id). "
            f"Nothing was written."
        )
    if gated_by is not None and session_id != gated_by:
        if session_id is None:
            raise SaveAuthorityError(
                f"This wrap was prepared under the consolidate gate by session {gated_by!r}, "
                f"so the save must come from that session, naming its session_id: without one the "
                f"baton is never re-checked. Finish it under that session id (CLI: "
                f"save-continuity --session-id; MCP: the server started with that --session-id; "
                f"library: session_id=), or abandon it with wrap-cancel --wrap-token <this "
                f"wrap's token> (MCP: wrap_cancel with wrap_token), which discards the "
                f"compression. Nothing was written."
            )
        raise SaveAuthorityError(
            f"This wrap was prepared by session {gated_by!r}, and only that session may "
            f"commit it; {session_id!r} cannot, even as the current baton holder. Abandon it "
            f"(wrap-cancel --wrap-token <this wrap's token>, or MCP wrap_cancel with wrap_token, "
            f"which discards the compression) and prepare_wrap again from {session_id!r}. "
            f"Nothing was written."
        )
    if wrap_token is None:
        raise ValueError(
            "validated_save_continuity: pass the wrap_token that prepare_wrap returned: "
            "without it a save cannot tell which step the reply answers; nothing was saved."
        )
    if snapshot["token"] != wrap_token:
        raise ValueError(
            f"wrap_token mismatch: caller passed '{wrap_token[:8]}…' but the in-progress wrap "
            f"has token '{snapshot['token'][:8]}…'. The token is stale (the wrap was already "
            f"completed or cancelled), from a different prepare_wrap call, or from a "
            f"concurrent process. Re-run prepare_wrap to start a new wrap with a fresh token."
        )
    # The limit is the step's: the one its prompt stated, kept with the wrap.
    step_limit = store.wrap_max_chars() or rolling.DEFAULT_MAX_CHARS
    if explicit_limit is not None and explicit_limit != step_limit:
        raise ValueError(
            f"validated_save_continuity: max_chars is {explicit_limit:,} but this step was "
            f"prepared with {step_limit:,}. Pass the same value, or none to use the step's. "
            f"Nothing was saved."
        )
    # The whole payload check.
    try:
        body = rolling.check_reply(text, step_limit)
    except ValueError as exc:
        raise ValueError(f"validated_save_continuity: {exc}") from None
    saved_text = body

    frozen_ids = list(snapshot["episode_ids"])
    path = str(store.continuity_path)
    content_hash = hashlib.sha256(saved_text.encode("utf-8")).hexdigest()
    # The wraps row's pair_id (``<token12>-<uuid8>``) is recorded per attempt.
    pair_id = f"{snapshot['token'][:12]}-{uuid.uuid4().hex[:8]}"

    with store._batch():
        # First under the batch's write lock: a concurrent save with the same idempotency
        # key (its result is returned; nothing is written yet), then the lease fence.
        if idempotency_key is not None:
            idem_status, idem_stored = store._idempotency_lookup(idempotency_key, idem_hash)
            if idem_status == "match":
                return _save_replay(store, idem_stored)
            if idem_status == "conflict":
                store._idempotency_conflict(idempotency_key, "save_continuity")
        lease_prepared = store._wrap_lease_prepared()
        store._lease_gate("validated_save_continuity", lease_token, lease_prepared)

        meta = store.load_meta()
        meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
        wrap_result = store.wrap_completed(
            episodes_compressed=len(frozen_ids),
            continuity_chars=len(saved_text),
            episode_ids=frozen_ids,
            wrap_token=snapshot["token"],
            unattended=unattended,
            model=model,
            content_hash=content_hash,
            pair_id=pair_id,
        )
        # The memory file itself, in this transaction with the wraps row above.
        store._insert_identity_version(
            saved_text, meta_json=store._meta_json(meta),
            source="wrap", wrap_id=wrap_result.wrap_id,
            lease_token=lease_token, lease_prepared=lease_prepared,
        )
        if idempotency_key is not None:
            # The key row's result, in this transaction: the non-content subset only,
            # which a replay returns marked ``replayed``.
            store._idempotency_insert(
                idempotency_key, "save_continuity", idem_hash, json.dumps({
                    "wrap_id": wrap_result.wrap_id, "chars": len(saved_text),
                    "path": path, "content_hash": content_hash,
                    "episodes_compressed": wrap_result.episodes_compressed,
                }))
        # The authoritative baton check, deliberately the LAST statement before the commit:
        # wrap_completed's DML above means this connection holds SQLite's write lock, so a
        # policy change from any other connection has either committed already (and this
        # re-read sees it) or waits for this transaction to end. Raising rolls the whole
        # batch back.
        _check_save_authority(store, session_id, wrap_token, allow_sole_live)
        # The continuity_saved event is queued inside the batch, so it is an
        # ``audit_outbox`` row in this transaction: a kill after the commit cannot lose the
        # record of the committed memory file.
        if store._audit is not None:
            store._audit_log_after_commit(
                "continuity_saved",
                {"chars": len(saved_text), "content_hash": content_hash},
                method="validated_save_continuity",
                committed="the wrap",
            )
        # The batch context manager commits here on a successful exit.

    # The batch committed; project the memory file and its meta sidecar from the committed
    # row. A failure warns and the save still succeeds: the DB holds the file and the next
    # open or load_continuity re-projects.
    store._project_after_commit("the wrap")

    # Auto-prune when retention is configured. The batched save suppresses prune inside
    # the batch, so it runs here, after the commit. The wrap has committed, so a prune
    # failure must not make the caller believe the save failed.
    if store._retention_days is not None:
        try:
            pruned = store.prune()
        except Exception as exc:
            detail = type(exc).__name__
            try:
                detail = f"{detail}: {exc}"
            except Exception:
                pass
            store._record_prune_failure(f"validated_save_continuity: {detail}")
            _warn_after_commit(
                f"Auto-prune failed after the wrap committed; the save succeeded, "
                f"pruned_count is reported as 0 and may undercount ({detail}). "
                f"Retention runs again on the next prune."
            )
        else:
            wrap_result.pruned_count = pruned

    return SaveContinuityResult(
        path=path,
        chars=len(saved_text),
        episodes_compressed=wrap_result.episodes_compressed,
        pending_count=_pending_count(store),
        content_hash=content_hash,
        wrap_result=_wrap_result_dict(wrap_result),
    )


def _wrap_result_dict(wrap_result: Any) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(wrap_result)
