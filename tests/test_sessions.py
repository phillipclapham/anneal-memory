"""Tests for AM-CONSOLIDATE-EFFERENT (spore-194): the consolidate-efferent gate.

Two layers: (1) the sidecar session registry + the consolidate baton
(``anneal_memory.sessions``); (2) the ``prepare_wrap`` efferent gate that downgrades an
unauthorized parallel consolidate to capture-only instead of recomposing the shared felt
layer. The invariant under test: a consolidate proceeds IFF (sole live session) OR (holds the
baton); else it auto-downgrades + leaves the store untouched.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from anneal_memory import (
    AnnealMemoryError,
    Store,
    felt_currency,
    prepare_wrap,
    sessions,
    validated_save_continuity,
)
from anneal_memory.types import EpisodeType

# A minimal valid default-schema continuity for driving a real consolidate to completion.
_WRAP_TEXT = (
    "# T — Memory (v1)\n\n"
    "## State\nActive.\n\n"
    "## Patterns\nNone yet.\n\n"
    "## Decisions\nNone.\n\n"
    "## Context\nFirst session.\n"
)


@pytest.fixture
def cp(tmp_path):
    # A continuity-path anchor; the leaf need not exist (resolve tolerates a missing file).
    return tmp_path / "memory.continuity.md"


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "wrap.db"), project_name="GateTest")


# -- the session registry --


def test_register_and_live(cp):
    sessions.register_session(cp, "s1", label="conv-1")
    live = sessions.live_sessions(cp)
    assert [s["session_id"] for s in live] == ["s1"]
    assert live[0]["label"] == "conv-1"
    assert live[0]["pid"] == os.getpid()


def test_no_registry_dir_is_empty(cp):
    assert sessions.live_sessions(cp) == []


def test_heartbeat_autoregisters_and_keeps_alive(cp):
    sessions.heartbeat(cp, "s1")  # never registered → auto-registers
    assert [s["session_id"] for s in sessions.live_sessions(cp)] == ["s1"]
    sessions.heartbeat(cp, "s1")  # touch existing
    assert len(sessions.live_sessions(cp)) == 1


def test_ttl_expiry(cp):
    sessions.register_session(cp, "s1")
    f = sessions._session_file(cp, "s1")
    past = time.time() - 10_000
    os.utime(f, (past, past))
    assert sessions.live_sessions(cp) == []  # default ttl 5400 < 10_000 elapsed → reaped
    assert [s["session_id"] for s in sessions.live_sessions(cp, ttl=20_000)] == ["s1"]


def test_close_session(cp):
    sessions.register_session(cp, "s1")
    sessions.close_session(cp, "s1")
    assert sessions.live_sessions(cp) == []
    sessions.close_session(cp, "nonexistent")  # best-effort: no raise


def test_corrupt_fresh_session_file_fails_closed(cp):
    # A corrupt FRESH peer file is authorization-relevant UNKNOWN → the gate must fail CLOSED
    # (not silently treat the peer as absent → a false sole-grant). [codex L3 MED-1]
    sessions.register_session(cp, "s1")
    sessions._session_file(cp, "s1").write_text("{ not json", encoding="utf-8")
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-registry-error"
    # by default liveness decides nothing: still no grant, for want of the baton
    assert sessions.consolidate_authorized(cp, "me")["authorized"] is False


def test_corrupt_stale_session_file_skipped(cp):
    # A DEAD session's corruption is irrelevant (TTL skips it before the read) → gate proceeds.
    sessions.register_session(cp, "s1")
    f = sessions._session_file(cp, "s1")
    f.write_text("{ not json", encoding="utf-8")
    os.utime(f, (time.time() - 10_000, time.time() - 10_000))
    assert sessions.live_sessions(cp) == []
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is True  # me is sole; the dead corrupt peer is ignored
    assert auth["reason"] == "sole-live-session"
    # and by default it is the missing baton that downgrades, not a registry error
    assert sessions.consolidate_authorized(cp, "me")["reason"] == "downgraded-no-baton"


def test_live_sessions_coerces_untrusted_field_types(cp):
    sessions.register_session(cp, "s1")
    f = sessions._session_file(cp, "s1")
    f.write_text(
        json.dumps(
            {"session_id": "s1", "label": 123, "pid": "oops", "registered_at": "nope"}
        ),
        encoding="utf-8",
    )
    live = sessions.live_sessions(cp)
    assert len(live) == 1
    assert live[0]["label"] is None  # int → None
    assert live[0]["pid"] is None  # str → None
    assert live[0]["registered_at"] == 0.0  # non-numeric → 0.0


def test_register_requires_session_id(cp):
    with pytest.raises(ValueError):
        sessions.register_session(cp, "")


# -- the consolidate baton --


def test_baton_claim_holds_release(cp):
    assert sessions.baton_holder(cp) is None
    claim = sessions.claim_baton(cp, "s1")
    assert claim["previous_holder"] is None
    assert claim["session_id"] == "s1"
    assert sessions.holds_baton(cp, "s1")
    assert not sessions.holds_baton(cp, "s2")
    assert sessions.baton_holder(cp) == "s1"
    assert sessions.release_baton(cp, "s1") is True
    assert sessions.baton_holder(cp) is None


def test_baton_take_records_previous_holder(cp):
    sessions.claim_baton(cp, "s1")
    claim = sessions.claim_baton(cp, "s2", take=True)
    assert claim["previous_holder"] == "s1"
    assert sessions.holds_baton(cp, "s2")


def test_claim_over_another_holder_needs_take(cp):
    # ⚖ Phill 2026-09-24 (flow spore-1169): taking another session's baton is deliberate.
    first = sessions.claim_baton(cp, "s1")
    with pytest.raises(sessions.BatonHeldError) as ei:
        sessions.claim_baton(cp, "s2")
    assert ei.value.holder == "s1" and ei.value.unreadable is False
    assert isinstance(ei.value, AnnealMemoryError)
    assert sessions.holds_baton(cp, "s1")  # the refusal changed nothing
    assert json.loads(sessions._baton_path(cp).read_text())["claimed_at"] == first["claimed_at"]


def test_holder_reclaim_is_a_noop_success(cp):
    first = sessions.claim_baton(cp, "s1")
    before = sessions._baton_path(cp).read_bytes()
    again = sessions.claim_baton(cp, "s1")
    assert again["previous_holder"] == "s1"
    assert again["claimed_at"] == first["claimed_at"]
    assert sessions._baton_path(cp).read_bytes() == before  # nothing rewritten


def _no_lock(monkeypatch):
    # Simulate a platform where flock is unavailable (Windows, some NFS): _baton_lock yields
    # False, so claim_baton falls back to an exclusive create for an unheld baton.
    import contextlib

    @contextlib.contextmanager
    def unlocked(_cp):
        yield False

    monkeypatch.setattr(sessions, "_baton_lock", unlocked)


def test_unheld_claim_is_create_only_without_the_lock(cp, monkeypatch):
    # Two sessions racing for an UNHELD baton: the loser must be refused, not overwrite.
    # Simulate the race by landing s1's claim between s2's read and s2's create.
    _no_lock(monkeypatch)
    real_read = sessions._read_baton

    def read_then_race(path):
        data = real_read(path)
        if data is None and not sessions._baton_path(cp).exists():
            monkeypatch.setattr(sessions, "_read_baton", real_read)
            sessions.claim_baton(cp, "s1")
        return data

    monkeypatch.setattr(sessions, "_read_baton", read_then_race)
    with pytest.raises(sessions.BatonHeldError) as ei:
        sessions.claim_baton(cp, "s2")
    assert ei.value.holder == "s1"
    assert sessions.holds_baton(cp, "s1")
    leftovers = [q.name for q in cp.parent.iterdir() if q.name.startswith(".")]
    assert leftovers == []  # the loser's tmp file was cleaned up


def test_release_only_by_holder(cp):
    sessions.claim_baton(cp, "s1")
    assert sessions.release_baton(cp, "s2") is False  # a non-holder can't drop it
    assert sessions.holds_baton(cp, "s1")


def test_close_session_releases_held_baton(cp):
    sessions.register_session(cp, "s1")
    sessions.claim_baton(cp, "s1")
    sessions.close_session(cp, "s1")
    assert sessions.baton_holder(cp) is None  # a held baton must not outlive its session


def test_corrupt_baton_fails_closed(cp):
    # A corrupt/unreadable baton is UNKNOWN authorization state → baton_holder RAISES and the
    # gate fails CLOSED (not silently "unheld" → a false sole-grant). [codex L3 MED-1]
    bp = sessions._baton_path(cp)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text("{ nope", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        sessions.baton_holder(cp)
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-registry-error"
    # a claim over it is refused without take (whether someone holds it is unknown) ...
    with pytest.raises(sessions.BatonHeldError) as ei:
        sessions.claim_baton(cp, "newhead")
    assert ei.value.unreadable is True and ei.value.holder is None
    # ... and replaces it atomically with take
    claim = sessions.claim_baton(cp, "newhead", take=True)
    assert claim["previous_holder"] is None
    assert sessions.holds_baton(cp, "newhead")


@pytest.mark.parametrize(
    "payload",
    [b"[]", b"null", b'"a string"', b"42", b"{}", b'{"session_id": ""}',
     b'{"session_id": 7}', b"\xff\xfe not utf-8"],
    ids=["list", "null", "string", "number", "empty-object", "empty-id", "int-id", "not-utf8"],
)
def test_wrong_shape_baton_is_unreadable_not_a_crash(cp, payload):
    # flow spore-1169 item 2: valid JSON of the wrong shape raised AttributeError, which no
    # caller caught, so claim_baton --take recovery was wedged and the gate crashed.
    bp = sessions._baton_path(cp)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_bytes(payload)
    with pytest.raises(json.JSONDecodeError):  # CorruptSidecarError is one
        sessions.baton_holder(cp)
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert (auth["authorized"], auth["reason"]) == (False, "downgraded-registry-error")
    assert sessions.release_baton(cp, "me") is False  # unowned: left for a deliberate take
    assert bp.read_bytes() == payload
    sessions.close_session(cp, "me")  # the release inside must not raise either
    with pytest.raises(sessions.BatonHeldError):
        sessions.claim_baton(cp, "me")
    claim = sessions.claim_baton(cp, "me", take=True)
    assert claim["previous_holder"] is None and sessions.holds_baton(cp, "me")


@pytest.mark.parametrize("payload", [b"[]", b"null", b'"s"', b'{"label": "x"}', b"\xff"])
def test_wrong_shape_fresh_session_file_fails_closed(cp, payload):
    # The same class in the registry: a fresh peer file of the wrong shape is an unknown peer.
    sessions.register_session(cp, "s1")
    sessions._session_file(cp, "s1").write_bytes(payload)
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert (auth["authorized"], auth["reason"]) == (False, "downgraded-registry-error")


# -- the efferent decision (consolidate_authorized) --


def test_sole_session_without_baton_downgrades_by_default(cp):
    # ⚖ Phill 2026-09-24 (flow spore-1105 (b), spore-1169): every consolidate needs the baton.
    sessions.register_session(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me")
    assert (auth["authorized"], auth["reason"]) == (False, "downgraded-no-baton")
    assert auth["live_session_ids"] == ["me"] and auth["baton_holder"] is None


def test_authorized_when_sole_only_if_opted_in(cp):
    sessions.register_session(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is True
    assert auth["reason"] == "sole-live-session"


def test_authorized_when_no_sessions_registered_only_if_opted_in(cp):
    assert sessions.consolidate_authorized(cp, "me")["reason"] == "downgraded-no-baton"
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)  # me not registered
    assert auth["authorized"] is True  # no OTHER live session
    assert auth["reason"] == "sole-live-session"


def test_downgrade_when_another_session_live(cp):
    sessions.register_session(cp, "other")
    assert sessions.consolidate_authorized(cp, "me")["reason"] == "downgraded-no-baton"
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-not-baton-holder"
    assert "other" in auth["live_session_ids"]
    assert auth["baton_holder"] is None


def test_baton_overrides_other_live_sessions(cp):
    sessions.register_session(cp, "other")
    sessions.claim_baton(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is True
    assert auth["reason"] == "holds-baton"
    assert auth["baton_holder"] == "me"


def test_consolidate_authorized_requires_session_id(cp):
    with pytest.raises(ValueError):
        sessions.consolidate_authorized(cp, "")


# -- prepare_wrap integration (the gate live in the pipeline) --


def test_gate_inert_without_session_id(store):
    store.record("obs", EpisodeType.OBSERVATION)
    result = prepare_wrap(store)  # no session_id → gate never engages (backward-compat)
    assert result["status"] == "ready"


def test_gate_sole_session_without_baton_downgrades(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    result = prepare_wrap(store, session_id="me")
    assert result["status"] == "downgraded"
    assert "downgraded-no-baton" in result["message"]
    assert not store.status().wrap_in_progress


def test_gate_sole_session_ready_when_opted_in(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    result = prepare_wrap(store, session_id="me", allow_sole_live=True)
    assert result["status"] == "ready"
    assert store.status().wrap_in_progress


def test_gate_downgrades_unbatoned_parallel(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "other")
    result = prepare_wrap(store, session_id="me")
    assert result["status"] == "downgraded"
    assert result["package"] is None
    assert result["wrap_token"] is None
    assert "downgraded" in result["message"].lower()
    # the store is left UNTOUCHED — no wrap marked in progress
    assert not store.status().wrap_in_progress


def test_gate_baton_holder_proceeds(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "other")
    sessions.claim_baton(store.continuity_path, "me")
    result = prepare_wrap(store, session_id="me")
    assert result["status"] == "ready"
    assert store.status().wrap_in_progress


def test_downgrade_does_not_strand_a_later_consolidate(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "other")
    assert prepare_wrap(store, session_id="me")["status"] == "downgraded"
    # me claims the baton → a fresh prepare must succeed (nothing stranded)
    sessions.claim_baton(store.continuity_path, "me")
    assert prepare_wrap(store, session_id="me")["status"] == "ready"


def test_downgrade_does_not_clear_another_sessions_inflight_wrap(store):
    store.record("obs", EpisodeType.OBSERVATION)
    # A (holding the baton) legitimately starts a wrap
    sessions.register_session(store.continuity_path, "A")
    sessions.claim_baton(store.continuity_path, "A")
    assert prepare_wrap(store, session_id="A")["status"] == "ready"
    assert store.status().wrap_in_progress
    # B comes online (parallel); its prepare downgrades and must NOT clear A's in-flight wrap
    sessions.register_session(store.continuity_path, "B")
    assert prepare_wrap(store, session_id="B")["status"] == "downgraded"
    assert store.status().wrap_in_progress  # A's wrap survives B's downgrade


# -- felt_currency (the seal-watermark read) --


def test_felt_currency_never_consolidated(store):
    fc = felt_currency(store)
    assert fc["sealed_at"] is None
    assert fc["episodes_since_seal"] == 0
    assert fc["is_current"] is True  # nothing captured, nothing un-consolidated


def test_felt_currency_pending_episodes_before_first_consolidate(store):
    store.record("obs", EpisodeType.OBSERVATION)
    fc = felt_currency(store)
    assert fc["sealed_at"] is None
    assert fc["episodes_since_seal"] == 1
    assert fc["is_current"] is False  # an un-consolidated episode exists


def test_felt_currency_current_after_consolidate(store):
    store.record("obs", EpisodeType.OBSERVATION)
    r = prepare_wrap(store)
    validated_save_continuity(
        store, _WRAP_TEXT, wrap_token=r["wrap_token"], today="2026-06-27"
    )
    fc = felt_currency(store)
    assert fc["sealed_at"] is not None
    assert fc["episodes_since_seal"] == 0
    assert fc["is_current"] is True


def test_felt_currency_stale_after_post_consolidate_capture(store):
    # The Slice-B-after-EOD case: consolidate, then capture more work.
    store.record("obs", EpisodeType.OBSERVATION)
    r = prepare_wrap(store)
    validated_save_continuity(
        store, _WRAP_TEXT, wrap_token=r["wrap_token"], today="2026-06-27"
    )
    store.record("work captured after the consolidate", EpisodeType.OBSERVATION)
    fc = felt_currency(store)
    assert fc["sealed_at"] is not None
    assert fc["episodes_since_seal"] == 1
    assert fc["is_current"] is False


def test_felt_currency_reports_wrap_in_progress(store):
    store.record("obs", EpisodeType.OBSERVATION)
    assert felt_currency(store)["wrap_in_progress"] is False
    prepare_wrap(store)  # opens a wrap (no save) → consolidate underway
    fc = felt_currency(store)
    assert fc["wrap_in_progress"] is True
    assert fc["is_current"] is False  # still stale until the save commits


# -- the baton/TTL interaction (the L1+L2 MED fix) --


def test_live_baton_holder_blocks_other(cp):
    sessions.register_session(cp, "head")
    sessions.claim_baton(cp, "head")
    auth = sessions.consolidate_authorized(cp, "lane")
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-not-baton-holder"
    assert auth["baton_holder"] == "head"


def test_stale_baton_holder_blocks_effectively_sole_session(cp):
    # The MED: a designated baton-holder goes TTL-stale (idle); a parallel session that is now
    # effectively sole must STILL downgrade — NOT silently consolidate unbidden.
    sessions.register_session(cp, "head")
    sessions.claim_baton(cp, "head")
    f = sessions._session_file(cp, "head")
    past = time.time() - 10_000
    os.utime(f, (past, past))  # head idles past TTL → reaped from live_sessions
    assert sessions.live_sessions(cp) == []
    auth = sessions.consolidate_authorized(cp, "lane")  # lane is now effectively sole
    assert auth["authorized"] is False  # the human designation outranks the TTL inference
    assert auth["reason"] == "downgraded-stale-baton-holder"
    assert auth["baton_holder"] == "head"


def test_no_baton_sole_authorized_only_when_opted_in(cp):
    # The stale-holder fix must not block the sole-no-baton case for a caller that opted in.
    sessions.register_session(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is True
    assert auth["reason"] == "sole-live-session"


def test_stale_baton_recovered_by_reclaim(cp):
    sessions.register_session(cp, "head")
    sessions.claim_baton(cp, "head")
    f = sessions._session_file(cp, "head")
    past = time.time() - 10_000
    os.utime(f, (past, past))  # head dead/idle
    sessions.register_session(cp, "lane")
    with pytest.raises(sessions.BatonHeldError):
        sessions.claim_baton(cp, "lane")  # a stale holder is still a holder
    sessions.claim_baton(cp, "lane", take=True)  # the human re-designates → recovery
    auth = sessions.consolidate_authorized(cp, "lane")
    assert auth["authorized"] is True
    assert auth["reason"] == "holds-baton"


def test_registry_error_fails_closed(cp, monkeypatch):
    sessions.register_session(cp, "me")

    def boom(*a, **k):
        raise PermissionError("registry unreadable")

    monkeypatch.setattr(sessions, "live_sessions", boom)
    auth = sessions.consolidate_authorized(cp, "me", allow_sole_live=True)
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-registry-error"
    assert sessions.consolidate_authorized(cp, "me")["authorized"] is False


def test_heartbeat_rescues_stale_session(cp):
    sessions.register_session(cp, "s1")
    f = sessions._session_file(cp, "s1")
    past = time.time() - 10_000
    os.utime(f, (past, past))
    assert sessions.live_sessions(cp) == []  # reaped
    sessions.heartbeat(cp, "s1")  # touches mtime back to now (file still present)
    assert [s["session_id"] for s in sessions.live_sessions(cp)] == ["s1"]  # rescued


def test_two_unregistered_callers_both_authorized_cooperative_gap(cp):
    # The cooperative-protocol gap (BY DESIGN, complement L3 INFO-2): two callers that never
    # register are invisible to each other → both authorized. The save-side wrap_token CAS is
    # the hard backstop on the COMMIT; the gate is a throttle, not the corruption guard.
    # Integration (flow) closes this by registering BEFORE prepare on every consolidate-capable
    # conversation.
    # Under allow_sole_live only: by default neither is authorized without the baton.
    assert sessions.consolidate_authorized(cp, "A", allow_sole_live=True)["authorized"] is True
    assert sessions.consolidate_authorized(cp, "B", allow_sole_live=True)["authorized"] is True
    assert sessions.consolidate_authorized(cp, "A")["authorized"] is False


def test_prepare_wrap_empty_session_id_raises_clear_error(store):
    # An empty session_id is a programmer error, surfaced clearly at the public API boundary
    # rather than deep inside the gate (complement L3 LOW-2).
    store.record("obs", EpisodeType.OBSERVATION)
    with pytest.raises(ValueError, match="non-empty"):
        prepare_wrap(store, session_id="")


# -- the store-level require-baton policy (flow spore-1169) --


def test_policy_is_off_by_default_and_audited_when_set(tmp_path):
    db = tmp_path / "p.db"
    store = Store(str(db))
    assert store.consolidate_requires_baton() is False
    store.set_consolidate_requires_baton(True)
    assert store.consolidate_requires_baton() is True
    store.set_consolidate_requires_baton(False)
    assert store.consolidate_requires_baton() is False
    store.close()
    entries = [json.loads(line) for line in
               (tmp_path / "p.audit.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [e["data"] for e in entries if e["event"] == "consolidate_policy_set"]
    assert events == [{"requires_baton": True, "was": False},
                      {"requires_baton": False, "was": True}]
    reopened = Store(str(db))
    reopened.set_consolidate_requires_baton(True)
    reopened.close()
    assert Store(str(db)).consolidate_requires_baton() is True  # persisted in the store


def test_policy_corrupt_value_reads_as_required(store):
    store._conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("consolidate_requires_baton", "yes"),
    )
    store._conn.commit()
    assert store.consolidate_requires_baton() is True


def test_policy_setter_refuses_non_bool_and_batch(store):
    with pytest.raises(TypeError):
        store.set_consolidate_requires_baton(1)  # type: ignore[arg-type]
    from anneal_memory import StoreError

    with store._batch():
        with pytest.raises(StoreError):
            store.set_consolidate_requires_baton(True)
    assert store.consolidate_requires_baton() is False


def test_policy_downgrades_a_caller_with_no_session_id(store):
    store.record("obs", EpisodeType.OBSERVATION)
    assert prepare_wrap(store)["status"] == "ready"  # policy off: the gate is inert
    store.wrap_cancelled()
    store.set_consolidate_requires_baton(True)
    result = prepare_wrap(store)
    assert result["status"] == "downgraded"
    assert "downgraded-baton-required" in result["message"]
    assert result["wrap_token"] is None and result["package"] is None
    assert not store.status().wrap_in_progress  # store untouched


def test_policy_overrides_allow_sole_live(store):
    store.record("obs", EpisodeType.OBSERVATION)
    store.set_consolidate_requires_baton(True)
    sessions.register_session(store.continuity_path, "me")
    result = prepare_wrap(store, session_id="me", allow_sole_live=True)
    assert result["status"] == "downgraded"
    assert "downgraded-no-baton" in result["message"]


def test_policy_save_needs_the_prepare_token(store):
    store.record("obs", EpisodeType.OBSERVATION)
    store.set_consolidate_requires_baton(True)
    sessions.claim_baton(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me")
    assert prep["status"] == "ready"
    with pytest.raises(ValueError, match="baton-protected"):
        validated_save_continuity(store, _WRAP_TEXT)
    with pytest.raises(ValueError, match="baton-protected"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"])
    assert store.status().wrap_in_progress  # the refusal wrote nothing
    assert store.get_wrap_history() == []
    validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me")
    assert not store.status().wrap_in_progress
    assert len(store.get_wrap_history()) == 1  # the token-carrying save committed


def test_policy_cli_prepare_wrap_json_reports_the_downgrade(tmp_path):
    import subprocess
    import sys

    db = str(tmp_path / "cli.db")
    s = Store(db)
    s.record("obs", EpisodeType.OBSERVATION)
    s.set_consolidate_requires_baton(True)
    s.close()
    run = subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", db, "prepare-wrap", "--json"],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["status"] == "downgraded" and payload["wrap_token"] is None
    assert "downgraded-baton-required" in payload["message"]


# -- L1/L2 review fixes (2026-09-24) --


@pytest.mark.parametrize(
    "payload",
    [
        '{"session_id": "s1", "claimed_at": ' + "9" * 5000 + "}",  # int digit limit → ValueError
        "[" * 200000 + "]" * 200000,  # nesting → RecursionError
    ],
    ids=["int-digit-limit", "deep-nesting"],
)
def test_parser_failures_are_unreadable_not_a_crash(cp, payload):
    # L1 MED-1: json.loads raises ValueError / RecursionError for these, which the
    # (OSError, JSONDecodeError) catches missed, wedging take=True and crashing the gate.
    bp = sessions._baton_path(cp)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(payload, encoding="utf-8")
    with pytest.raises(sessions.CorruptSidecarError):
        sessions.baton_holder(cp)
    assert sessions.consolidate_authorized(cp, "me")["reason"] == "downgraded-registry-error"
    assert sessions.release_baton(cp, "me") is False
    sessions.close_session(cp, "me")
    claim = sessions.claim_baton(cp, "me", take=True)  # the recovery is not wedged
    assert claim["previous_holder"] is None and sessions.holds_baton(cp, "me")
    # and the same payload in a fresh peer file fails closed rather than raising
    sessions.register_session(cp, "peer")
    sessions._session_file(cp, "peer").write_text(payload, encoding="utf-8")
    auth = sessions.consolidate_authorized(cp, "x", allow_sole_live=True)
    assert auth["reason"] == "downgraded-registry-error"


def test_oversized_numbers_coerce_instead_of_raising(cp):
    # L1 MED-2: float() of a 400-digit int raises OverflowError.
    big = int("9" * 400)
    bp = sessions._baton_path(cp)
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps({"session_id": "s1", "claimed_at": big}), encoding="utf-8")
    assert sessions.claim_baton(cp, "s1")["claimed_at"] == 0.0  # holder no-op still works
    sessions.register_session(cp, "peer")
    sessions._session_file(cp, "peer").write_text(
        json.dumps({"session_id": "peer", "registered_at": big, "pid": True}), encoding="utf-8"
    )
    [info] = sessions.live_sessions(cp)
    assert info["registered_at"] == 0.0 and info["pid"] is None


def test_a_bad_peer_file_does_not_block_the_holder(cp):
    # L1 LOW-MED-5: by default liveness decides nothing, so a corrupt peer is not a reason
    # to refuse the baton holder.
    sessions.claim_baton(cp, "me")
    sessions.register_session(cp, "peer")
    sessions._session_file(cp, "peer").write_text('{"label": "x"}', encoding="utf-8")
    auth = sessions.consolidate_authorized(cp, "me")
    assert (auth["authorized"], auth["reason"]) == (True, "holds-baton")
    assert auth["live_session_ids"] == []  # unknown, and said so by being empty
    # ... but under allow_sole_live, where liveness can authorize, it still fails closed
    other = sessions.consolidate_authorized(cp, "x", allow_sole_live=True)
    assert other["reason"] == "downgraded-registry-error"


def test_release_holds_the_baton_lock_across_check_and_unlink(cp, monkeypatch):
    # L2 M2: a take landing between release's check and its unlink was deleted, leaving the
    # baton unheld and claimable by anyone. Prove the check runs under the exclusive lock:
    # from inside it, a non-blocking flock on the lock file must fail.
    fcntl = pytest.importorskip("fcntl")
    import errno as _errno

    sessions.claim_baton(cp, "A")
    real = sessions.holds_baton
    seen = []

    def probe(path, sid):
        lock = sessions._anchor(cp).with_name(sessions._anchor(cp).name + ".baton.lock")
        fd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append("free")
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            assert exc.errno in (_errno.EWOULDBLOCK, _errno.EAGAIN)
            seen.append("held")
        finally:
            os.close(fd)
        return real(path, sid)

    monkeypatch.setattr(sessions, "holds_baton", probe)
    assert sessions.release_baton(cp, "A") is True
    assert seen == ["held"]


def test_exceptions_pickle_round_trip(tmp_path):
    import pickle

    e = sessions.CorruptSidecarError(tmp_path / "x.baton", "why")
    back = pickle.loads(pickle.dumps(e))
    assert str(back) == "x.baton: why" and back.path == e.path
    assert isinstance(back, json.JSONDecodeError) and isinstance(back, AnnealMemoryError)
    h = sessions.BatonHeldError("me", "other", False)
    hb = pickle.loads(pickle.dumps(h))
    assert (hb.session_id, hb.holder, hb.unreadable) == ("me", "other", False)
    assert str(hb) == str(h)


def test_save_rechecks_the_baton_when_the_session_names_itself(store):
    # L2 M3: a take mid-wrap revokes the old holder's commit, for a caller that names itself.
    store.record("obs", EpisodeType.OBSERVATION)
    cp = store.continuity_path
    sessions.claim_baton(cp, "A")
    prep = prepare_wrap(store, session_id="A")
    sessions.claim_baton(cp, "B", take=True)
    with pytest.raises(ValueError, match="does not hold the consolidate baton"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"],
                                  session_id="A")
    assert store.status().wrap_in_progress  # nothing written
    # Strict match (0.9.15): the new holder may not commit the revoked holder's wrap either;
    # it abandons it and prepares its own.
    with pytest.raises(ValueError, match="only that session may commit it"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="B")
    store.wrap_cancelled()
    prep_b = prepare_wrap(store, session_id="B")
    validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep_b["wrap_token"], session_id="B")
    assert len(store.get_wrap_history()) == 1


def test_policy_save_with_a_borrowed_token_is_refused(store):
    # L2 H1: the token is not a secret (wrap-token-current prints it). On a protected store a
    # save that does not name the baton holder is refused even with the right token.
    store.record("obs", EpisodeType.OBSERVATION)
    store.set_consolidate_requires_baton(True)
    sessions.claim_baton(store.continuity_path, "A")
    prepare_wrap(store, session_id="A")
    token = store.load_wrap_snapshot()["token"]  # what any other process can read
    with pytest.raises(ValueError, match="baton-protected"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=token)
    with pytest.raises(ValueError, match="does not hold"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=token, session_id="automation")
    assert store.get_wrap_history() == []


def test_status_reports_the_policy_on_every_transport(tmp_path):
    import subprocess
    import sys

    db = str(tmp_path / "st.db")
    s = Store(db)
    assert s.status().consolidate_requires_baton is False
    s.set_consolidate_requires_baton(True)
    assert s.status().consolidate_requires_baton is True
    s.close()
    run = subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", db, "status", "--json"],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["consolidate_requires_baton"] is True
    text = subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", db, "status"],
        capture_output=True, text=True,
    ).stdout
    assert "Baton-protected" in text


# -- L3 round-1 fixes (codex + complement, 2026-09-24) --


def _lock_is_held_elsewhere(cp) -> bool:
    fcntl = pytest.importorskip("fcntl")
    import errno as _errno

    lock = sessions._anchor(cp).with_name(sessions._anchor(cp).name + ".baton.lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        assert exc.errno in (_errno.EWOULDBLOCK, _errno.EAGAIN)
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)




def test_non_bool_take_and_allow_sole_live_are_refused(store):
    cp = store.continuity_path
    sessions.claim_baton(cp, "A")
    with pytest.raises(TypeError):
        sessions.claim_baton(cp, "B", take="false")  # type: ignore[arg-type]
    assert sessions.holds_baton(cp, "A")
    with pytest.raises(TypeError):
        sessions.consolidate_authorized(cp, "B", allow_sole_live="false")  # type: ignore[arg-type]
    store.record("obs", EpisodeType.OBSERVATION)
    with pytest.raises(TypeError):
        prepare_wrap(store, session_id="B", allow_sole_live="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError):  # checked at entry, on every path
        prepare_wrap(store, allow_sole_live="false")  # type: ignore[arg-type]
    assert not store.status().wrap_in_progress


def test_release_without_a_baton_touches_no_lock_file(cp):
    assert sessions.release_baton(cp, "me") is False
    sessions.close_session(cp, "me")
    assert not any(p.name.endswith(".baton.lock") for p in cp.parent.iterdir())


def test_a_take_during_the_save_rolls_the_commit_back(store, monkeypatch):
    # codex L3 HIGH (round 1): the early baton check was a one-shot TOCTOU. The authoritative
    # re-check runs inside the batch, after wrap_completed, and a failure rolls the batch back.
    store.record("obs", EpisodeType.OBSERVATION)
    cp = store.continuity_path
    sessions.claim_baton(cp, "A")
    prep = prepare_wrap(store, session_id="A")
    real = store.wrap_completed

    def take_then_complete(*a, **k):
        sessions.claim_baton(cp, "B", take=True)  # lands after the early check
        return real(*a, **k)

    monkeypatch.setattr(store, "wrap_completed", take_then_complete)
    before = store.continuity_path.read_text() if store.continuity_path.exists() else None
    with pytest.raises(ValueError, match="does not hold the consolidate baton"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"],
                                  session_id="A")
    assert store.get_wrap_history() == []
    assert store.status().wrap_in_progress  # rolled back to what prepare left
    after = store.continuity_path.read_text() if store.continuity_path.exists() else None
    assert after == before
    leftovers = [q.name for q in store.continuity_path.parent.iterdir() if ".tmp" in q.name]
    assert leftovers == []


def test_a_policy_flip_during_a_tokenless_save_rolls_it_back(store, tmp_path, monkeypatch):
    # codex L3 HIGH (round 1): the policy could be switched on between a save's read and its
    # commit. A second connection flips it mid-save; the in-transaction re-read catches it.
    # Flipped in the gap after the early check and before the batch (the tmp staging step).
    # Inside the batch the other connection cannot write at all: this save holds the lock.
    store.record("obs", EpisodeType.OBSERVATION)
    prepare_wrap(store)
    real = store._prepare_continuity_write

    def flip_then_stage(*a, **k):
        other = Store(str(store.path))
        other.set_consolidate_requires_baton(True)
        other.close()
        return real(*a, **k)

    monkeypatch.setattr(store, "_prepare_continuity_write", flip_then_stage)
    with pytest.raises(ValueError, match="baton-protected"):
        validated_save_continuity(store, _WRAP_TEXT)
    assert store.get_wrap_history() == []
    assert store.status().wrap_in_progress



def test_an_unexpected_link_error_is_not_swallowed(cp, monkeypatch):
    import errno as _errno

    _no_lock(monkeypatch)

    def broken(src, dst):
        raise OSError(_errno.EIO, "io error")

    monkeypatch.setattr(sessions.os, "link", broken)
    with pytest.raises(OSError) as ei:
        sessions.claim_baton(cp, "s1")
    assert ei.value.errno == _errno.EIO
    assert not sessions._baton_path(cp).exists()


def test_close_session_warns_when_the_release_fails(cp, monkeypatch):
    sessions.claim_baton(cp, "me")

    def boom(*a, **k):
        raise PermissionError("lock file not ours")

    monkeypatch.setattr(sessions, "release_baton", boom)
    with pytest.warns(UserWarning, match="could not release the baton"):
        sessions.close_session(cp, "me")


# -- L3 round-3 fixes --



def test_close_session_releases_even_when_the_unlink_fails(cp, monkeypatch):
    sessions.register_session(cp, "me")
    sessions.claim_baton(cp, "me")
    real_unlink = sessions.Path.unlink

    def no_unlink(self, *a, **k):
        if self.suffix == ".session":
            raise PermissionError("read-only registry")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(sessions.Path, "unlink", no_unlink)
    with pytest.raises(PermissionError):
        sessions.close_session(cp, "me")
    assert sessions.baton_holder(cp) is None  # the release still ran


def test_close_session_warning_cannot_raise(cp, monkeypatch):
    import warnings as _w

    sessions.claim_baton(cp, "me")
    monkeypatch.setattr(sessions, "release_baton",
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError("x")))
    with _w.catch_warnings():
        _w.simplefilter("error")
        sessions.close_session(cp, "me")  # -W error: must not raise


def test_no_flock_no_hardlinks_fails_closed_and_take_still_works(cp, monkeypatch):
    # L3 rounds 2-4: the O_EXCL fallback kept producing HIGHs (racer deletion, wedged baton).
    # Withdrawn: without flock and hard links an unheld claim fails closed, nothing written.
    import errno as _errno

    _no_lock(monkeypatch)
    monkeypatch.setattr(sessions.os, "link",
                        lambda s, d: (_ for _ in ()).throw(OSError(_errno.EPERM, "no links")))
    with pytest.raises(OSError) as ei:
        sessions.claim_baton(cp, "s1")
    assert ei.value.errno == _errno.EPERM  # the real cause, not a relabel
    assert not sessions._baton_path(cp).exists()
    assert [q.name for q in cp.parent.iterdir() if q.name.startswith(".")] == []
    assert sessions.claim_baton(cp, "s1", take=True)["previous_holder"] is None
    assert sessions.holds_baton(cp, "s1")


def test_close_session_survives_a_broken_log_handler(cp, monkeypatch):
    import logging as _logging
    import warnings as _w

    class Broken(_logging.Handler):
        def emit(self, record):
            raise OSError("handler down")

    logger = _logging.getLogger("anneal_memory.sessions")
    handler = Broken()
    logger.addHandler(handler)
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(_logging, "raiseExceptions", False, raising=False)
    try:
        monkeypatch.setattr(sessions, "release_baton",
                            lambda *a, **k: (_ for _ in ()).throw(PermissionError("x")))
        with _w.catch_warnings():
            _w.simplefilter("error")
            sessions.close_session(cp, "me")
    finally:
        logger.removeHandler(handler)


# -- sole-live prepare -> save (Diogenes 2026-09-25) --


def test_sole_live_prepare_then_save_succeeds(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")  # no baton claimed
    prep = prepare_wrap(store, session_id="me", allow_sole_live=True)
    assert prep["status"] == "ready"
    validated_save_continuity(
        store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me", allow_sole_live=True
    )
    assert not store.status().wrap_in_progress
    assert len(store.get_wrap_history()) == 1


def test_sole_live_save_without_the_flag_names_the_never_held_cause(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me", allow_sole_live=True)
    with pytest.raises(ValueError, match="no baton is claimed") as exc:
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me")
    assert "unreadable" not in str(exc.value) and "taken" not in str(exc.value)
    assert store.status().wrap_in_progress and store.get_wrap_history() == []


def test_sole_live_save_refused_when_a_second_session_goes_live(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me", allow_sole_live=True)
    sessions.register_session(store.continuity_path, "peer")
    with pytest.raises(ValueError, match="no longer authorized"):
        validated_save_continuity(
            store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me",
            allow_sole_live=True,
        )
    assert store.status().wrap_in_progress and store.get_wrap_history() == []


def test_save_names_the_current_holder_when_the_baton_was_taken(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.claim_baton(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me")
    sessions.claim_baton(store.continuity_path, "other", take=True)
    with pytest.raises(ValueError, match="held by 'other'"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me")


def test_policy_store_ignores_allow_sole_live_at_save(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.claim_baton(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me")
    store.set_consolidate_requires_baton(True)
    sessions.release_baton(store.continuity_path, "me")
    sessions.register_session(store.continuity_path, "me")  # sole live, no baton
    with pytest.raises(ValueError, match="does not hold the consolidate baton") as exc:
        validated_save_continuity(
            store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me",
            allow_sole_live=True,
        )
    assert "baton-protected" in str(exc.value)
    assert "must pass allow_sole_live=True" not in str(exc.value)  # false advice here
    assert store.status().wrap_in_progress


@pytest.mark.parametrize("bad", ["false", 1, None])
def test_save_refuses_a_non_bool_allow_sole_live(store, bad):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    prep = prepare_wrap(store, session_id="me", allow_sole_live=True)
    with pytest.raises(TypeError, match="allow_sole_live must be a bool"):
        validated_save_continuity(
            store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="me",
            allow_sole_live=bad,
        )
    assert store.status().wrap_in_progress


# -- three L3 findings on 0.9.14 (codex + complement, 2026-09-25) --


def _ready_wrap_for(store, sid):
    ep = store.record("obs", EpisodeType.OBSERVATION)
    sessions.claim_baton(store.continuity_path, sid)
    prep = prepare_wrap(store, session_id=sid)
    assert prep["status"] == "ready"
    return ep, prep


def test_unauthorized_caller_cannot_cancel_the_holders_wrap_via_the_empty_path(store):
    ep, prep = _ready_wrap_for(store, "holder")
    assert store.delete(ep.id)  # the window is now empty while the wrap is open
    sessions.register_session(store.continuity_path, "intruder")
    result = prepare_wrap(store, session_id="intruder")
    assert result["status"] == "downgraded"
    assert store.status().wrap_in_progress  # NOT cancelled
    assert store.load_wrap_snapshot()["token"] == prep["wrap_token"]


def test_policy_store_sessionless_caller_cannot_cancel_via_the_empty_path(store):
    ep, prep = _ready_wrap_for(store, "holder")
    store.set_consolidate_requires_baton(True)
    assert store.delete(ep.id)
    result = prepare_wrap(store)  # no session_id
    assert result["status"] == "downgraded"
    assert store.status().wrap_in_progress


def test_the_holder_still_recovers_an_emptied_wrap_via_the_empty_path(store):
    ep, _ = _ready_wrap_for(store, "holder")
    assert store.delete(ep.id)
    assert prepare_wrap(store, session_id="holder")["status"] == "empty"
    assert not store.status().wrap_in_progress  # authorized recovery unchanged


def test_save_omitting_session_id_is_refused_when_prepare_was_gated(store):
    _, prep = _ready_wrap_for(store, "A")
    sessions.claim_baton(store.continuity_path, "B", take=True)  # A's baton is revoked
    with pytest.raises(ValueError, match="must come from that session"):
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"])
    assert store.status().wrap_in_progress and store.get_wrap_history() == []
    assert store.wrap_gated_session() == "A"


def test_an_ungated_wrap_still_saves_without_session_id(store):
    store.record("obs", EpisodeType.OBSERVATION)
    prep = prepare_wrap(store)  # no session_id, no policy: the opt-in gate stays off
    assert store.wrap_gated_session() is None
    validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"])
    assert len(store.get_wrap_history()) == 1


def test_gated_session_key_clears_on_complete_and_cancel(store):
    _, prep = _ready_wrap_for(store, "A")
    assert store.wrap_gated_session() == "A"
    validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="A")
    assert store.wrap_gated_session() is None
    store.record("obs2", EpisodeType.OBSERVATION)
    prepare_wrap(store, session_id="A")
    assert store.wrap_gated_session() == "A"
    store.wrap_cancelled()
    assert store.wrap_gated_session() is None


def test_a_baton_taken_during_the_package_build_does_not_start_a_wrap(store, monkeypatch):
    from anneal_memory import continuity

    store.record("obs", EpisodeType.OBSERVATION)
    sessions.claim_baton(store.continuity_path, "A")
    real = continuity._build_wrap_package

    def build_then_lose_the_baton(*a, **kw):
        out = real(*a, **kw)
        sessions.claim_baton(store.continuity_path, "B", take=True)
        return out

    monkeypatch.setattr(continuity, "_build_wrap_package", build_then_lose_the_baton)
    result = prepare_wrap(store, session_id="A")
    assert result["status"] == "downgraded" and result["wrap_token"] is None
    assert not store.status().wrap_in_progress  # wrap_started never ran


@pytest.mark.parametrize("bad", ["", 123])
def test_wrap_started_validates_gated_session_id(store, bad):
    with pytest.raises(ValueError, match="gated_session_id"):
        store.wrap_started(token="t" * 32, episode_ids=[], gated_session_id=bad)
    assert not store.status().wrap_in_progress


def test_sessionless_caller_cannot_cancel_a_gated_wrap_via_the_empty_path(store):
    ep, prep = _ready_wrap_for(store, "holder")
    assert store.delete(ep.id)
    result = prepare_wrap(store)  # default store, no session_id: ungated by omission
    assert result["status"] == "downgraded"
    assert "downgraded-gated-wrap-open" in result["message"]
    assert store.status().wrap_in_progress
    assert store.load_wrap_snapshot()["token"] == prep["wrap_token"]


# -- 0.9.15 residue: strict match, empty-path CAS, audit records --


def _audit_events(store):
    path = store.path.with_suffix(".audit.jsonl")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_strict_match_a_new_holder_cannot_commit_the_previous_holders_wrap(store):
    _, prep = _ready_wrap_for(store, "A")
    sessions.claim_baton(store.continuity_path, "B", take=True)
    with pytest.raises(ValueError, match="only that session may commit it") as exc:
        validated_save_continuity(store, _WRAP_TEXT, wrap_token=prep["wrap_token"], session_id="B")
    assert "prepare_wrap again" in str(exc.value)
    assert store.status().wrap_in_progress and store.get_wrap_history() == []


def test_empty_prepare_on_an_idle_store_writes_nothing(store, monkeypatch):
    def never(*a, **kw):
        raise AssertionError("an idle store must not be cancelled")

    monkeypatch.setattr(store, "wrap_cancelled", never)
    assert prepare_wrap(store)["status"] == "empty"
    assert prepare_wrap(store, session_id="anyone")["status"] in ("empty", "downgraded")


def test_empty_path_does_not_cancel_a_wrap_started_after_the_window_was_read(store, monkeypatch):
    # L1: the peer's wrap starts AFTER the empty-window read; the snapshot was observed first,
    # so the cancel is a CAS on the OLD observation and cannot destroy the peer's wrap.
    from anneal_memory import continuity

    ep, _ = _ready_wrap_for(store, "holder")
    assert store.delete(ep.id)
    real = store.episodes_since_wrap
    peer = {}

    def window_then_a_peer_starts_a_wrap():
        out = real()
        assert out == []
        store.wrap_cancelled()
        ep2 = store.record("fresh", EpisodeType.OBSERVATION)
        peer["token"] = "e" * 32
        store.wrap_started(token=peer["token"], episode_ids=[ep2.id])
        return out

    monkeypatch.setattr(store, "episodes_since_wrap", window_then_a_peer_starts_a_wrap)
    result = prepare_wrap(store, session_id="holder")
    monkeypatch.undo()
    assert result["status"] == "downgraded"
    assert store.load_wrap_snapshot()["token"] == peer["token"]


def test_a_stale_gated_key_with_no_wrap_is_inert(store):
    store._conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('wrap_gated_session', 'ghost')"
    )
    store._conn.commit()
    assert store.wrap_gated_session() is None
    assert prepare_wrap(store)["status"] == "empty"  # not a phantom gated-wrap downgrade


def test_cancel_audit_records_the_gated_session(store):
    _ready_wrap_for(store, "A")
    store.wrap_cancelled()
    ev = [e for e in _audit_events(store) if e["event"] == "wrap_cancelled"][-1]
    assert ev["data"]["wrap_gated_session"] == "A"


def test_zero_edge_compost_is_audited(store):
    assert store.sever_pattern_concept("ghost") == 0
    ev = [e for e in _audit_events(store) if e["event"] == "pattern_concept_severed"]
    assert ev and ev[-1]["data"] == {"name": "ghost", "severed": 0}


def test_policy_change_takes_the_write_lock_before_reading_the_previous_value(store):
    stmts = []
    store._conn.set_trace_callback(stmts.append)
    store.set_consolidate_requires_baton(True)
    store._conn.set_trace_callback(None)
    begin = next(i for i, x in enumerate(stmts) if x.strip().upper().startswith("BEGIN IMMEDIATE"))
    read = next(i for i, x in enumerate(stmts) if "SELECT" in x.upper() and "consolidate_requires_baton" in x)
    assert begin < read


def test_policy_change_audit_records_the_real_previous_value(store):
    store.set_consolidate_requires_baton(True)
    store.set_consolidate_requires_baton(False)
    evs = [e["data"] for e in _audit_events(store) if e["event"] == "consolidate_policy_set"]
    assert evs[-2:] == [{"requires_baton": True, "was": False}, {"requires_baton": False, "was": True}]


def test_cli_wrap_status_shows_the_preparing_session(tmp_path):
    import subprocess
    import sys

    db = str(tmp_path / "ws.db")
    s = Store(db)
    s.record("obs", EpisodeType.OBSERVATION)
    sessions.claim_baton(s.continuity_path, "A")
    prepare_wrap(s, session_id="A")
    s.close()
    run = subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", db, "wrap-status", "--json"],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["wrap_gated_session"] == "A"
