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
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-registry-error"


def test_corrupt_stale_session_file_skipped(cp):
    # A DEAD session's corruption is irrelevant (TTL skips it before the read) → gate proceeds.
    sessions.register_session(cp, "s1")
    f = sessions._session_file(cp, "s1")
    f.write_text("{ not json", encoding="utf-8")
    os.utime(f, (time.time() - 10_000, time.time() - 10_000))
    assert sessions.live_sessions(cp) == []
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is True  # me is sole; the dead corrupt peer is ignored
    assert auth["reason"] == "sole-live-session"


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


def test_baton_reclaim_records_previous_holder(cp):
    sessions.claim_baton(cp, "s1")
    claim = sessions.claim_baton(cp, "s2")
    assert claim["previous_holder"] == "s1"
    assert sessions.holds_baton(cp, "s2")


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
    # but a fresh claim still works over a corrupt baton (the previous-holder read is defensive)
    claim = sessions.claim_baton(cp, "newhead")
    assert claim["previous_holder"] is None
    assert sessions.holds_baton(cp, "newhead")


# -- the efferent decision (consolidate_authorized) --


def test_authorized_when_sole(cp):
    sessions.register_session(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is True
    assert auth["reason"] == "sole-live-session"


def test_authorized_when_no_sessions_registered(cp):
    auth = sessions.consolidate_authorized(cp, "me")  # me not even registered
    assert auth["authorized"] is True  # no OTHER live session
    assert auth["reason"] == "sole-live-session"


def test_downgrade_when_another_session_live(cp):
    sessions.register_session(cp, "other")
    auth = sessions.consolidate_authorized(cp, "me")
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


def test_gate_sole_session_ready(store):
    store.record("obs", EpisodeType.OBSERVATION)
    sessions.register_session(store.continuity_path, "me")
    result = prepare_wrap(store, session_id="me")
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
    # other closes → me is now sole → a fresh prepare must succeed (nothing stranded)
    sessions.close_session(store.continuity_path, "other")
    assert prepare_wrap(store, session_id="me")["status"] == "ready"


def test_downgrade_does_not_clear_another_sessions_inflight_wrap(store):
    store.record("obs", EpisodeType.OBSERVATION)
    # A (sole at the time) legitimately starts a wrap
    sessions.register_session(store.continuity_path, "A")
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


def test_no_baton_sole_still_authorized(cp):
    # The fix must NOT block the legitimate sole-no-baton case (the common single-session day).
    sessions.register_session(cp, "me")
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is True
    assert auth["reason"] == "sole-live-session"


def test_stale_baton_recovered_by_reclaim(cp):
    sessions.register_session(cp, "head")
    sessions.claim_baton(cp, "head")
    f = sessions._session_file(cp, "head")
    past = time.time() - 10_000
    os.utime(f, (past, past))  # head dead/idle
    sessions.register_session(cp, "lane")
    sessions.claim_baton(cp, "lane")  # the human re-designates → recovery
    auth = sessions.consolidate_authorized(cp, "lane")
    assert auth["authorized"] is True
    assert auth["reason"] == "holds-baton"


def test_registry_error_fails_closed(cp, monkeypatch):
    sessions.register_session(cp, "me")

    def boom(*a, **k):
        raise PermissionError("registry unreadable")

    monkeypatch.setattr(sessions, "live_sessions", boom)
    auth = sessions.consolidate_authorized(cp, "me")
    assert auth["authorized"] is False
    assert auth["reason"] == "downgraded-registry-error"


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
    assert sessions.consolidate_authorized(cp, "A")["authorized"] is True
    assert sessions.consolidate_authorized(cp, "B")["authorized"] is True


def test_prepare_wrap_empty_session_id_raises_clear_error(store):
    # An empty session_id is a programmer error, surfaced clearly at the public API boundary
    # rather than deep inside the gate (complement L3 LOW-2).
    store.record("obs", EpisodeType.OBSERVATION)
    with pytest.raises(ValueError, match="non-empty"):
        prepare_wrap(store, session_id="")
