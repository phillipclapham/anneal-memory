"""Generation 3, design B3: idempotency keys bound to the whole request, and the
fenced consolidation lease (holder, epoch, expiry and identity version in one
predicate). The runs behind these tests are in
project_memory/seat_1023_28/b3_repro.txt.
"""

from __future__ import annotations

import time
import warnings
from unittest import mock

import pytest

from anneal_memory import (
    IdempotencyConflictError,
    LeaseFencedError,
    SaveContinuityReplayResult,
    Store,
    prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.types import EpisodeType

_TEXT = (
    "# T — Memory (v1)\n\n"
    "## State\nActive.\n\n"
    "## Patterns\nNone yet.\n\n"
    "## Decisions\nNone.\n\n"
    "## Context\nFirst session.\n"
)


def _count(store: Store, table: str) -> int:
    return store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_t4_an_idempotency_key_reused_for_another_request_is_refused(tmp_path):
    s = Store(str(tmp_path / "idem.db"), project_name="T")
    first = s.record("idem one", EpisodeType.OBSERVATION, idempotency_key="req-1")
    # The lost-response retry: same key, same request, the same episode, no write.
    assert s.record("idem one", EpisodeType.OBSERVATION, idempotency_key="req-1").id == first.id
    with pytest.raises(IdempotencyConflictError):
        s.record("idem two", EpisodeType.OBSERVATION, idempotency_key="req-1")
    with pytest.raises(IdempotencyConflictError):  # an argument the old key compared not at all
        s.record("idem one", EpisodeType.OBSERVATION, metadata={"x": 1}, idempotency_key="req-1")
    assert _count(s, "episodes") == 1
    # B3 L3 r1 U3: ids derive from content and timestamp, so a deleted episode
    # recreated with another type reuses the id; the keyed retry is refused.
    ts = "2026-10-23T01:00:00.000000Z"
    gone = s.record("aba", EpisodeType.OBSERVATION, timestamp=ts, idempotency_key="req-aba")
    s.delete(gone.id)
    assert s.record("aba", EpisodeType.DECISION, timestamp=ts).id == gone.id
    with pytest.raises(IdempotencyConflictError) as erased:
        s.record("aba", EpisodeType.OBSERVATION, timestamp=ts, idempotency_key="req-aba")
    assert erased.value.reason == "effect_erased"
    # B3 L3 r2 V2: set_trust after a keyed record is not an erasure; the retry
    # replays the same episode.
    trusted = s.record("trust moves", EpisodeType.OBSERVATION, idempotency_key="req-trust")
    s.set_trust(trusted.id, "external")
    assert s.record("trust moves", EpisodeType.OBSERVATION,
                    idempotency_key="req-trust").id == trusted.id

    # The save: a keyed retry gets its result where an unkeyed one is a phantom
    # re-save, and the same key with other text changes nothing.
    p = prepare_wrap(s, idempotency_key="prep-1")
    assert prepare_wrap(s, idempotency_key="prep-1") == p  # rebuilt from the open snapshot
    # r1 T4: the key row keeps the wrap token, never the package's episode text.
    row = s._conn.execute("SELECT result_json FROM idempotency_keys WHERE key = 'prep-1'").fetchone()
    assert "idem one" not in row[0] and p["wrap_token"] in row[0]
    # r1 T3: a retry whose early read raced its first call is a replay at every
    # later wrap-in-progress check, prepare_wrap's own and wrap_started's.
    with mock.patch.object(Store, "_idempotency_replay", side_effect=[(False, None), (True, row[0])]):
        assert prepare_wrap(s, idempotency_key="prep-1")["wrap_token"] == p["wrap_token"]
    with mock.patch.object(Store, "_idempotency_replay", return_value=(False, None)), \
            mock.patch.object(Store, "get_wrap_started_at", return_value=None):
        assert prepare_wrap(s, idempotency_key="prep-1")["wrap_token"] == p["wrap_token"]
    saved = validated_save_continuity(
        s, _TEXT, wrap_token=p["wrap_token"], today="2026-10-23",
        compost=iter(["gone_pattern"]), idempotency_key="save-1")
    assert saved["composted"] == {"gone_pattern": 0}  # r1 U6: the iterator was hashed, not spent
    again = validated_save_continuity(
        s, _TEXT, wrap_token=p["wrap_token"], today="2026-10-23",
        compost=["gone_pattern"], idempotency_key="save-1")
    assert again["wrap_id"] == saved["wrap_id"] and again["replayed"] is True
    # B3 L3 r2 V3: the replay is the allowlist, nothing more.
    assert set(again) == set(SaveContinuityReplayResult.__annotations__)
    with pytest.raises(IdempotencyConflictError):
        validated_save_continuity(
            s, _TEXT.replace("Active.", "Other."), wrap_token=p["wrap_token"],
            today="2026-10-23", compost=["gone_pattern"], idempotency_key="save-1")
    assert (_count(s, "wraps"), _count(s, "identity_versions")) == (1, 1)
    assert s.load_continuity() == _TEXT
    with pytest.raises(IdempotencyConflictError) as closed:  # r1 T4: never a stale package
        prepare_wrap(s, idempotency_key="prep-1")
    assert closed.value.reason == "wrap_closed"
    # B3 L3 r1 U4: a replay that would rebuild another package is refused.
    s.record("idem three", EpisodeType.OBSERVATION)
    prepare_wrap(s, idempotency_key="prep-2")
    s.save_continuity(_TEXT.replace("Active.", "Moved."))
    with pytest.raises(IdempotencyConflictError) as moved:
        prepare_wrap(s, idempotency_key="prep-2")
    assert moved.value.reason == "sources_moved"
    s.close()


def test_t5_a_save_prepared_under_an_expired_lease_is_fenced(tmp_path):
    path = str(tmp_path / "lease.db")
    a = Store(path, project_name="T")
    b = Store(path, project_name="T")  # another process on the same store
    a.record("obs", EpisodeType.OBSERVATION)
    ga = a.lease_acquire("A", ttl_seconds=0.6)
    assert ga.epoch == 1 and ga.token not in repr(ga)
    with pytest.raises(LeaseFencedError) as tokenless:  # B3 L3 r1 U1
        prepare_wrap(b)
    assert tokenless.value.reason == "lease_held"
    pa = prepare_wrap(a, lease_token=ga.token)
    assert a.wrap_lease_fence() == {"holder": "A", "epoch": 1, "identity_version": None}

    def save(store, token, text=_TEXT, wrap=pa["wrap_token"]):
        return validated_save_continuity(
            store, text, wrap_token=wrap, today="2026-10-23", lease_token=token)

    # r1 T1: with A's lease live, another writer presenting no token or the wrong
    # one cannot commit A's wrap (draft 1 checked it as the preparer and let B in).
    for token in (None, "0" * 32):
        with pytest.raises(LeaseFencedError) as fenced:
            save(b, token, text=_TEXT.replace("Active.", "B's text."))
        assert fenced.value.reason == "token_mismatch"
    with pytest.raises(LeaseFencedError):  # one holder name is not one tenure
        b.lease_acquire("A", ttl_seconds=60)
    # r1 T2: every identity commit on a leased store presents the token.
    with pytest.raises(LeaseFencedError) as raw:
        b.save_continuity(_TEXT.replace("Active.", "raw."))
    assert raw.value.reason == "lease_held"
    assert _count(a, "identity_versions") == 0
    time.sleep(0.7)

    # Expired with no one taking over: refused all the same (draft 2 checked the
    # epoch only, and let this commit).
    with pytest.raises(LeaseFencedError):
        save(a, ga.token)
    gb = b.lease_acquire("B", ttl_seconds=60)
    assert gb.epoch == 2
    with pytest.raises(LeaseFencedError):
        save(a, ga.token)
    with pytest.raises(LeaseFencedError):  # the wrap token is not ownership
        save(b, gb.token, text=_TEXT.replace("Active.", "B's text."))
    with pytest.raises(LeaseFencedError):
        validated_save_continuity(b, _TEXT, wrap_token=pa["wrap_token"], today="2026-10-23")
    assert (_count(a, "wraps"), _count(a, "identity_versions")) == (0, 0)
    with pytest.raises(LeaseFencedError):
        a.lease_acquire("A", ttl_seconds=60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Integration L3 I4: a leased wrap under a live lease is cancelled only
        # by a token that owns it (here the live holder's).
        with pytest.raises(LeaseFencedError) as uncancelled:
            b.wrap_cancelled(force=True)
        assert uncancelled.value.reason == "token_mismatch"
        b.wrap_cancelled(lease_token=gb.token)
    pb = prepare_wrap(b, lease_token=gb.token)
    # Integration L3 I1: a leased wrap records the crystal document. The holder's
    # own crystal change (its token) moves the record with it; a change outside
    # the fence refuses the save (version_moved).
    from anneal_memory import CrystalStore

    CrystalStore(tmp_path / "lease.crystal.json").crystallize(
        name="the_wraps_own", level=2, explanation="crystallized mid-wrap",
        lease_token=gb.token)
    own = a._conn.execute("SELECT content_hash FROM crystal_document").fetchone()[0]
    a._conn.execute("UPDATE crystal_document SET content_hash = 'elsewhere'")
    a._conn.commit()
    with pytest.raises(LeaseFencedError) as moved:
        save(b, gb.token, wrap=pb["wrap_token"])
    assert moved.value.reason == "version_moved"
    a._conn.execute("UPDATE crystal_document SET content_hash = ?", (own,))
    a._conn.commit()
    assert save(b, gb.token, wrap=pb["wrap_token"])["wrap_id"] == 1
    assert (_count(a, "wraps"), _count(a, "identity_versions")) == (1, 1)
    # Integration L3 r2 (codex MED, reproduced 10-24): the holder folds receipts
    # under its own lease; a fold presenting no token is refused, folding nothing.
    from anneal_memory.worth import fold_surfaced

    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text(
        '{"event_id": "e1", "ts": "2026-10-02T10:00:00Z", "exposed": '
        '[{"pattern": "the_wraps_own", "rank": 0, "source": "evidence_edge"}]}\n')
    crystal = CrystalStore(tmp_path / "lease.crystal.json")
    with pytest.raises(LeaseFencedError) as unfolded:
        fold_surfaced(crystal, [receipts])
    assert unfolded.value.reason == "lease_held"
    assert fold_surfaced(crystal, [receipts], lease_token=gb.token).exposures_counted == 1

    # B3 L3 r1 U1: every save carries the identity version it was prepared on,
    # leased or not, so an unleased wrap never overwrites a later version.
    assert b.lease_release(gb.token)
    a.record("obs 2", EpisodeType.OBSERVATION)
    pu = prepare_wrap(a)
    b.save_continuity(_TEXT.replace("Active.", "C1 by B."))
    with pytest.raises(LeaseFencedError) as stale:
        save(a, None, text=_TEXT.replace("Active.", "A's text."), wrap=pu["wrap_token"])
    assert stale.value.reason == "version_moved"
    assert "C1 by B." in a.load_continuity() and _count(a, "identity_versions") == 2
    a.close()
    b.close()
