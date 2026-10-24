"""Generation 3, design B3: idempotency keys bound to the whole request, and the
fenced consolidation lease (holder, epoch, expiry and identity version in one
predicate). The runs behind these tests are in
project_memory/seat_1023_28/b3_repro.txt.
"""

from __future__ import annotations

import time
import warnings

import pytest

from anneal_memory import (
    IdempotencyConflictError,
    LeaseFencedError,
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

    # The save: a keyed retry gets its result where an unkeyed one is a phantom
    # re-save, and the same key with other text changes nothing.
    p = prepare_wrap(s, idempotency_key="prep-1")
    assert prepare_wrap(s, idempotency_key="prep-1")["wrap_token"] == p["wrap_token"]
    saved = validated_save_continuity(
        s, _TEXT, wrap_token=p["wrap_token"], today="2026-10-23", idempotency_key="save-1")
    again = validated_save_continuity(
        s, _TEXT, wrap_token=p["wrap_token"], today="2026-10-23", idempotency_key="save-1")
    assert again["wrap_id"] == saved["wrap_id"] and "provisional" not in again
    with pytest.raises(IdempotencyConflictError):
        validated_save_continuity(
            s, _TEXT.replace("Active.", "Other."), wrap_token=p["wrap_token"],
            today="2026-10-23", idempotency_key="save-1")
    assert (_count(s, "wraps"), _count(s, "identity_versions")) == (1, 1)
    assert s.load_continuity() == _TEXT
    s.close()


def test_t5_a_save_prepared_under_an_expired_lease_is_fenced(tmp_path):
    path = str(tmp_path / "lease.db")
    a = Store(path, project_name="T")
    b = Store(path, project_name="T")  # another process on the same store
    a.record("obs", EpisodeType.OBSERVATION)
    assert a.lease_acquire("A", ttl_seconds=0.3) == 1
    pa = prepare_wrap(a, lease_holder="A")
    assert a.wrap_lease_fence() == {"holder": "A", "epoch": 1, "identity_version": None}
    time.sleep(0.4)

    def save(store, holder, text=_TEXT, token=pa["wrap_token"]):
        return validated_save_continuity(
            store, text, wrap_token=token, today="2026-10-23", lease_holder=holder)

    # Expired with no one taking over: refused all the same (draft 2 checked the
    # epoch only, and let this commit).
    with pytest.raises(LeaseFencedError):
        save(a, "A")
    assert b.lease_acquire("B", ttl_seconds=60) == 2
    with pytest.raises(LeaseFencedError):
        save(a, "A")
    with pytest.raises(LeaseFencedError):  # the token is not ownership
        save(b, "B", text=_TEXT.replace("Active.", "B's text."))
    with pytest.raises(LeaseFencedError):
        validated_save_continuity(b, _TEXT, wrap_token=pa["wrap_token"], today="2026-10-23")
    assert (_count(a, "wraps"), _count(a, "identity_versions")) == (0, 0)
    with pytest.raises(LeaseFencedError):
        a.lease_acquire("A", ttl_seconds=60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b.wrap_cancelled()
    pb = prepare_wrap(b, lease_holder="B")
    assert save(b, "B", token=pb["wrap_token"])["wrap_id"] == 1
    assert (_count(a, "wraps"), _count(a, "identity_versions")) == (1, 1)
    a.close()
    b.close()
