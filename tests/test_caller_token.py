"""prepare_wrap(wrap_token=...): a caller-supplied token, and the token-bound wrap it opens.

A caller that mints its own token holds the wrap's identity before prepare_wrap returns, so
every cancel it makes can compare tokens; a cancel that names no token cannot end such a wrap
(WrapCancelBoundError) except with force. Levain 0.5.7's out-of-band exit path is the caller
this exists for: a tokenless cancel there could end a peer's wrap (codex + gemini, 10-03).
The interleaved two-process run behind these tests is
project_memory/reference/caller_token_run_1004.log.
"""

from __future__ import annotations

import pickle
import uuid

import pytest

from anneal_memory import (
    Store,
    WrapCancelBoundError,
    WrapOwnershipError,
    prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.types import EpisodeType

_WRAP_TEXT = (
    "# T — Memory (v1)\n\n"
    "## State\nActive.\n\n"
    "## Patterns\nNone yet.\n\n"
    "## Decisions\nNone.\n\n"
    "## Context\nFirst session.\n"
)


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "wrap.db"), project_name="TokenTest")
    s.record("obs", EpisodeType.OBSERVATION)
    yield s
    s.close()


def test_caller_token_is_the_wrap_token_and_saves(store):
    with pytest.raises(TypeError):
        prepare_wrap(store, wrap_token=123)  # type: ignore[arg-type]
    for bad in ("", "ABCDEF" + "0" * 26, "0" * 31, str(uuid.uuid4())):
        with pytest.raises(ValueError):
            prepare_wrap(store, wrap_token=bad)
    assert store.get_wrap_started_at() is None  # refused before any write

    t = uuid.uuid4().hex
    r = prepare_wrap(store, wrap_token=t)
    assert r["status"] == "ready" and r["wrap_token"] == t
    assert store.load_wrap_snapshot()["token"] == t
    assert store.wrap_bound_token() == t
    validated_save_continuity(store, _WRAP_TEXT, wrap_token=t, today="2026-10-04")
    assert store.get_wrap_started_at() is None
    assert store.wrap_bound_token() is None


def test_cancel_by_another_token_is_refused_and_changes_nothing(store):
    t = uuid.uuid4().hex
    prepare_wrap(store, wrap_token=t)
    other = uuid.uuid4().hex
    with pytest.raises(WrapOwnershipError) as exc:
        store.wrap_cancelled(expect_token=other)
    assert exc.value.actual == t and exc.value.bound is True
    assert "without" not in str(exc.value)  # no tokenless override is offered
    again = pickle.loads(pickle.dumps(exc.value))
    assert again.bound is True and str(again) == str(exc.value)
    assert store.load_wrap_snapshot()["token"] == t
    # The holder's own token clears it.
    assert store.wrap_cancelled(expect_token=t).token == t
    assert store.get_wrap_started_at() is None


def test_tokenless_cancel_of_a_bound_wrap_is_refused(store):
    t = uuid.uuid4().hex
    prepare_wrap(store, wrap_token=t)
    for kwargs in ({}, {"session_id": "anyone"}):
        with pytest.raises(WrapCancelBoundError) as exc:
            store.wrap_cancelled(**kwargs)
        assert pickle.loads(pickle.dumps(exc.value)).session_id == kwargs.get("session_id")
    assert store.load_wrap_snapshot()["token"] == t

    # prepare_wrap's empty-window path does not cancel it either, unless it holds the token.
    for ep in store.episodes_since_wrap():  # empty the window, as a delete or prune can
        store.delete(ep.id)
    r = prepare_wrap(store)
    assert r["status"] == "downgraded" and "downgraded-bound-wrap-open" in r["message"]
    assert store.load_wrap_snapshot()["token"] == t
    assert prepare_wrap(store, wrap_token=t)["status"] == "empty"
    assert store.get_wrap_started_at() is None

    # The operator's force still clears a bound wrap.
    store.record("obs2", EpisodeType.OBSERVATION)
    prepare_wrap(store, wrap_token=uuid.uuid4().hex)
    assert store.wrap_cancelled(force=True).token is not None
    assert store.get_wrap_started_at() is None


def test_no_token_passed_keeps_the_old_behaviour(store):
    r = prepare_wrap(store)
    assert r["status"] == "ready" and len(r["wrap_token"]) == 32
    assert store.wrap_bound_token() is None
    assert store.wrap_cancelled().token == r["wrap_token"]  # tokenless clear still works

    # A wrap_bound_token that does not equal the live token (left by a binary that
    # predates the key) binds nothing.
    r = prepare_wrap(store)
    with store._conn:
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('wrap_bound_token', ?)",
            ("f" * 32,),
        )
    assert store.wrap_bound_token() is None
    assert store.wrap_cancelled().token == r["wrap_token"]
