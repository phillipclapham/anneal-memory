"""A tokenless wrap-cancel may not end a wrap prepared under the consolidate gate
by another session (spore-699 bound). Reproduced on 0.9.21 before the fix: a
tokenless ``wrap-cancel`` from a session with no claim cleared the holder's wrap.
"""

from __future__ import annotations

import pickle

import pytest

from anneal_memory import Store, WrapCancelGatedError
from anneal_memory.server import Server


def _gated(store: Store, token: str, session: str | None = "holder") -> None:
    e = store.record("x" * 90 + token, "observation")
    store.wrap_started(token=token, episode_ids=[e.id], gated_session_id=session)


def test_gated_wrap_needs_its_session_token_or_force(tmp_path, monkeypatch):
    store = Store(tmp_path / "m.db")
    _gated(store, "a" * 32)
    for kw in ({}, {"session_id": "other"}, {"force": 1}):
        with pytest.raises(WrapCancelGatedError) as exc:
            store.wrap_cancelled(**kw)
        assert exc.value.gated_session == "holder"
        assert store.wrap_gated_session() == "holder"  # nothing changed
    err = pickle.loads(pickle.dumps(exc.value))
    assert err.gated_session == "holder"

    assert store.wrap_cancelled(session_id="holder").token == "a" * 32
    _gated(store, "b" * 32)
    assert store.wrap_cancelled(force=True).token == "b" * 32
    _gated(store, "c" * 32)
    assert store.wrap_cancelled(expect_token="c" * 32, session_id="other").token == "c" * 32
    _gated(store, "d" * 32, session=None)  # ungated: tokenless cancel unchanged
    assert store.wrap_cancelled().token == "d" * 32

    # The MCP surface carries the same bound.
    _gated(store, "e" * 32)
    server = Server.__new__(Server)
    server._store = store
    refused = server._tool_wrap_cancel({"session_id": "other"})
    text = refused["content"][0]["text"]
    # the refusal carries no recipe: its reader is the caller the bound stops
    assert refused["isError"] and "holder" not in text and "force" not in text
    assert store.wrap_gated_session() == "holder"
    assert not server._tool_wrap_cancel({"session_id": "holder"}).get("isError")
    assert store.wrap_gated_session() is None

    # prepare_wrap's empty-window path observes PARTIAL state and cancels it
    # tokenlessly; if a peer's gated wrap lands first, the store refuses, and
    # prepare_wrap must downgrade rather than raise.
    from anneal_memory import prepare_wrap

    store = Store(tmp_path / "m2.db")
    for k, v in (("wrap_started_at", "2026-09-25T00:00:00Z"), ("wrap_gated_session", "A")):
        store._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (k, v))
    store._conn.commit()

    def peer_landed(**_kw):
        raise WrapCancelGatedError(gated_session="A", session_id=None)

    monkeypatch.setattr(store, "wrap_cancelled", peer_landed)
    result = prepare_wrap(store)
    assert result["status"] != "ready" and "replaced" in result["message"]
