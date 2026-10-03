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


def test_gated_wrap_needs_its_session_token_or_force(tmp_path):
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



def test_empty_window_recovery_never_clears_a_wrap_started_mid_call(tmp_path):
    # Reproduced on the branch before the fix (2026-10-02): prepare_wrap observed
    # PARTIAL state with an empty window; before its tokenless cancel ran, a peer
    # cleared the partial state and started a healthy UNGATED wrap, and the cancel
    # destroyed it. The cancel is now a compare-and-swap on "still partial".
    from anneal_memory import prepare_wrap

    path = tmp_path / "m.db"
    a = Store(path)
    a._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES "
                    "('wrap_started_at', '2026-09-25T00:00:00Z')")
    a._conn.commit()
    real = a.wrap_cancelled

    def peer_then_cancel(**kw):
        b = Store(path)
        b.wrap_cancelled()  # the peer clears the partial state...
        e = b.record("y" * 90, "observation")
        b.wrap_started(token="b" * 32, episode_ids=[e.id])  # ...and starts its own wrap
        b.close()
        return real(**kw)

    a.wrap_cancelled = peer_then_cancel  # type: ignore[method-assign]
    result = prepare_wrap(a)
    assert result["status"] != "ready" and "replaced" in result["message"]
    assert Store(path).load_wrap_snapshot()["token"] == "b" * 32  # the peer's wrap survives
    with pytest.raises(ValueError):
        real(expect_partial=True, expect_token="b" * 32)
