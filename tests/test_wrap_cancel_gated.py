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


def test_a_mismatch_refusal_never_offers_an_override_the_gate_refuses(tmp_path):
    """Diogenes 10-03 (edd780d2e03e): a wrong-token cancel of a gated wrap told the
    caller to cancel again without a token, which WrapCancelGatedError then refused.
    Reproduced on 636558f through the MCP handler and the CLI. Each surface's
    advice is followed here, not just read."""
    import subprocess
    import sys

    from anneal_memory import WrapInProgressError, WrapOwnershipError

    db = tmp_path / "m.db"
    store = Store(db)
    _gated(store, "a" * 32)
    wrong = "0" * 32

    with pytest.raises(WrapOwnershipError) as exc:
        store.wrap_cancelled(expect_token=wrong)
    assert exc.value.gated_session == "holder"
    assert "without expect_token" not in str(exc.value)
    with pytest.raises(WrapInProgressError) as wip:
        store.wrap_started(token="1" * 32, episode_ids=[])
    assert "consolidate gate" in str(wip.value)

    server = Server(store)
    text = server._tool_wrap_cancel({"wrap_token": wrong})["content"][0]["text"]
    assert "WITHOUT wrap_token" not in text and "operator's decision" in text
    # The holder's own mismatch still gets the override, and it works for the holder.
    held = server._tool_wrap_cancel({"wrap_token": wrong, "session_id": "holder"})
    assert "WITHOUT wrap_token to override (keep session_id)" in held["content"][0]["text"]
    store.close()

    import os
    from pathlib import Path

    import anneal_memory

    # The subprocess must import THIS tree, not a site-packages copy in the venv.
    env = {**os.environ, "PYTHONPATH": str(Path(anneal_memory.__file__).parent.parent)}

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(cli + list(args), capture_output=True, text=True, env=env)

    cli = [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), "wrap-cancel"]
    out = run("--wrap-token", wrong)
    assert out.returncode == 1
    assert "without --wrap-token" not in out.stderr and "operator's decision" in out.stderr
    out = run("--wrap-token", wrong, "--session-id", "holder")
    assert "re-run without --wrap-token (keep --session-id)" in out.stderr
    out = run("--session-id", "holder")
    assert out.returncode == 0, out.stderr
    assert Store(db).wrap_gated_session() is None
