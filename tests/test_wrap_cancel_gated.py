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
    # The library holder gets the override, as MCP and CLI do (L2, 1003+16).
    with pytest.raises(WrapOwnershipError) as own:
        store.wrap_cancelled(expect_token=wrong, session_id="holder")
    assert "without expect_token, keeping session_id" in str(own.value)
    assert "another session" not in str(own.value)
    clone = pickle.loads(pickle.dumps(own.value))
    assert (clone.session_id, clone.gated_session, str(clone)) == ("holder", "holder", str(own.value))

    with pytest.raises(WrapInProgressError) as wip:
        store.wrap_started(token="1" * 32, episode_ids=[])
    assert "a plain cancel of it is refused" in str(wip.value)
    with pytest.raises(WrapCancelGatedError):  # true for the holder's own plain cancel too
        store.wrap_cancelled()

    server = Server(store)
    text = server._tool_wrap_cancel({"wrap_token": wrong})["content"][0]["text"]
    assert "WITHOUT wrap_token" not in text and "operator's decision" in text
    # The holder's own mismatch still gets the override, and it works for the holder.
    held = server._tool_wrap_cancel({"wrap_token": wrong, "session_id": "holder"})
    held_text = held["content"][0]["text"]
    assert "WITHOUT wrap_token, keeping session_id" in held_text
    assert "different session" not in held_text
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
    assert "re-run without --wrap-token, keeping --session-id" in out.stderr
    out = run("--session-id", "holder")
    assert out.returncode == 0, out.stderr
    after = Store(db)
    assert after.wrap_gated_session() is None
    after.close()


def test_partial_state_recovery_is_the_partial_only_clear(tmp_path):
    """L3 r1 glm / r2 codex / r3 complement+codex+glm (1003+16, each run): a partial
    wrap's recovery advice was first false ("no usable token"), then a plain cancel
    that ends a healthy wrap a peer starts before the retry, then a token retry
    that fails for a non-canonical token or a replacement reusing the token. The
    advice is now expect_partial / partial=true / --partial, followed here."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import anneal_memory
    from anneal_memory import WrapOwnershipError

    def make_partial(st: Store, token: str) -> None:
        e = st.record("x" * 90 + token, "observation")
        st.wrap_started(token=token, episode_ids=[e.id], gated_session_id="holder")
        st._conn.execute("UPDATE metadata SET value='' WHERE key='wrap_episode_ids'")
        st._conn.commit()

    db = tmp_path / "m.db"
    store = Store(db)
    make_partial(store, "Legacy-Token")  # a token no transport accepts
    with pytest.raises(WrapOwnershipError) as exc:
        store.wrap_cancelled(expect_token="0" * 32)
    assert exc.value.partial_state and "expect_partial=True" in str(exc.value)
    assert "no usable token" not in str(exc.value)
    server = Server(store)
    text = server._tool_wrap_cancel({"wrap_token": "0" * 32})["content"][0]["text"]
    assert "partial=true" in text and "WITHOUT wrap_token" not in text
    both = server._tool_wrap_cancel({"wrap_token": "0" * 32, "partial": True})
    assert both["isError"] and "cannot be combined" in both["content"][0]["text"]
    assert server._tool_wrap_cancel({"partial": True})["isError"] is False  # advice works
    assert not store.status().wrap_in_progress

    # The race: a peer clears the partial state and starts a healthy wrap, here
    # REUSING the token (allow_restart), before the retry. partial=true refuses.
    make_partial(store, "a" * 32)
    peer = Store(db)
    peer.wrap_cancelled(expect_partial=True)
    e2 = peer.record("y" * 90, "observation")
    peer.wrap_started(token="a" * 32, episode_ids=[e2.id])
    late = server._tool_wrap_cancel({"partial": True})
    assert late["isError"] and "not touched" in late["content"][0]["text"]
    with pytest.raises(WrapOwnershipError) as lib:
        store.wrap_cancelled(expect_partial=True)
    assert lib.value.expect_partial and "no longer holds partial" in str(lib.value)
    clone = pickle.loads(pickle.dumps(lib.value))
    assert clone.expect_partial and str(clone) == str(lib.value)
    # L3 r4 codex (run): a token that happens to equal the old marker string is an
    # ordinary mismatch, not a partial-only refusal.
    with pytest.raises(WrapOwnershipError) as odd:
        store.wrap_cancelled(expect_token="(partial state)")
    assert not odd.value.expect_partial
    assert "expect_partial" not in str(odd.value) and "no longer holds" not in str(odd.value)
    assert peer.wrap_cancelled(expect_token="a" * 32).token == "a" * 32  # B survived
    idle = server._tool_wrap_cancel({"partial": True})["content"][0]["text"]
    assert "no wrap is in progress" in idle

    # CLI parity, following its own advice.
    make_partial(store, "a" * 32)
    peer.close()
    store.close()
    env = {**os.environ, "PYTHONPATH": str(Path(anneal_memory.__file__).parent.parent)}
    cli = [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), "wrap-cancel"]
    out = subprocess.run(cli + ["--wrap-token", "0" * 32], capture_output=True, text=True, env=env)
    assert out.returncode == 1 and "--partial" in out.stderr
    out = subprocess.run(
        cli + ["--wrap-token", "0" * 32, "--partial"], capture_output=True, text=True, env=env
    )
    assert out.returncode == 1 and "cannot be combined" in out.stderr
    out = subprocess.run(cli + ["--partial"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    out = subprocess.run(cli + ["--partial"], capture_output=True, text=True, env=env)
    assert out.returncode == 1 and "no longer holds partial" in out.stderr
    after = Store(db)
    assert not after.status().wrap_in_progress
    after.close()


def test_force_with_a_stale_token_says_force_was_ignored(tmp_path):
    """L3 codex (1003+16, run): force=True with a stale token is a deliberate
    operator act, and the refusal said only "the operator's decision"."""
    import copy

    from anneal_memory import WrapOwnershipError

    store = Store(tmp_path / "m.db")
    _gated(store, "a" * 32)
    with pytest.raises(WrapOwnershipError) as exc:
        store.wrap_cancelled(expect_token="0" * 32, force=True)
    assert exc.value.force and "force is ignored" in str(exc.value)
    for clone in (pickle.loads(pickle.dumps(exc.value)), copy.deepcopy(exc.value)):
        assert (clone.force, clone.session_id, str(clone)) == (True, None, str(exc.value))
    text = Server(store)._tool_wrap_cancel({"wrap_token": "0" * 32, "force": True})
    assert "force is ignored" in text["content"][0]["text"]
    assert store.wrap_gated_session() == "holder"  # nothing changed
    assert store.wrap_cancelled(force=True).token == "a" * 32
