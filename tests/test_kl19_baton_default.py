"""KL-19: consolidation is the operator's act by default (Phill 10-12, ruling A).

A store `init` creates requires the consolidate baton. Agents record freely; a wrap
consolidates only under a session id the operator claimed the baton for, and the
downgrade and `status` say how many episodes wait and the one command that claims.
The MCP server's session id comes from its launch config, never from a tool argument.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from anneal_memory import EpisodeType, Store, sessions
from anneal_memory.server import Server

_WRAP_TEXT = (
    "# T — Memory (v1)\n\n"
    "## State\nActive.\n\n"
    "## Patterns\nNone yet.\n\n"
    "## Decisions\nNone.\n\n"
    "## Context\nFirst session.\n"
)


def _cli(db, *argv):
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *argv],
        capture_output=True, text=True,
    )


def _text(result):
    return result["content"][0]["text"]


@pytest.fixture
def init_db(tmp_path):
    db = tmp_path / "memory.db"
    run = _cli(db, "init")
    assert run.returncode == 0, run.stderr
    return db


def _record(db, n=2):
    s = Store(str(db))
    try:
        for i in range(n):
            s.record(f"obs {i}", EpisodeType.OBSERVATION)
    finally:
        s.close()


def test_init_turns_the_policy_on_and_no_baton_leaves_it_off(tmp_path, init_db):
    s = Store(str(init_db))
    try:
        assert s.consolidate_requires_baton() is True
    finally:
        s.close()
    open_db = tmp_path / "open.db"
    run = _cli(open_db, "init", "--no-baton", "--json")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["consolidate_requires_baton"] is False
    s = Store(str(open_db))
    try:
        assert s.consolidate_requires_baton() is False
    finally:
        s.close()


def test_a_library_created_store_is_unchanged(tmp_path):
    s = Store(str(tmp_path / "lib.db"))
    try:
        assert s.consolidate_requires_baton() is False
    finally:
        s.close()


def test_init_created_store_downgrades_an_unclaimed_mcp_wrap(init_db):
    _record(init_db, 3)
    store = Store(str(init_db))
    try:
        out = _text(Server(store)._tool_prepare_wrap({}))
        assert "downgraded-baton-required" in out
        assert "3 episode(s) since the last wrap" in out
        assert "baton claim NAME" in out
        assert "Wrap token:" not in out
        assert store.status().wrap_in_progress is False
        status = _text(Server(store)._tool_status({}))
        assert "3 episode(s) since the last wrap" in status
        assert "--session-id NAME" in status
    finally:
        store.close()


def test_mcp_wrap_consolidates_only_after_the_operator_claims_for_its_id(init_db):
    _record(init_db)
    store = Store(str(init_db))
    try:
        desk = Server(store, session_id="desk")
        before = _text(desk._tool_prepare_wrap({}))
        assert "downgraded-no-baton" in before
        assert "baton claim desk" in before
        assert store.status().wrap_in_progress is False

        claim = _cli(init_db, "baton", "claim", "desk")
        assert claim.returncode == 0, claim.stderr

        other = _text(Server(store, session_id="other")._tool_prepare_wrap({}))
        # An MCP server never registers as a live session (close_session would drop the
        # operator's standing claim on exit), so the holder reads as stale here; the
        # refusal is what matters.
        assert "Consolidate downgraded" in other and "baton holder = desk" in other
        assert store.status().wrap_in_progress is False

        ready = _text(desk._tool_prepare_wrap({}))
        token = ready.rsplit("Wrap token: ", 1)[1].strip()
        saved = desk._tool_save_continuity({"text": _WRAP_TEXT, "wrap_token": token})
        assert not saved.get("isError"), _text(saved)
        assert len(store.get_wrap_history()) == 1
        assert "holds the baton and may consolidate" in _text(desk._tool_status({}))
    finally:
        store.close()


def test_mcp_session_id_is_not_a_tool_argument(init_db):
    """An agent cannot choose who it wraps as: a session_id tool argument is ignored."""
    _record(init_db)
    assert _cli(init_db, "baton", "claim", "desk").returncode == 0
    store = Store(str(init_db))
    try:
        out = _text(Server(store)._tool_prepare_wrap({"session_id": "desk"}))
        assert "downgraded-baton-required" in out
    finally:
        store.close()


def test_server_reads_the_session_id_from_the_environment(init_db, monkeypatch):
    from anneal_memory import server as server_mod

    seen = {}

    class _Spy(server_mod.Server):
        def run(self):
            seen["sid"] = self._session_id

    monkeypatch.setattr(server_mod, "Server", _Spy)
    monkeypatch.setattr(sys.stdin, "reconfigure", lambda **k: None, raising=False)
    monkeypatch.setattr(sys.stdout, "reconfigure", lambda **k: None, raising=False)
    monkeypatch.setenv("ANNEAL_MEMORY_SESSION_ID", "from-env")
    server_mod.start_server(db_path=str(init_db), skip_integrity=True)
    assert seen["sid"] == "from-env"
    server_mod.start_server(db_path=str(init_db), skip_integrity=True, session_id="flag")
    assert seen["sid"] == "flag"


def test_cli_round_trip_under_a_claimed_session(init_db, tmp_path):
    _record(init_db)
    unclaimed = _cli(init_db, "prepare-wrap", "--json", "--session-id", "me")
    assert unclaimed.returncode == 3
    assert "baton claim me" in json.loads(unclaimed.stdout)["message"]

    assert _cli(init_db, "baton", "claim", "me").returncode == 0
    prep = _cli(init_db, "prepare-wrap", "--json", "--session-id", "me")
    assert prep.returncode == 0, prep.stderr
    token = json.loads(prep.stdout)["wrap_token"]
    out = tmp_path / "wrap.md"
    out.write_text(_WRAP_TEXT)
    no_session = _cli(init_db, "save-continuity", str(out), "--wrap-token", token)
    assert no_session.returncode == 1 and "baton-protected" in no_session.stderr
    saved = _cli(init_db, "save-continuity", str(out), "--wrap-token", token,
                 "--session-id", "me")
    assert saved.returncode == 0, saved.stderr


def test_baton_cli_claim_take_status_release(init_db):
    _record(init_db, 4)
    st = json.loads(_cli(init_db, "baton", "status", "--json").stdout)
    assert st == {"consolidate_requires_baton": True, "baton_holder": None,
                  "baton_readable": True, "episodes_since_wrap": 4}
    text_status = _cli(init_db, "status").stdout
    assert "no holder: 4 episode(s) since the last wrap" in text_status

    assert _cli(init_db, "baton", "claim", "a").returncode == 0
    held = _cli(init_db, "baton", "claim", "b")
    assert held.returncode == 1 and "--take" in held.stderr
    moved = _cli(init_db, "baton", "claim", "b", "--take")
    assert moved.returncode == 0 and "from 'a' to 'b'" in moved.stdout
    assert json.loads(_cli(init_db, "status", "--json").stdout)["baton_holder"] == "b"

    assert _cli(init_db, "baton", "release", "a").returncode == 1  # not the holder
    assert _cli(init_db, "baton", "release", "b").returncode == 0
    s = Store(str(init_db))
    try:
        assert sessions.baton_holder(s.continuity_path) is None
    finally:
        s.close()


def test_migration_entry_tells_existing_adopters_the_agent_never_claims():
    from anneal_memory.migration import MIGRATION_MANIFEST, _version_tuple
    from anneal_memory import __version__

    [entry] = [e for e in MIGRATION_MANIFEST if e["feature"] == "KL-19-OPERATOR-CONSOLIDATES"]
    durable = next(i for i, e in enumerate(MIGRATION_MANIFEST) if e["feature"] == "AM-DURABLE-FACTS")
    assert MIGRATION_MANIFEST.index(entry) > durable  # appended, never reordered
    assert _version_tuple(entry["version"]) <= _version_tuple(__version__)
    assert "baton claim NAME" in entry["summary"]
    assert "never claim the baton yourself" in entry["suggested_edit"]


def _start(monkeypatch, db, **kw):
    from anneal_memory import server as server_mod

    class _NoRun(server_mod.Server):
        def run(self):
            pass

    monkeypatch.setattr(server_mod, "Server", _NoRun)
    monkeypatch.setattr(sys.stdin, "reconfigure", lambda **k: None, raising=False)
    monkeypatch.setattr(sys.stdout, "reconfigure", lambda **k: None, raising=False)
    server_mod.start_server(db_path=str(db), skip_integrity=True, **kw)


def test_a_store_the_server_creates_is_baton_protected(tmp_path, monkeypatch):
    db = tmp_path / "fresh.db"
    _start(monkeypatch, db)
    s = Store(str(db))
    try:
        assert s.consolidate_requires_baton() is True
    finally:
        s.close()


def test_the_server_never_changes_an_existing_stores_policy(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    s = Store(str(db))
    s.record("before", EpisodeType.OBSERVATION)
    s.close()
    _start(monkeypatch, db)
    s = Store(str(db))
    try:
        assert s.consolidate_requires_baton() is False
    finally:
        s.close()
    open_db = tmp_path / "open.db"
    assert _cli(open_db, "init", "--no-baton").returncode == 0
    _start(monkeypatch, open_db)
    s = Store(str(open_db))
    try:
        assert s.consolidate_requires_baton() is False
    finally:
        s.close()


# -- L1/L2 round 1 fixes --

def test_an_existing_open_store_is_not_gated_by_a_configured_session_id(tmp_path):
    """L2 HIGH: an open store whose config gains --session-id keeps wrapping."""
    db = tmp_path / "open.db"
    s = Store(str(db))
    s.record("before", EpisodeType.OBSERVATION)
    try:
        out = _text(Server(s, session_id="claude-code")._tool_prepare_wrap({}))
        assert "Wrap token:" in out, out
        s.wrap_cancelled()
    finally:
        s.close()
    run = _cli(db, "prepare-wrap", "--json", "--session-id", "me")
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["wrap_token"]


def test_mcp_wrap_cancel_acts_as_the_server_and_refuses_another_name(init_db):
    _record(init_db)
    assert _cli(init_db, "baton", "claim", "desk").returncode == 0
    store = Store(str(init_db))
    try:
        desk = Server(store, session_id="desk")
        assert "Wrap token:" in _text(desk._tool_prepare_wrap({}))
        borrowed = desk._tool_wrap_cancel({"session_id": "someone-else"})
        assert borrowed.get("isError") and "this server's own" in _text(borrowed)
        assert store.status().wrap_in_progress is True
        own = desk._tool_wrap_cancel({})  # no argument: cancels as 'desk'
        assert not own.get("isError"), _text(own)
        assert store.status().wrap_in_progress is False
    finally:
        store.close()


def test_wrap_status_complete_line_names_the_session(init_db):
    _record(init_db)
    assert _cli(init_db, "baton", "claim", "me").returncode == 0
    assert _cli(init_db, "prepare-wrap", "--session-id", "me").returncode == 0
    out = _cli(init_db, "wrap-status").stdout
    assert "save-continuity --session-id me" in out


def test_an_unreadable_baton_is_reported_as_unreadable_not_as_a_holder(init_db):
    s = Store(str(init_db))
    try:
        sessions._baton_path(s.continuity_path).write_text("[]")
    finally:
        s.close()
    text = _cli(init_db, "status").stdout
    assert "UNREADABLE" in text and "only session" not in text
    st = json.loads(_cli(init_db, "status", "--json").stdout)
    assert st["baton_holder"] is None and st["baton_readable"] is False
    s = Store(str(init_db))
    try:
        mcp = _text(Server(s, session_id="desk")._tool_status({}))
        assert "unreadable" in mcp and "--take" in mcp
    finally:
        s.close()


def test_an_empty_session_id_is_an_argument_error(init_db):
    for argv in (("baton", "claim", ""), ("prepare-wrap", "--session-id", ""),
                 ("save-continuity", "-", "--session-id", "")):
        run = _cli(init_db, *argv)
        assert run.returncode == 2 and "Traceback" not in run.stderr, argv


def test_the_recipe_is_for_the_human_and_takes_only_from_another_holder(init_db):
    _record(init_db)
    unheld = _cli(init_db, "prepare-wrap", "--json", "--session-id", "me")
    msg = json.loads(unheld.stdout)["message"]
    assert "do not run it yourself" in msg and "baton claim me" in msg and "--take" not in msg
    assert _cli(init_db, "baton", "claim", "other").returncode == 0
    held = json.loads(_cli(init_db, "prepare-wrap", "--json", "--session-id", "me").stdout)
    assert "baton claim me --take" in held["message"]
    s = Store(str(init_db))
    try:
        mcp = _text(Server(s, session_id="me")._tool_status({}))
        assert "(another session holds the baton)" in mcp
        assert "'other'" not in mcp and "holder: other" not in mcp
    finally:
        s.close()


def test_an_empty_store_says_nothing_is_waiting(init_db):
    s = Store(str(init_db))
    try:
        out = _text(Server(s)._tool_prepare_wrap({}))
        assert "Nothing is waiting" in out and "baton claim" not in out
    finally:
        s.close()
