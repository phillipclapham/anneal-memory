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
                  "episodes_since_wrap": 4}
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
