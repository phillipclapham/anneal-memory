"""The 4.0 rolling wrap: the mechanisable part, and nothing more.

The model's side of a wrap (is the memory file any good) is read by a person, not
asserted here. These tests pin what code decides: which episodes make a step, what a
save refuses, that a save answers the step prepare opened, that every version is kept,
that an old store opens, and which MCP tools exist.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from datetime import timezone

import pytest

from anneal_memory import Store, register_writer_schema, rolling
from anneal_memory.continuity import prepare_wrap, validated_save_continuity
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server


def _record_days(store: Store, days: list[str], per_day: int = 2) -> None:
    for d in days:
        for i in range(per_day):
            store.record(
                f"note {i} of {d}", "observation", timestamp=f"{d}T12:0{i}:00.000000Z"
            )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


def test_the_prompt_is_the_measured_prompt():
    """The prompt text is the product: a change to it is a change to the method."""
    system, user = rolling.build_prompt("MEM", rolling.Step("2026-01-02", ["a"], "LOG", 0, 1))
    assert hashlib.sha256(system.encode()).hexdigest() == (
        "789588a94f01aca255055bf4dbc43890191f5b7a87cc7de7c8af051107c3a3f6"
    )
    assert hashlib.sha256(user.encode()).hexdigest() == (
        "af2b9b511e7e091a3506cf33a2f8c26e541f4adb53cfc6a90ce059580433806d"
    )
    assert user.endswith(
        "(The current memory file is 3 characters. Aim for about 20,000 characters; "
        "the hard limit is 26,000.)"
    )


def test_a_catch_up_is_one_step_per_day_never_one_merged_log(store):
    _record_days(store, ["2026-03-01", "2026-03-02", "2026-03-04"])
    seen = []
    for expect_after in (2, 1, 0):
        prep = prepare_wrap(store, tz="UTC")
        step = prep["step"]
        assert prep["status"] == "ready" and step["steps_remaining"] == expect_after
        seen.append(step["day"])
        # only this day's log is in the prompt, never an earlier or later one
        assert step["day"] in step["user"]
        # the log line is as the method was measured: (observation), never EpisodeType.OBSERVATION
        assert ") note 0 of" in step["user"] and "(observation) note 0 of" in step["user"]
        assert "EpisodeType" not in step["user"]
        others = [d for d in ("2026-03-01", "2026-03-02", "2026-03-04") if d != step["day"]]
        assert not any(f"note 0 of {d}" in step["user"] for d in others)
        validated_save_continuity(
            store, f"memory after {step['day']}", wrap_token=prep["wrap_token"]
        )
    assert seen == ["2026-03-01", "2026-03-02", "2026-03-04"]
    # the second step starts from the file the first one saved
    assert prepare_wrap(store, tz="UTC")["status"] == "empty"
    # a day is cut at the zone's midnight, and a long day is sliced
    rows = [("a", "2026-03-01T23:30:00Z", "observation", "x" * 50),
            ("b", "2026-03-02T00:30:00Z", "observation", "y" * 50)]
    assert [s.day for s in rolling.plan_steps(rows, tz=timezone.utc)] == ["2026-03-01", "2026-03-02"]
    assert [s.day for s in rolling.plan_steps(rows, tz=rolling.resolve_tz("Asia/Tokyo"))] == ["2026-03-02"]
    long_day = [(f"e{i}", f"2026-03-01T10:00:{i:02d}Z", "observation", "z" * 100) for i in range(5)]
    sliced = rolling.plan_steps(long_day, tz=timezone.utc, chunk_max=300)
    assert [len(s.episode_ids) for s in sliced] == [2, 2, 1] and sliced[0].chunks == 3


def test_a_save_refuses_empty_and_over_the_limit_and_keeps_the_step_open(store):
    _record_days(store, ["2026-03-01"])
    prep = prepare_wrap(store, tz="UTC", max_chars=1_000)
    token = prep["wrap_token"]
    for bad in ("", "   \n ", "x" * 1_001):
        with pytest.raises(ValueError):
            validated_save_continuity(store, bad, wrap_token=token, max_chars=1_000)
    # nothing was written and the step is still open: the retry lands
    assert store.get_wrap_history() == []
    out = validated_save_continuity(store, "y" * 1_000, wrap_token=token, max_chars=1_000)
    # the stored file is exactly the checked text, so "at most 1,000" is true of it
    assert out["chars"] == 1_000
    assert store.load_continuity() == "y" * 1_000
    with pytest.raises(ValueError, match="at least 1000"):
        prepare_wrap(store, max_chars=10)


def test_a_save_must_answer_the_step_prepare_opened(store):
    _record_days(store, ["2026-03-01", "2026-03-02"])
    with pytest.raises(ValueError, match="No wrap in progress"):
        validated_save_continuity(store, "memory")
    prep = prepare_wrap(store, tz="UTC")
    with pytest.raises(ValueError, match="mismatch"):
        validated_save_continuity(store, "memory", wrap_token="0" * 32)
    validated_save_continuity(store, "memory one", wrap_token=prep["wrap_token"])
    # the token is spent: the same reply cannot be saved twice
    with pytest.raises(ValueError, match="No wrap in progress"):
        validated_save_continuity(store, "memory one", wrap_token=prep["wrap_token"])
    # only the step's own episodes were wrapped; the next day is still owed
    assert store.count_episodes_since_wrap() == 2


def test_every_version_is_kept_and_a_dropped_fact_is_recoverable(store):
    _record_days(store, ["2026-03-01", "2026-03-02"])
    texts = ["fact A and fact B", "fact B only"]
    for text in texts:
        prep = prepare_wrap(store, tz="UTC")
        validated_save_continuity(store, text, wrap_token=prep["wrap_token"])
    rows = sqlite3.connect(store.path).execute(
        "SELECT text FROM identity_versions WHERE source = 'wrap' ORDER BY version"
    ).fetchall()
    assert [r[0].strip() for r in rows] == texts
    assert len(store.get_wrap_history()) == 2
    assert store.load_continuity().strip() == "fact B only"


def test_an_old_store_opens_and_its_file_seeds_the_first_rolling_wrap(tmp_path):
    db = tmp_path / "old.db"
    old = Store(db, project_name="T")
    old.record("before the upgrade", "observation", timestamp="2026-03-01T12:00:00.000000Z")
    old.save_continuity("# T — Memory (v1)\n\n## Patterns\nsome_pattern | 3x\n")
    old.close()
    conn = sqlite3.connect(db)
    register_writer_schema(conn)
    conn.execute(
        "INSERT INTO pattern_history (pattern_name, max_level_reached, last_seen_at) "
        "VALUES ('some_pattern', 3, '2026-03-01')"
    )
    conn.commit()
    conn.close()
    store = Store(db, project_name="T")
    try:
        prep = prepare_wrap(store, tz="UTC")
        assert "some_pattern | 3x" in prep["step"]["user"]  # the old file is the seed
        validated_save_continuity(store, "rolled", wrap_token=prep["wrap_token"])
    finally:
        store.close()
    # no table was dropped and no row lost
    kept = sqlite3.connect(db).execute("SELECT pattern_name FROM pattern_history").fetchall()
    assert kept == [("some_pattern",)]


def test_the_mcp_tools_are_capture_recall_read_wrap_and_prospective_memory(store):
    names = {t["name"] for t in TOOLS}
    assert {"record", "recall", "read_continuity", "prepare_wrap", "save_continuity",
            "wrap_cancel", "status"} <= names
    assert not {n for n in names if n.startswith("crystal")}
    server = Server(store)
    listed = {t["name"] for t in server._handle_tools_list({})["tools"]}
    assert listed == names
    _record_days(store, ["2026-03-01"])
    prep = server._handle_tools_call({"name": "prepare_wrap", "arguments": {"tz": "UTC"}})
    text = prep["content"][0]["text"]
    token = text.split("Wrap token: ")[1].split()[0]
    assert "=== SYSTEM ===" in text and not prep.get("isError")
    saved = server._handle_tools_call(
        {"name": "save_continuity", "arguments": {"text": "remembered", "wrap_token": token}})
    assert not saved.get("isError")
    read = server._handle_tools_call({"name": "read_continuity", "arguments": {}})
    assert "remembered" in read["content"][0]["text"]
    json.dumps(TOOLS)  # the manifest stays serialisable


def test_a_wrap_left_open_by_3x_with_its_old_keys_cancels_and_a_new_prepare_opens(store):
    """3.x wrote ``wrap_section_schema`` and ``wrap_derive_roots`` with an open wrap; 4.0
    never writes them, but a cancel must still clear them so the wrap does not stick."""
    _record_days(store, ["2026-03-01"])
    assert prepare_wrap(store, tz="UTC")["status"] == "ready"
    for key, value in (("wrap_section_schema", '[{"heading": "State", "role": "live-state"}]'),
                       ("wrap_derive_roots", '{"a": "b"}')):
        store._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
    store._conn.commit()
    store.wrap_cancelled()
    for key in ("wrap_section_schema", "wrap_derive_roots", "wrap_started_at", "wrap_token"):
        assert store._get_metadata(key) in ("", None), key
    assert prepare_wrap(store, tz="UTC")["status"] == "ready"


def _cli(db, *args):
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *args],
        capture_output=True, text=True,
    )


def test_a_memory_file_that_mentions_the_placeholders_is_not_rewritten():
    """The prompt is filled in one pass: a memory file that contains the text
    ``{day}`` or ``{episodes}`` reaches the model as written."""
    memory = "The template uses {episodes} and {day} and {current_memory} placeholders."
    system, user = rolling.build_prompt(memory, rolling.Step("2026-01-02", ["a"], "LOG LINE", 0, 1))
    assert memory in user
    assert user.count("LOG LINE") == 1 and "## WORK LOG FOR 2026-01-02:\nLOG LINE\n" in user


def test_a_save_needs_the_steps_token_on_every_path(store, tmp_path):
    _record_days(store, ["2026-03-01"])
    prep = prepare_wrap(store, tz="UTC")
    with pytest.raises(ValueError, match="pass the wrap_token that prepare_wrap returned"):
        validated_save_continuity(store, "a stale reply")
    assert store.status().wrap_in_progress and store.get_wrap_history() == []
    # MCP: required in the schema, and refused when missing
    save_tool = next(t for t in TOOLS if t["name"] == "save_continuity")
    assert "wrap_token" in save_tool["inputSchema"]["required"]
    refused = Server(store)._handle_tools_call(
        {"name": "save_continuity", "arguments": {"text": "a stale reply"}})
    assert refused.get("isError") and "wrap_token" in refused["content"][0]["text"]
    assert store.status().wrap_in_progress
    # CLI: refused before the store is opened
    reply = tmp_path / "reply.md"
    reply.write_text("a stale reply", encoding="utf-8")
    r = _cli(store.path, "save-continuity", str(reply))
    assert r.returncode == 1 and "--wrap-token is required" in r.stderr and "Traceback" not in r.stderr
    assert store.status().wrap_in_progress
    validated_save_continuity(store, "the real reply", wrap_token=prep["wrap_token"])


def test_the_cap_is_bound_to_the_step(store):
    _record_days(store, ["2026-03-01", "2026-03-02", "2026-03-03"])
    prep = prepare_wrap(store, tz="UTC", max_chars=1_000)
    token = prep["wrap_token"]
    assert store.wrap_max_chars() == 1_000
    # a bare save enforces the cap the prompt stated, not the default
    with pytest.raises(ValueError, match="over the 1,000 limit"):
        validated_save_continuity(store, "z" * 1_500, wrap_token=token)
    # an explicit cap must equal the step's, and the refusal names both
    with pytest.raises(ValueError, match=r"1,500.*1,000"):
        validated_save_continuity(store, "z" * 10, wrap_token=token, max_chars=1_500)
    validated_save_continuity(store, "z" * 1_000, wrap_token=token, max_chars=1_000)
    assert store.wrap_max_chars() is None  # cleared with the wrap
    # a higher cap on prepare is honoured by the bare save
    prep = prepare_wrap(store, tz="UTC", max_chars=40_000)
    out = validated_save_continuity(store, "y" * 30_000, wrap_token=prep["wrap_token"])
    assert out["chars"] == 30_000
    # the next prompt's size line is the stored file's size, exactly
    nxt = prepare_wrap(store, tz="UTC")
    assert "The current memory file is 30,000 characters" in nxt["step"]["user"]
    # a wrap opened before the key existed falls back to the default
    store._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('wrap_max_chars', '')")
    store._conn.commit()
    assert store.wrap_max_chars() is None
    with pytest.raises(ValueError, match="over the 26,000 limit"):
        validated_save_continuity(store, "w" * 26_001, wrap_token=nxt["wrap_token"])
    validated_save_continuity(store, "w" * 26_000, wrap_token=nxt["wrap_token"])


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs time.tzset (POSIX)")
def test_the_default_zone_is_asked_per_instant_so_a_dst_change_is_honoured(monkeypatch):
    monkeypatch.setenv("TZ", "EST5EDT,M3.2.0,M11.1.0")  # a POSIX rule: no zone database needed
    time.tzset()
    try:
        zone = rolling.resolve_tz(None)
        assert zone is None  # the machine's zone, never one offset fixed at call time
        # 04:30Z is 00:30 in summer (EDT) and 23:30 the day before in winter (EST)
        assert rolling.episode_day("2026-07-01T04:30:00Z", zone) == "2026-07-01"
        assert rolling.episode_day("2026-01-01T04:30:00Z", zone) == "2025-12-31"
        rows = [("a", "2026-07-01T04:30:00Z", "observation", "x"),
                ("b", "2026-01-01T04:30:00Z", "observation", "y")]
        assert [st.day for st in rolling.plan_steps(rows, tz=zone)] == ["2025-12-31", "2026-07-01"]
    finally:
        monkeypatch.undo()
        time.tzset()


def test_an_unreadable_timestamp_never_blocks_a_wrap(store):
    rows = [
        ("a", "yesterday", "observation", "no date at all"),
        ("b", "2026-03-01 garbage", "observation", "a date prefix"),
        ("c", "2026-03-01T10:00:00+05:00", "observation", "05:00Z"),
        ("d", "2026-03-01T06:00:00Z", "observation", "06:00Z"),
    ]
    steps = {st.day: st for st in rolling.plan_steps(rows, tz=timezone.utc)}
    assert set(steps) == {"0000-00-00", "2026-03-01"}
    assert steps["0000-00-00"].episode_ids == ["a"] and steps["0000-00-00"].unreadable == 1
    # within a day: by the parsed instant (05:00Z before 06:00Z, though the text says otherwise),
    # and the line whose time cannot be read goes last
    assert steps["2026-03-01"].episode_ids == ["c", "d", "b"]
    assert steps["2026-03-01"].unreadable == 1
    # through prepare: it opens, and the message names the count
    store.record("odd one", "observation", timestamp="yesterday")
    prep = prepare_wrap(store, tz="UTC")
    assert prep["status"] == "ready" and prep["step"]["day"] == "0000-00-00"
    assert "1 episode(s) have a timestamp that could not be read" in prep["message"]
    validated_save_continuity(store, "kept", wrap_token=prep["wrap_token"])


def test_utc_needs_no_zone_database_and_a_missing_zone_names_tzdata():
    assert rolling.resolve_tz("UTC") is timezone.utc and rolling.resolve_tz("utc") is timezone.utc
    with pytest.raises(ValueError, match="tzdata"):
        rolling.resolve_tz("Not/AZone")


def test_diff_shows_a_wrap_against_the_version_just_before_it(tmp_path):
    db = tmp_path / "d.db"
    s = Store(db, project_name="T")
    s.save_continuity("seed line one\nseed line two\n")  # the file a 3.x store holds
    s.record("work", "observation", timestamp="2026-03-01T12:00:00.000000Z")
    prep = prepare_wrap(s, tz="UTC")
    validated_save_continuity(s, "new memory", wrap_token=prep["wrap_token"])
    assert s.wrap_identity_before_texts([1]) == {1: "seed line one\nseed line two\n"}
    s.close()
    out = _cli(db, "diff", "--json")
    assert out.returncode == 0, out.stderr
    wrap = json.loads(out.stdout)["wraps"][-1]
    assert (wrap["lines_added"], wrap["lines_removed"]) == (1, 2)
    assert "-seed line one" in wrap["text_diff"] and "+new memory" in wrap["text_diff"]
    zero = _cli(db, "diff", "--wraps", "0")
    assert zero.returncode == 1 and "at least 1" in zero.stderr


def test_a_fenced_save_or_cancel_says_what_to_do_and_prints_no_traceback(store, tmp_path):
    _record_days(store, ["2026-03-01"])
    prep = prepare_wrap(store, tz="UTC")
    token = prep["wrap_token"]
    store.save_continuity("a hand edit while the step ran")  # another identity version
    with pytest.raises(Exception) as fenced:
        validated_save_continuity(store, "the reply", wrap_token=token)
    assert type(fenced.value).__name__ == "LeaseFencedError"
    reply = tmp_path / "reply.md"
    reply.write_text("the reply", encoding="utf-8")
    r = _cli(store.path, "save-continuity", str(reply), "--wrap-token", token)
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert "Nothing was written" in r.stderr and f"wrap-cancel --wrap-token {token}" in r.stderr
    mcp = Server(store)._tool_save_continuity({"text": "the reply", "wrap_token": token})
    assert mcp.get("isError") and f"wrap_cancel with wrap_token {token}" in mcp["content"][0]["text"]
    # the advice works: cancel, then prepare again
    assert _cli(store.path, "wrap-cancel", "--wrap-token", token).returncode == 0
    assert prepare_wrap(store, tz="UTC")["status"] == "ready"


def test_a_cancel_under_a_live_lease_is_a_clean_refusal_not_a_traceback(store):
    _record_days(store, ["2026-03-01"])
    grant = store.lease_acquire("holder", 300)
    prepare_wrap(store, tz="UTC", lease_token=grant.token)
    r = _cli(store.path, "wrap-cancel")
    assert r.returncode == 1 and "Traceback" not in r.stderr
    assert "lease" in r.stderr and "Nothing was changed" in r.stderr
    mcp = Server(store)._tool_wrap_cancel({})
    assert mcp.get("isError") and "lease" in mcp["content"][0]["text"]
    store.wrap_cancelled(lease_token=grant.token)  # the library path with the token works


def test_the_declared_result_keys_exist_in_a_real_result(store):
    from anneal_memory.types import PrepareWrapResult, SaveContinuityResult, WrapStepDict

    _record_days(store, ["2026-03-01"])
    prep = prepare_wrap(store, tz="UTC")
    assert set(prep) == set(PrepareWrapResult.__annotations__)
    assert set(prep["step"]) == set(WrapStepDict.__annotations__)
    saved = validated_save_continuity(store, "kept", wrap_token=prep["wrap_token"])
    assert set(saved) == set(SaveContinuityResult.__annotations__)


def test_a_batch_holds_the_write_lock_from_its_first_statement(store):
    """``validated_save_continuity`` reads the idempotency key inside ``_batch``; that is
    safe only because the batch opens ``BEGIN IMMEDIATE``, not at its first write."""
    with store._batch():
        other = sqlite3.connect(store.path, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        finally:
            other.close()


def test_the_prepare_text_puts_the_operator_note_before_the_system_marker(store):
    _record_days(store, ["2026-03-01"])
    text = Server(store)._tool_prepare_wrap({"tz": "UTC"})["content"][0]["text"]
    note, marker, prompt = text.partition("=== SYSTEM ===")
    assert marker and "Wrap token:" in note and "run only that" in note
    assert "Wrap token:" not in prompt and "run only that" not in prompt


# -- L3 r1 fixes -----------------------------------------------------------------------


def test_a_keyed_save_replay_carries_the_pending_count_it_has_now(store):
    _record_days(store, ["2026-03-01", "2026-03-02"], per_day=1)
    prep = prepare_wrap(store, tz="UTC")
    first = validated_save_continuity(
        store, "day one", wrap_token=prep["wrap_token"], idempotency_key="save-day-1")
    assert first["pending_count"] == 1
    again = validated_save_continuity(
        store, "day one", wrap_token=prep["wrap_token"], idempotency_key="save-day-1")
    assert again["replayed"] is True and again["pending_count"] == 1


def test_a_replayed_prepare_meets_the_baton_gate_again(store):
    from anneal_memory import sessions
    _record_days(store, ["2026-03-01"])
    path = store.continuity_path
    sessions.register_session(path, "s1")
    sessions.claim_baton(path, "s1")
    first = prepare_wrap(store, tz="UTC", session_id="s1", idempotency_key="prep-1")
    assert first["status"] == "ready"
    sessions.register_session(path, "s2")
    sessions.claim_baton(path, "s2", take=True)
    again = prepare_wrap(store, tz="UTC", session_id="s1", idempotency_key="prep-1")
    assert again["status"] == "downgraded" and again["step"] is None
    assert again["wrap_token"] is None


def test_tz_must_be_a_name_or_a_tzinfo_and_one_zone_is_one_request(store):
    with pytest.raises(TypeError, match="tz"):
        rolling.resolve_tz(5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tz"):
        prepare_wrap(store, tz=5)  # type: ignore[arg-type]
    _record_days(store, ["2026-03-01"])
    first = prepare_wrap(store, tz="utc", idempotency_key="prep-tz")
    again = prepare_wrap(store, tz="UTC", idempotency_key="prep-tz")
    assert again["wrap_token"] == first["wrap_token"]


def test_a_replayed_prepare_refuses_a_step_whose_episode_is_gone(store):
    from anneal_memory.store import IdempotencyConflictError
    _record_days(store, ["2026-03-01"])
    first = prepare_wrap(store, tz="UTC", idempotency_key="prep-gone")
    assert first["status"] == "ready"
    frozen = store.load_wrap_snapshot()["episode_ids"]
    assert store.delete(frozen[0]) is True
    with pytest.raises(IdempotencyConflictError, match="episodes changed") as err:
        prepare_wrap(store, tz="UTC", idempotency_key="prep-gone")
    assert err.value.reason == "sources_moved"
