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
    assert out["chars"] == 1_001  # the file plus its closing newline
    assert store.load_continuity() == "y" * 1_000 + "\n"
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
