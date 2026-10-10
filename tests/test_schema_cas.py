"""The section schema prepare_wrap reads cannot be replaced before wrap_started
freezes it (the frozen-schema race, reproduced 2026-10-04 on 0.9.28)."""

from __future__ import annotations

import warnings

import pytest

from anneal_memory import Store, WrapSchemaMovedError, prepare_wrap
from anneal_memory.schema import name_for_schema, schema_by_name


@pytest.fixture
def two(tmp_path):
    a = Store(tmp_path / "m.db", project_name="T")
    a.set_section_schema(schema_by_name("partnership"))
    a.record("an episode", "observation")
    b = Store(tmp_path / "m.db", project_name="T")
    yield a, b
    a.close()
    b.close()


def test_a_schema_change_between_read_and_wrap_started_opens_no_wrap(two):
    a, b = two
    orig = a.wrap_started

    def interleaved(*args, **kw):
        b.set_section_schema(schema_by_name("project"))
        return orig(*args, **kw)

    a.wrap_started = interleaved
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = prepare_wrap(a)
    assert result["status"] != "ready"
    assert "downgraded-schema-changed" in result["message"]
    assert a.get_wrap_started_at() is None
    assert name_for_schema(a.section_schema) == "project"


def test_wrap_started_refuses_a_passed_schema_that_is_not_live(two):
    a, b = two
    read = a.section_schema_for_wrap()
    b.set_section_schema(schema_by_name("project"))
    with pytest.raises(WrapSchemaMovedError):
        a.wrap_started(token="t" * 32, episode_ids=[], section_schema=read)
    assert a.get_wrap_started_at() is None


def test_set_section_schema_refuses_under_the_lock_when_the_fast_check_missed(two, monkeypatch):
    a, b = two
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert prepare_wrap(a)["status"] == "ready"
    monkeypatch.setattr(b, "get_wrap_started_at", lambda: None)  # a stale fast read
    with pytest.raises(ValueError, match="wrap is in progress"):
        b.set_section_schema(schema_by_name("project"))
    assert name_for_schema(b.section_schema) == "partnership"


def test_an_omitted_schema_is_read_under_the_lock(two, monkeypatch):
    # A direct caller that passes no schema freezes whatever is live once the write lock
    # is held, so a change committed just before the lock is the one frozen.
    a, b = two
    real_boundary = a._db_boundary

    def boundary(op):
        if op == "wrap_started":
            b.set_section_schema(schema_by_name("project"))
        return real_boundary(op)

    monkeypatch.setattr(a, "_db_boundary", boundary)
    a.wrap_started(token="t" * 32, episode_ids=[])
    assert name_for_schema(a.section_schema_for_wrap()) == "project"


def test_the_refusal_survives_pickle_and_copy():
    import copy
    import pickle

    err = WrapSchemaMovedError()
    assert str(pickle.loads(pickle.dumps(err))) == str(err)
    assert str(copy.deepcopy(err)) == str(err)
