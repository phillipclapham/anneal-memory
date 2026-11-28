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






def test_set_section_schema_refuses_under_the_lock_when_the_fast_check_missed(two, monkeypatch):
    a, b = two
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert prepare_wrap(a)["status"] == "ready"
    monkeypatch.setattr(b, "get_wrap_started_at", lambda: None)  # a stale fast read
    with pytest.raises(ValueError, match="wrap is in progress"):
        b.set_section_schema(schema_by_name("project"))
    assert name_for_schema(b.section_schema) == "partnership"




def test_the_refusal_survives_pickle_and_copy():
    import copy
    import pickle

    err = WrapSchemaMovedError()
    assert str(pickle.loads(pickle.dumps(err))) == str(err)
    assert str(copy.deepcopy(err)) == str(err)
