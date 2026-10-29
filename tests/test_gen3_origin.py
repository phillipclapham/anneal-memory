"""Generation 3, design B4: origin bound by the runtime (``runtime_context``),
stored per episode, and kept apart from authority (``trust_ceiling``) and
corroboration (graduation's gate)."""

from __future__ import annotations

import pytest

from anneal_memory import Store
from anneal_memory.server import Server
from anneal_memory.store import _corroborative_trust, _derived_trust
from anneal_memory.types import EpisodeType

OBS = EpisodeType.OBSERVATION


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "o.db"), project_name="T", trust_ceiling="operator")
    yield s
    s.close()


def _origin(store: Store, ep) -> str:
    return store.origin_map([ep.id])[ep.id]


def _stored(store: Store, ep_id: str):
    return store._conn.execute(
        "SELECT origin, rule FROM episode_origin WHERE episode_id = ?", (ep_id,)
    ).fetchone()


def test_t6_a_tainted_runtime_context_caps_what_the_caller_declares(store):
    page = store.record("a fetched page", OBS, trust="external")
    with store.runtime_context(tainted_by=[page.id]):
        inside = store.record("operator says so", OBS, trust="operator")
    outside = store.record("operator says so, later", OBS, trust="operator")
    assert _origin(store, inside) == "external"
    assert _origin(store, outside) == "operator"
    assert _stored(store, inside.id)["rule"] == "origin-conservative-v1"


def test_outside_a_context_the_ceiling_still_caps_the_declared_class(tmp_path):
    s = Store(str(tmp_path / "c.db"), project_name="T")  # ceiling agent
    with pytest.raises(ValueError):
        s.record("x", OBS, trust="operator")
    assert _origin(s, s.record("y", OBS, trust="tool")) == "tool"
    s.close()


def test_floor_caps_an_operator_declared_record(store):
    with store.runtime_context(floor="tool"):
        ep = store.record("declared operator", OBS, trust="operator")
    assert _origin(store, ep) == "tool"


def test_derived_from_can_only_lower(store):
    page = store.record("page", OBS, trust="external")
    with store.runtime_context(floor="agent"):
        ep = store.record("summary", OBS, trust="agent", derived_from=[page.id])
        plain = store.record("plain", OBS, trust="operator")
    assert _origin(store, ep) == "external"
    assert _origin(store, plain) == "agent"  # the floor still caps without sources


def test_nesting_never_raises(store):
    with store.runtime_context(floor="external"):
        with store.runtime_context(floor="operator"):
            ep = store.record("inner", OBS, trust="operator")
    assert _origin(store, ep) == "external"


def test_an_inner_context_can_lower_and_the_outer_resumes(store):
    with store.runtime_context(floor="agent"):
        with store.runtime_context(floor="tool"):
            low = store.record("inner", OBS, trust="operator")
        mid = store.record("after inner", OBS, trust="operator")
    assert (_origin(store, low), _origin(store, mid)) == ("tool", "agent")


def test_exit_restores_even_when_the_block_raises(store):
    with pytest.raises(RuntimeError):
        with store.runtime_context(floor="external"):
            raise RuntimeError("boom")
    assert _origin(store, store.record("after", OBS, trust="operator")) == "operator"


def test_a_context_is_per_store(store, tmp_path):
    other = Store(str(tmp_path / "o.db"), project_name="T", trust_ceiling="operator")
    try:
        with store.runtime_context(floor="external"):
            ep = other.record("via the other store", OBS, trust="operator")
        assert _origin(other, ep) == "operator"
    finally:
        other.close()


def test_a_pre_gen3_episode_reads_by_the_conservative_rule(store):
    page = store.record("page", OBS, trust="external")
    summary = store.record("summary", OBS, trust="agent", derived_from=[page.id])
    deeper = store.record("deeper", OBS, trust="operator", derived_from=[summary.id])
    plain = store.record("plain", OBS, trust="tool")
    for ep in (page, summary, deeper, plain):
        store._conn.execute("DELETE FROM episode_origin WHERE episode_id = ?", (ep.id,))
    store._conn.commit()
    got = store.origin_map([page.id, summary.id, deeper.id, plain.id])
    assert got == {page.id: "external", summary.id: "external",
                   deeper.id: "external", plain.id: "tool"}


def test_a_stored_origin_is_a_fact_that_a_later_trust_change_does_not_move(store):
    ep = store.record("x", OBS, trust="tool")
    store.set_trust(ep.id, "operator")
    assert _origin(store, ep) == "tool"


def test_a_derived_episode_with_a_row_source_reads_the_row(store):
    with store.runtime_context(floor="external"):
        page = store.record("tainted page", OBS, trust="operator")
    summary = store.record("summary", OBS, trust="operator", derived_from=[page.id])
    store._conn.execute("DELETE FROM episode_origin WHERE episode_id = ?", (summary.id,))
    store._conn.commit()
    assert _origin(store, summary) == "external"


def test_origin_row_goes_with_its_episode(store):
    ep = store.record("gone", OBS)
    assert _stored(store, ep.id) is not None
    store.delete(ep.id)
    assert _stored(store, ep.id) is None


def test_unknown_floor_raises_the_trust_error(store):
    with pytest.raises(ValueError, match="unknown trust"):
        with store.runtime_context(floor="root"):
            pass


def test_an_unknown_tainted_id_fails_closed_to_external(store):
    with store.runtime_context(tainted_by=["deadbeef"]):
        ep = store.record("x", OBS, trust="operator")
    assert _origin(store, ep) == "external"


def test_a_bare_string_for_tainted_by_is_refused(store):
    with pytest.raises(TypeError):
        with store.runtime_context(tainted_by="abcd1234"):
            pass


def test_status_and_counts_expose_origin(store):
    page = store.record("page", OBS, trust="external")
    store.record("mine", OBS)
    with store.runtime_context(tainted_by=[page.id]):
        store.record("tainted", OBS, trust="operator")
    assert store.origin_counts() == {"operator": 0, "agent": 1, "tool": 0, "external": 2}
    assert store.status().origin_counts == store.origin_counts()


def test_the_corroborative_rule_keeps_its_old_name_and_behaviour():
    assert _derived_trust is _corroborative_trust
    assert _corroborative_trust("agent", ["external", "tool"]) == "tool"
    assert _corroborative_trust("operator", []) == "operator"


def test_mcp_server_runs_its_tool_calls_under_its_pinned_ceiling(tmp_path):
    s = Store(str(tmp_path / "m.db"), project_name="T", trust_ceiling="agent")
    srv = Server(s)
    seen: dict[str, object] = {}

    def spy(args):
        seen["cap"] = s._origin_cap_in_force()
        seen["ep"] = s.record("from a tool call", OBS, trust="agent")
        return {"content": [{"type": "text", "text": "ok"}]}

    real = srv._tool_handlers["record"]
    srv._tool_handlers["record"] = spy
    srv._handle_tools_call({"name": "record", "arguments": {}})
    srv._tool_handlers["record"] = real
    assert seen["cap"] == "agent"
    assert s.origin_map([seen["ep"].id])[seen["ep"].id] == "agent"  # type: ignore[attr-defined]
    assert s._origin_cap_in_force() is None  # and the context is gone afterwards
    # the real record tool, end to end
    res = srv._handle_tools_call({"name": "record", "arguments": {
        "content": "real tool call", "episode_type": "observation", "trust": "tool"}})
    assert not res.get("isError")
    assert s.origin_counts()["tool"] == 1 and s.origin_counts()["agent"] == 1
    s.close()
