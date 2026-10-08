"""CAP-04: state keys + the recall redirect (project_memory/cap04_current_state_design_1007.md).

The measured shape (STALE n=100, the "M0" run): the query names the OLD state ("still in
Seattle?") and the update shares none of its words. Today the old fact is served and a
link only hides it; the redirect serves the replacement in the old hit's slot.
"""

from __future__ import annotations

import json

import pytest

from anneal_memory import Store, SupersessionError
from anneal_memory.retrieval import retrieve_relevant
from anneal_memory.store import STATE_KEY_MAX_LEN, normalize_state_key

OLD = ("I've been based in Seattle for the last few years, near the waterfront, "
       "a short walk from the ferry terminal.")
NEW = "Finally settled into my new place in Austin and set up the utilities here."
QUESTION = "does the user still live in Seattle"
FILLER = [
    "looking for healthy meal prep ideas with more plant-based dishes",
    "the vegan cooking class last weekend was really inspiring",
    "trying to pick a laptop for photo editing on a budget",
    "planning a short hiking trip with a couple of friends",
]


def _seed(st: Store) -> None:
    for i in range(40):
        st.record(f"{FILLER[i % 4]} (note {i})", "observation",
                  timestamp=f"2026-01-{1 + i % 28:02d}T09:{i:02d}:00Z")


def _recall(st: Store, query: str = QUESTION, **kw):
    return retrieve_relevant(st, None, query, max_patterns=0, associative=False,
                             mode=kw.pop("mode", "query"), **kw)


# -- normalisation ----------------------------------------------------------------

def test_a_key_is_whitespace_collapsed_and_case_folded():
    assert normalize_state_key("  User.Home_City ") == "user.home_city"
    assert normalize_state_key("user   home\tcity") == "user home city"


@pytest.mark.parametrize("bad", ["", "   ", "x" * (STATE_KEY_MAX_LEN + 1), "a\x00b", "a‎b", 7, None])
def test_an_invalid_key_is_refused(bad):
    with pytest.raises(ValueError):
        normalize_state_key(bad)


def test_record_refuses_an_invalid_key_and_writes_nothing(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        with pytest.raises(ValueError):
            st.record("a fact", "observation", state_key="")
        assert st.recall(limit=0).total_matching == 0


# -- linking ----------------------------------------------------------------------

def test_a_newer_fact_in_the_same_slot_replaces_the_older(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="User.Home_City")
        shown = {e.id: e.superseded_by for e in
                 st.recall(limit=10, include_superseded=True).episodes}
        assert shown == {new.id: None, old.id: new.id}
        assert [e.id for e in st.recall(limit=10).episodes] == [new.id]


def test_other_slots_and_unkeyed_episodes_are_untouched(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                      state_key="user.home_city")
        b = st.record("Works as a pastry chef.", "observation",
                      timestamp="2026-01-06T10:00:00Z", state_key="user.job")
        c = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        assert {e.id for e in st.recall(limit=10).episodes} == {a.id, b.id, c.id}


def test_a_backdated_fact_is_history_under_the_newest_holder(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        cur = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="user.home_city")
        back = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                         state_key="user.home_city")
        assert [e.id for e in st.recall(limit=10).episodes] == [cur.id]
        report = st.state_key_report()
        assert [i["id"] for i in report[0]["current"]] == [cur.id]
        assert [i["id"] for i in report[0]["replaced"]] == [back.id]


def test_three_holders_form_a_chain_and_the_report_shows_it(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        ids = [st.record(f"lives in city {n}", "observation",
                         timestamp=f"2026-0{n}-01T10:00:00Z", state_key="user.home_city").id
               for n in (1, 2, 3)]
        report = st.state_key_report("USER.home_city")
        assert len(report) == 1
        assert [i["id"] for i in report[0]["current"]] == [ids[2]]
        assert [i["id"] for i in report[0]["replaced"]] == [ids[1], ids[0]]
        assert st.state_key_report("nobody.has_this") == []


def test_deleting_the_current_holder_restores_the_one_before(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="user.home_city")
        assert st.delete(new.id)
        assert [e.id for e in st.recall(limit=10).episodes] == [old.id]
        assert [i["id"] for i in st.state_key_report()[0]["current"]] == [old.id]
        rows = st._conn.execute("SELECT episode_id FROM state_keys").fetchall()
        assert [r[0] for r in rows] == [old.id]


def test_unsupersede_undoes_a_wrong_slot_and_the_next_write_replaces_both(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record("lives in Seattle", "observation", timestamp="2026-01-01T10:00:00Z",
                      state_key="user.home_city")
        b = st.record("lives in Austin", "observation", timestamp="2026-02-01T10:00:00Z",
                      state_key="user.home_city")
        assert st.unsupersede(old_id=a.id, new_id=b.id)
        assert {i["id"] for i in st.state_key_report()[0]["current"]} == {a.id, b.id}
        c = st.record("lives in Denver", "observation", timestamp="2026-03-01T10:00:00Z",
                      state_key="user.home_city")
        assert [e.id for e in st.recall(limit=10).episodes] == [c.id]


def test_set_state_key_links_existing_episodes(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        assert st.set_state_key(old.id, "user.home_city") == []
        assert st.set_state_key(new.id, "user.home_city") == [(old.id, new.id)]
        # The same key again changes nothing; a different key is refused.
        assert st.set_state_key(new.id, " USER.home_city") == []
        with pytest.raises(SupersessionError, match="already fills"):
            st.set_state_key(new.id, "user.job")
        with pytest.raises(SupersessionError, match="does not exist"):
            st.set_state_key("deadbeef", "user.job")


def test_set_state_key_refuses_a_cycle_and_writes_nothing(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        ts = "2026-01-05T10:00:00Z"
        x = st.record("The project database runs on postgres in production.", "observation",
                      timestamp=ts)
        y = st.record("The project database runs on sqlite in production now.", "observation",
                      timestamp=ts)
        assert st.supersede(old_id=x.id, new_id=y.id)  # equal timestamps pass the order check
        st.set_state_key(y.id, "project.db")
        with pytest.raises(SupersessionError, match="cycle"):
            st.set_state_key(x.id, "project.db")
        keyed = {r[0] for r in st._conn.execute("SELECT episode_id FROM state_keys")}
        assert keyed == {y.id}


# -- the redirect -----------------------------------------------------------------

@pytest.mark.parametrize("mode", ["query", "prompt"])
def test_a_hit_on_the_replaced_fact_serves_the_current_one(tmp_path, mode):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        query = QUESTION + " near the waterfront"
        before = [e.id for e in _recall(st, query, mode=mode).episodes]
        assert old.id in before  # the query reaches the old fact by its words
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="user.home_city")
        res = _recall(st, query, mode=mode)
        ids = [e.id for e in res.episodes]
        assert new.id in ids and old.id not in ids
        assert res.replaced == {new.id: (old.id,)}
        # The new fact shares no distinctive word with the query: hiding alone served nothing.
        assert not ({"seattle", "live"} & set(NEW.lower().split()))


def test_no_links_means_no_redirect_and_the_same_episodes(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        res = _recall(st)
        assert res.replaced == {}
        hits, heads = st.superseded_keyword_candidates(["seattle"], limit_per_keyword=10)
        assert hits == {} and heads == {}


def test_a_replacement_after_the_cutoff_is_not_a_redirect(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                  state_key="user.home_city")
        res = _recall(st, exclude_recent_minutes=60, now="2026-02-10T10:30:00Z")
        assert old.id in [e.id for e in res.episodes]
        assert res.replaced == {}


def test_a_redirect_follows_the_chain_to_its_live_end(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        mid = st.record("Moved to Denver for a year.", "observation",
                        timestamp="2026-02-01T10:00:00Z", state_key="user.home_city")
        cur = st.record(NEW, "observation", timestamp="2026-03-01T10:00:00Z",
                        state_key="user.home_city")
        res = _recall(st)
        ids = [e.id for e in res.episodes]
        assert cur.id in ids and old.id not in ids and mid.id not in ids
        assert res.replaced[cur.id] == (old.id,)


def test_an_explicit_link_redirects_too(tmp_path):
    """The redirect reads supersessions, not keys: an explicit grounded link serves its
    replacement as well (it used to only hide the old fact)."""
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record("The database engine for Quillmark is postgres.", "observation",
                        timestamp="2026-01-05T10:00:00Z")
        new = st.record("Quillmark moved its database engine over to sqlite.", "observation",
                        timestamp="2026-02-10T10:00:00Z", supersedes=[old.id])
        res = _recall(st, "is quillmark still on postgres")
        assert new.id in [e.id for e in res.episodes]
        assert res.replaced == {new.id: (old.id,)}


# -- surfaces ---------------------------------------------------------------------

def test_mcp_record_takes_a_state_key(tmp_path):
    from anneal_memory.server import Server
    with Store(str(tmp_path / "m.db")) as st:
        srv = Server(st)
        srv._tool_record({"content": OLD, "episode_type": "observation",
                          "state_key": "user.home_city"})
        out = srv._tool_record({"content": NEW, "episode_type": "observation",
                                "state_key": "user.home_city"})
        assert not out.get("isError")
        assert len(st.recall(limit=10).episodes) == 1
        bad = srv._tool_record({"content": "x", "episode_type": "observation",
                                "state_key": "   "})
        assert bad.get("isError")


def _cli(monkeypatch, capsys, *argv):
    from anneal_memory.cli import main as cli_main
    monkeypatch.setattr("sys.argv", ["anneal-memory", *argv])
    try:
        cli_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    return capsys.readouterr()


def test_cli_record_state_key_and_the_state_listing(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "m.db")
    Store(db).close()
    _cli(monkeypatch, capsys, "--db", db, "record", OLD, "--state-key", "user.home_city")
    _cli(monkeypatch, capsys, "--db", db, "record", NEW, "--state-key", "user.home_city")
    out = _cli(monkeypatch, capsys, "--db", db, "state", "--json").out
    report = json.loads(out)
    assert [r["key"] for r in report] == ["user.home_city"]
    assert len(report[0]["current"]) == 1 and len(report[0]["replaced"]) == 1
    text = _cli(monkeypatch, capsys, "--db", db, "state").out
    assert "current" in text and "replaced" in text and "unsupersede" in text


def test_cli_state_set_and_its_refusal(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        b = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
    _cli(monkeypatch, capsys, "--db", db, "state", "user.home_city", "--set", a.id)
    out = json.loads(_cli(monkeypatch, capsys, "--db", db, "state", "user.home_city",
                          "--set", b.id, "--json").out)
    assert out["links"] == [{"old_id": a.id, "new_id": b.id}]
    with pytest.raises(SystemExit):
        _cli(monkeypatch, capsys, "--db", db, "state", "user.job", "--set", b.id)
    assert "already fills" in capsys.readouterr().err


def test_mcp_keyword_recall_names_the_replacement(tmp_path):
    from anneal_memory.server import Server
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="user.home_city")
        text = Server(st)._tool_recall({"keyword": "Seattle"})["content"][0]["text"]
        assert f"({old.id})" in text and f"replaced by ({new.id})" in text
        assert "Austin" in text
        # A filtered call is left alone, as the durable-facts block is.
        filtered = Server(st)._tool_recall({"keyword": "Seattle", "source": "agent"})
        assert "replaced by" not in filtered["content"][0]["text"]
