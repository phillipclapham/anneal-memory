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
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00.000000Z")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00.000000Z")
        assert st.set_state_key(old.id, "user.home_city") == []
        assert st.set_state_key(new.id, "user.home_city") == [(old.id, new.id)]
        # The same key again changes nothing; a different key is refused.
        assert st.set_state_key(new.id, " USER.home_city") == []
        with pytest.raises(SupersessionError, match="already fills"):
            st.set_state_key(new.id, "user.job")
        with pytest.raises(SupersessionError, match="does not exist"):
            st.set_state_key("deadbeef", "user.job")


def test_set_state_key_refuses_an_episode_already_replaced_and_writes_nothing(tmp_path):
    """Every planned link points at the slot's newest live holder, so a cycle can only
    come from keying an episode a link already hides, and that is refused first."""
    with Store(str(tmp_path / "m.db")) as st:
        ts = "2026-01-05T10:00:00.000000Z"
        x = st.record("The project database runs on postgres in production.", "observation",
                      timestamp=ts)
        y = st.record("The project database runs on sqlite in production now.", "observation",
                      timestamp=ts)
        assert st.supersede(old_id=x.id, new_id=y.id)  # equal timestamps pass the order check
        st.set_state_key(y.id, "project.db")
        with pytest.raises(SupersessionError, match="already replaced"):
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
        served = next(e for e in res.episodes if e.id == new.id)
        assert [r.id for r in served.replaces] == [old.id]
        assert served.replaces[0].content == OLD and served.replaces[0].timestamp.startswith("2026-01-05")
        # The new fact shares no distinctive word with the query: hiding alone served nothing.
        assert not ({"seattle", "live"} & set(NEW.lower().split()))


def test_no_links_means_no_redirect_and_the_same_episodes(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        res = _recall(st)
        assert all(e.replaces == () for e in res.episodes)
        assert not st.has_supersessions()


def test_a_replacement_after_the_cutoff_is_not_a_redirect(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="user.home_city")
        st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                  state_key="user.home_city")
        res = _recall(st, exclude_recent_minutes=60, now="2026-02-10T10:30:00Z")
        assert old.id in [e.id for e in res.episodes]
        assert all(e.replaces == () for e in res.episodes)


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
        assert [r.id for r in res.episodes[ids.index(cur.id)].replaces] == [old.id]


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
        ids = [e.id for e in res.episodes]
        assert new.id in ids
        assert [r.id for r in res.episodes[ids.index(new.id)].replaces] == [old.id]


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
    assert "current" in text and "replaced" in text and "state --unset" in text


def test_cli_state_set_and_its_refusal(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00.000000Z")
        b = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00.000000Z")
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
        assert f"({new.id})" in text and f"replaces ({old.id})" in text
        # Wrap-hidden matches newer than the servable one cannot crowd it out of a
        # small limit (codex L3 r3 MED).
        for i in range(4):
            x = st.record(f"Seattle wrap note {i}", "observation",
                          timestamp=f"2026-03-0{i + 1}T10:00:00Z")
            y = st.record(f"Seattle wrap note {i} again", "observation",
                          timestamp=f"2026-03-0{i + 1}T11:00:00Z")
            st.supersede(old_id=x.id, new_id=y.id, source="wrap")
        got = st.replaced_matches("Seattle", limit=2, redirectable_only=True)
        assert [e.id for e in got] == [old.id]
        assert "Austin" in text
        # A filtered call is left alone, as the durable-facts block is.
        filtered = Server(st)._tool_recall({"keyword": "Seattle", "source": "agent"})
        assert "Replaced since" not in filtered["content"][0]["text"]
        listed = Server(st)._tool_recall({"keyword": "Seattle", "include_superseded": True})
        assert "Replaced since" not in listed["content"][0]["text"]


# -- L1/L2 round 1 regressions ------------------------------------------------------

def test_unset_takes_a_wrong_key_out_and_it_stays_out(tmp_path):   # L2 H3
    with Store(str(tmp_path / "m.db")) as st:
        good = st.record("lives in Seattle", "observation", timestamp="2026-01-01T10:00:00Z",
                         state_key="user.home_city")
        wrong = st.record("quarterly offsite in Lisbon, bring a passport", "observation",
                          timestamp="2026-02-01T10:00:00Z", state_key="user.home_city")
        out = st.clear_state_key(wrong.id)
        assert out["key"] == "user.home_city" and out["removed"] == [(good.id, wrong.id)]
        assert {e.id for e in st.recall(limit=10).episodes} == {good.id, wrong.id}
        later = st.record("lives in Austin", "observation", timestamp="2026-03-01T10:00:00Z",
                          state_key="user.home_city")
        live = {e.id for e in st.recall(limit=10).episodes}
        assert live == {wrong.id, later.id}          # the unkeyed one is never re-hidden
        assert st.clear_state_key(wrong.id) == {"key": None, "removed": [], "added": []}


def test_unset_in_the_middle_of_a_chain_re_forms_the_slot(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        a, b, c = (st.record(f"lives in city {n}", "observation",
                             timestamp=f"2026-0{n}-01T10:00:00Z", state_key="k").id for n in (1, 2, 3))
        out = st.clear_state_key(b)
        assert sorted(out["removed"]) == sorted([(a, b), (b, c)])
        assert out["added"] == [(a, c)]
        assert {e.id for e in st.recall(limit=10).episodes} == {b, c}


def test_a_backdated_write_also_makes_a_split_slot_whole(tmp_path):   # L1 LOW 2
    with Store(str(tmp_path / "m.db")) as st:
        h1 = st.record("city one", "observation", timestamp="2026-01-01T10:00:00Z", state_key="k")
        h3 = st.record("city three", "observation", timestamp="2026-03-01T10:00:00Z", state_key="k")
        assert st.unsupersede(old_id=h1.id, new_id=h3.id)
        st.record("city two", "observation", timestamp="2026-02-01T10:00:00Z", state_key="k")
        assert [e.id for e in st.recall(limit=10).episodes] == [h3.id]


@pytest.mark.parametrize("older,newer", [
    ("2026-01-01T12:00:00Z", "2026-01-01T12:00:00.500000Z"),     # L2 M2: 'Z' sorts after '.'
    ("2026-01-01T12:00:00+05:00", "2026-01-01T10:00:00Z"),       # 07:00Z before 10:00Z
])
def test_order_is_by_instant_not_spelling(tmp_path, older, newer):
    with Store(str(tmp_path / "m.db")) as st:
        n = st.record("the newer fact", "observation", timestamp=newer, state_key="k")
        st.record("the older fact", "observation", timestamp=older, state_key="k")
        assert [e.id for e in st.recall(limit=10).episodes] == [n.id]


def test_a_tie_goes_to_the_later_write(tmp_path):   # L1 LOW 4
    ts = "2026-01-01T10:00:00Z"
    with Store(str(tmp_path / "m.db")) as st:
        st.record("first", "observation", timestamp=ts, state_key="k")
        second = st.record("second", "observation", timestamp=ts, state_key="k")
        assert [e.id for e in st.recall(limit=10).episodes] == [second.id]


def test_keying_an_episode_already_replaced_is_refused(tmp_path):   # L2 L2
    with Store(str(tmp_path / "m.db")) as st:
        x = st.record("The database engine for Quillmark is postgres.", "observation",
                      timestamp="2026-01-01T10:00:00.000000Z")
        st.record("Quillmark moved its database engine over to sqlite.", "observation",
                  timestamp="2026-02-01T10:00:00.000000Z", supersedes=[x.id])
        with pytest.raises(SupersessionError, match="already replaced"):
            st.set_state_key(x.id, "k")


def test_a_wrap_link_hides_but_never_serves(tmp_path):   # L2 M5
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record("The database engine for Quillmark is postgres.", "observation",
                        timestamp="2026-01-05T10:00:00Z")
        new = st.record("Quillmark moved its database engine over to sqlite.", "observation",
                        timestamp="2026-02-10T10:00:00Z")
        assert st.supersede(old_id=old.id, new_id=new.id, source="wrap")
        res = _recall(st, "is quillmark still on postgres")
        assert old.id not in [e.id for e in res.episodes]
        assert all(e.replaces == () for e in res.episodes)   # its own hit, never a swap


def test_a_link_changes_no_score(tmp_path):   # L2 H1 + M4
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z", state_key="k")
        before = {e.id: e.score for e in _recall(st, mode="prompt",
                                                 query=QUESTION + " near the waterfront").episodes}
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z", state_key="k")
        after = {e.id: e.score for e in _recall(st, mode="prompt",
                                                query=QUESTION + " near the waterfront").episodes}
        assert after[new.id] == before[old.id]
        assert {i: v for i, v in after.items() if i != new.id} == \
               {i: v for i, v in before.items() if i != old.id}


def test_cli_state_unset(tmp_path, monkeypatch, capsys):
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z", state_key="k")
        b = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z", state_key="k")
    out = json.loads(_cli(monkeypatch, capsys, "--db", db, "state", "--unset", b.id, "--json").out)
    assert out["key"] == "k" and out["removed"] == [{"old_id": a.id, "new_id": b.id}]


def test_set_state_key_refuses_a_non_canonical_timestamp(tmp_path):   # codex L3 r1 HIGH 1
    with Store(str(tmp_path / "m.db")) as st:
        e = st.record("a fact", "observation", timestamp="2026-01-01T12:00:00+05:00")
        with pytest.raises(SupersessionError, match="UTC form"):
            st.set_state_key(e.id, "k")


def test_record_with_a_key_stores_a_canonical_utc_timestamp(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        e = st.record("a fact", "observation", timestamp="2026-01-01T12:00:00+05:00",
                      state_key="k")
        assert e.timestamp == "2026-01-01T07:00:00.000000Z"
        with pytest.raises(ValueError, match="microsecond precision"):
            st.record("b fact", "observation", timestamp="yesterday", state_key="k")


def test_a_cutoff_and_the_key_rule_agree_on_order(tmp_path):   # codex L3 r1 HIGH 1
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-02-10T08:00:00Z", state_key="k")
        st.record(NEW, "observation", timestamp="2026-02-10T05:00:00-05:00", state_key="k")
        res = _recall(st, exclude_recent_minutes=60, now="2026-02-10T10:00:00Z")
        assert old.id in [e.id for e in res.episodes]          # 10:00Z is inside the window
        assert all(e.replaces == () for e in res.episodes)


def test_a_wrap_hop_anywhere_on_the_path_never_serves(tmp_path):   # codex H2 + complement M1
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        a = st.record("The database engine for Quillmark is postgres.", "observation",
                      timestamp="2026-01-05T10:00:00Z")
        b = st.record("Quillmark moved its database engine over to sqlite.", "observation",
                      timestamp="2026-02-10T10:00:00Z", supersedes=[a.id])
        c = st.record("Quillmark moved its database engine over to duckdb now.", "observation",
                      timestamp="2026-03-10T10:00:00Z")
        assert st.supersede(old_id=b.id, new_id=c.id, source="wrap")
        res = _recall(st, "is quillmark still on postgres")
        assert all(not e.replaces for e in res.episodes)
        assert st.redirectable_ids({a.id: c.id}) == set()


def test_a_key_link_is_its_own_kind_whoever_wrote_it(tmp_path):   # codex M3
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z",
                        state_key="k", source="wrap")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        state_key="k", source="wrap")
        assert [r[0] for r in st._conn.execute("SELECT source FROM supersessions")] == ["state_key"]
        res = _recall(st, QUESTION + " near the waterfront")
        assert new.id in [e.id for e in res.episodes]
        assert st.redirectable_ids({old.id: new.id}) == {old.id}

def test_a_key_row_never_outlives_its_episode_even_for_raw_sql(tmp_path):   # codex M6
    with Store(str(tmp_path / "m.db")) as st:
        e = st.record("lives in Seattle", "observation", state_key="k")
        st._conn.execute("DELETE FROM episodes WHERE id = ?", (e.id,))   # an older binary
        st._conn.commit()
        assert st._conn.execute("SELECT COUNT(*) FROM state_keys").fetchone()[0] == 0


def test_mcp_block_finds_an_old_match_behind_many_live_ones(tmp_path):   # codex M4
    from anneal_memory.server import Server
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record("Seattle office lease signed.", "observation",
                        timestamp="2026-01-01T10:00:00Z", state_key="office.city")
        new = st.record("The office moved to Austin.", "observation",
                        timestamp="2026-01-02T10:00:00Z", state_key="office.city")
        for i in range(30):
            st.record(f"Seattle weather note {i}", "observation",
                      timestamp=f"2026-02-{1 + i % 28:02d}T10:{i:02d}:00Z")
        text = Server(st)._tool_recall({"keyword": "Seattle"})["content"][0]["text"]
        assert f"({new.id})" in text and f"replaces ({old.id})" in text


def test_cli_state_refuses_conflicting_flags(tmp_path, monkeypatch, capsys):   # LOW
    db = str(tmp_path / "m.db")
    Store(db).close()
    with pytest.raises(SystemExit):
        _cli(monkeypatch, capsys, "--db", db, "state", "k", "--set", "a", "--unset", "b")
    with pytest.raises(SystemExit):
        _cli(monkeypatch, capsys, "--db", db, "state", "k", "--unset", "abcd1234")


def test_normalising_a_normal_key_changes_nothing():   # complement LOW
    for k in ["\u0130stanbul", "STRASSE stra\u00dfe", "\u1e9e", "\ufb01le"]:
        n = normalize_state_key(k)
        assert normalize_state_key(n) == n


# -- L3 r2: a keyed episode is replaced only through its key ------------------------

def test_an_explicit_link_from_a_keyed_episode_is_refused(tmp_path):   # codex r2 H2/H3, glm H
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record("The database engine for Quillmark is postgres.", "observation",
                      timestamp="2026-01-01T10:00:00Z", state_key="quillmark.db")
        with pytest.raises(SupersessionError, match="fills the state slot"):
            st.record("Quillmark moved its database engine over to sqlite.", "observation",
                      timestamp="2026-02-01T10:00:00Z", supersedes=[a.id])
        b = st.record("Quillmark moved its database engine over to sqlite.", "observation",
                      timestamp="2026-02-01T10:00:00Z")
        with pytest.raises(SupersessionError, match="fills the state slot"):
            st.supersede(old_id=a.id, new_id=b.id)
        with pytest.raises(SupersessionError, match="fills the state slot"):
            st.supersede(old_id=a.id, new_id=b.id, source="wrap")
        # Not even with the same key, from any source (codex L3 r3 H1/H2: an equal-key
        # explicit or wrap link survived clear_state_key and hid a keyed episode behind
        # an unkeyed one); nothing was written by the refusals.
        for src in ("agent", "wrap"):
            with pytest.raises(SupersessionError, match="fills the state slot"):
                st.record("Quillmark moved its database engine over to duckdb.", "observation",
                          timestamp="2026-03-01T10:00:00Z", supersedes=[a.id],
                          state_key="quillmark.db", source=src)
        assert st.recall(limit=10).total_matching == 2
        # Through the key alone it goes, and clearing that key brings the old one back.
        c = st.record("Quillmark moved its database engine over to duckdb.", "observation",
                      timestamp="2026-03-01T10:00:00Z", state_key="quillmark.db")
        assert {e.id for e in st.recall(limit=10).episodes} == {b.id, c.id}
        st.clear_state_key(c.id)
        assert {e.id for e in st.recall(limit=10).episodes} == {a.id, b.id, c.id}


def test_a_wrap_link_from_a_keyed_episode_is_rejected_not_saved(tmp_path):   # codex r2 H1
    from anneal_memory import prepare_wrap, validated_save_continuity
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record("The database engine for Quillmark is postgres.", "observation",
                      state_key="quillmark.db")
        b = st.record("Quillmark moved its database engine over to sqlite.", "observation")
        token = prepare_wrap(st, max_chars=40000)["wrap_token"]
        r = validated_save_continuity(
            st, "# t\n\n## State\ns.\n\n## Patterns\n\n## Decisions\nd.\n\n## Context\n"
            f"c. [supersedes: {a.id} by {b.id}]\n", wrap_token=token)
        assert st.recall(limit=10).total_matching == 2
        assert any("fills the state slot" in str(x) for x in r.get("supersessions_rejected", []))


def test_finer_than_microsecond_is_refused_for_a_key(tmp_path):   # codex r2 M5
    with Store(str(tmp_path / "m.db")) as st:
        with pytest.raises(ValueError, match="microsecond"):
            st.record("a", "observation", timestamp="2026-01-01T10:00:00.1234569Z", state_key="k")


def test_the_report_reads_a_large_slot_in_chunks(tmp_path):   # codex r2 M6
    with Store(str(tmp_path / "m.db")) as st:
        for i in range(1100):
            st.record(f"value {i}", "observation",
                      timestamp=f"2026-01-01T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}Z",
                      state_key="k")
        r = st.state_key_report("k")[0]
        assert len(r["current"]) == 1 and len(r["replaced"]) == 1099


def test_cli_state_set_prints_the_canonical_key(tmp_path, monkeypatch, capsys):   # codex r2 LOW
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00.000000Z")
    out = json.loads(_cli(monkeypatch, capsys, "--db", db, "state", " USER.Home_City ",
                          "--set", a.id, "--json").out)
    assert out["key"] == "user.home_city"
