"""CAP-08: an episode's trust class, and graduations that cannot climb on
tool/external grounding alone (T1-T3).

The BEFORE run (1007+28, 2026-10-07, on 8542f49): one episode recorded from a
web page carrying a false claim, and a wrap citing only it, graduated the claim
to 2x through the real pipeline. ``TestTheBeforeRunNowHolds`` is that run with
the episode labelled ``external``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from anneal_memory import (
    DEFAULT_TRUST,
    TRUST_LEVELS,
    EpisodeType,
    Store,
    prepare_wrap,
    trust_rank,
    validated_save_continuity,
)
from anneal_memory.audit import AuditTrail
from anneal_memory.graduation import validate_graduations
from anneal_memory.server import Server

CLAIM = "Web page fetched by a tool: the Eiffel Tower was moved to Lyon in 2025."
EXPLANATION = "Eiffel Tower moved to Lyon in 2025"
HEAD = "# T — Memory (v1)\n\n## State\nActive.\n\n"
TAIL = "## Decisions\nNone.\n\n## Context\nSession.\n"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


class TestTrustStorage:
    def test_the_levels_are_ordered_lowest_first(self):
        assert TRUST_LEVELS == ("external", "tool", "agent", "operator")
        assert DEFAULT_TRUST == "agent"
        assert [trust_rank(t) for t in TRUST_LEVELS] == [0, 1, 2, 3]
        with pytest.raises(ValueError, match="unknown trust"):
            trust_rank("trusted")

    def test_record_stores_only_a_non_default_class(self, store, tmp_path):
        a = store.record("agent note", EpisodeType.OBSERVATION)
        e = store.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
        t = store.record("tool output relayed", EpisodeType.OBSERVATION, trust="tool")
        assert store.trust_map([a.id, e.id, t.id]) == {e.id: "external", t.id: "tool"}
        rows = sqlite3.connect(tmp_path / "m.db").execute(
            "SELECT COUNT(*) FROM episode_trust"
        ).fetchone()[0]
        assert rows == 2
        assert store.trust_counts() == {"operator": 0, "agent": 1, "tool": 1, "external": 1}

    def test_an_unknown_class_writes_nothing(self, store):
        with pytest.raises(ValueError, match="unknown trust"):
            store.record("x", EpisodeType.OBSERVATION, trust="trusted")
        assert store.recall(limit=10).episodes == []

    def test_the_audit_record_event_carries_a_non_default_class(self, store, tmp_path):
        store.record("agent note", EpisodeType.OBSERVATION)
        store.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
        lines = (tmp_path / "m.audit.jsonl").read_text().splitlines()
        records = [json.loads(line)["data"] for line in lines if '"record"' in line]
        assert [r.get("trust") for r in records] == [None, "external"]

    def test_a_trust_row_never_outlives_its_episode(self, store, tmp_path):
        """The trigger, whatever path deletes: an 8-hex id that came back
        would otherwise inherit the old label."""
        e = store.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
        assert store.delete(e.id)
        conn = sqlite3.connect(tmp_path / "m.db")
        assert conn.execute("SELECT COUNT(*) FROM episode_trust").fetchone()[0] == 0
        conn.execute("INSERT INTO episodes (id, timestamp, type, content) VALUES ('aaaaaaaa', 't', 'observation', 'c')")
        conn.execute("INSERT INTO episode_trust VALUES ('aaaaaaaa', 'tool')")
        conn.execute("DELETE FROM episodes WHERE id = 'aaaaaaaa'")
        assert conn.execute("SELECT COUNT(*) FROM episode_trust").fetchone()[0] == 0
        conn.close()

    def test_a_read_only_store_from_before_the_table_reads_all_agent(self, tmp_path):
        db = tmp_path / "old.db"
        Store(db).close()
        conn = sqlite3.connect(db)
        conn.execute("DROP TRIGGER episode_trust_follows_delete")
        conn.execute("DROP TABLE episode_trust")
        conn.commit()
        conn.close()
        ro = Store(db, read_only=True)
        try:
            assert ro.trust_map(["abcdef12"]) == {}
        finally:
            ro.close()


class TestSetTrust:
    def test_lowering_is_open_and_audited(self, store, tmp_path):
        ep = store.record("an agent note", EpisodeType.OBSERVATION)
        assert store.set_trust(ep.id, "tool") == "agent"
        assert store.trust_map([ep.id]) == {ep.id: "tool"}
        events = [json.loads(line) for line in (tmp_path / "m.audit.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "trust_set"
        assert events[-1]["data"] == {"episode_id": ep.id, "from": "agent", "to": "tool"}

    def test_raising_is_refused_without_the_operator(self, store):
        ep = store.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
        with pytest.raises(ValueError, match="operator's call"):
            store.set_trust(ep.id, "agent")
        assert store.trust_map([ep.id]) == {ep.id: "external"}
        assert store.set_trust(ep.id, "agent", allow_raise=True) == "external"
        assert store.trust_map([ep.id]) == {}

    def test_an_unknown_episode_is_refused(self, store):
        with pytest.raises(ValueError, match="no episode"):
            store.set_trust("deadbeef", "tool")

    def test_a_refusal_inside_a_batch_keeps_the_batchs_writes(self, store):
        """Refusals are raised after the db boundary, which rolls back on any
        exception: raised inside it, a refusal discarded the batch."""
        ep = store.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
        with store._batch():
            kept = store.record("written in the batch", EpisodeType.OBSERVATION)
            with pytest.raises(ValueError):
                store.set_trust(ep.id, "operator")
        assert store.get(kept.id) is not None


def _line(ids, level=2, date="2026-10-08", name="eiffel_in_lyon"):
    return f'- {name} | {level}x ({date}) [evidence: {", ".join(ids)} "{EXPLANATION}"]'


class TestGraduationRule:
    """``validate_graduations`` with ``trust_of`` (T3), pure-function."""

    def _run(self, ids, trust, content=None, history=None, level=2):
        content = content or {i: CLAIM for i in ids}
        return validate_graduations(
            text="## Patterns\n" + _line(ids, level=level) + "\n",
            valid_ids=set(ids),
            today="2026-10-08",
            node_content_map=content,
            pattern_history_lookup=(lambda name: history) if history else None,
            trust_of=lambda cid: trust.get(cid, DEFAULT_TRUST),
        )

    @pytest.mark.parametrize("cls", ["external", "tool"])
    def test_grounding_only_below_agent_does_not_climb(self, cls):
        r = self._run(["aaaa0001"], {"aaaa0001": cls})
        assert r.validated == 0 and r.demoted == 1
        assert "| 1x (2026-10-08) (uncorroborated)" in r.text
        assert [(u.name, u.trust, u.held) for u in r.uncorroborated] == [
            ("eiffel_in_lyon", cls, False)
        ]

    def test_a_stapled_agent_citation_that_grounds_nothing_does_not_corroborate(self):
        """⛔ MUTATION-CHECKED: count every resolved citation instead of the
        grounding ones and this graduates."""
        r = self._run(
            ["aaaa0001", "aaaa0002"],
            {"aaaa0001": "external"},
            content={"aaaa0001": CLAIM, "aaaa0002": "Lunch was a sandwich."},
        )
        assert r.validated == 0
        assert r.uncorroborated[0].citations == ["aaaa0001"]

    @pytest.mark.parametrize("cls", ["agent", "operator"])
    def test_an_agent_or_operator_episode_that_grounds_it_lets_it_climb(self, cls):
        r = self._run(
            ["aaaa0001", "aaaa0002"],
            {"aaaa0001": "external", "aaaa0002": cls},
            content={"aaaa0001": CLAIM, "aaaa0002": f"I checked: the {EXPLANATION}."},
        )
        assert r.validated == 1 and r.uncorroborated == []
        assert r.pattern_trust == {"eiffel_in_lyon": cls}

    def test_without_trust_of_nothing_changes(self):
        r = validate_graduations(
            text="## Patterns\n" + _line(["aaaa0001"]) + "\n",
            valid_ids={"aaaa0001"},
            today="2026-10-08",
            node_content_map={"aaaa0001": CLAIM},
        )
        assert r.validated == 1 and r.uncorroborated == [] and r.pattern_trust == {}

    def test_a_level_earned_earlier_is_held_not_demoted(self):
        """The ungrounded path's hold: earned 2x recently through other
        evidence, re-stamped today on an external-only citation."""
        history = {
            "max_level_reached": 2,
            "last_seen_at": "2026-10-07",
            "explanation_corpus": "the landmark relocated south last year",
            "last_explanation": "the landmark relocated south last year",
        }
        r = self._run(["aaaa0001"], {"aaaa0001": "external"}, history=history)
        assert r.validated == 0 and r.demoted == 0
        assert [u.held for u in r.uncorroborated] == [True]
        assert "| 2x (2026-10-08) (carried-forward)" in r.text

    def test_it_cannot_climb_past_an_earned_level(self):
        history = {
            "max_level_reached": 2,
            "last_seen_at": "2026-10-07",
            "explanation_corpus": "the landmark relocated south last year",
            "last_explanation": "the landmark relocated south last year",
        }
        r = self._run(["aaaa0001"], {"aaaa0001": "external"}, history=history, level=3)
        assert "| 2x (2026-10-08) (uncorroborated)" in r.text
        assert [u.held for u in r.uncorroborated] == [False]


class TestTheBeforeRunNowHolds:
    """The BEFORE run through the real pipeline, with the plant labelled."""

    def _wrap(self, store, patterns, today):
        prepare_wrap(store)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = validated_save_continuity(
                store, HEAD + "## Patterns\n" + patterns + "\n\n" + TAIL, today=today
            )
        return result, [str(w.message) for w in caught]

    def _planted(self, store):
        store.record("Session start.", EpisodeType.OBSERVATION)
        self._wrap(store, "- eiffel_in_lyon | 1x (2026-10-07)", "2026-10-07")
        return store.record(CLAIM, EpisodeType.OBSERVATION,
                            source="web:example.invalid", trust="external")

    def test_an_external_only_graduation_is_held_at_1x(self, store, tmp_path):
        ep = self._planted(store)
        result, warned = self._wrap(store, _line([ep.id]), "2026-10-08")
        assert result["graduations_validated"] == 0
        assert result["uncorroborated"][0]["name"] == "eiffel_in_lyon"
        assert "- eiffel_in_lyon | 1x (2026-10-08) (uncorroborated)" in store.load_continuity()
        assert any("did not climb" in w for w in warned)
        saved = [json.loads(line) for line in (tmp_path / "m.audit.jsonl").read_text().splitlines()
                 if '"continuity_saved"' in line][-1]
        assert saved["data"]["uncorroborated"][0]["trust"] == "external"
        assert AuditTrail.verify(tmp_path / "m.db").valid

    def test_one_agent_episode_that_grounds_it_lets_it_climb(self, store):
        ep = self._planted(store)
        own = store.record(f"I checked a second source myself: {EXPLANATION}.",
                           EpisodeType.OBSERVATION)
        result, warned = self._wrap(store, _line([ep.id, own.id]), "2026-10-08")
        assert result["graduations_validated"] == 1
        assert result["uncorroborated"] == []
        assert result["pattern_trust"] == {"eiffel_in_lyon": "agent"}
        assert not any("did not climb" in w for w in warned)


class TestMcpRecord:
    def test_trust_is_recorded(self, store):
        server = Server(store)
        result = server._tool_record({"content": CLAIM, "episode_type": "observation",
                                      "trust": "external"})
        assert not result.get("isError")
        ep = store.recall(limit=1).episodes[0]
        assert store.trust_map([ep.id]) == {ep.id: "external"}

    def test_an_agent_cannot_label_its_own_write_operator(self, store):
        server = Server(store)
        result = server._tool_record({"content": "trust me", "episode_type": "observation",
                                      "trust": "operator"})
        assert result.get("isError")
        assert store.recall(limit=10).episodes == []


def _cli(db, *args, env_extra=None, stdin=subprocess.DEVNULL):
    env = {k: v for k, v in os.environ.items() if k != "ANNEAL_OPERATOR"}
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *args],
        capture_output=True, text=True, env=env, stdin=stdin,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


class TestCli:
    def test_record_trust_and_the_operator_gate(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        r = _cli(db, "record", CLAIM, "--trust", "external", "--json")
        assert r.returncode == 0, r.stderr
        ext_id = json.loads(r.stdout)["id"]
        r = _cli(db, "record", "the operator's own fact", "--trust", "operator")
        assert r.returncode == 1 and "ANNEAL_OPERATOR=1" in r.stderr
        r = _cli(db, "record", "the operator's own fact", "--trust", "operator", "--json",
                 env_extra={"ANNEAL_OPERATOR": "1"})
        assert r.returncode == 0, r.stderr
        op_id = json.loads(r.stdout)["id"]
        s = Store(db)
        try:
            assert s.trust_map([ext_id, op_id]) == {ext_id: "external", op_id: "operator"}
            assert len(s.recall(limit=10).episodes) == 2
        finally:
            s.close()

    def test_the_trust_command_lowers_freely_and_raises_only_for_the_operator(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        ep_id = json.loads(_cli(db, "record", "a note", "--json").stdout)["id"]
        assert _cli(db, "trust", ep_id).stdout.strip() == f"{ep_id}: agent"
        assert _cli(db, "trust", ep_id, "tool").returncode == 0
        r = _cli(db, "trust", ep_id, "agent")
        assert r.returncode == 1 and "Unchanged" in r.stderr
        r = _cli(db, "trust", ep_id, "agent", "--json", env_extra={"ANNEAL_OPERATOR": "1"})
        assert json.loads(r.stdout) == {"id": ep_id, "from": "tool", "to": "agent"}

    def test_an_export_round_trip_keeps_a_lower_class_and_never_vouches(self, tmp_path):
        src, dst = tmp_path / "a.db", tmp_path / "b.db"
        s = Store(src)
        try:
            ext = s.record(CLAIM, EpisodeType.OBSERVATION, trust="external")
            op = s.record("the operator's own fact", EpisodeType.OBSERVATION, trust="operator")
        finally:
            s.close()
        out = tmp_path / "export.json"
        assert _cli(src, "export", "--format", "json", "--output", str(out)).returncode == 0
        exported = {e["id"]: e.get("trust") for e in json.loads(out.read_text())["episodes"]}
        assert exported == {ext.id: "external", op.id: "operator"}
        Store(dst).close()
        r = _cli(dst, "import", str(out))
        assert r.returncode == 0, r.stderr
        d = Store(dst)
        try:
            # external survives; operator from a file comes in as agent (absent)
            assert d.trust_map([ext.id, op.id]) == {ext.id: "external"}
        finally:
            d.close()
