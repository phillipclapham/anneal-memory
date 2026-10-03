"""B1: durable facts survive by mechanism, not by the composer's memory.

Reproduced failure (InMind; a levain long run on 0.9.25): a fact the
continuity carried was dropped by a later wrap, at well under the size budget.
Each test here wraps a store, then saves text that loses or changes a durable
line, and checks what the save does about it.
"""

from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import pytest

from anneal_memory import (
    DEFAULT_SCHEMA,
    FLOW_SCHEMA,
    Store,
    prepare_wrap,
    validate_structure,
    validated_save_continuity,
)
from anneal_memory.continuity import format_wrap_package_text
from anneal_memory.durable import DurableFact, parse_durable_facts
from anneal_memory.schema import (
    default_max_chars,
    durable_budget,
    name_for_schema,
    required_headings,
    schema_by_name,
    schema_role_warning,
    validate_schema,
)

TODAY = "2026-10-03"

# The schemas a store persisted before Durable Facts existed.
OLD_FLOW6 = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Active Threads", "role": "live-state"},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
    {"heading": "Understanding", "role": "narrative-timeless"},
]
OLD_DEFAULT4 = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Patterns", "role": "graduating"},
    {"heading": "Decisions", "role": "decisions"},
    {"heading": "Context", "role": "narrative"},
]

PENDING = (
    "- The nightly bank export calls fmt_row52; it switches to fmt_row64 only at "
    "the bank cutover, which has not happened (as of 2026-10-03) — cues: cutover, "
    "bank, export, nightly, formatter"
)
ALLERGY = "- tree nut allergy — cues: restaurant, dinner, recipe, food, menu"


def partnership_text(durable: str | None, *, context: str = "c") -> str:
    """A FLOW-shaped continuity; ``durable`` is the section body (no header),
    or None for a text with no Durable Facts section."""
    block = "" if durable is None else f"## Durable Facts\n{durable}\n\n"
    return (
        "# T — Memory (v1)\n\n"
        "## State\nworking\n\n"
        "## Active Threads\n- a thread\n\n"
        f"{block}"
        "## Patterns\n- x | 1x (2026-10-03)\n\n"
        "## Decisions\n- d\n\n"
        f"## Context\n{context}\n\n"
        "## Understanding\nu\n"
    )


def default_text(durable: str | None) -> str:
    block = "" if durable is None else f"## Durable Facts\n{durable}\n\n"
    return (
        "# T — Memory (v1)\n\n"
        "## State\nworking\n\n"
        f"{block}"
        "## Patterns\n- x | 1x (2026-10-03)\n\n"
        "## Decisions\n- d\n\n"
        "## Context\nc\n"
    )


def wrap(store: Store, text: str, n: int = 0) -> tuple[dict, list[str]]:
    """One prepare + save; returns the result and the warnings it raised."""
    store.record(f"episode {n} {text[:10]}", "observation")
    assert prepare_wrap(store)["status"] == "ready"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validated_save_continuity(store, text, today=TODAY)
    return result, [str(w.message) for w in caught]


def durable_warnings(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("Durable facts:")]


@pytest.fixture
def pstore(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T", section_schema=FLOW_SCHEMA)
    yield s
    s.close()


# -- The invariant at save ---------------------------------------------------


class TestCarryForward:
    def test_omitted_line_is_reinserted_and_warned(self, pstore):
        wrap(pstore, partnership_text(f"{ALLERGY}\n{PENDING}"), 1)
        _, msgs = wrap(pstore, partnership_text(ALLERGY), 2)
        saved = pstore.load_continuity()
        assert PENDING in saved
        assert ALLERGY in saved
        got = durable_warnings(msgs)
        assert len(got) == 1 and "re-inserted verbatim" in got[0] and PENDING in got[0]
        # Re-inserted INSIDE the section, after the composer's own lines.
        facts = parse_durable_facts(saved, FLOW_SCHEMA)
        assert [f.line for f in facts] == [ALLERGY, PENDING]

    def test_omitted_section_is_recreated_at_schema_position(self, pstore):
        wrap(pstore, partnership_text(PENDING), 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        saved = pstore.load_continuity()
        headers = [l for l in saved.split("\n") if l.startswith("## ")]
        assert headers == [
            "## State", "## Active Threads", "## Durable Facts", "## Patterns",
            "## Decisions", "## Context", "## Understanding",
        ]
        assert PENDING in saved
        assert any("re-created" in m for m in durable_warnings(msgs))

    def test_pending_transition_survives_three_omitting_wraps(self, pstore):
        wrap(pstore, partnership_text(PENDING), 1)
        for n in range(2, 5):
            _, msgs = wrap(pstore, partnership_text(None), n)
            assert durable_warnings(msgs)
        assert parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA)[0].line == PENDING

    def test_kept_line_saves_composer_bytes_and_no_warning(self, pstore):
        wrap(pstore, partnership_text(PENDING), 1)
        text = partnership_text(PENDING, context="c2")
        _, msgs = wrap(pstore, text, 2)
        assert pstore.load_continuity() == text
        assert durable_warnings(msgs) == []

    def test_whitespace_difference_is_not_an_omission(self, pstore):
        # Only the line really left out comes back; the respaced one is kept.
        wrap(pstore, partnership_text(f"{ALLERGY}\n{PENDING}"), 1)
        spaced = ALLERGY.replace("tree nut", "tree   nut") + "  "
        _, msgs = wrap(pstore, partnership_text(spaced), 2)
        saved = pstore.load_continuity()
        assert saved.count("tree") == 1 and PENDING in saved
        [got] = durable_warnings(msgs)
        assert PENDING in got and "tree" not in got

    def test_default_schema_store_also_carries(self, tmp_path):
        store = Store(tmp_path / "d.db", project_name="T")  # fresh default
        try:
            wrap(store, default_text(ALLERGY), 1)
            _, msgs = wrap(store, default_text(None), 2)
            assert ALLERGY in store.load_continuity()
            assert durable_warnings(msgs)
        finally:
            store.close()


class TestDropMarker:
    def test_marker_drops_line_records_audit_and_leaves_no_marker(self, tmp_path):
        store = Store(tmp_path / "m.db", project_name="T", section_schema=FLOW_SCHEMA)
        wrap(store, partnership_text(f"{ALLERGY}\n{PENDING}"), 1)
        marker = f"[drop-durable: {PENDING}]"
        _, msgs = wrap(store, partnership_text(f"{ALLERGY}\n{marker}"), 2)
        saved = store.load_continuity()
        store.close()
        assert PENDING not in saved
        assert "drop-durable" not in saved
        assert ALLERGY in saved
        assert durable_warnings(msgs) == []
        entries = [
            json.loads(l)
            for l in (tmp_path / "m.audit.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        saves = [e for e in entries if e["event"] == "continuity_saved"]
        assert saves[-1]["data"]["durable_dropped"] == [PENDING]
        assert "durable_dropped" not in saves[0]["data"]

    def test_marker_naming_the_fact_part_only(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        wrap(pstore, partnership_text("[drop-durable: tree nut allergy]"), 2)
        assert "tree nut" not in pstore.load_continuity()

    def test_marker_that_also_keeps_the_line_still_drops_it(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        wrap(pstore, partnership_text(f"{ALLERGY}\n- [drop-durable: {ALLERGY}]"), 2)
        assert "tree nut" not in pstore.load_continuity()

    def test_unknown_marker_is_removed_and_warned(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        _, msgs = wrap(pstore, partnership_text(f"{ALLERGY}\n[drop-durable: - no such fact]"), 2)
        saved = pstore.load_continuity()
        assert "drop-durable" not in saved and ALLERGY in saved
        got = durable_warnings(msgs)
        assert len(got) == 1 and "names no line" in got[0]

    def test_empty_marker_is_removed_and_warned(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        _, msgs = wrap(pstore, partnership_text(f"{ALLERGY}\n[drop-durable: ]"), 2)
        assert "drop-durable" not in pstore.load_continuity()
        assert any("names no line" in m for m in durable_warnings(msgs))

    def test_marker_outside_the_section_is_ignored(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        text = partnership_text(ALLERGY, context=f"c\n[drop-durable: {ALLERGY}]")
        _, msgs = wrap(pstore, text, 2)
        assert ALLERGY in pstore.load_continuity()
        assert any("outside" in m for m in durable_warnings(msgs))


class TestSizeNeverRefuses:
    def test_over_budget_warns_and_keeps_every_line(self, pstore):
        budget = durable_budget(default_max_chars(FLOW_SCHEMA))
        lines = [f"- fact number {i} " + "x" * 80 for i in range(budget // 80 + 5)]
        wrap(pstore, partnership_text("\n".join(lines)), 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        facts = parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA)
        assert [f.line for f in facts] == lines
        assert any("over its" in m for m in durable_warnings(msgs))

    def test_reinsertion_never_refuses_a_text_at_the_size_limit(self, pstore):
        big_durable = "\n".join(f"- durable fact {i} " + "y" * 90 for i in range(25))
        max_chars = default_max_chars(FLOW_SCHEMA)
        wrap(pstore, partnership_text(big_durable), 1)
        filler = "z" * (max_chars - len(partnership_text(None)) - 10)
        result, _ = wrap(pstore, partnership_text(None, context=filler), 2)
        assert result["chars"] > max_chars  # durable on top of a full text
        assert len(parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA)) == 25

    def test_omitting_a_large_section_is_not_a_shrink_refusal(self, pstore):
        # Most of the prior's non-graduating mass is Durable Facts. Before B1 a
        # wrap that left the section out fell under the shrink gate's
        # whole-document floor and was REFUSED; now the lines are carried back
        # before the gate reads the text.
        big_durable = "\n".join(f"- durable fact {i} " + "y" * 90 for i in range(25))
        wrap(pstore, partnership_text(big_durable), 1)
        wrap(pstore, partnership_text(None), 2)
        assert len(parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA)) == 25


class TestNearDuplicate:
    def test_reworded_without_marker_warns(self, pstore):
        old = "- The nightly bank export calls fmt_row52 until the bank cutover"
        new = "- The nightly bank export calls fmt_row64 after the bank cutover"
        wrap(pstore, partnership_text(old), 1)
        _, msgs = wrap(pstore, partnership_text(new), 2)
        saved = pstore.load_continuity()
        assert old in saved and new in saved
        assert any("looks reworded" in m for m in durable_warnings(msgs))


# -- Cue words ---------------------------------------------------------------


class TestCues:
    @pytest.mark.parametrize("marker", [" — cues:", " -- cues:", " | cues:", " | CUES:"])
    def test_parse_marker_forms(self, marker):
        text = default_text(f"- tree nut allergy{marker} Restaurant,  dinner , ,menu")
        [fact] = parse_durable_facts(text, DEFAULT_SCHEMA)
        assert fact == DurableFact(
            line=f"- tree nut allergy{marker} Restaurant,  dinner , ,menu",
            fact="tree nut allergy",
            cues=("restaurant", "dinner", "menu"),
        )

    def test_parse_no_marker_and_non_fact_lines(self):
        text = default_text("- plain fact\nprose line\n\n[drop-durable: x]\n- ")
        facts = parse_durable_facts(text, DEFAULT_SCHEMA)
        assert facts == [DurableFact(line="- plain fact", fact="plain fact", cues=())]
        assert parse_durable_facts(text, OLD_DEFAULT4) == []
        assert parse_durable_facts(None, DEFAULT_SCHEMA) == []

    def test_cue_only_change_is_carried_without_reinsert(self, pstore):
        # The cue-updated line replaces the old one; the line really left out
        # (PENDING) is the only re-insert.
        wrap(pstore, partnership_text(f"{ALLERGY}\n{PENDING}"), 1)
        updated = "- tree nut allergy — cues: restaurant, bakery, snack"
        _, msgs = wrap(pstore, partnership_text(updated), 2)
        saved = pstore.load_continuity()
        assert updated in saved and ALLERGY not in saved and PENDING in saved
        [got] = durable_warnings(msgs)
        assert "tree nut" not in got

    def test_cue_line_reinserted_verbatim(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        wrap(pstore, partnership_text(None), 2)
        [fact] = parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA)
        assert fact.line == ALLERGY
        assert fact.cues == ("restaurant", "dinner", "recipe", "food", "menu")

    def test_more_than_eight_cues_warns(self, pstore):
        line = "- tree nut allergy — cues: " + ", ".join(f"c{i}" for i in range(9))
        _, msgs = wrap(pstore, partnership_text(line), 1)
        assert any("more than 8 cues" in m for m in durable_warnings(msgs))
        _, msgs = wrap(pstore, partnership_text(ALLERGY), 2)  # 5 cues, and a cue update
        assert not any("cues on" in m for m in durable_warnings(msgs))


# -- Wrap package -------------------------------------------------------------


class TestPackage:
    def test_guidance_and_size_against_budget(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        pstore.record("e", "observation")
        text = format_wrap_package_text(prepare_wrap(pstore))
        budget = durable_budget(default_max_chars(FLOW_SCHEMA))
        section = len("## Durable Facts\n") + len(ALLERGY) + 1 + 1
        assert f"Durable Facts: {section} / {budget} chars" in text
        assert "[drop-durable: <exact line text>]" in text
        assert "would change what advice or answer you give" in text
        assert "cutover, bank, export, nightly, formatter" in text
        assert "`## Durable Facts` (optional)" in text
        assert (
            f"Stay within {default_max_chars(FLOW_SCHEMA)} characters, not counting "
            f"`## Durable Facts`"
        ) in text


# -- Persisted schema is the authority ---------------------------------------


class TestPersistedSchema:
    def test_old_flow6_store_unaffected(self, tmp_path):
        store = Store(tmp_path / "m.db", project_name="T", section_schema=OLD_FLOW6)
        try:
            assert name_for_schema(store.section_schema) == "partnership"
            assert schema_role_warning(store.section_schema) is None
            text = partnership_text(None)
            wrap(store, text, 1)
            store.record("e", "observation")
            pkg = format_wrap_package_text(prepare_wrap(store))
            assert "Durable" not in pkg and "drop-durable" not in pkg
            assert f"Stay within {default_max_chars(OLD_FLOW6)} characters.\n" in pkg
            # A Durable Facts heading is just an unknown header here: no
            # carrying, no warning, bytes saved as written.
            validated_save_continuity(store, partnership_text(ALLERGY), today=TODAY)
            _, msgs = wrap(store, text, 2)
            assert store.load_continuity() == text
            assert durable_warnings(msgs) == []
        finally:
            store.close()

    def test_three_persistence_cases(self, tmp_path):
        # (1) persisted the old default -> no section.
        old = Store(tmp_path / "old.db", section_schema=OLD_DEFAULT4)
        assert [s["heading"] for s in old.section_schema] == [s["heading"] for s in OLD_DEFAULT4]
        assert name_for_schema(old.section_schema) == "default"
        old.close()
        # (2) a new partnership store gets it.
        new = Store(tmp_path / "new.db", section_schema=schema_by_name("partnership"))
        assert "Durable Facts" in [s["heading"] for s in new.section_schema]
        new.close()
        # (3) no persisted schema -> DEFAULT_SCHEMA -> gets it.
        p = str(tmp_path / "none.db")
        Store(p).close()
        conn = sqlite3.connect(p)
        conn.execute("DELETE FROM metadata WHERE key='section_schema'")
        conn.commit()
        conn.close()
        legacy = Store(p, read_only=True)
        assert "Durable Facts" in [s["heading"] for s in legacy.section_schema]
        legacy.close()

    def test_name_for_schema_old_and_new(self):
        assert name_for_schema(OLD_FLOW6) == "partnership"
        assert name_for_schema(OLD_DEFAULT4) == "default"
        assert name_for_schema(FLOW_SCHEMA) == "partnership"
        assert name_for_schema(DEFAULT_SCHEMA) == "default"
        assert "Durable Facts" in [s["heading"] for s in FLOW_SCHEMA]
        # Normalising a persisted schema keeps the optional flag (it is what the
        # store writes back).
        assert validate_schema(FLOW_SCHEMA) == FLOW_SCHEMA

    def test_old_flow6_misrole_still_warns(self):
        misroled = [dict(s) for s in OLD_FLOW6]
        misroled[-1]["role"] = "narrative"
        assert schema_role_warning(misroled) is not None


class TestSchemaValidation:
    def test_optional_is_not_required(self):
        assert "Durable Facts" in [s["heading"] for s in FLOW_SCHEMA]
        assert "Durable Facts" not in required_headings(FLOW_SCHEMA)
        assert validate_structure(partnership_text(None), FLOW_SCHEMA)
        assert validate_structure(partnership_text(ALLERGY), FLOW_SCHEMA)

    def test_optional_must_be_bool(self):
        bad = [dict(s) for s in DEFAULT_SCHEMA]
        bad[1]["optional"] = "yes"
        with pytest.raises(ValueError, match="must be a bool"):
            validate_schema(bad)

    def test_only_durable_may_be_optional(self):
        bad = [dict(s) for s in OLD_DEFAULT4]
        bad[0]["optional"] = True
        with pytest.raises(ValueError, match="only a 'durable' section"):
            validate_schema(bad)

    def test_one_durable_section(self):
        bad = [dict(s) for s in DEFAULT_SCHEMA] + [
            {"heading": "More Facts", "role": "durable", "optional": True}
        ]
        with pytest.raises(ValueError, match="at most one 'durable'"):
            validate_schema(bad)

    def test_merged_durable_header_is_ambiguous(self):
        merged = partnership_text(None).replace("## Patterns", "## Durable Facts and Patterns")
        assert validate_structure(merged, FLOW_SCHEMA) is False

    def test_budgets_unchanged_by_the_optional_section(self):
        assert default_max_chars(DEFAULT_SCHEMA) == default_max_chars(OLD_DEFAULT4) == 20000
        assert default_max_chars(FLOW_SCHEMA) == default_max_chars(OLD_FLOW6)
        assert durable_budget(20000) == 3000
