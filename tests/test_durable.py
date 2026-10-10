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
        # Every marker drop is reported (scoped round on 7c161d3).
        assert durable_warnings(msgs) == [f"Durable facts: dropped by marker: {PENDING}"]
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
        assert "`## Durable Facts` may be left out." in text
        assert "(optional)" not in text
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

    def test_header_containing_the_optional_heading_is_not_ambiguous(self):
        # L3 ruling: an optional heading counts toward ambiguity only as an
        # exact header. "## Durable Facts and Patterns" is a Patterns header,
        # and it is not a durable section either.
        merged = partnership_text(None).replace("## Patterns", "## Durable Facts and Patterns")
        assert validate_structure(merged, FLOW_SCHEMA) is True
        assert parse_durable_facts(merged.replace("- x | 1x", "- y\n- x | 1x"), FLOW_SCHEMA) == []
        both = partnership_text(None).replace("## Patterns", "## Patterns and Understanding")
        assert validate_structure(both, FLOW_SCHEMA) is False

    def test_budgets_unchanged_by_the_optional_section(self):
        assert default_max_chars(DEFAULT_SCHEMA) == default_max_chars(OLD_DEFAULT4) == 20000
        assert default_max_chars(FLOW_SCHEMA) == default_max_chars(OLD_FLOW6)
        assert durable_budget(20000) == 3000


# -- Fix round (L1 + L2 review findings) -------------------------------------
#
# Each test reproduces a finding against 11d8d92 (it fails there) and pins the
# fix.

CUTOVER_OLD = (
    "- The nightly bank export calls fmt_row52; it switches to fmt_row64 only at "
    "the bank cutover, which has not happened — cues: cutover, bank, export, "
    "nightly, formatter"
)
CUTOVER_NEW = (
    "- The nightly bank export now calls fmt_row64; the bank cutover happened — "
    "cues: cutover, bank, export, nightly, formatter"
)


def saved_bytes(store: Store) -> bytes:
    return Path(store.continuity_path).read_bytes()


class TestFixRoundSilentLoss:
    def test_h1_second_durable_section_is_protected(self, pstore):
        two = partnership_text(ALLERGY).replace(
            "## Patterns", "## Durable Facts\n- second section fact\n\n## Patterns"
        )
        assert [f.fact for f in parse_durable_facts(two, FLOW_SCHEMA)] == [
            "tree nut allergy", "second section fact",
        ]
        wrap(pstore, two, 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        saved = pstore.load_continuity()
        assert "- second section fact" in saved and ALLERGY in saved
        assert saved.count("## Durable Facts") == 1

    def test_h1_new_text_with_two_sections_is_merged_and_warned(self, pstore):
        two = partnership_text(ALLERGY).replace(
            "## Patterns", "## Durable Facts\n- second section fact\n\n## Patterns"
        )
        _, msgs = wrap(pstore, two, 1)
        saved = pstore.load_continuity()
        assert saved.count("## Durable Facts") == 1
        facts = [f.fact for f in parse_durable_facts(saved, FLOW_SCHEMA)]
        assert facts == ["tree nut allergy", "second section fact"]
        headers = [l for l in saved.split("\n") if l.startswith("## ")]
        assert headers.index("## Durable Facts") == headers.index("## Active Threads") + 1
        assert any("merged" in m for m in durable_warnings(msgs))

    def test_m1_star_and_numbered_bullets_and_continuations(self, pstore):
        body = "* star fact\n1. numbered fact\n- wrapped fact that\n  continues here"
        assert [f.fact for f in parse_durable_facts(partnership_text(body), FLOW_SCHEMA)] == [
            "star fact", "numbered fact", "wrapped fact that continues here",
        ]
        wrap(pstore, partnership_text(body), 1)
        wrap(pstore, partnership_text(None), 2)
        saved = pstore.load_continuity()
        assert "* star fact\n1. numbered fact\n- wrapped fact that\n  continues here" in saved

    def test_m1_untracked_prose_line_warns(self, pstore):
        _, msgs = wrap(pstore, partnership_text(f"{ALLERGY}\nSome prose about facts."), 1)
        assert any(
            "'Some prose about facts.'" in m and "not tracked" in m
            for m in durable_warnings(msgs)
        )

    def test_m3_crlf_marker_is_a_marker_and_reinsert_takes_crlf(self, pstore):
        wrap(pstore, partnership_text(f"{ALLERGY}\n{PENDING}").replace("\n", "\r\n"), 1)
        crlf = partnership_text(f"- [drop-durable: {ALLERGY}]").replace("\n", "\r\n")
        _, msgs = wrap(pstore, crlf, 2)
        raw = saved_bytes(pstore)
        assert b"drop-durable" not in raw and b"tree nut" not in raw
        assert PENDING.encode() in raw  # re-inserted
        assert b"\n" not in raw.replace(b"\r\n", b"")  # every line ending is CRLF

    def test_m4_large_section_is_bounded_and_summarised(self, pstore):
        import time
        old = "\n".join(f"- service s{i} runs on host alpha in region east zone" for i in range(300))
        new = "\n".join(f"- service s{i} runs on host alpha in region east area" for i in range(300))
        wrap(pstore, partnership_text(old), 1)
        t0 = time.perf_counter()
        _, msgs = wrap(pstore, partnership_text(new), 2)
        assert time.perf_counter() - t0 < 2.0
        got = durable_warnings(msgs)
        assert sum("looks reworded as" in m for m in got) == 20
        assert any(m.startswith("Durable facts: and ") and "more" in m for m in got)


class TestFixRoundSupersede:
    def test_l2_resolved_transition_is_a_contradiction_candidate(self, pstore):
        wrap(pstore, partnership_text(CUTOVER_OLD), 1)
        _, msgs = wrap(pstore, partnership_text(CUTOVER_NEW), 2)
        got = durable_warnings(msgs)
        assert any(
            "may be superseded by" in m and "fmt_row52" in m and "fmt_row64; the bank" in m
            and "[drop-durable: " in m
            for m in got
        )

    def test_package_lists_pending_transitions(self, pstore):
        wrap(pstore, partnership_text(f"{ALLERGY}\n{CUTOVER_OLD}"), 1)
        pstore.record("e", "observation")
        text = format_wrap_package_text(prepare_wrap(pstore))
        check = text.index("Check whether these pending changes have happened")
        tail = text[check:check + 600]
        assert CUTOVER_OLD in tail and ALLERGY not in tail


class TestFixRoundTexts:
    def test_guidance_wording_and_dateless_example(self, pstore):
        pstore.record("e", "observation")
        text = format_wrap_package_text(prepare_wrap(pstore))
        block = text[text.index("**Durable Facts**"):]
        assert "Most sessions add zero or one line." in block
        assert "Patterns belong in ## Patterns, not here." in block
        assert "(as of" not in block

    def test_pattern_shaped_line_warns(self, pstore):
        _, msgs = wrap(pstore, partnership_text("- x_rule | 2x (2026-10-03) [evidence: abcd1234]"), 1)
        assert any("belong in ## Patterns" in m for m in durable_warnings(msgs))

    def test_reinsert_warning_says_how_to_drop_a_changed_fact(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        assert any("If a fact changed, drop the old line" in m for m in durable_warnings(msgs))

    def test_unknown_drop_names_closest_prior_line(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        _, msgs = wrap(pstore, partnership_text(f"{ALLERGY}\n[drop-durable: tree nut allergies]"), 2)
        [m] = [m for m in durable_warnings(msgs) if "names no line" in m]
        assert f"closest prior line is {ALLERGY!r}" in m and "exact" in m

    def test_rebuilt_section_collapses_blank_runs(self, pstore):
        wrap(pstore, partnership_text(f"{ALLERGY}\n{PENDING}"), 1)
        wrap(pstore, partnership_text(f"{ALLERGY}\n\n\n\n[drop-durable: {PENDING}]\n\n"), 2)
        section = pstore.load_continuity().split("## Durable Facts")[1].split("## Patterns")[0]
        assert "\n\n\n" not in section

    def test_marker_deleting_own_new_line_warns(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        _, msgs = wrap(pstore, partnership_text(f"{ALLERGY}\n[drop-durable: {ALLERGY}]"), 2)
        assert any("which this wrap wrote itself" in m for m in durable_warnings(msgs))

    def test_save_result_carries_durable_warnings(self, pstore, tmp_path):
        wrap(pstore, partnership_text(ALLERGY), 1)
        result, msgs = wrap(pstore, partnership_text(None), 2)
        assert result["durable_warnings"] == durable_warnings(msgs) != []
        old = Store(tmp_path / "old.db", project_name="T", section_schema=OLD_FLOW6)
        try:
            res_old, _ = wrap(old, partnership_text(None), 1)
            assert "durable_warnings" not in res_old
        finally:
            old.close()

    def test_en_dash_cue_separator(self):
        [f] = parse_durable_facts(default_text("- tree nut allergy – cues: dinner, menu"), DEFAULT_SCHEMA)
        assert f.fact == "tree nut allergy" and f.cues == ("dinner", "menu")

    def test_two_prior_lines_sharing_a_fact_warn(self, pstore):
        two = f"{ALLERGY}\n- tree nut allergy — cues: bakery, snack"
        wrap(pstore, partnership_text(two), 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        assert any("two lines share the fact 'tree nut allergy'" in m for m in durable_warnings(msgs))

    def test_migration_entry_names_the_set_schema_rerun(self):
        from anneal_memory.migration import MIGRATION_MANIFEST
        [entry] = [e for e in MIGRATION_MANIFEST if e["feature"] == "AM-DURABLE-FACTS"]
        assert (
            "re-run `anneal-memory --db <path> set-schema <its schema name>` "
            "(e.g. partnership)"
        ) in entry["summary"]


# -- L3 round (complement + codex on dbf9cdf) ---------------------------------
#
# Each test fails on dbf9cdf.


class TestL3Round:
    def test_dropping_durable_mass_by_marker_is_not_a_shrink_refusal(self, pstore):
        # Non-graduating mass is mostly durable; dropping it all by marker
        # tripped the whole-document backstop on dbf9cdf (ValueError).
        lines = [f"- durable fact {i} " + "y" * 90 for i in range(25)]
        wrap(pstore, partnership_text("\n".join(lines)), 1)
        markers = "\n".join(f"[drop-durable: {l}]" for l in lines)
        result, _ = wrap(pstore, partnership_text(markers), 2)
        assert parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA) == []
        assert result["chars"] < 400

    def test_continuation_under_a_cue_line_is_part_of_the_fact(self, pstore):
        first = "- fact — cues: a, b"
        cont = "  load-bearing condition"
        [f] = parse_durable_facts(partnership_text(f"{first}\n{cont}"), FLOW_SCHEMA)
        assert "load-bearing condition" in f.fact
        wrap(pstore, partnership_text(f"{first}\n{cont}"), 1)
        wrap(pstore, partnership_text(first), 2)  # continuation left out
        assert f"{first}\n{cont}" in pstore.load_continuity()

    def test_cue_marker_only_on_the_last_line(self):
        text = partnership_text("- fact — cues: a, b\n  more — cues: c")
        [f] = parse_durable_facts(text, FLOW_SCHEMA)
        assert f.cues == ("c",) and f.fact == "fact — cues: a, b more"

    def test_header_with_durable_words_is_not_refused(self, pstore):
        doc = partnership_text(ALLERGY).replace("## Decisions", "## Decisions (durable facts)")
        wrap(pstore, doc, 1)
        assert "## Decisions (durable facts)" in pstore.load_continuity()

    def test_reinsert_is_byte_for_byte(self, pstore):
        fact = "- hard break here  \n  continued line \t"
        wrap(pstore, partnership_text(fact), 1)
        wrap(pstore, partnership_text(None), 2)
        assert fact.encode() + b"\n" in saved_bytes(pstore)

    def test_many_unknown_markers_are_bounded(self, pstore):
        lines = [f"- fact number {i} about the service" for i in range(500)]
        wrap(pstore, partnership_text("\n".join(lines)), 1)
        markers = "\n".join(f"[drop-durable: no such line {i}]" for i in range(500))
        _, msgs = wrap(pstore, partnership_text("\n".join(lines) + "\n" + markers), 2)
        got = durable_warnings(msgs)
        assert sum("names no line" in m for m in got) == 20
        assert any("and 480 more drop marker(s)" in m for m in got)

    def test_audit_records_the_whole_fact(self, tmp_path):
        store = Store(tmp_path / "m.db", project_name="T", section_schema=FLOW_SCHEMA)
        two_line = "- wrapped fact\n  second line"
        wrap(store, partnership_text(f"{two_line}\n{ALLERGY}"), 1)
        wrap(store, partnership_text(ALLERGY), 2)  # two_line re-inserted
        wrap(store, partnership_text(f"{ALLERGY}\n{two_line}\n[drop-durable: - wrapped fact]"), 3)
        store.close()
        saves = [
            json.loads(l)["data"]
            for l in (tmp_path / "m.audit.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["event"] == "continuity_saved"
        ]
        assert saves[1]["durable_reinserted"] == [two_line]
        assert saves[2]["durable_dropped"] == [two_line]


# -- fix-diff round on fe242cc -------------------------------------------------


def _crlf_doc(durable_body: str) -> str:
    return (
        "# T\n## State\ns\n## Active Threads\n- t\n"
        f"## Durable Facts\n{durable_body}\n"
        "## Patterns\n- x | 1x (2026-10-03)\n## Decisions\nd\n## Context\nc\n"
        "## Understanding\nu\n"
    ).replace("\n", "\r\n")


class TestFixDiffRound:
    def test_archived_durable_heading_is_not_a_durable_section(self, pstore):
        wrap(pstore, partnership_text(ALLERGY), 1)
        archived = partnership_text(ALLERGY).replace(
            "## Patterns",
            f"## Archived Durable Facts\n[drop-durable: {ALLERGY}]\n- old archived fact\n\n## Patterns",
        )
        assert [f.fact for f in parse_durable_facts(archived, FLOW_SCHEMA)] == ["tree nut allergy"]
        wrap(pstore, archived, 2)
        assert ALLERGY in pstore.load_continuity()

    def test_crlf_doc_dropping_its_durable_lines_is_not_a_shrink_refusal(self):
        from anneal_memory.continuity import _check_no_catastrophic_shrink
        facts = [f"- fact {i}" for i in range(500)]
        prior = _crlf_doc("\n".join(facts))
        new = _crlf_doc("")
        _check_no_catastrophic_shrink(prior, new, FLOW_SCHEMA, allow_shrink=False)

    def test_crlf_save_dropping_500_facts_by_marker_saves(self, pstore):
        facts = [f"- fact {i}" for i in range(500)]
        wrap(pstore, _crlf_doc("\n".join(facts)), 1)
        markers = "\n".join(f"[drop-durable: {f}]" for f in facts)
        wrap(pstore, _crlf_doc(markers), 2)
        assert parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA) == []

    def test_multiline_fact_renders_on_one_line_in_warnings(self, pstore):
        wrap(pstore, partnership_text("- wrapped fact\n  second line"), 1)
        _, msgs = wrap(pstore, partnership_text(None), 2)
        [m] = [m for m in durable_warnings(msgs) if "re-inserted verbatim" in m]
        assert "\n" not in m and "- wrapped fact / second line" in m

    def test_marker_matching_several_facts_warns(self, pstore):
        two = "- wrapped fact\n  variant a\n- wrapped fact\n  variant b"
        wrap(pstore, partnership_text(two), 1)
        _, msgs = wrap(pstore, partnership_text("[drop-durable: - wrapped fact]"), 2)
        assert "wrapped fact" not in pstore.load_continuity()
        assert any("matched 2 prior facts" in m for m in durable_warnings(msgs))



# -- scoped round on 7c161d3 ---------------------------------------------------


class TestScopedRound:
    def test_lowercase_durable_header_is_protected(self, pstore):
        lower = partnership_text(ALLERGY).replace("## Durable Facts", "## durable facts")
        assert [f.fact for f in parse_durable_facts(lower, FLOW_SCHEMA)] == ["tree nut allergy"]
        wrap(pstore, lower, 1)
        wrap(pstore, partnership_text(None), 2)
        assert ALLERGY in pstore.load_continuity()

    def test_every_marker_drop_is_reported(self, pstore):
        two_line = "- wrapped fact\n  second line"
        wrap(pstore, partnership_text(f"{ALLERGY}\n{two_line}"), 1)
        fenced = f"{ALLERGY}\n```\n[drop-durable: - wrapped fact]\n```"
        result, msgs = wrap(pstore, partnership_text(fenced), 2)
        assert "wrapped fact" not in pstore.load_continuity()
        assert "Durable facts: dropped by marker: - wrapped fact / second line" in result[
            "durable_warnings"
        ]

    def test_section_chars_splits_like_measure_sections(self):
        from anneal_memory.continuity import measure_sections
        from anneal_memory.durable import section_chars
        text = partnership_text("- a\r## Not a header\n- b")
        assert section_chars(text, FLOW_SCHEMA) == measure_sections(text)["Durable Facts"]

    def test_dropped_fact_attributed_to_first_marker_only(self, pstore):
        two = "- wrapped fact\n  variant a\n- wrapped fact\n  variant b"
        wrap(pstore, partnership_text(two), 1)
        markers = "[drop-durable: wrapped fact variant a]\n[drop-durable: - wrapped fact]"
        _, msgs = wrap(pstore, partnership_text(markers), 2)
        got = durable_warnings(msgs)
        assert "wrapped fact" not in pstore.load_continuity()
        assert not any("matched 2 prior facts" in m for m in got)
        assert not any("names no line" in m for m in got)


# -- three-lineage round on f89470d ---------------------------------------------


class TestThreeLineageRound:
    def test_heading_fold_matches_the_schema_duplicate_check(self):
        # validate_schema folds with lower(), so it accepts these as distinct headings;
        # the exact-heading rule must fold the same way or every document is ambiguous.
        sch = validate_schema([
            {"heading": "State", "role": "live-state"},
            {"heading": "STRASSE", "role": "graduating"},
            {"heading": "Straße", "role": "durable", "optional": True},
        ])
        doc = "## State\nx\n\n## STRASSE\ny\n\n## Straße\n- fact\n"
        assert validate_structure(doc, sch)
        assert [f.fact for f in parse_durable_facts(doc, sch)] == ["fact"]

    def test_marker_drop_warnings_are_capped_and_summarised(self, pstore):
        facts = [f"- fact {i}" for i in range(25)]
        wrap(pstore, partnership_text("\n".join(facts)), 1)
        markers = "\n".join(f"[drop-durable: {f}]" for f in facts)
        result, msgs = wrap(pstore, partnership_text(markers), 2)
        got = [m for m in result["durable_warnings"] if "dropped by marker" in m]
        assert len(got) == 21
        assert got[-1].startswith("Durable facts: and 5 more fact(s) dropped by marker")
        assert parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA) == []


# -- 0.9.28: every per-item warning list is bounded; near-miss headers are named ---------


class TestBoundedWarningsAndNearMiss:
    def test_one_marker_matching_25_facts_names_at_most_20(self, pstore):
        facts = "\n".join(f"- shared\n  variant {i}" for i in range(25))
        wrap(pstore, partnership_text(facts), 1)
        result, _ = wrap(pstore, partnership_text("[drop-durable: - shared]"), 2)
        [multi] = [m for m in result["durable_warnings"] if "matched 25 prior facts" in m]
        assert multi.count("variant") == 20 and "| and 5 more" in multi
        assert parse_durable_facts(pstore.load_continuity(), FLOW_SCHEMA) == []

    def test_decorated_durable_header_is_named(self, pstore):
        text = partnership_text(None).replace(
            "## Patterns", "## Durable Facts (pinned)\n\n- tree nut allergy\n\n## Patterns")
        result, _ = wrap(pstore, text, 1)
        assert any("'## Durable Facts (pinned)' is not the durable heading" in m
                   for m in result["durable_warnings"])

    def test_a_schema_header_naming_durable_facts_is_not_a_near_miss(self, pstore):
        text = partnership_text(ALLERGY).replace("## Decisions", "## Decisions (durable facts)")
        result, _ = wrap(pstore, text, 1)
        assert not any("is not the durable heading" in m for m in result["durable_warnings"])


class TestGuidanceHeadingsParse:
    def test_every_heading_in_the_section_list_parses_when_copied(self, pstore):
        # A composer copies the section list; each item, backticks stripped, must be a
        # header the parser accepts, or the durable section silently parses to nothing.
        pstore.record("e", "observation")
        text = format_wrap_package_text(prepare_wrap(pstore))
        [line] = [ln for ln in text.splitlines() if "EXACTLY these sections" in ln]
        listed = line.split("in order: ", 1)[1].split(".", 1)[0].split(", ")
        headers = [item.replace("`", "") for item in listed]
        assert all(h.startswith("## ") for h in headers)
        doc = "# T — Memory (v1)\n\n" + "\n\n".join(
            h + ("\n- tree nut allergy" if h == "## Durable Facts" else "\nx") for h in headers
        ) + "\n"
        assert "## Durable Facts" in headers
        assert [f.fact for f in parse_durable_facts(doc, FLOW_SCHEMA)] == ["tree nut allergy"]

    def test_archived_durable_facts_header_is_named_but_durable_factsheet_is_not(self, pstore):
        text = partnership_text(ALLERGY).replace(
            "## Patterns", "## Durable Factsheet\n\nnotes\n\n## Patterns")
        result, _ = wrap(pstore, text, 1)
        assert not any("is not the durable heading" in m for m in result["durable_warnings"])
