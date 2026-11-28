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
    validated_save_continuity,
)
from anneal_memory.continuity import format_wrap_package_text
from anneal_memory.store import register_writer_schema


def _wconnect(*args, **kwargs):
    """A raw writer fixture: the gen-2 guard triggers need the writer-schema function."""
    conn = sqlite3.connect(*args, **kwargs)
    register_writer_schema(conn)
    return conn

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





# -- Wrap package -------------------------------------------------------------




# -- Persisted schema is the authority ---------------------------------------


class TestPersistedSchema:

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
        conn = _wconnect(p)
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






class TestFixRoundTexts:







    def test_en_dash_cue_separator(self):
        [f] = parse_durable_facts(default_text("- tree nut allergy – cues: dinner, menu"), DEFAULT_SCHEMA)
        assert f.fact == "tree nut allergy" and f.cues == ("dinner", "menu")


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


    def test_cue_marker_only_on_the_last_line(self):
        text = partnership_text("- fact — cues: a, b\n  more — cues: c")
        [f] = parse_durable_facts(text, FLOW_SCHEMA)
        assert f.cues == ("c",) and f.fact == "fact — cues: a, b more"






# -- fix-diff round on fe242cc -------------------------------------------------


def _crlf_doc(durable_body: str) -> str:
    return (
        "# T\n## State\ns\n## Active Threads\n- t\n"
        f"## Durable Facts\n{durable_body}\n"
        "## Patterns\n- x | 1x (2026-10-03)\n## Decisions\nd\n## Context\nc\n"
        "## Understanding\nu\n"
    ).replace("\n", "\r\n")





# -- scoped round on 7c161d3 ---------------------------------------------------




# -- three-lineage round on f89470d ---------------------------------------------




# -- 0.9.28: every per-item warning list is bounded; near-miss headers are named ---------




