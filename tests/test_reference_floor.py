"""The reference floor: a protected layer may not fall below REFERENCE_FLOOR_PERCENT of
the largest it was in the last 30 days, which a save-to-save comparison cannot see.

One test per behaviour. The end-to-end ones run the real save pipeline on a
FLOW_SCHEMA store (the same shape ``TestCatastrophicShrinkGate`` uses)."""

from __future__ import annotations

import pytest

from anneal_memory import FLOW_SCHEMA, Store, prepare_wrap, validated_save_continuity
from anneal_memory import continuity as C


def _doc(understanding: int, patterns: int = 900, context: int = 20) -> str:
    """A FLOW_SCHEMA continuity whose Understanding is ~`understanding` chars."""
    return (
        "# flow — Memory (test)\n\n"
        "## State\nparity-observe mode.\n\n"
        "## Active Threads\n- migration\n\n"
        "## Patterns\n" + ("verify_or_surface grounds truth. " * (patterns // 33)) + "\n\n"
        "## Decisions\n[decided] diet-first.\n\n"
        "## Context\n" + ("recent work. " * (context // 13 + 1)) + "\n\n"
        "## Understanding\n" + ("the felt layer, who we are together. " * (understanding // 37)) + "\n"
    )


def _masses(text: str) -> dict[str, int]:
    return C._floor_role_masses(text, FLOW_SCHEMA)


def _save(store: Store, text: str, today: str, **kw):
    store.record(f"episode {today}: observation.", "observation")
    pkg = prepare_wrap(store, max_chars=40000)
    return validated_save_continuity(
        store, text, today=today, wrap_token=pkg["wrap_token"], **kw)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "s.db", project_name="flow")
    s.set_section_schema(FLOW_SCHEMA)
    yield s
    s.close()


























