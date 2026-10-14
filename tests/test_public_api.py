"""The public surface is frozen (docs/stability.md). This test fails when it moves.

A failure means one of: the change is an addition or a deliberate break that
needs the snapshot regenerated (``python scripts/gen_public_api_snapshot.py``)
plus a CHANGELOG line, or it is an accident to undo. The diff names the entry.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_GEN = ROOT / "scripts" / "gen_public_api_snapshot.py"
_SNAP = ROOT / "tests" / "data" / "public_api_snapshot.json"

if not _GEN.exists():  # a source distribution ships tests, not scripts
    pytest.skip("scripts/gen_public_api_snapshot.py is not in this tree", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("_gen_public_api_snapshot", _GEN)
assert _spec and _spec.loader
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

_expected = json.loads(_SNAP.read_text(encoding="utf-8"))
_actual = json.loads(_gen.render(_gen.collect()))


def _diff(a: dict, b: dict) -> str:
    keys = sorted(set(a) | set(b))
    return ", ".join(k for k in keys if a.get(k) != b.get(k)) or "none"


def test_all_is_unchanged() -> None:
    assert _actual["all"] == _expected["all"]


@pytest.mark.parametrize("name", _expected["all"])
def test_export_unchanged(name: str) -> None:
    assert _actual["exports"].get(name) == _expected["exports"][name]


def test_cli_tree_unchanged() -> None:
    assert _actual["cli"] == _expected["cli"]


def test_mcp_tool_names_unchanged() -> None:
    assert sorted(_actual["mcp_tools"]) == sorted(_expected["mcp_tools"])


@pytest.mark.parametrize("tool", sorted(_expected["mcp_tools"]))
def test_mcp_input_schema_unchanged(tool: str) -> None:
    assert _actual["mcp_tools"].get(tool) == _expected["mcp_tools"][tool]


def test_snapshot_file_is_what_the_generator_writes() -> None:
    assert _SNAP.read_text(encoding="utf-8") == _gen.render(_gen.collect())
