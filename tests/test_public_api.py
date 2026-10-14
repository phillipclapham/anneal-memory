"""The public surface is frozen (docs/stability.md). This test fails when it moves.

A failure means one of: the change is an addition or a deliberate break that
needs the snapshot regenerated (``python scripts/gen_public_api_snapshot.py``)
plus a CHANGELOG line, or it is an accident to undo. The assertion message
names the entry.

The snapshot froze the exports in ``__all__`` (classes with bases, inherited
members, enum members and TypedDict keys; function signatures; constant
values), the CLI parser tree, and the MCP tool input schemas. It was generated
on one Python version; the comparison tests skip on any other major.minor.

The ``--json`` output keys of each CLI command are part of the stability
promise (docs/stability.md) and are NOT checked here. That is a known edge:
the snapshot covers the parser, not what a command prints.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_GEN = ROOT / "scripts" / "gen_public_api_snapshot.py"
_SNAP = ROOT / "tests" / "data" / "public_api_snapshot.json"

if not _GEN.exists():  # a source distribution ships tests, not scripts
    pytest.skip("scripts/gen_public_api_snapshot.py is not in this tree", allow_module_level=True)
if not _SNAP.exists():
    pytest.skip("tests/data/public_api_snapshot.json is not in this tree", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("_gen_public_api_snapshot", _GEN)
assert _spec and _spec.loader
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

_expected = json.loads(_SNAP.read_text(encoding="utf-8"))
_PY = f"{sys.version_info[0]}.{sys.version_info[1]}"
_same_python = pytest.mark.skipif(
    _expected.get("python") != _PY,
    reason=f"snapshot generated on Python {_expected.get('python')}; running {_PY}",
)
# collect() is only compared on the generating interpreter, so it only runs there
_actual = json.loads(_gen.render(_gen.collect())) if _expected.get("python") == _PY else {}


def _unified(a: str, b: str) -> str:
    lines = difflib.unified_diff(
        a.splitlines(), b.splitlines(), "committed snapshot", "regenerated", lineterm="", n=2
    )
    return "\n".join(list(lines)[:60])


def _show(label: str, actual: object, expected: object) -> str:
    return f"{label} moved:\n" + _unified(
        json.dumps(expected, indent=1, sort_keys=True), json.dumps(actual, indent=1, sort_keys=True)
    )


@_same_python
def test_all_is_unchanged() -> None:
    assert _actual["all"] == _expected["all"], _show("__all__", _actual["all"], _expected["all"])


@_same_python
@pytest.mark.parametrize("name", _expected["all"])
def test_export_unchanged(name: str) -> None:
    got = _actual["exports"].get(name)
    assert got == _expected["exports"][name], _show(f"export {name}", got, _expected["exports"][name])


@_same_python
def test_cli_tree_unchanged() -> None:
    assert _actual["cli"] == _expected["cli"], _show("the CLI tree", _actual["cli"], _expected["cli"])


@_same_python
def test_mcp_tool_names_unchanged() -> None:
    assert sorted(_actual["mcp_tools"]) == sorted(_expected["mcp_tools"])


@_same_python
@pytest.mark.parametrize("tool", sorted(_expected["mcp_tools"]))
def test_mcp_input_schema_unchanged(tool: str) -> None:
    got = _actual["mcp_tools"].get(tool)
    assert got == _expected["mcp_tools"][tool], _show(f"MCP tool {tool}", got, _expected["mcp_tools"][tool])


def test_mcp_tool_names_are_unique() -> None:
    from anneal_memory.integrity import TOOLS

    names = [t["name"] for t in TOOLS]
    assert len(names) == len(set(names)), sorted(n for n in set(names) if names.count(n) > 1)


def test_snapshot_has_no_machine_paths() -> None:
    text = _SNAP.read_text(encoding="utf-8")
    assert "/Users/" not in text and "/home/" not in text


@_same_python
def test_snapshot_file_is_what_the_generator_writes() -> None:
    committed = _SNAP.read_text(encoding="utf-8")
    fresh = _gen.render(_gen.collect())
    assert committed == fresh, _unified(committed, fresh)


@_same_python
def test_snapshot_is_independent_of_home_and_env() -> None:
    # The child writes to a file, not a pipe: on Windows CI the piped stdout came
    # back None with returncode 0 (run 38621285354), so the pipe is not trusted.
    with tempfile.TemporaryDirectory() as home:
        out = os.path.join(home, "snapshot.out")
        env = {
            **os.environ,
            "HOME": home,
            "USERPROFILE": home,
            "ANNEAL_MEMORY_DB": os.path.join(home, "junk.db"),
            "ANNEAL_MEMORY_SOURCE": "junk-source",
        }
        code = (
            "import importlib.util,sys;"
            f"s=importlib.util.spec_from_file_location('g',{str(_GEN)!r});"
            "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
            f"sys.path.insert(0,{str(ROOT)!r});"
            f"open({out!r},'w',encoding='utf-8',newline='\\n').write(g.render(g.collect()))"
        )
        proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        regenerated = Path(out).read_text(encoding="utf-8")
    committed = _SNAP.read_text(encoding="utf-8")
    assert regenerated == committed, _unified(committed, regenerated)
