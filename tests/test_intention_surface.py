"""The INTENTION names for the spore layer: additive, one store, either id spelling.

Every old ``spore`` name stays (docs/stability.md); these pin the additions:
the aliases, the ``id_prefix`` mint, a mixed-prefix store, the MCP tools and the
CLI group beside the spore ones, and that no second store file is ever made.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import anneal_memory
from anneal_memory import spores
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.spores import SporeError, SporeStore
from anneal_memory.store import Store

_PAIRS = {
    "IntentionStore": "SporeStore",
    "IntentionType": "SporeType",
    "IntentionError": "SporeError",
    "IntentionApply": "SporeApply",
    "IntentionApplyResult": "SporeApplyResult",
    "IntentionOp": "SporeOp",
    "IntentionDict": "SporeDict",
    "intention_version": "spore_version",
    "INTENTION_VERSION_EXCLUDED": "SPORE_VERSION_EXCLUDED",
    "normalize_intention_field": "normalize_spore_field",
}

# spore tool -> its intention name (the verb map: descend -> close, ascend -> promote)
_TOOL_PAIRS = {
    "spore_add": "intention_add",
    "spore_get": "intention_get",
    "spore_list": "intention_list",
    "spore_touch": "intention_touch",
    "spore_update": "intention_update",
    "spore_descend": "intention_close",
    "spore_ascend": "intention_promote",
    "spore_surface": "intention_surface",
}


def test_intention_names_are_the_spore_objects():
    for new, old in _PAIRS.items():
        assert getattr(spores, new) is getattr(spores, old), new
        assert getattr(anneal_memory, new) is getattr(anneal_memory, old), new
        assert new in anneal_memory.__all__, new


def test_id_prefix_mints_and_the_counter_runs_across_both(tmp_path):
    path = tmp_path / "m.spores.json"
    assert SporeStore(path).add(type="task", text="a")["id"] == "spore-001"  # default unchanged
    inew = SporeStore(path, id_prefix="intention")
    assert inew.add(type="task", text="b")["id"] == "intention-002"  # a number is never reused
    assert SporeStore(path).add(type="task", text="c")["id"] == "spore-003"
    with pytest.raises(ValueError):
        SporeStore(path, id_prefix="seed")


def test_either_spelling_reads_updates_and_resolves_in_one_store(tmp_path):
    path = tmp_path / "m.spores.json"
    old = SporeStore(path).add(type="task", text="old record")      # spore-001
    new = SporeStore(path, id_prefix="intention")
    assert new.add(type="question", text="new record")["id"] == "intention-002"
    # an install that switched prefix: the old spore-N record still reads, updates, closes
    assert new.get("intention-001")["id"] == old["id"] == "spore-001"
    assert new.update("intention-001", text="edited")["text"] == "edited"
    assert new.touch("spore-001")["id"] == "spore-001"
    assert new.descend("intention-001", kind="done")["status"] == "resolved"
    # and the default-prefix reader sees the intention-N record under either spelling
    default = SporeStore(path)
    assert default.get("spore-002")["id"] == "intention-002"
    assert default.update("spore-002", text="x")["id"] == "intention-002"
    assert default.get("intention-001")["resolution"]["kind"] == "done"  # resolved set too
    with pytest.raises(SporeError, match="already resolved"):
        default.descend("spore-001", kind="done")
    assert default.get("spore-009") is None


def test_an_exact_spelling_wins_when_a_store_holds_both_numbers(tmp_path):
    path = tmp_path / "m.spores.json"
    rows = [
        {"id": "spore-005", "type": "task", "text": "s", "status": "open", "tier": "warm"},
        {"id": "intention-005", "type": "task", "text": "i", "status": "open", "tier": "warm"},
    ]
    path.write_text(json.dumps({"spores": rows, "resolved": [], "schema_version": 1}))
    store = SporeStore(path)
    assert store.get("spore-005")["text"] == "s"
    assert store.get("intention-005")["text"] == "i"
    assert store.update("intention-005", text="i2")["text"] == "i2"
    assert store.get("spore-005")["text"] == "s"


def test_mcp_has_both_tool_families_on_one_store_file(tmp_path):
    names = {t["name"] for t in TOOLS}
    for old, new in _TOOL_PAIRS.items():
        assert old in names and new in names
    by_name = {t["name"]: t for t in TOOLS}
    def shape(node):  # the schema without its prose: every key, type, enum, bound, required
        if isinstance(node, dict):
            return {k: shape(v) for k, v in node.items() if k != "description"}
        if isinstance(node, list):
            return [shape(v) for v in node]
        return node

    for old, new in _TOOL_PAIRS.items():  # same schema bar the id's name
        a = json.loads(json.dumps(shape(by_name[old]["inputSchema"])).replace("spore_id", "intention_id"))
        assert a == shape(by_name[new]["inputSchema"]), new

    store = Store(path=tmp_path / "t.db", project_name="T")
    try:
        server = Server(store)
        assert set(server._tool_handlers) == names

        def call(name, args):
            return server._handle_tools_call({"name": name, "arguments": args})["content"][0]["text"]

        added = call("intention_add", {"type": "task", "text": "bridge"})
        iid = added.split()[1]
        assert iid == "intention-001"
        assert json.loads(call("spore_get", {"spore_id": iid}))["text"] == "bridge"
        assert json.loads(call("spore_get", {"spore_id": "spore-001"}))["id"] == iid
        assert call("spore_add", {"type": "task", "text": "old way"}).split()[1] == "spore-002"
        assert "Closed" in call("intention_close", {"intention_id": "spore-002", "kind": "done"})
        assert "Promoted" in call("intention_promote",
                                  {"intention_id": iid, "kind": "project", "ref": "p/x"})
        assert call("intention_list", {}) == "No open intentions match."
        assert call("intention_get", {}) == "Error: intention_id is required"
    finally:
        store.close()


def test_cli_intention_group_is_an_alias_of_spore_and_adds_no_second_file(tmp_path):
    db = str(tmp_path / "m.db")

    def run(*args, check=True):
        r = subprocess.run([sys.executable, "-m", "anneal_memory.cli", "--db", db, *args],
                           capture_output=True, text=True, encoding="utf-8")
        if check:
            assert r.returncode == 0, (args, r.stdout, r.stderr)
        return r

    assert "intention-001" in run("intention", "add", "--type", "task", "--text", "x").stdout
    assert "spore-002" in run("spore", "add", "--type", "task", "--text", "y").stdout
    assert json.loads(run("spore", "get", "intention-001", "--json").stdout)["text"] == "x"
    assert json.loads(run("intention", "get", "spore-002", "--json").stdout)["text"] == "y"
    assert "resolved down" in run("intention", "close", "intention-001", "--kind", "done").stdout
    assert "resolved down" in run("intention", "descend", "spore-002", "--kind", "dropped").stdout
    # the filename decision: one sibling file, the spore one, never a second live store
    assert sorted(p.name for p in Path(tmp_path).iterdir() if "spore" in p.name or "intention" in p.name
                  if not p.name.endswith(".lock")) == ["m.spores.json"]
