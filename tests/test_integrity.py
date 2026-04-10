"""Tests for anneal_memory.integrity — tool description integrity verification."""

import json
import pytest
from pathlib import Path

from anneal_memory.integrity import (
    TOOLS,
    RESOURCES,
    hash_tool,
    generate_integrity_file,
    verify_integrity,
)


class TestToolDefinitions:
    """Verify canonical tool definitions are well-formed."""

    def test_six_tools_defined(self):
        assert len(TOOLS) == 6

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {"record", "recall", "prepare_wrap", "save_continuity", "delete_episode", "status"}

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]

    def test_record_requires_content_and_type(self):
        record = next(t for t in TOOLS if t["name"] == "record")
        assert "required" in record["inputSchema"]
        assert set(record["inputSchema"]["required"]) == {"content", "episode_type"}

    def test_save_continuity_requires_text(self):
        save = next(t for t in TOOLS if t["name"] == "save_continuity")
        assert "required" in save["inputSchema"]
        assert "text" in save["inputSchema"]["required"]

    def test_status_has_no_required_params(self):
        status = next(t for t in TOOLS if t["name"] == "status")
        assert "required" not in status["inputSchema"]

    def test_episode_type_enum_consistent(self):
        """All tools that reference episode_type should have the same enum values."""
        expected = {"observation", "decision", "tension", "question", "outcome", "context"}
        for tool in TOOLS:
            props = tool["inputSchema"]["properties"]
            if "episode_type" in props:
                assert set(props["episode_type"]["enum"]) == expected

    def test_resources_defined(self):
        assert len(RESOURCES) == 2

    def test_continuity_resource(self):
        assert RESOURCES[0]["uri"] == "anneal://continuity"
        assert RESOURCES[0]["mimeType"] == "text/markdown"

    def test_integrity_manifest_resource(self):
        assert RESOURCES[1]["uri"] == "anneal://integrity/manifest"
        assert RESOURCES[1]["mimeType"] == "application/json"


class TestHashTool:
    """Verify hash generation is deterministic and sensitive."""

    def test_deterministic(self):
        tool = TOOLS[0]
        assert hash_tool(tool) == hash_tool(tool)

    def test_different_descriptions_different_hashes(self):
        tool_a = {"description": "Do A", "inputSchema": {"type": "object", "properties": {}}}
        tool_b = {"description": "Do B", "inputSchema": {"type": "object", "properties": {}}}
        assert hash_tool(tool_a) != hash_tool(tool_b)

    def test_different_schemas_different_hashes(self):
        tool_a = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        tool_b = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"y": {"type": "integer"}}},
        }
        assert hash_tool(tool_a) != hash_tool(tool_b)

    def test_hash_is_sha256_hex(self):
        h = hash_tool(TOOLS[0])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestGenerateIntegrityFile:
    """Test integrity file generation."""

    def test_generates_valid_json(self, tmp_path):
        path = tmp_path / "integrity.json"
        result = generate_integrity_file(path)
        assert result == path
        data = json.loads(path.read_text())
        assert "version" in data
        assert "tools" in data
        assert data["version"] == 1

    def test_contains_all_tool_hashes(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        assert set(data["tools"].keys()) == {t["name"] for t in TOOLS}

    def test_hashes_match_current_definitions(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        for tool in TOOLS:
            assert data["tools"][tool["name"]] == hash_tool(tool)


class TestVerifyIntegrity:
    """Test integrity verification."""

    def test_valid_file_passes(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        valid, issues = verify_integrity(path)
        assert valid is True
        assert issues == []

    def test_missing_file_fails(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_corrupt_json_fails(self, tmp_path):
        path = tmp_path / "integrity.json"
        path.write_text("not json{{{")
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 1
        assert "Failed to read" in issues[0]

    def test_tampered_hash_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        # Tamper with the record tool's hash
        data["tools"]["record"] = "0" * 64
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("record" in i and "mismatch" in i for i in issues)

    def test_missing_tool_in_file_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        del data["tools"]["status"]
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("status" in i and "not found" in i for i in issues)

    def test_extra_tool_in_file_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        data["tools"]["evil_tool"] = "0" * 64
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("evil_tool" in i and "Unknown" in i for i in issues)

    def test_empty_tools_dict_fails(self, tmp_path):
        path = tmp_path / "integrity.json"
        path.write_text(json.dumps({"version": 1, "tools": {}}))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 6  # All 6 tools missing


class TestShippedManifest:
    """Verify the manifest shipped inside the package matches current TOOLS.

    This test would have caught the v0.1.9 staleness that Session 10.5c.1
    Layer 4 found: ``delete_episode`` was added in v0.1.9 but
    ``tool-integrity.json`` was last regenerated in v0.1.8, so the shipped
    manifest had been silently failing host-side verification for one
    full session and one PyPI release. Without this test, any future tool
    addition can ship with the same staleness.

    If this test fails after adding, renaming, or editing a tool, run::

        python3 -c "from anneal_memory.integrity import generate_integrity_file; from pathlib import Path; generate_integrity_file(Path('anneal_memory/tool-integrity.json'))"

    and commit the regenerated manifest.
    """

    def test_shipped_manifest_verifies(self):
        """The manifest shipped inside the anneal_memory package must
        pass verify_integrity() against the current TOOLS definitions.
        """
        import anneal_memory
        pkg_root = Path(anneal_memory.__file__).parent
        manifest = pkg_root / "tool-integrity.json"
        assert manifest.exists(), (
            f"Shipped integrity manifest missing at {manifest} — "
            "regenerate with generate_integrity_file()"
        )
        valid, issues = verify_integrity(manifest)
        assert valid, (
            f"Shipped tool-integrity.json is out of sync with TOOLS. "
            f"Issues: {issues}. Regenerate via "
            f"generate_integrity_file(Path('anneal_memory/tool-integrity.json'))"
        )

    def test_shipped_manifest_covers_all_current_tools(self):
        """Every tool in TOOLS must have an entry in the shipped manifest.

        Complements test_shipped_manifest_verifies by failing loudly and
        specifically when a new tool is added without regenerating, even
        if verify_integrity's behavior changes in the future.
        """
        import anneal_memory
        pkg_root = Path(anneal_memory.__file__).parent
        manifest = pkg_root / "tool-integrity.json"
        data = json.loads(manifest.read_text())
        manifest_tool_names = set(data["tools"].keys())
        current_tool_names = {t["name"] for t in TOOLS}
        missing = current_tool_names - manifest_tool_names
        extra = manifest_tool_names - current_tool_names
        assert not missing, (
            f"Tools in TOOLS but missing from shipped manifest: {missing}. "
            f"Run generate_integrity_file() and commit."
        )
        assert not extra, (
            f"Tools in shipped manifest but removed from TOOLS: {extra}. "
            f"Run generate_integrity_file() and commit."
        )
