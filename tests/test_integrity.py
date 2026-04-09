"""Tests for anneal_memory.integrity — tool description integrity verification."""

import json
import pytest
from pathlib import Path

from anneal_memory.integrity import (
    TOOLS,
    RESOURCES,
    _hash_tool,
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

    def test_one_resource_defined(self):
        assert len(RESOURCES) == 1

    def test_resource_uri(self):
        assert RESOURCES[0]["uri"] == "anneal://continuity"
        assert RESOURCES[0]["mimeType"] == "text/markdown"


class TestHashTool:
    """Verify hash generation is deterministic and sensitive."""

    def test_deterministic(self):
        tool = TOOLS[0]
        assert _hash_tool(tool) == _hash_tool(tool)

    def test_different_descriptions_different_hashes(self):
        tool_a = {"description": "Do A", "inputSchema": {"type": "object", "properties": {}}}
        tool_b = {"description": "Do B", "inputSchema": {"type": "object", "properties": {}}}
        assert _hash_tool(tool_a) != _hash_tool(tool_b)

    def test_different_schemas_different_hashes(self):
        tool_a = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        tool_b = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"y": {"type": "integer"}}},
        }
        assert _hash_tool(tool_a) != _hash_tool(tool_b)

    def test_hash_is_sha256_hex(self):
        h = _hash_tool(TOOLS[0])
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
            assert data["tools"][tool["name"]] == _hash_tool(tool)


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
