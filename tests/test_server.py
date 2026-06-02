"""Tests for anneal_memory.server — MCP server tool handlers and protocol.

Tests the Server class by calling handler methods directly (not via stdio).
Transport layer tests (_read_message) use mocked stdin/stdout.
"""

import io
import json
import uuid

import pytest
from pathlib import Path

from anneal_memory import __version__
from unittest.mock import patch

from anneal_memory.server import (
    Server, _tool_result, _response, _error_response,
    _read_message, _write_message, _EOF, _MAX_MESSAGE_SIZE,
)
from anneal_memory.store import Store
from anneal_memory.integrity import TOOLS, RESOURCES


@pytest.fixture
def store(tmp_path):
    """Create a fresh Store for each test."""
    db_path = tmp_path / "test.db"
    s = Store(path=db_path, project_name="TestProject")
    yield s
    s.close()


@pytest.fixture
def server(store):
    """Create a Server backed by the test store."""
    return Server(store)


# -- Helper Assertions --


def _text_from_result(result: dict) -> str:
    """Extract text from a tool result dict."""
    return result["content"][0]["text"]


def _is_error(result: dict) -> bool:
    """Check if a tool result is an error."""
    return result.get("isError", False)


# -- Protocol Handlers --


class TestInitialize:
    def test_returns_protocol_version(self, server):
        result = server._handle_initialize({})
        assert result["protocolVersion"] == "2024-11-05"

    def test_returns_capabilities(self, server):
        result = server._handle_initialize({})
        assert "tools" in result["capabilities"]
        assert "resources" in result["capabilities"]

    def test_returns_server_info(self, server):
        result = server._handle_initialize({})
        assert result["serverInfo"]["name"] == "anneal-memory"
        assert result["serverInfo"]["version"] == __version__


class TestPing:
    def test_returns_empty(self, server):
        assert server._handle_ping({}) == {}


class TestToolsList:
    def test_returns_all_tools(self, server):
        result = server._handle_tools_list({})
        assert len(result["tools"]) == 14  # 6 core + 8 spore tools

    def test_tools_match_integrity_definitions(self, server):
        result = server._handle_tools_list({})
        assert result["tools"] is TOOLS


class TestResourcesList:
    def test_returns_resources(self, server):
        result = server._handle_resources_list({})
        assert len(result["resources"]) == 2
        assert result["resources"][0]["uri"] == "anneal://continuity"
        assert result["resources"][1]["uri"] == "anneal://integrity/manifest"


class TestResourcesRead:
    def test_no_continuity_returns_placeholder(self, server):
        result = server._handle_resources_read({"uri": "anneal://continuity"})
        assert len(result["contents"]) == 1
        assert "No continuity file yet" in result["contents"][0]["text"]

    def test_existing_continuity_returned(self, server, store):
        store.save_continuity("# Test\n## State\nActive\n## Patterns\n## Decisions\n## Context\n")
        result = server._handle_resources_read({"uri": "anneal://continuity"})
        assert "# Test" in result["contents"][0]["text"]

    def test_integrity_manifest_returns_hashes(self, server):
        result = server._handle_resources_read({"uri": "anneal://integrity/manifest"})
        assert len(result["contents"]) == 1
        content = result["contents"][0]
        assert content["mimeType"] == "application/json"
        import json
        manifest = json.loads(content["text"])
        assert manifest["algorithm"] == "SHA-256"
        assert "tools" in manifest
        assert "record" in manifest["tools"]
        assert "delete_episode" in manifest["tools"]
        assert "spore_add" in manifest["tools"]
        assert len(manifest["tools"]) == 14  # 6 core + 8 spore tools

    def test_unknown_uri_returns_empty(self, server):
        result = server._handle_resources_read({"uri": "anneal://unknown"})
        assert result["contents"] == []


class TestToolsCall:
    def test_unknown_tool_returns_error(self, server):
        result = server._handle_tools_call({"name": "nonexistent", "arguments": {}})
        assert _is_error(result)
        assert "Unknown tool" in _text_from_result(result)


# -- Tool: record --


class TestToolRecord:
    def test_record_observation(self, server):
        result = server._tool_record({
            "content": "Noticed a pattern in the data",
            "episode_type": "observation",
        })
        assert not _is_error(result)
        text = _text_from_result(result)
        assert "Recorded observation" in text
        assert "(" in text  # Contains episode ID

    def test_record_all_types(self, server):
        for t in ["observation", "decision", "tension", "question", "outcome", "context"]:
            result = server._tool_record({
                "content": f"Test {t}",
                "episode_type": t,
            })
            assert not _is_error(result)
            assert t in _text_from_result(result)

    def test_record_with_source(self, server):
        result = server._tool_record({
            "content": "External observation",
            "episode_type": "observation",
            "source": "user",
        })
        assert not _is_error(result)

    def test_record_with_metadata(self, server):
        result = server._tool_record({
            "content": "Observation with metadata",
            "episode_type": "observation",
            "metadata": {"confidence": 0.9},
        })
        assert not _is_error(result)

    def test_record_empty_content_fails(self, server):
        result = server._tool_record({
            "content": "",
            "episode_type": "observation",
        })
        assert _is_error(result)

    def test_record_missing_content_fails(self, server):
        result = server._tool_record({
            "episode_type": "observation",
        })
        assert _is_error(result)

    def test_record_missing_type_fails(self, server):
        result = server._tool_record({
            "content": "Something",
        })
        assert _is_error(result)

    def test_record_invalid_type_fails(self, server):
        result = server._tool_record({
            "content": "Something",
            "episode_type": "invalid_type",
        })
        assert _is_error(result)


# -- Tool: delete_episode --


class TestToolDeleteEpisode:
    def test_delete_existing_episode(self, server, store):
        ep = store.record("Test episode", "observation")
        result = server._tool_delete_episode({"episode_id": ep.id})
        assert not _is_error(result)
        assert ep.id in _text_from_result(result)
        assert "Deleted" in _text_from_result(result)
        # Verify actually deleted
        recalled = store.recall(keyword="Test episode")
        assert len(recalled.episodes) == 0

    def test_delete_nonexistent_episode(self, server):
        result = server._tool_delete_episode({"episode_id": "deadbeef"})
        assert _is_error(result)
        assert "not found" in _text_from_result(result)

    def test_delete_empty_id_fails(self, server):
        result = server._tool_delete_episode({"episode_id": ""})
        assert _is_error(result)
        assert "required" in _text_from_result(result)

    def test_delete_missing_id_fails(self, server):
        result = server._tool_delete_episode({})
        assert _is_error(result)
        assert "required" in _text_from_result(result)

    def test_delete_cascades_associations(self, server, store):
        ep1 = store.record("Episode one", "observation")
        ep2 = store.record("Episode two", "observation")
        ep3 = store.record("Episode three", "observation")
        store.record_associations(
            direct_pairs={(ep1.id, ep2.id), (ep1.id, ep3.id)},
        )
        # Verify associations exist
        assocs = store.get_associations([ep1.id])
        assert len(assocs) == 2

        # Delete ep1 via MCP tool
        result = server._tool_delete_episode({"episode_id": ep1.id})
        assert not _is_error(result)
        assert "2 associated link(s)" in _text_from_result(result)

        # Associations gone
        assocs = store.get_associations([ep1.id])
        assert len(assocs) == 0

    def test_delete_no_associations_message(self, server, store):
        ep = store.record("Solo episode", "decision")
        result = server._tool_delete_episode({"episode_id": ep.id})
        text = _text_from_result(result)
        assert "associated link" not in text
        assert "audit trail" in text

    def test_delete_whitespace_id_fails(self, server):
        result = server._tool_delete_episode({"episode_id": "   "})
        assert _is_error(result)
        assert "required" in _text_from_result(result)


# -- Tool: recall --


class TestToolRecall:
    def test_recall_empty_store(self, server):
        result = server._tool_recall({})
        text = _text_from_result(result)
        assert "No matching episodes" in text

    def test_recall_after_recording(self, server):
        server._tool_record({
            "content": "First observation",
            "episode_type": "observation",
        })
        server._tool_record({
            "content": "A decision was made",
            "episode_type": "decision",
        })
        result = server._tool_recall({})
        text = _text_from_result(result)
        assert "Found 2 episodes" in text
        assert "First observation" in text
        assert "A decision was made" in text

    def test_recall_by_type(self, server):
        server._tool_record({"content": "Obs 1", "episode_type": "observation"})
        server._tool_record({"content": "Dec 1", "episode_type": "decision"})
        result = server._tool_recall({"episode_type": "decision"})
        text = _text_from_result(result)
        assert "Found 1 episodes" in text
        assert "Dec 1" in text
        assert "Obs 1" not in text

    def test_recall_by_keyword(self, server):
        server._tool_record({"content": "SQLite is great", "episode_type": "observation"})
        server._tool_record({"content": "Redis is fast", "episode_type": "observation"})
        result = server._tool_recall({"keyword": "SQLite"})
        text = _text_from_result(result)
        assert "SQLite" in text
        assert "Redis" not in text

    def test_recall_with_limit(self, server):
        for i in range(5):
            server._tool_record({"content": f"Episode {i}", "episode_type": "observation"})
        result = server._tool_recall({"limit": 2})
        text = _text_from_result(result)
        assert "Found 5 episodes (showing 2)" in text

    def test_recall_by_source(self, server):
        server._tool_record({"content": "Agent obs", "episode_type": "observation", "source": "agent"})
        server._tool_record({"content": "User obs", "episode_type": "observation", "source": "user"})
        result = server._tool_recall({"source": "user"})
        text = _text_from_result(result)
        assert "User obs" in text
        assert "Agent obs" not in text


# -- Tool: status --


class TestToolStatus:
    def test_empty_store_status(self, server):
        result = server._tool_status({})
        text = _text_from_result(result)
        assert "Episodes: 0 total" in text
        assert "Wraps: 0 completed" in text
        assert "Last wrap: never" in text
        assert "not yet created" in text

    def test_status_after_recording(self, server):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        server._tool_record({"content": "Test2", "episode_type": "decision"})
        result = server._tool_status({})
        text = _text_from_result(result)
        assert "Episodes: 2 total" in text
        assert "observation: 1" in text
        assert "decision: 1" in text

    def test_status_after_wrap(self, server, store):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        # Manually complete a wrap to test status
        store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])
        store.wrap_completed(episodes_compressed=1, continuity_chars=100)
        result = server._tool_status({})
        text = _text_from_result(result)
        assert "Wraps: 1 completed" in text

    # -- Audit health visibility (Diogenes ARCH finding, Apr 13 2026) --

    def test_status_surfaces_audit_enabled(self, server):
        """Default fixture enables audit — status must show it."""
        server._tool_record({"content": "Test", "episode_type": "observation"})
        result = server._tool_status({})
        text = _text_from_result(result)
        assert "Audit: enabled" in text
        assert "entries" in text
        assert ".audit.jsonl" in text
        assert "retention unlimited" in text
        assert "anneal-memory verify" in text

    def test_status_surfaces_audit_disabled(self, tmp_path):
        """audit=False: status must say so in one clean line."""
        db_path = str(tmp_path / "noaudit.db")
        s = Store(path=db_path, project_name="NoAudit", audit=False)
        try:
            srv = Server(s)
            result = srv._tool_status({})
            text = _text_from_result(result)
            assert "Audit: disabled" in text
            assert ".audit.jsonl" not in text
        finally:
            s.close()

    def test_status_surfaces_audit_retention_days(self, tmp_path):
        """Finite retention window surfaces in the line."""
        db_path = str(tmp_path / "retain.db")
        s = Store(
            path=db_path,
            project_name="Retain",
            audit=True,
            audit_retention_days=30,
        )
        try:
            srv = Server(s)
            srv._tool_record({"content": "Test", "episode_type": "observation"})
            result = srv._tool_status({})
            text = _text_from_result(result)
            assert "retention 30d" in text
        finally:
            s.close()


# -- Tool: prepare_wrap --


class TestToolPrepareWrap:
    def test_no_episodes_returns_message(self, server):
        result = server._tool_prepare_wrap({})
        text = _text_from_result(result)
        assert "No episodes since last wrap" in text

    def test_returns_episodes_and_instructions(self, server):
        server._tool_record({"content": "Pattern noticed", "episode_type": "observation"})
        server._tool_record({"content": "Chose SQLite", "episode_type": "decision"})
        result = server._tool_prepare_wrap({})
        text = _text_from_result(result)
        assert not _is_error(result)
        # Should contain instructions
        assert "## State" in text
        assert "## Patterns" in text
        # Should contain episodes
        assert "Pattern noticed" in text
        assert "Chose SQLite" in text
        # Should mention first wrap
        assert "first wrap" in text.lower() or "No existing continuity" in text

    def test_includes_existing_continuity(self, server, store):
        continuity = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            "## Patterns\nthought: something | 1x (2026-03-31)\n"
            "## Decisions\n"
            "## Context\nDid stuff.\n"
        )
        store.save_continuity(continuity)
        # Need to complete a wrap so episodes_since_wrap works correctly
        store.wrap_completed(episodes_compressed=0, continuity_chars=len(continuity))
        # Record new episodes
        server._tool_record({"content": "New finding", "episode_type": "observation"})
        result = server._tool_prepare_wrap({})
        text = _text_from_result(result)
        assert "Current Continuity File" in text
        assert "thought: something" in text

    def test_custom_max_chars(self, server):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        result = server._tool_prepare_wrap({"max_chars": 5000})
        text = _text_from_result(result)
        assert "5000" in text


# -- Tool: save_continuity --


class TestToolSaveContinuity:
    def test_save_valid_continuity(self, server):
        server._tool_record({"content": "Test obs", "episode_type": "observation"})
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive on test\n"
            "## Patterns\nthought: testing works | 1x (2026-03-31)\n"
            "## Decisions\n[decided(rationale: \"test\", on: \"2026-03-31\")] Use SQLite\n"
            "## Context\nFirst session. Tested things.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        assert not _is_error(result)
        output = _text_from_result(result)
        assert "Continuity saved" in output
        assert "Episodes compressed: 1" in output

    def test_save_empty_text_fails(self, server):
        result = server._tool_save_continuity({"text": ""})
        assert _is_error(result)

    def test_save_missing_text_fails(self, server):
        result = server._tool_save_continuity({})
        assert _is_error(result)

    def test_save_invalid_structure_fails(self, server):
        server._tool_record({"content": "Test obs", "episode_type": "observation"})
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": "# Just a title\nNo sections."})
        assert _is_error(result)
        assert "must contain all sections" in _text_from_result(result)

    def test_save_missing_one_section_fails(self, server):
        server._tool_record({"content": "Test obs", "episode_type": "observation"})
        server._tool_prepare_wrap({})
        text = (
            "# Test\n"
            "## State\nActive\n"
            "## Patterns\n"
            "## Decisions\n"
            # Missing ## Context
        )
        result = server._tool_save_continuity({"text": text})
        assert _is_error(result)

    def test_save_updates_store(self, server, store):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            "## Patterns\n\n"
            "## Decisions\n\n"
            "## Context\nDid a thing.\n"
        )
        server._tool_prepare_wrap({})
        server._tool_save_continuity({"text": text})
        # Verify stored
        saved = store.load_continuity()
        assert saved is not None
        assert "Active" in saved
        # Verify wrap recorded
        status = store.status()
        assert status.total_wraps == 1

    def test_save_reports_section_sizes(self, server):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nVery active on important work\n"
            "## Patterns\nthought: deep insight | 1x (2026-03-31)\n"
            "## Decisions\n[decided(rationale: \"yes\", on: \"2026-03-31\")] Choice\n"
            "## Context\nLong narrative about what happened in the session.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        output = _text_from_result(result)
        assert "Section sizes:" in output

    def test_save_without_prepare_wrap_is_refused(self, server, store):
        """save_continuity with no wrap in progress is refused (v0.3.1).

        Before v0.3.1 a no-prepare save fell through to a
        skipped_prepare path and warned; v0.3.1 makes it a hard error
        — the structural fix for phantom re-saves.
        """
        server._tool_record({"content": "Test", "episode_type": "observation"})
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            "## Patterns\n\n"
            "## Decisions\n\n"
            "## Context\nFirst session.\n"
        )
        result = server._tool_save_continuity({"text": text})
        assert _is_error(result)
        assert "No wrap in progress" in _text_from_result(result)
        assert store.status().total_wraps == 0


# -- 10.5c.4: MCP token round-trip --


class TestMCPWrapTokenRoundTrip:
    """The MCP tool contract for wrap_token: prepare_wrap response
    text ends with a 'Wrap token: <hex>' trailer, save_continuity
    accepts an optional wrap_token argument, and mismatch detection
    fires with is_error=True tool results."""

    def test_prepare_wrap_text_includes_token_trailer(self, server):
        server._tool_record(
            {"content": "For token trailer", "episode_type": "observation"}
        )
        result = server._tool_prepare_wrap({})
        text = _text_from_result(result)
        assert "Wrap token:" in text
        # Extract and validate shape.
        token = None
        for line in reversed(text.splitlines()):
            if line.startswith("Wrap token:"):
                token = line.split("Wrap token:", 1)[1].strip()
                break
        assert token is not None
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_save_accepts_correct_token(self, server):
        """Happy path: token from prepare_wrap round-trips through
        the save_continuity tool argument and the save completes."""
        server._tool_record(
            {"content": "Round-trip test", "episode_type": "observation"}
        )
        prep = _text_from_result(server._tool_prepare_wrap({}))
        token = None
        for line in reversed(prep.splitlines()):
            if line.startswith("Wrap token:"):
                token = line.split("Wrap token:", 1)[1].strip()
                break

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nToken round-trip validated.\n"
            "## Patterns\nthought: token works | 1x (2026-04-10)\n"
            "## Decisions\n"
            "[decided(rationale: \"test\", on: \"2026-04-10\")] ok\n"
            "## Context\nRound-tripped through MCP.\n"
        )
        result = server._tool_save_continuity(
            {"text": text, "wrap_token": token}
        )
        assert not _is_error(result)

    def test_save_rejects_wrong_token(self, server):
        """A wrong token produces an is_error=True tool result with
        a 'wrap_token mismatch' message."""
        server._tool_record(
            {"content": "Wrong token test", "episode_type": "observation"}
        )
        server._tool_prepare_wrap({})

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nWrong token.\n"
            "## Patterns\n\n"
            "## Decisions\n\n"
            "## Context\nShould fail.\n"
        )
        result = server._tool_save_continuity(
            {"text": text, "wrap_token": "0" * 32}
        )
        assert _is_error(result)
        assert "wrap_token mismatch" in _text_from_result(result)

    def test_save_rejects_non_string_token(self, server):
        """wrap_token must be a string if provided."""
        server._tool_record(
            {"content": "Type test", "episode_type": "observation"}
        )
        server._tool_prepare_wrap({})

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nType check.\n"
            "## Patterns\n\n"
            "## Decisions\n\n"
            "## Context\nShould fail on type.\n"
        )
        # Integer token — MCP clients could pass the wrong type.
        result = server._tool_save_continuity(
            {"text": text, "wrap_token": 12345}
        )
        assert _is_error(result)
        assert "wrap_token must be a string" in _text_from_result(result)

    def test_save_without_token_still_uses_snapshot(self, server, store):
        """Omitting wrap_token does not disable the snapshot filter —
        a TOCTOU episode recorded after prepare_wrap must still be
        deferred to the next wrap."""
        # Snapshot episode.
        server._tool_record(
            {"content": "Snapshot episode", "episode_type": "observation"}
        )
        server._tool_prepare_wrap({})
        # TOCTOU episode — recorded through the record tool between
        # prepare and save.
        server._tool_record(
            {
                "content": "TOCTOU episode should be deferred",
                "episode_type": "observation",
            }
        )
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nImplicit snapshot.\n"
            "## Patterns\n\n"
            "## Decisions\n\n"
            "## Context\nSnapshot only.\n"
        )
        result = server._tool_save_continuity({"text": text})
        output = _text_from_result(result)
        assert not _is_error(result)
        # Only the snapshot episode should have been compressed.
        assert "Episodes compressed: 1" in output

        # Next prepare_wrap picks up the TOCTOU episode.
        next_prep = _text_from_result(server._tool_prepare_wrap({}))
        assert "TOCTOU episode should be deferred" in next_prep


# -- Graduation Validation Through save_continuity --


class TestGraduationValidation:
    """Test that save_continuity correctly validates graduation citations."""

    def _get_today(self):
        from datetime import date
        return date.today().isoformat()

    def test_valid_citation_passes(self, server, store):
        """A 2x graduation citing a real episode ID should validate."""
        result = server._tool_record({
            "content": "SQLite handles concurrent reads well via WAL mode",
            "episode_type": "observation",
        })
        ep_id = _text_from_result(result).split("(")[1].split(")")[0]
        today = self._get_today()

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            f'## Patterns\nthought: SQLite WAL is reliable | 2x ({today}) '
            f'[evidence: {ep_id} "concurrent reads WAL mode"]\n'
            "## Decisions\n\n"
            "## Context\nTested SQLite.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        assert not _is_error(result)
        output = _text_from_result(result)
        assert "Citations validated: 1" in output

    def test_fake_citation_gets_demoted(self, server, store):
        """A 2x graduation citing a non-existent episode ID should be demoted."""
        server._tool_record({"content": "Real episode", "episode_type": "observation"})
        today = self._get_today()

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            f'## Patterns\nthought: bogus claim | 2x ({today}) '
            f'[evidence: deadbeef "fabricated evidence"]\n'
            "## Decisions\n\n"
            "## Context\nTest.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        assert not _is_error(result)
        output = _text_from_result(result)
        assert "Citations demoted (bad evidence): 1" in output

        # Verify the saved text was actually modified (demoted to 1x)
        saved = store.load_continuity()
        assert "| 1x" in saved
        assert "(ungrounded)" in saved

    def test_1x_pattern_needs_no_citation(self, server):
        """1x patterns don't need citations — they're first observations."""
        server._tool_record({"content": "Something", "episode_type": "observation"})
        today = self._get_today()

        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            f"## Patterns\nthought: new insight | 1x ({today})\n"
            "## Decisions\n\n"
            "## Context\nTest.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        assert not _is_error(result)
        # No validation messages — 1x doesn't need citations
        output = _text_from_result(result)
        assert "Citations validated" not in output
        assert "demoted" not in output

    def test_explanation_overlap_required(self, server, store):
        """Citation explanation must reference actual episode content."""
        result = server._tool_record({
            "content": "PostgreSQL chosen for ACID compliance in production",
            "episode_type": "decision",
        })
        ep_id = _text_from_result(result).split("(")[1].split(")")[0]
        today = self._get_today()

        # Explanation references completely different content
        text = (
            "# TestProject — Memory (v1)\n"
            "## State\nActive\n"
            f'## Patterns\nthought: Redis is fast | 2x ({today}) '
            f'[evidence: {ep_id} "Redis caching performance benchmarks"]\n'
            "## Decisions\n\n"
            "## Context\nTest.\n"
        )
        server._tool_prepare_wrap({})
        result = server._tool_save_continuity({"text": text})
        output = _text_from_result(result)
        assert "demoted" in output


# -- Stale Wrap Recovery --


class TestStaleWrapRecovery:
    def test_prepare_wrap_clears_stale_flag(self, server, store):
        """prepare_wrap with no episodes should clear stale wrap_in_progress."""
        store.wrap_started(
            token=uuid.uuid4().hex, episode_ids=[]
        )  # Simulate abandoned wrap
        assert store.status().wrap_in_progress is True
        server._tool_prepare_wrap({})  # No episodes → clears flag
        assert store.status().wrap_in_progress is False


# -- Recall Edge Cases --


class TestRecallEdgeCases:
    def test_negative_limit_clamped_to_zero(self, server):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        result = server._tool_recall({"limit": -1})
        assert "No matching episodes" in _text_from_result(result)

    def test_negative_offset_clamped_to_zero(self, server):
        server._tool_record({"content": "Test", "episode_type": "observation"})
        result = server._tool_recall({"offset": -1})
        assert not _is_error(result)


# -- Full Wrap Cycle --


class TestFullWrapCycle:
    """Test the complete prepare_wrap -> save_continuity flow."""

    def test_full_cycle(self, server, store):
        # Record episodes
        server._tool_record({"content": "Discovered SQLite is zero-dep", "episode_type": "observation"})
        server._tool_record({"content": "Use SQLite for episodic store", "episode_type": "decision"})
        server._tool_record({"content": "How to handle pruning?", "episode_type": "question"})

        # Prepare wrap
        wrap_result = server._tool_prepare_wrap({})
        assert not _is_error(wrap_result)
        wrap_text = _text_from_result(wrap_result)
        assert "3" in wrap_text  # 3 episodes

        # Save continuity
        continuity = (
            "# TestProject — Memory (v1)\n"
            "## State\nBuilding episodic store with SQLite.\n"
            "## Patterns\n{architecture:\n"
            "  thought: SQLite zero-dep is key advantage | 1x (2026-03-31)\n"
            "}\n"
            "## Decisions\n"
            "[decided(rationale: \"zero-dep, indexed queries\", on: \"2026-03-31\")] SQLite for episodes\n"
            "## Context\nFirst session. Evaluated storage options, settled on SQLite.\n"
            "Open question: pruning strategy.\n"
        )
        save_result = server._tool_save_continuity({"text": continuity})
        assert not _is_error(save_result)

        # Verify status
        status_result = server._tool_status({})
        status_text = _text_from_result(status_result)
        assert "Wraps: 1 completed" in status_text
        assert "Episodes: 3 total, 0 since last wrap" in status_text

        # Verify continuity resource
        resource = server._handle_resources_read({"uri": "anneal://continuity"})
        assert "SQLite" in resource["contents"][0]["text"]

    def test_second_wrap_cycle(self, server, store):
        """Two consecutive wrap cycles work correctly."""
        # First cycle
        server._tool_record({"content": "First session work", "episode_type": "observation"})
        continuity_v1 = (
            "# TestProject — Memory (v1)\n"
            "## State\nSession 1 complete.\n"
            "## Patterns\nthought: testing matters | 1x (2026-03-31)\n"
            "## Decisions\n\n"
            "## Context\nFirst session done.\n"
        )
        server._tool_prepare_wrap({})
        server._tool_save_continuity({"text": continuity_v1})

        # Second cycle
        server._tool_record({"content": "Second session work", "episode_type": "observation"})
        continuity_v2 = (
            "# TestProject — Memory (v1)\n"
            "## State\nSession 2 complete.\n"
            "## Patterns\nthought: testing matters | 1x (2026-03-31)\n"
            "## Decisions\n\n"
            "## Context\nTwo sessions done.\n"
        )
        server._tool_prepare_wrap({})
        server._tool_save_continuity({"text": continuity_v2})

        status = store.status()
        assert status.total_wraps == 2
        assert status.total_episodes == 2
        assert status.episodes_since_wrap == 0


# -- Response Helpers --


class TestResponseHelpers:
    def test_tool_result_success(self):
        result = _tool_result("Success message")
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Success message"
        assert result.get("isError", False) is False

    def test_tool_result_error(self):
        result = _tool_result("Error!", is_error=True)
        assert result["isError"] is True

    def test_response_format(self):
        resp = _response(42, {"tools": []})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 42
        assert resp["result"] == {"tools": []}

    def test_error_response_format(self):
        resp = _error_response(42, -32601, "Method not found")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 42
        assert resp["error"]["code"] == -32601
        assert resp["error"]["message"] == "Method not found"


# -- Transport Layer --


def _make_stdin(line: str) -> io.StringIO:
    """Build a mock stdin with a newline-delimited JSON line."""
    return io.StringIO(line + "\n")


class TestReadMessageBombGuard:
    """Tests for message size bomb guard."""

    def test_oversized_message_rejected(self):
        huge_line = '{"method":"x","x":"' + "a" * (_MAX_MESSAGE_SIZE + 1) + '"}'
        with patch("sys.stdin", _make_stdin(huge_line)):
            result = _read_message()
        assert isinstance(result, str)
        assert "too large" in result.lower()

    def test_exactly_max_size_accepted(self):
        """Exactly _MAX_MESSAGE_SIZE should not trigger the guard."""
        base = json.dumps({"jsonrpc": "2.0", "method": "ping"})
        padding_needed = _MAX_MESSAGE_SIZE - len(base) - 1  # -1 for closing }
        # Pad with a long string value inside valid JSON
        padded = '{"jsonrpc": "2.0", "method": "ping", "pad": "' + "a" * (padding_needed - len(', "pad": ""')) + '"}'
        assert len(padded) <= _MAX_MESSAGE_SIZE
        with patch("sys.stdin", _make_stdin(padded)):
            result = _read_message()
        assert isinstance(result, dict)
        assert result["method"] == "ping"


class TestNotificationHandling:
    """Tests for MCP notification path (messages without id)."""

    def test_notification_produces_no_response(self, server, store):
        """A JSON-RPC notification (no id field) should not produce a response."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        mock_stdin = io.StringIO(json.dumps(notification) + "\n")
        output = io.StringIO()

        with patch("sys.stdin", mock_stdin):
            with patch("sys.stdout", output):
                server.run()

        # No output should have been written for the notification
        assert output.getvalue() == ""


# -- Spore tools (prospective-intention layer) --


def _call(server, name, arguments=None):
    """Drive a tool through the real dispatch path (exercises the
    exception -> is_error conversion in _handle_tools_call)."""
    return server._handle_tools_call({"name": name, "arguments": arguments or {}})


class TestSporeTools:
    def test_all_spore_tools_registered_and_in_integrity(self, server):
        names = {t["name"] for t in TOOLS}
        expected = {
            "spore_add", "spore_get", "spore_list", "spore_touch",
            "spore_update", "spore_descend", "spore_ascend", "spore_surface",
        }
        assert expected <= names
        assert expected <= set(server._tool_handlers)

    def test_spore_store_is_sibling_of_db(self, server, store):
        # <stem>.spores.json beside the episodic db, mirroring continuity.
        assert server._spore_store.path == store.path.parent / f"{store.path.stem}.spores.json"

    def test_add_then_list_then_get(self, server):
        r = _call(server, "spore_add", {"type": "task", "text": "ship the MCP surface", "tier": "hot", "salience": 2})
        assert not _is_error(r)
        assert "spore-001" in _text_from_result(r)

        r = _call(server, "spore_list", {})
        assert not _is_error(r)
        assert "ship the MCP surface" in _text_from_result(r)

        r = _call(server, "spore_get", {"spore_id": "spore-001"})
        assert not _is_error(r)
        body = json.loads(_text_from_result(r))
        assert body["id"] == "spore-001"
        assert body["tier"] == "hot"
        # get annotates computed germination (the tool description promises it).
        assert body["germination"] == "growing"

    def test_list_invalid_enum_is_error_not_empty(self, server):
        _call(server, "spore_add", {"type": "task", "text": "real task"})
        # A typo'd filter must error loudly, not silently return "no matches".
        r = _call(server, "spore_list", {"type": "tasks"})
        assert _is_error(r)
        assert "invalid type" in _text_from_result(r)

    def test_surface_non_boolean_top_of_mind_does_not_enable_tom(self, server):
        _call(server, "spore_add", {"type": "task", "text": "hot one", "tier": "hot"})
        _call(server, "spore_add", {"type": "task", "text": "parked one", "tier": "parked"})
        # The string "false" is truthy in Python; `is True` must keep it OFF, so
        # this returns ALL open spores (including the parked one), not the ToM subset.
        r = _call(server, "spore_surface", {"top_of_mind": "false"})
        assert not _is_error(r)
        text = _text_from_result(r)
        assert "hot one" in text and "parked one" in text

    def test_add_invalid_salience_is_error(self, server):
        # 2.9 must NOT be silently coerced to 2 (no int() masking in the tool).
        assert _is_error(_call(server, "spore_add", {"type": "task", "text": "x", "salience": 2.9}))

    def test_add_invalid_type_is_error(self, server):
        r = _call(server, "spore_add", {"type": "nonsense", "text": "x"})
        assert _is_error(r)

    def test_add_missing_text_is_error(self, server):
        r = _call(server, "spore_add", {"type": "task"})
        assert _is_error(r)

    def test_get_missing_and_nonexistent(self, server):
        assert _is_error(_call(server, "spore_get", {}))
        assert _is_error(_call(server, "spore_get", {"spore_id": "spore-999"}))

    def test_touch_updates_seen(self, server):
        _call(server, "spore_add", {"type": "task", "text": "t"})
        r = _call(server, "spore_touch", {"spore_id": "spore-001"})
        assert not _is_error(r)
        assert "Touched spore-001" in _text_from_result(r)

    def test_update_only_passes_given_fields(self, server):
        _call(server, "spore_add", {"type": "question", "text": "q", "domain": "design"})
        r = _call(server, "spore_update", {"spore_id": "spore-001", "tier": "hot"})
        assert not _is_error(r)
        # domain left intact (not cleared) because it wasn't passed
        body = json.loads(_text_from_result(_call(server, "spore_get", {"spore_id": "spore-001"})))
        assert body["tier"] == "hot"
        assert body["domain"] == "design"

    def test_update_no_fields_is_error(self, server):
        _call(server, "spore_add", {"type": "task", "text": "t"})
        assert _is_error(_call(server, "spore_update", {"spore_id": "spore-001"}))

    def test_update_clears_with_empty_string(self, server):
        _call(server, "spore_add", {"type": "task", "text": "t", "pointer": "p/x"})
        _call(server, "spore_update", {"spore_id": "spore-001", "pointer": ""})
        body = json.loads(_text_from_result(_call(server, "spore_get", {"spore_id": "spore-001"})))
        assert body["pointer"] is None

    def test_descend_valid_and_invalid_kind(self, server):
        _call(server, "spore_add", {"type": "task", "text": "t"})
        # 'answered' is a question-kind, invalid for a task
        assert _is_error(_call(server, "spore_descend", {"spore_id": "spore-001", "kind": "answered"}))
        r = _call(server, "spore_descend", {"spore_id": "spore-001", "kind": "done"})
        assert not _is_error(r)
        assert "resolved down" in _text_from_result(r)
        # resolved spore no longer open
        assert "No open spores" in _text_from_result(_call(server, "spore_list", {}))

    def test_ascend_records_ref_and_requires_ref(self, server):
        _call(server, "spore_add", {"type": "thought", "text": "an idea"})
        assert _is_error(_call(server, "spore_ascend", {"spore_id": "spore-001", "kind": "essay"}))
        r = _call(server, "spore_ascend", {"spore_id": "spore-001", "kind": "essay", "ref": "essays/idea.md"})
        assert not _is_error(r)
        assert "essays/idea.md" in _text_from_result(r)

    def test_surface_top_of_mind(self, server):
        _call(server, "spore_add", {"type": "task", "text": "hot one", "tier": "hot"})
        # A freshly-added spore is 'growing' (seen today) regardless of tier, so to
        # test exclusion use 'parked' — germination='parked', tier != hot.
        _call(server, "spore_add", {"type": "task", "text": "parked one", "tier": "parked"})
        r = _call(server, "spore_surface", {"top_of_mind": True})
        assert not _is_error(r)
        text = _text_from_result(r)
        assert "hot one" in text           # tier == hot -> included
        assert "parked one" not in text    # parked: not hot, not growing -> excluded
