"""Tests for the CLI interface.

Tests the CLI as a module — calls the subcommand handlers directly
with parsed args, or invokes via subprocess for integration tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from argparse import Namespace
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from anneal_memory import Store, __version__
from anneal_memory.cli import (
    build_parser,
    cmd_associations,
    cmd_audit,
    cmd_continuity,
    cmd_delete,
    cmd_diff,
    cmd_episodes,
    cmd_export,
    cmd_get,
    cmd_graph,
    cmd_history,
    cmd_import,
    cmd_init,
    cmd_prepare_wrap,
    cmd_prune,
    cmd_record,
    cmd_save_continuity,
    cmd_search,
    cmd_stats,
    cmd_status,
    cmd_verify,
    cmd_wrap_cancel,
    cmd_wrap_status,
    cmd_wrap_token_current,
    main,
    parse_duration,
)
from anneal_memory.store import StoreError


# -- Fixtures --

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp DB path and return it."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store_with_data(tmp_db):
    """Create a store with some test episodes."""
    store = Store(tmp_db, project_name="TestProject")
    store.record("Found a pattern in the data", episode_type="observation", source="agent")
    store.record("Decided to use SQLite", episode_type="decision", source="agent")
    store.record("Is Redis better for this?", episode_type="question", source="cli")
    store.record("The test passed successfully", episode_type="outcome", source="test-runner")
    store.close()
    return tmp_db


@pytest.fixture
def base_args(tmp_db):
    """Base args namespace with common fields."""
    return Namespace(db=tmp_db, project_name="Agent", json=False)


@pytest.fixture
def base_args_with_data(store_with_data):
    """Base args namespace pointing to a store with data."""
    return Namespace(db=store_with_data, project_name="TestProject", json=False)


# -- parse_duration tests --

class TestParseDuration:
    def test_seconds(self):
        result = parse_duration("30s")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        assert abs((now - ts).total_seconds() - 30) < 2

    def test_minutes(self):
        result = parse_duration("5m")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        assert abs((now - ts).total_seconds() - 300) < 2

    def test_hours(self):
        result = parse_duration("24h")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        assert abs((now - ts).total_seconds() - 86400) < 2

    def test_days(self):
        result = parse_duration("3d")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - ts
        assert abs(delta.days - 3) <= 1

    def test_weeks(self):
        result = parse_duration("1w")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - ts
        assert abs(delta.days - 7) <= 1

    def test_iso_timestamp_passthrough(self):
        ts = "2026-04-01T12:00:00Z"
        assert parse_duration(ts) == ts

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("abc")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("3x")

    def test_case_insensitive(self):
        result = parse_duration("3D")
        ts = datetime.fromisoformat(result.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - ts
        assert abs(delta.days - 3) <= 1


# -- cmd_init tests --

class TestCmdInit:
    def test_init_creates_store(self, base_args, capsys):
        cmd_init(base_args)
        captured = capsys.readouterr()
        assert "Initialized anneal-memory store" in captured.out
        assert Path(base_args.db).exists()

    def test_init_already_exists(self, store_with_data, capsys):
        args = Namespace(db=store_with_data, project_name="Agent", json=False)
        with pytest.raises(SystemExit):
            cmd_init(args)

    def test_init_with_project_name(self, base_args, capsys):
        base_args.project_name = "MyProject"
        cmd_init(base_args)
        captured = capsys.readouterr()
        assert "MyProject" in captured.out


# -- cmd_status tests --

class TestCmdStatus:
    def test_status_empty_store(self, base_args, capsys):
        # Create the store first
        store = Store(base_args.db)
        store.close()
        cmd_status(base_args)
        captured = capsys.readouterr()
        assert "Episodes:   0 total" in captured.out
        assert "Wraps:      0 total" in captured.out

    def test_status_with_data(self, base_args_with_data, capsys):
        cmd_status(base_args_with_data)
        captured = capsys.readouterr()
        assert "Episodes:   4 total" in captured.out
        assert "observation" in captured.out

    def test_status_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        cmd_status(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_episodes"] == 4
        assert "episodes_by_type" in data

    def test_status_shows_version(self, base_args_with_data, capsys):
        cmd_status(base_args_with_data)
        captured = capsys.readouterr()
        assert __version__ in captured.out


# -- cmd_episodes tests --

class TestCmdEpisodes:
    def test_list_all(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "4 total matching" in captured.out

    def test_filter_by_type(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = "decision"
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 total matching" in captured.out
        assert "SQLite" in captured.out

    def test_filter_by_source(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = "cli"
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 total matching" in captured.out
        assert "Redis" in captured.out

    def test_since_filter(self, base_args_with_data, capsys):
        base_args_with_data.since = "1h"
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "4 total matching" in captured.out

    def test_episodes_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_matching"] == 4
        assert len(data["episodes"]) == 4

    def test_empty_results(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = "tension"
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "No episodes found" in captured.out

    def test_limit_and_offset(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 2
        base_args_with_data.offset = 0
        base_args_with_data.json = True
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["episodes"]) == 2
        assert data["total_matching"] == 4


# -- cmd_record tests --

class TestCmdRecord:
    def test_record_basic(self, base_args, capsys):
        # Init store first
        Store(base_args.db).close()
        base_args.content = "Test observation"
        base_args.type = "observation"
        base_args.source = "cli"
        base_args.tags = None
        cmd_record(base_args)
        captured = capsys.readouterr()
        assert "Recorded episode" in captured.out
        assert "observation" in captured.out

    def test_record_with_tags(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.content = "Tagged observation"
        base_args.type = "observation"
        base_args.source = "cli"
        base_args.tags = "test,experiment"
        cmd_record(base_args)
        captured = capsys.readouterr()
        assert "Recorded episode" in captured.out

        # Verify tags stored
        with Store(base_args.db) as store:
            result = store.recall(limit=1)
            assert result.episodes[0].metadata == {"tags": ["test", "experiment"]}

    def test_record_json(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.json = True
        base_args.content = "JSON record test"
        base_args.type = "decision"
        base_args.source = "test"
        base_args.tags = None
        cmd_record(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["type"] == "decision"
        assert data["source"] == "test"
        assert "id" in data

    def test_record_from_stdin(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.content = "-"
        base_args.type = "observation"
        base_args.source = "cli"
        base_args.tags = None
        with mock.patch("sys.stdin", StringIO("Content from stdin")):
            cmd_record(base_args)
        captured = capsys.readouterr()
        assert "Recorded episode" in captured.out

    def test_record_different_types(self, base_args, capsys):
        Store(base_args.db).close()
        for ep_type in ["observation", "decision", "tension", "question", "outcome", "context"]:
            base_args.content = f"Test {ep_type}"
            base_args.type = ep_type
            base_args.source = "cli"
            base_args.tags = None
            cmd_record(base_args)
        captured = capsys.readouterr()
        assert captured.out.count("Recorded episode") == 6


# -- cmd_search tests --

class TestCmdSearch:
    @staticmethod
    def _set_search_args(args, query, **kwargs):
        args.query = query
        args.since = kwargs.get("since")
        args.type = kwargs.get("type")
        args.source = kwargs.get("source")
        args.limit = kwargs.get("limit", 20)

    def test_search_found(self, base_args_with_data, capsys):
        self._set_search_args(base_args_with_data, "pattern")
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 episodes matching" in captured.out
        assert "pattern" in captured.out.lower()

    def test_search_not_found(self, base_args_with_data, capsys):
        self._set_search_args(base_args_with_data, "nonexistent_xyz")
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        assert "No episodes matching" in captured.out

    def test_search_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        self._set_search_args(base_args_with_data, "SQLite")
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_matching"] == 1
        assert "SQLite" in data["episodes"][0]["content"]

    def test_search_case_insensitive(self, base_args_with_data, capsys):
        self._set_search_args(base_args_with_data, "sqlite")
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 episodes matching" in captured.out


# -- cmd_continuity tests --

class TestCmdContinuity:
    def test_no_continuity_yet(self, base_args, capsys):
        Store(base_args.db).close()
        with pytest.raises(SystemExit):
            cmd_continuity(base_args)

    def test_continuity_exists(self, base_args, capsys):
        with Store(base_args.db) as store:
            store.save_continuity("## State\nTest continuity content\n## Patterns\n\n## Decisions\n\n## Context\n")
        cmd_continuity(base_args)
        captured = capsys.readouterr()
        assert "Test continuity content" in captured.out

    def test_continuity_json(self, base_args, capsys):
        text = "## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n"
        with Store(base_args.db) as store:
            store.save_continuity(text)
        base_args.json = True
        cmd_continuity(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["chars"] == len(text)
        assert "text" in data


# -- cmd_associations tests --

class TestCmdAssociations:
    def test_stats_empty(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.stats = True
        base_args.episode = None
        base_args.min_strength = 0.0
        base_args.limit = 20
        cmd_associations(base_args)
        captured = capsys.readouterr()
        assert "Total links:     0" in captured.out

    def test_stats_json(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.json = True
        base_args.stats = True
        base_args.episode = None
        base_args.min_strength = 0.0
        base_args.limit = 20
        cmd_associations(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_links"] == 0

    def test_episode_no_associations(self, base_args_with_data, capsys):
        # Get an episode ID
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            result = store.recall(limit=1)
            ep_id = result.episodes[0].id

        base_args_with_data.stats = False
        base_args_with_data.episode = ep_id
        base_args_with_data.min_strength = 0.0
        base_args_with_data.limit = 20
        cmd_associations(base_args_with_data)
        captured = capsys.readouterr()
        assert "No associations found" in captured.out

    def test_no_episode_no_stats_errors(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.stats = False
        base_args.episode = None
        base_args.min_strength = 0.0
        base_args.limit = 20
        with pytest.raises(SystemExit):
            cmd_associations(base_args)


# -- cmd_delete tests --

class TestCmdDelete:
    def test_delete_with_force(self, base_args_with_data, capsys):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            result = store.recall(limit=1)
            ep_id = result.episodes[0].id

        base_args_with_data.episode_id = ep_id
        base_args_with_data.force = True
        cmd_delete(base_args_with_data)
        captured = capsys.readouterr()
        assert f"Deleted episode {ep_id}" in captured.out

        # Verify deletion
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            assert store.get(ep_id) is None

    def test_delete_nonexistent(self, base_args_with_data, capsys):
        base_args_with_data.episode_id = "00000000"
        base_args_with_data.force = True
        with pytest.raises(SystemExit):
            cmd_delete(base_args_with_data)

    def test_delete_json(self, base_args_with_data, capsys):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            result = store.recall(limit=1)
            ep_id = result.episodes[0].id

        base_args_with_data.json = True
        base_args_with_data.episode_id = ep_id
        base_args_with_data.force = True
        cmd_delete(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["deleted"] == ep_id


# -- cmd_verify tests --

class TestCmdVerify:
    def test_verify_valid(self, base_args_with_data, capsys):
        cmd_verify(base_args_with_data)
        captured = capsys.readouterr()
        assert "Audit trail valid" in captured.out

    def test_verify_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        cmd_verify(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["valid"] is True
        assert data["total_entries"] > 0

    def test_verify_empty_store(self, base_args, capsys):
        Store(base_args.db).close()
        cmd_verify(base_args)
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower()


# -- Parser tests --

class TestParser:
    def test_version(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_status_command(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_episodes_with_filters(self):
        parser = build_parser()
        args = parser.parse_args(["episodes", "--since", "3d", "--until", "1d", "--type", "observation", "--keyword", "test", "--limit", "10"])
        assert args.command == "episodes"
        assert args.since == "3d"
        assert args.until == "1d"
        assert args.type == "observation"
        assert args.keyword == "test"
        assert args.limit == 10

    def test_get_command(self):
        parser = build_parser()
        args = parser.parse_args(["get", "abc123"])
        assert args.command == "get"
        assert args.episode_id == "abc123"

    def test_prune_command(self):
        parser = build_parser()
        args = parser.parse_args(["prune", "--older-than", "90", "--dry-run"])
        assert args.command == "prune"
        assert args.older_than == 90
        assert args.dry_run is True

    def test_search_with_filters(self):
        parser = build_parser()
        args = parser.parse_args(["search", "query", "--since", "3d", "--type", "observation", "--source", "cli"])
        assert args.since == "3d"
        assert args.type == "observation"
        assert args.source == "cli"

    def test_json_on_subparser(self):
        """--json works at subcommand level for all commands with it."""
        parser = build_parser()
        for cmd in ["status", "episodes", "get abc123", "continuity", "verify", "associations --stats"]:
            args_list = cmd.split() + ["--json"]
            args = parser.parse_args(args_list)
            assert args.json is True, f"--json failed for subcommand: {cmd}"

    def test_record_command(self):
        parser = build_parser()
        args = parser.parse_args(["record", "Test content", "--type", "decision"])
        assert args.command == "record"
        assert args.content == "Test content"
        assert args.type == "decision"

    def test_search_command(self):
        parser = build_parser()
        args = parser.parse_args(["search", "test query"])
        assert args.command == "search"
        assert args.query == "test query"

    def test_delete_with_force(self):
        parser = build_parser()
        args = parser.parse_args(["delete", "abc123", "--force"])
        assert args.command == "delete"
        assert args.episode_id == "abc123"
        assert args.force is True

    def test_db_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--db", "/tmp/test.db", "status"])
        assert args.db == "/tmp/test.db"
        assert args.command == "status"

    def test_json_flag_top_level(self):
        parser = build_parser()
        args = parser.parse_args(["--json", "status"])
        assert args.json is True

    def test_json_flag_subcommand_level(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--json"])
        assert args.json is True

    def test_init_command(self):
        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_associations_stats(self):
        parser = build_parser()
        args = parser.parse_args(["associations", "--stats"])
        assert args.command == "associations"
        assert args.stats is True

    def test_associations_episode(self):
        parser = build_parser()
        args = parser.parse_args(["associations", "--episode", "abc123", "--min-strength", "0.5"])
        assert args.episode == "abc123"
        assert args.min_strength == 0.5

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--skip-integrity", "--no-audit"])
        assert args.command == "serve"
        assert args.skip_integrity is True
        assert args.no_audit is True


# -- Integration tests (main entry point) --

class TestMainEntryPoint:
    def test_main_status(self, store_with_data, capsys):
        with mock.patch("sys.argv", ["anneal-memory", "--db", store_with_data, "status"]):
            main()
        captured = capsys.readouterr()
        assert "Episodes:   4 total" in captured.out

    def test_main_episodes_json(self, store_with_data, capsys):
        with mock.patch("sys.argv", ["anneal-memory", "--db", store_with_data, "episodes", "--json"]):
            main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_matching"] == 4

    def test_main_record_and_search(self, store_with_data, capsys):
        with mock.patch("sys.argv", [
            "anneal-memory", "--db", store_with_data,
            "record", "Unique test content xyz123", "--type", "observation",
        ]):
            main()

        with mock.patch("sys.argv", [
            "anneal-memory", "--db", store_with_data,
            "search", "xyz123",
        ]):
            main()
        captured = capsys.readouterr()
        assert "xyz123" in captured.out


# -- Subprocess integration test --

class TestSubprocess:
    def test_cli_version(self):
        """Test the CLI works as a real subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_cli_init_and_status(self, tmp_path):
        """Test init + status as real subprocess calls."""
        db_path = str(tmp_path / "test.db")

        # Init
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", db_path, "init"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Initialized" in result.stdout

        # Status
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", db_path, "status"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Episodes:   0 total" in result.stdout

    def test_cli_record_and_search(self, tmp_path):
        """Test record + search round-trip as real subprocess calls."""
        db_path = str(tmp_path / "test.db")

        # Init
        subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", db_path, "init"],
            capture_output=True,
        )

        # Record
        result = subprocess.run(
            [
                sys.executable, "-m", "anneal_memory.cli",
                "--db", db_path,
                "record", "The subprocess test worked perfectly",
                "--type", "outcome",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Recorded episode" in result.stdout

        # Search
        result = subprocess.run(
            [
                sys.executable, "-m", "anneal_memory.cli",
                "--db", db_path,
                "search", "subprocess",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "subprocess" in result.stdout.lower()

    def test_cli_json_output(self, tmp_path):
        """Test JSON output works in subprocess."""
        db_path = str(tmp_path / "test.db")

        subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", db_path, "init"],
            capture_output=True,
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "anneal_memory.cli",
                "--db", db_path,
                "status", "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["total_episodes"] == 0

    def test_cli_verify(self, tmp_path):
        """Test verify command in subprocess."""
        db_path = str(tmp_path / "test.db")

        subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", db_path, "init"],
            capture_output=True,
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "anneal_memory.cli",
                "--db", db_path,
                "verify",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_python_m_package(self, tmp_path):
        """Test python -m anneal_memory works via __main__.py."""
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert __version__ in result.stdout


# -- cmd_get tests --

class TestCmdGet:
    def test_get_episode(self, base_args_with_data, capsys):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            result = store.recall(limit=1)
            ep_id = result.episodes[0].id
            ep_content = result.episodes[0].content

        base_args_with_data.episode_id = ep_id
        cmd_get(base_args_with_data)
        captured = capsys.readouterr()
        assert f"Episode {ep_id}" in captured.out
        assert ep_content in captured.out

    def test_get_nonexistent(self, base_args_with_data, capsys):
        base_args_with_data.episode_id = "00000000"
        with pytest.raises(SystemExit):
            cmd_get(base_args_with_data)

    def test_get_json(self, base_args_with_data, capsys):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            result = store.recall(limit=1)
            ep_id = result.episodes[0].id

        base_args_with_data.json = True
        base_args_with_data.episode_id = ep_id
        cmd_get(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["id"] == ep_id
        assert "content" in data
        assert "type" in data


# -- cmd_prune tests --

class TestCmdPrune:
    def test_prune_nothing(self, base_args_with_data, capsys):
        """All episodes are recent — nothing to prune."""
        base_args_with_data.older_than = 1
        base_args_with_data.dry_run = False
        cmd_prune(base_args_with_data)
        captured = capsys.readouterr()
        assert "Pruned 0 episodes" in captured.out

    def test_prune_dry_run(self, base_args_with_data, capsys):
        base_args_with_data.older_than = 1
        base_args_with_data.dry_run = True
        cmd_prune(base_args_with_data)
        captured = capsys.readouterr()
        assert "Would prune" in captured.out

    def test_prune_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.older_than = 1
        base_args_with_data.dry_run = False
        cmd_prune(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "pruned" in data
        assert data["older_than_days"] == 1


# -- Episodes extended filters tests --

class TestEpisodesExtendedFilters:
    def test_keyword_filter(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.until = None
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = "SQLite"
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 total matching" in captured.out

    def test_until_filter(self, base_args_with_data, capsys):
        """--until with a far future time should include all episodes."""
        base_args_with_data.since = None
        base_args_with_data.until = "2099-01-01T00:00:00Z"
        base_args_with_data.type = None
        base_args_with_data.source = None
        base_args_with_data.keyword = None
        base_args_with_data.limit = 50
        base_args_with_data.offset = 0
        cmd_episodes(base_args_with_data)
        captured = capsys.readouterr()
        assert "4 total matching" in captured.out


# -- Search with filters tests --

class TestSearchWithFilters:
    def test_search_with_type_filter(self, base_args_with_data, capsys):
        base_args_with_data.query = "Redis"
        base_args_with_data.since = None
        base_args_with_data.type = "question"
        base_args_with_data.source = None
        base_args_with_data.limit = 20
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        assert "1 episodes matching" in captured.out

    def test_search_with_type_filter_no_match(self, base_args_with_data, capsys):
        base_args_with_data.query = "Redis"
        base_args_with_data.since = None
        base_args_with_data.type = "decision"
        base_args_with_data.source = None
        base_args_with_data.limit = 20
        cmd_search(base_args_with_data)
        captured = capsys.readouterr()
        assert "No episodes matching" in captured.out


# -- Backward compatibility tests --

class TestBackwardCompat:
    def test_no_subcommand_parses_correctly(self):
        """When no subcommand given, parse_known_args should detect it."""
        parser = build_parser()
        args, remaining = parser.parse_known_args([])
        assert args.command is None

    def test_server_flags_without_subcommand_dont_error(self):
        """Server-specific flags without 'serve' should not crash parse_known_args."""
        parser = build_parser()
        args, remaining = parser.parse_known_args(["--no-audit", "--skip-integrity"])
        assert args.command is None
        # These flags are unknown to the CLI parser — they'll be in remaining
        assert "--no-audit" in remaining
        assert "--skip-integrity" in remaining

    def test_no_subcommand_calls_server_main(self):
        """main() with no subcommand should delegate to server.main()."""
        with mock.patch("sys.argv", ["anneal-memory"]):
            with mock.patch("anneal_memory.server.main") as mock_server:
                main()
                mock_server.assert_called_once()


# -- Init JSON tests --

class TestCmdInitJson:
    def test_init_json(self, base_args, capsys):
        base_args.json = True
        cmd_init(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "database" in data
        assert "project" in data


# -- Environment variable tests --

class TestEnvVars:
    def test_anneal_memory_db_env(self):
        with mock.patch.dict(os.environ, {"ANNEAL_MEMORY_DB": "/tmp/custom.db"}):
            from anneal_memory.cli import _default_db
            assert _default_db() == "/tmp/custom.db"

    def test_default_db_without_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            from anneal_memory.cli import _default_db
            result = _default_db()
            assert "anneal-memory" in result
            assert "memory.db" in result

    def test_anneal_memory_source_env(self):
        """ANNEAL_MEMORY_SOURCE env var sets default source for record."""
        with mock.patch.dict(os.environ, {"ANNEAL_MEMORY_SOURCE": "ci-pipeline"}):
            parser = build_parser()
            args = parser.parse_args(["record", "test content"])
            assert args.source == "ci-pipeline"

    def test_source_flag_overrides_env(self):
        """--source flag overrides ANNEAL_MEMORY_SOURCE env var."""
        with mock.patch.dict(os.environ, {"ANNEAL_MEMORY_SOURCE": "ci-pipeline"}):
            parser = build_parser()
            args = parser.parse_args(["record", "test content", "--source", "manual"])
            assert args.source == "manual"


# -- Record stdin empty tests --

class TestRecordStdinEmpty:
    def test_empty_stdin_errors(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.content = "-"
        base_args.type = "observation"
        base_args.source = "cli"
        base_args.tags = None
        with mock.patch("sys.stdin", StringIO("")):
            with pytest.raises(SystemExit):
                cmd_record(base_args)




# -- cmd_export tests --

class TestCmdExport:
    def test_export_json_stdout(self, base_args_with_data, capsys):
        base_args_with_data.format = "json"
        base_args_with_data.output = None
        cmd_export(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["anneal_memory_export"] is True
        assert data["format_version"] == 1
        assert len(data["episodes"]) == 4
        assert data["project_name"] == "TestProject"
        assert "continuity" in data
        assert "wraps" in data

    def test_export_json_to_file(self, base_args_with_data, tmp_path, capsys):
        out = str(tmp_path / "export.json")
        base_args_with_data.format = "json"
        base_args_with_data.output = out
        cmd_export(base_args_with_data)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert len(data["episodes"]) == 4

    def test_export_json_to_file_json_mode(self, base_args_with_data, tmp_path, capsys):
        out = str(tmp_path / "export.json")
        base_args_with_data.json = True
        base_args_with_data.format = "json"
        base_args_with_data.output = out
        cmd_export(base_args_with_data)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert result["format"] == "json"
        assert result["episodes"] == 4

    def test_export_markdown_stdout(self, base_args_with_data, capsys):
        base_args_with_data.format = "markdown"
        base_args_with_data.output = None
        cmd_export(base_args_with_data)
        captured = capsys.readouterr()
        assert "# anneal-memory Export" in captured.out
        assert "## Episodes" in captured.out
        assert "Found a pattern" in captured.out

    def test_export_markdown_to_file(self, base_args_with_data, tmp_path, capsys):
        out = str(tmp_path / "export.md")
        base_args_with_data.format = "markdown"
        base_args_with_data.output = out
        cmd_export(base_args_with_data)
        assert Path(out).exists()
        text = Path(out).read_text()
        assert "## Episodes" in text

    def test_export_sqlite(self, base_args_with_data, tmp_path, capsys):
        out = str(tmp_path / "copy.db")
        base_args_with_data.format = "sqlite"
        base_args_with_data.output = out
        cmd_export(base_args_with_data)
        assert Path(out).exists()
        # Verify the copy is a valid DB with same data
        with Store(out, project_name="TestProject") as s:
            assert s.status().total_episodes == 4

    def test_export_sqlite_json(self, base_args_with_data, tmp_path, capsys):
        out = str(tmp_path / "copy.db")
        base_args_with_data.json = True
        base_args_with_data.format = "sqlite"
        base_args_with_data.output = out
        cmd_export(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["format"] == "sqlite"
        assert data["size_bytes"] > 0


# -- cmd_import tests --

class TestCmdImport:
    def _make_export(self, store_path, tmp_path):
        """Create a JSON export file from a store."""
        with Store(store_path, project_name="TestProject") as store:
            result = store.recall(limit=100)
            episodes = [
                {
                    "id": ep.id,
                    "timestamp": ep.timestamp,
                    "type": ep.type.value,
                    "content": ep.content,
                    "source": ep.source,
                    "metadata": ep.metadata,
                }
                for ep in result.episodes
            ]
        export_path = tmp_path / "export.json"
        export_path.write_text(json.dumps({
            "anneal_memory_export": True,
            "episodes": episodes,
        }))
        return str(export_path)

    def test_import_into_new_store(self, store_with_data, tmp_path, capsys):
        export_path = self._make_export(store_with_data, tmp_path)
        new_db = str(tmp_path / "new.db")
        Store(new_db).close()  # init empty store

        args = Namespace(db=new_db, project_name="Agent", json=False, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        assert "4 imported" in captured.out

        with Store(new_db) as store:
            assert store.status().total_episodes == 4

    def test_import_skips_duplicates(self, store_with_data, tmp_path, capsys):
        export_path = self._make_export(store_with_data, tmp_path)
        # Import into the SAME store — all should be skipped
        args = Namespace(db=store_with_data, project_name="TestProject", json=False, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        assert "4 skipped" in captured.out
        assert "0 imported" in captured.out

    def test_import_json_output(self, store_with_data, tmp_path, capsys):
        export_path = self._make_export(store_with_data, tmp_path)
        new_db = str(tmp_path / "new.db")
        Store(new_db).close()

        args = Namespace(db=new_db, project_name="Agent", json=True, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["imported"] == 4
        assert data["skipped"] == 0

    def test_import_invalid_file(self, base_args, tmp_path, capsys):
        bad_path = str(tmp_path / "bad.json")
        Path(bad_path).write_text('{"not_an_export": true}')
        Store(base_args.db).close()

        args = Namespace(db=base_args.db, project_name="Agent", json=False, path=bad_path)
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_import_file_not_found(self, base_args, capsys):
        args = Namespace(db=base_args.db, project_name="Agent", json=False, path="/nonexistent.json")
        with pytest.raises(SystemExit):
            cmd_import(args)

    def test_import_empty_episodes(self, base_args, tmp_path, capsys):
        export_path = str(tmp_path / "empty.json")
        Path(export_path).write_text(json.dumps({"anneal_memory_export": True, "episodes": []}))
        Store(base_args.db).close()

        args = Namespace(db=base_args.db, project_name="Agent", json=False, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        assert "No episodes to import" in captured.out


# -- cmd_audit tests --

class TestCmdAudit:
    def test_audit_shows_entries(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.event = None
        base_args_with_data.limit = 50
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        assert "Audit trail:" in captured.out
        assert "record" in captured.out

    def test_audit_filter_by_event(self, base_args_with_data, capsys):
        base_args_with_data.since = None
        base_args_with_data.event = "record"
        base_args_with_data.limit = 50
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        assert "record" in captured.out

    def test_audit_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.since = None
        base_args_with_data.event = None
        base_args_with_data.limit = 50
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "entries" in data
        assert data["total"] > 0
        assert all(e["event"] for e in data["entries"])

    def test_audit_filter_by_event_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.since = None
        base_args_with_data.event = "record"
        base_args_with_data.limit = 50
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert all(e["event"] == "record" for e in data["entries"])

    def test_audit_limit(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.since = None
        base_args_with_data.event = None
        base_args_with_data.limit = 2
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["entries"]) <= 2
        assert data["total"] >= len(data["entries"])

    def test_audit_empty_store(self, base_args, capsys):
        # No audit files yet
        base_args.since = None
        base_args.event = None
        base_args.limit = 50
        cmd_audit(base_args)
        captured = capsys.readouterr()
        assert "No audit trail files" in captured.out

    def test_audit_since_filter(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.since = "1h"
        base_args_with_data.event = None
        base_args_with_data.limit = 50
        cmd_audit(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # All entries should be recent
        assert data["total"] > 0


# -- cmd_diff tests --

class TestCmdDiff:
    def test_diff_no_wraps(self, base_args_with_data, capsys):
        base_args_with_data.wraps = 5
        cmd_diff(base_args_with_data)
        captured = capsys.readouterr()
        assert "No wraps to compare" in captured.out

    def test_diff_no_wraps_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.wraps = 5
        cmd_diff(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["wraps"] == []

    def test_diff_with_wraps(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            store.record("ep1", episode_type="observation")
            store.wrap_completed(1, 100)
            store.record("ep2", episode_type="observation")
            store.wrap_completed(1, 150, graduations_validated=1)

        args = Namespace(db=db, project_name="Agent", json=False, wraps=5)
        cmd_diff(args)
        captured = capsys.readouterr()
        assert "Wrap progression" in captured.out
        assert "(+50)" in captured.out  # chars delta

    def test_diff_with_wraps_json(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            store.record("ep1", episode_type="observation")
            store.wrap_completed(1, 100)
            store.record("ep2", episode_type="observation")
            store.wrap_completed(1, 200)

        args = Namespace(db=db, project_name="Agent", json=True, wraps=5)
        cmd_diff(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["wraps"]) == 2
        assert "delta" in data["wraps"][1]
        assert data["wraps"][1]["delta"]["continuity_chars"] == 100


# -- cmd_graph tests --

class TestCmdGraph:
    def test_graph_empty(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.format = "json"
        base_args_with_data.output = None
        base_args_with_data.min_strength = 0.0
        cmd_graph(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_graph_with_associations(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            ep1 = store.record("Episode one", episode_type="observation")
            ep2 = store.record("Episode two", episode_type="decision")
            store.record_associations(
                direct_pairs={(ep1.id, ep2.id)},
            )

        args = Namespace(db=db, project_name="Agent", json=True, format="json", output=None, min_strength=0.0)
        cmd_graph(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_graph_dot_format(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            ep1 = store.record("Episode one", episode_type="observation")
            ep2 = store.record("Episode two", episode_type="decision")
            store.record_associations(
                direct_pairs={(ep1.id, ep2.id)},
            )

        args = Namespace(db=db, project_name="Agent", json=False, format="dot", output=None, min_strength=0.0)
        cmd_graph(args)
        captured = capsys.readouterr()
        assert "graph associations" in captured.out
        assert ep1.id in captured.out
        assert ep2.id in captured.out

    def test_graph_dot_to_file(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            ep1 = store.record("Episode one", episode_type="observation")
            ep2 = store.record("Episode two", episode_type="decision")
            store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        out = str(tmp_path / "graph.dot")
        args = Namespace(db=db, project_name="Agent", json=False, format="dot", output=out, min_strength=0.0)
        cmd_graph(args)
        assert Path(out).exists()
        text = Path(out).read_text()
        assert "graph associations" in text

    def test_graph_min_strength_filter(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            ep1 = store.record("Episode one", episode_type="observation")
            ep2 = store.record("Episode two", episode_type="decision")
            store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        # Set min_strength higher than any edge
        args = Namespace(db=db, project_name="Agent", json=False, format="json", output=None, min_strength=99.0)
        cmd_graph(args)
        captured = capsys.readouterr()
        assert "No associations above strength" in captured.out


# -- cmd_stats tests --

class TestCmdStats:
    def test_stats_empty(self, base_args, capsys):
        Store(base_args.db).close()
        cmd_stats(base_args)
        captured = capsys.readouterr()
        assert "Detailed Statistics" in captured.out
        assert "Total:        0" in captured.out

    def test_stats_with_data(self, base_args_with_data, capsys):
        cmd_stats(base_args_with_data)
        captured = capsys.readouterr()
        assert "Detailed Statistics" in captured.out
        assert "Total:        4" in captured.out
        assert "By type:" in captured.out
        assert "By age:" in captured.out
        assert "By source:" in captured.out

    def test_stats_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        cmd_stats(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["episodes"]["total"] == 4
        assert "by_type" in data["episodes"]
        assert "by_age" in data["episodes"]
        assert "by_source" in data["episodes"]
        assert "continuity" in data
        assert "wraps" in data
        assert "associations" in data

    def test_stats_with_wraps(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            store.record("ep1", episode_type="observation")
            store.wrap_completed(1, 100, graduations_validated=2, graduations_demoted=1)
            store.record("ep2", episode_type="decision")
            store.wrap_completed(1, 150)

        args = Namespace(db=db, project_name="Agent", json=True)
        cmd_stats(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["wraps"]["total_wraps"] == 2
        assert data["wraps"]["total_graduations"] == 2
        assert data["wraps"]["total_demotions"] == 1


# -- cmd_history tests --

class TestCmdHistory:
    def test_history_no_wraps(self, base_args_with_data, capsys):
        base_args_with_data.limit = 20
        cmd_history(base_args_with_data)
        captured = capsys.readouterr()
        assert "No wrap history" in captured.out

    def test_history_no_wraps_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.limit = 20
        cmd_history(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["wraps"] == []

    def test_history_with_wraps(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            store.record("ep1", episode_type="observation")
            store.wrap_completed(1, 100)
            store.record("ep2", episode_type="observation")
            store.wrap_completed(1, 150, associations_formed=3, associations_decayed=1)

        args = Namespace(db=db, project_name="Agent", json=False, limit=20)
        cmd_history(args)
        captured = capsys.readouterr()
        assert "Wrap history" in captured.out
        assert "Wrap 1" in captured.out
        assert "Wrap 2" in captured.out

    def test_history_json(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            store.record("ep1", episode_type="observation")
            store.wrap_completed(1, 100)

        args = Namespace(db=db, project_name="Agent", json=True, limit=20)
        cmd_history(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["wraps"]) == 1
        assert data["wraps"][0]["continuity_chars"] == 100

    def test_history_limit(self, tmp_path, capsys):
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            for i in range(5):
                store.record(f"ep{i}", episode_type="observation")
                store.wrap_completed(1, 100 + i * 10)

        args = Namespace(db=db, project_name="Agent", json=True, limit=3)
        cmd_history(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["wraps"]) == 3
        # total should reflect ALL wraps, not just the limited window
        assert data["total"] == 5


# -- Parser tests for new commands --

class TestParserNewCommands:
    def test_export_command(self):
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "json", "--output", "out.json"])
        assert args.command == "export"
        assert args.format == "json"
        assert args.output == "out.json"

    def test_export_default_format(self):
        parser = build_parser()
        args = parser.parse_args(["export"])
        assert args.format == "json"
        assert args.output is None

    def test_import_command(self):
        parser = build_parser()
        args = parser.parse_args(["import", "data.json"])
        assert args.command == "import"
        assert args.path == "data.json"

    def test_audit_command(self):
        parser = build_parser()
        args = parser.parse_args(["audit", "--since", "3d", "--event", "record", "--limit", "10"])
        assert args.command == "audit"
        assert args.since == "3d"
        assert args.event == "record"
        assert args.limit == 10

    def test_audit_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["audit"])
        assert args.since is None
        assert args.event is None
        assert args.limit == 50

    def test_diff_command(self):
        parser = build_parser()
        args = parser.parse_args(["diff", "--wraps", "10"])
        assert args.command == "diff"
        assert args.wraps == 10

    def test_diff_default_wraps(self):
        parser = build_parser()
        args = parser.parse_args(["diff"])
        assert args.wraps == 5

    def test_graph_command(self):
        parser = build_parser()
        args = parser.parse_args(["graph", "--format", "dot", "--output", "g.dot", "--min-strength", "0.5"])
        assert args.command == "graph"
        assert args.format == "dot"
        assert args.output == "g.dot"
        assert args.min_strength == 0.5

    def test_graph_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["graph"])
        assert args.format == "json"
        assert args.min_strength == 0.0

    def test_stats_command(self):
        parser = build_parser()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"

    def test_history_command(self):
        parser = build_parser()
        args = parser.parse_args(["history", "--limit", "5"])
        assert args.command == "history"
        assert args.limit == 5

    def test_history_default_limit(self):
        parser = build_parser()
        args = parser.parse_args(["history"])
        assert args.limit == 20

    def test_json_on_new_subcommands(self):
        parser = build_parser()
        for cmd in ["export", "import data.json", "audit", "diff", "graph", "stats", "history"]:
            args_list = cmd.split() + ["--json"]
            args = parser.parse_args(args_list)
            assert args.json is True, f"--json failed for subcommand: {cmd}"


# -- Export/Import round-trip integration test --

class TestExportImportRoundTrip:
    def test_full_round_trip(self, store_with_data, tmp_path, capsys):
        """Export from one store, import to another, verify data matches."""
        export_path = str(tmp_path / "roundtrip.json")

        # Export
        args = Namespace(
            db=store_with_data, project_name="TestProject", json=False,
            format="json", output=export_path,
        )
        cmd_export(args)

        # Import to new store
        new_db = str(tmp_path / "imported.db")
        Store(new_db).close()
        args = Namespace(db=new_db, project_name="Agent", json=False, path=export_path)
        cmd_import(args)

        # Verify
        with Store(new_db) as store:
            result = store.recall(limit=100)
            assert result.total_matching == 4
            types = {ep.type.value for ep in result.episodes}
            assert "observation" in types
            assert "decision" in types


# -- Additional test coverage from Layer 1+2 review --

class TestImportMalformedData:
    def test_import_missing_content_key(self, tmp_path, capsys):
        """Import with episodes missing required fields reports errors."""
        export_path = str(tmp_path / "bad_episodes.json")
        Path(export_path).write_text(json.dumps({
            "anneal_memory_export": True,
            "episodes": [
                {"id": "abc12345", "type": "observation"},  # missing content
                {"id": "def67890", "content": "Valid content", "type": "observation", "timestamp": "2026-04-09T12:00:00.000000Z"},
            ],
        }))
        db = str(tmp_path / "test.db")
        Store(db).close()

        args = Namespace(db=db, project_name="Agent", json=True, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["imported"] == 1
        assert data["errors"] == 1

    def test_import_invalid_episode_type(self, tmp_path, capsys):
        """Import with invalid episode type reports errors."""
        export_path = str(tmp_path / "bad_type.json")
        Path(export_path).write_text(json.dumps({
            "anneal_memory_export": True,
            "episodes": [
                {"id": "abc12345", "content": "Test", "type": "INVALID_TYPE", "timestamp": "2026-04-09T12:00:00.000000Z"},
            ],
        }))
        db = str(tmp_path / "test.db")
        Store(db).close()

        args = Namespace(db=db, project_name="Agent", json=True, path=export_path)
        cmd_import(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["errors"] == 1
        assert data["imported"] == 0


class TestAuditGzipPath:
    def test_audit_reads_gzipped_files(self, tmp_path, capsys):
        """Verify audit can read gzipped rotated audit files."""
        import gzip as gz

        db = str(tmp_path / "test.db")
        stem = "test"

        # Create a store with some audit entries
        with Store(db) as store:
            store.record("Episode 1", episode_type="observation")
            store.record("Episode 2", episode_type="decision")

        # Simulate a rotated audit file by creating a gzipped JSONL
        sealed_entries = [
            json.dumps({"v": 1, "seq": 0, "ts": "2026-04-01T00:00:00.000000Z", "event": "record", "actor": "agent", "prev_hash": "sha256:GENESIS", "data": {"episode_id": "old1", "type": "observation"}}),
            json.dumps({"v": 1, "seq": 1, "ts": "2026-04-01T01:00:00.000000Z", "event": "record", "actor": "test", "prev_hash": "sha256:fake", "data": {"episode_id": "old2", "type": "decision"}}),
        ]
        gz_path = tmp_path / f"{stem}.audit.2026-W14.jsonl.gz"
        with gz.open(gz_path, "wt", encoding="utf-8") as f:
            for entry in sealed_entries:
                f.write(entry + "\n")

        # Create a manifest pointing to the gz file
        manifest = {
            "version": 1,
            "db_path": f"{stem}.db",
            "active_file": f"{stem}.audit.jsonl",
            "files": [{"filename": gz_path.name, "period": "2026-W14", "entries": 2, "first_ts": "2026-04-01T00:00:00Z", "last_ts": "2026-04-01T01:00:00Z", "last_hash": "", "sha256_file": ""}],
        }
        manifest_path = tmp_path / f"{stem}.audit.manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Now read audit — should include entries from both gz and active files
        args = Namespace(db=db, project_name="Agent", json=True, since=None, event=None, limit=100)
        cmd_audit(args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        # Should have entries from the gz file AND the active file
        assert data["total"] >= 4  # 2 from gz + 2+ from active
        events = [e["event"] for e in data["entries"]]
        assert "record" in events


class TestGraphJsonOutput:
    def test_graph_json_output_to_file(self, tmp_path, capsys):
        """Graph JSON to file with --json flag outputs metadata."""
        db = str(tmp_path / "test.db")
        with Store(db) as store:
            ep1 = store.record("Episode one", episode_type="observation")
            ep2 = store.record("Episode two", episode_type="decision")
            store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        out = str(tmp_path / "graph.json")
        args = Namespace(db=db, project_name="Agent", json=True, format="json", output=out, min_strength=0.0)
        cmd_graph(args)
        captured = capsys.readouterr()
        # With --json + --output, should get JSON metadata on stdout
        meta = json.loads(captured.out)
        assert meta["format"] == "json"
        assert meta["nodes"] == 2
        assert meta["edges"] == 1
        # And the file should contain the actual graph
        graph = json.loads(Path(out).read_text())
        assert len(graph["nodes"]) == 2


# -- cmd_prepare_wrap tests --

class TestCmdPrepareWrap:
    def test_prepare_wrap_no_episodes(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.max_chars = 20000
        base_args.staleness_days = 7
        cmd_prepare_wrap(base_args)
        captured = capsys.readouterr()
        assert "No episodes since last wrap" in captured.out

    def test_prepare_wrap_no_episodes_json(self, base_args, capsys):
        Store(base_args.db).close()
        base_args.json = True
        base_args.max_chars = 20000
        base_args.staleness_days = 7
        cmd_prepare_wrap(base_args)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "empty"

    def test_prepare_wrap_with_episodes(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        captured = capsys.readouterr()
        # Should contain the compression instructions and episodes
        assert "Compress your session episodes" in captured.out
        assert "Episodes This Session (4)" in captured.out
        assert "Found a pattern" in captured.out

    def test_prepare_wrap_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["episode_count"] == 4
        assert "instructions" in data
        assert "episodes" in data

    def test_prepare_wrap_sets_in_progress(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        # Verify wrap is marked in progress
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            assert store.status().wrap_in_progress is True


# -- cmd_save_continuity tests --

class TestCmdSaveContinuity:
    @staticmethod
    def _valid_continuity(today: str | None = None) -> str:
        """Build a minimal valid 4-section continuity with dynamic date.

        Uses ``date.today().isoformat()`` by default so that
        ``validated_save_continuity`` — which reads wall-clock ``today``
        internally — does not silently skip graduation validation on
        citation date mismatch. Previously a hardcoded ``2026-04-09``
        string caused this test fixture to decay silently against the
        pipeline's internal clock (Diogenes Finding #3, Apr 10 2026).
        """
        if today is None:
            today = date.today().isoformat()
        return (
            "# Agent — Memory (v1)\n\n"
            "## State\nTest state\n\n"
            f"## Patterns\nthought: test | 1x ({today})\n\n"
            f"## Decisions\n[decided(rationale: \"test\", on: \"{today}\")] Test\n\n"
            "## Context\nTest context\n"
        )

    def test_save_from_file(self, base_args_with_data, tmp_path, capsys):
        # First prepare wrap to set the in-progress flag
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()  # clear

        # Write continuity to a temp file
        cont_file = tmp_path / "continuity.md"
        cont_file.write_text(self._valid_continuity())

        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        cmd_save_continuity(base_args_with_data)
        captured = capsys.readouterr()
        assert "Continuity saved" in captured.out
        assert "Episodes compressed: 4" in captured.out

    def test_save_from_stdin(self, base_args_with_data, tmp_path, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        base_args_with_data.file = "-"
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        with mock.patch("sys.stdin", StringIO(self._valid_continuity())):
            cmd_save_continuity(base_args_with_data)
        captured = capsys.readouterr()
        assert "Continuity saved" in captured.out

    def test_save_json_output(self, base_args_with_data, tmp_path, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        cont_file = tmp_path / "continuity.md"
        cont_file.write_text(self._valid_continuity())

        base_args_with_data.json = True
        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        cmd_save_continuity(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["saved"] is True
        assert data["episodes_compressed"] == 4
        assert "sections" in data

    def test_save_with_affect(self, base_args_with_data, tmp_path, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        cont_file = tmp_path / "continuity.md"
        cont_file.write_text(self._valid_continuity())

        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = "engaged"
        base_args_with_data.affect_intensity = 0.8
        cmd_save_continuity(base_args_with_data)
        captured = capsys.readouterr()
        assert "Continuity saved" in captured.out

    def test_save_invalid_structure(self, base_args_with_data, tmp_path, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        cont_file = tmp_path / "bad.md"
        cont_file.write_text("# No sections here\nJust text.")

        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        with pytest.raises(SystemExit):
            cmd_save_continuity(base_args_with_data)

    def test_save_warns_without_prepare(self, base_args_with_data, tmp_path, capsys):
        """Saving without prepare-wrap should warn."""
        cont_file = tmp_path / "continuity.md"
        cont_file.write_text(self._valid_continuity())

        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        cmd_save_continuity(base_args_with_data)
        captured = capsys.readouterr()
        assert "prepare-wrap was not called first" in captured.out

    def test_save_records_wrap_in_history(self, base_args_with_data, tmp_path, capsys):
        """After save-continuity, wrap should appear in history."""
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        cont_file = tmp_path / "continuity.md"
        cont_file.write_text(self._valid_continuity())

        base_args_with_data.file = str(cont_file)
        base_args_with_data.affect_tag = None
        base_args_with_data.affect_intensity = 0.5
        cmd_save_continuity(base_args_with_data)
        capsys.readouterr()

        # Verify wrap appears in history
        base_args_with_data.json = True
        base_args_with_data.limit = 20
        cmd_history(base_args_with_data)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["wraps"]) == 1
        assert data["wraps"][0]["episodes_compressed"] == 4


# -- Parser tests for new commands --

class TestParserPrepareWrapSaveContinuity:
    def test_prepare_wrap_command(self):
        parser = build_parser()
        args = parser.parse_args(["prepare-wrap", "--max-chars", "10000"])
        assert args.command == "prepare-wrap"
        assert args.max_chars == 10000

    def test_prepare_wrap_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["prepare-wrap"])
        assert args.max_chars == 20000
        assert args.staleness_days == 7

    def test_save_continuity_command(self):
        parser = build_parser()
        args = parser.parse_args(["save-continuity", "cont.md"])
        assert args.command == "save-continuity"
        assert args.file == "cont.md"

    def test_save_continuity_with_affect(self):
        parser = build_parser()
        args = parser.parse_args(["save-continuity", "cont.md", "--affect-tag", "curious", "--affect-intensity", "0.7"])
        assert args.affect_tag == "curious"
        assert args.affect_intensity == 0.7

    def test_save_continuity_stdin(self):
        parser = build_parser()
        args = parser.parse_args(["save-continuity", "-"])
        assert args.file == "-"

    def test_json_on_new_commands(self):
        parser = build_parser()
        for cmd in ["prepare-wrap", "save-continuity cont.md"]:
            args_list = cmd.split() + ["--json"]
            args = parser.parse_args(args_list)
            assert args.json is True, f"--json failed for: {cmd}"


# ---------------------------------------------------------------------------
# 10.5c.4 — CLI cross-process TOCTOU handshake token
# ---------------------------------------------------------------------------
#
# The hard part of the session-handshake token design: CLI subcommands
# are separate processes. The ``prepare-wrap`` invocation returns the
# token via stdout, a wrapping script captures it, and the
# ``save-continuity`` invocation passes it back via ``--wrap-token``.
# The shared state across the process boundary is the SQLite store
# metadata — the token and frozen episode ID list are persisted by
# the first process and read by the second.
#
# These tests exercise that cross-process handshake with REAL
# subprocesses (not in-process calls) to lock the actual ship shape.


class TestCLICrossProcessTOCTOU:
    """10.5c.4 — CLI subprocess handshake across prepare → record → save."""

    # Valid 4-section continuity template — callers fill in the cited ID.
    # FlowScript braces are doubled so ``str.format(cited=...)`` doesn't
    # misinterpret them as format fields.
    _TEMPLATE = (
        "# CrossProcess — Memory (v1)\n\n"
        "## State\nCross-process test.\n\n"
        "## Patterns\n"
        "{{core:\n"
        "  thought: token round-trips via --wrap-token "
        "| 1x (2026-04-10)\n"
        "}}\n\n"
        "## Decisions\n"
        "[decided(rationale: \"test\", on: \"2026-04-10\")] ok\n\n"
        "## Context\nCited {cited}.\n"
    )

    @staticmethod
    def _run(*args, check=True):
        """Helper: run ``anneal_memory.cli`` as a subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", *args],
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise AssertionError(
                f"CLI call failed ({result.returncode}):\n"
                f"args={args}\nstdout={result.stdout}\nstderr={result.stderr}"
            )
        return result

    @staticmethod
    def _extract_token_from_text(stdout: str) -> str:
        """Pull the 'Wrap token: <hex>' trailer out of prepare-wrap text."""
        for line in reversed(stdout.splitlines()):
            if line.startswith("Wrap token:"):
                return line.split("Wrap token:", 1)[1].strip()
        raise AssertionError(
            "No 'Wrap token:' trailer in prepare-wrap output:\n" + stdout
        )

    def test_prepare_wrap_text_output_has_token_trailer(self, tmp_path):
        """prepare-wrap text output ends with the 'Wrap token: <hex>'
        trailer so wrapping scripts can grep/awk the value."""
        db_path = str(tmp_path / "trailer.db")
        self._run("--db", db_path, "init")
        self._run(
            "--db", db_path,
            "record", "Trailer test", "--type", "observation",
        )
        out = self._run("--db", db_path, "prepare-wrap").stdout
        token = self._extract_token_from_text(out)
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_prepare_wrap_json_output_has_wrap_token(self, tmp_path):
        """prepare-wrap --json output includes wrap_token as a
        top-level key on the package dict."""
        db_path = str(tmp_path / "json_token.db")
        self._run("--db", db_path, "init")
        self._run(
            "--db", db_path,
            "record", "JSON test", "--type", "observation",
        )
        out = self._run("--db", db_path, "prepare-wrap", "--json").stdout
        payload = json.loads(out)
        assert "wrap_token" in payload
        assert len(payload["wrap_token"]) == 32

    def test_frozen_snapshot_holds_across_process_boundary(self, tmp_path):
        """THE cross-process TOCTOU test. Three separate subprocess
        invocations: prepare-wrap → record → save-continuity. The
        recorded episode must NOT be absorbed into the in-progress
        wrap, regardless of the process boundary between prepare
        and save.
        """
        db_path = str(tmp_path / "cross.db")
        self._run("--db", db_path, "init")
        # Record one snapshot episode.
        rec = self._run(
            "--db", db_path,
            "record", "Snapshot-only episode",
            "--type", "observation",
        )
        # Pull the recorded episode ID from the stdout. Format is
        # "Recorded episode <id>: ..."
        ep_id = None
        for line in rec.stdout.splitlines():
            if line.startswith("Recorded episode"):
                ep_id = line.split("Recorded episode", 1)[1].split(":")[0].strip()
                break
        assert ep_id is not None, f"Could not parse episode id from: {rec.stdout}"

        # Prepare wrap — captures the token.
        prep = self._run("--db", db_path, "prepare-wrap")
        token = self._extract_token_from_text(prep.stdout)

        # TOCTOU — another subprocess records a NEW episode between
        # prepare and save.
        self._run(
            "--db", db_path,
            "record", "TOCTOU leak attempt",
            "--type", "observation",
        )

        # Write the continuity file that the agent would have
        # produced, citing only the snapshot episode.
        continuity_path = tmp_path / "cross.md"
        continuity_path.write_text(
            self._TEMPLATE.format(cited=ep_id), encoding="utf-8"
        )

        # Save with the token — JSON mode so we can assert exact
        # metrics.
        save = self._run(
            "--db", db_path,
            "save-continuity", str(continuity_path),
            "--wrap-token", token,
            "--json",
        )
        result = json.loads(save.stdout)
        assert result["episodes_compressed"] == 1, (
            f"TOCTOU episode leaked across process boundary — "
            f"expected 1 episode compressed, got {result['episodes_compressed']}"
        )

        # Next prepare-wrap should pick up the TOCTOU episode.
        next_prep = self._run(
            "--db", db_path, "prepare-wrap", "--json"
        )
        next_payload = json.loads(next_prep.stdout)
        # Empty status means the TOCTOU episode somehow got absorbed
        # into the previous wrap — fail loudly.
        assert next_payload.get("status") != "empty", (
            "TOCTOU episode did not survive to the next wrap — "
            "cross-process snapshot filter is broken"
        )
        assert "TOCTOU leak attempt" in next_payload.get("episodes", "")

    def test_wrong_token_across_process_boundary_rejected(self, tmp_path):
        """Passing the wrong --wrap-token across a process boundary
        fails with a non-zero exit and the in-progress wrap is
        preserved for retry."""
        db_path = str(tmp_path / "cross_wrong.db")
        self._run("--db", db_path, "init")
        self._run(
            "--db", db_path,
            "record", "A real episode",
            "--type", "observation",
        )
        self._run("--db", db_path, "prepare-wrap")

        continuity_path = tmp_path / "cross_wrong.md"
        continuity_path.write_text(
            self._TEMPLATE.format(cited="deadbeef"), encoding="utf-8"
        )

        result = self._run(
            "--db", db_path,
            "save-continuity", str(continuity_path),
            "--wrap-token", "0" * 32,
            check=False,
        )
        assert result.returncode != 0
        assert "wrap_token mismatch" in result.stderr

        # Wrap is still in progress — retry without token (implicit
        # snapshot) should succeed.
        retry = self._run(
            "--db", db_path,
            "save-continuity", str(continuity_path),
            check=False,
        )
        assert retry.returncode == 0

    def test_implicit_snapshot_across_process_boundary(self, tmp_path):
        """Without --wrap-token, the CLI still uses the persisted
        snapshot because the library consults it whenever it's
        present. Cross-process single-user common case needs no
        ceremony.
        """
        db_path = str(tmp_path / "implicit_cross.db")
        self._run("--db", db_path, "init")
        rec = self._run(
            "--db", db_path,
            "record", "Real episode",
            "--type", "observation",
        )
        ep_id = None
        for line in rec.stdout.splitlines():
            if line.startswith("Recorded episode"):
                ep_id = line.split("Recorded episode", 1)[1].split(":")[0].strip()
                break
        self._run("--db", db_path, "prepare-wrap")

        # TOCTOU record between processes.
        self._run(
            "--db", db_path,
            "record", "Should be deferred",
            "--type", "observation",
        )

        continuity_path = tmp_path / "implicit_cross.md"
        continuity_path.write_text(
            self._TEMPLATE.format(cited=ep_id), encoding="utf-8"
        )

        # Save without --wrap-token.
        save = self._run(
            "--db", db_path,
            "save-continuity", str(continuity_path),
            "--json",
        )
        result = json.loads(save.stdout)
        assert result["episodes_compressed"] == 1


# -- 10.5c.4a operator surface subcommands --

class TestCmdWrapStatus:
    """wrap-status: display wrap-in-progress state with recovery hints."""

    def test_idle_store_text(self, base_args_with_data, capsys):
        cmd_wrap_status(base_args_with_data)
        captured = capsys.readouterr()
        assert "no wrap in progress" in captured.out

    def test_idle_store_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        cmd_wrap_status(base_args_with_data)
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "idle"
        assert data["wrap_token"] is None
        assert data["wrap_episode_count"] == 0
        assert data["wrap_episode_ids"] == []
        assert data["wrap_started_at"] is None

    def test_in_progress_text(self, base_args_with_data, capsys):
        # Prime a wrap via prepare_wrap, then inspect.
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()  # clear

        cmd_wrap_status(base_args_with_data)
        out = capsys.readouterr().out
        assert "wrap in progress since" in out
        assert "token:" in out
        assert "episodes: 4" in out  # base_args_with_data has 4 episodes
        assert "save-continuity --wrap-token" in out
        assert "wrap-cancel" in out

    def test_in_progress_json(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        base_args_with_data.json = True
        cmd_wrap_status(base_args_with_data)
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "in_progress"
        assert data["wrap_started_at"] is not None
        assert len(data["wrap_token"]) == 32
        assert data["wrap_episode_count"] == 4
        assert len(data["wrap_episode_ids"]) == 4

    def test_partial_state_recovery_hint(self, base_args_with_data, capsys):
        """StoreError partial-state failure surfaces recovery hint, not traceback."""
        # Manually induce partial state: wrap_started_at set but
        # wrap_token empty. This bypasses the canonical pipeline —
        # exactly the scenario the operator subcommand exists to recover.
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            store._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("wrap_started_at", "2026-04-10T00:00:00.000000Z"),
            )
            store._conn.commit()

        with pytest.raises(SystemExit) as exc_info:
            cmd_wrap_status(base_args_with_data)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "partial wrap-in-progress state" in captured.err
        assert "wrap-cancel" in captured.err

    def test_partial_state_json(self, base_args_with_data, capsys):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            store._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("wrap_started_at", "2026-04-10T00:00:00.000000Z"),
            )
            store._conn.commit()

        base_args_with_data.json = True
        with pytest.raises(SystemExit) as exc_info:
            cmd_wrap_status(base_args_with_data)
        assert exc_info.value.code == 1

        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "partial_state"
        assert data["operation"] == "load_wrap_snapshot"
        assert "wrap-cancel" in data["recovery"]


class TestCmdWrapCancel:
    """wrap-cancel: clear wrap state without recording a completed wrap."""

    def test_cancel_idle_store(self, base_args_with_data, capsys):
        cmd_wrap_cancel(base_args_with_data)
        out = capsys.readouterr().out
        assert "no wrap was in progress" in out

    def test_cancel_in_progress_reports_token(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)

        # Capture the token that was minted so we can verify it's echoed.
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            snapshot = store.load_wrap_snapshot()
        assert snapshot is not None
        expected_token = snapshot["token"]

        capsys.readouterr()  # clear prepare_wrap output
        cmd_wrap_cancel(base_args_with_data)
        out = capsys.readouterr().out
        assert "wrap cancelled" in out
        assert expected_token in out

        # Verify the state was actually cleared.
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            assert store.load_wrap_snapshot() is None
            assert store.get_wrap_started_at() is None

    def test_cancel_json(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        base_args_with_data.json = True
        cmd_wrap_cancel(base_args_with_data)
        data = json.loads(capsys.readouterr().out)
        assert data["status"] == "cancelled"
        assert len(data["cancelled_token"]) == 32

    def test_cancel_recovers_partial_state(self, base_args_with_data, capsys):
        """wrap-cancel must work on partial-state stores — that's the whole point."""
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            store._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("wrap_started_at", "2026-04-10T00:00:00.000000Z"),
            )
            store._conn.commit()

        # Should NOT raise — cancel is the escape hatch.
        cmd_wrap_cancel(base_args_with_data)
        out = capsys.readouterr().out
        assert "state cleared" in out or "wrap cancelled" in out

        # Verify wrap-status now sees the store as idle.
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            assert store.get_wrap_started_at() is None


class TestCmdWrapTokenCurrent:
    """wrap-token-current: shell-pipeline-friendly token extraction."""

    def test_idle_prints_nothing(self, base_args_with_data, capsys):
        cmd_wrap_token_current(base_args_with_data)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_idle_json(self, base_args_with_data, capsys):
        base_args_with_data.json = True
        cmd_wrap_token_current(base_args_with_data)
        data = json.loads(capsys.readouterr().out)
        assert data["wrap_token"] is None

    def test_in_progress_prints_only_token(self, base_args_with_data, capsys):
        base_args_with_data.max_chars = 20000
        base_args_with_data.staleness_days = 7
        cmd_prepare_wrap(base_args_with_data)
        capsys.readouterr()

        cmd_wrap_token_current(base_args_with_data)
        out = capsys.readouterr().out.strip()
        # Pipeline-friendly: just the token, nothing else.
        assert len(out) == 32
        assert all(c in "0123456789abcdef" for c in out)

    def test_partial_state_exits_nonzero_with_stderr_hint(
        self, base_args_with_data, capsys
    ):
        with Store(base_args_with_data.db, project_name="TestProject") as store:
            store._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("wrap_started_at", "2026-04-10T00:00:00.000000Z"),
            )
            store._conn.commit()

        with pytest.raises(SystemExit) as exc_info:
            cmd_wrap_token_current(base_args_with_data)
        assert exc_info.value.code == 1

        captured = capsys.readouterr()
        # stdout stays clean for pipeline safety; diagnostic on stderr.
        assert captured.out == ""
        assert "partial wrap-in-progress" in captured.err
        assert "wrap-cancel" in captured.err


class TestWrapOperatorSubcommandsSubprocess:
    """End-to-end subprocess test covering the full operator flow."""

    @staticmethod
    def _run(*args, check=True):
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", *args],
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            raise AssertionError(
                f"CLI call failed ({result.returncode}):\n"
                f"args={args}\nstdout={result.stdout}\nstderr={result.stderr}"
            )
        return result

    def test_prepare_status_token_cancel_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "ops.db")

        self._run("--db", db_path, "init")
        self._run("--db", db_path, "record", "ep1", "--type", "observation")
        self._run("--db", db_path, "record", "ep2", "--type", "observation")

        # Idle status via subprocess.
        status = self._run("--db", db_path, "--json", "wrap-status")
        assert json.loads(status.stdout)["status"] == "idle"

        # Token-current prints empty on idle.
        tok_idle = self._run("--db", db_path, "wrap-token-current")
        assert tok_idle.stdout.strip() == ""

        # Prime a wrap.
        self._run("--db", db_path, "prepare-wrap")

        # Status reports in-progress.
        status = self._run("--db", db_path, "--json", "wrap-status")
        status_data = json.loads(status.stdout)
        assert status_data["status"] == "in_progress"
        assert len(status_data["wrap_token"]) == 32
        assert status_data["wrap_episode_count"] == 2

        # Token-current yields the exact same token (shell-pipeline use).
        tok_active = self._run("--db", db_path, "wrap-token-current")
        assert tok_active.stdout.strip() == status_data["wrap_token"]

        # Cancel and verify idle again.
        cancel = self._run("--db", db_path, "--json", "wrap-cancel")
        assert json.loads(cancel.stdout)["cancelled_token"] == status_data["wrap_token"]

        status = self._run("--db", db_path, "--json", "wrap-status")
        assert json.loads(status.stdout)["status"] == "idle"
