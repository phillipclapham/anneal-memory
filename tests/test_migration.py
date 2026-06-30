"""Tests for migration-notify (``anneal_memory.migration`` + the ``migrate`` CLI).

Self-migration notices: on upgrade, ``migrate check`` proposes edits to the
adopter's own instruction files (never writes them); ``migrate ack`` advances
a per-store acknowledgement marker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anneal_memory import __version__
from anneal_memory.cli import build_parser, cmd_migrate_ack, cmd_migrate_check
from anneal_memory.migration import (
    MIGRATION_MANIFEST,
    MigrationEntry,
    _version_tuple,
    marker_path_for,
    pending_migrations,
    read_marker,
    write_marker,
)


# -- _version_tuple --

class TestVersionTuple:
    @pytest.mark.parametrize("value,expected", [
        ("0.4.7", (0, 4, 7)),
        ("0.4.7rc1", (0, 4, 7)),         # pre-release suffix stripped
        ("0.4.7+ubuntu.1", (0, 4, 7)),   # dotted build suffix must NOT leak
        ("0.4.7rc.8", (0, 4, 7)),        # dotted pre-release suffix must NOT leak
        ("0.5", (0, 5)),
        ("1.2.3.4", (1, 2, 3, 4)),
        ("0.04.7", (0, 4, 7)),     # leading zero
        ("10.20.30", (10, 20, 30)),
        ("  0.4.7  ", (0, 4, 7)),  # surrounding whitespace
        ("", ()),
        ("garbage", ()),
        ("v1.2", ()),              # leading 'v' is non-digit -> empty
    ])
    def test_parse(self, value: str, expected: tuple[int, ...]) -> None:
        assert _version_tuple(value) == expected

    def test_ordering(self) -> None:
        assert _version_tuple("0.4.6") < _version_tuple("0.4.7")
        assert _version_tuple("0.4.7") < _version_tuple("0.5.0")
        assert _version_tuple("0.4") < _version_tuple("0.4.6")  # shorter prefix < longer
        assert _version_tuple("0.4.7") == _version_tuple("0.4.7")


# -- pending_migrations --

SYNTHETIC: list[MigrationEntry] = [
    {"version": "0.4.7", "feature": "A", "summary": "s", "suggested_edit": "e",
     "files": ["CLAUDE.md"]},
    {"version": "0.5.0", "feature": "B", "summary": "s", "suggested_edit": "e",
     "files": ["CLAUDE.md"]},
]


class TestPendingMigrations:
    def test_none_ack_shows_all_installed(self) -> None:
        out = pending_migrations(None, current_version="0.5.0", manifest=SYNTHETIC)
        assert [e["feature"] for e in out] == ["A", "B"]

    def test_ack_filters_older(self) -> None:
        out = pending_migrations("0.4.7", current_version="0.5.0", manifest=SYNTHETIC)
        assert [e["feature"] for e in out] == ["B"]

    def test_ack_current_shows_nothing(self) -> None:
        out = pending_migrations("0.5.0", current_version="0.5.0", manifest=SYNTHETIC)
        assert out == []

    def test_withholds_entries_newer_than_installed(self) -> None:
        # installed 0.4.7 -> the 0.5.0 entry is not yet installed; withhold it.
        out = pending_migrations(None, current_version="0.4.7", manifest=SYNTHETIC)
        assert [e["feature"] for e in out] == ["A"]

    def test_installed_behind_all_entries(self) -> None:
        out = pending_migrations(None, current_version="0.4.6", manifest=SYNTHETIC)
        assert out == []

    def test_ack_ahead_of_installed_is_safe(self) -> None:
        # Acknowledged a version ahead of what's installed -> nothing pending.
        out = pending_migrations("0.9.0", current_version="0.4.7", manifest=SYNTHETIC)
        assert out == []

    def test_real_manifest_from_clean_at_current(self) -> None:
        # A clean adopter at the installed version sees every entry <= installed.
        out = pending_migrations(None)  # current_version defaults to __version__
        expected = [e for e in MIGRATION_MANIFEST
                    if _version_tuple(e["version"]) <= _version_tuple(__version__)]
        assert len(out) == len(expected)
        assert len(out) >= 1  # the shipped manifest is non-empty for the current line

    def test_real_manifest_entries_not_keyed_ahead_of_version(self) -> None:
        # Structural guard: no shipped manifest entry may be keyed to a version
        # newer than __version__, or it would be permanently invisible (withheld
        # as "not yet installed") until a future release bump. Catches the
        # release-time chicken-and-egg between the manifest and the version.
        for entry in MIGRATION_MANIFEST:
            assert _version_tuple(entry["version"]) <= _version_tuple(__version__), (
                f"manifest entry {entry['feature']} keyed to {entry['version']} "
                f"is ahead of __version__ {__version__}"
            )

    def test_real_manifest_entries_have_all_fields(self) -> None:
        for entry in MIGRATION_MANIFEST:
            assert entry["version"] and entry["feature"]
            assert entry["summary"] and entry["suggested_edit"]
            assert entry["files"] and all(isinstance(f, str) for f in entry["files"])


# -- AM-WRAP-GENERATED: the first file-RETIREMENT entry (spore-217) --

class TestWrapGeneratedEntry:
    """Locks AM-WRAP-GENERATED — the first manifest entry whose guidance RETIRES
    a methodology companion file rather than reconciling a description of an
    anneal feature. The content invariant is load-bearing: an AI ACTS on this
    against a real operator's files, so it must lead with the live-memory
    boundary, guide ARCHIVE via an IMPERATIVE prohibition (never an erase),
    preserve adopter-specific custom steps, and stay filename-agnostic. The
    asserts pin the safety-critical phrases AND forbid additive destructive
    imperatives — L3 (codex + L1 converged): the regression guard must catch
    ADDITION, not just replacement."""

    def _entry(self) -> MigrationEntry:
        matches = [e for e in MIGRATION_MANIFEST if e["feature"] == "AM-WRAP-GENERATED"]
        assert len(matches) == 1, "exactly one AM-WRAP-GENERATED entry expected"
        return matches[0]

    def test_present_and_versioned(self) -> None:
        # Keyed at the first PUBLIC release version (0.9.6) so it is visible
        # (not withheld as "not yet installed") yet not retroactively keyed
        # under a shipped release an adopter may already have acknowledged.
        # The feature landed in code at the never-published label 0.9.5; it
        # first ships publicly in 0.9.6, so 0.9.6 is the honest key — and it
        # also surfaces for a local 0.9.5-acked operator upgrading to 0.9.6
        # (a 0.9.5 ack marker does not record which 0.9.5 build it saw).
        assert self._entry()["version"] == "0.9.6"

    def test_live_memory_guard_leads(self) -> None:
        # complement L3: the hard boundary must LEAD — an AI executing
        # incrementally could begin acting before reading a guard left at the end.
        edit = self._entry()["suggested_edit"].lower()
        assert "do not touch any file that is your live memory" in edit
        assert "live memory" in edit[:160], "the live-memory guard must lead, not trail"

    def test_archive_not_erase_is_an_imperative_prohibition(self) -> None:
        # The safety invariant: an IMPERATIVE prohibition (not a two-branch
        # "refusal is correct" validation), AND no additive destructive imperative
        # can slip in. The bad-phrase list is verified absent today, so it catches
        # a future REGRESSION (codex LOW + L1 MED converged), not a false positive.
        edit = self._entry()["suggested_edit"].lower()
        assert "must not happen" in edit
        assert "archive, never erase" in edit
        for bad in ("delete the file", "delete it", "delete the original",
                    "safe to delete", "you can delete", "you may delete", "rm -"):
            assert bad not in edit, f"destructive phrasing leaked in: {bad!r}"

    def test_preserves_custom_steps_before_archiving(self) -> None:
        # L1 L3: prepare_wrap generates ONLY the mechanics; a mixed doc's
        # adopter-specific steps (commit/digest/baton/dev-archival) must be
        # RELOCATED, not archived away with the shell.
        edit = self._entry()["suggested_edit"].lower()
        assert "relocate" in edit
        assert "dev-archival" in edit

    def test_reads_claim_scoped_to_anneal_and_warns_on_imports(self) -> None:
        # complement L3: "nothing reads it" overclaims — an adopter whose harness
        # @imports / auto-loads the file still reads it. Scope to anneal + warn.
        edit = self._entry()["suggested_edit"].lower()
        assert "no anneal-memory mechanism reads" in edit
        assert "@import" in edit

    def test_filename_agnostic_anchored_on_prepare_wrap(self) -> None:
        # Anchored on the anneal capability (prepare_wrap); WRAP_PROTOCOL.md
        # appears ONLY in the summary as a recognition example — the actionable
        # edit stays agnostic so the guidance reaches every harness, not just flow.
        entry = self._entry()
        assert "prepare_wrap" in entry["summary"]
        assert "prepare_wrap" in entry["suggested_edit"]
        assert "wrap_protocol.md" not in entry["suggested_edit"].lower()

    def test_surfaces_for_migrating_adopter_at_prior_release(self) -> None:
        # The Tony scenario: an adopter acknowledged at the prior PyPI release
        # (0.8.5), upgrading to the current line, sees the new retirement entry,
        # and every surfaced entry is genuinely newer than what they acked.
        out = pending_migrations("0.8.5", current_version="0.9.6")
        assert "AM-WRAP-GENERATED" in [e["feature"] for e in out]
        assert all(_version_tuple(e["version"]) > _version_tuple("0.8.5") for e in out)

    def test_not_surfaced_for_fresh_install_acked_at_current(self) -> None:
        # spore-216 compose (negative case): a fresh install whose init auto-acked
        # the current version never had the file, so the entry is correctly
        # invisible to it — it surfaces only on a genuine migration from below.
        out = pending_migrations("0.9.6", current_version="0.9.6")
        assert "AM-WRAP-GENERATED" not in [e["feature"] for e in out]


# -- marker read / write --

class TestMarker:
    def test_marker_path_for(self) -> None:
        assert marker_path_for(Path("/x/memory.db")) == Path("/x/memory.migrate.json")

    def test_marker_path_expands_user(self) -> None:
        assert "~" not in str(marker_path_for(Path("~/x/memory.db")))

    def test_read_absent_returns_none(self, tmp_path: Path) -> None:
        assert read_marker(tmp_path / "nope.json") is None

    def test_round_trip(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        write_marker(m, "0.4.7")
        assert read_marker(m) == "0.4.7"

    def test_write_creates_parent(self, tmp_path: Path) -> None:
        m = tmp_path / "sub" / "dir" / "memory.migrate.json"
        write_marker(m, "0.4.7")
        assert read_marker(m) == "0.4.7"

    def test_write_overwrites(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        write_marker(m, "0.4.6")
        write_marker(m, "0.4.7")
        assert read_marker(m) == "0.4.7"

    def test_read_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        m.write_text("{not json", encoding="utf-8")
        assert read_marker(m) is None

    def test_read_non_dict_returns_none(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        m.write_text('"just a string"', encoding="utf-8")
        assert read_marker(m) is None

    def test_read_missing_key_returns_none(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        m.write_text('{"other": "x"}', encoding="utf-8")
        assert read_marker(m) is None

    def test_read_non_str_version_returns_none(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        m.write_text('{"acknowledged_version": 47}', encoding="utf-8")
        assert read_marker(m) is None

    def test_read_empty_version_returns_none(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        m.write_text('{"acknowledged_version": "  "}', encoding="utf-8")
        assert read_marker(m) is None

    def test_write_leaves_no_tmp_behind(self, tmp_path: Path) -> None:
        m = tmp_path / "memory.migrate.json"
        write_marker(m, "0.4.7")
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


# -- CLI commands (in-process) --

def _args(db: Path, *, json_out: bool = False, version: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(db=str(db), json=json_out, version=version)


class TestMigrateCLI:
    def test_check_no_marker_lists_pending(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_migrate_check(_args(tmp_path / "memory.db"))
        out = capsys.readouterr().out
        assert "self-migration proposal" in out
        assert "AM-SPORES-BOUNDARY" in out
        assert "never edits your files" in out

    def test_check_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_migrate_check(_args(tmp_path / "memory.db", json_out=True))
        data = json.loads(capsys.readouterr().out)
        assert data["installed_version"] == __version__
        assert data["acknowledged_version"] is None
        assert len(data["pending"]) >= 1
        assert data["pending"][0]["feature"]

    def test_ack_then_check_clean(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "memory.db"
        cmd_migrate_ack(_args(db))
        capsys.readouterr()
        cmd_migrate_check(_args(db))
        assert "up to date" in capsys.readouterr().out.lower()

    def test_ack_writes_marker_at_current(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "memory.db"
        cmd_migrate_ack(_args(db))
        assert read_marker(marker_path_for(db)) == __version__

    def test_ack_explicit_version(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "memory.db"
        cmd_migrate_ack(_args(db, version="0.4.7"))
        assert read_marker(marker_path_for(db)) == "0.4.7"

    def test_ack_rejects_ahead_of_installed(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # Acknowledging a version ahead of installed would silently suppress all
        # real proposals — refuse it, and write nothing.
        db = tmp_path / "memory.db"
        with pytest.raises(SystemExit) as exc:
            cmd_migrate_ack(_args(db, version="9.9.9"))
        assert exc.value.code == 1
        assert "ahead of the installed version" in capsys.readouterr().err
        assert read_marker(marker_path_for(db)) is None  # nothing written

    def test_ack_rejects_unparseable_version(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "memory.db"
        with pytest.raises(SystemExit) as exc:
            cmd_migrate_ack(_args(db, version="wat"))
        assert exc.value.code == 1
        assert "not a parseable version" in capsys.readouterr().err
        assert read_marker(marker_path_for(db)) is None

    def test_ack_rejects_explicit_empty_version(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # An explicit empty string is malformed input — it must NOT silently
        # fall through to the installed-version default (only a genuinely-absent
        # arg, version=None, does). codex L3.
        db = tmp_path / "memory.db"
        with pytest.raises(SystemExit) as exc:
            cmd_migrate_ack(_args(db, version=""))
        assert exc.value.code == 1
        assert "not a parseable version" in capsys.readouterr().err
        assert read_marker(marker_path_for(db)) is None

    def test_ack_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "memory.db"
        cmd_migrate_ack(_args(db, json_out=True))
        data = json.loads(capsys.readouterr().out)
        assert data["acknowledged_version"] == __version__
        assert data["marker"].endswith("memory.migrate.json")

    def test_check_does_not_create_db(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        db = tmp_path / "nonexistent.db"
        cmd_migrate_check(_args(db))
        out = capsys.readouterr().out
        assert "proposal" in out or "up to date" in out
        assert not db.exists()  # migrate-check never touches the db


# -- parser wiring --

class TestMigrateParser:
    def test_parser_wires_check(self) -> None:
        ns = build_parser().parse_args(["migrate", "check"])
        assert ns.func is cmd_migrate_check

    def test_parser_wires_ack_default_version(self) -> None:
        ns = build_parser().parse_args(["migrate", "ack"])
        assert ns.func is cmd_migrate_ack
        assert ns.version is None

    def test_parser_wires_ack_explicit_version(self) -> None:
        ns = build_parser().parse_args(["migrate", "ack", "0.4.7"])
        assert ns.version == "0.4.7"

    def test_bare_migrate_requires_subcommand(self) -> None:
        ns = build_parser().parse_args(["migrate"])
        with pytest.raises(SystemExit) as exc:
            ns.func(ns)
        assert exc.value.code == 1


# -- subprocess (real entry point + exit codes) --

class TestMigrateSubprocess:
    def _env(self, tmp_path: Path) -> dict[str, str]:
        return {**os.environ, "ANNEAL_MEMORY_DB": str(tmp_path / "m.db")}

    def test_migrate_no_subcommand_exits_1(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "migrate"],
            capture_output=True, text=True, env=self._env(tmp_path),
        )
        assert result.returncode == 1
        assert "requires a subcommand" in result.stderr

    def test_check_then_ack_then_clean(self, tmp_path: Path) -> None:
        env = self._env(tmp_path)
        check = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "migrate", "check"],
            capture_output=True, text=True, env=env,
        )
        assert check.returncode == 0
        assert "AM-SPORES-BOUNDARY" in check.stdout

        ack = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "migrate", "ack"],
            capture_output=True, text=True, env=env,
        )
        assert ack.returncode == 0

        check2 = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "migrate", "check"],
            capture_output=True, text=True, env=env,
        )
        assert check2.returncode == 0
        assert "up to date" in check2.stdout.lower()
