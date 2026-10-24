"""Gen 2: a sidecar written by a NEWER anneal-memory is refused, never read as
this release's format, never quarantined, never rewritten.

Built from the 1014+17 S2 RUN (hand-written newer files driven through the real
CLI on the gen-2 branch before the fix):

- continuity meta ``format_version`` 99: a wrap read it as its own, bumped
  ``sessions_produced`` and wrote it back; ``continuity --json`` and ``export``
  printed it as this release's meta.
- audit manifest ``version`` 99: read as version 1 (verify valid). With a field
  shape 1.0 rejects (``files`` an object), the next write QUARANTINED it and
  ``audit-repair`` rebuilt a version-1 manifest over it.
- ``derive-trust.json`` ``version`` 99: trusted on read; ``allow`` and ``revoke``
  rewrote it as version 2, dropping what the newer release had written.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import anneal_memory.audit as audit_module
from anneal_memory import prepare_wrap, validated_save_continuity
from anneal_memory.audit import AuditTrail, _parse_manifest_bytes
from anneal_memory.rederive import _load_trust, allow_store, revoke_store, trusted_roots
from anneal_memory.server import Server
from anneal_memory.store import Store, StoreError, _NewerSidecarError
from anneal_memory.types import EpisodeType

_REPO = Path(__file__).resolve().parents[1]

_CONT = (
    "# Scratch — Memory (v1)\n\n"
    "## State\nWorking on sidecar checks.\n\n"
    "## Patterns\n{sidecars:\n  thought: newer files refuse | 1x (2026-10-14)\n}\n\n"
    "## Decisions\n[decided(rationale: \"clean refusal\", on: \"2026-10-14\")] Refuse newer sidecars\n\n"
    "## Context\nS2 run.\n"
)

_NEWER_META = {"format_version": 99, "sessions_produced": 7, "citations_seen": True,
               "field_from_the_future": {"x": 1}}


def _cli(db: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    # The child writes UTF-8 and the parent reads UTF-8 on every platform, so
    # the assertions on stderr never depend on the console code page (console
    # encodings are tested in test_cli.py, not here).
    full_env = dict(os.environ, PYTHONPATH=str(_REPO), PYTHONIOENCODING="utf-8")
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *args],
        capture_output=True, text=True, encoding="utf-8", env=full_env, timeout=120,
    )


@contextlib.contextmanager
def _unreadable(path: Path):
    """Mode 000 for the block; the mode is restored on every exit so tmp_path
    cleanup works. Skipped where chmod cannot make a file unreadable."""
    path.chmod(0o000)
    try:
        yield
    finally:
        path.chmod(0o600)


def _newer_meta(store: Store) -> None:
    """Generation 3: the meta is the latest identity row's ``meta_json`` and the
    sidecar its projection, so a newer writer's meta is a row plus the file it
    projects (writing the file alone is a foreign edit, kept aside at open)."""
    import sqlite3
    from anneal_memory.store import register_writer_schema

    conn = sqlite3.connect(str(store.path))
    register_writer_schema(conn)
    conn.execute(
        "UPDATE identity_versions SET meta_json = ? WHERE version = "
        "(SELECT MAX(version) FROM identity_versions)",
        (json.dumps(_NEWER_META, sort_keys=True),))
    conn.commit()
    conn.close()
    store.meta_path.write_text(json.dumps(_NEWER_META, indent=2, sort_keys=True) + "\n")


_NO_CHMOD = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="mode 000 does not block reads on Windows or as root",
)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mem.db", project_name="Scratch")
    yield s
    s.close()


# -- continuity meta --------------------------------------------------------


class TestContinuityMeta:
    def _wrapped(self, store: Store) -> None:
        store.record("first episode", EpisodeType.OBSERVATION)
        prepare_wrap(store)
        validated_save_continuity(store, _CONT)

    def test_this_release_writes_format_version_1_and_reads_it(self, store):
        self._wrapped(store)
        assert json.loads(store.meta_path.read_text())["format_version"] == 1
        assert store.load_meta()["sessions_produced"] == 1

    def test_load_meta_refuses_a_newer_format_version(self, store):
        store.meta_path.write_text(json.dumps(_NEWER_META))
        with pytest.raises(_NewerSidecarError, match="written by a newer anneal-memory") as ei:
            store.load_meta()
        assert isinstance(ei.value, StoreError)
        assert ei.value.operation == "load_meta"
        assert ei.value.path == str(store.meta_path)

    @pytest.mark.parametrize("value", [True, "99", 1.5, None, 0, 1])
    def test_only_an_integer_greater_than_1_refuses(self, store, value):
        # bool is excluded (True == 1 in Python); a non-int keeps the old reading.
        store.meta_path.write_text(json.dumps({"format_version": value}))
        assert store.load_meta()["format_version"] == value

    def test_missing_and_garbled_meta_still_read_as_defaults(self, store):
        assert store.load_meta()["format_version"] == 1
        store.meta_path.write_text("{not json")
        assert store.load_meta() == {"sessions_produced": 0, "citations_seen": False,
                                     "format_version": 1}
        # L3 r2 S2: a version row whose meta_json does not parse is not
        # projected, and neither read raises out of the reconcile.
        store.meta_path.unlink()
        self._wrapped(store)
        store._conn.execute(
            "UPDATE identity_versions SET meta_json = 'not json' WHERE version = "
            "(SELECT MAX(version) FROM identity_versions)")
        store._conn.commit()
        assert store.load_continuity() == _CONT
        assert store.load_meta() == {"sessions_produced": 0, "citations_seen": False,
                                     "format_version": 1}

    def test_a_wrap_over_a_newer_meta_is_refused_and_writes_nothing(self, store):
        """RUN case M: the save read format_version 99 as its own and wrote it
        back with sessions_produced 8."""
        self._wrapped(store)
        store.record("second episode", EpisodeType.OBSERVATION)
        prepare_wrap(store)
        _newer_meta(store)
        meta_before = store.meta_path.read_bytes()
        cont_before = store.continuity_path.read_bytes()
        with pytest.raises(_NewerSidecarError):
            validated_save_continuity(store, _CONT + "Second wrap.\n")
        assert store.meta_path.read_bytes() == meta_before
        assert store.continuity_path.read_bytes() == cont_before
        assert store.status().wrap_in_progress  # the wrap stays open
        assert not [p for p in store.path.parent.iterdir() if ".tmp" in p.name]

    def test_mcp_save_returns_the_refusal_not_an_io_failure(self, store):
        self._wrapped(store)
        server = Server(store)
        server._tool_record({"content": "second", "episode_type": "observation"})
        server._tool_prepare_wrap({})
        _newer_meta(store)
        result = server._tool_save_continuity({"text": _CONT + "Second.\n"})
        assert result.get("isError")
        text = result["content"][0]["text"]
        assert "written by a newer anneal-memory" in text
        assert "I/O failure" not in text

    def test_prepare_wrap_refuses_before_any_wrap_opens(self, store):
        self._wrapped(store)
        _newer_meta(store)
        before = store.meta_path.read_bytes()
        store.record("second episode", EpisodeType.OBSERVATION)
        with pytest.raises(_NewerSidecarError):
            prepare_wrap(store)
        assert not store.status().wrap_in_progress
        assert store.meta_path.read_bytes() == before
        # The MCP tool returns the refusal, not an I/O failure.
        result = Server(store)._tool_prepare_wrap({})
        assert result.get("isError")
        text = result["content"][0]["text"]
        assert "written by a newer anneal-memory" in text and "I/O failure" not in text
        assert not store.status().wrap_in_progress

    def test_cli_prepare_wrap_refuses_and_opens_no_wrap(self, store):
        self._wrapped(store)
        store.record("second episode", EpisodeType.OBSERVATION)
        _newer_meta(store)
        db = store.path
        store.close()
        _cli(db, "baton", "claim", "S2")
        r = _cli(db, "prepare-wrap", "--session-id", "S2")
        assert r.returncode == 1
        assert r.stderr.startswith("Error: ") and "written by a newer anneal-memory" in r.stderr
        assert "Traceback" not in r.stderr and "I/O failure" not in r.stderr
        with Store(db) as again:
            assert not again.status().wrap_in_progress

    def test_cli_save_continuity_json_continuity_and_export_refuse(self, store, tmp_path):
        self._wrapped(store)
        db = store.path
        store.close()
        _cli(db, "baton", "claim", "S2")
        assert _cli(db, "record", "second episode", "--type", "observation").returncode == 0
        out = _cli(db, "prepare-wrap", "--session-id", "S2").stdout
        token = out.split("Wrap token:")[1].split()[0]
        _newer_meta(store)
        before = store.meta_path.read_bytes()
        r = _cli(db, "--json", "continuity")
        assert r.returncode == 1 and "written by a newer anneal-memory" in r.stderr
        assert "Traceback" not in r.stderr
        r = _cli(db, "export", "--output", str(tmp_path / "out.json"))
        assert r.returncode == 1 and "written by a newer anneal-memory" in r.stderr
        assert not (tmp_path / "out.json").exists()
        # The plain text read does not touch the meta and still works.
        assert _cli(db, "continuity").returncode == 0
        # The CLI wrap: refused as a refusal, not "store I/O failure".
        (tmp_path / "c.md").write_text(_CONT + "Again.\n", encoding="utf-8")
        r = _cli(db, "save-continuity", str(tmp_path / "c.md"), "--session-id", "S2",
                 "--wrap-token", token)
        assert r.returncode == 1
        assert "written by a newer anneal-memory" in r.stderr
        assert "I/O failure" not in r.stderr
        assert store.meta_path.read_bytes() == before


    # -- L3 r1: an unreadable meta is not an absent one; a malformed one is -----

    @_NO_CHMOD
    def test_an_unreadable_meta_is_refused_not_defaulted(self, store):
        """L3 r1 F-A (codex HIGH), RUN: chmod 000 on a real meta read as the
        defaults, the wrap opened and the save renamed over the file."""
        store.meta_path.write_text(json.dumps(
            {"format_version": 1, "sessions_produced": 5, "citations_seen": True}))
        with _unreadable(store.meta_path):
            with pytest.raises(StoreError) as ei:
                store.load_meta()
            assert not isinstance(ei.value, _NewerSidecarError)
            assert ei.value.operation == "load_meta"
            assert ei.value.path == str(store.meta_path)
            store.record("first episode", EpisodeType.OBSERVATION)
            with pytest.raises(StoreError, match="Cannot read"):
                prepare_wrap(store)
            assert not store.status().wrap_in_progress
        assert json.loads(store.meta_path.read_text())["sessions_produced"] == 5

    @_NO_CHMOD
    def test_a_save_over_an_unreadable_meta_projection_keeps_the_db_meta(self, store):
        # Generation 3: once a version row exists the meta is read from the DB,
        # so an unreadable projection cannot reset the counters (the gen-2
        # hazard); the save commits and re-projects a readable sidecar.
        self._wrapped(store)
        store.record("second episode", EpisodeType.OBSERVATION)
        prepare_wrap(store)
        with _unreadable(store.meta_path):
            validated_save_continuity(store, _CONT + "Second.\n")
        assert store.load_meta()["sessions_produced"] == 2
        assert json.loads(store.meta_path.read_text())["sessions_produced"] == 2

    @_NO_CHMOD
    def test_cli_paths_work_over_an_unreadable_meta_projection(self, store, tmp_path):
        # Generation 3: the CLI reads the meta from the DB; the unreadable file
        # is kept aside at the first open and re-projected.
        self._wrapped(store)
        db = store.path
        store.close()
        _cli(db, "baton", "claim", "S2")
        assert _cli(db, "record", "second episode", "--type", "observation").returncode == 0
        token = _cli(db, "prepare-wrap", "--session-id", "S2").stdout.split(
            "Wrap token:")[1].split()[0]
        (tmp_path / "c.md").write_text(_CONT + "Again.\n", encoding="utf-8")
        with _unreadable(store.meta_path):
            runs = [
                _cli(db, "--json", "continuity"),
                _cli(db, "export", "--output", str(tmp_path / "out.json")),
                _cli(db, "save-continuity", str(tmp_path / "c.md"), "--session-id", "S2",
                     "--wrap-token", token),
            ]
        for r in runs:
            assert r.returncode == 0, r.stderr
            assert "Traceback" not in r.stderr
        assert (tmp_path / "out.json").exists()
        assert json.loads(store.meta_path.read_text())["sessions_produced"] == 2
        assert len(list(tmp_path.glob("mem.continuity.meta.json.diverged-*"))) == 1

    @pytest.mark.parametrize("content", [b"[]", b"null", b"42", b'"text"', b"\xff\xfe\x00{"])
    def test_a_meta_that_is_not_a_json_object_reads_as_defaults(self, store, content):
        """L3 r1 F-C, RUN: `[]`/`null` crashed `{**defaults, **data}` with a
        TypeError, non-UTF-8 bytes with a UnicodeDecodeError, at the top of
        prepare_wrap and in `continuity --json` / `export`."""
        store.meta_path.write_bytes(content)
        defaults = {"sessions_produced": 0, "citations_seen": False, "format_version": 1}
        assert store.load_meta() == defaults
        store.record("first episode", EpisodeType.OBSERVATION)
        prepare_wrap(store)
        validated_save_continuity(store, _CONT)
        assert store.load_meta()["sessions_produced"] == 1

    def test_cli_json_continuity_and_export_survive_a_malformed_meta(self, store, tmp_path):
        self._wrapped(store)
        db = store.path
        store.close()
        for content in (b"[]", b"\xff\xfe\x00{"):
            store.meta_path.write_bytes(content)
            r = _cli(db, "--json", "continuity")
            assert r.returncode == 0, r.stderr
            assert json.loads(r.stdout)["meta"]["format_version"] == 1
            out = tmp_path / "out.json"
            r = _cli(db, "export", "--output", str(out))
            assert r.returncode == 0, r.stderr
            out.unlink()


# -- audit manifest ---------------------------------------------------------


def _manifest(db: Path) -> Path:
    return db.parent / f"{db.stem}.audit.manifest.json"


def _markers(db: Path) -> list[str]:
    return [p.name for p in db.parent.iterdir() if ".corrupt-" in p.name]


class TestAuditManifest:
    def test_parse_refuses_a_newer_version_as_unavailable(self):
        raw = json.dumps({"version": 2, "files": []}).encode()
        with pytest.raises(audit_module._ManifestNewer) as ei:
            _parse_manifest_bytes(raw, "mem")
        assert isinstance(ei.value, audit_module._ManifestUnavailable)
        assert not isinstance(ei.value, audit_module._ManifestQuarantined)
        # An OSError: none of the catches that lead to quarantine take it.
        assert not isinstance(ei.value, audit_module._UNPARSEABLE_JSON)

    def test_the_version_is_checked_before_any_field(self):
        # A newer shape this release would reject still reads as NEWER.
        raw = json.dumps({"version": 99, "files": {"by_period": {}}}).encode()
        with pytest.raises(audit_module._ManifestNewer):
            _parse_manifest_bytes(raw, "mem")

    @pytest.mark.parametrize("value", [1, True, "2", None])
    def test_only_an_integer_greater_than_1_is_newer(self, value):
        raw = json.dumps({"version": value, "files": []}).encode()
        assert _parse_manifest_bytes(raw, "mem")["version"] == value

    def _trail_with_newer_manifest(self, tmp_path, reshape: bool) -> tuple[Path, bytes]:
        db = tmp_path / "mem.db"
        AuditTrail(db).log("record", {"id": "1"})
        data = json.loads(_manifest(db).read_text())
        data["version"] = 99
        if reshape:
            data["files"] = {"by_period": {}}
        _manifest(db).write_text(json.dumps(data))
        return db, _manifest(db).read_bytes()

    @pytest.mark.parametrize("reshape", [False, True])
    def test_verify_reports_unavailable(self, tmp_path, reshape):
        db, before = self._trail_with_newer_manifest(tmp_path, reshape)
        result = AuditTrail.verify(db)
        assert not result.valid and not result.anchor_trusted
        assert result.error.startswith("Manifest unavailable:")
        assert "newer anneal-memory" in result.error
        assert _manifest(db).read_bytes() == before

    @pytest.mark.parametrize("reshape", [False, True])
    def test_a_write_never_quarantines_or_rewrites_it(self, tmp_path, reshape):
        """RUN case A2: the reshaped newer manifest was quarantined by the
        next record and audit-repair rebuilt version 1 over it."""
        db, before = self._trail_with_newer_manifest(tmp_path, reshape)
        AuditTrail(db).log("record", {"id": "2"})  # a fresh process's append
        assert _manifest(db).read_bytes() == before
        assert _markers(db) == []

    def test_an_append_that_needs_the_manifest_is_refused(self, tmp_path):
        # No active file: the first append seeds its chain from the manifest.
        db = tmp_path / "mem.db"
        _manifest(db).write_text(json.dumps({"version": 99, "files": {"x": 1}}))
        before = _manifest(db).read_bytes()
        with pytest.raises(audit_module._ManifestNewer):
            AuditTrail(db).log("record", {"id": "1"})
        assert _manifest(db).read_bytes() == before
        assert _markers(db) == []

    def test_a_week_rotation_under_a_newer_manifest_seals_nothing(self, tmp_path):
        db, before = self._trail_with_newer_manifest(tmp_path, reshape=False)
        trail = AuditTrail(db)
        active = db.parent / f"{db.stem}.audit.jsonl"
        trail.log("record", {"id": "2"})  # initializes: _initialize sets _last_week itself
        active_before = active.read_bytes()
        trail._last_week = "1999-W01"  # the week rolled over, AFTER the init
        reached = []
        real = trail._rotate_locked
        trail._rotate_locked = lambda *a, **k: (reached.append(a), real(*a, **k))[1]
        trail.log("record", {"id": "3"})
        assert len(reached) == 1, "the log must reach the rotation body, or this proves nothing"
        assert _manifest(db).read_bytes() == before
        assert list(db.parent.glob("*.jsonl.gz")) == []
        assert active.read_bytes().startswith(active_before)  # not sealed away
        assert _markers(db) == []

    @pytest.mark.parametrize("reshape", [False, True])
    def test_audit_repair_refuses(self, tmp_path, reshape):
        db, before = self._trail_with_newer_manifest(tmp_path, reshape)
        result = AuditTrail.repair_manifest(db)
        assert not result.repaired
        assert "newer anneal-memory" in result.error
        assert _manifest(db).read_bytes() == before
        assert _markers(db) == []

    def test_audit_repair_refuses_a_newer_manifest_beside_a_marker(self, tmp_path):
        # A marker skips the manifest read and rebuilds over the path: a newer
        # manifest saved there is refused, never rebuilt over.
        db, before = self._trail_with_newer_manifest(tmp_path, reshape=False)
        marker = db.parent / f"{_manifest(db).name}.corrupt-20261014T000000000000Z"
        marker.write_text("{not json")
        result = AuditTrail.repair_manifest(db)
        assert not result.repaired
        assert "newer anneal-memory" in result.error and "nothing was written" in result.error
        assert _manifest(db).read_bytes() == before
        assert marker.exists()

    @_NO_CHMOD
    def test_audit_repair_refuses_an_unreadable_manifest_beside_a_marker(self, tmp_path):
        """L3 r1 F-B (codex HIGH), RUN: the marker peek swallowed every error, so a
        mode-000 manifest beside a marker was rebuilt over."""
        db, before = self._trail_with_newer_manifest(tmp_path, reshape=False)
        marker = db.parent / f"{_manifest(db).name}.corrupt-20261014T000000000000Z"
        marker.write_text("{not json")
        with _unreadable(_manifest(db)):
            result = AuditTrail.repair_manifest(db)
        assert not result.repaired
        assert "could not be read" in result.error and "nothing was rebuilt" in result.error
        assert _manifest(db).read_bytes() == before
        assert marker.exists() and not list(db.parent.glob("*.repaired"))

    @pytest.mark.parametrize("manifest", [None, b"{not json", b"[]"])
    def test_audit_repair_still_rebuilds_an_absent_or_invalid_manifest_beside_a_marker(
        self, tmp_path, manifest
    ):
        db = tmp_path / "mem.db"
        AuditTrail(db).log("record", {"id": "1"})
        if manifest is None:
            _manifest(db).unlink()
        else:
            _manifest(db).write_bytes(manifest)
        marker = db.parent / f"{_manifest(db).name}.corrupt-20261014T000000000000Z"
        marker.write_text("{not json")
        result = AuditTrail.repair_manifest(db)
        assert result.repaired, result.error

    def test_cli_audit_says_newer_not_corrupt(self, tmp_path):
        db, _ = self._trail_with_newer_manifest(tmp_path, reshape=True)
        r = _cli(db, "audit", "--limit", "1")
        assert r.returncode == 0
        assert "written by a newer anneal-memory" in r.stderr
        assert "corrupt" not in r.stderr


# -- derive-trust.json ------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design")
class TestDeriveTrust:
    @pytest.fixture
    def trust(self, tmp_path, monkeypatch):
        d = tmp_path / "trustdir"
        d.mkdir(mode=0o700)
        tf = d / "derive-trust.json"
        monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(tf))
        root = tmp_path / "root"
        root.mkdir(mode=0o700)
        db = tmp_path / "mem.db"
        allow_store(db, root)
        return tf, db, root

    def _set_version(self, tf: Path, value) -> bytes:
        data = json.loads(tf.read_text())
        data["version"] = value
        data["field_from_the_future"] = True
        tf.write_text(json.dumps(data))
        os.chmod(tf, 0o600)
        return tf.read_bytes()

    def test_this_release_writes_version_2_and_trusts_it(self, trust):
        tf, db, root = trust
        assert json.loads(tf.read_text())["version"] == 2
        assert trusted_roots(db)

    @pytest.mark.parametrize("value,message", [
        (3, "written by a newer anneal-memory"),
        (99, "written by a newer anneal-memory"),
        (True, "unrecognised version"),
        ("2", "unrecognised version"),
        (2.0, "unrecognised version"),
        (None, "unrecognised version"),
    ])
    def test_a_version_not_an_int_le_2_trusts_nothing(self, trust, value, message):
        tf, db, _ = trust
        self._set_version(tf, value)
        assert _load_trust(tf) == []
        assert trusted_roots(db) == {}
        with pytest.raises(ValueError, match=message):
            _load_trust(tf, strict=True)

    @pytest.mark.parametrize("value", [1, 2])
    def test_versions_this_release_reads(self, trust, value):
        tf, db, _ = trust
        self._set_version(tf, value)
        assert len(_load_trust(tf)) == 1

    def test_no_version_key_keeps_the_old_reading(self, trust):
        tf, db, _ = trust
        data = json.loads(tf.read_text())
        del data["version"]
        tf.write_text(json.dumps(data))
        os.chmod(tf, 0o600)
        assert len(_load_trust(tf)) == 1

    def test_allow_and_revoke_never_overwrite_a_newer_file(self, trust, tmp_path):
        """RUN case T: both rewrote a version-99 file as version 2."""
        tf, db, _ = trust
        before = self._set_version(tf, 99)
        root2 = tmp_path / "root2"
        root2.mkdir(mode=0o700)
        with pytest.raises(ValueError, match="written by a newer anneal-memory"):
            allow_store(db, root2)
        assert tf.read_bytes() == before
        with pytest.raises(ValueError, match="written by a newer anneal-memory"):
            revoke_store(db)
        assert tf.read_bytes() == before


# -- the generation ties the sidecar formats together -----------------------


def test_a_sidecar_format_constant_moves_only_with_a_generation_decision():
    from anneal_memory import store as store_module
    from anneal_memory.rederive import _TRUST_VERSION

    assert (store_module._SCHEMA_VERSION, store_module._META_FORMAT_VERSION,
            audit_module._MANIFEST_VERSION, _TRUST_VERSION) == (3, 1, 1, 2), (
        "changing a store-attached sidecar constant (meta, manifest) requires "
        "raising _SCHEMA_VERSION; _TRUST_VERSION is per-user, not covered by "
        "the generation, and needs its own newer-refusal kept."
    )
