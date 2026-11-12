"""Schema v2 of the spore/intention store: open rows under ``intentions``; a v1 store
(open rows under ``spores``) is refused with an instruction and converted once by
``SporeStore.migrate_to_v2`` / ``anneal-memory intention migrate-key``.
"""

from __future__ import annotations

import json
from datetime import date
import subprocess
import sys
from pathlib import Path

import pytest

from anneal_memory.spores import (
    LEGACY_OPEN_KEY,
    OPEN_KEY,
    SPORE_SCHEMA_VERSION,
    SporeError,
    SporeSchemaError,
    SporeStore,
    spore_version,
)

T0 = "2026-11-01"
D0 = date(2026, 11, 1)


def _v1_doc() -> dict:
    """A v1 document as the 1.0.7 store wrote it: a custom top-level key, a tombstone
    list, resolved rows, and the open key BEFORE 'resolved' (position is preserved)."""
    open_row = {
        "id": "intention-001", "type": "task", "text": "open one", "domain": "", "tier": "warm",
        "salience": 1, "seen": T0, "created": T0, "status": "open", "origin_key": "a" * 32,
    }
    open_two = dict(open_row, id="spore-002", text="é open two", origin_key="b" * 32)
    done = dict(open_row, id="intention-003", status="resolved", origin_key="c" * 32,
                resolution={"direction": "descend", "kind": "done", "on": T0})
    return {
        "spores": [open_row, open_two],
        "resolved": [done],
        "deleted": [{"id": "spore-004", "origin_key": "d" * 32}],
        "host_note": {"keep": ["me"]},
        "schema_version": 1,
    }


@pytest.fixture
def v1_store(tmp_path: Path) -> SporeStore:
    path = tmp_path / "x.spores.json"
    path.write_text(json.dumps(_v1_doc(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SporeStore(path)


def test_constants():
    assert (SPORE_SCHEMA_VERSION, OPEN_KEY, LEGACY_OPEN_KEY) == (2, "intentions", "spores")


def test_a_new_store_is_born_v2(tmp_path):
    store = SporeStore(tmp_path / "n.json")
    item = store.add(type="task", text="t", today=D0)
    doc = json.loads(store.path.read_text())
    assert doc["schema_version"] == 2
    assert [r["id"] for r in doc["intentions"]] == [item["id"]]
    assert "spores" not in doc


def test_a_v1_store_is_refused_with_the_instruction_and_never_touched(v1_store):
    before = v1_store.path.read_bytes()
    for call in (
        lambda: v1_store.list_open(),
        lambda: v1_store.get("intention-001"),
        lambda: v1_store.add(type="task", text="t", today=D0),
        lambda: v1_store.touch("intention-001"),
        lambda: v1_store.surface(),
    ):
        with pytest.raises(SporeSchemaError) as exc:
            call()
        msg = str(exc.value)
        assert "schema v1" in msg and "migrate-key" in msg and str(v1_store.path) in msg
    assert v1_store.path.read_bytes() == before


def test_the_refusal_is_a_spore_error(v1_store):
    with pytest.raises(SporeError):
        v1_store.list_open()


def test_a_v1_store_with_no_version_field_is_refused_not_read_as_empty(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"spores": [{"id": "spore-001", "type": "task"}]}))
    with pytest.raises(SporeSchemaError):
        SporeStore(path).list_open()


def test_an_empty_object_is_a_fresh_v2_store(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("{}")
    assert SporeStore(path).list_open() == []


def test_a_v2_store_still_carrying_the_old_key_is_refused(tmp_path):
    path = tmp_path / "both.json"
    path.write_text(json.dumps({"intentions": [], "spores": [], "resolved": [], "schema_version": 2}))
    with pytest.raises(SporeSchemaError, match="still carries"):     # a SCHEMA refusal: the CLI prints it as one clean line
        SporeStore(path).list_open()


def test_a_newer_schema_is_still_refused(tmp_path):
    path = tmp_path / "n.json"
    path.write_text(json.dumps({"intentions": [], "resolved": [], "schema_version": 3}))
    with pytest.raises(SporeSchemaError, match="newer"):
        SporeStore(path).list_open()


def test_migrate_converts_in_place_and_keeps_every_row_and_key(v1_store, tmp_path):
    v1 = _v1_doc()
    before_version = spore_version(v1["spores"][0])
    result = v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    assert result["status"] == "migrated"
    assert (result["open"], result["resolved"], result["deleted"]) == (2, 1, 1)
    doc = json.loads(v1_store.path.read_text(encoding="utf-8"))
    assert list(doc) == ["intentions", "resolved", "deleted", "host_note", "schema_version"]
    assert doc["intentions"] == v1["spores"] and doc["resolved"] == v1["resolved"]
    assert doc["deleted"] == v1["deleted"] and doc["host_note"] == v1["host_note"]
    assert doc["schema_version"] == 2
    # ids, origin keys and versions a caller holds stay valid
    assert spore_version(v1_store.get("intention-001")) == before_version
    assert v1_store.get("spore-002")["origin_key"] == "b" * 32
    # and it is a working v2 store afterwards: the counter runs across the deleted id
    assert v1_store.add(type="task", text="n", today=D0)["id"].endswith("-005")


def test_migrate_backup_is_byte_exact_and_never_overwritten(v1_store, tmp_path):
    raw = v1_store.path.read_bytes()
    result = v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    backup = Path(result["backup"])
    assert backup.parent == tmp_path / "bk" and backup.name.startswith("x.spores.json.v1-")
    assert backup.read_bytes() == raw
    if sys.platform != "win32":
        assert (backup.stat().st_mode & 0o777) == 0o600


def test_migrate_default_backup_sits_beside_the_file(v1_store):
    result = v1_store.migrate_to_v2()
    assert Path(result["backup"]).parent == v1_store.path.parent


def test_migrate_twice_is_a_no_op(v1_store, tmp_path):
    v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    after = v1_store.path.read_bytes()
    again = v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    assert again["status"] == "already-current" and again["backup"] is None
    assert v1_store.path.read_bytes() == after
    assert len(list((tmp_path / "bk").iterdir())) == 1


def test_dry_run_writes_nothing_not_even_a_backup(v1_store, tmp_path):
    before = v1_store.path.read_bytes()
    result = v1_store.migrate_to_v2(backup_dir=tmp_path / "bk", dry_run=True)
    assert result["status"] == "would-migrate" and result["schema_version"] == 1
    assert (result["open"], result["resolved"]) == (2, 1)
    assert v1_store.path.read_bytes() == before
    assert not (tmp_path / "bk").exists()


@pytest.mark.parametrize("doc", [
    {"spores": [], "intentions": [], "resolved": [], "schema_version": 1},   # both keys
    {"spores": ["x"], "resolved": [], "schema_version": 1},                   # a row that is no object
    {"spores": [], "resolved": {}, "schema_version": 1},                      # resolved is not a list
    {"spores": [], "resolved": [], "schema_version": 3},                      # a schema it does not know
    {"spores": [], "resolved": [], "schema_version": True},                   # a bool is not a version
    {"spores": [], "resolved": [], "schema_version": 0},                      # an explicit version below 1 is unknown
    {"spores": [], "resolved": [], "schema_version": -1},
    {"spores": [], "resolved": [], "deleted": ["x"], "schema_version": 1},    # a corrupt tombstone list would fail the next add
    {"resolved": [], "schema_version": 1},                                    # no open key at all
])
def test_migrate_refuses_what_it_cannot_convert_and_leaves_it_alone(tmp_path, doc):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    before = path.read_bytes()
    with pytest.raises(SporeError):
        SporeStore(path).migrate_to_v2(backup_dir=tmp_path / "bk")
    assert path.read_bytes() == before
    assert not (tmp_path / "bk").exists()


def test_migrate_refuses_a_missing_or_unreadable_file(tmp_path):
    with pytest.raises(SporeError, match="does not exist"):
        SporeStore(tmp_path / "none.json").migrate_to_v2()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(SporeError, match="not readable JSON"):
        SporeStore(bad).migrate_to_v2()


def test_migrate_holds_the_store_lock_through_the_rewrite(v1_store, monkeypatch):
    fcntl = pytest.importorskip("fcntl")             # POSIX only; Windows runs unlocked
    import os

    seen = {}
    real_save = SporeStore._save

    def probing_save(self, data):
        fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen["held"] = False
            except BlockingIOError:
                seen["held"] = True
        finally:
            os.close(fd)
        return real_save(self, data)

    monkeypatch.setattr(SporeStore, "_save", probing_save)
    v1_store.migrate_to_v2()
    assert seen == {"held": True}


def test_a_conversion_that_does_not_match_raises_naming_the_backup(v1_store, monkeypatch, tmp_path):
    real_save = SporeStore._save

    def lossy_save(self, data):
        data = dict(data, intentions=data["intentions"][:1])  # drops a row
        return real_save(self, data)

    monkeypatch.setattr(SporeStore, "_save", lossy_save)
    with pytest.raises(SporeError, match="does not match") as exc:
        v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    backup = next((tmp_path / "bk").iterdir())
    assert str(backup) in str(exc.value)


ROOT = Path(__file__).resolve().parent.parent


def _cli(*argv: str, cwd: Path):
    # run from the checkout so `-m anneal_memory` is THIS tree, not an installed copy
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory", *argv], capture_output=True, text=True, cwd=ROOT)


def test_cli_migrate_key_json_and_the_clean_refusal(tmp_path):
    db = tmp_path / "m.db"
    store = tmp_path / "m.spores.json"
    store.write_text(json.dumps(_v1_doc()))
    refused = _cli("--db", str(db), "intention", "list", cwd=tmp_path)
    assert refused.returncode == 1 and refused.stdout == ""
    assert refused.stderr.startswith("Error: ") and "migrate-key" in refused.stderr
    assert "Traceback" not in refused.stderr
    dry = _cli("--db", str(db), "intention", "migrate-key", "--dry-run", "--json", cwd=tmp_path)
    assert dry.returncode == 0 and json.loads(dry.stdout)["status"] == "would-migrate"
    done = _cli("--db", str(db), "intention", "migrate-key", "--json", "--backup-dir", str(tmp_path / "bk"),
                cwd=tmp_path)
    assert done.returncode == 0
    out = json.loads(done.stdout)
    assert out["status"] == "migrated" and Path(out["backup"]).exists()
    listed = _cli("--db", str(db), "intention", "list", "--json", cwd=tmp_path)
    assert listed.returncode == 0 and len(json.loads(listed.stdout)) == 2


def test_cli_migrate_key_takes_an_explicit_file(v1_store, tmp_path):
    out = _cli("intention", "migrate-key", "--file", str(v1_store.path), cwd=tmp_path)
    assert out.returncode == 0 and out.stdout.startswith("migrated: ")
    missing = _cli("intention", "migrate-key", "--file", str(tmp_path / "none.json"), cwd=tmp_path)
    assert missing.returncode == 1 and "does not exist" in missing.stderr


def test_load_and_migrate_classify_a_document_the_same_way(tmp_path):
    """_load tells the operator to run migrate-key only for a document migrate converts."""
    from anneal_memory.spores import _is_convertible_v1

    docs = {
        "no_version": {"spores": []},
        "null_version": {"spores": [], "schema_version": None},
        "v0": {"spores": [], "schema_version": 0},
        "v1": {"spores": [], "schema_version": 1},
        "v1_no_spores_key": {"resolved": [], "schema_version": 1},
        "both_keys_v1": {"spores": [], "intentions": [], "schema_version": 1},
        "float_version": {"spores": [], "schema_version": 1.0},
        "negative_version": {"spores": [], "schema_version": -1},
        "non_object_row": {"spores": ["x"], "schema_version": 1},
        "resolved_not_a_list": {"spores": [], "resolved": {}, "schema_version": 1},
        "corrupt_tombstones": {"spores": [], "deleted": ["x"], "schema_version": 1},
    }
    for name, doc in docs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(SporeSchemaError) as exc:
            SporeStore(path).list_open()
        says_migrate = "intention migrate-key" in str(exc.value)
        assert says_migrate == _is_convertible_v1(doc), name
        if says_migrate:
            assert SporeStore(path).migrate_to_v2(backup_dir=tmp_path / "bk")["status"] == "migrated", name
        else:
            before = path.read_bytes()
            with pytest.raises(SporeError):
                SporeStore(path).migrate_to_v2(backup_dir=tmp_path / "bk")
            assert path.read_bytes() == before, name


def test_migrate_of_an_empty_object_is_already_current(tmp_path):
    path = tmp_path / "e.json"
    path.write_text("{}")
    assert SporeStore(path).migrate_to_v2()["status"] == "already-current"


def test_a_retry_in_the_same_second_takes_the_next_backup_name(v1_store, tmp_path, monkeypatch):
    real_save = SporeStore._save
    calls = {"n": 0}

    def failing_once(self, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_save(self, data)

    monkeypatch.setattr(SporeStore, "_save", failing_once)
    raw = v1_store.path.read_bytes()
    with pytest.raises(SporeError, match="disk full"):
        v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    assert v1_store.path.read_bytes() == raw            # the failed attempt changed nothing
    second = v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    assert second["status"] == "migrated"
    assert len(list((tmp_path / "bk").iterdir())) == 2   # two distinct backups, no FileExistsError


def test_dry_run_and_a_missing_file_plant_no_directory_or_lock(tmp_path):
    ghost = tmp_path / "typo" / "dir" / "x.json"
    with pytest.raises(SporeError, match="does not exist"):
        SporeStore(ghost).migrate_to_v2(dry_run=True)
    assert not (tmp_path / "typo").exists()
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(_v1_doc()))
    SporeStore(path).migrate_to_v2(dry_run=True)
    assert not path.with_name(path.name + ".lock").exists()


def test_the_instruction_quotes_a_path_with_a_space(tmp_path):
    path = tmp_path / "with space" / "s.spores.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_v1_doc()))
    with pytest.raises(SporeSchemaError) as exc:
        SporeStore(path).list_open()
    assert f"--file '{path}'" in str(exc.value)


def test_a_backup_that_cannot_complete_leaves_no_partial_file_and_the_store_untouched(v1_store, tmp_path, monkeypatch):
    from anneal_memory import spores

    def boom(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(spores, "_full_fsync", boom)
    before = v1_store.path.read_bytes()
    with pytest.raises(SporeError, match="could not write the backup"):
        v1_store.migrate_to_v2(backup_dir=tmp_path / "bk")
    assert v1_store.path.read_bytes() == before
    assert list((tmp_path / "bk").iterdir()) == []         # the partial .bak was removed, not left as a recovery-looking artifact
