"""Tests for AM-CONTLOCK — the shared cross-process continuity lock.

`continuity_lock` is the coordination primitive an external editor (e.g. Levain's
governed State write) and anneal's own consolidate both take so a wrap landing
between the editor's read and its `os.replace` can't silently clobber. These tests
prove: it's a REAL exclusive flock (same-process two-fd AND true cross-process),
it releases on exit, it degrades to a no-op without fcntl, and both anneal
continuity writers (the canonical pipeline + the bare write) actually take it.
"""

import contextlib
import os
import subprocess
import sys
import textwrap
import time

import pytest

import anneal_memory.store as store_module
from anneal_memory import Store, continuity_lock, prepare_wrap, validated_save_continuity

fcntl = pytest.importorskip("fcntl")  # the whole point is POSIX advisory locking


# A minimal valid default-schema continuity text for driving a real wrap.
_WRAP_TEXT = (
    "# T — Memory (v1)\n\n"
    "## State\nActive.\n\n"
    "## Patterns\nNone yet.\n\n"
    "## Decisions\nNone.\n\n"
    "## Context\nFirst session.\n"
)


def _lock_path(continuity_path) -> str:
    return str(continuity_path) + ".lock"


def test_lock_creates_sidecar_and_releases(tmp_path):
    cp = tmp_path / "memory.continuity.md"
    lock_path = _lock_path(cp)
    with continuity_lock(cp):
        assert os.path.exists(lock_path), "the .lock sidecar is created on acquire"
    # After release, a fresh non-blocking exclusive acquire must succeed.
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must NOT raise
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_lock_is_exclusive_same_process(tmp_path):
    """While held, a second open file description cannot take it (real LOCK_EX)."""
    cp = tmp_path / "memory.continuity.md"
    lock_path = _lock_path(cp)
    with continuity_lock(cp):
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


def test_lock_excludes_across_processes(tmp_path):
    """The load-bearing guarantee: a DIFFERENT process holding the lock blocks us.
    A child acquires + holds it; while it's held the parent's non-blocking acquire
    must fail; once the child exits the parent's acquire succeeds."""
    cp = tmp_path / "memory.continuity.md"
    cp.write_text("seed", encoding="utf-8")
    acquired = tmp_path / "acquired"
    release = tmp_path / "release"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import os, time
                from anneal_memory import continuity_lock
                with continuity_lock({str(cp)!r}):
                    open({str(acquired)!r}, "w").close()
                    # hold until the parent signals (bounded so a wedged parent
                    # can't strand the child forever).
                    for _ in range(200):
                        if os.path.exists({str(release)!r}):
                            break
                        time.sleep(0.01)
                """
            ),
        ]
    )
    try:
        for _ in range(500):  # up to ~5s for the child to acquire
            if acquired.exists():
                break
            time.sleep(0.01)
        assert acquired.exists(), "child failed to acquire the lock"

        fd = os.open(_lock_path(cp), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(BlockingIOError):  # child holds it cross-process
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    finally:
        release.write_text("go", encoding="utf-8")
        child.wait(timeout=10)

    # Child gone → the lock is free.
    fd = os.open(_lock_path(cp), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_lock_degrades_to_noop_without_fcntl(tmp_path, monkeypatch):
    """On a non-POSIX / lock-less platform (fcntl is None) the lock is a working
    no-op context manager and creates no sidecar — atomic os.replace still protects
    a single writer; cross-process serialization is simply not available."""
    monkeypatch.setattr(store_module, "fcntl", None)
    cp = tmp_path / "memory.continuity.md"
    with continuity_lock(cp):
        pass  # must not raise
    assert not os.path.exists(_lock_path(cp)), "no lockfile is opened when fcntl is None"


def test_store_method_keys_on_continuity_path(tmp_path):
    store = Store(str(tmp_path / "memory.db"))
    with store.continuity_lock():
        assert os.path.exists(_lock_path(store.continuity_path))


def test_validated_save_takes_the_lock(tmp_path, monkeypatch):
    """The canonical pipeline wraps its Phase-3 externalization in the lock."""
    store = Store(str(tmp_path / "memory.db"), project_name="T")
    store.record("An observation worth at least a few words here.", "observation")

    entered = []
    real = store.continuity_lock

    @contextlib.contextmanager
    def spy():
        entered.append(True)
        with real():
            yield

    monkeypatch.setattr(store, "continuity_lock", spy)
    prepare_wrap(store)
    validated_save_continuity(store, _WRAP_TEXT)
    assert entered == [True], "validated_save_continuity entered the continuity lock exactly once"
    assert os.path.exists(_lock_path(store.continuity_path))


def test_bare_save_continuity_takes_the_lock(tmp_path, monkeypatch):
    """The bare (immune-bypassing) write is a secondary continuity writer and takes
    the SAME lock — the invariant is 'every continuity-file writer holds the lock'."""
    store = Store(str(tmp_path / "memory.db"))

    entered = []
    real = store_module.continuity_lock

    @contextlib.contextmanager
    def spy(path):
        entered.append(str(path))
        with real(path):
            yield

    monkeypatch.setattr(store_module, "continuity_lock", spy)
    store.save_continuity("# T\n\n## State\nx\n")
    assert entered == [str(store.continuity_path)]
    assert os.path.exists(_lock_path(store.continuity_path))


# --- codex L3 HIGH: the Phase-1 tmp write is unique per save attempt ---------
# A deterministic token-only tmp path let two concurrent saves sharing one wrap
# snapshot collide — the loser's bytes could be renamed under the winner's
# committed DB row (silent file/DB divergence), since the tmp write precedes the
# CAS and sits outside the Phase-3 flock. The fix: a <token12>-<uuid8> pair id.

def test_canonical_save_uses_unique_pair_id(tmp_path, monkeypatch):
    """The canonical pipeline passes a <token12>-<uuid8> pair id (NOT the bare
    deterministic wrap token) to BOTH tmp writes — unique per attempt, still
    paired for operator recovery."""
    store = Store(str(tmp_path / "memory.db"), project_name="T")
    store.record("an observation with enough characters to be a valid episode.", "observation")

    captured: dict[str, str] = {}
    realc = store._prepare_continuity_write
    realm = store._prepare_meta_write

    def cspy(text, token_hex=None):
        captured["cont"] = token_hex
        return realc(text, token_hex=token_hex)

    def mspy(meta, token_hex=None):
        captured["meta"] = token_hex
        return realm(meta, token_hex=token_hex)

    monkeypatch.setattr(store, "_prepare_continuity_write", cspy)
    monkeypatch.setattr(store, "_prepare_meta_write", mspy)
    res = prepare_wrap(store)
    validated_save_continuity(store, _WRAP_TEXT)

    pair = captured["cont"]
    assert pair == captured["meta"], "continuity + meta tmp share ONE pair id (recovery pairing)"
    prefix, sep, uniq = pair.partition("-")
    assert sep == "-" and len(prefix) == 12 and len(uniq) == 8 and uniq
    assert prefix == res["wrap_token"][:12], "the pair-id prefix IS the wrap-token prefix"


def test_orphan_detection_pairs_by_pair_id_and_filters_by_token_prefix(tmp_path):
    """Orphan grouping keys on the FULL pair id (so two crashed saves sharing a
    token prefix stay separate); the active-wrap filter matches on the token
    PREFIX (the part before '-'), since the live wrap is known only by its token."""
    store = Store(str(tmp_path / "memory.db"), project_name="T")
    store.record("an observation with enough characters to be a valid episode.", "observation")
    cp = store.continuity_path

    # The parser returns the full pair id; its first '-'-component is the token prefix.
    md_tmp = cp.with_name(f"{cp.stem}.abc123def456-0a1b2c3d.md.tmp")
    assert store._token_from_orphan(md_tmp) == "abc123def456-0a1b2c3d"
    assert store._token_from_orphan(md_tmp).split("-", 1)[0] == "abc123def456"

    # In-flight (active wrap) tmp is filtered by token prefix; a different crashed
    # pair is still reported.
    res = prepare_wrap(store)
    token = res["wrap_token"]
    inflight = cp.with_name(f"{cp.stem}.{token[:12]}-deadbeef.md.tmp")
    inflight.write_text("in flight", encoding="utf-8")
    orphan = cp.with_name(f"{cp.stem}.999999999999-cafebabe.md.tmp")
    orphan.write_text("orphan", encoding="utf-8")
    found = {o.name for o in store._find_orphan_tmp_files()}
    assert inflight.name not in found, "in-flight tmp (active token prefix) is filtered out"
    assert orphan.name in found, "a different-token crashed tmp is still reported"


# --- codex L3 round-4 MEDIUM: degrade on a lock-LESS POSIX FS (runtime flock fail) ---
# `continuity_lock` only no-op'd when `fcntl is None` (import fails). A lock-less FS
# (some NFS) imports fcntl but `flock` raises ENOLCK/EOPNOTSUPP — that must ALSO
# degrade to a no-op, not propagate, since validated_save_continuity takes the lock
# AFTER its DB commit (a propagating error would brick every wrap there).

def test_lock_degrades_when_flock_raises_lock_unavailable(tmp_path, monkeypatch):
    """ENOLCK from flock → degrade to a no-op (yield unlocked), do not propagate."""
    import errno as _errno

    class _NoLockFcntl:
        LOCK_EX = fcntl.LOCK_EX

        @staticmethod
        def flock(fd, op):
            raise OSError(_errno.ENOLCK, "no locks available")

    monkeypatch.setattr(store_module, "fcntl", _NoLockFcntl)
    entered = False
    with continuity_lock(tmp_path / "memory.continuity.md"):
        entered = True
    assert entered, "continuity_lock yielded (degraded) despite flock ENOLCK"


def test_lock_reraises_non_lock_unavailable_flock_error(tmp_path, monkeypatch):
    """A flock OSError that is NOT lock-unavailable (e.g. EBADF) is a real bug →
    it must propagate, not be swallowed by the degradation path."""
    import errno as _errno

    class _BadFdFcntl:
        LOCK_EX = fcntl.LOCK_EX

        @staticmethod
        def flock(fd, op):
            raise OSError(_errno.EBADF, "bad file descriptor")

    monkeypatch.setattr(store_module, "fcntl", _BadFdFcntl)
    with pytest.raises(OSError):
        with continuity_lock(tmp_path / "memory.continuity.md"):
            pass


# --- spore-091 #1: the sidecar derives from the RESOLVED file identity ---------
# (weekly-code-review codex catch 2026-06-15) Pre-fix `continuity_lock` derived the
# `.lock` from the path STRING, so a symlinked continuity file → two `.lock`
# spellings → two inodes → silent NON-serialization (the read→replace lost-update
# class). The fix `.resolve()`s inside the primitive so every caller lands on one
# lock inode regardless of spelling.

def test_lock_resolves_symlinked_continuity_path_to_one_inode(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = real_dir / "memory.continuity.md"
    real.write_text("seed", encoding="utf-8")
    link = tmp_path / "linked.continuity.md"
    link.symlink_to(real)

    resolved_lock = str(real) + ".lock"
    with continuity_lock(link):  # the symlinked spelling
        # the sidecar sits next to the RESOLVED target, never next to the symlink
        assert os.path.exists(resolved_lock), "lock not created at the resolved path"
        assert not os.path.exists(str(link) + ".lock"), "lock leaked to the symlink spelling"
        # a second acquirer using the resolved spelling is EXCLUDED — same inode.
        fd = os.open(resolved_lock, os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)


# --- spore-091 #2: require=True fails CLOSED; the manager yields acquired:bool ---
# A governed external editor (Levain's State write) with no 2PC / recovery net of
# its own must NOT proceed unserialized on a lock-less FS. require=True raises
# instead of degrading; require=False keeps anneal's best-effort no-op (yields False).

def test_lock_yields_true_when_held(tmp_path):
    with continuity_lock(tmp_path / "memory.continuity.md") as held:
        assert held is True


def test_require_false_yields_false_on_degradation(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "fcntl", None)
    with continuity_lock(tmp_path / "memory.continuity.md") as held:
        assert held is False  # degraded — best-effort, not a raise


def test_require_true_raises_when_fcntl_none(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "fcntl", None)
    with pytest.raises(store_module.ContinuityLockUnavailable):
        with continuity_lock(tmp_path / "memory.continuity.md", require=True):
            pass


def test_require_true_raises_on_lockless_fs(tmp_path, monkeypatch):
    import errno as _errno

    class _NoLockFcntl:
        LOCK_EX = fcntl.LOCK_EX

        @staticmethod
        def flock(fd, op):
            raise OSError(_errno.ENOLCK, "no locks available")

    monkeypatch.setattr(store_module, "fcntl", _NoLockFcntl)
    with pytest.raises(store_module.ContinuityLockUnavailable):
        with continuity_lock(tmp_path / "memory.continuity.md", require=True):
            pass


def test_store_method_require_true_raises_on_degradation(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "fcntl", None)
    store = Store(str(tmp_path / "memory.db"))
    with pytest.raises(store_module.ContinuityLockUnavailable):
        with store.continuity_lock(require=True):
            pass
