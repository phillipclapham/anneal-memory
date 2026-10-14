"""Direct tests for the shared ``_fsync_dir`` helper (1014+8 seat).

The helper's one load-bearing property is that it NEVER raises: every
``OSError`` it can meet — ``os.open``, ``os.fsync`` and ``os.close``
alike — is swallowed. The close-swallow exists because audit.py's former
duplicate left ``os.close`` unguarded in its ``finally`` and could escape
an ``OSError`` to callers the docstring promised best-effort; these tests
lock the property so a future copy-drift back to that shape fails here,
not under audit. The explicit ``os.name != "posix"`` guard (instead of
the exception-propelled no-op the crystal/migration duplicates relied
on) is locked the same way.
"""

import os

from anneal_memory._fsdir import _fsync_dir


def test_fsync_dir_swallows_open_failure(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("open failed")

    monkeypatch.setattr(os, "open", boom)
    _fsync_dir(tmp_path)


def test_fsync_dir_swallows_fsync_failure(tmp_path, monkeypatch):
    def boom(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", boom)
    _fsync_dir(tmp_path)


def test_fsync_dir_swallows_close_failure(tmp_path, monkeypatch):
    def boom(fd):
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", boom)
    _fsync_dir(tmp_path)


def test_fsync_dir_is_plain_noop_when_platform_is_not_posix(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: opened.append(args) or 3)
    monkeypatch.setattr(os, "name", "nt")
    _fsync_dir(tmp_path)
    assert opened == []