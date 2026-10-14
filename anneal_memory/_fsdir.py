"""Best-effort POSIX directory fsync — the one shared implementation.

``store``, ``audit``, ``crystal`` and ``migration`` (and their callers
``cli`` and ``continuity``) all import :func:`_fsync_dir` from this LEAF
module: stdlib-only, importing nothing from the package, so no caller
gains a cycle or a heavy sibling. Before this module existed the helper
was duplicated byte-for-idiom four times, and the copies had drifted:

- ``audit`` left ``os.close`` unguarded in its ``finally``, so a rare
  ``OSError`` from close (or one masking a swallowed fsync error) could
  escape to callers that were promised best-effort.
- ``crystal`` and ``migration`` swallowed every error but reached their
  Windows no-op only through an exception-propelled path, without the
  explicit platform guard.
- ``store`` was strictest on both axes and is the surviving semantics.

The load-bearing property this helper must keep: it NEVER raises.
"""

from __future__ import annotations

import os
from pathlib import Path


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync for crash-safe atomic writes.

    On POSIX filesystems (ext4, xfs, btrfs, APFS), ``fsync(file)``
    guarantees the file's data blocks are on disk but does NOT
    guarantee the containing directory's entry for the file is
    durable. A crash between the file fsync and a subsequent
    ``rename(2)`` can leave the directory without an entry for the
    tmp file — the rename then fails with ``ENOENT`` even though the
    data was durable. Syncing the parent directory closes this gap.

    Windows does NOT support ``os.open`` on a directory for fsync;
    on that platform this function is a no-op (an explicit
    ``os.name != "posix"`` check, not an exception-driven one). macOS
    ``fsync`` is weaker than Linux (true durability needs
    ``fcntl.F_FULLFSYNC``, which this helper does not call;
    ``spores._full_fsync`` does) but the directory entry sync still
    helps reduce the residual crash window.

    Swallows all OSError — this is a best-effort durability
    reinforcement, NOT a hard guarantee. Callers must not rely on
    this function either succeeding or raising; it's one layer of
    defense among several.
    """
    if os.name != "posix":
        return
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass