"""Migration-notify: tell a driving AI what changed on upgrade so it can
self-migrate its own instruction files (``CLAUDE.md`` / ``AGENTS.md`` /
``GEMINI.md``).

The problem this solves. An operator hand-tunes their agent's core
instruction files, the substrate (anneal-memory) evolves underneath them,
and the hand-tuned files silently drift from the current version. A new
feature can then land as a *conflict* with stale instructions rather than as
an addition — which is exactly how the ``spores`` prospective layer first
read as conflicting with a FLOW-style methodology instead of orthogonal to
it. There was no path from "the substrate changed" to "here's the core-file
edit that change suggests."

The fix (propose-diff, never clobber). Each release that wants to ping
adopters adds a structured :class:`MigrationEntry` to
:data:`MIGRATION_MANIFEST` describing what changed and the core-file edit it
suggests. ``anneal-memory migrate check`` emits the entries newer than the
adopter's last-acknowledged version (a per-store marker file); the driving
AI applies them to its own core files under operator review, then
``anneal-memory migrate ack`` advances the marker. **anneal never edits the
adopter's files** — it only proposes; the operator's hand-customizations are
never clobbered.

Granularity. The acknowledgement marker is a ``<db-stem>.migrate.json``
sibling of the episodic db (mirroring the spore store's sibling-file
pattern), so it tracks migration state per memory store — i.e. per project /
per entity, which is the granularity at which core instruction files live.
If several projects share ONE store, they share one marker, so an ``ack``
from one acknowledges for all — give each project its own store to track
them independently. ``ack`` also advances by *version*, so all manifest
entries sharing a version are acknowledged together (co-shipped features are
meant to be reviewed together; per-feature acknowledgement is out of scope
for this layer).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from . import __version__

if TYPE_CHECKING:
    from typing import Sequence


class MigrationEntry(TypedDict):
    """One self-migration proposal, keyed to the release that ships it.

    ``version`` is the anneal-memory version that introduced *this guidance*
    (the release shipping the entry), not necessarily the version that
    introduced the underlying feature — an adopter who upgraded across a
    feature without reconciling their core files still needs the proposal,
    and the marker compares against the guidance version.
    """

    version: str
    feature: str
    summary: str
    suggested_edit: str
    files: list[str]


# The default core instruction files a proposal can touch, across harnesses.
CORE_FILES: list[str] = ["CLAUDE.md", "AGENTS.md", "GEMINI.md"]


# Ordered oldest -> newest. APPEND new entries; never reorder or rewrite a
# shipped one (an adopter may have already acknowledged up to it).
MIGRATION_MANIFEST: list[MigrationEntry] = [
    {
        "version": "0.4.7",
        "feature": "AM-SPORES-BOUNDARY",
        "summary": (
            "anneal-memory 0.4.0 added `spores` — a parallel store for "
            "prospective open loops (typed tasks/questions/thoughts that must "
            "resolve), distinct from the four retrospective memory layers. If "
            "your instruction files describe the memory system, they likely "
            "predate spores and may read the feature as conflicting with your "
            "methodology rather than orthogonal to it."
        ),
        "suggested_edit": (
            "In the section of your file that describes anneal-memory, add a "
            "short 'Spores — prospective layer' note. Suggested text (use it "
            "directly so adopters don't each paraphrase a different version): "
            "\"anneal-memory also ships a parallel spore store for prospective "
            "open loops — tasks/questions/thoughts that must resolve. It is a "
            "parallel store, NOT a fifth memory layer. Two cuts: memory vs "
            "spores is a LIFECYCLE cut (memory accretes and never completes; a "
            "spore must close); methodology vs both is a PROCEDURE-vs-ITEM cut "
            "(your methodology is the process you run; memory and spores are "
            "the state it operates on). Your methodology OPERATES the spore "
            "store rather than competing with it.\" "
            "RECONCILE PRACTICE, NOT JUST VOCABULARY: if your methodology "
            "already captures open loops (next-actions at wrap, a scratchpad), "
            "surface to your operator that those can route into the typed spore "
            "store instead of an ad-hoc list or the continuity tail — don't "
            "leave prospective items living in a retrospective layer. "
            "Full reference: the anneal-memory agent-instructions snippet "
            "(examples/agent-instructions.example)."
        ),
        "files": list(CORE_FILES),
    },
    {
        "version": "0.4.7",
        "feature": "AM-MIGRATE-NOTIFY",
        "summary": (
            "anneal-memory now ships self-migration notices: when you upgrade, "
            "`migrate check` reports what changed and the core-file edits it "
            "suggests, so your hand-tuned instruction files don't silently "
            "drift from the substrate."
        ),
        "suggested_edit": (
            "Add one line to your upgrade habit (in your instruction files or "
            "your operator runbook): 'After upgrading anneal-memory, run "
            "`anneal-memory migrate check`, apply the proposals that fit — "
            "showing your operator the diff for anything load-bearing — then "
            "`anneal-memory migrate ack`.'"
        ),
        "files": list(CORE_FILES),
    },
    {
        "version": "0.4.8",
        "feature": "AM-PRESERVE-BARE-PATH",
        "summary": (
            "Demotion fix: a graduated Proven carried forward WITHOUT a fresh "
            "citation (a `name | Nx (date)` line with no `[evidence:]` tag, "
            "re-stamped to today) is now HELD at its level when it is at/below "
            "its earned high-water mark and was grounded within the warm window "
            "— instead of being demoted one level every wrap. This completes "
            "AM-CARRYFORWARD (0.4.6), which had protected only the cited "
            "demotion path. Transparent: carried-forward patterns stop eroding "
            "by session-domain; no data was lost (only the level marker moved)."
        ),
        "suggested_edit": (
            "No instruction-file edit is required — this is a transparent engine "
            "fix. One optional relaxation: if your wrap guidance tells the agent "
            "it MUST re-cite every Proven each wrap or lose its level, you can "
            "soften that — a Proven carried forward bare (re-stamped to today, no "
            "citation) at/below its earned level and still warm is now held, not "
            "eroded. Brand-new bald `name | Nx` claims with no earned history "
            "still demote, so inflation is unaffected."
        ),
        "files": list(CORE_FILES),
    },
]


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a version string into a comparable tuple of its leading numeric
    dotted components.

    Tolerant of pre-release/build suffixes — parsing stops at the FIRST
    non-numeric character (whole-parse, not merely per-chunk), so a dotted
    suffix cannot leak spurious trailing components:
    ``"0.4.7"`` -> ``(0, 4, 7)``, ``"0.4.7rc1"`` -> ``(0, 4, 7)``,
    ``"0.4.7+ubuntu.1"`` -> ``(0, 4, 7)``, ``"0.5"`` -> ``(0, 5)``. A version
    that starts non-numeric (``""``, ``"garbage"``, ``"v1.2"``) yields ``()``
    (sorts before every real version), so it degrades rather than raising.
    """
    parts: list[int] = []
    for chunk in version.strip().split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
        if len(digits) != len(chunk):
            # A partially-numeric chunk ("7rc1", "7+ubuntu") is a pre-release /
            # build-suffix boundary — stop the whole parse so a later dotted
            # segment ("...+ubuntu.1") cannot leak a spurious component.
            break
    return tuple(parts)


def pending_migrations(
    acknowledged_version: str | None,
    *,
    current_version: str = __version__,
    manifest: Sequence[MigrationEntry] = MIGRATION_MANIFEST,
) -> list[MigrationEntry]:
    """Return the manifest entries the adopter has not yet acknowledged AND
    that the installed version actually ships.

    An entry is pending when ``acknowledged_version < entry.version <=
    current_version``. ``acknowledged_version is None`` (never acknowledged)
    means everything up to ``current_version`` is pending. Entries newer than
    ``current_version`` are withheld — you cannot migrate to a feature you
    have not installed (guards a stale-manifest/downgrade edge).
    """
    current = _version_tuple(current_version)
    ack = _version_tuple(acknowledged_version) if acknowledged_version else None
    pending: list[MigrationEntry] = []
    for entry in manifest:
        entry_v = _version_tuple(entry["version"])
        if entry_v > current:
            continue  # feature not installed yet — nothing to migrate to
        if ack is not None and entry_v <= ack:
            continue  # already acknowledged at or above this version
        pending.append(entry)
    return pending


def marker_path_for(db_path: Path) -> Path:
    """The acknowledgement-marker path for a given episodic db path.

    A ``<db-stem>.migrate.json`` sibling of the db, mirroring the spore
    store's ``<db-stem>.spores.json`` sibling. Independent of the db itself —
    the db need not exist (a fresh adopter has no marker, so everything is
    pending until they ack).
    """
    db_path = Path(db_path).expanduser()
    return db_path.parent / f"{db_path.stem}.migrate.json"


def read_marker(marker_path: Path) -> str | None:
    """Read the acknowledged version from a marker file.

    Returns ``None`` if the marker is absent, unreadable, or malformed — a
    migration marker is advisory, and a corrupt/missing one safely degrades to
    "acknowledged nothing" (so ``migrate check`` re-proposes edits the AI can
    no-op, rather than crashing the adopter's workflow). Never raises.
    """
    try:
        raw = marker_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("acknowledged_version")
    if not isinstance(version, str) or not version.strip():
        return None
    return version


def _fsync_dir(dir_path: Path) -> None:
    """Best-effort POSIX directory fsync so the rename itself is durable. Mirrors
    the episodic store's / spore store's ``_fsync_dir``; on macOS ``os.fsync``
    is weaker than ``F_FULLFSYNC`` — a documented platform limit, not a bug
    here. No-ops where a directory fd can't be opened/synced (e.g. Windows)."""
    try:
        fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_marker(marker_path: Path, version: str) -> None:
    """Atomically write the acknowledged version to a marker file.

    Unique-tmp + file fsync + ``os.replace`` + directory fsync — the full
    atomic-write invariant the episodic / spore stores use, so a concurrent
    writer or a crash mid-write cannot leave a torn or non-durable marker. The
    raw fd from ``mkstemp`` is closed explicitly if ``os.fdopen`` fails to take
    ownership (mirrors ``spores.py``), and the tmp sidecar is never leaked.
    """
    marker_path = Path(marker_path).expanduser()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"acknowledged_version": version}, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{marker_path.stem}.",
        suffix=".migrate.json.tmp",
        dir=str(marker_path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)  # fdopen didn't take ownership — close the raw fd
            raise
        with fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, marker_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(marker_path.parent)
