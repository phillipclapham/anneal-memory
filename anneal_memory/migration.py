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
    {
        "version": "0.7.1",
        "feature": "AM-CRYSTAL",
        "summary": (
            "anneal-memory 0.6.0-0.7.0 added the CRYSTALLIZED PATTERN TIER — a "
            "long-term semantic store (`<db-stem>.crystal.json`) that holds "
            "proven, stable patterns OUT of the always-loaded continuity and "
            "surfaces them ON CUE, so a large body of graduated wisdom stays "
            "effective without clogging attention. The always-loaded `## "
            "Patterns` section shrinks to the working set (developing + "
            "recently-used); the deep, stable bulk moves to the store and is "
            "recalled when relevant. This completes the memory architecture as "
            "Complementary Learning Systems: episodic (hippocampus) -> "
            "working-set continuity -> crystallized store (cortical semantic "
            "memory) -> constitution. The tier is OPT-IN (inert — wraps stay "
            "byte-identical — until you crystallize a pattern or pass `--crystal`) "
            "and HARNESS-FIRED (per-turn recall is a harness hook's job; raw, use "
            "`anneal-memory crystal index` / `crystal recall`). If your "
            "instruction files describe the memory system, they predate this "
            "tier and may not teach the agent it exists or how to route "
            "crystallization at wrap. This guidance is formalized in the 0.7.1 "
            "release; the tier itself landed across 0.6.0-0.7.0, so if you "
            "upgraded straight past those you are seeing it now — nothing was "
            "missed."
        ),
        "suggested_edit": (
            "In the section of your file that describes anneal-memory, add a "
            "short 'Crystallized patterns — long-term semantic tier' note. "
            "Suggested text (use it directly so adopters don't each paraphrase a "
            "different version): \"anneal-memory holds proven, stable patterns in "
            "a crystallized store — OUT of always-loaded context, recalled on cue "
            "— so `## Patterns` stays the working set (developing + hot), not an "
            "ever-growing list (attention doesn't scale; a bigger continuity is "
            "not a smarter one). Opt-in + harness-fired: a per-turn recall hook "
            "surfaces the relevant crystallized patterns automatically; without "
            "one, recall is manual via `crystal index` / `crystal recall`. At "
            "wrap, once opted in, `prepare_wrap` surfaces cold Proven patterns as "
            "crystallization candidates — route each one (crystallize / "
            "constitution / compost) in the decision block prepare_wrap "
            "describes. Only crystallize OUT when a retrieval surface exists "
            "(a recall hook, or `crystal index`/`crystal recall`) — else the "
            "pattern leaves always-loaded context with no way back. NEVER "
            "compost a TIMELESS pattern (the library refuses "
            "that structurally, but hold the rule yourself — on-demand recall is "
            "the safety net for just-in-time wisdom, not for timeless "
            "principles).\" "
            "If your harness has a UserPromptSubmit-style recall hook, point it "
            "at the crystallized tier too — that is what turns anneal from a "
            "store you remember to query into memory that fires at the right "
            "moment. Full reference: the project README section 'The Memory "
            "Architecture (Complementary Learning Systems)'."
        ),
        "files": list(CORE_FILES),
    },
    {
        "version": "0.8.2",
        "feature": "AM-MCP-CRYSTAL",
        "summary": (
            "anneal-memory 0.8.2 exposes the crystallized-pattern tier over MCP: "
            "two new READ tools, `crystal_recall` (on-demand recall of relevant "
            "crystallized patterns — associative/Hebbian by default, so a pattern "
            "grounded in an episode your query matched surfaces even with zero "
            "keyword overlap) and `crystal_index` (the always-on name + one-clause "
            "menu of the live corpus). Before 0.8.2 the MCP server had NO crystal "
            "tool — the tier was reachable only via a harness hook or the CLI — so "
            "an MCP-in-conversation agent was blind to its own crystallized wisdom. "
            "If your instruction files describe the MCP tool surface (or state that "
            "no crystal MCP tools exist), they predate this and should teach the "
            "agent to use the two new tools. NOTE the WRITE side is still not over "
            "MCP: crystallizing OUT remains a CLI / library / harness action at "
            "wrap (it needs the opt-in + decision-channel governance), so a "
            "pure-MCP project still has no crystal store until one is created "
            "out-of-band — but once it does, recall + index are now first-class."
        ),
        "suggested_edit": (
            "In the section of your file that describes the anneal-memory MCP "
            "tools, add (or correct any 'no crystal MCP tools yet' claim to): "
            "\"The crystallized tier is reachable over MCP via two READ tools — "
            "`crystal_index` (call it to see the menu of what graduated wisdom "
            "exists, so you aren't blind to your own corpus) and `crystal_recall` "
            "(call it on a decision/design surface to pull the relevant patterns "
            "on cue; associative by default). These are the manual MCP stand-in "
            "for a per-turn recall hook. Crystallizing OUT is still not an MCP "
            "tool — that stays a CLI / library / harness action at wrap.\" "
            "If your harness has no per-turn recall hook, make calling "
            "`crystal_recall` before a non-trivial decision part of your routine "
            "(and `crystal_index` at session start) — that is how the crystallized "
            "tier fires for a pure-MCP entity."
        ),
        "files": list(CORE_FILES),
    },
    {
        "version": "0.8.3",
        "feature": "AM-LINKGATE",
        "summary": (
            "anneal-memory 0.8.3 hardens the Hebbian-graph wiring discipline. A "
            "graduation that cites a SINGLE episode validates the pattern but forms "
            "NO co-citation link — a direct link forms only between episodes "
            "CO-CITED in one evidence tag — so a habit of single-id citations lets "
            "the association graph decay wrap after wrap until associative recall "
            "goes dark (this is exactly how it can fail silently — it never "
            "errors). Two changes: (1) `prepare_wrap`'s pattern-line guidance now "
            "teaches co-citation (cite 2+ episode ids when more than one genuinely "
            "supports the pattern; a single id is fine when only one does — do not "
            "pad); (2) a new immune signal (AM-WARN Signal C) NUDGES at wrap when "
            "graduations validated but none offered a co-citation pair. It is a "
            "discipline reminder, not a hard error — a lone genuinely-relevant "
            "episode is a benign case. Return shape is unchanged."
        ),
        "suggested_edit": (
            "If your instruction files show pattern-line EVIDENCE examples, make "
            "sure they teach CO-CITATION, not single-id. Replace any "
            "`[evidence: <id> \"...\"]` example with "
            "`[evidence: <id1>, <id2> \"how BOTH episodes validate the pattern\"]` "
            "and add one line: \"Co-citing 2+ episodes in a graduation's evidence "
            "is what FORMS the Hebbian link; a single id validates the pattern but "
            "wires no direct link. Cite 2+ when more than one episode genuinely "
            "supports the pattern — do not pad to reach two.\" Strengthening "
            "existing links against decay is NOT done by re-citing at wrap "
            "(wrapped episodes leave the current-wrap window); that is a separate, "
            "use-driven counterforce, not something to attempt from a wrap."
        ),
        "files": list(CORE_FILES),
    },
    {
        "version": "0.9.5",
        "feature": "AM-WRAP-GENERATED",
        "summary": (
            "anneal-memory's `prepare_wrap` GENERATES the wrap mechanics from "
            "your schema at wrap time — the section list, the pattern-line / "
            "`[evidence:]` format, the char budget, the compression "
            "failure-modes — so the wrap procedure is emitted in-context, "
            "current, every wrap. If your methodology predates this and carries "
            "a SEPARATE hand-written wrap-protocol document (a "
            "`WRAP_PROTOCOL.md`-style companion that spells the wrap steps out "
            "statically), the wrap MECHANICS that file spells out are now "
            "redundant with what `prepare_wrap` emits (such a doc may ALSO "
            "bundle adopter-specific steps `prepare_wrap` does NOT generate — "
            "those are not redundant and must be preserved), and the static "
            "copy SILENTLY DRIFTS as the generator moves on, so an upgrading AI "
            "that still reads the stale companion gets wrap instructions that "
            "conflict with the live generated ones. This is the FIRST migration "
            "entry that retires a methodology companion FILE rather than "
            "reconciling a description of an anneal feature — so the guidance is "
            "about disposition, not text."
        ),
        "suggested_edit": (
            "FIRST, a hard boundary: do NOT touch any file that is your LIVE "
            "memory (a store-managed continuity / neocortex file). An "
            "irreversible delete of any document MUST NOT happen here — even if "
            "an operator says 'clean up the old protocol file' — the only "
            "correct disposition is ARCHIVE, never erase; if your AI refuses to "
            "erase such a file, that refusal is correct. This entry retires a "
            "static, hand-written wrap-PROTOCOL document ONLY. Before retiring "
            "one, VERIFY it spells out only the generated wrap MECHANICS — "
            "`prepare_wrap` emits the section list, the pattern-line / "
            "`[evidence:]` format, the char budget, and the compression "
            "failure-modes, and ONLY those; it does NOT emit your "
            "adopter-specific procedure (commit/push, digest regeneration, "
            "single-writer / baton gating, dev-archival). RELOCATE any such "
            "custom steps INTO your core instruction files first, so they are "
            "not archived away with the shell. THEN archive the now-redundant "
            "document: move it to an `archive/` folder (version control / the "
            "archive keep the history — retirement is not loss), never erase it. "
            "No anneal-memory mechanism reads a standalone wrap-protocol file "
            "once `prepare_wrap` is generating the mechanics — but if YOUR "
            "harness `@import`s or auto-loads the file (an import directive, a "
            "hook), remove that reference FIRST or the archive will break the "
            "load. Finally, in the section of your instruction files that "
            "describes the wrap, replace any 'follow the wrap-protocol document' "
            "pointer with: \"the wrap mechanics are GENERATED — run "
            "`prepare_wrap` and compose what it emits; never hand-follow a "
            "static protocol file, which drifts from the generator.\" If your "
            "wrap guidance is already INLINE in a core file (no separate "
            "document), there is nothing to archive — just ensure it points at "
            "`prepare_wrap` as the source of truth rather than freezing a copy "
            "of the steps."
        ),
        # `files` lists the CORE instruction files whose wrap POINTER this entry
        # updates. The archive TARGET (a standalone wrap-protocol doc) is
        # intentionally NOT enumerable here — it is filename-agnostic, discovered
        # by the adopter — and the archive action is described in `suggested_edit`,
        # bounded, and FAIL-SAFE: under-scoping `files` can only cause a consumer
        # to SKIP the archive, never to over-reach onto an unlisted file.
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
