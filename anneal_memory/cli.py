"""CLI interface for anneal-memory.

Operator interface for inspecting, debugging, and managing agent memory.
Thin wrapper over Store — all logic lives in the library.

Usage:
    anneal-memory status                          # Store overview
    anneal-memory episodes --since 3d --type observation
    anneal-memory get abc123                      # Single episode by ID
    anneal-memory continuity                      # Print continuity file
    anneal-memory record "Discovered X" --type observation
    anneal-memory search "pattern"
    anneal-memory associations --episode abc123
    anneal-memory associations --stats
    anneal-memory verify                          # Audit trail integrity
    anneal-memory delete abc123
    anneal-memory prune --older-than 90           # Prune old episodes
    anneal-memory prepare-wrap                    # Get compression package (agent-driven)
    anneal-memory save-continuity cont.md         # Save agent's compression (validated)
    anneal-memory init                            # Initialize new store
    anneal-memory serve                           # Start MCP server
    anneal-memory export --format json -o out.json  # Export store data
    anneal-memory import out.json                 # Import from export
    anneal-memory audit --since 3d --event record # Read audit trail
    anneal-memory diff --wraps 5                  # Wrap metric progression
    anneal-memory graph --format dot -o graph.dot # Association graph
    anneal-memory stats                           # Detailed analytics
    anneal-memory history --limit 10              # Wrap history timeline

Environment variables:
    ANNEAL_MEMORY_DB       Default database path (overridden by --db)
    ANNEAL_MEMORY_SOURCE   Default source for 'record' command (overridden by --source)
Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .associations import process_wrap_associations
from .audit import AuditTrail, _iter_lines as _iter_audit_lines
from .continuity import measure_sections, prepare_wrap_package, validate_structure
from .graduation import validate_graduations
from .store import Store
from .types import AffectiveState, AssociationStats, EpisodeType


# -- Time parsing --

_TIME_PATTERN = re.compile(r"^(\d+)([smhdw])$")

_TIME_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_duration(value: str) -> str:
    """Parse a human duration like '3d', '24h', '1w' into ISO 8601 UTC timestamp.

    Returns an ISO timestamp representing (now - duration).
    """
    match = _TIME_PATTERN.match(value.strip().lower())
    if not match:
        # Try parsing as ISO timestamp directly
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        except ValueError:
            raise ValueError(
                f"Invalid duration: {value!r}. "
                f"Use format like '3d', '24h', '1w', or an ISO 8601 timestamp."
            )

    amount = int(match.group(1))
    unit = _TIME_UNITS[match.group(2)]
    delta = timedelta(**{unit: amount})
    since = datetime.now(timezone.utc) - delta
    return since.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# -- Output formatting --

def _print_json(data: Any) -> None:
    """Print JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _format_timestamp(ts: str | None) -> str:
    """Format an ISO timestamp for human display."""
    if not ts:
        return "never"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        total_sec = delta.total_seconds()
        if total_sec < 0:
            return f"in the future ({dt.strftime('%Y-%m-%d %H:%M')} UTC)"
        if total_sec < 60:
            return f"{int(total_sec)}s ago"
        elif total_sec < 3600:
            return f"{int(total_sec / 60)}m ago"
        elif total_sec < 86400:
            return f"{int(total_sec / 3600)}h ago"
        else:
            return f"{delta.days}d ago ({dt.strftime('%Y-%m-%d %H:%M')} UTC)"
    except (ValueError, TypeError):
        return ts or "unknown"


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _default_db() -> str:
    """Get the default database path (env var or ~/.anneal-memory/memory.db)."""
    env_db = os.environ.get("ANNEAL_MEMORY_DB")
    if env_db:
        return env_db
    return str(Path("~/.anneal-memory/memory.db").expanduser())


# -- Shared --json argparse parent --

def _json_parent() -> argparse.ArgumentParser:
    """Create a parent parser with --json flag for subcommands.

    Uses argparse.SUPPRESS as default so the subparser's --json doesn't
    overwrite the top-level --json=True when both are present.
    Allows both `anneal-memory --json status` and `anneal-memory status --json`.
    """
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Output in JSON format",
    )
    return parent


# -- Store factory --

def _open_store(args: argparse.Namespace) -> Store:
    """Open a Store from CLI args."""
    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        print(
            "Run 'anneal-memory init' to create a new store, "
            "or use --db to specify the path.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Store(
        path=db_path,
        project_name=getattr(args, "project_name", "Agent"),
        audit=True,
    )


# -- Episode dict helper --

def _episode_dict(ep: Any) -> dict[str, Any]:
    """Convert an Episode to a JSON-serializable dict."""
    return {
        "id": ep.id,
        "timestamp": ep.timestamp,
        "type": ep.type.value,
        "content": ep.content,
        "source": ep.source,
        "session_id": ep.session_id,
        "metadata": ep.metadata,
    }


# -- Subcommand handlers --

def cmd_init(args: argparse.Namespace) -> None:
    """Initialize a new memory store."""
    db_path = Path(args.db).expanduser()
    if db_path.exists():
        print(f"Store already exists: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Ensure parent directory exists (first-run UX)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    project = getattr(args, "project_name", "Agent")
    store = Store(path=db_path, project_name=project, audit=True)
    store.close()

    if args.json:
        _print_json({
            "database": str(db_path),
            "continuity": str(db_path.parent / f"{db_path.stem}.continuity.md"),
            "project": project,
        })
        return

    print(f"Initialized anneal-memory store:")
    print(f"  Database: {db_path}")
    print(f"  Continuity: {db_path.parent / f'{db_path.stem}.continuity.md'}")
    print(f"  Project: {project}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show store status."""
    with _open_store(args) as store:
        status = store.status()

        if args.json:
            _print_json({
                "total_episodes": status.total_episodes,
                "episodes_since_wrap": status.episodes_since_wrap,
                "total_wraps": status.total_wraps,
                "last_wrap_at": status.last_wrap_at,
                "wrap_in_progress": status.wrap_in_progress,
                "tombstone_count": status.tombstone_count,
                "continuity_chars": status.continuity_chars,
                "episodes_by_type": status.episodes_by_type,
                "association_stats": _assoc_stats_dict(status.association_stats),
            })
            return

        print(f"anneal-memory v{__version__}")
        print(f"  Database: {store.path}")
        print(f"  Project:  {store.project_name}")
        print()
        print(f"Episodes:   {status.total_episodes} total, {status.episodes_since_wrap} since last wrap")
        if status.episodes_by_type:
            types_str = ", ".join(
                f"{t}: {c}" for t, c in sorted(status.episodes_by_type.items())
            )
            print(f"  By type:  {types_str}")
        print(f"Wraps:      {status.total_wraps} total, last {_format_timestamp(status.last_wrap_at)}")
        if status.wrap_in_progress:
            print(f"  !! Wrap in progress")
        if status.continuity_chars is not None:
            print(f"Continuity: {status.continuity_chars:,} chars")
        else:
            print(f"Continuity: not yet created")
        if status.tombstone_count > 0:
            print(f"Tombstones: {status.tombstone_count}")

        if status.association_stats and status.association_stats.total_links > 0:
            a = status.association_stats
            print()
            print(f"Associations: {a.total_links} links")
            print(f"  Avg strength:    {a.avg_strength:.3f}")
            print(f"  Max strength:    {a.max_strength:.3f}")
            print(f"  Global density:  {a.density:.4f}")
            print(f"  Local density:   {a.local_density:.4f}")
            if a.strongest_pairs:
                print(f"  Strongest pairs:")
                for p in a.strongest_pairs[:5]:
                    print(f"    {p.episode_a} <-> {p.episode_b}  strength={p.strength:.3f}  citations={p.co_citations}")


def cmd_episodes(args: argparse.Namespace) -> None:
    """List and filter episodes."""
    with _open_store(args) as store:
        since = parse_duration(args.since) if args.since else None
        until = parse_duration(args.until) if args.until else None
        ep_type = args.type if args.type else None

        result = store.recall(
            since=since,
            until=until,
            episode_type=ep_type,
            source=args.source,
            keyword=args.keyword,
            limit=args.limit,
            offset=args.offset,
        )

        if args.json:
            _print_json({
                "episodes": [_episode_dict(ep) for ep in result.episodes],
                "total_matching": result.total_matching,
            })
            return

        if not result.episodes:
            print("No episodes found.")
            return

        print(f"Episodes: {result.total_matching} total matching")
        print()
        for ep in result.episodes:
            age = _format_timestamp(ep.timestamp)
            content_preview = _truncate(ep.content.replace("\n", " "), 100)
            print(f"  [{ep.id}] {ep.type.value:<12} {age}")
            print(f"           {content_preview}")
            if ep.source != "agent":
                print(f"           source: {ep.source}")
            print()


def cmd_get(args: argparse.Namespace) -> None:
    """Show a single episode by ID."""
    with _open_store(args) as store:
        episode = store.get(args.episode_id)
        if episode is None:
            print(f"Episode {args.episode_id} not found.", file=sys.stderr)
            sys.exit(1)

        if args.json:
            _print_json(_episode_dict(episode))
            return

        print(f"Episode {episode.id}")
        print(f"  Type:      {episode.type.value}")
        print(f"  Source:    {episode.source}")
        print(f"  Time:      {_format_timestamp(episode.timestamp)}")
        print(f"  Session:   {episode.session_id or 'none'}")
        if episode.metadata:
            print(f"  Metadata:  {json.dumps(episode.metadata)}")
        print()
        print(episode.content)


def cmd_continuity(args: argparse.Namespace) -> None:
    """Print the current continuity file."""
    with _open_store(args) as store:
        text = store.load_continuity()
        if text is None:
            print("No continuity file yet. Run a wrap first.", file=sys.stderr)
            sys.exit(1)

        if args.json:
            meta = store.load_meta()
            _print_json({
                "text": text,
                "chars": len(text),
                "meta": meta,
            })
            return

        print(text)


def cmd_record(args: argparse.Namespace) -> None:
    """Record a new episode."""
    # Read from stdin if content is "-"
    if args.content == "-":
        content = sys.stdin.read().strip()
        if not content:
            print("Error: no content provided on stdin.", file=sys.stderr)
            sys.exit(1)
    else:
        content = args.content

    with _open_store(args) as store:
        metadata = None
        if args.tags:
            metadata = {"tags": [t.strip() for t in args.tags.split(",")]}

        episode = store.record(
            content=content,
            episode_type=args.type,
            source=args.source,
            metadata=metadata,
        )

        if args.json:
            _print_json({
                "id": episode.id,
                "timestamp": episode.timestamp,
                "type": episode.type.value,
                "source": episode.source,
            })
            return

        print(f"Recorded episode {episode.id} ({episode.type.value})")


def cmd_search(args: argparse.Namespace) -> None:
    """Search episodes by keyword."""
    with _open_store(args) as store:
        since = parse_duration(args.since) if args.since else None
        ep_type = args.type if args.type else None

        result = store.recall(
            keyword=args.query,
            since=since,
            episode_type=ep_type,
            source=args.source,
            limit=args.limit,
        )

        if args.json:
            _print_json({
                "episodes": [_episode_dict(ep) for ep in result.episodes],
                "total_matching": result.total_matching,
            })
            return

        if not result.episodes:
            print(f"No episodes matching '{args.query}'.")
            return

        print(f"Found {result.total_matching} episodes matching '{args.query}':")
        print()
        for ep in result.episodes:
            age = _format_timestamp(ep.timestamp)
            content = _truncate(ep.content.replace("\n", " "), 100)
            print(f"  [{ep.id}] {ep.type.value:<12} {age}")
            print(f"           {content}")
            print()


def cmd_associations(args: argparse.Namespace) -> None:
    """Query Hebbian associations."""
    with _open_store(args) as store:
        if args.stats:
            stats = store.association_stats()
            if args.json:
                _print_json(_assoc_stats_dict(stats))
                return

            print(f"Association Network Stats:")
            print(f"  Total links:     {stats.total_links}")
            print(f"  Avg strength:    {stats.avg_strength:.3f}")
            print(f"  Max strength:    {stats.max_strength:.3f}")
            print(f"  Global density:  {stats.density:.4f}")
            print(f"  Local density:   {stats.local_density:.4f}")
            if stats.strongest_pairs:
                print()
                print(f"  Strongest pairs:")
                for p in stats.strongest_pairs[:10]:
                    tag_str = f"  affect={p.affective_tag}({p.affective_intensity:.1f})" if p.affective_tag else ""
                    print(f"    {p.episode_a} <-> {p.episode_b}  strength={p.strength:.3f}  citations={p.co_citations}{tag_str}")
            return

        if not args.episode:
            print("Error: provide --episode ID or use --stats", file=sys.stderr)
            sys.exit(1)

        pairs = store.get_associations(
            episode_ids=[args.episode],
            min_strength=args.min_strength,
            limit=args.limit,
        )

        if args.json:
            _print_json([
                {
                    "episode_a": p.episode_a,
                    "episode_b": p.episode_b,
                    "strength": p.strength,
                    "co_citations": p.co_citations,
                    "first_linked": p.first_linked,
                    "last_strengthened": p.last_strengthened,
                    "affective_tag": p.affective_tag,
                    "affective_intensity": p.affective_intensity,
                }
                for p in pairs
            ])
            return

        if not pairs:
            print(f"No associations found for episode {args.episode}.")
            return

        print(f"Associations for episode {args.episode}:")
        print()
        for p in pairs:
            other = p.episode_b if p.episode_a == args.episode else p.episode_a
            tag_str = f"  affect={p.affective_tag}({p.affective_intensity:.1f})" if p.affective_tag else ""
            print(f"  <-> {other}  strength={p.strength:.3f}  citations={p.co_citations}{tag_str}")
            linked = store.get(other)
            if linked:
                print(f"      {_truncate(linked.content.replace(chr(10), ' '), 90)}")
            print()


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete an episode."""
    with _open_store(args) as store:
        episode = store.get(args.episode_id)
        if episode is None:
            print(f"Episode {args.episode_id} not found.", file=sys.stderr)
            sys.exit(1)

        if not args.force:
            print(f"Episode {episode.id} ({episode.type.value}):")
            print(f"  {_truncate(episode.content.replace(chr(10), ' '), 100)}")
            print()
            confirm = input("Delete this episode? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

        store.delete(args.episode_id)

        if args.json:
            _print_json({"deleted": args.episode_id})
        else:
            print(f"Deleted episode {args.episode_id}")


def cmd_prune(args: argparse.Namespace) -> None:
    """Prune old episodes."""
    with _open_store(args) as store:
        if args.dry_run:
            # Use the same timestamp calculation as store.prune() for accuracy.
            # store.prune uses: datetime.now(utc) - timedelta(days=N)
            cutoff = (datetime.now(timezone.utc) - timedelta(days=args.older_than)).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            result = store.recall(until=cutoff)
            count = result.total_matching
            if args.json:
                _print_json({"would_prune": count, "older_than_days": args.older_than})
            else:
                print(f"Would prune {count} episodes older than {args.older_than} days.")
            return

        pruned = store.prune(older_than_days=args.older_than)

        if args.json:
            _print_json({"pruned": pruned, "older_than_days": args.older_than})
            return

        print(f"Pruned {pruned} episodes older than {args.older_than} days.")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify audit trail integrity."""
    db_path = Path(args.db).expanduser()
    result = AuditTrail.verify(db_path)

    if args.json:
        _print_json({
            "valid": result.valid,
            "total_entries": result.total_entries,
            "files_verified": result.files_verified,
            "chain_break_at": result.chain_break_at,
            "chain_break_file": result.chain_break_file,
            "skipped_lines": result.skipped_lines,
            "error": result.error,
        })
        return

    if result.valid:
        print(f"Audit trail valid: {result.total_entries} entries across {result.files_verified} file(s)")
        if result.skipped_lines > 0:
            print(f"  ({result.skipped_lines} malformed lines skipped)")
    else:
        print(f"Audit trail INVALID: {result.error}", file=sys.stderr)
        if result.chain_break_at is not None:
            print(
                f"  Chain broke at seq {result.chain_break_at} in {result.chain_break_file}",
                file=sys.stderr,
            )
        sys.exit(1)


def cmd_prepare_wrap(args: argparse.Namespace) -> None:
    """Output the compression package for agent-driven wraps.

    This is the CLI equivalent of the MCP prepare_wrap tool.
    The agent reads this output, compresses in its own reasoning,
    then saves via save-continuity. Compression IS cognition —
    the agent's judgment during compression is where identity forms.
    """
    with _open_store(args) as store:
        episodes = store.episodes_since_wrap()

        if not episodes:
            store.wrap_cancelled()
            if args.json:
                _print_json({"status": "empty", "message": "No episodes since last wrap"})
            else:
                print("No episodes since last wrap. Nothing to compress.")
            return

        existing = store.load_continuity()
        package = prepare_wrap_package(
            episodes=episodes,
            existing_continuity=existing,
            project_name=store.project_name,
            max_chars=args.max_chars,
            staleness_days=args.staleness_days,
        )

        # Mark wrap as in progress
        store.wrap_started()

        if args.json:
            _print_json(package)
            return

        # Build readable output for the agent (mirrors MCP prepare_wrap)
        parts: list[str] = [package["instructions"], "\n---\n"]
        parts.append(f"## Episodes This Session ({package['episode_count']})")
        parts.append(package["episodes"])

        if package["continuity"]:
            parts.append("\n---\n## Current Continuity File")
            parts.append(package["continuity"])
        else:
            parts.append(
                "\n---\n(No existing continuity file — this is the first wrap.)"
            )

        if package["stale_patterns"]:
            parts.append("\n---\n## Stale Patterns (consider removing)")
            for sp in package["stale_patterns"]:
                parts.append(
                    f"- Line {sp['line']}: {sp['content']}"
                    f" ({sp['days_stale']}d stale)"
                )

        # Include Hebbian association context
        episode_ids = [ep.id for ep in episodes]
        assoc_context = store.get_association_context(episode_ids)
        if assoc_context:
            parts.append("\n---\n" + assoc_context)

        print("\n".join(parts))


def cmd_save_continuity(args: argparse.Namespace) -> None:
    """Save agent-compressed continuity with full validation.

    This is the CLI equivalent of the MCP save_continuity tool.
    Reads the agent's compressed text from a file or stdin, then runs
    the same validation pipeline: structure check, graduation citation
    validation, Hebbian association formation, decay.
    """
    # Read continuity text from file or stdin
    if args.file == "-":
        text = sys.stdin.read().strip()
    else:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: file not found: {file_path}", file=sys.stderr)
            sys.exit(1)
        text = file_path.read_text(encoding="utf-8").strip()

    if not text:
        print("Error: empty continuity text.", file=sys.stderr)
        sys.exit(1)

    with _open_store(args) as store:
        # Parse optional affective state
        affective_state = None
        if args.affect_tag:
            affective_state = AffectiveState(
                tag=args.affect_tag,
                intensity=args.affect_intensity,
            )

        # Check if prepare-wrap was called first
        skipped_prepare = not store.status().wrap_in_progress

        # Validate structure (4 required sections)
        if not validate_structure(text):
            print(
                "Error: continuity must contain all 4 sections: "
                "## State, ## Patterns, ## Decisions, ## Context",
                file=sys.stderr,
            )
            sys.exit(1)

        # Get current session's episodes for citation validation
        episodes = store.episodes_since_wrap()
        valid_ids = {ep.id[:8].lower() for ep in episodes}
        node_content_map = {ep.id[:8].lower(): ep.content for ep in episodes}

        # Check citation history
        meta = store.load_meta()
        citations_seen = meta.get("citations_seen", False)

        # Validate graduations
        from datetime import date as _date
        today = _date.today().isoformat()
        grad_result = validate_graduations(
            text=text,
            valid_ids=valid_ids,
            today=today,
            node_content_map=node_content_map,
            citations_seen=citations_seen,
        )

        # Save continuity
        path = store.save_continuity(grad_result.text)

        # Record Hebbian associations + decay
        assoc_formed, assoc_strengthened, assoc_decayed = \
            process_wrap_associations(store, grad_result, affective_state)

        # Update metadata
        if grad_result.validated > 0 or grad_result.citation_counts:
            meta["citations_seen"] = True
        meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
        store.save_meta(meta)

        # Record wrap completion
        sections = measure_sections(grad_result.text)
        patterns = len(re.findall(r"\|\s*\d+x", grad_result.text))
        wrap_result = store.wrap_completed(
            episodes_compressed=len(episodes),
            continuity_chars=len(grad_result.text),
            graduations_validated=grad_result.validated,
            graduations_demoted=grad_result.demoted + grad_result.bare_demoted,
            citation_reuse_max=grad_result.citation_reuse_max,
            patterns_extracted=patterns,
            associations_formed=assoc_formed,
            associations_strengthened=assoc_strengthened,
            associations_decayed=assoc_decayed,
        )

        if args.json:
            _print_json({
                "saved": True,
                "path": path,
                "chars": len(grad_result.text),
                "episodes_compressed": len(episodes),
                "graduations_validated": grad_result.validated,
                "graduations_demoted": grad_result.demoted + grad_result.bare_demoted,
                "associations_formed": assoc_formed,
                "associations_strengthened": assoc_strengthened,
                "associations_decayed": assoc_decayed,
                "skipped_prepare": skipped_prepare,
                "sections": {name: chars for name, chars in sorted(sections.items())},
            })
            return

        print(f"Continuity saved ({len(grad_result.text):,} chars) to {path}")
        if skipped_prepare:
            print(
                "Note: prepare-wrap was not called first — "
                "continuity may not reflect current episodes."
            )
        print(f"Episodes compressed: {len(episodes)}")

        if grad_result.validated:
            print(f"Citations validated: {grad_result.validated}")
        if grad_result.demoted:
            print(f"Citations demoted (bad evidence): {grad_result.demoted}")
        if grad_result.bare_demoted:
            print(f"Bare graduations demoted (no evidence): {grad_result.bare_demoted}")
        if grad_result.gaming_suspects:
            print(f"Citation gaming suspects: {', '.join(grad_result.gaming_suspects)}")

        if assoc_formed or assoc_strengthened:
            print(f"Associations: {assoc_formed} formed, {assoc_strengthened} strengthened")
        if assoc_decayed:
            print(f"Associations decayed: {assoc_decayed}")

        print("\nSection sizes:")
        for name, chars in sorted(sections.items()):
            print(f"  {name}: {chars} chars")


def cmd_export(args: argparse.Namespace) -> None:
    """Export store data."""
    fmt = args.format

    if fmt == "sqlite":
        # Use SQLite backup API for a consistent snapshot (safe even mid-transaction)
        db_path = Path(args.db).expanduser()
        if not db_path.exists():
            print(f"Error: database not found: {db_path}", file=sys.stderr)
            sys.exit(1)
        out = Path(args.output) if args.output else Path(f"anneal-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db")
        src_conn = sqlite3.connect(str(db_path))
        try:
            dst_conn = sqlite3.connect(str(out))
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()
        if args.json:
            _print_json({"format": "sqlite", "path": str(out), "size_bytes": out.stat().st_size})
        else:
            print(f"Exported SQLite database to {out} ({out.stat().st_size:,} bytes)", file=sys.stderr)
        return

    with _open_store(args) as store:
        # Gather all data
        result = store.recall(limit=100000)
        episodes = [_episode_dict(ep) for ep in result.episodes]
        continuity = store.load_continuity()
        meta = store.load_meta()
        assoc_stats = store.association_stats()

        # Get all associations
        all_ep_ids = [ep["id"] for ep in episodes]
        associations = []
        if all_ep_ids:
            pairs = store.get_associations(all_ep_ids, limit=100000)
            associations = [
                {
                    "episode_a": p.episode_a,
                    "episode_b": p.episode_b,
                    "strength": p.strength,
                    "co_citations": p.co_citations,
                    "first_linked": p.first_linked,
                    "last_strengthened": p.last_strengthened,
                    "affective_tag": p.affective_tag,
                    "affective_intensity": p.affective_intensity,
                }
                for p in pairs
            ]

        # Get wrap history
        wraps = _query_wraps(store)

        if fmt == "json":
            export_data = {
                "anneal_memory_export": True,
                "format_version": 1,
                "version": __version__,
                "exported_at": _now_utc_str(),
                "project_name": store.project_name,
                "episodes": episodes,
                "associations": associations,
                "wraps": wraps,
                "continuity": continuity,
                "meta": meta,
            }
            if args.output:
                out = Path(args.output)
                out.write_text(json.dumps(export_data, indent=2, default=str), encoding="utf-8")
                if args.json:
                    _print_json({"format": "json", "path": str(out), "episodes": len(episodes)})
                else:
                    print(f"Exported {len(episodes)} episodes to {out}", file=sys.stderr)
            else:
                _print_json(export_data)

        elif fmt == "markdown":
            lines = [f"# anneal-memory Export — {store.project_name}", ""]
            lines.append(f"Exported: {_now_utc_str()}")
            lines.append(f"Version: {__version__}")
            lines.append(f"Episodes: {len(episodes)}")
            lines.append(f"Associations: {len(associations)}")
            lines.append(f"Wraps: {len(wraps)}")
            lines.append("")

            if continuity:
                lines.append("---")
                lines.append("## Continuity File")
                lines.append("")
                lines.append(continuity)
                lines.append("")

            lines.append("---")
            lines.append("## Episodes")
            lines.append("")
            for ep in episodes:
                lines.append(f"### [{ep['id']}] {ep['type']} — {ep['timestamp']}")
                if ep.get("source") and ep["source"] != "agent":
                    lines.append(f"Source: {ep['source']}")
                lines.append("")
                lines.append(ep["content"])
                lines.append("")

            if associations:
                lines.append("---")
                lines.append("## Associations")
                lines.append("")
                for a in associations:
                    tag_str = f" [{a['affective_tag']}, {a['affective_intensity']:.1f}]" if a.get("affective_tag") else ""
                    lines.append(f"- {a['episode_a']} ↔ {a['episode_b']}: strength={a['strength']:.3f}, co-cited={a['co_citations']}{tag_str}")
                lines.append("")

            text = "\n".join(lines)
            if args.output:
                out = Path(args.output)
                out.write_text(text, encoding="utf-8")
                if args.json:
                    _print_json({"format": "markdown", "path": str(out), "episodes": len(episodes)})
                else:
                    print(f"Exported {len(episodes)} episodes to {out}", file=sys.stderr)
            else:
                print(text)


def cmd_import(args: argparse.Namespace) -> None:
    """Import episodes from a JSON export file.

    Imports episodes only — not associations, continuity, or wrap history.
    This is intentional: associations form through the agent's own consolidation
    acts during wraps, not through external injection. Importing someone else's
    cognitive topology would undermine the system's thesis that identity emerges
    from the agent's own compression and citation patterns. Import the raw
    material, let the cognitive system rebuild its own conclusions.
    """
    import_path = Path(args.path)
    if not import_path.exists():
        print(f"Error: file not found: {import_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = json.loads(import_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    if not data.get("anneal_memory_export"):
        print("Error: file is not an anneal-memory export (missing 'anneal_memory_export' key).", file=sys.stderr)
        sys.exit(1)

    # Check format version compatibility
    _SUPPORTED_FORMAT_VERSION = 1
    file_version = data.get("format_version", 1)
    if file_version > _SUPPORTED_FORMAT_VERSION:
        print(
            f"Warning: export format version {file_version} is newer than supported version "
            f"{_SUPPORTED_FORMAT_VERSION}. Some data may be lost or misinterpreted.",
            file=sys.stderr,
        )

    episodes = data.get("episodes", [])
    if not episodes:
        if args.json:
            _print_json({"imported": 0, "skipped": 0, "errors": 0})
        else:
            print("No episodes to import.")
        return

    with _open_store(args) as store:
        imported = 0
        skipped = 0
        errors = 0

        for ep_data in episodes:
            try:
                # Check if episode already exists
                existing = store.get(ep_data["id"])
                if existing is not None:
                    skipped += 1
                    continue

                store.record(
                    content=ep_data["content"],
                    episode_type=ep_data["type"],
                    source=ep_data.get("source", "import"),
                    metadata=ep_data.get("metadata"),
                    timestamp=ep_data.get("timestamp"),
                )
                imported += 1
            except Exception as e:
                errors += 1
                if not args.json:
                    print(f"  Error importing episode {ep_data.get('id', '?')}: {e}", file=sys.stderr)

        if args.json:
            _print_json({"imported": imported, "skipped": skipped, "errors": errors})
        else:
            print(f"Import complete: {imported} imported, {skipped} skipped (already exist), {errors} errors")


def cmd_audit(args: argparse.Namespace) -> None:
    """Read and filter audit trail entries."""
    db_path = Path(args.db).expanduser()
    stem = db_path.stem
    audit_dir = db_path.parent
    active_path = audit_dir / f"{stem}.audit.jsonl"
    manifest_path = audit_dir / f"{stem}.audit.manifest.json"

    # Collect all audit files in chronological order
    files_to_read: list[Path] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for f in manifest.get("files", []):
                fpath = audit_dir / f["filename"]
                if fpath.exists():
                    files_to_read.append(fpath)
        except (json.JSONDecodeError, KeyError):
            pass
    if active_path.exists():
        files_to_read.append(active_path)

    if not files_to_read:
        if args.json:
            _print_json({"entries": [], "total": 0})
        else:
            print("No audit trail files found.")
        return

    # Parse since filter
    since_ts = parse_duration(args.since) if args.since else None

    # Read and filter entries
    entries: list[dict] = []
    for fpath in files_to_read:
        for line in _iter_audit_lines(fpath):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Apply filters
            if args.event and entry.get("event") != args.event:
                continue
            if since_ts and entry.get("ts", "") < since_ts:
                continue

            entries.append(entry)

    # Apply limit (from the end — most recent)
    total = len(entries)
    if args.limit and len(entries) > args.limit:
        entries = entries[-args.limit:]

    if args.json:
        _print_json({"entries": entries, "total": total})
        return

    if not entries:
        print("No matching audit entries.")
        return

    print(f"Audit trail: {total} entries" + (f" (showing last {len(entries)})" if len(entries) < total else ""))
    print()
    for entry in entries:
        ts = _format_timestamp(entry.get("ts"))
        event = entry.get("event", "?")
        actor = entry.get("actor", "")
        seq = entry.get("seq", "")
        data = entry.get("data", {})

        # Format data summary based on event type
        summary = ""
        if event == "record":
            summary = f"episode={data.get('episode_id', '?')} type={data.get('type', '?')}"
        elif event == "delete":
            summary = f"episode={data.get('episode_id', '?')}"
        elif event == "wrap_completed":
            summary = f"episodes={data.get('episodes_compressed', '?')} chars={data.get('continuity_chars', '?')}"
        elif event == "continuity_saved":
            summary = f"chars={data.get('chars', '?')}"
        elif event == "associations_updated":
            summary = f"formed={data.get('formed', 0)} strengthened={data.get('strengthened', 0)}"
        elif event == "associations_decayed":
            summary = f"decayed={data.get('decayed', 0)}"
        elif event == "prune":
            summary = f"count={data.get('count', '?')}"

        actor_str = f" [{actor}]" if actor and actor != "agent" else ""
        print(f"  [{seq:>4}] {event:<24} {ts}{actor_str}")
        if summary:
            print(f"         {summary}")


def cmd_diff(args: argparse.Namespace) -> None:
    """Show wrap-over-wrap metric changes."""
    with _open_store(args) as store:
        wraps = _query_wraps(store)

        if not wraps:
            if args.json:
                _print_json({"wraps": [], "message": "No wraps yet"})
            else:
                print("No wraps to compare.")
            return

        # Limit to last N wraps
        n = args.wraps
        if len(wraps) > n:
            wraps = wraps[-n:]

        if args.json:
            # Include deltas between consecutive wraps
            diffs = []
            for i, w in enumerate(wraps):
                entry: dict[str, Any] = {**w}
                if i > 0:
                    prev = wraps[i - 1]
                    entry["delta"] = {
                        "episodes_compressed": (w.get("episodes_compressed") or 0) - (prev.get("episodes_compressed") or 0),
                        "continuity_chars": (w.get("continuity_chars") or 0) - (prev.get("continuity_chars") or 0),
                        "graduations_validated": (w.get("graduations_validated") or 0) - (prev.get("graduations_validated") or 0),
                        "graduations_demoted": (w.get("graduations_demoted") or 0) - (prev.get("graduations_demoted") or 0),
                    }
                diffs.append(entry)
            _print_json({"wraps": diffs})
            return

        print(f"Wrap progression (last {len(wraps)}):")
        print()
        # Header
        print(f"  {'Wrap':>4}  {'When':<20}  {'Episodes':>8}  {'Chars':>7}  {'Grad':>4}  {'Demoted':>7}  {'Formed':>6}  {'Decayed':>7}")
        print(f"  {'----':>4}  {'----':<20}  {'--------':>8}  {'-------':>7}  {'----':>4}  {'-------':>7}  {'------':>6}  {'-------':>7}")

        for i, w in enumerate(wraps):
            wrap_id = w.get("id", "?")
            when = _format_timestamp(w.get("wrapped_at"))
            eps = w.get("episodes_compressed") or 0
            chars = w.get("continuity_chars") or 0
            grad = w.get("graduations_validated") or 0
            demoted = w.get("graduations_demoted") or 0
            formed = w.get("associations_formed") or 0
            decayed = w.get("associations_decayed") or 0

            # Show delta for continuity chars
            chars_delta = ""
            if i > 0:
                prev_chars = wraps[i - 1].get("continuity_chars") or 0
                diff = chars - prev_chars
                if diff > 0:
                    chars_delta = f" (+{diff})"
                elif diff < 0:
                    chars_delta = f" ({diff})"

            print(f"  {wrap_id:>4}  {when:<20}  {eps:>8}  {chars:>7}{chars_delta}  {grad:>4}  {demoted:>7}  {formed:>6}  {decayed:>7}")


def cmd_graph(args: argparse.Namespace) -> None:
    """Export association graph."""
    with _open_store(args) as store:
        # Get all episodes to build node list
        result = store.recall(limit=100000)
        all_ids = [ep.id for ep in result.episodes]

        if not all_ids:
            if args.json:
                _print_json({"nodes": [], "edges": []})
            else:
                print("No episodes in store.")
            return

        pairs = store.get_associations(all_ids, min_strength=args.min_strength, limit=100000)

        if not pairs:
            if args.json:
                _print_json({"nodes": [], "edges": [], "message": "No associations above threshold"})
            else:
                print(f"No associations above strength {args.min_strength}.")
            return

        # Collect connected nodes
        connected_ids: set[str] = set()
        for p in pairs:
            connected_ids.add(p.episode_a)
            connected_ids.add(p.episode_b)

        # Build node info
        nodes: dict[str, dict] = {}
        for ep in result.episodes:
            if ep.id in connected_ids:
                nodes[ep.id] = {
                    "id": ep.id,
                    "type": ep.type.value,
                    "label": _truncate(ep.content.replace("\n", " "), 40),
                }

        fmt = args.format

        if fmt == "json":
            graph_data = {
                "nodes": list(nodes.values()),
                "edges": [
                    {
                        "source": p.episode_a,
                        "target": p.episode_b,
                        "strength": p.strength,
                        "co_citations": p.co_citations,
                        "affective_tag": p.affective_tag,
                        "affective_intensity": p.affective_intensity,
                    }
                    for p in pairs
                ],
            }
            if args.output:
                out = Path(args.output)
                out.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")
                if args.json:
                    _print_json({"format": "json", "path": str(out), "nodes": len(nodes), "edges": len(pairs)})
                else:
                    print(f"Graph exported to {out} ({len(nodes)} nodes, {len(pairs)} edges)", file=sys.stderr)
            else:
                _print_json(graph_data)

        elif fmt == "dot":
            # Type -> color mapping
            type_colors = {
                "observation": "#4A90D9",
                "decision": "#D94A4A",
                "tension": "#D9A04A",
                "question": "#9B59B6",
                "outcome": "#27AE60",
                "context": "#95A5A6",
            }

            lines = ["graph associations {"]
            lines.append('  graph [rankdir=LR, overlap=false, splines=true];')
            lines.append('  node [shape=box, style="rounded,filled", fontsize=10];')
            lines.append("")

            for nid, ninfo in nodes.items():
                color = type_colors.get(ninfo["type"], "#CCCCCC")
                label = ninfo["label"].replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'  "{nid}" [label="{nid}\\n{label}", fillcolor="{color}", fontcolor="white"];')

            lines.append("")
            for p in pairs:
                # Width proportional to strength
                width = max(0.5, min(4.0, p.strength))
                label = f"{p.strength:.1f}"
                if p.affective_tag:
                    label += f"\\n{p.affective_tag}"
                lines.append(f'  "{p.episode_a}" -- "{p.episode_b}" [penwidth={width:.1f}, label="{label}"];')

            lines.append("}")
            text = "\n".join(lines)

            if args.output:
                out = Path(args.output)
                out.write_text(text, encoding="utf-8")
                if args.json:
                    _print_json({"format": "dot", "path": str(out), "nodes": len(nodes), "edges": len(pairs)})
                else:
                    print(f"DOT graph exported to {out} ({len(nodes)} nodes, {len(pairs)} edges)", file=sys.stderr)
            else:
                print(text)


def cmd_stats(args: argparse.Namespace) -> None:
    """Show detailed store statistics."""
    with _open_store(args) as store:
        status = store.status()
        wraps = _query_wraps(store)

        # Episode age distribution
        all_eps = store.recall(limit=100000)
        now = datetime.now(timezone.utc)
        age_buckets = {"<1h": 0, "1-24h": 0, "1-7d": 0, "7-30d": 0, ">30d": 0}
        for ep in all_eps.episodes:
            try:
                dt = datetime.fromisoformat(ep.timestamp.replace("Z", "+00:00"))
                age_secs = (now - dt).total_seconds()
                if age_secs < 3600:
                    age_buckets["<1h"] += 1
                elif age_secs < 86400:
                    age_buckets["1-24h"] += 1
                elif age_secs < 604800:
                    age_buckets["1-7d"] += 1
                elif age_secs < 2592000:
                    age_buckets["7-30d"] += 1
                else:
                    age_buckets[">30d"] += 1
            except (ValueError, TypeError):
                pass

        # Source distribution
        source_counts: dict[str, int] = {}
        for ep in all_eps.episodes:
            source_counts[ep.source] = source_counts.get(ep.source, 0) + 1

        # Wrap metrics
        wrap_stats: dict[str, Any] = {}
        if wraps:
            eps_per_wrap = [w.get("episodes_compressed") or 0 for w in wraps]
            wrap_stats = {
                "total_wraps": len(wraps),
                "avg_episodes_per_wrap": sum(eps_per_wrap) / len(eps_per_wrap) if eps_per_wrap else 0,
                "total_graduations": sum(w.get("graduations_validated") or 0 for w in wraps),
                "total_demotions": sum(w.get("graduations_demoted") or 0 for w in wraps),
                "total_associations_formed": sum(w.get("associations_formed") or 0 for w in wraps),
                "total_associations_decayed": sum(w.get("associations_decayed") or 0 for w in wraps),
            }
            # First and last wrap timestamps
            wrap_stats["first_wrap"] = wraps[0].get("wrapped_at")
            wrap_stats["last_wrap"] = wraps[-1].get("wrapped_at")

        if args.json:
            _print_json({
                "episodes": {
                    "total": status.total_episodes,
                    "since_wrap": status.episodes_since_wrap,
                    "by_type": status.episodes_by_type,
                    "by_age": age_buckets,
                    "by_source": source_counts,
                    "tombstones": status.tombstone_count,
                },
                "continuity": {
                    "chars": status.continuity_chars,
                },
                "wraps": wrap_stats,
                "associations": _assoc_stats_dict(status.association_stats),
            })
            return

        print(f"anneal-memory v{__version__} — Detailed Statistics")
        print(f"  Database: {store.path}")
        print(f"  Project:  {store.project_name}")
        print()

        # Episodes
        print("Episodes:")
        print(f"  Total:        {status.total_episodes}")
        print(f"  Since wrap:   {status.episodes_since_wrap}")
        if status.tombstone_count:
            print(f"  Tombstones:   {status.tombstone_count}")
        print()
        if status.episodes_by_type:
            print("  By type:")
            for t, c in sorted(status.episodes_by_type.items()):
                bar = "█" * min(c, 40)
                print(f"    {t:<12} {c:>4}  {bar}")
            print()
        print("  By age:")
        for bucket, count in age_buckets.items():
            if count > 0:
                bar = "█" * min(count, 40)
                print(f"    {bucket:<8} {count:>4}  {bar}")
        print()
        if source_counts:
            print("  By source:")
            for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
                print(f"    {src:<20} {count:>4}")
            print()

        # Continuity
        if status.continuity_chars is not None:
            print(f"Continuity: {status.continuity_chars:,} chars")
        else:
            print("Continuity: not yet created")
        print()

        # Wraps
        if wrap_stats:
            print("Wraps:")
            print(f"  Total:          {wrap_stats['total_wraps']}")
            print(f"  Avg episodes:   {wrap_stats['avg_episodes_per_wrap']:.1f} per wrap")
            print(f"  Graduations:    {wrap_stats['total_graduations']} validated, {wrap_stats['total_demotions']} demoted")
            print(f"  Associations:   {wrap_stats['total_associations_formed']} formed, {wrap_stats['total_associations_decayed']} decayed")
            print(f"  First wrap:     {_format_timestamp(wrap_stats['first_wrap'])}")
            print(f"  Last wrap:      {_format_timestamp(wrap_stats['last_wrap'])}")
        else:
            print("Wraps: none")
        print()

        # Associations
        a = status.association_stats
        if a and a.total_links > 0:
            print(f"Associations:")
            print(f"  Total links:    {a.total_links}")
            print(f"  Avg strength:   {a.avg_strength:.3f}")
            print(f"  Max strength:   {a.max_strength:.3f}")
            print(f"  Global density: {a.density:.4f}")
            print(f"  Local density:  {a.local_density:.4f}")
        else:
            print("Associations: none")


def cmd_history(args: argparse.Namespace) -> None:
    """Show wrap history."""
    with _open_store(args) as store:
        wraps = _query_wraps(store)

        if not wraps:
            if args.json:
                _print_json({"wraps": []})
            else:
                print("No wrap history.")
            return

        # Capture total before limiting
        total_wraps = len(wraps)

        # Apply limit
        if args.limit and len(wraps) > args.limit:
            wraps = wraps[-args.limit:]

        if args.json:
            _print_json({"wraps": wraps, "total": total_wraps})
            return

        showing = f" (showing last {len(wraps)})" if len(wraps) < total_wraps else ""
        print(f"Wrap history ({total_wraps} wraps){showing}:")
        print()
        for w in wraps:
            wrap_id = w.get("id", "?")
            when = _format_timestamp(w.get("wrapped_at"))
            eps = w.get("episodes_compressed") or 0
            chars = w.get("continuity_chars") or 0
            grad = w.get("graduations_validated") or 0
            demoted = w.get("graduations_demoted") or 0
            formed = w.get("associations_formed") or 0
            strengthened = w.get("associations_strengthened") or 0
            decayed = w.get("associations_decayed") or 0

            print(f"  Wrap {wrap_id} — {when}")
            print(f"    Episodes compressed: {eps}")
            print(f"    Continuity size:     {chars:,} chars")
            if grad or demoted:
                print(f"    Graduations:         {grad} validated, {demoted} demoted")
            if formed or strengthened or decayed:
                print(f"    Associations:        {formed} formed, {strengthened} strengthened, {decayed} decayed")
            print()


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server.

    Delegates to server.start_server() with explicit parameters —
    no sys.argv reconstruction needed.
    """
    from .server import start_server

    db_path = str(Path(args.db).expanduser())
    audit_ret = getattr(args, "audit_retention_days", 0)

    # Handle one-shot server modes
    if getattr(args, "generate_integrity", False):
        from .integrity import generate_integrity_file
        out = Path(__file__).parent / "tool-integrity.json"
        generate_integrity_file(out)
        print(f"Generated {out}", file=sys.stderr)
        return

    if getattr(args, "verify_audit", False):
        cmd_verify(args)
        return

    start_server(
        db_path=db_path,
        project_name=getattr(args, "project_name", "Agent"),
        skip_integrity=getattr(args, "skip_integrity", False),
        no_audit=getattr(args, "no_audit", False),
        audit_retention_days=audit_ret if audit_ret > 0 else None,
    )


# -- Helpers --

def _now_utc_str() -> str:
    """Current time as ISO 8601 UTC string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _query_wraps(store: Store) -> list[dict[str, Any]]:
    """Query all wraps from the store's database.

    Direct SQL because wraps are a CLI-level concern (history, diff, stats).
    The Store API doesn't expose wrap history — it's a monitoring concern.
    """
    try:
        rows = store._conn.execute(
            """SELECT id, wrapped_at, episodes_compressed,
                      graduations_validated, graduations_demoted,
                      citation_reuse_max, continuity_chars, patterns_extracted,
                      associations_formed, associations_strengthened, associations_decayed
               FROM wraps ORDER BY id ASC"""
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": row["id"],
            "wrapped_at": row["wrapped_at"],
            "episodes_compressed": row["episodes_compressed"],
            "graduations_validated": row["graduations_validated"],
            "graduations_demoted": row["graduations_demoted"],
            "citation_reuse_max": row["citation_reuse_max"],
            "continuity_chars": row["continuity_chars"],
            "patterns_extracted": row["patterns_extracted"],
            "associations_formed": row["associations_formed"],
            "associations_strengthened": row["associations_strengthened"],
            "associations_decayed": row["associations_decayed"],
        }
        for row in rows
    ]


def _assoc_stats_dict(stats: AssociationStats | None) -> dict[str, Any] | None:
    """Convert AssociationStats to a JSON-serializable dict."""
    if stats is None:
        return None
    return {
        "total_links": stats.total_links,
        "avg_strength": stats.avg_strength,
        "max_strength": stats.max_strength,
        "density": stats.density,
        "local_density": stats.local_density,
        "strongest_pairs": [
            {
                "episode_a": p.episode_a,
                "episode_b": p.episode_b,
                "strength": p.strength,
                "co_citations": p.co_citations,
                "affective_tag": p.affective_tag,
                "affective_intensity": p.affective_intensity,
            }
            for p in (stats.strongest_pairs or [])
        ],
    }


# -- Argument parser --

_EPISODE_TYPE_HELP = (
    "Episode type: observation (pattern/insight), decision (committed choice), "
    "tension (conflict/tradeoff), question (open question), "
    "outcome (result of action), context (environmental info)"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    json_parent = _json_parent()

    parser = argparse.ArgumentParser(
        prog="anneal-memory",
        description="Living memory for AI agents. Episodes compress into identity.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"anneal-memory {__version__}",
    )

    default_db = _default_db()

    # Shared arguments
    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Path to the SQLite database (default: {default_db}). "
             f"Set ANNEAL_MEMORY_DB env var to change the default.",
    )
    parser.add_argument(
        "--project-name",
        default="Agent",
        dest="project_name",
        help="Project name for continuity file header (default: Agent)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )

    subparsers = parser.add_subparsers(dest="command")

    # -- init --
    sub = subparsers.add_parser("init", help="Initialize a new memory store", parents=[json_parent])
    sub.set_defaults(func=cmd_init)

    # -- status --
    sub = subparsers.add_parser("status", help="Show store status", parents=[json_parent])
    sub.set_defaults(func=cmd_status)

    # -- episodes --
    sub = subparsers.add_parser("episodes", help="List and filter episodes", parents=[json_parent])
    sub.add_argument("--since", help="Show episodes after duration (e.g. 3d, 24h, 1w)")
    sub.add_argument("--until", help="Show episodes before duration (e.g. 1d = yesterday+)")
    sub.add_argument("--type", choices=[t.value for t in EpisodeType], help=_EPISODE_TYPE_HELP)
    sub.add_argument("--source", help="Filter by source")
    sub.add_argument("--keyword", help="Search content by keyword")
    sub.add_argument("--limit", type=int, default=50, help="Max episodes to return (default: 50)")
    sub.add_argument("--offset", type=int, default=0, help="Skip first N episodes")
    sub.set_defaults(func=cmd_episodes)

    # -- get --
    sub = subparsers.add_parser("get", help="Show a single episode by ID", parents=[json_parent])
    sub.add_argument("episode_id", help="Episode ID (8-char hex)")
    sub.set_defaults(func=cmd_get)

    # -- continuity --
    sub = subparsers.add_parser("continuity", help="Print current continuity file", parents=[json_parent])
    sub.set_defaults(func=cmd_continuity)

    # -- record --
    sub = subparsers.add_parser("record", help="Record a new episode", parents=[json_parent])
    sub.add_argument("content", help="Episode content (use '-' to read from stdin)")
    sub.add_argument(
        "--type",
        choices=[t.value for t in EpisodeType],
        default="observation",
        help=f"Episode type (default: observation). {_EPISODE_TYPE_HELP}",
    )
    sub.add_argument(
        "--source",
        default=os.environ.get("ANNEAL_MEMORY_SOURCE", "cli"),
        help="Source attribution (default: cli, or ANNEAL_MEMORY_SOURCE env var)",
    )
    sub.add_argument("--tags", help="Comma-separated tags")
    sub.set_defaults(func=cmd_record)

    # -- search --
    sub = subparsers.add_parser("search", help="Search episodes by keyword", parents=[json_parent])
    sub.add_argument("query", help="Search keyword")
    sub.add_argument("--since", help="Filter results to after duration (e.g. 3d)")
    sub.add_argument("--type", choices=[t.value for t in EpisodeType], help="Filter by episode type")
    sub.add_argument("--source", help="Filter by source")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_search)

    # -- associations --
    sub = subparsers.add_parser("associations", help="Query Hebbian associations", parents=[json_parent])
    sub.add_argument("--episode", help="Episode ID to query associations for")
    sub.add_argument("--stats", action="store_true", help="Show network statistics")
    sub.add_argument("--min-strength", type=float, default=0.0, help="Minimum strength filter")
    sub.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    sub.set_defaults(func=cmd_associations)

    # -- delete --
    sub = subparsers.add_parser("delete", help="Delete an episode", parents=[json_parent])
    sub.add_argument("episode_id", help="Episode ID to delete")
    sub.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    sub.set_defaults(func=cmd_delete)

    # -- prune --
    sub = subparsers.add_parser("prune", help="Prune old episodes", parents=[json_parent])
    sub.add_argument("--older-than", type=int, required=True, help="Prune episodes older than N days")
    sub.add_argument("--dry-run", action="store_true", help="Show what would be pruned without pruning")
    sub.set_defaults(func=cmd_prune)

    # -- verify --
    sub = subparsers.add_parser("verify", help="Verify audit trail hash chain integrity", parents=[json_parent])
    sub.set_defaults(func=cmd_verify)

    # -- prepare-wrap --
    sub = subparsers.add_parser(
        "prepare-wrap",
        help="Output compression package for agent-driven wraps",
        parents=[json_parent],
    )
    sub.add_argument("--max-chars", type=int, default=20000, help="Max continuity size (default: 20000)")
    sub.add_argument("--staleness-days", type=int, default=7, help="Days before flagging stale patterns (default: 7)")
    sub.set_defaults(func=cmd_prepare_wrap)

    # -- save-continuity --
    sub = subparsers.add_parser(
        "save-continuity",
        help="Save agent-compressed continuity with validation",
        parents=[json_parent],
    )
    sub.add_argument("file", help="Path to continuity markdown file (use '-' for stdin)")
    sub.add_argument("--affect-tag", help="Affective state tag (e.g. 'engaged', 'uncertain')")
    sub.add_argument("--affect-intensity", type=float, default=0.5, help="Affective intensity 0.0-1.0 (default: 0.5)")
    sub.set_defaults(func=cmd_save_continuity)

    # -- serve --
    sub = subparsers.add_parser("serve", help="Start the MCP server", parents=[json_parent])
    sub.add_argument("--skip-integrity", action="store_true", help="Skip integrity verification")
    sub.add_argument("--no-audit", action="store_true", help="Disable audit trail")
    sub.add_argument("--generate-integrity", action="store_true", help="Generate tool-integrity.json and exit")
    sub.add_argument("--verify-audit", action="store_true", help="Verify audit trail and exit")
    sub.add_argument("--audit-retention-days", type=int, default=0, help="Auto-cleanup rotated audit files")
    sub.set_defaults(func=cmd_serve)

    # -- export --
    sub = subparsers.add_parser("export", help="Export store data", parents=[json_parent])
    sub.add_argument(
        "--format", "-f",
        choices=["json", "markdown", "sqlite"],
        default="json",
        help="Export format (default: json)",
    )
    sub.add_argument("--output", "-o", help="Output file path (default: stdout for json/markdown, auto-named for sqlite)")
    sub.set_defaults(func=cmd_export)

    # -- import --
    sub = subparsers.add_parser("import", help="Import episodes from JSON export", parents=[json_parent])
    sub.add_argument("path", help="Path to JSON export file")
    sub.set_defaults(func=cmd_import)

    # -- audit --
    sub = subparsers.add_parser("audit", help="Read and filter audit trail entries", parents=[json_parent])
    sub.add_argument("--since", help="Show entries after duration (e.g. 3d, 24h)")
    sub.add_argument("--event", help="Filter by event type (record, delete, wrap_completed, etc.)")
    sub.add_argument("--limit", type=int, default=50, help="Max entries to show (default: 50, from most recent)")
    sub.set_defaults(func=cmd_audit)

    # -- diff --
    sub = subparsers.add_parser("diff", help="Show wrap-over-wrap metric changes", parents=[json_parent])
    sub.add_argument("--wraps", type=int, default=5, help="Number of recent wraps to compare (default: 5)")
    sub.set_defaults(func=cmd_diff)

    # -- graph --
    sub = subparsers.add_parser("graph", help="Export association graph", parents=[json_parent])
    sub.add_argument(
        "--format", "-f",
        choices=["dot", "json"],
        default="json",
        help="Graph format (default: json)",
    )
    sub.add_argument("--output", "-o", help="Output file path (default: stdout)")
    sub.add_argument("--min-strength", type=float, default=0.0, help="Minimum edge strength (default: 0.0)")
    sub.set_defaults(func=cmd_graph)

    # -- stats --
    sub = subparsers.add_parser("stats", help="Detailed store statistics", parents=[json_parent])
    sub.set_defaults(func=cmd_stats)

    # -- history --
    sub = subparsers.add_parser("history", help="Show wrap history", parents=[json_parent])
    sub.add_argument("--limit", type=int, default=20, help="Max wraps to show (default: 20)")
    sub.set_defaults(func=cmd_history)

    return parser


def main() -> None:
    """CLI entry point for anneal-memory.

    Backward compatibility: when no subcommand is given, delegates entirely
    to server.main() which parses sys.argv with its own parser. This means
    existing MCP host configs (``anneal-memory --db /path --no-audit``) work
    unchanged — the server's parser handles all server-specific flags.
    """
    # Detect whether a CLI subcommand is present.
    # Use parse_known_args so server-specific flags (--no-audit, --skip-integrity)
    # don't cause errors when no subcommand is given.
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    if args.command is None:
        # No subcommand — full backward compat: delegate to server.main().
        # server.main() has its own complete argparse and handles all
        # server-specific flags (--no-audit, --skip-integrity, etc.).
        from .server import main as server_main
        server_main()
        return

    # CLI subcommand present — reject any unrecognized arguments
    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    # Dispatch to subcommand handler
    args.func(args)


if __name__ == "__main__":
    main()
