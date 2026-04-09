"""CLI interface for anneal-memory.

Operator interface for inspecting, debugging, and managing agent memory.
Thin wrapper over Store + Engine — all logic lives in the library.

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
    anneal-memory wrap                            # Run compression (needs Engine)
    anneal-memory init                            # Initialize new store
    anneal-memory serve                           # Start MCP server

Environment variables:
    ANNEAL_MEMORY_DB       Default database path (overridden by --db)
    ANNEAL_MEMORY_SOURCE   Default source for 'record' command (overridden by --source)
    ANNEAL_MEMORY_MODEL    Default model for 'wrap' command (overridden by --model)
    ANTHROPIC_API_KEY      API key for 'wrap' command

Zero dependencies beyond Python stdlib (except 'wrap' which needs [engine]).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audit import AuditTrail
from .store import Store
from .types import AssociationStats, EpisodeType


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


def cmd_wrap(args: argparse.Namespace) -> None:
    """Run a compression wrap."""
    try:
        from .engine import Engine
    except ImportError:
        print(
            "Error: Engine requires the [engine] extra.\n"
            "Install with: pip install anneal-memory[engine]",
            file=sys.stderr,
        )
        sys.exit(1)

    with _open_store(args) as store:
        engine = Engine(
            store,
            api_key=args.api_key,
            model=args.model,
            max_chars=args.max_chars,
            characterize_affect=args.affect,
        )

        result = engine.wrap()

        if args.json:
            _print_json({
                "saved": result.saved,
                "chars": result.chars,
                "episodes_compressed": result.episodes_compressed,
                "graduations_validated": result.graduations_validated,
                "graduations_demoted": result.graduations_demoted,
                "patterns_extracted": result.patterns_extracted,
                "associations_formed": result.associations_formed,
                "associations_strengthened": result.associations_strengthened,
                "associations_decayed": result.associations_decayed,
                "pruned_count": result.pruned_count,
            })
            return

        if not result.saved:
            print("No episodes to compress (or validation failed).")
            return

        print(f"Wrap complete:")
        print(f"  Episodes compressed: {result.episodes_compressed}")
        print(f"  Continuity size:     {result.chars:,} chars")
        print(f"  Patterns extracted:  {result.patterns_extracted}")
        if result.graduations_validated or result.graduations_demoted:
            print(f"  Graduations:         {result.graduations_validated} validated, {result.graduations_demoted} demoted")
        if result.associations_formed or result.associations_strengthened or result.associations_decayed:
            print(f"  Associations:        {result.associations_formed} formed, {result.associations_strengthened} strengthened, {result.associations_decayed} decayed")
        if result.pruned_count:
            print(f"  Pruned:              {result.pruned_count} old episodes")


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

    # -- wrap --
    sub = subparsers.add_parser("wrap", help="Run compression wrap (requires [engine] extra)", parents=[json_parent])
    sub.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    sub.add_argument(
        "--model",
        default=os.environ.get("ANNEAL_MEMORY_MODEL", "claude-sonnet-4-6"),
        help="Model for compression (default: claude-sonnet-4-6, or ANNEAL_MEMORY_MODEL env var)",
    )
    sub.add_argument("--max-chars", type=int, default=20000, help="Max continuity size (default: 20000)")
    sub.add_argument("--affect", action="store_true", help="Enable affective state characterization")
    sub.set_defaults(func=cmd_wrap)

    # -- serve --
    sub = subparsers.add_parser("serve", help="Start the MCP server", parents=[json_parent])
    sub.add_argument("--skip-integrity", action="store_true", help="Skip integrity verification")
    sub.add_argument("--no-audit", action="store_true", help="Disable audit trail")
    sub.add_argument("--generate-integrity", action="store_true", help="Generate tool-integrity.json and exit")
    sub.add_argument("--verify-audit", action="store_true", help="Verify audit trail and exit")
    sub.add_argument("--audit-retention-days", type=int, default=0, help="Auto-cleanup rotated audit files")
    sub.set_defaults(func=cmd_serve)

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
