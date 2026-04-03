"""MCP server for anneal-memory.

Implements the Model Context Protocol over stdio transport (JSON-RPC 2.0,
newline-delimited). 5 tools + 1 resource. Zero dependencies beyond Python
stdlib.

Usage:
    anneal-memory --db /path/to/memory.db [--project-name "My Agent"]
    anneal-memory --generate-integrity  # Generate tool-integrity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from . import __version__
from .continuity import measure_sections, prepare_wrap_package, validate_structure
from .graduation import validate_graduations
from .integrity import RESOURCES, TOOLS, generate_integrity_file, verify_integrity
from .store import Store

logger = logging.getLogger("anneal-memory")

# MCP protocol version
_PROTOCOL_VERSION = "2024-11-05"

# Maximum message size (10MB) — prevents memory exhaustion from oversized lines
_MAX_MESSAGE_SIZE = 10 * 1024 * 1024


# -- Stdio Transport (newline-delimited JSON per MCP 2024-11-05 spec) --


# Sentinel to distinguish EOF from parse errors in _read_message
_EOF = object()


def _read_message() -> dict[str, Any] | object:
    """Read a JSON-RPC message from stdin (newline-delimited).

    Returns:
        Parsed dict on success, _EOF sentinel on stdin close,
        or a string error message on parse failure.
    """
    while True:
        line = sys.stdin.readline()
        if not line:
            return _EOF
        line = line.strip()
        if not line:
            continue  # Skip blank lines

        if len(line) > _MAX_MESSAGE_SIZE:
            return f"Message too large ({len(line)} bytes, max {_MAX_MESSAGE_SIZE})"

        try:
            return json.loads(line)
        except (json.JSONDecodeError, ValueError) as e:
            return f"JSON parse error: {e}"


def _write_message(msg: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout (newline-delimited)."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _response(msg_id: int | str | None, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error_response(
    msg_id: int | str | None, code: int, message: str
) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _tool_result(text: str, is_error: bool = False) -> dict[str, Any]:
    """Format a tool call result per MCP spec."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


# -- JSON-RPC Error Codes --
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


class Server:
    """MCP server backed by an anneal-memory Store.

    Handles the MCP protocol: initialize handshake, tool dispatch,
    resource serving. Single-threaded, synchronous.

    Args:
        store: An open Store instance.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self._handlers: dict[str, Any] = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
        }
        self._tool_handlers: dict[str, Any] = {
            "record": self._tool_record,
            "recall": self._tool_recall,
            "prepare_wrap": self._tool_prepare_wrap,
            "save_continuity": self._tool_save_continuity,
            "status": self._tool_status,
        }

    def run(self) -> None:
        """Main server loop. Reads messages from stdin, dispatches, responds."""
        while True:
            msg = _read_message()
            if msg is _EOF:
                break
            if isinstance(msg, str):
                # Parse error — respond per JSON-RPC spec and continue
                _write_message(_error_response(None, _PARSE_ERROR, msg))
                continue

            method = msg.get("method", "")
            msg_id = msg.get("id")

            # Notifications (no id) don't get responses
            if msg_id is None:
                logger.debug("Notification: %s", method)
                continue

            handler = self._handlers.get(method)
            if handler:
                try:
                    result = handler(msg.get("params") or {})
                    _write_message(_response(msg_id, result))
                except Exception as e:
                    logger.exception("Handler error for %s", method)
                    _write_message(
                        _error_response(msg_id, _INTERNAL_ERROR, str(e))
                    )
            else:
                _write_message(
                    _error_response(
                        msg_id, _METHOD_NOT_FOUND, f"Method not found: {method}"
                    )
                )

    # -- Protocol Handlers --

    def _handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
            },
            "serverInfo": {
                "name": "anneal-memory",
                "version": __version__,
            },
        }

    def _handle_ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"tools": TOOLS}

    def _handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name", "")
        arguments = params.get("arguments") or {}

        handler = self._tool_handlers.get(name)
        if not handler:
            return _tool_result(f"Unknown tool: {name}", is_error=True)

        try:
            return handler(arguments)
        except Exception as e:
            logger.exception("Tool error for %s", name)
            return _tool_result(f"Error: {e}", is_error=True)

    def _handle_resources_list(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"resources": RESOURCES}

    def _handle_resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri", "")
        if uri == "anneal://continuity":
            text = self._store.load_continuity()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": text
                        or "(No continuity file yet — record episodes and wrap to create one.)",
                    }
                ],
            }
        return {"contents": []}

    # -- Tool Implementations --

    def _tool_record(self, args: dict[str, Any]) -> dict[str, Any]:
        content = args.get("content", "")
        episode_type = args.get("episode_type", "")
        source = args.get("source", "agent")
        metadata = args.get("metadata")

        if not content:
            return _tool_result("Error: content is required", is_error=True)
        if not episode_type:
            return _tool_result("Error: episode_type is required", is_error=True)

        try:
            ep = self._store.record(
                content=content,
                episode_type=episode_type,
                source=source,
                metadata=metadata,
            )
        except ValueError as e:
            return _tool_result(f"Error: {e}", is_error=True)

        return _tool_result(
            f"Recorded {ep.type.value} ({ep.id}) at {ep.timestamp}"
        )

    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self._store.recall(
            since=args.get("since"),
            until=args.get("until"),
            episode_type=args.get("episode_type"),
            source=args.get("source"),
            keyword=args.get("keyword"),
            limit=max(0, args.get("limit", 100)),
            offset=max(0, args.get("offset", 0)),
        )

        if not result.episodes:
            return _tool_result("No matching episodes found.")

        lines = [
            f"Found {result.total_matching} episodes"
            f" (showing {len(result.episodes)}):"
        ]
        for ep in result.episodes:
            source_info = f" [{ep.source}]" if ep.source != "agent" else ""
            lines.append(
                f"- ({ep.id}) [{ep.type.value}] {ep.timestamp}"
                f"{source_info}: {ep.content}"
            )

        return _tool_result("\n".join(lines))

    def _tool_prepare_wrap(self, args: dict[str, Any]) -> dict[str, Any]:
        max_chars = args.get("max_chars", 20000)
        staleness_days = args.get("staleness_days", 7)

        episodes = self._store.episodes_since_wrap()

        if not episodes:
            # Clear any stale wrap_in_progress flag from an abandoned previous wrap
            self._store.wrap_cancelled()
            return _tool_result(
                "No episodes since last wrap. Nothing to compress."
            )

        existing = self._store.load_continuity()
        package = prepare_wrap_package(
            episodes=episodes,
            existing_continuity=existing,
            project_name=self._store.project_name,
            max_chars=max_chars,
            staleness_days=staleness_days,
        )

        # Mark wrap as in progress (clears any stale flag from abandoned wrap)
        self._store.wrap_started()

        # Build readable output for the agent
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

        return _tool_result("\n".join(parts))

    def _tool_save_continuity(self, args: dict[str, Any]) -> dict[str, Any]:
        text = args.get("text", "")
        if not text:
            return _tool_result("Error: text is required", is_error=True)

        # Check if prepare_wrap was called first
        skipped_prepare = not self._store.status().wrap_in_progress

        # Validate structure (4 required sections)
        if not validate_structure(text):
            return _tool_result(
                "Error: continuity must contain all 4 sections: "
                "## State, ## Patterns, ## Decisions, ## Context",
                is_error=True,
            )

        # Get current session's episodes for citation validation
        # Lowercase IDs to match graduation.py's normalization of cited IDs
        episodes = self._store.episodes_since_wrap()
        valid_ids = {ep.id[:8].lower() for ep in episodes}
        node_content_map = {ep.id[:8].lower(): ep.content for ep in episodes}

        # Check if citations have been seen before (bare graduation sunset)
        meta = self._store.load_meta()
        citations_seen = meta.get("citations_seen", False)

        # Validate graduations (demotes bad citations in-place)
        today = date.today().isoformat()
        grad_result = validate_graduations(
            text=text,
            valid_ids=valid_ids,
            today=today,
            node_content_map=node_content_map,
            citations_seen=citations_seen,
        )

        # Save the (possibly modified) continuity text
        path = self._store.save_continuity(grad_result.text)

        # Update metadata
        if grad_result.validated > 0 or grad_result.citation_counts:
            meta["citations_seen"] = True
        meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
        self._store.save_meta(meta)

        # Record wrap completion in the store
        sections = measure_sections(grad_result.text)
        patterns = len(re.findall(r"\|\s*\d+x", grad_result.text))
        self._store.wrap_completed(
            episodes_compressed=len(episodes),
            continuity_chars=len(grad_result.text),
            graduations_validated=grad_result.validated,
            graduations_demoted=grad_result.demoted + grad_result.bare_demoted,
            citation_reuse_max=grad_result.citation_reuse_max,
            patterns_extracted=patterns,
        )

        # Build response
        lines = [f"Continuity saved ({len(grad_result.text)} chars) to {path}"]
        if skipped_prepare:
            lines.append(
                "Note: prepare_wrap was not called first — "
                "continuity may not reflect current episodes."
            )
        lines.append(f"Episodes compressed: {len(episodes)}")

        if grad_result.validated:
            lines.append(f"Citations validated: {grad_result.validated}")
        if grad_result.demoted:
            lines.append(
                f"Citations demoted (bad evidence): {grad_result.demoted}"
            )
        if grad_result.bare_demoted:
            lines.append(
                f"Bare graduations demoted (no evidence): "
                f"{grad_result.bare_demoted}"
            )
        if grad_result.gaming_suspects:
            lines.append(
                f"Citation gaming suspects: "
                f"{', '.join(grad_result.gaming_suspects)}"
            )

        lines.append("\nSection sizes:")
        for name, chars in sorted(sections.items()):
            lines.append(f"  {name}: {chars} chars")

        return _tool_result("\n".join(lines))

    def _tool_status(self, args: dict[str, Any]) -> dict[str, Any]:
        status = self._store.status()

        lines = [
            f"Episodes: {status.total_episodes} total, "
            f"{status.episodes_since_wrap} since last wrap",
            f"Wraps: {status.total_wraps} completed",
        ]

        if status.last_wrap_at:
            lines.append(f"Last wrap: {status.last_wrap_at}")
        else:
            lines.append("Last wrap: never")

        if status.wrap_in_progress:
            lines.append(
                "Wrap in progress (prepare_wrap called, save_continuity pending)"
            )

        if status.continuity_chars is not None:
            lines.append(f"Continuity: {status.continuity_chars} chars")
        else:
            lines.append("Continuity: not yet created")

        if status.tombstone_count:
            lines.append(f"Tombstones: {status.tombstone_count}")

        if status.episodes_by_type:
            lines.append("\nBy type:")
            for type_name, count in sorted(status.episodes_by_type.items()):
                lines.append(f"  {type_name}: {count}")

        return _tool_result("\n".join(lines))


def main() -> None:
    """CLI entry point for the anneal-memory MCP server."""
    parser = argparse.ArgumentParser(
        prog="anneal-memory",
        description="Two-layer memory MCP server for AI agents.",
    )
    default_db = str(Path("~/.anneal-memory/memory.db").expanduser())
    parser.add_argument(
        "--db",
        default=default_db,
        help=f"Path to the SQLite database file (default: {default_db})",
    )
    parser.add_argument(
        "--project-name",
        default="Agent",
        help="Project name for continuity file header (default: Agent)",
    )
    parser.add_argument(
        "--generate-integrity",
        action="store_true",
        help="Generate tool-integrity.json and exit",
    )
    parser.add_argument(
        "--skip-integrity",
        action="store_true",
        help="Skip integrity verification on startup",
    )
    parser.add_argument(
        "--verify-audit",
        action="store_true",
        help="Verify audit trail hash chain integrity and exit",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable hash-chained JSONL audit trail",
    )
    parser.add_argument(
        "--audit-retention-days",
        type=int,
        default=0,
        help="Auto-cleanup rotated audit files older than N days (default: 0=keep forever)",
    )

    args = parser.parse_args()

    # Expand ~ in db path (user-provided or default)
    args.db = str(Path(args.db).expanduser())

    # Generate integrity file mode
    if args.generate_integrity:
        out = Path(__file__).parent / "tool-integrity.json"
        generate_integrity_file(out)
        print(f"Generated {out}", file=sys.stderr)
        return

    # Verify audit trail mode
    if args.verify_audit:
        from .audit import AuditTrail as _AT
        result = _AT.verify(args.db)
        if result.valid:
            print(
                f"Audit trail valid: {result.total_entries} entries "
                f"across {result.files_verified} file(s)",
                file=sys.stderr,
            )
        else:
            print(f"Audit trail INVALID: {result.error}", file=sys.stderr)
            if result.chain_break_at is not None:
                print(
                    f"  Chain broke at seq {result.chain_break_at} "
                    f"in {result.chain_break_file}",
                    file=sys.stderr,
                )
            sys.exit(1)
        return

    # Force UTF-8 on stdio — locale encoding can corrupt non-ASCII memories
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    # Logging to stderr (stdout is the MCP transport)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="[anneal-memory] %(levelname)s: %(message)s",
    )

    # Integrity verification — hard stop on failure (use --skip-integrity for dev)
    if not args.skip_integrity:
        integrity_path = Path(__file__).parent / "tool-integrity.json"
        if integrity_path.exists():
            valid, issues = verify_integrity(integrity_path)
            if not valid:
                for issue in issues:
                    logger.error("Integrity: %s", issue)
                logger.error(
                    "Tool description integrity check failed. "
                    "Use --skip-integrity to bypass."
                )
                sys.exit(1)
        # Missing file is not an error — first run or dev mode

    # Open store and run server
    audit_retention = args.audit_retention_days if args.audit_retention_days > 0 else None
    store = Store(
        path=args.db,
        project_name=args.project_name,
        audit=not args.no_audit,
        audit_retention_days=audit_retention,
    )

    try:
        server = Server(store)
        server.run()
    finally:
        store.close()


if __name__ == "__main__":
    main()
