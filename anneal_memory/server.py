"""MCP server for anneal-memory.

Implements the Model Context Protocol over stdio transport (JSON-RPC 2.0,
newline-delimited). 6 tools + 2 resources. Zero dependencies beyond Python
stdlib.

Usage:
    anneal-memory --db /path/to/memory.db [--project-name "My Agent"]
    anneal-memory --generate-integrity  # Generate tool-integrity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

# Module-level wrap-token shape constant. Mirrors the JSON schema
# pattern in ``integrity.py`` for the ``save_continuity`` tool's
# ``wrap_token`` field. Kept as a module-level compiled regex so the
# transport doesn't re-compile on every call AND so a future second
# transport (WebSocket, gRPC, etc.) can import the same constant
# instead of duplicating the pattern inline — contrarian Layer 3 F1
# flagged the inline ``import re`` pattern as a rule-of-three risk.
_WRAP_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")

from . import __version__
from .continuity import (
    format_wrap_package_text,
    prepare_wrap as _lib_prepare_wrap,
    validated_save_continuity as _lib_validated_save_continuity,
)
from .integrity import RESOURCES, TOOLS, hash_tool, generate_integrity_file, verify_integrity
from .store import Store, StoreError
from .types import AffectiveState

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
            "delete_episode": self._tool_delete_episode,
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
        if uri == "anneal://integrity/manifest":
            manifest = {
                "version": 1,
                "algorithm": "SHA-256",
                "canonicalization": "deterministic sorted-keys JSON",
                "tools": {tool["name"]: hash_tool(tool) for tool in TOOLS},
            }
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(manifest, indent=2, sort_keys=True),
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

    def _tool_delete_episode(self, args: dict[str, Any]) -> dict[str, Any]:
        episode_id = args.get("episode_id", "").strip()
        if not episode_id:
            return _tool_result("Error: episode_id is required", is_error=True)

        # Count associations before delete (CASCADE will remove them)
        # High limit: we need accurate count, not just top results
        assoc_count = len(self._store.get_associations([episode_id], limit=10000))

        deleted = self._store.delete(episode_id)
        if not deleted:
            return _tool_result(
                f"Episode {episode_id} not found. Use recall to find valid IDs.",
                is_error=True,
            )

        msg = f"Deleted episode {episode_id}"
        if assoc_count > 0:
            msg += f" and {assoc_count} associated link(s)"
        msg += ". This action is logged in the audit trail."
        return _tool_result(msg)

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
        """MCP transport adapter for the library prepare_wrap pipeline.

        Parses max_chars and staleness_days from MCP tool args, delegates
        to the library canonical pipeline, and formats the returned
        package as text via format_wrap_package_text. The library
        handles the full lifecycle (wrap_cancelled on empty,
        wrap_started on ready) so this function stays transport-only.

        On ``status == "ready"`` the library mints a session-handshake
        token (``wrap_token``) and persists a frozen snapshot of the
        episode IDs in store metadata. The token is appended to the
        agent-facing text so the agent can round-trip it back on the
        ``save_continuity`` call for explicit mismatch detection. The
        frozen-snapshot filter at save time applies regardless of
        whether the token is round-tripped — the token is the
        verification layer, not the snapshot enabler.
        """
        max_chars = args.get("max_chars", 20000)
        staleness_days = args.get("staleness_days", 7)

        result = _lib_prepare_wrap(
            self._store,
            max_chars=max_chars,
            staleness_days=staleness_days,
        )

        text = format_wrap_package_text(result)
        if result["status"] == "ready" and result["wrap_token"]:
            # Surface the token in a stable, machine-parseable line at
            # the end of the agent-facing text. The agent reads this
            # and passes it back on save_continuity. Format is
            # deliberately boring ("Wrap token: <hex>") so a simple
            # regex or endswith check can extract it; more elaborate
            # structure would overfit one parsing pattern.
            text = f"{text}\n\n---\nWrap token: {result['wrap_token']}"
        return _tool_result(text)

    def _tool_save_continuity(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP transport adapter for the library validated_save_continuity.

        Parses text and optional affective_state from MCP tool args,
        delegates the entire save pipeline (structure validation,
        graduation, associations, decay, metadata, wrap completion) to
        the library, and formats the returned dict as an MCP text
        response. ValueError from the library (empty text or missing
        sections) becomes an is_error=True tool result.
        """
        text = args.get("text", "")
        if not text:
            return _tool_result("Error: text is required", is_error=True)

        # Parse optional affective state (limbic layer).
        # JSON-to-float coercion is transport-specific (MCP receives
        # arbitrary JSON types; CLI's argparse already coerces). Once
        # we have a Python float, clamping is delegated to
        # AffectiveState.__post_init__ — the single source of truth
        # that both transports rely on, matching CLI behavior.
        affective_state: AffectiveState | None = None
        affect_raw = args.get("affective_state")
        if affect_raw and isinstance(affect_raw, dict):
            tag = affect_raw.get("tag", "")
            try:
                intensity = float(affect_raw.get("intensity", 0.0))
            except (ValueError, TypeError):
                intensity = 0.0
            if tag and isinstance(tag, str) and tag.strip():
                affective_state = AffectiveState(tag=tag, intensity=intensity)

        # Optional session-handshake token. When the agent round-trips
        # the token from the prior prepare_wrap response, the library
        # verifies it matches the persisted wrap and rejects stale or
        # wrong-wrap tokens. Omitting the token is fine for the
        # single-agent common case — the frozen-snapshot filter still
        # applies because the library consults the persisted snapshot
        # whenever it's present.
        #
        # Shape validation at the MCP boundary: must be a 32-char
        # hex string if present. The JSON schema also declares the
        # pattern but some MCP clients skip schema validation; this
        # explicit check is belt-and-suspenders so the library stays
        # free of regex. Uses the module-level ``_WRAP_TOKEN_RE``
        # constant (shared with any future transport) rather than an
        # inline pattern.
        wrap_token = args.get("wrap_token")
        if wrap_token is not None:
            if not isinstance(wrap_token, str):
                return _tool_result(
                    "Error: wrap_token must be a string if provided",
                    is_error=True,
                )
            if wrap_token == "":
                # Normalize empty string to None — same as "no token
                # passed." An empty-string token would fall into the
                # library's mismatch path with a confusing error.
                wrap_token = None
            elif not _WRAP_TOKEN_RE.fullmatch(wrap_token):
                return _tool_result(
                    "Error: wrap_token must be a 32-char hex string "
                    f"(got {len(wrap_token)} chars)",
                    is_error=True,
                )

        try:
            result = _lib_validated_save_continuity(
                self._store,
                text,
                affective_state=affective_state,
                wrap_token=wrap_token,
            )
        except ValueError as exc:
            return _tool_result(f"Error: {exc}", is_error=True)
        except StoreError as exc:
            # Surface the structured I/O error with operation + path
            # context so the agent sees which file / what operation
            # failed rather than a bare traceback.
            return _tool_result(
                f"Error: store I/O failure during {exc.operation} "
                f"at {exc.path}: {exc}",
                is_error=True,
            )

        # Format the library result dict as the MCP text response
        lines = [
            f"Continuity saved ({result['chars']} chars) to {result['path']}"
        ]
        if result["skipped_prepare"]:
            lines.append(
                "Note: prepare_wrap was not called first — "
                "continuity may not reflect current episodes."
            )
        lines.append(f"Episodes compressed: {result['episodes_compressed']}")

        if result["graduations_validated"]:
            lines.append(f"Citations validated: {result['graduations_validated']}")
        if result["demoted"]:
            lines.append(f"Citations demoted (bad evidence): {result['demoted']}")
        if result["bare_demoted"]:
            lines.append(
                f"Bare graduations demoted (no evidence): {result['bare_demoted']}"
            )
        if result["gaming_suspects"]:
            lines.append(
                f"Citation gaming suspects: {', '.join(result['gaming_suspects'])}"
            )

        if result["associations_formed"] or result["associations_strengthened"]:
            lines.append(
                f"Associations: {result['associations_formed']} formed, "
                f"{result['associations_strengthened']} strengthened"
            )
        if result["associations_decayed"]:
            lines.append(f"Associations decayed: {result['associations_decayed']}")

        lines.append("\nSection sizes:")
        for name, chars in sorted(result["sections"].items()):
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

        if status.association_stats and status.association_stats.total_links > 0:
            a = status.association_stats
            density_str = f"density {a.density:.4f}"
            if a.local_density > 0:
                density_str += f" (local {a.local_density:.4f})"
            lines.append(
                f"\nAssociations: {a.total_links} links, "
                f"avg strength {a.avg_strength:.2f}, "
                f"max {a.max_strength:.1f}, "
                f"{density_str}"
            )

        return _tool_result("\n".join(lines))


def start_server(
    *,
    db_path: str,
    project_name: str = "Agent",
    skip_integrity: bool = False,
    no_audit: bool = False,
    audit_retention_days: int | None = None,
) -> None:
    """Start the MCP server with the given configuration.

    Called by main() (standalone entry point) and by the CLI dispatcher's
    ``serve`` subcommand. Factored out so callers don't need to reconstruct
    sys.argv — just pass explicit parameters.
    """
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
    if not skip_integrity:
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
    store = Store(
        path=db_path,
        project_name=project_name,
        audit=not no_audit,
        audit_retention_days=audit_retention_days,
    )

    try:
        server = Server(store)
        server.run()
    finally:
        store.close()


def main() -> None:
    """CLI entry point for the anneal-memory MCP server.

    Parses command-line arguments and delegates to start_server()
    or handles one-shot modes (--generate-integrity, --verify-audit).
    """
    parser = argparse.ArgumentParser(
        prog="anneal-memory",
        description="Living memory MCP server for AI agents.",
    )
    env_db = os.environ.get("ANNEAL_MEMORY_DB")
    default_db = env_db if env_db else str(Path("~/.anneal-memory/memory.db").expanduser())
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

    audit_retention = args.audit_retention_days if args.audit_retention_days > 0 else None
    start_server(
        db_path=args.db,
        project_name=args.project_name,
        skip_integrity=args.skip_integrity,
        no_audit=args.no_audit,
        audit_retention_days=audit_retention,
    )


if __name__ == "__main__":
    main()
