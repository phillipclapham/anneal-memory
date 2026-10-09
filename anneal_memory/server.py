"""MCP server for anneal-memory.

Implements the Model Context Protocol over stdio transport (JSON-RPC 2.0,
newline-delimited). 17 tools + 2 resources. Zero dependencies beyond Python
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
import sys
from pathlib import Path
from typing import Any, cast

from . import __version__
from .continuity import (
    format_wrap_package_text,
    prepare_wrap as _lib_prepare_wrap,
    validated_save_continuity as _lib_validated_save_continuity,
)
from .integrity import RESOURCES, TOOLS, hash_tool, generate_integrity_file, verify_integrity
# ``_WRAP_TOKEN_RE`` lives in ``store.py`` as of 10.5c.5 — shared
# shape constant importable by any transport (CLI, MCP, future
# WebSocket/gRPC). The earlier server-module home was an
# architectural wart that made the CLI load the entire MCP server
# module just to reach one compiled regex.
from .spores import (
    VALID_GERMINATIONS,
    VALID_TIERS,
    VALID_TYPES,
    SporeStore,
    germination_tier,
)
from .crystal import CrystalError, CrystalStore
from .retrieval import (
    MAX_PATTERNS,
    MIN_KEYWORDS,
    QUERY_MIN_KEYWORDS,
    RETRIEVAL_MODES,
    durable_facts_for,
    EpisodeMatch,
    RetrievalMode,
    extract_keywords,
    retrieve_patterns,
    retrieve_relevant,
    search_episodes_counted,
)
from .store import (
    Store,
    StoreDatabaseError,
    _is_write_lock_contention,
    StoreError,
    WrapCancelBoundError,
    WrapCancelGatedError,
    WrapOwnershipError,
    _WRAP_TOKEN_RE,
)
from .types import AffectiveState, EpisodeType, RelevantFact, RelevantPattern

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


def _truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with an ellipsis (parity with the CLI ``_truncate``)."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# -- JSON-RPC Error Codes --
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603


# Word-match presentation in MCP ``recall`` (see ``Server._recall_word_fallback``).
_FALLBACK_DEFAULT_CAP = 10   # word matches listed when the caller passed no ``limit``
_EXACT_RESULTS_ENOUGH = 3    # an exact result this small is topped up with word matches
_ALSO_MATCHING_MAX = 5       # how many word matches are appended to such a result
_RECALL_DEFAULT_LIMIT = 100  # MCP recall's ``limit`` when the caller passes none


def _as_int(value: object) -> int | None:
    """``value`` as an integer, or ``None`` if it is not one. A whole-number float
    counts (3.0 -> 3); a bool, a fractional or non-finite float and every other type
    do not."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _durable_block(facts: list[RelevantFact]) -> str:
    """The reply block for durable facts a query cued, or ``""`` for none: each fact's
    text (not its cue list), then the words that brought it up: ``cue: ...`` for matched
    cue words and ``matches: ...`` for matched words of the fact text, both when both
    took part."""
    if not facts:
        return ""
    lines = ["Durable facts matching your words:"]
    for f in facts:
        parts = []
        if f.cue_matched:
            parts.append(f"cue: {', '.join(f.cue_matched)}")
        if f.fact_matched:
            parts.append(f"matches: {', '.join(f.fact_matched)}")
        if not parts:  # a RelevantFact built without the split fields
            parts.append(f"{'cue' if f.source == 'cue' else 'matches'}: {', '.join(f.matched)}")
        lines.append(f"- {f.fact} ({'; '.join(parts)})")
    return "\n".join(lines)


def _word_match_line(match: EpisodeMatch, word_count: int) -> str:
    """One ``recall`` reply line for a word-by-word match, in the exact path's shape
    plus how many of the query's words the episode matched."""
    ep = match.episode
    source_info = f" [{ep.source}]" if ep.source != "agent" else ""
    replaced = f" (superseded by {match.superseded_by})" if match.superseded_by else ""
    return (
        f"- ({ep.id}) [{ep.type}] {ep.timestamp}{source_info}{replaced}"
        f" (matched {len(match.matched)}/{word_count}: {', '.join(match.matched)}):"
        f" {ep.content}"
    )


class Server:
    """MCP server backed by an anneal-memory Store.

    Handles the MCP protocol: initialize handshake, tool dispatch,
    resource serving. Single-threaded, synchronous.

    Args:
        store: An open Store instance.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        # The prospective (spore) store is a JSON sibling of the episodic db:
        # ``<stem>.spores.json``, mirroring the ``<stem>.continuity.md`` sibling.
        # Created on first write, so it need not pre-exist.
        base = Path(store.path) if store.path is not None else Path("memory.db")
        self._spore_store = SporeStore(base.parent / f"{base.stem}.spores.json")
        # The crystallized-pattern store is another JSON sibling
        # (``<stem>.crystal.json``) — the on-demand graduated tier (AM-CRYSTAL).
        # AM-CRYSTAL-OPTIN: store the PATH only; the wrap handlers pass it into
        # prepare/save ONLY when the file already exists (the opt-in signal), so an
        # MCP operator who never crystallized is never dropped into the
        # crystallize-OUT regime — byte-identical pre-crystal wrap. Persistent
        # opt-in = `crystal crystallize` writes the file (then every wrap, CLI +
        # MCP, auto-enables); `prepare-wrap --crystal` enables a single CLI wrap
        # but does NOT itself create the store.
        self._crystal_path = base.parent / f"{base.stem}.crystal.json"
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
            "wrap_cancel": self._tool_wrap_cancel,
            "delete_episode": self._tool_delete_episode,
            "status": self._tool_status,
            "crystal_recall": self._tool_crystal_recall,
            "crystal_index": self._tool_crystal_index,
            "spore_add": self._tool_spore_add,
            "spore_get": self._tool_spore_get,
            "spore_list": self._tool_spore_list,
            "spore_touch": self._tool_spore_touch,
            "spore_update": self._tool_spore_update,
            "spore_descend": self._tool_spore_descend,
            "spore_ascend": self._tool_spore_ascend,
            "spore_surface": self._tool_spore_surface,
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

            # After the EOF + str guards, _read_message() guarantees a
            # parsed JSON-RPC dict. Narrow for the type checker — with an
            # explicit raise, not an ``assert``: asserts are stripped under
            # ``python -O``, so the narrowing (and any protection it implies)
            # silently disappears in exactly the deployment most likely to run
            # optimised. The two siblings in continuity.py were converted for
            # this reason; this was the third and last.
            if not isinstance(msg, dict):
                raise TypeError(
                    f"parsed message must be a dict, got {type(msg).__name__}"
                )
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
                supersedes=args.get("supersedes"),
            )
        except ValueError as e:  # SupersessionError is a ValueError
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
        """Episode recall with the durable-fact tier on top: when a ``keyword`` is given
        on the first page, the durable facts its words cue are listed first (see
        :func:`_durable_block`), and a call that matched no episode but cued a fact
        returns the facts, followed by "No matching episodes found."."""
        # ``limit`` and ``offset`` reach SQLite and slice indices: a whole-number float (a
        # client that serializes 3 as 3.0) is read as the integer, and anything else that
        # is not an integer is refused by name rather than by a driver error. A negative
        # value reads as 0. Normalised ONCE, here, so the episode tier and the facts gate
        # below see the same numbers.
        args = dict(args)
        for name in ("limit", "offset"):
            if name in args:
                value = _as_int(args[name])
                if value is None:
                    return _tool_result(
                        f"Error: {name} must be an integer", is_error=True
                    )
                args[name] = max(0, value)
        result = self._recall_episodes(args)
        keyword = args.get("keyword")
        # Durable facts are not episodes: they go on a plain keyword recall's first page,
        # and not on a call that filters episodes (since/until/source/episode_type) or one
        # that asks for none (limit 0, which returns nothing at all, facts included).
        if (
            result.get("isError")
            or not isinstance(keyword, str)
            or args.get("offset", 0) != 0
            or args.get("limit", _RECALL_DEFAULT_LIMIT) <= 0
            or any(args.get(f) for f in ("since", "until", "source", "episode_type"))
        ):
            return result
        block = _durable_block(self._cued_facts(keyword, "query"))
        if not block:
            return result
        text = result["content"][0]["text"]
        return _tool_result(block + "\n\n" + text)

    def _cued_facts(self, query: str, mode: RetrievalMode) -> list[RelevantFact]:
        """The durable facts of this server's store that ``query`` cues."""
        return durable_facts_for(self._store, query, mode=mode)

    def _recall_episodes(self, args: dict[str, Any]) -> dict[str, Any]:
        episode_type = args.get("episode_type")
        if episode_type is not None and (
            not isinstance(episode_type, str)
            or episode_type not in {t.value for t in EpisodeType}
        ):
            valid = ", ".join(t.value for t in EpisodeType)
            return _tool_result(
                f"Error: episode_type {episode_type!r} is not one of: {valid}.",
                is_error=True,
            )
        result = self._store.recall(
            since=args.get("since"),
            until=args.get("until"),
            episode_type=args.get("episode_type"),
            source=args.get("source"),
            keyword=args.get("keyword"),
            limit=max(0, args.get("limit", _RECALL_DEFAULT_LIMIT)),
            offset=max(0, args.get("offset", 0)),
            include_superseded=args.get("include_superseded") is True,
        )

        if not result.episodes:
            fallback = self._recall_word_fallback(args, result.total_matching)
            if fallback is not None:
                return fallback
            return _tool_result("No matching episodes found.")

        lines = [
            f"Found {result.total_matching} episodes"
            f" (showing {len(result.episodes)}):"
        ]
        for ep in result.episodes:
            source_info = f" [{ep.source}]" if ep.source != "agent" else ""
            replaced = f" (superseded by {ep.superseded_by})" if ep.superseded_by else ""
            lines.append(
                f"- ({ep.id}) [{ep.type.value}] {ep.timestamp}"
                f"{source_info}{replaced}: {ep.content}"
            )

        # A phrase that hit only a little, from a keyword with three or more words,
        # probably missed the episode that holds most of those words. Exact results stay
        # first and unchanged; the word matches the exact search did not already show
        # follow them, only on the first page of an exhausted exact result.
        keyword = args.get("keyword")
        if (
            isinstance(keyword, str)
            and args.get("offset", 0) == 0
            and result.total_matching < _EXACT_RESULTS_ENOUGH
            and len(result.episodes) == result.total_matching
        ):
            words = extract_keywords(keyword, mode="query")
            if len(words) >= 3:
                shown = {ep.id for ep in result.episodes}
                room = min(
                    _ALSO_MATCHING_MAX, args.get("limit", _RECALL_DEFAULT_LIMIT) - len(shown)
                )
                extra = [
                    m for m in self._word_matches(args, keyword)[0]
                    if m.episode.id not in shown
                ][:max(0, room)]
                if extra:
                    lines.append("")
                    lines.append("Also matching by words:")
                    lines.extend(_word_match_line(m, len(words)) for m in extra)

        return _tool_result("\n".join(lines))

    def _word_matches(
        self, args: dict[str, Any], keyword: str
    ) -> tuple[list[EpisodeMatch], bool]:
        """Every word-by-word match for ``keyword`` under the call's filters, best
        first, with whether the read was cut short by the per-keyword ceiling (the
        caller caps and counts)."""
        return search_episodes_counted(
            self._store,
            keyword,
            episode_type=args.get("episode_type"),
            source=args.get("source"),
            since=args.get("since"),
            until=args.get("until"),
            limit=sys.maxsize,
            include_superseded=args.get("include_superseded") is True,
        )

    def _recall_word_fallback(
        self, args: dict[str, Any], total_matching: int
    ) -> dict[str, Any] | None:
        """Word-by-word ranking for a multi-word ``keyword`` the exact-phrase recall
        missed, or ``None`` when the fallback does not apply (the caller then reports
        "No matching episodes found." as before).

        It applies only when the exact query matched NOTHING (``total_matching == 0`` —
        a ``limit`` of 0 or an ``offset`` past the matches is not a miss), the first
        page was asked for, the ``keyword`` is two or more whitespace-separated tokens,
        and it reduces to at least one distinctive word (a phrase that reduces to one
        word still falls back on that word). A single-token keyword has nothing to
        split, so it never falls back. The same filters go through. The list is capped
        at :data:`_FALLBACK_DEFAULT_CAP` unless the caller passed a ``limit``, which is
        then honoured; when more matched than are shown the reply says so. It names the
        words, and per episode how many matched, so the agent can tell this ranked list
        from an exact match."""
        keyword = args.get("keyword")
        if total_matching != 0 or not isinstance(keyword, str):
            return None
        if args.get("offset", 0) > 0 or len(keyword.split()) < 2:
            return None
        words = extract_keywords(keyword, mode="query")
        if not words:
            return None
        cap = args["limit"] if "limit" in args else _FALLBACK_DEFAULT_CAP
        if cap <= 0:
            return None
        matches, truncated = self._word_matches(args, keyword)
        if not matches:
            return None
        shown = matches[:cap]
        # The candidate read per keyword has a ceiling; when a keyword had more matches
        # than were read, the count is a floor. (The search reports it from the counts it
        # already took; nothing is counted again here.)
        read_all = not truncated
        head = (
            "No episode contains the exact phrase; ranked by matching words "
            f"({', '.join(words)})."
        )
        if len(matches) > len(shown) or not read_all:
            count = f"{len(matches)}" if read_all else f"at least {len(matches)}"
            head += (
                f" Showing top {len(shown)} of {count} word matches; pass a "
                "rarer word or a higher limit for more."
            )
        else:
            head += f" Showing {len(shown)}:"
        lines = [head]
        lines.extend(_word_match_line(m, len(words)) for m in shown)
        return _tool_result("\n".join(lines))

    def _crystal_store_for_wrap(self) -> CrystalStore | None:
        """AM-CRYSTAL-OPTIN: the crystal tier on the MCP wrap path is opt-in.

        Returns the ``CrystalStore`` only when its file already exists (the
        opt-in signal — created when a pattern is first crystallized via the CLI
        ``crystal crystallize``; ``CrystalStore(path)`` is lazy and does not
        create it on open); otherwise ``None`` ⇒ byte-identical pre-crystal
        wrap. An MCP operator who never crystallized is never dropped into the
        crystallize-OUT regime. MCP has no per-call flag, so file existence is
        the only opt-in signal here — bootstrap by crystallizing via the CLI.
        """
        return (
            CrystalStore(self._crystal_path)
            if self._crystal_path.exists() else None
        )

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
        # AM-SCHEMA-BUDGET (v0.4.2): omit max_chars (None) -> the library
        # derives a schema-aware budget (20000 for the ops DEFAULT_SCHEMA,
        # larger for a richer schema like FLOW_SCHEMA). Do NOT inject a flat
        # 20000 default here — that would silently defeat schema-aware budgeting
        # for a partnership entity wrapping over MCP. An explicit caller value
        # still overrides.
        max_chars = args.get("max_chars")
        staleness_days = args.get("staleness_days", 7)

        result = _lib_prepare_wrap(
            self._store,
            max_chars=max_chars,
            staleness_days=staleness_days,
            # AM-CRYSTAL-OPTIN: opt-in gate — pass the store only when it exists.
            crystal_store=self._crystal_store_for_wrap(),
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

        # Optional catastrophic-shrink override (v0.3.5). Defaults to
        # False; only a deliberate diet / migration recompression should
        # set it. Require a genuine JSON boolean ``true`` — for a SAFETY
        # override, a stringy ``"false"`` (or any other non-bool JSON)
        # must NOT silently disable the gate, so anything that is not
        # literally ``True`` leaves it enabled.
        allow_shrink = args.get("allow_shrink", False) is True
        # Deprecated no-op since 0.9.26: the AM-LINKGATE refusal it overrode
        # was removed. Still accepted and passed through unchanged.
        allow_unlinked = args.get("allow_unlinked", False) is True

        try:
            result = _lib_validated_save_continuity(
                self._store,
                text,
                affective_state=affective_state,
                wrap_token=wrap_token,
                allow_shrink=allow_shrink,
                allow_unlinked=allow_unlinked,
                # AM-CRYSTAL-OPTIN: opt-in gate — pass the store only when it exists.
                crystal_store=self._crystal_store_for_wrap(),
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
        lines.append(f"Episodes compressed: {result['episodes_compressed']}")
        if result.get("stale_state"):
            lines.append(
                "State lines that do not hold (fix them next wrap): "
                + "; ".join(result["stale_state"])
            )
        lines.append(
            f"Citation spread: {result['citation_spread']} distinct episode(s) "
            f"cited on today's 2x-and-up graduation lines (counted before grounding checks)"
        )

        if result["graduations_validated"]:
            lines.append(f"Citations validated: {result['graduations_validated']}")
        if result["demoted"]:
            lines.append(f"Citations demoted (bad evidence): {result['demoted']}")
        if result["bare_demoted"]:
            lines.append(
                f"Bare graduations demoted (no evidence): {result['bare_demoted']}"
            )
        if result["skipped_non_today"]:
            # Carried-forward graduations from prior sessions are
            # normal. A non-zero count alongside a failing test or
            # unexpected validation gap is the Finding #3 test-drift
            # class — surfacing it here gives operators parity with
            # the library return shape.
            lines.append(
                f"Graduations skipped (non-today date): "
                f"{result['skipped_non_today']}"
            )
        if result["gaming_suspects"]:
            lines.append(
                f"Citation gaming suspects: {', '.join(result['gaming_suspects'])}"
            )
        # The prior-state bound's cuts travel in the text: the UserWarning is
        # post-commit and never reaches an MCP client (L2 r1, run).
        for cap in cast("dict[str, Any]", result).get("level_capped") or []:
            lines.append(
                f"Level capped: {cap['name']} {cap['written_level']}x -> "
                f"{cap['capped_to']}x (a new pattern enters at 1x; a validated Nx "
                f"becomes (N+1)x)"
            )

        if result["associations_formed"] or result["associations_strengthened"]:
            lines.append(
                f"Associations: {result['associations_formed']} formed, "
                f"{result['associations_strengthened']} strengthened"
            )
        if result["associations_decayed"]:
            lines.append(f"Associations decayed: {result['associations_decayed']}")
        if result["supersessions_recorded"]:
            lines.append(f"Supersessions recorded: {result['supersessions_recorded']}")
        for rej in result["supersessions_rejected"]:
            lines.append(
                f"Supersession rejected ({rej['old_id']} by {rej['new_id']}): {rej['reason']}"
            )

        if result["association_warning"]:
            # A post-commit UserWarning never reaches an MCP client, so the
            # AM-WARN signal travels in the result text (codex L3 MED, 0.9.26:
            # with the AM-LINKGATE refusal removed, a dead association write
            # path would otherwise be silent over MCP).
            lines.append(f"Association warning: {result['association_warning']}")
        if allow_unlinked is True:
            lines.append(
                "allow_unlinked is deprecated and did nothing: the AM-LINKGATE save "
                "refusal it overrode was removed in 0.9.26."
            )

        # Durable-fact save warnings (a re-inserted line, a drop marker that named
        # nothing) are post-commit and never reach an MCP client as a UserWarning, so
        # they travel in the result text too. Read leniently: a result without the key
        # has none.
        raw_warnings: Any = cast("dict[str, Any]", result).get("durable_warnings") or []
        durable_warnings = [w for w in raw_warnings if isinstance(w, str) and w]
        if durable_warnings:
            lines.append("\nDurable facts:")
            prefix = "Durable facts: "
            for w in durable_warnings:
                lines.append(f"  - {w[len(prefix):] if w.startswith(prefix) else w}")

        lines.append("\nSection sizes:")
        for name, chars in sorted(result["sections"].items()):
            lines.append(f"  {name}: {chars} chars")

        return _tool_result("\n".join(lines))

    def _tool_wrap_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP transport adapter for Store.wrap_cancelled().

        The operator escape hatch for a stuck wrap, which until 0.9.8
        existed only as the ``wrap-cancel`` CLI subcommand and the
        ``Store.wrap_cancelled()`` Python method — neither reachable by
        an MCP client, so an agent that hit ``WrapInProgressError`` had
        no in-band way out and the wrap stayed stuck until a human
        opened a terminal. Reported from the field by Alex De Groodt,
        who sat locked for three days with 31 episodes stranded.

        Reports from the RECEIPT ``wrap_cancelled()`` returns, which is read
        inside the same transaction as the clear — true only because that
        method opens with ``BEGIN IMMEDIATE`` (added after 0.9.8). Remove it and
        this sentence goes back to being false.

        ⚠ IT DELIBERATELY DOES NOT ``load_wrap_snapshot()`` FIRST, AND THAT IS
        THE WHOLE POINT. The first version did, to have something to report.
        That is a TOCTOU: between the read and the clear, a peer session can
        finish the wrap this call saw and start a NEW one — and the clear is
        unconditional, so it destroys the new wrap while the response names the
        old token and count. The operator is told they cleaned up a corpse when
        they killed a live peer. Found by codex and glm independently at L3.
        Reading inside the transaction removes the window rather than narrowing
        it, and the receipt cannot disagree with what was cleared. ⚠ 0.9.8
        shipped that sentence WITHOUT the ``BEGIN IMMEDIATE`` that makes it
        true: bare SELECTs open no transaction under ``isolation_level=""``, so
        the window was narrowed, not removed, and the race was reproduced on
        2026-09-03. The guarantee lives in ``Store.wrap_cancelled``, not here.

        The partial-state path (``StoreError`` territory for
        ``load_wrap_snapshot``) needs no special handling here any more:
        ``wrap_cancelled()`` clears raw metadata and never parses it, so it
        works on exactly the corrupt states that method refuses — and the
        receipt says ``partial_state`` so the response can tell the operator
        what really happened instead of "no wrap was in progress", which was
        false precisely on the recovery path this tool most exists to serve.
        """
        expect_token = args.get("wrap_token")
        if expect_token is not None:
            # fullmatch, not match: `$` matches before a trailing newline, so
            # `match` accepts "a"*32 + "\n" — 33 characters — and the store then
            # reports a misleading ownership MISMATCH instead of an invalid
            # token. save_continuity's validator already used fullmatch; this
            # one silently did not. codex + glm, L3.
            if not isinstance(expect_token, str) or not _WRAP_TOKEN_RE.fullmatch(
                expect_token
            ):
                return _tool_result(
                    "wrap_token must be the 32-character hex token prepare_wrap "
                    "returned.",
                    is_error=True,
                )

        session_id = args.get("session_id")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            return _tool_result("session_id must be a non-empty string.", is_error=True)
        force = args.get("force", False)
        if not isinstance(force, bool):
            return _tool_result("force must be true or false.", is_error=True)
        partial = args.get("partial", False)
        if not isinstance(partial, bool):
            return _tool_result("partial must be true or false.", is_error=True)
        if partial and expect_token is not None:
            return _tool_result(
                "partial and wrap_token cannot be combined: partial clears the "
                "store only while it holds partial wrap state, whatever its token.",
                is_error=True,
            )

        try:
            # expect_partial only when asked, so the plain call is unchanged for a
            # Store subclass or stub that predates the keyword.
            receipt = self._store.wrap_cancelled(
                expect_token=expect_token,
                session_id=session_id,
                force=force,
                **({"expect_partial": True} if partial else {}),
            )
        except WrapCancelBoundError:
            # No recipe, as for the gated refusal below.
            return _tool_result(
                "Refused: the wrap in progress was opened with a token its preparer "
                "holds, and a cancel that names no token cannot end it. Cancelling "
                "it discards that caller's compression, which is the operator's "
                "decision. Nothing was changed.",
                is_error=True,
            )
        except WrapCancelGatedError as exc:
            # No recipe here, on purpose: the reader of a refusal is the caller the
            # bound exists to stop.
            return _tool_result(
                "Refused: the wrap in progress was prepared under the consolidate "
                "gate by another session. Cancelling it discards that session's "
                "compression, which is the operator's decision. Nothing was changed.",
                is_error=True,
            )
        except WrapOwnershipError as exc:
            # ⭐ THE REFUSAL IS THE FEATURE, so it reports what is true and what
            # to do — not a bare mismatch. `actual is None` is a DIFFERENT fact
            # from "a peer owns it": the first means the caller's own wrap has
            # already finished (retry-safe, nothing to do), the second means
            # cancelling would destroy someone else's compression.
            if partial:
                # The partial-only clear found no partial state: a healthy wrap
                # replaced it (whatever its token) or the store went idle.
                now = (
                    "a healthy wrap is in progress now, so it was not touched"
                    if exc.actual else "no wrap is in progress now"
                )
                return _tool_result(
                    f"Nothing was changed: the store no longer holds partial wrap "
                    f"state ({now}).",
                    is_error=True,
                )
            if exc.partial_state:
                # ⛔ THE THIRD STATE, AND OMITTING IT WAS A REAL LOCKOUT. Partial
                # metadata (wrap_started_at set, wrap_token empty) can match NO
                # token, so no proven cancel can ever succeed. Reporting it as
                # "nothing in progress" left wrap_started_at standing and the
                # next prepare_wrap failed — Alex's original three-day lockout,
                # through the guard written to prevent it. L3 consensus.
                # The recovery is partial=true, not a plain cancel: a peer may
                # clear the state and start a healthy wrap before the retry, and
                # a plain cancel would end it; a surviving token may be one this
                # tool cannot accept (L3 r2/r3, 1003+16, run).
                return _tool_result(
                    "The store holds PARTIAL wrap state — a crash or a hand edit "
                    "left it half-written, and it cannot be saved. prepare_wrap "
                    "will keep refusing until it is cleared. Nothing was changed. "
                    "Call wrap_cancel again with partial=true and no wrap_token to "
                    "clear it; that refuses if a healthy wrap has replaced it.",
                    is_error=True,
                )
            if exc.actual is None:
                return _tool_result(
                    "Nothing to cancel: the wrap you named has already completed "
                    "or been cancelled, and no wrap is in progress now. Nothing "
                    "was changed — prepare_wrap will start a fresh one.",
                    is_error=True,
                )
            if exc.bound:
                # A call without wrap_token would hit WrapCancelBoundError, so no
                # override is offered, and no recipe.
                return _tool_result(
                    "Refused: the wrap in progress is NOT the one you named, and it "
                    "was opened with a token its preparer holds. Cancelling it "
                    "discards that caller's compression, which is the operator's "
                    "decision. "
                    + ("force is ignored while wrap_token is given. " if exc.force else "")
                    + "Nothing was changed. Call `status` to see when it started.",
                    is_error=True,
                )
            if exc.gated_session and exc.gated_session != session_id and exc.force:
                return _tool_result(
                    "Refused: the wrap in progress is NOT the one you named, and "
                    "another session prepared it under the consolidate gate. force "
                    "is ignored while wrap_token is given, so nothing was changed.",
                    is_error=True,
                )
            if exc.gated_session and exc.gated_session != session_id:
                # The override below would hit WrapCancelGatedError, so it is not
                # offered; and, as there, no recipe (Diogenes 10-03, edd780d2e03e).
                return _tool_result(
                    "Refused: the wrap in progress is NOT the one you named, and "
                    "another session prepared it under the consolidate gate. "
                    "Cancelling it discards that session's compression, which is "
                    "the operator's decision. Nothing was changed. Call `status` "
                    "to see when that wrap started.",
                    is_error=True,
                )
            if exc.gated_session:
                return _tool_result(
                    "Refused: the wrap in progress is NOT the one you named; it is "
                    "a different wrap prepared under your own session. Nothing was "
                    "changed. Call `status` to see when it started; to end it, call "
                    "wrap_cancel again WITHOUT wrap_token, keeping session_id.",
                    is_error=True,
                )
            return _tool_result(
                "Refused: the wrap in progress is NOT the one you named, so it "
                "belongs to a different session and cancelling it would destroy "
                "its compression. Nothing was changed. Call `status` to see when "
                "that wrap started; if it really is abandoned, call wrap_cancel "
                "again WITHOUT wrap_token to override.",
                is_error=True,
            )
        except StoreDatabaseError as exc:
            # ⚠ NOT A NEW FAILURE MODE — I ASSERTED THAT AND IT WAS FALSE.
            # The L2 seat MEASURED the pre-BEGIN-IMMEDIATE shape under the same
            # contention: the bare SELECTs succeed instantly (WAL readers are
            # never blocked), then the FIRST INSERT blocks and raises the
            # identical "database is locked" at 5.19s, versus 5.20s now. Same
            # error, same class, same latency — the lock statement only moved it
            # ahead of the reads. So there is no loudness-for-safety trade here;
            # the loud failure already existed and the silent corruption is gone.
            # What was always missing is a REACHABLE NEXT ACTION on the one tool
            # an agent calls when it is already stuck — 0.9.8's own defect class
            # (a recovery message naming a path its caller cannot take).
            #
            # ⛔ AND THE FIRST DRAFT OF THIS MESSAGE OVERCLAIMED — codex, L3.
            # It said the lock was "evidence the wrap is LIVE, not stranded".
            # It is not. A write lock proves only that SOMEONE is writing: an
            # unrelated `record`, a prune, a schema init on a fresh open, or an
            # abandoned transaction holds it identically while an old wrap stays
            # stranded. Ownership is exactly what SQLite locks do not carry.
            # ⚡ The test made the point by accident — it holds a RAW
            # BEGIN IMMEDIATE on an unrelated connection, i.e. the counterexample
            # to the claim it was asserting. Establishing liveness needs
            # persisted owner/lease data, which is spore-699, not a lock.
            # So the message reports what is actually known and lets `status`
            # settle ownership.
            if _is_write_lock_contention(exc):
                return _tool_result(
                    "Could not cancel: another process holds this store's "
                    "write lock, so something else is writing RIGHT NOW. "
                    "Nothing was changed and the store is not corrupt. ⚠ This "
                    "does NOT tell you whether the pending wrap belongs to that "
                    "writer — a lock carries no ownership, and an unrelated "
                    "write holds it the same way a live peer's wrap does. Do "
                    "NOT retry blindly: a wrap that IS live is destroyed by "
                    "cancelling it. Wait a few seconds, then call `status`. If "
                    "`wrap_in_progress` is gone, the writer finished and there "
                    "is nothing to cancel. If a wrap is still open, compare its "
                    "start time: unchanged across several checks means it is "
                    "probably stranded and safe to cancel; moving means a peer "
                    "is actively working and you should leave it alone.",
                    is_error=True,
                )
            raise

        if receipt.partial_state:
            token = f" (token: {receipt.token})" if receipt.token else ""
            return _tool_result(
                f"Wrap state was CORRUPT (partial wrap-in-progress metadata) "
                f"and has been cleared{token}. This is the recovery case: the "
                f"store was left half-way through a wrap. Your episodes are "
                f"intact — prepare_wrap will now start a fresh wrap and pick "
                f"them all up."
            )

        if receipt.token is None:
            return _tool_result(
                "No wrap was in progress (wrap state cleared anyway). "
                "prepare_wrap will now start a fresh wrap."
            )

        count = len(receipt.episode_ids) if receipt.episode_ids is not None else 0
        started = f", started {receipt.started_at}" if receipt.started_at else ""
        return _tool_result(
            f"Wrap cancelled (token: {receipt.token}{started}). "
            f"{count} episode(s) released from the frozen snapshot "
            f"— they are NOT deleted and the next prepare_wrap picks them up."
        )

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
            # The START TIME is what makes the wrap_cancel guidance actionable.
            # That tool's description tells the agent to check status first and
            # not to cancel a wrap that began moments ago (it is probably a live
            # peer mid-compression, not a corpse) — and until 0.9.8 status
            # reported only a boolean, so the check it named could not be
            # performed. Advice pointing at a surface that cannot answer it is
            # the same defect this release exists to close (codex L3).
            started_at = self._store.get_wrap_started_at()
            lines.append(
                "Wrap in progress (prepare_wrap called, save_continuity pending)"
                + (f" — started {started_at}" if started_at else "")
            )

        if status.consolidate_requires_baton:
            lines.append(
                "Baton-protected store: only the session holding the consolidate baton "
                "can consolidate; prepare_wrap from this server downgrades"
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

        # Audit layer visibility — tamper-evident trail is load-bearing
        # for the "grounded memory" claim. Agents and operators need to
        # be able to see that it's running without shelling out.
        lines.append("")
        if status.audit_enabled:
            if status.audit_entry_count is not None:
                audit_line = (
                    f"Audit: enabled — {status.audit_entry_count} entries"
                )
            else:
                audit_line = "Audit: enabled — entry count unavailable"
            if status.audit_retention_days is not None:
                audit_line += (
                    f", retention {status.audit_retention_days}d"
                )
            else:
                audit_line += ", retention unlimited"
            if status.audit_write_failures:
                audit_line += (
                    f" — ⚠ {status.audit_write_failures} write(s) FAILED and "
                    f"were dropped over this store's LIFETIME; the trail is "
                    f"INCOMPLETE (verify() cannot see a missing entry)"
                )
                if status.audit_last_failure:
                    audit_line += f", last: {status.audit_last_failure}"
            lines.append(audit_line)
            if status.audit_log_path is not None:
                lines.append(f"Audit log: {status.audit_log_path}")
            lines.append(
                "Audit chain: run `anneal-memory verify` to validate"
            )
        else:
            lines.append("Audit: disabled")

        # Post-commit auto-prune health (Diogenes 2026-09-17, codex L3
        # 2026-09-17 MED — wiring this in was the finding: a field on
        # ``StoreStatus`` is not a surface, per the audit precedent this
        # section is a copy of). Wired HERE ONLY, not into the CLI ``status``
        # command: ``status.prune_failures`` is instance-local on
        # ``self._store`` with no metadata-table flush point (unlike
        # ``audit_write_failures``), and the MCP server holds ONE Store
        # instance for its whole lifetime — the exact "long-running process"
        # this counter was built for. A CLI invocation opens a fresh Store
        # per command, so it would structurally always print 0 there, which
        # reads as "retention is healthy" and is not (the audit field made
        # this same mistake before it was made durable — see
        # tests/test_audit.py::TestDegradedAuditHealthReachesEveryTransport).
        # Full durability, so the CLI can carry this too, was deferred on
        # 2026-09-17.
        if status.prune_failures:
            prune_line = (
                f"⚠ {status.prune_failures} post-commit auto-prune "
                f"failure(s) this session — "
                + (
                    "retention may be behind"
                    if status.prune_behind
                    else "a later prune completed, retention has caught up"
                )
            )
            if status.prune_last_failure:
                prune_line += f", last: {status.prune_last_failure}"
            lines.append(prune_line)

        return _tool_result("\n".join(lines))

    # -- Crystallized-pattern tools (AM-CRYSTAL — the on-demand graduated tier) --

    def _tool_crystal_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP transport adapter for on-demand crystallized-pattern recall (AM-CRYSTAL).

        The crystallized tier's READ surface for MCP-in-conversation adopters — the
        parity of the CLI ``crystal recall`` and of the per-turn recall hook a harness
        fires. Associative by DEFAULT (AM-CRYSTAL-RECALL, 0.8.0; the evidence edge, see
        ``retrieval.py``): a pattern
        grounded in an episode the query matched surfaces even with ZERO query-keyword
        overlap (the keyword-orthogonal miss keyword-only recall cannot reach). It
        reuses the server's already-open episodic ``self._store`` for the seed
        episodes (a read-only ACCESS PATTERN over the server's normal read-write handle —
        ``retrieve_relevant`` only READS the store; the store itself is not opened
        read-only). Unlike the CLI (a cold subprocess that may have no db, hence its
        separate ``Store(read_only=True)`` open) the MCP server is the single process
        holding the handle, so there is no second-handle contention to avoid and no
        extra open. ``associative=false`` forces the keyword-only path (the pre-0.8.0
        behavior).

        Precision-biased: a thin query (< 2 distinctive keywords) or nothing clearing
        the threshold returns no patterns, by design — surface nothing rather than
        noise. Fail-CLOSED on a corrupt CRYSTAL store (an error result ⇒ the caller
        treats it as no recall); a faulting EPISODIC query degrades QUIETLY to
        keyword-only (logged breadcrumb on stderr, the JSON-RPC stdout channel and the
        result contract stay pristine) — the associative tier is best-effort
        augmentation, never a hard requirement.
        """
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            # require a non-empty STRING — a non-str (e.g. a JSON number) would slip
            # past a bare `not query` truthy check into extract_keywords / the backend
            return _tool_result(
                "Error: query must be a non-empty string", is_error=True
            )
        query = query.strip()
        max_patterns = args.get("max_patterns", MAX_PATTERNS)
        if not isinstance(max_patterns, int) or isinstance(max_patterns, bool):
            return _tool_result(
                "Error: max_patterns must be an integer", is_error=True
            )
        # Reject a non-bool ``associative`` rather than coercing it (a JSON-stringy
        # "false" is truthy → would silently STAY associative, the opposite of intent).
        # Mirrors the strict bool handling on the sibling MCP flags (save_continuity's
        # allow_shrink, spore_surface's top_of_mind).
        associative = args.get("associative", True)
        if not isinstance(associative, bool):
            return _tool_result(
                "Error: associative must be a boolean", is_error=True
            )

        raw_mode = args.get("mode", "prompt")
        if not isinstance(raw_mode, str) or raw_mode not in RETRIEVAL_MODES:
            return _tool_result(
                f"Error: mode must be one of {list(RETRIEVAL_MODES)}", is_error=True
            )
        mode: RetrievalMode = "query" if raw_mode == "query" else "prompt"

        try:
            crystal_store = CrystalStore(self._crystal_path)
            if max_patterns <= 0:
                # Parity with retrieve_patterns' own short-circuit — "no patterns"
                # without touching the episodic store, so a no-op recall can't emit a
                # spurious degrade breadcrumb.
                patterns = []
            elif not associative:
                patterns = retrieve_patterns(
                    crystal_store, query, max_patterns=max_patterns, mode=mode
                )
            else:
                patterns = self._crystal_recall_associative(
                    crystal_store, query, max_patterns, mode
                )
        except (CrystalError, OSError) as exc:
            # Fail-CLOSED on the crystal store (corruption / unreadable file): the
            # crystal tier is the primary, so a fault is an error result, not a
            # silent empty. An EPISODIC fault never lands here — it degrades inside
            # _crystal_recall_associative (which catches StoreError). The OSError arm
            # is defensive: CrystalStore._load already normalizes OSError -> CrystalError,
            # so in practice only CrystalError arrives, but the broader catch keeps CLI
            # parity (cli.cmd_crystal_recall) and is belt-and-suspenders at the boundary.
            return _tool_result(f"Error: {exc}", is_error=True)

        # The durable facts the query cues come first, ahead of the patterns (a no-op
        # call, max_patterns <= 0, asks for nothing, so it gets nothing here either).
        facts_block = (
            _durable_block(self._cued_facts(query, mode)) if max_patterns > 0 else ""
        )
        if not patterns:
            # Disambiguate the retry signal for an LLM consumer: a thin query (the
            # library floors recall at MIN_KEYWORDS distinctive keywords) is fixable by
            # rephrasing; a genuine miss is not. Only when we actually attempted recall
            # (max_patterns > 0) — a capped-out call isn't a "thin query".
            floor = QUERY_MIN_KEYWORDS if mode == "query" else MIN_KEYWORDS
            miss = "No crystallized patterns matched."
            if max_patterns > 0 and len(extract_keywords(query, mode=mode)) < floor:
                miss = (
                    "No crystallized patterns matched (query too thin — give it at "
                    f"least {floor} distinctive keyword{'' if floor == 1 else 's'}, "
                    "or check crystal_index "
                    "for what exists)."
                )
            # The miss line stays, after the facts block, so the caller knows no pattern
            # matched.
            return _tool_result(facts_block + "\n\n" + miss if facts_block else miss)
        lines = [f"Found {len(patterns)} crystallized pattern(s):"]
        for p in patterns:
            tag_info = f" [{', '.join(p.tags)}]" if p.tags else ""
            lines.append(
                f"- {p.name} ({p.level}x, {p.activation}, score={p.score:.1f})"
                f"{tag_info}: {p.explanation}"
            )
        if facts_block:
            lines = [facts_block, "", *lines]
        return _tool_result("\n".join(lines))

    def _crystal_recall_associative(
        self,
        crystal_store: CrystalStore,
        query: str,
        max_patterns: int,
        mode: RetrievalMode = "prompt",
    ) -> list[RelevantPattern]:
        """Associative crystal recall (the evidence edge) over the server's OPEN episodic store,
        degrading to keyword-only when an episodic query faults.

        Mirrors the CLI ``_crystal_recall_associative`` but reuses ``self._store``
        (already open, read-only usage) instead of opening a read-only subprocess
        ``Store``. Routes to ``retrieve_relevant`` with ``max_episodes=0`` (patterns
        only). A crystal fault (:class:`CrystalError` / file ``OSError``) raised inside
        ``retrieve_relevant`` is NOT caught here — it propagates to the caller's
        fail-closed handler. Only an episodic :class:`StoreError` degrades: the
        seed episodes are then unavailable, so fall back to keyword-only
        :func:`retrieve_patterns` and leave a breadcrumb so a genuinely broken backend
        isn't INVISIBLE (the operator who expected the associative cure but silently
        got keyword-only forever = the invisible_infrastructure_failure shape)."""
        try:
            return retrieve_relevant(
                self._store,
                crystal_store,
                query,
                max_patterns=max_patterns,
                max_episodes=0,
                associative=True,
                mode=mode,
                durable=False,
            ).patterns
        except StoreError as exc:
            logger.warning(
                "episodic store unavailable (%s: %s); "
                "crystal recall degraded to keyword-only.",
                type(exc).__name__,
                exc,
            )
            return retrieve_patterns(
                crystal_store, query, max_patterns=max_patterns, mode=mode
            )

    def _tool_crystal_index(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP transport adapter for the always-on crystallized INDEX (AM-CRYSTAL-INDEX).

        A name + one-clause menu of the LIVE crystal corpus so an MCP adopter isn't
        blind to its own crystallized wisdom; the bodies fill on cue via
        ``crystal_recall``. Deliberately THIN (name + clause ONLY — no level /
        activation / id): the menu is meant to be always-loaded, so inflating it
        re-creates the attention-doesn't-scale disease the crystal tier exists to cure.
        Sorted by name for a stable, churn-free artifact. Fail-CLOSED on a corrupt
        crystal store (error result)."""
        try:
            crystal_store = CrystalStore(self._crystal_path)
            items = sorted(
                crystal_store.active(), key=lambda c: str(c.get("name", ""))
            )
        except (CrystalError, OSError) as exc:
            return _tool_result(f"Error: {exc}", is_error=True)
        rows = [
            (
                str(c.get("name", "")),
                _truncate(str(c.get("explanation", "")), 100),
            )
            for c in items
        ]
        if not rows:
            return _tool_result("No crystallized patterns.")
        return _tool_result(
            "\n".join(f"{name}: {clause}" for name, clause in rows)
        )

    # -- Spore tools (prospective-intention layer) --
    # SporeError / ValueError raised by the store propagate to
    # _handle_tools_call, which converts them to an is_error tool result.

    def _tool_spore_add(self, args: dict[str, Any]) -> dict[str, Any]:
        # Pass salience through RAW (no int() coercion) so the library validates
        # type+range — coercing here would silently turn 2.9 into 2, "" into 0,
        # etc., diverging from the declared integer/0-3 schema.
        item = self._spore_store.add(
            type=args.get("type", ""),
            text=args.get("text", ""),
            domain=args.get("domain", "") or "",
            tier=args.get("tier", "warm"),
            salience=args.get("salience", 0),
            next=args.get("next"),
            pointer=args.get("pointer"),
        )
        return _tool_result(
            f"Planted {item['id']} ({item['type']}/{item['tier']}): {item['text']}"
        )

    def _tool_spore_get(self, args: dict[str, Any]) -> dict[str, Any]:
        spore_id = (args.get("spore_id") or "").strip()
        if not spore_id:
            return _tool_result("Error: spore_id is required", is_error=True)
        item = self._spore_store.get(spore_id)
        if item is None:
            return _tool_result(f"Spore {spore_id} not found.", is_error=True)
        # Annotate computed germination — the tool description promises it, and
        # list_open's silent equality filter doesn't add it to the stored row.
        row = dict(item)
        row["germination"] = germination_tier(item)
        return _tool_result(json.dumps(row, indent=2, ensure_ascii=False))

    def _tool_spore_list(self, args: dict[str, Any]) -> dict[str, Any]:
        # The library equality-filters without validating, so an invalid enum
        # (e.g. type='tasks') would silently return "no matches". Validate against
        # the declared schema enums and fail loudly, mirroring the CLI's choices=.
        for field, valid in (
            ("type", VALID_TYPES),
            ("tier", VALID_TIERS),
            ("germination", VALID_GERMINATIONS),
        ):
            v = args.get(field)
            if v is not None and v not in valid:
                return _tool_result(
                    f"Error: invalid {field} {v!r}; valid: {list(valid)}", is_error=True
                )
        items = self._spore_store.list_open(
            type=args.get("type"),
            tier=args.get("tier"),
            domain=args.get("domain"),
            germination=args.get("germination"),
        )
        if not items:
            return _tool_result("No open spores match.")
        lines = [
            f"[{s['id']}] {s.get('type')}/{s.get('tier')}/{germination_tier(s)} "
            f"salience={s.get('salience')}: {s.get('text', '')}"
            for s in items
        ]
        return _tool_result("\n".join(lines))

    def _tool_spore_touch(self, args: dict[str, Any]) -> dict[str, Any]:
        spore_id = (args.get("spore_id") or "").strip()
        if not spore_id:
            return _tool_result("Error: spore_id is required", is_error=True)
        item = self._spore_store.touch(spore_id)
        return _tool_result(
            f"Touched {item['id']} (seen -> {item['seen']}, {germination_tier(item)})"
        )

    def _tool_spore_update(self, args: dict[str, Any]) -> dict[str, Any]:
        spore_id = (args.get("spore_id") or "").strip()
        if not spore_id:
            return _tool_result("Error: spore_id is required", is_error=True)
        # Pass only the keys the caller actually sent — presence in `args`
        # distinguishes "omitted (leave)" from "empty string (clear)".
        kwargs: dict[str, Any] = {}
        for field in ("tier", "next", "text", "salience", "domain", "pointer"):
            if field in args:
                kwargs[field] = args[field]
        if args.get("add_note"):
            kwargs["add_note"] = args["add_note"]
        if not kwargs:
            return _tool_result("Error: no fields to update", is_error=True)
        item = self._spore_store.update(spore_id, **kwargs)
        return _tool_result(f"Updated {item['id']}")

    def _tool_spore_descend(self, args: dict[str, Any]) -> dict[str, Any]:
        spore_id = (args.get("spore_id") or "").strip()
        if not spore_id:
            return _tool_result("Error: spore_id is required", is_error=True)
        kind = args.get("kind", "")
        item = self._spore_store.descend(spore_id, kind=kind)
        return _tool_result(f"Descended {item['id']} ({kind}) -> resolved down.")

    def _tool_spore_ascend(self, args: dict[str, Any]) -> dict[str, Any]:
        spore_id = (args.get("spore_id") or "").strip()
        if not spore_id:
            return _tool_result("Error: spore_id is required", is_error=True)
        kind = args.get("kind", "")
        ref = args.get("ref", "")
        item = self._spore_store.ascend(spore_id, kind=kind, ref=ref)
        return _tool_result(f"Ascended {item['id']} -> {kind}: {ref}")

    def _tool_spore_surface(self, args: dict[str, Any]) -> dict[str, Any]:
        # `is True` (not bool()) so a non-boolean truthy value like the string
        # "false" can't flip on the ToM subset — matches the save_continuity
        # allow_shrink convention and fails safe to "all open spores".
        items = self._spore_store.surface(top_of_mind=args.get("top_of_mind") is True)
        if not items:
            return _tool_result("(no open spores)")
        lines = [
            f"({s.get('type')}/{s.get('tier')}/{germination_tier(s)}) {s.get('text', '')}"
            for s in items
        ]
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
    # Force UTF-8 on stdio — locale encoding can corrupt non-ASCII memories.
    # stdlib typeshed types sys.stdin/stdout as TextIO (no .reconfigure) but
    # at runtime they are io.TextIOWrapper which does expose the method.
    sys.stdin.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

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
        from .audit import AuditTrail as _AT, set_aside_report_lines
        result = _AT.verify(args.db)
        # The CLI's gap lines, from the same function (L2 10-03: this surface
        # printed "valid" over a set-aside week and never named it).
        for line in set_aside_report_lines(result.set_aside, args.db):
            print(f"  {line}", file=sys.stderr)
        if result.valid:
            anchor_note = "" if result.anchor_trusted else (
                " (chain anchor recovered by audit-repair; entries before it "
                "cannot be verified)"
            )
            print(
                f"Audit trail valid: {result.total_entries} entries "
                f"across {result.files_verified} file(s){anchor_note}",
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
