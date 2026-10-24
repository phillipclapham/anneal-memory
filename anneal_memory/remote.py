"""Remote MCP over Streamable HTTP that never writes the store (experimental;
not part of the frozen public surface in docs/stability.md).

Serves a store to a remote MCP client (a claude.ai custom connector) over the
Streamable HTTP transport, JSON responses only, no SSE stream, no session id.
Stdlib only, like the stdio server.

What keeps the store read-only is structure, not a check:

- The store is opened per request with ``Store(read_only=True)``: the
  connection runs ``PRAGMA query_only=ON`` and the open writes nothing.
- The only store tools a request can reach are the names in :data:`REMOTE_TOOLS`,
  bound one by one below to read methods of :class:`~anneal_memory.server.Server`.
  No store write or consolidate handler is referenced from this module, so no
  argument, method name or path can route to one.
- ``resources/*`` is not offered.

The one exception to "reads only" is ``capture``, and only when ``serve
--outbox PATH`` is given: it appends one JSON line per episode to that outbox
file (and its ``.lock``), separate files the store never reads, and it does not
open the store at all. The operator's own process moves outbox lines into
memory later, so the store keeps one writer. Without ``--outbox`` the tool is
not listed and calling it answers "Unknown tool".

Known edges of ``--outbox`` (the operator's own configuration reaches them;
nothing a client sends does): a hardlink to the db placed outside the db's
directory, or a db path whose last component is a symlink, is not detected as
an alias; a FIFO at the outbox path blocks capture; a ``close()`` failing after
a successful fsync reports an error for a line that was kept. On Windows there
is no cross-process outbox lock.

Authentication is a per-connector bearer token, accepted in either of two
places, because claude.ai's "Request headers" field was a beta limited to some
organizations when this was written (claude.com/docs/connectors/custom/remote-mcp,
read 2026-10-24):

- ``Authorization: Bearer <token>`` on ``POST /mcp``, or
- the token as the last path segment, ``POST /mcp/<token>``, for a connector
  added with "No sign-in" (the URL is then the secret).

The token file holds ``name:sha256(token)`` lines; tokens themselves are never
stored, logged or echoed. Each connector gets its own line, so one can be
revoked without touching the others.

Usage::

    python -m anneal_memory.remote token-add claude-ai --token-file tokens
    python -m anneal_memory.remote token-revoke claude-ai --token-file tokens
    python -m anneal_memory.remote serve --db memory.db --token-file tokens \\
        [--host 127.0.0.1] [--port 8765] [--outbox outbox.jsonl]

Bind to loopback and put a TLS-terminating proxy in front (Tailscale Funnel,
``tailscale funnel --bg 8765``); this server speaks plain HTTP.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
import tempfile
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

from . import __version__
from ._fsdir import _fsync_dir
from .origin import strip_hidden_controls
from .server import Server, _tool_result
from .store import Store
from .types import EpisodeType

logger = logging.getLogger("anneal-memory.remote")

# Newest first. A client asking for one of these gets it back; anything else
# gets the newest (MCP lifecycle: the server answers with a version it supports).
# Listed because the initialize/tools shapes this server sends do not differ
# between them [judged by 1024+9 anneal seat, 2026-10-24, against the spec
# changelogs; tests/test_remote.py drives 2025-06-18 only]. 2025-03-26 is left
# out: it still allowed JSON-RPC batches, which this server refuses.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18")

MAX_BODY_BYTES = 1024 * 1024
MAX_CONCURRENT = 8
MAX_CONNECTIONS = 32
MAX_REPLY_BYTES = 4 * 1024 * 1024
MAX_RECALL_LIMIT = 200

MAX_CAPTURE_CHARS = 4000
MAX_OUTBOX_BYTES = 8 * 1024 * 1024
CAPTURES_PER_HOUR = 60
OUTBOX_LOCK_WAIT_S = 2.0


class _OutboxBusy(Exception):
    """The outbox lock was not free in time (a puller is moving the file)."""

# Browsers send Origin; claude.ai's backend calls are expected not to. A present
# Origin outside this set is refused (Streamable HTTP: 403 on an invalid Origin).
ALLOWED_ORIGINS = frozenset({"https://claude.ai", "https://claude.com"})
REQUEST_TIMEOUT_S = 15
# The methods a request can succeed with; GET and DELETE answer 405 (no SSE stream, no sessions).
ALLOWED_METHODS = "POST, HEAD, OPTIONS"

_TOKEN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

READ_CONTINUITY_TOOL: dict[str, Any] = {
    "name": "read_continuity",
    "description": (
        "Read the current continuity file: the agent's compressed long-term "
        "memory (state, active threads, patterns, decisions, context, "
        "understanding). Read-only."
    ),
    "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
}


# MCP tool annotations (spec 2025-03-26+): every tool in REMOTE_TOOLS only reads. Clients that
# gate writes on them (ChatGPT confirms any tool without readOnlyHint) then call
# these without a prompt. Hints are advisory to the client; the allowlist above and
# the query-only open are what enforce read-only here.
_READ_ONLY_ANNOTATIONS = {"readOnlyHint": True, "destructiveHint": False,
                          "idempotentHint": True, "openWorldHint": False}


def _remote_tool_defs() -> list[dict[str, Any]]:
    """The tool definitions served remotely: the stdio definitions of the
    allowlisted read tools plus ``read_continuity``, each a copy carrying
    the read-only annotations."""
    from .integrity import TOOLS

    by_name = {t["name"]: t for t in TOOLS}
    defs = [by_name[name] for name in _SERVER_READ_TOOLS] + [READ_CONTINUITY_TOOL]
    return [{**d, "annotations": {**d.get("annotations", {}), **_READ_ONLY_ANNOTATIONS}}
            for d in defs]


CAPTURE_TOOL: dict[str, Any] = {
    "name": "capture",
    "description": (
        "Save one new episode to memory: something the user wants remembered "
        "(an observation, a decision, an outcome). Append-only: it cannot edit "
        "or delete anything. It reaches memory at the next sync, not at once, "
        "so do not expect recall to find it in this conversation."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "minLength": 1, "maxLength": MAX_CAPTURE_CHARS,
                        "description": "The episode text, self-contained."},
            "episode_type": {"type": "string", "enum": [t.value for t in EpisodeType]},
        },
        "required": ["content", "episode_type"],
        "additionalProperties": False,
    },
    # A write: clients that gate on readOnlyHint (ChatGPT) ask the user first.
    "annotations": {"readOnlyHint": False, "destructiveHint": False,
                    "idempotentHint": False, "openWorldHint": False},
}

CAPTURED_REPLY = ("Captured. It reaches the memory at the next sync; it is not "
                  "recallable before then.")


def line_for(connector: str, episode_type: str, content: str) -> str:
    """One outbox line: the format a puller re-validates key by key."""
    return json.dumps({
        "v": 1, "id": secrets.token_hex(16), "connector": connector,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episode_type": episode_type, "content": content,
    }, ensure_ascii=True) + "\n"


# The stdio tools exposed remotely. Each name maps to the Server method that
# implements it in :meth:`RemoteSession._bind`; the only other tool is ``capture``
# (outbox only, never the store), and only with ``--outbox``.
_SERVER_READ_TOOLS = ("recall", "crystal_recall", "crystal_index")
REMOTE_TOOLS = (*_SERVER_READ_TOOLS, "read_continuity")


# -- Tokens --


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def load_tokens(path: str | Path) -> dict[str, str]:
    """``{name: sha256hex}`` from a token file. Malformed lines are refused
    loudly rather than skipped: a skipped line is a token that silently stops
    working, or one that silently keeps working after a bad edit."""
    tokens: dict[str, str] = {}
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, digest = line.partition(":")
        if not sep or not _TOKEN_NAME_RE.match(name) or not _SHA256_HEX_RE.match(digest):
            raise ValueError(f"{path}:{lineno}: expected name:sha256hex")
        if name in tokens:
            raise ValueError(f"{path}:{lineno}: duplicate token name {name!r}")
        tokens[name] = digest
    return tokens


def match_token(tokens: dict[str, str], presented: str | None) -> str | None:
    """The connector name whose token this is, or None. Every stored digest is
    compared, in constant time each, so the answer's timing does not depend on
    which line (if any) matched."""
    if not presented:
        return None
    digest = _hash_token(presented)
    found = None
    for name, stored in tokens.items():
        if hmac.compare_digest(digest, stored):
            found = name
    return found


@contextlib.contextmanager
def _token_file_locked(path: Path) -> Iterator[dict[str, str]]:
    """Load, let the caller mutate, and durably rewrite the token file, all under
    an exclusive lock on a sibling lock file (L3 r1 codex: two unlocked edits
    could resurrect a revoked token). Comment lines are not preserved. Where
    fcntl does not exist (Windows) the edit runs unlocked: single writer only."""
    try:
        import fcntl
    except ImportError:
        fcntl = None  # type: ignore[assignment]
    lock_fd = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tokens = load_tokens(path) if path.exists() else {}
        yield tokens
        body = "".join(f"{n}:{d}\n" for n, d in tokens.items())
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        _fsync_dir(path.parent)
    finally:
        os.close(lock_fd)


def token_add(path: str | Path, name: str) -> str:
    """Create a token for ``name``, store its hash, return the token (shown once)."""
    if not _TOKEN_NAME_RE.match(name):
        raise ValueError("token name: letters, digits, '_', '.', '-'; max 64")
    with _token_file_locked(Path(path)) as tokens:
        if name in tokens:
            raise ValueError(f"token {name!r} exists; revoke it first")
        token = secrets.token_urlsafe(32)
        tokens[name] = _hash_token(token)
    return token


def token_revoke(path: str | Path, name: str) -> None:
    with _token_file_locked(Path(path)) as tokens:
        if name not in tokens:
            raise ValueError(f"no token named {name!r}")
        del tokens[name]


# -- JSON-RPC over one request --


def _rpc_result(msg_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _rpc_error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


class RemoteSession:
    """Answers one JSON-RPC message against a read-only open of ``db_path``.

    A fresh read-only Store per message: no connection outlives a request, so
    a replica swapped underneath is picked up by the next call, and threads
    never share a connection (Store is single-thread)."""

    def __init__(self, db_path: str | Path, outbox: str | Path | None = None) -> None:
        self._db_path = Path(db_path)
        self._outbox = Path(outbox) if outbox is not None else None
        self._rate_lock = threading.Lock()
        self._append_lock = threading.Lock()
        self._recent: dict[str, list[float]] = {}

    def _tools(self) -> tuple[str, ...]:
        return (*REMOTE_TOOLS, "capture") if self._outbox is not None else REMOTE_TOOLS

    def _capture(self, connector: str | None, arguments: dict[str, Any]) -> dict[str, Any]:
        """Append one episode line to the outbox. Never touches the store."""
        assert self._outbox is not None
        if connector is None:
            return _tool_result("Error: capture needs an authenticated connector", is_error=True)
        if set(arguments) != {"content", "episode_type"}:
            return _tool_result("Error: arguments are exactly content and episode_type",
                                is_error=True)
        content, etype = arguments["content"], arguments["episode_type"]
        if not isinstance(etype, str) or etype not in {t.value for t in EpisodeType}:
            return _tool_result("Error: unknown episode_type", is_error=True)
        if not isinstance(content, str):
            return _tool_result("Error: content must be a string", is_error=True)
        # Also the invisible format and separator characters (zero-width, BOM,
        # U+2028/2029) the shared filter keeps: they dodge keyword filters and
        # can render as line breaks downstream (L2).
        content = "".join(ch for ch in strip_hidden_controls(content)
                          if unicodedata.category(ch) not in ("Cf", "Zl", "Zp")).strip()
        if not 1 <= len(content) <= MAX_CAPTURE_CHARS:
            return _tool_result(f"Error: content must be 1..{MAX_CAPTURE_CHARS} characters",
                                is_error=True)
        now = time.monotonic()
        with self._rate_lock:
            recent = [t for t in self._recent.get(connector, []) if now - t < 3600]
            if len(recent) >= CAPTURES_PER_HOUR:
                self._recent[connector] = recent
                return _tool_result("Error: capture rate limit reached; try later", is_error=True)
            recent.append(now)
            self._recent[connector] = recent
        written = False
        try:
            written = self._append(line_for(connector, etype, content))
        except _OutboxBusy:
            return _tool_result("Error: capture is busy (a sync is running); try again",
                                is_error=True)
        finally:
            if not written:  # a refused or failed write gives its slot back
                with self._rate_lock:
                    slots = self._recent.get(connector, [])
                    if now in slots:
                        slots.remove(now)
        if not written:
            return _tool_result("Error: the capture outbox is full until the next sync",
                                is_error=True)
        return _tool_result(CAPTURED_REPLY)

    def _append(self, line: str) -> bool:
        """Append ``line`` whole or not at all; False when the outbox is full.
        Raises :class:`_OutboxBusy` when the lock is not free within OUTBOX_LOCK_WAIT_S."""
        assert self._outbox is not None
        data = line.encode("ascii")
        try:
            import fcntl
        except ImportError:
            fcntl = None  # type: ignore[assignment]
        # The thread lock serializes this process even where fcntl is missing; the
        # flock is the contract for a puller: rename the outbox only while holding
        # it. The append opens by path under the lock, so it lands wholly before or after.
        # One deadline for both locks, so requests queued behind a held outbox lock
        # cannot convoy past it and keep every request slot (L3 r2, both seats).
        deadline = time.monotonic() + OUTBOX_LOCK_WAIT_S
        if not self._append_lock.acquire(timeout=OUTBOX_LOCK_WAIT_S):
            raise _OutboxBusy()
        try:
            lock_fd = os.open(self._outbox.with_name(self._outbox.name + ".lock"),
                              os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if fcntl is not None:
                    while True:
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise _OutboxBusy() from None
                            time.sleep(0.05)
                flags = (os.O_WRONLY | os.O_APPEND | os.O_CREAT
                         | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
                fd = os.open(self._outbox, flags, 0o600)
                try:
                    size = os.fstat(fd).st_size
                    if size + len(data) > MAX_OUTBOX_BYTES:
                        return False
                    try:
                        view = memoryview(data)
                        while view:
                            n = os.write(fd, view)
                            if n <= 0:
                                raise OSError("outbox write made no progress")
                            view = view[n:]
                        os.fsync(fd)
                    except BaseException:
                        # A torn or undurable line is cut back off, so a refusal
                        # never leaves a fragment or a line the client will retry.
                        with contextlib.suppress(OSError):
                            os.ftruncate(fd, size)
                            os.fsync(fd)
                        raise
                finally:
                    os.close(fd)
                if size == 0:  # possibly a new file: make its directory entry durable
                    _fsync_dir(self._outbox.parent)
            finally:
                os.close(lock_fd)
        finally:
            self._append_lock.release()
        return True

    def _bind(self, server: Server, store: Store) -> dict[str, Callable[[dict[str, Any]], Any]]:
        def read_continuity(_args: dict[str, Any]) -> dict[str, Any]:
            text = store.load_continuity()
            return _tool_result(text or "(No continuity file yet.)")

        return {
            "recall": server._tool_recall,
            "crystal_recall": server._tool_crystal_recall,
            "crystal_index": server._tool_crystal_index,
            "read_continuity": read_continuity,
        }

    def handle(self, msg: dict[str, Any], connector: str | None = None) -> dict[str, Any] | None:
        """The response for ``msg``, or None for a notification. ``connector`` is
        the authenticated token name; a capture records it and needs it."""
        method = msg.get("method")
        msg_id = msg.get("id")
        if msg.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return _rpc_error(msg_id if "id" in msg else None, -32600, "Invalid request")
        if "id" not in msg:
            return None
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return _rpc_error(msg_id, -32602, "params must be an object")

        if method == "initialize":
            asked = params.get("protocolVersion")
            version = asked if asked in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
            return _rpc_result(msg_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "anneal-memory-remote", "version": __version__},
                "instructions": (
                    "Read access to an anneal-memory store. read_continuity "
                    "returns the compressed long-term memory; recall searches "
                    "episodes; crystal_recall/crystal_index read crystallized patterns."
                    + (" capture appends a new episode, which reaches memory at the "
                       "next sync." if self._outbox is not None else "")
                ),
            })
        if method == "ping":
            return _rpc_result(msg_id, {})
        if method == "tools/list":
            defs = _remote_tool_defs()
            if self._outbox is not None:
                defs.append(CAPTURE_TOOL)
            return _rpc_result(msg_id, {"tools": defs})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _rpc_error(msg_id, -32602, "arguments must be an object")
            if name not in self._tools():
                return _rpc_result(msg_id, _tool_result(f"Unknown tool: {name}", is_error=True))
            if name == "capture":
                try:
                    return _rpc_result(msg_id, self._capture(connector, arguments))
                except OSError as e:
                    logger.error("capture failed: %s", type(e).__name__)
                    return _rpc_result(msg_id, _tool_result(f"Error: {type(e).__name__}",
                                                            is_error=True))
            if name == "recall":
                # The stdio recall has no ceiling on limit; a remote reply does.
                limit = arguments.get("limit")
                integral = (isinstance(limit, int) and not isinstance(limit, bool)) or (
                    isinstance(limit, float) and limit.is_integer())
                if integral and limit is not None and limit > MAX_RECALL_LIMIT:
                    arguments = {**arguments, "limit": MAX_RECALL_LIMIT}
            try:
                # Resolve the directory once per request, keeping the configured file
                # name: the db and its sibling files (named from that stem) then come from
                # one published generation's directory even if the link is swapped after
                # this line. A generation deleted mid-request fails the call; it never mixes.
                db = self._db_path.parent.resolve() / self._db_path.name
                with Store(db, read_only=True, audit=False) as store:
                    server = Server(store)
                    result = self._bind(server, store)[name](arguments)
            except Exception as e:  # a tool error is a result, not a transport error
                # Class only: the message can carry paths or query text.
                logger.error("tool %s failed: %s", name, type(e).__name__)
                return _rpc_result(msg_id, _tool_result(f"Error: {type(e).__name__}", is_error=True))
            return _rpc_result(msg_id, result)
        return _rpc_error(msg_id, -32601, f"Method not found: {method}")


# -- HTTP --


class _Handler(BaseHTTPRequestHandler):
    server_version = "anneal-memory-remote"
    sys_version = ""
    timeout = REQUEST_TIMEOUT_S
    protocol_version = "HTTP/1.1"

    # Set on the class by make_server.
    tokens_path: Path
    session: RemoteSession
    slots: threading.BoundedSemaphore

    def log_message(self, format: str, *args: Any) -> None:
        # The default line carries the request path, which may be the token.
        pass

    def _send(self, status: int, body: bytes = b"", ctype: str | None = None,
              extra: dict[str, str] | None = None) -> None:
        # One request per connection. A refusal sent before the body is read
        # would otherwise leave that body on a kept-alive socket to be parsed
        # as the next request (L1/L2 1024+9: request smuggling through the
        # proxy's connection pool), and no early return can forget this.
        self.close_connection = True
        if status >= 400:
            logger.info("refused %d %s", status, self.command)
        self.send_response(status)
        self.send_header("Connection", "close")
        if ctype:
            self.send_header("Content-Type", ctype)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if status != 204:  # RFC 9110 §8.6: no Content-Length on a 204
            self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body and self.command != "HEAD":  # a HEAD reply never carries a body, refusals included
            self.wfile.write(body)

    def _send_json(self, status: int, obj: Any) -> None:
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

    def _authorize(self) -> str | None:
        """The connector name, or None after sending a refusal."""
        origin = self.headers.get("Origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            self._send(403)
            return None
        path = self.path.split("?", 1)[0]
        presented: str | None = None
        if path == "/mcp":
            auth = self.headers.get("Authorization", "")
            scheme, _, value = auth.partition(" ")
            if scheme.lower() == "bearer":
                presented = value.strip()
        elif path.startswith("/mcp/") and "/" not in path[5:]:
            presented = path[5:]
        else:
            self._send(404)
            return None
        try:
            tokens = load_tokens(self.tokens_path)
        except (OSError, ValueError):
            logger.exception("token file unreadable; refusing every request")
            self._send(503)
            return None
        name = match_token(tokens, presented)
        if name is None:
            self._send(401, b'{"error":"unauthorized"}', "application/json",
                       {"WWW-Authenticate": 'Bearer realm="anneal-memory"'})
            return None
        return name

    def _read_body(self, length: int) -> bytes | None:
        """Exactly ``length`` bytes within one deadline for the whole body (a
        per-recv timeout alone lets a 1-byte-per-14s sender hold the thread for
        days), or None after a refusal."""
        deadline = time.monotonic() + REQUEST_TIMEOUT_S
        chunks: list[bytes] = []
        got = 0
        try:
            while got < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._send(408)
                    return None
                self.connection.settimeout(remaining)
                chunk = self.rfile.read1(min(65536, length - got))
                if not chunk:
                    self._send_json(400, _rpc_error(None, -32700, "Body shorter than Content-Length"))
                    return None
                chunks.append(chunk)
                got += len(chunk)
        except (TimeoutError, OSError):
            self.close_connection = True
            return None
        self.connection.settimeout(REQUEST_TIMEOUT_S)
        return b"".join(chunks)

    def do_GET(self) -> None:
        self._get()

    def do_DELETE(self) -> None:
        self._get()

    def do_HEAD(self) -> None:
        # The Gemini app probes the server URL with HEAD before anything else;
        # the default 501 read as "server cannot be reached" there (10-24).
        if self._authorize() is None:
            return
        self._send(200, extra={"Allow": ALLOWED_METHODS})

    def do_OPTIONS(self) -> None:
        if self._authorize() is None:
            return
        # Method discovery only; no CORS preflight (every client here calls server-side).
        self._send(204, extra={"Allow": ALLOWED_METHODS})

    def _get(self) -> None:
        if self._authorize() is None:
            return
        # No server-initiated SSE stream and no sessions to end.
        self._send(405, extra={"Allow": ALLOWED_METHODS})

    def do_POST(self) -> None:
        try:
            self._post()
        except Exception as e:  # never a dropped connection with no answer
            logger.error("request failed: %s", type(e).__name__)
            try:
                self._send_json(500, _rpc_error(None, -32603, "Internal error"))
            except OSError:
                pass

    def _post(self) -> None:
        connector = self._authorize()
        if connector is None:
            return
        if self.headers.get("Transfer-Encoding"):
            self._send(411)
            return
        values = self.headers.get_all("Content-Length") or []
        # Digits only, exactly one header: int() also takes "+5", " 5", "1_0".
        if len(values) != 1 or not (values[0].isascii() and values[0].isdigit()):
            self._send(411)
            return
        if len(values[0]) > 8 or int(values[0]) > MAX_BODY_BYTES:
            self._send(413)
            return
        raw = self._read_body(int(values[0]))
        if raw is None:
            return
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            self._send_json(400, _rpc_error(None, -32700, "Parse error"))
            return
        if not isinstance(msg, dict):
            # Batches were removed from the protocol in 2025-06-18.
            self._send_json(400, _rpc_error(None, -32600, "Invalid request"))
            return
        if "method" not in msg:
            if "result" in msg or "error" in msg:
                # A client's response to a server request; this server sends none.
                self._send(202)
            else:
                self._send_json(400, _rpc_error(msg.get("id"), -32600, "Invalid request"))
            return
        version = self.headers.get("MCP-Protocol-Version")
        if (msg.get("method") != "initialize" and version is not None
                and version not in SUPPORTED_PROTOCOL_VERSIONS):
            self._send_json(400, _rpc_error(msg.get("id"), -32600,
                                            "Unsupported MCP-Protocol-Version"))
            return
        if not self.slots.acquire(blocking=False):
            self._send(503, extra={"Retry-After": "2"})
            return
        try:
            reply = self.session.handle(msg, connector)
        finally:
            self.slots.release()
        method = msg.get("method")
        params = msg.get("params")
        tool = params.get("name") if method == "tools/call" and isinstance(params, dict) else None
        logger.info("connector=%s method=%s%s", connector,
                    method if isinstance(method, str) and method.isprintable() else "?",
                    f" tool={tool}" if isinstance(tool, str) and tool in self.session._tools() else "")
        if reply is None:
            self._send(202)
            return
        body = json.dumps(reply).encode("utf-8")
        if len(body) > MAX_REPLY_BYTES:
            reply = _rpc_result(reply.get("id"), _tool_result(
                f"Error: reply over {MAX_REPLY_BYTES} bytes; narrow the query or lower limit",
                is_error=True))
            body = json.dumps(reply).encode("utf-8")
        self._send(200, body, "application/json")


class _BoundedServer(ThreadingHTTPServer):
    """At most MAX_CONNECTIONS handler threads; a connection past that is closed
    at once (L3 r1 codex: MAX_CONCURRENT bounded tool work, not threads). It is
    refused, not queued: a blocking wait would let slow connections stall the
    accept loop for everyone (L3 r2 complement)."""

    daemon_threads = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._connections = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


def make_server(db_path: str | Path, tokens_path: str | Path,
                host: str = "127.0.0.1", port: int = 8765,
                outbox: str | Path | None = None) -> ThreadingHTTPServer:
    """A ready (not yet serving) HTTP server. Fails fast on a bad token file, a
    store that cannot be opened read-only, or an outbox whose directory is missing."""
    load_tokens(tokens_path)
    if outbox is not None:
        out_dir = Path(outbox).parent
        if not out_dir.is_dir():
            raise ValueError(f"outbox directory does not exist: {out_dir}")
        # The store and its sidecars live beside the db; an outbox there could be
        # one of them (or a link to one), and capture must never write the store.
        if out_dir.resolve() == Path(db_path).parent.resolve():
            raise ValueError("the outbox must not be in the store's directory")
    with Store(db_path, read_only=True, audit=False):
        pass
    handler = type("Handler", (_Handler,), {
        "tokens_path": Path(tokens_path),
        "session": RemoteSession(db_path, outbox),
        "slots": threading.BoundedSemaphore(MAX_CONCURRENT),
    })
    return _BoundedServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anneal_memory.remote",
                                     description="Remote MCP over Streamable HTTP: store read-only, optional append-only capture outbox")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--db", required=True)
    s.add_argument("--token-file", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--outbox", default=None,
                   help="enable the append-only capture tool, writing to this file")
    for cmd in ("token-add", "token-revoke"):
        t = sub.add_parser(cmd)
        t.add_argument("name")
        t.add_argument("--token-file", required=True)
    args = parser.parse_args(argv)

    try:
        if args.cmd == "token-add":
            print(token_add(args.token_file, args.name))
            return 0
        if args.cmd == "token-revoke":
            token_revoke(args.token_file, args.name)
            return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    httpd = make_server(args.db, args.token_file, args.host, args.port, args.outbox)
    logger.info("serving on http://%s:%d/mcp (capture %s)", args.host, args.port,
                "on" if args.outbox else "off")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
