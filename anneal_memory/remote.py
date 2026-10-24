"""Remote, read-only MCP over Streamable HTTP (experimental; not part of the
frozen public surface in docs/stability.md).

Serves a store to a remote MCP client (a claude.ai custom connector) over the
Streamable HTTP transport, JSON responses only, no SSE stream, no session id.
Stdlib only, like the stdio server.

What makes it read-only is structure, not a check:

- The store is opened per request with ``Store(read_only=True)``: the
  connection runs ``PRAGMA query_only=ON`` and the open writes nothing.
- The only tools a request can reach are the names in :data:`REMOTE_TOOLS`,
  bound one by one below to read methods of :class:`~anneal_memory.server.Server`.
  No write or consolidate handler is referenced from this module, so no
  argument, method name or path can route to one.
- ``resources/*`` is not offered.

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
        [--host 127.0.0.1] [--port 8765]

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

from . import __version__
from ._fsdir import _fsync_dir
from .server import Server, _tool_result
from .store import Store

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

# Browsers send Origin; claude.ai's backend calls are expected not to. A present
# Origin outside this set is refused (Streamable HTTP: 403 on an invalid Origin).
ALLOWED_ORIGINS = frozenset({"https://claude.ai", "https://claude.com"})
REQUEST_TIMEOUT_S = 15

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


# MCP tool annotations (spec 2025-03-26+): every remote tool only reads. Clients that
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
    return [{**d, "annotations": dict(_READ_ONLY_ANNOTATIONS)} for d in defs]


# The stdio tools exposed remotely. Each name maps to the Server method that
# implements it in :meth:`RemoteSession._bind`; nothing else is reachable.
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

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

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

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """The response for ``msg``, or None for a notification."""
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
                    "Read-only access to an anneal-memory store. read_continuity "
                    "returns the compressed long-term memory; recall searches "
                    "episodes; crystal_recall/crystal_index read crystallized patterns."
                ),
            })
        if method == "ping":
            return _rpc_result(msg_id, {})
        if method == "tools/list":
            return _rpc_result(msg_id, {"tools": _remote_tool_defs()})
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _rpc_error(msg_id, -32602, "arguments must be an object")
            if name not in REMOTE_TOOLS:
                return _rpc_result(msg_id, _tool_result(f"Unknown tool: {name}", is_error=True))
            if name == "recall":
                # The stdio recall has no ceiling on limit; a remote reply does.
                limit = arguments.get("limit")
                integral = (isinstance(limit, int) and not isinstance(limit, bool)) or (
                    isinstance(limit, float) and limit.is_integer())
                if integral and limit is not None and limit > MAX_RECALL_LIMIT:
                    arguments = {**arguments, "limit": MAX_RECALL_LIMIT}
            try:
                # Resolve once per request: a db reached through a swapped symlink
                # (a published generation) and its sibling files then come from one
                # generation, even if the link moves mid-request.
                with Store(self._db_path.resolve(), read_only=True, audit=False) as store:
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
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
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

    def _get(self) -> None:
        if self._authorize() is None:
            return
        # No server-initiated SSE stream and no sessions to end.
        self._send(405, extra={"Allow": "POST"})

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
            reply = self.session.handle(msg)
        finally:
            self.slots.release()
        method = msg.get("method")
        params = msg.get("params")
        tool = params.get("name") if method == "tools/call" and isinstance(params, dict) else None
        logger.info("connector=%s method=%s%s", connector,
                    method if isinstance(method, str) and method.isprintable() else "?",
                    f" tool={tool}" if isinstance(tool, str) and tool in REMOTE_TOOLS else "")
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
                host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """A ready (not yet serving) HTTP server. Fails fast on a bad token file or
    a store that cannot be opened read-only."""
    load_tokens(tokens_path)
    with Store(db_path, read_only=True, audit=False):
        pass
    handler = type("Handler", (_Handler,), {
        "tokens_path": Path(tokens_path),
        "session": RemoteSession(db_path),
        "slots": threading.BoundedSemaphore(MAX_CONCURRENT),
    })
    return _BoundedServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m anneal_memory.remote",
                                     description="Read-only remote MCP over Streamable HTTP")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--db", required=True)
    s.add_argument("--token-file", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
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
    httpd = make_server(args.db, args.token_file, args.host, args.port)
    logger.info("serving read-only on http://%s:%d/mcp", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
