"""Remote read-only MCP over Streamable HTTP (anneal_memory.remote), driven over a
real loopback socket."""

from __future__ import annotations

import json
import logging
import socket
import threading
from datetime import date
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from anneal_memory import remote
from anneal_memory.crystal import CrystalStore
from anneal_memory.store import Store


@pytest.fixture
def served(tmp_path: Path):
    db = tmp_path / "memory.db"
    with Store(db) as s:
        s.record("the replica answers from argushub over funnel", "observation")
        s.save_continuity("# Agent — Memory (v1)\n\n## State\nremote test state\n")
    CrystalStore(tmp_path / "memory.crystal.json").crystallize(
        name="funnel_pattern", level=2, explanation="the replica answers over funnel",
        today=date(2026, 1, 1))
    tokens = tmp_path / "tokens"
    a = remote.token_add(tokens, "claude-ai")
    b = remote.token_add(tokens, "other")
    httpd = remote.make_server(db, tokens, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path: str, body=None, auth: str | None = None, origin: str | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=data, method="POST" if data else "GET")
        req.add_header("Content-Type", "application/json")
        if auth:
            req.add_header("Authorization", auth)
        if origin:
            req.add_header("Origin", origin)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return r.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            return e.code, None

    try:
        call.port = httpd.server_address[1]  # type: ignore[attr-defined]
        yield call, a, b, tokens, tmp_path
    finally:
        httpd.shutdown()
        httpd.server_close()


_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


def _tool(name: str, args: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": name, "arguments": args or {}}}


def test_refused_without_a_valid_token(served):
    call, a, _b, _t, _p = served
    assert call("/mcp")[0] == 401
    assert call("/mcp", _LIST)[0] == 401
    assert call("/mcp", _LIST, "Bearer wrong")[0] == 401
    assert call("/mcp", _LIST, a)[0] == 401  # the token without its scheme
    assert call("/mcp/wrong", _LIST)[0] == 401
    assert call(f"/mcp/{a}/x", _LIST)[0] == 404
    assert call("/other", _LIST, f"Bearer {a}")[0] == 404
    assert call("/mcp", _LIST, f"Bearer {a}")[0] == 200
    assert call(f"/mcp/{a}", _LIST)[0] == 200
    assert call(f"/mcp/{a}", _LIST, origin="https://evil.example")[0] == 403
    assert call(f"/mcp/{a}", _LIST, origin="https://claude.ai")[0] == 200
    # A refused request's unread body must not be parsed as a second request on a
    # kept-alive socket: exactly one response, then the server closes.
    inner = (f"POST /mcp/{a} HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{{}}").encode()
    raw = (b"POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Length: " + str(len(inner)).encode()
           + b"\r\n\r\n" + inner)
    with socket.create_connection(("127.0.0.1", call.port), timeout=10) as sock:
        sock.sendall(raw)
        got = b""
        while chunk := sock.recv(65536):
            got += chunk
    assert got.count(b"HTTP/1.") == 1 and got.startswith(b"HTTP/1.1 401")
    assert b"Connection: close" in got


def test_no_write_or_consolidate_tool_is_reachable(served):
    call, a, _b, _t, _p = served
    _s, r = call(f"/mcp/{a}", _LIST)
    assert {t["name"] for t in r["result"]["tools"]} == set(remote.REMOTE_TOOLS)
    for name in ("record", "prepare_wrap", "save_continuity", "wrap_cancel",
                 "delete_episode", "spore_add", "spore_update", "spore_touch", "status"):
        _s, r = call(f"/mcp/{a}", _tool(name, {"content": "x", "episode_type": "observation"}))
        assert r["result"]["isError"] is True
        assert r["result"]["content"][0]["text"] == f"Unknown tool: {name}"
    _s, r = call(f"/mcp/{a}", {"jsonrpc": "2.0", "id": 3, "method": "resources/read",
                               "params": {"uri": "anneal://continuity"}})
    assert r["error"]["code"] == -32601


def test_recall_round_trip_writes_nothing(served):
    call, a, _b, _t, path = served
    before = {p.name: p.stat().st_mtime_ns for p in path.iterdir()}
    _s, r = call("/mcp", _tool("recall", {"keyword": "funnel"}), f"Bearer {a}")
    assert "the replica answers from argushub over funnel" in r["result"]["content"][0]["text"]
    _s, r = call("/mcp", _tool("read_continuity"), f"Bearer {a}")
    assert "remote test state" in r["result"]["content"][0]["text"]
    _s, r = call("/mcp", _tool("crystal_index"), f"Bearer {a}")
    assert "funnel_pattern" in r["result"]["content"][0]["text"]
    _s, r = call("/mcp", _tool("crystal_recall", {"query": "replica funnel"}), f"Bearer {a}")
    assert r["result"].get("isError") is not True
    after = {p.name: p.stat().st_mtime_ns for p in path.iterdir()}
    assert after == before


def test_tokens_are_per_connector_and_never_stored(served, caplog):
    call, a, b, tokens, _p = served
    caplog.set_level(logging.DEBUG)
    call(f"/mcp/{a}", _tool("recall", {"keyword": "funnel"}))
    call(f"/mcp/{a}x", _LIST)
    assert a not in caplog.text and "/mcp/" not in caplog.text
    text = tokens.read_text()
    assert a not in text and b not in text
    remote.token_revoke(tokens, "other")
    assert call(f"/mcp/{b}", _LIST)[0] == 401
    assert call(f"/mcp/{a}", _LIST)[0] == 200


def test_head_and_options_answer_behind_auth(served):
    call, a, _b, _t, _p = served
    for method, ok in (("HEAD", 200), ("OPTIONS", 204)):
        def go(path, auth=None):
            req = urllib.request.Request(f"http://127.0.0.1:{call.port}{path}", method=method)
            if auth:
                req.add_header("Authorization", auth)
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, b""
        assert go("/mcp")[0] == 401
        assert go("/mcp/wrong")[0] == 401
        assert go("/elsewhere")[0] == 404
        assert go(f"/mcp/{a}") == (ok, b"")
        assert go("/mcp", f"Bearer {a}") == (ok, b"")
    # On the wire: a refused HEAD sends headers only; a 204 has no Content-Length.
    for request, status in ((b"HEAD /mcp HTTP/1.1", b"401"), (f"OPTIONS /mcp/{a} HTTP/1.1".encode(), b"204")):
        with socket.create_connection(("127.0.0.1", call.port), timeout=10) as sock:
            sock.sendall(request + b"\r\nHost: x\r\n\r\n")
            got = b""
            while chunk := sock.recv(65536):
                got += chunk
        head, _, rest = got.partition(b"\r\n\r\n")
        assert head.split()[1] == status and rest == b""
        assert (b"Content-Length" in head) is (status != b"204")
