"""The one-user OAuth front of the remote server (anneal_memory.remote_oauth), driven
over a real loopback socket the way the Gemini app's custom-app flow runs."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from anneal_memory import remote
from anneal_memory.store import Store

ISS = "https://hub.example:8443"
REDIRECT = "https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-x"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # type: ignore[no-untyped-def]
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


@pytest.fixture
def oauth_server(tmp_path: Path):
    db = tmp_path / "memory.db"
    with Store(db) as s:
        s.record("a fact the projection holds", "observation")
    tokens = tmp_path / "tokens"
    gem = remote.token_add(tokens, "gemini")
    httpd = remote.make_server(db, tokens, port=0, oauth_issuer=ISS)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def req(method: str, path: str, data: bytes | None = None, headers: dict | None = None):
        r = urllib.request.Request(base + path, data=data, method=method, headers=headers or {})
        try:
            with _OPENER.open(r, timeout=30) as resp:
                return resp.status, dict(resp.headers), resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode()

    try:
        yield req, gem, tokens
    finally:
        httpd.shutdown()
        httpd.server_close()


def _pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _register(req) -> dict:
    st, _h, body = req("POST", "/register", json.dumps({
        "redirect_uris": [REDIRECT], "token_endpoint_auth_method": "client_secret_basic"}).encode())
    assert st == 201
    return json.loads(body)


def _sign_in(req, q: dict, token: str) -> tuple[int, str]:
    st, _h, page = req("POST", "/authorize", urllib.parse.urlencode({**q, "token": token}).encode(),
                       {"Content-Type": "application/x-www-form-urlencoded"})
    return st, page


def _code_from(page: str) -> dict[str, str]:
    target = re.search(r'content="0;url=([^"]+)"', page).group(1).replace("&amp;", "&")  # type: ignore[union-attr]
    assert target.startswith(REDIRECT + "?")
    return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(target).query).items()}


def _exchange(req, reg: dict, code: str, verifier: str, redirect: str = REDIRECT):
    basic = base64.b64encode(f"{reg['client_id']}:{reg['client_secret']}".encode()).decode()
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect,
            "code_verifier": verifier}
    return req("POST", "/token", urllib.parse.urlencode(form).encode(),
               {"Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded"})


def test_discovery_to_bearer_ends_with_a_token_the_file_already_holds(oauth_server):
    req, gem, _tokens = oauth_server
    list_tools = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    st, h, _ = req("POST", "/mcp", list_tools)
    assert st == 401 and h["WWW-Authenticate"] == (
        f'Bearer resource_metadata="{ISS}/.well-known/oauth-protected-resource/mcp"')
    st, _h, body = req("GET", "/.well-known/oauth-protected-resource/mcp")
    assert st == 200 and json.loads(body)["authorization_servers"] == [ISS]
    st, _h, body = req("GET", "/.well-known/oauth-authorization-server")
    meta = json.loads(body)
    assert meta["registration_endpoint"] == ISS + "/register"
    assert meta["code_challenge_methods_supported"] == ["S256"]
    reg = _register(req)
    verifier, challenge = _pkce()
    state = "s" * 2000  # Gemini's state runs ~1,400-2,400 characters
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": state}
    st, h, page = req("GET", "/authorize?" + urllib.parse.urlencode(q))
    assert st == 200 and "form-action 'self'" in h["Content-Security-Policy"]
    st, page = _sign_in(req, q, gem)
    assert st == 200 and gem not in page  # a 200 page that navigates, never a 302
    back = _code_from(page)
    assert back["state"] == state
    st, _h, body = _exchange(req, reg, back["code"], verifier)
    tok = json.loads(body)
    assert st == 200 and tok == {"access_token": gem, "token_type": "Bearer"}
    st, _h, body = req("POST", "/mcp", list_tools, {"Authorization": f"Bearer {gem}"})
    assert st == 200 and "recall" in body
    # The same code is single use.
    assert _exchange(req, reg, back["code"], verifier)[0] == 400


def test_oauth_refusals(oauth_server, tmp_path):
    req, gem, tokens = oauth_server
    st, _h, _ = req("POST", "/register", json.dumps(
        {"redirect_uris": ["https://evil.example/cb"]}).encode())
    assert st == 400
    reg = _register(req)
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x"}
    for bad in ({"redirect_uri": "https://evil.example/cb"}, {"code_challenge_method": "plain"},
                {"response_type": "token"}):
        assert req("GET", "/authorize?" + urllib.parse.urlencode({**q, **bad}))[0] == 400, bad
        assert _sign_in(req, {**q, **bad}, gem)[0] == 400, bad  # the submit re-checks
    # A token this server's file does not hold is refused, and the page never echoes it.
    st, page = _sign_in(req, q, "not-a-token-here")
    assert st == 401 and "not-a-token-here" not in page
    # A wrong verifier burns the code.
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    assert _exchange(req, reg, code, "x" * 43)[0] == 400
    assert _exchange(req, reg, code, verifier)[0] == 400
    # A redirect URI other than the signed-in one is refused.
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    assert _exchange(req, reg, code, verifier, REDIRECT + "2")[0] == 400
    # A token revoked between sign-in and exchange is not handed out.
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    remote.token_revoke(tokens, "gemini")
    st, _h, body = _exchange(req, reg, code, verifier)
    assert st == 400 and gem not in body
    # Without --oauth-issuer none of these routes exist.
    db = tmp_path / "memory.db"
    plain = remote.make_server(db, tokens, port=0)
    threading.Thread(target=plain.serve_forever, daemon=True).start()
    try:
        for path in ("/.well-known/oauth-authorization-server", "/authorize"):
            r = urllib.request.Request(f"http://127.0.0.1:{plain.server_address[1]}{path}")
            with pytest.raises(urllib.error.HTTPError) as e:
                _OPENER.open(r, timeout=30)
            assert e.value.code == 404, path
    finally:
        plain.shutdown()
        plain.server_close()
