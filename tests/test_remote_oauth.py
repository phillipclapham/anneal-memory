"""The one-user OAuth front of the remote server (anneal_memory.remote_oauth), driven
over a real loopback socket the way the Gemini app's custom-app flow runs."""

from __future__ import annotations

import base64
import hashlib
import json
import os
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
    # The completed sign-in pinned its redirect: another app on the same Google
    # host (someone else's custom app) is refused from now on, the pinned one is not.
    pin = _tokens.with_name(_tokens.name + ".oauth-redirect")
    assert pin.read_text().strip() == REDIRECT
    if os.name == "posix":  # Windows st_mode does not carry POSIX permissions
        assert pin.stat().st_mode & 0o777 == 0o600
    other = "https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-attacker"
    st, _h, _ = req("POST", "/register", json.dumps({"redirect_uris": [other]}).encode())
    assert st == 400
    assert req("GET", "/authorize?" + urllib.parse.urlencode({**q, "redirect_uri": other}))[0] == 400
    assert _sign_in(req, {**q, "redirect_uri": other}, gem)[0] == 400
    assert _register(req)["redirect_uris"] == [REDIRECT]
    assert req("GET", "/authorize?" + urllib.parse.urlencode(q))[0] == 200


def test_gemini_registration_with_hosts_outside_the_set_keeps_the_allowed_ones(oauth_server):
    """The Gemini app's registration as reported 2026-10-15 (BerriAI/litellm#46640):
    /r/ and /a/ on three Google hosts, one of them (oauth-redirect-test) not allowed.
    Refusing the whole list was the live failure of 10-24."""
    req, gem, _tokens = oauth_server
    hosts = ("oauth-redirect-sandbox", "oauth-redirect-test", "oauth-redirect")
    sent = [f"https://{h}.googleusercontent.com/{p}/user_bound_custom-mcp-x"
            for p in ("r", "a") for h in hosts]
    st, _h, body = req("POST", "/register", json.dumps(
        {"redirect_uris": sent, "token_endpoint_auth_method": "client_secret_basic"}).encode())
    reg = json.loads(body)
    assert st == 201 and REDIRECT in reg["redirect_uris"]
    assert reg["redirect_uris"] == [u for u in sent if "oauth-redirect-test" not in u]
    test_only = [u for u in sent if "oauth-redirect-test" in u]
    assert req("POST", "/register", json.dumps({"redirect_uris": test_only}).encode())[0] == 400
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x"}
    st, _h, body = _exchange(req, reg, _code_from(_sign_in(req, q, gem)[1])["code"], verifier)
    assert st == 200 and json.loads(body)["access_token"] == gem


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
    # A non-ASCII client_id is a refusal, not a server error.
    form = {"grant_type": "authorization_code", "code": "x", "code_verifier": verifier,
            "client_id": "é"}
    assert req("POST", "/token", urllib.parse.urlencode(form).encode())[0] == 400
    # An unreadable token file is reported as the server's fault, never a wrong token.
    saved = tokens.read_bytes()
    tokens.write_text("not a token line\n")
    st, page = _sign_in(req, q, gem)
    assert st == 503 and "not valid" not in page
    tokens.write_bytes(saved)
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


def test_the_pin_binds_codes_issued_before_it_and_resource_is_bound(oauth_server):
    """Two sign-ins while nothing is pinned (two apps): the first exchange pins, the
    second app's live code is refused. A token is never issued for another resource."""
    req, gem, _tokens = oauth_server
    reg = _register(req)
    verifier, challenge = _pkce()
    other = "https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-other"
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x",
         "resource": ISS + "/mcp"}
    code_a = _code_from(_sign_in(req, q, gem)[1])["code"]
    st, page = _sign_in(req, {**q, "redirect_uri": other}, gem)
    assert st == 200
    target = re.search(r'content="0;url=([^"]+)"', page).group(1).replace("&amp;", "&")  # type: ignore[union-attr]
    code_b = urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)["code"][0]
    assert _exchange(req, reg, code_a, verifier)[0] == 200
    st, _h, body = _exchange(req, reg, code_b, verifier, other)
    assert st == 400 and gem not in body
    # RFC 8707: a resource other than this server is refused at sign-in and at exchange.
    for wrong in ("https://evil.example/mcp", ISS + "/mcp//"):
        assert req("GET", "/authorize?" + urllib.parse.urlencode({**q, "resource": wrong}))[0] == 400
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    basic = base64.b64encode(f"{reg['client_id']}:{reg['client_secret']}".encode()).decode()
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT,
            "code_verifier": verifier, "resource": "https://evil.example/mcp"}
    st, _h, body = req("POST", "/token", urllib.parse.urlencode(form).encode(),
                       {"Authorization": f"Basic {basic}"})
    assert st == 400 and json.loads(body)["error"] == "invalid_target"


@pytest.mark.parametrize("issuer", ["http://h", "https://h/x", "https://u@h", "https://h:bad",
                                    "https://h\r\nX: y", "https://h?q", "https:// h",
                                    "https://h///"])
def test_the_issuer_must_be_a_bare_https_origin(issuer):
    from anneal_memory.remote_oauth import OAuth
    with pytest.raises(ValueError):
        OAuth(issuer)
    assert OAuth("https://h.example:8443/").issuer == "https://h.example:8443"
    assert OAuth("https://[2001:db8::1]:8443").issuer == "https://[2001:db8::1]:8443"
