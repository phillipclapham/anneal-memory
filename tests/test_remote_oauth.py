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

from anneal_memory import remote, remote_oauth
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
    # The completed sign-in pinned its redirect for this connector: another app on
    # the same Google host (someone else's custom app) is refused that token from
    # now on, the pinned one is not.
    pin = _tokens.with_name(_tokens.name + ".oauth-redirect.gemini")
    assert pin.read_text().strip() == REDIRECT
    assert not _tokens.with_name(_tokens.name + ".oauth-redirect").exists()
    if os.name == "posix":  # Windows st_mode does not carry POSIX permissions
        assert pin.stat().st_mode & 0o777 == 0o600
    other = "https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-attacker"
    st, page = _sign_in(req, {**q, "redirect_uri": other}, gem)
    assert st == 400 and "already connected" in page
    assert _code_from(_sign_in(req, q, gem)[1])["code"]


def _sign_in_code(req, q: dict, token: str, redirect: str) -> str:
    st, page = _sign_in(req, {**q, "redirect_uri": redirect}, token)
    assert st == 200, page
    target = re.search(r'content="0;url=([^"]+)"', page).group(1).replace("&amp;", "&")  # type: ignore[union-attr]
    assert target.startswith(redirect + "?")
    return urllib.parse.parse_qs(urllib.parse.urlsplit(target).query)["code"][0]


def _public_exchange(req, client_id: str, code: str, verifier: str, redirect: str, **extra):
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect,
            "code_verifier": verifier, "client_id": client_id, **extra}
    return req("POST", "/token", urllib.parse.urlencode(form).encode(),
               {"Content-Type": "application/x-www-form-urlencoded"})


CLAUDE = "https://claude.ai/api/mcp/auth_callback"
CHATGPT = "https://chatgpt.com/connector_platform_oauth_redirect"
CHATGPT_CB = "https://chatgpt.com/connector/oauth/cb_abc123"


def test_claude_and_chatgpt_connectors_each_sign_in_and_pin_their_own_redirect(oauth_server):
    """The two documented callbacks register as public clients, sign in with their own
    connector token, get that token back, list tools; each token stays with its app."""
    req, _gem, tokens = oauth_server
    claude_tok = remote.token_add(tokens, "claude-ai")
    gpt_tok = remote.token_add(tokens, "chatgpt")
    list_tools = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    for token, name, redirect, resource in (
            (claude_tok, "claude-ai", CLAUDE, ISS + "/mcp"),
            (gpt_tok, "chatgpt", CHATGPT_CB, ISS)):  # ChatGPT may name the bare origin
        st, _h, body = req("POST", "/register", json.dumps({
            "redirect_uris": [redirect], "token_endpoint_auth_method": "none"}).encode())
        assert st == 201, body
        reg = json.loads(body)
        assert "client_secret" not in reg and reg["redirect_uris"] == [redirect]
        verifier, challenge = _pkce()
        q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": redirect,
             "code_challenge": challenge, "code_challenge_method": "S256", "state": "s",
             "resource": resource, "scope": "openid"}
        assert req("GET", "/authorize?" + urllib.parse.urlencode(q))[0] == 200
        code = _sign_in_code(req, q, token, redirect)
        st, _h, body = _public_exchange(req, reg["client_id"], code, verifier, redirect,
                                        resource=resource)
        assert st == 200 and json.loads(body)["access_token"] == token
        st, _h, body = req("POST", "/mcp", list_tools, {"Authorization": f"Bearer {token}"})
        assert st == 200 and "recall" in body
        assert tokens.with_name(f"{tokens.name}.oauth-redirect.{name}").read_text().strip() == redirect
    # Each token is now bound to its own app: the claude-ai token is refused to ChatGPT
    # and the chatgpt token to claude.ai, at the sign-in submit.
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "cid", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "s"}
    for token, wrong in ((claude_tok, CHATGPT_CB), (gpt_tok, CLAUDE), (gpt_tok, CHATGPT)):
        st, page = _sign_in(req, {**q, "redirect_uri": wrong}, token)
        assert st == 400 and token not in page, wrong


def test_a_code_for_a_second_app_is_refused_once_its_connector_pins(oauth_server):
    """Two sign-ins with one token before anything is pinned (two apps): the first
    exchange pins that connector, the second app's live code is refused; another
    connector is untouched."""
    req, gem, tokens = oauth_server
    claude_tok = remote.token_add(tokens, "claude-ai")
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "cid", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "s"}
    code_a = _sign_in_code(req, q, claude_tok, CLAUDE)
    code_b = _sign_in_code(req, q, claude_tok, CHATGPT)
    assert _public_exchange(req, "cid", code_a, verifier, CLAUDE)[0] == 200
    st, _h, body = _public_exchange(req, "cid", code_b, verifier, CHATGPT)
    assert st == 400 and claude_tok not in body
    assert not tokens.with_name(tokens.name + ".oauth-redirect.gemini").exists()
    code = _sign_in_code(req, q, gem, REDIRECT)
    assert _public_exchange(req, "cid", code, verifier, REDIRECT)[0] == 200


def test_a_server_wide_pin_still_binds_every_connector(oauth_server):
    """A pin file written before per-connector pins (the Gemini unit's) keeps its
    meaning: registration must include it and no connector signs in elsewhere."""
    req, gem, tokens = oauth_server
    claude_tok = remote.token_add(tokens, "claude-ai")
    legacy = tokens.with_name(tokens.name + ".oauth-redirect")
    legacy.write_text(REDIRECT + "\n")
    st, _h, _ = req("POST", "/register", json.dumps({"redirect_uris": [CLAUDE]}).encode())
    assert st == 400
    assert _register(req)["redirect_uris"] == [REDIRECT]
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "cid", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "s"}
    assert req("GET", "/authorize?" + urllib.parse.urlencode({**q, "redirect_uri": CLAUDE}))[0] == 400
    assert _sign_in(req, {**q, "redirect_uri": CLAUDE}, claude_tok)[0] == 400
    page = req("GET", "/authorize?" + urllib.parse.urlencode({**q, "redirect_uri": REDIRECT}))[2]
    assert "becomes the only" not in page
    code = _sign_in_code(req, q, gem, REDIRECT)
    assert _public_exchange(req, "cid", code, verifier, REDIRECT)[0] == 200
    assert not tokens.with_name(tokens.name + ".oauth-redirect.gemini").exists()
    # An unreadable server-wide pin refuses everything, as before.
    legacy.write_bytes(b"\xff\n")
    assert _sign_in(req, {**q, "redirect_uri": REDIRECT}, gem)[0] == 400


@pytest.mark.parametrize("uri,ok", [
    (CLAUDE, True), (CHATGPT, True), (CHATGPT_CB, True),
    ("https://claude.ai/", False), ("https://claude.ai/api/mcp/auth_callback/x", False),
    ("https://claude.ai/api/mcp/auth_callback?next=/x", False),
    ("https://chatgpt.com/share/abc", False), ("https://chatgpt.com/connector/oauth/a/b", False),
    ("https://chatgpt.com/connector/oauth/", False),
    (REDIRECT, True), (REDIRECT + "?q=1", True),  # the Google hosts keep their rule
])
def test_whole_app_hosts_pass_only_their_documented_callback_paths(uri, ok):
    assert remote_oauth.redirect_ok(uri) is ok


def test_connector_pin_refusals_never_log_the_token_and_the_pin_file_is_lowercased(
        oauth_server, caplog):
    req, _gem, tokens = oauth_server
    tok = remote.token_add(tokens, "Claude-AI")
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "cid", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "s"}
    code = _sign_in_code(req, q, tok, CLAUDE)
    assert _public_exchange(req, "cid", code, verifier, CLAUDE)[0] == 200
    pin = tokens.with_name(tokens.name + ".oauth-redirect.claude-ai")
    assert pin.read_text().strip() == CLAUDE
    caplog.set_level("INFO", logger="anneal-memory.remote")
    assert _sign_in(req, {**q, "redirect_uri": CHATGPT}, tok)[0] == 400  # pinned elsewhere
    pin.write_bytes(b"\xff\n")
    assert _sign_in(req, {**q, "redirect_uri": CLAUDE}, tok)[0] == 400  # unreadable pin
    assert "pinned for Claude-AI" in caplog.text and "unreadable" in caplog.text
    assert tok not in caplog.text


def test_the_server_wide_pin_outranks_a_connector_pin_and_a_bad_connector_pin_says_so(oauth_server):
    req, gem, tokens = oauth_server
    claude_tok = remote.token_add(tokens, "claude-ai")
    connector_pin = tokens.with_name(tokens.name + ".oauth-redirect.claude-ai")
    connector_pin.write_text(CLAUDE + "\n")
    legacy = tokens.with_name(tokens.name + ".oauth-redirect")
    legacy.write_text(REDIRECT + "\n")
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "cid", "code_challenge": challenge,
         "code_challenge_method": "S256", "state": "s"}
    assert _sign_in(req, {**q, "redirect_uri": CLAUDE}, claude_tok)[0] == 400
    legacy.unlink()
    assert _sign_in(req, {**q, "redirect_uri": CLAUDE}, claude_tok)[0] == 200
    connector_pin.write_bytes(b"\xff\n")
    st, page = _sign_in(req, {**q, "redirect_uri": CLAUDE}, claude_tok)
    assert st == 400 and "cannot be read" in page and claude_tok not in page


def test_a_connector_name_that_cannot_be_a_file_suffix_never_pins(tmp_path: Path):
    oauth = remote_oauth.OAuth(ISS, tmp_path / "tokens.oauth-redirect")
    assert oauth.pinned("../x") == remote_oauth._INVALID_PIN
    with oauth._lock:
        assert oauth._ensure_pinned(CLAUDE, "../x") is False
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    # The suffix rule is the rule token names are created under.
    assert remote_oauth._CONNECTOR_RE.pattern == remote._TOKEN_NAME_RE.pattern.strip("^$")


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


def test_refusals_are_logged_without_query_or_userinfo(oauth_server, caplog):
    """The operator's journal says why a sign-in step was refused and which redirect,
    and never carries a query (state) or userinfo."""
    req, gem, _tokens = oauth_server
    caplog.set_level("INFO", logger="anneal-memory.remote")
    secret_q = "?state=SECRETSTATE"
    test_host = "https://u:pw@oauth-redirect-test.googleusercontent.com/r/x" + secret_q
    st, _h, _ = req("POST", "/register", json.dumps(
        {"redirect_uris": [REDIRECT, test_host, 5]}).encode())
    assert st == 201
    assert req("POST", "/register", json.dumps({"redirect_uris": [test_host]}).encode())[0] == 400
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": "c", "redirect_uri": test_host,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x"}
    assert req("GET", "/authorize?" + urllib.parse.urlencode(q))[0] == 400
    text = caplog.text
    assert "oauth register accepted: 1 of 3" in text and "<int>" in text
    assert "oauth register refused (no redirect URI" in text
    assert ("oauth authorize refused (page): redirect_uri is not allowed" in text
            and "oauth-redirect-test.googleusercontent.com/r/x" in text)
    assert "SECRETSTATE" not in text and "pw@" not in text and gem not in text


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


def test_every_request_and_the_token_grant_are_logged_without_secrets(oauth_server, caplog):
    """Each answered request leaves one line (status, method, path, Host, User-Agent),
    2xx included, and /token names the grant's shape; no code, verifier, secret,
    state or token reaches the journal (10-24: the Gemini app stopped after its
    token and the journal could not say where it went)."""
    req, gem, _tokens = oauth_server
    caplog.set_level("INFO", logger="anneal-memory.remote")
    reg = _register(req)
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "SECRETSTATE"}
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    assert _exchange(req, reg, code, verifier)[0] == 200
    assert req("HEAD", "/mcp", headers={"Authorization": f"Bearer {gem}",
                                        "User-Agent": "Google-Probe/1 \u00e9"})[0] == 200
    text = caplog.text
    assert "answered 200 HEAD /mcp host=127.0.0.1:" in text
    assert "ua=Google-Probe/1 \\xe9" in text  # escaped: the journal stays ASCII
    assert "answered 200 POST /token" in text and "answered 201 POST /register" in text
    assert ("oauth token issued (client auth basic; fields "
            "code,code_verifier,grant_type,redirect_uri; grant_type authorization_code)") in text
    for secret in (gem, code, verifier, reg["client_secret"], "SECRETSTATE"):
        assert secret not in text


def test_a_token_in_an_unknown_path_or_resource_never_reaches_the_journal(oauth_server, caplog):
    """Only known paths are named in the journal; a prefixed or doubled token URL,
    the likeliest shape of a half-configured client, is logged by length."""
    req, gem, _tokens = oauth_server
    caplog.set_level("INFO", logger="anneal-memory.remote")
    for path in (f"/gemini/mcp/{gem}", f"//mcp/{gem}", f"/mcp%2F{gem}", f"/mcp/{gem}?x=1",
                 f"/.well-known/oauth-protected-resource/mcp/{gem}", f"/MCP/{gem}"):
        assert req("POST", path, b"{}", {"Content-Type": "application/json"})[0] in (400, 401, 404)
    reg = _register(req)
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x"}
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    basic = base64.b64encode(f"{reg['client_id']}:{reg['client_secret']}".encode()).decode()
    form = {"grant_type": "authorization_code", "code": code, "code_verifier": verifier,
            "resource": f"{ISS}/mcp/{gem}"}
    st, _h, _b = req("POST", "/token", urllib.parse.urlencode(form).encode(),
                     {"Authorization": f"Basic {basic}",
                      "Content-Type": "application/x-www-form-urlencoded"})
    assert st == 400
    text = caplog.text
    assert gem not in text and "(other path, " in text
    assert f"resource {ISS}/mcp(token-in-path) is not this server's" in text
    other = f"https://{gem}.example/mcp"
    assert remote_oauth._url_for_log(other, ISS) == f"(other origin, {len(other)} chars)"
    assert remote_oauth.path_for_log("/.well-known/oauth-authorization-server") == \
        "/.well-known/oauth-authorization-server"
    assert remote_oauth.path_for_log("/a\nb") == "(other path, 4 chars)"


def test_a_json_body_to_token_is_counted_not_echoed(oauth_server, caplog):
    """A JSON body parsed as a form is one field named by the whole body (L2 1024+22):
    the code and verifier in it never reach the journal, and the code stays usable."""
    req, gem, _tokens = oauth_server
    caplog.set_level("INFO", logger="anneal-memory.remote")
    reg = _register(req)
    verifier, challenge = _pkce()
    q = {"response_type": "code", "client_id": reg["client_id"], "redirect_uri": REDIRECT,
         "code_challenge": challenge, "code_challenge_method": "S256", "state": "x"}
    code = _code_from(_sign_in(req, q, gem)[1])["code"]
    body = json.dumps({"grant_type": "authorization_code", "code": code,
                       "code_verifier": verifier, "client_secret": reg["client_secret"]})
    assert req("POST", "/token", body.encode(), {"Content-Type": "application/json"})[0] == 400
    smuggled = f"grant_type=authorization_code;code={code}"
    assert req("POST", "/token", smuggled.encode())[0] == 400
    text = caplog.text
    assert "oauth token refused: unsupported grant (client auth none; fields - +1 other (" in text
    assert "grant_type (other, " in text
    for secret in (code, verifier, reg["client_secret"], code[:16]):
        assert secret not in text
    assert _exchange(req, reg, code, verifier)[0] == 200


def _raw(port: int, head: str, body: bytes = b"") -> int:
    import contextlib
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=20) as s:
        with contextlib.suppress(ConnectionError):  # a refusal may come before the body is read
            s.sendall(head.encode() + b"\r\n" + body)
        data = b""
        while b"\r\n" not in data:
            got = s.recv(4096)
            if not got:
                break
            data += got
    return int(data.split(b" ", 2)[1])


def test_a_chunked_tool_call_is_answered(tmp_path: Path):
    """Funnel forwards an HTTP/2 request that carries no length as a chunked
    HTTP/1.1 body; the Gemini app's tool calls arrived that way (10-24: 411)."""
    db = tmp_path / "memory.db"
    with Store(db) as s:
        s.record("a fact the projection holds", "observation")
    tokens = tmp_path / "tokens"
    gem = remote.token_add(tokens, "gemini")
    httpd = remote.make_server(db, tokens, port=0, oauth_issuer=ISS)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    head = (f"POST /mcp HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {gem}\r\n"
            "Content-Type: application/json\r\nAccept: application/json, text/event-stream\r\n")
    msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode()
    one, two = msg[:10], msg[10:]
    chunked = (f"{len(one):x};ext=1\r\n".encode() + one + b"\r\n"
               + f"{len(two):X}\r\n".encode() + two + b"\r\n0\r\nX-Trailer: y\r\n\r\n")
    try:
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n", chunked) == 200
        big = b"%x\r\n" % (remote.MAX_BODY_BYTES + 1)
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n", big) == 413
        assert _raw(port, head + "Transfer-Encoding: chunked\r\nContent-Length: 5\r\n",
                    chunked) == 411
        assert _raw(port, head + "Transfer-Encoding: gzip, chunked\r\n", chunked) == 411
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n", b"zz\r\n") == 400
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n", b"+5\r\nhello\r\n0\r\n\r\n") == 400
        assert _raw(port, head) == 400  # no length at all: an empty body, not JSON
        sixteen = b"".join(b"X-T%d: v\r\n" % i for i in range(16))
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n",
                    chunked.replace(b"X-Trailer: y\r\n", sixteen)) == 200
        seventeen = sixteen + b"X-T16: v\r\n"
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n",
                    chunked.replace(b"X-Trailer: y\r\n", seventeen)) == 400
        tiny = b"".join(b"1\r\n" + bytes([c]) + b"\r\n" for c in msg) + b"0\r\n\r\n"
        assert _raw(port, head + "Transfer-Encoding: chunked\r\n", tiny) == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_trickled_chunked_body_meets_one_deadline(tmp_path: Path, monkeypatch):
    """One byte at a time must not hold the thread past the whole-body deadline
    (L3 1024+22: a per-line socket timeout let a trickle last for hours; the
    OAuth routes read their bodies before any auth)."""
    import socket
    import time as _time
    monkeypatch.setattr(remote, "REQUEST_TIMEOUT_S", 1)
    tokens = tmp_path / "tokens"
    remote.token_add(tokens, "gemini")
    db = tmp_path / "memory.db"
    Store(db).close()
    httpd = remote.make_server(db, tokens, port=0, oauth_issuer=ISS)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(("127.0.0.1", httpd.server_address[1]), timeout=10) as s:
            s.sendall(b"POST /register HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n")
            start = _time.monotonic()
            answer = b""
            s.settimeout(0.25)
            while _time.monotonic() - start < 6 and not answer:
                try:
                    s.sendall(b"0")  # a size line that never ends
                    answer = s.recv(64)
                except socket.timeout:
                    continue
                except ConnectionError:
                    break
            elapsed = _time.monotonic() - start
        assert answer.startswith(b"HTTP/1.1 408") and elapsed < 3, (answer, elapsed)
    finally:
        httpd.shutdown()
        httpd.server_close()
