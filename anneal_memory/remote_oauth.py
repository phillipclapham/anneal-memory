"""A one-user OAuth 2.1 front for :mod:`anneal_memory.remote` (experimental; not
part of the frozen public surface in docs/stability.md).

Some MCP clients (the Gemini app's custom apps) refuse a server without OAuth.
This adds exactly what such a client needs to end up holding one of the bearer
tokens the server already accepts, and nothing that could mint any other:

- Discovery: RFC 9728 protected-resource metadata and RFC 8414 authorization
  server metadata, and ``WWW-Authenticate: Bearer resource_metadata=...`` on a 401.
- ``POST /register`` (RFC 7591): stateless. It returns a fresh client id and
  secret and stores neither; anyone may register, so a client secret proves
  nothing and is not checked. What binds a client is the authorization code.
- ``GET /authorize``: a sign-in page. ``POST /authorize``: the operator pastes an
  existing connector token; it is checked against the server's own token file.
  Only then is a one-time code (single use, :data:`CODE_TTL_S`, PKCE S256 bound,
  client id and redirect URI bound) issued, delivered by a 200 page that
  navigates to the redirect URI (a 302 there was reported blocked by a
  ``form-action`` policy in Gemini's callback chain).
- ``POST /token``: exchanges the code, checks PKCE, re-checks the token is still
  in the token file, and returns that same token as the access token.

So the only credential this flow ever hands out is one the operator typed and
the token file already holds: a server run with a different token file can
neither mint nor accept another server's tokens. There is no refresh token and
no expiry beyond the token's own revocation (``token-revoke``).

Redirect URIs must be https on a host in :data:`REDIRECT_HOSTS`. Every Gemini app
returns through those same hosts, so anyone's own app could be pointed here and
its sign-in link sent to the operator. So the first sign-in that completes an
exchange pins its exact redirect URI to a file (created exclusively, mode 0600),
and from then on only that URI is accepted at registration and at sign-in; the
sign-in page shows the full URI and, while nothing is pinned, says that the app
allowed now becomes the only one. Deleting the pin file re-opens the first use.
Codes live in memory; a restart between sign-in and exchange fails that sign-in only.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any


def _fsync_dir_strict(path: Path) -> None:
    """Make a new directory entry durable, or raise. Unlike the shared best-effort
    helper, a failure here must stop the sign-in: a pin lost in a crash after the
    token went out would re-open first use. Windows has no directory fsync."""
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


CODE_TTL_S = 120
MAX_PENDING_CODES = 32
MAX_STATE_CHARS = 4096
MAX_FORM_BYTES = 16 * 1024
MAX_REGISTER_BYTES = 16 * 1024
# The hosts the Gemini app registers as redirect URIs (oauth-redirect,
# oauth-redirect-sandbox), read from public reports 2026-10-24
# (github.com/doobidoo/mcp-memory-service/issues/1553; discuss.ai.google.dev
# t/186405). A redirect anywhere else is refused at registration and at sign-in.
REDIRECT_HOSTS = frozenset({"oauth-redirect.googleusercontent.com",
                            "oauth-redirect-sandbox.googleusercontent.com"})
_VERIFIER_RE = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")
_CHALLENGE_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # base64url(sha256), unpadded
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
AUTH_METHODS = ("client_secret_basic", "client_secret_post", "none")
# What a ``check_token`` callback returns when the token file cannot be read,
# so a broken server is never reported to the operator as a wrong token.
UNAVAILABLE = object()
_INVALID_PIN = "\x00invalid pin"  # matches no redirect URI


def redirect_ok(uri: Any) -> bool:
    if not isinstance(uri, str) or len(uri) > 2048:
        return False
    try:
        parts = urllib.parse.urlsplit(uri)
        port = parts.port
    except ValueError:
        return False
    return (parts.scheme == "https" and parts.hostname in REDIRECT_HOSTS
            and port in (None, 443) and not parts.username and not parts.password
            and not parts.fragment and uri.isascii() and uri.isprintable())


def s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class OAuth:
    """Endpoints for one issuer. Each handler returns ``(status, headers, body)``."""

    def __init__(self, issuer: str, pin_path: str | os.PathLike[str] | None = None) -> None:
        # It reaches metadata and a response header: only an exact https://host[:port].
        bad = ValueError("the OAuth issuer must be a bare https origin, https://host[:port]")
        if not issuer.isascii() or not issuer.isprintable() or any(c in issuer for c in ' "\\'):
            raise bad
        parts = urllib.parse.urlsplit(issuer)
        try:
            port = parts.port
        except ValueError:
            raise bad from None
        host = parts.hostname or ""
        if ":" in host:  # an IPv6 literal keeps its brackets
            host = f"[{host}]"
        origin = f"https://{host}" + (f":{port}" if port is not None else "")
        if parts.scheme != "https" or not parts.hostname or \
                issuer.lower() not in (origin, origin + "/"):
            raise bad
        self.issuer = origin
        self.resource = self.issuer + "/mcp"
        self._codes: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._pin_path = Path(pin_path) if pin_path is not None else None

    # -- the redirect pin --

    def pinned(self) -> str | None:
        """The redirect URI the first completed sign-in pinned, or None before one."""
        if self._pin_path is None:
            return None
        try:
            text = self._pin_path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        except (OSError, ValueError):  # unreadable or not ASCII: refuse every redirect
            return _INVALID_PIN
        return text if redirect_ok(text) else _INVALID_PIN

    def _redirect_allowed(self, uri: Any) -> bool:
        pin = self.pinned()
        return redirect_ok(uri) and (pin is None or uri == pin)

    def _ensure_pinned(self, uri: str) -> bool:
        """True when ``uri`` is the pinned redirect, pinning it if nothing is yet.
        The pin is written whole to a temporary file and linked into place, so it
        appears complete or not at all and the first link wins across processes;
        the directory is synced before a token is released. Raises OSError when it
        cannot be written. Call under ``self._lock``."""
        if self._pin_path is None:
            return True
        current = self.pinned()
        if current is not None:
            return current == uri
        tmp = self._pin_path.with_name(f"{self._pin_path.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as f:
                f.write(uri + "\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, self._pin_path)
            except FileExistsError:
                pass  # another process pinned first; compare below
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        _fsync_dir_strict(self._pin_path.parent)
        return self.pinned() == uri

    # -- discovery --

    def resource_metadata_url(self) -> str:
        return self.issuer + "/.well-known/oauth-protected-resource/mcp"

    def challenge_header(self) -> str:
        return f'Bearer resource_metadata="{self.resource_metadata_url()}"'

    def protected_resource(self) -> dict[str, Any]:
        return {"resource": self.resource, "authorization_servers": [self.issuer],
                "bearer_methods_supported": ["header"]}

    def server_metadata(self) -> dict[str, Any]:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.issuer + "/authorize",
            "token_endpoint": self.issuer + "/token",
            "registration_endpoint": self.issuer + "/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": list(AUTH_METHODS),
        }

    # -- registration --

    def register(self, body: bytes) -> tuple[int, dict[str, Any]]:
        try:
            meta = json.loads(body)
        except (ValueError, RecursionError):
            return 400, {"error": "invalid_client_metadata"}
        if not isinstance(meta, dict):
            return 400, {"error": "invalid_client_metadata"}
        uris = meta.get("redirect_uris")
        if not isinstance(uris, list) or not uris or len(uris) > 16 \
                or not all(redirect_ok(u) for u in uris) \
                or (self.pinned() is not None and self.pinned() not in uris):
            return 400, {"error": "invalid_redirect_uri",
                         "error_description": "redirect URIs must be https on an allowed "
                                              "host, and include the pinned one once pinned"}
        method = meta.get("token_endpoint_auth_method", "client_secret_basic")
        if method not in AUTH_METHODS:
            method = "client_secret_basic"
        out: dict[str, Any] = {
            "client_id": secrets.token_urlsafe(24),
            "client_id_issued_at": int(time.time()),
            "redirect_uris": uris,
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": method,
        }
        if method != "none":
            out["client_secret"] = secrets.token_urlsafe(32)
            out["client_secret_expires_at"] = 0
        name = meta.get("client_name")
        if isinstance(name, str) and len(name) <= 200:
            out["client_name"] = name
        return 201, out

    # -- authorization --

    def _request_problem(self, q: dict[str, str]) -> str | None:
        """Why an authorization request is refused, or None. Checked on the page
        and again on the submit, which carries the same fields."""
        if q.get("response_type") != "code":
            return "response_type must be code"
        if not _CLIENT_ID_RE.fullmatch(q.get("client_id", "")):
            return "missing or malformed client_id"
        if not self._redirect_allowed(q.get("redirect_uri")):
            return "redirect_uri is not allowed (not the app this server was first connected to)"
        if q.get("code_challenge_method") != "S256" or \
                not _CHALLENGE_RE.fullmatch(q.get("code_challenge", "")):
            return "PKCE with S256 is required"
        state = q.get("state", "")
        if len(state) > MAX_STATE_CHARS or not state.isprintable():
            return "state is too long or not printable"
        if "resource" in q and not self._resource_ok(q["resource"]):
            return "resource is not this server"  # RFC 8707: never a token for another
        return None

    def _resource_ok(self, value: str) -> bool:
        return value in (self.resource, self.resource + "/")

    @staticmethod
    def _fields(raw: dict[str, list[str]]) -> dict[str, str] | None:
        """One value per field, or None when any field repeats."""
        if any(len(v) != 1 for v in raw.values()):
            return None
        return {k: v[0] for k, v in raw.items()}

    def authorize_page(self, query: str) -> tuple[int, str]:
        try:
            q = self._fields(urllib.parse.parse_qs(query, keep_blank_values=True,
                                                   max_num_fields=16))
        except ValueError:
            return 400, _page("Sign-in refused", "<p>malformed query</p>")
        problem = "a field is repeated" if q is None else self._request_problem(q)
        if problem or q is None:  # never redirect on a bad request (RFC 6749 §4.1.2.1)
            return 400, _page("Sign-in refused", f"<p>{html.escape(problem or '')}</p>")
        return 200, self._form(q, "")

    def _form(self, q: dict[str, str], error: str) -> str:
        keep = ("response_type", "client_id", "redirect_uri", "code_challenge",
                "code_challenge_method", "state", "resource", "scope")
        hidden = "".join(
            f'<input type="hidden" name="{k}" value="{html.escape(q[k], quote=True)}">'
            for k in keep if k in q)
        err = f'<p class="err">{html.escape(error)}</p>' if error else ""
        where = html.escape(q["redirect_uri"], quote=True)
        first = ("" if self.pinned() is not None else
                 "<p><b>No app is connected yet: the one you allow now becomes the only "
                 "one this server accepts.</b> Only continue if you started this from "
                 "your own app.</p>")
        return _page("Connect to anneal memory", (
            f"<p>An app is asking for read access to this memory. It returns to "
            f"<code>{where}</code>.</p>{first}{err}"
            f'<form method="post" action="/authorize">{hidden}'
            '<label>Connector token<br><input type="password" name="token" '
            'autocomplete="off" required></label><br>'
            '<button type="submit">Allow</button></form>'))

    def authorize_submit(self, body: bytes, check_token: Any) -> tuple[int, str]:
        """``check_token(token) -> connector name | None`` against the token file."""
        try:
            raw = urllib.parse.parse_qs(body.decode("ascii"), keep_blank_values=True,
                                        max_num_fields=16)
        except (UnicodeDecodeError, ValueError):
            return 400, _page("Sign-in refused", "<p>malformed form</p>")
        q = self._fields(raw)
        problem = "a field is repeated" if q is None else self._request_problem(q)
        if problem:
            return 400, _page("Sign-in refused", f"<p>{html.escape(problem)}</p>")
        assert q is not None
        token = q.get("token", "").strip()
        checked = check_token(token) if token else None
        if checked is UNAVAILABLE:
            return 503, _page("Unavailable", "<p>The server cannot read its token file.</p>")
        if checked is None:  # tokens are 32 random bytes: no delay buys anything
            return 401, self._form(q, "That token is not valid here.")
        code = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            for c in [c for c, v in self._codes.items() if v["exp"] <= now]:
                del self._codes[c]
            if len(self._codes) >= MAX_PENDING_CODES:
                return 503, _page("Busy", "<p>Too many sign-ins in progress; try again shortly.</p>")
            self._codes[code] = {"client_id": q["client_id"], "redirect_uri": q["redirect_uri"],
                                 "challenge": q["code_challenge"], "token": token,
                                 "resource": q.get("resource"),
                                 "exp": now + CODE_TTL_S}
        params = {"code": code}
        if "state" in q:
            params["state"] = q["state"]
        target = _with_query(q["redirect_uri"], params)
        esc = html.escape(target, quote=True)
        return 200, _page("Connected", (
            f'<p>Returning to the app. <a href="{esc}">Continue</a> if nothing happens.</p>'),
            refresh=target)

    # -- token --

    def token(self, body: bytes, basic: str | None, check_token: Any) -> tuple[int, dict[str, Any]]:
        try:
            raw = urllib.parse.parse_qs(body.decode("ascii"), keep_blank_values=True,
                                        max_num_fields=16)
        except (UnicodeDecodeError, ValueError):
            return 400, {"error": "invalid_request"}
        q = self._fields(raw)
        if q is None:
            return 400, {"error": "invalid_request"}
        if q.get("grant_type") != "authorization_code":
            return 400, {"error": "unsupported_grant_type"}
        client_id = q.get("client_id")
        if basic is not None:
            try:
                padded = basic + "=" * (-len(basic) % 4)
                user = base64.b64decode(padded, validate=True).decode("ascii").split(":", 1)[0]
                basic_id = urllib.parse.unquote_plus(user)  # RFC 6749 §2.3.1: form-encoded
            except (ValueError, UnicodeDecodeError):
                return 401, {"error": "invalid_client"}
            if client_id is not None and client_id != basic_id:
                return 400, {"error": "invalid_request"}
            client_id = basic_id
        if not isinstance(client_id, str) or not _CLIENT_ID_RE.fullmatch(client_id):
            return 400, {"error": "invalid_request"}  # also keeps compare_digest on ASCII
        with self._lock:
            entry = self._codes.pop(q.get("code", ""), None)  # single use, even when refused
        verifier = q.get("code_verifier", "")
        if (entry is None or entry["exp"] <= time.monotonic()
                or not hmac.compare_digest(client_id, entry["client_id"])
                # optional with PKCE (OAuth 2.1); when sent it must be the signed-in one
                or q.get("redirect_uri", entry["redirect_uri"]) != entry["redirect_uri"]
                or not _VERIFIER_RE.fullmatch(verifier)
                or not hmac.compare_digest(s256(verifier), entry["challenge"])):
            return 400, {"error": "invalid_grant"}
        resource = q.get("resource")
        if resource is not None and not self._resource_ok(resource) or \
                entry["resource"] is not None and resource is not None \
                and resource.rstrip("/") != entry["resource"].rstrip("/"):  # both already exact
            return 400, {"error": "invalid_target"}
        still = check_token(entry["token"])
        if still is UNAVAILABLE:
            return 503, {"error": "server_error"}
        if still is None:  # revoked since the sign-in
            return 400, {"error": "invalid_grant"}
        # The pin binds codes issued before it existed too: a second app's code,
        # signed in while nothing was pinned, is refused once the first one pinned.
        try:
            with self._lock:
                pinned_here = self._ensure_pinned(entry["redirect_uri"])
        except OSError:
            return 503, {"error": "server_error"}
        if not pinned_here:
            return 400, {"error": "invalid_grant"}
        return 200, {"access_token": entry["token"], "token_type": "Bearer"}


def _with_query(uri: str, params: dict[str, str]) -> str:
    parts = urllib.parse.urlsplit(uri)
    query = "&".join(p for p in (parts.query, urllib.parse.urlencode(params)) if p)
    return urllib.parse.urlunsplit(parts._replace(query=query))


def _page(title: str, body: str, refresh: str | None = None) -> str:
    meta = (f'<meta http-equiv="refresh" content="0;url={html.escape(refresh, quote=True)}">'
            if refresh else "")
    return ("<!doctype html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"{meta}<title>{html.escape(title)}</title><style>"
            "body{font:16px system-ui,sans-serif;max-width:28rem;margin:3rem auto;padding:0 1rem}"
            "input[type=password]{width:100%;padding:.5rem;margin:.5rem 0}"
            "button{padding:.5rem 1.2rem}.err{color:#b00}</style></head>"
            f"<body><h1>{html.escape(title)}</h1>{body}</body></html>")


# Sent with every page: nothing loads from anywhere, the page cannot be framed,
# and the form may only post back here. The navigation to the redirect URI is a
# meta refresh, which form-action does not govern.
PAGE_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; style-src 'unsafe-inline'; "
                                "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"),
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
