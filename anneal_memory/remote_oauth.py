"""A one-user OAuth 2.1 front for :mod:`anneal_memory.remote` (experimental; not
part of the frozen public surface in docs/stability.md).

Some MCP clients (the Gemini app's custom apps) refuse a server without OAuth, and
others (claude.ai and ChatGPT connectors) otherwise need the token in the URL.
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

Redirect URIs must be https on a host in :data:`REDIRECT_HOSTS`; a registration
keeps the URIs that pass and drops the rest, and is refused only when none pass
(the Gemini app also registers hosts outside the set). Every app of one vendor
has its redirects on the same hosts, so anyone's own app could be pointed here and
its sign-in link sent to the operator. So the first sign-in that completes an
exchange for a connector token pins its exact redirect URI to that connector's
pin file (``<pin_path>.<connector name, lowercased>``, created exclusively, mode 0600), and
from then on sign-in with that token accepts only that URI: after its first
completed sign-in, each connector token is accepted only for the redirect URI it
pinned. Nothing ties a token's
name to a vendor. The pin follows the name, so a token revoked and re-added under
the same name keeps its app. The sign-in page shows the full URI. Deleting a pin
file re-opens the first use for its connector.
A pin file at ``pin_path`` itself (the one pin servers wrote before per-connector pins) still
binds every connector, and registration must then include its URI; renaming it to
``<pin_path>.<connector name, lowercased>`` makes it that connector's pin and frees
the others.
Known edges: two connector names that differ only by case share one pin; a name
long enough to push the pin file name past the filesystem's limit can never pin.
For claude.ai every user's app shares one callback URI, so a pin there separates
vendors, not accounts: the operator typing the token only into a sign-in they
started from their own connector is what separates accounts (whether claude.ai
binds a sign-in's state to the browser session that started it is not documented).
Codes live in memory; a restart between sign-in and exchange fails that sign-in only.

The sign-in window: a code is issued only while the operator has opened sign-in,
so a sign-in link sent to them (phishing) fails at any other time. The window file
(``<token file>.oauth-window``, written by ``python -m anneal_memory.remote
oauth-window``) holds one integer, the UTC expiry epoch. Sign-in is open while that
file exists, parses, and its expiry is in the future and no more than
:data:`MAX_WINDOW_S` ahead (a far-future value is refused, not trusted); a missing,
unreadable or garbled file is closed. The first completed sign-in (a code issued)
deletes the file, so one opening allows one sign-in. Only the step that issues a
code is gated: registration, metadata and ``/token`` are not, so a code issued
inside the window still exchanges after it closes.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import html
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

logger = logging.getLogger("anneal-memory.remote")


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
MAX_WINDOW_S = 3600  # a window further out than this is refused, never trusted
_WINDOW_RE = re.compile(r"[0-9]{1,12}")
_WINDOW_CLOSED_PAGE = ("<p>Sign-in is closed: ask the operator to open it.</p>"
                       "<p>The operator opens sign-in on the server for a few minutes at a "
                       "time. If you did not start this sign-in from your own app, stop "
                       "here.</p>")
MAX_PENDING_CODES = 32
MAX_STATE_CHARS = 4096
MAX_FORM_BYTES = 16 * 1024
MAX_REGISTER_BYTES = 16 * 1024
# The hosts the Gemini app registers as redirect URIs (oauth-redirect,
# oauth-redirect-sandbox), read from public reports 2026-10-24
# (github.com/doobidoo/mcp-memory-service/issues/1553; discuss.ai.google.dev
# t/186405). The app also registers oauth-redirect-test URIs in the same request
# (github.com/BerriAI/litellm/issues/46640, 2026-10-15); those are dropped at
# registration, never added here. claude.ai, read 2026-10-25 from
# claude.com/docs/connectors/building/authentication.md ("register exactly this
# redirect URI: https://claude.ai/api/mcp/auth_callback"; Claude Code's http loopback
# redirect is not served here). ChatGPT, read 2026-10-25 from
# developers.openai.com/apps-sdk/build/auth (https://chatgpt.com/connector_platform_oauth_redirect,
# or https://chatgpt.com/connector/oauth/{callback_id} when the server does not do
# RFC 9207 issuer identification, which this one does not). Sign-in refuses any
# redirect outside this set.
REDIRECT_HOSTS = frozenset({"oauth-redirect.googleusercontent.com",
                            "oauth-redirect-sandbox.googleusercontent.com",
                            "claude.ai", "chatgpt.com"})
# claude.ai and chatgpt.com serve whole apps, not only redirects, so on them only
# the documented callback paths pass (the Google hosts above are redirect-only).
# A ChatGPT callback_id may differ when the connector is re-added; the OpenAI
# guide does not say. Then sign-in is refused until that connector's pin file is
# deleted.
REDIRECT_PATHS = {
    "claude.ai": re.compile(r"/api/mcp/auth_callback"),
    "chatgpt.com": re.compile(r"/connector_platform_oauth_redirect|/connector/oauth/[A-Za-z0-9_-]{1,128}"),
}
_VERIFIER_RE = re.compile(r"^[A-Za-z0-9._~-]{43,128}$")
_CHALLENGE_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")  # base64url(sha256), unpadded
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# A connector name as a pin file suffix, so never a "/"; a test holds it equal to
# the rule token names are created under.
_CONNECTOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
AUTH_METHODS = ("client_secret_basic", "client_secret_post", "none")
# What a ``check_token`` callback returns when the token file cannot be read,
# so a broken server is never reported to the operator as a wrong token.
UNAVAILABLE = object()
_INVALID_PIN = "\x00invalid pin"  # matches no redirect URI


def _uri_for_log(uri: Any) -> str:
    """scheme://host/path of a refused redirect URI, for the operator's log: no
    query, fragment or userinfo, so no state or credential reaches the journal."""
    if not isinstance(uri, str):
        return f"<{type(uri).__name__}>"
    try:
        parts = urllib.parse.urlsplit(uri)
        host = parts.hostname or ""
    except ValueError:
        return "<unparseable>"
    return ascii(f"{parts.scheme}://{host}{parts.path}"[:120])


# The paths a journal line may name verbatim. Any other path is logged by its
# length only, because a mistyped or prefixed token URL (/x/mcp/<token>) is
# exactly what a half-configured client sends.
_LOGGABLE_PATHS = frozenset({
    "/", "/mcp", "/register", "/authorize", "/token", "/favicon.ico",
    "/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp",
    "/.well-known/oauth-authorization-server", "/.well-known/oauth-authorization-server/mcp",
    "/.well-known/openid-configuration", "/.well-known/openid-configuration/mcp",
    "/mcp/.well-known/openid-configuration"})


def path_for_log(path: str) -> str:
    """A request path as a journal may hold it: never a query, never a token."""
    path = path.split("?", 1)[0]
    if path in _LOGGABLE_PATHS:
        return path
    if path.startswith("/mcp/"):
        return "/mcp(token-in-path)"  # that segment IS the token
    return f"(other path, {len(path)} chars)"


def _url_for_log(url: str, origin: str) -> str:
    """A client-sent URL for the journal: :func:`path_for_log` of it when its origin
    is ``origin``, else only its length (L3 1024+22: a host or scheme can carry a
    secret too)."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return f"(unparseable, {len(url)} chars)"
    if f"{parts.scheme}://{parts.netloc}".lower() != origin.lower():
        return f"(other origin, {len(url)} chars)"
    return origin + path_for_log(parts.path or "/")


# Token-request field names and grant types a journal line may name. Anything
# else is counted, never echoed: a JSON body sent as a form parses into ONE field
# whose name is the whole body, code and verifier included.
_TOKEN_FIELDS = frozenset({"grant_type", "code", "code_verifier", "redirect_uri", "client_id",
                           "client_secret", "resource", "refresh_token", "scope"})
_GRANT_TYPES = frozenset({"authorization_code", "refresh_token", "client_credentials",
                          "urn:ietf:params:oauth:grant-type:device_code"})


def _grant_shape(q: dict[str, str], basic: bool) -> str:
    """What a client sent to /token, by known name and shape only."""
    auth = ("basic" if basic else "post" if "client_secret" in q else "none") + (
        "+post" if basic and "client_secret" in q else "")
    known = sorted(k for k in q if k in _TOKEN_FIELDS)
    other = [k for k in q if k not in _TOKEN_FIELDS]
    fields = (",".join(known) or "-") + (f" +{len(other)} other ({sum(map(len, other))} chars)"
                                         if other else "")
    grant = q.get("grant_type")
    shown = "-" if grant is None else grant if grant in _GRANT_TYPES else \
        f"(other, {len(grant)} chars)"
    return f"client auth {auth}; fields {fields}; grant_type {shown}"


def _log_authorize_refusal(step: str, problem: str | None, q: dict[str, str] | None) -> None:
    redirect = _uri_for_log(q.get("redirect_uri")) if q is not None else "-"
    logger.info("oauth authorize refused (%s): %s; redirect %s", step, problem, redirect)


def redirect_ok(uri: Any) -> bool:
    if not isinstance(uri, str) or len(uri) > 2048:
        return False
    try:
        parts = urllib.parse.urlsplit(uri)
        port = parts.port
    except ValueError:
        return False
    path_rule = REDIRECT_PATHS.get(parts.hostname or "")
    # A bare trailing "?" or "#" parses as an empty query or fragment, so the raw
    # string is checked too.
    if path_rule is not None and ("?" in uri or "#" in uri
                                  or not path_rule.fullmatch(parts.path)):
        return False
    return (parts.scheme == "https" and parts.hostname in REDIRECT_HOSTS
            and port in (None, 443) and not parts.username and not parts.password
            and not parts.fragment and uri.isascii() and uri.isprintable())


def s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class OAuth:
    """Endpoints for one issuer. Each handler returns ``(status, headers, body)``."""

    def __init__(self, issuer: str, pin_path: str | os.PathLike[str] | None = None,
                 window_path: str | os.PathLike[str] | None = None) -> None:
        """``window_path`` is the sign-in window file; without one sign-in is
        always closed (an auth boundary fails closed)."""
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
        self._window_path = Path(window_path) if window_path is not None else None

    # -- the sign-in window --

    def window_closed(self) -> str | None:
        """Why sign-in is closed now, or None while the window is open. The reason
        is one of a few fixed phrases, safe for the journal."""
        if self._window_path is None:
            return "no sign-in window is configured"
        try:
            with open(self._window_path, "rb") as f:
                raw = f.read(65)
        except FileNotFoundError:
            return "the sign-in window is not open"
        except OSError:
            return "the sign-in window file is unreadable"
        try:
            text = raw.decode("ascii").strip()
        except UnicodeDecodeError:
            text = ""
        if len(raw) > 64 or not _WINDOW_RE.fullmatch(text):
            return "the sign-in window file is garbled"
        expiry, now = int(text), time.time()
        if expiry <= now:
            return "the sign-in window has expired"
        if expiry > now + MAX_WINDOW_S:
            return "the sign-in window is more than an hour ahead (refused)"
        return None

    def _use_window(self) -> str | None:
        """Close the window for the sign-in about to get a code: None when this
        sign-in took it, else why not. Deleting is the claim, so of two sign-ins
        racing (threads or processes) only the one whose unlink succeeds proceeds.
        Call under ``self._lock``."""
        closed = self.window_closed()
        if closed is not None:
            return closed
        assert self._window_path is not None
        try:
            os.unlink(self._window_path)
        except FileNotFoundError:
            return "the sign-in window was used by another sign-in"
        except OSError as e:
            # A window that cannot be deleted would allow a second sign-in, so
            # this one is refused too and the operator is told why.
            logger.warning("oauth sign-in window could not be closed (%s); sign-in refused",
                           type(e).__name__)
            return "the sign-in window file could not be deleted"
        # Best effort: a crash before this sync can bring the file back, and it
        # still expires by its own time.
        with contextlib.suppress(OSError):
            _fsync_dir_strict(self._window_path.parent)
        logger.info("oauth sign-in window used: closed until the operator opens it again")
        return None

    # -- the redirect pin --

    @staticmethod
    def _read_pin(path: Path) -> str | None:
        try:
            text = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        except (OSError, ValueError):  # unreadable or not ASCII: refuse every redirect
            return _INVALID_PIN
        return text if redirect_ok(text) else _INVALID_PIN

    def _connector_pin_path(self, connector: str) -> Path | None:
        """Where ``connector``'s pin lives, or None when its name cannot be a file
        name suffix (then nothing can pin, and :meth:`pinned` refuses every redirect)."""
        assert self._pin_path is not None
        if not isinstance(connector, str) or not _CONNECTOR_RE.fullmatch(connector):
            return None
        # Lowercased so the file is the same one on a case-insensitive filesystem.
        return self._pin_path.with_name(f"{self._pin_path.name}.{connector.lower()}")

    def pinned(self, connector: str | None = None) -> str | None:
        """The redirect URI a completed sign-in pinned, or None before one: the
        server-wide pin when that file exists, else ``connector``'s own (None when
        no connector is named). An unreadable pin, or a connector name that cannot
        be a file suffix, gives ``_INVALID_PIN``, which matches no redirect."""
        if self._pin_path is None:
            return None
        legacy = self._read_pin(self._pin_path)
        if legacy is not None or connector is None:
            return legacy
        path = self._connector_pin_path(connector)
        return _INVALID_PIN if path is None else self._read_pin(path)

    def _redirect_allowed(self, uri: Any, connector: str | None = None) -> bool:
        pin = self.pinned(connector)
        return redirect_ok(uri) and (pin is None or uri == pin)

    def _ensure_pinned(self, uri: str, connector: str) -> bool:
        """True when ``uri`` is ``connector``'s pinned redirect, pinning it if nothing
        is yet. The pin is written whole to a temporary file and linked into place,
        so it appears complete or not at all and the first link wins across
        processes; the directory is synced before a token is released. Raises
        OSError when it cannot be written. Call under ``self._lock``."""
        if self._pin_path is None:
            return True
        current = self.pinned(connector)
        if current is not None:
            if current != uri:
                return False
            # Synced on every release, not only the first: a pin linked by an
            # attempt whose directory sync failed is visible but maybe not durable.
            _fsync_dir_strict(self._pin_path.parent)
            return True
        path = self._connector_pin_path(connector)
        if path is None:
            return False
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="ascii") as f:
                f.write(uri + "\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass  # another process pinned first; compare below
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        _fsync_dir_strict(self._pin_path.parent)
        return self.pinned(connector) == uri

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
        if not isinstance(uris, list) or not uris or len(uris) > 16:
            logger.info("oauth register refused: redirect_uris is not a list of 1-16")
            return 400, {"error": "invalid_redirect_uri",
                         "error_description": "redirect_uris must be a list of 1 to 16 URIs"}
        # The Gemini app registers its redirect on several Google hosts at once, some
        # outside REDIRECT_HOSTS. The client is registered with the URIs that pass and
        # the rest are dropped (RFC 7591 §3.2.1 lets the server replace requested
        # metadata); sign-in still accepts only an allowed, and once pinned the pinned, URI.
        kept: list[str] = []
        dropped: list[Any] = []
        for u in uris:
            (kept if redirect_ok(u) else dropped).append(u)
        pinned = self.pinned()
        if not kept:
            refused = "no redirect URI is https on an allowed host"
        elif pinned is not None and pinned not in kept:
            refused = ("the pin file is unreadable" if pinned == _INVALID_PIN else
                       f"the pinned redirect {_uri_for_log(pinned)} is not among them; "
                       "rename the server-wide pin file to <its name>.<connector name, "
                       "lowercased> to pin only that connector")
        else:
            refused = ""
        shown = ", ".join(_uri_for_log(u) for u in dropped[:6]) + \
            (f" and {len(dropped) - 6} more" if len(dropped) > 6 else "")
        logger.info("oauth register %s: %d of %d redirect URIs allowed%s",
                    f"refused ({refused})" if refused else "accepted", len(kept), len(uris),
                    f"; dropped {shown}" if dropped else "")
        if refused:
            return 400, {"error": "invalid_redirect_uri",
                         "error_description": "redirect URIs must be https on an allowed "
                                              "host, and include the pinned one once pinned"}
        uris = kept
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
            return ("redirect_uri is not allowed (not https on an allowed callback, or not "
                    "the server-wide pinned one)")
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
        # The bare origin names this same server. OpenAI's auth guide (read
        # 2026-10-25) gives a bare-origin resource as its example; whether ChatGPT
        # keeps an entered /mcp path is not stated there, so both are accepted.
        return value in (self.resource, self.resource + "/", self.issuer, self.issuer + "/")

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
            _log_authorize_refusal("page", problem, q)
            return 400, _page("Sign-in refused", f"<p>{html.escape(problem or '')}</p>")
        closed = self.window_closed()
        if closed is not None:  # never ask for a token the submit would refuse
            _log_authorize_refusal("page", closed, q)
            return 403, _page("Sign-in is closed", _WINDOW_CLOSED_PAGE)
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
                 "<p><b>The first app a connector token signs in to becomes the only "
                 "app that token can connect.</b> Apps of one vendor can share a return "
                 "address, so only continue if you started this from your own app.</p>")
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
            _log_authorize_refusal("submit", problem, q)
            return 400, _page("Sign-in refused", f"<p>{html.escape(problem)}</p>")
        assert q is not None
        token = q.get("token", "").strip()
        checked = check_token(token) if token else None
        if checked is UNAVAILABLE:
            return 503, _page("Unavailable", "<p>The server cannot read its token file.</p>")
        if checked is None:  # tokens are 32 random bytes: no delay buys anything
            return 401, self._form(q, "That token is not valid here.")
        if not self._redirect_allowed(q["redirect_uri"], checked):
            if self.pinned(checked) == _INVALID_PIN:
                _log_authorize_refusal("submit", f"the pin file for {checked} is unreadable", q)
                return 400, _page("Sign-in refused", (
                    f"<p>The pin file for the {html.escape(checked)} connector cannot be "
                    "read. Fix or delete it on the server.</p>"))
            _log_authorize_refusal("submit", f"redirect is not the one pinned for {checked}", q)
            return 400, _page("Sign-in refused", (
                f"<p>The {html.escape(checked)} connector token is already connected to "
                "another app. Delete its pin file on the server to connect a new one.</p>"))
        code = secrets.token_urlsafe(32)
        now = time.monotonic()
        with self._lock:
            for c in [c for c, v in self._codes.items() if v["exp"] <= now]:
                del self._codes[c]
            if len(self._codes) >= MAX_PENDING_CODES:
                return 503, _page("Busy", "<p>Too many sign-ins in progress; try again shortly.</p>")
            # The window gates the code itself: checked and used (deleted) here,
            # under the lock and after every other check, so a refusal above never
            # spends it and no code is issued without taking it.
            closed = self._use_window()
            if closed is not None:
                _log_authorize_refusal("submit", closed, q)
                return 403, _page("Sign-in is closed", _WINDOW_CLOSED_PAGE)
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
            logger.info("oauth token refused: the form is not ASCII form-encoding")
            return 400, {"error": "invalid_request"}
        q = self._fields(raw)
        if q is None:
            logger.info("oauth token refused: a field is repeated")
            return 400, {"error": "invalid_request"}
        sent = _grant_shape(q, basic is not None)
        if q.get("grant_type") != "authorization_code":
            logger.info("oauth token refused: unsupported grant (%s)", sent)
            return 400, {"error": "unsupported_grant_type"}
        client_id = q.get("client_id")
        if basic is not None:
            try:
                padded = basic + "=" * (-len(basic) % 4)
                user = base64.b64decode(padded, validate=True).decode("ascii").split(":", 1)[0]
                basic_id = urllib.parse.unquote_plus(user)  # RFC 6749 §2.3.1: form-encoded
            except (ValueError, UnicodeDecodeError):
                logger.info("oauth token refused: Basic credentials do not decode (%s)", sent)
                return 401, {"error": "invalid_client"}
            if client_id is not None and client_id != basic_id:
                logger.info("oauth token refused: client_id differs from Basic (%s)", sent)
                return 400, {"error": "invalid_request"}
            client_id = basic_id
        if not isinstance(client_id, str) or not _CLIENT_ID_RE.fullmatch(client_id):
            logger.info("oauth token refused: missing or malformed client_id (%s)", sent)
            return 400, {"error": "invalid_request"}  # also keeps compare_digest on ASCII
        with self._lock:
            entry = self._codes.pop(q.get("code", ""), None)  # single use, even when refused
        verifier = q.get("code_verifier", "")
        if entry is None or entry["exp"] <= time.monotonic():
            why = "unknown, used or expired code"
        elif not hmac.compare_digest(client_id, entry["client_id"]):
            why = "client_id is not the signed-in one"
        # optional with PKCE (OAuth 2.1); when sent it must be the signed-in one
        elif q.get("redirect_uri", entry["redirect_uri"]) != entry["redirect_uri"]:
            why = "redirect_uri is not the signed-in one"
        elif not _VERIFIER_RE.fullmatch(verifier) \
                or not hmac.compare_digest(s256(verifier), entry["challenge"]):
            why = "PKCE verifier does not match"
        else:
            why = ""
        if why:
            logger.info("oauth token refused: %s (%s)", why, sent)
            return 400, {"error": "invalid_grant"}
        assert entry is not None  # the first branch above refuses a missing entry
        resource = q.get("resource")
        if resource is not None and not self._resource_ok(resource) or \
                entry["resource"] is not None and resource is not None \
                and resource.rstrip("/") != entry["resource"].rstrip("/"):  # both already exact
            logger.info("oauth token refused: resource %s is not this server's (%s)",
                        _url_for_log(resource, self.issuer), sent)
            return 400, {"error": "invalid_target"}
        still = check_token(entry["token"])
        if still is UNAVAILABLE:
            return 503, {"error": "server_error"}
        if still is None:  # revoked since the sign-in
            logger.info("oauth token refused: the token was revoked after sign-in")
            return 400, {"error": "invalid_grant"}
        # The pin binds codes issued before it existed too: a second app's code for
        # the same connector, signed in while nothing was pinned, is refused once the
        # first one pinned.
        try:
            with self._lock:
                pinned_here = self._ensure_pinned(entry["redirect_uri"], still)
        except OSError:
            return 503, {"error": "server_error"}
        if not pinned_here:
            logger.info("oauth token refused: redirect %s is not the one pinned for %s",
                        _uri_for_log(entry["redirect_uri"]), still)
            return 400, {"error": "invalid_grant"}
        logger.info("oauth token issued (%s)", sent)
        logger.info("oauth sign-in completed for %s (connector %s)",
                    _uri_for_log(entry["redirect_uri"]), still)
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
