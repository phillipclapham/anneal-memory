"""anneal_memory.rederive — STATE lines that prove themselves when read.

A ``derived-state`` section (the ``project`` schema's ``## State``) holds
present-tense claims, each ending with an annotation:

    [derive: COMMAND => EXPECTED]   value claim: stdout == EXPECTED (exit 0 or 1)
    [derive: COMMAND]               truth claim: exit 0 agrees, exit 1 disagrees
    (any other exit is an error in both forms)
    [judged: WHO, WHEN, AGAINST]    a judgement; accepted, never executed

:func:`rederive_continuity` runs each command and flags the line inline;
:func:`check_state_for_save` is the save-time gate.

⚠ SECURITY. This module executes commands stored in a memory file that an
agent wrote, so every command is untrusted input. The containment is designed
in ``docs/rederive.md``; the rules it enforces here are:

1. nothing runs unless the store's database path is bound to a root directory
   in the per-user trust file (outside the store; no MCP tool writes it);
2. no shell: ``shlex`` + ``shell=False``, and shell metacharacters are refused;
3. an argument-by-argument allowlist of read-only command forms;
4. file arguments must resolve inside the root, and grep/wc read only files
   git tracks, never their content (grep answers with a count or yes/no);
5. a from-scratch environment, the root as cwd, stdin closed, one pinned ref;
6. per-command timeout and output cap, per-load time budget and line cap;
7. command output reaches the text only as a sanitised, truncated one-liner.

Widening the allowlist is a security change: read ``docs/rederive.md`` first.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .schema import SectionSpec

__all__ = [
    "DeriveRefused",
    "GIT_SUBCOMMANDS",
    "PROGRAMS",
    "strip_flag",
    "strip_rederive_output",
    "Annotation",
    "LineResult",
    "RederiveReport",
    "parse_annotation",
    "validate_command",
    "derived_state_lines",
    "rederive_text",
    "rederive_continuity",
    "check_state_for_save",
    "trust_file_path",
    "trusted_root",
    "allow_store",
    "revoke_store",
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_LOAD_BUDGET",
    "DEFAULT_MAX_LINES",
    "DEFAULT_OUTPUT_CAP",
]

DEFAULT_COMMAND_TIMEOUT = 5.0  # seconds per command
DEFAULT_LOAD_BUDGET = 30.0  # seconds per load, across all lines
DEFAULT_MAX_LINES = 64  # derive commands run per load
DEFAULT_OUTPUT_CAP = 64 * 1024  # bytes of stdout (and of stderr) per command
_FLAG_VALUE_MAX = 120  # chars of command output shown in an inline flag

TRUST_ENV = "ANNEAL_MEMORY_DERIVE_TRUST"
REF_TOKEN = "@REF"


class DeriveRefused(ValueError):
    """A derive command is outside the allowlist. Never executed."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_DERIVE_OPEN = "[derive:"
_JUDGED_OPEN = "[judged:"
_EXPECT_SEP = " => "


@dataclass(frozen=True)
class Annotation:
    kind: Literal["derive", "judged"]
    claim: str
    command: str = ""
    expected: str | None = None  # None = truth claim (exit status)
    judgement: str = ""


# The inline flag rederive_text appends ("  ✓", "  ⚠ STALE (…)", …), so text
# loaded with --rederive and carried into a wrap still parses.
_APPENDED_FLAG = re.compile(r"  (?:✓|⚠ [A-Z][A-Z ]*[A-Z]|⛔ REFUSED)(?: \(.*\))?$")


def strip_flag(line: str) -> str:
    """``line`` without a flag :func:`rederive_text` appended to it; any other
    line comes back byte for byte."""
    m = _APPENDED_FLAG.search(line)
    return line[: m.start()] if m else line


# Every text rederive_text returns starts with this tag. At save the header is
# removed only when the whole line matches what rederive_text writes, so an
# authored note is never deleted; flags are removed from State lines always,
# since nothing may follow an annotation's closing "]" (a forged "  ✓" is
# never persisted).
_ENVELOPE = "> [anneal re-derive] "
_HEADER_LINE = re.compile(
    r"^> \[anneal re-derive\] (?:"
    r"[1-9]\d* STATE line\(s\) in [^\n]+?(?: at [0-9a-f]{40})?"
    r"(?:: \d+ [a-z]+(?:, \d+ [a-z]+)*)?"
    r"\. git lines read the pinned ref only where they use @REF; grep, wc and test "
    r"read the working tree\."
    r"|not enabled for this store; STATE lines were not checked "
    r"\(see `anneal-memory derive allow`\)\.)$"
)


def strip_rederive_output(text: str, schema: list[SectionSpec]) -> str:
    """Remove everything :func:`rederive_text` added: its header line (and the
    blank line after it) and every flag it appended to a State line. Saving
    loaded text back must never persist a verdict that is true only at load."""
    lines = text.split("\n")
    # Exactly what rederive_text writes: the header line, then a blank line.
    # It is removed where re-derive puts it (the first line) or where a
    # composer who keeps the "# " title first puts it (right under the title):
    # prepare_wrap hands it to a composer (spore-1233). Anywhere else a
    # matching line is authored text and is kept.
    for at in (0, 1):
        if at == 1 and not (lines and lines[0].startswith("# ")):
            break
        if (
            len(lines) > at + 1
            and _HEADER_LINE.match(lines[at].rstrip("\r"))
            and not lines[at + 1].strip()
        ):
            del lines[at : at + 2]
            break
    joined = "\n".join(lines)
    for idx, line in derived_state_lines(joined, schema):
        cr = "\r" if line.endswith("\r") else ""
        lines[idx] = strip_flag(line[:-1] if cr else line) + cr
    return "\n".join(lines)


def parse_annotation(line: str) -> Annotation | None:
    """Parse the trailing annotation of a State line, or ``None`` if absent.

    The annotation is the LAST ``[derive:`` / ``[judged:`` on the line and must
    close with the line's final ``]`` (after any re-derive flag is removed).
    """
    if line.endswith("\r"):
        line = line[:-1]
    stripped = strip_flag(line)
    if not stripped.endswith("]"):
        return None
    d = stripped.rfind(_DERIVE_OPEN)
    j = stripped.rfind(_JUDGED_OPEN)
    start = max(d, j)
    if start < 0:
        return None
    claim = stripped[:start].rstrip()
    body = stripped[start + len(_DERIVE_OPEN if start == d else _JUDGED_OPEN) : -1].strip()
    if start == j:
        if not body:
            return None
        return Annotation(kind="judged", claim=claim, judgement=body)
    if _EXPECT_SEP in body:
        command, expected = body.rsplit(_EXPECT_SEP, 1)  # a pattern may hold " => "
        return Annotation(
            kind="derive", claim=claim, command=command.strip(), expected=expected.strip()
        )
    return Annotation(kind="derive", claim=claim, command=body)


# --------------------------------------------------------------------------
# The allowlist (containment rules 2-4)
# --------------------------------------------------------------------------

# Rule 2: none of these is needed by an allowed form; their presence is an
# attempt, refused before shlex ever sees the command.
_FORBIDDEN_CHARS = set(";&|$`<>\\")


@dataclass(frozen=True)
class _GitForm:
    bare: frozenset[str] = frozenset()  # flags taking no value
    valued: frozenset[str] = frozenset()  # flags accepted only as --flag=value
    digit_count: bool = False  # accepts -<N> (e.g. log -1)


# Only forms whose output is an exit status, a commit id, a count, a tag
# name or a tracked path: nothing git prints here can carry file or commit
# content, whatever the repository's configuration says. log, cat-file -p,
# ls-tree and for-each-ref were removed after review twice found a new way
# for repository config or metadata to reach past the root through them
# (format.pretty, mailmap, alternates); the construct shrinks rather than
# growing another filter (spore-813).
_GIT_FORMS: dict[str, _GitForm] = {
    "rev-parse": _GitForm(
        bare=frozenset({"--verify", "-q", "--quiet", "--short", "--abbrev-ref", "--symbolic-full-name"}),
        valued=frozenset({"--short"}),
    ),
    "describe": _GitForm(
        bare=frozenset({"--tags", "--exact-match", "--always", "--long", "--contains"}),
        valued=frozenset({"--abbrev", "--match", "--exclude"}),
    ),
    "rev-list": _GitForm(
        bare=frozenset({"--count", "--first-parent", "--merges", "--no-merges"}),
        valued=frozenset({"--max-count", "--since", "--until", "--author", "--grep"}),
        digit_count=True,
    ),
    "merge-base": _GitForm(bare=frozenset({"--is-ancestor"})),
    "cat-file": _GitForm(bare=frozenset({"-e"})),
    "ls-files": _GitForm(bare=frozenset({"--error-unmatch", "--cached"})),
}

GIT_SUBCOMMANDS: frozenset[str] = frozenset(_GIT_FORMS)

# grep may only answer with a count or a yes/no, never print file content:
# one of the answer flags is required, and the rest only shape the match.
_GREP_ANSWER = set("cqlL")
_GREP_SHORT = _GREP_ANSWER | set("FEiwxsv")
_WC_SHORT = set("lcwm")
_TEST_OPS = frozenset({"-e", "-f", "-d", "-s"})


def _check_path(arg: str, root: Path | None) -> None:
    """Rule 4: a file argument is relative and resolves inside ``root``."""
    if not arg or arg.startswith("-"):
        raise DeriveRefused(f"path argument {arg!r} must not be empty or begin with '-'")
    p = Path(arg)
    if p.is_absolute() or arg.startswith("~"):
        raise DeriveRefused(f"path argument {arg!r} must be relative to the root")
    if ".." in p.parts:
        raise DeriveRefused(f"path argument {arg!r} must not contain '..'")
    if any(part.casefold() == ".git" for part in p.parts):
        raise DeriveRefused(f"path argument {arg!r} must not be inside .git")
    if p.drive or p.root:
        raise DeriveRefused(f"path argument {arg!r} must be relative to the root")
    if root is not None:
        root_real = os.path.realpath(root)
        real = os.path.realpath(os.path.join(root_real, arg))
        try:
            inside = os.path.commonpath([real, root_real]) == root_real
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            raise DeriveRefused(f"path argument {arg!r} resolves outside the root")
        # No symlink anywhere on the path: a TRACKED symlink (link -> .env,
        # link -> .git/config) resolves inside the root and passes the tracked
        # check, yet opens a file the rules promise is unreadable (L3).
        walk = root_real
        for part in p.parts:
            walk = os.path.join(walk, part)
            if os.path.islink(walk):
                raise DeriveRefused(f"path argument {arg!r} passes through a symlink")


def _validate_git(args: list[str]) -> None:
    if not args:
        raise DeriveRefused("git needs a subcommand")
    sub, rest = args[0], args[1:]
    form = _GIT_FORMS.get(sub)
    if form is None:
        # Also catches every global option (-c, -C, --exec-path, --git-dir, ...):
        # the subcommand must be the very first argument.
        raise DeriveRefused(
            f"git {sub!r} is not an allowed subcommand "
            f"(allowed: {', '.join(sorted(_GIT_FORMS))})"
        )
    after_dashdash = False
    for a in rest:
        if after_dashdash:
            if a.startswith("-"):
                raise DeriveRefused(f"path {a!r} must not begin with '-'")
            continue
        if a == "--":
            after_dashdash = True
            continue
        if a.startswith("-"):
            if form.digit_count and re.fullmatch(r"-[0-9]+", a):
                continue
            if "=" in a:
                name, value = a.split("=", 1)
                if name not in form.valued:
                    raise DeriveRefused(f"git {sub} flag {name!r}= is not allowed")
                continue
            if a not in form.bare:
                raise DeriveRefused(f"git {sub} flag {a!r} is not allowed")
            continue
        # A positional (revision, object or path). Git keeps it inside the repo.
    if sub == "merge-base" and "--is-ancestor" not in rest:
        raise DeriveRefused("git merge-base is allowed only with --is-ancestor")
    if sub == "cat-file" and "-e" not in rest:
        raise DeriveRefused("git cat-file is allowed only with -e")


def _validate_short_flags(prog: str, flag: str, allowed: set[str]) -> None:
    if not re.fullmatch(r"-[A-Za-z]+", flag) or not set(flag[1:]) <= allowed:
        raise DeriveRefused(f"{prog} flag {flag!r} is not allowed")


def _validate_grep(args: list[str], root: Path | None) -> None:
    i = 0
    flags: set[str] = set()
    while i < len(args) and args[i].startswith("-"):
        _validate_short_flags("grep", args[i], _GREP_SHORT)
        flags |= set(args[i][1:])
        i += 1
    if not flags & _GREP_ANSWER:
        raise DeriveRefused("grep needs one of -c, -q, -l, -L (it may not print file content)")
    rest = args[i:]
    if len(rest) < 2:
        raise DeriveRefused("grep needs a PATTERN and at least one PATH")
    if len(rest) - 1 > _MAX_PATHS:
        raise DeriveRefused(f"grep takes at most {_MAX_PATHS} paths")
    for path in rest[1:]:
        _check_path(path, root)


def _validate_wc(args: list[str], root: Path | None) -> None:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        _validate_short_flags("wc", args[i], _WC_SHORT)
        i += 1
    if i == len(args):
        raise DeriveRefused("wc needs at least one PATH")
    if len(args) - i > _MAX_PATHS:
        raise DeriveRefused(f"wc takes at most {_MAX_PATHS} paths")
    for path in args[i:]:
        _check_path(path, root)


def _validate_test(args: list[str], root: Path | None) -> None:
    if len(args) != 2 or args[0] not in _TEST_OPS:
        raise DeriveRefused(
            f"test is allowed only as 'test {'|'.join(sorted(_TEST_OPS))} PATH'"
        )
    _check_path(args[1], root)


_VALIDATORS = {
    "git": lambda args, root: _validate_git(args),
    "grep": _validate_grep,
    "wc": _validate_wc,
    "test": _validate_test,
}
PROGRAMS: frozenset[str] = frozenset(_VALIDATORS)


def _check_repo_shape(root: str) -> str | None:
    """Why ``root`` is not a plain repository, or ``None``. Git follows
    repository metadata wherever it points (a gitfile, commondir, a symlinked
    object store, alternates), so git runs only where all of it is a real
    directory tree inside the root (codex L3 round 2)."""
    git = os.path.join(root, ".git")
    if os.path.islink(git) or not os.path.isdir(git):
        return "the root's .git is not a plain directory (a worktree or gitfile?)"
    for rel in ("objects", "refs"):
        if os.path.islink(os.path.join(git, rel)):
            return f".git/{rel} is a symlink"
    for rel in ("commondir", os.path.join("objects", "info", "alternates"),
                os.path.join("objects", "info", "http-alternates")):
        if os.path.lexists(os.path.join(git, rel)):
            return f".git/{rel} exists; git would read outside the root"
    # The repository's config may not pull in another file.
    try:
        with open(os.path.join(git, "config"), encoding="utf-8", errors="replace") as f:
            if re.search(r"^\s*\[\s*include", f.read(), re.IGNORECASE | re.MULTILINE):
                return ".git/config includes another file; git would read outside the root"
    except FileNotFoundError:
        pass
    except OSError as e:
        return f".git/config is unreadable ({e})"
    # Git follows a symlink anywhere in its metadata (a loose ref, HEAD,
    # packed-refs, an objects fan-out directory): one that leaves the root
    # is a read outside it (gpt-oss L3 round 3, reproduced).
    root_real = os.path.realpath(root)
    for dirpath, dirnames, filenames in os.walk(git, followlinks=False):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full) and not _inside(os.path.realpath(full), root_real):
                return f"{os.path.relpath(full, root)} is a symlink out of the root"
    return None


class _TrackError(Exception):
    """The tracked check itself failed (no git, no repository, timeout)."""


class _NotTracked(Exception):
    """A lexically allowed path that git does not track: the claim is judged
    stale (the file left git, or never entered it), and nothing is read."""


_MAX_PATHS = 8  # path arguments per grep/wc line
_MAX_COMMAND_CHARS = 400


def _require_tracked(argv: list[str], root: str, timeout: float, shape: str | None) -> None:
    """grep and wc read only files git tracks, so an untracked or ignored
    file inside the root (a .env, a key) is never read. Needs a git root."""
    if argv[0] == "grep":
        i = 1
        while argv[i].startswith("-"):
            i += 1
        paths = argv[i + 1 :]
    elif argv[0] == "wc":
        paths = [a for a in argv[1:] if not a.startswith("-")]
    else:
        return
    if shape:
        raise _TrackError(shape)
    for path in paths:
        try:
            st = os.lstat(os.path.join(root, path))
        except FileNotFoundError:
            continue  # judged below: not a tracked file
        if stat.S_ISDIR(st.st_mode):
            raise DeriveRefused(f"path argument {path!r} is a directory; name a file")
        if not stat.S_ISREG(st.st_mode):
            raise DeriveRefused(f"path argument {path!r} is not a regular file")
    res = _run_bounded(
        ["git", *_GIT_HARDENING, "ls-files", "-s", "-z", "--", *paths],
        root, timeout, 256 * 1024,
    )
    if res.spawn_error or res.timed_out or res.overflowed or res.returncode != 0:
        why = res.spawn_error or ("timed out" if res.timed_out else f"exit {res.returncode}")
        raise _TrackError(f"cannot check that the paths are tracked ({why})")
    # Exact names only, as regular files: a gitlink (160000), a tracked
    # symlink (120000) or a directory prefix match is not "tracked".
    regular: set[str] = set()
    for entry in res.stdout.split(b"\0"):
        if not entry:
            continue
        meta, _, name = entry.partition(b"\t")
        mode = meta.split(b" ", 1)[0]
        if mode in (b"100644", b"100755"):
            regular.add(name.decode("utf-8", "surrogateescape"))
    for path in paths:
        if os.path.normpath(path) not in regular:
            raise _NotTracked("a path is not a regular file git tracks in the root")


def validate_command(command: str, root: Path | None = None) -> list[str]:
    """Return the argv for an allowed ``command``; raise :class:`DeriveRefused`.

    With ``root=None`` only the lexical checks run (the save-time static gate
    on an untrusted store); with a root, path arguments are also resolved
    through symlinks and must stay inside it.
    """
    if not command or not command.strip():
        raise DeriveRefused("empty derive command")
    if len(command) > _MAX_COMMAND_CHARS:
        raise DeriveRefused(f"command longer than {_MAX_COMMAND_CHARS} characters")
    bad = sorted({c for c in command if c in _FORBIDDEN_CHARS or (ord(c) < 32) or ord(c) == 127})
    if bad:
        raise DeriveRefused(
            "shell metacharacters or control characters are not allowed: "
            + " ".join(repr(c) for c in bad)
        )
    try:
        argv = shlex.split(command)
    except ValueError as e:
        raise DeriveRefused(f"unparseable command: {e}") from None
    if not argv:
        raise DeriveRefused("empty derive command")
    prog, args = argv[0], argv[1:]
    validator = _VALIDATORS.get(prog)
    if validator is None:
        raise DeriveRefused(
            f"program {prog!r} is not allowed (allowed: {', '.join(sorted(_VALIDATORS))})"
        )
    validator(args, root)
    return argv


# --------------------------------------------------------------------------
# Running (containment rules 5-6)
# --------------------------------------------------------------------------

_GIT_HARDENING = [
    "--no-pager",
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "protocol.allow=never",
    "-c", "log.showSignature=false",
    "-c", "log.mailmap=false",
    "-c", "mailmap.file=",
    "-c", "mailmap.blob=",
    "-c", "gpg.program=false",
    "-c", "gpg.ssh.program=false",
    "-c", "gpg.x509.program=false",
]


def _child_env(root: str | None = None) -> dict[str, str]:
    # Absolute PATH entries only: a relative entry (".", "bin") would resolve
    # against the root and run a program committed to the repository.
    path = os.pathsep.join(
        p for p in os.environ.get("PATH", os.defpath).split(os.pathsep) if os.path.isabs(p)
    )
    env = {
        "PATH": path,
        "LC_ALL": "C",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
        # Overrides repo-level protocol.<name>.allow, which -c protocol.allow
        # does not (L1, measured 2026-09-30).
        "GIT_ALLOW_PROTOCOL": "none",
        # A path argument is a file name, never a glob or :(magic) pathspec.
        "GIT_LITERAL_PATHSPECS": "1",
        # Only the repository's own config: no ~/.gitconfig, XDG or system
        # file (codex L3 round 3).
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    home = os.environ.get("HOME")
    if home:
        env["HOME"] = home
    if root is not None:
        # No PATH entry inside the root: git's own helpers (gpg, ...) are
        # found through PATH too, and must never be repository files.
        root_real = os.path.realpath(root)
        env["PATH"] = os.pathsep.join(
            p for p in env["PATH"].split(os.pathsep)
            if not _inside(os.path.realpath(p), root_real)
        )
        # git must not discover a repository above the root (a parent repo,
        # a dotfiles repo in $HOME): the root is the boundary.
        env["GIT_CEILING_DIRECTORIES"] = os.path.dirname(root)
    if os.name == "nt":  # Windows needs these to start processes at all
        for k in ("SYSTEMROOT", "COMSPEC", "PATHEXT", "USERPROFILE"):
            if k in os.environ:
                env[k] = os.environ[k]
    return env


@dataclass
class _RunOutcome:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    overflowed: bool = False
    spawn_error: str | None = None


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_bounded(argv: list[str], cwd: str, timeout: float, cap: int) -> _RunOutcome:
    env = _child_env(cwd)
    exe = shutil.which(argv[0], path=env["PATH"])
    if exe is None:
        return _RunOutcome(None, b"", b"", spawn_error=f"{argv[0]!r} not found on PATH")
    exe_real = os.path.realpath(exe)
    cwd_real = os.path.realpath(cwd)
    try:
        exe_inside = os.path.commonpath([exe_real, cwd_real]) == cwd_real
    except ValueError:
        exe_inside = False
    if exe_inside:
        return _RunOutcome(None, b"", b"", spawn_error=f"{argv[0]!r} resolves inside the root")
    if os.name == "nt" and not exe_real.lower().endswith(".exe"):
        # a .bat/.cmd wrapper would run through cmd.exe, a shell
        return _RunOutcome(None, b"", b"", spawn_error=f"{argv[0]!r} is not a .exe")
    kw: dict = dict(
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        shell=False,
    )
    if os.name == "posix":
        kw["start_new_session"] = True  # so a kill reaches the whole group
    try:
        proc = subprocess.Popen([exe, *argv[1:]], **kw)
    except OSError as e:
        return _RunOutcome(None, b"", b"", spawn_error=str(e))

    out, err = bytearray(), bytearray()
    overflow = threading.Event()
    # Popen was given stdout=stderr=PIPE, so both streams exist; say so for the type checker.
    assert proc.stdout is not None and proc.stderr is not None

    def pump(fd: int, buf: bytearray) -> None:
        # ``fd`` is this thread's own dup, closed here, so closing the Popen
        # stream below can never leave this read on a recycled descriptor.
        try:
            _pump(fd, buf)
        finally:
            os.close(fd)

    def _pump(fd: int, buf: bytearray) -> None:
        while True:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            if len(buf) > cap:
                overflow.set()  # the main thread alone kills and reaps
                break

    threads = [
        threading.Thread(target=pump, args=(os.dup(proc.stdout.fileno()), out), daemon=True),
        threading.Thread(target=pump, args=(os.dup(proc.stderr.fileno()), err), daemon=True),
    ]
    for t in threads:
        t.start()
    timed_out = False
    end = time.monotonic() + timeout
    while True:  # only this thread waits, kills or reaps, so a kill never hits a reused pid
        try:
            proc.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            pass
        if overflow.is_set() or time.monotonic() >= end:
            timed_out = not overflow.is_set()
            _kill(proc)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            break
    for t in threads:
        t.join(timeout=2)
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except OSError:
            pass
    return _RunOutcome(
        proc.returncode,
        bytes(out[:cap]),
        bytes(err[:cap]),
        timed_out=timed_out,
        overflowed=overflow.is_set(),
    )


def _one_line(data: bytes | str, limit: int = _FLAG_VALUE_MAX) -> str:
    """Rule 7: output reaches the document only as a quoted one-liner."""
    s = data.decode("utf-8", "replace") if isinstance(data, bytes) else data
    s = "".join(ch if ch.isprintable() else " " for ch in s)
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[: limit - 1] + "…"
    return repr(s)


def _normalise(s: str) -> str:
    return " ".join(s.split())


# --------------------------------------------------------------------------
# Section walking
# --------------------------------------------------------------------------


def _derived_headings(schema: list[SectionSpec]) -> list[str]:
    return [s["heading"].lower() for s in schema if s["role"] == "derived-state"]


def derived_state_lines(text: str, schema: list[SectionSpec]) -> list[tuple[int, str]]:
    """``(index, line)`` for every content line inside a derived-state section.

    Blank lines are skipped; every other line, a subheading included, is a
    claim and needs an annotation. A section ends at the next ``## `` header.
    """
    headings = _derived_headings(schema)
    if not headings:
        return []
    out: list[tuple[int, str]] = []
    inside = False
    for i, line in enumerate(text.split("\n")):
        if line.startswith("## "):
            low = line[3:].strip().lower()
            inside = any(
                re.search(rf"(?<!\w){re.escape(h)}(?!\w)", low) for h in headings
            )
            continue
        if not inside or not line.strip():
            continue
        out.append((i, line))
    return out


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

Status = Literal["ok", "stale", "error", "refused", "judged", "unannotated", "skipped"]

_FLAGS: dict[str, str] = {
    "ok": "✓",
    "stale": "⚠ STALE",
    "error": "⚠ DERIVE ERROR",
    "refused": "⛔ REFUSED",
    "judged": "",
    "unannotated": "⚠ NO DERIVE",
    "skipped": "⚠ NOT DERIVED",
}


@dataclass(frozen=True)
class LineResult:
    index: int  # 0-based line index in the continuity text
    line: str
    status: Status
    detail: str = ""
    # True only when a check actually ran and answered: the command exited with
    # a status it could be judged on, or the tracked-file check found the file
    # gone from git. A refusal, a shape or ref error, a spawn failure, a timeout
    # or an overflow leaves it False (spore-1233).
    ran: bool = False

    @property
    def flag(self) -> str:
        base = _FLAGS[self.status]
        if not base:
            return ""
        return f"{base} ({self.detail})" if self.detail else base


@dataclass
class RederiveReport:
    text: str  # the continuity with inline flags
    enabled: bool  # False = store not opted in; nothing ran
    ref: str | None = None  # commit id @REF resolved to
    root: str | None = None
    results: list[LineResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    @property
    def ran(self) -> int:
        """Lines whose check actually ran and answered (``LineResult.ran``).
        Zero means nothing was checked (spore-1233)."""
        return sum(1 for r in self.results if r.ran)

    @property
    def clean(self) -> bool:
        """At least one State check ran, and every line holds or is judged.
        False when nothing ran."""
        return (
            self.enabled
            and self.ran > 0
            and all(r.status in ("ok", "judged") for r in self.results)
        )


def _judge_line(
    idx: int,
    line: str,
    root: str,  # the only caller passes the resolved allowed root; a None root never executes
    ref: str | None,
    ref_error: str | None,
    timeout: float,
    cap: int,
    shape: str | None = None,
) -> LineResult:
    ann = parse_annotation(line)
    if ann is None:
        return LineResult(idx, line, "unannotated", "no [derive: …] or [judged: …]")
    if ann.kind == "judged":
        return LineResult(idx, line, "judged")
    try:
        argv = validate_command(ann.command, Path(root) if root else None)
        if root is not None:
            _require_tracked(argv, root, timeout, shape)
    except DeriveRefused as e:
        return LineResult(idx, line, "refused", str(e))
    except _NotTracked as e:
        return LineResult(idx, line, "stale", str(e), ran=True)
    except _TrackError as e:
        return LineResult(idx, line, "error", str(e))
    if argv[0] == "git":
        if shape:
            return LineResult(idx, line, "error", shape)
        if any(REF_TOKEN in a for a in argv):
            if ref is None:
                return LineResult(idx, line, "error", f"{REF_TOKEN} unresolved: {ref_error}")
            argv = [a.replace(REF_TOKEN, ref) for a in argv]
        argv = ["git", *_GIT_HARDENING, *argv[1:]]
    res = _run_bounded(argv, root, timeout, cap)
    if res.spawn_error:
        return LineResult(idx, line, "error", res.spawn_error)
    if res.timed_out:
        return LineResult(idx, line, "error", f"timed out after {timeout:g}s")
    if res.overflowed:
        return LineResult(idx, line, "error", f"output exceeded {cap} bytes")
    rc = res.returncode
    if ann.expected is None:
        if rc == 0:
            return LineResult(idx, line, "ok", ran=True)
        if rc == 1:
            return LineResult(idx, line, "stale", "exit 1", ran=True)
        return LineResult(idx, line, "error", f"exit {rc}: {_one_line(res.stderr)}", ran=True)
    # Exit 1 is "false / no match" for every allowed program (grep -c prints
    # 0 and exits 1), so a value claim still compares its output; 2+ is an
    # error (git uses 128/129).
    if rc not in (0, 1):
        return LineResult(idx, line, "error", f"exit {rc}: {_one_line(res.stderr)}", ran=True)
    got = _normalise(res.stdout.decode("utf-8", "replace"))
    if got == _normalise(ann.expected):
        return LineResult(idx, line, "ok", ran=True)
    return LineResult(
        idx, line, "stale", f"now {_one_line(got)}, claimed {_one_line(ann.expected)}", ran=True
    )


def _resolve_ref(root: str, ref: str | None, timeout: float) -> tuple[str | None, str | None]:
    target = ref or "HEAD"
    try:
        validate_command(f"git rev-parse --verify {shlex.quote(target)}")
    except DeriveRefused as e:
        return None, str(e)
    if target.startswith("-"):
        return None, f"ref {target!r} must not begin with '-'"
    argv = ["git", *_GIT_HARDENING, "rev-parse", "--verify", "--end-of-options", f"{target}^{{commit}}"]
    res = _run_bounded(argv, root, timeout, 4096)
    if res.spawn_error or res.timed_out or res.returncode != 0:
        why = res.spawn_error or ("timed out" if res.timed_out else _one_line(res.stderr))
        return None, f"cannot resolve {target!r}: {why}"
    return res.stdout.decode().strip(), None


def rederive_text(
    text: str,
    schema: list[SectionSpec],
    root: str | os.PathLike | None,
    *,
    ref: str | None = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
    budget: float = DEFAULT_LOAD_BUDGET,
    max_lines: int = DEFAULT_MAX_LINES,
    output_cap: int = DEFAULT_OUTPUT_CAP,
) -> RederiveReport:
    """Re-derive every derived-state line of ``text`` against ``root``.

    ``root=None`` means the store is not opted in: nothing executes, and the
    returned text carries a one-line notice instead of flags.
    """
    text = strip_rederive_output(text, schema)  # re-deriving is idempotent
    lines = text.split("\n")
    targets = derived_state_lines(text, schema)
    if root is None:
        notice = (
            f"{_ENVELOPE}not enabled for this store; STATE lines were not "
            "checked (see `anneal-memory derive allow`)."
        )
        return RederiveReport(text=notice + "\n\n" + text if targets else text, enabled=False)

    deadline = time.monotonic() + budget
    root_s = os.path.realpath(root)
    needs_ref = any(REF_TOKEN in (parse_annotation(l) or Annotation("judged", "")).command for _, l in targets)
    # The repository's shape is checked once per load, before git runs at all.
    shape = _check_repo_shape(root_s) if targets else None
    ref_sha, ref_err = (None, None)
    if shape and (needs_ref or ref is not None):
        ref_err = shape
    elif needs_ref or ref is not None:
        ref_sha, ref_err = _resolve_ref(root_s, ref, timeout)

    results: list[LineResult] = []
    ran = 0
    for idx, line in targets:
        ann = parse_annotation(line)
        is_cmd = ann is not None and ann.kind == "derive"
        if is_cmd and (ran >= max_lines or time.monotonic() >= deadline):
            why = f"line cap {max_lines}" if ran >= max_lines else f"load budget {budget:g}s"
            results.append(LineResult(idx, line, "skipped", why))
            continue
        remaining = deadline - time.monotonic()
        if is_cmd and remaining < 2 * timeout:
            # A line may spend one timeout on the tracked-path check and one on
            # the command; with less than that left, a timeout would be the
            # budget's, not the command's, so do not start it.
            results.append(LineResult(idx, line, "skipped", f"load budget {budget:g}s"))
            continue
        r = _judge_line(idx, line, root_s, ref_sha, ref_err, timeout, output_cap, shape)
        if is_cmd:  # refused lines count too, so the cap bounds the work
            ran += 1
        results.append(r)

    for r in results:
        if r.flag:
            body = strip_flag(lines[r.index].rstrip("\r"))
            cr = "\r" if lines[r.index].endswith("\r") else ""
            lines[r.index] = f"{body}  {r.flag}{cr}"
    counts = {s: sum(1 for r in results if r.status == s) for s in _FLAGS}
    summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    header = (
        f"{_ENVELOPE}{len(results)} STATE line(s) in {' '.join(root_s.split())}"
        + (f" at {ref_sha}" if ref_sha else "")
        + (f": {summary}" if summary else "")
        + ". git lines read the pinned ref only where they use @REF; grep, wc "
        "and test read the working tree."
    )
    out = "\n".join(lines)
    if results:
        out = header + "\n\n" + out
    return RederiveReport(text=out, enabled=True, ref=ref_sha, root=root_s, results=results)


# --------------------------------------------------------------------------
# Trust file (containment rule 1)
# --------------------------------------------------------------------------


def trust_file_path() -> Path:
    env = os.environ.get(TRUST_ENV)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".anneal-memory" / "derive-trust.json"


_SUPPORTED = os.name == "posix"  # ownership, O_NOFOLLOW, flock, killpg


def _load_trust(path: Path, *, strict: bool = False) -> list[dict]:
    """The bindings in the trust file. A missing file is empty. A file that is
    unreadable, a symlink, not owned by this user, or writable by others (or
    whose directory is) trusts nothing on read, and ``strict`` (the write
    paths) raises instead, so ``allow`` and ``revoke`` never overwrite
    bindings they could not read. The checks and the read use ONE descriptor,
    so the file cannot be swapped between them."""

    def untrusted(why: str) -> list[dict]:
        if strict:
            raise ValueError(f"trust file {path}: {why}; fix or remove it first")
        return []

    if not _SUPPORTED:
        return untrusted("re-derive is supported on POSIX systems only")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        if strict:  # about to create it: its directory must already be safe
            try:
                pst = os.stat(os.path.dirname(os.path.abspath(path)))
            except OSError:
                return []
            if pst.st_uid != os.geteuid() or pst.st_mode & 0o022:
                return untrusted("its directory is not owned by this user, or is writable by others")
        return []
    except OSError as e:
        return untrusted(f"cannot open without following links ({e})")
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid() or st.st_mode & 0o022:
            return untrusted("not owned by this user, or writable by others")
        # Whoever can write the directory can replace the file.
        pst = os.stat(os.path.dirname(os.path.abspath(path)))
        if pst.st_uid != os.geteuid() or pst.st_mode & 0o022:
            return untrusted("its directory is not owned by this user, or is writable by others")
        with os.fdopen(os.dup(fd), "r", encoding="utf-8") as f:
            data = json.loads(f.read())
    except (OSError, ValueError):
        return untrusted("unreadable")
    finally:
        os.close(fd)
    stores = data.get("stores") if isinstance(data, dict) else None
    if not isinstance(stores, list):
        return untrusted("has no 'stores' list")
    return [s for s in stores if isinstance(s, dict) and isinstance(s.get("db"), str) and isinstance(s.get("root"), str)]


def _write_trust(path: Path, stores: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".derive-trust.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "stores": stores}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update_trust(path: Path, change, verify=None) -> list[dict]:
    """Read-modify-write the trust file under an exclusive lock on a stable
    sibling, so a concurrent allow and revoke cannot lose either update."""
    if not _SUPPORTED:
        raise ValueError("re-derive is supported on POSIX systems only")
    import fcntl

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_fd = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        before = _load_trust(path, strict=True)
        after = change(before)
        if after is not before:
            _write_trust(path, after)
            # Checked while still holding the lock, and undone by restoring
            # exactly what was there, so a concurrent caller's binding is
            # never touched (glm-5.3 L3 round 3).
            if verify is not None and not verify():
                _write_trust(path, before)
                raise ValueError(f"the binding was written to {path} but is not honoured on read")
        return before
    finally:
        os.close(lock_fd)  # releases the flock


def _db_key(db_path: str | os.PathLike) -> str | None:
    s = str(db_path)
    if not s or s == ":memory:":
        return None
    return os.path.realpath(s)


def trusted_root(db_path: str | os.PathLike, trust_file: Path | None = None) -> str | None:
    """The root this store is allowed to re-derive in, or ``None``."""
    key = _db_key(db_path)
    if key is None:
        return None
    tf = trust_file or trust_file_path()
    for s in _load_trust(tf):
        if s["db"] == key:
            root = s["root"]
            if not os.path.isabs(root) or not os.path.isdir(root):
                return None
            # A trust file inside a bound root could have arrived with it.
            if _inside(os.path.realpath(tf), root):
                return None
            return root
    return None


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def allow_store(db_path: str | os.PathLike, root: str | os.PathLike, trust_file: Path | None = None) -> str:
    """Bind a store to a root directory. Returns the resolved root."""
    key = _db_key(db_path)
    if key is None:
        raise ValueError("an in-memory store cannot be allowed to re-derive")
    root_s = os.path.realpath(root)
    if not os.path.isdir(root_s):
        raise ValueError(f"root {root_s!r} is not a directory")
    path = trust_file or trust_file_path()
    if _inside(os.path.realpath(path), root_s):
        raise ValueError(
            f"the trust file {path} is inside the root; a binding there would be ignored"
        )
    entry = {
        "db": key,
        "root": root_s,
        "allowed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _update_trust(
        path,
        lambda stores: [s for s in stores if s["db"] != key] + [entry],
        verify=lambda: trusted_root(key, path) == root_s,  # never report an opt-in that does not hold
    )
    return root_s


def revoke_store(db_path: str | os.PathLike, trust_file: Path | None = None) -> bool:
    """Remove a store's binding. Returns whether one existed."""
    key = _db_key(db_path)
    path = trust_file or trust_file_path()

    def drop(stores: list[dict]) -> list[dict]:
        kept = [s for s in stores if s["db"] != key]
        return stores if len(kept) == len(stores) else kept

    before = _update_trust(path, drop)
    return any(s["db"] == key for s in before)


# --------------------------------------------------------------------------
# Store-level entry points
# --------------------------------------------------------------------------


def rederive_continuity(store, *, ref: str | None = None, trust_file: Path | None = None, **limits) -> RederiveReport | None:
    """Load ``store``'s continuity with every STATE line re-derived.

    Returns ``None`` when the store has no continuity yet. On a store that is
    not opted in (:func:`allow_store`), nothing executes.
    """
    text = store.load_continuity()
    if text is None:
        return None
    root = trusted_root(store.path, trust_file)
    return rederive_text(text, store.section_schema, root, ref=ref, **limits)


def check_state_for_save(
    text: str,
    schema: list[SectionSpec],
    db_path: str | os.PathLike,
    trust_file: Path | None = None,
) -> RederiveReport | None:
    """The save gate for derived-state sections.

    Raises ``ValueError`` (nothing is written) when a State line has no
    annotation or a refused command, and, on an opted-in store, when a command
    errors. Returns the report (``None`` when the schema has no derived-state
    section) so the caller can surface stale lines without refusing.
    """
    if not _derived_headings(schema):
        return None
    problems: list[str] = []
    for idx, line in derived_state_lines(text, schema):
        ann = parse_annotation(line)
        if ann is None:
            problems.append(f"line {idx + 1}: no [derive: …] or [judged: …] annotation")
        elif ann.kind == "derive":
            try:
                validate_command(ann.command)
            except DeriveRefused as e:
                problems.append(f"line {idx + 1}: refused: {e}")
    if problems:
        raise ValueError("State section refused:\n  " + "\n  ".join(problems))
    root = trusted_root(db_path, trust_file)
    report = rederive_text(text, schema, root)
    if not report.enabled:
        return report
    bad = [r for r in report.results if r.status in ("error", "refused")]
    if bad:
        raise ValueError(
            "State section refused, derive commands failed:\n  "
            + "\n  ".join(f"line {r.index + 1}: {r.flag}" for r in bad)
        )
    return report
