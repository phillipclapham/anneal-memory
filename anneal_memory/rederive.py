"""anneal_memory.rederive — STATE lines that prove themselves when read.

A ``derived-state`` section (the ``project`` schema's ``## State``) holds
present-tense claims, each ending with an annotation:

    [derive: COMMAND => EXPECTED]   value claim: stdout == EXPECTED (exit 0; grep also 1)
    [derive: COMMAND]               truth claim: exit 0 agrees, exit 1 disagrees
    (any other exit is an error in both forms)
    [judged: WHO, WHEN, AGAINST]    a judgement; accepted, never executed
    [derive@LABEL: …]               either derive form, run in the root bound to LABEL

:func:`rederive_continuity` runs each command and flags the line inline;
:func:`check_state_for_save` is the save-time gate.

⚠ SECURITY. This module executes commands stored in a memory file that an
agent wrote, so every command is untrusted input. The containment is designed
in ``docs/rederive.md``; the rules it enforces here are:

1. nothing runs unless the store's database path is bound to a root directory
   in the per-user trust file (outside the store; no MCP tool writes it); a
   line names a further root only by a label that file binds, never a path;
2. no shell: ``shlex`` + ``shell=False``, and shell metacharacters are refused;
3. an argument-by-argument allowlist of read-only command forms;
4. file arguments must resolve inside the root, and grep/wc read only files
   git tracks, never their content (grep answers with a count or yes/no);
5. a from-scratch environment, the root as cwd, stdin closed, one pinned ref
   per root;
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
from collections.abc import Mapping
from typing import Any, Literal

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
    "trusted_roots",
    "root_identities",
    "has_derive_lines",
    "VISIBILITIES",
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
# An annotation opener: "[derive:", "[derive@LABEL:" or "[judged:". The label
# is captured loosely here and checked against _LABEL by the caller, so a
# malformed one is refused rather than read as some other opener.
_OPENER = re.compile(r"\[(?:derive(?:@([^:\]\s]*))?|judged):")
# A root label names a root bound in the trust file; it is never a path.
_LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
VISIBILITIES = ("public", "private")


@dataclass(frozen=True)
class Annotation:
    kind: Literal["derive", "judged"]
    claim: str
    command: str = ""
    expected: str | None = None  # None = truth claim (exit status)
    judgement: str = ""
    root: str | None = None  # the root label of "[derive@LABEL: …]"; None = the default root


def _check_label(label: str) -> None:
    if not _LABEL.fullmatch(label):
        raise DeriveRefused(
            f"root label {label!r} is not a label (lowercase letters, digits and '-', "
            "at most 32 characters, starting with a letter or digit)"
        )


# The inline flag rederive_text appends ("  ✓", "  ⚠ STALE (…)", …), so text
# loaded with --rederive and carried into a wrap still parses.
_APPENDED_FLAG = re.compile(r"  (?:✓|⚠ [A-Z][A-Z ]*[A-Z]|⛔ REFUSED)(?: \(.*\))?")


_FLAG_BASE = re.compile(r"  (?:✓|⚠ [A-Z][A-Z ]*[A-Z]|⛔ REFUSED)")


def _flag_at(line: str, i: int) -> bool:
    """Whether a flag rederive_text could have appended starts at ``i`` and
    runs to the end of the line: the same shape as ``_APPENDED_FLAG``, checked
    without scanning the rest of the line, so a line of many flag-shaped
    fragments costs linear time, not quadratic (codex L3 2026-10-01: 40k
    fragments took 2.2s, doubling size quadrupled it)."""
    m = _FLAG_BASE.match(line, i)
    if m is None:
        return False
    return m.end() == len(line) or (line.startswith(" (", m.end()) and line.endswith(")"))


def strip_flag(line: str) -> str:
    """``line`` without a flag :func:`rederive_text` appended to it; any other
    line comes back byte for byte.

    A claim may itself hold a flag-shaped run ("Release gate  ⚠ BLOCKED (see
    Open) [derive: …]"), and so may a flag's detail, so the cut is the
    rightmost flag that follows an annotation's closing "]". Only a line with
    no such flag (an unannotated one) falls back to the leftmost (Diogenes
    2026-10-01)."""
    cuts = [m.start() for m in re.finditer("  (?=[✓⚠⛔])", line) if _flag_at(line, m.start())]
    if not cuts:
        return line
    after_bracket = [i for i in cuts if i > 0 and line[i - 1] == "]"]  # an index, not a copied prefix
    return line[: (after_bracket[-1] if after_bracket else cuts[0])]


# Every text rederive_text returns starts with this tag. At save the header is
# removed only when the whole line matches what rederive_text writes, so an
# authored note is never deleted; flags are removed from State lines always,
# since nothing may follow an annotation's closing "]" (a forged "  ✓" is
# never persisted).
_ENVELOPE = "> [anneal re-derive] "
_HEADER_LINE = re.compile(
    r"^> \[anneal re-derive\] (?:"
    # One unstructured span between a fixed start and a fixed end: the
    # per-root list made a nested repetition that backtracked exponentially on
    # a crafted first line, on every save (L2 2026-10-01, reproduced: 20
    # segments took 0.58s, doubling per segment). Python 3.10 has no atomic
    # groups, so the ambiguity is removed rather than bounded.
    r"[1-9]\d* STATE line\(s\) in [^\n]+"
    r"\. git lines read the pinned ref only where they use @REF; grep, wc and test "
    r"read the working tree\."
    r"|not enabled for this store; STATE lines were not checked "
    r"\(see `anneal-memory derive allow`\)\.)$"
)


def drop_header(text: str) -> str:
    """``text`` without the header line :func:`rederive_text` puts first (and
    the blank line after it). Only that exact line in that place is removed, so
    an authored note is never deleted."""
    lines = text.split("\n")
    if len(lines) > 1 and _HEADER_LINE.match(lines[0].rstrip("\r")) and not lines[1].strip():
        return "\n".join(lines[2:])
    return text


def strip_rederive_output(text: str, schema: list[SectionSpec]) -> str:
    """Remove everything :func:`rederive_text` added: its header line (and the
    blank line after it) and every flag it appended to a State line. Saving
    loaded text back must never persist a verdict that is true only at load."""
    lines = drop_header(text).split("\n")
    joined = "\n".join(lines)
    for idx, line in derived_state_lines(joined, schema):
        cr = "\r" if line.endswith("\r") else ""
        lines[idx] = strip_flag(line[:-1] if cr else line) + cr
    return "\n".join(lines)


def parse_annotation(line: str) -> Annotation | None:
    """Parse the trailing annotation of a State line, or ``None`` if absent.

    The annotation is the LAST ``[derive:`` / ``[derive@LABEL:`` /
    ``[judged:`` on the line and must close with the line's final ``]`` (after
    any re-derive flag is removed). A label is returned as written; whether it
    is a valid label is checked where the command is (:func:`_check_label`).
    """
    if line.endswith("\r"):
        line = line[:-1]
    stripped = strip_flag(line)
    if not stripped.endswith("]"):
        return None
    m = None
    for m in _OPENER.finditer(stripped):
        pass
    if m is None:
        return None
    claim = stripped[: m.start()].rstrip()
    body = stripped[m.end() : -1].strip()
    if m.group(0) == _JUDGED_OPEN:
        if not body:
            return None
        return Annotation(kind="judged", claim=claim, judgement=body)
    label = m.group(1)  # None for "[derive:"; a string (maybe empty) for "[derive@…:"
    if _EXPECT_SEP in body:
        command, expected = body.rsplit(_EXPECT_SEP, 1)  # a pattern may hold " => "
        return Annotation(
            kind="derive", claim=claim, command=command.strip(), expected=expected.strip(), root=label
        )
    return Annotation(kind="derive", claim=claim, command=body, root=label)


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


def _check_repo_shape(root: str, timeout: float | None = None) -> str | None:
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
    # Git follows a symlink anywhere in its metadata (a loose ref, HEAD,
    # packed-refs, an objects fan-out directory): one that leaves the root
    # is a read outside it (gpt-oss L3 round 3, reproduced).
    root_real = os.path.realpath(root)
    walk_deadline = None if timeout is None else time.monotonic() + timeout
    # Bounded by the load budget, checked while each directory is read (not
    # after os.walk has listed it whole): a .git of millions of entries would
    # otherwise hold a load for as long as the walk takes (codex L3,
    # spore-1233 r2 and 2026-10-01; measured: 60k entries overran a 0.05s
    # budget by 3.6x, 200k in one directory a 0.01s one by 13x).
    seen = 0
    stack = [git]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    seen += 1
                    if walk_deadline is not None and seen % 256 == 0 and time.monotonic() > walk_deadline:
                        return ".git could not be checked within the load budget"
                    if entry.is_symlink():
                        if not _inside(os.path.realpath(entry.path), root_real):
                            return f"{os.path.relpath(entry.path, root)} is a symlink out of the root"
                    elif entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
        except OSError as e:
            return f".git could not be read ({e})"
    if walk_deadline is not None and time.monotonic() > walk_deadline:
        return ".git could not be checked within the load budget"
    # The repository's config may not pull in another file. Git's own parser
    # decides what an include is (a regex missed "[core][include]" on one
    # line; Diogenes 2026-10-01); --file reads that one file and follows no
    # include. _GIT_HARDENING's --no-pager is load-bearing here: without it,
    # git's pager lookup loads the repository config, includes and all,
    # before --file applies (L2, measured on git 2.50). Needs git >= 2.25 for
    # --name-only. config.worktree is a second config file git reads when
    # extensions.worktreeConfig is set, so it may not exist at all.
    if os.path.lexists(os.path.join(git, "config.worktree")):
        return ".git/config.worktree exists; its config is not checked"
    config = os.path.join(git, "config")
    if os.path.lexists(config):
        try:
            if not stat.S_ISREG(os.stat(config).st_mode):
                return ".git/config is not a regular file"
        except OSError as e:  # a dangling link, or gone since lexists (L3)
            return f".git/config is unreadable ({e})"
        left = _SHAPE_TIMEOUT if walk_deadline is None else min(_SHAPE_TIMEOUT, walk_deadline - time.monotonic())
        if left <= 0:
            return ".git could not be checked within the load budget"
        out = _run_bounded(
            ["git", *_GIT_HARDENING, "config", "--file", os.path.join(git, "config"),
             "--name-only", "--list"],
            root, left, _SHAPE_CAP,
        )
        if out.returncode != 0 or out.timed_out or out.overflowed:
            why = out.spawn_error or out.stderr.decode("utf-8", "replace").strip() or "no answer"
            return f".git/config could not be parsed ({why})"
        # Only a path key includes a file; "[include] enabled = true" does not (L3).
        for key in out.stdout.decode("utf-8", "replace").splitlines():
            key = key.lower()
            if key == "include.path" or (key.startswith("includeif.") and key.endswith(".path")):
                return ".git/config includes another file; git would read outside the root"
    return None


# Bounds for the one git call the shape check makes (reading .git/config).
_SHAPE_TIMEOUT = 10.0
_SHAPE_CAP = 1 << 20


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

Status = Literal["ok", "stale", "error", "refused", "judged", "unannotated", "skipped", "unbound"]

_FLAGS: dict[str, str] = {
    "ok": "✓",
    "stale": "⚠ STALE",
    "error": "⚠ DERIVE ERROR",
    "refused": "⛔ REFUSED",
    "judged": "",
    "unannotated": "⚠ NO DERIVE",
    "skipped": "⚠ NOT DERIVED",
    "unbound": "⚠ UNBOUND",
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
        # Whitespace collapsed: a detail echoing a command token could hold a
        # two-space run, the boundary strip_flag cuts at, or a line break
        # (L1, reproduced with a refused "x]  ⚠ FOO (y)" token).
        detail = " ".join(self.detail.split())
        return f"{base} ({detail})" if detail else base


@dataclass
class RederiveReport:
    text: str  # the continuity with inline flags
    enabled: bool  # False = store not opted in; nothing ran
    ref: str | None = None  # commit id @REF resolved to
    root: str | None = None  # the default root
    results: list[LineResult] = field(default_factory=list)
    roots: dict = field(default_factory=dict)  # root label (None = default) -> root
    refs: dict = field(default_factory=dict)  # root label -> commit id its @REF resolved to

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
        if rc == 1 and argv[0] != "wc":
            return LineResult(idx, line, "stale", "exit 1", ran=True)
        return LineResult(idx, line, "error", f"exit {rc}: {_one_line(res.stderr)}", ran=True)
    # Only grep's exit 1 means "no match" with a complete answer (grep -c
    # prints 0 and exits 1), so only grep's output is compared on exit 1. For
    # every other program exit 1 can follow partial output: wc after counting
    # the files it could read (codex L3, spore-1233 r2), git ls-files
    # --error-unmatch after listing the tracked ones (L1 2026-10-01); both
    # reproduced passing a false claim. Anything else is an error (git uses
    # 128/129).
    if rc not in (0, 1) or (rc == 1 and argv[0] != "grep"):
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


def _line_needs_repo(ann: Annotation | None) -> bool:
    """Whether a line could need its root to be a plain repository: anything
    but a judged line, an unannotated one or a plain ``test`` (fail-safe: a
    command whose first token merely looks like another program still
    counts, since ``"git"`` validates to git; L3 2026-10-01)."""
    return ann is not None and ann.kind == "derive" and ann.command.split()[:1] != ["test"]


def rederive_text(
    text: str,
    schema: list[SectionSpec],
    root: str | os.PathLike | Mapping | None,
    *,
    ref: str | None = None,
    timeout: float = DEFAULT_COMMAND_TIMEOUT,
    budget: float = DEFAULT_LOAD_BUDGET,
    max_lines: int = DEFAULT_MAX_LINES,
    output_cap: int = DEFAULT_OUTPUT_CAP,
) -> RederiveReport:
    """Re-derive every derived-state line of ``text`` against its root.

    ``root`` is the store's default root, or a mapping of root label to root
    as :func:`trusted_roots` returns it (``None`` keys the default). A
    ``[derive: …]`` line runs in the default root, a ``[derive@LABEL: …]``
    line in that label's root, and a label with no root is ``⚠ UNBOUND``.
    ``ref`` pins the default root only; each other root's ``@REF`` is its own
    HEAD. ``root=None`` (or a mapping with no default) means the store is not
    opted in: nothing executes, and the returned text carries a one-line
    notice instead of flags.
    """
    text = strip_rederive_output(text, schema)  # re-deriving is idempotent
    lines = text.split("\n")
    targets = derived_state_lines(text, schema)
    if isinstance(root, Mapping):
        # A key that is not a valid label is never a root: its lines are refused.
        roots = {
            k: os.path.realpath(v) for k, v in root.items() if k is None or _LABEL.fullmatch(k)
        }
    else:
        roots = {None: os.path.realpath(root)} if root is not None else {}
    if None not in roots:
        notice = (
            f"{_ENVELOPE}not enabled for this store; STATE lines were not "
            "checked (see `anneal-memory derive allow`)."
        )
        return RederiveReport(text=notice + "\n\n" + text if targets else text, enabled=False)

    deadline = time.monotonic() + budget
    anns = {idx: parse_annotation(line) for idx, line in targets}

    def key_of(ann: Annotation | None) -> str | None:
        return ann.root if ann is not None and ann.kind == "derive" else None

    # Each root used by a line is checked once per load, in a fixed order
    # (default first), before git runs in it: its shape when any of its lines
    # could need the repository, and its @REF. All of it is bounded by the one
    # load budget (docs/rederive.md, multi-root).
    # One pass groups the lines the line cap can reach by root; a root only
    # lines past the cap would use is never checked (codex L3 2026-10-01).
    by_root: dict[str | None, list[Annotation]] = {None: []}
    reachable = 0
    for idx, _ in targets:
        a = anns[idx]
        if a is None or a.kind != "derive":
            continue
        reachable += 1
        if reachable > max_lines:
            break
        if a.root is None or (a.root in roots and _LABEL.fullmatch(a.root)):
            by_root.setdefault(a.root, []).append(a)
    used = [k for k in roots if k in by_root]
    shapes: dict[str | None, str | None] = {}
    refs: dict[str | None, tuple[str | None, str | None]] = {}
    for k in used:
        root_k = roots[k]
        mine = by_root[k]
        pin = ref if k is None else None
        needs_ref = any(REF_TOKEN in a.command for a in mine)
        needs_repo = needs_ref or pin is not None or any(_line_needs_repo(a) for a in mine)
        remaining = deadline - time.monotonic()
        if needs_repo and remaining <= 0:
            # The budget went on earlier roots: this root is not checked, so
            # nothing runs in it (L2 2026-10-01: 40 roots overran 0.05s by 33x).
            shapes[k] = "the load budget was spent before this root was checked"
        else:
            shapes[k] = _check_repo_shape(root_k, timeout=remaining) if needs_repo else None
        if needs_ref or pin is not None:
            left = min(timeout, deadline - time.monotonic())
            if shapes[k]:
                refs[k] = (None, shapes[k])
            elif left <= 0:
                refs[k] = (None, "the load budget was spent before the ref was resolved")
            else:
                refs[k] = _resolve_ref(root_k, pin, left)
        else:
            refs[k] = (None, None)

    results: list[LineResult] = []
    ran = 0
    for idx, line in targets:
        ann = anns[idx]
        is_cmd = ann is not None and ann.kind == "derive"
        label = ann.root if ann is not None and ann.kind == "derive" else None
        if label is not None:
            try:
                _check_label(label)
            except DeriveRefused as e:
                ran += 1  # counts toward the cap like any refused line
                results.append(LineResult(idx, line, "refused", str(e)))
                continue
            if label not in roots:
                # Counted like every derive line, so the cap counts exactly the
                # lines the preflight above grouped (an uncounted line would let
                # a later line reach a root that was never checked).
                ran += 1
                results.append(LineResult(
                    idx, line, "unbound",
                    f"no root is bound to label {label!r} for this store (see `anneal-memory derive allow --label`)",
                ))
                continue
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
        k = key_of(ann)
        if is_cmd and k not in shapes:
            # Fail-safe: a root the preflight did not check never runs a line.
            results.append(LineResult(idx, line, "skipped", f"line cap {max_lines}"))
            continue
        ref_sha, ref_err = refs.get(k, (None, None))
        r = _judge_line(idx, line, roots[k], ref_sha, ref_err, timeout, output_cap, shapes.get(k))
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

    def where(k: str | None) -> str:
        sha = refs.get(k, (None, None))[0]
        return f"{' '.join(roots[k].split())}" + (f" at {sha}" if sha else "")

    header = (
        f"{_ENVELOPE}{len(results)} STATE line(s) in {where(None)}"
        + "".join(f"; {k} in {where(k)}" for k in used if k is not None)
        + (f": {summary}" if summary else "")
        + ". git lines read the pinned ref only where they use @REF; grep, wc "
        "and test read the working tree."
    )
    out = "\n".join(lines)
    if results:
        out = header + "\n\n" + out
    return RederiveReport(
        text=out, enabled=True, ref=refs[None][0], root=roots[None], results=results,
        roots=dict(roots), refs={k: v[0] for k, v in refs.items()},
    )


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
            json.dump({"version": 2, "stores": stores}, f, indent=2)
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


def _usable_root(root: object, tf: Path) -> str | None:
    if not isinstance(root, str) or not os.path.isabs(root):
        return None
    # Resolved, so a hand-written symlink cannot pass the nesting and
    # trust-file checks under another name (L2 2026-10-01, reproduced). A
    # malformed path (a NUL byte) is unusable, never a crash (codex L3).
    try:
        root = os.path.realpath(root)
        if not os.path.isdir(root):
            return None
    except (OSError, ValueError, TypeError):
        return None
    # A trust file inside a bound root could have arrived with it.
    if _inside(os.path.realpath(tf), root):
        return None
    return root


def _nested(a: str, b: str) -> bool:
    return _inside(a, b) or _inside(b, a)


def trusted_roots(db_path: str | os.PathLike, trust_file: Path | None = None) -> dict[str | None, str]:
    """Every root this store may re-derive in, keyed by label (``None`` keys
    the default root). Empty when the store has no usable default root.

    A labelled root is dropped, so its lines read ``⚠ UNBOUND``, when it is
    not a usable directory, when it is nested in (or holds) another of the
    store's roots, or when the store's roots do not all declare the same
    visibility. The rules :func:`allow_store` enforces are checked again
    here, so a hand-edited trust file cannot widen what a store reaches."""
    key = _db_key(db_path)
    if key is None:
        return {}
    tf = trust_file or trust_file_path()
    for s in _load_trust(tf):
        if s["db"] != key:
            continue
        default = _usable_root(s["root"], tf)
        if default is None:
            return {}
        roots: dict[str | None, str] = {None: default}
        labels = s.get("labels")
        if not isinstance(labels, dict) or not labels:
            return roots
        vis = s.get("visibility")
        found: dict[str, str] = {}
        for name, entry in labels.items():
            if not isinstance(name, str) or not _LABEL.fullmatch(name) or not isinstance(entry, dict):
                continue
            r = _usable_root(entry.get("root"), tf)
            if r is None or vis not in VISIBILITIES or entry.get("visibility") != vis:
                continue
            found[name] = r
        # Every root that nests with any other is dropped, whatever the order
        # the file lists them in (codex L3 2026-10-01: dropping only the later
        # one let JSON key order choose which root text could reach).
        for name, r in found.items():
            others = [default] + [o for n, o in found.items() if n != name]
            if not any(_nested(r, o) for o in others):
                roots[name] = r
        return roots
    return {}


def trusted_root(db_path: str | os.PathLike, trust_file: Path | None = None) -> str | None:
    """The default root this store is allowed to re-derive in, or ``None``."""
    return trusted_roots(db_path, trust_file).get(None)


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def _check_root(root_s: str, path: Path) -> None:
    if not os.path.isdir(root_s):
        raise ValueError(f"root {root_s!r} is not a directory")
    # A repository root whose shape git lines refuse (a linked worktree, a
    # gitfile, an include) would make every git, grep and wc line an error,
    # and an opted-in store refuses every save with one (Diogenes 2026-10-01).
    # A root with no .git at all stays allowed: its test lines run.
    if os.path.lexists(os.path.join(root_s, ".git")):
        shape = _check_repo_shape(root_s)
        if shape:
            hint = " Allow the main checkout instead." if "plain directory" in shape else ""
            raise ValueError(
                f"root {root_s!r} is not a repository re-derive can run git in: {shape}.{hint}"
            )
    if _inside(os.path.realpath(path), root_s):
        raise ValueError(
            f"the trust file {path} is inside the root; a binding there would be ignored"
        )


def allow_store(
    db_path: str | os.PathLike,
    root: str | os.PathLike,
    trust_file: Path | None = None,
    *,
    label: str | None = None,
    visibility: str | None = None,
) -> str:
    """Bind a store to a root directory. Returns the resolved root.

    With no ``label`` this binds the store's default root (the one
    ``[derive: …]`` lines run in), keeping any labelled roots. With a
    ``label`` it binds a further root for ``[derive@LABEL: …]`` lines; the
    store needs a default root first. ``visibility`` (``public`` or
    ``private``) is the operator's declaration about the root, which anneal
    cannot check: a store with more than one root must declare it on every
    root, and every root must declare the same one (one visibility class per
    store, so a public store's text never reaches a private repo's files).
    """
    key = _db_key(db_path)
    if key is None:
        raise ValueError("an in-memory store cannot be allowed to re-derive")
    if label is not None and not _LABEL.fullmatch(label):
        raise ValueError(
            f"label {label!r} must be lowercase letters, digits and '-', at most 32 "
            "characters, starting with a letter or digit"
        )
    if visibility is not None and visibility not in VISIBILITIES:
        raise ValueError(f"visibility must be one of {', '.join(VISIBILITIES)}")
    root_s = os.path.realpath(root)
    path = trust_file or trust_file_path()
    _check_root(root_s, path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def change(stores: list[dict]) -> list[dict]:
        current = next((s for s in stores if s["db"] == key), None)
        rest = [s for s in stores if s["db"] != key]
        raw = current.get("labels") if current else None
        labels = {
            n: e for n, e in (raw.items() if isinstance(raw, dict) else ())
            if isinstance(n, str) and isinstance(e, dict) and isinstance(e.get("root"), str)
        }
        if label is None:
            vis = visibility or (current.get("visibility") if current and current["root"] == root_s else None)
            for name, e in labels.items():
                if _nested(root_s, e.get("root", "")):
                    raise ValueError(f"root {root_s!r} is nested with the root of label {name!r}")
                if vis is None or e.get("visibility") != vis:
                    raise ValueError(
                        "this store has labelled roots, so its default root needs --visibility "
                        f"{e.get('visibility')}, matching theirs (one visibility class per store)"
                    )
            entry = {"db": key, "root": root_s, "allowed_at": now}
            if vis is not None:
                entry["visibility"] = vis
        else:
            if current is None:
                raise ValueError("bind the store's default root first (allow without --label)")
            if visibility is None:
                raise ValueError("a labelled root needs --visibility (public or private)")
            if current.get("visibility") is None:
                raise ValueError(
                    "the store's default root declares no visibility; re-allow it with "
                    f"--visibility first (one visibility class per store, and this root is {visibility!r})"
                )
            if current.get("visibility") != visibility:
                raise ValueError(
                    f"the store's default root is declared {current['visibility']!r} and this root "
                    f"{visibility!r}: one visibility class per store"
                )
            pairs: list[tuple[str | None, str]] = [(None, current["root"])]
            pairs += [(n, e.get("root", "")) for n, e in labels.items() if n != label]
            for who, other in pairs:
                if _nested(root_s, other):
                    raise ValueError(
                        f"root {root_s!r} is nested with the "
                        + ("default root" if who is None else f"root of label {who!r}")
                    )
            labels[label] = {"root": root_s, "allowed_at": now, "visibility": visibility}
            entry = {k: v for k, v in current.items() if k != "labels"}
        if labels:
            entry["labels"] = labels
        return rest + [entry]

    _update_trust(
        path,
        change,
        verify=lambda: trusted_roots(key, path).get(label) == root_s,  # never report an opt-in that does not hold
    )
    return root_s


def revoke_store(db_path: str | os.PathLike, trust_file: Path | None = None, *, label: str | None = None) -> bool:
    """Remove a store's binding (every root), or with ``label`` only that
    labelled root. Returns whether one existed."""
    key = _db_key(db_path)
    path = trust_file or trust_file_path()

    def drop(stores: list[dict]) -> list[dict]:
        if label is None:
            kept = [s for s in stores if s["db"] != key]
            return stores if len(kept) == len(stores) else kept
        out, hit = [], False
        for s in stores:
            labels = s.get("labels") if s["db"] == key else None
            if isinstance(labels, dict) and label in labels:
                hit = True
                s = dict(s)
                s["labels"] = {n: e for n, e in labels.items() if n != label}
                if not s["labels"]:
                    del s["labels"]
            out.append(s)
        return out if hit else stores

    before = _update_trust(path, drop)
    if label is None:
        return any(s["db"] == key for s in before)
    return any(s["db"] == key and label in (s.get("labels") or {}) for s in before)


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
    roots = trusted_roots(store.path, trust_file)
    return rederive_text(text, store.section_schema, roots, ref=ref, **limits)


_NO_FROZEN_ROOTS = object()


def _derive_roots_used(text: str, schema: list[SectionSpec]) -> set:
    """The root labels (``None`` for the default root) the derive lines of
    ``text`` run in."""
    return {
        a.root
        for _, ln in derived_state_lines(text, schema)
        if (a := parse_annotation(ln)) is not None and a.kind == "derive"
    }


def has_derive_lines(text: str, schema: list[SectionSpec]) -> bool:
    """Whether any derived-state line of ``text`` is a ``[derive…]`` line."""
    return bool(_derive_roots_used(text, schema))


def _identity(root: str) -> str:
    """``root`` with the identity of the directory and of its ``.git`` as
    "DEV:INO[/BIRTH]:DEV:INO[/BIRTH]:PATH". The path alone does not name a repository: a
    directory deleted and replaced by another at the same path keeps it (L2
    2026-10-01, reproduced). Fields that cannot be read are "-"."""
    def ids(path: str, follow: bool) -> str:
        try:
            st = os.stat(path) if follow else os.lstat(path)
        except (OSError, ValueError):
            return "-:-"
        # Where the platform reports a creation time it joins the inode, so an
        # inode reused for the replacement still differs (complement L3 r1:
        # ext4, xfs and tmpfs hand a freed inode to the next create, and Linux
        # Python reports no creation time, so there the inode alone can match).
        born = getattr(st, "st_birthtime", None)
        return f"{st.st_dev}:{st.st_ino}" + ("" if born is None else f"/{born!r}")
    return f"{ids(root, True)}:{ids(os.path.join(root, '.git'), False)}:{root}"


def root_identities(roots: dict) -> dict:
    """The value a wrap freezes for a root map (:func:`trusted_roots`): each
    root with its directory identity, see :func:`_identity`."""
    return {k: _identity(v) for k, v in roots.items()}


def _roots_moved(frozen: dict, now: dict) -> list[str]:
    """Each root bound now (``now``, as :func:`root_identities` gives it) that
    is not the one ``frozen`` held for its label: rebound, bound to a label
    that had none, or replaced at the same path. The caller passes only the
    roots its lines run in. A root unbound since is not listed: it
    certifies nothing (a revoked label reads UNBOUND and refuses on its own;
    with no root left, no command runs)."""
    def name(k):
        return "the default root" if k is None else f"label {k!r}"

    def path(ident):
        return ident.split(":", 4)[-1] if ident else "unbound"

    out = []
    for k in sorted(now, key=lambda k: (k is not None, k or "")):
        was, is_ = frozen.get(k), now[k]
        if was == is_:
            continue
        if was is not None and path(was) == path(is_):
            out.append(f"{name(k)}: {path(is_)} was replaced by a different directory at the same path")
        else:
            out.append(f"{name(k)}: {path(was)} -> {path(is_)}")
    return out


def check_state_for_save(
    text: str,
    schema: list[SectionSpec],
    db_path: str | os.PathLike,
    trust_file: Path | None = None,
    *,
    frozen_roots: Any = _NO_FROZEN_ROOTS,
    cancel_hint: str = "cancel this wrap and run prepare_wrap again",
) -> RederiveReport | None:
    """The save gate for derived-state sections.

    Raises ``ValueError`` (nothing is written) when a State line has no
    annotation or a refused command or label, and, on an opted-in store, when
    a command errors or names a label with no bound root. Returns the report (``None`` when the schema has no derived-state
    section) so the caller can surface stale lines without refusing.

    ``frozen_roots`` is the root map the wrap's prepare read and showed the
    composer, as :func:`root_identities` encodes it
    (:meth:`Store.wrap_derive_roots`). When it is passed, the map is read once
    here, compared with it, and that same map is the one the commands run
    against: a root a derive line here runs in that is not the frozen one for
    its label refuses the save (spore-1282, a label rebound while the composer
    worked; see :func:`_roots_moved`). ``None`` means the wrap froze no map,
    which refuses when any root is bound and any derive line would run.
    Leaving it out skips the comparison.
    ``cancel_hint`` finishes each refusal with how to recover.
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
                if ann.root is not None:
                    _check_label(ann.root)
                validate_command(ann.command)
            except DeriveRefused as e:
                problems.append(f"line {idx + 1}: refused: {e}")
    if problems:
        raise ValueError("State section refused:\n  " + "\n  ".join(problems))
    roots = trusted_roots(db_path, trust_file)
    if frozen_roots is not _NO_FROZEN_ROOTS:
        used = _derive_roots_used(text, schema)
        if frozen_roots is None and roots and used:
            raise ValueError(
                "State section refused: the wrap in progress recorded no re-derive root "
                "map (it was prepared by an older anneal-memory, or opened with "
                "Store.wrap_started directly), so nothing shows the roots bound now are "
                "the ones its State lines were written against. To recover, "
                + cancel_hint + "."
            )
        # Only roots a derive line in this text runs in can certify anything, so
        # only those are compared (L1 W2: an unrelated label bound mid-compose
        # refused a save no line of which was at risk).
        moved = [] if frozen_roots is None else _roots_moved(
            frozen_roots, {k: v for k, v in root_identities(roots).items() if k in used}
        )
        if moved:
            raise ValueError(
                "State section refused: the re-derive roots changed after prepare_wrap ("
                + "; ".join(moved)
                + "), so these State lines would be checked in a repository the composer "
                "did not see. "
                "To recover, " + cancel_hint + "."
            )
    report = rederive_text(text, schema, roots)
    if frozen_roots is not _NO_FROZEN_ROOTS and frozen_roots is not None:
        # The identities are taken again after the commands ran: a root replaced
        # while they ran refuses (codex L3 r1). This narrows the check-then-use
        # window and does not close it: a directory swapped away and back while
        # a command runs is not seen. Running in descriptors opened and checked
        # once is the deferred fd-pinning (spore-1272).
        after = _roots_moved(
            frozen_roots, {k: v for k, v in root_identities(roots).items() if k in used}
        )
        if after:
            raise ValueError(
                "State section refused: a re-derive root changed while its commands ran ("
                + "; ".join(after) + "). To recover, " + cancel_hint + "."
            )
    if not report.enabled:
        return report
    # An unbound label is not clean: the claim names a root nobody bound, so
    # it was not checked (absence of signal is never health).
    bad = [r for r in report.results if r.status in ("error", "refused", "unbound")]
    if bad:
        raise ValueError(
            "State section refused, derive commands failed:\n  "
            + "\n  ".join(f"line {r.index + 1}: {r.flag}" for r in bad)
        )
    return report
