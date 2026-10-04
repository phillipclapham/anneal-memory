"""Team ledger import: a teammate's decisions enter an engineer's store as
provenance-carrying episodes.

A team ledger is a set of append-only JSONL files, one hash chain per file; the
writer is Levain's team layer, and :func:`canonical` / :func:`chain_hash` here are the
rule it hashes by (``test_golden_vector`` in this repo pins the bytes; a copy of that
vector in Levain's tests is what would tie the two packages together). This
module is the READ side: it re-verifies every chain, maps each entry to an
episode that names its author, and hands the batch to
:meth:`Store.import_team_entries`, which inserts idempotently in one
transaction.

What it guarantees:

- **Chain-verified.** An entry is imported only if its hash matches
  ``sha256(prev + canonical(entry))`` and its ``prev`` is the hash of the entry
  before it, back to the chain start (``prev == ""``). Entries after a break, a
  fork or a gap are refused and reported; the verified prefix imports.
- **Never re-attributed.** ``source`` is ``team:<author>`` from the entry's own
  ``author`` field, and a chain whose entries name different authors is cut at
  the first change. The chain proves the file was not edited after the fact; it
  does not prove who wrote it (the git host's access control does that).
- **Idempotent by ledger id.** Re-importing changes nothing; the same id with a
  different hash is reported as a conflict and never overwritten.
- **Acks are not memory.** An ``ack`` entry is counted and skipped.
- **Hiding is authorised, not assumed.** A supersession by the SAME author applies.
  One by a different author applies only when that author matches a
  ``link_authority`` pattern the caller names (a team lead, the pack authors);
  otherwise it is reported in full and nothing is hidden.

Limits that are inherent, stated so nobody has to rediscover them: the author is
self-declared, so authentication is the git host's job (branch protection, signed
commits) and nothing here sees git committers; anyone who can write to the ledger can
append a line under any author the chain of its own file allows, or a fork that freezes
a teammate's chain at the fork point (reported, never silent); a directory read binds
each file to the author directory it sits in, a stream on stdin binds nothing and is one
trust unit the exporter must vouch for; an entry removed from the ledger stays in an
engineer's store, because only a signed ``retire`` reaches it; an ``ack`` leaves no
record, so a later entry reusing its id in another call is not seen as a clash; the
first importer of an id wins; and a local writer with ``Store.record`` can plant a
``team:`` row that claims an entry id.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .store import Store

__all__ = [
    "canonical",
    "chain_hash",
    "import_ledger",
    "read_ledger_lines",
    "TeamImportReport",
]

SCHEMA_VERSION = 1
TYPES = ("decision", "constraint", "finding", "question", "tension", "ack", "retire")
KINDS = ("ruling", "practice")
# decision/constraint need a kind; finding/question/tension take their own episode type.
_EPISODE_TYPE = {
    "decision": "decision",
    "constraint": "decision",
    "finding": "observation",
    "question": "question",
    "tension": "tension",
    # retire only retires: its episode is the anchor the supersession links hang
    # on, and it states the retirement, not a decision of its own.
    "retire": "context",
}
_MAX_LINE_CHARS = 1_000_000
_HANDLE = re.compile(r"[A-Za-z0-9._@:+-]{1,200}")
# Levain's writer builds an id as <P>-<stamp>-<hex> with P derived from the author
# below; an id that does not start with its own author's P is a squatter's, and is
# refused. The whole shape is checked, so an author "alice" cannot claim an id of
# author "alice-bob". Two authors whose handles reduce to the same prefix (pack:x and
# pack-x) are not told apart; the second to import reports a conflict.
_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]")
_LEDGER_ID = re.compile(r"[A-Za-z0-9._-]{1,64}-[0-9]{14}-[0-9a-f]{8}", re.ASCII)
_AGENT = re.compile(r"[A-Za-z0-9._:@-]{1,128}")


def _id_prefix(author: str) -> str:
    return _ID_SAFE.sub("-", author).strip("-.")[:64] or "x"

_TEXT_FIELDS = {"words": 4000, "summary": 2000, "reason": 4000, "recheck": 1000}
_MAX_PATHS = 100
_MAX_PATH_CHARS = 300
_MAX_LINES = 200_000
_MAX_FILE_BYTES = 64 * 1024 * 1024
_FUTURE_SKEW = timedelta(days=1)
_TS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def canonical(entry: dict) -> str:
    """The exact bytes the writer hashes: every field but ``hash``, sorted keys,
    compact separators, non-ASCII kept."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_hash(prev: str, entry: dict) -> str:
    return hashlib.sha256((prev + canonical(entry)).encode("utf-8")).hexdigest()


@dataclass
class TeamImportReport:
    imported: list[dict] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    skipped_ack: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    chain_problems: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    links_made: list[dict] = field(default_factory=list)
    links_pending: list[dict] = field(default_factory=list)
    links_refused: list[dict] = field(default_factory=list)
    links_unauthorized: list[dict] = field(default_factory=list)
    dry_run: bool = False

    @property
    def clean(self) -> bool:
        """True when nothing was refused or in conflict. Pending links are not a
        problem: a target not fetched yet completes on a later import."""
        return not (self.rejected or self.chain_problems or self.conflicts
                    or self.links_refused or self.links_unauthorized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "clean": self.clean,
            "imported": len(self.imported),
            "already_present": len(self.already_present),
            "skipped_ack": len(self.skipped_ack),
            "rejected": self.rejected,
            "chain_problems": self.chain_problems,
            "conflicts": self.conflicts,
            "links_made": len(self.links_made),
            "cross_author_links": [
                {"id": l["id"], "target": l["target"], "by": l["source"]}
                for l in self.links_made if l["cross_author"] == "true"
            ],
            "links_pending": self.links_pending,
            "links_refused": self.links_refused,
            "links_unauthorized": self.links_unauthorized,
            "imported_ids": [i["id"] for i in self.imported],
        }


def read_ledger_lines(paths: Iterable[str | Path]) -> list[str | FileStart]:
    """Lines of every ledger file given, each file preceded by a :class:`FileStart`.

    A directory is read as ``<root>/<author>/<file>.jsonl``: only files exactly one
    level down are read, and each is bound to the author directory it sits in. A file
    given directly is bound to its parent directory's name. A file that cannot be read
    (not a regular file, wrong depth, over the size cap, not UTF-8) is reported and
    skipped, so one odd file cannot block everyone else's entries."""
    lines: list[str | FileStart] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            root = p.resolve()
            files = []
            for f in sorted(p.rglob("*.jsonl")):
                rel = f.relative_to(p).parts
                label = f"{root.name}/{'/'.join(rel)}"
                files.append((f, FileStart(label, rel[0] if len(rel) == 2 else None,
                                           None if len(rel) == 2 else "expected <author>/<file>.jsonl")))
        else:
            files = [(p, FileStart(f"{p.resolve().parent.name}/{p.name}", p.resolve().parent.name))]
        for f, start in files:
            if start.problem:
                lines.append(start)
                continue
            try:
                if not f.is_file():
                    raise ValueError("not a regular file")
                with f.open("rb") as fh:
                    raw_bytes = fh.read(_MAX_FILE_BYTES + 1)
                if len(raw_bytes) > _MAX_FILE_BYTES:
                    raise ValueError(f"larger than {_MAX_FILE_BYTES} bytes")
                try:
                    text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"not UTF-8 text ({exc.reason})") from exc
            except (OSError, ValueError) as exc:
                lines.append(FileStart(start.label, None, str(exc)))
                continue
            lines.append(start)
            # Split on newline only, as the writer frames lines: str.splitlines also
            # breaks on U+2028, U+0085 and others that may sit inside a JSON string.
            lines.extend(ln.rstrip("\r") for ln in text.split("\n"))
    return lines


def _normalize_ts(ts: object) -> str | None:
    if not isinstance(ts, str) or not _TS.match(ts):
        return None
    base = ts[:-1]
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in base else "%Y-%m-%dT%H:%M:%S"
    try:
        dt = datetime.strptime(base, fmt).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _unsafe_text(text: str) -> bool:
    """Control, format, line/paragraph-separator and unassigned characters survive
    JSON quoting (U+2028, U+0085, bidi overrides, tag characters) and are refused."""
    return any(
        unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Co", "Cn") and c not in "\n\r\t"
        for c in text
    )


def _is_str_list(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _entry_problem(e: dict) -> str | None:
    """Why a chain-valid entry cannot become an episode, or None."""
    if e.get("v") != SCHEMA_VERSION:
        return f"schema version {e.get('v')!r} is not {SCHEMA_VERSION}"
    for f in ("id", "author"):
        if not isinstance(e.get(f), str) or not _HANDLE.fullmatch(e[f]):
            return f"{f} is missing or not a plain handle"
    if not re.fullmatch(re.escape(_id_prefix(e["author"])) + r"-[0-9]{14}-[0-9a-f]{8}", e["id"], re.ASCII):
        return "id must be <author handle>-<14 digits>-<8 hex>"
    for f in ("agent", "session"):
        if e.get(f) is not None and not (
            isinstance(e[f], str) and _AGENT.fullmatch(e[f])
        ):
            return f"{f} must be a plain handle"
    # An owner may be a client's display name (client:Acme Corp), so it is printable
    # text of bounded length, and it is rendered quoted.
    if e.get("owner") is not None and not (
        isinstance(e["owner"], str) and e["owner"].strip()
        and len(e["owner"]) <= 200 and e["owner"].isprintable()
        and not _unsafe_text(e["owner"])
    ):
        return "owner must be printable text of at most 200 characters"
    if e.get("type") not in TYPES:
        return f"type {e.get('type')!r} is not one of {', '.join(TYPES)}"
    ts = _normalize_ts(e.get("ts"))
    if ts is None:
        return "ts is not an ISO-8601 UTC timestamp"
    if datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc) \
            > datetime.now(timezone.utc) + _FUTURE_SKEW:
        return "ts is more than a day in the future"
    if e["type"] in ("decision", "constraint") and e.get("kind") not in KINDS:
        return f"a {e['type']} needs kind: ruling or practice"
    if e.get("kind") is not None and e["kind"] not in KINDS:
        return "kind must be ruling or practice"
    for f, cap in _TEXT_FIELDS.items():
        if e.get(f) is not None:
            if not isinstance(e[f], str):
                return f"{f} must be a string"
            if len(e[f]) > cap:
                return f"{f} is longer than {cap} characters"
            if _unsafe_text(e[f]):
                return f"{f} contains control, format or separator characters"
    if e.get("kind") == "ruling" and not (e.get("words") or "").strip():
        return "a ruling needs the decider's own words"
    for f in ("paths", "supersedes", "refs"):
        if e.get(f) is not None and not _is_str_list(e[f]):
            return f"{f} must be a list of non-empty strings"
    if len(e.get("paths") or []) > _MAX_PATHS or any(
        len(p) > _MAX_PATH_CHARS for p in e.get("paths") or []
    ):
        return f"paths: at most {_MAX_PATHS} globs of {_MAX_PATH_CHARS} characters"
    if any(_unsafe_text(p) for p in e.get("paths") or []):
        return "paths contain control, format or separator characters"
    if len(e.get("supersedes") or []) > _MAX_PATHS:
        return f"supersedes names more than {_MAX_PATHS} entries"
    for f in ("supersedes", "refs"):
        if not all(_LEDGER_ID.fullmatch(x) for x in e.get(f) or []):
            return f"{f} must name ledger ids"
    for p in e.get("paths") or []:
        if p.startswith("/") or ".." in p.split("/"):
            return f"paths are repo-relative globs, got {p!r}"
    if e["type"] == "retire" and not e.get("supersedes"):
        return "a retire entry must name what it retires in 'supersedes'"
    if e["type"] == "ack" and e.get("supersedes"):
        return "an ack may not supersede anything"
    if e["id"] in (e.get("supersedes") or []):
        return "an entry cannot supersede itself"
    if e["type"] not in ("ack", "retire") and not any(
        (e.get(f) or "").strip() for f in ("words", "reason", "summary")
    ):
        return "an entry with no words, reason or summary says nothing"
    return None


def _q(text: str) -> str:
    """A free-text field as a JSON string: quotes and newlines are escaped, so a
    field can never close its own quote or start a line that reads as another
    entry's header."""
    return json.dumps(text.strip(), ensure_ascii=False)


def render_content(e: dict) -> str:
    """The episode text. Every free-text field is a quoted, escaped string after a
    fixed label; the author, agent and owner are validated handles. The decider's
    words are attributed; a summary is always labelled as the author's summary,
    never as the decision."""
    author = e["author"]
    if e["type"] == "retire":
        text = f"[team ledger] retire by {author}: retired {', '.join(e['supersedes'])}."
        if (e.get("reason") or "").strip():
            text += f" Reason: {_q(e['reason'])}"
        return text
    kind = f" ({e['kind']})" if e.get("kind") else ""
    head = f"[team ledger] {e['type']}{kind}, entered by {author}"
    if e.get("agent"):
        head += f" via {e['agent']}"
    parts = [head + "."]
    if (e.get("words") or "").strip():
        owner = f" (owner {_q(e['owner'])})" if e.get("owner") else ""
        parts.append(f"Decider's words{owner}: {_q(e['words'])}.")
    elif e.get("owner"):
        parts.append(f"Owner: {_q(e['owner'])}.")
    if (e.get("reason") or "").strip():
        parts.append(f"Reason: {_q(e['reason'])}")
    if e.get("paths"):
        parts.append(f"Paths: {', '.join(_q(p) for p in e['paths'])}.")
    if (e.get("recheck") or "").strip():
        parts.append(f"Recheck: {_q(e['recheck'])}")
    if (e.get("summary") or "").strip():
        parts.append(f"Summary by {author}: {_q(e['summary'])}")
    return " ".join(parts)


@dataclass(frozen=True)
class FileStart:
    """Starts a new file in a stream of lines, naming its path.

    Only :func:`read_ledger_lines` makes one, and it is an OBJECT in the stream, not
    text, so the contents of a ledger file can never produce one: a line that merely
    looks like a marker is just a line that is not an entry. It binds the authors that
    follow to the directory the path sits in. A stream of plain strings (stdin) has no
    paths, so it binds nothing and is one trust unit: the caller vouches for it."""

    label: str
    # The author this file may hold entries for, taken from where the file sits
    # (``<root>/<author>/<file>.jsonl``), not parsed back out of the label. None
    # binds nothing.
    author: str | None = None
    # Set when the file was found but cannot be read as a ledger file (not a regular
    # file, wrong depth, unreadable). It is reported and the rest of the import goes on.
    problem: str | None = None


def _path_author_ok(expected: str | None, author: object) -> bool:
    """With an expected author known (from where the file sits), an entry's author
    must be it. Without one nothing can be bound."""
    if expected is None:
        return True
    return isinstance(author, str) and expected in (author, _id_prefix(author))


def _verified_chains(
    lines: Iterable[str | FileStart], report: TeamImportReport
) -> list[dict]:
    """Entries that pass the chain walk, in stream order.

    A chain is ONE CONTIGUOUS RUN: it starts at a ``prev == ""`` root and each
    following line must name the hash of the line accepted just before it. Lines
    are never stitched across files or lines, so an entry cannot extend a chain
    that lives elsewhere, and a line that does not continue the run is refused
    (with every later line of that run, which names it as ``prev``)."""
    seen_hash: dict[str, dict] = {}
    out: list[dict] = []
    expected: str | None = None
    labelled = False
    last_hash: str | None = None
    run_author: object = None
    file_started = False  # a labelled file may hold ONE chain: one root, then its children
    for n, line in enumerate(lines, 1):
        if n > _MAX_LINES:
            report.chain_problems.append(
                f"more than {_MAX_LINES} lines; the rest were not read"
            )
            break
        if isinstance(line, FileStart):
            expected, labelled, last_hash, run_author, file_started = (
                line.author, True, None, None, False)
            if line.problem:
                report.chain_problems.append(f"{line.label}: {line.problem}, not read")
                expected, labelled = "\0skip", True  # nothing in this file can bind
            continue
        if not line.strip():
            continue
        if len(line) > _MAX_LINE_CHARS:
            report.chain_problems.append(f"line {n}: longer than {_MAX_LINE_CHARS} characters")
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as exc:
            report.chain_problems.append(f"line {n}: not JSON ({exc.msg})")
            continue
        except (RecursionError, ValueError):
            report.chain_problems.append(f"line {n}: not parseable as an entry")
            continue
        if not isinstance(e, dict) or not isinstance(e.get("hash"), str) \
                or not isinstance(e.get("prev"), str):
            report.chain_problems.append(f"line {n}: not an entry with prev and hash")
            continue
        who = f"line {n} ({e.get('id')!r})"
        try:
            good = e["hash"] == chain_hash(e["prev"], e)
        except (UnicodeEncodeError, ValueError, TypeError, RecursionError):
            good = False
        if not good:
            report.chain_problems.append(
                f"{who}: hash mismatch, the entry was edited after it was written"
            )
            continue
        if e["hash"] in seen_hash:
            # The same line twice (a file given twice) is harmless. A repeat moves no
            # chain state, so a copied line cannot be used to carry a run across files.
            if seen_hash[e["hash"]] != e:
                report.chain_problems.append(f"{who}: repeats a hash with different content")
            elif e["prev"] == last_hash or (e["prev"] == "" and not file_started):
                # A file that starts by repeating an earlier chain (a copy, then an
                # append) continues from the repeat. Authority is not at stake: an
                # entry still has to carry the author its own file is bound to.
                last_hash = e["hash"]
                file_started = True
            continue
        is_root = e["prev"] == ""
        if is_root:
            if labelled and file_started:
                report.chain_problems.append(
                    f"{who}: a second chain root inside one file, not imported"
                )
                continue
        elif e["prev"] != last_hash:
            report.chain_problems.append(
                f"{who}: does not continue the entry before it (a gap, a fork, or "
                "another file's chain), not imported"
            )
            continue
        if not is_root and run_author is not None and e.get("author") != run_author:
            report.chain_problems.append(
                f"{who}: names author {e.get('author')!r} inside {run_author!r}'s "
                "chain; the chain is cut here"
            )
            continue
        if not _path_author_ok(expected, e.get("author")):
            report.chain_problems.append(
                f"{who}: author {e.get('author')!r} is not the author this file is "
                f"bound to ({expected!r}), not imported"
            )
            continue
        run_author = e.get("author")
        last_hash = e["hash"]
        file_started = True
        seen_hash[e["hash"]] = e
        out.append(e)
    return out


def import_ledger(
    store: Store,
    lines: Iterable[str | FileStart],
    *,
    dry_run: bool = False,
    link_authority: Iterable[str] = (),
) -> TeamImportReport:
    """Import ledger lines into ``store``. See the module docstring.

    Never raises for bad ledger content: every refusal is in the report. A store
    error (locked database, corrupt file) raises as it does everywhere else.
    """
    report = TeamImportReport(dry_run=dry_run)
    valid: list[dict] = []
    for e in _verified_chains(lines, report):
        problem = _entry_problem(e)
        if problem:
            report.rejected.append({"id": e.get("id"), "reason": problem})
        else:
            valid.append(e)
    # The same ledger id with two different hashes inside one batch (an ack and a
    # retire included): import neither, so the outcome does not depend on line order.
    by_id: dict[str, set[str]] = {}
    for e in valid:
        by_id.setdefault(e["id"], set()).add(e["hash"])
    clash = {i for i, hs in by_id.items() if len(hs) > 1}
    for i in sorted(clash):
        report.conflicts.append(
            {"id": i, "reason": "two different entries in this import carry this id"}
        )
    # An entry dropped for a clash would leave its descendants orphaned, so every
    # entry of an author involved in a clash is dropped with it.
    clash_authors = {e["author"] for e in valid if e["id"] in clash}
    records: list[dict[str, Any]] = []
    for e in valid:
        if e["author"] in clash_authors:
            continue
        if e["type"] == "ack":
            report.skipped_ack.append(e["id"])
            continue
        records.append({
            "entry_id": e["id"],
            "hash": e["hash"],
            "type": _EPISODE_TYPE[e["type"]],
            "source": f"team:{e['author']}",
            "timestamp": _normalize_ts(e["ts"]),
            "content": render_content(e),
            "metadata": {"team": {**e, "entry_id": e["id"]}},
            "supersedes": list(e.get("supersedes") or []),
        })
    result = store.import_team_entries(
        records, dry_run=dry_run, link_authority=tuple(link_authority),
    )
    report.imported = result["imported"]
    report.already_present = result["already_present"]
    report.conflicts.extend(result["conflicts"])
    report.links_made = result["links_made"]
    report.links_pending = result["links_pending"]
    report.links_refused = result["links_refused"]
    report.links_unauthorized = result["links_unauthorized"]
    return report
