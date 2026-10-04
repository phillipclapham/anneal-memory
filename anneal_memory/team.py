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
commits) and nothing here sees git committers or file paths; the input is one trust
unit the exporter vouches for (Levain's ``team export --jsonl`` on stdin; a ledger
directory is deliberately not read directly); anyone who can write to the ledger can
append a line under an author whose chain they extend, or a fork that freezes a
teammate's chain at the fork point (reported, never silent); an entry removed from the
ledger stays in an engineer's store, because only a signed ``retire`` reaches it; an
``ack`` leaves no record, so a later entry reusing its id in another call is not seen as
a clash; the first importer of an id wins; and a local writer with ``Store.record`` can
plant a ``team:`` row that claims an entry id.
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
_MAX_JSON_DEPTH = 32  # a ledger entry nests two levels; deeper is never an entry
_MAX_FILE_BYTES = 64 * 1024 * 1024
_FUTURE_SKEW = timedelta(days=1)
_TS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z", re.ASCII)


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


def read_ledger_lines(paths: Iterable[str | Path]) -> list[str]:
    """Lines of the ledger FILES named, one after another, as one stream.

    A named file is read as the stream it is: nothing binds its entries to an author
    by where it sits, so give it only files you vouch for (see the module docstring).
    A directory is refused: reading a ledger tree directly is not supported, because
    the paths in it are not a safe place to take an author from. Line framing is the
    writer's: newline only (``str.splitlines`` also breaks on U+2028 and U+0085, which
    may sit inside a JSON string). Raises ``ValueError`` for a path that cannot be read."""
    lines: list[str] = []
    for raw in paths:
        try:
            p = Path(raw).expanduser()
        except RuntimeError as exc:
            raise ValueError(f"{raw}: cannot expand the path ({exc})") from exc
        if p.is_dir():
            raise ValueError(
                f"{p}: a directory is not read directly; pipe the ledger's exporter "
                "(`levain team export --jsonl`) into team-import -"
            )
        if not p.is_file():
            raise ValueError(f"{p}: not a regular file")
        try:
            with p.open("rb") as fh:
                data = fh.read(_MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise ValueError(f"{p}: cannot be read ({exc.strerror or exc})") from exc
        if len(data) > _MAX_FILE_BYTES:
            raise ValueError(f"{p}: larger than {_MAX_FILE_BYTES} bytes")
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{p}: not UTF-8 text ({exc.reason})") from exc
        lines.extend(ln.rstrip("\r") for ln in text.split("\n"))
    return lines


def _normalize_ts(ts: object) -> str | None:
    if not isinstance(ts, str) or not _TS.fullmatch(ts):
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


_NO_RUN = object()  # no chain run has started: distinct from a root whose author is null


def _refuse_constant(name: str) -> None:
    """NaN and Infinity are not JSON; the entry would store as text other readers refuse."""
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise ValueError("a number outside the float range is not accepted")
    return value


def _nests_too_deep(line: str) -> bool:
    """True when the line's brackets nest deeper than an entry ever does. Checked
    BEFORE ``json.loads``: on Windows a few hundred thousand open brackets made the
    parser run for over twenty minutes (CI hang, 2026-10-04), where Linux and macOS
    raise RecursionError at once."""
    depth = 0
    in_str = False
    esc = False
    for ch in line:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            depth += 1
            if depth > _MAX_JSON_DEPTH:
                return True
        elif ch in "]}":
            depth -= 1
    return False


def _is_str_list(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _entry_problem(e: dict) -> str | None:
    """Why a chain-valid entry cannot become an episode, or None."""
    if type(e.get("v")) is not int or e["v"] != SCHEMA_VERSION:
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


def _verified_chains(
    lines: Iterable[str], report: TeamImportReport
) -> list[dict]:
    """Entries that pass the chain walk, in stream order.

    The input is ONE TRUST UNIT: a stream the caller vouches for (Levain's export on
    stdin), with the files of a ledger one after another. A chain is a contiguous
    run: it starts at a ``prev == ""`` root and each following line must name the
    hash of the line accepted just before it, so a line that does not continue its
    run is refused (with every later line of that run, which names it as ``prev``).
    An exact repeat of an earlier line is skipped and moves nothing, so a copied line
    cannot be used to attach a run to another author's chain; a copy of a chain's
    prefix followed by an append is continued only when nothing else sits between the
    copy's source and it. The stream carries no file names, so no author can be bound
    to a path."""
    seen_hash: dict[str, dict] = {}
    out: list[dict] = []
    last_hash: str | None = None
    run_author: object = _NO_RUN
    for n, line in enumerate(lines, 1):
        if n > _MAX_LINES:
            report.chain_problems.append(
                f"more than {_MAX_LINES} lines; the rest were not read"
            )
            break
        if not line.strip():
            continue
        if len(line) > _MAX_LINE_CHARS:
            report.chain_problems.append(f"line {n}: longer than {_MAX_LINE_CHARS} characters")
            continue
        if _nests_too_deep(line):
            report.chain_problems.append(f"line {n}: nested deeper than an entry can be")
            continue
        try:
            e = json.loads(line, parse_constant=_refuse_constant, parse_float=_finite_float)
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
            continue  # an exact repeat moves no chain state: nothing can be carried across it
        is_root = e["prev"] == ""
        if not is_root and e["prev"] != last_hash:
            report.chain_problems.append(
                f"{who}: does not continue the entry before it (a gap, a fork, or "
                "another file's chain), not imported"
            )
            continue
        if not is_root and e.get("author") != run_author:
            report.chain_problems.append(
                f"{who}: names author {e.get('author')!r} inside {run_author!r}'s "
                "chain; the chain is cut here"
            )
            continue
        run_author = e.get("author")
        last_hash = e["hash"]
        seen_hash[e["hash"]] = e
        out.append(e)
    return out


def import_ledger(
    store: Store,
    lines: Iterable[str],
    *,
    dry_run: bool = False,
    link_authority: Iterable[str] = (),
) -> TeamImportReport:
    """Import ledger lines into ``store``. See the module docstring.

    Never raises for bad ledger content: every refusal is in the report. A store
    error (locked database, corrupt file) raises as it does everywhere else.
    """
    if isinstance(link_authority, (str, bytes)):
        raise TypeError("link_authority is a collection of patterns, not one string")
    report = TeamImportReport(dry_run=dry_run)
    valid: list[dict] = []
    verified = _verified_chains(lines, report)
    for e in verified:
        problem = _entry_problem(e)
        if problem:
            rid = e.get("id")
            report.rejected.append(
                {"id": rid[:100] if isinstance(rid, str) else repr(rid)[:100], "reason": problem}
            )
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
    # An entry dropped for a clash takes its descendants with it (they name it, or one
    # of them, as prev), so nothing is imported orphaned and nobody else's chain goes.
    dropped: set[str] = set()
    records: list[dict[str, Any]] = []
    valid_hashes = {e["hash"] for e in valid}
    for e in verified:
        # Descent is read over every chain-verified entry: a semantically rejected
        # middle entry must not let its descendants slip past a dropped ancestor.
        if (e["hash"] in valid_hashes and e["id"] in clash) or e["prev"] in dropped:
            dropped.add(e["hash"])
            continue
        if e["hash"] not in valid_hashes:
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
