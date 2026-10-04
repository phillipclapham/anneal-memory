"""Team ledger import: a teammate's decisions enter an engineer's store as
provenance-carrying episodes.

A team ledger is a set of append-only JSONL files, one hash chain per file; the
writer is Levain's team layer, and :func:`canonical` / :func:`chain_hash` here are the
rule it hashes by (a golden-vector test pins the bytes). This
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
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
_HANDLE = re.compile(r"^[A-Za-z0-9._@:+-]{1,200}$")
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
    dry_run: bool = False

    @property
    def clean(self) -> bool:
        """True when nothing was refused or in conflict. Pending links are not a
        problem: a target not fetched yet completes on a later import."""
        return not (self.rejected or self.chain_problems or self.conflicts
                    or self.links_refused)

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
            "imported_ids": [i["id"] for i in self.imported],
        }


def read_ledger_lines(paths: Iterable[str | Path]) -> list[str]:
    """Lines of every ledger file given; a directory contributes every
    ``*.jsonl`` under it, in sorted order."""
    lines: list[str] = []
    for raw in paths:
        p = Path(raw).expanduser()
        files = sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
        for f in files:
            lines.extend(f.read_text(encoding="utf-8").splitlines())
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


def _is_str_list(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x.strip() for x in v)


def _entry_problem(e: dict) -> str | None:
    """Why a chain-valid entry cannot become an episode, or None."""
    if e.get("v") != SCHEMA_VERSION:
        return f"schema version {e.get('v')!r} is not {SCHEMA_VERSION}"
    for f in ("id", "author"):
        if not isinstance(e.get(f), str) or not _HANDLE.match(e[f]):
            return f"{f} is missing or not a plain handle"
    if e.get("type") not in TYPES:
        return f"type {e.get('type')!r} is not one of {', '.join(TYPES)}"
    if _normalize_ts(e.get("ts")) is None:
        return "ts is not an ISO-8601 UTC timestamp"
    if e["type"] in ("decision", "constraint") and e.get("kind") not in KINDS:
        return f"a {e['type']} needs kind: ruling or practice"
    if e.get("kind") is not None and e["kind"] not in KINDS:
        return "kind must be ruling or practice"
    for f in ("words", "summary", "reason", "recheck", "owner", "agent", "session"):
        if e.get(f) is not None and not isinstance(e[f], str):
            return f"{f} must be a string"
    if e.get("kind") == "ruling" and not (e.get("words") or "").strip():
        return "a ruling needs the decider's own words"
    for f in ("paths", "supersedes", "refs"):
        if e.get(f) is not None and not _is_str_list(e[f]):
            return f"{f} must be a list of non-empty strings"
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


def render_content(e: dict) -> str:
    """The episode text. The decider's words are quoted and attributed; a
    summary is always labelled as the author's summary, never as the decision."""
    author = e["author"]
    if e["type"] == "retire":
        text = f"Team retire by {author}: retired {', '.join(e['supersedes'])}."
        if (e.get("reason") or "").strip():
            text += f" Reason: {e['reason'].strip()}"
        return text
    kind = f" ({e['kind']})" if e.get("kind") else ""
    head = f"Team {e['type']}{kind}, entered by {author}"
    if e.get("agent"):
        head += f" via {e['agent']}"
    parts = [head + "."]
    if (e.get("words") or "").strip():
        owner = f" ({e['owner']})" if e.get("owner") else ""
        parts.append(f"Decider's words{owner}: \"{e['words'].strip()}\".")
    elif e.get("owner"):
        parts.append(f"Owner: {e['owner']}.")
    if (e.get("reason") or "").strip():
        parts.append(f"Reason: {e['reason'].strip()}")
    if e.get("paths"):
        parts.append(f"Paths: {', '.join(e['paths'])}.")
    if (e.get("recheck") or "").strip():
        parts.append(f"Recheck: {e['recheck'].strip()}")
    if (e.get("summary") or "").strip():
        parts.append(f"Summary by {author}: {e['summary'].strip()}")
    return " ".join(parts)


def _verified_chains(
    lines: Iterable[str], report: TeamImportReport
) -> list[dict]:
    """Entries that pass the chain walk, oldest first within each chain.

    Chains are rebuilt from the ``prev`` -> ``hash`` links, so the input needs no
    file names and may be several files concatenated."""
    by_prev: dict[str, list[dict]] = {}
    seen_hash: dict[str, dict] = {}
    all_entries: list[dict] = []
    for n, line in enumerate(lines, 1):
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
        if not isinstance(e, dict) or not isinstance(e.get("hash"), str) \
                or not isinstance(e.get("prev"), str):
            report.chain_problems.append(f"line {n}: not an entry with prev and hash")
            continue
        if e["hash"] in seen_hash:
            report.chain_problems.append(f"line {n} ({e.get('id')}): repeats a hash already seen")
            continue
        if e["hash"] != chain_hash(e["prev"], e):
            report.chain_problems.append(
                f"line {n} ({e.get('id')}): hash mismatch, the entry was edited after it was written"
            )
            continue
        seen_hash[e["hash"]] = e
        by_prev.setdefault(e["prev"], []).append(e)
        all_entries.append(e)

    out: list[dict] = []
    reached: set[str] = set()
    for start in by_prev.get("", []):
        cur: dict | None = start
        author = start.get("author")
        while cur is not None:
            if cur.get("author") != author:
                report.chain_problems.append(
                    f"chain of {author!r}: entry {cur.get('id')} names author "
                    f"{cur.get('author')!r}; the chain is cut here"
                )
                break
            nxt = by_prev.get(cur["hash"], [])
            out.append(cur)
            reached.add(cur["hash"])
            if len(nxt) > 1:
                report.chain_problems.append(
                    f"fork after {cur.get('id')}: {len(nxt)} entries claim it as prev; "
                    "the chain is cut here"
                )
                break
            cur = nxt[0] if nxt else None
    for e in all_entries:
        if e["hash"] not in reached:
            report.chain_problems.append(
                f"entry {e.get('id')}: not reachable from a chain start "
                "(a gap, a fork or a cut chain), not imported"
            )
    return out


def import_ledger(
    store: Store, lines: Iterable[str], *, dry_run: bool = False
) -> TeamImportReport:
    """Import ledger lines into ``store``. See the module docstring.

    Never raises for bad ledger content: every refusal is in the report. A store
    error (locked database, corrupt file) raises as it does everywhere else.
    """
    report = TeamImportReport(dry_run=dry_run)
    records: list[dict[str, Any]] = []
    for e in _verified_chains(lines, report):
        problem = _entry_problem(e)
        if problem:
            report.rejected.append({"id": e.get("id"), "reason": problem})
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
    result = store.import_team_entries(records, dry_run=dry_run)
    report.imported = result["imported"]
    report.already_present = result["already_present"]
    report.conflicts = result["conflicts"]
    report.links_made = result["links_made"]
    report.links_pending = result["links_pending"]
    report.links_refused = result["links_refused"]
    return report
