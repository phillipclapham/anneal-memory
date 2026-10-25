"""Reader scopes: build a projected replica of a store that a scoped reader may open.

    anneal-memory scope-project --src DIR --out ROOT --policy FILE

A scope is a policy file (JSON). The projection is filtered when it is BUILT, so its store
never holds what the scope denies; a server for a scoped reader opens only
``<ROOT>/current/``. A read-time filter alone is not this module's model.

Policy (version 1)::

    {"version": 1, "name": "reader-b",
     "allow": ["upper acme"],
     "rules": [
       {"name": "client-security", "all_of": [["acme", "zorblax"], ["dkim", "unencrypted"]],
        "window": "line"},
       {"name": "widgetco", "any_of": ["widgetco", "bramble"]}],
     "deny_episode_ids": ["<episode id>"]}

Term semantics: lowercase; runs of space/'_'/'-' compare as one space; a term of 6+
characters matches as a substring, a shorter one needs a boundary on both sides; text inside
an ``allow`` phrase is blanked (an occurrence is excused only when it lies wholly inside one
allow match), and an allow phrase that would hide a term refuses the policy.

``any_of`` hits when any term occurs. ``all_of`` hits when every group has a term inside the
same window: ``line`` (one newline-separated line) or ``document`` (the whole text). What a
"document" is: one table cell; one crystal (its strings, one per line); and in the continuity
file one LINE, since a neocortex line is one item and a whole-file window would match any file
that names a client in one section and a security term in another.

``deny_episode_ids`` are computed by the caller (a host derives them from its own tags), so
this package needs no host tag store.

Publishing is by generation (releases/current): each build is written whole into its own
``gen-<stamp>/``, checked, and the ``current`` symlink is swapped to it in one rename. A
reader sees one complete generation or none. Everything, policy loading included, runs under
one lock, and any failure UNPUBLISHES (removes ``current``): serving nothing is the
fail-closed state.

Fail-closed, in order:
1. Policy: a malformed file refuses (see :func:`parse_policy`).
2. Episodes: one whose content, source, session id or metadata a rule hits, or whose id is
   denied, is deleted (the store's AFTER DELETE triggers clear derived rows); association,
   supersession and retired-origin-key rows naming a deleted episode go too.
3. Matching rows of the pattern/wrap-history tables in ROW_DROP_TABLES are deleted.
4. The continuity file loses every " · "-separated segment that a rule hits (a line's own
   label is kept); a line that still hits once its segments are filtered goes whole. This is
   INTERIM (``continuity: lexical-segment (interim)``): prose that refers to a scoped fact
   without naming a term survives. A crystal any rule hits is dropped whole; the meta JSON
   must hit nothing.
5. POST-CHECK, the gate: every text value of every table, every JSON key and value and the
   decoded text of every output file are rescanned with every rule, the raw db bytes with the
   ``any_of`` long terms; no episode with a denied id may remain. Any hit and nothing is published.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .store import register_writer_schema

__all__ = [
    "ScopeError", "Matcher", "Rule", "Policy", "build_matcher", "parse_policy",
    "load_policy", "project", "unpublish", "CONTINUITY_MODE",
]

STEM = "memory"
DB = f"{STEM}.db"
CONTINUITY = f"{STEM}.continuity.md"
META = f"{STEM}.continuity.meta.json"
CRYSTAL = f"{STEM}.crystal.json"
SCOPE_FILE = "scope.json"
CURRENT = "current"
CONTINUITY_MODE = "lexical-segment (interim)"

# Tables whose rows carry pattern names or wrap-history prose. No remote tool serves them; a
# matching row is deleted. A hit in any table NOT listed here (or the episodes table) refuses
# the build until someone decides how that table is handled.
ROW_DROP_TABLES = (
    "pattern_history", "wrap_graduations", "pattern_levels", "pattern_grounding",
    "pattern_aliases", "pattern_associations", "state_keys",
)

LONG_TERM = 6  # terms this long or longer match as substrings; shorter ones need boundaries

POLICY_VERSION = 1
_POLICY_KEYS = frozenset({"version", "name", "allow", "rules", "deny_episode_ids"})
_RULE_KEYS = frozenset({"name", "any_of", "all_of", "window"})
_WINDOWS = ("line", "document")
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# A line's own label ("[decided, date, Phill]" or "- **▶▶ MONEY**"), kept when its first segment goes.
_LINE_LABEL = re.compile(r"^\s*(?:\[[^\]]{1,80}\]|-\s+\*\*[^*]{1,60}\*\*)")
_SEP = re.compile(r"[\s_-]+")
# Joins a JSON object's strings into one document: one string per line, and a NUL line
# between them so no term can run from one value into the next.
_JOIN = "\n\x00\n"


class ScopeError(Exception):
    """The projection was refused (and, once building began, unpublished)."""


# -- matching --


def _norm(text: str) -> str:
    """Lowercase, every run of spaces/'_'/'-' as one space: the form terms are compared in."""
    return _SEP.sub(" ", text.lower())


@dataclass(frozen=True)
class Matcher:
    long_terms: tuple[str, ...]  # substrings of the normalised text (str "in": C speed)
    short: re.Pattern[str] | None  # short terms, bounded; a lookahead, so every start is seen
    allow: re.Pattern[str] | None  # allow phrases, bounded

    def _hit(self, text: str, short: bool) -> bool:
        t = _norm(text)
        spans = [a.span() for a in self.allow.finditer(t)] if self.allow is not None else []
        if not spans:
            return any(x in t for x in self.long_terms) or (
                short and self.short is not None and self.short.search(t) is not None)

        # A term occurrence is excused only when it lies wholly inside one allow match:
        # "upper acme" excuses "acme", never "acme dental" running past it.
        def free(i: int, j: int) -> bool:
            return not any(a <= i and j <= b for a, b in spans)

        for x in self.long_terms:
            i = t.find(x)
            while i != -1:
                if free(i, i + len(x)):
                    return True
                i = t.find(x, i + 1)
        if short and self.short is not None:
            for mm in self.short.finditer(t):
                if free(mm.start(1), mm.end(1)):
                    return True
        return False

    def search(self, text: str) -> bool:
        return self._hit(text, short=True)

    def search_long(self, text: str) -> bool:
        return self._hit(text, short=False)


def build_matcher(terms: set[str], allow: set[str] | None = None) -> Matcher:
    normed = {_norm(t).strip() for t in terms} - {""}
    if not normed:
        raise ScopeError("empty term list: refusing (it would pass everything)")
    long_terms = tuple(sorted(t for t in normed if len(t) >= LONG_TERM))
    short = sorted((t for t in normed if len(t) < LONG_TERM), key=len, reverse=True)
    short_re = (re.compile(r"(?<![a-z0-9])(?=(" + "|".join(map(re.escape, short)) + r")(?![a-z]))")
                if short else None)
    allow_n = sorted({_norm(a).strip() for a in (allow or ())} - {""}, key=len, reverse=True)
    allow_re = (re.compile(r"(?<![a-z0-9])(?:" + "|".join(map(re.escape, allow_n)) + r")(?![a-z0-9])")
                if allow_n else None)
    m = Matcher(long_terms, short_re, allow_re)
    swallowed = sorted(t for t in normed if not m.search(t))
    if swallowed:
        raise ScopeError(f"an allow phrase hides term(s): {swallowed}")
    return m


# -- policy --


@dataclass(frozen=True)
class Rule:
    name: str
    groups: tuple[Matcher, ...]  # any_of = one group; all_of = every group must hit
    window: str  # "line" | "document" (any_of: "document"; the window cannot change its answer)
    any_of: bool

    def search(self, text: str) -> bool:
        if self.any_of:
            return self.groups[0].search(text)
        if self.window == "document":
            return all(g.search(text) for g in self.groups)
        return any(all(g.search(line) for g in self.groups) for line in text.split("\n"))

    def search_json(self, obj: Any) -> bool:
        return self.search(_JOIN.join(_json_strings(obj)))


@dataclass(frozen=True)
class Policy:
    name: str
    rules: tuple[Rule, ...]
    deny_episode_ids: frozenset[str]
    sha256: str

    def hit(self, text: str) -> str | None:
        """The name of the first rule that hits ``text``, or None."""
        for r in self.rules:
            if r.search(text):
                return r.name
        return None

    def hits(self, text: str) -> list[str]:
        return [r.name for r in self.rules if r.search(text)]

    def hit_json(self, obj: Any) -> bool:
        return any(r.search_json(obj) for r in self.rules)


def _str_list(v: Any, where: str) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ScopeError(f"policy: {where} must be a list of strings")
    return v


def parse_policy(raw: bytes) -> Policy:
    """Parse and validate a policy; every malformation refuses with ScopeError."""
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ScopeError(f"policy: not valid UTF-8 JSON: {e}") from e
    if not isinstance(doc, dict):
        raise ScopeError("policy: the top level must be an object")
    extra = sorted(set(doc) - _POLICY_KEYS)
    if extra:
        raise ScopeError(f"policy: unknown key(s) {extra}")
    if doc.get("version") != POLICY_VERSION or isinstance(doc.get("version"), bool):
        raise ScopeError(f"policy: version must be {POLICY_VERSION}, got {doc.get('version')!r}")
    name = doc.get("name")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ScopeError(f"policy: name must match {_NAME.pattern}, got {name!r}")
    allow = set(_str_list(doc.get("allow", []), "allow"))
    ids = _str_list(doc.get("deny_episode_ids", []), "deny_episode_ids")
    if any(not i.strip() for i in ids):
        raise ScopeError("policy: deny_episode_ids holds an empty id")
    raw_rules = doc.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ScopeError("policy: rules must be a list")
    rules: list[Rule] = []
    seen: set[str] = set()
    for i, r in enumerate(raw_rules):
        if not isinstance(r, dict):
            raise ScopeError(f"policy: rule {i} must be an object")
        extra = sorted(set(r) - _RULE_KEYS)
        if extra:
            raise ScopeError(f"policy: rule {i} has unknown key(s) {extra}")
        rname = r.get("name")
        if not isinstance(rname, str) or not _NAME.match(rname):
            raise ScopeError(f"policy: rule {i} name must match {_NAME.pattern}, got {rname!r}")
        if rname in seen:
            raise ScopeError(f"policy: duplicate rule name {rname!r}")
        seen.add(rname)
        if ("any_of" in r) == ("all_of" in r):
            raise ScopeError(f"policy: rule {rname!r} needs exactly one of any_of / all_of")
        if "any_of" in r:
            if "window" in r:
                raise ScopeError(f"policy: rule {rname!r}: window applies to all_of only")
            terms = _str_list(r["any_of"], f"rule {rname!r} any_of")
            if not terms:
                raise ScopeError(f"policy: rule {rname!r} has an empty term list")
            rules.append(Rule(rname, (build_matcher(set(terms), allow),), "document", True))
        else:
            window = r.get("window")
            if window not in _WINDOWS:
                raise ScopeError(f"policy: rule {rname!r} window must be one of {list(_WINDOWS)}")
            groups = r["all_of"]
            if not isinstance(groups, list) or not groups:
                raise ScopeError(f"policy: rule {rname!r} all_of must be a non-empty list of groups")
            ms = []
            for j, g in enumerate(groups):
                terms = _str_list(g, f"rule {rname!r} group {j}")
                if not terms:
                    raise ScopeError(f"policy: rule {rname!r} group {j} is empty")
                ms.append(build_matcher(set(terms), allow))
            rules.append(Rule(rname, tuple(ms), window, False))
    if not rules and not ids:
        raise ScopeError("policy: no rules and no deny_episode_ids (it would pass everything)")
    return Policy(name, tuple(rules), frozenset(ids), hashlib.sha256(raw).hexdigest())


def load_policy(path: Path) -> Policy:
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise ScopeError(f"policy: cannot read {path}: {e}") from e
    return parse_policy(raw)


# -- building --


def _text_hits(conn: sqlite3.Connection, p: Policy) -> list[str]:
    hits = []
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'")]
    for t in tables:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
        for row in conn.execute(f'SELECT * FROM "{t}"'):
            for col, v in zip(cols, row):
                if isinstance(v, (bytes, bytearray)):
                    v = bytes(v).decode("utf-8", "replace")
                if isinstance(v, str) and p.hit(v):
                    hits.append(f"{t}.{col}")
    return hits


def _denied_left(conn: sqlite3.Connection, p: Policy) -> int:
    if not p.deny_episode_ids:
        return 0
    ids = sorted(p.deny_episode_ids)
    return sum(
        conn.execute(f"SELECT count(*) FROM episodes WHERE id IN ({','.join('?' * len(chunk))})",
                     chunk).fetchone()[0]
        for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)))


def _json_strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _json_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_strings(v)


def _json_units(obj: Any) -> Iterator[Any]:
    """A JSON file's units for a windowed rule: each top-level key, each element of a
    top-level list value, each other top-level value; a non-object document is one unit."""
    if not isinstance(obj, dict):
        yield obj
        return
    for k, v in obj.items():
        yield str(k)
        if isinstance(v, list):
            yield from v
        else:
            yield v


def _raw_long_hit(data: bytes, p: Policy) -> bool:
    text = data.decode("utf-8", "ignore")
    return any(r.any_of and r.groups[0].search_long(text) for r in p.rules)


def project_db(src: Path, dst: Path, p: Policy, workdir: Path) -> dict[str, Any]:
    """Write the filtered db to ``dst``; return the drop counts."""
    work = workdir / "work.db"
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    w = sqlite3.connect(work)
    try:
        s.backup(w)
        # The generation guard triggers refuse writers that do not declare the store's
        # generation. A store newer than this package refuses here with sqlite3.Error,
        # reported as a ScopeError below.
        register_writer_schema(w)
        rows = w.execute("SELECT id, content, source, session_id, metadata FROM episodes").fetchall()
        by_rule: dict[str, int] = {r.name: 0 for r in p.rules}
        by_id = 0
        drop = []
        for eid, *fields in rows:
            names = sorted({n for v in fields if isinstance(v, str) for n in p.hits(v)})
            denied = eid in p.deny_episode_ids
            if names or denied:
                drop.append(eid)
                for n in names:
                    by_rule[n] += 1
                by_id += denied
        w.execute("CREATE TEMP TABLE drop_ids (id TEXT PRIMARY KEY)")
        w.executemany("INSERT INTO drop_ids VALUES (?)", [(d,) for d in drop])
        w.execute("DELETE FROM associations WHERE episode_a IN drop_ids OR episode_b IN drop_ids")
        w.execute("DELETE FROM supersessions WHERE old_id IN drop_ids OR new_id IN drop_ids")
        w.execute("DELETE FROM episodes WHERE id IN drop_ids")
        # the delete trigger records each dropped episode's origin key: a count and a hash
        # of what was filtered, so it goes too
        w.execute("DELETE FROM retired_origin_keys WHERE episode_id IN drop_ids")
        for t in ROW_DROP_TABLES:
            bad = [
                rowid for rowid, *vals in w.execute(f'SELECT rowid, * FROM "{t}"')
                if any(isinstance(v, str) and p.hit(v) for v in vals)
            ]
            w.executemany(f'DELETE FROM "{t}" WHERE rowid = ?', [(r,) for r in bad])
        w.commit()
        hits = _text_hits(w, p)
        if hits:
            raise ScopeError(f"post-check: scoped terms remain in {sorted(set(hits))}")
        if _denied_left(w, p):
            raise ScopeError("post-check: a denied episode id remains")
        w.execute("VACUUM INTO ?", (str(dst),))
    except sqlite3.Error as e:
        raise ScopeError(f"sqlite: {type(e).__name__}: {e}") from e
    finally:
        w.close()
        s.close()
    v = sqlite3.connect(dst)
    try:
        v.execute("PRAGMA journal_mode=DELETE")
        if v.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ScopeError("projection quick_check failed")
        hits = _text_hits(v, p)
        left = _denied_left(v, p)
    finally:
        v.close()
    if hits:
        raise ScopeError(f"post-check (written db): {sorted(set(hits))}")
    if left:
        raise ScopeError("post-check (written db): a denied episode id remains")
    # Raw bytes too (free pages, anything outside cell values). any_of long terms only: short
    # terms hit random binary in page data, which would refuse at random, and an all_of window
    # has no meaning in page bytes.
    if _raw_long_hit(dst.read_bytes(), p):
        raise ScopeError("post-check (raw db bytes): a scoped term remains")
    return {"total": len(drop), "by_rule": by_rule, "by_id": by_id}


def project_continuity(text: str, p: Policy) -> tuple[str, int]:
    """(filtered text, segments dropped). A line is the unit every rule is applied to."""
    out = []
    dropped = 0
    for line in text.splitlines():
        if not p.hit(line):
            out.append(line)
            continue
        segs = line.split(" · ")
        kept = [seg for seg in segs if not p.hit(seg)]
        if p.hit(" · ".join(kept)):
            # an all_of hit spread across segments that no single segment carries: the line goes
            kept = []
        dropped += len(segs) - len(kept)
        if segs and p.hit(segs[0]) or not kept:
            label = _LINE_LABEL.match(segs[0])
            if label and not p.hit(label.group(0)):
                kept.insert(0, label.group(0) + " (scoped item removed)")
        if any(seg.strip() for seg in kept):
            out.append(" · ".join(kept))
    return "\n".join(out) + ("\n" if text.endswith("\n") else ""), dropped


def project_crystals(doc: dict, p: Policy) -> tuple[dict, int]:
    doc = dict(doc)
    dropped = 0
    for key in ("crystal", "retired"):
        if isinstance(doc.get(key), list):
            kept = [c for c in doc[key] if not p.hit_json(c)]
            dropped += len(doc[key]) - len(kept)
            doc[key] = kept
    return doc, dropped


def _postcheck_files(gen: Path, p: Policy) -> None:
    for name in (CONTINUITY, META, CRYSTAL, SCOPE_FILE):
        f = gen / name
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        if name == CONTINUITY:
            # the continuity file's unit is the line (module doc); any_of is also run over the
            # whole text, so a term broken across a newline is still seen
            bad = any(p.hit(line) for line in text.splitlines()) or any(
                r.any_of and r.search(text) for r in p.rules)
        else:
            # one unit per crystal (the filter's unit) and per other top-level value, so a
            # document window never spans two crystals; any_of also runs over the raw text
            bad = any(r.any_of and r.search(text) for r in p.rules) or any(
                p.hit_json(u) for u in _json_units(json.loads(text)))
        if bad:
            raise ScopeError(f"post-check: scoped terms remain in {name}")


def _lock(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


_ROOT_ENTRY = re.compile(
    r"^(?:\.lock|current|\.current\.tmp|gen-.+|\.stage\..+|\.work\..+|\.DS_Store)$")  # Finder's, on a Mac


def _check_root(root: Path) -> None:
    """Refuse a root holding anything this builder did not make: a wrong --out (the source
    replica itself, say) that the sweep would otherwise touch."""
    odd = sorted(x.name for x in root.iterdir() if not _ROOT_ENTRY.match(x.name))
    if odd:
        raise ScopeError(f"projection root {root} holds unexpected entries {odd}; clear it by hand")


def _sweep(root: Path, keep: str | None) -> None:
    for x in [*root.glob(".stage.*"), *root.glob(".work.*")]:
        shutil.rmtree(x, ignore_errors=True)
    for x in root.glob("gen-*"):
        if x.name != keep:
            shutil.rmtree(x, ignore_errors=True)


def _swap_current(root: Path, gen_name: str) -> None:
    tmp = root / f".{CURRENT}.tmp"
    tmp.unlink(missing_ok=True)
    os.symlink(gen_name, tmp)
    os.replace(tmp, root / CURRENT)


def _unpublish_locked(root: Path) -> None:
    (root / CURRENT).unlink(missing_ok=True)
    _sweep(root, keep=None)


def unpublish(root: Path) -> None:
    """Remove the published projection: serving nothing is the fail-closed state."""
    fd = _lock(root)
    try:
        _unpublish_locked(root)
    finally:
        os.close(fd)


def project(src: Path, root: Path, policy: Policy | Path | str) -> dict[str, Any]:
    """Build and publish a generation of ``src`` under ``policy``, or raise ScopeError having
    UNPUBLISHED, all under the lock. Pass the policy as a path to have it loaded under the
    lock too (so a build can never publish with a stale policy over a newer one); a malformed
    policy then also unpublishes."""
    src, root = Path(src), Path(root)
    if root.is_dir():
        _check_root(root)  # before the lock file too: a foreign dir gets nothing written into it
    fd = _lock(root)
    try:
        try:
            _check_root(root)
            p = policy if isinstance(policy, Policy) else load_policy(Path(policy))
            cur = root / CURRENT
            _sweep(root, keep=os.readlink(cur) if cur.is_symlink() else None)
            stage = Path(tempfile.mkdtemp(dir=root, prefix=".stage."))
            workdir = Path(tempfile.mkdtemp(dir=root, prefix=".work."))  # 0700; this root's sweep owns it
            try:
                if not (src / DB).is_file():
                    raise ScopeError(f"source store missing: {src / DB}")
                episodes = project_db(src / DB, stage / DB, p, workdir)
                seg_dropped = 0
                if (src / CONTINUITY).exists():
                    text = (src / CONTINUITY).read_text(encoding="utf-8")
                    out, seg_dropped = project_continuity(text, p)
                    (stage / CONTINUITY).write_text(out, encoding="utf-8")
                if (src / META).exists():
                    meta = json.loads((src / META).read_text(encoding="utf-8"))
                    (stage / META).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
                crystals_dropped = 0
                if (src / CRYSTAL).exists():
                    doc = json.loads((src / CRYSTAL).read_text(encoding="utf-8"))
                    doc, crystals_dropped = project_crystals(doc, p)
                    (stage / CRYSTAL).write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
                (stage / SCOPE_FILE).write_text(json.dumps(
                    {"name": p.name, "policy_sha256": p.sha256, "continuity": CONTINUITY_MODE},
                    indent=2) + "\n", encoding="utf-8")
                _postcheck_files(stage, p)
                # unique (never the live one); the UTC build stamp is what an expiry job ages
                gen = f"gen-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{secrets.token_hex(4)}"
                os.rename(stage, root / gen)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
                if stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
            _swap_current(root, gen)
            _sweep(root, keep=gen)
            return {
                "scope": p.name,
                "policy_sha256": p.sha256,
                "generation": gen,
                "episodes_dropped": episodes,
                "crystals_dropped": crystals_dropped,
                "continuity_segments_dropped": seg_dropped,
                "continuity": CONTINUITY_MODE,
            }
        except ScopeError:
            _unpublish_locked(root)
            raise
        except Exception as e:  # any failure of the boundary build is a refusal
            _unpublish_locked(root)
            raise ScopeError(f"{type(e).__name__}: {e}") from e
    finally:
        os.close(fd)
