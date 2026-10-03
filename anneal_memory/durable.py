"""anneal_memory.durable — durable facts: the one parser and the save invariant.

A schema section with the ``durable`` role (``## Durable Facts`` in the named
schemas) holds facts that must survive by mechanism, not by the composer's
memory. Measured on the InMind bench: a fact present in the continuity right
after injection was gone a few wraps later at well under the size budget, so the
loss was the composer's compression choice, not budget pressure.

One fact per bullet line (``- ``, ``* `` or ``1. ``); an indented line right
under it continues it. A fact may end with composer-written cue words for the
situations where it matters::

    - tree nut allergy — cues: restaurant, dinner, recipe, food, menu

The cue marker is ``— cues:`` (em dash), ``– cues:`` (en dash), ``-- cues:`` or
``| cues:`` (``cues:`` in any case); cues are comma-separated, trimmed and
lowercased.

At save (:func:`enforce_durable_facts`) every fact of the PRIOR continuity's
durable section(s) must appear in the NEW text's, compared by its
whitespace-normalised FACT part, so a changed cue list is an allowed update. A
missing fact is re-inserted verbatim (cues and continuation lines included) and
warned about after the commit; it is never a refusal, because an omission must
never make a store unwritable. The only way to drop a fact is a marker line in
the new durable section, ``[drop-durable: <exact line text>]``, naming the
fact or its full first line; the save removes the marker and records the drop in
the audit chain.

The parser is fence-unaware, like ``validate_structure``: a ``## `` line
inside a code block is a header like any other. CR and CRLF line endings are read as LF; a rebuilt text takes the dominant
line ending of the text it was built from.

This module imports only :mod:`anneal_memory.schema`, so the save pipeline and
any reader of durable facts can share this one parser without an import cycle.
"""

from __future__ import annotations

import dataclasses
import difflib
import re

from .schema import SectionSpec, default_max_chars, durable_budget

__all__ = [
    "DurableFact",
    "parse_durable_facts",
    "MAX_CUES",
]

#: More cues than this on one line draws a save warning (cue sprawl).
MAX_CUES = 8

# The cue marker. The fact is everything before the FIRST marker.
_CUES_RE = re.compile(r"[ \t]*(?:—|–|--|\|)[ \t]*cues:[ \t]*", re.IGNORECASE)
# A fact line: a ``- `` / ``* `` bullet or a ``1. `` numbered item.
_BULLET_RE = re.compile(r"^[ \t]*(?:[-*]|\d+\.)[ \t]+(.*?)[ \t]*$")
# A drop marker line, optionally written as a bullet: always a MARKER, never a
# fact. The target runs to the LAST ``]`` on the line, so a fact that itself
# contains brackets can be named.
DROP_DURABLE_RE = re.compile(
    r"^[ \t]*(?:(?:[-*]|\d+\.)[ \t]+)?\[drop-durable:[ \t]*(.*?)[ \t]*\][ \t]*$",
    re.IGNORECASE,
)
_NEWLINES_RE = re.compile(r"\r\n|\r")
# A graduation line (``name | 2x (date)``) or an evidence tag: pattern syntax.
_PATTERN_SHAPE_RE = re.compile(r"\|[ \t]*\d+x\b|\[evidence:", re.IGNORECASE)
# A fact that describes a change that has not happened yet.
_PENDING_RE = re.compile(
    r"not happened|has not|hasn't|not yet|\buntil\b|only at|pending|switches to|will switch",
    re.IGNORECASE,
)
# A reworded fact looks like this much of the old one (normalised-token Jaccard).
_NEAR_DUP_JACCARD = 0.6
# Each class of pairwise warning names at most this many pairs, then one
# summary line; the scan itself stops after this many comparisons.
_PAIR_WARN_LIMIT = 20
_PAIR_SCAN_LIMIT = 200_000
# Function words of six letters or more, which would otherwise count as rare.
_STOPWORDS = frozenset(
    """
    about above across after again against almost already although always
    among another anyone anything around because become before behind being
    below beside besides between beyond cannot during either enough except
    further having however itself little mostly myself neither nobody nothing
    others otherwise really should something sometimes somewhere through
    toward towards unless upon whatever whenever where whether which while
    within without would yourself
    """.split()
)


@dataclasses.dataclass(frozen=True)
class DurableFact:
    """One fact of a durable section.

    Attributes:
        line: the raw first line, byte-for-byte (LF line endings).
        fact: the fact text, bullet and cue marker removed, continuation lines
            joined with single spaces.
        cues: the lowercased, trimmed cue words of a cue marker on the fact's
            LAST physical line; ``()`` with none.
        continuation: the raw indented lines that continue the fact,
            byte-for-byte.
    """

    line: str
    fact: str
    cues: tuple[str, ...]
    continuation: tuple[str, ...] = ()

    @property
    def raw_lines(self) -> list[str]:
        """Every raw line of the fact, first line first."""
        return [self.line, *self.continuation]

    @property
    def raw(self) -> str:
        """The whole fact as written: its raw lines joined with ``\\n``."""
        return "\n".join(self.raw_lines)


def match_headings(line_lower: str, headings: set[str]) -> list[str]:
    """The lowercased schema headings that one lowercased ``## `` header line
    satisfies, as a word-bounded phrase (``## State of Mind`` satisfies
    ``state``; ``## Interstate`` does not). More than one means the header is
    ambiguous."""
    return [
        h
        for h in headings
        if re.search(rf"(?<!\w){re.escape(h)}(?!\w)", line_lower)
    ]


def durable_spec(schema: list[SectionSpec]) -> SectionSpec | None:
    """The schema's ``durable`` section, or ``None`` (validate_schema allows at
    most one)."""
    for spec in schema:
        if spec["role"] == "durable":
            return spec
    return None


def _normalise(text: str) -> str:
    return _NEWLINES_RE.sub("\n", text)


def _dominant_newline(text: str) -> str:
    crlf = text.count("\r\n")
    cr = text.count("\r") - crlf
    lf = text.count("\n") - crlf
    # max() keeps the first of equals, so LF wins a tie.
    return max((lf, "\n"), (crlf, "\r\n"), (cr, "\r"), key=lambda p: p[0])[1]


def _headers(lines: list[str]) -> list[int]:
    return [i for i, line in enumerate(lines) if line.startswith("## ")]


def is_exact_heading(title: str, heading: str) -> bool:
    """The one rule for an optional heading: the header's stripped text equals
    the heading, case-insensitively (casefold). ``## durable facts`` is the
    durable heading; ``## Archived Durable Facts`` and
    ``## Decisions (durable facts)`` are not."""
    return title.strip().casefold() == heading.strip().casefold()


def _is_durable_header(line: str, heading: str) -> bool:
    return line.startswith("## ") and is_exact_heading(line[3:], heading)


def section_spans(lines: list[str], schema: list[SectionSpec]) -> list[tuple[int, int]]:
    """``(header_index, end_index)`` of EVERY durable section in ``lines`` (LF
    lines): each header that is exactly the durable heading, up to the next
    ``## `` header (or the end). ``[]`` when the schema or the text has none."""
    spec = durable_spec(schema)
    if spec is None:
        return []
    heading = spec["heading"]
    headers = _headers(lines)
    spans: list[tuple[int, int]] = []
    for k, i in enumerate(headers):
        if _is_durable_header(lines[i], heading):
            end = headers[k + 1] if k + 1 < len(headers) else len(lines)
            spans.append((i, end))
    return spans


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def split_cues(text: str) -> tuple[str, tuple[str, ...]]:
    """Split a fact body (no bullet) into ``(fact, cues)``."""
    m = _CUES_RE.search(text)
    if m is None:
        return text.strip(), ()
    cues = tuple(c.strip().lower() for c in text[m.end():].split(",") if c.strip())
    return text[: m.start()].strip(), cues


def _make_fact(first: str, body: str, continuation: list[str]) -> DurableFact | None:
    """A fact from its raw physical lines. A cue suffix is recognised only on
    the LAST physical line, so a continuation under a line that ends in cues
    is part of the fact (and of its identity), never swallowed as cues. The raw
    lines are kept byte-for-byte; only the comparison key is normalised."""
    parts = [body, *(c.strip() for c in continuation)]
    last_fact, cues = split_cues(parts[-1])
    fact = _norm_ws(" ".join([*parts[:-1], last_fact]))
    if not fact:
        return None
    return DurableFact(
        line=first,
        fact=fact,
        cues=cues,
        continuation=tuple(continuation),
    )


@dataclasses.dataclass
class _Item:
    """One element of a durable section body, in order."""

    kind: str  # "fact" | "marker" | "blank" | "other"
    lines: list[str]
    fact: DurableFact | None = None
    target: str = ""


def _items(body: list[str]) -> list[_Item]:
    """Classify a section body's lines: facts (with their continuation lines),
    drop markers, blanks, and anything else."""
    items: list[_Item] = []
    pending: tuple[str, str, list[str]] | None = None  # first line, body, continuation

    def _close() -> None:
        nonlocal pending
        if pending is None:
            return
        first, text, cont = pending
        fact = _make_fact(first, text, cont)
        if fact is None:
            items.append(_Item("other", [first, *cont]))
        else:
            items.append(_Item("fact", [first, *cont], fact=fact))
        pending = None

    for line in body:
        if not line.strip():
            _close()
            items.append(_Item("blank", [line]))
            continue
        m = DROP_DURABLE_RE.match(line)
        if m is not None:
            _close()
            items.append(_Item("marker", [line], target=m.group(1).strip()))
            continue
        b = _BULLET_RE.match(line)
        if b is not None:
            _close()
            if b.group(1):
                pending = (line, b.group(1), [])
            else:
                items.append(_Item("blank", [line]))  # an empty bullet holds nothing
            continue
        if pending is not None and line[:1] in (" ", "\t"):
            pending[2].append(line)
            continue
        _close()
        items.append(_Item("other", [line]))
    _close()
    return items


def _section_items(
    lines: list[str], spans: list[tuple[int, int]]
) -> list[_Item]:
    out: list[_Item] = []
    for start, end in spans:
        out += _items(lines[start + 1:end])
    return out


def parse_durable_facts(
    continuity_text: str | None, schema: list[SectionSpec]
) -> list[DurableFact]:
    """Every fact of every durable section of ``continuity_text``, in order.

    ``[]`` when the schema has no durable section, the text is empty, or the
    text has no such section.
    """
    if not continuity_text:
        return []
    lines = _normalise(continuity_text).split("\n")
    spans = section_spans(lines, schema)
    return [
        it.fact
        for it in _section_items(lines, spans)
        if it.kind == "fact" and it.fact is not None
    ]


def pending_transitions(
    continuity_text: str | None, schema: list[SectionSpec]
) -> list[str]:
    """First lines of the durable facts that describe a change that has not
    happened yet (``not yet``, ``until``, ``only at``, ``switches to`` ...)."""
    return [
        f.line
        for f in parse_durable_facts(continuity_text, schema)
        if _PENDING_RE.search(f.fact)
    ]


def section_chars(text: str | None, schema: list[SectionSpec]) -> int:
    """RAW chars of the durable section(s) of ``text``: header included, lines
    split on ``\n`` as ``measure_sections`` splits them (CRLF counts two). This
    is the same basis as ``len(text)``, so the shrink gate can subtract it.
    0 when the schema or the text has none."""
    if not text:
        return 0
    # Split like measure_sections: on "\n" only, so a lone "\r" is not a line
    # break and a CRLF line keeps its "\r" in its length.
    parts = text.split("\n")
    seps = ["\n"] * (len(parts) - 1) + [""]  # the last line has no ending
    return sum(
        len(parts[i]) + len(seps[i])
        for start, end in section_spans(parts, schema)
        for i in range(start, end)
    )


def fact_key(fact: DurableFact) -> str:
    """The comparison key of a fact: its fact part, whitespace-normalised."""
    return _norm_ws(fact.fact)


def _tokens(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", s.lower()))


def _rare(tokens: frozenset[str]) -> frozenset[str]:
    """Identifier-like or uncommon tokens: a digit, an underscore, or a
    non-stopword of six letters or more."""
    return frozenset(
        t for t in tokens
        if any(c.isdigit() for c in t) or "_" in t or (len(t) >= 6 and t not in _STOPWORDS)
    )


def _collapse_blanks(body: list[str]) -> list[str]:
    out: list[str] = []
    for line in body:
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return out


@dataclasses.dataclass
class DurableReport:
    """What :func:`enforce_durable_facts` did to one save's text."""

    heading: str
    reinserted: list[str]
    dropped: list[str]
    unknown_drops: list[tuple[str, str | None]]  # (target, closest prior line)
    unknown_drops_more: int
    multi_drops: list[tuple[str, list[str]]]  # (marker target, facts it dropped)
    own_lines_dropped: list[str]
    recreated: bool
    merged_sections: int
    untracked: list[str]
    near_duplicates: list[tuple[str, str]]
    near_duplicates_more: int
    contradictions: list[tuple[str, str]]
    contradictions_more: int
    pair_scan_stopped: bool
    stray_markers: list[str]
    cue_sprawl: list[str]
    pattern_shaped: list[str]
    shared_facts: list[tuple[str, list[str]]]
    chars: int
    budget: int


def enforce_durable_facts(
    prior_text: str | None,
    new_text: str,
    schema: list[SectionSpec],
) -> tuple[str, DurableReport | None]:
    """Apply the durable-facts invariant to ``new_text``.

    Returns the text to save and a report, or ``(new_text, None)`` unchanged
    when the schema has no durable section. The text is rebuilt only when a
    marker was removed, a fact re-inserted or several durable sections merged,
    so a save that needs none of these saves the composer's exact bytes.
    """
    spec = durable_spec(schema)
    if spec is None:
        return new_text, None
    heading = spec["heading"]
    all_lower = {s["heading"].lower() for s in schema}

    prior_facts = parse_durable_facts(prior_text, schema)
    prior: list[tuple[DurableFact, str]] = []
    seen: set[str] = set()
    for f in prior_facts:
        k = fact_key(f)
        if k not in seen:
            seen.add(k)
            prior.append((f, k))

    newline = _dominant_newline(new_text)
    lines = _normalise(new_text).split("\n")
    spans = section_spans(lines, schema)
    items = _section_items(lines, spans)

    in_section = {i for s, e in spans for i in range(s + 1, e)}
    stray = [
        line.strip()
        for i, line in enumerate(lines)
        if i not in in_section and DROP_DURABLE_RE.match(line)
    ]

    # A marker names the fact part or the full first line (with or without its
    # bullet), so it carries both keys.
    marker_keys: dict[str, str] = {}
    empty_markers: list[str] = []
    for it in items:
        if it.kind != "marker":
            continue
        target = it.target
        tb = _BULLET_RE.match(target)
        named = tb.group(1) if tb is not None else target
        keys = [k for k in (_norm_ws(named), _norm_ws(split_cues(named)[0])) if k]
        if not keys:
            empty_markers.append(target)
        for key in keys:
            marker_keys.setdefault(key, target)

    def _first_body(f: DurableFact) -> str:
        b = _BULLET_RE.match(f.line)
        return _norm_ws(b.group(1)) if b is not None else ""

    dropped_keys = {
        k for f, k in prior if k in marker_keys or _first_body(f) in marker_keys
    }
    dropped = [f.raw for f, k in prior if k in dropped_keys]
    # Which prior facts each marker dropped: a marker naming a first line can
    # match several facts that share it (their continuations differ). Each
    # dropped fact is attributed to the FIRST marker (in text order) that names
    # it; every marker that names some prior fact counts as matched.
    marker_rank: dict[str, int] = {}
    for it in items:
        if it.kind == "marker":
            marker_rank.setdefault(it.target, len(marker_rank))
    matches_by_target: dict[str, list[str]] = {}
    matched_targets: set[str] = set()
    for f, k in prior:
        if k not in dropped_keys:
            continue
        naming = {marker_keys[key] for key in (k, _first_body(f)) if key in marker_keys}
        matched_targets |= naming
        first = min(naming, key=lambda t: marker_rank.get(t, len(marker_rank)))
        matches_by_target.setdefault(first, []).append(f.raw)
    unknown_targets = list(dict.fromkeys(
        [t for t in marker_keys.values() if t not in matched_targets] + empty_markers
    ))
    # The closest prior line to a marker that matched nothing, compared on the
    # fact part and on the whole first line (stdlib difflib, no model).
    candidates: dict[str, str] = {}
    for f, k in prior:
        candidates.setdefault(k, f.line)
        candidates.setdefault(_first_body(f), f.line)
    # Bounded: a closest-line hint for at most _PAIR_WARN_LIMIT markers (each
    # costs a pass over the prior lines); the rest go into one summary line.
    unknown: list[tuple[str, str | None]] = []
    unknown_more = max(len(unknown_targets) - _PAIR_WARN_LIMIT, 0)
    for t in unknown_targets[:_PAIR_WARN_LIMIT]:
        tb2 = _BULLET_RE.match(t)
        probe = _norm_ws(split_cues(tb2.group(1) if tb2 is not None else t)[0])
        close = difflib.get_close_matches(probe, list(candidates), n=1, cutoff=0.5)
        unknown.append((t, candidates[close[0]] if close else None))

    # The new section(s) without markers, and without a fact a marker dropped
    # (the marker is the explicit instruction, even if the line was also kept).
    body: list[str] = []
    new_facts: list[DurableFact] = []
    own_dropped: list[str] = []
    untracked: list[str] = []
    for it in items:
        if it.kind == "marker":
            continue
        if it.kind == "fact" and it.fact is not None:
            if fact_key(it.fact) in dropped_keys:
                own_dropped.append(it.fact.line)
                continue
            new_facts.append(it.fact)
        elif it.kind == "other":
            untracked.append(it.lines[0].strip())
        body += it.lines
    new_keys = {fact_key(f) for f in new_facts}
    missing = [f for f, k in prior if k not in new_keys and k not in dropped_keys]

    rebuild = bool(missing) or any(it.kind == "marker" for it in items) or bool(
        own_dropped
    ) or len(spans) > 1
    recreated = False
    if spans and rebuild:
        if missing:
            last = len(body)
            while last > 0 and not body[last - 1].strip():
                last -= 1
            body[last:last] = [ln for f in missing for ln in f.raw_lines]
        body = _collapse_blanks(body)
        out: list[str] = []
        idx = 0
        for n, (start, end) in enumerate(spans):
            out += lines[idx:start]
            if n == 0:
                out += [lines[start], *body]
            idx = end
        out += lines[idx:]
        lines = out
    elif not spans and missing:
        recreated = True
        # At the durable section's schema position: before the first section
        # that follows it in the schema, else at the end.
        order = [s["heading"].lower() for s in schema]
        later = set(order[order.index(heading.lower()) + 1:])
        insert_at: int | None = None
        for i in _headers(lines):
            matched = match_headings(lines[i].lower(), all_lower)
            if len(matched) == 1 and matched[0] in later:
                insert_at = i
                break
        block = [f"## {heading}", *(ln for f in missing for ln in f.raw_lines)]
        if insert_at is None:
            end = len(lines)
            while end > 0 and not lines[end - 1].strip():
                end -= 1
            lines[end:end] = ["", *block]
        else:
            lines[insert_at:insert_at] = [*block, ""]
        rebuild = True
    text = newline.join(lines) if rebuild else new_text

    # Pairwise checks of each re-inserted fact against the new ones: a reword
    # (near duplicate) or a fact that may replace it (shared cues or shared
    # identifier-like tokens). Token sets are computed once, and the scan is
    # bounded so a very large section cannot stall the save.
    near: list[tuple[str, str]] = []
    contra: list[tuple[str, str]] = []
    near_n = contra_n = 0
    scanned = 0
    stopped = False
    new_tok = [(nf, _tokens(nf.fact)) for nf in new_facts]
    new_info = [(nf, t, _rare(t), set(nf.cues)) for nf, t in new_tok]
    for f in missing:
        if stopped:
            break
        ft = _tokens(f.fact)
        fr = _rare(ft)
        fc = set(f.cues)
        fk = fact_key(f)
        for nf, nt, nr, nc in new_info:
            scanned += 1
            if scanned > _PAIR_SCAN_LIMIT:
                stopped = True
                break
            if fact_key(nf) == fk or not ft or not nt:
                continue
            if len(ft & nt) / len(ft | nt) >= _NEAR_DUP_JACCARD:
                near_n += 1
                if len(near) < _PAIR_WARN_LIMIT:
                    near.append((f.line, nf.line))
            elif len(fc & nc) >= 2 or len(fr & nr) >= 2:
                contra_n += 1
                if len(contra) < _PAIR_WARN_LIMIT:
                    contra.append((f.line, nf.line))

    # Two lines sharing a fact, within the prior section or within the saved
    # one (a prior line and its own update are not a pair). Only the first is
    # protected, since the save compares by fact.
    final = parse_durable_facts(text, schema)
    shared: list[tuple[str, list[str]]] = []
    for group in (prior_facts, final):
        by_key: dict[str, list[str]] = {}
        for f in group:
            lines_for = by_key.setdefault(fact_key(f), [])
            if f.line not in lines_for:
                lines_for.append(f.line)
        for key, ls in by_key.items():
            if len(ls) > 1 and (key, ls) not in shared:
                shared.append((key, ls))

    return text, DurableReport(
        heading=heading,
        reinserted=[f.raw for f in missing],
        dropped=dropped,
        unknown_drops=unknown,
        unknown_drops_more=unknown_more,
        multi_drops=[(t, r) for t, r in matches_by_target.items() if len(r) > 1],
        own_lines_dropped=own_dropped,
        recreated=recreated,
        merged_sections=len(spans) if len(spans) > 1 else 0,
        untracked=untracked,
        near_duplicates=near,
        near_duplicates_more=near_n - len(near),
        contradictions=contra,
        contradictions_more=contra_n - len(contra),
        pair_scan_stopped=stopped,
        stray_markers=stray,
        cue_sprawl=[f.line for f in final if len(f.cues) > MAX_CUES],
        pattern_shaped=[f.line for f in final if _PATTERN_SHAPE_RE.search(f.line)],
        shared_facts=shared,
        chars=section_chars(text, schema),
        budget=durable_budget(default_max_chars(schema)),
    )


def _one_line(raw: str) -> str:
    """A fact's raw lines on one line, for a warning: joined with `` / ``."""
    return " / ".join(part.strip() for part in raw.split("\n"))


def report_warnings(report: DurableReport) -> list[str]:
    """The post-commit warning texts for one save's :class:`DurableReport`.
    A multi-line fact is shown on one line, its lines joined with `` / ``."""
    h = report.heading
    marker = "`[drop-durable: <exact line text>]`"
    out: list[str] = []
    if report.merged_sections:
        out.append(
            f"Durable facts: this wrap had {report.merged_sections} `## {h}` sections; "
            f"their lines were merged into the first one."
        )
    if report.reinserted:
        where = (
            f"`## {h}` was missing from this wrap, so it was re-created with"
            if report.recreated
            else f"`## {h}` in this wrap left out"
        )
        out.append(
            f"Durable facts: {where} {len(report.reinserted)} line(s) of the prior "
            f"continuity, re-inserted verbatim: "
            + " | ".join(_one_line(r) for r in report.reinserted)
            + f". If a fact changed, drop the old line with the marker, {marker} "
            f"in `## {h}`; that marker is the only way a durable line is removed."
        )
    for target, closest in report.unknown_drops:
        hint = (
            f" The closest prior line is {closest!r}."
            if closest is not None
            else " No prior line is close to it."
        )
        out.append(
            f"Durable facts: `[drop-durable: {target}]` names no line of the prior "
            f"`## {h}` section; the marker was removed and nothing else changed. "
            f"Matching is exact (whitespace aside), on the fact or the whole line."
            + hint
        )
    if report.unknown_drops_more:
        out.append(
            f"Durable facts: and {report.unknown_drops_more} more drop marker(s) "
            f"named no line of the prior `## {h}` section; they were removed and "
            f"nothing else changed."
        )
    for raw in report.dropped:
        out.append(f"Durable facts: dropped by marker: {_one_line(raw)}")
    for target, raws in report.multi_drops:
        out.append(
            f"Durable facts: `[drop-durable: {target}]` matched {len(raws)} prior "
            f"facts and dropped them all: "
            + " | ".join(_one_line(r) for r in raws)
            + ". To drop only one, name its whole fact."
        )
    for line in report.own_lines_dropped:
        out.append(
            f"Durable facts: a drop marker also removed {line!r}, which this wrap "
            f"wrote itself. If the fact should stay, write it again next wrap."
        )
    for old, new in report.near_duplicates:
        out.append(
            f"Durable facts: the re-inserted line {old!r} looks reworded as {new!r}: "
            f"if the new line replaces the old, drop the old with "
            f"`[drop-durable: {old}]`."
        )
    if report.near_duplicates_more:
        out.append(
            f"Durable facts: and {report.near_duplicates_more} more re-inserted "
            f"line(s) that look reworded."
        )
    for old, new in report.contradictions:
        out.append(
            f"Durable facts: re-inserted {old!r} may be superseded by {new!r}; if the "
            f"fact changed, drop the old line with [drop-durable: {old}]."
        )
    if report.contradictions_more:
        out.append(
            f"Durable facts: and {report.contradictions_more} more re-inserted "
            f"line(s) that may be superseded by a new one."
        )
    if report.pair_scan_stopped:
        out.append(
            "Durable facts: the reword / supersede check stopped early on a very "
            "large section; some pairs were not compared."
        )
    for line in report.untracked:
        out.append(
            f"Durable facts: {line!r} in `## {h}` is not tracked as a durable fact; "
            f"write it as a `- ` line."
        )
    if report.stray_markers:
        out.append(
            f"Durable facts: drop marker(s) outside `## {h}` were ignored and left "
            f"in the text: " + " | ".join(report.stray_markers)
        )
    for line in report.cue_sprawl:
        out.append(
            f"Durable facts: more than {MAX_CUES} cues on {line!r}. Keep the "
            f"situations where the fact should come to mind, not synonyms of it."
        )
    for line in report.pattern_shaped:
        out.append(
            f"Durable facts: {line!r} has the shape of a pattern line; pattern lines "
            f"belong in ## Patterns, where citations are validated."
        )
    for fact, lines in report.shared_facts:
        out.append(
            f"Durable facts: two lines share the fact {fact!r}: "
            + " | ".join(lines)
            + ". The save compares lines by fact, so only one of them is protected; "
            "keep one line per fact."
        )
    if report.chars > report.budget:
        out.append(
            f"Durable facts: `## {h}` is {report.chars} chars, over its "
            f"{report.budget}-char budget. Every line was kept; drop facts that no "
            f"longer hold with {marker}."
        )
    return out
