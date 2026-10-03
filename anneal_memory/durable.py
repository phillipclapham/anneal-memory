"""anneal_memory.durable — durable facts: the one parser and the save invariant.

A schema section with the ``durable`` role (``## Durable Facts`` in the named
schemas) holds facts that must survive by mechanism, not by the composer's
memory. Measured on the InMind bench: a fact present in the continuity right
after injection was gone a few wraps later at well under the size budget, so the
loss was the composer's compression choice, not budget pressure.

One fact per ``- `` line. A line may end with composer-written cue words for
the situations where the fact matters::

    - tree nut allergy — cues: restaurant, dinner, recipe, food, menu

The cue marker is ``— cues:`` (em dash), ``-- cues:`` or ``| cues:``
(``cues:`` in any case); cues are comma-separated, trimmed and lowercased.

At save (:func:`enforce_durable_facts`) every line of the PRIOR continuity's
durable section must appear in the NEW text's durable section, compared by its
whitespace-normalised FACT part, so a changed cue list is an allowed update. A
missing line is re-inserted verbatim (cues included) and warned about after the
commit; it is never a refusal, because an omission must never make a store
unwritable. The only way to drop a line is a marker line in the new durable
section, ``[drop-durable: <exact line text>]``, naming the line's fact or its
full text; the save removes the marker and records the drop in the audit chain.

This module imports only :mod:`anneal_memory.schema`, so the save pipeline and
any reader of durable facts can share this one parser without an import cycle.
"""

from __future__ import annotations

import dataclasses
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
_CUES_RE = re.compile(r"[ \t]*(?:—|--|\|)[ \t]*cues:[ \t]*", re.IGNORECASE)
# A drop marker line, optionally written as a bullet. The target runs to the
# LAST ``]`` on the line, so a fact that itself contains brackets can be named.
DROP_DURABLE_RE = re.compile(
    r"^[ \t]*(?:[-*][ \t]+)?\[drop-durable:[ \t]*(.*?)[ \t]*\][ \t]*$"
)
# A reworded fact looks like this much of the old one (normalised-token Jaccard).
_NEAR_DUP_JACCARD = 0.6


@dataclasses.dataclass(frozen=True)
class DurableFact:
    """One ``- `` line of a durable section.

    Attributes:
        line: the raw line (trailing whitespace removed).
        fact: the text between the ``- `` bullet and the cue marker, stripped.
        cues: the lowercased, trimmed cue words; ``()`` with no cue marker.
    """

    line: str
    fact: str
    cues: tuple[str, ...]


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


def section_span(lines: list[str], schema: list[SectionSpec]) -> tuple[int, int] | None:
    """``(header_index, end_index)`` of the durable section in ``lines``: the
    first ``## `` header that matches the durable heading and no other schema
    heading, up to the next ``## `` header (or the end). ``None`` when the
    schema or the text has no durable section."""
    spec = durable_spec(schema)
    if spec is None:
        return None
    target = spec["heading"].lower()
    all_lower = {s["heading"].lower() for s in schema}
    start: int | None = None
    for i, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        if start is not None:
            return start, i
        if match_headings(line.lower(), all_lower) == [target]:
            start = i
    return None if start is None else (start, len(lines))


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def split_cues(text: str) -> tuple[str, tuple[str, ...]]:
    """Split a line body (no bullet) into ``(fact, cues)``."""
    m = _CUES_RE.search(text)
    if m is None:
        return text.strip(), ()
    cues = tuple(c.strip().lower() for c in text[m.end():].split(",") if c.strip())
    return text[: m.start()].strip(), cues


def parse_line(line: str) -> DurableFact | None:
    """A :class:`DurableFact` for a ``- `` line, ``None`` for any other line
    (blank, prose, a drop marker, a bullet with no fact)."""
    stripped = line.strip()
    if not stripped.startswith("- ") or DROP_DURABLE_RE.match(line):
        return None
    fact, cues = split_cues(stripped[2:])
    if not fact:
        return None
    return DurableFact(line=line.rstrip(), fact=fact, cues=cues)


def fact_key(fact: DurableFact) -> str:
    """The comparison key of a fact: its fact part, whitespace-normalised."""
    return _norm_ws(fact.fact)


def parse_durable_facts(
    continuity_text: str | None, schema: list[SectionSpec]
) -> list[DurableFact]:
    """Every fact line of the durable section of ``continuity_text``, in order.

    ``[]`` when the schema has no durable section, the text is empty, or the
    text has no such section.
    """
    if not continuity_text:
        return []
    lines = continuity_text.split("\n")
    span = section_span(lines, schema)
    if span is None:
        return []
    out: list[DurableFact] = []
    for line in lines[span[0] + 1:span[1]]:
        fact = parse_line(line)
        if fact is not None:
            out.append(fact)
    return out


def section_chars(text: str | None, schema: list[SectionSpec]) -> int:
    """Chars of the durable section (header included, as ``measure_sections``
    counts a section), 0 when the schema or the text has none."""
    if not text:
        return 0
    lines = text.split("\n")
    span = section_span(lines, schema)
    if span is None:
        return 0
    return sum(len(line) + 1 for line in lines[span[0]:span[1]])


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclasses.dataclass
class DurableReport:
    """What :func:`enforce_durable_facts` did to one save's text."""

    heading: str
    reinserted: list[str]
    dropped: list[str]
    unknown_drops: list[str]
    recreated: bool
    near_duplicates: list[tuple[str, str]]
    stray_markers: list[str]
    cue_sprawl: list[str]
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
    marker was removed or a line re-inserted, so a save that needs neither
    saves the composer's exact bytes.
    """
    spec = durable_spec(schema)
    if spec is None:
        return new_text, None
    heading = spec["heading"]
    all_lower = {s["heading"].lower() for s in schema}

    # Prior facts, in order, first occurrence of each fact key.
    prior: list[tuple[DurableFact, str]] = []
    seen: set[str] = set()
    for f in parse_durable_facts(prior_text, schema):
        k = fact_key(f)
        if k not in seen:
            seen.add(k)
            prior.append((f, k))

    lines = new_text.split("\n")
    span = section_span(lines, schema)

    # Drop markers count only inside the durable section; one anywhere else is
    # left in place and reported. A marker may name the fact part or the full
    # line (with or without its bullet), so it carries both keys.
    marker_keys: dict[str, str] = {}  # key -> the marker's target as written
    empty_markers: list[str] = []  # a marker naming nothing at all
    marker_idx: set[int] = set()
    stray: list[str] = []
    for i, line in enumerate(lines):
        m = DROP_DURABLE_RE.match(line)
        if m is None:
            continue
        if span is not None and span[0] < i < span[1]:
            marker_idx.add(i)
            target = m.group(1).strip()
            body = target[2:] if target.startswith("- ") else target
            keys = [k for k in (_norm_ws(body), _norm_ws(split_cues(body)[0])) if k]
            if not keys:
                empty_markers.append(target)
            for key in keys:
                marker_keys.setdefault(key, target)
        else:
            stray.append(line.strip())

    def _dropped_by_marker(f: DurableFact, k: str) -> bool:
        stripped = f.line.strip()
        return k in marker_keys or _norm_ws(stripped[2:]) in marker_keys

    dropped_keys = {k for f, k in prior if _dropped_by_marker(f, k)}
    dropped = [f.line for f, k in prior if k in dropped_keys]
    matched_targets = {
        marker_keys[key]
        for f, k in prior
        if k in dropped_keys
        for key in (k, _norm_ws(f.line.strip()[2:]))
        if key in marker_keys
    }
    unknown = list(dict.fromkeys(
        [t for t in marker_keys.values() if t not in matched_targets] + empty_markers
    ))

    # The new section without its markers, and without a line a marker dropped
    # (the marker is the explicit instruction, even if the line was also kept).
    body_lines: list[str] = []
    removed_any = bool(marker_idx)
    if span is not None:
        for i in range(span[0] + 1, span[1]):
            if i in marker_idx:
                continue
            parsed = parse_line(lines[i])
            if parsed is not None and fact_key(parsed) in dropped_keys:
                removed_any = True
                continue
            body_lines.append(lines[i])
    new_facts = [f for f in (parse_line(line) for line in body_lines) if f is not None]
    new_keys = {fact_key(f) for f in new_facts}
    missing = [f for f, k in prior if k not in new_keys and k not in dropped_keys]

    recreated = False
    if span is not None and (missing or removed_any):
        if missing:
            last = len(body_lines)
            while last > 0 and not body_lines[last - 1].strip():
                last -= 1
            body_lines[last:last] = [f.line for f in missing]
        lines = lines[: span[0] + 1] + body_lines + lines[span[1]:]
    elif span is None and missing:
        recreated = True
        # At the durable section's schema position: before the first section
        # that follows it in the schema, else at the end.
        order = [s["heading"].lower() for s in schema]
        later = set(order[order.index(heading.lower()) + 1:])
        insert_at: int | None = None
        for i, line in enumerate(lines):
            if line.startswith("## "):
                matched = match_headings(line.lower(), all_lower)
                if len(matched) == 1 and matched[0] in later:
                    insert_at = i
                    break
        block = [f"## {heading}", *(f.line for f in missing)]
        if insert_at is None:
            end = len(lines)
            while end > 0 and not lines[end - 1].strip():
                end -= 1
            lines[end:end] = ["", *block]
        else:
            lines[insert_at:insert_at] = [*block, ""]
    text = "\n".join(lines) if (missing or removed_any) else new_text

    near: list[tuple[str, str]] = []
    for f in missing:
        for nf in new_facts:
            if fact_key(nf) != fact_key(f) and _token_jaccard(f.fact, nf.fact) >= _NEAR_DUP_JACCARD:
                near.append((f.line, nf.line))

    sprawl = [f.line for f in parse_durable_facts(text, schema) if len(f.cues) > MAX_CUES]

    return text, DurableReport(
        heading=heading,
        reinserted=[f.line for f in missing],
        dropped=dropped,
        unknown_drops=unknown,
        recreated=recreated,
        near_duplicates=near,
        stray_markers=stray,
        cue_sprawl=sprawl,
        chars=section_chars(text, schema),
        budget=durable_budget(default_max_chars(schema)),
    )


def report_warnings(report: DurableReport) -> list[str]:
    """The post-commit warning texts for one save's :class:`DurableReport`."""
    h = report.heading
    out: list[str] = []
    if report.reinserted:
        where = (
            f"`## {h}` was missing from this wrap, so it was re-created with"
            if report.recreated
            else f"`## {h}` in this wrap left out"
        )
        out.append(
            f"Durable facts: {where} {len(report.reinserted)} line(s) of the prior "
            f"continuity, re-inserted verbatim: "
            + " | ".join(report.reinserted)
            + f". A durable line is removed only by a marker line in `## {h}`: "
            f"`[drop-durable: <exact line text>]`."
        )
    for target in report.unknown_drops:
        out.append(
            f"Durable facts: `[drop-durable: {target}]` names no line of the prior "
            f"`## {h}` section; the marker was removed and nothing else changed."
        )
    for old, new in report.near_duplicates:
        out.append(
            f"Durable facts: the re-inserted line {old!r} looks reworded as {new!r}: "
            f"if the new line replaces the old, drop the old with "
            f"`[drop-durable: {old}]`."
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
    if report.chars > report.budget:
        out.append(
            f"Durable facts: `## {h}` is {report.chars} chars, over its "
            f"{report.budget}-char budget. Every line was kept; drop facts that no "
            f"longer hold with `[drop-durable: <exact line text>]`."
        )
    return out
