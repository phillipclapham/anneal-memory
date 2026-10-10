"""CAP-06 drift probes: the operator's instrument for meaning drift in the continuity.

The operator declares what must survive consolidation (a Proven pattern, or a fact).
After every save anneal checks each probe LEXICALLY against the saved text and reports
held / changed / weakened / crystallized / lost. It never judges truth, never blocks a
save and never rewrites the file: semantic truth is the operator's to hold, and this is
the tool they hold it with. Probes are not part of the wrap package, so they measure
drift rather than obedience (the save result does report the probes not held).

What the check can and cannot see (L2 review, 2026-10-07, run): a fact is matched
within one sentence or bullet, never across a whole paragraph line; numbers of any
length count; negators (not, no, nor, never, without, n't) are compared separately, so
a flipped negation reads ``changed``. A role swap that keeps every word ("freedom over
security" / "security over freedom") still reads ``held``: that residue is the operator's.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .graduation import _SYCOPHANCY_STOP_WORDS

PROBE_KINDS = ("pattern", "fact")
PROBE_STATUSES = ("held", "changed", "weakened", "crystallized", "lost", "unchecked")

NEGATORS = frozenset({"not", "no", "nor", "never", "without"})
# Words that order or place things carry the claim in a fact ("deploy before migrate"),
# so a probe keeps them (L3 1007, complement).
_ORDERING = frozenset({"before", "after", "above", "below", "over", "under", "up", "down",
                       "out", "off", "more", "most", "few", "only", "same", "other"})
_STOP = _SYCOPHANCY_STOP_WORDS - NEGATORS - _ORDERING
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?;])[*_`\"')\]]*\s+")
# A negator negates the next few content words, not the whole sentence: "and no seat
# offers to run it" must not flip "he runs it" (residue run, 2026-10-07).
_NEGATION_WINDOW = 2


def _forms(tok: str) -> frozenset[str]:
    """A token and its light stems (``-ing``, ``-ed``, ``-es``, ``-s``)."""
    out = {tok}
    for suffix, keep in (("ing", 4), ("ed", 4), ("es", 4), ("s", 3)):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= keep:
            out.add(tok[: -len(suffix)])
    return frozenset(out)


def _tokens(text: str) -> list[str]:
    text = re.sub(r"n't\b", " not", text.lower()).replace("cannot", "can not")
    return [t for t in re.split(r"[^a-z0-9]+", text)
            if t and (any(c.isdigit() for c in t) or (len(t) > 2 and t not in _STOP)
                      or t in NEGATORS or t in _ORDERING)]


def _units(lines: list[str]) -> list[str]:
    """Sentences and bullets: a new unit at a blank line, a bullet or a sentence end;
    wrapped lines of one sentence are joined."""
    blocks: list[list[str]] = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            blocks.append([])
            continue
        if _BULLET.match(line) or not blocks:
            blocks.append([])
        blocks[-1].append(line.strip())
    units: list[str] = []
    for b in blocks:
        if b:
            units.extend(u for u in _SENTENCE_END.split(" ".join(b)) if u.strip())
    return units


def _section_lines(text: str, heading: str | None) -> list[str]:
    """Lines of the ``## heading`` section (case-insensitive, header excluded), or of
    the whole text when ``heading`` is None."""
    lines = text.split("\n")
    if heading is None:
        return lines
    want = heading.strip().lstrip("#").strip().lower()
    out: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            inside = line[3:].strip().lower() == want
            continue
        if inside:
            out.append(line)
    return out


def _negated(toks: list[str], words: set[str]) -> bool:
    """Whether a negator in ``toks`` sits within ``_NEGATION_WINDOW`` tokens before a
    token whose forms meet ``words``."""
    for i, t in enumerate(toks):
        if t in NEGATORS and any(_forms(n) & words
                                 for n in toks[i + 1:i + 1 + _NEGATION_WINDOW]):
            return True
    return False


def _check_fact(text: str, fact: str, section: str | None) -> tuple[str, str]:
    toks = _tokens(fact)
    content = [t for t in toks if t not in NEGATORS]
    content_forms = set().union(*(_forms(t) for t in content)) if content else set()
    negated = _negated(toks, content_forms)
    best, best_unit, best_negated = -1, "", False
    for unit in _units(_section_lines(text, section)):
        utoks = _tokens(unit)
        uforms = set().union(*(_forms(t) for t in utoks)) if utoks else set()
        shared = sum(1 for t in content if _forms(t) & uforms)
        unit_negated = _negated(utoks, content_forms)
        # On a tie, the unit whose negation agrees wins (L3 1007, complement).
        if shared > best or (shared == best and best_negated != negated
                             and unit_negated == negated):
            best, best_unit, best_negated = shared, unit, unit_negated
    hint = best_unit.strip()[:200]
    if content and best == len(content):
        if negated != best_negated:
            return "changed", f"every word kept but the negation differs: {hint!r}"
        return "held", hint
    missing = len(content) - max(best, 0)
    return "lost", f"{missing} of {len(content)} words missing; closest: {hint!r}"


def evaluate_probes(
    text: str,
    probes: Iterable[dict[str, Any]],
    *,
    pattern_levels: dict[str, int],
    live_crystals: Mapping[str, Any] | Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Check each probe against the saved continuity ``text``.

    ``pattern_levels`` maps each pattern named in the graduating section(s) to its
    level (the caller parses them as graduation does). ``live_crystals`` maps each
    live crystallized pattern to its level (or names them): a pattern that left the
    file for the crystal store at its level moved, it was not lost; below its level it
    is ``weakened`` (L3 1007, codex). Returns one dict per probe:
    ``{probe_id, kind, subject, status, detail}``.
    """
    crystals: dict[str, Any] = (dict(live_crystals) if isinstance(live_crystals, Mapping)
                                else {n: None for n in live_crystals})
    results: list[dict[str, Any]] = []
    for p in probes:
        try:
            results.append(_evaluate_one(text, p, pattern_levels, crystals))
        except Exception as exc:  # noqa: BLE001 - an instrument must never become a gate
            # (L1 1007, run: an unknown kind from a newer version, or a NULL text, used
            # to refuse every save on the store.)
            results.append({"probe_id": p.get("id"), "kind": p.get("kind"),
                            "subject": p.get("name") or p.get("text"),
                            "status": "unchecked",
                            "detail": f"{type(exc).__name__}: {str(exc)[:160]}"})
    return results


def _evaluate_one(text: str, p: dict[str, Any], pattern_levels: dict[str, int],
                  crystals: dict[str, Any]) -> dict[str, Any]:
    kind = p["kind"]
    if kind == "pattern":
        name = p["name"]
        need = int(p.get("min_level") or 2)
        level = pattern_levels.get(name)
        if level is not None and level >= need:
            status, detail = "held", f"{level}x (needs {need}x)"
        elif level is not None:
            status, detail = "weakened", f"{level}x (needs {need}x)"
        elif name in crystals:
            clevel = crystals[name]
            if isinstance(clevel, int) and clevel < need:
                status, detail = "weakened", f"crystallized at {clevel}x (needs {need}x)"
            else:
                status, detail = "crystallized", "not in the file; live in the crystal store"
        else:
            status, detail = "lost", "not in the file or the crystal store"
        subject = name
    elif kind == "fact":
        status, detail = _check_fact(text, p["text"], p.get("section"))
        subject = p["text"]
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return {"probe_id": p["id"], "kind": kind, "subject": subject,
            "status": status, "detail": detail}


def fact_has_words(fact: str) -> bool:
    """Whether a fact has at least one word a probe can check."""
    return any(t not in NEGATORS for t in _tokens(fact))
