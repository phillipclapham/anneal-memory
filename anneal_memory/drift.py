"""CAP-06 drift probes: the operator's instrument for meaning drift in the continuity.

The operator declares what must survive consolidation (a Proven pattern, or a fact).
After every save anneal checks each probe LEXICALLY against the saved text and reports
held / weakened / crystallized / lost. It never judges truth, never blocks a save and
never rewrites the file: semantic truth is the operator's to hold, and this is the tool
they hold it with. Design: ``project_memory/cap06_drift_probe_design_1007.md`` (private
repo); the probes are deliberately not shown to the composer, so they measure drift
rather than obedience.
"""

from __future__ import annotations

from typing import Any, Iterable

from .graduation import _meaningful_words

PROBE_KINDS = ("pattern", "fact")
PROBE_STATUSES = ("held", "weakened", "crystallized", "lost")


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


def evaluate_probes(
    text: str,
    probes: Iterable[dict[str, Any]],
    *,
    pattern_levels: dict[str, int],
    live_crystals: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Check each probe against the saved continuity ``text``.

    ``pattern_levels`` maps each pattern named in the graduating section(s) to its
    level (the caller parses them as graduation does). ``live_crystals`` names the
    live crystallized patterns: a pattern that left the file for the crystal store
    moved, it was not lost. Returns one dict per probe:
    ``{probe_id, kind, subject, status, detail}``.
    """
    crystals = set(live_crystals)
    results: list[dict[str, Any]] = []
    for p in probes:
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
                status, detail = "crystallized", "not in the file; live in the crystal store"
            else:
                status, detail = "lost", "not in the file or the crystal store"
            subject = name
        elif kind == "fact":
            words = _meaningful_words(p["text"])
            best, best_line = -1, ""
            for line in _section_lines(text, p.get("section")):
                shared = len(words & _meaningful_words(line))
                if shared > best:
                    best, best_line = shared, line
            if words and best == len(words):
                status, detail = "held", best_line.strip()[:200]
            else:
                status = "lost"
                missing = len(words) - max(best, 0)
                detail = (f"{missing} of {len(words)} words missing; closest line: "
                          f"{best_line.strip()[:200]!r}")
            subject = p["text"]
        else:
            raise ValueError(f"unknown probe kind {kind!r}")
        results.append({"probe_id": p["id"], "kind": kind, "subject": subject,
                        "status": status, "detail": detail})
    return results
