"""Graduation and demotion logic for anneal-memory.

Handles temporal graduation validation:
- Citation checking (do graduated patterns cite real episodes?)
- Explanation overlap (does the evidence actually support the pattern?)
- Citation decay detection (which patterns haven't been reinforced?)
- Citation gaming detection (suspicious reuse of single episodes)
- Demotion (demote ungrounded or stale graduations)

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Matches graduated patterns (2x or 3x) WITH [evidence: <id> "explanation"] citations.
# Captures: (1) level, (2) date, (3) cited IDs, (4) optional explanation in quotes.
_GRADUATION_RE = re.compile(
    r"\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*\[evidence:\s*"
    r"([a-fA-F0-9][a-fA-F0-9, ]*)"  # one or more hex IDs
    r'(?:\s+"([^"]*)")?\s*\]'  # optional quoted explanation
)

# Matches bare graduations (2x or 3x) WITHOUT [evidence:] tags.
_BARE_GRADUATION_RE = re.compile(
    r"\|\s*([23])x\s*\((\d{4}-\d{2}-\d{2})\)\s*(?!\[evidence:)"
)

# Matches any pattern with temporal marker (Nx)
_PATTERN_RE = re.compile(
    r"\|\s*(\d+)x\s*\((\d{4}-\d{2}-\d{2})\)"
)

# Stop words for explanation overlap checking
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could this that these those "
    "it its he she they we you i me my our his her their in on at to "
    "for of by with from and or but not no nor so if as".split()
)


@dataclass
class GraduationResult:
    """Result of graduation validation."""

    text: str  # Possibly modified text (demoted patterns)
    validated: int  # Citations that checked out
    demoted: int  # Citations that failed -> demoted
    citation_reuse_max: int  # Max times any single node was cited
    bare_demoted: int  # Bare graduations demoted (no evidence tag)
    citation_counts: dict[str, int] = field(default_factory=dict)  # ep_id -> times cited
    gaming_suspects: list[str] = field(default_factory=list)  # IDs cited >= threshold


@dataclass
class StalenessInfo:
    """Information about a potentially stale pattern."""

    line_number: int
    content: str  # The pattern line
    level: int  # Current graduation level
    last_date: str  # Date of last validation
    days_stale: int  # Days since last validation


def validate_graduations(
    text: str,
    valid_ids: set[str],
    today: str,
    node_content_map: dict[str, str] | None = None,
    citations_seen: bool = False,
) -> GraduationResult:
    """Validate evidence citations on graduated patterns.

    Scans the ## Patterns section for 2x/3x lines with [evidence: <id> "explanation"].
    Only validates citations whose date matches today (newly graduated this session).
    Carried-forward patterns from previous sessions pass through unchanged.

    Validation checks (all must pass):
    1. At least one cited ID exists in the current session's episode set
    2. If an explanation is provided and node_content_map is available,
       the explanation must reference actual content from the cited episode

    If validation fails, demotes the graduation (3x->2x, 2x->1x) and marks (ungrounded).

    Args:
        text: The continuity file text (full markdown).
        valid_ids: Set of 8-char hex IDs from current session's episodes.
        today: Today's date as YYYY-MM-DD string.
        node_content_map: Optional mapping of episode ID -> content for overlap checking.
        citations_seen: If True, bare graduations (no evidence tag) are also demoted.

    Returns:
        GraduationResult with possibly modified text and validation counts.
    """
    lines = text.split("\n")
    in_patterns = False
    validated = 0
    demoted = 0
    bare_demoted = 0
    citation_counts: dict[str, int] = {}

    for i, line in enumerate(lines):
        # Track section boundaries
        if line.startswith("## "):
            in_patterns = "pattern" in line.lower()
            continue
        if not in_patterns:
            continue

        # Check for citations with evidence tags
        match = _GRADUATION_RE.search(line)
        if match:
            level = int(match.group(1))
            date_str = match.group(2)
            cited_raw = match.group(3)
            explanation = match.group(4)

            # Only validate today's citations
            if date_str != today:
                continue

            # Normalize cited IDs
            cited_ids = {
                cid.strip().lower()[:8]
                for cid in re.split(r"[,\s]+", cited_raw)
                if cid.strip()
            }

            # Track citation frequency
            for cid in cited_ids & valid_ids:
                citation_counts[cid] = citation_counts.get(cid, 0) + 1

            # Check 1: at least one cited ID exists
            ids_valid = bool(cited_ids & valid_ids)

            # Check 2: explanation references ANY cited episode's content
            explanation_valid = True
            if ids_valid and explanation and node_content_map:
                # Check all valid cited IDs — pass if ANY has content overlap
                explanation_valid = False
                for cid in cited_ids & valid_ids:
                    node_content = node_content_map.get(cid, "")
                    if node_content and check_explanation_overlap(explanation, node_content):
                        explanation_valid = True
                        break

            if ids_valid and explanation_valid:
                validated += 1
            else:
                demoted += 1
                lines[i] = _demote_line(line, match, level)
            continue

        # Fail-safe sunset: bare graduations without evidence
        if not citations_seen:
            continue

        bare_match = _BARE_GRADUATION_RE.search(line)
        if not bare_match:
            continue

        bare_level = int(bare_match.group(1))
        bare_date = bare_match.group(2)
        if bare_date != today:
            continue

        bare_demoted += 1
        old_marker = bare_match.group(0)
        new_marker = old_marker.replace(
            f"| {bare_level}x", f"| {bare_level - 1}x"
        )
        lines[i] = line.replace(old_marker, new_marker + " (needs-evidence)")

    reuse_max = max(citation_counts.values()) if citation_counts else 0
    gaming_suspects = detect_citation_gaming(citation_counts)

    return GraduationResult(
        text="\n".join(lines),
        validated=validated,
        demoted=demoted,
        citation_reuse_max=reuse_max,
        bare_demoted=bare_demoted,
        citation_counts=dict(citation_counts),
        gaming_suspects=gaming_suspects,
    )


def check_explanation_overlap(explanation: str, episode_content: str) -> bool:
    """Check if an explanation references actual content from the cited episode.

    Uses word overlap (excluding stop words). At least one meaningful word
    from the explanation must appear in the episode content. Prevents
    generic explanations like "confirms pattern" while allowing paraphrasing.

    Args:
        explanation: The quoted explanation from the evidence tag.
        episode_content: The full content of the cited episode.

    Returns:
        True if the explanation references the episode content.
    """
    def meaningful_words(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-zA-Z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP_WORDS
        }

    explanation_words = meaningful_words(explanation)
    episode_words = meaningful_words(episode_content)
    return bool(explanation_words & episode_words)


def detect_stale_patterns(text: str, today: str, staleness_days: int = 7) -> list[StalenessInfo]:
    """Find patterns that haven't been validated recently.

    Scans the ## Patterns section for patterns with dates older than
    staleness_days. These are candidates for removal or demotion.

    Args:
        text: The continuity file text.
        today: Today's date as YYYY-MM-DD string.
        staleness_days: Days without validation before flagging. Default 7.

    Returns:
        List of StalenessInfo for patterns exceeding the staleness threshold.
    """
    from datetime import datetime

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    stale: list[StalenessInfo] = []

    lines = text.split("\n")
    in_patterns = False

    for i, line in enumerate(lines):
        if line.startswith("## "):
            in_patterns = "pattern" in line.lower()
            continue
        if not in_patterns:
            continue

        match = _PATTERN_RE.search(line)
        if not match:
            continue

        level = int(match.group(1))
        date_str = match.group(2)

        try:
            pattern_dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        days_old = (today_dt - pattern_dt).days
        if days_old >= staleness_days:
            stale.append(StalenessInfo(
                line_number=i + 1,  # 1-indexed for human readability
                content=line.strip(),
                level=level,
                last_date=date_str,
                days_stale=days_old,
            ))

    return stale


def detect_citation_gaming(citation_counts: dict[str, int], threshold: int = 3) -> list[str]:
    """Detect episodes cited suspiciously many times.

    If a single episode is cited as evidence for many different patterns,
    it may indicate the LLM is gaming the citation system rather than
    finding genuine independent evidence.

    Args:
        citation_counts: Mapping of episode ID -> citation count.
        threshold: Number of citations before flagging. Default 3.

    Returns:
        List of episode IDs that exceed the threshold.
    """
    return [
        ep_id for ep_id, count in citation_counts.items()
        if count >= threshold
    ]


# -- Internal helpers --


def _demote_line(line: str, match: re.Match, level: int) -> str:
    """Demote a graduated pattern line (3x->2x or 2x->1x) and mark ungrounded."""
    old_marker = match.group(0)
    new_marker = old_marker.replace(f"| {level}x", f"| {level - 1}x")
    # Replace evidence tag with ungrounded marker
    new_marker = re.sub(
        r'\[evidence:\s*[a-fA-F0-9][a-fA-F0-9, ]*(?:\s+"[^"]*")?\s*\]',
        "(ungrounded)", new_marker
    )
    return line.replace(old_marker, new_marker)
