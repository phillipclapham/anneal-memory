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
from datetime import datetime as _datetime
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
    # Graduation-format lines whose date != today (carried-forward
    # patterns from prior sessions, OR a test-authoring bug where a
    # hardcoded date drifted from wall-clock). Callers that intend
    # to exercise validation (tests, deterministic experiments) can
    # assert this is zero as a structural invariant — catches the
    # Finding #3 class of test drift without any log/warning noise
    # in production where non-today skips are the normal case.
    skipped_non_today: int = 0
    citation_counts: dict[str, int] = field(default_factory=dict)  # ep_id -> times cited
    gaming_suspects: list[str] = field(default_factory=list)  # IDs cited >= threshold
    direct_co_citations: list[tuple[str, str]] = field(default_factory=list)  # Pairs from same line
    all_validated_ids: list[set[str]] = field(default_factory=list)  # Per-line validated ID sets


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
    skipped_non_today = 0
    citation_counts: dict[str, int] = {}
    direct_co_citations: list[tuple[str, str]] = []
    all_validated_ids: list[set[str]] = []

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

            # Only validate today's citations. Non-today graduations
            # are either legitimately carried-forward from a prior
            # session (normal) or a test-authoring bug where a
            # hardcoded date drifted from wall-clock (Diogenes
            # Finding #3 class — recurred 3x before the counter
            # was added). Increment ``skipped_non_today`` so callers
            # that care can assert on it; production ignores it.
            if date_str != today:
                skipped_non_today += 1
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
                # Extract co-citation pairs for Hebbian associations
                # Only from VALIDATED citations — immune system protects the graph
                valid_cited = sorted(cited_ids & valid_ids)
                if len(valid_cited) >= 2:
                    for idx_a in range(len(valid_cited)):
                        for idx_b in range(idx_a + 1, len(valid_cited)):
                            direct_co_citations.append(
                                (valid_cited[idx_a], valid_cited[idx_b])
                            )
                # Track all validated IDs per line for session co-citation
                if valid_cited:
                    all_validated_ids.append(set(valid_cited))
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
            skipped_non_today += 1
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
        skipped_non_today=skipped_non_today,
        citation_counts=dict(citation_counts),
        gaming_suspects=gaming_suspects,
        direct_co_citations=direct_co_citations,
        all_validated_ids=all_validated_ids,
    )


def check_explanation_overlap(explanation: str, episode_content: str) -> bool:
    """Check if an explanation references actual content from the cited episode.

    Uses word overlap (excluding stop words). At least 2 meaningful words
    from the explanation must appear in the episode content. Single-word
    overlap is too easy to game (e.g., "database" matching any DB-related
    episode). Two words provide meaningful grounding while still allowing
    paraphrasing.

    Args:
        explanation: The quoted explanation from the evidence tag.
        episode_content: The full content of the cited episode.

    Returns:
        True if the explanation sufficiently references the episode content.
    """
    def meaningful_words(text: str) -> set[str]:
        return {
            w for w in re.split(r"[^a-zA-Z0-9]+", text.lower())
            if len(w) > 2 and w not in _STOP_WORDS
        }

    explanation_words = meaningful_words(explanation)
    episode_words = meaningful_words(episode_content)
    return len(explanation_words & episode_words) >= 2


def extract_session_co_citations(
    all_validated_ids: list[set[str]],
) -> set[tuple[str, str]]:
    """Extract session-level co-citation pairs from validated graduation IDs.

    Session co-citations are pairs of episodes cited in DIFFERENT pattern
    lines during the same wrap. These represent a weaker association signal
    than direct co-citations (same line).

    Args:
        all_validated_ids: List of sets, each containing validated episode IDs
            from a single pattern line.

    Returns:
        Set of canonical (id_a, id_b) pairs where id_a < id_b.
    """
    # Pre-build index: episode_id -> set of line indices it appears in
    # O(n*k) where n = number of lines, k = avg IDs per line
    id_to_lines: dict[str, set[int]] = {}
    all_ids: set[str] = set()
    for idx, id_set in enumerate(all_validated_ids):
        for ep_id in id_set:
            all_ids.add(ep_id)
            if ep_id not in id_to_lines:
                id_to_lines[ep_id] = set()
            id_to_lines[ep_id].add(idx)

    # Generate pairs between IDs that appear in DIFFERENT lines
    # O(n^2) over unique IDs with O(1) line-set lookups
    pairs: set[tuple[str, str]] = set()
    id_list = sorted(all_ids)
    for i in range(len(id_list)):
        a_lines = id_to_lines[id_list[i]]
        for j in range(i + 1, len(id_list)):
            b_lines = id_to_lines[id_list[j]]
            if a_lines != b_lines or len(a_lines) > 1:
                # They appear in at least one different line context
                pairs.add((id_list[i], id_list[j]))

    return pairs


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
    today_dt = _datetime.strptime(today, "%Y-%m-%d")
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
            pattern_dt = _datetime.strptime(date_str, "%Y-%m-%d")
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
    """Demote a graduated pattern line (3x->2x or 2x->1x) and mark ungrounded.

    Uses positional replacement via match span to avoid fragility
    from str.replace on LLM-generated text that might contain
    duplicate marker-like substrings.
    """
    old_marker = match.group(0)
    new_marker = old_marker.replace(f"| {level}x", f"| {level - 1}x")
    # Replace evidence tag with ungrounded marker
    new_marker = re.sub(
        r'\[evidence:\s*[a-fA-F0-9][a-fA-F0-9, ]*(?:\s+"[^"]*")?\s*\]',
        "(ungrounded)", new_marker
    )
    # Positional replacement — immune to duplicate marker text elsewhere in line
    start, end = match.span()
    return line[:start] + new_marker + line[end:]
