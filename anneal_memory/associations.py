"""Hebbian association logic for anneal-memory.

The third cognitive layer: lateral links between episodes that strengthen
via co-citation during graduation. Deep Hebbian — associations form from
semantic judgment during consolidation (the agent cites episodes together
because they're meaningfully related), not from temporal proximity.

Two tiers of co-citation:
- Direct: episodes cited on the same pattern line (strength += 1.0)
- Session: episodes cited in the same wrap but different patterns (strength += 0.3)

Decay: associations weaken each wrap where not reinforced (strength *= decay_factor).
Links below the cleanup threshold are pruned. The cognitive structure evolves —
old connections fade, new ones form, strong connections persist.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .types import AffectiveState, AssociationPair, AssociationStats

# Import GraduationResult lazily to avoid circular imports
# (graduation.py doesn't import associations.py, but the orchestration
# function below receives GraduationResult as input)

# Strength contributions
DIRECT_CO_CITATION_STRENGTH = 1.0
SESSION_CO_CITATION_STRENGTH = 0.3

# Affective modulation: intensity scales strength by up to this factor.
# At 0.5: no affect = 1.0x, low (0.2) = 1.1x, high (0.8) = 1.4x, max = 1.5x.
# Semantic signal stays primary; affect modulates, doesn't dominate.
AFFECTIVE_MODULATION_FACTOR = 0.5

# Maximum strength cap — prevents calcification where long-lived pairs
# become "immortal" and resist decay. Without a cap, a link co-cited 20x
# (strength 20.0) would need ~50 wraps of non-reinforcement to decay below
# threshold. With cap at 10.0, max decay-to-death is ~44 wraps — still long
# for genuinely strong connections, but bounded. The cap also prevents the
# "strongest pairs" display from being dominated by historical accumulation.
MAX_STRENGTH = 10.0

# Default decay: 0.9 per wrap where not reinforced
DEFAULT_DECAY_FACTOR = 0.9

# Links below this threshold are deleted (dead weight)
DEFAULT_CLEANUP_THRESHOLD = 0.1

# Schema for the associations table.
# New DBs get all columns from CREATE TABLE. Existing DBs (0.1.7 without
# affective columns) get them via _migrate_add_affective_columns().
ASSOCIATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS associations (
    episode_a TEXT NOT NULL,
    episode_b TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0,
    first_linked TEXT NOT NULL,
    last_strengthened TEXT NOT NULL,
    co_citations INTEGER NOT NULL DEFAULT 1,
    affective_tag TEXT,
    affective_intensity REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (episode_a, episode_b),
    CHECK (episode_a < episode_b),
    FOREIGN KEY (episode_a) REFERENCES episodes(id) ON DELETE CASCADE,
    FOREIGN KEY (episode_b) REFERENCES episodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assoc_a ON associations(episode_a);
CREATE INDEX IF NOT EXISTS idx_assoc_b ON associations(episode_b);
CREATE INDEX IF NOT EXISTS idx_assoc_strength ON associations(strength DESC);
"""


def migrate_add_affective_columns(conn: sqlite3.Connection) -> None:
    """Add affective columns to existing associations tables.

    Safe to call on tables that already have the columns (checks first).
    Handles migration for databases where the associations table exists
    but lacks the affective_tag and affective_intensity columns
    (added in Session 9.5, limbic layer).
    """
    # Check if columns already exist
    cursor = conn.execute("PRAGMA table_info(associations)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if "affective_tag" not in existing_cols:
        conn.execute("ALTER TABLE associations ADD COLUMN affective_tag TEXT")
    if "affective_intensity" not in existing_cols:
        conn.execute(
            "ALTER TABLE associations ADD COLUMN affective_intensity REAL NOT NULL DEFAULT 0.0"
        )
    conn.commit()


def canonical_pair(id_a: str, id_b: str) -> tuple[str, str] | None:
    """Return a canonically ordered pair (smaller ID first), or None for self-pairs.

    This ensures each pair is stored exactly once regardless of
    which order the IDs were encountered. Self-pairs (same ID twice)
    return None — an episode cannot be associated with itself.
    """
    if id_a == id_b:
        return None
    if id_a < id_b:
        return (id_a, id_b)
    return (id_b, id_a)


def record_associations(
    conn: sqlite3.Connection,
    direct_pairs: set[tuple[str, str]],
    session_pairs: set[tuple[str, str]],
    timestamp: str,
    affective_state: AffectiveState | None = None,
) -> tuple[int, int]:
    """Record or strengthen association links from co-citation.

    Args:
        conn: SQLite connection with associations table.
        direct_pairs: Pairs of episode IDs co-cited on the same pattern line.
        session_pairs: Pairs cited in the same wrap but different patterns.
        timestamp: ISO 8601 UTC timestamp for this wrap.
        affective_state: Optional agent functional state during this consolidation.
            When provided, intensity modulates association strength (up to 1.5x
            at max intensity) and the tag is stored on the association record.

    Returns:
        Tuple of (links_formed, links_strengthened).
    """
    formed = 0
    strengthened = 0

    # Calculate affective modulation
    affect_multiplier = 1.0
    if affective_state is not None:
        clamped = max(0.0, min(1.0, affective_state.intensity))
        affect_multiplier = 1.0 + clamped * AFFECTIVE_MODULATION_FACTOR

    # Canonicalize all pairs upfront for consistent set operations.
    # Filter out self-pairs (canonical_pair returns None for id_a == id_b).
    canon_direct: set[tuple[str, str]] = set()
    for pair in direct_pairs:
        cp = canonical_pair(*pair)
        if cp is not None:
            canon_direct.add(cp)

    canon_session: set[tuple[str, str]] = set()
    for pair in session_pairs:
        cp = canonical_pair(*pair)
        if cp is not None:
            canon_session.add(cp)

    # Process direct co-citations (stronger signal)
    for a, b in canon_direct:
        was_new = _upsert_association(
            conn, a, b,
            DIRECT_CO_CITATION_STRENGTH * affect_multiplier,
            timestamp, affective_state,
        )
        if was_new:
            formed += 1
        else:
            strengthened += 1

    # Process session co-citations (weaker signal)
    # Exclude pairs already processed as direct co-citations.
    # Both sets are now canonical, so set subtraction is safe.
    for a, b in canon_session - canon_direct:
        was_new = _upsert_association(
            conn, a, b,
            SESSION_CO_CITATION_STRENGTH * affect_multiplier,
            timestamp, affective_state,
        )
        if was_new:
            formed += 1
        else:
            strengthened += 1

    conn.commit()
    return formed, strengthened


def decay_associations(
    conn: sqlite3.Connection,
    strengthened_pairs: set[tuple[str, str]],
    decay_factor: float = DEFAULT_DECAY_FACTOR,
    cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD,
) -> int:
    """Decay associations not reinforced this wrap.

    Associations that were strengthened during this wrap are skipped.
    All others have their strength multiplied by decay_factor.
    Links that fall below cleanup_threshold are deleted.

    Args:
        conn: SQLite connection with associations table.
        strengthened_pairs: Canonical pairs strengthened this wrap (skip these).
        decay_factor: Multiplier for unreinforced links. Default 0.9.
        cleanup_threshold: Delete links weaker than this. Default 0.1.

    Returns:
        Number of associations that decayed (including deleted ones).
    """
    # Get all associations
    rows = conn.execute(
        "SELECT episode_a, episode_b, strength FROM associations"
    ).fetchall()

    decayed = 0
    to_delete: list[tuple[str, str]] = []

    for row in rows:
        pair = (row[0], row[1])  # Already canonical in DB
        if pair in strengthened_pairs:
            continue  # Reinforced this wrap — no decay

        new_strength = row[2] * decay_factor
        if new_strength < cleanup_threshold:
            to_delete.append(pair)
            decayed += 1
        else:
            conn.execute(
                "UPDATE associations SET strength = ? WHERE episode_a = ? AND episode_b = ?",
                (new_strength, pair[0], pair[1]),
            )
            decayed += 1

    # Delete dead links
    for a, b in to_delete:
        conn.execute(
            "DELETE FROM associations WHERE episode_a = ? AND episode_b = ?",
            (a, b),
        )

    conn.commit()
    return decayed


def get_associations(
    conn: sqlite3.Connection,
    episode_ids: list[str],
    min_strength: float = 0.0,
    limit: int = 50,
) -> list[AssociationPair]:
    """Get associations for the given episodes.

    Returns associations where at least one side is in episode_ids,
    ordered by strength descending.

    Args:
        conn: SQLite connection with associations table.
        episode_ids: Episode IDs to query associations for.
        min_strength: Minimum strength to include. Default 0.0 (all).
        limit: Maximum associations to return.

    Returns:
        List of AssociationPair ordered by strength descending.
    """
    if not episode_ids:
        return []

    placeholders = ",".join("?" for _ in episode_ids)
    rows = conn.execute(
        f"""SELECT episode_a, episode_b, strength, co_citations,
                   first_linked, last_strengthened,
                   affective_tag, affective_intensity
            FROM associations
            WHERE (episode_a IN ({placeholders}) OR episode_b IN ({placeholders}))
              AND strength >= ?
            ORDER BY strength DESC
            LIMIT ?""",
        [*episode_ids, *episode_ids, min_strength, limit],
    ).fetchall()

    return [
        AssociationPair(
            episode_a=row[0],
            episode_b=row[1],
            strength=row[2],
            co_citations=row[3],
            first_linked=row[4],
            last_strengthened=row[5],
            affective_tag=row[6],
            affective_intensity=row[7] or 0.0,
        )
        for row in rows
    ]


def get_association_context(
    conn: sqlite3.Connection,
    episode_ids: list[str],
    min_strength: float = 0.5,
    limit: int = 20,
) -> str:
    """Format association context for inclusion in a wrap package.

    Produces human-readable text describing which episodes have been
    thought about together before. Includes content snippets so the
    LLM can evaluate whether connections still hold without needing
    to cross-reference opaque IDs.

    Args:
        conn: SQLite connection with associations + episodes tables.
        episode_ids: Episode IDs from the current wrap window.
        min_strength: Minimum strength to surface. Default 0.5.
        limit: Maximum associations to include.

    Returns:
        Formatted string for the wrap package, or empty string if no associations.
    """
    assocs = get_associations(conn, episode_ids, min_strength=min_strength, limit=limit)

    if not assocs:
        return ""

    # Build content snippet map for referenced episodes
    all_assoc_ids: set[str] = set()
    for a in assocs:
        all_assoc_ids.add(a.episode_a)
        all_assoc_ids.add(a.episode_b)

    snippet_map: dict[str, str] = {}
    if all_assoc_ids:
        placeholders = ",".join("?" for _ in all_assoc_ids)
        rows = conn.execute(
            f"SELECT id, content FROM episodes WHERE id IN ({placeholders})",
            list(all_assoc_ids),
        ).fetchall()
        for row in rows:
            # Truncate to first 80 chars for readability
            content = row[1][:80].replace("\n", " ")
            if len(row[1]) > 80:
                content += "..."
            snippet_map[row[0]] = content

    lines = ["## Episode Associations (from previous graduations)"]
    lines.append(
        "These episodes have been cited together in past compressions. "
        "Consider whether these connections still hold.\n"
    )
    for a in assocs:
        snip_a = snippet_map.get(a.episode_a, "(pruned)")
        snip_b = snippet_map.get(a.episode_b, "(pruned)")
        affect_info = ""
        if a.affective_tag:
            affect_info = f" [{a.affective_tag}, {a.affective_intensity:.1f}]"
        lines.append(
            f"- ({a.episode_a}) \"{snip_a}\" ↔ ({a.episode_b}) \"{snip_b}\": "
            f"co-cited {a.co_citations}x, strength {a.strength:.1f}{affect_info}"
        )

    return "\n".join(lines)


def association_stats(
    conn: sqlite3.Connection,
    total_episodes: int,
    top_n: int = 5,
) -> AssociationStats:
    """Compute association network health metrics.

    Args:
        conn: SQLite connection with associations table.
        total_episodes: Total episode count (for density calculation).
        top_n: Number of strongest pairs to include.

    Returns:
        AssociationStats with network metrics.
    """
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(AVG(strength), 0), COALESCE(MAX(strength), 0) "
        "FROM associations"
    ).fetchone()

    total_links = row[0]
    avg_strength = row[1]
    max_strength = row[2]

    # Density: links / possible_links
    # possible_links = n*(n-1)/2 for n episodes
    possible = total_episodes * (total_episodes - 1) / 2 if total_episodes > 1 else 1
    density = total_links / possible if possible > 0 else 0.0

    # Top N strongest pairs
    top_rows = conn.execute(
        """SELECT episode_a, episode_b, strength, co_citations,
                  first_linked, last_strengthened,
                  affective_tag, affective_intensity
           FROM associations
           ORDER BY strength DESC
           LIMIT ?""",
        (top_n,),
    ).fetchall()

    strongest = [
        AssociationPair(
            episode_a=r[0], episode_b=r[1], strength=r[2],
            co_citations=r[3], first_linked=r[4], last_strengthened=r[5],
            affective_tag=r[6], affective_intensity=r[7] or 0.0,
        )
        for r in top_rows
    ]

    return AssociationStats(
        total_links=total_links,
        avg_strength=avg_strength,
        max_strength=max_strength,
        density=density,
        strongest_pairs=strongest,
    )


def process_wrap_associations(
    store: Any,
    grad_result: Any,
    affective_state: AffectiveState | None = None,
) -> tuple[int, int, int]:
    """Orchestrate the full association pipeline after graduation validation.

    Shared by both MCP server and Engine paths to eliminate duplication.
    Extracts co-citation pairs from graduation results, records associations
    (with optional affective modulation), and triggers decay.

    Args:
        store: A Store instance (duck-typed to avoid circular imports).
        grad_result: A GraduationResult from validate_graduations().
        affective_state: Optional agent functional state during this wrap.

    Returns:
        Tuple of (formed, strengthened, decayed).
    """
    # Lazy import to avoid circular dependency
    from .graduation import extract_session_co_citations

    direct_pairs = set(grad_result.direct_co_citations)
    session_pairs = extract_session_co_citations(
        grad_result.all_validated_ids
    )

    formed, strengthened = 0, 0
    if direct_pairs or session_pairs:
        formed, strengthened = store.record_associations(
            direct_pairs=direct_pairs,
            session_pairs=session_pairs,
            affective_state=affective_state,
        )

    # Build canonical set of all strengthened pairs for decay exclusion
    all_strengthened: set[tuple[str, str]] = set()
    for pair in direct_pairs | session_pairs:
        cp = canonical_pair(*pair)
        if cp is not None:
            all_strengthened.add(cp)

    decayed = store.decay_associations(
        strengthened_pairs=all_strengthened
    )

    return formed, strengthened, decayed


# -- Internal --


def _upsert_association(
    conn: sqlite3.Connection,
    episode_a: str,
    episode_b: str,
    strength_delta: float,
    timestamp: str,
    affective_state: AffectiveState | None = None,
) -> bool:
    """Insert or strengthen an association link.

    When affective_state is provided, the tag and intensity are stored
    on the association record (most recent wins — audit trail preserves
    the full history of affective states across wraps).

    Returns True if a new link was created, False if an existing link
    was strengthened.
    """
    existing = conn.execute(
        "SELECT strength, co_citations FROM associations "
        "WHERE episode_a = ? AND episode_b = ?",
        (episode_a, episode_b),
    ).fetchone()

    affect_tag = affective_state.tag if affective_state else None
    affect_intensity = affective_state.intensity if affective_state else 0.0

    if existing is None:
        conn.execute(
            """INSERT INTO associations
               (episode_a, episode_b, strength, first_linked, last_strengthened,
                co_citations, affective_tag, affective_intensity)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
            (episode_a, episode_b, min(strength_delta, MAX_STRENGTH),
             timestamp, timestamp, affect_tag, affect_intensity),
        )
        return True
    else:
        new_strength = min(existing[0] + strength_delta, MAX_STRENGTH)
        conn.execute(
            """UPDATE associations
               SET strength = ?,
                   last_strengthened = ?,
                   co_citations = co_citations + 1,
                   affective_tag = COALESCE(?, affective_tag),
                   affective_intensity = CASE WHEN ? IS NOT NULL THEN ? ELSE affective_intensity END
               WHERE episode_a = ? AND episode_b = ?""",
            (new_strength, timestamp, affect_tag, affect_tag, affect_intensity,
             episode_a, episode_b),
        )
        return False
