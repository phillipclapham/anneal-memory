"""Pattern-level associative graph for anneal-memory (AM-LINKGATE-DECAY, Slice B).

The CORTICAL sibling of the episode-level Hebbian layer (``associations.py``).
Where that layer links EPISODES (keyed on volatile ids, strengthened on
co-CITATION at graduation, decayed per-WRAP), this layer links graduated
PATTERNS by their stable NAMES, strengthened on co-RETRIEVAL, decayed on
calendar-DISUSE. The three differences are deliberate and load-bearing:

  episode layer (associations.py)      pattern layer (this module)
  ─────────────────────────────────    ─────────────────────────────────────
  keyed on episode ids (volatile)      keyed on pattern NAMES (stable) → dodges
                                       the dead-id problem STRUCTURALLY: a
                                       graduated pattern sheds its founding
                                       episodes as it matures, but its name
                                       persists, so the edge never dangles.
  strengthens on co-citation           strengthens on co-RETRIEVAL — the recall
   (a graduation EVENT, sparse)         hook logs which patterns surfaced
                                        together; retrieval is CONTINUOUS, so the
                                        graph stays live between graduations.
  decays per-wrap (0.9/wrap)           decays on calendar-DISUSE, lazy-on-touch.
                                        Wrap-count is regime-VARIANT (a high-wrap
                                        ops entity ages its whole graph faster in
                                        calendar terms — the Argus disease one
                                        layer up). Calendar time is the
                                        regime-INVARIANT decay clock.

CLS framing (Complementary Learning Systems): episode↔episode = hippocampal
(co-citation-formed, decay-prone; EXISTS). pattern↔pattern = CORTICAL
(co-retrieval-strengthened, survives episode consolidation; this module).

SHADOW MODE (Slice B). Nothing READS this graph for recall yet. The producer
(the harness recall hook logging co-surface events) and the consumer-of-log
(the wrap drain, here) build + telemeter the graph; recall stays
graph-independent. While recall does not consult the graph, the echo-chamber
(a pattern-pair the graph itself caused to co-surface voting up its own edge)
CANNOT form — textbook open-loop bring-up. Slice C (graph-consuming recall) is
GATED on a validation oracle. The ``formation_source`` / event ``basis`` fields
exist so that when Slice C lands, the drain can already distinguish an
INDEPENDENT co-surface (keyword / episode-evidence retrieval — true
co-activation) from a GRAPH-MEDIATED one (the graph pulled the second pattern
in — must NOT strongly reinforce, or the graph votes for itself). In Slice B
every basis is independent, so the gate is a no-op — but it is built, not bolted
on later.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import math
import sqlite3
from itertools import combinations
from typing import Any

from .types import PatternAssociationPair, PatternAssociationStats

# -- Tunable constants (oracle-tuned in the shadow phase; sane starting values) --

# Co-graduation seeding: weak link formed when 2+ patterns graduate the same
# wrap. The recall-INDEPENDENT exploration / selection-bias-corrector channel
# (graduation does not depend on keyword recall, so co-graduation seeds the
# pairs keyword-recall is blind to). MANDATORY per the design — without it the
# graph inherits the recall hook's keyword blind spots.
SEED_STRENGTH = 0.3

# Co-surface reinforcement: the per-(pair, session) base delta. Burst-damped by
# log1p of the in-session co-surface count so a 16x within-session burst
# contributes ~log1p(16)≈2.8x base, NOT 16x base. Distinct sessions each count
# (cross-session recurrence is the real signal; within-session repetition is one
# train of thought).
CO_SURFACE_BASE = 0.5

# Per-edge strength cap — prevents a long-lived pair becoming "immortal" and
# resisting decay (mirrors the episode layer's MAX_STRENGTH=10.0).
MAX_STRENGTH = 10.0

# Calendar lazy-decay: effective = stored * DECAY_PER_DAY ** days_since_materialized.
# Gentle (range 0.93–0.98). 0.96/day means a pair survives ~weeks of disuse,
# which the measured recurring-pair gap distribution (median-1 / p90-6 / max-8
# days) comfortably outlives — the graph holds a real recurring pair across its
# natural quiet stretches but lets a one-off fade.
DECAY_PER_DAY = 0.96

# Retrieval threshold: effective strength below this → the edge is too weak to
# surface for recall (Slice C consumes this). SEPARATE from the GC threshold:
# a sub-retrieval edge stays in the table (it may re-strengthen) — it just goes
# silent.
RETRIEVAL_THRESHOLD = 0.2

# GC threshold: effective strength below this → the row is dead weight, deleted.
# Disuse-decay IS the retirement-GC; there is no separate explicit GC pass.
GC_THRESHOLD = 0.03

# Homeostatic per-node outgoing-strength budget. A bare per-edge cap bounds
# magnitude but NOT concentration — one hub pattern could still accumulate
# unbounded TOTAL association mass (the Matthew effect). After reinforcement,
# each touched node's outgoing edges are scaled so their sum ≤ this budget.
NODE_OUTGOING_BUDGET = 15.0

# Provenance gate (Slice C echo-chamber structural fix). Independent co-surfaces
# (keyword / episode-evidence basis = true co-activation) reinforce at full
# weight; graph-mediated co-surfaces (the graph itself caused the co-surface)
# reinforce at this factor — 0.0 = the graph can never vote for its own edges.
# In Slice B no co-surface is graph-mediated, so this never fires; it is the
# structural guarantee that survives the Slice-C loop-close.
GRAPH_MEDIATED_REINFORCE_FACTOR = 0.0
_INDEPENDENT_BASES = frozenset({"keyword", "assoc_hop", "episode_evidence", "assoc_degraded"})

# How long a drained co-surface event_id is remembered for idempotency. The
# harness truncates its spool after a successful drain; this table is the
# crash-safety backstop (if truncation fails and the harness re-feeds the same
# events, they are filtered). Pruned beyond this horizon — a re-feed cannot
# happen after the spool has rotated.
PROCESSED_EVENT_RETENTION_DAYS = 30


# Schema. Appended to the Store's _init_schema via executescript, so existing
# DBs upgrade safely via CREATE IF NOT EXISTS (additive). The pattern graph
# lives in the episodic DB alongside the episode `associations` table but is a
# SEPARATE table (NOT typed rows in `associations`): pattern names and episode
# ids are different namespaces with different lifecycles, and mixing them invites
# the nullable-FK weirdness codex flagged. There is deliberately NO foreign key
# to a patterns table — pattern NAMES are the namespace (the crystal store +
# the neocortex `## Patterns`), and name stability is the whole point.
PATTERN_ASSOCIATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS pattern_associations (
    name_a TEXT NOT NULL,
    name_b TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.0,
    -- ACT-R fan-effect affordance: v1 treats edges as undirected, but the
    -- column exists NOW so a directed/asymmetric Sji can be added without a
    -- migration. 'undirected' until Slice C measures whether direction pays.
    direction TEXT NOT NULL DEFAULT 'undirected',
    first_linked_at TEXT NOT NULL,        -- ISO date (logical clock)
    last_strengthened_at TEXT NOT NULL,   -- ISO date — last reinforcement (any source)
    last_co_surfaced_at TEXT NOT NULL,    -- ISO date — last co-RETRIEVAL (telemetry)
    last_decay_at TEXT NOT NULL,          -- ISO date — strength is materialized as-of THIS date
    co_surface_count INTEGER NOT NULL DEFAULT 0,         -- raw co-surface events
    co_surface_session_count INTEGER NOT NULL DEFAULT 0, -- distinct sessions (burst-damped)
    formation_source TEXT NOT NULL,       -- 'co_graduation' | 'co_surface' | 'co_touch'
    PRIMARY KEY (name_a, name_b),
    CHECK (name_a < name_b)
);

CREATE INDEX IF NOT EXISTS idx_pattern_assoc_a ON pattern_associations(name_a);
CREATE INDEX IF NOT EXISTS idx_pattern_assoc_b ON pattern_associations(name_b);
CREATE INDEX IF NOT EXISTS idx_pattern_assoc_strength ON pattern_associations(strength DESC);

-- Rename / generation aliases. A pattern rename is first-class: the old name
-- aliases to the canonical (current) name so its earned edges follow the
-- rename instead of dangling. ``generation`` disambiguates the HOMONYM case
-- (the killer catch): re-crystallizing the SAME name for a DIFFERENT concept
-- mints a new generation, so the old concept's edges never silently re-attach
-- to the new one (semantic_homonym_edge_contamination — worse than a dead id
-- because it looks valid). ``canonical_name`` is NULL for a generation-bump
-- row (a tombstone marking "names at gen < N are a different concept").
CREATE TABLE IF NOT EXISTS pattern_aliases (
    alias TEXT NOT NULL,
    canonical_name TEXT,
    generation INTEGER NOT NULL DEFAULT 1,
    kind TEXT NOT NULL,                   -- 'rename' | 'generation'
    created_at TEXT NOT NULL,             -- ISO date (logical clock)
    PRIMARY KEY (alias, generation)
);

-- Idempotency backstop for the wrap drain. Records every co-surface event_id
-- already applied to the graph, so a re-fed spool (harness truncation crashed
-- between drain-commit and spool-clear) cannot double-count. Inserted in the
-- SAME transaction as the strengthening — atomic all-or-nothing.
CREATE TABLE IF NOT EXISTS processed_co_surface_events (
    event_id TEXT PRIMARY KEY,
    drained_at TEXT NOT NULL              -- ISO date (logical clock) — for retention pruning
);

CREATE INDEX IF NOT EXISTS idx_processed_cosurface_drained ON processed_co_surface_events(drained_at);
"""


# -- Date helpers (logical-clock discipline — spore-081). All dates are the
#    canonical fixed-width ``YYYY-MM-DD`` form so lexicographic compare is
#    chronological; never wall-clock inside the mechanics — the caller threads a
#    logical ``today``. --


def _date_part(iso: str) -> str:
    """The ``YYYY-MM-DD`` date prefix of any ISO string (date or datetime)."""
    return iso[:10]


def _days_between(date_from: str, date_to: str) -> int:
    """Whole calendar days from ``date_from`` to ``date_to`` (both ``YYYY-MM-DD``),
    floored at 0. A backwards or same-day span is 0 — decay never runs in
    reverse (guards the local/UTC evening-skew and backdated-run cases that bit
    spore-081)."""
    from datetime import date

    try:
        a = date.fromisoformat(_date_part(date_from))
        b = date.fromisoformat(_date_part(date_to))
    except ValueError:
        return 0
    return max(0, (b - a).days)


def effective_strength(
    stored: float, last_decay_at: str, today: str, decay_per_day: float = DECAY_PER_DAY
) -> float:
    """Strength after lazy calendar decay from its materialization date to ``today``.

    ``stored`` is the strength as-of ``last_decay_at``; this applies
    ``decay_per_day ** (calendar days since)`` without mutating the row. The
    single source of truth for "what is this edge worth right now" — reads and
    pre-reinforcement materialization both go through it.

    Corruption fail-safe (L1 LOW-2): a malformed/empty ``last_decay_at`` makes
    ``_days_between`` return 0, so the edge holds its stored value rather than
    crashing or over-decaying. The cost is that such a row never decays AND never
    GCs (the GC path uses this same function) — it is effectively immortal until a
    rename/sever rewrites its date. Acceptable for shadow-mode telemetry (fail
    toward preservation, not loss); revisit if corrupted dates ever appear.
    """
    days = _days_between(last_decay_at, today)
    if days <= 0:
        return stored
    return stored * (decay_per_day**days)


def canonical_pair(name_a: str, name_b: str) -> tuple[str, str] | None:
    """Canonically ordered pair (lexicographically smaller name first), or None
    for a self-pair. A pattern cannot associate with itself."""
    if name_a == name_b:
        return None
    return (name_a, name_b) if name_a < name_b else (name_b, name_a)


# -- Formation: co-graduation seeding (recall-independent exploration channel) --


def seed_co_graduation(
    conn: sqlite3.Connection,
    graduated_names: list[str],
    today: str,
    seed_strength: float = SEED_STRENGTH,
    commit: bool = True,
) -> int:
    """Seed weak links between patterns that graduated in the SAME wrap.

    Co-graduation is TRUE co-activation (the agent judged these patterns
    worth promoting together, on the same evidence window) and — critically —
    it is recall-INDEPENDENT, so it seeds the pairs the keyword recall hook is
    structurally blind to. This is the graph's exploration / selection-bias
    corrector; without it the graph compounds the recall hook's blind spots.

    Only seeds pairs that do NOT already exist (a real co-retrieval signal
    should not be diluted/reset by a weak seed); existing edges are left
    untouched. Returns the number of new links seeded.
    """
    formed = 0
    for a, b in combinations(sorted(set(graduated_names)), 2):
        pair = canonical_pair(a, b)
        if pair is None:
            continue
        ca, cb = pair
        exists = conn.execute(
            "SELECT 1 FROM pattern_associations WHERE name_a = ? AND name_b = ?",
            (ca, cb),
        ).fetchone()
        if exists is not None:
            continue
        conn.execute(
            """INSERT INTO pattern_associations
                   (name_a, name_b, strength, direction, first_linked_at,
                    last_strengthened_at, last_co_surfaced_at, last_decay_at,
                    co_surface_count, co_surface_session_count, formation_source)
               VALUES (?, ?, ?, 'undirected', ?, ?, ?, ?, 0, 0, 'co_graduation')""",
            (ca, cb, min(seed_strength, MAX_STRENGTH), today, today, today, today),
        )
        formed += 1
    if commit:
        conn.commit()
    return formed


# -- Strengthening: the idempotent co-surface drain --


def _aggregate_events(
    events: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], dict[str, int]], set[str]]:
    """Aggregate raw co-surface events into per-pair reinforcement deltas.

    Burst-damp is PER-SESSION: within one session a pair's repeated co-surfaces
    are damped by ``log1p(count)``; distinct sessions each contribute their own
    damped delta. Provenance-gated: an event whose ``basis`` is graph-mediated
    reinforces at ``GRAPH_MEDIATED_REINFORCE_FACTOR`` (0 in Slice B's no-op gate;
    the structural guarantee for Slice C).

    Returns (delta_by_pair, counts_by_pair{raw, sessions}, seen_event_ids).
    Events without an ``event_id`` or with < 2 surfaced names are ignored for
    pairing (a 0/1-pattern recall opportunity is still logged by the producer so
    that absence is unambiguous, but it forms no pair).
    """
    # (pair, session) -> [raw_count, reinforce_weight] ; weight is the per-event
    # provenance factor (independent=1, graph=factor), summed so a session with
    # mixed bases damps on the weighted count.
    per_session: dict[tuple[tuple[str, str], str], list[float]] = {}
    raw_total: dict[tuple[str, str], int] = {}
    sessions_seen: dict[tuple[str, str], set[str]] = {}
    seen_event_ids: set[str] = set()

    for ev in events:
        event_id = ev.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        # Dedup by event_id WITHIN this batch too (codex/kimi L3): the
        # cross-drain idempotency table only filters event_ids already committed,
        # so a duplicate event_id appearing twice in one ``events`` list (a
        # producer retry, a future re-feed path) would otherwise be aggregated
        # twice and double-count. Fresh uuid-per-event makes this unreachable
        # today, but the invariant should hold unconditionally.
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)
        names = ev.get("names") or []
        if not isinstance(names, list) or len(names) < 2:
            continue
        session = str(ev.get("session") or "")
        basis = str(ev.get("basis") or "keyword")
        factor = 1.0 if basis in _INDEPENDENT_BASES else GRAPH_MEDIATED_REINFORCE_FACTOR
        clean = sorted({str(n) for n in names if isinstance(n, str) and n})
        for a, b in combinations(clean, 2):
            pair = canonical_pair(a, b)
            if pair is None:
                continue
            key = (pair, session)
            slot = per_session.setdefault(key, [0.0, 0.0])
            slot[0] += 1.0  # raw co-surface count in this (pair, session)
            slot[1] += factor  # provenance-weighted count
            raw_total[pair] = raw_total.get(pair, 0) + 1
            sessions_seen.setdefault(pair, set()).add(session)

    delta_by_pair: dict[tuple[str, str], float] = {}
    for (pair, _session), (_raw, weighted) in per_session.items():
        if weighted <= 0:
            continue
        # log1p burst-damp on the provenance-weighted in-session count.
        delta_by_pair[pair] = delta_by_pair.get(pair, 0.0) + CO_SURFACE_BASE * math.log1p(weighted)

    counts_by_pair = {
        pair: {"raw": raw_total.get(pair, 0), "sessions": len(sessions_seen.get(pair, set()))}
        for pair in delta_by_pair
    }
    return delta_by_pair, counts_by_pair, seen_event_ids


def drain_co_surface_events(
    conn: sqlite3.Connection,
    events: list[dict[str, Any]],
    today: str,
    decay_per_day: float = DECAY_PER_DAY,
    commit: bool = True,
) -> dict[str, int]:
    """Drain a batch of co-surface events into the pattern graph — idempotently.

    The consumer-of-log half of the producer/consumer split. The harness recall
    hook appends co-surface events to a spool; at wrap time the harness reads the
    spool and calls this. Steps, all in ONE transaction (atomic — a crash rolls
    back everything, and a re-fed spool is filtered on the next drain):

      1. Filter out event_ids already in ``processed_co_surface_events`` (the
         idempotency backstop — a re-fed spool never double-counts).
      2. Aggregate the remaining events per-pair (per-session burst-damp,
         provenance gate) → reinforcement deltas.
      3. For each pair: materialize lazy calendar decay to ``today``, add the
         delta (capped), forming the edge if new.
      4. Homeostatic normalization: scale each touched node's outgoing edges so
         their sum ≤ NODE_OUTGOING_BUDGET (bounds CONCENTRATION, not just
         per-edge magnitude).
      5. Opportunistic GC: delete any touched edge whose effective strength fell
         below GC_THRESHOLD.
      6. Record every seen event_id (incl. the already-processed ones — harmless
         INSERT OR IGNORE) so the spool can be safely truncated.
      7. Prune processed-event rows beyond the retention horizon.

    Returns a metrics dict: {events_seen, events_applied, pairs_formed,
    pairs_strengthened, pairs_gc'd}.

    Known limitation (codex L3 MED-4): the per-session burst-damp + the
    ``co_surface_session_count`` are computed PER-DRAIN, not over persisted
    history. In the common case (one wrap per session → a session's events all
    drain together) this IS per-session. But a session that spans MULTIPLE wraps
    (a mid-session checkpoint wrap, then more prompts under the same session id)
    has its events split across drains: each drain damps its own slice
    (``2·base·log1p(1)`` instead of ``base·log1p(2)``) and increments
    ``co_surface_session_count`` once per slice, so a multi-wrap session
    over-strengthens slightly and inflates the session count. Acceptable for a
    strength-accumulating SHADOW graph (the distortion is small, the session count
    is telemetry not load-bearing, and multi-wrap sessions are uncommon); if they
    become common, a persisted per-(pair, session) accumulator is the fix.
    """
    delta_by_pair, counts_by_pair, seen_event_ids = _aggregate_events(events)

    # 1. Idempotency: which event_ids were already drained?
    already: set[str] = set()
    if seen_event_ids:
        placeholders = ",".join("?" for _ in seen_event_ids)
        rows = conn.execute(
            f"SELECT event_id FROM processed_co_surface_events WHERE event_id IN ({placeholders})",
            list(seen_event_ids),
        ).fetchall()
        already = {r[0] for r in rows}

    # Re-aggregate over ONLY the unprocessed events (so an already-drained
    # event contributes no delta). Cheap: re-filter and re-run the aggregator.
    if already:
        fresh_events = [e for e in events if e.get("event_id") not in already]
        delta_by_pair, counts_by_pair, _ = _aggregate_events(fresh_events)

    formed = strengthened = gced = 0
    touched_nodes: set[str] = set()

    # 3. Apply reinforcement (decay-materialize then add).
    for (ca, cb), delta in delta_by_pair.items():
        row = conn.execute(
            "SELECT strength, last_decay_at, co_surface_count, co_surface_session_count "
            "FROM pattern_associations WHERE name_a = ? AND name_b = ?",
            (ca, cb),
        ).fetchone()
        counts = counts_by_pair[(ca, cb)]
        if row is None:
            new_strength = min(delta, MAX_STRENGTH)
            conn.execute(
                """INSERT INTO pattern_associations
                       (name_a, name_b, strength, direction, first_linked_at,
                        last_strengthened_at, last_co_surfaced_at, last_decay_at,
                        co_surface_count, co_surface_session_count, formation_source)
                   VALUES (?, ?, ?, 'undirected', ?, ?, ?, ?, ?, ?, 'co_surface')""",
                (ca, cb, new_strength, today, today, today, today,
                 counts["raw"], counts["sessions"]),
            )
            formed += 1
        else:
            stored, last_decay_at = row[0], row[1]
            decayed = effective_strength(stored, last_decay_at, today, decay_per_day)
            new_strength = min(decayed + delta, MAX_STRENGTH)
            conn.execute(
                """UPDATE pattern_associations
                       SET strength = ?, last_decay_at = ?, last_strengthened_at = ?,
                           last_co_surfaced_at = ?,
                           co_surface_count = co_surface_count + ?,
                           co_surface_session_count = co_surface_session_count + ?
                       WHERE name_a = ? AND name_b = ?""",
                (new_strength, today, today, today, counts["raw"], counts["sessions"], ca, cb),
            )
            strengthened += 1
        touched_nodes.add(ca)
        touched_nodes.add(cb)

    # 4. Homeostatic per-node normalization over touched nodes.
    if touched_nodes:
        _normalize_nodes(conn, touched_nodes, today, decay_per_day)

    # 5. Opportunistic GC on touched edges (decay may have pushed one under the
    #    floor; the +delta just applied keeps a live pair safely above it).
    gced = _gc_touched(conn, touched_nodes, today, decay_per_day)

    # 6. Record processed event_ids (INSERT OR IGNORE — re-seen ids are no-ops).
    for event_id in seen_event_ids:
        conn.execute(
            "INSERT OR IGNORE INTO processed_co_surface_events (event_id, drained_at) VALUES (?, ?)",
            (event_id, today),
        )

    # 7. Prune the idempotency table beyond the retention horizon.
    _prune_processed_events(conn, today)

    if commit:
        conn.commit()

    return {
        "events_seen": len(seen_event_ids),
        "events_applied": len(seen_event_ids) - len(already),
        "pairs_formed": formed,
        "pairs_strengthened": strengthened,
        "pairs_gc": gced,
    }


def _normalize_nodes(
    conn: sqlite3.Connection, nodes: set[str], today: str, decay_per_day: float
) -> None:
    """Scale each touched node's outgoing edges so their effective-strength sum ≤
    NODE_OUTGOING_BUDGET. Bounds CONCENTRATION (the Matthew effect a bare per-edge
    cap misses).

    SINGLE-PASS + DETERMINISTIC (L1 LOW-1 / L2 M1): each node's scale factor is
    computed against the PRE-normalization snapshot, then each incident edge is
    scaled exactly ONCE by the MIN of its two endpoints' factors. An edge between
    two over-budget hubs is therefore bounded by the tighter budget rather than
    double-scaled by sequential per-node passes — and the result is independent of
    iteration order (so the oracle's ``concentration`` canary is stable). The
    node-sum invariant still holds: ``Σ eff_i·min(s_node, s_other_i) ≤ Σ
    eff_i·s_node = budget``. Operates on materialized (decayed-to-today) strengths;
    only edges that actually scale are written (decay stays lazy otherwise)."""
    if not nodes:
        return
    placeholders = ",".join("?" for _ in nodes)
    rows = conn.execute(
        f"""SELECT name_a, name_b, strength, last_decay_at FROM pattern_associations
            WHERE name_a IN ({placeholders}) OR name_b IN ({placeholders})""",
        [*nodes, *nodes],
    ).fetchall()
    if not rows:
        return
    eff: dict[tuple[str, str], float] = {}
    node_total: dict[str, float] = {}
    for name_a, name_b, strength, last_decay_at in rows:
        e = effective_strength(strength, last_decay_at, today, decay_per_day)
        eff[(name_a, name_b)] = e
        if name_a in nodes:
            node_total[name_a] = node_total.get(name_a, 0.0) + e
        if name_b in nodes:
            node_total[name_b] = node_total.get(name_b, 0.0) + e
    scale = {
        n: (NODE_OUTGOING_BUDGET / t if t > NODE_OUTGOING_BUDGET else 1.0)
        for n, t in node_total.items()
    }
    for (name_a, name_b), e in eff.items():
        s = min(scale.get(name_a, 1.0), scale.get(name_b, 1.0))
        if s < 1.0:
            conn.execute(
                "UPDATE pattern_associations SET strength = ?, last_decay_at = ? "
                "WHERE name_a = ? AND name_b = ?",
                (e * s, today, name_a, name_b),
            )


def _gc_touched(
    conn: sqlite3.Connection, nodes: set[str], today: str, decay_per_day: float
) -> int:
    """Delete touched edges whose effective strength fell below GC_THRESHOLD.
    Opportunistic — disuse-decay IS the retirement GC; there is no separate
    sweep. Only checks edges incident to nodes touched this drain."""
    if not nodes:
        return 0
    gced = 0
    placeholders = ",".join("?" for _ in nodes)
    rows = conn.execute(
        f"""SELECT name_a, name_b, strength, last_decay_at FROM pattern_associations
            WHERE name_a IN ({placeholders}) OR name_b IN ({placeholders})""",
        [*nodes, *nodes],
    ).fetchall()
    for name_a, name_b, strength, last_decay_at in rows:
        if effective_strength(strength, last_decay_at, today, decay_per_day) < GC_THRESHOLD:
            conn.execute(
                "DELETE FROM pattern_associations WHERE name_a = ? AND name_b = ?",
                (name_a, name_b),
            )
            gced += 1
    return gced


def _prune_processed_events(conn: sqlite3.Connection, today: str) -> None:
    """Drop processed-event rows older than the retention horizon."""
    from datetime import date, timedelta

    try:
        cutoff = (date.fromisoformat(_date_part(today)) - timedelta(days=PROCESSED_EVENT_RETENTION_DAYS)).isoformat()
    except ValueError:
        return
    conn.execute("DELETE FROM processed_co_surface_events WHERE drained_at < ?", (cutoff,))


# -- Full-graph maintenance (periodic; NOT per-wrap) --


def gc_pattern_associations(
    conn: sqlite3.Connection, today: str, decay_per_day: float = DECAY_PER_DAY, commit: bool = True
) -> int:
    """Sweep the WHOLE graph and delete edges below GC_THRESHOLD after decay-to-today.

    Lazy decay means a never-touched dead edge lingers until something near it is
    drained; this is the optional periodic sweep that reclaims those. It is NOT
    called per-wrap (that would re-introduce the regime-variant per-wrap cost the
    calendar model exists to avoid) — a harness calls it occasionally."""
    rows = conn.execute(
        "SELECT name_a, name_b, strength, last_decay_at FROM pattern_associations"
    ).fetchall()
    gced = 0
    for name_a, name_b, strength, last_decay_at in rows:
        if effective_strength(strength, last_decay_at, today, decay_per_day) < GC_THRESHOLD:
            conn.execute(
                "DELETE FROM pattern_associations WHERE name_a = ? AND name_b = ?",
                (name_a, name_b),
            )
            gced += 1
    if commit:
        conn.commit()
    return gced


# -- Lifecycle: rename (first-class alias) + homonym guard --


def _current_generation(conn: sqlite3.Connection, name: str) -> int:
    """The current concept generation for ``name`` (default 1). A homonym sever
    bumps this; edges formed under an older generation are a DIFFERENT concept."""
    row = conn.execute(
        "SELECT MAX(generation) FROM pattern_aliases WHERE alias = ?", (name,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 1


def rename_pattern(
    conn: sqlite3.Connection, old_name: str, new_name: str, today: str, commit: bool = True
) -> int:
    """Rename a pattern, carrying its earned edges to the new name (first-class
    alias). The SAME concept persists under a new name — so its association
    history must follow, not dangle. Edges incident to ``old_name`` are re-keyed
    to ``new_name``; if a re-keyed pair already exists, the two MERGE (max
    strength / counts, earliest first_linked, latest recency) rather than one
    clobbering the other; a self-pair created by the rename (old was linked to
    new) is dropped. Records a ``rename`` alias row.

    Design note (L2 L1): the ``rename`` alias row is an AUDIT record, not a
    runtime resolution table — the read/seed/drain paths do NOT resolve names
    through it, because the rename re-keys edges EAGERLY (they already live under
    ``new_name``) and the producer emits the neocortex's CURRENT name (so a
    post-rename co-surface already uses ``new_name``). ``pattern_aliases`` is read
    at runtime only for its ``generation`` column (the homonym guard, via
    ``_current_generation``). Returns edges re-keyed."""
    if old_name == new_name:
        return 0
    rows = conn.execute(
        """SELECT name_a, name_b, strength, first_linked_at, last_strengthened_at,
                  last_co_surfaced_at, last_decay_at, co_surface_count,
                  co_surface_session_count, formation_source
           FROM pattern_associations WHERE name_a = ? OR name_b = ?""",
        (old_name, old_name),
    ).fetchall()
    rekeyed = 0
    for r in rows:
        other = r[1] if r[0] == old_name else r[0]
        # Materialize each edge's stored strength to ``today`` so a merge compares
        # like-with-like (and the surviving row's last_decay_at is consistent).
        eff = effective_strength(r[2], r[6], today)
        pair = canonical_pair(new_name, other)
        # Delete the old edge first (its key changes).
        conn.execute(
            "DELETE FROM pattern_associations WHERE name_a = ? AND name_b = ?",
            (r[0], r[1]),
        )
        if pair is None:
            # old_name was linked to new_name → a self-pair after rename; drop it.
            continue
        ca, cb = pair
        existing = conn.execute(
            "SELECT strength, first_linked_at, last_strengthened_at, last_co_surfaced_at, "
            "last_decay_at, co_surface_count, co_surface_session_count "
            "FROM pattern_associations WHERE name_a = ? AND name_b = ?",
            (ca, cb),
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO pattern_associations
                       (name_a, name_b, strength, direction, first_linked_at,
                        last_strengthened_at, last_co_surfaced_at, last_decay_at,
                        co_surface_count, co_surface_session_count, formation_source)
                   VALUES (?, ?, ?, 'undirected', ?, ?, ?, ?, ?, ?, ?)""",
                # first_linked_at PRESERVES the original (r[3]) — a rename is the
                # same concept under a new name, so its edge was first linked when
                # the old edge was, not at rename time (L2 H1). last_decay_at=today
                # because ``eff`` is materialized to today.
                (ca, cb, min(eff, MAX_STRENGTH), r[3], r[4], r[5], today,
                 r[7], r[8], r[9]),
            )
        else:
            ex_eff = effective_strength(existing[0], existing[4], today)
            conn.execute(
                """UPDATE pattern_associations
                       SET strength = ?, last_decay_at = ?,
                           first_linked_at = MIN(first_linked_at, ?),
                           last_strengthened_at = MAX(last_strengthened_at, ?),
                           last_co_surfaced_at = MAX(last_co_surfaced_at, ?),
                           co_surface_count = co_surface_count + ?,
                           co_surface_session_count = co_surface_session_count + ?
                       WHERE name_a = ? AND name_b = ?""",
                (min(ex_eff + eff, MAX_STRENGTH), today, r[3], r[4], r[5],
                 r[7], r[8], ca, cb),
            )
        rekeyed += 1
    if rows:
        conn.execute(
            "INSERT OR IGNORE INTO pattern_aliases (alias, canonical_name, generation, kind, created_at) "
            "VALUES (?, ?, ?, 'rename', ?)",
            (old_name, new_name, _current_generation(conn, old_name), today),
        )
    if commit:
        conn.commit()
    return rekeyed


def sever_pattern_concept(
    conn: sqlite3.Connection, name: str, today: str, commit: bool = True
) -> int:
    """Homonym guard (codex's killer catch). When a pattern's concept LEAVES
    (composted / retired), delete its association edges and bump its generation —
    so a future pattern that re-uses the same NAME for a DIFFERENT concept starts
    with a clean slate instead of silently inheriting the old concept's edges
    (``semantic_homonym_edge_contamination`` — name-keying dodges dead ids but
    opens name-reuse; an old edge re-attaching to a new concept is WORSE than a
    dead id because it looks valid). Re-graduating the SAME concept simply
    re-forms its edges, so a genuine revival loses only the (now-stale) strength,
    not correctness. Returns edges severed."""
    severed = conn.execute(
        "SELECT COUNT(*) FROM pattern_associations WHERE name_a = ? OR name_b = ?",
        (name, name),
    ).fetchone()[0]
    conn.execute(
        "DELETE FROM pattern_associations WHERE name_a = ? OR name_b = ?", (name, name)
    )
    # Record the concept boundary: gen N retired, future edges are gen N+1.
    next_gen = _current_generation(conn, name) + 1
    conn.execute(
        "INSERT OR IGNORE INTO pattern_aliases (alias, canonical_name, generation, kind, created_at) "
        "VALUES (?, NULL, ?, 'generation', ?)",
        (name, next_gen, today),
    )
    if commit:
        conn.commit()
    return severed


# -- Reads / telemetry --


def get_pattern_associations(
    conn: sqlite3.Connection,
    names: list[str],
    today: str,
    min_strength: float = RETRIEVAL_THRESHOLD,
    decay_per_day: float = DECAY_PER_DAY,
    limit: int = 50,
) -> list[PatternAssociationPair]:
    """Pattern associations incident to any of ``names``, ranked by EFFECTIVE
    (decayed-to-today) strength descending, filtered to effective ≥
    ``min_strength``. Read-only — does NOT materialize decay (a read never
    mutates the graph). This is the surface Slice C will consume; in Slice B it
    is telemetry only."""
    if not names:
        return []
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"""SELECT name_a, name_b, strength, direction, first_linked_at,
                   last_strengthened_at, last_co_surfaced_at, last_decay_at,
                   co_surface_count, co_surface_session_count, formation_source
            FROM pattern_associations
            WHERE name_a IN ({placeholders}) OR name_b IN ({placeholders})""",
        [*names, *names],
    ).fetchall()
    out = []
    for r in rows:
        eff = effective_strength(r[2], r[7], today, decay_per_day)
        if eff < min_strength:
            continue
        out.append(
            PatternAssociationPair(
                name_a=r[0], name_b=r[1], strength=eff, stored_strength=r[2],
                direction=r[3], first_linked_at=r[4], last_strengthened_at=r[5],
                last_co_surfaced_at=r[6], last_decay_at=r[7],
                co_surface_count=r[8], co_surface_session_count=r[9],
                formation_source=r[10],
            )
        )
    out.sort(key=lambda p: (p.strength, p.name_a, p.name_b), reverse=True)
    return out[:limit]


def pattern_association_stats(
    conn: sqlite3.Connection, today: str, decay_per_day: float = DECAY_PER_DAY, top_n: int = 10
) -> PatternAssociationStats:
    """Pattern-graph health metrics — the shadow-phase telemetry surface.

    Reports stored AND effective (decayed) aggregates so the oracle can watch
    the graph live without a separate read path. ``concentration`` is the share
    of total effective strength held by the top edge (the echo-chamber canary:
    if it climbs monotonically the graph is collapsing onto a few pairs)."""
    rows = conn.execute(
        "SELECT name_a, name_b, strength, direction, first_linked_at, "
        "last_strengthened_at, last_co_surfaced_at, last_decay_at, "
        "co_surface_count, co_surface_session_count, formation_source "
        "FROM pattern_associations"
    ).fetchall()
    pairs = [
        PatternAssociationPair(
            name_a=r[0], name_b=r[1],
            strength=effective_strength(r[2], r[7], today, decay_per_day),
            stored_strength=r[2], direction=r[3], first_linked_at=r[4],
            last_strengthened_at=r[5], last_co_surfaced_at=r[6], last_decay_at=r[7],
            co_surface_count=r[8], co_surface_session_count=r[9], formation_source=r[10],
        )
        for r in rows
    ]
    total_links = len(pairs)
    effs = [p.strength for p in pairs]
    total_eff = sum(effs)
    avg_strength = (total_eff / total_links) if total_links else 0.0
    max_strength = max(effs) if effs else 0.0
    retrievable = sum(1 for e in effs if e >= RETRIEVAL_THRESHOLD)
    concentration = (max_strength / total_eff) if total_eff > 0 else 0.0
    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p.formation_source] = by_source.get(p.formation_source, 0) + 1
    pairs.sort(key=lambda p: (p.strength, p.name_a, p.name_b), reverse=True)
    return PatternAssociationStats(
        total_links=total_links,
        retrievable_links=retrievable,
        avg_strength=avg_strength,
        max_strength=max_strength,
        concentration=concentration,
        by_formation_source=by_source,
        strongest_pairs=pairs[:top_n],
    )
