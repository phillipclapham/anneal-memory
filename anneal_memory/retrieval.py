"""On-demand relevance retrieval — the unified recall surface (AM-CRYSTAL-RECALL).

:func:`retrieve_relevant` is the **v2-consumer contract**: a harness's per-turn
recall hook calls ONE library function with the current prompt and gets back both
kinds of relevant memory — distilled crystallized PATTERNS and raw EPISODES —
already scored, ranked, and capped, with zero harness-side query logic. The hook
shell stays a thin trigger+format adapter; all the retrieval intelligence lives
here, harness-neutral.

This is the library-ization of the scoring that flow prototyped in its
``recall_injection_hook.py`` (the documented "swap retrieve_episodes for an anneal
call" seam). That hook tokenizes the prompt, OR-matches keywords against episode
content, and scores by weighted keyword overlap (distinctive snake_case /
hyphenated / long terms weigh more) — working around plain substring-LIKE
brittleness. Moving it into the library means every adopter (flow today, Levain v2
tomorrow) inherits the same precision-biased retrieval, and the SAME scan now also
surfaces crystallized patterns — the on-demand graduated tier — through the one
hook.

PRECISION BIAS — better to surface NOTHING than NOISE. Most prompts surface
nothing; the function only speaks when the store holds something that genuinely
bears on the query. Injecting noise would dilute the exact salience on-demand
recall exists to protect. Hence: a minimum distinctive-keyword count, a per-item
≥2-keyword-hit floor, and a weighted-overlap threshold.

RETRIEVAL BACKEND — keyword PLUS associative (Hebbian), LIVE. The episode tier is
weighted keyword overlap and is reliable (episodes carry rich, varied vocabulary).
The crystallized PATTERN tier is not: a graduated pattern is compressed to a sparse
name + clause, so its relevance to a query is usually SEMANTIC, not lexical — and
keyword scoring is blind to that (flow's dogfood measured it firing on ~2% of
relevant prompts, surfacing the wrong patterns on coincidental token overlap while
the genuinely-relevant ones stayed cold). So pattern retrieval is AUGMENTED with the
associative backend: query → keyword-matched episodes (the seed set) → their Hebbian
co-cited episodes (one hop) → the patterns whose ``evidence`` cites any of them. A
pattern grounded in an episode the query matched surfaces even with zero query-keyword
overlap — the keyword-orthogonal miss is fixed.

It is strictly additive (``retrieve_relevant(..., associative=True)``, default on):
the associative pass UNIONS extra patterns under the SAME precision gate + cap, never
removing a keyword hit, and it inherits the episode tier's precision (a pattern can
surface only if the query first matched an episode → no seed, no associative pattern).
That makes it regime-adaptive WITHOUT a flag: on an entity-dense corpus where keyword
already works, the associative pass surfaces what keyword found (or nothing clears the
gate); on a conceptual/partnership corpus it surfaces the cold patterns keyword can't
reach. The result shape and the consumer do not change — one backend swap under this
function lights up every harness (flow, Chip, the Levain/OpenHands adapters). It needs
the episodic :class:`Store` (the association graph lives there), so the Store-free
:func:`retrieve_patterns` stays keyword-only by design.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from math import log

from .crystal import CrystalDict, CrystalStore, activation_tier
from .store import Store
from .types import Episode, EpisodeType, RelevantPattern, RelevantResult, ScoredEpisode

# --- Tuning (a Levain adapter would expose these as config; defaults match the
# flow recall-hook prototype so flow's measured keyword baseline carries over) ---
MAX_PATTERNS = 3          # cap on crystallized patterns surfaced
MAX_EPISODES = 3          # cap on episodes surfaced
MIN_KEYWORDS = 2          # queries with fewer distinctive keywords → surface nothing
MAX_KEYWORDS = 14         # cap query breadth
MIN_KEYWORD_LEN = 4       # shorter tokens are rarely distinctive
SCORE_THRESHOLD = 2.5     # weighted-overlap floor to surface at all (precision bias)
MIN_HITS = 2              # require ≥ this many DISTINCT keyword hits, always
MIN_EPISODE_LEN = 80      # skip trivially short episodes (not applied to patterns)
CANDIDATE_LIMIT_PER_KEYWORD = 400  # per-keyword recall fetch cap before scoring

# --- Associative (Hebbian) pattern retrieval (AM-CRYSTAL-RECALL backend) ---
# The fix for keyword-ORTHOGONAL pattern relevance: a pattern whose distilled text
# shares no distinctive keyword with the query, but which was GROUNDED in an episode
# the query matched (or one co-cited with it). The keyword episode tier is reliable
# (rich, varied episode vocabulary); ``pattern.evidence`` + the co-citation graph
# carry that reliability into the sparse pattern tier. A Levain adapter would expose
# these as config; defaults are precision-first (better to miss than to flood).
ASSOC_SEED_LIMIT = 20          # cap keyword-matched episodes used as graph seeds
ASSOC_FETCH_LIMIT = 200        # cap on associations fetched for the one Hebbian hop.
# Fetched as ONE strength-ranked query across all seeds, so on a DENSE graph a
# high-degree seed could monopolize the budget and starve other seeds' hops — a
# recall-only bound (precision is unaffected, and the hop is a marginal extension over
# the direct evidence-citation path). 200 is ample headroom for current corpus sizes;
# per-seed-fair fetching is the follow-up IF the graph densifies AND the hop proves
# load-bearing (it is near-dead on today's sparse, decayed graphs — avg strength ~0.27).
ASSOC_MIN_STRENGTH = 0.5       # ignore decayed/noise-level links (mirrors wrap-context default)
ASSOC_HOP_FACTOR = 0.6         # discount one Hebbian hop vs a direct seed citation
ASSOC_STRENGTH_NORM = 2.0      # link strength at/above this passes full hop weight
ASSOC_SCORE_THRESHOLD = SCORE_THRESHOLD  # summed-reach floor to surface (same precision bar)

# Episode types that carry higher-signal prior thinking than the rest — the anneal
# analog of the prototype hook's "findings/decisions weigh more": a committed
# DECISION and a realized OUTCOME beat ambient OBSERVATION/CONTEXT for recall.
_HIGH_SIGNAL_TYPES = frozenset({EpisodeType.DECISION, EpisodeType.OUTCOME})
_TYPE_BOOST = 0.5

# Compact function-word stoplist (mirrors the flow hook — deliberately lean;
# flow-/memory-meaningful words like recall/continuity/memory are NOT here).
_STOPWORDS = frozenset("""
a an the this that these those there here it its it's is are was were be been being
am do does did doing have has had having will would shall should can could may might must
of in on at to from by for with about into over under again further then once
and or but nor so yet if because as until while of off out up down
i you he she we they me him her us them my your his our their mine yours
what which who whom whose where when why how all any both each few more most other some such
no not only own same than too very just also even still way ways thing things stuff
get got getting go going gone goes make makes made making want wants wanted need needs needed
think thinks thought know knows knew let lets let's really maybe kinda sorta gonna wanna
one two three first next last new old good bad big small lot lots bit
like likes liked use uses used using see sees saw look looks looking
me dude ok okay yeah yep nope hey hi
""".split())

# Tokens kept whole even though they contain punctuation (domain terms).
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{2,}")


def extract_keywords(query: str) -> list[str]:
    """Distinctive lowercased keywords from a query. Preserves snake_case /
    hyphenated domain terms whole. Stopwords + short tokens dropped, deduped, capped,
    order-preserving (the first/most-salient terms survive the cap)."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TOKEN_RE.findall(query.lower()):
        if len(tok) < MIN_KEYWORD_LEN or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= MAX_KEYWORDS:
            break
    return out


def _keyword_weight(kw: str) -> float:
    """Distinctive terms weigh more. Generic 4-7 char words = 1.0; long words and
    snake_case/hyphenated domain terms weigh up — a cheap IDF proxy, no corpus query."""
    w = 1.0
    if len(kw) >= 8:
        w += 0.5
    if "_" in kw or "-" in kw:
        w += 1.0
    return w


def _score_text(text: str, keywords: list[str], weights: dict[str, float]) -> tuple[float, int]:
    """Weighted keyword-overlap score for one item's searchable text. Returns
    (score, distinct_hit_count). Substring match (``kw in text``) so a query word
    matches inside a snake_case name/term."""
    lc = text.lower()
    score = 0.0
    hits = 0
    for kw in keywords:
        if kw in lc:
            score += weights[kw]
            hits += 1
    return score, hits


def _pattern_tags(crystal: CrystalDict) -> list[str]:
    """Coerce a crystal row's ``tags`` to ``list[str]`` for the read path. The library
    treats a corrupt store as a first-class operational state, so a hand-edited row may
    carry a non-list ``tags`` or non-str members. Normalize rather than (a) let a raw
    ``TypeError`` from ``' '.join`` escape the library's documented error boundary, or
    (b) emit a :class:`RelevantPattern` whose ``tags`` violates its ``list[str]``
    contract. A bare string becomes a SINGLE tag (not char-joined into "a p p a r…").
    Structural store corruption still raises ``CrystalError`` upstream in ``_load`` —
    this only guards a malformed field inside an otherwise-valid row."""
    raw = crystal.get("tags") or []
    if isinstance(raw, str):
        return [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    return [t for t in raw if isinstance(t, str) and t]


def _pattern_text(crystal: CrystalDict) -> str:
    """The searchable text of a crystallized pattern: its (high-signal snake_case)
    name + explanation + tags."""
    tags = _pattern_tags(crystal)
    return f"{crystal.get('name', '')} {crystal.get('explanation', '')} {' '.join(tags)}"


def _score_patterns(
    crystal_store: CrystalStore,
    keywords: list[str],
    weights: dict[str, float],
    *,
    max_patterns: int,
    today: date,
) -> list[RelevantPattern]:
    """Score the live crystallized corpus against the query. Same precision bias as
    episodes (≥MIN_HITS distinct keyword hits + SCORE_THRESHOLD), but NO length floor
    — a pattern's name alone can be a strong, short signal."""
    scored: list[RelevantPattern] = []
    for c in crystal_store.active():
        score, hits = _score_text(_pattern_text(c), keywords, weights)
        if hits < MIN_HITS or score < SCORE_THRESHOLD:
            continue
        _lvl = c.get("level")
        scored.append(
            RelevantPattern(
                name=str(c.get("name", "")),
                # bool is an int subclass — exclude it so a hand-corrupted level=True
                # doesn't read as 1 (write-path forbids bool; this guards a bad store).
                level=_lvl if isinstance(_lvl, int) and not isinstance(_lvl, bool) else 0,
                explanation=str(c.get("explanation", "")),
                tags=_pattern_tags(c),
                activation=activation_tier(c, today),
                score=round(score, 2),
            )
        )
    # Best score first; higher level then name break ties (a total order →
    # deterministic cap selection, not SQLite/insertion-order dependent).
    scored.sort(key=lambda p: (-p.score, -p.level, p.name))
    return scored[:max_patterns]


def _scored_episode_candidates(
    store: Store,
    keywords: list[str],
    weights: dict[str, float],
    *,
    until: str | None,
) -> list[ScoredEpisode]:
    """The FULL ranked set of episodes clearing the precision bar (NOT capped). Fetch
    candidates via the public ``Store.recall`` (one bounded LIKE query per keyword,
    unioned by id), then score by full weighted overlap. Reuses the public API — no
    new SQL, no Store internals.

    Returned uncapped because it serves two consumers: the displayed episode tier
    (sliced to ``max_episodes`` by :func:`_score_episodes`) AND the SEED set for
    associative pattern reach (which wants the wider relevant set, not just the top 3)."""
    candidates: dict[str, Episode] = {}
    for kw in keywords:
        result = store.recall(keyword=kw, until=until, limit=CANDIDATE_LIMIT_PER_KEYWORD)
        for ep in result.episodes:
            candidates.setdefault(ep.id, ep)

    scored: list[ScoredEpisode] = []
    for ep in candidates.values():
        content = ep.content or ""
        if len(content) < MIN_EPISODE_LEN:
            continue
        score, hits = _score_text(content, keywords, weights)
        if hits < MIN_HITS:
            continue
        if ep.type in _HIGH_SIGNAL_TYPES:
            score += _TYPE_BOOST
        if score < SCORE_THRESHOLD:
            continue
        scored.append(
            ScoredEpisode(
                id=ep.id,
                timestamp=ep.timestamp,
                type=ep.type.value if isinstance(ep.type, EpisodeType) else str(ep.type),
                source=ep.source or "",
                content=content,
                score=round(score, 2),
            )
        )
    # Best score first; recency then id break ties (a total order → deterministic
    # selection under the display cap and the associative seed slice).
    scored.sort(key=lambda e: (e.score, e.timestamp, e.id), reverse=True)
    return scored


def _score_episodes(
    store: Store,
    keywords: list[str],
    weights: dict[str, float],
    *,
    max_episodes: int,
    until: str | None,
) -> list[ScoredEpisode]:
    """The displayed episode tier: :func:`_scored_episode_candidates` capped at
    ``max_episodes``. Kept as the stable named entry for the episode-only path."""
    return _scored_episode_candidates(store, keywords, weights, until=until)[:max_episodes]


def _associative_patterns(
    store: Store,
    crystal_store: CrystalStore,
    seed_episodes: list[ScoredEpisode],
    *,
    max_patterns: int,
    today: date,
    exclude_names: set[str],
) -> list[RelevantPattern]:
    """Surface crystallized patterns the keyword pass MISSED, by reach through the
    Hebbian substrate: query → keyword-matched episodes (the seeds) → their co-cited
    episodes (one Hebbian hop) → the patterns whose ``evidence`` cites any of them.

    This is the canonical-Hebbian fix for keyword-ORTHOGONAL relevance — a pattern
    whose compressed text shares no distinctive keyword with the query, but which was
    GROUNDED in an episode the query matched (the ``pattern.evidence`` edge), or in
    one co-cited with such an episode during a past graduation (the association graph).

    Precision is INHERITED from the episode tier: a pattern surfaces ONLY if the query
    first matched an episode (no seed → empty ``reach`` → nothing), and only when its
    STRONGEST IDF-weighted episode reach clears ``ASSOC_SCORE_THRESHOLD``. Three guards
    keep it precision-first: (1) a directly-cited seed (the clean signal — every seed
    cleared the episode ``SCORE_THRESHOLD``) can clear the bar, but its contribution is
    down-weighted by an **evidence-IDF** so a *hub* episode (evidence for many patterns)
    the query merely brushed cannot float them all; (2) surfacing is decided by the
    single strongest reach, NOT the sum (summing rewards citation breadth over
    relevance), with multiplicity a small bounded rank bonus only; (3) a lone weak
    one-hop reach is discounted (strength × ``ASSOC_HOP_FACTOR``) below the bar.
    ``exclude_names`` drops patterns the keyword pass already surfaced (no double-count)."""
    if not seed_episodes:
        return []
    # episode id -> reach weight. A directly keyword-matched seed contributes its own
    # episode score (it already cleared the episode precision bar, so >= SCORE_THRESHOLD).
    reach: dict[str, float] = {e.id: e.score for e in seed_episodes}
    # Only the top-N seeds fan out a Hebbian hop (a cost bound); the FULL reach dict
    # above is kept for DIRECT evidence scoring below, so the hop cap can never drop a
    # directly-cited pattern. seed_episodes arrives score-sorted from the caller, so this
    # takes the highest-scoring seeds (made explicit, not insertion-order-implicit).
    seed_ids = [e.id for e in seed_episodes[:ASSOC_SEED_LIMIT]]
    seed_set = set(seed_ids)
    # The FULL seed set, not just the top-N that fan out a hop: a seed's DIRECT keyword
    # reach is authoritative and must never be overwritten by an indirect hop. seed_set
    # ⊆ all_seed_set, so a pair between a top-N seed and seed #N+1 (absent from seed_set)
    # would otherwise treat #N+1 as a hop `dst` and clobber its direct reach.
    all_seed_set = {e.id for e in seed_episodes}

    # One Hebbian hop: episodes co-cited with a seed during past graduations. The
    # non-seed side of each link is REACHED, weighted by its seed neighbour's score ×
    # the normalized link strength × a one-hop discount — so a hop alone rarely clears
    # the gate (precision-first), but a strong/multiply-reached one can boost a pattern.
    pairs = store.get_associations(
        seed_ids, min_strength=ASSOC_MIN_STRENGTH, limit=ASSOC_FETCH_LIMIT
    )
    for p in pairs:
        a_seed, b_seed = p.episode_a in seed_set, p.episode_b in seed_set
        if a_seed == b_seed:
            continue  # both seeds (already weighted) or neither (unreachable) — skip
        src, dst = (p.episode_a, p.episode_b) if a_seed else (p.episode_b, p.episode_a)
        hop = (
            reach.get(src, 0.0)
            * min(p.strength / ASSOC_STRENGTH_NORM, 1.0)
            * ASSOC_HOP_FACTOR
        )
        if dst in all_seed_set:
            continue  # dst is itself a seed — its direct keyword reach is authoritative
        # max-merge (NOT sum): a destination reached by several seeds keeps the single
        # strongest hop, so multiple weak hops can't accumulate past the gate.
        if hop > reach.get(dst, 0.0):
            reach[dst] = hop

    # evidence-IDF: an episode cited as evidence by MANY patterns is a weak relevance
    # discriminator — a hub episode the query merely brushed must not float every
    # pattern citing it. Count citations across the live corpus once and down-weight a
    # reached episode's contribution by its popularity (citing[e] >= 1 for any episode a
    # pattern cites, so log(1)=0 → a distinctive citation keeps full weight).
    active = list(crystal_store.active())
    citing: Counter[str] = Counter()
    for c in active:
        ev = c.get("evidence")
        if isinstance(ev, list):
            for e in ev:
                if isinstance(e, str):
                    citing[e] += 1

    scored: list[RelevantPattern] = []
    for c in active:
        name = str(c.get("name", ""))
        if name in exclude_names:
            continue
        evidence = c.get("evidence")
        if not isinstance(evidence, list):  # defensive: a hand-corrupted row
            continue
        weighted = [
            reach[e] / (1.0 + log(citing[e]))
            for e in evidence
            if isinstance(e, str) and e in reach
        ]
        if not weighted:
            continue
        # max-aggregation: surfacing is decided by the SINGLE strongest distinctive
        # reach (NOT the sum — summing rewards citation breadth over relevance and lets
        # several weak reaches accumulate past the gate). Multiplicity is only a small
        # bounded rank bonus, never enough to clear the gate on its own.
        strongest = max(weighted)
        if strongest < ASSOC_SCORE_THRESHOLD:
            continue
        score = strongest + min(len(weighted) - 1, 3) * 0.1
        _lvl = c.get("level")
        scored.append(
            RelevantPattern(
                name=name,
                level=_lvl if isinstance(_lvl, int) and not isinstance(_lvl, bool) else 0,
                explanation=str(c.get("explanation", "")),
                tags=_pattern_tags(c),
                activation=activation_tier(c, today),
                score=round(score, 2),
            )
        )
    scored.sort(key=lambda p: (-p.score, -p.level, p.name))
    return scored[:max_patterns]


def retrieve_relevant(
    store: Store,
    crystal_store: CrystalStore | None,
    query: str,
    *,
    max_patterns: int = MAX_PATTERNS,
    max_episodes: int = MAX_EPISODES,
    exclude_recent_minutes: int | None = None,
    now: str | None = None,
    today: date | None = None,
    associative: bool = True,
) -> RelevantResult:
    """Surface the memory relevant to ``query`` — crystallized patterns AND episodes
    — scored, ranked, and capped. THE on-demand recall contract a harness hook calls.

    Args:
        store: the episodic :class:`Store` to scan for relevant episodes.
        crystal_store: the :class:`CrystalStore` for the on-demand graduated tier, or
            ``None`` (an entity with no crystallized patterns yet → episodes only).
        query: the prompt / text to find relevant memory for.
        max_patterns / max_episodes: caps per kind (precision bias).
        exclude_recent_minutes: if set, episodes newer than this are excluded — the
            harness's "don't re-surface the live session's own echo" knob (the flow
            hook passes 45). Patterns are unaffected (they're distilled, not live).
        now: ISO-8601 UTC instant for the recent-exclusion cutoff (+ determinism);
            defaults to wall-clock. Only consulted when ``exclude_recent_minutes`` is set.
        today: logical date for crystallized-pattern activation tiers (+ determinism);
            defaults to ``date.today()``.
        associative: when ``True`` (default), pattern retrieval is AUGMENTED with the
            Hebbian backend — patterns whose ``evidence`` cites a keyword-matched
            episode (or one co-cited with it) surface even with zero query-keyword
            overlap, fixing the keyword-orthogonal miss. Strictly additive: it unions
            extra patterns under the SAME precision gate + cap, so it never removes a
            keyword hit and (a) needs the episodic ``Store`` (the association graph
            lives there) and (b) is a no-op when nothing keyword-matched an episode.
            Set ``False`` for pure keyword scoring (the pre-backend behavior).

    Returns:
        :class:`RelevantResult` with ``patterns`` + ``episodes`` (each scored/ranked)
        and the ``query_keywords`` the query reduced to. Empty (both lists ``[]``) when
        the query has fewer than :data:`MIN_KEYWORDS` distinctive keywords or nothing
        clears the precision threshold — surface nothing rather than noise.
    """
    today = today or date.today()
    keywords = extract_keywords(query)
    if len(keywords) < MIN_KEYWORDS:
        return RelevantResult(patterns=[], episodes=[], query_keywords=keywords)

    weights = {kw: _keyword_weight(kw) for kw in keywords}

    # The keyword-matched episode candidates are computed ONCE and serve two roles:
    # the displayed episode tier (capped) AND the SEED set for associative pattern
    # reach. The associative path needs them even when episodes aren't displayed
    # (max_episodes=0), so compute whenever EITHER consumer wants them.
    want_assoc = associative and crystal_store is not None and max_patterns > 0
    seed_episodes: list[ScoredEpisode] = []
    if max_episodes > 0 or want_assoc:
        until = _recent_cutoff(exclude_recent_minutes, now)
        seed_episodes = _scored_episode_candidates(store, keywords, weights, until=until)
    episodes = seed_episodes[:max_episodes] if max_episodes > 0 else []

    patterns: list[RelevantPattern] = []
    if crystal_store is not None and max_patterns > 0:
        patterns = _score_patterns(
            crystal_store, keywords, weights, max_patterns=max_patterns, today=today
        )
        # Keyword-first: the associative pass fills only the slots the keyword pass left
        # UNUSED. A keyword hit (overlap on the pattern's OWN text) is strictly
        # higher-confidence than evidence-mediated reach, so it can never be displaced
        # by a numerically-larger associative score — "strictly additive" by
        # construction, and regime-adaptive (a dense corpus fills its slots on keyword,
        # so the associative pass naturally no-ops).
        remaining = max_patterns - len(patterns)
        if want_assoc and seed_episodes and remaining > 0:
            patterns += _associative_patterns(
                store,
                crystal_store,
                seed_episodes,
                max_patterns=remaining,
                today=today,
                exclude_names={p.name for p in patterns},
            )

    return RelevantResult(patterns=patterns, episodes=episodes, query_keywords=keywords)


def retrieve_patterns(
    crystal_store: CrystalStore | None,
    query: str,
    *,
    max_patterns: int = MAX_PATTERNS,
    today: date | None = None,
) -> list[RelevantPattern]:
    """Patterns-only on-demand recall — the crystallized tier WITHOUT an episodic Store.

    :func:`retrieve_relevant` is the full contract (patterns AND episodes), but it
    requires a :class:`Store` even when ``max_episodes=0`` — so a harness hook that
    only wants the graduated-pattern tier (its episodes live elsewhere, or it wants
    none) would otherwise construct and open an episodic ``Store`` on EVERY turn
    purely to satisfy the signature. That per-turn open is a real cost + a
    write-lock contention risk against a concurrent single-writer wrap. This is that
    hook's contract: the SAME ``_score_patterns`` scoring and precision bias as the
    pattern half of :func:`retrieve_relevant`, with no ``Store`` touched.

    Args:
        crystal_store: the :class:`CrystalStore` for the on-demand graduated tier,
            or ``None`` (an entity with no crystallized patterns yet → ``[]``).
        query: the prompt / text to find relevant patterns for.
        max_patterns: cap (precision bias).
        today: logical date for crystallized-pattern activation tiers (+ determinism);
            defaults to ``date.today()``.

    Returns:
        a scored/ranked ``list[RelevantPattern]`` (best score first, a higher
        graduation level breaking ties), capped at ``max_patterns``. Empty when
        ``crystal_store`` is ``None``, ``max_patterns <= 0``, the query has fewer
        than :data:`MIN_KEYWORDS` distinctive keywords, or nothing clears the
        precision threshold — surface nothing rather than noise.

    Raises:
        CrystalError: if the crystal store is structurally corrupt or written by a
            newer schema (``CrystalStore._load`` deliberately surfaces a corrupt store
            rather than silently treating it as empty memory).
        OSError: on a filesystem access failure (permission denied, I/O error).

        This does NOT fail soft — a harness hook that wants "no recall beats a crash"
        wraps the call at ITS layer (``try: ... except Exception: return []``); hiding
        corruption in the library would defeat the fail-closed-on-corruption design.

    Parity:
        For a valid ``str`` query this equals ``retrieve_relevant(<any store>,
        crystal_store, query, max_patterns=max_patterns, max_episodes=0,
        associative=False, today=today).patterns`` — same keywords, weights, and
        ``_score_patterns`` call — but builds no episodic Store. The ``associative=False``
        is load-bearing: with the default ``associative=True`` the full function ALSO
        does Hebbian pattern reach (which needs the Store), so this Store-free entry is
        keyword-only by design — there is no associative path here. Two further caveats:
        a ``None`` ``crystal_store`` short-circuits to ``[]`` here without inspecting the
        query, and pass the SAME explicit ``today`` to both if comparing outputs (each
        defaults it independently, so a midnight-straddling pair can label ``activation``
        differently).
    """
    today = today or date.today()
    if crystal_store is None or max_patterns <= 0:
        return []
    keywords = extract_keywords(query)
    if len(keywords) < MIN_KEYWORDS:
        return []
    weights = {kw: _keyword_weight(kw) for kw in keywords}
    return _score_patterns(
        crystal_store, keywords, weights, max_patterns=max_patterns, today=today
    )


def _recent_cutoff(exclude_recent_minutes: int | None, now: str | None) -> str | None:
    """ISO-8601 cutoff for the recent-episode exclusion, or None to disable it. A
    parse failure on a caller-supplied ``now`` disables the exclusion (fail-open —
    never silently drop ALL episodes by producing a bogus cutoff)."""
    if not exclude_recent_minutes or exclude_recent_minutes <= 0:
        return None
    base: datetime
    if now:
        try:
            base = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except ValueError:
            return None
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
    else:
        base = datetime.now(timezone.utc)
    cutoff = base - timedelta(minutes=exclude_recent_minutes)
    return cutoff.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
