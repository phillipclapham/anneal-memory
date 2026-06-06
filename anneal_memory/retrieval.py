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

RETRIEVAL BACKEND — keyword-first, associative STAGED. v1 is weighted keyword
overlap. The reliability story for the crystallized tier wants *associative*
retrieval (a query → relevant episodes → Hebbian co-cited episodes → the
crystallized patterns whose evidence cites them), which removes the
keyword-guessing failure mode. That backend is deliberately deferred behind the
measurement (``stability_is_observed_not_declared``): the keyword hit-rate flow
dogfoods IS the baseline an associative backend must beat, and is itself
decision-relevant (if keyword retrieval can't find the right pattern, that is the
empirical case for wiring the Hebbian layer in here). The seam: a future
``retrieve_relevant(..., associative=True)`` expands the candidate set via
:meth:`Store.get_associations` before scoring; the result shape and the consumer
do not change.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

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
    # Best score first; a higher graduation level breaks ties (3x before 2x).
    scored.sort(key=lambda p: (p.score, p.level), reverse=True)
    return scored[:max_patterns]


def _score_episodes(
    store: Store,
    keywords: list[str],
    weights: dict[str, float],
    *,
    max_episodes: int,
    until: str | None,
) -> list[ScoredEpisode]:
    """Fetch episode candidates via the public ``Store.recall`` (one bounded LIKE
    query per keyword, unioned by id), then score by full weighted overlap. Reuses
    the public API — no new SQL, no Store internals."""
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
    # Best score first; recency breaks ties.
    scored.sort(key=lambda e: (e.score, e.timestamp), reverse=True)
    return scored[:max_episodes]


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

    patterns: list[RelevantPattern] = []
    if crystal_store is not None and max_patterns > 0:
        patterns = _score_patterns(
            crystal_store, keywords, weights, max_patterns=max_patterns, today=today
        )

    episodes: list[ScoredEpisode] = []
    if max_episodes > 0:
        until = _recent_cutoff(exclude_recent_minutes, now)
        episodes = _score_episodes(
            store, keywords, weights, max_episodes=max_episodes, until=until
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
        today=today).patterns`` — same keywords, weights, and ``_score_patterns``
        call — but builds no episodic Store. Two caveats: a ``None`` ``crystal_store``
        short-circuits to ``[]`` here without inspecting the query, and pass the SAME
        explicit ``today`` to both if comparing outputs (each defaults it independently,
        so a midnight-straddling pair can label ``activation`` differently).
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
