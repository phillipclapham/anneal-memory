"""Word-by-word episode search — the explicit-query retrieval path.

:func:`search_episodes` ranks episodes against a free-text query, word by word, the
counterpart of ``Store.recall(keyword=...)``'s whole-phrase substring match. The query
is reduced to its distinctive keywords (:func:`extract_keywords`), each is fetched from
the store with the caller's filters applied in SQL, and an episode surfaces if it
contains at least one of them. Rarer words (corpus-IDF weights) and more matched words
rank first; a decision or outcome gets a small boost; recency then id break ties.

The IDF weight comes from the document frequency the candidate fetch already counted,
so a term found in most episodes (a corpus-learned stopword) weighs little and a rare
one weighs a lot. Below :data:`IDF_MIN_CORPUS` episodes the counts are too noisy and a
length-proxy weight stands in.

The prompt-mode constants and gates (``extract_keywords(mode="prompt")``) stay for the
keyword extractor's default; the search path itself runs in ``"query"`` mode, which has
no score bar, no length floor and no distinctive-anchor gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import log
from typing import Any, Literal

from .store import Store
from .types import Episode, EpisodeType, ScoredEpisode

RetrievalMode = Literal["prompt", "query"]

# --- Tuning (a harness adapter would expose these as config; defaults match the
# original recall-hook prototype so its measured keyword baseline carries over) ---
MAX_KEYWORDS = 14         # cap query breadth
MIN_KEYWORD_LEN = 4       # shorter tokens are rarely distinctive
SCORE_THRESHOLD = 2.5     # weighted-overlap floor to surface at all (precision bias)
MIN_HITS = 2              # require ≥ this many DISTINCT keyword hits, always
MIN_EPISODE_LEN = 80      # skip trivially short episodes (prompt mode)
CANDIDATE_LIMIT_PER_KEYWORD = 400  # per-keyword recall fetch cap before scoring
QUERY_CANDIDATE_LIMIT = 5000  # the same cap in query mode: a bound, not a convenience —
# a common word on a large store must not materialise every episode that holds it. The
# per-keyword document frequency (``total_matching``) stays exact whatever the cap.

# --- The two retrieval modes. "prompt" is the every-turn hook path and keeps every gate
# above. "query" is an EXPLICIT question an agent or operator asked on purpose: it opens
# the gates that exist to keep an unasked-for injection quiet (the keyword floor, the
# hit floor, the weighted-overlap bar, the distinctive anchor) and keeps everything that
# shapes the ranking (the IDF weights, the type boost, the caps); it also drops the
# MIN_EPISODE_LEN floor and keeps short ALL-CAPS / digit tokens as keywords. These
# are the query-mode values; the prompt-mode values stay the constants above, and nothing
# in this module rewrites a constant at call time. ---
RETRIEVAL_MODES = ("prompt", "query")
QUERY_MIN_KEYWORDS = 1
QUERY_MIN_HITS = 1

# Episode types that carry higher-signal prior thinking than the rest — the anneal
# analog of the prototype hook's "findings/decisions weigh more": a committed
# DECISION and a realized OUTCOME beat ambient OBSERVATION/CONTEXT for recall.
_HIGH_SIGNAL_TYPES = frozenset({EpisodeType.DECISION, EpisodeType.OUTCOME})
_TYPE_BOOST = 0.5

# Compact function-word stoplist (deliberately lean;
# memory-meaningful words like recall/continuity/memory are NOT here).
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
# Query mode tokenizes the original-cased text and admits 2-character tokens (S3, k8, v2);
# which short ones survive is decided in extract_keywords.
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def extract_keywords(query: str, *, mode: RetrievalMode = "prompt") -> list[str]:
    """Distinctive lowercased keywords from a query. Preserves snake_case /
    hyphenated domain terms whole. Stopwords + short tokens dropped, deduped, capped,
    order-preserving (the first/most-salient terms survive the cap).

    ``mode="query"`` additionally keeps a token shorter than :data:`MIN_KEYWORD_LEN`
    when it is written ALL-CAPS in the query or contains a digit, ``_`` or ``-`` (``SQL``,
    ``API``, ``S3``, ``k8s``, ``v2``): an explicit question names its short technical
    terms on purpose. A plain lowercase short word, and any stopword however it is
    cased, stays out. ``mode="prompt"`` (the default) is unchanged."""
    _check_mode(mode)
    seen: set[str] = set()
    out: list[str] = []
    if mode == "query":
        for raw in _QUERY_TOKEN_RE.findall(query):
            tok = raw.lower()
            if tok in _STOPWORDS or tok in seen:
                continue
            if len(tok) < MIN_KEYWORD_LEN and not (
                (raw.isalpha() and raw.isupper())
                or any(c.isdigit() or c in "_-" for c in raw)
            ):
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= MAX_KEYWORDS:
                break
        return out
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
    snake_case/hyphenated domain terms weigh up — a cheap IDF PROXY, no corpus query.

    Used as the small-corpus fallback. The proxy is corpus-BLIND: it can't tell that ``work`` (in ~30% of a
    real corpus) is a domain stopword while ``poker`` (~2%) is distinctive — both
    read as 1.0. :func:`_idf_weight` is the corpus-aware replacement on the
    Store-aware path, where the real document frequencies are available."""
    w = 1.0
    if len(kw) >= 8:
        w += 0.5
    if "_" in kw or "-" in kw:
        w += 1.0
    return w


# --- Corpus-IDF keyword weighting (the precision fix for the promiscuous meta-process
# recall — spore-141 / spore-104 dep-2, 2026-06-21). The length-proxy above is blind
# to corpus frequency, so high-document-frequency PROCESS words (work/commit/project/
# load — the connective tissue of a work corpus) score like distinctive terms and
# float whatever meta-process episode they brush onto ~every prompt. A real inverse-
# document-frequency weight, measured from the episode corpus, deflates those
# corpus-learned stopwords while preserving genuinely rare terms. ---
IDF_MIN_CORPUS = 50  # below this many episodes, document frequencies are too noisy to
# beat the length-proxy heuristic → fall back to it (and a fresh adopter's tiny corpus,
# or any test fixture, keeps the pre-IDF behavior until the corpus matures).
IDF_FLOOR = 0.30     # numerical-safety floor for a near-ubiquitous term's weight (and a
# small support contribution to an already-anchored item's score). It does NOT prevent
# common-word noise on its own — the DISTINCTIVE ANCHOR below does that structurally; the
# floor only bounds how much a floored term adds to the sum of an item that ALREADY has a
# real distinctive match. Below the floor → clamped here.
# The IDF regime's precision bar, calibrated on one operator's REAL receipt corpus (the threshold
# sweep, 2026-06-21). It is LOWER than SCORE_THRESHOLD (2.5) because the IDF weight band is
# lower than the length-proxy band — a SEPARATE constant (not a rescale of the weight) so
# the additive _TYPE_BOOST / multiplicity bonus keep their calibrated proportion. The
# length-proxy / small-corpus / Store-free paths keep SCORE_THRESHOLD untouched, so the
# test suite and pre-IDF behavior are byte-unchanged.
# ⚠ CALIBRATED ON ONE OPERATOR'S CORPUS. IDF_SCORE_THRESHOLD and IDF_MIN_CORPUS are tuned to a
# conceptual/partnership corpus regime (much repeated abstract vocabulary). An adopter on
# a different regime (entity-dense: code identifiers, proper nouns → most terms naturally
# rare) should RE-SWEEP both against their own receipt corpus — `1.6` will under-tighten
# there. The DISTINCTIVE ANCHOR below is corpus-size-RELATIVE (√N) and needs no re-sweep.
# (Per-call override / a distribution-relative bar is the deferred enhancement — spore.)
IDF_SCORE_THRESHOLD = 1.6
# The distinctiveness anchor — the STRUCTURAL guard against common-word accumulation
# (structural_invariants_beat_discipline). An item surfaces in the IDF regime only if it
# matched at least one keyword AT OR ABOVE this weight — i.e. one genuinely distinctive
# term, not merely a pile of mid/low-frequency process words. 0.5 on the normalized IDF
# scale ⟺ df ≤ √N (the geometric-mean document frequency): "rarer than the geometric mean
# of the corpus." Corpus-size-RELATIVE by construction, so it does NOT need per-regime
# re-calibration the way the absolute bar does. Without it, a long process-word-only prompt
# (6 floored words = 1.8 > 1.6) reintroduces the promiscuous leak in a softened costume
# (measured 2026-06-21); with it, no-distinctive-term prompts surface nothing, structurally.
IDF_ANCHOR_WEIGHT = 0.5


def _idf_weight(df: int, corpus_n: int) -> float:
    """Inverse-document-frequency keyword weight from the episode corpus — the precision
    fix's lever.

    ``df`` = how many of the ``corpus_n`` episodes contain the keyword (the exact,
    uncapped ``RecallResult.total_matching``). A term in MOST episodes (a corpus-learned
    stopword — e.g. ``work``/``commit``/``project``) gets a low weight; a rare
    distinctive term a high one. Smoothed (``+1``) so a ``df`` of 0 can't divide by zero,
    normalized by ``log(n+1)`` into ``[0, 1]`` then floored at :data:`IDF_FLOOR`. Only the
    endpoints are corpus-size-stable (unique → ~1.0; ubiquitous → ~0); a fixed-FRACTION
    term drifts down as the corpus grows, so :data:`IDF_SCORE_THRESHOLD` is a one-corpus
    calibration (the :data:`IDF_ANCHOR_WEIGHT` guard, being a √N point, does not drift).
    A term at/above :data:`IDF_ANCHOR_WEIGHT` is the distinctive anchor an item needs to
    surface; common words accumulate toward :data:`IDF_SCORE_THRESHOLD` only as support."""
    if corpus_n <= 1:  # totality guard (unreachable via _query_weights' IDF_MIN_CORPUS
        return IDF_FLOOR  # gate, but keeps the function safe for a direct caller — log(≤2))
    # df and corpus_n come from separate Store.recall() calls (the candidate fetch vs the
    # corpus count), so they are NOT one SQLite snapshot — a concurrent writer between them
    # can make df > corpus_n (a prune race) → a negative log. Clamp the invariant df ≤ n at
    # the source: the prune race floors the term (correct — "in ≥ all episodes" = ubiquitous)
    # instead of going semantically wrong. The opposite race (writer ADDS between the calls)
    # leaves df slightly under-counted vs n → marginally higher weight → direction-SAFE
    # (toward recall, away from the over-pruning the precision fix guards). A single-snapshot
    # stats read would erase even that residual; benign here, deferred (spore-141 follow-up).
    df = min(df, corpus_n)
    raw = log((corpus_n + 1) / (df + 1)) / log(corpus_n + 1)
    return max(IDF_FLOOR, raw)


def _query_weights(
    store: Store,
    keywords: list[str],
    doc_freq: dict[str, int] | None,
    *,
    until: str | None = None,
    filters: dict[str, Any] | None = None,
    corpus_n: int | None = None,
) -> tuple[dict[str, float], bool]:
    """The per-keyword weights for scoring, and whether corpus-IDF was applied (so the
    caller picks the matching precision bar — :data:`IDF_SCORE_THRESHOLD` vs
    :data:`SCORE_THRESHOLD`). Corpus-IDF when an episode corpus large enough to estimate
    document frequencies is present (``doc_freq`` captured from the candidate fetch,
    ``corpus_n`` >= :data:`IDF_MIN_CORPUS`); the length-proxy otherwise — the Store-free
    path (``doc_freq is None``), a sub-threshold corpus, and every small test fixture.

    ``until`` MUST match the cutoff the candidate fetch used: ``doc_freq`` is counted
    as-of that cutoff, so ``corpus_n`` is too — DF and N share one population, a valid
    frequency ratio (otherwise an ``exclude_recent_minutes`` caller would under-count DF
    against a whole-corpus N and read terms as more distinctive than they are).
    ``filters`` (``since`` / ``episode_type`` / ``source`` / ``include_superseded``) carries
    the same narrowing the candidate fetch used, for the same reason: DF and N must be
    counted over one population. ``corpus_n``, when given, is that population's size
    read in the same snapshot as ``doc_freq`` (:meth:`Store.keyword_candidates`), and
    no query is made."""
    if doc_freq is None:
        return {kw: _keyword_weight(kw) for kw in keywords}, False
    if corpus_n is None:
        corpus_n = store.recall(until=until, limit=0, **(filters or {})).total_matching
    if corpus_n < IDF_MIN_CORPUS:
        return {kw: _keyword_weight(kw) for kw in keywords}, False
    return {kw: _idf_weight(doc_freq.get(kw, 0), corpus_n) for kw in keywords}, True


def _check_mode(mode: str) -> None:
    """Reject anything but a known retrieval mode — a typo must not silently run the
    precision-biased path when the caller asked for the open one (or the reverse)."""
    if not isinstance(mode, str) or mode not in RETRIEVAL_MODES:
        raise ValueError(
            f"mode must be one of {list(RETRIEVAL_MODES)}, got {mode!r}"
        )


def _min_hits(mode: str) -> int:
    """The distinct-keyword-hit floor for ``mode`` (:data:`QUERY_MIN_HITS` in query mode, else :data:`MIN_HITS`)."""
    return QUERY_MIN_HITS if mode == "query" else MIN_HITS


def _min_episode_len(mode: str) -> int:
    """The shortest episode content that can surface in ``mode``. Prompt mode skips
    anything under :data:`MIN_EPISODE_LEN` (a hook should not inject a one-line scrap);
    query mode has no floor, because a short episode ("User is allergic to tree nuts.")
    is exactly what an explicit question is looking for."""
    return 0 if mode == "query" else MIN_EPISODE_LEN


def _precision_bar(used_idf: bool, mode: str = "prompt") -> float:
    """The regime-matched precision threshold: the (lower) IDF bar when the weights are
    corpus-IDF, the length-proxy bar otherwise, so candidates are gated on the same
    scale their weights live on. ``0.0`` in query mode: an explicit
    query has no score bar, only the hit floor."""
    if mode == "query":
        return 0.0
    return IDF_SCORE_THRESHOLD if used_idf else SCORE_THRESHOLD


def _anchor_floor(used_idf: bool, mode: str = "prompt") -> float:
    """The distinctiveness-anchor requirement for the regime: the √N
    :data:`IDF_ANCHOR_WEIGHT` under corpus-IDF (an item must match one genuinely rare
    term to surface), ``0.0`` under the length-proxy (no anchor gate — the proxy band has
    no meaningful frequency signal, and this keeps the Store-free / small-corpus / test
    paths byte-unchanged). ``0.0`` in query mode: an explicit query needs no distinctive anchor."""
    if mode == "query":
        return 0.0
    return IDF_ANCHOR_WEIGHT if used_idf else 0.0


def _score_text(
    text: str, keywords: list[str], weights: dict[str, float]
) -> tuple[float, int, float]:
    """Weighted keyword-overlap score for one item's searchable text. Returns
    (score, distinct_hit_count, top_hit_weight). Substring match (``kw in text``) so a
    query word matches inside a snake_case name/term. ``top_hit_weight`` is the largest
    weight among the MATCHED keywords (0.0 if none) — the distinctiveness-anchor signal:
    in the IDF regime an item must clear :data:`IDF_ANCHOR_WEIGHT` on it to surface."""
    lc = text.lower()
    score = 0.0
    hits = 0
    top = 0.0
    for kw in keywords:
        if kw in lc:
            w = weights[kw]
            score += w
            hits += 1
            if w > top:
                top = w
    return score, hits, top


def _fetch_episode_candidates(
    store: Store,
    keywords: list[str],
    *,
    until: str | None,
    filters: dict[str, Any] | None = None,
    uncapped: bool = False,
) -> tuple[dict[str, Episode], dict[str, int], int]:
    """Per-keyword episode candidates → (unioned candidate episodes, per-keyword document
    frequency, corpus size). One :meth:`Store.keyword_candidates` call: a single scan
    for every keyword instead of a count and a fetch per keyword, with the corpus size
    for corpus-IDF (:func:`_query_weights`) read in the same snapshot. ``doc_freq[kw]``
    is the EXACT, uncapped match count, as ``RecallResult.total_matching`` reports it.
    ``filters`` (``since`` / ``episode_type`` / ``source`` / ``include_superseded``) go
    straight to ``Store.recall`` so a filtered query narrows the candidates in SQL.
    ``uncapped`` (query mode) fetches up to :data:`QUERY_CANDIDATE_LIMIT` matches of each
    keyword instead of the newest :data:`CANDIDATE_LIMIT_PER_KEYWORD`: an explicit query
    should not lose an older episode that matches several words to a modest pile of newer
    one-word matches, and the ceiling keeps a very common word from stalling the process."""
    fetch_limit = QUERY_CANDIDATE_LIMIT if uncapped else CANDIDATE_LIMIT_PER_KEYWORD
    return store.keyword_candidates(
        keywords, limit_per_keyword=fetch_limit, until=until, **(filters or {})
    )


def _score_candidate_episodes(
    candidates: dict[str, Episode],
    keywords: list[str],
    weights: dict[str, float],
    *,
    score_threshold: float = SCORE_THRESHOLD,
    require_anchor: float = 0.0,
    min_hits: int | None = None,
    min_len: int | None = None,
) -> list[ScoredEpisode]:
    """Score pre-fetched candidate episodes by weighted overlap — the FULL ranked set
    clearing the precision bar (NOT capped; the caller slices it). ``score_threshold``
    is the regime-matched bar — :data:`SCORE_THRESHOLD` for length-proxy weights,
    :data:`IDF_SCORE_THRESHOLD` for the lower-band corpus-IDF weights. ``require_anchor``
    (>0 only in the IDF regime) is the distinctiveness anchor: an episode must have
    MATCHED a keyword at/above it — so a process-word-only query surfaces nothing.
    ``min_hits``
    defaults to :data:`MIN_HITS` and ``min_len`` to :data:`MIN_EPISODE_LEN`, both read at
    call time."""
    floor = MIN_HITS if min_hits is None else min_hits
    len_floor = MIN_EPISODE_LEN if min_len is None else min_len
    scored: list[ScoredEpisode] = []
    for ep in candidates.values():
        content = ep.content or ""
        if len(content) < len_floor:
            continue
        score, hits, top = _score_text(content, keywords, weights)
        if hits < floor or top < require_anchor:
            continue
        if ep.type in _HIGH_SIGNAL_TYPES:
            score += _TYPE_BOOST
        if score < score_threshold:
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
    # selection under the caller's cap).
    scored.sort(key=lambda e: (e.score, e.timestamp, e.id), reverse=True)
    return scored


@dataclass(frozen=True)
class EpisodeMatch:
    """One episode :func:`search_episodes` returned: the scored episode, the query
    keywords found in its content (``matched``, in query order), and, when it was
    searched with ``include_superseded=True``, the id of the episode that replaced it
    (``superseded_by``, else ``None``)."""

    episode: ScoredEpisode
    matched: tuple[str, ...]
    superseded_by: str | None = None


def search_episodes(
    store: Store,
    query: str,
    *,
    episode_type: EpisodeType | str | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 10,
    include_superseded: bool = False,
) -> list[EpisodeMatch]:
    """Rank episodes against a free-text query, word by word — the explicit-query
    counterpart of ``Store.recall(keyword=...)``'s whole-phrase substring match.

    The scoring runs in ``"query"`` mode: the query
    is reduced to its distinctive keywords (:func:`extract_keywords`), each is fetched
    from the store with the filters below applied in SQL, and an episode surfaces if it
    contains at least one of them. Episodes carrying more of the query's words, and
    rarer words (corpus-IDF weights), rank first; a decision or outcome gets the same
    small boost it gets in recall; recency then id break ties. No length floor applies
    (a one-line episode can match), and a short token written ALL-CAPS or containing a
    digit, ``_`` or ``-`` counts as a keyword (see :func:`extract_keywords`).

    Args:
        store: the episodic :class:`Store`.
        query: the text to search for; it need not appear verbatim anywhere.
        episode_type / source / since / until / include_superseded: the same filters
            ``Store.recall`` takes. The IDF weights are counted over the filtered
            episodes, so a narrow filter ranks by what is distinctive inside it.
        limit: maximum matches returned (``<= 0`` returns none).

    Returns:
        a ranked ``list`` of :class:`EpisodeMatch`, best first, at most ``limit``.
        Empty when the query reduces to no keywords or no episode contains one.

    Raises:
        ValueError: ``episode_type`` is not an episode type (checked before anything
            else, so a bad type fails even with ``limit <= 0``).
    """
    return _search_episodes(
        store, query, episode_type=episode_type, source=source, since=since,
        until=until, limit=limit, include_superseded=include_superseded,
    )[0]


def search_episodes_counted(
    store: Store,
    query: str,
    *,
    episode_type: EpisodeType | str | None = None,
    source: str | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 10,
    include_superseded: bool = False,
) -> tuple[list[EpisodeMatch], bool]:
    """:func:`search_episodes` plus whether the read was cut short: ``(matches, truncated)``.
    ``truncated`` is true when some keyword has more matching episodes than
    :data:`QUERY_CANDIDATE_LIMIT`, so the ranking considered only the newest of them. It is
    taken from the exact per-keyword match counts the search already computed, so it costs
    no extra query."""
    return _search_episodes(
        store, query, episode_type=episode_type, source=source, since=since,
        until=until, limit=limit, include_superseded=include_superseded,
    )


def _search_episodes(
    store: Store,
    query: str,
    *,
    episode_type: EpisodeType | str | None,
    source: str | None,
    since: str | None,
    until: str | None,
    limit: int,
    include_superseded: bool,
) -> tuple[list[EpisodeMatch], bool]:
    if episode_type is not None:
        episode_type = EpisodeType(episode_type)
    if limit <= 0:
        return [], False
    keywords = extract_keywords(query, mode="query")
    if len(keywords) < QUERY_MIN_KEYWORDS:
        return [], False
    filters: dict[str, Any] = {
        "since": since,
        "episode_type": episode_type,
        "source": source,
        "include_superseded": include_superseded,
    }
    candidates, doc_freq, corpus_n = _fetch_episode_candidates(
        store, keywords, until=until, filters=filters, uncapped=True
    )
    weights, used_idf = _query_weights(
        store, keywords, doc_freq, until=until, filters=filters, corpus_n=corpus_n
    )
    scored = _score_candidate_episodes(
        candidates, keywords, weights,
        score_threshold=_precision_bar(used_idf, "query"),
        require_anchor=_anchor_floor(used_idf, "query"),
        min_hits=_min_hits("query"),
        min_len=_min_episode_len("query"),
    )
    matches = [
        EpisodeMatch(
            episode=e,
            matched=tuple(kw for kw in keywords if kw in e.content.lower()),
            superseded_by=candidates[e.id].superseded_by,
        )
        for e in scored[:limit]
    ]
    return matches, any(n > QUERY_CANDIDATE_LIMIT for n in doc_freq.values())

