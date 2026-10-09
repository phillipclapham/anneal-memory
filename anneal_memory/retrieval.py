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

RETRIEVAL BACKEND — keyword PLUS the evidence edge. The episode tier is weighted
keyword overlap and is reliable (episodes carry rich, varied vocabulary). The
crystallized PATTERN tier is not: a graduated pattern is compressed to a sparse
name + clause, so its relevance to a query is usually SEMANTIC, not lexical — and
keyword scoring is blind to that (flow's dogfood measured it firing on ~2% of
relevant prompts, surfacing the wrong patterns on coincidental token overlap while
the genuinely-relevant ones stayed cold). So pattern retrieval is AUGMENTED by an
associative pass: query → keyword-matched episodes (the seed set) → the patterns
whose ``evidence`` cites one of them (the evidence edge). A pattern grounded in an
episode the query matched surfaces even with zero query-keyword overlap.

Until 0.9.26 the pass also followed ONE Hebbian hop (seed → its co-cited episodes
→ the patterns citing those). It was removed after two measurements on flow's
store. On 2026-09-29, 0 of 788 production crystal exposures had come through it
(744 through the evidence edge, 44 by keyword), because the episodes the Hebbian
links connected and the episodes crystals cite were disjoint sets. On 2026-10-03,
a copy of the store was given the cheapest fix (link each crystal's own evidence
episodes to each other); replaying 1,000 real prompts, the hop added no pattern
that recall without it had not already surfaced, at the shipped constants and at
double strength. The Hebbian links still form at consolidation; recall does not read them.

Superseded episodes (a newer episode recorded as replacing them) are left out of
the candidate fetch, because it goes through ``Store.recall``'s default. So they
are not seeds either: a pattern whose evidence cites only a superseded episode no
longer surfaces through the evidence edge (keyword matching on the pattern's own
text is unaffected). That is chosen, not incidental: a pattern grounded only in a
replaced fact should not be resurfaced by it. A pattern that generalizes over the
old episode loses that path until it is re-grounded on current evidence.

It is strictly additive (``retrieve_relevant(..., associative=True)``, default on):
the associative pass UNIONS extra patterns under the SAME precision gate + cap, never
removing a keyword hit, and it inherits the episode tier's precision (a pattern can
surface only if the query first matched an episode → no seed, no associative pattern).
That makes it regime-adaptive WITHOUT a flag: on an entity-dense corpus where keyword
already works, the associative pass surfaces what keyword found (or nothing clears the
gate); on a conceptual/partnership corpus it surfaces the cold patterns keyword can't
reach. The result shape and the consumer do not change — one backend swap under this
function lights up every harness (flow, Chip, the Levain/OpenHands adapters). It needs
the episodic :class:`Store` (the seed episodes live there), so the Store-free
:func:`retrieve_patterns` stays keyword-only by design.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from collections import Counter
import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import log
from typing import Any, Literal

from .crystal import CrystalDict, CrystalStore, activation_tier
from .store import Store
from .durable import DurableFact, parse_durable_facts
from .types import (
    DEFAULT_TRUST,
    Episode,
    EpisodeType,
    RelevantFact,
    RelevantPattern,
    RelevantResult,
    REPLACED_CONTENT_MAX,
    ReplacedEpisode,
    ScoredEpisode,
)

RetrievalMode = Literal["prompt", "query"]

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

# --- Durable facts (the continuity's ``## Durable Facts`` section) ---
# The cue tier is deliberately NOT behind the gates above (the bar, the anchor, the hit
# floor, the keyword floor): a durable fact is one the composer wrote cue words for, and
# a one-word prompt ("restaurant?") must be able to bring it up. It runs on every prompt,
# so its precision guard is structural, six parts:
#   1. a cue matches by WHOLE-TOKEN equality (never a substring), after light stemming;
#      a token under three characters or a stopword never matches;
#   2. a cue or fact word that is INERT in this store never matches: it appears, as a
#      whole word, in more than DURABLE_GENERIC_DF of the store's own episodes ("time"
#      and "work" in a work store; "restaurant" stays a cue). The set of inert words is
#      computed when the continuity is saved (compute_durable_inert_tokens) and stored in
#      the metadata key INERT_TOKENS_KEY, tied to the continuity's hash; the prompt path
#      only READS it, and with no key or a stale hash applies no such filter (it never
#      counts document frequency itself). A store under IDF_MIN_CORPUS episodes has too
#      few to tell and gets an empty set;
#   3. a prompt with more than DURABLE_SHORT_PROMPT_TOKENS usable tokens needs TWO
#      distinct query tokens that matched (inflections of one token count once, and a
#      token matching both a cue and a fact word counts once); a short prompt still cues
#      on one;
#   4. the fact text alone cues a fact only through DURABLE_FACT_TEXT_MIN distinct
#      distinctive words of it (cue words alone are the primary path);
#   5. at most MAX_DURABLE_QUERY_TOKENS distinct usable query tokens are considered, so
#      a pasted document costs what a sentence costs;
#   6. at most MAX_DURABLE_FACTS surface per call, ranked by distinct matched query
#      tokens, then by section order.
MAX_DURABLE_FACTS = 2
MAX_DURABLE_QUERY_TOKENS = 12
DURABLE_GENERIC_DF = 0.10
DURABLE_SHORT_PROMPT_TOKENS = 2
DURABLE_FACT_TEXT_MIN = 2
INERT_TOKENS_KEY = "durable_inert_tokens"
_FACT_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FACT_MIN_TOKEN_LEN = 3
_EPISODE_PAGE = 1000  # episodes read per page when counting document frequency

# --- Associative pattern retrieval (the evidence edge; AM-CRYSTAL-RECALL backend) ---
# The fix for keyword-ORTHOGONAL pattern relevance: a pattern whose distilled text
# shares no distinctive keyword with the query, but which was GROUNDED in an episode
# the query matched. The keyword episode tier is reliable (rich, varied episode
# vocabulary); ``pattern.evidence`` carries that reliability into the sparse pattern
# tier. Defaults are precision-first (better to miss than to flood).
ASSOC_SCORE_THRESHOLD = SCORE_THRESHOLD  # reach floor — the DEFAULT only. retrieve_relevant
# passes the regime-matched bar (IDF_SCORE_THRESHOLD under corpus-IDF), so in production the
# effective associative gate tracks the episode/pattern bar, not this proxy-band constant.

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

    Used on the Store-FREE path (:func:`retrieve_patterns`) and as the small-corpus
    fallback. The proxy is corpus-BLIND: it can't tell that ``work`` (in ~30% of a
    real corpus) is a domain stopword while ``poker`` (~2%) is distinctive — both
    read as 1.0. :func:`_idf_weight` is the corpus-aware replacement on the
    Store-aware path, where the real document frequencies are available."""
    w = 1.0
    if len(kw) >= 8:
        w += 0.5
    if "_" in kw or "-" in kw:
        w += 1.0
    return w


# --- Corpus-IDF keyword weighting (the precision fix for the promiscuous flow-meta
# recall — spore-141 / spore-104 dep-2, 2026-06-21). The length-proxy above is blind
# to corpus frequency, so high-document-frequency PROCESS words (work/commit/project/
# load — the connective tissue of a work corpus) score like distinctive terms and
# float whatever flow-meta episode they brush onto ~every prompt. A real inverse-
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
# The IDF regime's precision bar, calibrated on flow's REAL receipt corpus (the threshold
# sweep, 2026-06-21). It is LOWER than SCORE_THRESHOLD (2.5) because the IDF weight band is
# lower than the length-proxy band — a SEPARATE constant (not a rescale of the weight) so
# the additive _TYPE_BOOST / multiplicity bonus keep their calibrated proportion. The
# length-proxy / small-corpus / Store-free paths keep SCORE_THRESHOLD untouched, so the
# test suite and pre-IDF behavior are byte-unchanged.
# ⚠ FLOW-CALIBRATED DEFAULTS. IDF_SCORE_THRESHOLD and IDF_MIN_CORPUS are tuned to flow's
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
    stopword — flow's ``work``/``commit``/``project``) gets a low weight; a rare
    distinctive term a high one. Smoothed (``+1``) so a ``df`` of 0 can't divide by zero,
    normalized by ``log(n+1)`` into ``[0, 1]`` then floored at :data:`IDF_FLOOR`. Only the
    endpoints are corpus-size-stable (unique → ~1.0; ubiquitous → ~0); a fixed-FRACTION
    term drifts down as the corpus grows, so :data:`IDF_SCORE_THRESHOLD` is a flow-N
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
    This is what makes :func:`retrieve_relevant` strictly more precise than the
    keyword-only :func:`retrieve_patterns` when a real corpus exists, while staying
    byte-identical to it on the tiny/empty corpora those tests use.

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


def _min_keywords(mode: str) -> int:
    """The keyword floor for ``mode``, read at call time so the module constant stays
    the one place a prompt-mode caller tunes it."""
    return QUERY_MIN_KEYWORDS if mode == "query" else MIN_KEYWORDS


def _min_hits(mode: str) -> int:
    """The distinct-keyword-hit floor for ``mode`` (see :func:`_min_keywords`)."""
    return QUERY_MIN_HITS if mode == "query" else MIN_HITS


def _min_episode_len(mode: str) -> int:
    """The shortest episode content that can surface in ``mode``. Prompt mode skips
    anything under :data:`MIN_EPISODE_LEN` (a hook should not inject a one-line scrap);
    query mode has no floor, because a short episode ("User is allergic to tree nuts.")
    is exactly what an explicit question is looking for."""
    return 0 if mode == "query" else MIN_EPISODE_LEN


def _precision_bar(used_idf: bool, mode: str = "prompt") -> float:
    """The regime-matched precision threshold: the (lower) IDF bar when the weights are
    corpus-IDF, the length-proxy bar otherwise. One bar for every tier a single
    :func:`retrieve_relevant` call scores, so episodes/patterns/associative-reach are
    gated on the same scale their weights live on. ``0.0`` in query mode: an explicit
    query has no score bar, only the hit floor."""
    if mode == "query":
        return 0.0
    return IDF_SCORE_THRESHOLD if used_idf else SCORE_THRESHOLD


def _anchor_floor(used_idf: bool, mode: str = "prompt") -> float:
    """The distinctiveness-anchor requirement for the regime: the √N
    :data:`IDF_ANCHOR_WEIGHT` under corpus-IDF (an item must match one genuinely rare
    term to surface), ``0.0`` under the length-proxy (no anchor gate — the proxy band has
    no meaningful frequency signal, and this keeps the Store-free / small-corpus / test
    paths byte-unchanged). The associative tier inherits it structurally: its seeds are
    the anchor-gated episodes, so a no-anchor query yields no seeds and no reach. ``0.0``
    in query mode: an explicit query needs no distinctive anchor."""
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
    score_threshold: float = SCORE_THRESHOLD,
    require_anchor: float = 0.0,
    min_hits: int | None = None,
) -> list[RelevantPattern]:
    """Score the live crystallized corpus against the query. Same precision bias as
    episodes (≥MIN_HITS distinct keyword hits + the bar), but NO length floor — a
    pattern's name alone can be a strong, short signal. ``score_threshold`` is the
    regime-matched bar (:data:`SCORE_THRESHOLD` proxy / :data:`IDF_SCORE_THRESHOLD` IDF);
    ``require_anchor`` (>0 only in the IDF regime) is the distinctiveness anchor — a
    pattern surfaces only if a MATCHED keyword clears it (a pile of common words can't).
    ``min_hits`` defaults to :data:`MIN_HITS`, read at call time."""
    floor = MIN_HITS if min_hits is None else min_hits
    scored: list[RelevantPattern] = []
    for c in crystal_store.active():
        score, hits, top = _score_text(_pattern_text(c), keywords, weights)
        if hits < floor or score < score_threshold or top < require_anchor:
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
                source="keyword",  # matched the pattern's OWN text (high-confidence)
            )
        )
    # Best score first; higher level then name break ties (a total order →
    # deterministic cap selection, not SQLite/insertion-order dependent).
    scored.sort(key=lambda p: (-p.score, -p.level, p.name))
    return scored[:max_patterns]


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
    clearing the precision bar (NOT capped). Serves two consumers: the displayed
    episode tier (sliced to ``max_episodes``) AND the SEED set for associative pattern
    reach (which wants the wider relevant set, not just the top 3). ``score_threshold``
    is the regime-matched bar — :data:`SCORE_THRESHOLD` for length-proxy weights,
    :data:`IDF_SCORE_THRESHOLD` for the lower-band corpus-IDF weights. ``require_anchor``
    (>0 only in the IDF regime) is the distinctiveness anchor: an episode must have
    MATCHED a keyword at/above it to seed — so a process-word-only query produces no
    seeds, and the associative pass it feeds inherits that structurally. ``min_hits``
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
    # selection under the display cap and the associative seed slice).
    scored.sort(key=lambda e: (e.score, e.timestamp, e.id), reverse=True)
    return scored


def _associative_patterns(
    crystal_store: CrystalStore,
    seed_episodes: list[ScoredEpisode],
    *,
    max_patterns: int,
    today: date,
    exclude_names: set[str],
    score_threshold: float = ASSOC_SCORE_THRESHOLD,
) -> list[RelevantPattern]:
    """Surface crystallized patterns the keyword pass MISSED, through the evidence
    edge: query → keyword-matched episodes (the seeds) → the patterns whose
    ``evidence`` cites one of them.

    This is the fix for keyword-ORTHOGONAL relevance — a pattern whose compressed
    text shares no distinctive keyword with the query, but which was GROUNDED in an
    episode the query matched.

    Precision is INHERITED from the episode tier: a pattern surfaces ONLY if the query
    first matched an episode it cites (no seed → empty ``reach`` → nothing), and only
    when its STRONGEST IDF-weighted seed reach clears ``score_threshold``. Two guards
    keep it precision-first: (1) a seed's contribution is down-weighted by an
    **evidence-IDF** so a *hub* episode (evidence for many patterns) the query merely
    brushed cannot float them all; (2) surfacing is decided by the single strongest
    reach, NOT the sum (summing rewards citation breadth over relevance), with
    multiplicity a small bounded rank bonus only. ``exclude_names`` drops patterns
    the keyword pass already surfaced (no double-count)."""
    if not seed_episodes:
        return []
    # episode id -> reach weight: a keyword-matched seed contributes its own episode
    # score (it already cleared the episode precision bar).
    reach: dict[str, float] = {e.id: e.score for e in seed_episodes}

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
        if strongest < score_threshold:
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
                source="evidence_edge",
            )
        )
    scored.sort(key=lambda p: (-p.score, -p.level, p.name))
    return scored[:max_patterns]


def _fact_tokens(text: str) -> list[str]:
    """Lowercase word tokens of ``text`` that may take part in a durable-fact match:
    at least three characters and not a stopword. Order-preserving, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in _FACT_TOKEN_RE.findall(text.lower()):
        if len(tok) < _FACT_MIN_TOKEN_LEN or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


@lru_cache(maxsize=65536)
def _token_forms(tok: str) -> frozenset[str]:
    """The token and its light stems: one trailing ``s`` removed while at least three
    characters remain, or one trailing ``es`` or ``ing`` removed while at least four
    remain (so ``rating``/``rat``, ``files``/``fil`` and ``lines``/``lin`` do not
    collide, and ``restaurants``/``restaurant`` and ``recipes``/``recipe`` agree).
    Two tokens match when their form sets intersect, so ``restaurateur`` agrees with
    neither."""
    forms = {tok}
    for suffix, keep in (("ing", 4), ("es", 4), ("s", 3)):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= keep:
            forms.add(tok[: -len(suffix)])
    return frozenset(forms)


def _canonical(tok: str) -> str:
    """One representative of a token's inflection family (its shortest form)."""
    return min(_token_forms(tok), key=lambda f: (len(f), f))


def continuity_hash(text: str | None) -> str:
    """The hash that ties a stored inert-token set to the continuity it was computed
    for: SHA-256 of the UTF-8 text with ``\\r\\n`` and lone ``\\r`` read as ``\\n``, which
    is what ``Store.load_continuity`` returns (universal newlines), so a continuity saved
    with CRLF line endings still matches itself on reload. An empty string for none."""
    canonical = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_durable_facts(store: Store) -> list[DurableFact]:
    """The durable facts of ``store``'s current continuity, parsed with the store's
    section schema. Never raises: no continuity file, no durable section, an unreadable
    file or schema all give ``[]`` (a missing memory tier must not break recall)."""
    try:
        return parse_durable_facts(store.load_continuity(), store.section_schema)
    except Exception:  # noqa: BLE001 - a recall tier fails soft by contract
        return []


def _durable_words(fact: DurableFact, mode: RetrievalMode) -> tuple[list[str], list[str]]:
    """A fact's matchable words: ``(cue words, fact-text words)``, each deduplicated."""
    cue_words = list(dict.fromkeys(w for cue in fact.cues for w in _fact_tokens(cue)))
    fact_words = list(dict.fromkeys(
        w for kw in extract_keywords(fact.fact, mode=mode) for w in _fact_tokens(kw)
    ))
    return cue_words, fact_words


def compute_durable_inert_tokens(store: Store, facts: list[DurableFact]) -> set[str]:
    """The cue and fact words of ``facts`` that are too common in ``store`` to cue
    anything: those found, as WHOLE words, in more than :data:`DURABLE_GENERIC_DF` of its
    episodes (``cat`` is not found in "category" or "concatenate"; ``rent`` is not found in
    "current"). An episode has a word when one of its own words shares a stem with it
    (:func:`_token_forms`), the same equality the cue match uses. The episodes are read in
    pages and tokenized in Python, nothing is held but counters, and the result is meant
    to be computed when a continuity is saved and stored under :data:`INERT_TOKENS_KEY`, not
    at prompt time. A store with fewer than :data:`IDF_MIN_CORPUS` episodes has too few to
    tell, and gets the empty set."""
    words: set[str] = set()
    for fact in facts:
        for mode in RETRIEVAL_MODES:
            cue_words, fact_words = _durable_words(fact, mode)  # type: ignore[arg-type]
            words.update(cue_words)
            words.update(fact_words)
    if not words:
        return set()
    wanted: dict[str, set[str]] = {}  # stem form -> the fact words that carry it
    for w in words:
        for form in _token_forms(w):
            wanted.setdefault(form, set()).add(w)
    seen_in: Counter[str] = Counter()
    total = 0
    offset = 0
    while True:
        page = store.recall(limit=_EPISODE_PAGE, offset=offset).episodes
        if not page:
            break
        offset += len(page)
        for ep in page:
            total += 1
            hit_words: set[str] = set()
            for tok in set(_FACT_TOKEN_RE.findall((ep.content or "").lower())):
                for form in _token_forms(tok):
                    if form in wanted:
                        hit_words.update(wanted[form])
            seen_in.update(hit_words)
    if total < IDF_MIN_CORPUS:
        return set()
    return {w for w in words if seen_in[w] / total > DURABLE_GENERIC_DF}


def _read_inert_tokens(store: Store, text: str | None) -> frozenset[str] | None:
    """The stored inert-token set when it is valid for this exact continuity, else ``None``.

    Valid means: the metadata key parses as an object, its ``continuity_hash`` is the hash
    of ``text``, its ``tokens`` are a list of strings, and its ``threshold`` is the current
    :data:`DURABLE_GENERIC_DF`. A missing, corrupt, stale or differently-tuned key is
    ``None``, and the caller then withholds the durable tier rather than serve facts that
    were never filtered (the next wrap writes a current key). One metadata read; never
    counts anything. A store with too few episodes to tell still has a valid key, with an
    empty set."""
    try:
        raw = store._get_metadata(INERT_TOKENS_KEY)
        if not raw:
            return None
        data = json.loads(raw)
        tokens = data.get("tokens")
        if (
            not isinstance(tokens, list)
            or not all(isinstance(t, str) for t in tokens)
            or data.get("continuity_hash") != continuity_hash(text)
            or data.get("threshold") != DURABLE_GENERIC_DF
            or type(data.get("episodes")) is not int
            or data["episodes"] < 0
        ):
            return None
        return frozenset(tokens)
    except Exception:  # noqa: BLE001 - a recall tier fails soft by contract
        return None


def match_durable_facts(
    facts: list[DurableFact],
    query: str,
    *,
    mode: RetrievalMode = "prompt",
    max_facts: int = MAX_DURABLE_FACTS,
    inert: frozenset[str] | set[str] = frozenset(),
) -> list[RelevantFact]:
    """The facts ``query`` cues, best first, at most ``max_facts``.

    A fact surfaces when a query token matches one of its cue tokens (cue phrases are
    split into word tokens), or when at least :data:`DURABLE_FACT_TEXT_MIN` distinct
    distinctive keywords of the fact text (:func:`extract_keywords` in ``mode``) match
    query tokens. Matching is whole-token equality after light stemming
    (:func:`_token_forms`), never a substring; a token under three characters or a
    stopword never matches, and neither does a cue or fact word in ``inert`` (the store's
    precomputed set, :func:`compute_durable_inert_tokens`). Only the first
    :data:`MAX_DURABLE_QUERY_TOKENS` distinct usable query tokens are considered. A prompt
    with more than :data:`DURABLE_SHORT_PROMPT_TOKENS` usable tokens needs matches on two
    DISTINCT query tokens: a token matching a cue and a fact word counts once, and so do
    inflections of one token. The retrieval gates (score bar, anchor, hit floor, keyword
    floor) do not apply to this tier, on purpose; its guard is the rule list at
    :data:`MAX_DURABLE_FACTS`. Ranking is by distinct matched query tokens, then section
    order. ``source`` is ``"cue"`` when any cue matched and ``"fact"`` otherwise;
    ``cue_matched`` and ``fact_matched`` keep the two kinds of match apart and ``matched``
    is their union."""
    _check_mode(mode)
    if max_facts <= 0 or not facts:
        return []
    # Tokens that are inert in this store are dropped BEFORE the cap, so a prompt that opens
    # with a dozen of them still reaches the cue behind them. What is left is the prompt for
    # the long-prompt rule: "time ... restaurant ... work" with time and work inert is a
    # short prompt.
    inert_forms: set[str] = set()
    for w in inert:
        inert_forms |= _token_forms(w)
    query_tokens = [
        t for t in _fact_tokens(query) if not (_token_forms(t) & inert_forms)
    ][:MAX_DURABLE_QUERY_TOKENS]
    if not query_tokens:
        return []
    need = 2 if len(query_tokens) > DURABLE_SHORT_PROMPT_TOKENS else 1
    query_forms = [(_canonical(tok), _token_forms(tok)) for tok in query_tokens]

    def matches(candidates: list[str]) -> list[tuple[str, str]]:
        """``(word, query token family)`` for each candidate that matches a query token;
        one entry per inflection family of candidate."""
        out: list[tuple[str, str]] = []
        families: set[str] = set()
        for w in candidates:
            if w in inert:
                continue
            canon_w = _canonical(w)
            if canon_w in families:
                continue
            forms = _token_forms(w)
            for canon_q, qf in query_forms:
                if forms & qf:
                    families.add(canon_w)
                    out.append((w, canon_q))
                    break
        return out

    ranked: list[tuple[int, int, RelevantFact]] = []
    for position, fact in enumerate(facts):
        cue_words, fact_words = _durable_words(fact, mode)
        cue_hits = matches(cue_words)
        cue_families = {_canonical(w) for w, _q in cue_hits}
        fact_hits = [
            (w, q) for w, q in matches(fact_words) if _canonical(w) not in cue_families
        ]
        if not cue_hits and len(fact_hits) < DURABLE_FACT_TEXT_MIN:
            continue
        distinct_query_tokens = {q for _w, q in cue_hits} | {q for _w, q in fact_hits}
        if len(distinct_query_tokens) < need:
            continue
        cue_matched = tuple(w for w, _q in cue_hits)
        fact_matched = tuple(w for w, _q in fact_hits)
        ranked.append((
            len(distinct_query_tokens),
            position,
            RelevantFact(
                fact=fact.fact,
                line=fact.line,
                matched=cue_matched + fact_matched,
                source="cue" if cue_hits else "fact",
                cue_matched=cue_matched,
                fact_matched=fact_matched,
            ),
        ))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return [r[2] for r in ranked[:max_facts]]


def durable_facts_for(
    store: Store, query: str, *, mode: RetrievalMode = "prompt"
) -> list[RelevantFact]:
    """The durable facts of ``store``'s continuity that ``query`` cues. Reads the store's
    stored inert-token set (:data:`INERT_TOKENS_KEY`); with no valid set for the current
    continuity (none written yet, corrupt, stale, or tuned differently) the tier is
    WITHHELD and this returns ``[]``, until the next wrap writes a current one.
    Structurally never raises: any failure at all gives ``[]``, because a
    recall tier on every prompt must not be able to break the prompt."""
    try:
        text = store.load_continuity()
        facts = parse_durable_facts(text, store.section_schema)
        if not facts:
            return []
        inert = _read_inert_tokens(store, text)
        if inert is None:  # fail closed: no valid set for this continuity, no facts
            return []
        return match_durable_facts(facts, query, mode=mode, inert=inert)
    except Exception:  # noqa: BLE001 - see the docstring
        return []


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
    mode: RetrievalMode = "prompt",
    durable: bool = True,
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
            harness's "don't re-surface the live session's own echo" knob. Patterns
            are unaffected (they're distilled, not live).
        now: ISO-8601 UTC instant for the recent-exclusion cutoff (+ determinism);
            defaults to wall-clock. Only consulted when ``exclude_recent_minutes`` is set.
        today: logical date for crystallized-pattern activation tiers (+ determinism);
            defaults to ``date.today()``.
        associative: when ``True`` (default), pattern retrieval is AUGMENTED with the
            associative pass — patterns whose ``evidence`` cites a keyword-matched
            episode (the evidence edge) surface even with zero query-keyword overlap.
            Strictly additive: it unions extra patterns under the SAME precision gate
            + cap, so it never removes a keyword hit and (a) needs the episodic
            ``Store`` (the seed episodes live there) and (b) is a no-op when nothing
            keyword-matched an episode.
            Set ``False`` for pure keyword scoring (the pre-backend behavior).
        mode: ``"prompt"`` (default) is the every-turn hook path: every precision gate
            as documented above, unchanged. ``"query"`` is an explicit question an agent
            or operator asked on purpose: a single distinctive keyword is enough
            (:data:`QUERY_MIN_KEYWORDS`), one keyword hit is enough
            (:data:`QUERY_MIN_HITS`), and neither the weighted-overlap bar nor the
            distinctive anchor applies, for episodes, patterns and the evidence edge
            alike, except that the evidence edge keeps the prompt-mode score bar. Query
            mode also drops the :data:`MIN_EPISODE_LEN` floor and keeps short ALL-CAPS
            or digit/``_``/``-`` tokens as keywords (:func:`extract_keywords`). The IDF
            weights, the ranking and the caps are the same in both modes. Anything
            else raises ``ValueError``.
        durable: when ``True`` (default), the result's ``facts`` holds the durable facts
            (the store's current continuity, ``## Durable Facts`` section) the query
            cues, at most :data:`MAX_DURABLE_FACTS`. This tier is NOT behind the gates
            above (it runs even for a one-word query, before the keyword floor) and is
            additive: ``patterns`` and ``episodes`` are the same with it on or off. Its
            precision guard replaces the gates: whole-token matching after light
            stemming (no substring, no stopword, no token under three characters); a
            query token that appears in more than :data:`DURABLE_GENERIC_DF` of the
            store's own episodes never matches (applied only to a store with
            :data:`IDF_MIN_CORPUS` or more episodes); a prompt with more than
            :data:`DURABLE_SHORT_PROMPT_TOKENS` usable tokens needs two distinct matched
            tokens; the fact text alone cues a fact only through
            :data:`DURABLE_FACT_TEXT_MIN` distinct words; and the cap. See
            :func:`match_durable_facts`. Never raises: a store with no continuity or no
            durable section gives ``[]``. ``False`` leaves ``facts`` empty.

    Returns:
        :class:`RelevantResult` with ``patterns`` + ``episodes`` (each scored/ranked)
        and the ``query_keywords`` the query reduced to. In prompt mode, empty (both
        lists ``[]``) when the query has fewer than :data:`MIN_KEYWORDS` distinctive
        keywords or nothing clears the precision threshold — surface nothing rather
        than noise.

    Consistency:
        A recall reads one committed state, as of its start: an episode deleted or
        erased before recall begins is never returned; a delete that commits while a
        recall runs may or may not be reflected, as with any database read. The episode
        half (candidates, weights, the supersession redirect) runs in one read
        transaction; the crystal store and durable facts are separate files, read outside.

    Raises:
        ValueError: ``mode`` is not ``"prompt"`` or ``"query"``.
    """
    _check_mode(mode)
    today = today or date.today()
    facts = durable_facts_for(store, query, mode=mode) if durable else []
    keywords = extract_keywords(query, mode=mode)
    if len(keywords) < _min_keywords(mode):
        return RelevantResult(
            patterns=[], episodes=[], query_keywords=keywords, facts=facts
        )

    # The keyword-matched episode candidates are computed ONCE and serve two roles:
    # the displayed episode tier (capped) AND the SEED set for associative pattern
    # reach. The associative path needs them even when episodes aren't displayed
    # (max_episodes=0), so compute whenever EITHER consumer wants them. The same fetch
    # captures the per-keyword document frequencies that weight the whole query by
    # corpus-IDF (the precision fix) — so the weighting is corpus-aware exactly when a
    # corpus is being scanned, and the length-proxy otherwise (the keyword-only path).
    want_assoc = associative and crystal_store is not None and max_patterns > 0
    # The episode half reads ONE committed state (snapshot isolation): candidate fetch,
    # corpus weights, the supersession check and the redirect all run in a single read
    # transaction (nested inside a caller's open one, it just joins it). Nothing on this
    # path writes. The crystal store and durable facts are other files, read outside.
    with store._db_boundary("recall"), store._read_snapshot():
        seed_episodes: list[ScoredEpisode] = []
        redirect = False
        if max_episodes > 0 or want_assoc:
            until = _recent_cutoff(exclude_recent_minutes, now)
            candidates, doc_freq, corpus_n = _fetch_episode_candidates(
                store, keywords, until=until, uncapped=mode == "query"
            )
            weights, used_idf = _query_weights(
                store, keywords, doc_freq, until=until, corpus_n=corpus_n
            )
            seed_episodes = _score_candidate_episodes(
                candidates, keywords, weights,
                score_threshold=_precision_bar(used_idf, mode),
                require_anchor=_anchor_floor(used_idf, mode),
                min_hits=_min_hits(mode),
                min_len=_min_episode_len(mode),
            )
            redirect = max_episodes > 0 and store.has_supersessions()
        else:
            # Keyword-only pattern path (no episode fetch): the length-proxy, byte-identical
            # to retrieve_patterns (the parity contract holds on this branch by construction).
            weights, used_idf = _query_weights(store, keywords, None)
        episodes = seed_episodes[:max_episodes] if max_episodes > 0 else []
        if redirect:
            episodes = _swap_replaced(store, keywords, seed_episodes, max_episodes,
                                      until=until, mode=mode)
        if episodes:
            # CAP-08 D3: each shown episode carries its effective trust, so a hook
            # can render relayed content as data. Read in the same snapshot, after
            # the redirect, so a replacing episode carries its own class, and each
            # replaced episode it carries (whose text it shows) carries its own.
            episode_trust = store.effective_trust_map(
                [ep.id for ep in episodes] + [r.id for ep in episodes for r in ep.replaces])
            episodes = [
                dataclasses.replace(
                    ep, trust=episode_trust.get(ep.id, DEFAULT_TRUST),
                    replaces=tuple(
                        dataclasses.replace(r, trust=episode_trust.get(r.id, DEFAULT_TRUST))
                        for r in ep.replaces))
                for ep in episodes
            ]
    # One regime-matched precision bar + anchor for every tier this call scores: the
    # lower IDF bar + the √N distinctiveness anchor when the weights are corpus-IDF, the
    # length-proxy bar + no anchor (0.0) otherwise.
    thr = _precision_bar(used_idf, mode)
    anchor = _anchor_floor(used_idf, mode)

    patterns: list[RelevantPattern] = []
    if crystal_store is not None and max_patterns > 0:
        patterns = _score_patterns(
            crystal_store, keywords, weights, max_patterns=max_patterns, today=today,
            score_threshold=thr, require_anchor=anchor, min_hits=_min_hits(mode),
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
                crystal_store,
                seed_episodes,
                max_patterns=remaining,
                today=today,
                exclude_names={p.name for p in patterns},
                # The evidence edge keeps the PROMPT bar in query mode too: the query
                # relaxation is for direct keyword matches, and one common word that
                # brushed an episode must not float every pattern citing it.
                score_threshold=_precision_bar(used_idf),
            )

    return RelevantResult(
        patterns=patterns, episodes=episodes, query_keywords=keywords, facts=facts
    )


def _swap_replaced(
    store: Store,
    keywords: list[str],
    live: list[ScoredEpisode],
    max_episodes: int,
    *,
    until: str | None,
    mode: str,
) -> list[ScoredEpisode]:
    """The displayed episode tier, CAP-04: the live hits exactly as ranked without this
    step, merged with the keyword hits on REPLACED episodes, each of which is swapped,
    in its own slot, for the live episode at the end of its chain. A query that names
    the old state ("still in Seattle?") is served the current fact, which by
    construction shares none of its words. Replaced hits are scored with weights counted
    over every episode, replaced ones included, so a link never raises a hit's score
    (counted over the live episodes only, a word that survives only in replaced text
    would read as maximally rare); live hits keep their own weights and scores. The
    replacement takes the slot, and the score, of the best hit it stands in for (or its
    own, if it ranks higher as a live hit): the slot the old fact would have held is the
    update's. It lists every replaced hit it stands in for in ``replaces``. A hit whose path to that episode
    crosses a wrap-proposed (or delete-rewired) link is dropped, not swapped: a wrap
    model proposes those in bulk (measured ~1.4% precise on STALE), so they hide but
    never serve."""
    every, doc_freq, corpus_n = _fetch_episode_candidates(
        store, keywords, until=until, uncapped=mode == "query",
        filters={"include_superseded": True},
    )
    replaced = {i: e for i, e in every.items() if e.superseded_by}
    swap_to = store.redirectable_ids(list(replaced), until) if replaced else {}
    replaced = {i: e for i, e in replaced.items() if i in swap_to}
    if not replaced:
        return live[:max_episodes]
    weights, used_idf = _query_weights(
        store, keywords, doc_freq, until=until, corpus_n=corpus_n,
        filters={"include_superseded": True},
    )
    old_hits = _score_candidate_episodes(
        replaced, keywords, weights,
        score_threshold=_precision_bar(used_idf, mode),
        require_anchor=_anchor_floor(used_idf, mode),
        min_hits=_min_hits(mode),
        min_len=_min_episode_len(mode),
    )
    ranked = sorted([*live, *old_hits], key=lambda e: (e.score, e.timestamp, e.id), reverse=True)
    slots: list[ScoredEpisode] = []
    refs: dict[str, list[ReplacedEpisode]] = {}
    for hit in ranked:
        if hit.id not in replaced:
            if hit.id not in refs:
                refs[hit.id] = []
                slots.append(hit)
        else:
            head = swap_to[hit.id]   # the row redirectable_ids read with its choice
            head_id = head.id
            if head_id not in refs:
                refs[head_id] = []
                slots.append(ScoredEpisode(
                    id=head.id, timestamp=head.timestamp,
                    type=head.type.value if isinstance(head.type, EpisodeType) else str(head.type),
                    source=head.source or "", content=head.content or "", score=hit.score,
                ))
            refs[head_id].append(ReplacedEpisode(
                id=hit.id, timestamp=hit.timestamp, content=hit.content[:REPLACED_CONTENT_MAX]))
        if len(slots) >= max_episodes:
            break
    return [dataclasses.replace(e, replaces=tuple(refs[e.id])) if refs[e.id] else e
            for e in slots]


def retrieve_patterns(
    crystal_store: CrystalStore | None,
    query: str,
    *,
    max_patterns: int = MAX_PATTERNS,
    today: date | None = None,
    mode: RetrievalMode = "prompt",
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
        mode: ``"prompt"`` (default) or ``"query"``, with the meaning given in
            :func:`retrieve_relevant`.

    Returns:
        a scored/ranked ``list[RelevantPattern]`` (best score first, a higher
        graduation level breaking ties), capped at ``max_patterns``. Empty when
        ``crystal_store`` is ``None``, ``max_patterns <= 0``, the query has fewer
        than the mode's distinctive-keyword floor, or nothing clears the mode's
        precision gates — surface nothing rather than noise.

    Raises:
        ValueError: ``mode`` is not ``"prompt"`` or ``"query"``.
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
        does evidence-edge pattern reach (which needs the Store), so this Store-free entry is
        keyword-only by design — there is no associative path here. Two further caveats:
        a ``None`` ``crystal_store`` short-circuits to ``[]`` here without inspecting the
        query, and pass the SAME explicit ``today`` to both if comparing outputs (each
        defaults it independently, so a midnight-straddling pair can label ``activation``
        differently).
    """
    _check_mode(mode)
    today = today or date.today()
    if crystal_store is None or max_patterns <= 0:
        return []
    keywords = extract_keywords(query, mode=mode)
    if len(keywords) < _min_keywords(mode):
        return []
    weights = {kw: _keyword_weight(kw) for kw in keywords}
    return _score_patterns(
        crystal_store, keywords, weights, max_patterns=max_patterns, today=today,
        score_threshold=_precision_bar(False, mode),
        require_anchor=_anchor_floor(False, mode),
        min_hits=_min_hits(mode),
    )


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

    This is :func:`retrieve_relevant`'s episode scoring in ``"query"`` mode: the query
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
