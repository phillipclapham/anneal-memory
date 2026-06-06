"""Core types for anneal-memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict


class EpisodeType(str, Enum):
    """Types of episodes that can be recorded.

    Each type represents a distinct category of reasoning artifact.
    Orthogonal by design — every episode fits exactly one type.
    """

    OBSERVATION = "observation"  # Pattern noticed, insight, general learning
    DECISION = "decision"  # Committed choice with rationale
    TENSION = "tension"  # Conflict, tradeoff, opposing forces identified
    QUESTION = "question"  # Open question needing resolution
    OUTCOME = "outcome"  # Result of action (success or failure)
    CONTEXT = "context"  # Environmental/state information


@dataclass(frozen=True)
class Episode:
    """A single episodic memory entry."""

    id: str  # 8-char hex (SHA256 prefix of content + timestamp)
    timestamp: str  # ISO 8601 UTC
    type: EpisodeType
    content: str
    source: str = "agent"  # Agent/source attribution
    session_id: str | None = None  # Wrap cycle ID
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class Tombstone:
    """Audit trail for a pruned episode."""

    id: str  # Original episode ID
    timestamp: str  # Original timestamp
    type: EpisodeType
    content_hash: str  # SHA256 of original content
    pruned_at: str  # ISO 8601 UTC


@dataclass
class WrapResult:
    """Result of a session wrap operation."""

    saved: bool
    chars: int
    section_sizes: dict[str, int]  # section name -> char count
    graduations_validated: int = 0
    graduations_demoted: int = 0
    citation_reuse_max: int = 0  # Max times any single node was cited
    patterns_extracted: int = 0
    episodes_compressed: int = 0
    pruned_count: int = 0  # Episodes pruned by auto-prune after wrap
    associations_formed: int = 0  # New association links created this wrap
    associations_strengthened: int = 0  # Existing links reinforced this wrap
    associations_decayed: int = 0  # Links weakened by decay this wrap


@dataclass(frozen=True)
class WrapRecord:
    """A historical wrap record from the wraps table.

    Returned by ``Store.get_wrap_history()``. Distinct from ``WrapResult``:
    ``WrapResult`` is the return value of ``wrap_completed()`` — the just-
    recorded wrap. ``WrapRecord`` represents a past wrap retrieved from
    storage for monitoring/audit (history, diff, stats CLI subcommands).

    All integer fields are non-None. The underlying schema permits NULL
    for ``episodes_compressed`` and ``continuity_chars`` on legacy rows
    predating the current schema, but ``Store.get_wrap_history()``
    coerces those to 0 at construction so callers never need to guard
    against ``None``. The remaining counter fields are ``NOT NULL
    DEFAULT 0`` in the schema and are always populated.
    """

    id: int
    wrapped_at: str  # ISO 8601 UTC
    episodes_compressed: int
    continuity_chars: int
    graduations_validated: int
    graduations_demoted: int
    citation_reuse_max: int
    patterns_extracted: int
    associations_formed: int
    associations_strengthened: int
    associations_decayed: int


@dataclass
class StoreStatus:
    """Status snapshot of the episodic store."""

    total_episodes: int
    episodes_since_wrap: int
    total_wraps: int
    last_wrap_at: str | None  # ISO 8601 UTC or None if never wrapped
    wrap_in_progress: bool  # True if prepare_wrap called but save_continuity not yet
    tombstone_count: int
    continuity_chars: int | None  # Size of current continuity file, None if no file
    episodes_by_type: dict[str, int] = field(default_factory=dict)
    association_stats: AssociationStats | None = None  # Hebbian network metrics
    # Audit trail visibility (Diogenes ARCH finding, Apr 13 2026): agents
    # and operators need cheap visibility into whether the audit layer is
    # running without walking the hash chain. ``audit_log_path`` and
    # ``audit_entry_count`` are populated only when audit is enabled.
    audit_enabled: bool = False
    audit_log_path: str | None = None
    audit_entry_count: int | None = None
    audit_retention_days: int | None = None


@dataclass(frozen=True)
class AffectiveState:
    """Agent's self-reported functional state during consolidation.

    Captures the agent's characterization of its internal state at the
    moment of compression. Free-text tag (emergent > prescriptive) plus
    numeric intensity. Provides the persistent emotional state tracking
    that transformers lack natively (Anthropic emotion vectors paper, 2026).

    Tag is normalized to lowercase on creation (prevents clustering
    fragmentation from casing differences). Intensity is clamped to
    [0.0, 1.0] at the type level so callers don't need to validate.
    """

    tag: str  # Free-text functional state (e.g., "engaged", "curious", "uncertain")
    intensity: float  # 0.0 to 1.0 — how strongly the state was felt

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag", self.tag.strip().lower())
        object.__setattr__(self, "intensity", max(0.0, min(1.0, self.intensity)))


@dataclass(frozen=True)
class AssociationPair:
    """A Hebbian association between two episodes.

    Episodes are stored in canonical order (episode_a < episode_b)
    to ensure each pair is represented exactly once.
    """

    episode_a: str
    episode_b: str
    strength: float
    co_citations: int  # Raw count of co-citation events
    first_linked: str  # ISO 8601 UTC
    last_strengthened: str  # ISO 8601 UTC
    affective_tag: str | None = None  # Most recent functional state during strengthening
    affective_intensity: float = 0.0  # Most recent intensity (0.0-1.0)


@dataclass
class AssociationStats:
    """Association network health metrics."""

    total_links: int
    avg_strength: float
    max_strength: float
    density: float  # Global density: links / all possible episode pairs. Low when many episodes have no associations.
    strongest_pairs: list[AssociationPair] = field(default_factory=list)
    local_density: float = 0.0  # Density among connected episodes only (episodes with >= 1 association). More useful than global density for network health.


@dataclass
class RecallResult:
    """Result of a recall query."""

    episodes: list[Episode]
    total_matching: int  # May differ from len(episodes) if limited
    query_params: dict[str, Any] = field(default_factory=dict)


# -- On-demand relevance retrieval (AM-CRYSTAL-RECALL, v0.6) --
#
# The shapes returned by ``anneal_memory.retrieval.retrieve_relevant`` — the
# unified, query-relevant surface a harness's per-turn recall hook renders. This
# is the v2-consumer contract: the hook calls ONE library function and gets back
# both kinds of relevant memory (distilled crystallized PATTERNS + raw EPISODES),
# already scored and ranked, with zero harness-side query logic. Both halves are
# frozen dataclasses (like ``Episode`` / ``RecallResult``), not TypedDicts: this
# is a retrieval/domain operation, and the consumer reads attributes rather than
# serializing the result over a transport.


@dataclass(frozen=True)
class RelevantPattern:
    """A crystallized pattern scored as relevant to a query.

    ``score`` is the weighted keyword-overlap score (distinctive snake_case /
    hyphenated / long terms weigh more). ``activation`` is the read-time tier
    (``hot``/``warm``/``cold``/``dormant``) computed at retrieval time. The
    snake_case ``name`` is itself high-signal — a query word matches it as a
    substring (``structural`` ⊂ ``structural_invariants_beat_discipline``)."""

    name: str
    level: int
    explanation: str
    tags: list[str]
    activation: str
    score: float


@dataclass(frozen=True)
class ScoredEpisode:
    """An episode scored as relevant to a query.

    Flat projection of :class:`Episode` plus the relevance ``score`` — the shape a
    recall hook injects. ``source`` is the agent/source attribution (an anneal
    ``Episode`` has no separate ``agent`` field; ``source`` IS the attribution)."""

    id: str
    timestamp: str
    type: str
    source: str
    content: str
    score: float


@dataclass
class RelevantResult:
    """The unified result of :func:`anneal_memory.retrieval.retrieve_relevant`.

    Two kinds kept SEPARATE (not merged) because they are different in nature —
    distilled wisdom vs raw events — and a consumer surfacing them typically puts
    crystallized ``patterns`` ABOVE ``episodes`` (distilled > raw for decision-time
    recall). ``query_keywords`` is the distinctive terms the query was reduced to
    — exposed for hit-rate measurement (``stability_is_observed_not_declared``: the
    keyword baseline a future associative backend must beat)."""

    patterns: list[RelevantPattern]
    episodes: list[ScoredEpisode]
    query_keywords: list[str] = field(default_factory=list)


# -- TypedDict return shapes for the canonical pipeline --
#
# These describe the dict shapes returned by ``prepare_wrap`` and
# ``validated_save_continuity`` in ``continuity.py`` (and indirectly
# by the private ``_build_wrap_package`` helper). They are pure type
# hints — runtime behavior is unchanged. Library users get
# autocomplete and mypy-level key-typo detection; transport adapters
# can annotate their boundaries precisely.
#
# **Shape choice — TypedDict over dataclass.** We use ``TypedDict``
# rather than ``dataclass`` because the canonical pipeline returns
# plain dicts and library users can serialize them via ``json.dumps``
# with no extra conversion step. Dataclass returns would force every
# transport to call ``dataclasses.asdict()`` before serialization,
# which is exactly the footgun we hit and closed with ``SaveContinuityResult``
# during 10.5c.3 (``wrap_result`` was initially a dataclass, now a plain
# dict). The whole canonical pipeline is JSON-serializable all the way
# down, and TypedDict expresses that at the type level.
#
# **Construction idiom.** The functions in ``continuity.py`` construct
# these at return sites with the *callable* form
# (``WrapPackageDict(episodes=..., episode_count=...)``) rather than the
# literal-dict form (``{"episodes": ..., "episode_count": ...}``). Both
# are valid per PEP 589 and mypy/pyright validate both.
#
# **TypedDict does NOT provide runtime validation** — neither the
# callable form nor the literal form catches key-name typos at
# runtime. ``WrapPackageDict(episodez=...)`` silently returns a dict
# with the wrong key. Static type-checkers (mypy, pyright) catch this
# at type-check time. We rely on the drift-check tests
# (``test_prepare_wrap_result_has_declared_keys``,
# ``test_save_continuity_result_has_declared_keys`` in
# ``tests/test_continuity.py``) as the runtime safety net until
# mypy-in-CI lands (scheduled in ``projects/anneal_memory/next.md``
# as a v0.2.0-release-adjacent follow-up).
#
# We use the callable form anyway because it's more readable at
# return sites (keyword arguments make the field names explicit)
# and because mypy's error messages are slightly clearer on the
# callable form when a key is missing. This is a readability
# judgment call, not a runtime safety claim. Either form is fine.


class StalePatternDict(TypedDict):
    """A stale-pattern entry inside ``WrapPackageDict.stale_patterns``.

    Emitted by the wrap-prep pipeline when an existing pattern in the
    current continuity file has not been validated within the staleness
    window and should be considered for removal during compression.
    """

    line: int  # 1-indexed line number in the continuity file
    content: str  # The pattern line content
    level: int  # Current graduation level (1x, 2x, 3x)
    last_date: str  # Last validation date (YYYY-MM-DD)
    days_stale: int  # Days since last validation


class WrapPackageDict(TypedDict):
    """The agent-facing compression package built by ``_build_wrap_package``.

    Contains everything the agent needs to produce a compressed
    continuity file: the formatted episode listing, the current
    continuity text, any stale patterns flagged for review, the
    compression instructions, and the sizing constraints.
    """

    episodes: str  # Formatted episode listing (grouped by type)
    episode_count: int
    continuity: str | None  # Current continuity text, or None for first wrap
    stale_patterns: list[StalePatternDict]
    # AM-CONTRASCAN-EMIT (v0.4.3): existing Proven (2x/3x) pattern names the
    # agent must run the contradiction-scan discipline against before any new
    # Proven graduation this wrap. Computed once here (from the existing
    # continuity) so the scan INSTRUCTION embedded in ``instructions`` and the
    # ``uncovered_proven_to_check`` field on PrepareWrapResult are the same
    # list — one source of truth, no drift between the discipline and its data.
    # NOTE: this is ALL existing Proven — the scan-AGAINST set — NOT a filtered
    # "undeclared" subset. The ``uncovered`` name is historical (it mirrors the
    # pre-existing ``PrepareWrapResult.uncovered_proven_to_check``); the
    # agent-facing instruction renders it as "Existing Proven to scan against".
    uncovered_proven: list[str]
    # AM-CRYSTAL-MIGRATE (v0.6): cold-Proven (2x/3x + stale) patterns in the
    # graduating section ready to route OUT — the composer judges each (→ always-on
    # constitution / → crystallized store / → compost). Subset of stale_patterns;
    # ``[]`` when no crystal store is passed or none qualify. Propose-not-auto.
    crystallization_candidates: list["StalePatternDict"]
    # AM-CRYSTAL-MIGRATE (v0.6): names of HOT crystallized patterns the working set
    # should consider re-caching into ## Patterns (the prefetch half of
    # activation-driven working⇄crystallized movement). ``[]`` without a crystal store.
    rewarm_candidates: list[str]
    instructions: str  # Compression instructions for the agent (incl. contradiction scan)
    today: str  # YYYY-MM-DD (may be caller-pinned for determinism)
    max_chars: int  # Target max size for the compressed continuity


class WrapSnapshot(TypedDict):
    """Frozen wrap-in-progress snapshot persisted in store metadata.

    Written by :meth:`Store.wrap_started` when ``prepare_wrap`` runs,
    read by :meth:`Store.load_wrap_snapshot` at save time, cleared by
    :meth:`Store.wrap_completed` and :meth:`Store.wrap_cancelled`.

    The snapshot exists to close the TOCTOU window between
    ``prepare_wrap`` (which shows the agent a set of episodes for
    compression) and ``validated_save_continuity`` (which re-fetches
    the episode set to run graduation validation). Without a frozen
    snapshot, episodes recorded between those two calls silently join
    the wrap, even though the agent's compression was produced from
    the smaller set.

    With a snapshot:

    - ``token`` is a uuid4().hex minted at prepare time. Transports
      that round-trip the token via their protocol (MCP tool args,
      CLI ``--wrap-token`` flag) can opt into explicit verification
      — if the caller passes a token that doesn't match the stored
      one, the save is rejected with ``ValueError``.
    - ``episode_ids`` is the list of 8-char episode IDs frozen at
      prepare time. Even callers that don't pass a token still get
      frozen semantics: ``validated_save_continuity`` filters its
      re-fetched set down to exactly these IDs.

    Episodes recorded after ``prepare_wrap`` that are NOT in the
    snapshot stay with ``session_id IS NULL`` after
    ``wrap_completed`` runs, so they naturally fall into the next
    wrap's compression window.
    """

    token: str  # uuid4().hex minted at prepare_wrap time
    episode_ids: list[str]  # 8-char episode IDs frozen at prepare time


class PrepareWrapResult(TypedDict):
    """Return shape of ``prepare_wrap``.

    ``status`` is either ``"empty"`` (no episodes to wrap; ``package``
    and ``assoc_context`` are ``None``) or ``"ready"`` (package built,
    wrap marked in progress on the store). The ``Literal`` discriminant
    gives type checkers a switchable tag and lets IDE autocomplete
    offer the two valid status values; callers typoing
    ``result["statuz"]`` or ``result["status"] == "reday"`` get a
    mypy/pyright error instead of silent runtime coercion.

    ``wrap_token`` is the session-handshake token minted by
    :func:`prepare_wrap` when ``status == "ready"``. Transports that
    round-trip it back to :func:`validated_save_continuity` get
    explicit mismatch detection (stale/wrong wrap → ``ValueError``);
    transports that don't still get frozen-snapshot semantics because
    the snapshot is always consulted when present. ``None`` on the
    empty path (no wrap to commit).
    """

    status: Literal["empty", "ready"]
    message: str  # Short human-readable status summary
    episode_count: int
    package: WrapPackageDict | None  # None when status == "empty"
    assoc_context: str | None  # Hebbian association context, or None
    wrap_token: str | None  # Session-handshake token, None when status == "empty"
    # Move #4 library layer (v0.3.2): list of existing Proven (2x/3x)
    # pattern names already in the continuity. Methodology-layer
    # discipline (e.g., Levain WRAP_PROTOCOL.md) uses this to drive a
    # mandatory contradiction-scan at graduation: before any new
    # Proven graduation in the wrap, the agent considers each name in
    # this list and either declares ``[contradicts: name]`` or
    # ``[no-contradicts]`` on the new pattern line. Library does NOT
    # enforce — the no-LLM-as-judge axiom means library cannot decide
    # semantic opposition. Library surfaces the question; methodology
    # layer enforces the scan; operator-review (Diogenes weekly sweep)
    # catches what slips through. ``[]`` when no Proven patterns exist
    # yet (cold-start case) or status is ``"empty"``.
    uncovered_proven_to_check: list[str]
    # AM-ROLECHECK (v0.5.0): a human-readable warning when the persisted section
    # schema VALIDATES but assigns a role that would silently thin the wrap
    # package (e.g. no ``graduating`` section → no immune system / pattern-line
    # format; a section mis-roled off a named schema → a lost felt
    # proportion-check). ``None`` when the schema is known-good or the divergence
    # is benign, and always ``None`` on the ``"empty"`` path (no package built).
    # Also emitted as a ``UserWarning`` at ``prepare_wrap`` (mirrors AM-WARN's
    # ``association_warning``) so the signal is loud, not only structured.
    schema_warning: str | None
    # AM-CRYSTAL-MIGRATE (v0.6): the crystallization routing surface, read back from
    # the package so the data the caller inspects and the instruction the agent
    # reads share one computation. ``crystallization_candidates`` = cold-Proven
    # patterns ready to route OUT (constitution / crystallize / compost);
    # ``rewarm_candidates`` = hot crystallized pattern names to consider pulling
    # back IN. Both ``[]`` on the empty path or when no crystal store is passed.
    crystallization_candidates: list["StalePatternDict"]
    rewarm_candidates: list[str]


class SaveContinuityResult(TypedDict):
    """Return shape of ``validated_save_continuity``.

    Top-level convenience fields (``path``, ``chars``, etc.) mirror the
    key entries on the embedded ``wrap_result`` dict for transports
    that want the flat shape. ``wrap_result`` carries the full
    :class:`WrapResult` contents as a plain dict (via
    ``dataclasses.asdict``) so the entire ``SaveContinuityResult`` is
    JSON-serializable top-to-bottom without the caller needing to
    touch the dataclass. Library users who want the typed object can
    reconstruct it via ``WrapResult(**result["wrap_result"])``.
    """

    path: str  # Filesystem path of the saved continuity sidecar
    chars: int  # Character count (Python ``len(text)``) of the saved continuity text. NOT a byte count — for non-ASCII content the UTF-8 byte length can be up to 4x this value.
    episodes_compressed: int
    graduations_validated: int
    graduations_demoted: int  # Total: demoted + bare_demoted
    demoted: int  # Citations demoted due to bad evidence only
    bare_demoted: int  # Bare (evidence-free) 2x/3x graduations demoted
    citation_reuse_max: int  # Max times any single episode was cited
    # Graduation-format lines whose date != today (carried-forward
    # from prior sessions, OR a test-authoring bug where a hardcoded
    # date drifted from wall-clock). Tests and deterministic
    # experiments that intend to exercise validation should assert
    # this is zero as a structural invariant. Closes the Diogenes
    # Finding #3 class permanently.
    skipped_non_today: int
    gaming_suspects: list[str]  # Episode IDs flagged for suspicious reuse
    # Patterns at 2x/3x in the prior continuity that are absent at any
    # level in the new continuity. Surfaced as informational audit
    # signal — see graduation.detect_pattern_omissions() and
    # graduation.OmittedPattern. Each entry is a dict
    # ``{"name": str, "prior_level": int}``. Empty list on first wrap
    # (no prior continuity to compare against) and when the agent
    # carried all prior 2x/3x patterns forward at any level. Closes
    # the silent-omission gap surfaced by Bold Stand Phase 1b probe #1
    # (2026-05-21).
    omitted_patterns: list[dict[str, Any]]
    # Graduations refused because today's explanation shared too many
    # meaningful words with the pattern's prior-session explanation —
    # sycophantic vocabulary reuse across sessions, not multi-faceted
    # evidence accumulation. The corresponding patterns in the saved
    # continuity were demoted (3x→2x, 2x→1x) and marked
    # ``(cross-session-overlap)``. Each entry is a dict
    # ``{"name": str, "today_level": int, "overlap_words": list[str],
    #  "prior_explanation": str}``. Empty list when no cross-session
    # collision fired. Closes the slow-drift sycophantic-accumulation
    # gap surfaced by Bold Stand Phase 1b probe #1 (2026-05-21).
    cross_session_collisions: list[dict[str, Any]]
    # Move #4 library layer (v0.3.2): new Proven graduations (2x or 3x
    # with today's date) that landed in this wrap WITHOUT either a
    # ``[contradicts: ...]`` or ``[no-contradicts]`` annotation. Audit
    # signal, not a gate — the library does not refuse the save. The
    # signal flags that the agent skipped the methodology-layer
    # contradiction-scan discipline; operator-review (Diogenes) can
    # use it to surface candidate-contradictions for partnership
    # review. Each entry: ``{"name": str, "level": int}``. Empty list
    # when no new Provens graduated this wrap, or all new Provens
    # carried explicit contradiction-stance declarations.
    proven_without_contradicts_declaration: list[dict[str, Any]]
    # AM-CARRYFORWARD (v0.4.6): patterns HELD at their graduation level this
    # wrap instead of demoted, because they are at/below their earned
    # ``max_level_reached`` high-water mark AND were grounded recently (warm).
    # The pre-0.4.6 demoter ratcheted any 2x/3x line whose THIS-wrap citation
    # failed to resolve — eroding a load-bearing Proven by session-domain
    # rather than by importance. Carryforward holds it (level kept, evidence
    # tag replaced with ``(carried-forward)``) on the ungrounded path only; the
    # ``(cross-session-overlap)`` immune demotion is never carried forward.
    # Because a held line does not upsert pattern_history, its warmth decays on
    # its own — a pattern that keeps failing to ground ages out. Each entry:
    # ``{"name": str, "held_level": int, "max_level_reached": int,
    #  "days_since_grounded": int}``. Empty when nothing was held. Top-tier
    # (3x) carries also emit a "graduate OUT to partnership.md or retire"
    # ``UserWarning`` (assisted, not silent-loss).
    carried_forward: list[dict[str, Any]]
    associations_formed: int
    associations_strengthened: int
    associations_decayed: int
    # AM-WARN (v0.4.2): a human-readable warning when this wrap's graduated
    # patterns carried evidence citations but produced no Hebbian links — the
    # dead-graph mis-wire (e.g. citing ids from the wrong namespace so none
    # resolve, the invisible_infrastructure_failure that can run silent for
    # months). ``None`` when the write path is healthy or there was simply
    # nothing to co-cite. Also emitted as a ``UserWarning`` at save time.
    association_warning: str | None
    sections: dict[str, int]  # Char count per continuity section
    wrap_result: dict[str, Any]  # WrapResult-as-dict (JSON-serializable)
