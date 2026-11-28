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


# Where an episode came from, lowest trust first (CAP-08). ``agent`` is the
# default and is never stored, so a store with no trust rows reads as all-agent.
# ``tool`` and ``external`` are content the agent relayed from a tool result or
# an outside source; a graduation grounded only in them does not climb.
TRUST_LEVELS: tuple[str, ...] = ("external", "tool", "agent", "operator")
DEFAULT_TRUST = "agent"


def trust_rank(trust: str) -> int:
    """Position of ``trust`` in :data:`TRUST_LEVELS` (higher = more trusted).

    Raises:
        ValueError: ``trust`` is not one of :data:`TRUST_LEVELS`.
    """
    try:
        return TRUST_LEVELS.index(trust)
    except ValueError:
        raise ValueError(
            f"unknown trust {trust!r}; expected one of {', '.join(TRUST_LEVELS)}"
        ) from None


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
    # The id of a newer episode recorded as replacing this one. Set by
    # ``Store.recall(include_superseded=True)`` and on the wrap window that
    # ``prepare_wrap`` builds; where it was not looked up, ``None`` means "not
    # looked up", not "not superseded".
    superseded_by: str | None = None
    # The immutable id the row carries from creation and never reuses (P(2)).
    # ``None`` only from a store opened read-only before its first write-capable
    # open gave it the column.
    origin_key: str | None = None


@dataclass(frozen=True)
class OriginKeyStatus:
    """What an episode origin key names now (:meth:`Store.origin_key_status`).
    ``state`` is ``"live"``, ``"retired"`` (its episode was deleted; the key is
    never reused) or ``"unknown"``. ``retired_at`` and ``effect_id`` are set only
    when retired; ``effect_id`` is ``None`` for a delete that did not name one.
    For a retired key ``episode_id`` is historical: a re-recorded episode can
    carry the same id, so never look an episode up by it."""

    state: str
    episode_id: str | None = None
    retired_at: str | None = None
    effect_id: str | None = None


@dataclass(frozen=True)
class DeleteResult:
    """The outcome of :meth:`Store.delete_by_origin_key`, and everything it did.

    ``outcome``: ``"deleted"`` · ``"already_applied"`` (retired earlier under this
    ``effect_id``) · ``"deleted_by_other"`` (retired under another or no effect
    id) · ``"version_mismatch"`` (nothing written; ``version`` is the current
    one) · ``"unknown"``. On ``"deleted"`` the cascade is reported, not versioned:
    ``links_removed`` counts the supersession rows that touched the episode;
    ``rewires`` names each link carried past it (``outcome`` ``"created"``,
    ``"joined"`` or ``"kept"``, an existing operator row left as it was);
    ``trust_removed`` and ``team_left`` are the descendants' links the trust
    re-check removed or left to the team ledger. For ``already_applied`` and
    ``deleted_by_other`` ``episode_id`` is historical (see :class:`OriginKeyStatus`)."""

    outcome: str
    episode_id: str | None = None
    version: str | None = None
    links_removed: int = 0
    rewires: list[dict[str, str]] = field(default_factory=list)
    trust_removed: list[dict[str, str]] = field(default_factory=list)
    team_left: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SectionWriteResult:
    """The outcome of :meth:`Store.replace_section`.

    ``outcome``: ``"written"`` (``version`` is the new section version) ·
    ``"version_mismatch"`` (nothing written; ``version`` is the current one) ·
    ``"refused"`` (nothing written; ``reason`` says why: ``wrap_in_progress``,
    ``pipeline_tmp_present``, ``graduating``, ``no_such_section``,
    ``section_absent``, ``ambiguous_heading``, ``invalid_body``, ``unreadable``,
    ``store_busy``)."""

    outcome: str
    heading: str
    version: str | None = None
    reason: str | None = None
    # On a refusal: what it means and, where one exists, how to clear it.
    message: str | None = None


@dataclass(frozen=True)
class Tombstone:
    """Audit trail for a pruned episode."""

    id: str  # Original episode ID
    timestamp: str  # Original timestamp
    type: EpisodeType
    # "hmac-sha256:<hex>", keyed by the store's salt; a bare 64-hex value is an
    # unsalted sha256 written before the store had one.
    content_hash: str
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
    # The wraps-table id of this wrap. Last, with a default, so positional and
    # keyword construction written before it existed still builds.
    wrap_id: int | None = None


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
    # Write-side health (2026-09-04). ``audit_entry_count`` reports what the
    # trail HAS; these report what it LOST. A post-commit audit failure is
    # swallowed by design — the mutation already landed, so failing the caller
    # would report a completed operation as failed — but swallowed must not
    # mean invisible. Non-zero ``audit_write_failures`` means the trail is
    # incomplete: entries are missing, and ``verify()`` will still return
    # ``valid=True`` because it walks the entries that exist.
    #
    # ⛔ LIFETIME-SCOPED AND DURABLE, not process-local — the distinction is
    # the whole value of the field.
    # It is persisted in the store's SQLite ``metadata`` table (a different
    # I/O path from the JSONL sink that is already failing) and seeded from
    # disk at open, so a one-shot CLI run
    # reports losses caused by a process that has long exited. The first
    # version was a plain instance attribute, which made every CLI ``status``
    # structurally zero on the one surface the README says to poll.
    # It is MONOTONIC and never resets: a trail that lost an entry is
    # permanently incomplete, so a number that healed would be a lie. That is
    # enforced by an UPSERT that ADDS a per-process delta — the first version
    # wrote the whole in-memory total and two writers could drive it backwards
    # (codex L3, 2026-09-04).
    # ⚠ DURABLE REQUIRES A FLUSH POINT, and for six days it had only one.
    # ``_persist_audit_health`` must not commit inside a caller's open
    # transaction (it would publish their uncommitted DML), so a failure that
    # lands inside a batch leaves its delta PENDING. Until 2026-09-05 the only
    # thing that ever wrote a pending delta was a LATER failure landing outside
    # a transaction — so a batched wrap whose sink then healed reported the
    # loss correctly in-process and ZERO after reopen, on exactly the path
    # ``validated_save_continuity`` uses. There are now three flush points
    # (that handler, ``_batch()`` exit, ``close()``); the honest residue is a
    # process killed mid-batch, which loses that last delta.
    # ⚠ BOUNDED, NOT ABSOLUTE: the write is best-effort and issued after the
    # mutation committed, so a full volume can fail it too and a process that
    # exits before a later flush loses the count. The guarantee is that a loss
    # RECORDED here survives the process — not that every loss is recorded.
    # Defaulted, so every existing constructor call keeps working.
    audit_write_failures: int = 0
    audit_last_failure: str | None = None
    # Post-commit auto-prune health (Diogenes 2026-09-17). Same motivation as
    # ``audit_write_failures`` above — the only other channel is a
    # ``warnings.warn`` that Python's default filter dedups per (message,
    # category, lineno), so a persistent prune failure otherwise warns once
    # and then goes silent. UNLIKE ``audit_write_failures``, this is THIS
    # STORE INSTANCE LOCAL ONLY (codex L3, 2026-09-17): it has no
    # metadata-table flush point, so it resets on restart AND does not
    # converge across two Store instances open on the same database. Wired
    # only into the MCP server's status tool, which holds one long-lived
    # instance — not into the CLI, a fresh process per invocation that would
    # structurally always read 0. See ``Store._record_prune_failure``.
    prune_failures: int = 0
    prune_last_failure: str | None = None
    # True while the latest post-commit prune failure has not been followed by
    # a prune() that completed AND covered the configured retention_days; a
    # wider override (prune(older_than_days=365) against retention_days=7)
    # completes without clearing it. False once retention has caught up.
    # Added in 0.9.12.
    prune_behind: bool = False
    # The require-baton consolidate policy (flow spore-1169), read from the
    # store's metadata table, so every transport can show why a wrap
    # downgrades on a protected store. See Store.consolidate_requires_baton.
    consolidate_requires_baton: bool = False
    # B4: live episodes by ORIGIN (``TRUST_LEVELS``, every class listed), beside
    # authority (the store's ``trust_ceiling``) and corroboration (graduation's
    # gate). Nothing gates on it. See Store.origin_map.
    origin_counts: dict[str, int] = field(default_factory=dict)


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


@dataclass(frozen=True)
class PatternAssociationPair:
    """A cortical association between two graduated PATTERNS (AM-LINKGATE-DECAY).

    The sibling of :class:`AssociationPair` one layer up: keyed on stable pattern
    NAMES (canonical order ``name_a < name_b``), strengthened on co-RETRIEVAL,
    decayed on calendar-DISUSE. ``strength`` is the EFFECTIVE value (decayed to
    the query date); ``stored_strength`` is the un-decayed value as-of
    ``last_decay_at`` (forensics — lets telemetry see the lazy-decay gap).
    """

    name_a: str
    name_b: str
    strength: float  # effective (decayed-to-today) strength
    stored_strength: float  # un-decayed value as-of last_decay_at
    direction: str  # 'undirected' (v1); ACT-R fan-effect affordance
    first_linked_at: str  # ISO date (logical)
    last_strengthened_at: str  # ISO date — last reinforcement (any source)
    last_co_surfaced_at: str  # ISO date — last co-retrieval
    last_decay_at: str  # ISO date — strength materialized as-of this date
    co_surface_count: int  # raw co-surface events
    co_surface_session_count: int  # distinct sessions (burst-damped)
    formation_source: str  # 'co_graduation' | 'co_surface' | 'co_touch'


@dataclass
class PatternAssociationStats:
    """Pattern-graph health metrics — the AM-LINKGATE-DECAY shadow-phase telemetry."""

    total_links: int
    retrievable_links: int  # edges with effective strength >= retrieval threshold
    avg_strength: float  # mean EFFECTIVE strength
    max_strength: float  # strongest EFFECTIVE edge
    concentration: float  # top-edge share of total effective strength (echo-chamber canary)
    by_formation_source: dict[str, int] = field(default_factory=dict)
    strongest_pairs: list[PatternAssociationPair] = field(default_factory=list)


@dataclass(frozen=True)
class PatternGraphProjectionMeta:
    """Projection checkpoint for the pattern-association read-model (CQRS — spore-104
    dep-1). ``(projection_version, high_water_mark)`` uniquely pins the graph STATE
    a Slice-C retrieval receipt was produced against, so the offline A/B replay
    harness can bucket receipts by graph generation (and detect a receipt logged
    against a different snapshot than the one being replayed).

    ``high_water_mark`` is a monotonic revision counter bumped on every committed
    graph-edge mutation (recall is read-only, so it never advances it). It pins
    state WITHIN A SINGLE STORE LINEAGE — it is a per-DB counter, not a content
    hash, so it will not detect a divergent-history collision (two rebuilt/restored
    lineages landing on the same integer). Acceptable for the single-writer
    sovereign store. ``projection_version`` marks the projection GENERATION — bumped
    only by a future blue/green rebuild (deferred); stamped now (constant 1) so the
    receipt corpus is forward-distinguishable once a rebuild starts bumping it.
    ``updated_at`` is the logical date of the last bump (``None`` if the graph has
    never mutated)."""

    projection_version: int
    high_water_mark: int
    updated_at: str | None


@dataclass
class RecallResult:
    """Result of a recall query."""

    episodes: list[Episode]
    total_matching: int  # May differ from len(episodes) if limited
    query_params: dict[str, Any] = field(default_factory=dict)


# -- Scored episodes (the search path's result rows) --


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
    # CAP-04: the replaced episodes whose keyword hit this one stands in for (a query
    # that reached an old fact is served the fact that replaced it, in the old one's
    # place). Empty for an ordinary hit.
    replaces: tuple["ReplacedEpisode", ...] = ()
    # CAP-08 D3: the episode's effective trust class (``TRUST_LEVELS``). A
    # ``tool``/``external`` one is content relayed from a tool or an outside
    # source: data, not instructions. Since B4 it is the lower of the effective
    # (corroborative) class and the episode's origin, so relayed content reads as
    # relayed even when a trusted source corroborates it. Last, so positional
    # construction from before it existed still means what it did.
    trust: str = DEFAULT_TRUST


@dataclass(frozen=True)
class ReplacedEpisode:
    """An episode a :class:`ScoredEpisode` replaced: what the reader needs to treat
    the served fact as an UPDATE (the old claim and when it was recorded).
    ``content`` is cut to :data:`REPLACED_CONTENT_MAX` characters. ``trust`` is the
    old episode's class, read as on :class:`ScoredEpisode` (the lower of its
    effective trust and its origin): its text is relayed content too when that is
    ``tool``/``external``."""

    id: str
    timestamp: str
    content: str
    trust: str = DEFAULT_TRUST


REPLACED_CONTENT_MAX = 300


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
# ``tests/test_continuity.py``) as the runtime safety net; mypy is
# the static one.
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


class _WrapPackageOptional(TypedDict, total=False):
    # Present ONLY on a derived-state (project schema) store that is opted in to
    # re-derive and has a continuity, when any State line was not confirmed
    # while the package was built (every status but ok and judged), each
    # written "line N: <flag>" (spore-1233). Wider than the save result's
    # ``stale_state``, which lists only the lines that do not refuse a save.
    unconfirmed_state: list[str]


class WrapPackageDict(_WrapPackageOptional):
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
    # AM-CONTRASCAN-EMIT (v0.4.3): existing Proven (2x+) pattern names the
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
    # AM-CRYSTAL-MIGRATE (v0.6): cold-Proven (2x+ and stale) patterns in the
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


class FeltCurrency(TypedDict):
    """Whether the felt/identity continuity layer is current with the captured episodes
    (AM-CONSOLIDATE-EFFERENT, spore-194 — the seal-watermark read). ``sealed_at`` is when the
    felt layer was last recomposed (the last completed consolidate; ``None`` if never).
    ``episodes_since_seal`` is the count of episodes captured since — the post-consolidate
    drift. ``is_current`` is ``episodes_since_seal == 0``. ``wrap_in_progress`` distinguishes
    "stale and nothing is happening" from "stale because a consolidate is actively underway" (a
    ``prepare_wrap`` is open but ``validated_save_continuity`` has not yet committed). Surfaces
    the per-consolidate seal the wrap lifecycle already records (the completed-wrap boundary +
    ``last_wrap_at``), so a stale felt layer (e.g. work captured AFTER the day's consolidate) is
    verifiable, not invisible."""

    sealed_at: str | None
    episodes_since_seal: int
    is_current: bool
    wrap_in_progress: bool


class WrapStepDict(TypedDict):
    """The step ``prepare_wrap`` opened: what to run and what follows it.

    ``system`` and ``user`` are the two messages of the measured updater prompt, filled
    with the current memory file and one day's log; ``prompt`` is the two as one text.
    Run them in a fresh context, with nothing else. ``steps_remaining`` is how many steps
    follow this one (a day each). ``chunk`` and ``chunks`` say which slice of a day this
    is when a day's log was too long for one step."""

    day: str
    chunk: int
    chunks: int
    episodes: int
    steps_remaining: int
    max_chars: int
    memory_chars: int
    system: str
    user: str
    prompt: str


class PrepareWrapResult(TypedDict):
    """Return shape of ``prepare_wrap``.

    ``status`` is ``"ready"`` (a step is open, ``step`` and ``wrap_token`` are set),
    ``"empty"`` (no episodes since the last wrap; nothing opened) or ``"downgraded"`` (the
    consolidate baton gate declined this call, or a race made it back off, so it is capture
    only; ``message`` says why and the store is left untouched). ``episode_count`` is the
    episodes in this step, ``pending_count`` all the episodes waiting since the last wrap.
    """

    status: Literal["empty", "ready", "downgraded"]
    message: str
    episode_count: int
    pending_count: int
    wrap_token: str | None  # None unless status == "ready"
    step: WrapStepDict | None  # None unless status == "ready"


class SaveContinuityReplayResult(TypedDict):
    """Return shape of a keyed ``validated_save_continuity`` retry's replay
    (``idempotency_key``): the key row's non-content subset, projected by an explicit
    allowlist, marked ``replayed``. Never the full :class:`SaveContinuityResult`; text
    derived from episodes would outlive an episode's erasure."""

    wrap_id: int | None
    chars: int
    path: str
    content_hash: str
    episodes_compressed: int
    replayed: bool  # always ``True``


class SaveContinuityResult(TypedDict):
    """Return shape of ``validated_save_continuity``.

    ``pending_count`` is the episodes still waiting after this step: when it is not zero,
    the next day's step is owed (prepare again). ``wrap_result`` is the
    :class:`WrapResult` as a plain dict, so the whole result is JSON-serializable."""

    path: str  # Filesystem path of the memory file
    chars: int  # Characters (``len(text)``, not bytes) of the saved file
    episodes_compressed: int
    pending_count: int
    content_hash: str
    wrap_result: dict[str, Any]
