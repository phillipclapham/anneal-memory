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
# an outside source.
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
    section_sizes: dict[str, int]  # 3.x field; empty on a 4.0 wrap (the file has no sections)
    # The five counts below are 3.x graduation and association counts, kept so the
    # shape stays stable; a 4.0 wrap leaves them 0.
    graduations_validated: int = 0
    graduations_demoted: int = 0
    citation_reuse_max: int = 0
    patterns_extracted: int = 0
    episodes_compressed: int = 0
    pruned_count: int = 0  # Episodes pruned by auto-prune after wrap
    associations_formed: int = 0
    associations_strengthened: int = 0
    associations_decayed: int = 0
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

    The graduation, citation, pattern and association counts are 3.x fields, kept
    for old rows and always 0 on a 4.0 wrap.

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
    # The require-baton consolidate policy (the require-baton ruling), read from the
    # store's metadata table, so every transport can show why a wrap
    # downgrades on a protected store. See Store.consolidate_requires_baton.
    consolidate_requires_baton: bool = False
    # B4: live episodes by ORIGIN (``TRUST_LEVELS``, every class listed), beside
    # authority (the store's ``trust_ceiling``). Nothing gates on it. See Store.origin_map.
    origin_counts: dict[str, int] = field(default_factory=dict)


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
# ``validated_save_continuity`` in ``continuity.py``. They are pure type
# hints: runtime behavior is unchanged, and TypedDict does no runtime
# validation. We use ``TypedDict`` rather than ``dataclass`` because the
# pipeline returns plain dicts that ``json.dumps`` serializes with no
# conversion step; the whole result is JSON-serializable down to
# ``SaveContinuityResult.wrap_result``. ``tests/test_rolling_wrap.py`` pins the
# declared keys of ``PrepareWrapResult``, ``WrapStepDict`` and ``SaveContinuityResult``
# against a real result at runtime; mypy checks them statically.


class WrapSnapshot(TypedDict):
    """Frozen wrap-in-progress snapshot persisted in store metadata.

    Written by :meth:`Store.wrap_started` when ``prepare_wrap`` runs,
    read by :meth:`Store.load_wrap_snapshot` at save time, cleared by
    :meth:`Store.wrap_completed` and :meth:`Store.wrap_cancelled`.

    The snapshot exists to close the TOCTOU window between
    ``prepare_wrap`` (which shows the agent a set of episodes for
    compression) and ``validated_save_continuity`` (which re-fetches
    the episode set to mark as wrapped). Without a frozen
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
    derived from episodes would outlive an episode's erasure. ``pending_count`` is the
    episodes waiting NOW, counted at the replay (not stored with the first call): when it
    is not zero the next day's step is owed."""

    wrap_id: int | None
    chars: int
    path: str
    content_hash: str
    episodes_compressed: int
    pending_count: int
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
