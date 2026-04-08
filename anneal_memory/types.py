"""Core types for anneal-memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    continuity_text: str | None = None  # The compressed continuity text (Engine only)
    associations_formed: int = 0  # New association links created this wrap
    associations_strengthened: int = 0  # Existing links reinforced this wrap
    associations_decayed: int = 0  # Links weakened by decay this wrap


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
