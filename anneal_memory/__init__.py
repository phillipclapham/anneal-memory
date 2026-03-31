"""anneal-memory: Two-layer memory for AI agents. Episodes compress into identity."""

__version__ = "0.1.0"

from .store import Store
from .types import Episode, EpisodeType, RecallResult, StoreStatus, Tombstone, WrapResult
from .continuity import validate_structure, prepare_wrap_package, build_engine_prompt
from .graduation import validate_graduations, check_explanation_overlap, detect_stale_patterns

__all__ = [
    "Store",
    "Episode",
    "EpisodeType",
    "RecallResult",
    "StoreStatus",
    "Tombstone",
    "WrapResult",
    "validate_structure",
    "prepare_wrap_package",
    "build_engine_prompt",
    "validate_graduations",
    "check_explanation_overlap",
    "detect_stale_patterns",
]
