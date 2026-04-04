"""anneal-memory: Two-layer memory for AI agents. Episodes compress into identity."""

__version__ = "0.1.6"

from .store import Store
from .types import Episode, EpisodeType, RecallResult, StoreStatus, Tombstone, WrapResult
from .audit import AuditTrail, AuditVerifyResult
from .continuity import validate_structure, prepare_wrap_package, build_engine_prompt
from .graduation import validate_graduations, check_explanation_overlap, detect_stale_patterns
from .integrity import TOOLS, RESOURCES, verify_integrity, generate_integrity_file
from .server import Server
from .engine import Engine

__all__ = [
    "Store",
    "Server",
    "Engine",
    "AuditTrail",
    "AuditVerifyResult",
    "Episode",
    "EpisodeType",
    "RecallResult",
    "StoreStatus",
    "Tombstone",
    "WrapResult",
    "TOOLS",
    "RESOURCES",
    "validate_structure",
    "prepare_wrap_package",
    "build_engine_prompt",
    "validate_graduations",
    "check_explanation_overlap",
    "detect_stale_patterns",
    "verify_integrity",
    "generate_integrity_file",
]
