"""anneal-memory: Living memory for AI agents. Episodes compress into identity."""

__version__ = "0.1.9"

from .store import Store
from .types import (
    AffectiveState,
    AssociationPair,
    AssociationStats,
    Episode,
    EpisodeType,
    RecallResult,
    StoreStatus,
    Tombstone,
    WrapResult,
)
from .audit import AuditTrail, AuditVerifyResult
from .continuity import validate_structure, prepare_wrap_package, validated_save_continuity
from .graduation import (
    validate_graduations,
    check_explanation_overlap,
    detect_stale_patterns,
    extract_session_co_citations,
)
from .integrity import TOOLS, RESOURCES, verify_integrity, generate_integrity_file
from .server import Server

__all__ = [
    "Store",
    "Server",
    "AuditTrail",
    "AuditVerifyResult",
    "AffectiveState",
    "AssociationPair",
    "AssociationStats",
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
    "validated_save_continuity",
    "validate_graduations",
    "check_explanation_overlap",
    "detect_stale_patterns",
    "extract_session_co_citations",
    "verify_integrity",
    "generate_integrity_file",
]
