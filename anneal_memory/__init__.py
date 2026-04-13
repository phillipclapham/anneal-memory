"""anneal-memory: Living memory for AI agents. Episodes compress into identity."""

__version__ = "0.1.9"

from .store import (
    AnnealMemoryError,
    Store,
    StoreDatabaseError,
    StoreError,
    StoreOperation,
)
from .types import (
    AffectiveState,
    AssociationPair,
    AssociationStats,
    Episode,
    EpisodeType,
    PrepareWrapResult,
    RecallResult,
    SaveContinuityResult,
    StalePatternDict,
    StoreStatus,
    Tombstone,
    WrapPackageDict,
    WrapRecord,
    WrapResult,
)
from .audit import AuditTrail, AuditVerifyResult
from .continuity import (
    format_wrap_package_text,
    prepare_wrap,
    prepare_wrap_package,
    validate_structure,
    validated_save_continuity,
)
from .graduation import (
    validate_graduations,
    check_explanation_overlap,
    detect_stale_patterns,
    extract_session_co_citations,
)
from .integrity import TOOLS, RESOURCES, verify_integrity, generate_integrity_file
from .server import Server

__all__ = [
    "AnnealMemoryError",
    "Store",
    "StoreDatabaseError",
    "StoreError",
    "StoreOperation",
    "Server",
    "AuditTrail",
    "AuditVerifyResult",
    "AffectiveState",
    "AssociationPair",
    "AssociationStats",
    "Episode",
    "EpisodeType",
    "PrepareWrapResult",
    "RecallResult",
    "SaveContinuityResult",
    "StalePatternDict",
    "StoreStatus",
    "Tombstone",
    "WrapPackageDict",
    "WrapRecord",
    "WrapResult",
    "TOOLS",
    "RESOURCES",
    "validate_structure",
    "prepare_wrap",
    "prepare_wrap_package",
    "format_wrap_package_text",
    "validated_save_continuity",
    "validate_graduations",
    "check_explanation_overlap",
    "detect_stale_patterns",
    "extract_session_co_citations",
    "verify_integrity",
    "generate_integrity_file",
]
