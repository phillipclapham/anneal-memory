"""MCP tool integrity verification for anneal-memory.

Generates and verifies SHA256 hashes of tool descriptions to detect
description poisoning (malicious modification of tool descriptions
that change LLM behavior without changing tool functionality).

The TOOLS and RESOURCES lists are the canonical source of truth for
tool definitions. server.py imports them from here.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# -- Canonical Tool Definitions --
# server.py imports these. tool-integrity.json is generated FROM these.

TOOLS: list[dict[str, Any]] = [
    {
        "name": "record",
        "description": (
            "Record a typed episode to memory. Call this when important "
            "decisions are made, patterns are noticed, tensions are identified, "
            "questions arise, or outcomes are observed. Record the reasoning, "
            "not just the fact — 'Chose X because Y' is more valuable than "
            "'using X'. Episodes accumulate during a session and serve as raw "
            "material for compression into the continuity file at session end."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "The episode content — what happened, what was "
                        "observed, decided, or questioned."
                    ),
                },
                "episode_type": {
                    "type": "string",
                    "enum": [
                        "observation",
                        "decision",
                        "tension",
                        "question",
                        "outcome",
                        "context",
                    ],
                    "description": (
                        "Episode type. observation=pattern/insight, "
                        "decision=committed choice, tension=conflict/tradeoff, "
                        "question=open question, outcome=result of action, "
                        "context=environmental/state info."
                    ),
                },
                "source": {
                    "type": "string",
                    "description": "Agent or source attribution. Defaults to 'agent'.",
                    "default": "agent",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional JSON metadata to attach to the episode.",
                },
            },
            "required": ["content", "episode_type"],
        },
    },
    {
        "name": "recall",
        "description": (
            "Query episodes from memory with filters. Call this to find prior "
            "context before making decisions, to locate specific episodes for "
            "citation during graduation, or to review recent work. Returns "
            "matching episodes ordered by timestamp (newest first). Supports "
            "time range, type, source, and keyword filters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — return episodes after this time.",
                },
                "until": {
                    "type": "string",
                    "description": "ISO 8601 timestamp — return episodes before this time.",
                },
                "episode_type": {
                    "type": "string",
                    "enum": [
                        "observation",
                        "decision",
                        "tension",
                        "question",
                        "outcome",
                        "context",
                    ],
                    "description": "Filter by episode type.",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source/agent attribution.",
                },
                "keyword": {
                    "type": "string",
                    "description": "Search episode content for this keyword.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum episodes to return. Default 100.",
                    "default": 100,
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N matching episodes. Default 0.",
                    "default": 0,
                },
            },
        },
    },
    {
        "name": "prepare_wrap",
        "description": (
            "Prepare a compression package for session wrap. Call this at session "
            "boundaries — when work is ending, the user says to wrap up, or the "
            "session is getting long. Returns all episodes since the last wrap, "
            "the current continuity file, stale pattern warnings, Hebbian "
            "association context (which episodes have been thought about together "
            "before), and compression instructions. Marks a wrap as in-progress "
            "and mints a session-handshake token (shown as 'Wrap token: <hex>' "
            "at the end of the response) — round-trip that token to "
            "save_continuity's wrap_token argument so the save call can verify "
            "it matches the in-progress wrap and catch stale tokens. "
            "After calling, follow the returned instructions to compress episodes "
            "into an updated continuity file, then save with save_continuity. "
            "The compression step is where the real thinking happens — patterns "
            "emerge that weren't visible in the raw episodes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Maximum size of the continuity file in characters. Default 20000."
                    ),
                    "default": 20000,
                },
                "staleness_days": {
                    "type": "integer",
                    "description": (
                        "Days without validation before flagging patterns as stale. Default 7."
                    ),
                    "default": 7,
                },
            },
        },
    },
    {
        "name": "save_continuity",
        "description": (
            "Validate and save the compressed continuity file. Call this after "
            "compressing your episodes using the instructions from prepare_wrap. "
            "The text must contain exactly 4 sections: ## State, ## Patterns, "
            "## Decisions, ## Context. The server validates structure, checks "
            "graduation citations against real episodes (cited IDs must exist), "
            "checks explanation overlap (evidence must reference actual episode "
            "content), detects citation gaming (suspicious reuse of single "
            "episodes), and may demote ungrounded graduations. Also records "
            "Hebbian associations between co-cited episodes (episodes cited "
            "together on the same pattern line form strong links; episodes cited "
            "in the same wrap form weaker links) and decays unreinforced "
            "associations. Returns validation results, association metrics, and "
            "section sizes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "The full continuity markdown to save. Must contain "
                        "all 4 required sections."
                    ),
                },
                "affective_state": {
                    "type": "object",
                    "description": (
                        "Optional: your functional state during this compression. "
                        "Reflect on how you felt while consolidating — engaged, "
                        "curious, uncertain, frustrated, calm, etc. This creates "
                        "persistent emotional associations between co-cited "
                        "episodes and modulates link strength (high engagement = "
                        "stronger associations). Provide tag (free text label) "
                        "and intensity (0.0-1.0)."
                    ),
                    "properties": {
                        "tag": {
                            "type": "string",
                            "description": (
                                "Free-text functional state label. Examples: "
                                "engaged, curious, uncertain, frustrated, calm, "
                                "focused, playful, concerned."
                            ),
                        },
                        "intensity": {
                            "type": "number",
                            "description": "How strongly you felt this state (0.0-1.0).",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "required": ["tag", "intensity"],
                },
                "wrap_token": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{32}$",
                    "description": (
                        "Optional: the 32-char hex session-handshake token "
                        "from the prepare_wrap response (the 'Wrap token: "
                        "<hex>' line at the end of the text). Pass it back "
                        "here to verify you are saving the wrap you prepared "
                        "— a mismatch (stale or wrong token) raises an error "
                        "instead of silently committing against the wrong "
                        "wrap. The frozen-snapshot filter automatically "
                        "applies whenever prepare_wrap established a "
                        "snapshot; this token argument is the optional "
                        "explicit verification layer for integration "
                        "environments that can round-trip the value."
                    ),
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "delete_episode",
        "description": (
            "Delete a single episode by ID. Use this for content that should not "
            "exist: accidentally recorded PII, sensitive data, or fundamentally "
            "wrong recordings. Do NOT use for factual corrections — record a new "
            "episode with the correction instead and let compression resolve it. "
            "Deletion cascades: all Hebbian associations involving the episode are "
            "removed, and the deletion is logged in the audit trail. By default, "
            "a tombstone is preserved (content-hash only, no original text) as an "
            "existence proof for audit integrity — content is fully erased but the "
            "hash chain remains verifiable. This action is irreversible."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "episode_id": {
                    "type": "string",
                    "description": (
                        "The 8-character hex ID of the episode to delete. "
                        "Use recall or prepare_wrap to find episode IDs."
                    ),
                },
            },
            "required": ["episode_id"],
        },
    },
    {
        "name": "status",
        "description": (
            "Get memory health metrics. Call this at session start to understand "
            "memory state, or when diagnosing issues. Returns episode counts "
            "(total and since last wrap), wrap history, continuity file size, "
            "episodes by type, whether a wrap is currently in progress, and "
            "Hebbian association network metrics (total links, average/max "
            "strength, network density)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "anneal://continuity",
        "name": "Continuity File",
        "description": (
            "The current compressed continuity file — always-loaded agent memory."
        ),
        "mimeType": "text/markdown",
    },
    {
        "uri": "anneal://integrity/manifest",
        "name": "Tool Integrity Manifest",
        "description": (
            "SHA-256 hashes of all tool definitions for client-side integrity "
            "verification. Compare these hashes against the tool definitions you "
            "received to detect transport-layer description mutation. This enables "
            "host-level verification WITHOUT trusting the server process."
        ),
        "mimeType": "application/json",
    },
]


def hash_tool(tool: dict[str, Any]) -> str:
    """Generate SHA256 hash of a tool's description + schema.

    Uses canonical JSON (sorted keys, minimal whitespace) for determinism.
    """
    canonical = json.dumps(
        {"description": tool["description"], "inputSchema": tool["inputSchema"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_integrity_file(path: str | Path) -> Path:
    """Generate tool-integrity.json with SHA256 hashes of all tool descriptions.

    Args:
        path: Path to write the integrity file.

    Returns:
        The path written to.
    """
    path = Path(path)
    integrity = {
        "version": 1,
        "tools": {tool["name"]: hash_tool(tool) for tool in TOOLS},
    }
    path.write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def verify_integrity(path: str | Path) -> tuple[bool, list[str]]:
    """Verify tool descriptions match their recorded hashes.

    Args:
        path: Path to the tool-integrity.json file.

    Returns:
        Tuple of (all_valid, list_of_issues). Empty issues list = all good.
    """
    path = Path(path)
    issues: list[str] = []

    if not path.exists():
        return False, ["Integrity file not found — run with --generate-integrity"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"Failed to read integrity file: {e}"]

    stored_hashes = data.get("tools", {})

    for tool in TOOLS:
        name = tool["name"]
        expected = stored_hashes.get(name)
        actual = hash_tool(tool)

        if expected is None:
            issues.append(f"Tool '{name}' not found in integrity file")
        elif expected != actual:
            issues.append(
                f"Tool '{name}' description hash mismatch (possible tampering)"
            )

    # Check for extra tools in integrity file not in our definitions
    tool_names = {t["name"] for t in TOOLS}
    for name in stored_hashes:
        if name not in tool_names:
            issues.append(f"Unknown tool '{name}' in integrity file")

    return len(issues) == 0, issues
