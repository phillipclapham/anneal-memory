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
            "Record a typed episode to memory. Episodes are timestamped, "
            "typed observations that accumulate during a session. They serve "
            "as raw material for compression into the continuity file."
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
            "Query episodes from memory with filters. Returns matching episodes "
            "ordered by timestamp (newest first). Use to find specific episodes "
            "for citation during graduation, or to review recent work."
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
            "Prepare a compression package for session wrap. Returns all episodes "
            "since the last wrap, the current continuity file, stale pattern warnings, "
            "and compression instructions. Marks a wrap as in-progress. Call this at "
            "session boundaries, then use the instructions to compress episodes into "
            "an updated continuity file, and save with save_continuity."
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
            "Validate and save the compressed continuity file. The text must contain "
            "exactly 4 sections: ## State, ## Patterns, ## Decisions, ## Context. "
            "Validates structure, checks graduation citations against real episodes, "
            "detects citation gaming, and may demote ungrounded graduations. "
            "Call after compressing with prepare_wrap."
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
            },
            "required": ["text"],
        },
    },
    {
        "name": "status",
        "description": (
            "Get memory health metrics. Returns episode counts (total and since "
            "last wrap), wrap history, continuity file size, episodes by type, "
            "and whether a wrap is currently in progress."
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
]


def _hash_tool(tool: dict[str, Any]) -> str:
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
        "tools": {tool["name"]: _hash_tool(tool) for tool in TOOLS},
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
        actual = _hash_tool(tool)

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
