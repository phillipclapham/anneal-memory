#!/usr/bin/env python3
"""
Hard-corpus variant of run_agent.py. Points at corpus_hard_for_agents.md,
writes outputs/{agent}_hard.txt. Shares dispatch logic with run_agent.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_agent import (  # type: ignore
    OUTPUT_DIR, LOG_DIR, run_complement, run_codex, run_gemini, TIMEOUT_SEC,
)

HERE = Path(__file__).parent
PROMPT_FILE = HERE / "prompt.md"
CORPUS_FILE = HERE / "corpus_hard_for_agents.md"

DISPATCHERS = {
    "complement": run_complement,
    "gemini": run_gemini,
    "codex": run_codex,
}


def build_full_prompt() -> str:
    return f"{PROMPT_FILE.read_text()}\n\n---\n\n{CORPUS_FILE.read_text()}"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in DISPATCHERS:
        print(f"usage: run_agent_hard.py {{{','.join(DISPATCHERS)}}}", file=sys.stderr)
        return 2
    agent = sys.argv[1]
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    full_prompt = build_full_prompt()
    print(f"[{agent}] HARD prompt chars: {len(full_prompt)}")
    print(f"[{agent}] dispatching...")

    try:
        text, duration_ms = DISPATCHERS[agent](full_prompt)
    except subprocess.TimeoutExpired:
        text = f"[TIMEOUT after {TIMEOUT_SEC}s]"
        duration_ms = TIMEOUT_SEC * 1000

    output_path = OUTPUT_DIR / f"{agent}_hard.txt"
    output_path.write_text(text)
    meta_path = LOG_DIR / f"{agent}_hard.meta.json"
    meta_path.write_text(
        json.dumps(
            {"agent": agent, "variant": "hard", "duration_ms": duration_ms,
             "chars": len(text)},
            indent=2,
        )
    )
    print(f"[{agent}] done in {duration_ms/1000:.1f}s ({len(text)} chars)")
    print(f"[{agent}] output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
