#!/usr/bin/env python3
"""
Direct agent dispatcher for Session 14a spike.

Bypasses consult.py because gemini takes 8+ min on this corpus size and
consult.py has a 300s timeout. Each agent gets the same full_prompt built
from prompt.md + corpus_for_agents.md, concatenated in the same order
regardless of which agent is being invoked.

Usage:
    python3 run_agent.py complement
    python3 run_agent.py codex
    python3 run_agent.py gemini

Writes outputs/{agent}.txt with raw response text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PROMPT_FILE = HERE / "prompt.md"
CORPUS_FILE = HERE / "corpus_for_agents.md"
OUTPUT_DIR = HERE / "outputs"
LOG_DIR = HERE / "logs"

CLAUDE_CLI = Path.home() / ".local" / "bin" / "claude"
GEMINI_CLI = Path("/opt/homebrew/bin/gemini")
CODEX_CLI = Path("/opt/homebrew/bin/codex")

# Env vars that would cause subprocess to leak the parent CC session.
STRIP_ENV = {
    "CLAUDECODE",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
    "CLAUDE_CODE_SESSION_ACCESS_TOKEN",
}

# 20 min ceiling — gemini ran in ~9 min direct, complement/codex should be faster.
TIMEOUT_SEC = 1200


def build_full_prompt() -> str:
    prompt = PROMPT_FILE.read_text()
    corpus = CORPUS_FILE.read_text()
    return f"{prompt}\n\n---\n\n{corpus}"


def clean_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k not in STRIP_ENV and not k.startswith("CLAUDE_CODE_")
    }


def run_complement(full_prompt: str) -> tuple[str, int]:
    """Claude Code CLI in print mode with a minimal system override."""
    system = (
        "You are running a one-shot compression task. Ignore any prior role. "
        "Follow the instructions in the user message exactly. Output ONLY the "
        "three required sections with no preamble or closing remarks."
    )
    cmd = [
        str(CLAUDE_CLI),
        "--print",
        "--model", "sonnet",
        "--append-system-prompt", system,
        "--permission-mode", "bypassPermissions",
        full_prompt,
    ]
    start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=TIMEOUT_SEC, env=clean_env(), cwd="/tmp",
    )
    duration = int((time.monotonic() - start) * 1000)
    if result.returncode != 0:
        return f"[ERROR code={result.returncode}]\n{result.stderr[:1000]}", duration
    return result.stdout.strip(), duration


def run_gemini(full_prompt: str) -> tuple[str, int]:
    cmd = [str(GEMINI_CLI), "-p", full_prompt, "--output-format", "json", "--yolo"]
    start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=TIMEOUT_SEC, env=clean_env(), cwd="/tmp",
    )
    duration = int((time.monotonic() - start) * 1000)
    if result.returncode != 0:
        return f"[ERROR code={result.returncode}]\n{result.stderr[:1000]}", duration
    try:
        data = json.loads(result.stdout)
        text = data.get("response") or data.get("result") or data.get("text") or ""
        if not text:
            return f"[ERROR: no response field]\n{result.stdout[:500]}", duration
        return text, duration
    except json.JSONDecodeError:
        return result.stdout.strip(), duration


def run_codex(full_prompt: str) -> tuple[str, int]:
    output_file = f"/tmp/codex_spike_{int(time.monotonic() * 1000)}.txt"
    cmd = [str(CODEX_CLI), "exec", "--full-auto", "-o", output_file, full_prompt]
    start = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=TIMEOUT_SEC, env=clean_env(),
        cwd=str(Path.home() / "Documents" / "flow"),  # codex needs a trusted dir
    )
    duration = int((time.monotonic() - start) * 1000)
    if result.returncode != 0:
        return f"[ERROR code={result.returncode}]\n{result.stderr[:1000]}", duration
    try:
        text = Path(output_file).read_text().strip()
    except FileNotFoundError:
        text = result.stdout.strip()
    finally:
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass
    return text or "(no output)", duration


DISPATCHERS = {
    "complement": run_complement,
    "gemini": run_gemini,
    "codex": run_codex,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in DISPATCHERS:
        print(f"usage: run_agent.py {{{','.join(DISPATCHERS)}}}", file=sys.stderr)
        return 2
    agent = sys.argv[1]
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    full_prompt = build_full_prompt()
    print(f"[{agent}] prompt chars: {len(full_prompt)}")
    print(f"[{agent}] dispatching...")

    try:
        text, duration_ms = DISPATCHERS[agent](full_prompt)
    except subprocess.TimeoutExpired:
        text = f"[TIMEOUT after {TIMEOUT_SEC}s]"
        duration_ms = TIMEOUT_SEC * 1000

    output_path = OUTPUT_DIR / f"{agent}.txt"
    output_path.write_text(text)
    meta_path = LOG_DIR / f"{agent}.meta.json"
    meta_path.write_text(
        json.dumps(
            {"agent": agent, "duration_ms": duration_ms, "chars": len(text)},
            indent=2,
        )
    )
    print(f"[{agent}] done in {duration_ms/1000:.1f}s ({len(text)} chars)")
    print(f"[{agent}] output: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
