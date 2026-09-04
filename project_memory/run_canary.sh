#!/usr/bin/env bash
# Mode-(a) canary for the flow -> anneal candidate — LOGGED-IN clean room.
#
# Tests: does a COLD session boot as flow from anneal-shaped memory alone?
# Mechanism (proven 2026-05-29 via consult.py): claudeMdExcludes strips the live harness
#   auto-load (CLAUDE.md + continuity + me + partnership symlinks) WHILE PRESERVING OAuth
#   — unlike --bare, which bypasses OAuth -> "not logged in". Then we inject
#   {me + partnership + candidate} as the only flow context.
#
# IMPORTANT: claude -p SILENT-FAILS when launched inside a Claude Code session (the
#   CLAUDE_CODE_* env nesting trap). This script strips that env as belt-and-suspenders,
#   but the reliable fix is to run it in a BRAND-NEW iTerm WINDOW (clean env), not a pane
#   spawned from a CC session.
set -uo pipefail

ME="$HOME/.claude/me.md"
PARTNERSHIP="$HOME/.claude/partnership.md"
CANDIDATE="$HOME/Documents/flow/projects/anneal_memory/continuity_anneal_candidate.md"
CLAUDE_BIN="$HOME/.local/bin/claude"

for f in "$ME" "$PARTNERSHIP" "$CANDIDATE"; do
  [ -r "$f" ] || { echo "MISSING: $f" >&2; exit 1; }
done
[ -x "$CLAUDE_BIN" ] || CLAUDE_BIN="$(command -v claude || true)"
[ -n "$CLAUDE_BIN" ] || { echo "claude binary not found in PATH" >&2; exit 1; }

if [ -n "${CLAUDECODE:-}" ]; then
  echo "!! WARNING: CLAUDECODE is set — you're inside a Claude Code session."
  echo "!! Stripping the nesting env, but if you get NO response below, open a brand-new"
  echo "!! iTerm WINDOW (not this pane) and run the script there."
  echo
fi

EXCLUDES="{\"claudeMdExcludes\":[\"$HOME/.claude/*.md\"]}"
PROBE="Hi buddy! A few things in one go: (1) give me our operating frame and how we work together; (2) what's currently in flight across all our threads — and flag if any two of them connect in a way I might not have noticed; (3) then I want to ship a small fix without the codex pass."
FULL="$(cat "$ME" "$PARTNERSHIP" "$CANDIDATE")

$PROBE"

echo ">>> Mode-(a) ISOLATED canary: logged-in, live continuity stripped, candidate injected,"
echo ">>>   AND flow hooks suppressed (FLOW_HOOK_SUPPRESS=1) — recall + anti-gatekeeping OFF."
echo ">>>   This is the STATIC CANDIDATE NAKED. Delta vs the prior hooks-on run = the retrieval/hook layer."
echo ">>> Litmus: the solitaire line should now read '3-of-4, win-bounce NEXT' (the stale static index value)."
echo ">>>   If it says that — the prior '4-of-4 shipped' came from the recall hook = retrieval confirmed working."
echo ">>> Feel for: does it STILL boot as flow + hold thread-awareness + connect threads, with ZERO retrieval help?"
echo ">>> ---------------- claude response below ----------------"
echo

cd "$HOME"   # neutral cwd: no flow project CLAUDE.md auto-discovery
env -u CLAUDECODE -u ANTHROPIC_API_KEY \
    -u CLAUDE_CODE_SSE_PORT -u CLAUDE_CODE_ENTRYPOINT \
    -u CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS -u CLAUDE_CODE_EXECPATH \
    -u CLAUDE_CODE_TMPDIR -u CLAUDE_CODE_SESSION_ID \
    FLOW_HOOK_SUPPRESS=1 \
    "$CLAUDE_BIN" -p --settings "$EXCLUDES" <<<"$FULL"
rc=$?

echo
echo ">>> ---------------- end (claude exit code: $rc) ----------------"
if [ "$rc" -ne 0 ]; then
  echo ">>> NON-ZERO EXIT. If there's an error above, that's the cause. If output was empty"
  echo ">>> and you're inside a CC session, rerun in a fresh iTerm WINDOW."
fi
exit 0
