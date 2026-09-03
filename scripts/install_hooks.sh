#!/usr/bin/env bash
# Install anneal-memory's git hooks. Run ONCE per clone:
#     bash scripts/install_hooks.sh
#
# This sets core.hooksPath, so a hook dropped into .git/hooks/ is IGNORED once
# this has run — the hooks that fire are the ones tracked in scripts/hooks/,
# which is the point: they are versioned and reviewable rather than per-machine.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
git -C "$root" config core.hooksPath scripts/hooks
chmod +x "$root"/scripts/hooks/* 2>/dev/null || true
echo "hooks installed: core.hooksPath=scripts/hooks"
echo "  pre-push -> release-stamp gate (spore-710). Bypass: git push --no-verify"
