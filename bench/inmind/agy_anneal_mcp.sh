#!/bin/sh
# anneal MCP server for the InMind agentic condition under agy (Antigravity CLI).
# agy's MCP config is global, so this entry serves a store ONLY when the bench sets
# ANNEAL_BENCH_DB for that agy process; any other agy session gets a server that exits.
[ -n "$ANNEAL_BENCH_DB" ] || exit 0
exec "${ANNEAL_BENCH_PY:-python3}" "$(dirname "$0")/agy_anneal_mcp.py"
