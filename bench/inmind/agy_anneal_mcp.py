"""anneal MCP server for the InMind agentic condition under agy, restricted to `recall`.

Same exposure as the Ollama agentic condition: the reader sees ONLY anneal's real `recall`
tool (its own schema and dispatch). Every recall call and its result text are appended to
$ANNEAL_BENCH_LOG as JSON lines, so the judge grades exactly what the reader saw.
"""
import json
import os
import sys

from anneal_memory import Store
from anneal_memory import server as srv

LOG = os.environ.get("ANNEAL_BENCH_LOG")
RECALL = [t for t in srv.TOOLS if t["name"] == "recall"]
# The same bounds the Ollama agentic condition puts on its reader (MAX_TOOL_ROUNDS in
# run.py): a capped number of recall calls per answer and a capped page, so a reader
# cannot page the whole store into its context. Measured 2026-10-03: unbounded, a flash
# reader made 14-19 calls with limit=100 and offsets, ~7% of a 5-hour quota per task.
MAX_CALLS = int(os.environ.get("ANNEAL_BENCH_MAX_CALLS", "6"))
MAX_LIMIT = int(os.environ.get("ANNEAL_BENCH_MAX_LIMIT", "10"))


class RecallOnly(srv.Server):
    def _handle_tools_list(self, params):
        return {"tools": RECALL}

    calls = 0

    def _handle_tools_call(self, params):
        if params.get("name") != "recall":
            return srv._tool_result("Unknown tool: only `recall` is available.", is_error=True)
        RecallOnly.calls += 1
        if RecallOnly.calls > MAX_CALLS:
            return srv._tool_result(
                f"recall call limit reached ({MAX_CALLS} per answer); answer now with what "
                "you have.", is_error=True)
        args = dict(params.get("arguments") or {})
        if args.get("offset"):
            return srv._tool_result(
                "paging is not available here; search with a more specific keyword instead.",
                is_error=True)
        try:
            limit = int(args.get("limit", MAX_LIMIT))
        except (TypeError, ValueError):
            limit = MAX_LIMIT
        args["limit"] = max(1, min(limit, MAX_LIMIT))
        params = {**params, "arguments": args}
        res = super()._handle_tools_call(params)
        if LOG:
            text = "".join(c.get("text", "") for c in res.get("content", []))
            with open(LOG, "a") as fh:
                fh.write(json.dumps({"args": params.get("arguments") or {}, "text": text}) + "\n")
        return res


def main() -> None:
    store = Store(os.environ["ANNEAL_BENCH_DB"], project_name="Assistant", audit=False)
    try:
        RecallOnly(store).run()
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
