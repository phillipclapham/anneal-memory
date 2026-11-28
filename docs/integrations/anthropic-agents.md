# Anthropic Agents SDK / Claude Code Integration

> The hook wiring was checked against `claude-agent-sdk 0.1.59` in April 2026. The anneal-memory calls were updated for 4.0 and not re-run against the SDK.

anneal-memory integrates with Anthropic's ecosystem through two complementary paths: the **CLAUDE.md snippet** (no code) and the **Claude Agent SDK hooks** (programmatic control).

The CLAUDE.md snippet is the path most users take. The agent reads instructions from a file, then uses the tools.

## Install

```
pip install anneal-memory
```

For the Agent SDK: `pip install claude-agent-sdk`

## Path 1: CLAUDE.md Snippet (Recommended)

This is the simplest integration. Copy the agent-instructions snippet into your project's `CLAUDE.md`, configure anneal-memory as an MCP server or CLI tool, and the agent follows the workflow automatically.

**For MCP (Claude Code):**
1. Add anneal-memory to your MCP config (see [README](../../README.md#mcp-server))
2. Copy [`examples/agent-instructions.lean.example`](../../examples/agent-instructions.lean.example) into your project's `CLAUDE.md`

**For CLI (agents with shell access):**
1. `pip install anneal-memory`
2. Copy [`examples/agent-instructions.lean.cli.example`](../../examples/agent-instructions.lean.cli.example) into your project's `CLAUDE.md`

The snippet teaches the agent:
- When to record episodes (during work, at meaningful moments)
- When to recall (before decisions that might have prior context)
- The wrap sequence at session end: prepare a step, run its prompt in a fresh context (in Claude Code, a subagent given only that prompt), save the whole reply
- How to handle a refused save or a stuck wrap

**Why this works:** Tool descriptions tell an agent what each tool does. The snippet tells it when to call them and in what order, which is what makes an agent record and recall on its own instead of waiting to be asked.

## Path 2: Claude Agent SDK Hooks (Programmatic)

For agents built with the Claude Agent SDK (`claude-agent-sdk`), hooks provide programmatic lifecycle control:

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity

store = Store("./memory.db", project_name="my-claude-agent")


async def on_session_stop(input_data, tool_use_id, context):
    """Wrap when the agent finishes: one fresh-context model call per step."""
    while True:
        wrap = prepare_wrap(store)
        if wrap["status"] != "ready":
            break
        step = wrap["step"]
        # run_model_fresh is yours: one model call that receives step["system"]
        # and step["user"] and nothing from this agent's conversation (for
        # example a plain Anthropic Messages API call, or a separate
        # claude-agent-sdk query with no project settings loaded). It returns
        # the reply text.
        reply = await run_model_fresh(step["system"], step["user"])
        result = validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"])
        if result["pending_count"] == 0:
            break
    return {}


async def on_tool_use(input_data, tool_use_id, context):
    """Record tool usage as episodes."""
    tool_name = input_data.get("tool_name", "")
    if tool_name in ("Read", "Edit", "Write", "Bash"):
        # Record significant tool interactions
        store.record(
            f"Used {tool_name}: {str(input_data.get('tool_input', ''))[:300]}",
            EpisodeType.OBSERVATION,
        )
    return {}


async def on_compact(input_data, tool_use_id, context):
    """Context window pressure: a good time to suggest a wrap."""
    # store.status() is the read-only peek at store state — use this for
    # threshold checks, not store.episodes_since_wrap() (that's a
    # private-ish API; status() is the public surface).
    if store.status().episodes_since_wrap > 10:
        return {
            "systemMessage": "Context is getting long. Consider wrapping your memory.",
        }
    return {}


options = ClaudeAgentOptions(
    setting_sources=["project"],  # Required to load CLAUDE.md
    hooks={
        "Stop": [HookMatcher(hooks=[on_session_stop])],
        "PostToolUse": [HookMatcher(hooks=[on_tool_use])],
        "PreCompact": [HookMatcher(hooks=[on_compact])],
    },
)

async for message in query(
    prompt="Help me refactor the database module",
    options=options,
):
    print(message)
```

### Key SDK Hooks

| Hook | Use |
|------|-----|
| `Stop` | Session end — trigger wrap sequence |
| `PreCompact` | Context pressure: suggest a wrap |
| `PostToolUse` | After tool execution — record episodes |
| `UserPromptSubmit` | New user input — recall relevant episodes |

### SDK Configuration Notes

- **`setting_sources=["project"]`** is required for the SDK to load `CLAUDE.md` files. Without this, the orchestration snippet is invisible.
- **`system_prompt`** can append memory context: `{"type": "preset", "preset": "claude_code", "append": continuity}`
- **Custom tools** via `@tool` decorator run in-process — anneal-memory's library functions can be exposed directly as tools.

## Key Considerations

- **Snippet vs SDK hooks:** The CLAUDE.md snippet works without any code changes — it's the right choice for most users. SDK hooks give programmatic control for custom agent frameworks built on the Claude Agent SDK.
- **The wrap runs in a fresh context on both paths.** With the snippet, the agent hands the prompt to a subagent. With SDK hooks, the hook makes the model call itself, as above. Either way the wrap prompt does not run inside the agent's own conversation.
- **Session boundaries:** In Claude Code, each conversation is a natural session. The snippet teaches the agent to detect session-end signals ("let's wrap up", "we're done") and run the wrap sequence. On a store that requires the baton, the wrap consolidates only after the operator has claimed the baton for the server's session id; until then `prepare_wrap` downgrades to capture-only and says what to run.
