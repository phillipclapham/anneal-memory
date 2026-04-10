# Anthropic Agents SDK / Claude Code Integration

anneal-memory integrates with Anthropic's ecosystem through two complementary paths: the **CLAUDE.md snippet** (proven at scale, zero code) and the **Claude Agent SDK hooks** (programmatic control).

The CLAUDE.md snippet is the most validated integration path — it's the same approach every Claude Code user relies on. The agent reads instructions from a file, then uses tools naturally.

## Install

```
pip install anneal-memory
```

For the Agent SDK: `pip install claude-agent-sdk`

## Path 1: CLAUDE.md Snippet (Recommended)

This is the simplest and most battle-tested integration. Copy the orchestration snippet into your project's `CLAUDE.md`, configure anneal-memory as an MCP server or CLI tool, and the agent follows the workflow automatically.

**For MCP (editors like Claude Code, Cursor):**
1. Add anneal-memory to your MCP config (see [README](../../README.md#mcp-server))
2. Copy [`examples/CLAUDE.md.example`](../../examples/CLAUDE.md.example) into your project's `CLAUDE.md`

**For CLI (agents with shell access):**
1. `pip install anneal-memory`
2. Copy [`examples/CLAUDE.md.cli.example`](../../examples/CLAUDE.md.cli.example) into your project's `CLAUDE.md`

The snippet teaches the agent:
- When to record episodes (during work, at meaningful moments)
- When to recall (before decisions that might have prior context)
- The full wrap sequence (prepare → compress → save) at session end
- How the immune system works (graduation, citations, decay)
- How to handle errors (missing sections, demoted patterns)

**Why this works:** The snippet teaches the cognitive WORKFLOW, not just tool descriptions. Tool descriptions tell agents WHAT each tool does. The snippet teaches WHEN and HOW to orchestrate them. This is the "skills paradigm" — instructions that make agents proactive, not just reactive.

## Path 2: Claude Agent SDK Hooks (Programmatic)

For agents built with the Claude Agent SDK (`claude-agent-sdk`), hooks provide programmatic lifecycle control:

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query
from anneal_memory import Store, EpisodeType, prepare_wrap_package

store = Store("./memory.db", project_name="my-claude-agent")


async def on_session_stop(input_data, tool_use_id, context):
    """Trigger wrap when agent finishes."""
    episodes = store.episodes_since_wrap()
    if episodes:
        continuity = store.load_continuity()
        package = prepare_wrap_package(episodes, continuity, store.project_name)
        # The agent should compress this — inject as system message
        return {
            "systemMessage": (
                f"Before you finish, compress your session memory:\n"
                f"{package['instructions']}\n"
                f"Episodes:\n{package['episodes']}\n"
                f"Current continuity:\n{package['continuity']}"
            ),
            "continue_": True,
        }
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
    """Context window pressure — good time to compress."""
    episodes = store.episodes_since_wrap()
    if len(episodes) > 10:
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
| `PreCompact` | Context pressure — prompt the agent to compress |
| `PostToolUse` | After tool execution — record episodes |
| `UserPromptSubmit` | New user input — recall relevant episodes |

### SDK Configuration Notes

- **`setting_sources=["project"]`** is required for the SDK to load `CLAUDE.md` files. Without this, the orchestration snippet is invisible.
- **`system_prompt`** can append memory context: `{"type": "preset", "preset": "claude_code", "append": continuity}`
- **Custom tools** via `@tool` decorator run in-process — anneal-memory's library functions can be exposed directly as tools.

## Key Considerations

- **Snippet vs SDK hooks:** The CLAUDE.md snippet works without any code changes — it's the right choice for most users. SDK hooks give programmatic control for custom agent frameworks built on the Claude Agent SDK.
- **Both paths preserve the cognitive loop:** The snippet teaches the agent to compress using its own judgment. The SDK hooks can inject wrap instructions but the agent still does the compression.
- **Session boundaries:** In Claude Code, each conversation is a natural session. The snippet teaches the agent to detect session-end signals ("let's wrap up", "we're done") and run the wrap sequence automatically.
