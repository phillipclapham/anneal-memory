# Pydantic AI Integration

anneal-memory integrates with Pydantic AI through a custom **Capability** — a composable unit that bundles hooks, tools, and instructions. Pydantic AI has the richest hook system of any framework (21 hook points), and its dependency injection makes the Store naturally available everywhere.

## Install

```
pip install anneal-memory pydantic-ai
```

## Integration Pattern

Pydantic AI's `Hooks` capability provides lifecycle hooks at every level:

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `before_run` | Before agent run starts | Load continuity, set up session |
| `after_run` | After agent run completes | Trigger wrap sequence |
| `after_model_request` | After each LLM response | Record reasoning as episodes |
| `after_tool_execute` | After each tool execution | Record tool results as episodes |
| `before_model_request` | Before each LLM call | Inject recalled episodes |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap_package, validated_save_continuity
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from dataclasses import dataclass


@dataclass
class Deps:
    """Dependencies injected into every hook, tool, and instruction."""
    user_id: str
    memory: Store


# Set up hooks
hooks = Hooks()


@hooks.on.before_run
async def load_memory(ctx: RunContext[Deps]) -> None:
    """Load continuity at run start."""
    continuity = ctx.deps.memory.load_continuity()
    # Continuity is available via deps for dynamic instructions


@hooks.on.after_tool_execute
async def record_tool_result(
    ctx: RunContext[Deps], tool_call, tool_def, args, result
):
    """Record tool results as episodes."""
    ctx.deps.memory.record(
        f"Tool '{tool_def.name}': {str(result)[:400]}",
        EpisodeType.OUTCOME,
    )
    return result  # Pass through unchanged


@hooks.on.after_model_request
async def record_response(ctx: RunContext[Deps], request_ctx, response):
    """Record LLM responses as episodes."""
    # Extract text from model response
    for part in response.parts:
        if hasattr(part, "content") and part.content:
            ctx.deps.memory.record(
                part.content[:500],
                EpisodeType.OBSERVATION,
            )
    return response


@hooks.on.after_run
async def wrap_session(ctx: RunContext[Deps], result):
    """Run wrap sequence when agent finishes."""
    store = ctx.deps.memory
    episodes = store.episodes_since_wrap()
    if episodes:
        continuity = store.load_continuity()
        package = prepare_wrap_package(episodes, continuity, store.project_name)
        # Compress via LLM and save
        # compressed = await compress_with_llm(package)
        # validated_save_continuity(store, compressed)
    return result


# Memory recall as a tool
@Agent.tool
async def recall_memory(ctx: RunContext[Deps], query: str) -> str:
    """Search memory for relevant prior episodes."""
    result = ctx.deps.memory.recall(keyword=query, limit=5)
    if not result.episodes:
        return "No relevant memories found."
    return "\n".join(
        f"[{ep.type}] ({ep.timestamp[:10]}) {ep.content}"
        for ep in result.episodes
    )


# Dynamic instructions that incorporate memory
def memory_instructions(ctx: RunContext[Deps]) -> str:
    continuity = ctx.deps.memory.load_continuity()
    base = "You are a helpful assistant with persistent memory across sessions."
    if continuity:
        return f"{base}\n\nYour memory from prior sessions:\n{continuity}"
    return f"{base}\n\nThis is your first session — no prior memory yet."


# Create agent with memory capability
agent = Agent(
    "anthropic:claude-sonnet-4-20250514",
    deps_type=Deps,
    instructions=memory_instructions,
    tools=[recall_memory],
    capabilities=[hooks],
)


# Run
async def main():
    store = Store("./memory.db", project_name="my-pydantic-agent")
    deps = Deps(user_id="user-123", memory=store)

    result = await agent.run("Help me design a caching strategy", deps=deps)
    print(result.output)
    store.close()
```

## Using wrap_run for Full Control

The `wrap_run` hook gives middleware-style control over the entire run:

```python
@hooks.on.wrap_run
async def memory_middleware(ctx: RunContext[Deps], handler):
    """Full middleware — setup before, teardown after."""
    store = ctx.deps.memory

    # Before: load memory
    continuity = store.load_continuity()

    # Run the agent
    result = await handler(ctx)

    # After: wrap session
    episodes = store.episodes_since_wrap()
    if episodes:
        package = prepare_wrap_package(episodes, continuity, store.project_name)
        # Compress and save...

    return result
```

## Building an AbstractCapability

For a reusable, distributable integration:

```python
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition


class AnnealMemoryCapability(AbstractCapability):
    """Reusable capability that adds persistent memory to any Pydantic AI agent."""

    def __init__(self, db_path: str, project_name: str):
        self.store = Store(db_path, project_name=project_name)
        self._hooks = Hooks()
        self._setup_hooks()

    def _setup_hooks(self):
        @self._hooks.on.before_run
        async def load(ctx):
            pass  # Load continuity

        @self._hooks.on.after_run
        async def wrap(ctx, result):
            pass  # Wrap sequence
            return result

    # Implement AbstractCapability interface...
```

## Key Considerations

- **Dependency injection:** The `deps` parameter on `agent.run()` is the natural home for the Store. It's available in every hook, tool, and instruction callback.
- **21 hook points:** Pydantic AI's hook system is the richest available. You can intercept at run, node, model request, tool validation, and tool execution levels — each with before/after/wrap/error variants.
- **No built-in memory:** Pydantic AI deliberately leaves memory to the user. anneal-memory fills this gap cleanly through the deps + hooks pattern.
- **Message history:** Pass `message_history` to `agent.run()` for conversation continuity (short-term). anneal-memory handles compressed knowledge (long-term). They're complementary.
- **The cognitive loop:** Use the `after_run` hook to trigger compression, but the compression itself should involve an LLM — agent judgment during compression is where identity forms.
