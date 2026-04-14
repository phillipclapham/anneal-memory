# OpenAI Agents SDK Integration

anneal-memory integrates with the OpenAI Agents SDK through **RunHooks** — lifecycle callbacks that fire at agent, tool, and LLM boundaries. The SDK's context object provides natural dependency injection for the Store.

> **Verified:** `anneal-memory` 0.2.0 · `openai-agents` 0.13.6 · Python 3.13
> (all imports + 5 `RunHooks` lifecycle methods + signature types + `ModelResponse.output` / `RunResult.final_output` fields verified against live classes)

## Install

```
pip install anneal-memory openai-agents
```

## Integration Pattern

The OpenAI Agents SDK provides `RunHooks` (global) and `AgentHooks` (per-agent):

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `on_agent_start` | Agent begins execution | Load continuity, recall relevant episodes |
| `on_llm_end` | After each LLM response | Record agent reasoning as episodes |
| `on_tool_end` | After each tool execution | Record tool results as episodes |
| `on_handoff` | Agent hands off to another | Record handoff, recall for new agent |
| `on_agent_end` | Agent finishes | Trigger wrap sequence |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from agents import Agent, Runner, RunHooks, RunContextWrapper
from agents import AgentHookContext, ModelResponse
from agents.tool import Tool
from dataclasses import dataclass


@dataclass
class AgentContext:
    """Dependency injection container — available in all hooks and tools."""
    user_id: str
    memory: Store


class AnnealMemoryHooks(RunHooks):
    """Lifecycle hooks that give OpenAI agents persistent memory."""

    async def on_agent_start(
        self, context: AgentHookContext[AgentContext], agent: Agent
    ) -> None:
        """Load continuity and recall relevant episodes at run start."""
        store = context.context.memory
        continuity = store.load_continuity()
        if continuity:
            # Continuity is available via the context for dynamic instructions
            pass

    async def on_llm_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent,
        response: ModelResponse,
    ) -> None:
        """Record agent responses as episodes.

        ModelResponse.output is a list[TResponseOutputItem] where each item
        is a union including ResponseOutputMessage (for assistant text).
        ResponseOutputMessage has .content which is a list of
        ResponseOutputText / ResponseOutputRefusal parts — the actual text
        lives on the text parts as .text. Two-level traversal required.
        """
        store = context.context.memory
        if not response.output:
            return
        for item in response.output:
            # item may be a tool call, reasoning item, etc. — skip
            # anything that doesn't carry content parts.
            content = getattr(item, "content", None)
            if not content:
                continue
            for part in content:
                text = getattr(part, "text", None)
                if text:
                    store.record(
                        text[:500],
                        EpisodeType.OBSERVATION,
                        source=agent.name,
                    )

    async def on_tool_end(
        self,
        context: RunContextWrapper[AgentContext],
        agent: Agent,
        tool: Tool,
        result: str,
    ) -> None:
        """Record tool results as episodes.

        ``Tool`` is a Union over FunctionTool, FileSearchTool,
        WebSearchTool, ComputerTool, HostedMCPTool, ShellTool,
        ApplyPatchTool, LocalShellTool, ImageGenerationTool,
        CodeInterpreterTool, and ToolSearchTool. Only a subset
        (FunctionTool, ShellTool, ApplyPatchTool) exposes a ``.name``
        attribute — the hosted/OpenAI-side tools don't. Fall back to
        the class name so attribution works for every variant.
        """
        store = context.context.memory
        tool_name = getattr(tool, "name", type(tool).__name__)
        store.record(
            f"Tool '{tool_name}': {result[:400]}",
            EpisodeType.OUTCOME,
            source=agent.name,
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[AgentContext],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        """Record agent handoffs — identity boundary events."""
        store = context.context.memory
        store.record(
            f"Handoff from {from_agent.name} to {to_agent.name}",
            EpisodeType.CONTEXT,
            source=from_agent.name,
        )

    async def on_agent_end(
        self, context: AgentHookContext[AgentContext], agent: Agent, output
    ) -> None:
        """Run wrap sequence when agent finishes."""
        store = context.context.memory
        wrap = prepare_wrap(store)
        if wrap["status"] == "ready":
            package = wrap["package"]
            # Feed the package to a compression sub-agent. Reusing
            # Runner.run keeps the compression inside the same SDK runtime:
            compressed = await Runner.run(
                compression_agent,  # a simple Agent() you define with compression instructions
                input=f"{package['instructions']}\n\nEpisodes:\n{package['episodes']}\n\nCurrent continuity:\n{package['continuity'] or ''}",
                context=context.context,
            )
            validated_save_continuity(store, compressed.final_output)


# Dynamic instructions that incorporate memory
def memory_instructions(ctx: RunContextWrapper[AgentContext]) -> str:
    continuity = ctx.context.memory.load_continuity()
    base = "You are a helpful assistant with persistent memory."
    if continuity:
        return f"{base}\n\nYour memory from prior sessions:\n{continuity}"
    return base


# Create agent with memory
agent = Agent(
    name="assistant",
    instructions=memory_instructions,  # Dynamic — reads memory each turn
    model="gpt-4o",
    tools=[...],
)

# Run with memory hooks
store = Store("./memory.db", project_name="my-openai-agent")
ctx = AgentContext(user_id="user-123", memory=store)

result = await Runner.run(
    agent,
    input="Help me design a database schema",
    context=ctx,
    hooks=AnnealMemoryHooks(),
)
```

## Using the Tracing System

For a lighter-weight integration, use a custom `TracingProcessor` that captures spans:

```python
from agents import TracingProcessor, Trace, Span, add_trace_processor
from anneal_memory import Store, EpisodeType


class AnnealMemoryTracer(TracingProcessor):
    def __init__(self, store: Store):
        self.store = store

    def on_trace_start(self, trace: Trace) -> None:
        pass

    def on_trace_end(self, trace: Trace) -> None:
        """Entire agent run completed — trigger consolidation."""
        wrap = prepare_wrap(self.store)
        if wrap["status"] == "ready":
            # Your compression + validated_save_continuity call here.
            # See AnnealMemoryHooks.on_agent_end above for the full pattern.
            pass

    def on_span_start(self, span: Span) -> None:
        pass

    def on_span_end(self, span: Span) -> None:
        """Record span data as episodes."""
        # span.span_data contains AgentSpanData, FunctionSpanData, etc.
        pass

    def shutdown(self) -> None:
        self.store.close()

    def force_flush(self) -> None:
        pass


store = Store("./memory.db", project_name="my-agent")
add_trace_processor(AnnealMemoryTracer(store))
```

## Cross-Run Session Tracking

Use `group_id` to link multiple runs in the same conversation — this maps to anneal-memory's session concept:

```python
from agents import RunConfig

config = RunConfig(
    group_id="conversation-456",  # Links traces across runs
    workflow_name="customer_support",
)

result = await Runner.run(agent, "First message", context=ctx, run_config=config)
# Later, same conversation:
result2 = await Runner.run(agent, "Follow up", context=ctx, run_config=config)
# Wrap at conversation end, not after each run
```

## Key Considerations

- **RunHooks vs AgentHooks:** `RunHooks` observe the entire run (all agents). `AgentHooks` are per-agent — useful for agent-specific episode recording in multi-agent setups. Both can be used simultaneously.
- **Context as dependency injection:** The `context` parameter on `Runner.run()` is available in every hook, tool, and dynamic instruction callback. Put the Store there.
- **Dynamic instructions:** Use `Agent(instructions=callable)` instead of static strings — the callable receives `RunContextWrapper` and can read current continuity.
- **Session management:** The SDK's built-in `SQLiteSession` handles conversation history (short-term). anneal-memory handles compressed knowledge (long-term). They're complementary.
- **The cognitive loop:** The tracing approach is passive (recording only). The hooks approach is active (can inject memory into agent state). Use hooks for full integration.
