# LlamaIndex Integration

> Verified against `llama-index-core 0.14.20` + `anneal-memory 0.2.0` (Apr 14, 2026).

anneal-memory integrates with LlamaIndex through the **instrumentation system** — event handlers and span handlers that wrap every component execution. LlamaIndex also offers a `BaseMemoryBlock` interface for deeper memory pipeline integration.

## Install

```
pip install anneal-memory llama-index-core
```

## Integration Pattern

LlamaIndex's instrumentation system (replaces the legacy callback system) provides:

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `BaseEventHandler.handle()` | On every event (LLM, tool, agent) | Record episodes from events |
| `BaseSpanHandler.prepare_to_exit_span()` | When a span completes | Record episodes from span data |
| `finalize()` on agent | After agent run completes | Trigger wrap sequence |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.tools import FunctionTool


store = Store("./memory.db", project_name="my-llamaindex-agent")


class AnnealMemoryHandler(BaseEventHandler):
    """Event handler that records episodes from LlamaIndex events."""

    @classmethod
    def class_name(cls) -> str:
        return "AnnealMemoryHandler"

    def handle(self, event: BaseEvent, **kwargs) -> None:
        if isinstance(event, LLMChatEndEvent):
            # Record LLM responses as episodes
            response = event.response
            if hasattr(response, "message") and response.message.content:
                store.record(
                    response.message.content[:500],
                    EpisodeType.OBSERVATION,
                )


# Register on the root dispatcher
dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(AnnealMemoryHandler())


# Memory recall as a tool
def recall_memory(query: str) -> str:
    """Search memory for relevant prior episodes."""
    result = store.recall(keyword=query, limit=5)
    if not result.episodes:
        return "No relevant memories found."
    return "\n".join(
        f"[{ep.type}] ({ep.timestamp[:10]}) {ep.content}"
        for ep in result.episodes
    )


recall_tool = FunctionTool.from_defaults(fn=recall_memory)

# Load continuity for agent instructions
continuity = store.load_continuity()
system_prompt = "You are a research assistant with persistent memory."
if continuity:
    system_prompt += f"\n\nYour memory from prior sessions:\n{continuity}"

# Create agent
agent = FunctionAgent(
    name="researcher",
    system_prompt=system_prompt,
    tools=[recall_tool, ...],
    llm=...,
)

# Run
result = await agent.run(user_msg="Analyze the dataset")

# Wrap after run — LlamaIndex's llm abstraction compresses the package:
wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    package = wrap["package"]
    from llama_index.core.llms import ChatMessage
    prompt = (
        f"{package['instructions']}\n\n"
        f"Episodes:\n{package['episodes']}\n\n"
        f"Current continuity:\n{package['continuity'] or ''}"
    )
    compressed = await llm.achat([ChatMessage(role="user", content=prompt)])
    validated_save_continuity(store, compressed.message.content)

store.close()
```

## Custom SpanHandler

For more granular control, use a span handler that captures the full execution tree:

```python
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler


class AnnealMemorySpanHandler(BaseSpanHandler):
    def new_span(self, id_, bound_args, instance=None, parent_span_id=None, tags=None):
        return None  # Let the default span handling work

    def prepare_to_exit_span(self, id_, bound_args, instance=None, result=None):
        """Record episodes when spans complete."""
        if result and isinstance(result, str):
            store.record(result[:500], EpisodeType.OBSERVATION)

    def prepare_to_drop_span(self, id_, bound_args, instance=None, err=None):
        """Record errors as tension episodes."""
        if err:
            store.record(f"Error: {err}", EpisodeType.TENSION)


dispatcher.add_span_handler(AnnealMemorySpanHandler())
```

## Custom MemoryBlock

For integration at the memory pipeline level, implement a `BaseMemoryBlock`:

```python
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.bridge.pydantic import Field


class AnnealMemoryBlock(BaseMemoryBlock):
    """Memory block backed by anneal-memory's episodic store."""

    # BaseMemoryBlock declares ``name`` as a required field; all three
    # framework-side subclasses must either pass it at construction
    # time or give it a default here.
    name: str = "anneal_memory"
    store_path: str = Field(description="Path to anneal-memory database")
    project_name: str = Field(default="agent")

    def _get_store(self) -> Store:
        return Store(self.store_path, project_name=self.project_name)

    async def _aget(self, messages=None, **block_kwargs):
        """Retrieve relevant memory context.

        LlamaIndex calls this with the recent chat history so the block
        can return any supplemental context the memory pipeline should
        surface. Return a list of ChatMessage objects (or a plain
        string) — LlamaIndex merges them into the prompt.
        """
        store = self._get_store()
        continuity = store.load_continuity()
        store.close()
        if continuity:
            from llama_index.core.llms import ChatMessage
            return [ChatMessage(role="system", content=f"Memory:\n{continuity}")]
        return []

    async def _aput(self, messages) -> None:
        """Record a batch of chat messages as episodes.

        LlamaIndex calls ``_aput`` with a *list* of ``ChatMessage``
        objects (the messages flushing out of short-term memory), not a
        single message. Record each one as an episode.
        """
        store = self._get_store()
        try:
            for message in messages:
                if hasattr(message, "content") and message.content:
                    store.record(message.content[:500], EpisodeType.OBSERVATION)
        finally:
            store.close()


# Use in Memory configuration
from llama_index.core.memory import Memory

memory = Memory.from_defaults(
    session_id="session-1",
    memory_blocks=[
        AnnealMemoryBlock(
            name="anneal_memory",
            store_path="./memory.db",
            project_name="my-agent",
        ),
    ],
)
```

## Key Considerations

- **Instrumentation vs legacy callbacks:** The instrumentation system (`BaseEventHandler`, `BaseSpanHandler`) is the current recommended approach. The legacy `CallbackManager` still works but is deprecated.
- **Async-first:** LlamaIndex's workflow agents are async. The instrumentation handlers receive events synchronously but can trigger async operations.
- **AgentWorkflow:** For multi-agent setups with handoffs, the `AgentWorkflow` class coordinates agents. Each agent can have its own memory context via different system prompts, while sharing the same anneal-memory store with different `source` values.
- **The cognitive loop:** The event handler records episodes automatically. Compression should happen after the agent run completes, using the agent's own LLM for judgment.
