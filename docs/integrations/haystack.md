# Haystack (deepset) Integration

> Verified against `haystack-ai 2.27.0` + `anneal-memory 0.2.0` (Apr 14, 2026).

anneal-memory integrates with Haystack through a custom **Tracer** implementation. Haystack's tracing system wraps every component execution in spans automatically — a custom tracer can record episodes from this data without modifying any user code.

## Install

```
pip install anneal-memory haystack-ai
```

## Integration Pattern

Haystack doesn't have traditional lifecycle hooks. Instead:

| Mechanism | Fires | anneal-memory use |
|-----------|-------|-------------------|
| Custom `Tracer` | Every component execution (spans) | Record episodes from component I/O |
| `snapshot_callback` | On breakpoints or errors | Capture state for debugging |
| Post-run extraction | After `Agent.run()` returns | Parse messages for episodes, trigger wrap |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from haystack import tracing
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from contextlib import contextmanager


store = Store("./memory.db", project_name="my-haystack-agent")


class AnnealMemoryTracer:
    """Tracer that records episodes from Haystack component execution.

    Implements the haystack.tracing.Tracer protocol:
    - ``trace(operation_name, tags=None, parent_span=None)`` — context
      manager yielding a Span.
    - ``current_span()`` — returns the active span or None.
    """

    def __init__(self):
        self._current_span: "AnnealMemorySpan | None" = None

    @contextmanager
    def trace(self, operation_name, tags=None, parent_span=None):
        """Wrap component execution in a span that records episodes."""
        span = AnnealMemorySpan(operation_name, dict(tags or {}))
        previous = self._current_span
        self._current_span = span
        try:
            yield span
        finally:
            self._current_span = previous
            # Record episode from span data
            if span.output_data:
                content = f"{operation_name}: {str(span.output_data)[:400]}"
                episode_type = EpisodeType.OBSERVATION
                if "agent" in operation_name.lower():
                    episode_type = EpisodeType.DECISION
                elif "tool" in operation_name.lower():
                    episode_type = EpisodeType.OUTCOME
                store.record(content, episode_type)

    def current_span(self):
        return self._current_span


class AnnealMemorySpan:
    """Span that captures component I/O.

    Implements the haystack.tracing.Span protocol: ``set_tag`` is the
    only abstract method, but Haystack internals also call
    ``set_tags``, ``set_content_tag``, ``raw_span``, and
    ``get_correlation_data_for_logs`` — provide stubs for all of them
    so the tracer survives every call site.
    """

    def __init__(self, name, tags):
        self.name = name
        self.tags = dict(tags)
        self.output_data = None

    def set_tag(self, key, value):
        self.tags[key] = value

    def set_tags(self, tags):
        self.tags.update(tags)

    def set_content_tag(self, key, value):
        self.output_data = value

    def raw_span(self):
        return self

    def get_correlation_data_for_logs(self):
        return {}


# Enable tracing
tracing.enable_tracing(AnnealMemoryTracer())


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


recall_tool = Tool(
    name="recall_memory",
    description="Search memory for relevant prior episodes",
    function=recall_memory,
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

# Load continuity for system prompt
continuity = store.load_continuity()
system_prompt = "You are a research assistant with persistent memory."
if continuity:
    system_prompt += f"\n\nYour memory from prior sessions:\n{continuity}"

# Create and run agent
agent = Agent(
    chat_generator=...,  # Your ChatGenerator
    tools=[recall_tool],
    system_prompt=system_prompt,
)

result = agent.run(messages=[ChatMessage.from_user("Analyze the dataset")])

# Wrap after run — reuse the agent's ChatGenerator for compression:
wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    package = wrap["package"]
    prompt_msg = ChatMessage.from_user(
        f"{package['instructions']}\n\n"
        f"Episodes:\n{package['episodes']}\n\n"
        f"Current continuity:\n{package['continuity'] or ''}"
    )
    result = agent.chat_generator.run(messages=[prompt_msg])
    compressed = result["replies"][0].text
    validated_save_continuity(store, compressed)

store.close()
```

## Post-Run Message Extraction

Haystack agents return full message history. Parse it for richer episodes:

```python
result = agent.run(messages=[ChatMessage.from_user("Design a schema")])

# Extract episodes from the conversation
for msg in result.get("messages", []):
    if msg.is_from("assistant"):
        store.record(msg.text[:500], EpisodeType.OBSERVATION)
    elif msg.is_from("tool"):
        store.record(f"Tool result: {msg.text[:400]}", EpisodeType.OUTCOME)
```

## Using Breakpoints for State Capture

Haystack's breakpoint system captures full pipeline state — useful for recording decision points. `AgentBreakpoint` is a two-field dataclass: `agent_name` (the component name of the agent in the surrounding pipeline, or any identifier you want logged) and `break_point` (a `Breakpoint` or `ToolBreakpoint`). The `snapshot_callback` is a separate keyword on `Agent.run` itself:

```python
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint

# Break after the chat_generator has been visited 3 times
result = agent.run(
    messages=[ChatMessage.from_user("Complex task")],
    break_point=AgentBreakpoint(
        agent_name="researcher",
        break_point=Breakpoint(
            component_name="chat_generator",
            visit_count=3,
        ),
    ),
    snapshot_callback=lambda snapshot: store.record(
        f"State at breakpoint: {str(snapshot)[:500]}",
        EpisodeType.CONTEXT,
    ),
)
```

## Key Considerations

- **Tracing is the primary hook:** Haystack wraps every component execution in spans. A custom tracer sees all I/O without modifying the pipeline.
- **No built-in long-term memory:** Haystack's `InMemoryChatMessageStore` is experimental and handles conversation history only. anneal-memory provides compressed knowledge, graduation, and the immune system.
- **Pipeline vs Agent:** For pipeline-based workflows (not agents), the same tracer approach works — each component execution creates a span.
- **State is ephemeral:** Haystack's `state_schema` provides key-value state within a single agent run. It doesn't persist. anneal-memory provides cross-session persistence.
- **The cognitive loop:** The tracer handles recording automatically. Compression should happen after `Agent.run()` returns, using the agent's ChatGenerator for the compression work.
