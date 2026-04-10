# Haystack (deepset) Integration

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
from anneal_memory import Store, EpisodeType, prepare_wrap_package
from haystack import tracing
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from contextlib import contextmanager


store = Store("./memory.db", project_name="my-haystack-agent")


class AnnealMemoryTracer:
    """Tracer that records episodes from Haystack component execution."""

    @contextmanager
    def trace(self, operation_name, tags=None):
        """Wrap component execution in a span that records episodes."""
        span = AnnealMemorySpan(operation_name, tags or {})
        try:
            yield span
        finally:
            # Record episode from span data
            if span.output_data:
                content = f"{operation_name}: {str(span.output_data)[:400]}"
                episode_type = EpisodeType.OBSERVATION
                if "agent" in operation_name.lower():
                    episode_type = EpisodeType.DECISION
                elif "tool" in operation_name.lower():
                    episode_type = EpisodeType.OUTCOME
                store.record(content, episode_type)


class AnnealMemorySpan:
    """Span that captures component I/O."""

    def __init__(self, name, tags):
        self.name = name
        self.tags = tags
        self.output_data = None

    def set_tag(self, key, value):
        self.tags[key] = value

    def set_content_tag(self, key, value):
        self.output_data = value


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

# Wrap after run
episodes = store.episodes_since_wrap()
if episodes:
    continuity_text = store.load_continuity()
    package = prepare_wrap_package(episodes, continuity_text, store.project_name)
    # Compress via LLM and save

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

Haystack's breakpoint system captures full pipeline state — useful for recording decision points:

```python
from haystack.dataclasses.breakpoints import AgentBreakpoint

# Break after 3 agent steps
result = agent.run(
    messages=[ChatMessage.from_user("Complex task")],
    break_point=AgentBreakpoint(break_after_agent_step=3),
    snapshot_callback=lambda snapshot: store.record(
        f"State at step 3: {str(snapshot)[:500]}",
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
