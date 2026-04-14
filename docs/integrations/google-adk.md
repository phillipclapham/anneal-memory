# Google ADK Integration

> Verified against `google-adk 1.30.0` + `anneal-memory 0.2.0` (Apr 14, 2026).

anneal-memory integrates with Google's Agent Development Kit through **agent callbacks** and a custom **MemoryService** implementation. ADK provides 6 lifecycle callbacks on the agent constructor plus a memory abstraction layer designed for exactly this kind of integration.

## Install

```
pip install anneal-memory google-adk
```

## Integration Pattern

ADK provides callbacks at three levels — agent, model, and tool:

| Callback | Fires | anneal-memory use |
|----------|-------|-------------------|
| `before_agent_callback` | Before agent runs | Load continuity, inject into state |
| `after_agent_callback` | After agent completes | Record agent output as episode |
| `before_model_callback` | Before each LLM call | Enrich with recalled episodes |
| `after_model_callback` | After each LLM response | Record reasoning as episodes |
| `before_tool_callback` | Before tool execution | (Optional) Record tool invocations |
| `after_tool_callback` | After tool execution | Record tool results as episodes |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.adk.runners import InMemoryRunner
from google.genai import types


store = Store("./memory.db", project_name="my-adk-agent")


def load_memory(callback_context: CallbackContext):
    """Load continuity and inject into agent state."""
    continuity = store.load_continuity()
    if continuity:
        callback_context.state["memory_context"] = continuity
    return None  # Return None to proceed normally


def record_output(callback_context: CallbackContext):
    """Record agent output as an episode."""
    # Access state to see what the agent produced
    agent_name = callback_context.agent_name
    # Record observation from the interaction
    return None


def record_tool_result(tool, args: dict, tool_context: ToolContext):
    """Record tool results as episodes."""
    store.record(
        f"Tool '{tool.name}' returned with args: {str(args)[:300]}",
        EpisodeType.OUTCOME,
        source=tool_context.agent_name,
    )
    return None  # Return None to keep original result


# Create agent with memory callbacks
agent = Agent(
    name="researcher",
    model="gemini-2.0-flash",
    instruction=(
        "You are a research assistant with persistent memory. "
        "Your memory context from prior sessions: {memory_context}"
    ),
    tools=[...],
    before_agent_callback=load_memory,
    after_agent_callback=record_output,
    after_tool_callback=record_tool_result,
)

# Run
runner = InMemoryRunner(agent=agent)
async for event in runner.run_async(
    user_id="user-1", session_id="session-1", new_message=types.Content(...)
):
    print(event)
```

## Custom MemoryService

For deeper integration, implement ADK's `BaseMemoryService` — this is the designed extension point for long-term memory:

```python
from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types as genai_types
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity


class AnnealMemoryService(BaseMemoryService):
    """ADK MemoryService backed by anneal-memory."""

    def __init__(self, db_path: str, project_name: str):
        self.store = Store(db_path, project_name=project_name)

    def add_session_to_memory(self, session) -> None:
        """Ingest a completed session into long-term memory."""
        # Extract episodes from session events
        for event in session.events:
            if hasattr(event, "content") and event.content:
                content = str(event.content)[:500]
                self.store.record(content, EpisodeType.OBSERVATION)

        # Trigger wrap
        wrap = prepare_wrap(self.store)
        if wrap["status"] == "ready":
            package = wrap["package"]
            # Use a Gemini model to compress. Reusing the ADK agent's
            # own LLM (via the Runner) keeps the compression inside the
            # model that did the work — compression IS the cognition.
            compressed = compress_with_gemini(package)  # your compression function
            validated_save_continuity(self.store, compressed)

    def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Search long-term memory for relevant episodes.

        ADK calls this with keyword-only args ``app_name`` / ``user_id`` /
        ``query`` and expects a ``SearchMemoryResponse`` with a
        ``memories`` list of ``MemoryEntry`` objects. ``MemoryEntry.content``
        is a ``google.genai.types.Content`` (not a plain string), so wrap
        the episode text in a ``Content(parts=[Part(text=...)])``.
        """
        result = self.store.recall(keyword=query, limit=10)
        memories = [
            MemoryEntry(
                content=genai_types.Content(
                    parts=[genai_types.Part(text=ep.content)],
                    role="user",
                ),
                author=getattr(ep, "source", None) or "anneal-memory",
                timestamp=ep.timestamp,
                custom_metadata={"episode_type": str(ep.type)},
            )
            for ep in result.episodes
        ]
        return SearchMemoryResponse(memories=memories)
```

## Using ToolContext.search_memory

ADK tools have built-in `search_memory()` on the `ToolContext`. With a custom `MemoryService`, your tools can query anneal-memory directly:

```python
def research_tool(query: str, tool_context: ToolContext) -> dict:
    """Research tool that checks memory before searching."""
    # ToolContext.search_memory(query) takes a single positional str and
    # returns a SearchMemoryResponse. ADK fills in app_name / user_id
    # from the invocation context before delegating to your
    # MemoryService.search_memory(*, app_name, user_id, query).
    prior = tool_context.search_memory(query)
    if prior.memories:
        return {
            "from_memory": [
                {
                    "content": str(entry.content),
                    "author": entry.author,
                    "timestamp": entry.timestamp,
                }
                for entry in prior.memories
            ]
        }
    # Fall back to external search...
```

## State Scoping

ADK's state system uses key prefixes for scoping — useful for memory context:

```python
# Session-scoped (default) — current conversation only
callback_context.state["memory_context"] = continuity

# User-scoped — persists across sessions for this user
callback_context.state["user:preferences"] = prefs

# App-scoped — global across all users
callback_context.state["app:shared_knowledge"] = knowledge
```

## Key Considerations

- **Callbacks vs MemoryService:** Callbacks give per-turn control (record at each step). `MemoryService` handles session-level ingestion. Use both: callbacks for real-time recording, MemoryService for session-end consolidation.
- **State templating:** ADK's `instruction` parameter supports `{state_key}` templates — `{memory_context}` in the instruction string automatically pulls from `state["memory_context"]`.
- **Workflow agents:** `SequentialAgent`, `ParallelAgent`, and `LoopAgent` compose sub-agents. Attach callbacks to individual sub-agents for per-agent memory, or to the top-level for aggregate memory.
- **The cognitive loop:** `add_session_to_memory()` is where the wrap sequence belongs. The compression should involve an LLM — anneal-memory's `prepare_wrap(store)` provides the raw material, your LLM provides the judgment.
