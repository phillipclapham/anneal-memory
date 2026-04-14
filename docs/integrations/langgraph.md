# LangGraph / LangChain Integration

anneal-memory integrates with LangGraph through **middleware** — the recommended extension mechanism in LangChain 1.0+. Middleware hooks fire at agent and model lifecycle boundaries, giving you full control over when to record, recall, and compress memory.

> **Verified:** `anneal-memory` 0.2.0 · `langchain` 1.2.15 · `langgraph` 1.1.6 · Python 3.14
> (end-to-end test: all four lifecycle hooks fire, `Store`/`record`/`recall`/`load_continuity`/`prepare_wrap`/`validated_save_continuity` all work)

## Install

```
pip install anneal-memory langchain langgraph
```

## Integration Pattern

LangGraph's `AgentMiddleware` provides four lifecycle hooks:

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `before_agent` | Once, before agent loop starts | Load continuity, inject memory context |
| `before_model` | Before each LLM call | Enrich prompt with recalled episodes |
| `after_model` | After each LLM response | Record agent decisions/observations as episodes |
| `after_agent` | Once, after agent loop ends | Trigger wrap sequence |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents import create_agent
from langgraph.runtime import Runtime


class AnnealMemoryMiddleware(AgentMiddleware):
    """Lifecycle middleware that gives LangGraph agents persistent memory."""

    def __init__(self, db_path: str, project_name: str):
        self.store = Store(db_path, project_name=project_name)

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Load continuity at session start."""
        continuity = self.store.load_continuity()
        if continuity:
            # Inject memory context into agent state
            return {"memory_context": continuity}
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Recall relevant episodes before each LLM call."""
        # Extract the latest user message for context-aware recall
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            # Search for relevant prior episodes
            result = self.store.recall(keyword=last_msg[:100], limit=5)
            if result.episodes:
                context = "\n".join(
                    f"[{ep.type}] {ep.content}" for ep in result.episodes
                )
                return {"recalled_context": context}
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Record the agent's response as an episode."""
        messages = state.get("messages", [])
        if messages:
            last = messages[-1]
            content = last.content if hasattr(last, "content") else str(last)
            if content and content.strip():
                # Determine episode type from content
                self.store.record(content[:500], EpisodeType.OBSERVATION)
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Run the wrap sequence at session end."""
        wrap = prepare_wrap(self.store)
        if wrap["status"] == "ready":
            package = wrap["package"]
            # Feed the package to your LLM. Adapt this to your runtime —
            # any LangChain chat model works. The agent's own LLM is ideal:
            # compression IS the cognition, so the model that did the work
            # should be the one reflecting on it.
            compressed = llm.invoke(
                f"Compress these episodes into continuity:\n"
                f"{package['instructions']}\n"
                f"Episodes:\n{package['episodes']}\n"
                f"Current continuity:\n{package['continuity'] or ''}"
            )
            validated_save_continuity(self.store, compressed.content)

        return None


# Usage
memory = AnnealMemoryMiddleware("./memory.db", "my-langgraph-agent")

agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[...],
    middleware=[memory],
)
```

## Using the Classic Callback System

If you're using LangGraph's `StateGraph` directly (not `create_agent`), the classic `BaseCallbackHandler` from `langchain_core` works as an observability layer:

```python
from langchain_core.callbacks.base import BaseCallbackHandler
from anneal_memory import Store, EpisodeType


class AnnealMemoryCallback(BaseCallbackHandler):
    def __init__(self, db_path: str, project_name: str):
        self.store = Store(db_path, project_name=project_name)

    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """Load memory context when graph starts."""
        pass  # Inject via state or prompt

    def on_tool_end(self, output, *, run_id, **kwargs):
        """Record tool results as episodes."""
        self.store.record(str(output)[:500], EpisodeType.OUTCOME)

    def on_chain_end(self, outputs, *, run_id, parent_run_id, **kwargs):
        """Record graph node outputs."""
        if parent_run_id is None:  # Top-level chain = full run complete
            # Trigger wrap sequence here
            pass


# Pass to graph invocation
graph.invoke(input, config={"callbacks": [AnnealMemoryCallback("./memory.db", "my-agent")]})
```

## Key Considerations

- **Middleware vs Callbacks:** Middleware can mutate state and control execution flow. Callbacks are read-only observers. Use middleware for full integration, callbacks for logging-only.
- **Store thread safety:** Each `Store` instance is safe for single-threaded use. For concurrent LangGraph nodes, create one Store per thread or use locks.
- **Compression timing:** The `after_agent` hook fires once per `create_agent` invocation — this is the natural session boundary for wraps.
- **The cognitive loop:** Don't automate compression. The wrap sequence should involve the agent's own LLM — the act of compressing IS where identity forms.
