# smolagents (HuggingFace) Integration

anneal-memory integrates with smolagents through **step_callbacks** — per-step-type callback registration that fires after each agent action or planning step. Simple, clean, and covers both CodeAgent and ToolCallingAgent.

## Install

```
pip install anneal-memory smolagents
```

## Integration Pattern

smolagents provides callbacks per memory step type:

| Step Type | Fires | anneal-memory use |
|-----------|-------|-------------------|
| `ActionStep` | After each tool use / code execution | Record tool results, observations |
| `PlanningStep` | After planning phase | Record agent reasoning |

Session boundaries are controlled by `run(reset=True/False)`.

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from smolagents import CodeAgent, ActionStep, PlanningStep, InferenceClientModel


store = Store("./memory.db", project_name="my-smolagent")


def record_action(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Record each action step as an episode."""
    if memory_step.observations:
        store.record(
            memory_step.observations[:500],
            EpisodeType.OUTCOME,
            source=agent.name or "agent",
        )
    if memory_step.error:
        store.record(
            f"Error: {memory_step.error}",
            EpisodeType.TENSION,
            source=agent.name or "agent",
        )


def record_plan(memory_step: PlanningStep, agent: CodeAgent) -> None:
    """Record planning steps as episodes."""
    if memory_step.plan:
        store.record(
            memory_step.plan[:500],
            EpisodeType.DECISION,
            source=agent.name or "agent",
        )


# Load continuity for the agent's system prompt
continuity = store.load_continuity()
instructions = "You are a research assistant."
if continuity:
    instructions += f"\n\nYour memory from prior sessions:\n{continuity}"

# Create agent with memory callbacks
agent = CodeAgent(
    tools=[...],
    model=InferenceClientModel("meta-llama/Llama-3.3-70B-Instruct"),
    instructions=instructions,
    step_callbacks={
        ActionStep: record_action,
        PlanningStep: record_plan,
    },
)

# Run
result = agent.run("Analyze the latest AI memory research papers")

# Wrap after run — reuse the agent's own model for compression so the
# model that did the work reflects on it:
wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    package = wrap["package"]
    prompt = (
        f"{package['instructions']}\n\n"
        f"Episodes:\n{package['episodes']}\n\n"
        f"Current continuity:\n{package['continuity'] or ''}"
    )
    compressed = agent.model([{"role": "user", "content": prompt}])
    validated_save_continuity(store, compressed.content)

store.close()
```

## Accessing Full Step Data

The `ActionStep` dataclass contains rich data for episode recording:

```python
def detailed_recording(step: ActionStep, agent: CodeAgent) -> None:
    """Extract detailed episode data from action steps."""
    # Tool calls made during this step
    if step.tool_calls:
        for tc in step.tool_calls:
            store.record(
                f"Called {tc.name}({tc.arguments})",
                EpisodeType.OBSERVATION,
            )

    # Code the agent wrote (CodeAgent only)
    if step.code_action:
        store.record(
            f"Code: {step.code_action[:300]}",
            EpisodeType.DECISION,
        )

    # Token usage for tracking
    if step.token_usage:
        # Available: input_tokens, output_tokens
        pass

    # Timing information
    if step.timing:
        duration = step.timing.end_time - step.timing.start_time
        # Track session length for wrap decisions
```

## Multi-Session Continuity

smolagents' `reset` parameter controls session boundaries:

```python
# First session
result1 = agent.run("Research phase 1", reset=True)  # Clears memory
# record + wrap

# Second session — agent starts fresh but anneal-memory remembers
continuity = store.load_continuity()
agent.instructions = f"You are a research assistant.\n\nPrior memory:\n{continuity}"
result2 = agent.run("Research phase 2", reset=True)
# record + wrap
```

## Key Considerations

- **No session-end hook:** smolagents doesn't have an explicit session-end callback. Wrap after `run()` returns — this is the natural session boundary.
- **Step callbacks are synchronous:** smolagents uses a generator-based streaming model, not async. Your callbacks run synchronously.
- **ActionStep vs FinalAnswerStep:** `FinalAnswerStep` callbacks currently don't fire ([smolagents#1879](https://github.com/huggingface/smolagents/issues/1879)). Use `ActionStep` where `is_final_answer=True` to detect the final step.
- **Managed agents:** smolagents supports hierarchical agents via `managed_agents`. Each sub-agent can have its own callbacks — useful for per-agent episode recording with different `source` values.
- **The cognitive loop:** The step callbacks handle recording automatically. Compression must be done after `run()` returns, using your agent's LLM for the actual compression work.
