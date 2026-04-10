# DSPy Integration

anneal-memory integrates with DSPy through **BaseCallback** — 12 lifecycle hooks (6 start/end pairs) that fire during module execution. DSPy's "programming, not prompting" paradigm makes it architecturally unique among frameworks — episodes capture module I/O rather than conversational turns.

## Install

```
pip install anneal-memory dspy
```

## Integration Pattern

DSPy's callback system provides hooks at every execution layer:

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `on_module_start` | Before any module executes | Recall relevant episodes |
| `on_module_end` | After module completes | Record module output as episode |
| `on_lm_end` | After each LLM call | Record raw LLM responses |
| `on_tool_end` | After tool execution (ReAct) | Record tool results |

## Complete Example

```python
import dspy
from dspy.utils.callback import BaseCallback
from anneal_memory import Store, EpisodeType, prepare_wrap_package


store = Store("./memory.db", project_name="my-dspy-program")


class AnnealMemoryCallback(BaseCallback):
    """Callback that records DSPy module execution as episodes."""

    def on_module_start(self, call_id: str, instance, inputs: dict):
        """Before module execution — opportunity to inject memory."""
        # Could modify inputs to include memory context
        pass

    def on_module_end(self, call_id: str, outputs, exception=None):
        """After module execution — record the output as an episode."""
        if exception:
            store.record(
                f"Module error: {exception}",
                EpisodeType.TENSION,
            )
            return

        if outputs is not None:
            # DSPy outputs are Prediction objects or dicts
            content = str(outputs)[:500]
            store.record(content, EpisodeType.OBSERVATION)

    def on_lm_end(self, call_id: str, outputs, exception=None):
        """After LLM call — record raw generation."""
        if outputs and not exception:
            # Record significant LLM outputs
            pass

    def on_tool_start(self, call_id: str, instance, inputs: dict):
        pass

    def on_tool_end(self, call_id: str, outputs, exception=None):
        """After tool execution (ReAct agents) — record results."""
        if outputs and not exception:
            store.record(
                f"Tool result: {str(outputs)[:400]}",
                EpisodeType.OUTCOME,
            )


# Configure DSPy with memory callback
lm = dspy.LM("anthropic/claude-sonnet-4-20250514")
dspy.configure(lm=lm, callbacks=[AnnealMemoryCallback()])


# Define a program with memory awareness
class MemoryAwareRAG(dspy.Module):
    """RAG module that incorporates anneal-memory context."""

    def __init__(self, num_passages=3):
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate = dspy.ChainOfThought("context, question, memory -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages

        # Load memory context
        continuity = store.load_continuity()
        memory = continuity if continuity else "No prior memory."

        return self.generate(
            context=context,
            question=question,
            memory=memory,
        )


# Use the program
rag = MemoryAwareRAG()
result = rag(question="What are the trends in AI memory systems?")
print(result.answer)

# Wrap after program execution
episodes = store.episodes_since_wrap()
if episodes:
    continuity = store.load_continuity()
    package = prepare_wrap_package(episodes, continuity, store.project_name)
    # Compress via LLM
    compress_module = dspy.ChainOfThought(
        "episodes, continuity, instructions -> compressed_continuity"
    )
    compressed = compress_module(
        episodes=package["episodes"],
        continuity=package["continuity"],
        instructions=package["instructions"],
    )
    validated_save_continuity(store, compressed.compressed_continuity)

store.close()
```

## ReAct Agent with Memory

DSPy's `ReAct` module loops over tools — episodes capture the reasoning chain:

```python
# Memory recall as a DSPy tool
def recall_memory(query: str) -> str:
    """Search memory for relevant prior episodes."""
    result = store.recall(keyword=query, limit=5)
    if not result.episodes:
        return "No relevant memories found."
    return "\n".join(
        f"[{ep.type}] ({ep.timestamp[:10]}) {ep.content}"
        for ep in result.episodes
    )


react = dspy.ReAct(
    "question -> answer",
    tools=[recall_memory, ...],
    max_iters=5,
)

result = react(question="Based on our prior analysis, what should we do next?")
```

## Memory as a Module Wrapper

Since DSPy programs compose via `forward()`, you can wrap any module to add memory:

```python
class WithMemory(dspy.Module):
    """Wrapper that adds memory context to any DSPy module."""

    def __init__(self, inner: dspy.Module, store: Store):
        self.inner = inner
        self.store = store

    def forward(self, **kwargs):
        # Inject memory context
        continuity = self.store.load_continuity()
        if continuity:
            kwargs["memory"] = continuity

        # Run inner module
        result = self.inner(**kwargs)

        # Record output
        self.store.record(str(result)[:500], EpisodeType.OBSERVATION)

        return result


# Usage
base_module = dspy.ChainOfThought("context, question, memory -> answer")
memory_module = WithMemory(base_module, store)
result = memory_module(context="...", question="...")
```

## With Optimizers

DSPy's optimizers compile programs using training data. Memory adds context across optimization runs:

```python
# The callback records episodes during optimization too
optimizer = dspy.MIPROv2(metric=my_metric, auto="light")
optimized = optimizer.compile(rag, trainset=train_data)

# Wrap after optimization to capture what was learned
episodes = store.episodes_since_wrap()
if episodes:
    # Compress the optimization session's learnings
    pass

# Save the optimized program
optimized.save("optimized_rag.json")
```

## Key Considerations

- **Stateless per-call:** DSPy modules are pure functions — each `forward()` call is independent. There are no built-in sessions. anneal-memory provides the cross-call continuity that DSPy lacks.
- **Callbacks are global:** Registered via `dspy.configure(callbacks=[...])` — they fire for ALL module executions in the program. Use `call_id` to track nesting and parent-child relationships.
- **Don't mutate in callbacks:** DSPy callbacks should not modify inputs or outputs. Create copies if you need to transform data. Recording to anneal-memory is a side effect, which is fine.
- **Signatures as episode types:** DSPy's typed signatures (`question -> answer`, `context, question -> answer, confidence`) provide natural episode type mapping — inputs are context, outputs are observations/decisions.
- **The cognitive loop:** DSPy's `ChainOfThought` can be used for the compression step itself — define a signature like `episodes, continuity, instructions -> compressed_continuity` and let DSPy handle the prompting.
