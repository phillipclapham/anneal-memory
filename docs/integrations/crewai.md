# CrewAI Integration

anneal-memory integrates with CrewAI through the **event bus** — a `BaseEventListener` that listens for task and crew lifecycle events. This is the cleanest integration path: non-invasive, no monkey-patching, captures everything.

## Install

```
pip install anneal-memory crewai
```

## Integration Pattern

CrewAI's event bus emits structured events at every lifecycle boundary:

| Event | Fires | anneal-memory use |
|-------|-------|-------------------|
| `CrewKickoffStartedEvent` | Crew begins | Load continuity, set up store |
| `TaskStartedEvent` | Before each task | Recall relevant episodes for task context |
| `TaskCompletedEvent` | After each task | Record task output as episode |
| `AgentExecutionCompletedEvent` | Agent finishes | Record agent-level observations |
| `ToolUsageFinishedEvent` | Tool completes | Record tool results |
| `CrewKickoffCompletedEvent` | Crew finishes | Trigger wrap sequence |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from crewai import Agent, Crew, Task, Process
from crewai.events import BaseEventListener
from crewai.events import (
    CrewKickoffStartedEvent,
    CrewKickoffCompletedEvent,
    TaskStartedEvent,
    TaskCompletedEvent,
    ToolUsageFinishedEvent,
)


class AnnealMemoryListener(BaseEventListener):
    """Event listener that gives CrewAI agents persistent memory."""

    def __init__(self, db_path: str, project_name: str):
        super().__init__()
        self.store = Store(db_path, project_name=project_name)

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_start(source, event):
            """Load continuity at crew kickoff."""
            continuity = self.store.load_continuity()
            if continuity:
                # Available for agents to reference via crew context
                print(f"Memory loaded: {len(continuity)} chars")

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_start(source, event):
            """Recall relevant episodes before each task."""
            if hasattr(event, "task_id"):
                result = self.store.recall(limit=5)
                if result.episodes:
                    print(f"Recalled {len(result.episodes)} episodes for task")

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_done(source, event):
            """Record task output as an episode."""
            if hasattr(event, "result_summary") and event.result_summary:
                self.store.record(
                    event.result_summary[:500],
                    EpisodeType.OUTCOME,
                    source=getattr(event, "worker_id", "crew"),
                )

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_done(source, event):
            """Record tool results as episodes."""
            # Tool results often contain valuable observations
            pass

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_crew_done(source, event):
            """Run wrap sequence when crew finishes."""
            wrap = prepare_wrap(self.store)
            if wrap["status"] == "ready":
                package = wrap["package"]
                # Feed the package to your LLM (CrewAI's `llm` from the
                # crew config is a natural fit — the agent's own model):
                compressed = llm.invoke(
                    f"Compress these episodes:\n{package['instructions']}\n"
                    f"Episodes:\n{package['episodes']}\n"
                    f"Current continuity:\n{package['continuity'] or ''}"
                )
                validated_save_continuity(self.store, compressed.content)


# Instantiate listener (registers automatically)
memory_listener = AnnealMemoryListener("./memory.db", "my-crew")

# Define crew as usual
researcher = Agent(
    role="Researcher",
    goal="Find and synthesize information",
    backstory="Expert research analyst",
)

task = Task(
    description="Research the latest trends in AI memory systems",
    expected_output="A comprehensive summary of findings",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[task],
    process=Process.sequential,
)

result = crew.kickoff()
```

## Alternative: Task and Step Callbacks

For simpler integrations, use CrewAI's built-in callback parameters:

```python
from anneal_memory import Store, EpisodeType

store = Store("./memory.db", project_name="my-crew")


def on_task_complete(output):
    """Record each task's output as an episode."""
    store.record(output.raw[:500], EpisodeType.OUTCOME, source=output.agent)


def on_agent_step(step):
    """Record each reasoning step."""
    if hasattr(step, "output") and step.output:
        store.record(str(step.output)[:300], EpisodeType.OBSERVATION)


crew = Crew(
    agents=[...],
    tasks=[...],
    task_callback=on_task_complete,
    step_callback=on_agent_step,
)
```

## Using with CrewBase Decorators

```python
from crewai.project import CrewBase, before_kickoff, after_kickoff, crew
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity


@CrewBase
class MyCrew:
    store = Store("./memory.db", project_name="my-crew")

    @before_kickoff
    def load_memory(self, inputs):
        """Inject memory context into crew inputs."""
        continuity = self.store.load_continuity()
        if continuity:
            inputs["memory_context"] = continuity
        return inputs

    @after_kickoff
    def save_memory(self, output):
        """Wrap session after crew completes."""
        wrap = prepare_wrap(self.store)
        if wrap["status"] == "ready":
            package = wrap["package"]
            # Your LLM compresses the package, then save with the full pipeline:
            compressed = llm.invoke(package["instructions"] + "\n" + package["episodes"])
            validated_save_continuity(self.store, compressed.content)
        return output

    @crew
    def crew(self) -> Crew:
        return Crew(agents=[...], tasks=[...])
```

## Key Considerations

- **Event bus vs callbacks:** The event bus (`BaseEventListener`) is richer — it captures tool usage, agent execution, and LLM calls in addition to task events. Use it for full integration. Callbacks are simpler for task-level-only recording.
- **CrewAI's built-in Memory:** CrewAI has its own `Memory` class with LLM-based extraction and consolidation. It does deduplication, not graduation. It has no immune system, citation decay, or anti-inbreeding defense. anneal-memory can run alongside it or replace it.
- **Multi-agent recording:** Use the `source` parameter on `store.record()` to attribute episodes to specific agents — this matters for per-agent identity tracking.
- **The cognitive loop:** The event listener captures episodes automatically, but compression should still involve an LLM — the agent's judgment during compression is where identity forms.
