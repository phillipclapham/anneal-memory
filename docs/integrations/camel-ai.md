# CAMEL-AI Integration

anneal-memory integrates with CAMEL-AI through **WorkforceCallback** for multi-agent orchestration and by wrapping `ChatAgent.step()` for single-agent use. CAMEL-AI's role-playing architecture makes it well-suited for per-agent memory with distinct identity tracking.

## Install

```
pip install anneal-memory camel-ai
```

## Integration Pattern

CAMEL-AI provides hooks at two levels:

| Mechanism | Fires | anneal-memory use |
|-----------|-------|-------------------|
| `WorkforceCallback` (11 methods) | Task lifecycle events | Record episodes from task completion/failure |
| `on_request_usage` on ChatAgent | After each LLM call | Track usage (limited data) |
| Wrap `ChatAgent.step()` | Each agent turn | Record input/output as episodes |

## Multi-Agent: WorkforceCallback

The `WorkforceCallback` is the richest integration surface for multi-agent setups:

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity
from camel.societies.workforce import Workforce
from camel.societies.workforce.workforce_callback import WorkforceCallback
from camel.societies.workforce.events import (
    LogEvent, TaskCreatedEvent, TaskCompletedEvent,
    TaskFailedEvent, AllTasksCompletedEvent,
    TaskAssignedEvent, TaskStartedEvent, TaskUpdatedEvent,
    TaskDecomposedEvent, WorkerCreatedEvent, WorkerDeletedEvent,
)
from camel.agents import ChatAgent


store = Store("./memory.db", project_name="my-camel-workforce")


class AnnealMemoryCallback(WorkforceCallback):
    """Workforce callback that records task events as episodes."""

    def log_message(self, event: LogEvent) -> None:
        pass  # Skip general log messages

    def log_task_created(self, event: TaskCreatedEvent) -> None:
        store.record(
            f"Task created: {event.task_id}",
            EpisodeType.CONTEXT,
        )

    def log_task_decomposed(self, event: TaskDecomposedEvent) -> None:
        store.record(
            f"Task decomposed: {event.task_id}",
            EpisodeType.DECISION,
        )

    def log_task_assigned(self, event: TaskAssignedEvent) -> None:
        store.record(
            f"Task {event.task_id} assigned to {event.worker_id}",
            EpisodeType.CONTEXT,
            source=event.worker_id or "coordinator",
        )

    def log_task_started(self, event: TaskStartedEvent) -> None:
        pass

    def log_task_updated(self, event: TaskUpdatedEvent) -> None:
        pass

    def log_task_completed(self, event: TaskCompletedEvent) -> None:
        store.record(
            f"Task {event.task_id} completed: {event.result_summary[:400] if event.result_summary else 'no summary'}",
            EpisodeType.OUTCOME,
            source=event.worker_id or "workforce",
        )

    def log_task_failed(self, event: TaskFailedEvent) -> None:
        store.record(
            f"Task {event.task_id} failed: {event.error_message}",
            EpisodeType.TENSION,
            source=event.worker_id or "workforce",
        )

    def log_worker_created(self, event: WorkerCreatedEvent) -> None:
        store.record(
            f"Worker created: {event.worker_id} ({event.role})",
            EpisodeType.CONTEXT,
        )

    def log_worker_deleted(self, event: WorkerDeletedEvent) -> None:
        pass

    def log_all_tasks_completed(self, event: AllTasksCompletedEvent) -> None:
        """All tasks done — trigger wrap sequence."""
        wrap = prepare_wrap(store)
        if wrap["status"] == "ready":
            package = wrap["package"]
            # Use a ChatAgent to compress, then save with the full pipeline:
            compressor = ChatAgent(system_message="You compress agent memory.")
            response = compressor.step(
                BaseMessage.make_user_message(
                    role_name="User",
                    content=f"{package['instructions']}\n\n{package['episodes']}",
                )
            )
            validated_save_continuity(store, response.msgs[0].content)


# Create workforce with memory callback
workforce = Workforce(
    description="Research team",
    children=[...],
    callbacks=[AnnealMemoryCallback()],
)

result = workforce.process_task(task)
```

## Single-Agent: Wrapping step()

For single `ChatAgent` use, wrap the `step()` method to capture each turn:

```python
from anneal_memory import Store, EpisodeType
from camel.agents import ChatAgent
from camel.messages import BaseMessage


store = Store("./memory.db", project_name="my-camel-agent")


class MemoryAgent(ChatAgent):
    """ChatAgent with anneal-memory episode recording."""

    def __init__(self, *args, memory_store: Store, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_store = memory_store

    def step(self, input_message, response_format=None):
        """Record each turn as an episode."""
        # Record the input
        if isinstance(input_message, BaseMessage):
            content = input_message.content
        else:
            content = str(input_message)
        self.memory_store.record(
            f"Input: {content[:300]}", EpisodeType.CONTEXT
        )

        # Run the step
        response = super().step(input_message, response_format)

        # Record the output
        if response.msgs and response.msgs[0].content:
            self.memory_store.record(
                response.msgs[0].content[:500],
                EpisodeType.OBSERVATION,
            )

        return response


# Load continuity for system message
continuity = store.load_continuity()
system_msg = "You are a research analyst."
if continuity:
    system_msg += f"\n\nYour memory from prior sessions:\n{continuity}"

agent = MemoryAgent(
    system_message=system_msg,
    memory_store=store,
)

# Run conversation
response = agent.step(BaseMessage.make_user_message(role_name="User", content="Analyze X"))
```

## Role-Playing with Memory

CAMEL-AI's `RolePlaying` creates two agents in conversation. Give each agent its own memory identity:

```python
from camel.societies import RolePlaying

# Each agent gets episodes attributed to its role
rp = RolePlaying(
    assistant_role_name="Researcher",
    user_role_name="Critic",
    task_prompt="Evaluate AI memory architectures",
)

init_msg = rp.init_chat()
while True:
    assistant_response, user_response = rp.step(init_msg)

    # Record with role attribution
    if assistant_response.msgs:
        store.record(
            assistant_response.msgs[0].content[:500],
            EpisodeType.OBSERVATION,
            source="Researcher",
        )
    if user_response.msgs:
        store.record(
            user_response.msgs[0].content[:500],
            EpisodeType.OBSERVATION,
            source="Critic",
        )

    if assistant_response.terminated or user_response.terminated:
        break
    init_msg = assistant_response.msgs[0]
```

## Key Considerations

- **WorkforceCallback vs step wrapping:** `WorkforceCallback` is the mature integration for multi-agent orchestration — 11 lifecycle hooks with structured Pydantic events. For single `ChatAgent` use, wrapping `step()` is necessary since ChatAgent has no lifecycle hooks.
- **Per-agent identity:** Use the `source` parameter on `store.record()` to attribute episodes to specific agents/roles. This matters for the flagship paper's identity-through-memory thesis.
- **Shared memory:** CAMEL-AI's `share_memory=True` flag on Workforce enables memory sharing among workers. With anneal-memory, all agents write to the same store but are distinguished by `source` — natural multi-agent memory.
- **The cognitive loop:** The WorkforceCallback handles recording automatically. Compression should happen in `log_all_tasks_completed()`, using an LLM for the actual compression judgment.
