# AutoGen / AG2 Integration

anneal-memory integrates with AutoGen/AG2 through **hooks** — four message lifecycle hooks registered via `register_hook()`. These fire during the reply generation process, giving you access to messages before and after they're sent between agents.

## Install

```
pip install anneal-memory ag2
```

(Also works with `pip install autogen` — same package.)

## Integration Pattern

AG2 provides 4 hooks on `ConversableAgent`, firing in this order:

| Hook | Fires | anneal-memory use |
|------|-------|-------------------|
| `update_agent_state` | First, before reply | Recall episodes, update agent state |
| `process_last_received_message` | Second, on final message only | Enrich incoming message with memory context |
| `process_all_messages_before_reply` | Third, on full history | Inject recalled episodes into context |
| `process_message_before_send` | After reply generated, before send | Record outgoing messages as episodes |

## Complete Example

```python
from anneal_memory import Store, EpisodeType, prepare_wrap_package
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent


store = Store("./memory.db", project_name="my-autogen-agent")


def inject_memory(agent: ConversableAgent, messages: list[dict]) -> None:
    """Load continuity and inject into agent state (fires first)."""
    continuity = store.load_continuity()
    if continuity and not getattr(agent, "_memory_loaded", False):
        # Update system message to include memory context
        original = agent.system_message
        agent.update_system_message(
            f"{original}\n\nYour memory from prior sessions:\n{continuity}"
        )
        agent._memory_loaded = True


def enrich_with_recall(messages: list[dict]) -> list[dict]:
    """Add recalled episodes to message context (fires third)."""
    if messages:
        last_content = messages[-1].get("content", "")
        if isinstance(last_content, str) and last_content:
            result = store.recall(keyword=last_content[:100], limit=3)
            if result.episodes:
                recall_text = "\n".join(
                    f"[{ep.type}] {ep.content}" for ep in result.episodes
                )
                # Prepend memory context to messages
                memory_msg = {
                    "role": "system",
                    "content": f"Relevant memories:\n{recall_text}",
                }
                return [memory_msg] + messages
    return messages


def record_outgoing(
    sender: ConversableAgent, message, recipient, silent: bool
):
    """Record every outgoing message as an episode (fires last)."""
    content = message.get("content", "") if isinstance(message, dict) else str(message)
    if content and content.strip():
        store.record(
            content[:500],
            EpisodeType.OBSERVATION,
            source=sender.name,
        )
    return message


# Create agents
assistant = AssistantAgent(
    name="researcher",
    system_message="You are a research analyst.",
    llm_config={"config_list": [...]},
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
)

# Register hooks
assistant.register_hook("update_agent_state", inject_memory)
assistant.register_hook("process_all_messages_before_reply", enrich_with_recall)
assistant.register_hook("process_message_before_send", record_outgoing)

# Also record user proxy messages
user_proxy.register_hook("process_message_before_send", record_outgoing)

# Run conversation
user_proxy.initiate_chat(
    assistant,
    message="Analyze the latest trends in AI memory systems",
)

# Wrap after conversation ends
episodes = store.episodes_since_wrap()
if episodes:
    continuity = store.load_continuity()
    package = prepare_wrap_package(episodes, continuity, store.project_name)
    # Compress via LLM and save

store.close()
```

## Group Chat with Per-Agent Memory

In multi-agent group chats, each agent can have its own memory attribution:

```python
from autogen import GroupChat, GroupChatManager

agents = [researcher, writer, critic]

# Register recording hook on all agents
for agent in agents:
    agent.register_hook("process_message_before_send", record_outgoing)
    agent.register_hook("update_agent_state", inject_memory)

group_chat = GroupChat(agents=agents, messages=[], max_round=10)
manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="Research and write a report")

# Episodes are attributed per-agent via source parameter
```

## Using ContextVariables (AG2 v0.9+)

AG2's ContextVariables provide shared state across agents in group chat:

```python
from autogen.agentchat.group import ContextVariables

context = ContextVariables(data={
    "memory_context": store.load_continuity() or "",
    "session_episodes": 0,
})

# Tools can access and update context
def memory_tool(query: str, context_variables: ContextVariables) -> str:
    """Search memory — context_variables injected automatically."""
    result = store.recall(keyword=query, limit=5)
    context_variables["session_episodes"] = context_variables.get("session_episodes", 0) + 1
    if not result.episodes:
        return "No relevant memories found."
    return "\n".join(f"[{ep.type}] {ep.content}" for ep in result.episodes)
```

## Key Considerations

- **Hook execution order:** `update_agent_state` → `process_last_received_message` → `process_all_messages_before_reply` → (LLM generates reply) → `process_message_before_send`. Memory injection goes in the first three; episode recording goes in the last.
- **`process_message_before_send` is permanent:** Changes to the message in this hook persist in chat history. This makes it the best place for recording — you see the final message that will be sent.
- **`process_all_messages_before_reply` is transient:** Changes here affect the current LLM call only, not the stored chat history. Good for injecting memory context without polluting the conversation.
- **No built-in persistence:** AG2 keeps conversation history in memory only (`chat_messages` dict). anneal-memory provides cross-session persistence and compressed knowledge.
- **The cognitive loop:** The hooks handle recording automatically. Compression should happen after `initiate_chat()` returns, using an LLM for the actual compression judgment.
