# Library Quickstart

anneal-memory's Python library is the core product. The CLI and MCP server are interfaces to it — the library is what you import when building agents in any framework.

## Install

```
pip install anneal-memory
```

Zero dependencies. Python 3.10+.

## Basic Usage

```python
from anneal_memory import Store, EpisodeType

# Create a store (auto-creates DB + directories)
store = Store("./memory.db", project_name="MyAgent")

# Record episodes during work
store.record("Connection pool is the real bottleneck", EpisodeType.OBSERVATION)
store.record("Chose PostgreSQL because ACID outweighs speed", EpisodeType.DECISION)
store.record("Latency vs consistency — can't optimize both", EpisodeType.TENSION)
store.record("Should we shard or add read replicas?", EpisodeType.QUESTION)
store.record("Migration done, 3x query improvement on hot path", EpisodeType.OUTCOME)
store.record("Production DB at 80% capacity, growing 5%/week", EpisodeType.CONTEXT)

# Query episodes
result = store.recall(keyword="database", limit=10)
for ep in result.episodes:
    print(f"[{ep.type}] {ep.content}")

# Filter by type, time, source
decisions = store.recall(episode_type=EpisodeType.DECISION)
recent = store.recall(since="2026-04-01T00:00:00Z")

store.close()
```

## The Wrap Sequence

This is the most important part. Wrapping is how raw episodes compress into working memory.

```python
from anneal_memory import Store, EpisodeType, prepare_wrap_package, validated_save_continuity

store = Store("./memory.db", project_name="MyAgent")

# 1. Get the compression package
episodes = store.episodes_since_wrap()
continuity = store.load_continuity()
package = prepare_wrap_package(episodes, continuity, "MyAgent")

# package contains:
#   package["episodes"]      — recent episodes as formatted text
#   package["continuity"]    — current continuity file (or empty template)
#   package["stale_patterns"] — patterns that haven't been cited recently
#   package["instructions"]  — compression instructions with marker reference
#   package["today"]         — current date for timestamps
#   package["associations"]  — association context for related episodes

# 2. Agent compresses (this is YOUR agent's job — the cognitive act)
# Feed the package to your LLM and ask it to compress episodes into
# an updated continuity file following the instructions.
compressed = my_llm_compress(package)  # your compression logic

# 3. Save with full validation pipeline
result = validated_save_continuity(store, compressed)

# validated_save_continuity runs the complete immune system:
#   - Validates 4-section structure (State, Patterns, Decisions, Context)
#   - Checks citation evidence on graduations (2x, 3x patterns)
#   - Demotes ungrounded patterns (anti-inbreeding)
#   - Records Hebbian associations between co-cited episodes
#   - Applies decay (0.9x) to unreinforced associations
#   - Updates metadata and records wrap completion
#   - Logs everything in the audit trail
#
# Returns: dict with episodes_compressed, graduations_validated,
#          graduations_demoted, associations_formed, etc.

store.close()
```

## Loading Continuity at Session Start

```python
store = Store("./memory.db", project_name="MyAgent")

continuity = store.load_continuity()
if continuity:
    # Feed this to your agent as context — it's the compressed
    # memory from all prior sessions
    print(continuity)
else:
    print("First session — no prior memory")
```

## Associations

After wraps, episodes that were cited together form lateral associations.

```python
# Get association statistics
stats = store.association_stats()
print(f"Total links: {stats.total_links}")
print(f"Avg strength: {stats.avg_strength:.2f}")
print(f"Density: {stats.density:.4f}")

# Find associations for specific episodes
assocs = store.get_associations(episode_ids=["abc123"], min_strength=0.5)
for a in assocs:
    print(f"{a.id_a} <-> {a.id_b} strength={a.strength:.2f}")
```

## Affective State

Record functional state during compression to track what the agent found engaging, uncertain, or surprising.

```python
# Via the MCP tool or CLI, affective state is passed as parameters.
# Via the library, it's recorded through the wrap metadata:
store.save_continuity(compressed)
# Affective state is typically passed through the MCP/CLI layer.
# For direct library use, store it in episode metadata:
store.record(
    "This architecture feels right — clean separation of concerns",
    EpisodeType.OBSERVATION,
    metadata={"affect": {"tag": "confident", "intensity": 0.8}}
)
```

## Audit Trail

On by default. Verify integrity programmatically:

```python
from anneal_memory import AuditTrail

result = AuditTrail.verify("./memory.db")
print(f"Valid: {result.valid}")
print(f"Total entries: {result.total_entries}")
if not result.valid:
    print(f"Break at: {result.break_at}")
```

Stream events to external systems:

```python
def my_handler(entry):
    # Send to SIEM, cloud logging, observability, etc.
    print(f"[{entry['event']}] {entry['timestamp']}")

store = Store("./memory.db", project_name="MyAgent", on_audit_event=my_handler)
```

## Context Manager

The Store supports context manager usage for automatic cleanup:

```python
with Store("./memory.db", project_name="MyAgent") as store:
    store.record("Something important", EpisodeType.OBSERVATION)
    # store.close() called automatically
```

## Configuration

```python
store = Store(
    "./memory.db",
    project_name="MyAgent",
    audit=True,                    # Enable audit trail (default: True)
    on_audit_event=my_handler,     # Optional: stream audit events
)
```

The database path can be absolute or relative. The continuity file lives alongside the database as `{db_name}.continuity.md`. Audit files live in `{db_dir}/audit/`.

## Next Steps

- **Framework integration:** See [integration guides](integrations/) for LangGraph, CrewAI, OpenAI Agents SDK, and 9 more frameworks
- **MCP setup:** See the [README](../README.md#mcp-server) for editor configuration
- **CLI usage:** See [`examples/CLAUDE.md.cli.example`](../examples/CLAUDE.md.cli.example) for the full agent workflow
- **Session hygiene:** See the [README](../README.md#session-hygiene) for why wraps matter
