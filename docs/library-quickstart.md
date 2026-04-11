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

> **Canonical entry points: `prepare_wrap(store)` + `validated_save_continuity(store, text)`.**
> These are the only functions you should call at session end. The library exposes
> lower-level primitives (`prepare_wrap_package()`, `store.save_continuity()`) for
> test use only — calling them directly bypasses the immune system and leaves
> `skipped_prepare=True` in the save result.
>
> `prepare_wrap_package()` is **deprecated since 0.2.0 and will be removed in 0.3.0.**
> Calling it emits a `DeprecationWarning` directing you to `prepare_wrap(store, ...)`.
> Advanced library users managing their own wrap lifecycle can use the private
> `_build_wrap_package()` helper instead — understanding that as a private symbol
> it carries no API stability guarantee across versions.

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity

store = Store("./memory.db", project_name="MyAgent")

# 1. Get the compression package.
#    prepare_wrap fetches episodes, loads the current continuity, detects stale
#    patterns, builds the compression package, attaches Hebbian association
#    context, and marks the wrap as in progress.
wrap = prepare_wrap(store)

# The returned dict has:
#   wrap["status"]         — "ready" (package built) or "empty" (no new episodes)
#   wrap["message"]        — human-readable status summary
#   wrap["episode_count"]  — number of episodes in the wrap window
#   wrap["package"]        — the compression package (dict, or None if empty)
#   wrap["assoc_context"]  — optional Hebbian association context (str or None)

if wrap["status"] == "ready":
    package = wrap["package"]
    # The package dict has:
    #   package["episodes"]       — recent episodes as formatted text
    #   package["episode_count"]  — same count as wrap["episode_count"]
    #   package["continuity"]     — current continuity file (None on first wrap)
    #   package["stale_patterns"] — patterns that haven't been cited recently
    #   package["instructions"]   — compression instructions with marker reference
    #   package["today"]          — current date for timestamps
    #   package["max_chars"]      — target continuity size

    # 2. Agent compresses (this is YOUR agent's job — the cognitive act).
    # Feed the package to your LLM and ask it to compress episodes into
    # an updated continuity file following the instructions.
    compressed = my_llm_compress(package, wrap["assoc_context"])  # your compression logic

    # 3. Save with the full validation pipeline.
    result = validated_save_continuity(store, compressed)
# "empty" status = nothing to wrap yet; skip and try again next session.

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
#          graduations_demoted, associations_formed, skipped_prepare, etc.

store.close()
```

**Why `prepare_wrap(store)` and not `prepare_wrap_package()`?** The store-aware version marks the wrap as in progress so the immune system knows this compression was preceded by a proper prepare step. Calling `prepare_wrap_package()` directly and then `validated_save_continuity()` will leave `skipped_prepare=True` in the save result — a signal the cognitive loop was bypassed. As of **0.2.0 `prepare_wrap_package()` is deprecated** and emits a `DeprecationWarning`; it will be removed in 0.3.0. Advanced library users managing their own lifecycle can use the private `_build_wrap_package()` helper (no API stability guarantee). Similarly, `store.save_continuity(text)` is the low-level file-write primitive — it bypasses graduation, associations, decay, and wrap metadata. Don't reach for either one.

## Affective State

Record functional state during compression to track what the agent found engaging, uncertain, or surprising. Pass an `AffectiveState` directly to `validated_save_continuity`:

```python
from anneal_memory import (
    Store, EpisodeType, AffectiveState,
    prepare_wrap, validated_save_continuity,
)

wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    compressed = my_llm_compress(wrap["package"], wrap["assoc_context"])
    result = validated_save_continuity(
        store,
        compressed,
        affective_state=AffectiveState(tag="engaged", intensity=0.8),
    )
```

The affective state is recorded on the Hebbian associations formed during this wrap — the intensity modulates association strength (up to 1.5x), and the tag is stored on each link for later retrieval. The tag is free-text (normalized to lowercase); intensity is clamped to `[0.0, 1.0]`. Transformers don't natively maintain persistent emotional state between sessions — this layer provides infrastructure for it.

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
    print(f"{a.episode_a} <-> {a.episode_b} strength={a.strength:.2f}")
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
