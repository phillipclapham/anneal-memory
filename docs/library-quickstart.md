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

## Error Handling

Every error raised by the library is a subclass of `AnnealMemoryError`. **Most callers should catch that base class at their outermost boundary, log, and let the failure propagate.** Branching per subclass is a minority pattern for long-running servers with retry budgets or transport authors writing their own error translation layer.

### The 80% case

```python
from anneal_memory import (
    AnnealMemoryError,
    prepare_wrap,
    validated_save_continuity,
)

try:
    package = prepare_wrap(store)
    continuity_text = agent_compress(package)
    result = validated_save_continuity(
        store, continuity_text, wrap_token=package["wrap_token"]
    )
except AnnealMemoryError as err:
    logger.error(
        "anneal-memory failed (op=%s path=%s): %s",
        getattr(err, "operation", "unknown"),
        getattr(err, "path", "unknown"),
        err,
        exc_info=True,
    )
    raise
```

`exc_info=True` captures the full traceback including the `__cause__` chain to the underlying `sqlite3` error (if any). That's almost always all you need.

### The hierarchy

```
AnnealMemoryError (Exception)
 └── StoreError
      └── StoreDatabaseError
```

- **`AnnealMemoryError`** — base class. Catch this at your outermost boundary. anneal-memory deliberately does NOT mirror PEP 249 / DB-API 2.0's nine-class hierarchy: this is a library that consumes a database internally, not a database driver, so callers branch on operational intent (log/retry/escalate), not on vendor-level failure taxonomy.
- **`StoreError`** — operation-level failure with structured context. Carries an `.operation` field (a `StoreOperation` literal like `"save_continuity"`, `"record"`, `"wrap_completed"`) and a `.path` field pointing at the store DB file. Covers file-write failures (continuity/meta atomic writes) and integrity failures (partial wrap-snapshot state).
- **`StoreDatabaseError`** — subclass of `StoreError` raised when a SQLite operation fails (locked database, disk full, integrity constraint violations after retries, corruption). The underlying `sqlite3.DatabaseError` is attached as `__cause__` at raise time (not preserved across pickle — see below). Because it subclasses `StoreError`, existing `except StoreError` handlers catch it unchanged.

### When (and when not) to retry

`StoreDatabaseError` is **not blanket-retryable**. The subclass wraps every runtime database failure — transient contention AND permanent conditions like disk-full, integrity violations, and corruption. Retrying the non-transient class wastes cycles or makes things worse. Check the cause type and message to decide:

```python
from anneal_memory import StoreDatabaseError

def is_retryable(err: StoreDatabaseError) -> bool:
    """Return True only for transient contention that a backoff-retry
    can reasonably clear. sqlite3 doesn't expose SQLite's extended
    error codes at the Python level (no SQLITE_BUSY constant), so we
    dispatch on the cause type name and then match the message — the
    ugly-but-stable approach.

    This helper works both in-process AND after a pickle round-trip
    because ``cause_type_name`` is a plain string field on the
    exception instance, not a live ``__cause__`` reference (see the
    Pickle section below for why that matters for cross-process
    consumers like ProcessPoolExecutor and pytest-xdist)."""
    if err.cause_type_name != "OperationalError":
        return False
    # The wrapped message embeds the sqlite3 error text, so we can
    # string-match on str(err) regardless of whether __cause__ is
    # still live.
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg
```

Then:

```python
try:
    result = validated_save_continuity(store, text, wrap_token=token)
except StoreDatabaseError as err:
    if is_retryable(err):
        retry_backoff()
    else:
        logger.error(
            "Non-retryable DB failure (op=%s): %s",
            err.operation, err.__cause__,
            exc_info=True,
        )
        raise
except StoreError as err:
    # File I/O or integrity — not retryable; surface to operator.
    logger.error(
        "Store failure (op=%s path=%s): %s",
        err.operation, err.path, err,
    )
    alert_operator(err)
except AnnealMemoryError:
    raise
```

### `.operation` field

The `.operation` value names the caller-facing unit of work (what you'd retry), not the individual SQL statement that failed. For most public methods it matches the method name exactly (`record`, `recall`, `wrap_completed`, `save_continuity`). Two exceptions surface internal sub-phases of `validated_save_continuity`:

- **`"schema_init"`** — a failure during `Store(path)` construction (connect, schema setup, PRAGMA configuration, or orphan-tmp detection).
- **`"batch_commit"`** — a failure during the outer commit of the two-phase-commit wrap pipeline. Surfaces to callers through `validated_save_continuity`, not through any method named `batch_commit`.

The full list of `StoreOperation` values is exported as a `Literal` from `anneal_memory`.

### Pickle / copy

Both error classes round-trip cleanly through `pickle`, `copy.copy`, and `copy.deepcopy` with their subclass identity and context fields preserved. The **`__cause__` chain does NOT survive pickle** — this is a standard Python limitation, not a library choice. Cross-process consumers (`ProcessPoolExecutor`, `pytest-xdist`, RPC transports) see a `StoreDatabaseError` with `__cause__ = None`; the cause text is embedded in `str(err)` for human-readable logs but the live sqlite3 exception object is local-scope only.

**Retry decisions in cross-process contexts — use `cause_type_name`, not `__cause__`:** the `is_retryable()` helper above deliberately dispatches on `err.cause_type_name` (a plain string field populated at raise time) rather than `err.__cause__` (the live exception reference). The reason: after a pickle round-trip — which is what every `ProcessPoolExecutor`, `pytest-xdist` worker, or Celery task queue does when propagating exceptions back to a supervisor — Python drops `__cause__`. A helper that checks `isinstance(err.__cause__, sqlite3.OperationalError)` returns `False` for every error after unpickling, regardless of whether the original cause was transient. Silent retry-storm bait.

`cause_type_name` is a plain string on the exception instance, populated by `_db_boundary` at raise time with `type(exc).__name__`. It survives pickle identically because there's no live reference to marshal. The `is_retryable()` example above works both in-process (with a live `__cause__`) and post-pickle (with `__cause__ = None`) via the same code path.

If you need the live exception object (for `errno` access or sqlite3 extended error codes) you still need in-process access — those attributes don't survive pickle and `cause_type_name` doesn't try to preserve them. Cross-process retry decisions should dispatch on type name + message text; in-process callers can additionally drill into `err.__cause__.args` for errno-level detail.

### Hash chain verification

`StoreError` / `StoreDatabaseError` are runtime operation errors. To validate the audit-log hash chain itself, run `anneal-memory verify` from the CLI — that walks every entry in the sidecar JSONL and returns an `AuditVerifyResult`.

## Next Steps

- **Framework integration:** See [integration guides](integrations/) for LangGraph, CrewAI, OpenAI Agents SDK, and 9 more frameworks
- **MCP setup:** See the [README](../README.md#mcp-server) for editor configuration
- **CLI usage:** See [`examples/CLAUDE.md.cli.example`](../examples/CLAUDE.md.cli.example) for the full agent workflow
- **Session hygiene:** See the [README](../README.md#session-hygiene) for why wraps matter
