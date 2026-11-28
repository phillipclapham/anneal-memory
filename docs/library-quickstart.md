# Library Quickstart

The Python library is the core of anneal-memory. The CLI and the MCP server are thin interfaces over it. This is what you import when you build an agent in any framework.

## Install

```
pip install anneal-memory
```

No dependencies. Python 3.10 or newer.

## Record and recall

```python
from anneal_memory import Store, EpisodeType

store = Store("./memory.db")   # creates the database and its directories

store.record("Connection pool is the real bottleneck", EpisodeType.OBSERVATION)
store.record("Chose PostgreSQL because ACID outweighs speed", EpisodeType.DECISION)
store.record("Latency vs consistency: can't optimize both", EpisodeType.TENSION)
store.record("Should we shard or add read replicas?", EpisodeType.QUESTION)
store.record("Migration done, hot-path queries are 3x faster", EpisodeType.OUTCOME)
store.record("Production DB at 80% capacity, growing 5%/week", EpisodeType.CONTEXT)

result = store.recall(keyword="PostgreSQL", limit=10)
for ep in result.episodes:
    print(f"[{ep.type}] {ep.content}")

decisions = store.recall(episode_type=EpisodeType.DECISION)
recent = store.recall(since="2026-04-01T00:00:00Z")

store.close()
```

### Searching with a multi-word query

`store.recall(keyword=...)` matches the keyword as one exact substring, so `"bank export nightly rows"` finds only an episode that contains those words in that order. To match word by word, use `search_episodes`. It reduces the query to its distinctive words, returns every episode that contains at least one, ranks by how many it contains (rarer words count more), and says which words each episode matched. It takes the same `episode_type`, `source`, `since` and `until` filters as `recall`.

```python
from anneal_memory import search_episodes

for m in search_episodes(store, "bank export fmt_row64 nightly rows", limit=5):
    print(m.episode.id, m.matched, m.episode.content)
```

The MCP `recall` tool does this for you. It tries the exact phrase first, and only when no episode contains it and the keyword has two or more distinctive words does it fall back to `search_episodes` with the same filters and limit. Its reply says when it did.

## The wrap

A wrap turns the episodes recorded since the last wrap into an updated memory file. It is one rolling rewrite and there is no other method.

`prepare_wrap(store)` opens **one step**: the oldest calendar day that has unwrapped episodes. It returns a fixed prompt, filled with the current memory file and that day's log, and a wrap token. You run the prompt in a **fresh context**, one that holds that prompt and nothing else, and pass the model's whole reply to `validated_save_continuity`. anneal-memory calls no model.

```python
import anthropic
from anneal_memory import Store, prepare_wrap, validated_save_continuity

client = anthropic.Anthropic()
MODEL = "your-model-name"   # a mid-size model at low effort is what the prompt is written for


def run_model(system: str, user: str) -> str:
    """One model call, nothing else in the context."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


store = Store("./memory.db")

while True:
    wrap = prepare_wrap(store)
    if wrap["status"] != "ready":
        break                      # "empty": nothing waiting. "downgraded": see below
    step = wrap["step"]
    print(f"wrapping {step['day']}: {step['episodes']} episodes, "
          f"{step['steps_remaining']} more step(s) after this one")
    reply = run_model(step["system"], step["user"])
    result = validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"])
    if result["pending_count"] == 0:
        break

store.close()
```

Any client works in place of `anthropic`; `run_model` only has to return the model's reply text.

### What `prepare_wrap` returns

A dict with `status`, `message`, `episode_count` (episodes in this step), `pending_count` (all episodes waiting), `wrap_token` and `step`.

- `status == "ready"`: a step is open. `wrap_token` and `step` are set.
- `status == "empty"`: no episodes since the last wrap. Nothing was opened; skip the save.
- `status == "downgraded"`: the consolidate baton gate declined the call, so it is capture-only. `message` says why and the store is untouched. See "The baton" below.

`step` holds `day`, `chunk` and `chunks` (a day over 400,000 characters of log is cut into slices), `episodes`, `steps_remaining`, `max_chars`, `memory_chars` (the size of the current file), and `system`, `user` and `prompt` (the two messages as one text, for a transport with a single slot).

Pass `tz="America/New_York"` (or any IANA name or `tzinfo`) to cut days in a zone other than the machine's, and `max_chars=` to change the 26,000-character cap. The step keeps that cap and the save enforces it: leave `max_chars` out of the save (a different value is refused).

### What the save checks

`validated_save_continuity(store, text, *, wrap_token=None, max_chars=None, ...)` checks that a wrap is open and is the one `wrap_token` names (the token is required; `None` is the signature's default only so the call can be refused with a clear message), that the caller may commit it, and that `text` is a non-empty string of at most the step's `max_chars` characters (the cap `prepare_wrap` was given). That is all. The file is not parsed and needs no particular layout. A refused save writes nothing and leaves the step open: run the step again and save again, or abandon it with `store.wrap_cancelled()`.

The result has `path`, `chars`, `episodes_compressed`, `pending_count`, `content_hash` and `wrap_result`. If `pending_count` is not zero, the next day's step is owed: prepare again.

A multi-day catch-up is several steps in order. Each step starts from the file the one before it saved. Do not merge days into one log.

### The baton

A store created by `anneal-memory init` requires the consolidate baton: a wrap runs only under a session id the operator has claimed (`anneal-memory baton claim NAME`). On such a store pass `session_id="NAME"` to both calls, and keep the token:

```python
wrap = prepare_wrap(store, session_id="NAME")
...
validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"], session_id="NAME")
```

A store you create with `Store(path)` has no such requirement, and there you leave `session_id` out. The CLI and the MCP server drop the id on such a store, but the library does not: a library call that passes `session_id=` is checked against the baton and downgrades unless that session holds it. `store.status()` and the downgrade message say how many episodes are waiting.

### Retries

Pass an `idempotency_key=` to make a retry safe. A key belongs to one operation, so give `prepare_wrap` and `validated_save_continuity` different keys. A retry of a committed save with the same key and text returns a short result (`wrap_id`, `chars`, `path`, `content_hash`, `episodes_compressed`, `replayed: True`) and writes nothing. A retry of `prepare_wrap` with its key returns the same step while that wrap is still open. Once the wrap is saved or cancelled, the retry raises `IdempotencyConflictError`; prepare again with a new key. The same key with a different request raises it too.

### Recovering an older version

Each saved file is a row in the `identity_versions` table. See [Recovering an older version](../README.md#recovering-an-older-version) in the README.

## Loading the memory file at session start

```python
store = Store("./memory.db")

text = store.load_continuity()
if text:
    print(text)   # feed this to your agent as context
else:
    print("First session: no memory file yet")
```

A store gets its first memory file at its first completed wrap.

## Audit trail

On by default. Verify it, or stream its events:

```python
from anneal_memory import AuditTrail, Store

result = AuditTrail.verify("./memory.db")
print(f"Valid: {result.valid}, entries: {result.total_entries}")
if not result.valid:
    print(f"Break at entry {result.chain_break_at} in {result.chain_break_file}")


def my_handler(entry):
    print(f"[{entry['event']}] {entry['timestamp']}")


store = Store("./memory.db", on_audit_event=my_handler)
```

## Context manager

```python
with Store("./memory.db") as store:
    store.record("Something important", EpisodeType.OBSERVATION)
    # store.close() is called for you
```

## Configuration

```python
store = Store(
    "./memory.db",
    audit=True,                  # audit trail on (the default)
    on_audit_event=my_handler,   # optional: stream audit events
)
```

The path can be absolute or relative. The memory file is kept beside the database as `<stem>.continuity.md`, a projection of the newest saved version. The audit trail is `<stem>.audit.jsonl` beside the database, with its weekly rotated files and manifest next to it.

## Error handling

The library raises errors in two groups.

**`AnnealMemoryError` and its subclasses** are store failures (`StoreError`, then `StoreDatabaseError`) and the wrap-lifecycle refusals `WrapInProgressError`, `WrapOwnershipError` and `WrapCancelGatedError`. Catch the base class at your outermost boundary, log, and let it propagate. Branch per subclass only in a long-running server with a retry budget.

**Plain `ValueError`** is raised by `validated_save_continuity` for a call that cannot succeed as written: no wrap in progress (`prepare_wrap` not called, or the step was already saved), no token, or one that is not the open wrap's (including a wrap that was cancelled and prepared again under the same token while the save was running), a `max_chars` that is not the step's, an empty reply, a reply over the step's limit, or an empty `session_id`. Fix the call. The empty and over-limit cases leave the step open, so you can run it again and save again.

**`SaveAuthorityError`**, a `ValueError` subclass exported at the top level, is the consolidate gate refusing a save. Test for it with `isinstance`, not the message text. Some of these are call fixes: a gated wrap's save that omits the `wrap_token` its `prepare_wrap` returned, or the preparing `session_id`; on a store that requires the baton, a save that omits `session_id` or `wrap_token`; a wrap prepared with `allow_sole_live=True` whose save does not repeat it. The rest are authority refusals: a `session_id` that does not, or no longer, hold the baton, or an unreadable baton file. For these, claim nothing yourself, because the operator decides who holds the baton. Surface the refusal, and either retry the same save once it is fixed or abandon the wrap with `store.wrap_cancelled(expect_token=...)`. A save from a session other than the one that prepared a gated wrap is refused even if that session now holds the baton.

`validated_save_continuity` raises `ValueError` when **no** wrap is open, and `prepare_wrap` raises `WrapInProgressError` when one **already** is. The consolidate is single-writer: finish or cancel the open wrap before starting another.

### The 80% case

```python
from anneal_memory import AnnealMemoryError, prepare_wrap, validated_save_continuity

try:
    wrap = prepare_wrap(store)
    if wrap["status"] == "ready":
        reply = run_model(wrap["step"]["system"], wrap["step"]["user"])
        result = validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"])
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

`exc_info=True` captures the traceback including the `__cause__` chain down to the underlying `sqlite3` error, if any. That is usually all you need.

### The hierarchy

```
AnnealMemoryError (Exception)
 ├── StoreError
 │    ├── StoreDatabaseError
 │    └── IdempotencyConflictError
 ├── WrapInProgressError
 ├── WrapOwnershipError
 └── WrapCancelGatedError
```

Other `AnnealMemoryError` subclasses are exported for narrower cases (`BatonHeldError`, `LeaseFencedError`, `WrapCancelBoundError`, `WrapRevertRefused`, `WrapContinuityMovedError`, `WrapWindowMovedError`, `ContinuityLockUnavailable`, `PostconditionFailed`, and the supersession, spore and intention errors). `IdempotencyConflictError` is raised when an idempotency key is reused for a different request.

- **`StoreError`**: an operation-level failure. It carries `.operation` (a `StoreOperation` literal such as `"save_continuity"`, `"record"` or `"wrap_completed"`) and `.path`, the database file. It covers file-write failures and integrity failures such as partial wrap-snapshot state.
- **`StoreDatabaseError`**: a SQLite operation failed (locked database, disk full, integrity violation after retries, corruption). The underlying `sqlite3.DatabaseError` is `__cause__` at raise time, though not across pickle. Existing `except StoreError` handlers still catch it.
- **`WrapInProgressError`**: a sibling of `StoreError`, not a subclass. `prepare_wrap` (and `Store.wrap_started`) raises it when a wrap is already open. It carries `.started_at`. Recovery: save the open wrap, or abandon it with `store.wrap_cancelled()` (for a gated wrap pass the preparing `session_id`, or `force=True`; pass `expect_token=` to abandon it only if it is still the one open), then prepare again.
- **`WrapOwnershipError`**: raised by `Store.wrap_cancelled(expect_token=...)` when the open wrap's token is not the one you named. It carries `.expected` and `.actual`; `actual is None` means no wrap is open at all (yours already finished, so it is safe to retry), as opposed to someone else's wrap being there (back off).
- **`WrapCancelGatedError`**: raised by `Store.wrap_cancelled()` with no `expect_token` when the open wrap was prepared under the baton by another session. Nothing changes. Pass that wrap's token, the preparing `session_id`, or `force=True` if that session is gone.
- **`SaveAuthorityError`** sits outside this tree, as described above.

### When to retry

`StoreDatabaseError` is not blanket-retryable. It wraps transient contention and permanent conditions alike, such as a full disk or corruption. Check the cause:

```python
from anneal_memory import StoreDatabaseError


def is_retryable(err: StoreDatabaseError) -> bool:
    """True only for transient contention that a backoff can clear.

    cause_type_name is a plain string on the exception, so this works in
    process and after a pickle round-trip (ProcessPoolExecutor, pytest-xdist),
    where __cause__ is dropped.
    """
    if err.cause_type_name != "OperationalError":
        return False
    msg = str(err).lower()
    return "locked" in msg or "busy" in msg
```

```python
from anneal_memory import StoreError, validated_save_continuity

try:
    result = validated_save_continuity(store, reply, wrap_token=token)
except StoreDatabaseError as err:
    if is_retryable(err):
        retry_backoff()
    else:
        logger.error("Non-retryable DB failure (op=%s): %s", err.operation, err.__cause__, exc_info=True)
        raise
except StoreError as err:
    # file I/O or integrity: not retryable, surface it to the operator
    logger.error("Store failure (op=%s path=%s): %s", err.operation, err.path, err)
    alert_operator(err)
```

`retry_backoff` and `alert_operator` are yours.

### The `.operation` field

`.operation` names the unit of work you would retry, not the SQL statement that failed. For most public methods it equals the method name. Four values name internal phases:

- `"schema_init"`: a failure while `Store(path)` was connecting, setting up the schema, or checking for orphan temp files.
- `"batch_commit"`: the outer commit of the wrap's two-phase pipeline. It surfaces through `validated_save_continuity`.
- `"batch_begin"`: the save could not take its write lock because another writer held it past the busy timeout. Nothing was written and the wrap stays open.
- `"supersession_repair"`: `Store(path)` opened read-write to repair supersession links an older version left dangling and hit a database error other than lock contention.

The full list of `StoreOperation` values is exported as a `Literal` from `anneal_memory`.

### Pickle and copy

The error classes round-trip through `pickle`, `copy.copy` and `copy.deepcopy` with their subclass and context fields. The `__cause__` chain does not survive pickle; that is a Python limit. Use `cause_type_name` and the message text for retry decisions across processes.

### Hash chain verification

`StoreError` and `StoreDatabaseError` are runtime errors. To check the audit log's hash chain itself, run `anneal-memory verify`, which walks every entry in the JSONL and returns an `AuditVerifyResult`.

## Next steps

- Framework guides: [docs/integrations](integrations/)
- MCP setup: the [README](../README.md#mcp-server)
- Agent workflow snippets: [`examples/agent-instructions.lean.cli.example`](../examples/agent-instructions.lean.cli.example)
- Session hygiene: the [README](../README.md#session-hygiene)
- Upgrading from 3.x: [MIGRATING_TO_4.md](../MIGRATING_TO_4.md)
