# Changelog

All notable changes to anneal-memory. Format is loosely [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses [SemVer](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-04-14

Documentation-verification pass + tooling hardening. Every one of the 12 framework integration guides has now been exercised end-to-end against a live `pip install` of the framework; every guide header is pinned with a "Verified against" line. Combined with 10.5d's prior run of the top 5 (LangGraph, CrewAI, OpenAI Agents SDK, Pydantic AI, smolagents), the final hit rate across all 12 guides was 9/12 had load-bearing drift — 15 integration bugs total fixed between v0.2.0 and v0.2.1.

No public API changes. No data migration. Drop-in upgrade from v0.2.0.

### Fixed — framework integration guides

- **google-adk.** `BaseMemoryService.search_memory` signature was wrong — the real ADK protocol is `(*, app_name: str, user_id: str, query: str) -> SearchMemoryResponse` (keyword-only three-tuple), not `(query) -> dict`. `MemoryEntry.content` is a `google.genai.types.Content` wrapper, not a plain string. Consumer-side `tool_context.search_memory(query)` returns `SearchMemoryResponse` with a `.memories` attribute, not a dict with `"results"`. Rewrote the `AnnealMemoryService` example and the `tool_context.search_memory` usage snippet against live `google-adk 1.30.0`. Verified with a real `MemoryEntry` round-trip.
- **llamaindex.** `BaseMemoryBlock.name` is a required field; guide's `AnnealMemoryBlock` failed to declare it and raised `ValidationError` at construction. `_aput` signature drift — real API is `_aput(self, messages: list[ChatMessage]) -> None` (a list of messages, no `**kwargs`). Guide had `_aput(self, message, **kwargs)` singular, which would have silently no-op'd on every flush (`hasattr(list, "content")` is False) and recorded zero episodes. Verified with a real `ChatMessage` list round-trip against `llama-index-core 0.14.20`.
- **haystack.** `Tracer.trace` signature missing `parent_span` — real protocol is `(operation_name, tags=None, parent_span=None)` and Haystack internals pass it as a kwarg; guide's two-arg signature would have raised `TypeError` on first real invocation. Guide's custom tracer was also missing `current_span()` (abstract on base `Tracer`) and the span class was missing `set_tags` / `raw_span` / `get_correlation_data_for_logs` (Haystack internals call all of them). `AgentBreakpoint` API was completely wrong — guide used `AgentBreakpoint(break_after_agent_step=3)` which is not a real kwarg; the real dataclass is `AgentBreakpoint(agent_name: str, break_point: Breakpoint | ToolBreakpoint)`. Rewrote the breakpoint example with the correct `Breakpoint(component_name, visit_count)` wrapper. Verified against `haystack-ai 2.27.0`.
- **camel-ai.** `TaskDecomposedEvent` has no `task_id` field — a decomposition describes the edge from a parent task into its children, so the real fields are `parent_task_id` and `subtask_ids`. Guide accessed `event.task_id` which would have raised `AttributeError` the first time a workforce decomposed a task. Rewrote the `log_task_decomposed` recording to surface `parent_task_id` + the list of `subtask_ids`. Verified with all 11 `WorkforceCallback` abstract methods implemented and real event objects against `camel-ai 0.2.90`.

### Verified clean (no drift)

- **autogen / ag2** — verified against `ag2 0.11.5`. All three hooks (`update_agent_state`, `process_all_messages_before_reply`, `process_message_before_send`) fire with the exact signatures the guide uses. Confirmed against `ConversableAgent._process_message_before_send` source.
- **dspy** — verified against `dspy 3.1.3`. All 5 `BaseCallback` lifecycle hooks, `dspy.configure(callbacks=[...])`, `ChainOfThought` / `ReAct` / `Retrieve` / `LM` / `MIPROv2` surface, and the `MemoryAwareRAG` subclass pattern all verified. `dspy.Module` tracks sub-modules without `super().__init__()`.
- **anthropic-agents (claude-agent-sdk)** — verified against `claude-agent-sdk 0.1.59`. `ClaudeAgentOptions(setting_sources, hooks)`, `HookMatcher(hooks=[callable])`, async hook callback signature, all four hook event names (`Stop`, `PostToolUse`, `PreCompact`, `UserPromptSubmit`) verified against the real `Literal` union, `query(prompt, options)` signature, `Store.status().episodes_since_wrap` accessor.

### Fixed — Diogenes overnight review (3 LOWs)

- `store.py` `_db_boundary` docstring opening line drift. Said "any `sqlite3.Error` subclass"; implementation catches `sqlite3.DatabaseError` specifically (`InterfaceError` API-misuse bugs propagate bare by design — documented lower in the same docstring). Opening line now matches implementation.
- `store.py` `close()` inside `_batch()` context. Previously raised `"Cannot batch_commit on a closed store"` which attributed the failure to the wrong operation. Added explicit guard at top of `close()` that raises `StoreError("Cannot close() while inside _batch() context", operation="close")`.
- `pyproject.toml` `[tool.pytest.ini_options]` gained `filterwarnings = ["error::DeprecationWarning"]`. Local dev has always caught `DeprecationWarning` via `-W error`; CI did not enforce the same gate. All 707 tests remain green with the flag set — CI now catches any deprecation regression the dev environment would catch.

### Internal notes

- `assert`-for-mypy-narrowing count held at 3 sites (`continuity.py:505`, `continuity.py:897`, `server.py:153`). No current risk (`python -O` is uncommon in this codebase's deployment). Gate: if this reaches 5+ sites, evaluate whether `if not X: raise StoreError(...)` is cleaner than the assert idiom.
- `StoreOperation` Literal unchanged from v0.2.0 (25 values). The 10.5c.6 drift test still runs on every CI matrix job.

### Meta

The v0.2.0 → v0.2.1 delta is almost entirely "documentation drift caught by exercising the docs against reality." This reinforces the framework-guide maintenance cadence committed in v0.2.0: every ~8 weeks OR on a major framework release, whichever comes first. 12-of-12 guides now have pinned version headers, so the next verification pass can diff against a concrete baseline instead of reading the prose cold.

## [0.2.0] — 2026-04-14

Library-first release. The library is now the canonical product; MCP server and CLI are thin adapters that delegate to the same pipeline. Every public integration guide was exercised against a live `pip install` before release (Session 10.5d). Substantial hardening of the wrap pipeline: session-handshake token closes the prepare/save TOCTOU window, two-phase file/DB commit survives crashes in every intermediate state, all SQLite failures surface through a typed exception hierarchy with pickle-safe cross-process retry dispatch.

### Breaking changes

- **Engine removed.** The automated compression engine is gone. Compression is cognition — it must involve the agent's own judgment, not a delegated "engine" LLM. Every access pattern (library, MCP, CLI) now runs the same agent-driven pipeline: `prepare_wrap(store)` → agent compresses → `validated_save_continuity(store, text, ...)`. If you were importing or subclassing `Engine`, it no longer exists; rewrite to the canonical pipeline.
- **Exception hierarchy rebuilt.** `StoreError` no longer subclasses `OSError`. New hierarchy:
  - `AnnealMemoryError(Exception)` — library base, catches everything
  - `StoreError(AnnealMemoryError)` — file/path failures (continuity write, meta sidecar, tmp sidecar)
  - `StoreDatabaseError(StoreError)` — SQLite failures (locked DB, disk full, corruption, integrity violations after retries)
  Code that caught `except OSError` on store operations must migrate to `except StoreError` or `except AnnealMemoryError`. Code that caught `except StoreError` continues to work unchanged (it now catches both file-origin and DB-origin failures via the subclass relationship). We deliberately do not mirror PEP 249 / DB-API 2.0 — anneal-memory is a library consuming a database, not a driver.
- **`validated_save_continuity` return shape.** `wrap_result` is now a plain `dict[str, Any]` (via `dataclasses.asdict`) instead of a `WrapResult` dataclass. The entire `SaveContinuityResult` is JSON-serializable top-to-bottom. Library users who want the typed object can reconstruct it via `WrapResult(**result["wrap_result"])`.
- **`Store.wrap_started()` signature.** The no-arg form now emits `DeprecationWarning`. Canonical form adds keyword-only `token` + `episode_ids` parameters, used by the library pipeline to persist the frozen wrap snapshot.
- **`prepare_wrap_package` deprecated.** The pure helper still works but emits `DeprecationWarning`. The canonical entry point is `prepare_wrap(store, *, max_chars=None, staleness_days=None)` which runs the full pipeline: snapshot freezing, handshake token, wrap lifecycle. Scheduled for removal in v0.3.0.

### Added

- **Library canonical + thin transport adapters.** `anneal_memory.continuity` holds the canonical `prepare_wrap` and `validated_save_continuity` pipelines. MCP server and CLI transports parse their transport-native inputs, call the library, and format their transport-native outputs. A cross-transport parity test locks equivalence as a structural invariant — any future drift between the three access patterns breaks CI. This was triggered by Diogenes catching the `validated_save_continuity` implementation diverging from MCP/CLI 12 hours after v0.1.9's `10.5c` doc migration shipped.
- **`Store.get_wrap_history() → list[WrapRecord]`** — public read API for wrap history. Replaces the prior private `Store._conn` access in the history/diff/stats/export CLI subcommands.
- **`Store.load_wrap_snapshot() → WrapSnapshot | None`** — returns the frozen wrap-in-progress snapshot (token + episode IDs) or `None` on the legacy skipped_prepare path. Raises `StoreError` on partial-state integrity failures. External callers can use it for stuck-wrap diagnostics.
- **`Store.get_wrap_started_at() → str | None`** — public read-only accessor for the currently-in-flight wrap timestamp. Used by the new `wrap-status` CLI subcommand.
- **`format_wrap_package_text(result) → str`** — canonical text formatter for the wrap package, extracted from the CLI transport so library users and custom transports can reuse it.
- **`today` parameter on `validated_save_continuity`** — optional deterministic date override for test/experiment runs. Defaults to wall-clock `date.today().isoformat()`.
- **`wrap_token` parameter on `validated_save_continuity`** — optional keyword for explicit commit-atomic verification. When the caller round-trips the token from the prior `prepare_wrap` response, a mismatch raises `ValueError`. Without the token, the frozen-snapshot filter still applies because the library consults the persisted snapshot whenever it's present.
- **Session-handshake token (`10.5c.4`).** `prepare_wrap` mints `uuid.uuid4().hex` and persists it alongside the frozen episode-ID list in store metadata via `Store.wrap_started(token=..., episode_ids=...)`. `validated_save_continuity` filters its re-fetched episode set down to the snapshot and verifies the token via compare-and-swap `UPDATE` inside `wrap_completed`. Episodes recorded between prepare and save stay with `session_id IS NULL` and appear in the NEXT wrap's compression window — no data loss, no silent absorption. Transports round-trip the token: MCP via `Wrap token: <hex>` trailer + `wrap_token` tool arg with JSON schema pattern constraint `^[0-9a-f]{32}$`; CLI via `Wrap token: <hex>` trailer + `--wrap-token` flag.
- **Audit chain-of-custody enrichment.** New audit fields on wrap lifecycle events: `wrap_token` on all three (`wrap_started`, `wrap_cancelled`, `wrap_completed`), plus `wrap_episode_ids` + `wrap_episode_count` on `wrap_started` and `wrap_cancelled`. The audit trail alone can now reconstruct "which episodes went into wrap #N" by matching tokens — useful for cross-process forensics.
- **Two-phase commit pipeline (`10.5c.5`).** `validated_save_continuity` is now fully transactional across continuity.md + meta.json + SQLite DML:
  - New `Store._batch()` context manager accumulates DML inside a single SQLite transaction and defers audit events until after commit.
  - `Store._prepare_continuity_write(text, token_hex)` and `Store._prepare_meta_write(meta, token_hex)` write tmp sidecars with uuid-paired suffixes (`<stem>.<12hex>.md.tmp` / `<stem>.<12hex>.json.tmp`) so concurrent writers cannot collide and operators can group tmp files by prefix on recovery.
  - `Store._fsync_dir(path)` POSIX durability helper fsyncs the containing directory after `os.replace()` so crashes don't revert externalized filenames.
  - `PRAGMA synchronous=FULL` on connection init so `commit()` fsyncs the WAL — required for the 2PC "DB commit ⇒ durable" invariant.
  - DB commit happens BEFORE the file renames. Any crash before commit → full rollback (DB reverts, tmp sidecars can be left or cleaned). Any crash after commit → operator recovery via the preserved tmp sidecars (they are NOT unlinked on post-commit failure paths).
  - Startup orphan detection emits `warnings.warn(UserWarning, ...)` for any orphan tmp files from prior crashed pipelines, grouped by wrap_token prefix, with ready-to-copy `mv` recovery commands. In-flight tmp files matching the currently-active `wrap_token` are filtered out to avoid false positives during another writer's batch window.
  - macOS durability caveat documented: `os.fsync` is weaker than Linux; true durability on macOS would need `fcntl(F_FULLFSYNC)` which stdlib doesn't expose.
  - `record()` and `delete()` are now batch-aware (check `_defer_commit` + route audit through `_audit_log`). Outside a batch their behavior is unchanged.
- **SQLite error wrapping (`10.5c.6`).** `Store._db_boundary(operation)` context manager wraps every SQLite-touching block in every public Store method. Any `sqlite3.DatabaseError` escaping the block (including `OperationalError`, `IntegrityError` after retry exhaustion, `DatabaseError`) surfaces as `StoreDatabaseError` with the underlying exception attached via `__cause__`. The boundary catches `sqlite3.DatabaseError` specifically — not `sqlite3.Error` — so `InterfaceError` API-misuse programming bugs propagate bare with accurate stack traces. `StoreOperation` `Literal` expanded from 4 values to 25 (one per wrapped method); a CI drift test grep-scans `store.py` for every `_db_boundary()` + `operation=` literal and cross-references against `typing.get_args(StoreOperation)` with bidirectional equality assertion — structural drift protection, not discipline.
- **`cause_type_name: str | None`** on `StoreError` / `StoreDatabaseError`. Populated at raise time with `type(exc).__name__` and survives pickle / `copy.deepcopy`. Callers who need to branch on "retryable OperationalError vs non-retryable IntegrityError" after the exception has crossed a process boundary can dispatch on this string. `__cause__` does NOT survive pickle (standard Python limitation) — `cause_type_name` is the pickle-safe bridge.
- **Single generic pickle reconstructor.** `_reconstruct_store_error(cls, ...)` replaces parallel per-subclass functions. Adding new `StoreError` subclasses no longer requires writing a new reconstructor.
- **Idempotent `close()`.** `Store.close()` uses a `_closed: bool` flag with post-close hierarchy guard. Calling `close()` multiple times is safe. We deliberately do NOT null `_conn` — doing so caused an earlier regression where post-close calls raised bare `AttributeError` before reaching the `_db_boundary`, bypassing the hierarchy contract.
- **Three operator-surface CLI subcommands for stuck-wrap recovery:**
  - `anneal-memory wrap-status` — shows current wrap lifecycle state (idle / in-progress, token, started_at).
  - `anneal-memory wrap-cancel` — cancels an in-progress wrap and clears the snapshot.
  - `anneal-memory wrap-token-current` — prints the currently-active wrap_token for scripted recovery flows.
- **CLI: 24 total subcommands.** Operator surface (status, integrity, export, import, history, diff, stats, wrap-status, wrap-cancel, wrap-token-current) plus agent-driven compression (`prepare-wrap`, `save-continuity`) plus the standard episode CRUD and associations/Hebbian/limbic queries. `CLAUDE.md` snippet teaches agents the full cognitive workflow.
- **Framework integration guides verified against live packages (`10.5d`).** Every one of the 12 integration guides has been pinned to a real version and the top 5 were exercised end-to-end with `pip install` + working example code:
  - **LangGraph / LangChain** — verified clean against `langchain 1.2.15` + `langgraph 1.1.6`
  - **CrewAI** — verified against `crewai 1.14.1`; fixed 2 load-bearing bugs (`event.result_summary` → `event.output.raw`, `event.worker_id` → `event.output.agent`) that had silently broken multi-agent attribution
  - **OpenAI Agents SDK** — verified against `openai-agents 0.13.6`; fixed 2 bugs (`response.output[i].text` single-level traversal → nested `item.content[j].text`, `tool.name` unconditional access → `getattr` with class-name fallback for hosted tool types)
  - **Pydantic AI** — verified against `pydantic-ai 1.81.0`; fixed 6 bugs (`wrap_run` attribute → `run`, all hook callbacks keyword-only after `ctx`, param renames `request_ctx` → `request_context` and `tool_call` → `call`, `WrapRunHandler` is zero-arg, `@Agent.tool` class-level decorator replaced with plain function)
  - **smolagents** — verified against `smolagents 1.24.0`; fixed 1 bug (`agent.model([{dict}])` → `agent.model.generate(messages=[ChatMessage(...)])`)
- **TypedDict return shapes.** `PrepareWrapResult`, `SaveContinuityResult`, `WrapPackageDict`, `StalePatternDict`, `WrapSnapshot`. Library users get autocomplete and mypy-level key-typo detection; transport adapters can annotate their boundaries precisely. Drift tests assert the declared keys match the runtime dict keys.

### Changed

- **Tmp sidecar filenames now include unique uuid suffixes** (`<stem>.<12hex>.md.tmp`). Previous naming would collide under concurrent writers.
- **Atomic-write invariant widened.** The broad-cleanup path now fires on `finally`, not just `except OSError` — non-OSError failures no longer leave orphan tmp files.
- **`get_wrap_history` legacy swallow tightened.** Still silently returns `[]` for "no such table" (legacy path preserved), but any other database failure now surfaces as `StoreDatabaseError` with `operation="get_wrap_history"` instead of bare `sqlite3.OperationalError`.
- **Structured context on transport-surfaced errors.** MCP `server.py` and CLI `cli.py` both catch `StoreError` and surface `.operation` / `.path` context in their transport-native error payloads.
- **Constructor wrap.** `Store.__init__` wraps `connect` + PRAGMAs + `_init_schema` + `_warn_orphan_tmp_files` in `_db_boundary("schema_init")` with a local try/except so orphan detection failure doesn't abort construction.
- **Tombstone description corrected.** `integrity.py`'s `delete_episode` integrity description + `Store.delete()` docstring rewritten with the honest field list + GDPR framing + opt-out pointer. Regenerated `tool-integrity.json` + regression test.
- **`_tool_status()` audit health visibility.** New `AuditTrail.stats()` + 4 new `StoreStatus` fields expose audit chain length / verification state / last-event timestamp via both the MCP `status` tool and the CLI `status` subcommand.

### Fixed

- **Diogenes: `test_carried_forward_not_validated skipped_non_today`** — assertion tightened to the invariant it was supposed to guard.
- **Diogenes: CLI `--wrap-token` format validation** — `_WRAP_TOKEN_RE` moved from `server.py` to `store.py`, imported into both transports (fixes cli→server import inversion).
- **`close()` nulling `_conn` regression.** An earlier iteration of the close-idempotency work nulled `self._conn`, which meant post-close calls raised bare `AttributeError` before reaching the hierarchy. Replaced with the `_closed` flag approach that preserves the hierarchy contract.
- **Replay race on wrap commit (L3 codex).** Verify-then-act sequence between `Store.load_wrap_snapshot()` and `Store.wrap_completed()` had a concurrent-replay window. Moved the verify INTO the commit's atomic boundary as a compare-and-swap `UPDATE` inside `wrap_completed`.
- **Auto-prune regression in 2PC.** The commit-ordering change briefly made the post-wrap auto-prune run before the DB commit, which could silently drop episodes on crash. Moved auto-prune to its correct phase.
- **Residual-window data loss in 2PC.** Early Pass-1 of the two-phase commit had a window between file rename and DB commit where a crash could leave the file visible with no DB record. Closed by flipping the phase ordering (DB commit first, then file rename).
- **Recoverability-identity pairing (L3 codex).** Pre-fix, multiple crashed wraps produced unpaired `.md.tmp` + `.json.tmp` files with no way to group them for operator recovery. The new uuid suffixes are derived from the `wrap_token`, so one deterministic batch_id pairs both tmp files — a single structural fix eliminated three surface-level symptoms (pairing + false-positive warnings + warning-text accuracy).

### Test coverage

- **707 passing tests** (from 514 at v0.1.9 — net +193 across the v0.2.0 arc).
- Full 5-review-pass on 10.5c.6: L1 session-code-review (11 findings) + L2 Python stdlib/exception design expert (10) in parallel + L3 3-agent consultation (10) + L3.5 codex post-fix (3, including a HIGH regression the fix itself introduced) + L4 integration semantics (21/21 pass). All 34 findings addressed before release.

### Session lineage

v0.1.9 → v0.2.0 arc: sessions 10.5a (core CLI) → 10.5b (extended CLI + engine removal) → 10.5c (library-first positioning) → 10.5c.1 (rule-of-three elimination) → 10.5c.2 (doc canonical-entry migration) → 10.5c.3 (exception hierarchy + TypedDict) → 10.5c.4 (session-handshake token) → 10.5c.4a (operator surface) → 10.5c.5 (two-phase commit) → 10.5c.6 (SQLite error wrapping) → 10.5d (framework integration verification).

---

## [0.1.9] — 2026-04-08

Pre-v0.2.0 stability release. `delete_episode` MCP tool, README rewrite (grounding thesis + production failures), security Layer 2 (host-verifiable manifest resource), Diogenes warmup fixes.

## [0.1.8] — 2026-04-07

Hebbian associations (v2 Deep Hebbian — co-citation during consolidation) + limbic layer (affective state tagging on associations).

## [0.1.6] — 2026-04-03

Hardening release.

## [0.1.5] — 2026-04-02

Compliance layer: hash-chained JSONL audit trail, weekly rotation, actor identity, content-hash-only (GDPR-compatible), `on_event` callback.

## [0.1.0] — 2026-04-01

Initial PyPI release. Two-layer memory (episodic SQLite + continuity markdown), graduation pipeline, immune system, MCP server (stdio transport, newline-delimited JSON).

[0.2.0]: https://github.com/phillipclapham/anneal-memory/compare/v0.1.9...v0.2.0
[0.1.9]: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.1.9
[0.1.8]: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.1.8
[0.1.6]: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.1.6
[0.1.5]: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.1.5
[0.1.0]: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.1.0
