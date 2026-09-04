# anneal-memory — Project Context

> Four-layer memory architecture (episodic + continuity + Hebbian + limbic) with cross-layer immune system for AI agents. Episodes compress into identity.
>
> ⚠ **THIS FILE IS ORIENTATION, NOT CURRENT STATE — and its reach is wider than it used to be.**
> It moved out of flow's project memory into this repo on 2026-09-04 (`e20ada6`), so it is now the
> repo's only `CLAUDE.md`: it orients every session here, and it is what anyone who clones or browses
> the repo reads. Staleness that used to be contained to one flow session now travels. Whether it also
> reaches the published artifacts is **not stated here** — read `[tool.hatch.build.targets.sdist]` and
> `[tool.hatch.build.targets.wheel]` in `pyproject.toml`, which are the only authority and have changed.
> **Canonical current state is `next_steps.md`.** Its sibling `projectbrief.md` is v0.5.0-era and says
> so. Everything in this file dated before today is a **receipt** — "this was true on that date" —
> never "this is true now".
>
> ⚖ **NUMBERS HERE ARE DERIVATIONS, NOT TRANSCRIPTIONS.** Every count in this file rotted at least
> once (tool count, CLI subcommand count — twice, disagreeing with itself — published version, test
> count). A copied number goes stale silently; a command cannot. Where a number is worth having, the
> command that produces it is written instead. If you find a bare number below, it is a receipt with
> a date on it, not a fact about now.
>
> Updated: 2026-09-04

## Orientation — run these, do not recall them

| Question | Live source |
|---|---|
| Version at HEAD | `grep __version__ anneal_memory/__init__.py` (and `version` in `pyproject.toml` — they must agree) |
| Latest published / all releases | `curl -s https://pypi.org/pypi/anneal-memory/json \| python3 -c "import json,sys;print(json.load(sys.stdin)['info']['version'])"` |
| MCP tools (names + count) | `anneal_memory/tool-integrity.json` — the shipped manifest, guarded by `tests/test_integrity.py::TestShippedManifest::test_shipped_manifest_verifies` (+ `..._covers_all_current_tools`). The repo-root `tool-integrity.json` is a byte-identical copy. |
| MCP resources | `RESOURCES` in `anneal_memory/server.py` |
| CLI subcommands | `anneal-memory --help` (⚠ the choice list counts `recall` as its own entry; it is an alias of `search`, so distinct subcommands = choices − 1) |
| Test count | `python3 -m pytest --collect-only -q \| tail -1` |
| What shipped, when, and what broke | `CHANGELOG.md` + `git log --oneline` |
| What is next / live design state | `project_memory/next_steps.md` |
| Open review findings | `project_memory/diogenes_*.md` (newest wins) |

## File Limit — a ROUTING target for `next_steps.md`, not a deletion quota

The nominal cap on `next_steps.md` is **200 lines**; the temporary 450-line override (Apr 10 2026)
was retired 2026-06-25. Check the reality with `wc -l project_memory/next_steps.md` — do not assume
it is under the cap, because it routinely is not, and that is sometimes correct. When it is over,
the only valid moves are **route** (shipped history → `CHANGELOG.md` + git + `COMPLETED_SESSIONS_ARCHIVE.md`;
reasoning → the design blocks; receipts → the archive) or **de-duplicate**. ⛔ A line budget may never
cost load-bearing pickup state — if the file is long because a lot is genuinely live, it stays long
and the budget yields. `next_steps.md` is the authority on its own length.

## Status — DATED RECEIPTS, newest first

⚠ **Nothing in this section is a claim about now.** Each entry was accurate on its own date and has
not been re-verified since. For current state use the Orientation table above and `next_steps.md`.
The 2026-06-30 entry below in particular is **superseded**: the 0.9.x line has published well past
0.9.6 (check PyPI), and HEAD has moved on (check `__init__.py`).

**SUPERSEDED — as of 2026-06-30: 0.9.6 SHIPPED PUBLIC — first PyPI publish of the 0.9.x line, DECOUPLED from Slice C (spore-222).** *(0.9.6 has NOT been the latest release since; see the Orientation table.)* `4a43c83`, tag `v0.9.6`, PyPI + GitHub release (https://pypi.org/project/anneal-memory/0.9.6/). PyPI was at 0.8.5; the whole 0.9.x line (0.9.0 shadow → 0.9.5) was main/editable-only, never published → collapsed into one public release for the Chris bundle (July 7). Bumped 0.9.5→0.9.6 (new public `anneal_memory.sessions` module/API — AM-CONSOLIDATE-EFFERENT — landed after the 0.9.5 label). Ships Slice B (`pattern_associations.py`) in SHADOW MODE (additive/inert; nothing reads the graph for recall). codex L3 release/packaging review (non-replaceable) → 1 Medium: the AM-WRAP-GENERATED migration entry was keyed at the never-published 0.9.5 → re-keyed to 0.9.6 (fail-safe; a local-0.9.5-acked operator upgrading now surfaces the wrap-protocol retirement). L4 e2e: a clean-venv install of the wheel AND a real `pip install anneal-memory==0.9.6` from PyPI both pass the full smoke (version, imports, shadow-table creation, prepare_wrap, the validated_save_continuity immune pipeline, the sessions/consolidate-efferent API, SporeStore). 1642 tests green, mypy clean, twine check passed. Glama/MCP-registry sync is auto. **▶ HANDOFF to the flow-repo/Levain convo: bump the Levain anneal pin `>=0.9.4`→`>=0.9.6`. Slice C (graph-consuming recall) stays the live track, §9.2-oracle-gated.** Detail → `next_steps.md` top.

**Prior (2026-06-23): AM-SPORE-CAS SHIPPED — `expect_disposition` CAS on `SporeStore.ascend`** (anneal `a3c9a2f`, `0.9.3`→`0.9.4`; shipped public in `v0.9.6`). Mirrors `update`'s optimistic CAS onto the upward-resolve path, checked inside the resolve transaction → closes the cross-process read-then-resolve TOCTOU a separate-step disposition guard otherwise has (a confirmed `loop→note` flip can no longer slip a now-note past a host's note-ascend guard). Additive + backward-compatible (`_UNSET` skips). **Surfaced by Levain's KEEP-NOTE LIFECYCLE L3 (codex HIGH); closed the same night per Phill's "deal with any anneal HIGH tonight"; codex re-verified "No findings."** Levain `_apply_spore_verb` threads the guard-read disposition through it; Levain pin → `>=0.9.4`. 4 ascend-CAS tests, anneal 1590 green, CHANGELOG [Unreleased]. Next-field CAS sibling = deferred MED (`spore-169`, cold). Detail → `next_steps.md` top.

**Prior (2026-06-21 PM): `spore-104` dep-1 SHIPPED — the pattern-graph PROJECTION-CHECKPOINT / high-water-mark primitive** (anneal `e9d1191` + flow `3729f54`, THIN pin-not-rebuild, full 4-layer apparatus; rides the 0.9.0 shadow → ships with Slice C, NO tag/PyPI). A `(projection_version, high_water_mark)` checkpoint over the pattern-association read-model so a Slice-C retrieval receipt PINS which graph state produced it: `high_water_mark` = a monotonic revision counter bumped per committed edge mutation (seed/drain/rename/sever/gc, gated, in-txn), `projection_version` = constant 1 (a future blue/green rebuild bumps it). Flow `recall_injection_hook` stamps the receipt from the SAME read-only Store open serving recall (no extra read; shadow invariant intact). **BOTH Slice-C upstream deps now DONE** (dep-2 source-tagging rode the AM-RECALL-IDF ship earlier today; `spore-104` composted). Apparatus highlight: codex L3 caught a `busy_timeout` FACT both Claude-lineage layers got backwards (read_only inherits Python's 5s default, not "none") → comment corrected + `spore-148` (AM-READONLY-FAILFAST). anneal 1586 tests + flow 30 + mypy clean. **▶ NEXT = the §9.2 A/B replay harness → re-measure the cleaned graph → Slice-C go/no-go — MEASURE receipts-per-hwm-bucket FIRST** (the pin-not-rebuild sufficiency condition).

**Prior (2026-06-21 AM): AM-RECALL-IDF SHIPPED** (the recall-PRECISION blocker CLEARED; `57511eb` 0.9.3 + flow `3ffaae7`) — corpus-IDF keyword weight + a √N distinctiveness anchor + per-pattern `RelevantPattern.source` (`spore-104` dep-2). Mechanism corrected (term-frequency bias, not degree-bias). Detail → `next_steps.md` top.

**Prior (2026-06-15): AM-PYTYPED SHIPPED + PUSHED** (`spore-097` done; main `7b638a8`, rides the 0.9.0 shadow → ships with Slice C, NO tag/PyPI). Empty `anneal_memory/py.typed` (PEP 561) — ships in the hatchling wheel automatically (verified `unzip -l`). Typed consumers (Levain/flow/OpenHands) no longer get `import-untyped`; mypy now checks anneal calls. Flipping it ON surfaced ONE anneal-side type-ergonomics gap → widened public `DESCEND_BY_TYPE`/`ASCEND_BY_TYPE` keys Literal→`str` (lookup tables queried by a runtime spore-type string from JSON; dogfood-passes). Consumer tails fixed same session: Levain `181acbe` (descend/ascend narrowing + `Literal` verb), flow `fba0168` (stale `patterns_extracted`→`episodes_compressed`). Apparatus L1 + L3-codex CLEAN; anneal mypy clean (17 files), 1540 tests. Levain template-mypy-config follow-up → `spore-099`.

**Prior (2026-06-15): spore-091 (AM-CONTLOCK hardening) SHIPPED + PUSHED** (main `a07f2bb`, rides the 0.9.0 shadow → ships with Slice C, NO tag/PyPI). The weekly cross-slice codex review's 2 latent gaps: `continuity_lock` now `.resolve()`s the `.lock` sidecar INSIDE the primitive (a symlinked continuity dir/file gave two cooperating processes two `.lock` spellings → silent non-serialization; invariant moved into the primitive, docstring overclaim corrected) + a new `require=` strict mode raises `ContinuityLockUnavailable` instead of degrading (anneal's own save keeps best-effort `require=False`; Levain's State write passes `require=True` → fails CLOSED). CM now yields `acquired:bool`. +6 tests (1540 pass). **Levain consumer SHIPPED same session (`7f5f15d`).**

**Prior (2026-06-14 eve): AM-SNAPSHOT ① SHIPPED — the durable recovery oracle; the Levain-2b-i coordination surface is now CLOSED** (main `bf3f0d6`, rides the 0.9.0 shadow → ships with Slice C, NO tag/PyPI). Persists the continuity `content_hash` + `<token12>-<uuid8>` `pair_id` in the `wraps` row INSIDE the Phase-2 commit (additive NULLable cols + idempotent migration), so orphan recovery SELF-CLASSIFIES (`committed_verified`/`committed_unverifiable`/`committed_hash_mismatch`/`debris`/`inconclusive`) even when a crash beats the Phase-4 `continuity_saved` audit. `_warn_orphan_tmp_files` emits a confident verdict + exact command but NEVER auto-acts; debris = a STRUCTURAL `<12hex>-<8hex>` match + STEM-ANCHORED parse (fail closed, never a false "safe to rm"); continuity tmp `newline=""` (byte-stable hash). Full 4-layer apparatus — **codex L3 caught an extra-dotted-parse fail-open both Claude layers missed** + a half-oracle gap; **L4 proved pipeline→crash→oracle→self-classify→recover LIVE.** +24 tests (1530 pass), mypy clean. Closes the AM-CONTLOCK ① follow-up. **gap ② — restore ❌ KILLED** (architecturally incoherent: the continuity text is a PROJECTION of the 5-layer store; restoring desyncs it) **; content-store ✅ RE-SCOPED 2026-06-15 → a COMMITTED projection-history viewer** (daemon disconfirmation `1e8bc012` → Phill consumer-reframe; the consumer is the operator in the governance-visible plane → the history teaches the architecture; v0 digest-delta now / v1 full content store self-paced & NOT Tony-gated — `spore-093`). **▶ NEXT anneal = the live track is Slice B→C** (AM-LINKGATE-DECAY shadow → graph-consuming recall, gated on the ~Jun-26 oracle `spore-085`); the **content-store v1 is committed-but-self-paced** (does NOT block B→C). Detail → `next_steps.md` 🆕 top. *(Prior: AM-CONTLOCK `7ab7991` — the shared `continuity_lock`.)*

**Prior (2026-06-12):** **v0.8.4 SHIPPED** on PyPI + GitHub (`v0.8.4` / main `4899423`, tag `v0.8.4`, argushub anneal-venv 0.8.4 fleet-deployed; flow runs the editable install). Two releases today: **0.8.3 = AM-LINKGATE** (the associative-layer gate-half — AM-WARN Signal C nudges single-id under-wiring as a discipline reminder, not a structural defect, + `prepare_wrap` co-citation guidance) and **0.8.4 = the `spore-081` fix** (`upsert_pattern_history` anchored `last_seen_at` to wall-clock → broke the AM-PRESERVE warm gate under deterministic/backdated + local/UTC-evening runs; now anchors to the pipeline's logical date). Both full 4-layer apparatus. **anneal main fully green (1463 tests).** The 0.6→0.8 arc landed the **crystallized-pattern tier (AM-CRYSTAL)** end-to-end: store-half (0.6.0) → opt-in + CLI (0.7.0) → AM-PRESERVE-VS-SYCOPHANCY (0.7.2) → **AM-CRYSTAL-RECALL Hebbian backend (0.8.0)** → CLI parity (0.8.1) → **MCP tools (0.8.2)** → **AM-LINKGATE (0.8.3)** → **AM-PRESERVE determinism fix (0.8.4)**. **→ UPDATE 2026-06-12 PM: main carries `0.9.0` (UNRELEASED — release deferred to Slice C).** v0.8.5 shipped Slice A (AM-PROVENANCE). Then **Slice B (AM-LINKGATE-DECAY) BUILT — SHADOW MODE — 0.9.0** (main `75b97ae`, pushed; NO tag/PyPI/fleet): the cortical pattern-association graph (graduated patterns linked by stable NAMES; co-graduation seeding + idempotent co-surface drain w/ per-session burst-damp + provenance gate; lazy calendar decay; rename/homonym lifecycle; telemetry + CLI). Producer (recall hook → spool) + drain (`anneal_dualwrite`) + oracle (`pattern_graph_oracle.py`) are flow-side. Full 4-layer apparatus — codex L3 caught 2 HIGH (batch-rollback seeding → post-commit txn; producer/rotation race → deferred-unlink), all fixed w/ regression tests; L4 11/11 incl. the shadow invariant verified live. **+ AM-STATUS-HARDEN (`spore-084`, main `2471ef0`): guarded the `status()` continuity read** (a corrupt/non-UTF8 continuity was faulting `status()` itself → broke any consumer incl. the Levain v2 dashboard; degrades to `None` now). **1504 tests, mypy clean. flow runs editable → shadow phase LIVE.** **▶ RELEASE (tag `v0.9.0` + PyPI + GitHub release + argushub fleet) DEFERRED to Slice C** (an unconsumed shadow subsystem = positioning-ahead-of-product; PyPI stays 0.8.5). **▶ NEXT = run the shadow phase (periodic `scripts/pattern_graph_oracle.py`, gate = `spore-085`) → Slice C (graph-CONSUMING recall) GATED on the oracle.** **Levain v2 Slice 1.5 runs in PARALLEL** (separate convo; Slice B NOT a Levain blocker). Deferred backlog: AM-PATTERN-ALIAS (`spore-047`, unify w/ Slice B's `pattern_aliases`), AM-GAUNTLET (`spore-048`), MCP crystal *lifecycle* (write) tools.

**Historical (v0.2.0-era, stale — every number in this paragraph is an April-2026 receipt; run the Orientation table for live values):** v0.1.9 on PyPI + GitHub. **v0.2.0 release UNBLOCKED through 10.5c.6.** CLI shipped (24 subcommands incl. operator surface — *as of then; `anneal-memory --help` is the live count and it is higher*). 707 tests. Sessions 1–10.5c.6 COMPLETE. Engine REMOVED (thesis-critical). **Rule-of-three eliminated (10.5c.1):** library canonical + thin transport adapters. Cross-transport parity test. **Library-first positioning in docs (10.5c.2).** **Exception hierarchy + TypedDict shapes + atomic-write invariant (10.5c.3).** **Session-handshake token + frozen snapshot + commit-atomic CAS (10.5c.4):** prepare/save TOCTOU window closed; audit chain-of-custody enrichment. **Operator surface subcommands (10.5c.4a):** `wrap-status` / `wrap-cancel` / `wrap-token-current` for stuck-wrap recovery. **Two-phase commit pipeline (10.5c.5):** file+DB atomicity via uuid-suffixed tmp sidecars paired by wrap_token prefix, batched SQLite transaction via new `Store._batch()` context manager, atomic rename ordering (DB-first-then-files), `_fsync_dir()` POSIX durability helper, `PRAGMA synchronous=FULL` WAL durability, startup orphan detection with active-wrap_token cross-reference to skip in-flight false positives, `db_committed` flag in pipeline preserves tmp files post-commit for operator recovery, `_audit_log()` batch-aware audit helper that queues events during batch and flushes with exception swallowing after commit. Full 4-layer review surfaced 20+ findings across L1 (auto-prune regression, residual-window data loss, cli→server import inversion, partial-state audit), L2 (CAS rollback ownership, audit-flush destroying committed state, fsync(dir), SIGINT poisoning, synchronous=FULL), L3 convergent (tmp collision, orphan detection gap), L3 per-agent (commit-failure rollback, Phase 4 audit try/except, -O safety, record() batch-aware) and a codex-retry pass (recoverability-identity pairing, warning text content, in-flight false positive filter) — all fixed + regression-gated. 36 new tests across 2PC + post-review + L3-fix + codex-fix classes. 10.5c.4a filed-then-shipped in the same session. Next: 10.5c.6 (optional SQLite error wrapping) → 10.5d (framework testing) → v0.2.0 release → Session 10 (identity experiment).

## Key Files

⚠ Paths are **relative to this repo root** unless prefixed `flow:`. Before 2026-09-04 this file lived
in flow's tree and its bare `contexts/…` paths resolved there; from here they do not. Anything marked
`flow:` is in `~/Briefcase/flow/` and is **not** part of this repo or its sdist — an adopter cannot
open it.

- **Code repo:** this repo, `~/Briefcase/anneal-memory/`. (`~/Documents/anneal-memory` is a symlink to it — legacy, still resolves; prefer the real path.)
- **Docs:** `docs/` — `library-quickstart.md`, `architecture.md`, and the integration guides in `docs/integrations/` (`ls docs/integrations/*.md | wc -l` for the count). All guides use the canonical `prepare_wrap` entry point; the 10.5c.2 migration is done.
- **Competitive intel:** `flow:contexts/archive/anneal_memory_competitive_intel.md`
- **Directory submission guide:** `flow:contexts/archive/anneal_memory_directory_submissions.md`
- **EU AI Act analysis:** `flow:contexts/eu_ai_act_analysis.md`
- **Experiment results:** `project_memory/experiment_results.md` (Claude-owned forensic analysis after each testbed run)
- **Session history:** `project_memory/COMPLETED_SESSIONS_ARCHIVE.md`

## Architecture

Four cognitive layers: episodic (SQLite) + continuity (markdown) + Hebbian associations (co-citation links during graduation, v0.1.8) + limbic (affective state tagging on associations, v0.1.8). **Single canonical pipeline** in `continuity.py` — `prepare_wrap(store)` and `validated_save_continuity(store, text)` are the only implementations. MCP and CLI transport adapters parse their transport-native input, delegate to the library, and format their transport-native output. **Surface sizes are derived, not stated here** — MCP tools from `anneal_memory/tool-integrity.json`, MCP resources from `server.py`'s `RESOURCES`, CLI subcommands from `anneal-memory --help` (see Orientation). Zero runtime dependencies — Python stdlib only (`[project].dependencies` in `pyproject.toml` is empty; `requires-python >= 3.10`). Newline-delimited JSON stdio transport; `_PROTOCOL_VERSION` in `server.py` is the authority (currently the MCP `2024-11-05` spec).

## Positioning

"The only AI agent memory system with an immune system." Library-first: the library is the product, CLI and MCP are interfaces to it. This is now architecturally true, not just a marketing claim — transports call the library functions directly. Three audiences see three things: developers (best agent memory DX), researchers (empirical identity thesis validation), enterprise (EU AI Act audit infrastructure). No competitor has citation-validated graduation, the explanation-grounding check, or active demotion of ungrounded citations. (Use the primitive names the README actually enumerates — `README.md` → *"Structural immune-system primitives at the citation layer"* and the compression-package primitives that follow it — rather than a count carried here. Verified 2026-09-04: "anti-inbreeding defense" and "principle demotion" appear nowhere in `README.md`, `docs/`, or `anneal_memory/`; they were retired 2026-05-21 as claims that mapped to no code path, and the retirement held.)

## Key Context

- **Canonical entry points (use these, not the low-level primitives):**
  - Library: `prepare_wrap(store, ...)` → agent compresses → `validated_save_continuity(store, text, ...)`.
    ⚠ **Do not copy signatures out of this file — read them off the code.** Both have grown keyword-only
    parameters since this section was written (`crystal_store`, `session_id`, `allow_shrink`,
    `carryforward_cold_days`). Live form: `python3 -c "import inspect, anneal_memory.continuity as c; print(inspect.signature(c.prepare_wrap)); print(inspect.signature(c.validated_save_continuity))"`
  - ⛔ `prepare_wrap_package()` is **GONE — removed in v0.3.0**, not deprecated-but-present. `from anneal_memory import prepare_wrap_package` raises `ImportError`. Use `prepare_wrap(store, ...)`. (The private `_build_wrap_package()` exists with no stability guarantee.)
  - Bare `store.save_continuity()` is a file write that bypasses graduation, associations, decay, meta, and wrap_completed. Docstring carries a prominent warning. Only legitimate caller is the library pipeline itself.
- `validated_save_continuity` accepts an optional `today` keyword for deterministic test/experiment runs; defaults to wall-clock `date.today().isoformat()`.
- `validated_save_continuity` accepts an optional `wrap_token` keyword for explicit commit-atomic verification. When the caller round-trips the token from the prior `prepare_wrap` response, a mismatch raises `ValueError`. Without the token, the frozen-snapshot filter still applies because the library consults the persisted snapshot whenever it's present.
- `Store.get_wrap_history() → list[WrapRecord]` is the public read API for wrap history (history/diff/stats/export CLI commands all use it). Replaces the old private `_conn` access.
- `Store.load_wrap_snapshot() → WrapSnapshot | None` returns the frozen wrap-in-progress snapshot (token + episode IDs) or None on the legacy skipped_prepare path. Raises `StoreError` on partial-state integrity failures. Used internally by `validated_save_continuity`; external callers can use it for stuck-wrap diagnostics.
- **TOCTOU window is CLOSED (10.5c.4):** `prepare_wrap` mints a session-handshake token and persists the frozen episode ID list in store metadata; `validated_save_continuity` filters its re-fetched episode set down to the snapshot and verifies the token via compare-and-swap UPDATE inside `wrap_completed`. Episodes recorded between prepare and save stay with `session_id IS NULL` and appear in the NEXT wrap's compression window. Framework integrations can still optionally round-trip the `wrap_token` for explicit mismatch detection. The remaining pipeline-atomicity gap (mid-pipeline crash between continuity file write and DB metadata commit) is the 10.5c.5 two-phase-commit work.
- Rewrite informed by FlowScript ContinuityManager — same concepts, fresh codebase.
- A simplified FlowScript subset is used for continuity compression. ⚠ The marker set is defined in code, not here — `anneal_memory/graduation.py` (the recognizer regex, with the marker-prefix comment block) and `anneal_memory/continuity.py:947` (*"the marker KINDS are a closed set"*) are the authority. A previous "9-marker" count in this file could not be reproduced from either.
- CLAUDE.md snippets (MCP + CLI) are the most important DX artifacts — teach agents the full cognitive workflow.
- Docker testbed *(2026-04-era receipt; NOT re-verified — the image is a local artifact, `docker images | grep anneal` is the only thing that can confirm it still exists, and the AM version baked into it is whatever was current when it was built, not what is on PyPI now)*: `anneal-testbed` image (native CC + AM 0.1.9 at build time, persistent volumes). Workspace: `~/Briefcase/anneal-testbed/` (`~/Documents/anneal-testbed` is a symlink to it).
- Testbed launch: `docker run -it -v anneal-auth:/root/.claude -v anneal-data:/root/.anneal-memory -v ~/Briefcase/anneal-testbed:/workspace anneal-testbed`
- !! Docker testbed entrypoint: MUST be `/bin/bash`. `docker commit` with `--entrypoint` overrides capture the override as the new default — always use `--change 'ENTRYPOINT ["/bin/bash"]'` when committing.
- Compliance layer SHIPPED (v0.1.5): hash-chained JSONL audit trail, 4-layer reviewed, actor identity, content-hash-only, `on_event` callback for cloud. Positioning: "tamper-evident audit infrastructure" NOT "compliance-grade" (per consultation review).
- ✓ EU AI Act provision-by-provision analysis COMPLETE at `flow:contexts/eu_ai_act_analysis.md` (flow repo — not in this repo or its sdist). Covers Articles 9, 12, 13, 14, 17, 26, 86. ⚠ The August 2, 2026 compliance deadline it was written against has **passed** — re-read the analysis before quoting it as forward-looking. What this repo actually claims about the Act is in `README.md` (Article 12 traceability; audit *infrastructure*, not certification) and `docs/architecture.md`.
- Two-layer compliance model: Layer 1 = memory audit (shipped). Layer 2 = compliance proxy (future, MCP transport interception).
- AnnealCloud future: compliance proxy hosting + witness services + encryption + external timestamps + multi-agent observability (parked until demand).

## Release Checklist (v0.2.0) — ⛔ ARCHIVED, SHIPPED, DO NOT FOLLOW

⛔ **v0.2.0 shipped 2026-04-13 and many releases have shipped since.** This block is kept verbatim as
the v0.2.0-era working record. **Every number, version, `✓ pending commit`, "New:"/"now"/"still"
statement, and API shape below is an April-2026 receipt** — several are already known-wrong at HEAD
(e.g. `StoreOperation` is stated as "expanded from 4 values to 21"; live is materially larger —
`python3 -c "import typing;from anneal_memory.store import StoreOperation;print(len(typing.get_args(StoreOperation)))"`
— and `_fsync_dir` is a module-level function in `store.py`, not a `Store` method as listed).
**Do not read anything below as current, and do not use it as a release procedure.** For what
actually shipped, read `CHANGELOG.md`; for how to cut a release, read `next_steps.md` and the
CHANGELOG's most recent entries.

1. Confirm 10.5c.2 landed (all guides migrated to `prepare_wrap(store, ...)`) — ✓ commit `2771a57`
2. Confirm 10.5c.3 landed (exception hierarchy + TypedDict + atomic writes) — ✓ commit `934a3ce`
3. Confirm 10.5c.4 landed (session-handshake token + CAS + audit enrichment) — ✓ commit `cadf9e9`
4. Confirm 10.5c.4a landed (operator surface: wrap-status/wrap-cancel/wrap-token-current) — ✓ pending commit
5. Confirm 10.5c.5 landed (file/DB two-phase commit + L1/L2/L3/L4 review fixes) — ✓ pending commit
6. Confirm 10.5c.6 landed (SQLite error wrapping — StoreDatabaseError + full 5-review-pass + deferral-fix pass) — ✓ pending commit
7. Run 10.5d framework tests (top 5 frameworks, verify integration guides actually work)
8. Version bump in `__init__.py`, `pyproject.toml`, `server.json`
8. Create CHANGELOG.md covering:
   - Engine removal (v0.1.9→v0.2.0 breaking note)
   - CLI shipped
   - Library canonical + thin transports
   - `prepare_wrap` canonical entry point
   - `WrapRecord` + `get_wrap_history()`
   - `today` parameter on `validated_save_continuity`
   - Widened return dict
   - **10.5c.3 BREAKING CHANGES:**
     - `validated_save_continuity` return: `wrap_result` is `dict[str, Any]` not `WrapResult` dataclass
     - `StoreError` no longer subclasses `OSError`; inherits from `AnnealMemoryError(Exception)` — callers catching `OSError` must migrate to `except StoreError` or `except AnnealMemoryError`
     - `prepare_wrap_package` emits `DeprecationWarning`, scheduled for removal in v0.3.0
   - New: `AnnealMemoryError` library exception base + `StoreOperation` Literal type
   - New: TypedDict return shapes (`PrepareWrapResult`, `SaveContinuityResult`, `WrapPackageDict`, `StalePatternDict`)
   - Fix: atomic-write invariant now holds for non-OSError failures
   - Transport adapters surface structured `StoreError.operation` / `.path` context
   - **10.5c.4 BREAKING CHANGES:**
     - `prepare_wrap` return adds `wrap_token: str | None` field (TypedDict drift for `PrepareWrapResult` consumers).
     - `validated_save_continuity` adds optional `wrap_token` kwarg.
     - `Store.wrap_started()` with no args emits `DeprecationWarning`; canonical form adds `token` + `episode_ids` keyword-only parameters.
     - `Store.wrap_completed()` adds optional `episode_ids` + `wrap_token` keyword-only parameters.
     - New `Store.load_wrap_snapshot() → WrapSnapshot | None` primitive.
     - New `WrapSnapshot` TypedDict in `anneal_memory.types`.
     - MCP `save_continuity` tool gains optional `wrap_token` argument (JSON schema pattern constraint `^[0-9a-f]{32}$`).
     - CLI `save-continuity` subcommand gains `--wrap-token` flag.
     - `load_wrap_snapshot` raises `StoreError` on partial-state (wrap_started_at set but wrap_token empty). Behavior change for callers using no-arg `wrap_started()` form followed by a canonical save — the canonical pipeline is unaffected.
     - TOCTOU prepare/save window is now CLOSED at the semantic layer: episodes recorded between `prepare_wrap` and `validated_save_continuity` are deferred to the next wrap (no data loss, no silent absorption).
     - New audit fields on wrap lifecycle events: `wrap_token` on all three (`wrap_started`, `wrap_cancelled`, `wrap_completed`), plus `wrap_episode_ids` + `wrap_episode_count` on `wrap_started` and `wrap_cancelled`. Audit trail alone can now reconstruct "which episodes went into wrap #N" by matching tokens.
   - **10.5c.4a NEW FEATURES (not breaking):**
     - New CLI subcommands: `wrap-status`, `wrap-cancel`, `wrap-token-current` for stuck-wrap operator recovery.
     - New `Store.get_wrap_started_at() → str | None` public accessor (read-only).
   - **10.5c.6 NEW FEATURES (non-breaking hierarchy expansion):**
     - New library exception class: `StoreDatabaseError(StoreError)` — raised when a SQLite operation fails inside the store (locked DB, disk full, integrity constraint violations after retries, corruption). Subclass of `StoreError` so every existing `except StoreError` handler catches it unchanged; callers who want to branch on "retryable DB error vs non-retryable file error" can catch the subclass specifically.
     - New internal primitive: `Store._db_boundary(operation)` context manager. Wraps every SQLite-touching block in every public Store method — record, get, delete, recall, episodes_since_wrap, status, wrap_started, wrap_cancelled, get_wrap_started_at, wrap_completed, load_wrap_snapshot, get_wrap_history, record_associations, decay_associations, get_associations, get_association_context, association_stats, prune, the constructor schema_init path, and the `_batch()` outer commit. Any `sqlite3.Error` (OperationalError, IntegrityError, DatabaseError, etc.) escaping the block surfaces as `StoreDatabaseError` with the underlying error attached via `__cause__`.
     - `StoreOperation` Literal expanded from 4 values to 21: adds `record`, `get`, `delete`, `recall`, `episodes_since_wrap`, `status`, `wrap_started`, `wrap_cancelled`, `get_wrap_started_at`, `get_wrap_history`, `record_associations`, `decay_associations`, `get_associations`, `get_association_context`, `association_stats`, `prune`, `schema_init`, `batch_commit`. New raise sites MUST add their identifier to the alias before raising (soft contract until mypy-in-CI lands).
     - Pickle/copy round-trip for `StoreDatabaseError`: parallel `_reconstruct_store_database_error` preserves subclass identity across `pickle.dumps` / `pickle.loads` / `copy.copy` / `copy.deepcopy`. `__cause__` chain does NOT survive pickle (standard Python limitation — see `StoreError` docstring).
     - `get_wrap_history` docstring updated — still silently returns `[]` for "no such table" (legacy path preserved), but any other database failure now surfaces as `StoreDatabaseError` with operation `get_wrap_history` instead of bare `sqlite3.OperationalError`.
     - `test_batch_commit_failure_rolls_back` migrated: now asserts `StoreDatabaseError` with `operation="batch_commit"` and verifies `__cause__` preserves the original `sqlite3.OperationalError`. Pre-existing test behavior drift is the only test-side change required by the hierarchy expansion.
     - 22 new regression tests in `TestDbBoundaryErrorWrapping` covering: every wrapped public method (individual fault injection via `_FlakyExecuteProxy`), the hierarchy catchpoint for both `StoreError` and `AnnealMemoryError`, the `__cause__` chain preservation (including for `IntegrityError` after retry exhaustion), pickle/copy round-trip, the constructor `schema_init` wrap, and one real-database-locked integration smoke test with two concurrent Store instances.
     - `docs/library-quickstart.md` gains a new `## Error Handling` section documenting the hierarchy, the `.operation` / `.path` / `__cause__` fields, the pickle behavior, and a code snippet showing how to branch on `StoreDatabaseError` vs `StoreError` at the catch boundary.
     - No breaking changes. No new operation names for existing methods. No caller-visible behavior change for anything that doesn't fail — the wrapping only affects the exception type surfaced when a failure occurs.
   - **10.5c.5 NEW FEATURES + INTERNAL CHANGES (no public API breaks):**
     - `validated_save_continuity` is now fully transactional across continuity.md + meta.json + SQLite DML. Any failure before the final renames restores the exact pre-wrap state (DB rollback + tmp cleanup). Post-commit failures PRESERVE tmp sidecars for operator recovery (do NOT unlink them).
     - New internal primitives on `Store` (underscore-prefixed, not part of public API): `_batch()` context manager for multi-step batched commits with audit deferral; `_prepare_continuity_write(text, token_hex)` / `_prepare_meta_write(meta, token_hex)` for tmp-sidecar writes; `_fsync_dir(path)` POSIX durability helper; `_audit_log(event, payload, **kwargs)` batch-aware audit helper; `_defer_commit` flag + `_deferred_audits` instance state; `_find_orphan_tmp_files()` + `_warn_orphan_tmp_files()` for startup detection.
     - Tmp sidecar filenames now use unique uuid suffixes (`<stem>.<12hex>.md.tmp` / `<stem>.<12hex>.json.tmp`) to avoid concurrent-writer collision. Paired pipeline writes share a wrap-token-derived suffix so continuity+meta tmps are operator-pairable by prefix.
     - `PRAGMA synchronous=FULL` set on connection init so `commit()` fsyncs the WAL — required for the 2PC "DB commit ⇒ durable" invariant.
     - Startup orphan detection emits `warnings.warn(UserWarning, ...)` for any orphan tmp files from prior crashed pipelines, with paired-file grouping + ready-to-copy `mv` recovery commands. In-flight tmp files (matching the currently-active `wrap_token`) are filtered out to avoid false positives during another writer's batch window.
     - `record()` and `delete()` are now batch-aware (check `_defer_commit` + route audit through `_audit_log`). They retain their immediate-commit behavior outside a batch — no caller-visible change.
     - `_WRAP_TOKEN_RE` shape constant moved from `server.py` to `store.py` so both CLI and MCP transports import from the store layer (fixes cli→server import inversion).
     - New `DeprecationWarning`: none. Explicit `StoreError` checks (not `assert`) for `meta_tmp is None` and `wrap_result is None` internal invariants — runs under `python -O`.
     - Not thread-safe, not task-safe, not reentrant — documented in `_batch()` docstring. Single-process design invariant.
     - macOS durability caveat documented: `os.fsync` is weaker than Linux; true durability needs `fcntl(F_FULLFSYNC)` which stdlib doesn't expose.
9. Regenerate `tool-integrity.json` (now guarded by `test_shipped_manifest_verifies`)
10. Commit, tag v0.2.0, push
11. `python3 -m build && twine upload` (use `PYPI_API_TOKEN2` from `.env.flow`)
12. GitHub release with notes
13. Trigger Glama sync
