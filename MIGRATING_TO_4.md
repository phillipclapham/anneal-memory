# Migrating to anneal-memory 4.0

4.0 is a breaking release. It cuts consolidation down to one method and removes everything that only served the old one. If you only record and recall, little changes for you. If you call `prepare_wrap` and `validated_save_continuity`, run the wrap from a script or an agent's instructions, or use the `crystal`, `associations` or `graph` commands, read this.

Your data is safe. **No table is dropped and no row is deleted.** A 3.x store opens in 4.0 as it is, and its current continuity file becomes the seed for the first wrap under the new method.

## What the wrap is now

`prepare_wrap` opens one step: the oldest calendar day of episodes that are not wrapped yet. It returns a fixed prompt filled with the current memory file and that day's log. You run the prompt in a fresh context, and you pass the model's whole reply to `validated_save_continuity`. A catch-up over several days is several steps in order. The README section "How the wrap works" has the full description.

The save checks that the reply is a non-empty string, at most 26,000 characters, and answers the step that was opened (the wrap token, which is required: without it a save cannot tell which step the reply answers). It checks nothing else. Every earlier version of the file is kept, so a dropped fact can be recovered.

## Who has to change what

| You... | Do this |
|---|---|
| Only record and recall (library, CLI or MCP) | Nothing. Episodes, recall, supersession, state keys, trust, spores and intentions, the audit trail, `export` and `import` work as before |
| Call `prepare_wrap` / `validated_save_continuity` | Change the call, as below. `wrap["package"]` is gone |
| Run `prepare-wrap` / `save-continuity` from a script or an agent's instructions | The output and the flags changed, as below |
| Use the MCP `prepare_wrap` / `save_continuity` tools | The reply now carries a prompt to run in a fresh context. Re-run `anneal-memory migrate check` and update your instructions |
| Call `crystal_recall` / `crystal_index` over MCP, or use the `crystal` CLI group | Removed. There is no replacement |
| Read graduation, pattern or association data | Removed from the package. The rows stay in the database file; read them with `sqlite3` if you need them |
| Pass `section_schema=` to `Store(...)` | Removed. The memory file has no required sections |
| Catch `ContinuityValidationError`, `SectionError`, `WrapSchemaMovedError` or `AssociationFormationOffError` | Removed. Nothing raises them |
| Parse the memory file for `## State`, `## Patterns` or `[evidence: ...]` | Nothing writes those on purpose any more. See "What the memory file looks like now" |

## Upgrading a 3.x store, step by step

1. **Stop the writers and back up.** Close agents and servers that use the store. Then `anneal-memory export --format sqlite --output memory.backup.db` writes a copy of the database. Keep your `<stem>.continuity.md` and `<stem>.crystal.json` files with it. 4.0 changes no table, but a copy costs nothing.
2. **Upgrade the package.** `pip install -U anneal-memory`, or let `uvx` fetch it. `anneal-memory --version` shows what runs.
3. **Look at the store.** `anneal-memory status` and `anneal-memory continuity` open it as it is. Nothing is migrated by reading.
4. **Cancel a wrap that 3.x left open.** `anneal-memory wrap-status` says whether one is in progress. If it is, 4.0 refuses to prepare a new step until it is closed, and a wrap that 3.x opened was built from the 3.x prompt, so cancel it instead of finishing it. Run `anneal-memory wrap-cancel`. Its episodes are released, not deleted, and the next `prepare-wrap` picks them up. On a baton-protected store a wrap prepared under the baton needs `--session-id` of the session that prepared it, or `--force` if that session is gone.
5. **Update what calls the wrap.** Change library calls, scripts and agent instructions as described below. `anneal-memory migrate check` lists the edits it proposes to your instruction files, and never edits them. While the 4.0.0 entry is pending it lists only that one, because it retires every earlier entry (they teach tools and sections 4.0 removed). Copy the new snippets from `examples/` if you prefer.
6. **Run the first wrap by hand.** `anneal-memory prepare-wrap` (with `--session-id NAME` if the store requires the baton and you have claimed it). Run the prompt in a fresh context, see below, write the model's reply to a file, and `anneal-memory save-continuity FILE --wrap-token TOKEN` (with the same `--session-id`). The token is also printed by `anneal-memory wrap-token-current`. The step covers the oldest day of episodes the last 3.x wrap did not reach; if several days are waiting, repeat until `pending_count` is 0.
7. **Read the result.** `anneal-memory continuity` prints the new file. `anneal-memory diff` shows what the wrap changed against the old file, and `anneal-memory history` lists the wraps, the 3.x ones included.
8. **Record that you are up to date.** `anneal-memory migrate ack`.

If a wrap went wrong, see "Recovering an earlier version" at the end.

### The first wrap after you upgrade

The old continuity file seeds the first step as it is. It may be longer than 26,000 characters, and that is allowed: the prompt tells the model the file's current size and the limit, and asks it to compress. The reply, not the seed, has to fit. If a reply is over the limit the save is refused, nothing is written, and the step stays open for another run.

Everything the old store held stays where it was: episodes, supersession links and state keys, trust classes, the audit trail, the baton and its policy, the spore and intention file, and the old wraps with their counts. The `<stem>.crystal.json` file and the outcome log, if you have them, are no longer read; leave them or delete them. Episode version tokens are not changed by 4.0.

### Running the step in a fresh context

The step is meant to run with the prompt and nothing else in the model's context: no conversation, no tools, no files. anneal-memory cannot enforce that. It is yours to arrange, and the agent that did the day's work is the wrong place to run it.

- **Claude Code and similar harnesses:** start a subagent and give it only the prompt text, then take its whole reply.
- **A new session or a plain model call:** send the `system` and `user` messages as they are. For a client with one text slot, `step.prompt` is both as one block.
- **From a script:** `prepare-wrap --json` gives `step.system`, `step.user` and `wrap_token`.

```python
import json, os, subprocess, sys, tempfile

SESSION = "me"            # omit --session-id below on a store that does not require the baton

while True:
    prep = subprocess.run(["anneal-memory", "prepare-wrap", "--json", "--session-id", SESSION],
                          capture_output=True, text=True)
    if prep.returncode == 3:                          # downgraded: the operator has to claim the baton
        sys.exit(json.loads(prep.stdout)["message"])
    wrap = json.loads(prep.stdout)
    if wrap["status"] != "ready":                     # "empty": nothing is waiting
        break
    step = wrap["step"]
    reply = run_model(step["system"], step["user"])   # yours: one model call, nothing else in context
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(reply)
    subprocess.run(["anneal-memory", "save-continuity", f.name, "--session-id", SESSION,
                    "--wrap-token", wrap["wrap_token"]], check=True)
    os.unlink(f.name)
```

`save-continuity --json` prints `{saved, path, chars, episodes_compressed, pending_count, wrap_id, content_hash}`. The library call returns the wrap id inside `wrap_result`.

## Library: before and after

3.x:

```python
wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    compressed = my_llm(wrap["package"])      # instructions, episodes, stale patterns...
    result = validated_save_continuity(store, compressed, wrap_token=wrap["wrap_token"])
    print(result["graduations_validated"], result["graduations_demoted"])
```

4.0:

```python
while True:
    wrap = prepare_wrap(store)
    if wrap["status"] != "ready":
        break
    step = wrap["step"]
    reply = run_model(step["system"], step["user"])    # a fresh context, this prompt only
    result = validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"])
    if result["pending_count"] == 0:
        break
```

### Signatures

```text
prepare_wrap(store, *, max_chars=None, tz=None, session_id=None, allow_sole_live=False,
             wrap_token=None, idempotency_key=None, lease_token=None) -> PrepareWrapResult

validated_save_continuity(store, text, *, wrap_token=None, max_chars=None, session_id=None,
                          allow_sole_live=False, idempotency_key=None, lease_token=None,
                          unattended=False, model=None)
```

- `max_chars` defaults to 26,000. Pass the same value to both calls if you change it.
- `tz` is an IANA name such as `"America/New_York"` or a `tzinfo`. The default is the machine's own zone.
- `model` and `unattended` are recorded on the wrap, for the record.
- `prepare_wrap` lost `staleness_days` and `crystal_store`. `validated_save_continuity` lost `affective_state` (it was the third positional argument, so a call that passed it positionally now breaks), `today`, `allow_shrink`, `allow_unlinked`, `carryforward_cold_days`, `crystal_store`, `compost` and `require_rederive`, and everything after `text` is keyword-only.
- A store with no baton requirement is wrapped without `session_id`. The CLI and the MCP server drop the id on such a store, but the library does not: a library call that passes one is judged against the baton.

### Result shapes

`prepare_wrap` returns `{status, message, episode_count, pending_count, wrap_token, step}`.

- `status` is `"ready"`, `"empty"` or `"downgraded"` (the baton gate declined, so the call is capture-only).
- `step` is `None` unless the status is `"ready"`. Then it holds `day`, `chunk`, `chunks`, `episodes`, `steps_remaining`, `max_chars`, `memory_chars`, `system`, `user` and `prompt` (`system` and `user` as one text).
- Gone from `PrepareWrapResult`: `package`, `assoc_context`, `uncovered_proven_to_check`, `schema_warning`, `crystallization_candidates` and `rewarm_candidates`.

`validated_save_continuity` returns `{path, chars, episodes_compressed, pending_count, content_hash, wrap_result}`. `SaveContinuityResult` lost its graduation, demotion, citation, association, pattern, schema, section and drift fields (`composted`, `stale_state`, `durable_warnings`, `drift`, `graduations_validated`, `graduations_demoted`, `demoted`, `bare_demoted`, `citation_reuse_max`, `citation_spread`, `omitted_patterns`, `cross_session_collisions`, `contradictions`, `carried_forward`, `associations_formed`, `associations_strengthened`, `associations_decayed`, `sections` and the rest of that group). A keyed retry returns the replay subset, marked `replayed`, as before; `SaveContinuityReplayResult` lost its two graduation counts.

Other shapes:

- `WrapResult` and `WrapRecord` keep every field they had, and the graduation, pattern, citation and association counts in them are always 0 for a wrap made under 4.0 (and `section_sizes` is empty). They stay so that wraps written by 3.x still read.
- `StoreStatus` lost `association_stats`.
- `WrapRevertReceipt` lost `pattern_state_exact`. A revert now undoes the wrap row, the identity version, the episode stamps and the idempotency record, and nothing else.
- `Store.wrap_started(*, token, episode_ids, allow_restart=False, gated_session_id=None, expect_last_wrap_id=None, token_bound=False, today=None, expect_continuity_sha256=None, lease_token=None, idempotency=None)` lost `section_schema` and `derive_roots`.
- `Store.wrap_completed(episodes_compressed, continuity_chars, *, episode_ids=None, wrap_token=None, content_hash=None, pair_id=None, unattended=False, model=None)` lost its graduation, citation, pattern, association and `section_sizes` arguments. Everything after `continuity_chars` is keyword-only. It still returns a `WrapResult`.
- `Store.load_continuity()` returns the stored text exactly as saved.
- `Store(section_schema=)` is gone. `project_name=` is still accepted and kept in the store's metadata and exports, but nothing writes it into the memory file.

### Removed Python names

Every one of these was in `anneal_memory.__all__` in 3.x and is not in 4.0. `anneal_memory.__all__` is the list for your version.

- **Exceptions:** `ContinuityValidationError`, `SectionError`, `WrapSchemaMovedError`, `AssociationFormationOffError`, `CrystalError`, `CrystalConflictError`, `DeriveRefused`, `OutcomeLogBusy`, `ForeignOutcomeLogError`.
- **Crystals:** `CrystalStore`, `CrystalSnapshot`, `CrystalDict`, `CrystalDecision`, `RetirementDict`, `Permanence`, `ActivationMode`, `Activation`, `Route`, `activation_tier`, `parse_crystal_decisions`, `CRYSTAL_SCHEMA_VERSION`, `RETIRE_KINDS`, `MIN_PROVEN_LEVEL`, `VALID_ACTIVATION_MODES`, `VALID_ACTIVATIONS`, `VALID_LEVELS`, `VALID_PERMANENCE`, `VALID_ROUTES`.
- **Graduation and patterns:** `validate_graduations`, `check_explanation_overlap`, `detect_stale_patterns`, `detect_pattern_omissions`, `detect_proven_without_declaration`, `extract_session_co_citations`, `extract_pattern_names`, `extract_pattern_summaries`, `extract_proven_patterns`, `extract_contradiction_declarations`, `PatternSummary`, `OmittedPattern`, `CrossSessionCollision`, `ProvenWithoutDeclaration`, `StalePatternDict`, `MAX_PATTERNS`, `RETRIEVAL_THRESHOLD`, `retrieve_relevant`, `retrieve_patterns`, `RelevantResult`, `RelevantPattern`, `RelevantFact`.
- **Associations:** `AssociationPair`, `AssociationStats`.
- **Section schemas and structure:** `SectionSpec`, `SectionRole`, `SectionWriteResult`, `DEFAULT_SCHEMA`, `FLOW_SCHEMA`, `PROJECT_SCHEMA`, `SCHEMA_NAMES`, `DEFAULT_GRADUATING`, `schema_by_name`, `name_for_schema`, `validate_schema`, `graduating_headings`, `required_headings`, `sections_by_role`, `schema_role_warning`, `heading_marker`, `hard_max_chars`, `validate_structure`, `canonical_section_markdown`.
- **Durable facts:** `DurableFact`, `parse_durable_facts`, `DROP_DURABLE_RE`, `section_chars`.
- **Rederive:** `RederiveReport`, `rederive_continuity`, `allow_store`, `revoke_store`, `derived_state_lines`, `parse_annotation`, `strip_flag`, `trusted_roots`, `trust_file_path`, `Annotation`, `TRUST_ENV`, `VISIBILITIES`.
- **Worth and the outcome log:** `OutcomeLog`, `outcome_log_path`, `fold_surfaced`, `receipt_crystal_names`, `ExposedRef`, `ExposureLabel`, `FoldResult`, `LogBinding`.
- **Other:** `AffectiveState`, `WrapPackageDict`.

`search_episodes` stays, as do `Store`, `EpisodeType`, `prepare_wrap`, `validated_save_continuity`, the audit, spore and intention names and the remaining exceptions.

Methods removed from `Store`: `add_drift_probe`, `list_drift_probes`, `retire_drift_probe`, `drift_status`, `association_formation`, `association_stats`, `decay_associations`, `drain_co_surface_events`, `gc_pattern_associations`, `get_association_context`, `get_associations`, `get_pattern_associations`, `pattern_association_stats`, `pattern_graph_projection_meta`, `record_associations`, `rename_pattern_association`, `sever_pattern_concept`, `get_pattern_history`, `pattern_history_names`, `upsert_pattern_history`, `saved_pattern_levels`, `seed_pattern_co_graduation`, `seed_pattern_max_level`, `pattern_grounding`, `ground_state`, `read_section`, `replace_section`, `section_schema`, `section_schema_for_wrap`, `set_section_schema` and `wrap_derive_roots`. `Store.wrap_identity_texts` is new.

## CLI: before and after

3.x:

```bash
anneal-memory prepare-wrap --session-id me        # printed a compression package
anneal-memory save-continuity out.md --session-id me --wrap-token "$WRAP_TOKEN" --allow-shrink
```

4.0:

```bash
anneal-memory prepare-wrap --session-id me        # prints "Wrap token: <hex>" and the prompt
# run the prompt in a fresh context; write the model's whole reply to out.md
anneal-memory save-continuity out.md --session-id me --wrap-token "$WRAP_TOKEN"
```

- `prepare-wrap` takes `--max-chars`, `--tz` and `--session-id`. `--json` gives `step.system`, `step.user` and `wrap_token` as fields. It still exits 3 when the wrap is downgraded.
- `save-continuity` takes `--wrap-token` (required), `--max-chars` (leave it out: the step keeps the cap `prepare-wrap` was given and the save uses it; a different value is refused), `--model`, `--unattended` and `--session-id`. `--allow-shrink`, `--allow-unlinked`, `--crystal`, `--require-rederive`, `--affect-tag` and `--affect-intensity` are gone, and `prepare-wrap` lost `--staleness-days` and `--crystal`.
- Removed commands: `crystal` (all of its subcommands), `associations`, `pattern-associations`, `graph`, `outcome`, `worth`, `probe` (with `add`, `list`, `retire` and `status`), `derive` (with `allow`, `revoke` and `status`), `set-schema`. Removed options: `init --schema`, `continuity --rederive` and `continuity --ref`.
- `--json` keys that went with them: `status` lost `association_stats`; `stats` lost `associations` and the graduation, demotion and association totals; `history` lost the graduation, pattern, citation and association counts per wrap; `diff` lost the same, and now gives `lines_added`, `lines_removed`, `delta` and, for the newest wrap, `text_diff`.
- Re-based: `history`, `diff` and `stats` now report what each wrap changed in the file (size, lines, a text diff), never graduation or association numbers. They read the same rows 3.x wrote.
- Kept: `init`, `status`, `episodes`, `get`, `continuity`, `record`, `search` (alias `recall`), `supersede`, `unsupersede`, `state`, `trust`, `delete`, `prune`, `verify`, `audit-repair`, `prepare-wrap`, `save-continuity`, `baton`, `wrap-status`, `wrap-cancel`, `revert-wrap`, `wrap-token-current`, `serve`, `export`, `import`, `team-import`, `team-status`, `team-forget-key`, `audit`, `diff`, `history`, `stats`, the `spore` and `intention` groups, and `migrate check` / `migrate ack`.

## MCP: before and after

The tools are `record`, `recall`, `read_continuity`, `prepare_wrap`, `save_continuity`, `wrap_cancel`, `delete_episode`, `status`, and the `spore_*` and `intention_*` families. `crystal_recall` and `crystal_index` are gone. `read_continuity` is new: it returns the memory file as a tool result, for clients that read tools and not resources (`anneal://continuity` still works). `prepare_wrap` lost `staleness_days` and gained `tz`. `save_continuity` lost `affective_state`, `allow_shrink` and `allow_unlinked` and gained `max_chars`. The hosted remote path's `orient` no longer carries a crystal index, and its read tools are `recall` alone.

`prepare_wrap` now returns the facts of the step, the wrap token, and the prompt, as one text. The agent runs the prompt in a fresh context, for example a subagent given only that prompt, and calls `save_continuity` with the model's whole reply and the token. Agent instructions written for 3.x tell the agent to compress in its own conversation and to cite episode ids. Replace them with the 4.0 snippets in `examples/`, or run `anneal-memory migrate check` and apply what it proposes. The tool descriptions changed, so a host that pins `tool-integrity.json` or the `anneal://integrity/manifest` hashes sees new values.

## What the memory file looks like now

**What the step shows the model.** The log lines carry only the time, the type and the text. An episode recorded with a lower trust class (`external`, `tool`) is not marked as such in the prompt (3.x tagged them), and whatever the model writes becomes the memory every session loads. Record only what you would put in the memory, and read each new memory file before it is loaded (`history` and `diff` show what changed).

The file is whatever the model wrote. There are no required `## State`, `## Patterns` or `## Decisions` sections, and the markers that 3.x parsed (level counts, `[evidence: ...]`, `[decided(...)]`) are no longer parsed or checked. A 3.x file that has them keeps them until a rewrite drops them, and the prompt does not ask for them. If your tooling reads those sections out of the file, it will find them only as long as the model keeps writing them.

## Recovering an earlier version

Every saved file is a row in the `identity_versions` table. `anneal-memory history` lists the wraps and `anneal-memory diff` shows what the newest ones changed, against the version in force just before each. To undo the newest wrap whole: `anneal-memory revert-wrap WRAP_ID --reason "..."`, where `WRAP_ID` is the id `history` shows for it. The file goes back to the version before it, saved as a new version, and that wrap's episodes are offered to the next step again. To read any older version, call `store.wrap_identity_texts([wrap_id, ...])` or query the table (the README section "Recovering an older version" has the code). To put one fact back, record it again as an episode so that the next wrap sees it.
