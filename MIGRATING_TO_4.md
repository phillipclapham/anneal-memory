# Migrating to anneal-memory 4.0

4.0 is a breaking release. It cuts consolidation down to one method and removes everything that only served the old one. If you only record and recall, little changes for you. If you call `prepare_wrap` and `validated_save_continuity`, or run the `crystal`, `associations` or `graph` commands, read this.

Your data is safe. **No table is dropped and no row is deleted.** A 3.x store opens in 4.0 as it is, and its current continuity file becomes the seed for the first wrap under the new method.

## What the wrap is now

`prepare_wrap` opens one step: the oldest calendar day of episodes that are not wrapped yet, and returns a fixed prompt filled with the current memory file and that day's log. You run the prompt in a fresh context, and you pass the model's whole reply to `validated_save_continuity`. A catch-up over several days is several steps in order. The README section "How the wrap works" has the full description.

The save checks that the reply is a non-empty string, at most 26,000 characters, and answers the step that was opened (the wrap token). It checks nothing else. Previous versions of the file are kept, so a dropped fact can be recovered.

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

```python
prepare_wrap(store, *, max_chars=None, tz=None, session_id=None, allow_sole_live=False,
             wrap_token=None, idempotency_key=None, lease_token=None) -> PrepareWrapResult

validated_save_continuity(store, text, *, wrap_token=None, max_chars=None, session_id=None,
                          allow_sole_live=False, idempotency_key=None, lease_token=None,
                          unattended=False, model=None)
```

- `max_chars` defaults to 26,000. Pass the same value to both calls if you change it.
- `tz` is an IANA name such as `"America/New_York"` or a `tzinfo`. The default is the machine's own zone.
- `model` and `unattended` are recorded on the wrap, for the record.
- `validated_save_continuity` lost `affective_state` (it was the third positional argument, so a call that passed it positionally now breaks), `today`, `allow_shrink`, `allow_unlinked`, `carryforward_cold_days`, `crystal_store`, `compost` and `require_rederive`. Everything after `text` is keyword-only. `prepare_wrap` lost its crystal-related arguments.

### Result shapes

`prepare_wrap` returns `{status, message, episode_count, pending_count, wrap_token, step}`.

- `status` is `"ready"`, `"empty"` or `"downgraded"` (the baton gate declined, so the call is capture-only).
- `step` is `None` unless the status is `"ready"`. Then it holds `day`, `chunk`, `chunks`, `episodes`, `steps_remaining`, `max_chars`, `memory_chars`, `system`, `user` and `prompt` (`system` and `user` as one text).

`validated_save_continuity` returns `{path, chars, episodes_compressed, pending_count, content_hash, wrap_result}`. The graduation, demotion, citation and association counts are gone. A keyed retry returns the replay subset, marked `replayed`, as before.

Other shapes changed with them: `WrapResult`, `WrapRecord`, `StoreStatus` and `WrapRevertReceipt` lost the fields that belonged to removed features, `Store.wrap_started` and `wrap_completed` have new signatures, and `Store(section_schema=)` is gone. If you drive the wrap lifecycle by hand, read those signatures in the source before upgrading.

Also removed from the public API: `CrystalStore` and its types, the graduation functions, `OutcomeLog` and the outcome/worth names, the rederive names, the schema names (`DEFAULT_SCHEMA`, `SectionSpec` and the rest), `retrieve_relevant` and `retrieve_patterns`, and the association types. `search_episodes` stays. `anneal_memory.__all__` is the list for your version.

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
- `save-continuity` takes `--wrap-token`, `--max-chars`, `--model`, `--unattended` and `--session-id`. `--allow-shrink`, `--allow-unlinked`, `--crystal`, `--require-rederive`, `--affect-tag` and `--affect-intensity` are gone, and `prepare-wrap` lost `--staleness-days` and `--crystal`.
- Removed commands: `crystal` (all of its subcommands), `associations`, `pattern-associations`, `graph`, `outcome`, `worth`, `probe`, `derive`, `set-schema`. Removed options: `init --schema`, `continuity --rederive`.
- Kept: `init`, `status`, `episodes`, `get`, `continuity`, `record`, `search` (alias `recall`), `supersede`, `unsupersede`, `state`, `trust`, `delete`, `prune`, `verify`, `audit-repair`, `prepare-wrap`, `save-continuity`, `baton`, `wrap-status`, `wrap-cancel`, `revert-wrap`, `wrap-token-current`, `serve`, `export`, `import`, `team-import`, `team-status`, `team-forget-key`, `audit`, `diff`, `history`, `stats`, the `spore` and `intention` groups, and `migrate check` / `migrate ack`.
## MCP: before and after

The tools are `record`, `recall`, `read_continuity`, `prepare_wrap`, `save_continuity`, `wrap_cancel`, `delete_episode`, `status`, and the `spore_*` and `intention_*` families. `crystal_recall` and `crystal_index` are gone.

`prepare_wrap` now returns the facts of the step, the wrap token, and the prompt, as one text. The agent runs the prompt in a fresh context, for example a subagent given only that prompt, and calls `save_continuity` with the model's whole reply and the token. Agent instructions written for 3.x tell the agent to compress in its own conversation and to cite episode ids. Replace them with the 4.0 snippets in `examples/`, or run `anneal-memory migrate check` and apply what it proposes.

## What the memory file looks like now

The file is whatever the model wrote. There are no required `## State`, `## Patterns`, `## Decisions` or `## Context` sections, and the markers (`1x`, `[evidence: ...]`, `[decided(...)]`) are no longer parsed or checked. If your tooling reads those sections out of the file, it will find them only as long as the model keeps writing them; the prompt does not ask for them.

## The first wrap after you upgrade

The old continuity file seeds the first step. It may be longer than 26,000 characters, and that is allowed: the prompt tells the model the file's current size and the limit, and asks it to compress. The reply, not the seed, has to fit. If a reply is over the limit the save is refused, nothing is written, and the step stays open for another run.

If you have many unwrapped episodes, the first wrap runs one step per calendar day. Keep preparing until `pending_count` is 0.

Everything the old store held stays where it was: episodes, supersession links and state keys, trust classes, the audit trail, the baton and its policy, the spore and intention file. The `<stem>.crystal.json` file, if you have one, is no longer read; leave it or delete it. Episode version tokens are not changed by 4.0.

## Upgrade steps

1. `pip install -U anneal-memory` (or let `uvx` fetch it).
2. `anneal-memory migrate check`. It lists the edits it proposes to your instruction files. It never edits them.
3. Update the wrap code or the agent instructions as above, or copy the new snippets from `examples/`.
4. `anneal-memory migrate ack`.
5. Run one wrap by hand and read the result with `anneal-memory continuity`, then `anneal-memory diff` to see what changed from the old file.

If something goes wrong, `anneal-memory revert-wrap WRAP_ID --reason "..."` undoes the newest wrap, and the previous version of the file is still in the `identity_versions` table.
