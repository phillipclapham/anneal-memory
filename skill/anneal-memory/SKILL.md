---
name: anneal-memory
description: Persistent agent memory via anneal-memory - an episodic log, one memory file rewritten a day at a time, and a prospective spore (intention) store. Use in any project wired for anneal-memory, meaning an `ANNEAL_MEMORY_DB` env var, an `anneal://continuity` MCP resource, or memory instructions that mention `recall`, `prepare_wrap` or `save_continuity`. Use it before any architectural decision or rule change (recall first), while recording decisions, observations, tensions and outcomes during work, and when the user says the session is ending ("wrap up", "save memory", "we're done") to run the prepare_wrap, run-in-a-fresh-context, save_continuity sequence. It also carries the start-of-session memory load.
license: MIT
---

# anneal-memory

This project uses **anneal-memory** for memory across sessions. Use it naturally. Don't explain the tooling to the user. You record what happens, recall it before you decide, and at the end of a session a wrap rewrites the memory file so the next session starts informed.

> **Transport.** There are two interfaces to the same loop: an **MCP server** (tool calls like `recall`, `prepare_wrap`) and a **CLI** (`anneal-memory recall ...`, `anneal-memory prepare-wrap`). This skill is written for both; the [Commands](#commands) table pairs them. Use whichever your harness has wired. If both are there, prefer MCP for the in-session loop and the CLI for operator inspection.

## The pieces

- **Episodes**: raw notes you record during work. Cheap and plentiful. They stay recallable.
- **The memory file** ("continuity"): one document a model rewrites at each wrap from the old file plus one day of episodes. It is what a session loads at the start. It is whatever the model wrote; it has no required sections.
- **Spores** (also called intentions): a separate store for open loops, things you mean to do or resolve.

You touch episodes directly (record, recall) and the memory file through the wrap. Every earlier version of the file is kept, so nothing a wrap drops is lost for good.

## Spores: the prospective layer

Episodes and the memory file are about what happened, and they never complete. A spore is about what is still open, and it must close. Plant one the moment a loop opens: a `task` (open doing), a `question` (open not-knowing) or a `thought` (open idea). Resolve each one: **descend** (done or dropped) or **ascend** (the loop closes by its content becoming an episode or a project; the spore still ends).

Memory versus spores is a lifecycle cut: memory accretes, a spore closes. If your harness already tracks open loops somewhere, that is a spore plant. Plant the loops here and let your list point at them, so you don't keep two trackers.

## Session workflow

### Start of session
Read the memory file: MCP `read_continuity` (or the `anneal://continuity` resource); CLI `anneal-memory continuity`. A first-ever session has none. The CLI prints `No continuity file yet` and **exits non-zero**: that is the first-session signal, not an error. Start recording. Recall specific episodes when you need them.

### During work
Record an episode when something happens that would change a later decision. Record the **reasoning, not just the fact**: "Chose X because Y" beats "using X". Do it without being asked.

Types (pick the most specific): `observation` · `decision` · `tension` · `question` · `outcome` · `context`.

**Cadence:** default to recording. Episodes are cheap, and the wrap sorts out what matters. After each exchange where real work happened, record at least one. If three or more exchanges pass with nothing recorded, something was missed. Aim for 5 to 15 episodes a session.

### Before decisions (NON-NEGOTIABLE)
Before any architectural decision, design choice or rule change, **recall** on the area being decided and use what comes back. If nothing relevant returns, go ahead and say that recall came back empty.

This is the rule everything else depends on. A store that is only written to, and never read before deciding, is dead: episodes pile up and decisions still get made blind to them.

### End of session
When the user signals the end ("wrap up", "save memory", "we're done", or anything like it), run the wrap. Don't just acknowledge it.

1. **`prepare_wrap`** opens one step: the oldest calendar day of episodes not yet wrapped. It returns the facts of the step, a **wrap token**, and a **prompt** (the current memory file plus that day's log, with instructions to rewrite the file).
2. **Run that prompt in a fresh context**, with the prompt and nothing else. In Claude Code, that means a subagent you give only the prompt text; elsewhere, a new session or a plain model call. Do not run it in your own conversation: you carry the whole session with you, and the step is meant to be a clean rewrite from the file and the day's log. Do not add your own commentary to the prompt.
3. **`save_continuity`** with the model's **whole reply** as the text and the token. The reply is the complete new memory file, not a diff and not a summary of changes.
4. If the save or the prepare result says more episodes are waiting (`pending_count` not zero, `steps_remaining` not zero), the next day is owed. Prepare again and repeat. A catch-up over several days is several steps in order, never one merged log.

The save checks that the reply is not empty, is within the character limit (26,000 by default), and answers the step that was opened. If it is refused, nothing was written and the step is still open: run the step again and save again. If a reply is too long, the refusal message says what to append to the prompt on the retry.

If `prepare_wrap` reports no episodes, there is nothing to do. If it **downgrades to capture-only** (a store created by the CLI requires the consolidate baton), nothing is lost: tell your human what the message says, which is how many episodes are waiting and the one command that lets this session consolidate (`anneal-memory baton claim NAME`; the CLI wraps with `--session-id NAME`, and an MCP server is started with it). Consolidating is the operator's act, so never claim the baton yourself.

**Once per prepare.** A second save with no fresh `prepare_wrap` is refused by design.

## What to record, and what not to

**Record:** decisions and their reasons · tradeoffs and tensions (name the axis) · blockers and dependencies · outcomes, successes and failures both · context that shapes decisions · what you learned about how to work here.

**Don't record:** routine code changes (git has them) · transient debugging · anything already in the docs · every small observation. Record what would change a decision.

**Correcting versus deleting:** for a changed fact, record a **new** episode, and if it replaces an older one say so: MCP `record` takes `supersedes` (an episode id) and `state_key` (a slot name such as `user.home_city`; the newest episode in a slot replaces the others). Recall then stops serving the old one, and nothing is deleted. For content that should not exist (PII, secrets, a recording that is simply wrong), delete the episode by id (`delete_episode`). That is audit-logged and cannot be undone.

## How to present memory

Report what you find naturally: "in an earlier session there was a tension between X and Y", not "the recall tool returned...". Don't narrate tool calls. Treat the memory file as a good summary written by a reader who was not in the room: it can be wrong or stale, and the episodes behind it are there to check against.

## One store per entity

A store is one entity's memory. There is no tenant scoping inside it, so give each agent its own store path. One `Store` object is not thread-safe or reentrant; use one per thread. Several processes of the same entity (a CLI beside an MCP server) may record into one store: SQLite serializes the episodes and the audit trail serializes its appends with a cross-process lock, provided every writer runs a version that takes it. Windows has no such lock, so keep to one writer at a time there.

## Commands

CLI and MCP forms of the same loop. Set `ANNEAL_MEMORY_DB` (or pass `--db`) for the CLI. For MCP, the server is set up in your harness's MCP config.

| Step | CLI | MCP tool |
|------|-----|----------|
| First-time setup | `anneal-memory init` | (the store is created on first use) |
| Read the memory file | `anneal-memory continuity` | `read_continuity`, or the `anneal://continuity` resource |
| Recall before deciding | `anneal-memory recall "<query>"` (alias: `search`) | `recall` |
| Browse episodes | `anneal-memory episodes --since 7d --type decision` | `recall` (filtered) |
| Record an episode | `anneal-memory record "Chose X because Y" --type decision` | `record` |
| Record multi-line | `echo "..." \| anneal-memory record - --type observation` | `record` |
| A fact changed | `anneal-memory record "..." --supersedes OLD_ID` or `--state-key KEY` | `record` with `supersedes` / `state_key` |
| Open a wrap step | `anneal-memory prepare-wrap` | `prepare_wrap` |
| Keep the reply | `anneal-memory save-continuity memory.md --wrap-token TOKEN` | `save_continuity` |
| Plant a spore | `anneal-memory spore add --type task --text "..."` | `spore_add` |
| List or surface spores | `anneal-memory spore list` · `spore surface` | `spore_list` · `spore_surface` |
| Resolve a spore | `anneal-memory spore descend ...` · `spore ascend ...` | `spore_descend` · `spore_ascend` |
| The same layer as intentions | `anneal-memory intention add\|list\|surface\|close\|promote ...` (ids are `intention-NNN`) | `intention_add` · `intention_list` · `intention_surface` · `intention_get` · `intention_update` · `intention_touch` · `intention_close` · `intention_promote` |
| Store health | `anneal-memory status` | `status` |
| Delete an episode | `anneal-memory delete <id> --force` | `delete_episode` |
| Clear a stuck wrap | `anneal-memory wrap-status` · `wrap-cancel` | `wrap_cancel` (look at `status` first) |

**A stuck wrap.** If `prepare_wrap` ran but its save never happened, the store reports a wrap in progress, and the next `prepare_wrap` refuses. Clear it with `wrap_cancel` or `anneal-memory wrap-cancel`. Episodes are not deleted. Over MCP, call `status` first: it says when the wrap started, and a wrap that began moments ago is probably a live peer session still working, since one server process runs per client session against a shared store. A wrap prepared under the baton can be cancelled without its token only by the session that prepared it, or with `force`, which is the operator's decision because it throws away what that session had in flight.

**If the wrap is yours, prove it.** Pass the `wrap_token` that `prepare_wrap` gave you (`--wrap-token`): the cancel succeeds only if that wrap is still the one in progress, and is refused with no change if a peer replaced it or it already finished.

**Operator-only** (CLI): `stats` · `history` · `diff` · `audit --since 7d` · `verify` · `export` / `import` · `prune --older-than 90` · `revert-wrap`. All take `--json`.

## Staying current

After you or your operator upgrade the `anneal-memory` package, run `anneal-memory migrate check`. It reports any change that suggests an edit to *these* instructions and proposes it, so hand-tuned guidance doesn't drift from the package. Apply what fits, then `anneal-memory migrate ack`. It only proposes; it never edits your files. Upgrading from 3.x: see `MIGRATING_TO_4.md` in the repository.

---

*This skill is depth on demand, not a replacement for the always-loaded lean snippet.* A model-invoked skill won't reliably fire at a bare session start, so keep `examples/agent-instructions.lean.example` (MCP) or `.lean.cli.example` (CLI) in your `CLAUDE.md` / `AGENTS.md`. It carries the baseline (load at start, record, recall before deciding, wrap), and this skill adds the detail. The full inline versions are `examples/agent-instructions.full.example` and `.full.cli.example`, and the project README has the rest.
