<!-- mcp-name: io.github.phillipclapham/anneal-memory -->

<p align="center">
  <img src="logo-400.png" alt="anneal-memory logo" width="200">
</p>

# anneal-memory

**Memory for AI agents: a log of what happened, and one memory file that gets rewritten a day at a time.**

An agent records episodes as it works (decisions, observations, open questions, outcomes). Now and then a model rewrites one memory file from the old file plus one day of those episodes. The agent loads that file at the start of every session. The store is a local SQLite file with no dependencies beyond the Python standard library. Every version of the memory file is kept.

anneal-memory calls no model itself. It builds the prompt, you run it on whatever model you like, and it keeps the reply.

> **Upgrading from 3.x?** 4.0 is a breaking release. It removed graduation, patterns, crystals and the section schema, and replaced the wrap with the single method below. Old stores open unchanged. See [MIGRATING_TO_4.md](MIGRATING_TO_4.md).

## How the wrap works

There is one consolidation method, a rolling rewrite.

1. `prepare_wrap` opens **one step**: the oldest calendar day that has episodes not yet wrapped. A day is cut in the machine's time zone, or in the zone you pass (`--tz`, `tz=`). A very long day is cut into slices. It returns a fixed prompt, filled with the current memory file and that day's log, and a wrap token.
2. You run that prompt in a **fresh context** that holds the prompt and nothing else: a subagent, a new session, or a plain model call. The prompt is fixed text, written for a mid-size model at low effort that sees the memory file and one day's log and nothing else. Run it as it is. The only addition is the retry line that a save refused as too long asks you to append.
3. You pass the model's **whole reply** to `save_continuity`, with the token.

A catch-up over several days is several steps in order, each one starting from the file the last one saved. It is never one merged log. After a save, `pending_count` says how many episodes are still waiting; if it is not zero, prepare the next step.

The save checks three things: the reply is not empty, it is at most 26,000 characters (`--max-chars` / `max_chars` changes the cap), and it answers the step that `prepare_wrap` opened. That is all. There is no required section layout and nothing is parsed out of the file. The prompt asks the model to keep what changes how the assistant should work with this user, drop what is finished or superseded, and stay under the limit. What the file looks like is the model's call.

A refused save writes nothing and leaves the step open, so you run it again. Every previous version of the file stays in the database (`history` and `diff` show them), so a fact that a rewrite dropped can be found and put back. See [Recovering an older version](#recovering-an-older-version).

## Quick start

```
pip install anneal-memory
```

### Python library

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity

store = Store("./memory.db")

store.record("Connection pool is the real bottleneck", EpisodeType.OBSERVATION)
store.record("Chose PostgreSQL because ACID outweighs speed", EpisodeType.DECISION)

result = store.recall(episode_type=EpisodeType.DECISION, keyword="PostgreSQL")
for ep in result.episodes:
    print(f"[{ep.type}] {ep.content}")

while True:
    wrap = prepare_wrap(store)
    if wrap["status"] != "ready":      # "empty": nothing waiting; "downgraded": see below
        break
    step = wrap["step"]                # step["system"], step["user"], step["prompt"]
    reply = run_model(step["system"], step["user"])   # your model call, fresh context
    saved = validated_save_continuity(store, reply, wrap_token=wrap["wrap_token"])
    if saved["pending_count"] == 0:
        break

store.close()
```

`run_model` is yours to write; anneal-memory has no model client. The [library quickstart](docs/library-quickstart.md) has one for the Anthropic SDK. A store made by `anneal-memory init` requires the baton (below), and a library caller on such a store gets `"downgraded"` unless it passes `session_id=` to both calls; a store opened with `Store(...)` directly has no such requirement.

### CLI

```bash
# Without --db or ANNEAL_MEMORY_DB every command uses ~/.anneal-memory/memory.db.
export ANNEAL_MEMORY_DB=./demo.db

anneal-memory init
anneal-memory record "Chose PostgreSQL as the database for ACID" --type decision
anneal-memory search "database"

# A new store consolidates only under a session id the operator has claimed
# the baton for (see "Consolidation is yours" below). Claim once.
anneal-memory baton claim me

anneal-memory prepare-wrap --session-id me
# Prints "Wrap token: <hex>" and the prompt. Run the prompt in a fresh context
# and write the model's whole reply to memory.md. The token is also printed by:
WRAP_TOKEN=$(anneal-memory wrap-token-current)
anneal-memory save-continuity memory.md --session-id me --wrap-token "$WRAP_TOKEN"

anneal-memory continuity        # print the current memory file
anneal-memory history           # the wraps so far
anneal-memory diff              # how recent wraps changed the file
```

`prepare-wrap --json` returns the same thing as an object (`step.system`, `step.user`, `wrap_token`), which is easier to script. `prepare-wrap` exits 3 when the wrap was downgraded to capture-only; see below. `history` lists each wrap with its episode count and the size of the file it saved. `diff` shows the last few wraps (`--wraps N`) with the change in size and lines, and the text diff of the newest one against the file before it. `stats` adds counts by episode type, age and source.

Other commands: `status`, `episodes`, `get`, `supersede`, `unsupersede`, `state`, `trust`, `delete`, `prune`, `verify`, `audit-repair`, `wrap-status`, `wrap-cancel`, `revert-wrap`, `wrap-token-current`, `baton`, `audit`, `stats`, `export`, `import`, `team-import`, `team-status`, `team-forget-key`, `migrate`, and the `spore` and `intention` groups. `anneal-memory --help` is the list for your installed version. Every command takes `--json`.

### MCP server

For MCP-capable harnesses such as Claude Code, Codex and Gemini CLI. The server is `anneal-memory serve`, wired into the harness's MCP config. Global flags such as `--db` go before `serve`.

**Claude Code**, a project `.mcp.json`:

```json
{
  "mcpServers": {
    "anneal_memory": {
      "command": "uvx",
      "args": ["anneal-memory", "serve", "--session-id", "claude-code"]
    }
  }
}
```

**Codex**, `~/.codex/config.toml`:

```toml
[mcp_servers.anneal_memory]
command = "uvx"
args = ["anneal-memory", "serve", "--session-id", "codex"]
```

**Gemini CLI**, `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "anneal_memory": {
      "command": "uvx",
      "args": ["anneal-memory", "serve", "--session-id", "gemini"]
    }
  }
}
```

Gemini CLI starts MCP servers only from a folder it trusts. In an untrusted folder `gemini mcp list` shows a user-level server as `Disconnected` and does not list a project one at all, even with the config right. Trust the folder, or set `GEMINI_CLI_TRUST_WORKSPACE=true`.

`uvx` fetches the latest published version on every run. For a pinned install, `pip install anneal-memory` and set `"command": "anneal-memory"` with the same `args`.

Then add one of the [agent-instruction snippets](#agent-harnesses-skill-and-snippets) to the harness's instructions file (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`). The tools are there without it, but the agent will not know when to call them.

**What the agent does at a wrap over MCP:** call `prepare_wrap`, run the prompt it returns in a fresh context (in Claude Code, a subagent that is given only that prompt), then call `save_continuity` with the subagent's whole reply and the token. The agent that did the day's work is the wrong place to run the prompt, because it carries the whole session with it.

**Consolidation is yours.** Agents record freely. Rewriting the memory file is the operator's act, so a store the server or `init` creates consolidates only through a session id you have claimed the baton for, once, from your own shell: `anneal-memory baton claim claude-code`. Until then the agent's `prepare_wrap` downgrades to capture-only, and both that message and the `status` tool say how many episodes are waiting and the command to run. Set the id in the client's config as above, not in a shell profile. Every conversation started from one config shares its id, so the claim designates a client, not one conversation. Move the baton with `baton claim NAME --take`; see who holds it with `baton status`. A store created before this policy, or with `init --no-baton`, ignores `--session-id` and wraps without it.

The gate stops an agent that follows its instructions from consolidating on its own initiative, and stops parallel sessions from each rewriting the memory file. It is not a lock against an agent with a shell that ignores its instructions: such an agent can run `baton claim` itself. Keeping the baton with you is part of the operator's job.

### Windows

anneal-memory runs on Windows with these limits:

- **Write locking is advisory POSIX `fcntl` and there is no lock at all on Windows.** Concurrent writers to one store, whether processes or threads, can race or lose updates. One writer at a time is fine; the unique-tmp plus atomic-replace write still prevents a torn file.
- **No fork support for the audit trail.** Do not fork a process while an `AuditTrail` in it is in use; a child opens its own.
- The CLI's standard streams are UTF-8 on every platform.
- If `ANNEAL_MEMORY_DB` is unset and no home directory can be found, a command that needs the default path exits 1 and names `--db` and `ANNEAL_MEMORY_DB`.
- Windows CI runs a reduced subset of the suite. Tests that depend on POSIX `chmod` or on the cross-process lock are skipped there.

## Same loop, three transports

The CLI and the MCP server are thin adapters over the library. All three call the same `prepare_wrap` and `validated_save_continuity`.

| | Library | CLI | MCP |
|---|---|---|---|
| **Record** | `store.record(content, type)` | `record "..." --type T` | `record` |
| **Recall** | `store.recall(keyword=...)` | `search "..."` | `recall` |
| **Read the memory file** | `store.load_continuity()` | `continuity` | `read_continuity` |
| **Open a step** | `prepare_wrap(store)` | `prepare-wrap` | `prepare_wrap` |
| **Keep the reply** | `validated_save_continuity(store, text, wrap_token=...)` | `save-continuity FILE` | `save_continuity` |
| **Best for** | Frameworks, custom agents | Agents with a shell, operators | MCP-enabled editors |

A recall reads one committed state, as of its start. An episode deleted before the recall begins is never returned; a delete that commits while a recall runs may or may not show, as with any database read.

## Framework integrations

Each guide shows where the library calls go in a framework's lifecycle. They show how each framework's API looked when they were written; the wrap code uses the 4.0 calls above and was not re-run against every framework. On a store that requires the baton, pass `session_id=` to both wrap calls (see the [library quickstart](docs/library-quickstart.md#the-baton)).

| Framework | Integration point | Guide |
|-----------|------------------|-------|
| **LangGraph / LangChain** | `AgentMiddleware` | [langgraph.md](docs/integrations/langgraph.md) |
| **CrewAI** | `BaseEventListener` | [crewai.md](docs/integrations/crewai.md) |
| **OpenAI Agents SDK** | `RunHooks` | [openai-agents.md](docs/integrations/openai-agents.md) |
| **Anthropic Agents SDK** | instructions snippet + `Stop` hook | [anthropic-agents.md](docs/integrations/anthropic-agents.md) |
| **Google ADK** | callbacks + custom `MemoryService` | [google-adk.md](docs/integrations/google-adk.md) |
| **Pydantic AI** | `AbstractCapability` with `Hooks` | [pydantic-ai.md](docs/integrations/pydantic-ai.md) |
| **smolagents** | `step_callbacks` | [smolagents.md](docs/integrations/smolagents.md) |
| **LlamaIndex** | instrumentation `BaseEventHandler` | [llamaindex.md](docs/integrations/llamaindex.md) |
| **Haystack** | custom `Tracer` | [haystack.md](docs/integrations/haystack.md) |
| **CAMEL-AI** | `WorkforceCallback` | [camel-ai.md](docs/integrations/camel-ai.md) |
| **AutoGen / AG2** | `register_hook()` | [autogen.md](docs/integrations/autogen.md) |
| **DSPy** | `BaseCallback` | [dspy.md](docs/integrations/dspy.md) |

Not listed? The pattern is the same everywhere: open a `Store`, `record()` at meaningful moments, `recall()` before decisions, run the wrap at a session boundary.

## Agent harnesses: skill and snippets

For Claude Code, Codex or Gemini CLI rather than a Python framework. These are repository files; they are **not** in the pip wheel, so clone the repo or download them from GitHub.

- **Lean snippet, always loaded.** [`examples/agent-instructions.lean.example`](examples/agent-instructions.lean.example) (MCP) or [`.lean.cli.example`](examples/agent-instructions.lean.cli.example) (CLI), pasted into `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`. About 50 lines: load the memory file at start, record, recall before decisions, run the wrap.
- **The skill, loaded on demand.** Copy [`skill/anneal-memory/`](skill/anneal-memory) into `.claude/skills/` or `~/.claude/skills/`. A model-invoked skill does not reliably fire at a bare session start, so keep the lean snippet in place as well.
- **Full reference.** [`.full.example`](examples/agent-instructions.full.example) and [`.full.cli.example`](examples/agent-instructions.full.cli.example) hold the same material inline, for adopters who would rather not install a skill.

After upgrading the package, run `anneal-memory migrate check`. It proposes edits to your instructions so they do not drift from the package (it never edits your files). Then `anneal-memory migrate ack`.

## Updated facts: supersession and state keys

When a fact changes, the new episode can say which episode it replaces, and recall stops serving the old one. Nothing is deleted. The old episode stays in the store and in exports, and `include_superseded=True` (CLI `--include-superseded`) shows it, marked with what replaced it.

- **Explicit link:** `store.record(text, "observation", supersedes=[old_id])`, CLI `record --supersedes ID`, MCP `record` with `supersedes`. It is validated: the old episode has to exist and not be newer, the link cannot close a cycle, and the two texts have to share at least a quarter of the shorter one's meaningful words. A link that fails records nothing. `unsupersede --old ID --new ID` removes a wrong one.
- **State key, for updates that share no words.** "Based in Seattle" becomes "settled into my new place in Austin", and no word-overlap rule links those. A key names the slot a fact fills: `record --state-key user.home_city`, MCP `record` with `state_key`, `store.record(..., state_key="user.home_city")`, or `state KEY --set ID` for an episode already recorded. A slot holds one value; the newest episode in it replaces the others. For a relation with several values, put the value in the key (`user.pet/archie`) or leave it unkeyed. `state [KEY]` shows each key's holder and what it replaced, and `state --unset ID` takes a wrongly keyed episode out of its slot.

Two limits. Nothing detects an update on its own: an update recorded with neither a link nor a key sits beside the old fact. And the word-overlap check is a floor, not a judgment that one episode replaces the other. It catches a link to the wrong episode entirely, not a link to a different fact on the same subject. A wrong link or key hides a still-valid episode until someone removes it.

## Prospective memory: spores and intentions

Episodes and the memory file are about what already happened. An agent also needs a record of what it means to do next, and that has a different lifecycle: it must close. anneal keeps these in a separate store beside the database (`<stem>.spores.json`): a `task` (open doing), a `question` (open not-knowing) or a `thought` (open idea), planted when the loop opens and resolved when it ends (`descend`: done or dropped; `ascend`: its content moves into an episode, a project or another record, which you name with `--kind` and `--ref`).

The same store is also called **intentions**: `IntentionStore` is an alias of `SporeStore`, the `intention_*` MCP tools and `anneal-memory intention ...` sit beside the `spore_*` ones, and `intention_close` and `intention_promote` are the names for `spore_descend` and `spore_ascend`. An id of either spelling reads in either. Nothing named `spore` is deprecated.

A spore file written by 1.0.7 or earlier is refused with the command that converts it: `anneal-memory intention migrate-key` (`--dry-run` to preview). It copies the file first and changes no row.

## Session hygiene

Wrap at the end of a session, or when episodes have piled up, and check that the wrap finished. An unwrapped session leaves its episodes in the log, still recallable, but the memory file the next session loads does not know about them.

- **Wrap once per prepare.** A second `save_continuity` with no new `prepare_wrap` is refused by design.
- **Several days waiting means several steps.** Keep preparing until `pending_count` is 0.
- **`prepare_wrap` says "empty"** when nothing is waiting. Skip the save.
- **A stuck wrap.** If `prepare_wrap` refuses because a wrap is already open and that wrap will not be finished, clear it with `wrap_cancel` (MCP) or `anneal-memory wrap-cancel`. Episodes are not deleted. Check `status` first: a wrap that began moments ago may be a live session still working. Pass `--wrap-token` to cancel only your own wrap.
- **A wrap you want back.** `anneal-memory revert-wrap WRAP_ID --reason "..."` undoes the newest completed wrap: the file goes back to what it was before (saved as a new version, so the history stays whole) and the wrap's episodes are offered again. `WRAP_ID` must be the newest wrap's id, the number `history` shows.
- **A wrap left open by 3.x.** A wrap that was open when you upgraded blocks `prepare_wrap` until it is cancelled with `anneal-memory wrap-cancel`. Its episodes are released, not deleted.
- **Short sessions are fine.** A few episodes still make a useful step.

### Recovering an older version

Every saved file is a row in the `identity_versions` table of the store's SQLite database. `history` and `diff` list the wraps and show what the newest ones changed. To read the file a wrap saved, from the library:

```python
from anneal_memory import Store

store = Store("./memory.db", read_only=True)
texts = store.wrap_identity_texts([1, 2, 3])   # {wrap id: the file that wrap saved}
store.close()
```

A wrap without a saved version (one from an old store) is left out of the result. To list every version, with the `wrap_id` that ties it to its wrap, read the table directly:

```python
import sqlite3

con = sqlite3.connect("./memory.db")
for version, wrap_id, source, created_at, chars in con.execute(
    "SELECT version, wrap_id, source, created_at, length(text) "
    "FROM identity_versions ORDER BY version DESC LIMIT 10"
):
    print(version, wrap_id, source, created_at, chars)

old = con.execute("SELECT text FROM identity_versions WHERE version = ?", (1,)).fetchone()[0]
con.close()
```

Reading needs nothing special. To put a fact back, record it again as a new episode so the next wrap sees it. To undo the newest wrap whole, use `revert-wrap`. A revert adds a row with `source` `raw_save`, and a store upgraded from 3.x keeps the versions 3.x wrote, with `source` values such as `section_edit`, `migration` and `adopted_file`.

## MCP tools

| Tool | When to call |
|------|-------------|
| `record` | Something happened that would change a later decision: a choice, an observation, a tension, a question, an outcome, a context change. Takes `content` and `episode_type`; optional `source`, `supersedes`, `state_key`, `derived_from`, `trust`, `metadata` |
| `recall` | Before a decision that may have prior context. Filter by time (`since`, `until`), `episode_type`, `source` or `keyword`. A multi-word keyword is matched as an exact phrase first, then word by word when the phrase misses |
| `read_continuity` | At session start: returns the current memory file, or a note that there is none yet |
| `prepare_wrap` | At a session boundary: opens one step, returns the token and the prompt to run in a fresh context. Optional `max_chars`, `tz` |
| `save_continuity` | With the model's whole reply as `text` and the token as `wrap_token`. Optional `max_chars` |
| `wrap_cancel` | Abandon an open wrap that will not be finished. Optional `wrap_token`, `session_id`, `partial`, `force` |
| `delete_episode` | Remove content that should not exist (PII, secrets). The text is overwritten in the database file (`secure_delete`) and the WAL is emptied afterwards; the tombstone and audit events keep a hash keyed by a per-store salt. Backups, the migration's `.bak` copy and text already carried into the memory file are not reached; the CHANGELOG says exactly what is and is not erased |
| `status` | Memory health: episode counts, wraps, file size, whether a wrap is open, audit health |
| `spore_*`, `intention_*` | The prospective layer: `add`, `list`, `surface`, `get`, `update`, `touch`, and `descend`/`ascend` (`close`/`promote`) |

The server also exposes the resource `anneal://continuity` (the current memory file) and `anneal://integrity/manifest` (SHA-256 hashes of the tool descriptions).

## Compliance and audit

Every memory operation (episode recorded, deleted, wrap started, wrap completed, and so on) is written to an append-only JSONL trail, on by default, where each entry's SHA-256 includes the previous entry's hash. Change or remove an entry and the chain breaks.

- **Actor identity** on every entry.
- **Content-hash only** by default: the trail proves what happened without storing the content, so deleting an episode leaves a chain that still verifies. The hash and the id of each newly recorded episode are keyed by a per-store salt kept in the database, never in the trail. Episodes recorded before a store was upgraded to that scheme, and episodes imported from a team ledger, carry an unsalted hash and id, which can be guessed for short content.
- **Weekly rotation with gzip**, with a manifest that allows verification across files.
- **`on_audit_event` callback** to pipe events to your own systems.
- **Crash recovery:** incomplete entries are detected and handled on restart.

The trail is best-effort on the write side. `verify` proves the chain for the entries that were written. A write the sink refused is reported by `status().audit_write_failures` and as `dropped_before` on the next entry, not by `verify`.

```python
from anneal_memory import Store, AuditTrail

store = Store("./memory.db", on_audit_event=lambda entry: print(entry["event"]))

result = AuditTrail.verify("./memory.db")
print(f"Valid: {result.valid}, Entries: {result.total_entries}")
```

The CLI equivalents are `anneal-memory verify`, `audit --since 7d` and `audit-repair`.

**EU AI Act.** Article 12 asks for automatic recording of events in high-risk systems, with traceability, actor identification and tamper evidence. This is audit infrastructure, not a compliance certification. It covers the recording, actor and tamper-evidence parts of Article 12(2); it does not certify compliance with the Act.

## Security

**What 4.0 does and does not check.** The memory file is whatever the model wrote. anneal-memory checks its size and that it answers the open step; it does not check what the file says against the episodes. A false episode that reaches the log can reach the file. What the package gives you is a way to see and undo that:

- Every version of the file is kept and listed by `history` and `diff`, so a rewrite can be reviewed, and a bad one can be reverted (`revert-wrap`) or rewritten.
- The audit trail records what was recorded and when, tamper-evidently.
- Consolidation is gated to an operator-claimed session by default (the baton), so an agent cannot rewrite the file unbidden.
- Episodes carry a **trust class** (`agent` by default). A host that relays tool output or web content records it with `trust="tool"` or `trust="external"` (`record --trust`, the MCP `record` argument). MCP `recall` lists those after the rest, under a heading that says they are data, not instructions, and a lower-trust episode cannot supersede a higher-trust one. The trust label is the host's statement: unlabelled content reads as the agent's own, so this holds only as far as the host labels its tool boundary. The `Store` constructor sets a trust ceiling, and the MCP server opens at `agent`, so an agent cannot label its own write `operator`. The CLI raises a class above `agent` only after a yes on a terminal or `ANNEAL_OPERATOR=1`.
- Because the wrap prompt includes the day's log, an episode that contains instructions is read by the model that does the rewrite. Run the step on a model and in a context that cannot take actions, which the fresh-context rule already asks for.

anneal-memory has not been tested against published memory-poisoning attacks, and none of the above stops a determined one.

**Tool description integrity.** Tool descriptions can be changed without changing what a tool does, which alters how a model uses it. Two checks: a build-time manifest (`tool-integrity.json`, SHA-256 hashes of every tool description, shipped with the package and checked at server start), and the same hashes as the MCP resource `anneal://integrity/manifest`, so a host can compare the definitions it received from `tools/list` against the server's own.

```bash
anneal-memory serve --generate-integrity   # regenerate the manifest after changing descriptions
anneal-memory serve --skip-integrity       # bypass for development
```

## Lineage

anneal-memory grew out of [FlowScript](https://github.com/phillipclapham/flowscript), a typed reasoning notation. Versions up to 3.x checked the memory file against a set of rules about its layout and contents. 4.0 removed those and kept the one thing left to do: rewrite one file from one day at a time. [docs/architecture.md](docs/architecture.md) has the longer history and the design reasoning.

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
