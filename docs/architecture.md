# anneal-memory: how it is put together, and where it came from

This is the depth behind the [README](../README.md). It covers what is stored where, why consolidation is a single method, what an audit trail does and does not give you, and where the project came from.

## What is stored where

One agent, one store. The store is a SQLite database plus a few files beside it.

| What | Where | Notes |
|---|---|---|
| Episodes | the database | Typed (`observation`, `decision`, `tension`, `question`, `outcome`, `context`), timestamped, append-only except for `delete` and `prune`. Recall reads these |
| Supersession links and state keys | the database | They hide an old fact from recall when a newer episode replaces it. Nothing is deleted |
| The memory file | the database (`identity_versions`), projected to `<stem>.continuity.md` | Every saved version is a row. The file beside the database is a copy of the newest one |
| Wraps | the database | One row per completed step: when, how many episodes, file size, model |
| The audit trail | `<stem>.audit.jsonl` beside the database | Hash-chained JSONL, rotated weekly into gzip files with a manifest (`<stem>.audit.manifest.json`) |
| Spores and intentions | `<stem>.spores.json` | The prospective store: open loops that must close |
| The baton | `<stem>.continuity.md.baton` | The session id the operator has claimed for consolidation, and when |

The episode log answers "what happened". The memory file answers "what should the next session know". They are separate on purpose: the file is small and always loaded, and the log is large and searched on demand.

## One consolidation method

A wrap is a rolling rewrite. For each calendar day that has unwrapped episodes, oldest first, a model is given the current memory file and that day's log and replies with the whole new file. anneal-memory builds the prompt and keeps the reply. It calls no model, so it works with any.

Three decisions follow from that.

**The step is one day.** A catch-up over a week is seven steps, each starting from the file the last one saved. Merging the days into one log would make the model decide at once what to do with a week of changes, and it would hide which day changed what. A day over 400,000 characters of log is cut into slices.

**The step runs in a fresh context.** The model sees the prompt and nothing from the session that asked for the wrap. The agent that did the day's work is attached to it. A fresh reader asks only what the file should say given the log.

**The save checks little.** The reply must be non-empty, at most the character cap (26,000 by default), and must answer the step that was opened. It is not parsed. Whatever the file contains is the model's choice, and the check that matters is a human or a later reader looking at it. That is why every version is kept: a rewrite that dropped something is found with `history` and `diff` and the dropped fact is read back out of `identity_versions`.

Earlier versions asked the model to write a fixed set of sections and checked the result against rules about its layout and contents, including a shrink gate. That added a lot of code and a set of rules the model had to satisfy, and 4.0 removed it. The README's [Security](../README.md#security) section says what that means for how far you can trust the file.

## The baton

Consolidation rewrites what the agent loads at the start of every session, so it is gated. A store created by the CLI requires the baton: a wrap runs only under a session id the operator claimed with `baton claim`. Agents record freely and cannot claim the baton through MCP. Leases and idempotency keys cover a consolidator that crashes or retries. The gate stops an agent that follows its instructions and parallel sessions overwriting each other. It does not stop an agent with a shell that ignores its instructions.

## Provenance and timestamps

The EU AI Act's Article 12 asks for automatic recording of events in high-risk systems, with traceability. A log can meet that at two levels: timestamps (which entry was written when) or a chain where each entry is bound to the one before it, so that removing or editing one is detectable. The audit trail is the second kind: each entry's SHA-256 includes the previous entry's hash. Episodes are timestamped and typed. Each wrap row records when it ran, how many episodes it covered, the size of the file it saved and, if you passed them, the model and whether it was unattended, and the audit trail records the wrap starting and completing.

What the chain does not tell you is why the model wrote a given line in the file. In 4.0 nothing links a sentence in the file to a specific episode. The log, the wrap record and the saved version give you the input and the output of each step, which is what you can audit. This is audit infrastructure, not a compliance certification.

## Lineage

anneal-memory did not start as a memory library. It started as **FlowScript**, a typed reasoning notation: a small set of markers (`->` for causation, `?` for an open question, `><` for a tension on an axis) written alongside ordinary prose, compiled to a queryable graph. The bet under it was about language: that the structure of a notation changes what can be thought in it. Stated as a thesis, not a finding, the short form was "language = consciousness = reality".

The notation turned out to carry things a memory system needs: content-hash ids, provenance, schema validation, typed relationships. The part that lasted was not the syntax. Asking someone to learn 21 markers is a barrier, and what you hand them is your artifact, not your practice. What lasted was the idea that compression is where the thinking happens: an agent that rewrites what it knows from what just happened ends up with something it did not have before. anneal-memory is that idea without the notation. FlowScript itself is archived at [github.com/phillipclapham/flowscript](https://github.com/phillipclapham/flowscript).

The version-by-version history, including what 4.0 removed, is in the [CHANGELOG](../CHANGELOG.md).

The reason for owning the store is the same one it started with. A human and an AI working in a shared, persistent memory can build on each other's thinking, and that is hard to do with a memory you cannot inspect, that a vendor can revoke, or that rewrites itself unbidden. So the store is local, has no dependencies, keeps every version, and is rewritten only under a claim the operator made.

## On benchmarks

anneal-memory has no score on LOCOMO or similar conversational-recall benchmarks. They measure whether an agent can retrieve facts from a long dialogue. anneal-memory's job is narrower: keep a small file current across sessions and keep a record that can be audited. If a comparison calls for a number, it will be run and reported with what it measures.
