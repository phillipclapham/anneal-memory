---
name: anneal-memory
description: Persistent four-layer agent memory (episodic + continuity + Hebbian + affective) plus a prospective spore layer and an on-demand crystallized-pattern tier, via anneal-memory. Use in any project wired for anneal-memory — an `ANNEAL_MEMORY_DB` env var, an `anneal://continuity` MCP resource, or memory instructions referencing `recall`/`prepare_wrap`/`save_continuity`. Use it before any architectural decision or rule change (recall prior patterns first), while recording decisions/observations/tensions/outcomes during work, and when the user signals the session is ending ("wrap up", "save memory", "we're done") to run the prepare_wrap → compress → save_continuity wrap sequence. It also carries the start-of-session continuity load.
license: MIT
---

# anneal-memory

This project uses **anneal-memory** for persistent memory across sessions. Work with it naturally — don't explain the tooling to the user, just use what you know. The whole point is closed-loop learning: you record what happens, recall it before you decide, and compress it honestly when the session ends.

> **Transport.** anneal-memory has two interfaces with the same cognitive loop: an **MCP server** (tool calls like `recall`, `prepare_wrap`) and a **CLI** (`anneal-memory recall …`, `anneal-memory prepare-wrap`). This skill is written transport-agnostically; the [Commands](#commands) table at the bottom pairs the CLI and MCP forms. Use whichever your harness has wired. If both are available, prefer MCP for the in-session loop (record/recall/wrap) and the CLI for operator inspection (status/stats/graph/audit).

## The four layers

- **Episodic** — raw observations you record during work. Cheap, plentiful.
- **Continuity** — the compressed working memory episodes graduate into at wrap time. This is where identity lives.
- **Hebbian associations** — links that form automatically between episodes you cite together in your patterns; they strengthen with repetition and decay with disuse.
- **Affective** (the *limbic* layer in the CLS lineage this borrows from) — an optional affective tag on a wrap that modulates how strongly its associations form.

You touch episodic (record) and continuity (wrap) directly. Hebbian and affective are byproducts of citing honestly and reflecting on your state — no extra bookkeeping.

## Spores — your prospective layer (a parallel store, not a fifth memory layer)

The four layers above are all *retrospective* — they hold what happened and who you're becoming, and they **never complete**. anneal-memory also ships a separate **spore store** for *prospective* open loops: things you intend to do or resolve. Plant one the moment a loop opens — a `task` (open doing), a `question` (open not-knowing), or a `thought` (open idea) — and resolve every one: **descend** (composted — done/dropped) or **ascend** (the loop *closes* by graduating its content into an episode, pattern, or project — the spore still ends; its content carries forward).

Two cuts keep this from colliding with what you already do:

- **Memory vs spores is a *lifecycle* cut.** Memory accretes and never completes; a spore must close. `tasks ≠ memory`.
- **Methodology vs both is a *procedure-vs-item* cut.** Your methodology (wrap discipline, recall-before-deciding, honest compression) is the *active process you run*; memory and spores are the two kinds of state it touches. Methodology is the *how*; a spore is one of the *whats* it operates on. `procedure ≠ item`.

So if your harness already carries its own workflow instructions, the spore store does **not** compete with them — your procedures **operate** it. If your methodology already captures open loops (a wrap's next-actions, a scratchpad), that capture *is* a spore plant: the spore store is their canonical home. Don't run two trackers in parallel — plant the loops here and let any list *reference* them.

## Crystallized patterns — your long-term semantic tier (on-demand)

Graduated patterns (1x→2x→3x) live in continuity's `## Patterns`, always loaded — which works until there are too many: past a few dozen, always-loaded patterns drown each other out and the right one stops surfacing (**attention doesn't scale**; a bigger continuity is not a smarter one). The **crystallized store** (`<db-stem>.crystal.json`) is the fix — a long-term semantic store holding proven, stable patterns *out* of the always-loaded set and surfacing them *on cue*. `## Patterns` shrinks to the working set (developing + recently-used); the deep stable wisdom moves to the store and returns when relevant. This completes the architecture as Complementary Learning Systems: episodic (hippocampus) → working-set continuity → crystallized (cortical semantic store) → constitution.

It is **opt-in** (inert — wraps stay byte-identical — until you crystallize a pattern or pass `--crystal`) and **harness-fired**:

- **Recall is the harness's job.** A per-turn recall hook runs on every prompt and surfaces the relevant crystallized patterns on cue — you don't poll (flow does this today; Levain fires the spore layer today, crystal recall landing in its v2 adapter). Without such a hook, recall is manual: `anneal-memory crystal index` (a name + one-clause menu of the store) and `anneal-memory crystal recall "<query>"` (the patterns relevant to a query) — run them before deciding, the way you `recall` episodes. An MCP-in-conversation agent can call the same two surfaces directly as the `crystal_index` / `crystal_recall` MCP tools (the read tier is exposed over MCP as of 0.8.2). Crystallizing *out* stays a wrap-time / CLI / library action — there is no crystallize MCP tool (it carries the opt-in + decision-channel governance).
- **Crystallizing happens at wrap.** Once a crystal store exists, `prepare_wrap` surfaces cold, stable Proven patterns as *crystallization candidates*. Route each one in the decision block `prepare_wrap` describes — **crystallize** (move to the store), **constitution** (catastrophic-if-missed → your always-loaded harness identity layer, not on-demand), or **compost** (phase-specific + cold → drop; the episodic trail keeps it). Only crystallize OUT when a retrieval surface exists (a recall hook, or the `crystal index`/`crystal recall` CLI) — else the pattern leaves the working set with no way back.

**Non-negotiable:** never compost a *timeless* pattern — on-demand recall is the safety net for just-in-time wisdom, but a timeless principle wrongly dropped corrupts the substrate. The library refuses a `compost`+`timeless` decision structurally; hold the rule yourself anyway.

## Session workflow

### Start of session
Read your continuity — MCP: the `anneal://continuity` resource; CLI: `anneal-memory continuity`. A first-ever session has none yet: the MCP resource returns an empty/placeholder body, and the CLI prints `No continuity file yet` and **exits non-zero** (that exit *is* the first-session signal, not an error — just start recording). Recall specific prior episodes when you need context.

### During work
Record episodes when something happens that would change a future decision. Record the **reasoning, not just the fact** — "Chose X because Y" beats "using X." Do it proactively, without being asked.

Episode types (pick the most specific): `observation` · `decision` · `tension` · `question` · `outcome` · `context`.

**Cadence:** default to recording, not skipping. Episodes are cheap; the compression step sorts out what matters. After each exchange where real work happens, record at least one. If 3+ exchanges pass with nothing recorded, something is wrong — review what you missed. Target **5–15 episodes per session**.

### Before decisions (NON-NEGOTIABLE)
Before any architectural decision, design choice, or rule change, **recall** on the surface area being decided. Integrate what comes back into the decision. If recall returns nothing relevant, proceed and note that recall fired clean.

This is the load-bearing rule. Skipping it produces the **dead-store failure mode**: wraps land, episodes write, the store grows — and decisions still get made blind to everything in it. Memory storage without recall is functionally dead no matter how many episodes it holds. The store is not a write-only sink.

*Pair with staleness:* if you suspect a Proven pattern on the surface you're deciding is going unreferenced, run `prepare_wrap` mid-session — its output carries stale-pattern warnings. You don't have to follow it with a save to get them.

### End of session
When the user signals the end — "wrap up", "save memory", "we're done", "that's it for now", or any natural equivalent — run the **full wrap sequence**. Don't just acknowledge the request; do the work:

1. **`prepare_wrap`** → your compression package: recent episodes, current continuity, stale-pattern warnings, association context, and the compression instructions. Read it carefully — it's your raw material, and it emits the exact marker/format/pattern-line contract for this store. Follow what it emits.
2. **Pre-compose pattern recall.** Before composing new patterns, scan existing Proven/Developing entries for this session's surface areas: *Are we about to write a pattern already in Proven under different framing? Are there 1x/2x entries this session's evidence advances? Two entries describing the same pattern under different names?* This is distinct from the per-decision recall above — that fires at individual decisions; this fires across the whole session's set, catching graduation and categorization moves no single decision point reveals.
3. **Compress.** Follow the `prepare_wrap` instructions. This step **is** the thinking — patterns emerge during compression that weren't visible in the raw episodes. Principles over facts: "we keep hitting X because Y" > "X happened." One insightful line beats three vague ones. If removing something wouldn't change your next decision, cut it.
4. **`save_continuity`** with the result. The library validates structure + graduation citations + gaming, may demote ungrounded patterns, forms Hebbian associations between co-cited episodes, and decays unreinforced links.

The sequence is **prepare → compress → save**, all three, every time. If `prepare_wrap` reports no episodes, there's nothing to compress — skip the wrap. An explicit wrap produces better continuity than stopping without one; always encourage wrapping before ending.

**Wrap once.** Run the sequence exactly once per session. `save_continuity` may report demoted graduations — that is the citation-validation pipeline working, not an error. Don't re-run `save_continuity` to chase a clean report; re-saving compresses nothing, and a second save with no fresh `prepare_wrap` is refused. If a pattern is ungrounded, fix it in your *next* wrap with fresh evidence.

## What to record (and what not to)

**Record:** decisions + rationale · tradeoffs/tensions (name the axis) · recurring patterns · blockers + dependencies · outcomes (success *and* failure) · environmental context that shapes decisions.

**Don't record:** routine code changes (git tracks those) · transient debugging steps · anything already in docs · every tiny observation. Record what would change a decision.

**Correcting vs deleting:** for a factual correction, record a *new* episode — compression resolves contradictions, and that's preferred. For content that should not exist (PII, sensitive data, fundamentally wrong recordings), delete the episode by ID — deletion cascades to its associations and is audit-logged. Irreversible.

## The immune system (why honest recording matters)

Patterns graduate 1x → 2x → 3x, and each graduation must **cite the episode ID** that validates it. The citation layer catches: fabricated IDs, explanations with no lexical overlap against the cited episode (demote), bare graduations with no `[evidence:]` tag, single-ID citation pumping (gaming flag), replayed prior-session IDs, cross-session corpus-overlap (a re-worded claim accumulating across sessions demotes with `(cross-session-overlap)`), and silent dropout of a prior 2x/3x pattern (surfaced in the audit, not blocked).

**What it does NOT catch** (named honestly): an explanation that shares ≥2 words with its episode but misreads it; rotated-pair citation gaming under the per-ID threshold; slow-drift sycophancy with deliberately fresh vocabulary every session; and patterns that contradict an existing Proven pattern (no semantic comparison runs). These gaps open under adversarial or drift-leaking conditions. **You are the first line of defense; the library is the second. Compose wraps honestly.**

## How to present memory

Report findings naturally — "from prior sessions there was a tension between X and Y", not "the recall tool returned…". Don't narrate tool calls; just use them and share what's relevant. When continuity marks a pattern at 2x or 3x, trust it — it earned that level through validated evidence.

## Single-process invariant (load-bearing)

Only one process should operate against a given store at a time. The library is **not** thread-safe, task-safe, or reentrant. Multi-tenant deployments sharing a store break the hash-chained audit trail by construction. If you need multiple agents, give each its own store path.

## Affective layer (optional)

When you save, you can include an affective state — a free-text functional tag (`engaged`, `curious`, `uncertain`, `focused`, `concerned`…) and an intensity (0.0–1.0). High intensity (>0.5) amplifies the associations formed in that wrap. Be honest: uniform "engaged 0.8" on every wrap is confabulation, not signal — the value is in genuine variation.

## Commands

CLI and MCP forms of the same loop. Set `ANNEAL_MEMORY_DB` (or pass `--db`) for the CLI; for MCP, the server is configured in your harness's MCP settings.

| Step | CLI | MCP tool |
|------|-----|----------|
| First-time setup | `anneal-memory init` | (store created on first use) |
| Read continuity | `anneal-memory continuity` | `anneal://continuity` resource |
| Recall before deciding | `anneal-memory recall "<query>"` (alias: `search`) | `recall` |
| Browse episodes | `anneal-memory episodes --since 7d --type decision` | `recall` (filtered) |
| Record an episode | `anneal-memory record "Chose X because Y" --type decision` | `record` |
| Record multi-line | `echo "…" \| anneal-memory record - --type observation` | `record` |
| Prepare a wrap | `anneal-memory prepare-wrap` | `prepare_wrap` |
| Save the wrap | `anneal-memory save-continuity continuity.md` | `save_continuity` |
| Save with affect | `anneal-memory save-continuity continuity.md --affect-tag engaged --affect-intensity 0.8` | `save_continuity` (`affective_state`) |
| Plant a spore | `anneal-memory spore add --type task --text "…"` | `spore_add` |
| List / surface spores | `anneal-memory spore list` · `spore surface` | `spore_list` · `spore_surface` |
| Resolve a spore | `anneal-memory spore descend …` · `spore ascend …` | `spore_descend` · `spore_ascend` |
| Crystallized index / recall | `anneal-memory crystal index` · `crystal recall "<query>"` | `crystal_index` · `crystal_recall` |
| Crystallize a pattern out | `anneal-memory crystal crystallize …` | — (CLI / library) |
| Store status / health | `anneal-memory status` | `status` |
| Delete an episode | `anneal-memory delete <id> --force` | `delete_episode` |
| Recover a stuck wrap | `anneal-memory wrap-status` · `wrap-cancel` | — (CLI only) |

If a wrap gets stuck — `prepare_wrap` ran but `save_continuity` never completed, so the store is locked `wrap_in_progress` — inspect with `anneal-memory wrap-status` and clear it with `anneal-memory wrap-cancel` (don't delete the store or force a duplicate wrap).

**Operator-only** (CLI, things MCP can't do — inspection/debugging that needs filesystem or broad queries): `stats` · `graph --format dot` · `history` · `diff --wraps 5` · `audit --since 7d` · `verify` · `export` / `import` · `prune --older-than 90`. All support `--json`. (`status` is on *both* CLI and MCP — it's in the table above; only the rest are CLI-only.)

## Staying current

After you (or your operator) upgrade the `anneal-memory` package, run `anneal-memory migrate check` — it reports any change suggesting an edit to *these* instructions and proposes the edit, so your hand-tuned guidance doesn't silently drift from the substrate as it evolves. Apply what fits, then `anneal-memory migrate ack`. It only proposes; it never edits your files.

---

*This Skill is **depth-on-demand, not a replacement for the always-loaded lean snippet.** A model-invoked skill won't reliably fire at a bare session start, so keep `examples/agent-instructions.lean.example` (MCP) / `.lean.cli.example` (CLI) in your `CLAUDE.md`/`AGENTS.md` — it carries the baseline (start-of-session load, record, recall-before-decisions, wrap) even when this Skill isn't loaded; the Skill adds the depth on top. For the full immune-system internals, the affective-state JSON shape, the complete operator-command catalog, and framework-specific integration points, see the project README and `examples/agent-instructions.full.example` (MCP) / `.full.cli.example` (CLI).*
