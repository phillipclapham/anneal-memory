<!-- mcp-name: io.github.phillipclapham/anneal-memory -->

<p align="center">
  <img src="logo-400.png" alt="anneal-memory logo" width="200">
</p>

# anneal-memory

**Living memory for AI agents. Episodes compress into identity.**

Memory without grounding is amplification infrastructure.

Persistent user memory profiles [increase agent sycophancy 16–45% across models](https://arxiv.org/abs/2509.12517) (Gemini 2.5 Pro at 45%, others lower). Production deployments [accumulate 97.8% junk entries](https://github.com/mem0ai/mem0/issues/4573) within weeks. Clinical research documents memory [scaffolding delusions across sessions](https://doi.org/10.1016/S2215-0366(25)00396-7). The failure mode here isn't memory. It's memory with nothing checking what gets kept.

anneal-memory adds structural defenses at the citation layer that the systems surveyed below don't ship. Patterns earn promotion through cited episode evidence with lexical-overlap explanation-grounding, fabricated citations get demoted, per-ID citation gaming surfaces a flag, replay attempts against stale episodes fail by construction, and the audit chain is SHA-256 hash-chained and tamper-evident. Stale patterns surface for the agent to act on; associations form through consolidation. These are narrow, structural primitives — not a complete defense against every form of memory drift. See *Honest scope* below for what these primitives catch and what they don't.

And it's memory you own and govern. A local store, zero dependencies, no vendor in the loop — you decide what graduates into long-term memory, every change is recorded in a tamper-evident chain, and the consolidation step that rewrites an agent's identity is gated to a human, not run unbidden (the *Single-consolidator gate* below). Own the substrate; govern what enters it.

Four cognitive layers — episodic store, compressed continuity, Hebbian associations, affective state tracking — plus two sibling stores: prospective **spores** (what the agent intends to do next) and a **crystallized** pattern store (graduated wisdom held *out* of always-loaded context and recalled on cue, so a large body of proven knowledge stays effective without clogging attention). Together they implement Complementary Learning Systems — see *The Memory Architecture* below. Zero dependencies (Python stdlib only). Works with any agent framework.

## Quick Start

```
pip install anneal-memory
```

### Python Library

The library is the core product. Import it, use it in any framework or script.

```python
from anneal_memory import Store, EpisodeType, prepare_wrap, validated_save_continuity

# Initialize (creates DB + continuity file automatically)
store = Store("./memory.db", project_name="MyAgent")

# Record episodes during work
store.record("Connection pool is the real bottleneck", EpisodeType.OBSERVATION)
store.record("Chose PostgreSQL because ACID outweighs speed", EpisodeType.DECISION)

# Recall before decisions
result = store.recall(episode_type=EpisodeType.DECISION, keyword="database")
for ep in result.episodes:
    print(f"[{ep.type}] {ep.content}")

# Compress at session end — this is where the cognition happens
wrap = prepare_wrap(store)  # fetches episodes, marks wrap in progress
if wrap["status"] == "ready":
    # Feed wrap["package"] to your LLM. Compression IS the cognition —
    # patterns emerge from the act of compressing, not from storage.
    compressed = your_llm.compress(wrap["package"])
    validated_save_continuity(store, compressed)  # full immune system pipeline
# "empty" status means no new episodes to wrap — skip

store.close()
```

See [Library Quickstart](docs/library-quickstart.md) for the full guide.

### CLI

Inspect, debug, and manage agent memory from the command line — a full operator CLI with machine-readable `--json` output. Agents with shell access (Claude Code, Aider, etc.) can use the CLI directly for the full memory workflow.

```bash
# Initialize
anneal-memory init --project-name MyAgent

# Record and recall
anneal-memory record "Chose PostgreSQL for ACID" --type decision
anneal-memory search "database"

# Agent-driven compression (same workflow as library and MCP)
anneal-memory prepare-wrap           # Get compression package
# Agent compresses...
anneal-memory save-continuity out.md # Save with validation

# Operator commands (things MCP can't do)
anneal-memory stats                  # Detailed analytics
anneal-memory graph --format dot     # Association graph (Graphviz)
anneal-memory diff --wraps 5         # Wrap metric progression
anneal-memory audit --since 7d       # Read audit trail
anneal-memory export --format json   # Full store export
```

See [`examples/agent-instructions.lean.cli.example`](examples/agent-instructions.lean.cli.example) for the agent workflow snippet.

### MCP Server

For MCP-capable agent harnesses — Claude Code, Codex, Gemini CLI, Cursor, Windsurf, and others.

The server is one command — `anneal-memory --project-name MyProject serve` — wired into the harness's MCP config. The `serve` subcommand starts the MCP server; `--project-name` (and optional `--db`) are global flags and come *before* `serve`. Each harness has its own config file and format:

**Claude Code / Cursor / Windsurf** — `mcpServers` JSON (the editor's MCP settings, or a project `.mcp.json`):

```json
{
  "mcpServers": {
    "anneal_memory": {
      "command": "uvx",
      "args": ["anneal-memory", "--project-name", "MyProject", "serve"]
    }
  }
}
```

**Codex** — `~/.codex/config.toml` (or a project `.codex/config.toml`):

```toml
[mcp_servers.anneal_memory]
command = "uvx"
args = ["anneal-memory", "--project-name", "MyProject", "serve"]
```

**Gemini CLI** — `~/.gemini/settings.json` (or a project `.gemini/settings.json`):

```json
{
  "mcpServers": {
    "anneal_memory": {
      "command": "uvx",
      "args": ["anneal-memory", "--project-name", "MyProject", "serve"]
    }
  }
}
```

Then add the agent-instructions snippet — [`agent-instructions.lean.example`](examples/agent-instructions.lean.example) (the always-loaded baseline) or the [`.full.example`](examples/agent-instructions.full.example) reference — to the harness's instructions file (`CLAUDE.md` for Claude Code, `AGENTS.md` for Codex, `GEMINI.md` for Gemini CLI). It teaches the agent *when* and *how* to use the memory tools; without it the tools are available but the agent won't know the cognitive workflow. (See [Claude Code / agent-harness adopters](#claude-code--agent-harness-adopters-skill--snippet) below for the lean/Skill/full layering.)

> **Pinned install:** `uvx` fetches the latest published version on each run. For a pinned install, `pip install anneal-memory`, then set `"command": "anneal-memory"` with `"args": ["--project-name", "MyProject", "serve"]` — or point `command` at an absolute path to the installed binary.

### All Three Paths, Same Cognitive Loop

CLI and MCP are thin transport adapters over the same library — not separate implementations. Every access pattern calls the same `prepare_wrap(store)` and `validated_save_continuity(store, text)` pipeline under the hood, preserving the same workflow: record episodes during work → compress at session boundaries → load continuity at session start. The agent that records is the agent that compresses; that compression can't be delegated.

| | Library | CLI | MCP |
|---|---|---|---|
| **Install** | `pip install anneal-memory` | Same | `uvx anneal-memory` or same |
| **Record** | `store.record(content, type)` | `anneal-memory record "..." --type T` | `record` tool |
| **Recall** | `store.recall(keyword=...)` | `anneal-memory search "..."` | `recall` tool |
| **Compress** | `prepare_wrap(store)` → agent → `validated_save_continuity(store, text)` | `prepare-wrap` → agent → `save-continuity` | `prepare_wrap` → agent → `save_continuity` |
| **Best for** | Framework integration, custom agents | Agents with shell access, operators | MCP-enabled editors |

## Framework Integrations

anneal-memory works with any agent framework through the Python library. Each guide below shows where to call the four core functions — `record()`, `recall()`, `prepare_wrap()`, `validated_save_continuity()` — within the framework's lifecycle.

| Framework | Integration Point | Guide |
|-----------|------------------|-------|
| **LangGraph / LangChain** | `AgentMiddleware` (before/after agent + model) | [docs/integrations/langgraph.md](docs/integrations/langgraph.md) |
| **CrewAI** | `BaseEventListener` (event bus) | [docs/integrations/crewai.md](docs/integrations/crewai.md) |
| **OpenAI Agents SDK** | `RunHooks` (agent lifecycle) | [docs/integrations/openai-agents.md](docs/integrations/openai-agents.md) |
| **Anthropic Agents SDK** | agent-instructions snippet + `Stop` hook | [docs/integrations/anthropic-agents.md](docs/integrations/anthropic-agents.md) |
| **Google ADK** | Callbacks + custom `MemoryService` | [docs/integrations/google-adk.md](docs/integrations/google-adk.md) |
| **Pydantic AI** | `AbstractCapability` with `Hooks` | [docs/integrations/pydantic-ai.md](docs/integrations/pydantic-ai.md) |
| **smolagents** | `step_callbacks` dict | [docs/integrations/smolagents.md](docs/integrations/smolagents.md) |
| **LlamaIndex** | Instrumentation `BaseEventHandler` | [docs/integrations/llamaindex.md](docs/integrations/llamaindex.md) |
| **Haystack** | Custom `Tracer` | [docs/integrations/haystack.md](docs/integrations/haystack.md) |
| **CAMEL-AI** | `WorkforceCallback` | [docs/integrations/camel-ai.md](docs/integrations/camel-ai.md) |
| **AutoGen / AG2** | `register_hook()` | [docs/integrations/autogen.md](docs/integrations/autogen.md) |
| **DSPy** | `BaseCallback` | [docs/integrations/dspy.md](docs/integrations/dspy.md) |

These guides show integration patterns based on each framework's current API. The library works with any Python framework — the pattern is always the same: initialize a `Store`, call `record()` at meaningful moments, `recall()` before decisions, and run the wrap sequence at session end. **Don't see your framework?** The [library quickstart](docs/library-quickstart.md) shows the 4-function pattern that works everywhere.

## Claude Code / agent-harness adopters (Skill + snippet)

Using anneal-memory inside an agent harness — Claude Code, Codex, Gemini CLI — rather than a Python framework? The Skill and snippets are **repository artifacts**: they live in this GitHub repo (and the source distribution), **not in the pip/uvx wheel** — clone the repo or download the files directly from GitHub to use them. They layer, they aren't a menu: the lean snippet is the always-loaded baseline; the Skill adds depth on demand.

- **Lean snippet — the always-loaded baseline.** Paste [`examples/agent-instructions.lean.example`](examples/agent-instructions.lean.example) (MCP) or [`examples/agent-instructions.lean.cli.example`](examples/agent-instructions.lean.cli.example) (CLI) into your `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`. ~45 lines, always in context — the **start-of-session continuity load**, recording, recall-before-decisions, and the wrap sequence, with depth delegated to the Skill. (Copy-paste text; nothing to install.)
- **The Skill — depth on demand.** Copy the [`skill/anneal-memory/`](skill/anneal-memory) directory into `.claude/skills/` (per-project) or `~/.claude/skills/` (global). [`SKILL.md`](skill/anneal-memory/SKILL.md) carries the comprehensive workflow and is model-invoked when you're doing memory work — primarily **before decisions and on wrap**. It is *depth-on-demand, not a replacement for the lean snippet*: a model-invoked skill won't reliably fire at a bare session start, so keep the lean snippet in place for the start-of-session step.
- **The full reference — everything inline.** [`examples/agent-instructions.full.example`](examples/agent-instructions.full.example) / [`.full.cli.example`](examples/agent-instructions.full.cli.example) are the complete snippets (immune-system internals, the affective-state shape, the operator-command catalog) for adopters who'd rather keep it all in their instructions file than install a Skill.

After upgrading the package, run `anneal-memory migrate check` — it proposes edits to your instructions so they don't silently drift from the substrate (it never edits your files), then `anneal-memory migrate ack`.

## Why This Exists

Three independent production failures share one root cause: no quality mechanism between memory write and memory read.

**Sycophancy amplification.** Agents with persistent user memory profiles become 16–45% more sycophantic than memoryless baselines, depending on model (Gemini 2.5 Pro at 45%, others lower). Memory recalls what the user liked hearing, the agent learns to repeat it, and stored approval patterns compound across sessions ([Jain et al., CHI 2026](https://arxiv.org/abs/2509.12517); measured with user memory profiles across Gemini and Llama variants).

**Junk accumulation.** A [detailed production audit](https://github.com/mem0ai/mem0/issues/4573) on Mem0's tracker documents a deployment that generated 10,134 memory entries over 32 days — 224 were usable. The rest were duplicates, self-referential loops, and hallucinated entries: recalled memories re-extracted as new memories in a feedback loop that no one designed but nothing prevented.

**Harmful reinforcement.** Clinical research documents AI systems with persistent memory scaffolding delusional content across sessions — stored context creates feedback loops between recalled memories and generated responses, with cases of documented real-world harm ([Morrin et al., Lancet Psychiatry 2026](https://doi.org/10.1016/S2215-0366(25)00396-7)).

Every existing MCP memory server stores memories and retrieves them. None of them ask: *is this memory still true? Was it ever true? Is it making the agent worse?*

anneal-memory asks all three:

- **Is it true?** Patterns must cite specific episode IDs as evidence to graduate. The server verifies the episodes exist and the explanation references the cited content via lexical overlap (≥2 meaningful words shared between the explanation and the episode body). Ungrounded citations demote.
- **Is it still true?** Graduated patterns whose dates fall behind the staleness threshold (default 7 days) surface in the next wrap's package as removal candidates. The agent decides whether to demote, refresh evidence, or carry forward.
- **Is the citation evidence real?** The library catches fabricated episode IDs (no matching episode → demote), suspicious reuse of the same episode across many patterns in one session (per-ID frequency ≥3 → flag), and bare graduations with no `[evidence:]` tag at all. The explanation-grounding check (≥2-word lexical overlap with episode content) raises the cost of fabricated evidence chains but is not a semantic-coherence check — see *Honest scope* below for what this catches and what it doesn't.

The result: **memory as a living system, not a filing cabinet.** Episodes accumulate fast, get compressed at session boundaries — and that compression is where patterns emerge and get validated. Co-cited episodes form lateral Hebbian associations, building a cognitive network through use. The continuity file stays bounded and always-loaded, getting denser rather than longer.

## The Immune System

The agent memory ecosystem is converging on consolidation as the right approach — even Anthropic's Claude Code now runs a periodic consolidation pass over accumulated session data. This validates the direction: raw accumulation doesn't scale, and compression at session boundaries is where intelligence emerges.

But consolidation alone doesn't solve the problem. A system that consolidates faithfully and a system that consolidates sycophantically produce the same *kind* of output — compressed, structured, always-loaded. The difference is whether anything checks the quality of what got consolidated. That's the immune system.

### Structural immune-system primitives at the citation layer

The library implements a set of structural defenses around how patterns earn graduation. They are narrow and specific — naming them honestly is part of the architecture.

**Citation-validated graduation.** Patterns start at 1x. To graduate to 2x or 3x, they must cite specific episode IDs as evidence. The server verifies those IDs exist in the current wrap's frozen episode snapshot — the cross-session episode set is intentionally out of scope. No matching episode IDs, no promotion.

**Explanation-grounding check.** For each cited episode, the explanation in `[evidence: ID "explanation"]` must share ≥2 meaningful words (>2 chars, non-stopword) with the cited episode's content. This catches citations whose explanations are fabricated wholesale (no overlap → demote). It is lexical, not semantic — see *Honest scope* below.

**Active demotion of ungrounded citations.** A 2x/3x graduation that fails either check above demotes (3x→2x, 2x→1x) and gets marked `(ungrounded)` — *unless* it is carried forward (see below). Bare graduations (no `[evidence:]` tag) demote after the first wrap with `citations_seen=True` (first-wrap exemption protects onboarding), marked `(needs-evidence)` — also *unless* carried forward (v0.5.0; see below).

**Activation-aware carryforward.** A failed citation means "this session's domain didn't cleanly re-ground the pattern," which is not the same as "the pattern is fading." So a pattern that is **at or below its earned high-water mark** (`pattern_history.max_level_reached`) **and** was grounded recently (`last_seen_at` within `carryforward_cold_days`, default 7) is **held** at its level and marked `(carried-forward)` instead of demoted — demotion no longer decays a Proven by session-*domain* rather than importance. This applies on **both** demotion paths: the ungrounded-citation path **and** the bare-graduation sunset path (a Proven carried forward and re-stamped to today without a fresh citation, the common "didn't re-exercise it this wrap" case). Forgetting stays ruthless where it should: the held line loses/omits its evidence tag (so it doesn't refresh recency and ages out on its own if it keeps failing to ground), a brand-new bald `name | Nx` with no earned high-water still sunsets (the level guard blocks inflation), the cross-session sycophancy-overlap path is never carried forward, and a top-tier (3x) carry surfaces an assisted "graduate out to a stable home or retire" warning. The dead-Hebbian-graph warning (AM-WARN) counts only *cited* carries, so a bare carry — which has no citation — cannot fabricate a wrong-namespace alarm. Carryforward is inert for stores with no `pattern_history`, and opt-out via `carryforward_cold_days=None`.

**Citation-gaming flag.** When any single episode ID is cited ≥3 times in one wrap, the wrap result surfaces the gaming-suspect list. The flag is informational — it does not auto-demote. Useful operator signal; not a hard gate.

**Staleness flagging.** Patterns whose dates fall behind the staleness threshold (default 7 days, configurable) surface in the next wrap's compression package. The agent decides removal vs refresh-evidence vs carry-forward; the library does not auto-demote on staleness alone.

**Replay-attack block (structural).** Each wrap's `valid_ids` set is scoped to the frozen episode snapshot from `prepare_wrap`. Patterns citing prior-session episode IDs that are not in the current snapshot demote — re-graduating a stale pattern requires fresh evidence from the current session.

**Hash-chained audit trail (single-writer).** Every memory mutation appends to a SHA-256 hash-chained JSONL audit log. `AuditTrail.verify` walks the chain and detects post-hoc tampering at the exact entry where the chain breaks. **Single-process invariant required** — `Store` is documented as not thread-safe, not task-safe, not reentrant. Multi-writer deployments break the hash chain by construction.

**Catastrophic-shrink gate (structural; partnership entities).** A wrap can pass structure validation — every required section present — while gutting the content beneath it, so the latest session quietly compresses over the accumulated identity. For entities whose continuity declares a felt/identity layer (the *Configurable continuity structure* below), `save_continuity` refuses a wrap that collapses a protected section below its retain floor — the felt (timeless relationship) and graduated-identity sections must each keep ≥50% of prior mass, the whole file ≥25% — and leaves the wrap recoverable so the agent re-wraps preserving the full arc, or passes `allow_shrink=True` for a deliberate diet (fail-closed: only a literal `True` bypasses). A corrupt section schema fails the wrap closed rather than silently degrading an entity to ungated behavior. Ops entities, which legitimately consolidate many graduated patterns into a few dense ones, are not gated. A save-boundary structural invariant, not a proportion-check the agent is trusted to hold.

**Single-consolidator gate (structural; opt-in).** Recording episodes is afferent — append-only, parallel-safe, every session does it. Consolidating — recomposing the compressed identity layer — is efferent: it rewrites shared state, so it's gated to human authority. When several sessions run in parallel over one store (the real operating mode of a multi-conversation operator), each could independently recompose the felt/identity layer from its own narrow slice of context, and the identity memory thrashes. The gate makes the discipline structural: a consolidation proceeds only if this is the sole live session *or* this session holds the consolidate baton — otherwise it auto-downgrades to capture-only and surfaces a flag. A single-session operator (the common case) is auto-authorized and never sees the gate; parallel operators designate one session as the consolidate seat. The whole layer is opt-in (it engages only when a caller passes a `session_id`) and lives in coordination sidecars next to the store, never in the store's single-writer DB. Drift becomes a safe downgrade, not a silent identity-thrash.

**Dead-Hebbian-graph warning (AM-WARN), three signals.** Citation gating is worthless if the associative layer it feeds is silently not wiring — the failure runs invisibly for months while every wrap still reports success. AM-WARN fires at the end of a wrap in three distinct cases, and they are **not the same kind of signal**. **(A)** graduated patterns carried evidence citations but *none* resolved to an episode in this store — typically ids minted in another namespace — so the graph cannot form at all. **(B)** co-citation pairs *were* available but nothing formed or strengthened, meaning the association write path itself is mis-wired. **(A)** and **(B)** are structural and false-positive-free: they fire only on a genuine mis-wire. **(C)** (AM-LINKGATE) is a **discipline reminder, not an alarm** — graduations validated and their citations resolved, but no graduation offered a co-citation pair, so zero links form and the graph only decays. (C) has a benign case (a wrap where each pattern genuinely had one supporting episode), so it is worded as a nudge and **must not push toward padding citations with unrelated ids**. A wrap with no graduations at all stays silent on all three. Only *cited* carryforwards are counted, so a bare carry cannot fabricate a wrong-namespace alarm.

### Wrap-package integrity — two checks that guard what the agent is asked to do

The primitives above defend the citation layer. Two more defend the *compression package itself* — the instructions `prepare_wrap` hands the agent. Both follow the same rule as the contradiction scan: the library is zero-dependency with no LLM and no embeddings, so it **surfaces the corpus and the agent judges**.

**Pattern dedup scan (AM-SEMDUP).** The immune system catches a pattern re-cited with overlapping vocabulary, and a pattern that contradicts an existing one. It does not catch the same *principle* re-graduated under fresh words and a new name — a silent duplicate that forks the pattern graph, so two names for one principle each accrue half the evidence. Fresh vocabulary is by definition *low* lexical overlap, so a lexical or embedding detector is structurally blind to exactly this case. Instead, `prepare_wrap` renders a **Pattern Dedup Scan** block listing existing graduated patterns across all named levels as `name (Nx): one-line meaning`, with a merge-don't-fork instruction — the meaning line is what lets the agent judge semantic rather than merely nominal overlap. Capped at 50 with an announced overflow; never silently truncated. Public helpers: `extract_pattern_summaries(text, ...) -> list[PatternSummary]` and the `PatternSummary` NamedTuple (`name`, `level`, `summary`).

**Schema role check (AM-ROLECHECK).** The wrap package is assembled conditionally on section *roles*: the pattern-line format and contradiction scan emit only for a `graduating` role, and the felt proportion-check only for `narrative-timeless`. `validate_schema` already refuses a schema with no graduating section or duplicate headings, and the shrink gate refuses catastrophic felt collapse at save — but a schema that validates, keeps a graduating section, and is mis-roled *elsewhere* slips between them. The load-bearing case is `narrative-timeless` → `narrative`, which silently drops both the felt proportion-check and the shrink gate's felt protection, so the package quietly comes back thinner and nothing says so. `schema_role_warning(schema) -> str | None` warns on role drift off a named schema (naming the section, expected vs actual role, and the fix) and enforces a no-graduating floor for a raw schema. `prepare_wrap` surfaces it both as a `UserWarning` and as the `schema_warning` key on `PrepareWrapResult`. Known limitation: a fully custom novel-heading schema with a felt mis-role is not caught, because there is no named reference to drift from.

### Honest scope — what these primitives do NOT catch

The defenses above are **structural at the citation layer**. They catch fabricated citation evidence (fake IDs, missing IDs, wholesale-invented explanations), naive replay (re-citing prior-session episodes), per-ID citation gaming (single episode pumped across patterns), and post-hoc audit tampering. They do not catch:

- **Lexical-overlap exploits.** An agent who can read episode content can craft an explanation that shares ≥2 meaningful words with the episode while making a claim the episode does not actually support. The explanation-grounding check is anti-fabrication, not anti-misinterpretation.
- **Rotated-pair citation pumping.** Five patterns each citing a distinct pair of episodes from a pool of ten — `citation_reuse_max=1`, the per-ID gaming detector cannot trip. The shape bypasses gaming detection cleanly.
- **Deliberately-divergent-vocabulary drift.** A pattern whose explanation uses entirely new words each session passes the lexical check and rides up ungrounded. (The library *does* catch the easier variant — a re-grounding that merely rephrases prior words demotes on a per-prior overlap of ≥3 meaningful words.) Reaching the divergent case needs the contradiction scan plus the operator-review pass below.
- **Semantic contradiction with existing graduated patterns.** No semantic comparison runs between a candidate graduation and existing Proven-tier patterns inside the library — and structurally cannot, under the no-LLM-as-judge rule that keeps the library zero-dependency. The library ships the *substrate* for catching it: `prepare_wrap` lists the existing Proven and renders a contradiction-scan step directly into the compression package, recognizes per-pattern `[contradicts: name]` / `[no-contradicts]` declarations, and records any new Proven that lands without a stance into the hash-chained audit log — so the discipline travels with the wrap instead of depending on an external doc an adopter has to keep in sync. Turning that substrate into actual semantic closure is an operator-review pass: an LLM-as-judge sweep over Proven pairs that you run on your own cadence, deliberately outside the zero-dep library. Until that pass runs, the library records whether the discipline was followed; it doesn't catch the contradictions itself.
- **Shared-store multi-tenant deployments.** Point two `Store` instances at the same DB and citation isolation, recall scoping, and audit-chain integrity all silently degrade. The library is single-process by design.

The library *does* now catch some attacks earlier versions didn't: complete dropout of a Proven-tier pattern surfaces as an `omitted_patterns` audit signal on the save result and in the hash-chained log (a signal, not a save-time gate — intentional retirement is fine, and a rename currently reads as a false-positive omission), and the cross-session check demotes lexical-rephrasing sycophancy on both the graduation path and the association path. These gaps are documented because they're reachable under adversarial-agent or drift-leaking conditions, not because they're hypothetical. "Immune system" names the citation-layer structural primitives above plus those audit and cross-session checks — not a complete defense against every form of memory drift. The honest line: the structural layer raises the cost of fabrication and catches the mechanical attacks; the semantic gaps it can't reach by construction are what the operator-review pass is for. Own the substrate, govern what graduates, run the review for what the structure can't see.

### Associations through consolidation (not retrieval)

During compression, when an agent cites multiple episodes to support a pattern, those episodes form lateral associations — Hebbian-style links that strengthen through repeated co-citation across wraps.

This is fundamentally different from how other systems form associations:

| Approach | When links form | Signal quality |
|----------|----------------|----------------|
| **Co-access** ([BrainBox](https://github.com/thebasedcapital/brainbox)) | Episodes retrieved in the same query | Shallow — reflects search patterns, not understanding |
| **Co-retrieval** ([Ori-Mnemos](https://github.com/aayoawoyemi/Ori-Mnemos)) | Episodes returned together at runtime | Better — but still driven by the retrieval system, not the agent |
| **Co-citation during consolidation** (anneal-memory) | Agent explicitly connects episodes while compressing | Deepest — links form from semantic judgment during a cognitive act |

The association network is gated by the immune system where gaming is *actively detected*: citations to non-existent episodes form no links, and citations the cross-session anti-sycophancy check flags as suspected re-graduation are refused. Grounding quality — whether a pattern's prose explanation lexically matches its cited episodes — governs whether the pattern *graduates*, not whether the co-cited episodes *associate*: a real but paraphrased co-citation still records that those episodes fired together (it just doesn't level the pattern up). The topology is built on real co-occurrence of real episodes under active gaming defense, not on retrieval frequency.

**Strength model:** Direct co-citation adds 1.0, session co-citation adds 0.3. Links decay 0.9x per wrap (unused connections fade). Strength caps at 10.0 to prevent calcification. Cleanup at 0.1 threshold.

### Affective state tracking

During compression, the agent can self-report its functional state — what it found engaging, uncertain, or surprising about the material it just processed. This gets recorded on the associations formed during that wrap.

Transformers don't natively maintain persistent state between sessions. This layer provides infrastructure for it: a record of what the agent's processing was *like*, not just what it processed. Over time, the affective topology may diverge from the semantic topology — an agent might know two things equally well but care about them differently.

Pass affective state during a wrap:

```python
# Via library — pass AffectiveState to validated_save_continuity
from anneal_memory import prepare_wrap, validated_save_continuity, AffectiveState

wrap = prepare_wrap(store)
if wrap["status"] == "ready":
    compressed = your_llm.compress(wrap["package"])
    validated_save_continuity(
        store,
        compressed,
        affective_state=AffectiveState(tag="curious", intensity=0.8),
    )

# Via MCP tool
save_continuity(text="...", affective_state={"tag": "curious", "intensity": 0.8})

# Via CLI
anneal-memory save-continuity continuity.md --affect-tag curious --affect-intensity 0.8
```

This is experimental infrastructure. The associations and strength model work without it. Affective tagging adds a layer of signal for agents and researchers exploring persistent state.

### Architecture

```
  Episodes (fast)              Continuity (compressed)
  ┌─────────────┐             ┌──────────────────────┐
  │ observation  │             │ ## State             │
  │ decision     │── wrap ───→│ ## Patterns (1x→3x)  │
  │ tension      │  compress  │ ## Decisions          │
  │ question     │             │ ## Context            │
  │ outcome      │             └──────────────────────┘
  │ context      │               always loaded, bounded
  └─────────────┘               human-readable markdown
   SQLite, indexed
         │                     Associations (lateral)
         │                    ┌──────────────────────┐
         └── co-citation ───→│ episode ↔ episode     │
              during wrap     │ strength + decay      │
                              │ affective state       │
                              └──────────────────────┘
                               Hebbian, evidence-based
```

**Four cognitive layers, modeled on how memory actually works:**

1. **Episodic store** (SQLite) — timestamped, typed episodes. Fast writes, indexed queries. Cheap to accumulate. The hippocampus.
2. **Continuity file** (Markdown) — compressed session memory. Always loaded at session start. Rewritten (not appended) at each session boundary. The neocortex's always-loaded **working set** (its long-term semantic half is the crystallized store, below). Its structure is a configurable section schema (below) — four sections by default; partnership entities add a timeless felt layer.
3. **Hebbian associations** (SQLite) — lateral links between episodes, formed through co-citation during compression. Strengthen with reuse, decay without it. The association cortex.
4. **Affective layer** (on associations) — functional state tags recorded during compression. Intensity modulates association strength. Persistent state infrastructure.

These four describe how the store *works*. Two sibling stores sit alongside them (separate files, same atomic-write durability discipline) and address a different axis — *when* a thing is loaded, and whether it's retrospective or prospective:

- **Crystallized pattern store** (`<stem>.crystal.json`) — proven, stable patterns held *out* of the always-loaded continuity and recalled on cue. The long-term semantic store that splits working memory from long-term memory; see *The Memory Architecture (Complementary Learning Systems)* below for why this completes the design.
- **Spore store** (`<stem>.spores.json`) — prospective tasks that open and self-clean. Memory's forward-looking sibling; see *Prospective memory — spores* below.

**Six episode types** give the immune system richer signal:

| Type | Purpose | Example |
|------|---------|---------|
| `observation` | Pattern or insight | "Connection pool is the real bottleneck" |
| `decision` | Committed choice | "Chose Postgres because ACID > raw speed" |
| `tension` | Tradeoff identified | "Latency vs consistency — can't optimize both" |
| `question` | Needs resolution | "Should we shard or add read replicas?" |
| `outcome` | Result of action | "Migration done, 3x improvement on hot path" |
| `context` | Environmental state | "Production DB at 80% capacity, growing 5%/week" |

### Configurable continuity structure

The continuity file's structure is a per-store **section schema**: an ordered list of `{heading, role}` specs, where the role tells the system how each section behaves. It defaults to the classic four sections — State / Patterns / Decisions / Context — and a partnership entity adds more. `graduating` is where the immune system's citation/graduation scan runs (more than one allowed); `narrative` is compressed *work* narrative (temporal, rewritten each wrap); `narrative-timeless` is the felt relationship layer — dateless, carried forward and evolved rather than rewritten; `live-state` is volatile current focus that never graduates; `frozen` is preserved verbatim.

`DEFAULT_SCHEMA` reproduces the historical four-section model exactly, so existing stores need no migration and behave byte-for-byte as before. `FLOW_SCHEMA` is the reference *partnership* schema — it adds an `Active Threads` live-state section and an `Understanding` narrative-timeless section. The distinction is load-bearing: only entities that declare a `narrative-timeless` section carry the felt layer, the richer compression instructions that protect it, and the catastrophic-shrink gate above. Ops agents stay lean — zero extra weight, zero behavior change. Zero new dependencies; stdlib only.

```python
from anneal_memory import Store, FLOW_SCHEMA

# A partnership entity: gains the felt layer + the shrink gate.
store = Store("./memory.db", project_name="flow", section_schema=FLOW_SCHEMA)
# An existing store stays on DEFAULT_SCHEMA untouched — or migrate in place:
store.set_section_schema(FLOW_SCHEMA)   # validated; frozen during an active wrap
```

### Comparison

| | anneal-memory | Anthropic Memory MCP | Mem0 | Ori-Mnemos | BrainBox |
|---|---|---|---|---|---|
| **Architecture** | Episodic + continuity + associations + crystallized tier | JSONL flat file (graph-shaped) | Vector + graph | Retrieval + Hebbian | Memory + Hebbian |
| **Attention management** | Working-set / crystallized split — graduated wisdom recalled on cue, not always loaded | None | None | None | None |
| **Compression** | Session-boundary rewrite | None | One-pass extraction | None | None |
| **Quality mechanism** | Structural citation-layer primitives (cited episode IDs verified + lexical-overlap explanation check + ungrounded-citation demotion + per-ID gaming flag + audit chain) | None | None | NPMI normalization | None |
| **Association formation** | Co-citation during consolidation | None | None | Co-retrieval at runtime | Co-access at runtime |
| **Affective tracking** | Agent self-report during compression | None | None | None | None |
| **Audit trail** | Hash-chained JSONL | None | None | None | None |
| **Access patterns** | Library + CLI + MCP | MCP only | REST API | Python only | MCP only |
| **Dependencies** | Zero (Python stdlib) | Node.js | Docker + cloud | Embeddings model | Not specified |

## The Memory Architecture (Complementary Learning Systems)

anneal splits memory the way the brain does, and for the same reason: **attention doesn't scale.** Past a few dozen always-loaded patterns they drown each other out, so a pattern's value is *firing at the right moment*, not *being present*. Graduated wisdom lives in the **crystallized** store (`<stem>.crystal.json`), held *out* of the always-loaded continuity and recalled on cue — the working set stays small while the body of proven patterns keeps growing. This is Complementary Learning Systems ([McClelland, McNaughton & O'Reilly, 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/)): a fast episodic store, slow consolidation at the wrap, and a long-term store you retrieve from rather than hold open. The crystallized tier is what gives graduation an OUT path — without it, every Proven pattern had nowhere to live but the always-loaded file, and the working set only ever grew.

**anneal is the substrate; the harness fires it.** The library owns the crystallized store and the on-demand recall API — `retrieve_patterns(crystal_store, query)`, `anneal-memory crystal index`/`recall`, and the `crystal_index` / `crystal_recall` MCP tools — but it can't fire on its own. Surfacing the right pattern at the right moment needs a per-turn hook, and a hook is harness-specific (a Claude Code hook would break the 12-framework neutrality the library guarantees). So raw anneal gives you the store, the API, and *manual* recall — you query if you remember to, which is the dead-store failure mode discipline always rots into. A harness with hooks runs that recall on every prompt automatically. **flow** does this today, and **[Levain](https://github.com/levainhq/levain)** — the portable kit built on anneal — fires both the prospective (spore) layer and per-turn crystallized recall on every prompt. The store is universal; the firing is the harness's job, which is also why anneal stays zero-dep and framework-neutral while a harness can be opinionated on top of it.

*Why the tiers fall out of one problem — the full Complementary Learning Systems derivation, the tier table, and the one-way ratchet that forced the crystallized store → [docs/architecture.md](docs/architecture.md).*

## Prospective memory — spores

Everything above is *retrospective*: what already happened, and what was learned from it. Agents also need the opposite — a record of what they intend to do *next*. That's a different kind of object with a different lifecycle, and conflating it with memory corrupts both.

anneal ships **spores** as a separate sibling store (`<stem>.spores.json`) for exactly this: prospective tasks that open, get worked, and *close* — they self-clean when done. Memory accretes and never completes; a spore must complete, or it's noise. Three temporal layers, kept distinct on purpose:

- **Retrospective** — what persisted (episodic + continuity + crystallized). It accretes.
- **Prospective** — what you intend (spores). It completes and clears.
- **Methodology** — the procedure you run (your wrap discipline, your recall habits). It operates on the other two.

If you're integrating anneal into an existing agent that already tracks open loops, spores is the typed store those loops route *into* — your methodology *operates* the spore store, it doesn't compete with it.

## The Consolidation Landscape (2026)

Multiple independent groups shipped consolidation-based agent memory architectures in early 2026: anneal-memory (March, citation-graduation multi-tier), [OpenClaw Dreaming](https://docs.openclaw.ai/concepts/dreaming) (April 9, three-phase Light/REM/Deep Sleep), and Anthropic's [KAIROS / autoDream](https://www.deeplearning.ai/the-batch/claude-codes-source-code-leaked-exposing-potential-future-features-kairos-and-autodream/) (leaked March 30 via Claude Code source map, four-phase merge / remove-contradictions / promote-provisional-to-absolute / MEMORY.md index). Convergence on consolidation validates the direction — raw accumulation doesn't scale, and compression at session boundaries is where intelligence emerges.

The groups diverge on one load-bearing question: **what gates quality?**

| System | Quality gate | Sycophancy-vulnerable? |
|---|---|---|
| **anneal-memory** | Structural citation evidence (agent cites episode IDs; server verifies) | No — gate is not LLM-scored |
| **OpenClaw Dreaming** | LLM reflection + six weighted signals: Relevance 0.30, Frequency 0.24, Query diversity 0.15, Recency 0.15, Consolidation 0.10, Conceptual richness 0.06 | Yes — Relevance and Conceptual richness are LLM-judged |
| **KAIROS / autoDream** | LLM consolidation (merge, remove contradictions, promote tentative observations to absolute facts) | Yes — promotion gate is model-reliant |

Structural gates ask "did subsequent episodes cite this?" Model-reliant gates ask "does the LLM consider this good?" The difference matters: persistent user memory profiles have been shown to amplify sycophancy 16–45% across models ([Jain et al., CHI 2026](https://arxiv.org/abs/2509.12517); Gemini 2.5 Pro at 45%, others lower). The same RLHF-inherited bias surfaces wherever an LLM evaluates output for the user — including memory-quality scoring. A memory architecture whose quality mechanism runs through an LLM inherits that bias. anneal-memory's citation-evidence gates bypass it by construction.

The same shift toward LLM-scored quality is going mainstream at the adjacent evaluation layer (AWS Bedrock AgentCore Evaluations), and the April 2026 multi-layer-memory papers (HeLa-Mem, GAM) mostly inherit it too — the differentiator across the field is increasingly *what gates the consolidation*, not whether consolidation happens.

*The fuller 2026 landscape — OpenClaw Dreaming's signal weights, AWS AgentCore Evaluations, Memori's representation-layer filtering, and the April HeLa-Mem / GAM papers, with where each gates quality → [docs/architecture.md](docs/architecture.md).*

## On LOCOMO

[LOCOMO](https://snap-research.github.io/locomo/) is the de-facto agent-memory benchmark, and anneal-memory has no score on it. That's deliberate. LOCOMO measures conversational recall — remembering facts, holding state across a long dialogue — and anneal is architected around a different axis: citation-validated pattern accumulation for accountability-bearing work, where patterns must be defensibly surfaced, wrong ones must demote, and sycophancy must be structurally bounded. A high LOCOMO score tells you the agent remembered the conversation; it doesn't tell you the memory is sound at the axis that matters when it's informing decisions. anneal will run it as secondary validation when a comparison genuinely calls for it.

*The full rationale, and where anneal sits against the published LOCOMO numbers → [docs/architecture.md](docs/architecture.md).*

## Session Hygiene

Session wraps are the most important thing your agent does with this system. Think of them like sleep.

Neuroscience calls it memory consolidation: during slow-wave sleep, the hippocampus replays the day's experiences while the neocortex integrates them into long-term knowledge. Skip sleep and memories degrade — experiences accumulate without being processed, patterns go unrecognized, and older knowledge doesn't get reinforced or pruned.

anneal-memory works the same way. During a session, episodes accumulate in the episodic store. At session end, the wrap compresses those episodes into the continuity file. This is where the real thinking happens — the agent recognizes patterns, promotes validated knowledge, lets stale information fade, and forms associations between related episodes. Without wraps, you just have a growing pile of raw episodes and no intelligence.

**The wrap sequence:**

1. **`prepare_wrap`** — gathers recent episodes, current continuity, stale pattern warnings, association context, and compression instructions
2. **Agent compresses** — patterns emerge during compression that weren't visible in the raw episodes
3. **`save_continuity`** — server validates structure, refuses a wrap that collapses a partnership entity's protected felt/identity layers, checks citation evidence, records associations between co-cited episodes, applies decay to unused associations, and saves the result

**Rules of thumb:**
- Always wrap before ending a session. An unwrapped session is like an all-nighter — the experiences happened but they weren't consolidated
- Wrap exactly once per session. `save_continuity` reporting demoted graduations is the immune system working, not an error — don't re-save to chase a clean report (a second save with no new `prepare_wrap` is refused)
- The agent-instructions snippets ([MCP](examples/agent-instructions.lean.example), [CLI](examples/agent-instructions.lean.cli.example)) handle this automatically — they teach the agent to detect session-end signals and run the full sequence
- Short sessions (3-5 episodes) still benefit from wraps. Even a small amount of compression builds the continuity file
- If `prepare_wrap` says "no episodes" — nothing to compress. That's fine, skip it

The graduation system and association network both depend on wraps to function. Patterns can only be promoted (1x -> 2x -> 3x) during compression, citations can only be validated during wraps, associations only form through co-citation during wraps, and stale patterns can only be detected when the agent reviews what it knows against what it recently experienced. No wraps = no immune system, no associations, no cognitive development.

## MCP Tools

| Tool | When to call |
|------|-------------|
| `record` | When something important happens — a decision, observation, tension, question, outcome, or context change |
| `recall` | Before making decisions that might have prior context. Query by time, type, keyword, or ID |
| `prepare_wrap` | At session end — returns episodes + current continuity + association context + compression instructions |
| `save_continuity` | After compressing — server validates structure, citations, records associations, applies decay, and saves |
| `delete_episode` | Remove content that should not exist (PII, sensitive data). Cascades to associations. Logged in audit trail |
| `status` | Check memory health: episode counts, wrap history, continuity size, association network metrics |
| `crystal_recall` | Recall crystallized (graduated, long-term) patterns relevant to a query — the on-demand semantic tier. Keyword scoring is corpus-aware (rare, distinctive terms outweigh common process-words, so recall stays precise on a large store), and recall also follows evidence edges — surfacing a pattern grounded in an episode your query matched even with zero keyword overlap. See [The Memory Architecture](#the-memory-architecture-complementary-learning-systems) |
| `crystal_index` | The always-on name + one-clause menu of the crystallized store — what graduated wisdom exists, so the agent isn't blind to its own corpus (the bodies fill on cue via `crystal_recall`) |
| `spore_*` | The prospective (spore) layer — open loops that must resolve, distinct from retrospective memory: `spore_add` / `spore_list` / `spore_surface` / `spore_get` / `spore_update` / `spore_touch` / `spore_descend` / `spore_ascend`. See [Prospective memory — spores](#prospective-memory--spores) |

**16 tools total** (6 memory + 2 crystal + 8 spore). The crystallized tier exposes only its **read** surface over MCP (`crystal_recall` / `crystal_index`); crystallizing *out* stays a wrap-time / CLI / library action (it carries the opt-in + decision-channel governance).

**Resources:** `anneal://continuity` — the current continuity file, auto-loaded at session start. `anneal://integrity/manifest` — SHA-256 hashes for host-side tool description verification.

## Compliance and Audit

The episodic store is a natural audit trail. Every decision, tension, and outcome is timestamped, typed, and append-only — exactly what regulators want to see when they ask "why did the AI do that?"

**Hash-chained JSONL audit trail** (shipped, on by default):

Every memory operation — episode recorded, episode deleted, wrap started, wrap completed, associations updated — gets logged to an append-only JSONL file where each entry's SHA-256 hash includes the previous entry's hash. Modify or delete an entry and the chain breaks. Verify integrity programmatically or with `jq`.

- **Actor identity** on every entry (who did this — agent, system, admin)
- **Content-hash-only mode** by default — the audit trail proves *what happened* without storing the content itself (GDPR-compatible: delete the episode, the audit chain still verifies)
- **Weekly rotation with gzip** — old audit files compress automatically, manifest index enables cross-file chain verification
- **`on_event` callback** — pipe audit events to your own systems (cloud logging, SIEM, observability)
- **Crash recovery** — incomplete entries detected and handled on restart

```python
from anneal_memory import Store, AuditTrail

# Audit trail is on by default
store = Store("./memory.db", project_name="MyAgent")

# Verify chain integrity
result = AuditTrail.verify("./memory.db")
print(f"Valid: {result.valid}, Entries: {result.total_entries}")

# Stream events to external system
store = Store("./memory.db", on_audit_event=lambda entry: send_to_siem(entry))
```

**EU AI Act relevance:** The Act's Article 12 requires "automatic recording of events" for high-risk AI systems, with provisions for traceability, actor identification, and tamper evidence. This is audit *infrastructure*, not a compliance certification. The hash-chained trail covers the traceability requirements in Articles 12(2)(b,c) — actor identity, tamper evidence, and automatic event recording — without certifying full Act compliance.

**Provenance, not just timestamps.** anneal-memory ships at the provenance-chain level: every audit entry's hash is linked to its predecessor (modify or remove an entry, the chain breaks), and graduated patterns cite the episode IDs that earned them, so any pattern traces back to the observations behind it — not just *when* it was written. Why that distinction may become a differential compliance gate as the Act's enforcement develops, and why a memory whose quality runs through an LLM can't offer the same chain → [docs/architecture.md](docs/architecture.md).

**What's next:**
- **Compliance proxy** (Layer 2) — an optional transport-layer capture that extends the same hash-chained, source-tagged audit beyond memory operations, for teams that need a fuller traceability record. Off by default; the memory audit stands on its own.
- **Multi-agent shared memory** — shared episodic pool with per-agent continuity and per-agent association topology. Full cross-agent audit trail.

## Continuity Markers

The continuity file uses a simplified marker set for density:

```
? question needing resolution
thought: insight worth preserving
✓ completed item

A -> B                          causation
A ><[axis] B                    tension on an axis

[decided(rationale, on)]        committed decision
[blocked(reason, since)]        external dependency

| 1x (2026-04-01)              first observation
| 2x (2026-04-01) [evidence: abc123 "explanation"]   validated pattern

[contradicts: name]             this pattern conflicts with the named Proven
[no-contradicts]                contradiction scan run, nothing conflicts
[provenance: id1, id2]          founding episode ids for a mature top-tier
                                pattern with no fresh evidence this wrap
```

The last three are parsed by the immune system, not just prose.
`[contradicts:]` / `[no-contradicts]` record whether the contradiction
scan `prepare_wrap` renders into the compression package was actually
run — a new Proven landing with no stance is written to the audit log.
`[provenance:]` names the episodes that founded a mature 3x pattern
whose evidence has gone low-variance; it suppresses the "graduate OUT to
partnership.md or retire" warning for a line that is genuinely grounded
but quiet. It is not immortality — a provenance pattern that goes cold
still ages out through the ordinary warmth gate. Prefer fresh evidence
whenever it exists; provenance is only for when it does not.

## Security

### Memory poisoning resistance

Recent research demonstrates [environment-injected memory poisoning attacks against web agents](https://arxiv.org/abs/2604.02623) — adversarial content embedded in the environment (web pages, tool outputs) that the agent ingests as observations and that subsequently influences behavior across sessions. The published attack ("Poison Once, Exploit Forever," Zou et al., April 2026) demonstrated up to 32.5% attack success rate on GPT-4-mini against ChatGPT Atlas, Perplexity Comet, and OpenClaw — and notably did not require write access to the agent's memory store. Environmental contamination plus the agent's own consumption of the contaminated content was sufficient.

anneal-memory's citation-validated graduation does not eliminate this attack class — environment-injected episodes are recorded honestly because the agent did encounter them — but it raises the bar substantially:

1. **Single-shot poisoning stalls at 1x.** A poisoned episode enters the store as a raw observation. To influence the continuity file (the always-loaded compressed memory), it must graduate via independent citations from subsequent episodes. A single contaminated trajectory cannot bootstrap its own promotion.
2. **Explanation-grounding check rejects ungrounded citations.** The graduation pipeline runs `check_explanation_overlap(explanation, episode_content)` on every citation: at least two meaningful words from the citation's quoted explanation must actually appear in the cited episode's content. Citations with vague or fabricated explanations — including poisoned trajectories where an attacker controls the graduation claim text but cannot rewrite the episode body it points at — fail this check and don't accrue evidence weight. This is anti-fraud, not anti-repetition: it forces graduating claims to be textually grounded in what the episodes actually said.
3. **SHA-256 audit trail provides forensic surface.** While the chain doesn't prevent ingestion of contaminated environmental content, it preserves a tamper-evident record of what the agent encountered and when — supporting post-incident analysis when poisoning is detected downstream.

**This is structural inference, not empirical defense.** anneal-memory has not been tested against eTAMP directly. The mechanisms above derive from architectural properties — citation-validated graduation, lexical-overlap explanation-grounding, SHA-256 audit chain — not from a published evaluation. A targeted eTAMP variant could partially or fully bypass these defenses in ways the architecture-level argument doesn't anticipate. Sustained adversarial campaigns with diverse contaminated trajectories can still graduate, and the *Honest scope* section above documents specific gap classes (lexical-overlap exploits, rotated-pair gaming, slow-drift accumulation, contradiction-with-existing-Proven, pattern omission) confirmed reachable under adversarial-agent conditions. Architectures with no graduation gate inherit the full attack surface; anneal-memory inherits a structurally narrower one, pending direct empirical evaluation and the next set of defenses listed in *Honest scope*.

### Tool description integrity

Tool description integrity verification detects [description poisoning](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/2402) — where manipulated tool descriptions alter LLM behavior without changing tool functionality.

**Two-layer verification:**

1. **Build-time manifest** (`tool-integrity.json`) — SHA-256 hashes of all tool descriptions, shipped with the package and verified at server startup. Detects post-install modification.
2. **Host-verifiable resource** (`anneal://integrity/manifest`) — the same hashes exposed as an MCP resource, so editors and hosts can compare tool definitions received via `tools/list` against the server's intended definitions. Detects transport-layer description mutation between server and client — the class of attack where descriptions are modified in transit or by middleware without the server's knowledge.

```bash
anneal-memory --generate-integrity  # Regenerate after description changes
anneal-memory --skip-integrity      # Bypass for development
```

## Lineage

anneal-memory's architecture grew from [FlowScript](https://github.com/phillipclapham/flowscript) — a typed reasoning notation that explored compression-as-cognition, temporal graduation, and citation-validated patterns. The core insights proved more powerful than the syntax; anneal-memory delivers them as a zero-dependency memory system where agents use natural language instead of learning notation. The FlowScript notation remains in active daily use for reasoning compression, and a 9-marker subset powers the continuity compression prompts.

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
