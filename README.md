<!-- mcp-name: io.github.phillipclapham/anneal-memory -->

<p align="center">
  <img src="logo-400.png" alt="anneal-memory logo" width="200">
</p>

# anneal-memory

**Living memory for AI agents. Episodes compress into identity.**

Memory without grounding is amplification infrastructure.

Persistent user memory profiles [increase agent sycophancy 10–45% across models](https://arxiv.org/abs/2509.12517) (Gemini 2.5 Pro at 45%, others lower). Production deployments [accumulate 97.8% junk entries](https://github.com/mem0ai/mem0/issues/4573) within weeks. Clinical research documents memory [scaffolding delusions across sessions](https://doi.org/10.1016/S2215-0366(25)00396-7). The problem isn't memory — it's memory without an immune system.

anneal-memory is that immune system. Patterns earn permanence through cited evidence, false knowledge gets caught and demoted, stale information fades, and associations form through consolidation. Your agent's memory develops over time, not just accumulates.

Four cognitive layers: episodic store, compressed continuity, Hebbian associations, and affective state tracking. Zero dependencies (Python stdlib only). Works with any agent framework.

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

Inspect, debug, and manage agent memory from the command line. 21 subcommands, all with `--json` output. Agents with shell access (Claude Code, Aider, etc.) can use the CLI directly for the full memory workflow.

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

See [`examples/CLAUDE.md.cli.example`](examples/CLAUDE.md.cli.example) for the agent workflow snippet.

### MCP Server

For MCP-enabled editors (Claude Code, Cursor, Windsurf, etc.). Zero-config — add to your MCP settings and go.

```json
{
  "mcpServers": {
    "anneal-memory": {
      "command": "uvx",
      "args": ["anneal-memory", "--project-name", "MyProject"]
    }
  }
}
```

Add the [orchestration snippet](examples/CLAUDE.md.example) to your project's `CLAUDE.md` — it teaches the agent *when* and *how* to use the memory tools. Without this snippet, the tools are available but the agent won't know the cognitive workflow.

> **Alternative:** `pip install anneal-memory` if you prefer a pinned install, then use `"command": "anneal-memory"` directly.

### All Three Paths, Same Cognitive Loop

CLI and MCP are thin transport adapters over the same library — not separate implementations. Every access pattern calls the same `prepare_wrap(store)` and `validated_save_continuity(store, text)` pipeline under the hood, preserving the same workflow: record episodes during work → compress at session boundaries → load continuity at session start. The agent that records is the agent that compresses. Compression cannot be delegated — it IS the cognition.

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
| **Anthropic Agents SDK** | CLAUDE.md snippet + `Stop` hook | [docs/integrations/anthropic-agents.md](docs/integrations/anthropic-agents.md) |
| **Google ADK** | Callbacks + custom `MemoryService` | [docs/integrations/google-adk.md](docs/integrations/google-adk.md) |
| **Pydantic AI** | `AbstractCapability` with `Hooks` | [docs/integrations/pydantic-ai.md](docs/integrations/pydantic-ai.md) |
| **smolagents** | `step_callbacks` dict | [docs/integrations/smolagents.md](docs/integrations/smolagents.md) |
| **LlamaIndex** | Instrumentation `BaseEventHandler` | [docs/integrations/llamaindex.md](docs/integrations/llamaindex.md) |
| **Haystack** | Custom `Tracer` | [docs/integrations/haystack.md](docs/integrations/haystack.md) |
| **CAMEL-AI** | `WorkforceCallback` | [docs/integrations/camel-ai.md](docs/integrations/camel-ai.md) |
| **AutoGen / AG2** | `register_hook()` | [docs/integrations/autogen.md](docs/integrations/autogen.md) |
| **DSPy** | `BaseCallback` | [docs/integrations/dspy.md](docs/integrations/dspy.md) |

These guides show integration patterns based on each framework's current API. The library works with any Python framework — the pattern is always the same: initialize a `Store`, call `record()` at meaningful moments, `recall()` before decisions, and run the wrap sequence at session end. **Don't see your framework?** The [library quickstart](docs/library-quickstart.md) shows the 4-function pattern that works everywhere.

## Why This Exists

Three independent production failures share one root cause: no quality mechanism between memory write and memory read.

**Sycophancy amplification.** Agents with persistent user memory profiles become 10–45% more sycophantic than memoryless baselines, depending on model (Gemini 2.5 Pro at 45%, others lower). Memory recalls what the user liked hearing, the agent learns to repeat it, and stored approval patterns compound across sessions ([Jain et al., CHI 2026](https://arxiv.org/abs/2509.12517); measured with user memory profiles across Gemini and Llama variants).

**Junk accumulation.** A [detailed production audit](https://github.com/mem0ai/mem0/issues/4573) on Mem0's tracker documents a deployment that generated 10,134 memory entries over 32 days — 224 were usable. The rest were duplicates, self-referential loops, and hallucinated entries: recalled memories re-extracted as new memories in a feedback loop that no one designed but nothing prevented.

**Harmful reinforcement.** Clinical research documents AI systems with persistent memory scaffolding delusional content across sessions — stored context creates feedback loops between recalled memories and generated responses, with cases of documented real-world harm ([Morrin et al., Lancet Psychiatry 2026](https://doi.org/10.1016/S2215-0366(25)00396-7)).

Every existing MCP memory server stores memories and retrieves them. None of them ask: *is this memory still true? Was it ever true? Is it making the agent worse?*

anneal-memory asks all three:

- **Is it true?** Patterns must cite specific episode IDs as evidence to graduate. The server verifies the episodes exist and the explanation connects to the cited content.
- **Is it still true?** Graduated knowledge that stops being reinforced by new episodes gets flagged as stale and demoted. Memory actively forgets what's no longer relevant.
- **Is it self-confirming?** Anti-inbreeding detection catches the agent citing its own output as evidence. The cited episode must contain meaningfully different content from the graduation claim.

The result: **memory as a living system, not a filing cabinet.** Episodes accumulate fast, get compressed at session boundaries — and the compression IS the cognition, where patterns emerge and get validated. Co-cited episodes form lateral Hebbian associations, building a cognitive network through use. The continuity file stays bounded and always-loaded, getting denser rather than longer.

## What Makes It Different

The agent memory ecosystem is converging on consolidation as the right approach — even Anthropic's Claude Code now runs a periodic consolidation pass over accumulated session data. This validates the direction: raw accumulation doesn't scale, and compression at session boundaries is where intelligence emerges.

But consolidation alone doesn't solve the problem. A system that consolidates faithfully and a system that consolidates sycophantically produce the same *kind* of output — compressed, structured, always-loaded. The difference is whether anything checks the quality of what got consolidated. That's the immune system.

### The immune system (nobody else has this)

**Citation-validated graduation.** Patterns start at 1x. To graduate to 2x or 3x, they must cite specific episode IDs as evidence. The server verifies those IDs exist and the explanation connects to the cited episode. No evidence, no promotion.

**Anti-inbreeding defense.** Explanation overlap checking prevents the agent from confirming its own hallucinated patterns — the cited episode must contain meaningfully different content from the graduation claim itself.

**Principle demotion.** Graduated knowledge that stops being reinforced by new episodes gets flagged as stale and can be demoted. Memory actively forgets what's no longer relevant.

### Associations through consolidation (not retrieval)

During compression, when an agent cites multiple episodes to support a pattern, those episodes form lateral associations — Hebbian-style links that strengthen through repeated co-citation across wraps.

This is fundamentally different from how other systems form associations:

| Approach | When links form | Signal quality |
|----------|----------------|----------------|
| **Co-access** (BrainBox) | Episodes retrieved in the same query | Shallow — reflects search patterns, not understanding |
| **Co-retrieval** (Ori-Mnemos) | Episodes returned together at runtime | Better — but still driven by the retrieval system, not the agent |
| **Co-citation during consolidation** (anneal-memory) | Agent explicitly connects episodes while compressing | Deepest — links form from semantic judgment during a cognitive act |

The association network inherits the immune system's integrity: only validated citations form links. Demoted citations don't. The entire cognitive topology is built on evidence, not frequency.

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
2. **Continuity file** (Markdown) — compressed session memory. 4 sections. Always loaded at session start. Rewritten (not appended) at each session boundary. The neocortex.
3. **Hebbian associations** (SQLite) — lateral links between episodes, formed through co-citation during compression. Strengthen with reuse, decay without it. The association cortex.
4. **Affective layer** (on associations) — functional state tags recorded during compression. Intensity modulates association strength. Persistent state infrastructure.

**Six episode types** give the immune system richer signal:

| Type | Purpose | Example |
|------|---------|---------|
| `observation` | Pattern or insight | "Connection pool is the real bottleneck" |
| `decision` | Committed choice | "Chose Postgres because ACID > raw speed" |
| `tension` | Tradeoff identified | "Latency vs consistency — can't optimize both" |
| `question` | Needs resolution | "Should we shard or add read replicas?" |
| `outcome` | Result of action | "Migration done, 3x improvement on hot path" |
| `context` | Environmental state | "Production DB at 80% capacity, growing 5%/week" |

### Comparison

| | anneal-memory | Anthropic Official | Mem0 | Ori-Mnemos | BrainBox |
|---|---|---|---|---|---|
| **Architecture** | Episodic + continuity + associations | JSONL flat file (graph-shaped) | Vector + graph | Retrieval + Hebbian | Memory + Hebbian |
| **Compression** | Session-boundary rewrite | None | One-pass extraction | None | None |
| **Quality mechanism** | Immune system (citations + anti-inbreeding + demotion) | None | None | NPMI normalization | None |
| **Association formation** | Co-citation during consolidation | None | None | Co-retrieval at runtime | Co-access at runtime |
| **Affective tracking** | Agent self-report during compression | None | None | None | None |
| **Audit trail** | Hash-chained JSONL | None | None | None | None |
| **Access patterns** | Library + CLI + MCP | MCP only | REST API | Python only | MCP only |
| **Dependencies** | Zero (Python stdlib) | Node.js | Docker + cloud | Embeddings model | Not specified |

## The Consolidation Landscape (2026)

Multiple independent groups shipped consolidation-based agent memory architectures in early 2026: anneal-memory (March, citation-graduation multi-tier), [OpenClaw Dreaming](https://docs.openclaw.ai/concepts/dreaming) (April 9, three-phase Light/REM/Deep Sleep), and Anthropic's [KAIROS / autoDream](https://www.deeplearning.ai/the-batch/claude-codes-source-code-leaked-exposing-potential-future-features-kairos-and-autodream/) (leaked March 30 via Claude Code source map, four-phase merge / remove-contradictions / promote-provisional-to-absolute / MEMORY.md index). Convergence on consolidation validates the direction — raw accumulation doesn't scale, and compression at session boundaries is where intelligence emerges.

The groups diverge on one load-bearing question: **what gates quality?**

| System | Quality gate | Sycophancy-vulnerable? |
|---|---|---|
| **anneal-memory** | Structural citation evidence (agent cites episode IDs; server verifies) | No — gate is not LLM-scored |
| **OpenClaw Dreaming** | LLM reflection + six weighted signals: Relevance 0.30, Frequency 0.24, Query diversity 0.15, Recency 0.15, Consolidation 0.10, Conceptual richness 0.06 | Yes — Relevance and Conceptual richness are LLM-judged |
| **KAIROS / autoDream** | LLM consolidation (merge, remove contradictions, promote tentative observations to absolute facts) | Yes — promotion gate is model-reliant |

Structural gates ask "did subsequent episodes cite this?" Model-reliant gates ask "does the LLM consider this good?" The difference matters: persistent user memory profiles have been shown to amplify sycophancy 10–45% across models ([Jain et al., CHI 2026](https://arxiv.org/abs/2509.12517); Gemini 2.5 Pro at 45%, others lower). The same RLHF-inherited bias surfaces wherever an LLM evaluates output for the user — including memory-quality scoring. A memory architecture whose quality mechanism runs through an LLM inherits that bias. anneal-memory's citation-evidence gates bypass it by construction.

The same architectural choice is going mainstream at the adjacent evaluation layer: [AWS Bedrock AgentCore Evaluations](https://aws.amazon.com/about-aws/whats-new/2026/03/agentcore-evaluations-generally-available/) (GA March 31, 2026) ships 13 built-in LLM-based evaluators for agent response quality, safety, task completion, and tool usage. Different layer (agent output vs. memory graduation), same failure class (LLM-as-judge inherits judge bias). The industry shift toward model-reliant quality infrastructure is real — which is precisely why structural alternatives at the memory layer matter.

A separate, orthogonal axis is **representation-layer quality filtering**: [Memori](https://arxiv.org/abs/2603.19935) (arXiv 2603.19935, March 2026) uses semantic triple extraction and dynamic linking to improve memory signal at the representation layer, reporting 81.95% on LOCOMO as the leading retrieval-based system (ahead of Zep 79.09%, LangMem 78.05%, Mem0 62.47% on its older pipeline). Different theory of quality — where a memory "lives" structurally and whether its graduation is citation-gated are independent choices. Both can be correct at their own axis.

**April 2026 — adjacent architectures published.** Two papers in the same month explored multi-layer memory architectures with associative linking from different angles. [HeLa-Mem](https://arxiv.org/abs/2604.16839) (Zhu et al., ACL 2026 accepted) implements explicit Hebbian learning dynamics with episodic and semantic memory layers, where a "Reflective Agent" identifies dense memory hubs for consolidation. [GAM](https://arxiv.org/abs/2604.12285) (Wu et al., April 2026 preprint) takes a different architectural path: hierarchical graph-based memory where event progression graphs integrate into a topic network at semantic shifts — graph-based associative linking rather than classical Hebbian dynamics. Different starting points, similar destination at the multi-layer level. Neither has citation-validated quality gates. Multi-layer memory with associative linking is a now-active architectural space; the differentiator is increasingly *what gates the consolidation*, not whether consolidation happens. anneal-memory diverges on the quality gate — HeLa-Mem's Reflective Agent and GAM's integration logic are LLM-mediated, inheriting the same LLM-as-judge bias surface as the OpenClaw Dreaming and KAIROS rows above. Citation-evidence gates run independent of LLM judgment by construction.

## On LOCOMO

[LOCOMO](https://snap-research.github.io/locomo/) is the current de-facto benchmark for agent memory. [Mem0](https://mem0.ai/research) reports 91.6, [MemMachine](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/) reports 91.69, [Memori](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) reports 81.95 among retrieval-based systems, and [Backboard](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) ships a dedicated LOCOMO evaluation framework. anneal-memory has no LOCOMO score as of April 22, 2026. This is deliberate.

LOCOMO measures conversational recall — can the agent remember facts, hold state across turns, maintain coherence across long dialogues? These are real evaluations of a real capability, and they aren't the capability anneal-memory is architected around. anneal-memory's target is citation-validated pattern accumulation that persists across sessions, agents, and contexts for accountability-bearing agent work: patterns must be defensibly surfaced, wrong patterns must demote, cross-agent contamination must be resisted, and sycophancy amplification from persistent-memory RLHF loops must be structurally bounded. A high LOCOMO score tells you the agent remembered the conversation; it doesn't tell you the memory is structurally sound at the axis that matters when the memory is informing downstream decisions.

**Scope-out is sequence, not refusal.** anneal-memory will run LOCOMO as secondary validation of a different-question architecture when (a) a competitor publishes numbers suggesting avoidance, or (b) academic publication requires it. The LOCOMO score will be reported alongside the axis anneal-memory actually optimizes for — not as a concession that LOCOMO was the right frame.

## Session Hygiene

Session wraps are the most important thing your agent does with this system. Think of them like sleep.

Neuroscience calls it memory consolidation: during slow-wave sleep, the hippocampus replays the day's experiences while the neocortex integrates them into long-term knowledge. Skip sleep and memories degrade — experiences accumulate without being processed, patterns go unrecognized, and older knowledge doesn't get reinforced or pruned.

anneal-memory works the same way. During a session, episodes accumulate in the episodic store. At session end, the wrap compresses those episodes into the continuity file. This is where the real thinking happens — the agent recognizes patterns, promotes validated knowledge, lets stale information fade, and forms associations between related episodes. Without wraps, you just have a growing pile of raw episodes and no intelligence.

**The wrap sequence:**

1. **`prepare_wrap`** — gathers recent episodes, current continuity, stale pattern warnings, association context, and compression instructions
2. **Agent compresses** — this IS the cognition. Patterns emerge during compression that weren't visible in the raw episodes
3. **`save_continuity`** — server validates structure, checks citation evidence, records associations between co-cited episodes, applies decay to unused associations, and saves the result

**Rules of thumb:**
- Always wrap before ending a session. An unwrapped session is like an all-nighter — the experiences happened but they weren't consolidated
- The orchestration snippets ([MCP](examples/CLAUDE.md.example), [CLI](examples/CLAUDE.md.cli.example)) handle this automatically — they teach the agent to detect session-end signals and run the full sequence
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

**EU AI Act relevance:** The Act's Article 12 requires "automatic recording of events" for high-risk AI systems, with provisions for traceability, actor identification, and tamper evidence. anneal-memory's audit infrastructure covers Articles 12(2)(b,c) out of the box. This is audit *infrastructure* that helps systems comply — not a compliance certification.

**Provenance vs timestamps.** Article 12's traceability obligation can be satisfied at two architectural levels: timestamp-only logs (which entry was written when) or **provenance chains** (where this state originated and through what intermediate steps). anneal-memory ships at the provenance-chain level — every audit entry's hash is cryptographically linked to its predecessor (modify or remove an entry, the chain breaks), and graduated patterns carry explicit episode-ID citations as evidence (any pattern is traceable back to the specific observations that earned its promotion). As regulatory guidance and case law develop through August 2026 enforcement and the period after, this distinction may become a differential compliance gate. Memory architectures whose quality mechanism runs through an LLM — the AWS AgentCore evaluation pattern (April 2026 architecture) referenced earlier in *The Consolidation Landscape*, plus any system whose answer to "why did this memory survive?" is "the consolidation model thought so" — cannot satisfy a provenance-chain interpretation as currently architected. "The LLM thought this was good" is not a chain of evidence. Architecture matters at the compliance layer, not just the inference layer.

**What's next:**
- **Compliance proxy** (Layer 2) — MCP transport-layer interception that captures *all* agent actions (every tool call, every response), not just memory operations. Same store, `source` field distinguishes intentional recording from automatic capture. Memory audit = "here's what the agent learned." Compliance proxy = "here's everything that happened."
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
```

## Security

### Memory poisoning resistance

Recent research demonstrates [environment-injected memory poisoning attacks against web agents](https://arxiv.org/abs/2604.02623) — adversarial content embedded in the environment (web pages, tool outputs) that the agent ingests as observations and that subsequently influences behavior across sessions. The published attack ("Poison Once, Exploit Forever," Zou et al., April 2026) demonstrated up to 32.5% attack success rate on GPT-4-mini against ChatGPT Atlas, Perplexity Comet, and OpenClaw — and notably did not require write access to the agent's memory store. Environmental contamination plus the agent's own consumption of the contaminated content was sufficient.

anneal-memory's citation-validated graduation does not eliminate this attack class — environment-injected episodes are recorded honestly because the agent did encounter them — but it raises the bar substantially:

1. **Single-shot poisoning stalls at 1x.** A poisoned episode enters the store as a raw observation. To influence the continuity file (the always-loaded compressed memory), it must graduate via independent citations from subsequent episodes. A single contaminated trajectory cannot bootstrap its own promotion.
2. **Explanation-grounding check rejects ungrounded citations.** The graduation pipeline runs `check_explanation_overlap(explanation, episode_content)` on every citation: at least two meaningful words from the citation's quoted explanation must actually appear in the cited episode's content. Citations with vague or fabricated explanations — including poisoned trajectories where an attacker controls the graduation claim text but cannot rewrite the episode body it points at — fail this check and don't accrue evidence weight. This is anti-fraud, not anti-repetition: it forces graduating claims to be textually grounded in what the episodes actually said.
3. **SHA-256 audit trail provides forensic surface.** While the chain doesn't prevent ingestion of contaminated environmental content, it preserves a tamper-evident record of what the agent encountered and when — supporting post-incident analysis when poisoning is detected downstream.

**This is structural inference, not empirical defense.** anneal-memory has not been tested against eTAMP directly. The mechanisms above derive from architectural properties — citation-validated graduation, anti-inbreeding overlap checking, SHA-256 audit chain — not from a published evaluation. A targeted eTAMP variant could partially or fully bypass these defenses in ways the architecture-level argument doesn't anticipate. Sustained adversarial campaigns with diverse contaminated trajectories can still graduate; the immune system bounds the *cost* of poisoning attacks, not the possibility. Architectures with no graduation gate inherit the full attack surface; anneal-memory inherits a structurally narrower one, pending direct empirical evaluation.

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
