<!-- mcp-name: io.github.phillipclapham/anneal-memory -->

<p align="center">
  <img src="logo-400.png" alt="anneal-memory logo" width="200">
</p>

# anneal-memory

**Two-layer memory for AI agents. Episodes compress into identity.**

The only MCP memory server with an immune system. Patterns earn permanence through evidence, false knowledge gets caught and demoted, and stale information fades — so your agent's memory gets smarter over time, not just bigger.

Zero dependencies. 5 tools. Works with any MCP client.

## Quick Start

**1. Run the server** (no install required):

```json
{
  "mcpServers": {
    "anneal-memory": {
      "command": "uvx",
      "args": ["anneal-memory", "--db", "./memory.db", "--project-name", "MyProject"]
    }
  }
}
```

Add this to `.mcp.json` in your project root (Claude Code) or your editor's MCP config.

> **Alternative:** `pip install anneal-memory` if you prefer a pinned install, then use `"command": "anneal-memory"` directly.

**2. Add the orchestration snippet** to your project's `CLAUDE.md`:

Copy the contents of [`examples/CLAUDE.md.example`](examples/CLAUDE.md.example) into your project's `CLAUDE.md`. This teaches the agent *when* and *how* to use the memory tools throughout a session — recording episodes during work, checking prior context before decisions, and running the full compression sequence at session end.

Without this snippet, the tools are available but the agent won't know the workflow. This is the most important setup step.

**3. Restart your editor.** That's it. The agent now records, recalls, and compresses memory across sessions.

## Why This Exists

Every MCP memory server we tested has the same problem: memory grows forever and nothing validates what it knows.

The Anthropic official server stores everything in a growing JSONL file with no pruning. Mem0 requires Docker and cloud for its best features. Others expose 15-33 tools that eat your context window. And none of them can tell you whether what they "remember" is still true.

anneal-memory takes a different approach: **memory as a living system, not a filing cabinet.**

- Episodes accumulate fast (append-only SQLite, typed by kind)
- At session boundaries, the agent compresses episodes into a continuity file — and the compression step IS the thinking, where patterns emerge
- Validated patterns strengthen. Stale patterns fade. False patterns get caught.
- The continuity file stays bounded and always-loaded, not growing linearly

## What Makes It Different

### The immune system (nobody else has this)

**Citation-validated graduation.** Patterns start at 1x. To graduate to 2x or 3x, they must cite specific episode IDs as evidence. The server verifies those IDs exist and the explanation connects to the cited episode. No evidence, no promotion.

**Anti-inbreeding defense.** Explanation overlap checking prevents the agent from confirming its own hallucinated patterns — the cited episode must contain meaningfully different content from the graduation claim itself.

**Principle demotion.** Graduated knowledge that stops being reinforced by new episodes gets flagged as stale and can be demoted. Memory actively forgets what's no longer relevant.

### Architecture

```
  Episodes (fast)              Continuity (compressed)
  ┌─────────────┐             ┌──────────────────────┐
  │ observation  │             │ ## State             │
  │ decision     │─── wrap ──→│ ## Patterns (1x→3x)  │
  │ tension      │  compress  │ ## Decisions          │
  │ question     │             │ ## Context            │
  │ outcome      │             └──────────────────────┘
  │ context      │               always loaded, bounded
  └─────────────┘               human-readable markdown
   SQLite, indexed
```

**Two layers, like how memory actually works:**
- **Episodic store** (SQLite) — timestamped, typed episodes. Fast writes, indexed queries. Cheap to accumulate.
- **Continuity file** (Markdown) — compressed session memory. 4 sections. Always loaded at session start. Rewritten (not appended) at each session boundary.

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

| | anneal-memory | Anthropic Official | Mem0 | mcp-memory-service | Hmem |
|---|---|---|---|---|---|
| **Architecture** | Episodic + continuity | Knowledge graph | Vector + graph | KG + vectors | 5-level hierarchy |
| **Compression** | Session-boundary rewrite | None | One-pass extraction | Dream consolidation | Hierarchical summary |
| **Quality mechanism** | Immune system | None | None | Conflict detection | None |
| **Graduation** | 1x→2x→3x with citations | None | None | None | None |
| **Dependencies** | Zero (Python stdlib) | Node.js | Docker + cloud | Embeddings model | Zero (Go binary) |
| **Tools** | 5 + 1 resource | 6 | Varies | 15+ | ~6 |
| **Orchestration snippet** | Yes (first-class) | No | No | Wiki pattern | No |

## Session Hygiene

Session wraps are the most important thing your agent does with this system. Think of them like sleep.

Neuroscience calls it memory consolidation: during slow-wave sleep, the hippocampus replays the day's experiences while the neocortex integrates them into long-term knowledge. Skip sleep and memories degrade — experiences accumulate without being processed, patterns go unrecognized, and older knowledge doesn't get reinforced or pruned.

anneal-memory works the same way. During a session, episodes accumulate in the episodic store (the hippocampus). At session end, the wrap compresses those episodes into the continuity file (the neocortex). This is where the real thinking happens — the agent recognizes patterns, promotes validated knowledge, and lets stale information fade. Without wraps, you just have a growing pile of raw episodes and no intelligence.

**The wrap sequence:**

1. **`prepare_wrap`** — gathers recent episodes, current continuity, stale pattern warnings, and compression instructions
2. **Agent compresses** — this IS the cognition. Patterns emerge during compression that weren't visible in the raw episodes
3. **`save_continuity`** — server validates structure, checks citation evidence, saves the result

**Rules of thumb:**
- Always wrap before ending a session. An unwrapped session is like an all-nighter — the experiences happened but they weren't consolidated
- The [CLAUDE.md snippet](examples/CLAUDE.md.example) handles this automatically — it teaches the agent to detect session-end signals ("let's wrap up," "we're done") and run the full sequence
- Short sessions (3-5 episodes) still benefit from wraps. Even a small amount of compression builds the continuity file
- If `prepare_wrap` says "no episodes" — nothing to compress. That's fine, skip it

The graduation system depends on wraps to function. Patterns can only be promoted (1x → 2x → 3x) during compression, citations can only be validated against episode IDs during wraps, and stale patterns can only be detected when the agent reviews what it knows against what it recently experienced. No wraps = no immune system.

## MCP Tools

| Tool | When to call |
|------|-------------|
| `record` | When something important happens — a decision, observation, tension, question, outcome, or context change |
| `recall` | Before making decisions that might have prior context. Query by time, type, keyword, or ID |
| `prepare_wrap` | At session end — returns episodes + current continuity + compression instructions |
| `save_continuity` | After compressing — server validates structure, citations, and saves |
| `status` | Check memory health: episode counts, wrap history, continuity size |

**Resource:** `anneal://continuity` — the current continuity file, auto-loaded at session start.

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

## Python API

```python
from anneal_memory import Store, EpisodeType

with Store("./memory.db", project_name="MyAgent") as store:
    store.record("User prefers PostgreSQL for ACID", EpisodeType.OBSERVATION)
    store.record("Chose pooling over caching", EpisodeType.DECISION)

    result = store.recall(episode_type=EpisodeType.DECISION, keyword="pooling")

    episodes = store.episodes_since_wrap()
    from anneal_memory import prepare_wrap_package
    package = prepare_wrap_package(episodes, store.load_continuity(), "MyAgent")
    # → episodes, continuity, stale_patterns, instructions, today
```

## Engine (Automated Compression)

For pipelines, cron jobs, or programmatic use without an interactive agent:

```bash
pip install anneal-memory[engine]
```

```python
from anneal_memory import Engine, Store

with Store("./memory.db", project_name="MyAgent") as store:
    store.record("Observed pattern in data", "observation")
    store.record("Chose approach X over Y", "decision")

    engine = Engine(store, api_key="sk-ant-...")  # or llm=my_callable
    result = engine.wrap()

    print(f"Compressed {result.episodes_compressed} episodes")
    print(f"Continuity: {result.chars} chars, {result.patterns_extracted} patterns")
```

The Engine gathers episodes, builds the compression prompt, calls the LLM, validates structure and citations, truncates if over budget, and saves — all in one call. Falls back to existing continuity on invalid LLM output (or rejects on first session — episodes stay for retry).

**Custom LLM** (zero additional dependencies): `Engine(store, llm=lambda prompt: my_llm(prompt))`

## Security

Tool description integrity verification is included. `tool-integrity.json` ships with the package containing SHA256 hashes of all tool descriptions, verified at server startup. This detects post-install modification of tool descriptions — a vector where manipulated descriptions alter LLM behavior without changing tool functionality.

```bash
anneal-memory --generate-integrity  # Regenerate after description changes
anneal-memory --skip-integrity      # Bypass for development
```

## Compliance and Audit (Coming Soon)

The episodic store is a natural audit trail. Every decision, tension, and outcome is timestamped, typed, and append-only — exactly what regulators want to see when they ask "why did the AI do that?"

**What exists today:**
- Typed episodes capture decisions with rationale, tradeoffs, outcomes, and context
- Append-only episodic store with optional tombstones (no silent deletion)
- Human-readable continuity file shows the agent's current understanding
- Tool description integrity verification (tamper detection)

**What's coming:**
- **JSONL audit sidecar** — append-only, rotatable export alongside the SQLite store. Readable with `cat` and `jq`, no special tooling required
- **Hash-chained episodes** — each entry's hash includes the previous entry's hash, creating a cryptographically tamper-evident chain. Modify or delete an episode and the chain breaks
- **Multi-agent shared memory** — shared episodic pool with per-agent continuity. Full cross-agent audit trail showing which agent made which decision and why

The two-layer architecture means compliance and intelligence don't compete: the episodic layer preserves everything (audit), while the continuity layer compresses intelligently (memory). Different layers for different readers.

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
