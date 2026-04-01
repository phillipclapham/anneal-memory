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

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
