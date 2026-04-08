<!-- mcp-name: io.github.phillipclapham/anneal-memory -->

<p align="center">
  <img src="logo-400.png" alt="anneal-memory logo" width="200">
</p>

# anneal-memory

**Living memory for AI agents. Episodes compress into identity.**

The only MCP memory server with an immune system. Patterns earn permanence through evidence, false knowledge gets caught and demoted, and stale information fades — so your agent's memory gets smarter over time, not just bigger.

Four cognitive layers: episodic store, compressed continuity, Hebbian associations, and affective state tracking. Zero dependencies. 5 tools. Works with any MCP client.

## Quick Start

**1. Run the server** (no install required):

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

Add this to `.mcp.json` in your project root (Claude Code) or your editor's MCP config.

The database defaults to `~/.anneal-memory/memory.db` (auto-created). Override with `--db /path/to/memory.db` for per-project storage.

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
- Episodes that get cited together form lateral associations — a cognitive network that develops through use
- The continuity file stays bounded and always-loaded, not growing linearly

## What Makes It Different

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
# Via MCP tool
save_continuity(text="...", affective_state={"tag": "curious", "intensity": 0.8})

# Via Engine (automated characterization)
engine = Engine(store, api_key="...", characterize_affect=True)
result = engine.wrap()  # LLM self-reports affect post-compression
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
| **Architecture** | Episodic + continuity + associations | Knowledge graph | Vector + graph | Retrieval + Hebbian | Memory + Hebbian |
| **Compression** | Session-boundary rewrite | None | One-pass extraction | None | None |
| **Quality mechanism** | Immune system (citations + anti-inbreeding + demotion) | None | None | NPMI normalization | None |
| **Association formation** | Co-citation during consolidation | None | None | Co-retrieval at runtime | Co-access at runtime |
| **Affective tracking** | Agent self-report during compression | None | None | None | None |
| **Audit trail** | Hash-chained JSONL | None | None | None | None |
| **Dependencies** | Zero (Python stdlib) | Node.js | Docker + cloud | Embeddings model | Not specified |
| **Tools** | 5 + 1 resource | 6 | Varies | Not MCP | MCP (mcpmarket) |

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
- The [CLAUDE.md snippet](examples/CLAUDE.md.example) handles this automatically — it teaches the agent to detect session-end signals ("let's wrap up," "we're done") and run the full sequence
- Short sessions (3-5 episodes) still benefit from wraps. Even a small amount of compression builds the continuity file
- If `prepare_wrap` says "no episodes" — nothing to compress. That's fine, skip it

The graduation system and association network both depend on wraps to function. Patterns can only be promoted (1x -> 2x -> 3x) during compression, citations can only be validated during wraps, associations only form through co-citation during wraps, and stale patterns can only be detected when the agent reviews what it knows against what it recently experienced. No wraps = no immune system, no associations, no cognitive development.

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
from anneal_memory import Store

# Audit trail is on by default
store = Store("./memory.db", project_name="MyAgent")

# Verify chain integrity
from anneal_memory import AuditTrail
result = AuditTrail.verify("./memory.db")
print(f"Valid: {result.valid}, Entries: {result.total_entries}")

# Stream events to external system
store = Store("./memory.db", on_audit_event=lambda entry: send_to_siem(entry))
```

**EU AI Act relevance:** The Act's Article 12 requires "automatic recording of events" for high-risk AI systems, with provisions for traceability, actor identification, and tamper evidence. anneal-memory's audit infrastructure covers Articles 12(2)(b,c) out of the box. This is audit *infrastructure* that helps systems comply — not a compliance certification.

**What's next:**
- **Compliance proxy** (Layer 2) — MCP transport-layer interception that captures *all* agent actions (every tool call, every response), not just memory operations. Same store, `source` field distinguishes intentional recording from automatic capture. Memory audit = "here's what the agent learned." Compliance proxy = "here's everything that happened."
- **Multi-agent shared memory** — shared episodic pool with per-agent continuity and per-agent association topology. Full cross-agent audit trail.

## MCP Tools

| Tool | When to call |
|------|-------------|
| `record` | When something important happens — a decision, observation, tension, question, outcome, or context change |
| `recall` | Before making decisions that might have prior context. Query by time, type, keyword, or ID |
| `prepare_wrap` | At session end — returns episodes + current continuity + association context + compression instructions |
| `save_continuity` | After compressing — server validates structure, citations, records associations, applies decay, and saves |
| `status` | Check memory health: episode counts, wrap history, continuity size, association network metrics |

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
    # -> episodes, continuity, stale_patterns, instructions, today

    # Associations
    stats = store.association_stats()
    print(f"Links: {stats.total_links}, Avg strength: {stats.avg_strength}")

    assocs = store.get_associations(episode_ids=["abc123"], min_strength=0.5)
    for a in assocs:
        print(f"{a.id_a} <-> {a.id_b} strength={a.strength}")
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
    print(f"Associations: {result.associations_formed} formed, {result.associations_strengthened} strengthened")
```

The Engine gathers episodes, builds the compression prompt, calls the LLM, validates structure and citations, records associations between co-cited episodes, applies decay, truncates if over budget, and saves — all in one call. Falls back to existing continuity on invalid LLM output (or rejects on first session — episodes stay for retry).

**Affective characterization:** `Engine(store, api_key="...", characterize_affect=True)` — after compression, the LLM self-reports its functional state during the wrap. This gets recorded on associations formed during that wrap. Experimental.

**Custom LLM** (zero additional dependencies): `Engine(store, llm=lambda prompt: my_llm(prompt))`

## Security

Tool description integrity verification is included. `tool-integrity.json` ships with the package containing SHA256 hashes of all tool descriptions, verified at server startup. This detects post-install modification of tool descriptions — a vector where manipulated descriptions alter LLM behavior without changing tool functionality.

```bash
anneal-memory --generate-integrity  # Regenerate after description changes
anneal-memory --skip-integrity      # Bypass for development
```

## Lineage

anneal-memory's architecture grew from [FlowScript](https://github.com/phillipclapham/flowscript) — a typed reasoning notation that explored compression-as-cognition, temporal graduation, and citation-validated patterns. The core insights proved more powerful than the syntax; anneal-memory delivers them as a zero-dependency MCP server where agents use natural language instead of learning notation. The FlowScript notation remains in active daily use for reasoning compression, and a 9-marker subset powers the continuity compression prompts.

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
