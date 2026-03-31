# anneal-memory

Two-layer memory for AI agents. Episodes compress into identity.

## What This Is

A memory system built on how memory actually works: fast accumulation of experiences (episodic layer) compressed into durable knowledge (continuity layer) at session boundaries. The act of compression is itself cognition — it forces pattern recognition, abstraction, and judgment. The system develops over time, getting smarter rather than just bigger.

**Two layers:**
- **Episodic store** (SQLite) — timestamped, typed episodes. Fast writes, indexed queries. The raw material.
- **Continuity file** (Markdown) — compressed session memory. 4 sections: State, Patterns, Decisions, Context. Always loaded. Human-readable and human-editable.

**What makes it different:**
- **Temporal graduation** — patterns earn permanence through evidence (1x → 2x → 3x). Citations must reference real episodes.
- **Anti-inbreeding defense** — explanation overlap checking prevents the LLM from confirming its own hallucinated patterns.
- **Principle demotion** — graduated knowledge that stops being reinforced by new episodes fades over time. Living memory, not archive.
- **Intelligent forgetting** — stale patterns are detected and flagged. The system doesn't just grow; it develops.

## Install

```bash
pip install anneal-memory
```

Zero runtime dependencies (Python stdlib only). Requires Python 3.10+.

For automated LLM compression (programmatic use without an interactive agent):

```bash
pip install anneal-memory[engine]
```

## Quick Start

```python
from anneal_memory import Store, EpisodeType

# Create a store (SQLite database + markdown continuity sidecar)
with Store("./agent_memory.db", project_name="MyAgent") as store:

    # Record episodes during a session
    store.record("User prefers PostgreSQL for ACID compliance", EpisodeType.OBSERVATION)
    store.record("Chose connection pooling over query caching", EpisodeType.DECISION)
    store.record("Latency vs consistency tradeoff", EpisodeType.TENSION)

    # Query episodes
    result = store.recall(episode_type=EpisodeType.DECISION, keyword="pooling")

    # At session end: get the compression package
    from anneal_memory import prepare_wrap_package
    episodes = store.episodes_since_wrap()
    package = prepare_wrap_package(episodes, store.load_continuity(), "MyAgent")
    # package contains: episodes, continuity, stale_patterns, instructions, today

    # After compression (by the agent or an LLM engine):
    # store.save_continuity(compressed_text)
    # store.wrap_completed(episodes_compressed=3, continuity_chars=len(compressed_text))
```

## Episode Types

| Type | Purpose |
|------|---------|
| `observation` | Pattern noticed, insight, general learning |
| `decision` | Committed choice with rationale |
| `tension` | Conflict, tradeoff, opposing forces |
| `question` | Open question needing resolution |
| `outcome` | Result of an action (success or failure) |
| `context` | Environmental/state information |

## Architecture

```
Session N episodes  ──→  prepare_wrap  ──→  LLM compression  ──→  save_continuity
       │                      │                    │                      │
  SQLite store          episodes +            Compression           Validated +
  (fast, indexed)    existing continuity    as cognition           saved markdown
                     + instructions        (the agent thinks      (4 sections)
                                           by compressing)
```

The continuity file uses a simplified marker set for density:
- `? question` / `thought: insight` / `✓ completed`
- `A -> B` (causation) / `A ><[axis] B` (tension)
- `[decided(rationale, on)]` / `[blocked(reason, since)]`
- `| Nx (date) [evidence: id "explanation"]` (temporal graduation)

## MCP Server

MCP server implementation coming soon. The library provides the foundation; the server will expose 5 tools (`record`, `recall`, `prepare_wrap`, `save_continuity`, `status`) and 1 resource (`anneal://continuity`).

## Security

anneal-memory includes tool description integrity verification. A `tool-integrity.json` file contains SHA256 hashes of all tool descriptions, verified at server startup. This prevents tool description poisoning — a class of attack where manipulated descriptions alter LLM behavior.

## License

MIT

## Author

Phill Clapham / [Clapham Digital LLC](https://claphamdigital.com)
