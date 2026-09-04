# anneal-memory

**Living memory for AI agents. Episodes compress into identity.**
**Status:** 🚢 **v0.5.0 SHIPPED 2026-06-05** — PyPI + GitHub (`v0.5.0` / main `6bbd16d`, tag `v0.5.0`, fleet-deployed). The 0.x arc ran 0.4.0→0.5.0 (spores · the immune-half-dead arc 0.4.5/0.4.6 · 0.4.7 Tony-committed + held batch · **AM-PRESERVE-BARE-PATH 0.4.8** · **AM-SEMDUP + AM-ROLECHECK 0.5.0** — the batched convenience release: a prepare-package merge-don't-fork dedup scan + a mis-roled-schema warning; full 4-layer apparatus, codex L3 to convergence across 4 passes, 1227 tests; AM-CHIPSCHEMA deferred to the Levain v2 reload). ⚠️ This brief.md is otherwise v0.3.x-era — **canonical current state is `next_steps.md`** ("## ✅ 0.5.0 SHIPPED"). Prior: 🚢 **v0.3.5 SHIPPED 2026-05-31** — PyPI + GitHub, the **catastrophic-shrink gate**: a partnership-entity-scoped STRUCTURAL refusal of felt/identity-layer collapse at the wrap pipeline (`validated_save_continuity(..., allow_shrink=...)`, fail-closed — `structural_invariants_beat_discipline` at the save boundary; origin = a Sonnet wrap recency-trapping the neocortex 19.7k→1.6k, now made structural). 4 cross-substrate review rounds (codex r2/r3 closed two protected-layer fail-opens past the Claude-lineage L1/L2), **851 tests + mypy clean**, commits `f297af9`/`f53e79b`, tag `v0.3.5`. **flow's editable install runs v0.3.5 → flow is protected by the live gate.** Publish gotcha: pin `hatchling==1.27` (1.30 emits metadata 2.5 twine rejects); `PYPI_API_TOKEN2` in `.env.flow`. Prior: 🚢 **v0.3.4 SHIPPED 2026-05-31** — PyPI + GitHub, pluggable continuity section schema (the flow-as-dogfood feature: `schema.py` + schema-aware `validate_structure`/`_build_wrap_instructions`/graduation-gate; `FLOW_SCHEMA` with timeless `## Understanding`), commit `30a376a`, tag `v0.3.4`, 829 tests + mypy clean, 4-layer apparatus'd — the anneal-side build of the flow→anneal migration Phase 2 (`flow_migration_scoping.md`). Prior: 🚢 **v0.3.1 SHIPPED 2026-05-17** — PyPI + GitHub, phantom-re-save fix (`validated_save_continuity` refuses a save with no wrap in progress), commit `3f79a00`, 707 tests; flow's primary build bet (un-archived May 17, 2026). **⚠️ The rest of this brief.md is v0.3.0-era and pending a full refresh at the build-out-roadmap session — see `next_steps.md` for current state.** Prior release: 🚢 v0.3.0 SHIPPED TO PyPI May 1, 2026 afternoon (release commit `f0a36ff` + tag `v0.3.0`). Deprecation cleanup release — removes the v0.2.0-deprecated `prepare_wrap_package()` public wrapper and tightens `Store.wrap_started()` to required keyword-only `token: str` + `episode_ids: list[str]` (legacy no-arg form gone). Bundles 2 Diogenes May 1 LOWs: `_batch()` `_conn=None` rollback guard + `StoreError` "Raised by" enumeration drift fix (replaces inline list with pointer to `_db_boundary` call sites as canonical SoT). 7 review-pass docstring/code-hygiene fixes from a four-layer review pipeline (session-code-review + domain-expert + complement/gemini/contrarian consultation + integration semantics smoke test). Two consecutive mechanism-accuracy errors caught in the same docstring rewrite — Layer 2 + Layer 3 each found one; both fixed before tag. Reinforces `feedback_consultation_blind_to_mechanism_errors.md` calibration discipline. 704 tests. Sessions 1–10.6 ship arcs all COMPLETE. Prior arcs: v0.2.0 (Apr 13 — library canonical + thin transport adapters + crash-safe wrap pipeline + exception hierarchy), v0.2.1 (Apr 14 — framework guide drift fixes), v0.2.2 (Apr 29 — README positioning expansion + 2 Diogenes LOWs), v0.2.3 (Apr 30 — mechanism-accuracy precision fix), v0.3.0 (May 1 — deprecation cleanup + signature tightening). **Current work horizon is gated or exploratory:** Session 10 identity experiment (paper gates cleared but architecture paper PARKED + Atrium S7 higher priority), 6 deferred Session 9/9.5 infrastructure items (gated on Session 10 data), Compliance Proxy ~Jun 1 decision point. **Unblocked anytime:** Session 12 (multi-agent shared memory / per-agent associations) + Session 13 (adversarial self-confirmation test — the explanation-grounding + cross-session anti-sycophancy checks; formerly "anti-inbreeding").
**Created:** March 31, 2026
**Updated:** June 1, 2026 (v0.3.5 ship entry added — catastrophic-shrink gate; brief had been tracking v0.3.4, diogenes 2026-06-01 doc-drift catch)
**Active arc (2026-05-20 onward):** **The Bold Stand** — public methodology claim arc. See `next_steps.md`. Separate from v0.4 (which remains gated by Levain's step-4 needs). Library is shipped + stable; active work is adversarial verification + positioning, not new features.

---

## What This Is

A memory system for AI agents built on four cognitive layers: an episodic store (fast, timestamped, searchable), a continuity file (compressed, always-loaded, rewritten at session boundaries), Hebbian co-citation associations (lateral links between episodes that strengthen through semantic judgment during graduation), and a limbic affective layer (functional-state tagging on associations, intensity-modulating association strength). A cross-layer immune system enforces citation-validated graduation, the explanation-grounding check, and active demotion of ungrounded citations across all four layers. (The README's ten enumerated primitives are canonical — "anti-inbreeding defense" and "principle demotion" were retired 2026-05-21 as names that mapped to no code path.) Episodes accumulate like raw material. Compression into continuity IS the cognition — the act of compressing forces pattern recognition, abstraction, and judgment. The system develops over time, getting smarter rather than just bigger.

Available as a Python library, CLI, and MCP server. Three access patterns: library (programmatic), CLI (operator/developer + agent-driven compression), MCP (agent-in-conversation). All zero-dep (Python stdlib only). All paths preserve the cognitive loop — the agent that records episodes is the agent that compresses them.

## Positioning

**"The only AI agent memory system with an immune system."**

[decided(rationale: "MCP backlash is structural (4-32x token overhead, 82% vuln rate, Perplexity/Nx/YC CEO moving away). Leading with 'MCP' turns off growing anti-MCP segment. Leading with 'memory system' loses nobody. Immune system differentiates regardless of transport.", on: "Apr 9")]

**Three access patterns (library is the product, CLI and MCP are interfaces):**
1. **Python library** — import, instantiate, call. Zero overhead. Works in any framework.
2. **CLI** — inspect, debug, audit, export, script. Operator interface. Unix composable. Things MCP can't do.
3. **MCP server** — real-time agent memory during conversation. Zero-dep. 5 tools + 1 resource.

**12 framework integration paths** (lifecycle callbacks for top 3, integration guides for all):
LangGraph, CrewAI, OpenAI Agents SDK (callbacks) | Google ADK, Pydantic AI, smolagents, LlamaIndex, Haystack, CAMEL-AI, Autogen/AG2, DSPy, Anthropic Agents SDK (guides/snippet)

Competitive landscape (413+ memory MCPs, detailed in `contexts/archive/anneal_memory_competitive_intel.md`):
- No competitor has citation-validated graduation (patterns must cite episode IDs)
- No competitor has the explanation-grounding check (`check_explanation_overlap`) or the cross-session anti-sycophancy check — the two primitives that used to be marketed as "anti-inbreeding defense" (name retired 2026-05-21 for naming honesty; use the shipped names)
- No competitor has active demotion of ungrounded citations, or staleness flagging (the two primitives that used to be marketed as "principle demotion" — same retirement; note the library flags staleness but does NOT auto-demote on staleness alone)
- No competitor has typed episodes (observation/decision/tension/question/outcome/context)
- No competitor has tamper-evident audit trail with hash chaining
- Closest threat: mcp-memory-service (dream consolidation) — but dependency-heavy, no evidence chains

**Three audiences see three things:**
1. **Developers:** Best AI agent memory system. Immune system, zero-dep, three access patterns, 12 framework integrations.
2. **Researchers/paper:** Empirical validation of identity-through-memory thesis.
3. **Enterprise/compliance:** Audit infrastructure for EU AI Act logging requirements (Articles 12, 13, 17, 26, 86). Honest framing: infrastructure that helps systems comply, not certification.

### Internal Strategic Frame (NOT public positioning)

anneal-memory = **reasoning-sovereignty layer of the cognitive sovereignty stack** (user owns HOW the agent thinks — association topology, graduated principles, developing judgment). Structurally the missing middle layer between user-owned inference (maturing local models, 25x/yr compression) and user-owned orchestration (harness frameworks). Together these constitute the off-ramp from techno-feudal cognitive capture for those who choose it.

This is the technical realization layer of Phill's decade-long cognitive liberation mission — not new framing, but substrate finally catching up to an architectural requirement that pre-dated the technology to satisfy it.

**Implications for architecture decisions:**
- Session 9 (Hebbian) = building the cognitive association topology users will own
- Session 9.5 (limbic) = affective layer = deepest cognitive property (what the agent learns to CARE about)
- Session 10 (identity experiment) = cognitive sovereignty demonstration (owned vs. unowned agent identity trajectory)
- Session 12+ (multi-agent shared memory) = communal sovereignty architecture (collectives > individuals historically)

**Framing discipline:** Internal vision (sovereignty infrastructure) ≠ external positioning (compliance + cognitive architecture + developer DX). Public positioning waits for product maturity. `positioning_ahead_of_product_kills_credibility` (3x Proven) applies here.

**Full strategic context:** `projects/archive/strategic_vision/vision.flowscript` Section 1.5 (SOVEREIGNTY_STACK) — strategic_vision archived May 18, 2026; live strategy lives in continuity.md.

## Architecture

**Four cognitive layers (CLS-backed — not a metaphor, convergent architecture):**
- **Episodic store (SQLite):** Timestamped, typed episodes. Append-heavy, indexed. The hippocampus.
- **Continuity file (Markdown):** Compressed session memory. 4 sections (State/Patterns/Decisions/Context). Always loaded. The neocortex. *(v0.4/v0.5 direction: pluggable per-section schema → roles `graduating`/`narrative`/`live-state`/`frozen`; partnership-entities add a second `narrative` section `## Understanding` — timeless relationship-shape, the felt-continuity layer ops-entities don't need. See `next_steps.md` flow-as-dogfood + `flow_migration_scoping.md`.)*
- **Hebbian associations (Session 9, v0.1.8):** Lateral links between episodes that strengthen via co-citation during graduation. Deep Hebbian — associations form from semantic judgment during consolidation, not temporal proximity. Two tiers: direct co-citation (+1.0) vs session co-citation (+0.3). Decay 0.9 per wrap, strength cap 10.0, cleanup at 0.1. Immune system extends to associations (only validated citations form links).
- **Limbic layer (Session 9.5, v0.1.8):** Affective state tagging on associations. Agent self-reports functional state during consolidation. Intensity modulates association strength (up to 1.5x). Provides the persistent emotional state tracking that transformers lack natively.

**Compression mechanism:** At session boundaries, recent episodes feed into the continuity file via LLM compression guided by a simplified 9-marker FlowScript subset. Temporal graduation (1x→2x→3x) with citation validation. Ungrounded citations are actively demoted; stale patterns are FLAGGED (`detect_stale_patterns`) — the library does not auto-demote on staleness alone, the agent decides. ("Citation decay" is a retired name.)

**Agent-driven compression (all access patterns preserve the cognitive loop):**
- **Library (canonical):** `prepare_wrap(store)` → agent compresses → `validated_save_continuity(store, text)`. This is the pipeline — structure validation + graduation + Hebbian associations + decay + metadata + wrap_completed all happen here.
- **MCP:** Agent calls `prepare_wrap` tool → compresses → `save_continuity` tool → transport adapter delegates to the library functions above.
- **CLI:** Agent runs `prepare-wrap` subcommand → compresses → `save-continuity` subcommand → transport adapter delegates to the library functions above.

All three access patterns go through the same library pipeline. There is one implementation, not three. The cross-transport parity test locks this equivalence as a structural invariant.

[decided(rationale: "Engine delegated compression to a separate LLM, removing the agent's judgment from consolidation. Compression IS cognition — every access pattern must preserve this. Dead code defined by workflows, not test count.", on: "Apr 9")] Engine removed. No automated compression bypass.

[decided(rationale: "Three parallel implementations of the save_continuity pipeline (MCP/CLI/library) had already diverged 12 hours after 10.5c shipped. Library becomes canonical; transports become thin adapters. Matches the 10.5c library-first positioning at the shape of the code, not just the shape of the docs.", on: "Apr 10")] Library canonical + thin transport adapters. Rule-of-three structurally eliminated.

**5 MCP tools + 1 resource. 21 CLI subcommands. Zero-dep install.**

**Compliance layer (v0.1.5):** Hash-chained JSONL audit trail. SHA-256 chain, weekly rotation with gzip, manifest index, actor identity, content-hash-only (GDPR-compatible), crash recovery, `on_event` callback for cloud/SIEM. Always-on by default. 308 tests. Full EU AI Act gap analysis at `contexts/eu_ai_act_analysis.md`.

## Two-Layer Compliance Model (Strategic Vision)

anneal-memory's current audit trail records what the memory system does (agent-directed episodes + internal mutations). This covers Article 12(2)(b,c) and much of the Act's logging requirements. But Article 12(1) requires "automatic recording of events" — the agent shouldn't have to choose to record compliance-relevant events.

**The gap:** If the agent makes a decision and doesn't record an episode about it, the audit trail has no idea it happened. The CLAUDE.md snippet guides recording, but guidance ≠ enforcement.

**The solution: two complementary compliance layers sharing the same store.**

| Layer | What it captures | How | Source field |
|-------|-----------------|-----|-------------|
| **Layer 1 — Memory audit** (shipped, v0.1.5) | Agent-directed episodes + internal mutations | Automatic sidecar of memory operations | `source="agent"` |
| **Layer 2 — Compliance proxy** (future) | ALL agent actions — every tool call, every response | MCP transport-layer interception, agent doesn't know | `source="system"` |

**Both write to the same episodic store.** The `source` field distinguishes intentional recording from automatic capture. Compression/graduation operates only on agent-directed episodes (you don't develop identity from raw API logs). The audit trail captures both layers.

**The FlowScript lineage:** FlowScript's wrapper approach (FlowScriptOpenAI/FlowScriptAnthropic transport interception) had the right thesis but wrong DX — required import changes, notation learning, one product doing too many things. MCP standardizes the transport (JSON-RPC over stdio), enabling a proxy that intercepts all tool calls transparently. Zero code changes for the agent developer. Same thesis, 10x better DX.

**With both layers:** Memory audit = "here's what the agent learned." Compliance proxy = "here's everything that happened." Together = full EU AI Act Article 12 coverage at the infrastructure level.

**Timing:** EU AI Act enforcement for Articles 9, 12, 13, 14, 17, 26, 86 begins **August 2, 2026** — roughly 15 weeks out from tonight's v0.2.0 ship (Apr 13 very late evening). Most AI tools have zero audit infrastructure. Early mover.

**Roadmap anchor (Aug 2, 2026):** this is a hard external date the roadmap respects. If enterprise interest in anneal-memory's audit positioning materializes before then, Compliance Proxy (Layer 2) build starts ~June to clear the enforcement window with shipped infrastructure, not mid-build infrastructure. Decision point: ~Jun 1, 2026. Signal threshold: any enterprise inbound interest, not "explicit paying customer" (that gate would miss the window). If no signal by Aug 1, revisit based on post-enforcement market. See `project_memory/next_steps.md` "~June 2026 — Compliance Proxy (Layer 2) decision point" for full rationale.

## Key Decisions

{decisions:
  [decided(rationale: "SQLite is stdlib, handles queries for citation validation. Continuity markdown is human-readable layer.", on: "Mar 31")] Episodic = SQLite, continuity = Markdown
  [decided(rationale: "Agent IS the LLM. Agent compresses, server validates. Purest compression-as-cognition.", on: "Mar 31")] MCP = agent compresses, server validates
  [decided(rationale: "Single repo, optional extras. Zero-dep MCP.", on: "Mar 31")] Single repo with optional [engine] extra
  [decided(rationale: "MCP 2024-11-05 spec = newline-delimited JSON. Content-Length framing caused 30s timeout.", on: "Apr 1")] Stdio transport: newline-delimited JSON (not Content-Length)
  [decided(rationale: "Ship first, experiment alongside. Real users strengthen the paper. LangChain interview Fri Apr 4.", on: "Apr 1")] Reordered: publish before experiments
  [decided(rationale: "Compliance is a VIEW of episodic store, not replacement. JSONL sidecar + hash chains. Don't force per-turn reasoning (FlowScript's mistake).", on: "Apr 1")] Compliance layer = JSONL audit sidecar with hash chaining
  [decided(rationale: "Compliance is VIEW of episodic store. JSONL sidecar + hash chains. Content by hash not duplication (GDPR). Retention default: keep forever.", on: "Apr 2")] Compliance = tamper-evident audit infrastructure (not 'compliance-grade' — honest framing per 4-layer review)
  [decided(rationale: "SQLCipher (SQLite) + cryptography lib (JSONL) both need deps. Natural fit for cloud tier, not zero-dep core.", on: "Apr 2")] Encryption at rest: deferred to cloud/premium tier
  [decided(rationale: "Current audit trail records memory mutations. Article 12(1) requires automatic recording of ALL events. Proxy layer closes the gap. FlowScript wrapper thesis reborn with MCP DX.", on: "Apr 2")] Two-layer compliance: memory audit (shipped) + compliance proxy (future). Both use same store. source field distinguishes.
  [decided(rationale: "Memory + compliance integrated is stronger product story. Proxy is latent in architecture (source field, on_event callback). Build when demand pulls.", on: "Apr 2")] Keep compliance tied to memory for now. Design for separation. Split when demand materializes.
  [decided(rationale: "AnnealCloud only if demand materializes. Compliance proxy + witness services + encryption + external timestamps = premium tier.", on: "Apr 2")] Premium cloud layer: parked until demand signals
}

## Repos

- **Code:** `~/Documents/anneal-memory/` (Python, MIT, public)
- **Project tracking:** `~/Documents/flow/project_memory/` (this directory)
- **GitHub:** `https://github.com/phillipclapham/anneal-memory`
- **PyPI:** [`anneal-memory`](https://pypi.org/project/anneal-memory/) 0.2.1

## Relationship to Other Projects

- **FlowScript:** anneal-memory is informed by FlowScript's ContinuityManager (same graduation concepts, fresh codebase) AND FlowScript's wrapper architecture (compliance proxy vision reborn with MCP DX). FlowScript repos ARCHIVED on GitHub Apr 8 with evolution narrative. flowscript.org LIVE as notation reference. Lineage section added to anneal-memory public README. The compliance hash-chaining adapted from FlowScript audit trail.
- **Flagship paper:** anneal-memory IS the experimental platform. Paper Claims Tracker in `experiment_results.md` — 5 validated, 3 partial, 3 untested. Compliance layer + proxy vision = paper's enterprise relevance section.
- **LangChain prep:** LangGraph research agent built as triple-function artifact. Architecture guide at `contexts/archive/langgraph_architecture_guide.md`.
- **Flow system:** anneal-memory mirrors what flow's continuity.md does manually. Convergent architecture, not shared code.
- **Harness-as-embodiment thesis (Apr 4):** anneal-memory is the MEMORY component of digital embodiment. Intelligence emerges from the harness (sensors + memory + identity + social + immune), not the LLM. Memory enables identity enables alignment. anneal-memory provides the encoding + consolidation + association infrastructure that makes the harness coherent enough to produce generalized intelligence. See `projects/agent_identity_research/` for full thesis.
- **EU AI Act:** Full provision-by-provision analysis at `contexts/eu_ai_act_analysis.md`. Compliance deadline August 2, 2026.
- **Atrium umbrella (NEW Apr 14 afternoon, REFRAMED Apr 14 evening):** anneal-memory is the first extracted COMPONENT of a larger meta-project called Atrium (codename locked). It ships **Components 1-3** of the Atrium harness taxonomy: individual Memory + Identity + individual Immune system. The afternoon scoping identified "Commons Foundation" as the next target (social + collective immune system) but the Apr 14 evening architectural correction — triggered by Session 14a.1 + 14a.2 spike data + a first-principles reset — revealed that Commons-as-memory-layer was a category error that would have violated the flagship paper's own `generator_independence_as_harness_precondition` mechanism. **The reframed Atrium component list is 8 core + 1 extension.** Core: Memory (1, shipped here), Identity (2, shipped here), Immune (3, shipped here), **Canon (4, NEW — externalized reference substrate with static + computed sub-layers)**, Sensors (5), Values (6), Self-observation (7), Host abstraction (8). Extension: Social (9, multi-agent only, covers transport + coordination + shared blackboard + aggregation functions + collective immune surveillance). anneal-memory is orthogonal to Canon and Social — those are their own Atrium components with their own Foundation documents to be written. Next Atrium extraction targets: Canon Foundation + Social Foundation, scoped in close temporal proximity. **AnnealCloud framing retired** — managed hosting becomes a service layer on top of the self-hostable complete stack, not a feature gate. See `projects/atrium/brief.md` (§Reframe note + §Nine Components) for the active architectural picture. The retired `project_memory/commons_foundation.md` carries a SUPERSEDED header and is preserved only as a lineage artifact.

---

*Updated: April 9, 2026*
