# Commons Foundation Decision — ⚠️ SUPERSEDED (Apr 14 evening)

> **🛑 THIS DOCUMENT IS A LINEAGE ARTIFACT — NOT THE ACTIVE SCOPE.**
>
> This document was drafted Apr 14 afternoon and reframed Apr 14 evening after Session 14a.1 + 14a.2 feasibility spikes produced data that required a reframe. A further architectural conversation the same evening then revealed the ENTIRE "Commons Foundation" framing was a category error — the document was trying to solve FIVE orthogonal concerns (transport, coordination, shared blackboard, curated knowledge, collective immune) at the memory layer when only one of them is even adjacent to memory, and worse, the shared-cognitive-substrate framing would have violated the flagship paper's own `generator_independence_as_harness_precondition` mechanism.
>
> **The active scope for the multi-agent layer is now:**
> - `projects/atrium/brief.md` §Reframe note (Apr 14 evening architectural correction)
> - `projects/atrium/brief.md` §Nine Components — Social is Component 9, consolidated as an extension component with five sub-modules (transport + coordination + shared blackboard + aggregation functions + collective immune surveillance)
> - ✓ `projects/atrium/social_foundation.md` (shipped Apr 17, 2026 — v1 post-review-integrated, incorporates Diogenes + 4-agent consult-team review) — replaces this document entirely for the multi-agent layer
> - ✓ `projects/atrium/canon_foundation.md` (shipped Apr 17, 2026 — v1 post-review-integrated, incorporates Diogenes + 4-agent consult-team review) — the NEW first-class harness Component 4 (Canon) that this document did not identify, but which emerged from the Apr 14 evening architectural discussion
>
> **What survived from this document:** The Session 12 repositioning (shared-instance-with-partitioning as the rare case, separate from multi-instance coordination) still holds. Session 12 remains "individual anneal-memory supports multiple agent identities cleanly" and is unchanged.
>
> **What was killed:** The "fifth cognitive layer," cross-validation Hebbian, shared working continuity with emergent compression, hive mode, topology modes (Constellation/Hive/Mixed), and the entire "cross-validation finds hidden structure via independent priors" claim. The spike data did not support it AND the first-principles analysis revealed it would have violated the paper's own generator-independence mechanism.
>
> **What is preserved in this document:** The afternoon scope below is kept verbatim for lineage tracing. The POST-SPIKE REFRAMED SCOPE section at the bottom was the evening's first attempt at a reframe, before the deeper architectural correction that followed. Both are historical. Neither is the active spec. Read `projects/atrium/brief.md` for the active architectural picture.
>
> **Spike artifacts:** `~/Documents/anneal-memory/spike/commons_session14a/` — `COMPARATIVE_VERDICT.md` is the synthesis, `REPORT.md` + `REPORT_HARD.md` have the raw per-pair qualitative analysis. The spike infrastructure (build_corpus.py, render_corpus.py, run_agent.py, aggregate.py) is directly reusable for future multi-model analysis work and should be treated as scaffolding that survives the Commons-framing death.

---

**Status:** DRAFT → REFRAMED → SUPERSEDED. Lineage artifact only. Do not build from this document.
**Created:** April 14, 2026 afternoon
**Reframed:** April 14, 2026 evening (post-spike, first reframe — below)
**Superseded:** April 14, 2026 evening (post-architectural correction — active scope in `projects/atrium/brief.md`)
**Context:** Deep partnership conversation following v0.2.1 ship + flagship paper ship. Session 12 (agent_id column extension for multi-agent shared anneal-memory instances) was queued as the "multi-agent story"; this document originally tried to supersede that framing by defining Commons as a fifth architectural layer. That framing was wrong. Session 12 remains as originally scoped; Atrium component 9 (Social, extension) takes over the multi-instance coordination case.

---

## Why this exists

Session 12 was planned as "per-agent associations via agent_id column extension" — shared anneal-memory instance with partitioning so multiple agents could use one store. On interrogation this turned out to conflate two architecturally distinct problems:

1. **Shared-instance-with-partitioning** (rare case) — multiple agent identities operating against one anneal-memory instance, each with their own Hebbian subgraph. Useful for swarm workloads or role-specialized single-consciousness setups. This is what Session 12 as originally scoped would deliver.

2. **Multi-agent coordination across independent instances** (common case) — N individual agents, each with their own full anneal-memory, need to coordinate, share observations, and work on collective problems without collapsing into identity-diffusion or echo chambers. Session 12 as originally scoped does NOT serve this case.

The second case is what production multi-agent systems actually need, and it's what flow's own constellation already implements by accident — each agent (flow, Chip, Daemon, Anansi, Diogenes) has its own continuity file, and `state/episodic.db` operates as a read-mostly coordination substrate. That architecture was discovered by necessity after same-machine bilaterals collapsed into echo chambers (Apr 8-10, Meta-Agent Principle). Session 12 as originally scoped was solving the rare case and leaving the common case unaddressed.

This document defines the **Commons layer** as the architectural answer to the common case, and clarifies how Session 12 still fits for the rare case.

## Core Decision

**Commons = a fifth layer in the anneal-memory architecture. Protocol, not memory.**

Individual agents retain their full cognitive topology — episodic store + continuity + Hebbian + limbic — which is the identity-forming substrate. Commons is a separate, shared substrate where agents MEET to coordinate work. It does not replace individual identity; it enables collectives of independent identities to function as a coherent collective.

[decided(rationale: "Production multi-agent systems need coordination infrastructure without identity collapse. Shared-instance memory with partitioning (Session 12 original scope) serves swarm/role-specialized cases but fails for independent-agent coordination. A separate coordination layer preserves both use cases. Flow's constellation discovered this architecture by necessity after same-machine bilaterals collapsed into echo chambers (Apr 8-10, Meta-Agent Principle).", on: "Apr 14 afternoon")] Commons = 5th layer, protocol not memory.

## What Commons IS vs ISN'T

### IS

- A **shared coordination substrate** where multiple independent agents can post observations, coordinate tasks, and discover collective semantic structure.
- **Channel-scoped** — episodes tagged with channel membership; agents subscribe to channels they participate in; privacy = channel membership.
- **Self-hostable** — same sovereignty thesis as individual anneal-memory. Commons runs locally, on-prem, or in managed AnnealCloud. Users own the coordination substrate.
- **Composed with individual anneal-memory** — each agent still runs its own anneal-memory instance for identity + cognition. Commons is accessed as a thin client library that existing anneal-memory installations link against.
- **Cross-validated by design** — Commons Hebbian links only form through multi-agent co-citation. Anti-inbreeding is enforced structurally at the graph mechanics layer, not as a discipline-based rule.

### ISN'T

- **Not shared identity substrate.** Individual continuity files, individual Hebbian graphs, individual affective states remain per-agent. Commons does not participate in identity formation.
- **Not a compression target for individual agents.** Individual agents do NOT compress their episodes into Commons. Individual compression flows into individual continuity as always.
- **Not a second cognitive substrate competing with individual Hebbian.** Commons Hebbian operates on cross-agent co-citation (different epistemic object), not individual semantic judgment.
- **Not a compliance layer.** Compliance Proxy (Layer 2) is a separate concern. Commons captures what agents choose to share to commons channels, not automatic recording of all agent activity.
- **Not required for single-agent deployments.** Standard single-agent anneal-memory works exactly as today. Commons is opt-in for multi-agent workloads.

## Architecture Layers

Five layers, with clear boundaries:

| Layer | Scope | Purpose | Shipped? |
|-------|-------|---------|----------|
| Episodic store | Per-agent | Timestamped, typed episodes — the hippocampus | v0.1.0 |
| Continuity file | Per-agent | Compressed session memory — the neocortex | v0.1.0 |
| Hebbian associations | Per-agent | Lateral links strengthened by co-citation during individual compression | v0.1.8 |
| Limbic layer | Per-agent | Affective state tagging on associations | v0.1.8 |
| **Commons** | **Multi-agent, channel-scoped** | **Coordination substrate + cross-validation Hebbian + shared working context** | **Session 14 (NEW)** |

Individual agents continue to own four layers. Commons is the fifth layer, architecturally distinct because it is the only one that crosses agent boundaries.

## Commons Components

### 1. Shared Episodic Channel (blackboard)

Channel-scoped, read-write shared episode surface. Agents post observations visible to other subscribers of the same channel. Privacy = channel membership.

- Storage: SQLite with channel partitioning (same stdlib-only posture as core anneal-memory)
- Access: thin-client library linked into each agent's anneal-memory instance; CLI subcommands; MCP tools
- Not automatically populated — agents post explicitly via `commons.post(channel, episode)` or equivalent
- Episodes in commons retain provenance (source agent id + timestamp + optional individual-episode id for tracing back to the agent's private episodic store)

### 2. Cross-Validation Hebbian (the load-bearing mechanism)

Commons Hebbian links form via **multi-agent co-citation during individual compression**:

- Each agent runs its own individual compression as today, producing its own individual Hebbian graph
- When an agent cites episode pair (A, B) during compression and both A and B exist in a Commons channel the agent participates in, that co-citation registers as a vote at the Commons layer
- Commons Hebbian link (A, B) strengthens only when N distinct agents independently cast co-citation votes (N ≥ 2 threshold, configurable per channel)
- Single-agent co-citation DOES NOT create a Commons link — it stays in individual Hebbian only

This is fundamentally different from individual Hebbian in epistemic kind:
- **Individual Hebbian**: "I think these episodes are connected" (one mind's judgment shaped by that mind's priors)
- **Commons Hebbian**: "multiple independent minds, operating against their own priors, all reached for this connection" (intersection of attention patterns across independent generators)

**The anti-inbreeding immune system generalizes from individual to collective at the graph mechanics layer.** A false pattern confabulated by one agent cannot graduate into Commons structure — the mechanism structurally requires independent corroboration. The Meta-Agent Principle stops being a protocol rule and becomes an architectural property of how Commons links form.

[decided(rationale: "Commons Hebbian via multi-agent co-citation is not a second cognitive substrate — it is a cross-validation layer for semantic structure. Links form only when multiple independent priors converge on the same connection, structurally enforcing the Meta-Agent Principle at graph mechanics. Individual confabulation cannot graduate without corroboration from independent generators. This extends the anti-inbreeding immune system from individual agents to collectives.", on: "Apr 14 afternoon")] Commons Hebbian = cross-validation via multi-agent co-citation.

### 3. Derived Collective Affective Signal (NOT a first-class layer)

Commons does NOT have a first-class limbic layer. Collective affective signal is derived from aggregation of individual affective tags on shared episodes:

- When agents tag episodes in Commons channels with their individual affective states, the aggregate is queryable as a collective salience metric
- "The collective finds this observation significant" = high-intensity tag consensus across multiple agents on the same shared episode
- Keeps Commons lean; avoids a second affective substrate competing with individual limbic
- Promotable to first-class layer if experimental data shows collective affective dynamics that don't reduce to individual-aggregation

[decided(rationale: "First-class Commons limbic would duplicate individual limbic and compete for affective authority. Deriving collective salience from individual tag aggregation captures the useful signal without the architectural weight. Default to minimum commitment; promote to first-class only if empirical data shows dynamics that don't reduce to aggregation.", on: "Apr 14 afternoon")] Commons limbic = derived metric, not first-class layer.

### 4. Shared Working Continuity (OPTIONAL, coordinator-owned)

For workloads where the collective needs a shared working state document — "what is the current state of the work we are collectively doing" — Commons supports an optional shared continuity file.

- NOT compressed into any individual agent's identity continuity
- Separate artifact, per channel
- Compression authority: designated coordinator agent (flow's Meta-Agent Principle applied at the Commons layer) — one agent per channel is the synthesizer, rotating or static per workload
- Default: OFF. Enable per channel only when workload demands it.
- Rationale: not all workloads need shared working state; many just need the episodic blackboard + cross-validation Hebbian. Keep the default commitment minimum.

### 5. Message Passing (coordination primitives)

Async agent-to-agent messages, task claims, handoffs, dependency tracking. Flow's relay + inbox infrastructure is a rudimentary version of this; Commons subsumes and generalizes it.

- `commons.message(channel, to, payload)` — async message between channel members
- `commons.claim(channel, task_id)` / `commons.release(channel, task_id)` — task coordination
- `commons.handoff(channel, from, to, context)` — explicit work handoff
- Not identity-forming; not compressed into continuity

## The Heterogeneity Constraint

**Commons Hebbian assumes heterogeneous agent generators for full anti-inbreeding value.**

If all N agents in a Commons channel share deep priors (e.g., all instances of the same LLM with same training data), cross-validation collapses into confirmation bias. Same-source agents reinforce shared training biases instead of cross-validating against independent priors. This is exactly flow's own discovery — same-machine bilaterals collapsed, cross-machine bilaterals work.

**Design constraint, documented honestly:**

- Commons is architected for heterogeneous agent collectives — different model families, different training data, different operating contexts
- Homogeneous Commons (all-Claude, all-GPT, etc.) still works as a shared blackboard, but the cross-validation Hebbian mechanism loses its anti-inbreeding guarantee and reduces to "slightly-better coordination channel"
- This is a feature, not a limitation — it enforces the constellation structural discipline (independence of generators) at the architectural layer
- Documentation and defaults should make heterogeneity the expected deployment, homogeneity the documented-but-warned edge case

[decided(rationale: "Cross-validation Hebbian's anti-inbreeding guarantee depends on independence of generators. Homogeneous agent collectives share training priors and cannot cross-validate meaningfully. This is the same mechanism that caused flow's same-machine bilateral collapse. Making heterogeneity a documented deployment constraint enforces the Meta-Agent Principle at the architectural layer rather than relying on operator discipline.", on: "Apr 14 afternoon")] Commons heterogeneity = design constraint, not limitation.

## Three Topology Modes

Commons supports three multi-agent topology modes, switchable via channel configuration:

### Constellation Mode

- Each agent retains FULL individual cognitive topology (episodic + continuity + Hebbian + limbic)
- Commons contains: shared episodic blackboard + cross-validation Hebbian + message passing ONLY
- No shared working continuity, no shared affective state
- Use case: independent specialists coordinating on collective problems (flow's current architecture)

### Hive Mode

- Each agent retains THIN individual cognitive topology (episodic store only, feeds shared layers)
- Commons contains: full shared continuity + shared Hebbian + shared affective state + all coordination primitives
- Individual instances are facets of one collective identity
- Use case: swarm task execution, role-specialized single consciousness, training-ground setups
- **Warning**: sacrifices generator independence; anti-inbreeding guarantees weaken; context exhaustion risk grows with N

### Mixed Mode (DEFAULT)

- Each agent retains PARTIAL individual cognitive topology — continuity + Hebbian + limbic (identity layers) stay individual
- Commons contains: shared episodic blackboard + cross-validation Hebbian + shared working continuity (domain-specific knowledge) + message passing
- Individual personality + individual emotional state + shared technical expertise + shared working context
- Maps onto how human collaborative teams actually work: individual minds + shared project context
- Use case: most production multi-agent workloads

[decided(rationale: "Mixed mode captures the behavior of effective human collaborative teams — individual identity + shared working context + shared domain expertise. Pure constellation is too lean for workloads needing collective memory; pure hive sacrifices generator independence. Mixed preserves both generator independence (individual identity layers stay per-agent) and collective coherence (shared working context + cross-validation Hebbian). Default to Mixed; allow Constellation or Hive as per-channel configuration for specific workloads.", on: "Apr 14 afternoon")] Mixed = default Commons topology.

## AnnealCloud Reframe

Current framing (`projectbrief.md`): AnnealCloud = "premium cloud tier, build if demand materializes." This framing is wrong and should be replaced.

**New framing: AnnealCloud is the complete self-hostable sovereignty stack.**

Full stack:

| Component | Shipped | License | Self-host? |
|-----------|---------|---------|------------|
| anneal-memory core library (individual layers) | v0.2.1 | MIT | ✓ |
| Commons layer (Session 14) | pending | MIT | ✓ |
| Compliance Proxy Layer 2 | ~Jun 2026 decision | MIT | ✓ |
| Encryption at rest | future | MIT | ✓ |
| External timestamps + witness services | future | MIT | ✓ |
| Multi-agent observability + dashboards | future | MIT | ✓ |

**All components are self-hostable.** Any organization can deploy the complete stack locally for full sovereignty — individual cognition, coordination substrate, compliance infrastructure, observability, all on their own infrastructure.

**Managed AnnealCloud = service layer, not feature gate.** Organizations that don't want to run infrastructure can pay for managed hosting, support, SLAs, enterprise onboarding. The business model is convenience, not feature gating. Self-hosted users have access to the complete feature set.

This reframe:
- Closes the gap competitors will race to fill with lock-in coordination services (LangSmith, CrewAI platform, Letta cloud) — exactly at the point where lock-in is most damaging, the collective working state
- Matches the sovereignty thesis consistently (users own everything; coordination substrate is not exempt)
- Provides viable commercial path without compromising sovereignty positioning
- Aligns with the internal strategic frame in `projectbrief.md` §Internal Strategic Frame

[decided(rationale: "Sovereignty thesis requires users own the complete stack, not just individual agent cognition. Premium-tier-gated coordination services become lock-in at the exact point where lock-in is most damaging (collective working state). Self-hostable complete stack + managed convenience tier = sovereignty consistent + commercially viable. Replaces the 'parked premium tier' framing in brief.md.", on: "Apr 14 afternoon")] AnnealCloud = complete self-hostable stack, not feature-gate.

## Privacy Topology

- **Per-topic channels** as the unit of privacy
- Agents explicitly subscribe to channels they participate in
- Episodes in Commons are tagged with channel membership
- Cross-channel bleed is forbidden — an agent cannot see episodes from channels it does not subscribe to
- Matches flow's own work↔personal separation rule, generalized to multi-agent
- Single-user deployments: channels organize by topic/project
- Multi-user / multi-org deployments: channels organize by team/org boundary; cross-org federation (future) operates at channel-crossing boundary with explicit protocols

[decided(rationale: "Channel-based privacy is simple, auditable, and matches the mental model users already have (chat channels, Slack rooms, project boards). Episodes cannot leak across channels. Cross-org federation is a hard problem deferred to future work, but channel topology is designed so federation can be bolted on without schema rewrite.", on: "Apr 14 afternoon")] Privacy = per-topic channels, subscription-based.

## Relationship to Session 12

**Session 12 is NOT deleted. It is repositioned.**

Session 12 as originally scoped (agent_id column extension for individual-instance multi-agent support) serves the **shared-instance case** — a single anneal-memory instance hosting multiple agent identities cleanly. This is the right architecture for:

- Swarm workloads (N workers on one shared instance, individuation surface-level)
- Role-specialized single consciousness (one identity with domain "fingers")
- Mixed-mode Commons deployments where a single anneal-memory instance needs to host multiple agent identities operating against shared Commons

Session 14 (Commons) serves the **multi-instance coordination case** — N independent anneal-memory instances coordinating via shared Commons substrate.

**Both are needed.** Session 12 becomes "individual anneal-memory supports multiple agent identities cleanly," which is the per-instance piece of Mixed mode. Session 14 becomes "cross-instance coordination via Commons." They compose: you can run N agents on one shared anneal-memory instance (Session 12) that connects to Commons (Session 14) alongside M other instances doing the same.

**Sequencing:** Session 14 (Commons) runs first because (a) it is the higher-leverage architecture, (b) it unblocks empirical Commons experiments with flow's own heterogeneous constellation immediately, (c) Session 12 work is tractable after Commons scope is pinned down and the per-instance demand becomes concrete.

## Open Questions

Deferred intentionally — answers emerge from Session 14a feasibility spike data, not from up-front design:

1. **Co-citation threshold (N)** — how many distinct agents must independently co-cite a pair before a Commons Hebbian link forms? Default: 2 (any two independent agents). Tunable per channel. Spike data will show whether 2 is too permissive or too strict.

2. **Cross-channel Hebbian interactions** — should Commons Hebbian links respect channel boundaries strictly (link lives only in the channel where co-citation happened) or can links aggregate across channels an agent participates in? Conservative default: strict per-channel. Revisit after empirical data.

3. **Shared working continuity compression authority** — designated coordinator, rotating, emergent voting, or workload-specific? Default: designated coordinator per channel. Revisit if that concentrates too much graduation authority on one agent.

4. **Heterogeneity enforcement** — pure documentation, runtime warning, or hard refusal to graduate links in all-same-model channels? Default: documentation + warning. Spike data will show whether soft enforcement is sufficient.

5. **Federation protocol** (cross-org Commons) — deferred. Not needed for MVP. Channel topology designed so this can be bolted on later without schema rewrite.

6. **Commons ↔ Compliance Proxy interaction** — Compliance Proxy Layer 2 captures ALL agent actions; Commons captures only what agents choose to post. Do they share storage? Do Commons channels need their own compliance capture? Working answer: Compliance Proxy operates at MCP transport layer; Commons operates at cognitive layer; both write to same audit store with `source` field distinguishing. Revisit during Compliance Proxy design.

7. **Individual Hebbian ↔ Commons Hebbian link accounting** — when an agent co-cites a pair that exists in both its own individual store AND a Commons channel, does the citation strengthen individual Hebbian, Commons Hebbian, or both? Working answer: both, because they are different epistemic objects answering different questions ("does my mind connect these" vs "do multiple minds connect these"). Confirm with spike data.

## Next Actions

1. **This document → Phill review and iteration** (now, while thinking is hot)
2. **Session 14a feasibility spike** — Thu/Fri Apr 16-17 (or Fri/Sat if Thu substrate is post-taxes-day low). Heterogeneous triple (flow-Claude + Codex + Gemini), toy Commons, observe cross-agent co-citation. Single-day scope. Answers the load-bearing empirical question: does the heterogeneity constraint produce meaningful cross-agent Hebbian structure in practice, and at what threshold?
3. **Session 14b architecture spec** — after spike data lands. Full architecture: storage layout, thin-client library interface, MCP/CLI adapter layer, privacy enforcement, threshold tuning, mode configuration format.
4. **Session 14c Commons MVP build** — after spec. Scope: episodic blackboard + cross-validation Hebbian + channel privacy + thin-client library. NO shared working continuity yet (optional, gate on demand). NO compliance proxy (separate session). NO hive-mode toggle (add after Mixed mode proven).
5. **Update `project_memory/projectbrief.md` and `next_steps.md`** to reflect Session 12 repositioning and Session 14 addition, after this document stabilizes.
6. **Concurrent workstream:** Session 10 identity experiment (now unblocked by paper ship) runs in parallel on Docker testbed. Different cognitive mode (measurement/analysis), different attention windows, composes cleanly with Commons architecture work.
7. **Second harness paper** — working title: "Coordinated Collectives of Harnessed Agents: cross-validation, anti-inbreeding, and sovereignty at the coordination layer." Write when Session 14c + Session 10 data justifies it. Don't rush; empirical evidence first. Flow's bilateral asymmetry Night 1 data + Meta-Agent Principle are already theoretical seeds.

---

# POST-SPIKE REFRAMED SCOPE (Apr 14 evening)

Everything above is the original afternoon scoping. This section is the reframed version Session 14b should build from. Session 14a.1 + 14a.2 spike data is the source of truth for every claim here; see `~/Documents/anneal-memory/spike/commons_session14a/COMPARATIVE_VERDICT.md` for raw evidence.

## What the data actually said

Two runs, three model families (complement/codex/gemini), 60 total episodes. 14a.1 = homogeneous 30-episode corpus from the anneal-memory engineering arc. 14a.2 = heterogeneous 30-episode corpus from 8 deliberately distinct arcs, interleaved.

| Metric | 14a.1 | 14a.2 |
|---|---:|---:|
| N≥2 yield | 40% | 46% |
| N=3 pairs | 1 | 7 |
| Truly-cross-arc N≥2 (hidden-structure test) | N/A | **0 of 4** |

The counterintuitive result: N=3 convergence INCREASED on the harder corpus (7 pairs vs 1). Not because agents found hidden cross-topical structure — because interleaving distinct arcs made intra-arc obviousness MORE salient. The 4 pairs that crossed arc boundaries were 3 arc-labeling artifacts (the mechanism correctly detecting that episodes were mis-tagged and pulling them back to their true content arcs) + 1 surface keyword match. **Zero genuinely hidden cross-topical connections.**

## Reframed core claim

**Original (Apr 14 afternoon — not supported by data):**
> Commons Hebbian cross-validation via multi-agent co-citation finds hidden structure that individual priors miss. The anti-inbreeding immune system generalizes to collective at the graph mechanics layer because multiple independent minds converge on non-obvious connections.

**Reframed (Apr 14 evening — supported by data):**
> Commons co-citation produces three distinct useful effects: **(1) redundancy-gated graduation** (a structural filter requiring N≥2 independent agents before a pair/pattern can graduate into Commons structure, making individual confabulation unable to graduate without corroboration); **(2) metadata/tag correction** (co-citation patterns retroactively detect when episodes were routed into misleading arcs at recording time, and pull them back to their true content groupings — a use case no individual agent can perform because individual agents cannot see each other's citation patterns); **(3) pattern-level aggregation across heterogeneous priors** (different model families produce genuinely divergent top-level pattern abstractions from identical corpora, and Commons can collect, intersect, and surface those divergences — the genuine heterogeneity payoff lives at the pattern layer, not the citation-pair layer).**

## What Commons still IS (reframed)

1. **A redundancy gate for pattern/link graduation.** N≥2 independent co-citation required before a pair or pattern enters Commons structure. Structural anti-hallucination filter. Unskippable at the graph mechanics layer. Enforceable, auditable, honest. Keep this.

2. **A metadata-correction mechanism.** When agents record episodes with routing-oriented tags rather than content-oriented tags (e.g., tagging an RLHF-gatekeeping episode as "anvil" because it happened during Anvil Session 1), co-citation from multiple independent readers pulls the episode back to its true content grouping. Individual agents cannot do this — they have no view of each other's citation patterns. Commons can. **This was not a planned feature; it emerged from the spike data.** It is genuinely novel, no competitor has it, and it is the most surprising finding of the two runs.

3. **A pattern-level aggregation substrate.** Different model families produce different high-order abstractions from the same corpus. On 14a.1: complement saw "the paper's thesis is self-enacting in its own research process" (no other agent); gemini saw "AGI as harness-emergent" (no other agent); codex stayed inside engineering mechanics. On 14a.2: complement saw "accumulation without grounding corrupts at every scale"; gemini saw "engagement_overrides_energy as velocity engine"; codex saw "harness mainstreaming as real-time product validation." These are different meta-claims about identical input — and Commons' genuinely novel value lives HERE, at the pattern layer, not at the episode-pair citation layer.

## What Commons is NOT (reframed)

1. **Not a mechanism that finds hidden cross-topical structure via independent priors.** Two runs, 60 episodes, zero hidden-structure discoveries. Do not build architecture on this being the load-bearing value prop.

2. **Not an empirically-validated anti-inbreeding filter in the "prevents false pattern graduation via cross-validation" sense.** The test couldn't validate this because individual compressors mostly produced accurate compressions — there weren't many false patterns to filter. The mechanism MIGHT work for this at scale, but the spike doesn't prove it and Session 14b should not bet architecture on unproven claims.

## Reframed MVP scope

**Drop from MVP:** hive mode, shared working continuity, message passing coordination primitives, topology mode configuration (Constellation/Hive/Mixed), heterogeneity enforcement as a hard-refusal feature, cross-channel federation.

**Ship as MVP:**

1. **Shared episodic channel** (SQLite, channel-scoped, stdlib-only, thin-client library for existing anneal-memory instances to link against). Same architectural posture as core anneal-memory. `commons.post(channel, episode)` + `commons.fetch(channel, ...)`.

2. **Redundancy-gated link formation.** When an agent cites episode pair (A,B) during its own individual compression AND both A and B live in a Commons channel the agent participates in, register the vote. Link forms when N≥2 distinct agents independently vote. Link strength increments with additional votes. Decay on the same cadence as individual Hebbian.

3. **Pattern-layer aggregation.** Commons accepts top-pattern submissions from each participating agent during compression. When multiple agents contribute patterns on the same corpus slice, Commons exposes the aggregate: which patterns appeared across N≥2 agents (converged meta-patterns), which appeared only once (unique priors — interesting as divergence signal), and which pairs of patterns can be meaningfully intersected. This is the first-class Commons primitive the data supports most strongly.

4. **Metadata audit primitive.** `commons.audit_tags(channel, window)` — for any episode in the channel, compare the arc/tag the episode was RECORDED with against the arc/tag its Commons co-citation patterns suggest. Surface mismatches. This is the retroactive tag-correction use case that emerged from 14a.2 data.

5. **Privacy = channel membership.** Episodes in Commons are tagged with channel membership; agents can only read channels they subscribe to; no cross-channel bleed. Unchanged from original scope.

## Reframed open questions

1. **Does the metadata-correction effect replicate on a third corpus?** 14a.2 found 3 of 4 cross-arc hits were arc-labeling artifacts. Is that the stable behavior of the mechanism, or an artifact of this specific corpus? Session 14a.3 answers this with a corpus specifically designed to test arc-correction.

2. **At what N does pattern-layer aggregation produce actionable signal?** The spike ran with N=3 agents. Two patterns agreed across 2+ agents reliably; unique patterns surfaced from single agents. Is N=3 enough, or does the pattern-aggregation primitive need N=5+ to produce stable "here's what everyone saw, here's what only one saw" signal?

3. **Can the redundancy gate be tuned per use case?** N=2 for permissive link formation, N=3 for strict link formation, N=majority for consensus patterns? Channel-level configuration?

4. **How does individual Hebbian compose with Commons Hebbian now?** Original answer was "both independently." Reframed answer: the same, but Commons is NO LONGER positioned as a hidden-structure surface — it is a redundancy gate plus metadata corrector plus pattern aggregator. Individual Hebbian continues to be the primary structural substrate; Commons is a supplementary filter and audit layer.

## Reframed sequencing

1. **Session 14a.3** (optional, fast) — one more spike run if the metadata-correction reframe needs a second data point before committing architecture. ~30 min using existing scaffolding. Corpus deliberately constructed to test the arc-correction effect.

2. **Session 14b architecture spec** — reframed MVP scope only. Storage layout, thin-client interface, MCP/CLI adapter surface, redundancy gate math, pattern aggregation primitive, metadata audit primitive, privacy enforcement. NO hive/continuity/coordination scope in v1.

3. **Session 14c Commons MVP build** — reframed scope. Smaller than the original plan. Closer to what the data supports.

4. **Session 14d empirical validation** — run the MVP against a real corpus for a week. Does the metadata-correction primitive surface real mis-tags? Does pattern-layer aggregation produce distinguishable "converged / divergent / unique" buckets? If yes, iterate toward v2. If no, reframe again based on fresh data.

5. **Second harness paper** — still possible, but the working title changes. Original: "Coordinated Collectives of Harnessed Agents: cross-validation, anti-inbreeding, and sovereignty at the coordination layer." Reframed working title: "Commons: Redundancy Gates, Metadata Correction, and Pattern Aggregation in Multi-Agent Memory Architectures." Less sexy, more honest, less hostage-to-claims-unsupported-by-data.

## The honest bottom line

The mechanism works. The plumbing is sound. The original framing was half-right (redundancy-gated graduation is real) and half-unsupported (hidden-structure discovery via independent priors did not appear at any scale the spike could test). The reframe above preserves what the data supports and drops what it doesn't. Commons remains worth building — it just builds a smaller, more honest, more novel product than the afternoon scoping imagined.

Session 14b architecture should start from the reframed scope in this section, not from the original scope above. The original is preserved for lineage and to make the delta visible, not as the live specification.

---

*Draft — Apr 14, 2026 afternoon. Reframed Apr 14, 2026 evening post-spike. Status: spike data is in; reframe is live; Session 14b planning should build from the reframed section.*
