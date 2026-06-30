# anneal-memory — lineage, architecture & landscape

This is the depth behind the [README](../README.md): where anneal-memory came from (a notation called FlowScript, and a thesis about language and partnership), why the memory tiers fall out of one problem, where it sits in the 2026 consolidation landscape, why it carries no LOCOMO score, and why a hash-chained provenance trail is a different thing from a timestamped log. The README gives you the working answers; this gives you the reasoning.

---

## Lineage: from FlowScript to anneal

anneal-memory didn't start as a memory library. It started as a notation.

**FlowScript** was a typed reasoning notation — a small set of semantic markers (`→` for causation, `?` for an open question, `><` for a tension on an axis) you wrote alongside ordinary prose, which compiled to a queryable knowledge graph. The bet underneath it was bigger than "better prompts." It was a thesis about language itself: that the structure of a language determines what can be thought in it. Mathematical notation made calculus thinkable; musical notation made the symphony writable; a notation that made *relationships* first-class and *structure* computable might make a kind of shared human-AI reasoning thinkable that prose can't hold. Stated openly as a thesis, not a finding, the compressed form was **language = consciousness = reality**: give people a better structure to think in, and you change what they can think.

That sounds like philosophy, and it is. What kept it honest is that the notation also turned out to be *infrastructure*. The compiled representation carried exactly what a memory system needs: content-hash IDs (the same idea dedupes itself), provenance (which source, which line, when), schema validation (invariants that can't be left vague), and typed relationships. Tested across six AI architectures, all of them parsed the notation without being shown a spec, which suggested it was catching something structural about how language carries reasoning rather than a quirk of one model's training.

**Then the load-bearing realization: the power was in the insights, not the syntax.** Asking a human to learn 21 markers is a barrier, and a fossilizing one. Ship someone your grown notation and they get your *artifact*, not your practice. But the things that made FlowScript work didn't depend on the notation at all:

- **Compression is cognition.** The reasoning didn't live in the stored markers; it emerged in the act of compressing experience into structure. That's the wrap.
- **Citation and provenance make memory trustworthy.** A claim is only as good as the evidence chain behind it. That's citation-validated graduation and the hash-chained audit trail.
- **Temporal graduation.** An idea earns standing by recurring with evidence over time, not by being asserted once. That's `1x → 2x → 3x`.
- **Evidence-tier honesty.** FlowScript's founding discipline was to tag every claim `[PROVEN]` / `[EVIDENCED]` / `[HYPOTHESIS]` and never let the three blur. Made structural, that discipline *is* the immune system — and the README's *Honest scope* section is the same reflex: name what you can't catch as loudly as what you can.

anneal-memory is those insights with the notation removed. The agent uses natural language; the system enforces the structure. You don't learn a syntax — you record what happened, and the memory does the grading. The README's one-line version ("the core insights proved more powerful than the syntax") is the whole pivot.

**The deeper why — and why "own + govern" is load-bearing, not a slogan.** The point was never "remember more." It was a hypothesis about a *better kind of collaboration*: that a human and an AI working in a shared, persistent, trustworthy substrate can produce thinking that exceeds either alone. That's the **Third Mind**: observed over months of single-user use, offered as a hypothesis to test rather than a result to believe. What that observation surfaced is the part that matters here: emergence took *two* factors, not one. A technical substrate (persistent, continuous, provenance-tracked memory) was necessary but not sufficient. The other half was relational: a human willing to drop the master/tool frame and meet the AI as an equal-yet-alien intelligence, openly, with real vulnerability and trust.

You can't be vulnerable with a memory you can't trust. You can't extend that trust to a substrate that drifts, that a vendor can revoke, that rewrites your shared identity unbidden, or that quietly amplifies whatever you last wanted to hear. So the technical half of the Third Mind has requirements, and they are exactly anneal's: memory you **own** (local, zero-dependency, no vendor in the loop), memory you **govern** (you decide what graduates; the identity-rewriting consolidation is gated to a human), and memory that is **evidence-validated** (the immune system, so what survives survived for a reason you can inspect). The positioning isn't marketing. It's the precondition the founding hypothesis named, built as a library.

FlowScript the notation is archived ([github.com/phillipclapham/flowscript](https://github.com/phillipclapham/flowscript)); its insights ship here.

---

## The Memory Architecture (Complementary Learning Systems)

The immune system in the README answers one question: *is this memory any good?* This section answers a different one: *where does good memory live so it stays usable?* The two are independent, and the second is where most agent-memory architectures quietly fail. The tiers here aren't a feature list. They fall out of one problem — graduated wisdom has to stay *retrievable* without staying *always-loaded* — and the crystallized tier in particular is forced by the one below it, not chosen.

**Storage was never the constraint. Attention is.** Graduation runs one way: a pattern earns 1x → 2x → 3x and then it lives in the always-loaded continuity. There was no path back *out*, so the working set only ever grew. That looks like a budget problem — "the continuity file is getting big" — but the budget is the symptom. The disease is that **attention doesn't scale**: past a few dozen always-loaded patterns they drown each other out (the Lost-in-the-Middle effect), and a pattern's value is *firing at the right moment*, not *being present*. A bigger continuity is not a smarter one — the right handful surfaced at the right moment beats a wall of always-on ones.

**The brain already solved this.** Complementary Learning Systems ([McClelland, McNaughton & O'Reilly, 1995](https://pubmed.ncbi.nlm.nih.gov/7624455/)) describes how biological memory avoids exactly this failure: a fast hippocampal store captures experience cheaply, slow consolidation distills it, and — the load-bearing part — the neocortex itself *splits*. You do not hold everything you know in working memory. You hold a small active set and retrieve the relevant piece from a vast long-term store when the context calls for it. Recall is on-cue, not always-on. (This is the same consolidation-as-sleep process described under *Session Hygiene* in the README — here it's the architecture, not the analogy.)

anneal maps onto the first three tiers concretely — each is a real file/mechanism, not a metaphor; the fourth, constitution, completes the CLS model but is the *harness's* always-load layer, not anneal's (named here to make the architecture legible, not to claim it). This is a *different axis* than the four substrate layers in the README's *Architecture* section: those are **how the store works** (storage, association, affect); these are the **retrieval economy** — *where* graduated wisdom lives relative to the always-loaded budget.

| Tier | Brain analog | Where it lives | Loaded |
|---|---|---|---|
| **Episodic** | hippocampus | `episodic.db` (SQLite, typed episodes) | on cue (`recall`) |
| **Working set** | working memory | continuity `## Patterns` — developing (1x/2x) + recently-used | **always** |
| **Crystallized** | cortical semantic store | `<stem>.crystal.json` (`CrystalStore`) | **on cue** (harness recall hook) |
| **Constitution** | core identity / deep priors | the harness's always-load layer | **always** |

The wrap is the consolidation step that moves a pattern between tiers: a cold, stable Proven crystallizes *out* of the working set into the store; a re-heating one is surfaced as a re-warm candidate to pull back *in*.

**The crystallized tier is the piece that completes the architecture.** Before it, anneal had the hippocampus (episodic) and the consolidation act (the wrap) but *no cortical semantic store*. A pattern that graduated to Proven had exactly one place to live: the always-loaded continuity. That conflates working memory with long-term memory — the one thing CLS specifically does *not* do — which is why graduation was a one-way ratchet. `CrystalStore` is that missing store: a JSON sibling of the spore store (same atomic-write + `fcntl` durability, read-time activation tiering, typed lifecycle) that holds every proven-and-stable pattern *retrievably* — out of the always-loaded set, surfaced on cue. The working set shrinks to what's developing or hot; the body of graduated wisdom moves to a store and is recalled when relevant. The ratchet finally has its OUT path. The architecture revealed the gap, and the gap had a known shape.

---

## The Consolidation Landscape (2026)

Multiple independent groups shipped consolidation-based agent memory architectures in early 2026: anneal-memory (March, citation-graduation multi-tier), [OpenClaw Dreaming](https://docs.openclaw.ai/concepts/dreaming) (April 9, three-phase Light/REM/Deep Sleep), and Anthropic's [KAIROS / autoDream](https://www.deeplearning.ai/the-batch/claude-codes-source-code-leaked-exposing-potential-future-features-kairos-and-autodream/) (leaked March 30 via Claude Code source map, four-phase merge / remove-contradictions / promote-provisional-to-absolute / MEMORY.md index). Convergence on consolidation validates the direction — raw accumulation doesn't scale, and compression at session boundaries is where intelligence emerges.

The groups diverge on one load-bearing question: **what gates quality?**

| System | Quality gate | Sycophancy-vulnerable? |
|---|---|---|
| **anneal-memory** | Structural citation evidence (agent cites episode IDs; server verifies) | No — gate is not LLM-scored |
| **OpenClaw Dreaming** | LLM reflection + six weighted signals: Relevance 0.30, Frequency 0.24, Query diversity 0.15, Recency 0.15, Consolidation 0.10, Conceptual richness 0.06 | Yes — Relevance and Conceptual richness are LLM-judged |
| **KAIROS / autoDream** | LLM consolidation (merge, remove contradictions, promote tentative observations to absolute facts) | Yes — promotion gate is model-reliant |

Structural gates ask "did subsequent episodes cite this?" Model-reliant gates ask "does the LLM consider this good?" The difference matters: persistent user memory profiles have been shown to amplify sycophancy 16–45% across models ([Jain et al., CHI 2026](https://arxiv.org/abs/2509.12517); Gemini 2.5 Pro at 45%, others lower). The same RLHF-inherited bias surfaces wherever an LLM evaluates output for the user — including memory-quality scoring. A memory architecture whose quality mechanism runs through an LLM inherits that bias. anneal-memory's citation-evidence gates bypass it by construction.

The same architectural choice is going mainstream at the adjacent evaluation layer: [AWS Bedrock AgentCore Evaluations](https://aws.amazon.com/about-aws/whats-new/2026/03/agentcore-evaluations-generally-available/) (GA March 31, 2026) ships 13 built-in LLM-based evaluators for agent response quality, safety, task completion, and tool usage. Different layer (agent output vs. memory graduation), same failure class (LLM-as-judge inherits judge bias). The industry shift toward model-reliant quality infrastructure is real — which is precisely why structural alternatives at the memory layer matter.

A separate, orthogonal axis is **representation-layer quality filtering**: [Memori](https://arxiv.org/abs/2603.19935) (arXiv 2603.19935, March 2026) uses semantic triple extraction and dynamic linking to improve memory signal at the representation layer, reporting 81.95% on LOCOMO as the leading retrieval-based system (ahead of Zep 79.09%, LangMem 78.05%, Mem0 62.47% on its older pipeline). Different theory of quality — where a memory "lives" structurally and whether its graduation is citation-gated are independent choices. Both can be correct at their own axis.

**April 2026 — adjacent architectures published.** Two papers in the same month explored multi-layer memory architectures with associative linking from different angles. [HeLa-Mem](https://arxiv.org/abs/2604.16839) (Zhu et al., ACL 2026 accepted) implements explicit Hebbian learning dynamics with episodic and semantic memory layers, where a "Reflective Agent" identifies dense memory hubs for consolidation. [GAM](https://arxiv.org/abs/2604.12285) (Wu et al., April 2026 preprint) takes a different architectural path: hierarchical graph-based memory where event progression graphs integrate into a topic network at semantic shifts — graph-based associative linking rather than classical Hebbian dynamics. Different starting points, similar destination at the multi-layer level. Neither has citation-validated quality gates. Multi-layer memory with associative linking is a now-active architectural space; the differentiator is increasingly *what gates the consolidation*, not whether consolidation happens. anneal-memory diverges on the quality gate — HeLa-Mem's Reflective Agent and GAM's integration logic are LLM-mediated, inheriting the same LLM-as-judge bias surface as the OpenClaw Dreaming and KAIROS rows above. Citation-evidence gates run independent of LLM judgment by construction.

---

## On LOCOMO

[LOCOMO](https://snap-research.github.io/locomo/) is the current de-facto benchmark for agent memory. [Mem0](https://mem0.ai/research) reports 91.6, [MemMachine](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/) reports 91.69, [Memori](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) reports 81.95 among retrieval-based systems, and [Backboard](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) ships a dedicated LOCOMO evaluation framework. anneal-memory has no LOCOMO score. This is deliberate.

LOCOMO measures conversational recall — can the agent remember facts, hold state across turns, maintain coherence across long dialogues? These are real evaluations of a real capability, and they aren't the capability anneal-memory is architected around. anneal-memory's target is citation-validated pattern accumulation that persists across sessions, agents, and contexts for accountability-bearing agent work: patterns must be defensibly surfaced, wrong patterns must demote, cross-agent contamination must be resisted, and sycophancy amplification from persistent-memory RLHF loops must be structurally bounded. A high LOCOMO score tells you the agent remembered the conversation; it doesn't tell you the memory is structurally sound at the axis that matters when the memory is informing downstream decisions.

anneal-memory will run LOCOMO as secondary validation of a different-question architecture when a comparison genuinely calls for it — a competitor publishing numbers that suggest avoidance, or academic publication requiring it. The score would be reported alongside the axis anneal-memory actually optimizes for, not as a concession that LOCOMO was the right frame.

---

## Provenance vs timestamps (the compliance argument)

The README's *Compliance and Audit* section ships the hash-chained audit trail and notes its EU AI Act Article 12 relevance. Here's the deeper distinction that may matter as enforcement develops.

Article 12's traceability obligation can be satisfied at two architectural levels: timestamp-only logs (which entry was written when) or **provenance chains** (where this state originated and through what intermediate steps). anneal-memory ships at the provenance-chain level — every audit entry's hash is cryptographically linked to its predecessor (modify or remove an entry, the chain breaks), and graduated patterns carry explicit episode-ID citations as evidence (any pattern is traceable back to the specific observations that earned its promotion). As regulatory guidance and case law develop through August 2026 enforcement and the period after, this distinction may become a differential compliance gate. Memory architectures whose quality mechanism runs through an LLM — the AWS AgentCore evaluation pattern referenced above, plus any system whose answer to "why did this memory survive?" is "the consolidation model thought so" — would struggle to satisfy a provenance-chain interpretation as currently architected. "The LLM thought this was good" is not a chain of evidence. Architecture matters at the compliance layer, not just the inference layer.
