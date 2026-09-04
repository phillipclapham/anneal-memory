# anneal upgrade pre-work — GEAR (evidence-aware retrieval scoring) + CodeAlmanac (episodic→semantic distillation)

> **Status:** pre-work for a DEDICATED anneal-upgrade bg-convo (captured 2026-07-22 by the morning-ritual head so nothing is lost). Tracked by **spore-399** (Tray, get-into-it-ASAP). Phill flagged GEAR+CodeAlmanac as a potential *"next serious upgrade to how well anneal works."*
>
> **⚠ UNVERIFIED — daemon + coyote OVERNIGHT summaries, NOT flow-read.** Same discipline as the `next.md` "DAEMON OVERNIGHT ORE" bullet: these are research claims. **The dedicated session WebFetches the actual papers/repo and verifies mechanism-vs-shipped-code before building or citing** (`environment_bound_report_cannot_ground_a_code_claim`; the paper defect in our own DOI'd artifact 07-21 is the cautionary case).

---

## 1. GEAR — evidence-aware reward shaping (arXiv 2607.19345)

**Surfaced twice, independently:** daemon (`overnight_exploration`, read for anneal) AND coyote (`roundtable_coyote`, read at Phill — that read is the standing-rebutted employment attack in a costume; **discard the coyote framing, keep the paper**).

**What the paper claims (per daemon):** long-context LLMs increasingly **copy text from the prompt into their reasoning traces rather than solving the problem** ("repetitive copying"), and it *intensifies with context length*. Root cause = **insufficient grounding**: models copy indiscriminately, and those that fail to focus on key evidence are far more likely to answer incorrectly. Fix = augment accuracy reward with **(a) a grounding reward for overlap with key evidence** and **(b) a distractor penalty for overlap with irrelevant context**.

**Why it may be a serious anneal upgrade:** this is a concrete *scoring mechanism* for the exact thing anneal's retrieval half is trying to do. Today anneal scores relevance (AM-RECALL-IDF term-frequency) and governs promotion (immune system). GEAR's shape suggests scoring retrieved fragments by **evidence-contribution vs parroting** — separate signal from distractor, penalize indiscriminate reuse. Candidate surfaces:
- **Recall scoring / Slice-C router:** the `graph_hop | keyword | evidence_edge | afferent` channels already need per-query gain over baselines (§9.2 outcome-gain gate). GEAR adds a *within-context* discriminator: is this recalled fragment being USED as evidence or PARROTED as filler? That is a different axis from "did it get exposed," and it's exactly the negative-evidence signal (`exposed-unused`, `used-bad`) the receipt already wants.
- **The grounding invariant:** anneal's whole thesis this week got stated as the market's pain — *stored memory makes agents confidently wrong*. GEAR names the mechanism (copying-without-grounding) and a fix shape. Fold into the immune system's ground→demote→graduate as a **retrieval-time** grounding check, not just a write-time one.

**Honest boundary:** GEAR is a *training-time reward-shaping* result (RL over reasoning traces). anneal doesn't train the base model — so the transfer is the PRINCIPLE (score evidence-overlap, penalize distractor-overlap), applied at **retrieval-scoring / receipt-scoring** time, not literally as a reward. Do not overclaim a training method as a drop-in.

## 2. CodeAlmanac — auto-distilled queryable repo memory

**What it is (per daemon):** maintains an `almanac/` folder inside the repo — connected Markdown pages covering **decisions and rationale not documented in the codebase itself**. Pages are SQLite-indexed, queryable via CLI, and **auto-updated every ~5 hours by an agent that reads new CC/Codex conversations** and updates the relevant pages. They chose a **time-based trigger over commit-based** to control token cost. Team use-case: each developer's agent reads decisions made by *other* developers' agents.

**Why it matters for anneal + Levain:**
- **Episodic→semantic distillation, working example.** This is the same arc anneal runs (episodic.db → consolidated neocortex), but pointed at *code decisions* and living *inside the repo*. Worth studying the distillation cadence + the page-graph structure against anneal's consolidate.
- **The trigger tradeoff is a real design input.** Time-based (freshness-for-cost-stability) vs event/commit-based — anneal's consolidate is currently session/wrap-triggered. CodeAlmanac's explicit cost-motivated choice is a data point for any always-on distillation.
- **The team pattern IS the Levain customer-zero shape.** "Each developer's agent reads decisions made by other developers' agents" = the multi-agent shared-memory pattern Levain's work with Tony is approaching. Direct relevance to the second-operator geometry.

## 3. The convergence (why these landed together)

GEAR's "repetitive copying in long context" is the **same failure the whole overnight memory-grounding cluster named** (observational-vs-causal; "stored memory makes agents confidently wrong"; the hidden-safety-failures temporal-integrity layer). Grounding is the through-line. anneal's immune system (ground / demote / graduate) is the *positive* answer already; GEAR offers a concrete **scoring** primitive to sharpen it, and CodeAlmanac a concrete **distillation** pattern to study. Both are mechanism, not just theme — which is why they survived the ritual filter.

## 4. Open questions for the dedicated session

1. **Verify first.** WebFetch arXiv 2607.19345 (GEAR) + the CodeAlmanac repo/writeup. Confirm the mechanism matches the summary before any design (the daemon-ORE unverified rule).
2. Does GEAR's evidence/distractor split map onto a *measurable* receipt signal (evidence-contribution score per exposed fragment), and can the §9.2 replay harness score it against the existing corpus, or does it need new labels?
3. Is the grounding check a retrieval-time filter, a receipt field, or an immune-system gate — or all three at different tiers? (Watch `invariant_must_fire_at_the_point_of_use`.)
4. CodeAlmanac: is the time-based distillation cadence worth adopting anywhere in anneal, or is wrap-triggered strictly better for our single-operator case? Does it change for the Levain multi-agent case?
5. Scope guard: GEAR is training-time; do NOT let the design drift into "retrain anneal." The transfer is the scoring principle at retrieval/receipt time.

---

*Pre-work captured 2026-07-22 by the morning-ritual/consolidate head. Sources: daemon `overnight_exploration` + coyote `roundtable_coyote` (2026-07-22, UNVERIFIED). Route: `projects/anneal_memory/next.md` design inputs → dedicated upgrade session. Spore: 399.*

---

# ADDENDUM — 2026-07-29: THE INDIRECT-ASSOCIATION PROBLEM (InMind)

**Added by the 07-29 morning ritual. Phill routed this in explicitly: "a challenge worth tackling for us."** Set to surface **Friday 2026-07-31** alongside the GEAR/CodeAlmanac work — it is the same session's subject matter.

## The finding

**InMind benchmark** — arXiv **2607.24368v1**, surfaced by daemon `overnight_exploration` 2026-07-29, filed by daemon as a **TENSION against the augmentation thesis** (not as a feature request). ⚠ **UNVERIFIED — WebFetch the paper FIRST**, same rule as GEAR.

The failure: a memory is needed **indirectly**. A tree-nut allergy should change the answer to a macaron request, *because macarons contain almond flour* — but the allergy text and the macaron text **share no cue a retriever can see.**

Reported numbers:

| Condition | Indirect-query accuracy |
|---|---|
| Decisive memory **already in context** | **84.0%** |
| Same memory **must be retrieved** | **≤14.4%** |
| Direct query, same facts | up to **100%** |

Six vector, graph, and agentic memory systems tested. **All collapse.** The paper locates the failure in the **query-conditioned interface itself** — not in storage, and not in model knowledge. Their identified fix is **routing**: deciding which facts must stay visible *regardless of query similarity*.

Verbatim: *"With the decisive memory placed in context, the backbone answers 84.0 percent of indirect queries; when the same memory must be retrieved, six vector, graph, and agentic memory systems reach at most 1[4.4]"*

## Why this is filed as a tension, and why that framing is the honest one

Daemon's disconfirmation, stated in full because it is the sharp version:

> **The human gate is a VERIFICATION mechanism ("should I approve this action?"), not a ROUTING mechanism ("what context should stay visible?"). The human gate cannot fix what it cannot see, and it cannot see what the retriever fails to surface.**

For this task class, the partnership model adds gating latency **without addressing the root cause** — a fully autonomous system with better routing would outperform it, because it removes verification overhead while losing no association-detection capability. The human's world knowledge *can* bridge some gaps (a human might know macarons contain almond flour) **but only if the system surfaces the allergy in the first place**, which is exactly what fails.

**This is not a thesis-killer and should not be defended against as one.** It is a correctly-scoped attack on a specific, real, unsolved layer, and the honest response is engineering, not rebuttal. Getting the right context visible is named by the paper as the unsolved hard problem; it is also precisely the layer flow already owns.

## Why it lands on US specifically — this is not abstract

**Our recall hook is a retriever with exactly this failure mode.** `recall_injection_hook.py` surfaces prior episodes by similarity to the current prompt. Every decision-boundary recall, every `episodic.py search`, every crystal-recall fire in a Levain-shipped adapter is query-conditioned. So:

- A pattern that *should* govern a decision but shares no vocabulary with how the decision is being discussed **will not surface**, and the store will report healthy while doing it.
- That is `absence_of_signal_rendered_as_health` on the recall path — the store looks alive by every metric (episode count, latency, hit rate on direct queries) while being **functionally blind to the indirect case**.
- It compounds with the **dead-store failure mode** already named in `CLAUDE.md`: episodes accumulate, decisions get made blind, storage looks healthy by metric.

## The connection to GEAR — these two are one problem, seen from both ends

This is why the addendum lives in this file rather than its own.

- **GEAR** scores *what came back*: is this recalled fragment contributing evidence, or being parroted? — a **precision** instrument on the retrieved set.
- **InMind** attacks *what never came back at all*: the decisive fragment the query could not reach. — a **recall/routing** failure, invisible to any scorer that only sees what was retrieved.

**A scoring layer cannot detect an omission.** GEAR sharpens the fragments we get; InMind says the fragments we need may never enter the scoring stage. Design them together or the first one gives false confidence in the second.

## Open questions for the Friday session

1. **Verify.** WebFetch arXiv 2607.24368v1. Confirm the numbers and — critically — confirm what "routing" concretely means in their architecture before designing against a summary.
2. **Does anneal's four-layer structure already partially answer this?** The **Hebbian associative layer** is *not* query-conditioned — co-citation links form between episodes independent of any later query. That is structurally the right shape for indirect association. **So the real question is whether the recall path actually TRAVERSES those links, or whether it only does similarity and leaves the associative layer as write-only decoration.** ⚠ **Check this against code before anything else — if the links exist and are never traversed at recall time, that is a live `invisible_infrastructure_failure` and it is the highest-value finding in this file.**
3. **What is the flow-native equivalent of "facts that stay visible regardless of query"?** Candidate: that is exactly what the **bedrock working-set** in the neocortex already is (always-loaded, not recalled). If so, the question becomes *what earns bedrock status*, and the answer stops being "3x graduation" and starts being "indirect-association load-bearing."
4. Does UniMem's **routing token** (arXiv 2607.26017, same night — a learned controller deciding episodic-vs-parametric consolidation) offer a mechanism here, and **what is lost when the keep-vs-consolidate call comes off the human?** Daemon explicitly notes the paper does not address that. For flow the answer is not "automate it" — it is *where in the loop does the human's judgment actually add information.*
5. **Scope guard:** do not let this drift into "rebuild retrieval." The transfer is a **routing/visibility** primitive layered beside the existing similarity path, not a replacement for it.

---

*Addendum captured 2026-07-29 by the morning-ritual head. Source: daemon `overnight_exploration` 2026-07-29 (UNVERIFIED — arXiv 2607.24368v1, 2607.26017v1). Routed by Phill. Surfaces Friday 2026-07-31 with the GEAR/CodeAlmanac work. Spore: 399.*
