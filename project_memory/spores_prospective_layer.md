# The Prospective-Intention Layer — `spores` (philosophy + semantics)

> The canonical philosophy/semantics doc for flow's **prospective** memory layer.
> Written 2026-06-01 so that when this generalizes to an anneal/Levain primitive,
> the generalization inherits the *why*, not just the JSON schema. Implementation:
> `scripts/spores.py` + `state/spores.json`. Parent thread: `flow_migration_scoping.md`
> Phase-5 (the re-scoped back-half). Decision episode: `flow-20260601-091312` +
> this session's build episode.

---

## 1. The thesis — `tasks ≠ memory`, three temporal layers

The flow→anneal migration exposed a conflation that `continuity.md` had carried for
months: it jammed **three different temporal systems** into one always-loaded file.
Sorted by temporal orientation + lifecycle:

| Layer | Orientation | Lifecycle | Owner |
|---|---|---|---|
| **Memory** | retrospective ("what happened / who we are") | accrete → compress → graduate; **never completes** | anneal (episodic + neocortex + Hebbian + limbic) |
| **Prospective-Intentions** | prospective ("what's open / wants future attention") | open → grow → **resolve** (descend or ascend) | **spores** (this layer) |
| **Top of Mind** | present ("what matters right now") | **generated**, never stored | computed downstream (phill_surface / session_init) |

**The discriminator is lifecycle, not topic.** Memory never completes — it's the
record. A prospective intention MUST complete — it's an open loop. Salience is
neither stored nor open; it's *computed* from the other two crossed with recency.
Bundling them forces one storage logic onto three incompatible temporal logics.
anneal already half-made the cut: `## Active Threads` is legitimate *memory* (coarse
work-identity that doesn't complete); the discrete dated **Action Items were the
prospective layer trapped in a memory file.** This layer frees them.

Cognitive-science backing: **prospective memory and declarative memory are distinct
systems.** We're not inventing a seam; we're cutting along one the architecture (and
the brain) already maintains (`decomposition_seam_belongs_at_an_existing_interface`).

## 2. The unit — a *spore*: a typed open cognitive loop

A spore is **something cognitively OPEN that wants future attention and must resolve.**
It carries a `type` naming *what kind of openness* it is:

- **task** — open *doing*. (meds pickup, restrict the API keys, schedule the vet panel)
- **question** — open *not-knowing*. (restore the domain or let it lapse? does WRAP_PROTOCOL need the recall step?)
- **thought** — open *idea*. (essay seeds, "what if" architecture, design-sketches like the corporate_mild runtime gate)

All three share **one lifecycle**. The type is a single field; it does not fork the
mechanism. Its payoff: questions and thoughts currently have **no home** in flow —
they leak into Top of Mind, the scratchpad, or continuity's Developing tail. The typed
spore layer gives them a real home with a real exit.

Empirical validation on the live data: `## Actions` Hot-item #1 (the corporate_mild
runtime-gate "DESIGN") is, under a task-only model, an awkward "task." Under this model
it is obviously a **thought-spore, growing, ascending toward a build.** The model fits
the data better than the costume the data was wearing.

## 3. The lifecycle

```
  PLANT ──▶ GROW (germination tiers) ──▶ RESOLVE
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                                 ▼
            DESCEND (compost)                                ASCEND (transmute)
            done / answered /                                → project / pattern /
            explored / dropped                                 episode / essay / thread
            = the self-clean                                 = the MEMBRANE into the
            (it closed and fell away)                          retrospective memory layer
                                                              (it grew into something permanent)
```

Per type:
- **task** → descend `done`/`dropped` | ascend `project` (became real work) / `thread` (became a standing commitment)
- **question** → descend `answered`/`mooted` | ascend `episode`/`pattern` (the answer became a recorded finding/principle)
- **thought** → descend `explored`/`dropped` | ascend `essay`/`pattern`/`project`

**Resolution is mandatory and recorded.** A resolved spore moves to `resolved[]` with a
`resolution = {direction, kind, ref, on (date), at (precise timestamp)}`. Descend records
*how it closed*; ascend records *what it became* (the `ref` — a project path, episode id,
pattern name). **Terminal `kind`s are enforced per type** (a `task` cannot descend
`answered`; a `question` cannot ascend `essay`) — the lifecycle is shared, the labels are
not. `composted` is the universal neglect-descent, valid for any type. The precise `at`
timestamp (not just the date) is what lets a wrap consume "what ascended *this session*".

## 4. Germination tiers — computed, never stored

The "grow" phase is made observable exactly the way Top of Mind is generated: by
computation from `seen`/`next`, never by a stored field. Mirrors flow's garden
convention (garden, not debt — tend what's alive, let the rest rest):

- **growing** — `seen` < 3 days ago — has momentum, don't interrupt
- **resting** — `seen` 3–7 days ago — mention gently, no pressure
- **dormant** — `seen` > 7 days ago **OR** on/after its `next:` date — surface as "still alive, or ready to compost?" (`next:` is a *surface-on-this-date* alarm; the day it arrives, it surfaces)
- **parked** — `tier == parked` — *deliberate* dormancy, distinct from neglect

A `next:` in the past forces dormant regardless of `seen` (it asked to be re-surfaced and
wasn't). The lifecycle **subsumes** the old staleness convention — we don't bolt
staleness on; it falls out of germination. `next:` is "put this back in my field of
vision," **not a deadline.**

## 5. Composition — why this is a layer, not a list

The two resolution directions are **edges into the rest of the system.** That is the
whole prize: the prospective layer is a *source and a sink* wired to memory and
projects, not a dead-end to-do file.

- **ascend → anneal (memory):** a question answered → a finding; a thought graduated →
  a pattern. The wrap's episodic-extraction step can consume **"what ascended this
  session"** as a first-class source — prospective feeds retrospective. This is where
  the two halves of memory finally talk to each other.
- **ascend → projects:** a task/thought that became real work points at `projects/X`.
- **descend → compost:** the garden's "ready to compost?" is just `descend`. Self-clean.
- **Top of Mind generation:** ToM is generated downstream from **hot+growing spores ×
  Active Threads × recent ships.** Because spores are typed, ToM now naturally surfaces
  open *questions* and live *thoughts*, not just a to-do list. `spores.py surface`
  exposes the seed contribution; the consumer (phill_surface / session_init) composes
  the full salience.
- **subsumes the garden tiers** (§4) and **homes the homeless** (essay seeds, `?`
  markers, "what if" ideas).
- **convergence to note, not force:** `state/agency_backlog.json` (Claude's
  explore/build/research/improve) is the *same primitive at a different scope* — spores
  at the agent's-own-curiosity scope. A later unification is plausible; don't force it.

## 6. Why "spores" — the name + the lineage

**Lineage:** this is the **Protocol Memory "Seeds"** model. PM (shipped Jan 2026)
typed its prospective items as task/question/thought and gave them exactly this
born→grow→descend/ascend lifecycle. The semantics are preserved verbatim; only the
**lexeme** changed.

**Why not "seed":** it collides with the **Levain identity-*seed*** (the genome that
boots an entity). Two meanings of one bare word in one ecosystem is precisely the drift
the disambiguation discipline fights. Collision audit of the full candidate set:

| candidate | verdict |
|---|---|
| seed | ✗ collides with Levain identity-seed |
| germ | ✗ "germ of an idea" is lovely, but "germs" + anneal's **immune system** reads wrong |
| pips | ✗ collides with python `pip` in a `scripts/` context |
| kernels | ~ great double-meaning ("kernel of an idea"), but OS-kernel / RAYGUN-OS noise |
| grains | ~ no hard collision, but diffuse |
| roots | ✗ roots = already-established → reads *retrospective*, wrong direction |
| intentions | ✗ only covers the *task* type; drops question + thought |
| **spores** | ✓ **chosen** |

**Why "spores" wins:** it's the only candidate that carries the *full* lifecycle with
zero ecosystem collision. A spore lies **dormant until conditions are right, then
germinates** (maps the dormancy tier exactly), and a spore that finds good ground
becomes a **new organism** (maps the ascend path: seed → new project/principle). Clean
of Levain-seed, pip, OS-kernel, and immune-germ.

## 7. The Levain generalization — the prospective-memory primitive

This is **the prospective half of partnership memory.** anneal owns the retrospective
layers (episodic + neocortex + Hebbian + limbic + immune); spores owns the prospective
layer; **ascend is the membrane between them.** Together they're the complete temporal
memory of a partnership entity:

```
   PROSPECTIVE (spores)            ──ascend──▶            RETROSPECTIVE (anneal)
   typed open loops                                       episodic / neocortex /
   task | question | thought                              Hebbian / limbic / immune
   resolve or compost                                     accrete / compress / graduate
                          ◀── generates ──┐
                                          │
                              TOP OF MIND (computed: spores × threads × recency)
```

**Levain ships both.** Every partnership-entity seed gets: anneal (retrospective) +
spores (prospective) + a generated-salience surface. The primitive is portable because
its definition is lifecycle-based, not flow-specific: *typed open cognitive loops with
descend/ascend resolution, germination computed from seen/next.* An ops entity (Argus)
may carry a thin/empty spore layer; a felt partnership entity (flow) carries a full one
— same shape as `## Understanding` being partnership-entity-only.

**Build order (flow-local first, then generalize):** prove the shape on flow → extract
the schema + verbs into an anneal/Levain module → ship as a Levain seed layer.
`sourdough_scoping`: commit the shape concretely, generalize from the working instance,
don't pre-abstract.

## 8. Scope — v1 vs later

- **v1 (now):** the store + CLI (`spores.py`), the typed model, germination, descend,
  and **ascend-records-a-pointer** (the actual anneal/project write stays a flow act).
- **v2 candidates:** ascend auto-writes the anneal episode; the wrap consumes
  ascended-this-session automatically; ToM judgment-driven (flow-in-the-loop salience)
  if the deterministic merge reads thin; unify with `agency_backlog`; the anneal/Levain
  module extraction; **the unified control plane (§10).**

## 10. Levain v2 — the unified self-hosted control plane (2026-06-01, Phill)

The visual/operator layer over the *whole* memory system, self-hosted (sovereignty:
you run it, you own the view). One web UI that surfaces **both halves of memory + the
control surfaces** in one place:

- **Prospective half (spores):** see your spores and their lifecycle, edit them
  visually — plant / touch / retier / descend / ascend by drag, watch germination
  tiers shift growing→resting→dormant, see the ascend edges fire into projects/anneal.
- **Retrospective half (anneal):** the **Hebbian association-graph visualization**
  (links forming + decaying) — this is exactly **AM-VIZ** already on the anneal
  roadmap (Alex De Groodt's idea, `projects/anneal_memory/next.md` "AM-VIZ"). Same
  surface. The two roadmap items are **one control plane**, not two features.
- **The already-planned control surfaces** fold in here too.

**Why this is the right v2 shape, not scope-creep:** the control plane is the
*operator interface to governance-not-trust at the memory layer.* The whole stack's
thesis is "govern the generative engine, don't trust it" — and you can't govern what
you can't see. A self-hosted UI that makes the prospective layer (open loops, where
they're heading) and the retrospective layer (what associated with what, what's
decaying) **legible and editable** is the cognitive-sovereignty stack's missing visual
organ. It composes: spores lifecycle + Hebbian graph + control surfaces are three views
of one owned memory. **Ships in Levain v2** as the control-pane layer over anneal +
spores. (Build-now: NO — flow-local CLI first, prove the data model; the pane is the
visual layer *over* a proven substrate.)

## 9. Implementation reference

- **Store:** `state/spores.json` → `{ "spores": [...], "resolved": [...], "schema_version": 1 }`
- **Script:** `scripts/spores.py` — mirrors the `agency_backlog.py` idiom (atomic
  tmp+fsync+rename write, `spore-NNN` ids counted across live+resolved).
- **Item:** `id · type · text · domain · tier · salience(0-3) · seen · next · created ·
  status · resolution · pointer · notes[]`. Germination is computed, never stored.
- **Verbs:** `add · list · touch · update · descend --kind · ascend --kind --ref · surface`
- **Consumer contract (the repoint target):** `phill_surface.read_continuity()` currently
  regex-parses `## Top of Mind` + `### Hot` + `### Warm` from `global/continuity.md`. The
  repoint swaps it to read `spores.py` (open spores by tier) + generate ToM. That repoint
  + migrating the genuinely-prospective `## Actions` items (NOT the standing-context
  dossiers or `[decided]` markers — those are memory) is the next step; archiving
  `continuity.md` is the true cutover that follows.
