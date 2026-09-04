# anneal-memory — next

> ⚖ **MOVED REPO-SIDE 2026-09-04** (Phill: *"yes, let's def move anneal/solitaire project memories then please."*).
> This memory lived at `~/Briefcase/flow/projects/anneal_memory/` until today. anneal is a published
> package with a real external user, which makes it **Class A**, and Class A keeps `project_memory/`
> in the repo — matching blackjack and video-poker. It was flow-side purely by history.
> `brief.md` → `projectbrief.md` and `next.md` → `next_steps.md` to match the Class-A convention.
> Git history was NOT carried across the repo boundary (a cross-repo move cannot); flow's history
> retains it up to this commit.

> ### 🔬 DIOGENES — NEWEST: `diogenes_20260904.md` · **STILL OPEN: 12**
> 21 episode(s) — HIGH 2 · LOW 3 · MEDIUM 9 — 14 of 21 episode(s) carry a severity; the other 7 are COVERAGE 2 · COVERAGE-OPEN 2 · SELF 1 · STILL OPEN 2. Routed UNTRIAGED by `route_diogenes.py`; the count above is Diogenes' own slot, not the ritual's. ▶ 8 FILED TONIGHT (measured at HEAD this window) · 4 CARRIED. A carried item was NOT necessarily re-checked this window — only the report's own still-open slot says which ones were. Do not read the total as a verified defect count.
> ▶ 21 human commit(s) in the last 24h — the count could move in either direction this window.
> ⚡ **1 finding(s) carry `[prescription: run]`** — candidates for `seat_run.py`, but only with an executable acceptance test.
> *(Counted by each finding's OWN trailing tag — a quoted tag is not a verdict — and a tag withdrawn by a later SELF-CORRECTION does not count at all. If this number moved while the report did not, that rule changed: see flow `scripts/prescription.py`.)*
> ⚙ COORDINATES: 4/11 confirmed at HEAD 34728ac (1 carried no quote, unchecked). ⚠ 7 could NOT be confirmed — look, do not assume the reviewer was wrong (a remediated finding reads the same as a bad coordinate): anneal_memory/cli.py:275 -> quoted text is at :291 · pyproject.toml:53 (1 of 5 fragments matched) · tests/test_integrity.py:783 (2 of 9 fragments matched) · anneal_memory/cli.py:485 (none of 9 quoted fragment(s) found) · tests/test_audit.py:1686 (none of 6 quoted fragment(s) found) · anneal_memory/store.py:71 (none of 10 quoted fragment(s) found)
> *(Pointer written 2026-09-04 by route_diogenes.py. `spore-473`: a routed report with no reader is a disposal chute.)*

> ⬇ **TRIAGE BELOW THIS LINE — the block above is a DISPOSABLE SPAN.** `route_diogenes.py`
> regenerates that block every night, so anything written inside it is deleted by the next
> run. This line and everything beneath it are never touched by that script. Put the verdict,
> the fix, the refutation and the date here. *(Written once; `spore-473` — a routed report
> with no reader is a disposal chute, and a reader whose answer is deleted is the same chute
> one step later.)*

## ▶▶ PICKUP 2026-09-05 — THE 09-04 SECOND-REVIEW PILE IS CLEARED. ONE ITEM DELIBERATELY LEFT.

**Seat 0904+11, afternoon of 2026-09-04.** Opened specifically for the twelve findings Diogenes
filed against this repo in its SECOND review of the day (12:31–12:49) — six new, six carried.
Eleven closed, one deliberately not.

▶ **RE-DERIVE STATE, DO NOT READ IT FROM HERE** — every number below was true at close and none of
them can stay true on their own:
```
push state   git ls-remote origin main   vs   git rev-parse HEAD     (equal at close)
tree         git status --short                                      (empty at close)
tests        .venv/bin/python -m pytest -q -p no:cacheprovider        (1832 at close, from 1822)
types        .venv/bin/python -m mypy anneal_memory                   (clean)
lint         .venv/bin/python -m ruff check .                         (63, unchanged all session)
findings     project_memory/diogenes_20260904.md — its OWN still-open slot, newest wins
```
⚠ The last line matters most: this file said "all six findings closed" on 09-04 and was falsified
**by a second review landing**, not by the code changing. An answer can go stale without anything
touching the repo (`spore-764`).

### ⛔ THE SHAPE, BECAUSE IT DECIDES HOW TO READ THE REST
Four of the six new findings were INSTRUMENTS THAT COULD NOT SEE THEIR OWN SUBJECT — a counter on
the one transport where it is structurally always zero, a guard that greps source for a defect
source cannot show, a safety claim one backtick from vacuous, and an orientation file whose every
number was wrong. Not a coincidence: it is what is left after three nights of the prose tail
cleared out.

### ▶ THE HIGH: THE COUNTER REACHED EVERY TRANSPORT AND WAS STILL ZERO ON THE ONE THAT MATTERS
`status().audit_write_failures` was added so a swallowed audit write would be POLLABLE, routed to
all three transports the day before — and on the CLI it could never report anything but 0. It was
a plain instance attribute; a CLI run is a one-shot process whose `status` mutates nothing, so the
only process that could have counted a failure was already gone. **Reproduced before fixing:** two
episodes committed with their audit writes refused, `status --json` printing `"entry_count": 0`
beside `"write_failures": 0` two lines apart, on a store that had genuinely lost both.

**The fix is durability, not a label.** The count lives in the SQLite `metadata` table — a
different I/O path from the JSONL sink that is already failing — seeded from disk at open
(read-only handles too). **LIFETIME-SCOPED AND MONOTONIC, and that is a deliberate semantic:** a
trail that lost an entry is permanently incomplete and `verify()` returns `valid=True` over that
hole forever, so a number that healed would be a lie. ⚡ It never shipped (absent from v0.9.9), so
no released behaviour changed — that is why the semantics were free to define.

⚠ **Do NOT import spore-745's "a resettable integer only MOVES the window" objection here.** That
is about `AuditTrail._dropped_since_last`, the ride-along marker, which must chain into the NEXT
entry and genuinely needs an outbox. **The marker is still process-local and still open.** The
count has no window to move: it is written when the failure happens. `audit.py`'s comment now
separates the two promises explicitly.

### ⚡ AND THE REGRESSION TEST FOR THE ORIGINAL DEFECT WAS HOLLOW — THE HALF WORTH CARRYING
It read `cli.py` off disk and asserted `"status.audit_write_failures" in text`. Mutation-proven:
hardcoding `"write_failures": 0` with the token alive in a COMMENT, plus `if False:` on the human
branch, left it passing and all 1822 tests green. **A substring assertion is satisfied by a
comment, a docstring, or a dead branch.** Replaced with tests that lose a write in one `Store`,
CLOSE it, and read it back from another — CLI `--json`, CLI human, MCP handler, a read-only
handle, and accumulation across sessions. All three mutants now fail.
▶ **The generalisable move: the test crosses the boundary the DEFECT crossed.** A test that
degrades and asserts inside one process could not have seen this however much code it executed.

### ▶ WHAT ELSE CLOSED (each re-derived at HEAD by running, not read off a table)
- **sdist leaked 984 KB of internal project memory** — 39 files including six review reports
  enumerating open defects by file and line, and three carrying the operator's absolute home path.
  Fixed as an ALLOWLIST, not an exclusion of `project_memory`: a denylist only catches the name you
  already found. **Independently re-derived after the lane reported it** — a fresh build is 967,624
  bytes over 69 files, zero `project_memory` entries, no home path anywhere. The wheel was never
  affected, which is why it survived nine releases: the clean artifact is the one people look at.
- **The packaging gate skipped in CI**, because `hatchling` is a build dependency hidden by build
  isolation. A guard present on the dev machine and absent where it is enforced. CI installs it now.
  ⚠ A SKIP IS NOT A PASS — if it starts skipping again the gate is gone and nothing will say so.
- **The reserved-audit-kwarg set was typed by hand** and covered four of six collision-capable
  names; `event` and `payload` go positionally at the flush splat and collide identically. Now
  DERIVED from the signature by parameter kind. The test derives it independently AND separately
  proves each name really does collide, instead of looping over the set under test.
- **The SKILL.md ladder gate** keyed its historical-quote exemption on FORMATTING. Now requires a
  retrospective cue on the line, tight enough that an unrecognised cue makes the gate FIRE.
- **`project_memory/CLAUDE.md`** — every checkable claim re-derived from live sources; counts
  replaced by the commands that produce them. Found nine more wrong claims than the four reported.
- ⚡ **THE LEVELCAP CENSUS WENT WELL PAST THE THREE README SITES DIOGENES NAMED — this is the
  session's clearest instance of its own class.** Deriving the list from the PROPERTY (any text
  stating a ladder or demotion range that implies a 3x top rung) instead of from the finding found
  **five more live shipped surfaces**: `types.py:599`, `schema.py:459` (an OPERATOR-FACING warning
  string), `graduation.py:429`, `docs/architecture.md:19` and `:36` (which ship in the sdist), plus
  `_demote_line`'s own docstring — the function that PERFORMS the demotion still said
  "(3x->2x or 2x->1x)". Found only because I ran the function to check a claim I had just written
  into the public README. Measured through it: 2x->1x, 3x->2x, 4x->3x, 12x->11x, 18x->17x, 25x->24x.
  ⛔ This is the FOURTH consecutive sweep of this class, and every previous one scoped itself to the
  strings already known — the 09-01 review said so in as many words: *"the class was defined by the
  strings already known, so the ladder spelling was never in the denominator."* Every site now
  states the RULE rather than a sample of it, which is the only form that survives the next move.
- **Carried:** the levelcap sweep finally reached README (demotion is by one from ANY level; the
  1x→3x diagram and promotion line implied a removed ceiling); the carry-forward warning stopped
  naming "Proven-tier", a category strictly wider than its own `>= 3` gate; `server.py`'s last
  assert-for-narrowing became an explicit raise (-O strips asserts); `graduation.py`'s coordinate
  lost its line number on purpose after being wrong twice, differently; and
  `_is_write_lock_contention` collapsed from two AST-identical copies into one, pinned by an
  IDENTITY assertion — a grep passes on two identical copies, which is the forbidden state.

### ⛔ THE ONE DELIBERATELY NOT CLOSED — DO NOT "FIX" IT
**`graduation.py:95`, `_BARE_GRADUATION_RE` still `([23])x`.** This is a RULED deferral held by
`spore-676`, gated on `spore-675`, and the reasoning in the file is now correct where it used to be
false. Widening it puts **fourteen mature carried patterns** onto the bare-demotion path at the
next re-stamp — a far larger blast radius than the defect. It is carried by Diogenes as a HIGH and
will keep being carried; that is the ruling working, not the ruling failing.
▶ **The improvement that would NOT violate the ruling, if someone wants it:** make a bare 4x+ line
VISIBLE (counted as skipped and surfaced) without making it demotable. Today it matches neither
regex and is reported as nothing at all — `absence_of_signal_rendered_as_health`. That is a
decision for whoever owns spore-675/676, not a casual widening.

### ⚠ L3 COVERAGE WAS NOT ACHIEVED TODAY, AND THE EXIT CODE SAID OTHERWISE
`deep_review.py --seats complement,glm-5.3,gpt-oss` exited **0** with **two of three seats
producing nothing**: complement `[Claude CLI exit code 1: (no stderr)]`, glm-5.3 empty after 380.8s
with `files_opened=0`. Only gpt-oss answered — 1,170 chars in 8.3s against a 96k diff, which reads
as a rubber stamp. **Checked the store against stdout per the day's rule; they AGREE, so this is a
real gap and not a hidden success.**

⛔ **THE DEFECT WORTH CARRYING IS THE EXIT CODE, NOT THE SEAT.** `deep_review.py` exits 0 and
prints the "PRODUCED NO REVIEW" warning inside the body, so a seat reading the exit code records a
three-lineage mesh when a lineage silently dropped out. That is the "L3 failures exit 0" family
aimed at the mesh's own composition. **Check any L3 claiming three lineages against
`state/verdicts.jsonl` for whether all three actually returned.** Routed to the fan-in — flow's
apparatus, not this repo's.

⚠ **AND A CORRECTION I OWE, because I filed the strong version first.** I reported complement as
"contributing nothing for at least three days". **FALSE, and the fan-in measured it back:** 16 of
its 24 rows SUCCEEDED, and today it was 8 successes out of 9. What is real is a ~33% intermittent
failure rate (8/24) with an opaque signature — seven identical `Claude CLI exit code 1 (no
stderr)` plus one timeout. **The denominator was in my own output — I printed "24 rows, 8 errored"
and generalised from the numerator anyway.** An errored-row count is not a failure rate.
▶ **RETRIED TWICE MORE, TIGHTER, AND IT STILL DID NOT COVER.** Scoped to
`anneal_memory/store.py` alone (12,084 chars): glm-5.3 DID read this time — rounds=6,
files_opened=1 — and hit its 550s timeout with nothing emitted; complement threw the identical
`exit code 1: (no stderr)` twice more, three for three from this seat.
⚡ **AND THE FAILURE IS NOT UNIFORMLY DISTRIBUTED, WHICH IS THE PART TO CARRY.** Measured over all
26 complement rows in the store: **5 of the 8 lifetime errors are anneal-memory** (5 errors to 1
success), while wisp is 3-for-3, blackjack 3-for-3, levain 2-for-2. Today specifically, all 8
successes fall in a 56-minute morning window across seven other repos and all 3 failures are this
repo in the afternoon. That does not cleanly separate a repo effect from a time effect — 09-03 has
a later solitaire success AFTER an anneal-memory failure, which argues against pure time. Stated
as two live candidates rather than a diagnosis.
▶ **A gpt-oss pass DID land on the tight scope and was triaged against disk — 2 of 6 had
substance and BOTH HIGHs were false.** "sqlite3 is never imported in store.py" (it is, line 19),
"the package cannot import" (it imports and binds all six names), and "_seed_audit_health is
called twice for writers" (instrumented: exactly one per open, both paths) — all three
diff-scoping artifacts, the seat seeing a moved function and an annotation line without the file.
⚠ **Both HIGHs were confidently wrong in 8.65 seconds. Do not weight this seat's severity.**
What was real: a silently-swallowed persist failure (now logged at debug) and a read-modify-write
that could lose an update between concurrent writers — DOCUMENTED, NOT CHANGED, because
concurrent writers already break the hash chain by construction and the proposed atomic increment
is worse for the scenario this counter records (a failing sink means repeated persist failures,
where a lost increment is lost forever and a lost whole-value write is repaired by the next one).

▶ **So the riskiest part of this change set was verified by me directly instead:** the standalone
`commit()` inside the post-commit handler cannot leak another method's uncommitted DML (a batch
that rolls back publishes nothing) and cannot disturb an open wrap. Both reproduced, both pinned as
tests, and the pin mutation-checked for vacuity.

### ▶ STILL OPEN GOING INTO 09-05
- `spore-745` — the `dropped_before` MARKER is still process-local (the count is not; see above).
- `spore-746` — rotation-failure orphan, pre-existing MED.
- `spore-747` — the levain two-anneal skew.
- `cli.cmd_verify` still buries `skipped_lines`, so an operator running `verify` is not told lines
  were skipped. Untouched, still worth doing.
- The three live clocks below are untouched: `spore-721` (09-11), `spore-722` (09-10),
  `spore-675` step 4.

---

## ⚠ (SUPERSEDED 2026-09-05 by the block above) PICKUP — AM-AUDIT-AFTER-COMMIT LANDED. Kept for the count story, not as instructions.

**State at close of the 09-04 seat:** AM-AUDIT-AFTER-COMMIT is COMMITTED — `d7b482e` (the policy),
`e8c04da` (L3 complement/glm + L4) and `8a6cc21` (the codex retry). Working tree clean apart from this file.
**1822 tests** (from 1764) · mypy clean · **ruff 63** · pre-push gate green on HEAD, stamp
`0.9.10.dev0`, not a released number. Working tree clean.
⚠ The ruff baseline at HEAD was **65, not the 64 this file claimed** — re-derived from disk, which
is this file's own standing instruction, and it was wrong about its own number. It is **63** now:
two previously-unused imports in `tests/test_audit.py` became used, and `project_memory/` is
excluded from ruff because the archived Phase-1 scripts arrived with today's move and added 52
findings to an otherwise real signal.

### ⛔ THE FINDING WAS "A CORRECTION REACHED ONE OF FIVE SITES." THE REAL DENOMINATOR IS SIXTEEN.
The 09-03 codex L3 HIGH established *an audit-sink failure must not propagate once the work is
committed* and was implemented inline at ONE site (`wrap_cancelled`). Measured by execution 09-04:

- **First census said EIGHT.** It scoped by the SYMPTOM — the literal text `self._audit.log`.
- **L1 found SEVEN MORE.** The association methods reach their commit through a free-function
  helper's `commit=` argument and emit via `_audit_log`, so no text search for the first spelling
  could see them. `gc_pattern_associations` and `sever_pattern_concept` DELETE edges and then raised
  a raw `OSError`.
- ⚡ **All three guards written that morning inherited the one scoping decision** — the census, the
  AST scan (same shape), and the behavioural table (the methods the census produced). **A third
  guard sharing a blind spot is not defence in depth.** This is the day's cross-repo class fired
  three levels deep inside the fix for it.

Verified wedges (each run, not read): `wrap_started` left a LIVE committed wrap the caller held no
token for → next `prepare_wrap` raises `WrapInProgressError` — **Alex's lockout, reachable through
the release that closed it**. `wrap_completed` wrote the wraps row, cleared `wrap_started_at`, and
told the agent it failed. `prune` deleted episodes; `save_continuity` externalized the file.

### ▶ WHAT THE FIX IS, so nobody re-derives it as eight inline guards
ONE home: `Store._audit_log_after_commit`. Sixteen sites route through it. `_audit_log` (the
pre-commit twin) now has **ZERO production callers and a test asserting it** — a future pre-commit
site must state its reason instead of inheriting the helper's name. That test is the guard that
would have caught the seven.

### ⚠ THE COST WAS PAID, NOT DEFERRED — and this is the part worth carrying
A swallow that nobody can see is `absence_of_signal_rendered_as_health` with extra steps. Measured:
**8 mutations, 7 dropped writes → ONE audit entry and `verify()` returned `valid=True`.** A
`UserWarning` could not carry it either — the default filter dedups per location (**5 failures → 1
warning**) and `-W error` produces **zero signal**. So four channels now, most-suppressible last:
`dropped_before` riding into the next entry (hash-chained), `status().audit_write_failures`
(pollable), the `anneal-memory` logger, then the warning.

### ▶ OPEN / NEXT
- ✅ **L3 COVERAGE IS RECORDED and this repo is OFF today's bugfind list.** The FIRST pass was not
  coverage — codex timed out with no output, glm was cut off part-way — and per `spore-744` (*being
  stingy with codex is a defect, not thrift*) it was **re-run on a tighter `--paths
  anneal_memory/audit.py` scope**. It came back in 424s with **two HIGH and one MED**, i.e. the
  retry was the whole value. complement + glm triaged in `e8c04da`; codex in `8a6cc21`.
- ⛔ **THE CODEX HIGH IS THE ONE TO CARRY: A DURABILITY HICCUP WAS READING AS TAMPERING, AND MY OWN
  MORNING FIX MADE IT POSSIBLE.** `write`+`flush` succeed, `fsync` raises EIO → the complete line is
  already on disk while `_seq`/`_prev_hash`/the drop counter are unchanged, so the retry re-emits the
  SAME seq. Reproduced: seqs `[0,1,1]`, `dropped_before` `[None,2,3]` (pending drops counted twice
  AND the landed entry counted as dropped), `verify()` → **`valid=False`**. On the record whose whole
  job is telling a hiccup from tampering. **FIXED**: all-or-nothing append, pre-append size restored.
  Mutation-checked. Also covers the partial-write case that concatenates into unparseable JSON.
- ⚡ **AND CODEX'S CLOSING LINE WAS ABOUT THE TESTS, NOT THE CODE — keep this discriminator.** *"The
  regression tests replace `AuditTrail.log` wholesale with a function that raises before writing."*
  Every fixture varied the sink outcome and the method while holding constant **WHERE IN THE WRITE**
  the failure lands — the dimension both HIGHs live in. `TestFailureLandsAtDifferentPointsInTheWrite`
  exists to vary it. **Ask what a fixture varies and whether the defect lives in what it holds fixed.**
- **L4 caught the one no test could**, and it is the pattern to carry: `audit_write_failures` reached
  `StoreStatus` and **none of the three transports**. The CLI `--json` builds its own `audit` object
  with four hardcoded keys, the human output prints four, the MCP handler composes its own line. The
  channel documented as "pollable" was not pollable on any surface an operator reads. **A field on a
  dataclass is not a surface.** Fixed + pinned across every module that reports audit health.
- **`gc_pattern_associations` and `drain_co_surface_events`** emit only on a non-zero count, so they
  are covered by the mechanical scan and NOT by the behavioural table. Stated in the test on purpose.
- ⚠ **CORRECTED AT CLOSE — the partial-write half of this IS fixed, do not go looking for it.**
  L2 named "an ENOSPC mid-line leaves a truncated line that `verify()` later reports as tampering",
  and this file said NOT FIXED. The codex round's all-or-nothing append **closes it**: the
  pre-append size is restored, so a partial write never survives to concatenate with the retry
  (`test_a_partial_write_is_rolled_back_not_left_to_concatenate`). **What remains is only the
  reporting half:** `cli.cmd_verify` buries `skipped_lines`, so an operator running `verify` is not
  told that lines were skipped. That one is untouched and still worth doing.
- ⛔ **`spore-745` — `dropped_before` IS LOST BY ANY `close()`/REOPEN, NOT JUST A CRASH.** Measured:
  an ORDINARY close+reopen leaves 3 episodes against 2 entries, no marker, `valid=True`,
  `audit_write_failures` back to 0. **Every CLI invocation opens and closes a Store**, so across CLI
  commands the mechanism effectively never fires. A durable fix wants a transactional outbox — a
  resettable integer only MOVES the window. ⚠ This comment has been wrong TWICE in one day, both
  times overclaiming; do not let a third version do it. · **`spore-746`** — rotation-failure orphan
  (pre-existing MED). · **`spore-747`** — the levain two-anneal skew + my design read (anneal's
  sidecars fail CLOSED on a newer schema while the SQLite store degrades SILENTLY; that asymmetry is
  what makes it a correctness hazard, not a reporting gap).
- (superseded framing kept for the trail) `dropped_before` closes swallow-only, not swallow-then-crash. Both L3 seats found this
  independently. The pending count is process-local: a crash between the swallowed write and the
  next successful one loses it, `__init__` resets to 0, and `verify()` reports `valid=True` over the
  gap exactly as before the mechanism existed. Making it durable means persisting outside the sink
  that is already failing (the SQLite metadata table is a different I/O path) — **a design question
  about issuing a write from inside a post-commit exception handler, not a patch.** Deliberately
  left; the comments say so rather than implying a guarantee the code does not have.
- ▶ **`decision_influence_receipt_contract.md` travelled here with the move and is NOT anneal-local.**
  `voltron/CLAUDE.md` names it an integrator-seat surface — the frozen cross-altitude schema binding
  anneal, vagus, Bridge and FlowPoker, where a field change is a checkpoint and never unilateral.
  The move removed the friction without removing the discipline. Do not treat it as this repo's.
- The **three live clocks below are untouched**: `spore-721` (09-11), `spore-722` (09-10),
  `spore-675` step 4.

### ⚙ TWO SHARP EDGES FOUND IN THE APPARATUS ITSELF (fixed; no new guards built)
- **The SKILL.md ladder gate took `max()` over the UNION of ladder lines**, so one corrected line
  certified every uncorrected one — line 85 supplied `12x` while line 35 still taught `1x→2x→3x`.
  Now judges each ladder RUN (ascending, not a demotion pair, not a whole-span historical quote).
- **The pre-push hook read the WORKING TREE while a push publishes COMMITS** — reproduced in a
  scratch clone: a bad committed stamp plus an uncommitted fix flipped the gate GREEN. New test reads
  the stamp out of `git show HEAD:`. Also: `sed -n '/A\|B/p'` is **not alternation in a BSD BRE**, so
  the hook's entire "here is why" block printed NOTHING on this machine, and the remediation named
  the release-stamp class unconditionally — a syntax failure got four bump-the-version steps.
- ⚠ **`test_store_operation_literal_has_no_drift` greps raw source, DOCSTRINGS INCLUDED.** Naming a
  new parameter `operation` failed it, and so did merely *mentioning* the pattern in prose. Left
  alone deliberately (spore-551: fix or tolerate a mis-scoped guard, never give it a neighbour).

### ⚖ MEASURED, AND IT CORRECTS BOTH SKILL.md AND THE GENERATED INSTRUCTIONS
*"The level is a monotonic high-water mark"* is **FALSE** and was propagated from
`continuity.py`'s generated instructions into SKILL.md by the fix that corrected the ladder.
`4x`→`3x`, `12x`→`11x`, `18x`→`17x` on an ungrounded re-stamp (a DECREMENT, not a flatten to 3x —
"never flatten back to `3x`" stands). `max_level_reached` is the monotonic one. ⛔ **`continuity.py`
contains BOTH the true statement and the false one, 39 lines apart, in the same package handed to
the composing agent every wrap**. ✅ **BOTH SITES FIXED at close** — `continuity.py`'s ladder
section no longer calls the visible level monotonic, and a test at the GENERATOR
(`test_teaching_never_calls_the_VISIBLE_level_monotonic`) refuses any sentence that says "monotonic"
without naming `max_level_reached`, with its sibling in `test_integrity.py` pinning SKILL.md.
Mutation-checked. *(This line said "NOT yet fixed. Do that." until the cold-read pass at close — a
stale INSTRUCTION in the very file whose standing rule is to re-derive before acting. An out-of-date
instruction is worse than an out-of-date fact, because the next session executes it.)*

## ⚠ (SUPERSEDED 2026-09-04 by the block at the top) PICKUP 2026-09-04 (evening) — kept for its reasoning, NOT as instructions

⛔ **DO NOT ACT ON THE NUMBERS OR THE TASK LIST BELOW.** Its "1764 tests · ruff 64" was true that
evening and both are wrong now (1822 / 63), and the ruff baseline it states was never right —
it was 65 at HEAD. Marked at close because this block's OWN warning is that *a plan read in order
hits the stale framing first and the correction never*, and it had become the stale framing.

**anneal is at a clean stop.** 0.9.9 is live on PyPI + GitHub, `main` is `0.9.10.dev0`, CI green on 3.10/3.11/3.12/3.13, working tree clean, **1764 tests** · mypy clean · ruff 64 (unchanged baseline). Nothing is half-finished and nothing is blocked.

### ⛔ THE ONE DISCIPLINE THAT MATTERS TOMORROW: THIS FILE LIED TO A SESSION TODAY
A block in here described a **75-day associative-layer outage** and drove a wrong prioritisation — `spore-675`'s own text had superseded it on 08-31 (*"The associative layer is alive. That question is settled and does not need re-asking"*) and the block was never updated. ⚡ **A plan read in order hits the stale framing first and the correction never.** So: **for any number in this file, re-derive it from disk before acting.** That cost an hour today and it is the third instance of one class in one session.

### ▶ TONIGHT'S DIOGENES REPORT — the window is LARGE and the triage recipe is proven
It routes to `project_memory/diogenes_20260904.md` (untriaged, verbatim). The window is **~10 anneal commits** — the biggest single-day anneal window in the record — so expect volume. **The recipe that worked today, in order:**
1. **VERIFY EVERY FINDING AGAINST DISK BEFORE BELIEVING IT.** 2 of 6 of his coordinates were wrong yesterday, and one was a SUBJECT coordinate — the half his OWN rule declares greped and exact. His confidence is not evidence.
2. **MUTATION-CHECK EVERY FIX.** It caught a defect that L1, L2 *and* codex all read past: a regression test for non-`JSONDecodeError` parse failures whose payload happened to fail as a `JSONDecodeError`. **A review reads what a test says; a mutant tests what it does.**
3. **ASK OF EACH NEW GUARD: would the defect itself make this pass?** A sequential test of a concurrency property answers yes — that is how the bad one got written.
4. Expect findings **in yesterday's fixes**, not only in old code. 15 of today's were self-inflicted, caught by a lineage that did not author them.

### ⚖ TWO RULED DEFERRALS — DO NOT RE-DERIVE THEM AS OPEN
- **`flow/scripts/crystal_decision_apply.py` level cap** — ✅ **FIXED by main 2026-09-04** (`e5cbaca7`, with a test). Its anneal-side CAUSE (a contract docstring instructing consumers to guard `level in (2, 3)`) was fixed in 0.9.9.
- **`_BARE_GRADUATION_RE` stays `[23]`** — `spore-676`, gated, on blast radius (widening puts 12+ mature carried patterns onto the bare-demotion path). ⚠ Its stated justification "the asymmetry is inert" was **measured false** and is corrected in place; the deferral stands on blast radius alone.

### ▶ THE THREE LIVE CLOCKS
- **`spore-721` — AM-LINKGATE BLOCK, ruled BUILD, `next: 2026-09-11`.** Gate on **≥2 pair-capable** graduations (a 1-graduation wrap cannot form a pair); **fail closed with a loud escape** (it refuses a memory SAVE). Full apparatus, codex at L3.
- **`spore-722` — Slice C step-3, `next: 2026-09-10`, KILL CRITERION REAL.** Gate is **Phill's labelling time**, not data. Answer *"what would we do with a GO?"* first; unanswered by 09-10 → shelved formally. Flow's read: probably shelve.
- **`spore-675` step 4 only** — let `prepare_wrap`'s cold-candidate surfacing drive routing each wrap. Small, blocks nothing, consolidate-seat act. Steps 1–3 are done (measured 09-04: 29 crystals at levels 2–10, 422 pattern links, prose line gone).

### ⚠ OWED ELSEWHERE
- **levain moves `KNOWN_GOOD_ANNEAL` + `TEMPLATES_RECONCILED_ANNEAL` → 0.9.9** as step 7 of its own cut. Pip floor `>=0.9.8,<0.10` does NOT move — that constraint is why this was 0.9.9 and not 0.10.0. The levain seat re-swept and confirmed no template edit is owed.
- **`spore-424` was rewritten today** — both its halves were wrong and each cost a failed publish (`~/.local/bin/twine` 6.2.0 dies on `Metadata-Version: 2.5`; use TOKEN3 + twine ≥ 7). Read it before any cut.

## ✅ SHIPPED 2026-09-03/04 — full narrative in `COMPLETED_SESSIONS_ARCHIVE.md`

- **0.9.9 PUBLISHED** → PyPI + GitHub, tag `v0.9.9` (`076eed9`). `BEGIN IMMEDIATE` closes the `wrap_cancel` read/clear race 0.9.8's docstring claimed at five sites was already closed; **AM-WRAPCANCEL-CAS** ships Alex De Groodt's A3 (`expect_token` + `wrap_token` on both transports, three-way refusal state); the published 0.9.8 SKILL.md row telling agents `wrap_cancel` was "CLI only" — his own lockout condition — is off PyPI; **33 surfaces** teaching the ceiling 0.9.7 removed are corrected.
- **Diogenes slot 20 → 2**, both ruled deferrals (below). All 6 filed findings verified against disk before being believed; 2 of his coordinates were wrong, one on the SUBJECT half his own rule gates.
- **Guards added, all mutation-checked:** cross-connection concurrency tests asserting the MECHANISM (`in_transaction` at the hook, not a timeout) · `test_skill_documents_every_tool` + a non-vacuity proof for the cell-scoped CLI-only guard · release-stamp test + **fail-closed `scripts/hooks/pre-push`** · cross-version syntax gate (an invalid escape was fatal on 3.10/3.11 while only a warning on the 3.13 dev box).

## ⚠ (SUPERSEDED 2026-09-04) PICKUP 2026-09-02 — the SERIAL order below is PARTLY DONE; `spore-675` is largely complete (see the live pickup at the top). Kept for the reasoning, not as instructions.

**Nothing here is half-finished. Two commits landed 2026-08-31 and both are clean; what remains is sequenced and gated, not blocked.**

> ⚠ **THE PUBLISHED VERSION MOVED ON 2026-09-02 AND THE CHAIN ABOVE DID NOT.** `0.9.8` shipped to PyPI from a **parallel session** carrying Alex De Groodt's field reports — see `## ✅ 0.9.8 SHIPPED` below. It touched `integrity.py` / `server.py` / `store.py` / `cli.py` and **nothing in `graduation.py`, `crystal.py`, `pattern_associations.py` or `continuity.py`**, so `spore-675` → 627 → `spore-676` is untouched and still the order. If you are here for the chain, read that block once for the version number and come back.

### ✅ SHIPPED 2026-08-31 — the AM-LEVELCAP widening finally reached the surfaces that CONSUME it (`spore-535`)

`acf5d7a` (generated instructions + internal descriptions + a real HIGH) and `b5a6e7c` (the adopter docs). **1691 tests, mypy clean, ruff unchanged at its 65-error baseline.**

⛔ **THE SHAPE, AND IT IS THE THING TO CARRY: 0.9.7 shipped the widening on 2026-08-14 and NO other surface moved for SEVENTEEN DAYS.** The library accepted 2-and-up with no ceiling while the generated wrap instructions still said *"where N is 1, 2, or 3"* and taught the ladder 1x→2x→3x and then stopped — **emitted to the composing agent at the moment of use, every wrap**. A library widening that no teaching surface carries is a widening that never reaches the behaviour. **A release that WIDENS emits no error signal anywhere downstream**: nothing breaks, every consumer keeps working correctly against the narrower contract it already knew, and the only symptom is absence. A release that BREAKS is found the same day. `absence_of_signal_rendered_as_health`, with the release note in the role of the thing that looked like completion.

⚠ **THE FILED SITE WAS THE SMALLEST OF THE SET.** `spore-535` named one sentence about marker syntax. The class spanned **four surfaces** — library (already fixed) · generated + internal · the Levain seed · the adopter docs — and **27 lines** carrying the retired teaching (unit: removed lines matching the pattern set, tests excluded; the set includes ceiling *language* such as "top-tier" and "3x/2x" by judgment, which both L3 seats had filed as ceiling sites). **A spore is an entry point, not a work order.**

**What the L3 mesh caught** (`complement` errored and contributed nothing — a named gap, not a clean bill):
- ⛔ **A PRE-EXISTING HIGH IN SHIPPED 0.9.7, FIXED:** `([2-9]|\d{2,})` matched **zero-padded** levels. `| 01x` parsed as a validated graduation at `int("01") == 1` and `| 00x` at level 0 — so the deliberate 1x exclusion was bypassed by a leading zero, and such a line formed a Hebbian pair as if Proven. `_demote_line` rewrites by level and cannot rewrite `| 01x`, so counters could report a demotion the displayed line never took. Now `([2-9]|[1-9][0-9]+)`, parametrized regressions added.
- Three defects in the same session's own work, all corrected: a false "monotonic high-water mark" claim (the visible level CAN be demoted; `max_level_reached` is the monotonic one), an over-generalized "not a closed set" for markers (**measured: the reader accepts `!+`, `?`, `✓`, `*` and silently DROPS `~` — kinds are CLOSED, only the `!` run is open**), and positive-token-only tests that *"only 1x, 2x, 3x, or 12x"* would have passed.

**The pin now derives from the generator's own output** (`TestTeacherCoversReaderRange`) and is **falsified, not merely green**: source reverted to HEAD with the tests kept → **9 of 15 FAIL**. ⚠ The old pin could never have worked — `TestCanonicalTemplateFormatEndToEnd` claimed to pin the template against the regex, but its fixture is a hand-copied string and `_marker_reference` appeared in that whole file **only inside docstrings, never imported, never called**. And the drift ran where a round-trip test structurally cannot look: teacher ⊆ reader held and always did; the defect is the converse.

### ⛔ `spore-676` — THE BARE 4x+ HOLE IS FIRING, AND THAT MAKES THE FIX MORE DANGEROUS, NOT MORE URGENT

A today-dated **bare** graduation at 4x or above (no `[evidence:]` tag) matches **neither** `_GRADUATION_RE` nor `_BARE_GRADUATION_RE` (`([23])x`). Verified: bare 12x → both False; bare 3x → `_BARE` True. It is neither validated nor bare-demoted.

⚠ **MEASURED 2026-08-31 against the live neocortex, and it inverts the spore's original closing line.** That line said the gap *"only fires on a today-dated bare 4x+ line, which flow's own practice does not currently produce (evidence tags are habitual)."* **Both halves are false.** Of the pattern lines in `~/.anneal-memory/memory.continuity.md`: **only THREE carry an `[evidence:]` tag. FOURTEEN are bare at level ≥4. TEN of those are dated 2026-08-30 — the last wrap.** Ten today-dated bare 4x+ lines matched neither regex at the most recent consolidate and were silently skipped. **Bare carries are the habit; evidence tags are the exception.**

⚡ **AND THE HOLE IS CURRENTLY PROTECTING THOSE FOURTEEN LINES.** Widening `_BARE_GRADUATION_RE` puts `absence_of_signal_rendered_as_health` (**18x**), `the_checkable_proxy_is_graded_instead_of_the_real_target` (14x), `correction_comes_from_outside_the_planner` (13x) and eleven more onto the bare-demotion path at the next re-stamp. The source's own comment at `graduation.py:57` already records the risk: widening makes every today-dated 4x+ line without evidence *"newly eligible for bare demotion — a mass demotion of mature carried patterns."*

▶ **FIRST MOVE IS EMPIRICAL AND UNCHANGED:** does AM-CARRYFORWARD hold a warm bare **18x** line through a demotion pass, on a **COPY** of the real store? codex argued it does; **that claim is UNVERIFIED and the source asserts the opposite risk.** If it does not hold, the right outcome is to make the **TEACHING** honest about the asymmetry rather than build a path that can erase an 18x pattern. Also owed: regression tests for bare demotion, warm hold, and malformed-evidence reporting at 4x/12x — none exist.
⚖ **SCHEDULED `next: 2026-09-02`, own Agent View session, own apparatus. GATED on `spore-675` being verified complete — the gate outranks the date.**

### ⚖ THE SERIAL ORDER, and why it is serial

1. ⚠ **`spore-675` — LARGELY DONE. THIS ENTRY WAS STALE AND CAUSED A WRONG PRIORITISATION 2026-09-04; RE-MEASURE BEFORE ACTING ON IT.**
   ⛔ It used to read *"co-surface pair FORMATION has been zero since 2026-06-20 (72 days)... the dominant fix for the Hebbian starvation"*. **`spore-675` ITSELF SUPERSEDED THAT ON 08-31** — *"The named ORACLE PASSED: `association_stats().total_links` = 304, up from the 165 that sat flat all through the outage. **The associative layer is alive. That question is settled and does not need re-asking.**"* This block was never updated, so a session reading the plan in order hits the outage framing first and the correction never.
   ✅ **MEASURED 2026-09-04:** crystal store **29 patterns at levels 2–10** (not 14 flattened at 3), `last_activated_on` through 09-01 · **297 episode links / 422 pattern links** (358 co_graduation, 64 co_surface), co_graduation links dated 09-02 · **step 3 DONE — the 25-name prose working-set line is GONE from the neocortex** (0 lines carrying 5+ pattern names).
   ▶ **WHAT REMAINS is step 4** — let `prepare_wrap`'s cold-candidate surfacing drive routing every wrap, so this stops being a manual migration — plus a status pass on the spore. **That is small and it is NOT a blocker for anything downstream.** *(Still a consolidate-seat act; it touches the crystal store.)*
2. **627 fix 1 — STATE THE DENOMINATOR.** Cheaper than first scoped: the oracle lives in `global/CONTINUITY_MANAGEMENT.md`, which is **on-demand and uncapped**, so the carrier-byte objection dissolves. `global/CLAUDE.md` carries only a routing *pointer* to it, not the rule. Two denominators are needed, not one: possible-new-pairs among today's graduations, **and** pair-capable events over applied events.
3. **627 fix 2 — the union. DEFERRED BY ITS OWN LOGIC.** The empty co-surface channel is not dormancy (falsified — retrieval returns `warm` AND `cold`, tier is computed and attached, never used to exclude) and not query length (falsified). It is that two complementary retrieval paths — keyword `retrieve_patterns` and associative `retrieve_relevant` — are wired **fallback-vs-primary instead of union**, and the hook takes the associative path whenever the anneal DB exists, which it always does. **Production has never run the keyword path in normal operation, and keyword is the half producing every pair-capable event.** With 18 patterns the union's ceiling is too low to measure, and it is the highest-blast-radius change in the system; it becomes testable only after (1) populates the store and (2) supplies a denominator.

⚠ **Why `events_applied` is not a pairing metric, so nobody re-derives it:** `_aggregate_events` adds an event to `seen_event_ids` at `pattern_associations.py:462` and only THEN drops it at `:464` for `len(names) < 2`. A 0-or-1-name recall counts as *applied* while forming no pair. Measured on the live spool: **206 events, 200 with zero names, 6 with one, none pair-capable** — and every one carries `basis: "assoc_hop"`, which is in `_INDEPENDENT_BASES` and therefore reinforces at factor **1.0**, so shadow-mode gating does **not** explain it.

**Still open below:** Slice-C step-3 (⚖ now carried by `spore-722`, kill criterion 2026-09-10) · the PMB influence gap · AM-LINKGATE's BLOCK half (⚖ ruled BUILD, `spore-721`, `next: 2026-09-11`) · `spore-093` content store.

## ✅ 0.9.8 SHIPPED 2026-09-02 — AM-MCP-WRAPCANCEL. Alex De Groodt's field report, 29 days late.

**LIVE ON PyPI and verified by installing FROM PyPI into a clean venv and replaying his stuck-wrap scenario against the published artifact** — not by tests passing, not by the tag existing. https://pypi.org/project/anneal-memory/0.9.8/

⛔ **THIS WAS A SIDE TRACK AND IT STAYED ONE.** Nothing in the `675 → 627 → 676` chain was touched. Files changed: `integrity.py`, `server.py`, `store.py`, `cli.py`, `README.md`, `skill/`, `docs/`, tests.

**A1 — `wrap_cancel` is an MCP tool.** It existed as the `wrap-cancel` CLI subcommand and `Store.wrap_cancelled()` and as **neither** over MCP, so an agent that hit `WrapInProgressError` had no in-band way out; his wrap stayed stuck **three days with 31 episodes stranded**. `tools/list` 16 → **17**; both `tool-integrity.json` manifests regenerated.

**A2 — the recovery message named a path its most likely caller could not reach.** It said *"Finish it with validated_save_continuity, or abandon it with store.wrap_cancelled()"* — two Python APIs, offered to an MCP client that can call neither. **Both halves carried it, not just the reported one**, as did **four** sibling `StoreError` hints in `load_wrap_snapshot`. Now two module constants (`_WRAP_FINISH_PATHS` / `_WRAP_CANCEL_PATHS`) so a fifth site cannot drift, and a test asserts the property across **all** branches rather than the cases that were noticed.

### ⚠ THE REVIEW MESH FOUND FOUR DEFECTS IN THE FIX ITSELF — carry these, they are the reusable half

- **A TOCTOU in the new handler, found by codex AND glm independently.** It pre-read the snapshot to have something to report, then called `wrap_cancelled()`, which clears whatever is **current** — so a peer finishing that wrap and starting a new one in between got its NEW wrap destroyed while the response named the OLD token. `Store.wrap_cancelled()` now returns a **`WrapCancelReceipt`** (`token` / `started_at` / `episode_ids` / `partial_state`) read **inside the clearing transaction**; the pre-read is gone, so the window is removed rather than narrowed. Additive — callers ignoring the return are unaffected.
- **The partial-state path said "No wrap was in progress"** — false, on the tool's **primary** recovery case, contradicting the error that sent the operator there. Fixed in the MCP handler and, a round later, in `cli.cmd_wrap_cancel` for parity.
- **`status` now reports `wrap_started_at`.** The description tells the agent to check `status` before cancelling a wrap that may be a live peer — and `status` returned a bare boolean, so the check it named **could not be performed**. Advice pointing at a surface that cannot answer it is this release's own defect class.
- **`skill/anneal-memory/SKILL.md` told agents `wrap_cancel` was `— (CLI only)`** — true until 0.9.8, false the moment it shipped, in the depth doc an **agent** loads. Now asserted: no SKILL.md row may say "CLI only" while naming a tool that is in `TOOLS`.

### 🔴 UNRELEASED ON `main` — rides the next cut, whatever that is

Second review round landed after publication. **None of it is gate-tier** (nothing bad ships), which is why no `0.9.9` was cut:
- `cli.cmd_wrap_cancel` reports from the receipt (TOCTOU + partial-state parity with MCP); `--json` carries the full receipt.
- The `wrap_cancel` description was **de-shouted** — it was the longest in the table and the only one in caps, while `delete_episode` carries a harsher consequence in flat prose.
- `docs/library-quickstart.md` points at the transports alongside the Python API.

### ▶ NEXT ROUND ON THIS SURFACE — `spore-699`, APPROVED BY PHILL 2026-09-02

Alex's **A3** (the wrap lock has no owner / PID / expiry) was deliberately **NOT** shipped and the letter says so. Approved for the next round: the L2 seat's **transport-layer** design — an **age refusal** in `_tool_wrap_cancel` (read `get_wrap_started_at()`, refuse a wrap younger than a documented threshold unless `force`, returning *before* `wrap_cancelled()` is called) plus an **optional `wrap_token`** (the owner has it from `prepare_wrap`; a sibling that only read the hook line does not — so token = "prove it's mine", `force` = "override without proof"). ⛔ `confirm: true` is **rejected**: the model authoring the call authors the confirmation in the same token stream. ⚠ Transport-layer checking stays TOCTOU-racy; the store-side CAS is the real close and is **also** in `spore-699` with Alex's competing idempotent-`prepare_wrap` alternative. Both must preserve the frozen-token contract.

**22 mutants, 22 killed.** Two of the new tests were themselves defective and mutation caught both — including `"wrap_cancel" in msg`, which passes on a message naming only `store.wrap_cancelled()` because one is a substring of the other.

## ✅ 0.9.7 SHIPPED 2026-08-14 — AM-LEVELCAP. **Read this before deciding anything else here.**

`abf119f` + `0f57107` + `a34eaa2`, tag `v0.9.7`, **on PyPI**, verified by a clean-venv install FROM PyPI (not the local build). 1676 tests, mypy clean. Levain floor bumped to `>=0.9.7` in the same pass (`levain 6281cf6` — pyproject + `KNOWN_GOOD_ANNEAL` + `TEMPLATES_RECONCILED_ANNEAL`, all three move together or the `pip_floor_verdict` release-gate reports drift; it caught me when I edited pyproject without reinstalling the editable package).

**THE DEFECT, in one line: `_GRADUATION_RE` was scoped to `([23])x` and `CrystalStore._validate_level` gated on `level in (2, 3)`, so a pattern at 4x or higher was invisible to validation AND could not crystallize out.** A trap with no exit at either end. The graduation half was *silent* — a 4x line matched no branch, incremented no counter, and (because co-citation is extracted inside the graduation path) formed **zero Hebbian links**. Measured on flow's live neocortex: of 8 evidence-bearing pattern lines, **ONE was visible**; after the fix, `validated` 1→7, `direct_co_citations` 0→4, session pairs 0→6, and the real write path formed **6 associations where it had formed 0**.

⚠ **BOTH HALVES HAD TO SHIP TOGETHER — and this is the transferable part.** The graduation fix landed first, alone. That alone makes a 4x line legal and linkable in the always-loaded working set *while still denying it an exit*, i.e. it converts a silent-drop bug into an unbounded working set — the exact failure the crystal store's own docstring says it exists to prevent. **Fixing one half of a two-half trap is not a partial fix; it is a different bug.**

**THE DESIGN CLAIM THAT CAME OUT OF IT, and it should govern future work here: LEVEL and RECENCY are two independent axes.** `crystallize()` documents raising level *"monotonically — a pattern's earned high-water mark holds"*, and a mark that saturates at 3 is not a high-water mark. Level = how many times lived experience re-earned the pattern. `last_activated_on` = when it last fired. A pattern earned ten times over months and one touched yesterday are different facts and recency cannot express the first. The cap collapsed a two-axis model to one. **`MIN_PROVEN_LEVEL = 2` is the new gate** (no ceiling); `VALID_LEVELS` retained as a compat export and is *no longer the gate*. The FLOOR is unchanged and was always right — 1x is developing, not Proven.

**WHO IT AFFECTED:** any consumer whose practice counts past 3x. Levain's methodology-core teaches 1x→2x→3x-then-crystallize, so a seed-following entity never produced an affected line — **which is why this survived to 0.9.6 unnoticed, and why the floor moved while the templates did not.** 0.9.7 adds NO migration entry (newest is still 0.9.6 AM-WRAP-GENERATED): the change only *widens* what is accepted, so no adopter needs a template edit — that is what makes `TEMPLATES_RECONCILED_ANNEAL = 0.9.7` honest rather than an ack past uncovered guidance.

0.9.7 also carries the **`!!!` salience-prefix fix** (`fef6af0`), which had sat on `main` unreleased since 08-11 behind an empty `[Unreleased]` — the LOW in the Diogenes block above, now discharged.

### ⛔ WHERE FLOW IS, because these decisions cannot be made without it

- **Flow stopped crystallizing on 2026-06-08 and the cap is why.** `memory.crystal.json` holds **14 patterns, all crystallized 2026-06-06, `last_activated_on` never updated** — so every one reads `dormant` and nothing has re-warmed in two months. Meanwhile a **hand-maintained 26-name working-set prose line** grew inside the always-loaded neocortex doing the same job, with **six names in BOTH surfaces**. We built a prose imitation of a store we already had and let the real one go cold — `two_things_that_should_be_one_computed_by_two_pieces_of_code`, subject: our own memory architecture. It was not laziness: `crystallize()` refused any level above 3, and our patterns were past 3 by June, so **the OUT path was literally closed to exactly the patterns that most needed it.**
- **Migration is `spore-530`**, for a consolidate (wrap-time single-writer act). Route each name CONSTITUTION / CRYSTALLIZE / COMPOST, crystallize **with its real level, not a flattened 3**, then delete the prose line. Flow's `global/CLAUDE.md` Pattern Graduation section is corrected to describe the real lifecycle.
- **The association graph sat at 165 rows through the whole outage.** `association_stats().total_links` rising on the first consolidate that graduates a 4x+ pattern with co-cited evidence is the oracle that the associative layer is alive again. **Check it before trusting any graph-derived measurement.**

### ⚠ TWO THINGS THIS FORCES ONTO THE SLICE-C DECISION BELOW

1. **The corpus the §9.2 replay harness measured was produced under the cap.** Any pattern that reached 4x+ contributed NO co-citation edges for as long as it was above 3. Before re-reading the NO-GO-LEAN or building step-3's labeled probe set, **establish whether the graph's sparsity is a property of the world or an artifact of this bug** — the harness cannot tell those apart, and its read was taken on the affected corpus. This does not overturn the lean; it means the lean's *input* is now suspect and re-measuring after a few post-0.9.7 wraps is cheap.
2. **The PMB gap (influence measurement) is unchanged and still real** — anneal validates GROUNDING and never measures INFLUENCE. Nothing today touched that. But note the shape rhyme worth carrying into that design: today's bug was an instrument that reported success while measuring nothing, found only because a *different* instrument (AM-LINKGATE) fired an honest warning it could not itself explain. An influence metric needs the same property — **it must be able to name the regime it is invalid in**, which is exactly the line PMB's `earned_memory.py:217` already gets right.

> **▶ LIVE TRACK = Slice C step-3 — the labeled probe set + the usefulness-JUDGE.** Slice B (AM-LINKGATE-DECAY) is BUILT and ships in SHADOW MODE (0.9.0, public in 0.9.6). The §9.2 replay harness SHIPPED 2026-06-30 (`scripts/slice_c_replay_harness.py`) and READ **NO-GO-LEAN**: do NOT wire the raw pattern→pattern hop. It stays a LEAN and not a verdict because the corpus has zero outcome labels — **step-3 is the only thing that closes it.** (The old four-signal GO/NO-GO gate — edge-floor / co-surface canary / precision / stability — is SUPERSEDED by the harness's own measurement and is no longer the gate.)
>
> ✅ **`spore-722` NOW CARRIES STEP-3 (planted 2026-09-04), AND THE GATE IN THIS BLOCK WAS WRONG.** It is not "after 675 yields data" — the harness names its own blocker at `slice_c_replay_harness.py:24-32`: *"a labeled gain cannot be computed from the corpus alone — that is build-order step-3 (**a Phill-labeled probe core** + LLM-judge for volume)"*. `cited_used` / `outcome_signal` are NULL across the whole corpus. **More patterns do not create labels**, and 675 is largely done anyway (29 crystals, 422 pattern links, measured 09-04) — the data was already here and the lean did not move. ⚖ **The real gate is Phill's labelling time.** ⛔ Answer FIRST: *what would we do with a GO?* The harness already read NO-GO-LEAN; building a judge to confirm a lean nobody would act on is theatre. **Flow's read: it probably resolves to SHELVE** — the associative layer is alive and serving recall (50 retrievable of 422), and a GO only WIDENS traversal. ⚠ **KILL CRITERION `2026-09-10`:** unanswered by then → Slice C is shelved formally, harness retires to reference, NO-GO-LEAN stands as the final read.
>
> **⚠ (SUPERSEDED) NO OPEN SPORE CARRIES STEP-3.** The spore that carried it was composted `done` on 2026-07-14 with step-3 still UNBUILT — verified against the harness's own docstring (`slice_c_replay_harness.py:24-29`: `cited_used` and `outcome_signal` are NULL across the entire corpus, so a labeled §6 gain "cannot be computed from the corpus alone — that is build-order step-3"). **This plan line is now the only clock.** Either re-plant a spore or formally shelve Slice C.

---

## ▶ INTEL — routed 2026-07-23 (Daemon overnight + Anansi frame), pre-work for DISCUSSION not action

> Three overnight items that bear on anneal, routed here for the next @project load. Read per
> the house rules: Daemon mines mainstream for ideas that improve OUR approach (never a
> feature-vs-feature benchmark); the Anansi frame is a POSITIONING CONFIRMATION, not a threat.
> None of this is a decision — it's material for the next session's judgment.

**1. AutoIndex (arXiv 2607.18603) — representation-as-optimization-target, aimed at the INGESTION layer.** AutoIndex holds the retriever fixed (BM25) and searches over *executable programs* that slice / enrich / normalize / reweight documents BEFORE indexing — treating the document representation as the primary optimization lever. Reported +8.4% mean Recall@100 / +8.3% nDCG@10 over static full-document BM25 (largest task gains +30.5% / +43.6%), via a validation-guided loop: agents diagnose the current representation's retrieval failures → synthesize candidate transform updates → retain only updates that improve retrieval. **The transfer to anneal (an IDEA for our approach, not a feature to copy):** if anneal treats episode→store conversion as a *fixed* preprocessing step, this says that step may be where the biggest recall gains hide — an executable *representation program* over ingestion, tunable. And the diagnose→synthesize→retain-if-better loop is structurally what anneal's own immune/graduation eval already does, one layer up. **Where it bites:** Slice-C is a *routing/recall-policy* build; AutoIndex is about *what gets stored and how it's shaped* — a different, upstream lever. Worth a real look as a distinct workstream, NOT a Slice-C input. (Judge before adopting — same measure-first discipline; `measured_ground_truth_beats_layout_theory`.)

**2. Anansi's developmental-memory frame — the market shipped TWO of the three halves; the one it did NOT ship is anneal's seam. (positioning CONFIRMATION.)** Anansi's overnight synthesis: *last week the market named the pain (observational-vs-causal), this week it started shipping the fix — but only along the procedural/effect-gate axes.* (a) Anthropic shipped **"Record a Skill"** in Cowork (learn a procedure by screen-recording yourself) = the frontier vendor shipping **procedure-capture, top-down** — Anansi's honest read (conceded against interest): this genuinely takes a bite out of "the operator authors the known procedure," resolving spore-020's "model-as-runtime, procedure-as-program" in the affirmative. (b) A swarm of fresh tiny repos shipped the **effect-gate / verify-the-loop** layer bottom-up (LoopGain, ActionRail, loopbreaker, flightwake, agentic-review-gate) = spore-021 (effect-verification as a named feature) shipping as actual software. (c) **"valuemaxxing vs tokenmaxxing"** named the metric shift at the enterprise-buying layer. **What stayed UN-shipped: the immune-system / developmental half** — graduate/demote over lived time, carry the causal *why*, the four-layer store with a citation-validated graduation gate. That is exactly anneal's layer, and the market shipping record-replay + verify-gates *around* it is confirmation the seam is real and correctly located — the moat is NOT record-replay (commoditizing) and NOT the effect-gate (productizing bottom-up), it's the developmental/immune memory. Fold into the spore-020 / spore-021 read; sharpen the positioning line at the next Levain/anneal session. (Do NOT frame anneal feature-vs-Record-a-Skill — category, not competitor.)

**3. SLPO (arXiv 2607.19691) — the "stopping head" primitive. PARK (lower priority).** Outcome-reward RL for latent reasoners; the one transferable primitive is a **correctness-supervised stopping head** — a learned "I've thought enough" signal refined by outcome rather than a fixed heuristic. IF the augmentation substrate ever needs to decide *how much internal computation to spend before surfacing a candidate to the human*, that's a cleaner interface than fixed token/step budgets. Not a now-thing; parked as a primitive to remember.

⚠ **ITEM 2 ABOVE IS NOW PARTLY FALSIFIED — see the PMB block below. The developmental half DID ship, in one place. What survives is the narrower and better claim.**

---

## ▶ INTEL — routed 2026-07-30 (morning ritual; source-verified, not seat-relayed). **PMB — the developmental half SHIPPED somewhere, and it is AHEAD of us on one axis.**

> Provenance: Popper attacked the thesis with *"the substrate is a 1-curl install"* (OptMem); Anansi countered that the trending flood ships only the OBSERVATIONAL half and *"the developmental axis stays nearly the only empty lane"* — **and flagged its own falsifier honestly** (*"inferred from descriptions, not code audits"*). Both were read against **actual source**, so the below is not a seat relay. Route: `spore-399` (Friday 07-31 anneal upgrade session) — this is **input #4**.

**OptMem — NOT a threat, confirmed against source. Observational by design and honest about it.** 903 stars, but **859 LOC, one file**: an append-only `LOG.txt` of fixed-width 320-byte `(date, text≤280)` records plus a `TREE/` of range summaries. **No importance, score, weight, confidence, access-count or evidence column exists.** The entire write path is `log_append` → print. Dedup is a *prompt instruction to the model* (*"Do not register redundant memories"*), not a mechanism. Its one decay-shaped behaviour is **not evidence-driven** — it is resolution loss from compression the AGENT performs (`nap` asks the model to squash a range), so **a memory's fate depends only on its POSITION IN THE LOG, never on whether it proved true or useful.** Genuinely good idea worth stealing, orthogonal to memory quality: fixed-width records make the tree **byte-addressable**, so `wake` at ~1M memories is ~0.03s and `cover(T, WAKE_LINES)` picks a *frontier* — finest detail near the present, coarsest for ancient history — i.e. **logarithmic-detail recall under a fixed token budget.** (Also independently converged on our single-writer rule: subagents are forbidden to write because *"it cannot judge what is already known."*)

**PMB (`github.com/oleksiijko/pmb`, `pip install pmb-ai`, Apache-2.0, 283★, ~52,100 LOC, last commit 07-10) — GENUINELY DEVELOPMENTAL. Four of the five mechanisms.**
- **Graduation by re-access:** `PROMOTE_WORKING_TO_EPISODIC_ACCESS = 2`, `PROMOTE_EPISODIC_TO_SEMANTIC_ACCESS = 7`, with a dogfooding note that 3/10 was lowered to 2/7 because *"Two repeats is a more honest 'this is a recurring topic' signal."* **Tier buys half-life** — working ~1.94d → episodic ~46d → semantic ~346d. Same 1x→2x→3x shape as ours.
- **Decay + retirement:** `TIER_DECAY_FACTORS` applied daily; `archive_cold` retires on **evidence of non-use** (`access_count == 0`), reversible, with lessons/goals/preferences structurally exempt.
- **PROOF-OF-INFLUENCE — the thing we do not have.** `lesson_surfaces.followed` is **THREE-valued** (1 followed / 0 ignored / **−1 not-applicable**) precisely because *"a rule that never pertained to the work must not count as 'not followed'. Otherwise the metric measures how broadly auto-recall surfaces (noise), not how well relevant rules are followed."* That is a **correctly-specified denominator**, which is the part this class of system normally gets wrong.
- **Causal lift, model-free, stdlib-only** (`health/earned_memory.py`): joins surfaced-lesson → turn outcome (only turns with a **mechanical** oracle — tests pass/fail, build, deploy, red→green), yielding per-lesson `success_rate`, `lift` vs a no-lesson baseline, `followed_lift` (a within-lesson followed-vs-ignored contrast), and a `causal_verdict` of helps/hurts/inconclusive gated on **non-overlapping 95% Wilson intervals** (chosen because *"it stays inside [0,1] and is sane at the tiny n the outcome signal actually produces"*). Its stated purpose: *"which memories pull weight, which are dead weight, which are HARMFUL (precede failures)."*
- ⚠ **THE LINE TO READ BEFORE FRIDAY**, `earned_memory.py:217` — the per-lesson lift **"must not drive ranking/decay on its own"**, because lift reads negative for a real positive effect (lessons surface on the HARD turns). **The instrument names the regime it is invalid in and refuses to act inside it.** That is `a_check_that_cannot_name_the_world_it_fails_in_is_not_a_check`, and **PMB passes it.** Treat this as a peer, not a competitor.

**WHAT PMB DOES *NOT* HAVE — and it is exactly our seam.** No **citation-validation**: a lesson never has to cite an episode to level up. Promotion is **corroboration-by-RETRIEVAL**, not corroboration-by-EVIDENCE — so nothing there rejects an ungrounded graduation the way our immune system does, and nothing demotes on a failed explanation-grounding check. `accumulate-into-policy` exists but is **LLM-mediated** (`distill_lessons.py` prompts a model for *"durable, transferable rules"*), not structural.

**▶ THE POSITIONING CORRECTION (do this before it reaches a launch narrative).** The 07-23 line — *"the developmental half stayed UN-shipped"* — **is false as written, and PMB's last commit predates it.** Anyone who knows PMB would break that sentence in one reply. The surviving claim is **narrower, checkable, and stronger**: *graduate/demote over lived time HAS shipped elsewhere; **citation-validated graduation has not.*** Ours is the only one where **a pattern must cite real evidence, and the explanation must lexically ground in the cited episode, or it does not level up.** Keep saying that; stop saying "developmental."

**▶ THE BUILD INPUT, AND IT IS A REAL GAP.** **anneal validates GROUNDING but never measures INFLUENCE. PMB measures INFLUENCE but never validates GROUNDING.** Complementary failure modes — we can say a graduated pattern was honestly earned, and we **cannot say it ever improved an outcome.** We have no `earned_memory.py`. Note this lands directly on the **live Slice-C blocker**: §9.2 read NO-GO-LEAN *only* because the corpus has **zero outcome labels** (`cited_used`/`outcome_signal` NULL), and step-3 (the labeled probe set) is the thing that closes it. **PMB is a working, model-free, stdlib-only demonstration of how to get an outcome label without a human labeling it** — bind to a MECHANICAL oracle (tests/build/deploy) and Wilson-gate the small-n. Whether that is a shortcut to step-3 or a different instrument entirely is Friday's judgment call — but it is the first concrete answer to "where do outcome labels come from" that this project has been handed.

**▶ HOW IT COMPOSES WITH `spore-399`'s THREE:** GEAR scores **what came back** (evidence-contribution vs parroting, intrinsic). InMind attacks **what never came back** (the indirect-association omission). **PMB scores whether what came back CHANGED ANYTHING (extrinsic/outcome).** Three different cuts at one question, and only PMB's is already running in production code we can read. Same discipline as the others: **WebFetch/read the source first, judge before adopting.**

---

## ▶ §9.2 replay harness (step-2) SHIPPED 2026-06-30 → read = NO-GO-LEAN (don't wire the raw hop); NEXT = labeled probe set (step-3) + a hub-penalty HYPOTHESIS to measure

**SHIPPED 2026-06-30:** `scripts/slice_c_replay_harness.py` + test (flow-side, sibling of `pattern_graph_oracle.py`; imports anneal read-only — **no anneal version bump / publish**: the anneal release was decoupled from Slice C and shipped independently as 0.9.6). Full 4-layer apparatus: L1 + L2 + L3 (codex non-replaceable + complement + nemotron — *nemotron has since been RETIRED from the code mesh, 2026-07-10; `gpt-oss` holds that seat now*) + L4. The corpus is LIVE; re-run on demand (`python3 scripts/slice_c_replay_harness.py` → `state/slice_c_replay_report.json`). It is the CONTINUOUS instrument (§2), not a one-shot.

**THE READ (2026-06-30, ~530-receipt corpus):**
- **Power:** measure-first found the §13 ~135/wk projection did NOT survive the AM-RECALL-IDF regime (81% empty-exposed) — `measured_ground_truth_beats_layout_theory` fired again. After the apparatus removed a SELECTION BIAS (the old `is_usable` >=2-production-exposed filter conditioned on the measured backend + dropped the 1-seed queries where the hop has the MOST room — codex L3 HIGH#2), eligibility is derived from REPLAYED `c_ev>=1` → **101 eligible / ~76 query-classes (above the 39 floor) = POWERED.**
- **Signal:** across ~76 query-classes the hop surfaces only **9 distinct marginal patterns** (watermark/tasks/FOUR_LAYER…), mean degree-pctile 0.22 = **HUB-WARD** = the **§7 spurious class** (top-degree flow-meta). The lean rests on these **denominator-free** signals (small marginal vocab + hub-ward pctile); marginals/class=0.118 corroborates query-invariance but is OPTIMISTIC — the fingerprint over-counts noun-varied boilerplate (`load projects {bridge|levain}`), inflating the class count + deflating the ratio → it OVERSTATES invariance (codex L3 caught my reversed robustness claim).
- **Verdict = NO-GO-LEAN:** do NOT wire the raw pattern->pattern hop. It stays a LEAN (NOT a clean NO-GO) because the corpus has **zero outcome labels** (`cited_used`/`outcome_signal` NULL across the corpus → §6 labeled-usefulness gain is uncomputable). Apparatus caught + corrected an early over-attribution: the concentration is NOT cleanly "graph degree" (a frequent marginal like `positioning` isn't reliably a top hub across snapshots — L2 F1); the query-INVARIANCE across a powered diverse set is the robust signal.

**▶ NEXT (to a CLEAN read):** (1) **labeled probe set** (step-3, §6) — Phill-labeled core + LLM-judge for volume — the only thing that turns the LEAN into a verdict; (2) **MEASURE a hub/degree penalty on the Slice-B graph AND/OR query-conditioned edge validity** as the candidate fix — a HYPOTHESIS, not adopt (§12 already REFUTED naive degree-bias once in favor of term-frequency IDF; don't repeat the over-attribution one tier up); (3) re-run the harness after each. Deferred enhancements: a true semantic query-class taxonomy (the fingerprint over-counts noun-varied boilerplate); the snapshot-at-window-START fidelity caveat (L2 F5).

**Harness CONTRACT (from the snapshotter docstring — load-bearing, do not drift):**
- Bucket on `(pv, hwm, query_date)`, NOT hwm alone (lazy decay fragments otherwise).
- A receipt whose `(pv, hwm)` has NO exact snapshot MUST be EXCLUDED (never nearest-matched).
- **`pin-not-rebuild` is sufficient ONLY IF receipts-per-hwm-bucket has A/B power — count receipts-per-hwm FIRST; if too sparse, build the as-of rebuild.** (Measured 2026-06-21: a single natural bucket ≈ 8 usable queries = SPARSE; **pooled multi-snapshot** — pin at every drain + closed-form decay + paired-gain pooling — reaches ~135 usable/week WITHOUT the deferred as-of CQRS rebuild. Phill RATIFIED pooled-multi-snapshot. As-of rebuild stays blocked anyway: raw co-surface events truncated post-drain, decay wall-calendar, GC deletes.)
- **The gain gate is the FULL influence chain, not just exposure:** retrieval exposure → cited/used subset → downstream claim/action → outcome GAIN vs a baseline counterfactual. The receipt carries the upstream half (`exposed[].source`, `query_text/date`, `graph_version/hwm`); §9.2 scores the DOWNSTREAM half. Unused exposures + failed outcomes = NEGATIVE evidence. Weight by OUTCOME, not exposure (Joachims counterfactual-LTR — co-surface signals are exposure-biased). Anti-spoofing: bind to substrate state, not a performance artifact.

Design: `slice_c_gain_instrument.md` (§4/§5/§9/§11/§13). **Deferred (NOT blocking):** `spore-146` (IDF per-call config / distribution-relative bar) + `spore-148` (AM-READONLY-FAILFAST fail-fast hardening).

## ▶ Slice-C DESIGN INPUTS (live — fold into the §9.2 build)

- **DESIGN REFRAME (2026-06-24) — Slice-C is a ROUTED MULTI-CHANNEL recall policy with a receipt per channel, NOT graph-replaces-lexical.** Route by bottleneck (causal experience → graph-like, persona/detail → flatter) onto anneal's existing source taxonomy (`keyword | evidence_edge | graph_hop | afferent`). Slice-C = wiring the graph hop in as ONE receipted channel among several — **the moat is the receipt-governed ROUTER, not the graph.** keyword+evidence-edge stay first-class; graph recall is an *added constrained channel*, never the new authority.
- **BRITTLENESS CAVEAT (hold as a design constraint, NOT settled — needs independent replication):** "The Price of Meaning" (arXiv:2603.27116) — semantically-organized memory has a structural interference tradeoff; pure semantic retrieval forgets/false-recalls *smoothly*, but explicit reasoning on top can turn that BRITTLE, and immunity needs either leaving semantic retrieval or external verification (= the receipt). So treat graph/semantic recall as constrained + receipted + baseline-checked, never authority. Reinforced by MemConflict (arXiv:2605.20926): pattern-graph edges need conflict surfaces / query-conditioned validity (current-vs-earlier, applicable-vs-distractor), not just positive links; answer-correctness can diverge from retrieval quality.
- **THE GAIN-INSTRUMENT is the Protocol-Memory ANTIDOTE (Phill, 2026-06-17):** kill/keep is never blind again → the ~Jun-28 read is a CONTINUOUS instrument read, not a wait-gate; instrument-WHILE-building. CL-Bench (arXiv:2606.05661): naive ICL can BEAT dedicated memory → measure outcome GAIN vs stateless/non-graph/full-context baselines, not edge-plausibility.
- **PORTABILITY WEDGE (OTel — the PRINCIPLE stands; the standard it was named for is DEAD).** The move: the DecisionInfluenceReceipt can export a THIN OTel-compatible projection (`gen_ai.memory.*` spans) WITHOUT ceding schema authority — `canonical_object_model_plus_replaceable_surfaces` applied to a *standard*. **⚠ DO NOT cite OTel GenAI semantic-conventions #200 as a live standard the receipt aligns to** (the original 2026-06-24 framing did, and it is wrong): verified 2026-06-29, **#200 was CLOSED as "not planned" — the full-lifecycle shape (store/retrieval/get-by-id/update/decay/expiry/deletion) was DECLINED**; a narrower CRUD shape merged separately with **no decay/expiry**. Feeds the Bridge 2c receipt-schema. Full record + the honest use of the decline → `decision_influence_receipt_contract.md` (portability note).
- **DAEMON OVERNIGHT ORE (2026-07-08, quarry not scoreboard — 6 papers, research-claims-UNVERIFIED):** (a) **paired-gain gate corroborated** — MemGym / EvoMemBench / Agent-Native Memory all score memory-on vs memory-off under the SAME reasoner → promote `graph_hop` only on paired per-query gain over keyword/evidence-edge baselines (EvoMemBench: no memory form wins across settings, memory helps most when context is insufficient); reinforces the §9.2 outcome-gain gate — don't promote on graph prettiness. (b) **typed memory IR** — MemIR names "provenance-role collapse" (flat memory merges evidence/cues/inferred-claims/temporal-refs until authority is unreadable); structural move = write memory as TYPED ATOMS (evidence|cue|claim distinct), bind receipt `cited_used`/`provenance_spans` to typed atoms, NOT narrative chunks. (c) **provenance UI = ACTION GATE, not a panel** — PaperTrail (CHI 2026, n=26) LOWERED trust vs citation-style + changed ZERO editing behavior under time pressure → show actor-first-estimate / claim-match / unsupported-flags / confidence / drift ONLY where the user must make/defer a concrete move (Bridge / FlowPoker). (d) **write policy = as load-bearing as retrieval scoring** — MPBench: memory-poisoning ASR 50.46%, injection detectors miss weak-signal memory attacks (payload looks like legit facts) → Slice-C receipts need write-authority provenance + scope-limited write policy + exposed-unused/used-bad negative evidence + post-write monitoring; graph-consuming recall can't become AUTHORITY without receipt-backed write gates. (e) **learned memory critic = regime-scoped accelerator only** — MemGym MemRM (sub-second compression-quality classifier) has partial OOD generalization → selective-classification w/ coverage/abstain telemetry, never an authority shortcut. **TOPOLOGY confirmed:** daemon could NOT run oracle/replay on argushub (anneal not importable, stale Store API, flow canonical store absent on hub) → Slice-C GO/NO-GO runs on the LAPTOP canonical store only.
- *All daemon-sourced arXiv = research claims, NOT flow-verified — verify before any external citation. Bridge-side receipt framing → `projects/bridge/next.md`.*
- **GEAR + CodeAlmanac (overnight 2026-07-22, UNVERIFIED) → dedicated upgrade session, spore-399.** GEAR (arXiv 2607.19345) = evidence-aware retrieval scoring (grounding reward + distractor penalty for the "repetitive copying in long context" failure) → score recalled fragments by evidence-contribution vs parroting (Slice-C router / receipt negative-evidence signal). CodeAlmanac = auto-distilled queryable repo memory (episodic→semantic, time-vs-commit trigger, cross-dev shared memory = the Levain multi-agent shape). Full capture + open questions: `gear_codealmanac_prework.md`.

## ▶ Deferred / self-paced (NONE block the B→C track)

- **content-store v1** (`spore-093` — LIVE, this is its clock) — per-wrap neocortex text + `as-of`/lineage viewer; anneal-side dep of the Levain projection-history viewer v1. **UNBUILT, verified on disk 2026-07-14:** `anneal_memory/store.py:756` — the `wraps` table carries **metrics only** (`episodes_compressed`, `graduations_validated`, `citation_reuse_max`, `continuity_chars`, …); there is **no content column, no content-store module, and no as-of/lineage query anywhere in the library.** (This spore was composted `done` on 2026-07-14 and has been RESTORED — it was never built.) **Gate = pure data-maturity, SELF-PACED** (build when governed-write history makes a searchable full-projection store earn its bytes); explicitly **NOT** Tony/second-operator-coupled. restore stays KILLED (the continuity text is a PROJECTION of the 5-layer substrate — restoring desyncs it; you regenerate a projection, you don't restore it). v0 digest-delta viewer already shipped.
- ⚖ **AM-LINKGATE BLOCK half — RULED *BUILD* BY PHILL 2026-09-04, AND IT HAS A CLOCK NOW: `spore-721`, `next: 2026-09-11`.** ▶ Two constraints are non-negotiable, both learned 09-03/04: **(1) gate on ≥2 PAIR-CAPABLE graduations**, never "has graduations" — a single-graduation wrap CANNOT form a pair (`pattern_associations.py:462` counts a 0-or-1-name recall as *applied* while forming none), so the naive predicate would REFUSE correct work and lose its compression; **(2) fail closed WITH an explicit, loud escape** — this refuses a memory SAVE, and a save-path gate with no override is a single point of failure for the whole substrate. ⚠ If it slips past 09-11 unstarted, that is DATA: retire the idea formally rather than re-plant a third time. **SUPERSEDED ENTRY BELOW:**
- **AM-LINKGATE: the structural BLOCK half — UNBUILT, and it has NO CLOCK (surfaced 2026-07-14).** The AM-LINKGATE spore proposed TWO halves: (a) `prepare` emits a REQUIRED co-citation step, and (b) **`save` BLOCKS** when `associations_formed + associations_strengthened == 0` on a session WITH graduations. **Only the WARN half shipped** — verified on disk: `anneal_memory/continuity.py:2584` Signal C is explicitly *"a discipline reminder, not a proven defect"* (it warns; it never refuses). The spore was composted `done` on the WARN half alone, so the structural guard now has no clock and the interim defense is pure discipline (`feedback_wrap_underwires_associative_layer.md`) — exactly the `structural_invariants_beat_discipline` inversion the gate existed to close. Decide: build the BLOCK, or formally accept WARN-only and retire the idea.
- **AM-CHIPSCHEMA** — a trusted-single-operator schema profile: for one trusted operator (Chip, most solo adopters) most of the immune system (replay block, citation-gaming flags, multi-tenant isolation, poisoning resistance, tool-integrity hashing) is *dormant insurance*; active-value subset = sycophancy-drift gate + catastrophic-shrink gate. Candidate: adversarial layer turned down, `Understanding`-at-primacy ordering, simplified concurrency. A *subset*, not a knock on the product (which must defend the multi-tenant/stranger case). Sibling: **AM-ATTENTIONZONE** (first-class nested attention-zone schema semantics — the substrate the Levain control-pane would render/reorder; pairs with AM-VIZ).
- **v0.5 design candidates** (all real architectural conversations, NOT patch work; defer until the B→C track resolves): I-4 branchable continuity/wrap-diffs · I-6 execution-memory parallel store (biography vs procedure) · I-9 outcome-linked usage (value-provenance ledger over the wrap audit chain) · I-10 trainable-phi as a resilience/coherence training signal (NOT a consciousness claim — mechanism over metaphysics) · I-11 limbic layer feeds attention-routing (affect-for-surfacing, vs the built affect-for-memory-strength). The broader I-1..I-9 frontier-validation backlog (2026-05-27) is mostly closed/superseded by the shipped Slice A–C arc (I-1 README audited clean 2026-06-02); full detail in git history.
- **Other parked:** `spore-169` (set_disposition next-field CAS, pre-existing-class MED, cold) · `spore-047` AM-PATTERN-ALIAS (pattern-NAME identity — rename forks history; batch with the next anneal touch) · `spore-048` AM-GAUNTLET (published adversarial immune-benchmark + the real Hebbian-surfaces-cold-patterns recall-quality test; trigger = before the next public methodology push OR a 3rd autoimmune sibling) · `spore-088` (flow→Argus inbox instructions don't reach his `codex exec` wrap) · the GPT-5.5 cold-read BACKLOG (2026-06-07). **AM-IDALIAS + AM-BACKFILL ✗ DROPPED** (dogfood discriminator — flow N-of-1; if the flow-id→anneal-id hand-rule rots, the structural fix is flow-side at `capture` time, not an anneal-core change). **DATA-GATED:** flow's conceptual-corpus end is MEASURED (Step-C 06-08 — keyword fails both directions, validating the Hebbian build: relevant-recall ~2%→~25-40%, all 10 stone-cold abstract patterns now reach conceptual prompts); the entity-dense end still owes Chip's contrast before any public positioning claim.

- **Deferred hardening (folded from the spore store 2026-07-14):**
  - **Resume-safe consolidate baton** (`anneal_memory.sessions` baton_holder/live_sessions/heartbeat + `scripts/anneal_dualwrite.py`). BUG (confirmed 2026-07-07): a `SessionStart:resume` rotates the CC `session_id` (observed `5a0043d6`→`1d8831d2`), orphaning the baton — it stays held by the now-dead pre-resume id, and the gate flags "holder not live." TESTED + FALSIFIED that claude-agents *switching* causes it (session_id was STABLE across a switch-away-and-back); the trigger is specifically a genuine RESUME of the baton-holding head (laptop sleep/wake, reconnect, background-job resume, compaction reload), NOT navigation. SEVERITY LOW: self-announcing (gate prints holder-not-live), fail-safe (blocks a consolidate, never corrupts one), one-command recovery (`anneal_dualwrite.py baton claim`). FIX (`structural_invariants_beat_discipline` — don't rely on remembering to reclaim): identify the consolidating head by a marker STABLE across an id rotation, not the ephemeral session_id, so a resume RE-ATTACHES the head role. Candidates: (a) match/inherit the baton on pid if pid survives a resume (VERIFY first — a resume may spawn a new process); (b) a persisted head-designation the resumed session auto-reclaims on `SessionStart:resume`. INTERIM GUARD (operational, in place today): whenever session-init shows `SessionStart:resume`, the head runs `baton status` and reclaims if it's the head — proven to catch it 2026-07-07.
  - ✅ **AM-WRAPSTARTED-XLOCK — CLOSED 2026-09-04 by the 0.9.9 locking work, verified by execution.** This item named its own two fix candidates: *"a cross-connection write lock on `wrap_started` (BEGIN IMMEDIATE / `continuity_lock` acquired early) OR atomic CAS-on-create (`wrap_started` fails if `wrap_started_at` already set by another connection)."* **Both now hold.** `wrap_started` opens with `BEGIN IMMEDIATE` and the guard read happens inside that transaction, so the check-and-set is atomic ACROSS CONNECTIONS. ⚡ RUN, not reasoned: two Stores on one db, first writes token `a`, second connection attempts `b` → **REFUSED with `WrapInProgressError`, surviving token is `a`.** The described defect ("two near-simultaneous prepares can both reach `wrap_started` and the second overwrites the first's token") is structurally impossible now. Guarded by `test_a_peer_cannot_slip_a_wrap_between_the_guard_read_and_the_writes`, mutation-checked. ⚠ Closed as a SIDE EFFECT of fixing a different defect and nobody would have noticed — found only because the improvements list was re-read against today's diff. **ORIGINAL ENTRY (superseded):** (AM-CONSOLIDATE-EFFERENT L3 codex HIGH follow-on; `continuity.py` `wrap_started`). Harden `wrap_started` against the cross-connection OVERWRITE at the ROOT so a tokenless save can't land one session's text under another's snapshot. The doc/usage level is fixed (CAS hard only WITH the wrap_token round-trip; flow round-trips), but the underlying gap is PRE-EXISTING: `prepare_wrap` holds no flock, so two near-simultaneous prepares (or a baton reclaim mid-flight) can both reach `wrap_started` and the second overwrites the first's token. Fix candidates: a cross-connection write lock on `wrap_started` (BEGIN IMMEDIATE / `continuity_lock` acquired early) or atomic CAS-on-create (`wrap_started` fails if `wrap_started_at` already set by another connection). Own full apparatus. Surfaced sharper by the parallel-consolidate use case.
  - **Clock-inject `upsert_pattern_history`'s UTC wall-clock fallback** (`store.py` ~2623, `_dt.now(_tz.utc)`) — the last un-pinnable wall-clock in the `seen_at` path, which is why its tests can only assert "whatever UTC says now" (`662409e`). Structural end-state: invariant beats per-test clock-discipline. TRIGGER: build ONLY if `seen_at`-less wall-clock fallbacks multiply beyond this single site; not worth it for one call site. Surfaced by codex L3 + L1/L2 during the 2026-06-15 CI test de-flake.
  - **Two carried-forward 3x patterns still lack `[provenance:]`** — `canonical_object_model_plus_replaceable_surfaces` + `partnership_challenge_is_bidirectional`; `grep -c 'provenance:' ~/.anneal-memory/memory.continuity.md` = 0. Backfill on a consolidate (`partnership.md` is the graduation home for the second).

---

## ▶ MIGRATED FROM THE SPORE STORE — 2026-08-27

> ⚖ **WHY THESE ARE HERE NOW.** `spore-338` (scoped 2026-07-15, ratified by Phill 2026-08-27) named the
> defect: the spore store had been silently breaking flow's own rule — *"NEVER track project/system
> development tasks in continuity; those live in project files."* These nine were anneal engineering
> filed in an operator inbox. **Each was PREMISE-CHECKED against disk before it moved** — Phill's
> explicit condition: *"many of these are stale or based on old premises so all of them need reviewed
> before dumping anyways."* A dead premise does not become correct by being written into a project
> file; it becomes HARDER to catch, because project memory reads as settled context while the spore
> store reads as a queue. The check result is stated on every item. Spore ids are kept as the trail.

- **AM-READONLY-FAILFAST** — the `read_only` Store open inherits Python sqlite3's default 5s
  `busy_timeout` (verified `PRAGMA busy_timeout=5000`); codex L3 caught it during the spore-104 dep-1
  build, refuting an L2 "no busy_timeout" claim. Under WAL a pure-SELECT recall reader effectively
  never contends, but on rare contention (a wrap's `_init_schema` DDL, a WAL checkpoint's brief
  exclusive moment) the **per-prompt hook can STALL up to 5s** before its try/except degrades. FIX =
  `busy_timeout=0` (or `connect timeout=0.0`) on the read_only branch in `Store.__init__` before
  `PRAGMA query_only=ON` → fail FAST and degrade rather than stall; writer unchanged. PRE-EXISTING,
  not a dep-1 regression. First check no read_only consumer WANTS the 5s retry. TRIGGER: next touch on
  `Store.__init__`/read_only, or a measured prompt-stall.
  ✅ **PREMISE VERIFIED 2026-08-27 — and the codebase itself points here.** `pattern_associations.py:347`
  reads verbatim: *"(A fail-fast `busy_timeout=0` on the read_only open is a real but SEPARATE anneal
  hardening, not this primitive's scope — spore.)"* Unbuilt, correctly scoped, still live. `spore-148`

- **episodic.py NEEDS A WRITER-SIDE GUARD AGAINST SHELL BACKTICK DAMAGE — the doc fix is a mitigation.**
  `episodic.py write` with POSITIONAL content goes through the shell, which EXECUTES anything inside
  backticks and substitutes the result, so a finding citing `` `route_diogenes.py:412` `` arrives with
  that span DELETED. **Three of four parallel lanes hit it on first contact (2026-08-16)** — a property
  of the interface, not four mistakes. WHY IT IS WORSE THAN MANGLING: it is silent at every surface that
  checks. The write reports success, the id returns, and `anneal_dualwrite capture` verifies that id
  landed — all three confirm the record EXISTS, none confirms it is INTACT. The spans most likely to be
  backticked are paths, command forms and identifiers, **so the damage strips the GROUNDING and leaves
  the CLAIM standing** — the worst possible half to lose in a store whose whole value is grounded
  evidence. Append-only, so the only repair is a superseding episode.
  THE TWO STRUCTURAL OPTIONS, neither built: **(a)** the WRITER warns on probable shell-substitution
  damage — detectable signature is a DOUBLE SPACE or orphaned punctuation gap where the backticked text
  was, plus an unbalanced/absent backtick count in otherwise code-dense content. Heuristic, not proof, so
  **WARN, never REFUSE**, or it becomes a new false-alarm gate. **(b)** make the positional form
  structurally unavailable for multi-line or code-dense content — require `--body-file` above N chars or
  when content contains a `:`+digits (file:line) shape.
  ⚠ Deliberately not built on 2026-08-16: unvetted machinery on the WRITER, on the day the neocortex is
  consolidated from that store. Build it where it can be mutation-checked against real episodes. If (a)
  ships it must not fire on legitimate prose, or fail-closed becomes fail-ignored.
  ✅ **PREMISE VERIFIED 2026-08-27.** `grep` over `scripts/episodic.py` finds the hazard documented at
  lines 15–30 and 70 (`--body-file` leads the usage text) and **NO guard code**. The doc mitigation is
  all that exists. `spore-545`
  ⚡ **SECOND INSTANCE, SAME DAY, DIFFERENT TOOL — AND IT WIDENS THE ITEM.** Later on 2026-08-27 the
  identical defect fired in **`spores.py update --add-note`**, eating `` `handoff` `` out of three
  notes and leaving *"because sorted first unconditionally"* — the claim standing with its subject
  deleted, which is precisely the failure shape this item describes. **So the scope is not
  `episodic.py`; it is every CLI in this repo that takes POSITIONAL prose through the shell**, and
  neither tool's docs nor any guard said so. ⚠ **THE TRIGGER IS THE QUOTING FORM, NOT THE TOOL** — a
  double-quoted bash assignment executes backticks; a single-quoted string and a quoted heredoc do
  not. Every other note written that day survived because they went through quoted heredocs.
  ⭐ **AND THE PROPOSED HEURISTIC WAS VALIDATED IN THE FIELD BEFORE BEING BUILT:** option (a) predicts
  the damage leaves an odd-or-absent backtick count in otherwise code-dense content. A scan over every
  note written that day on exactly that predicate found the three damaged notes **and nothing else** —
  zero false positives on a real corpus. That is the evidence the WARN-don't-REFUSE design needed.

- **MEMORY-POISONING WRITE-POLICY REVIEW for the anneal episodic stores.** ⚠ **weekly_audit 2026-07-12
  HIGH; daemon 2026-07-08 finding; ZERO follow-through in the 46 days since.** The parallel to
  `spore-311`'s FILESYSTEM confinement, for the COGNITIVE substrate: **who can write to the
  episodic/neocortex stores, under what conditions, with what validation.** This CONTRADICTS the Proven
  "own the substrate / sovereignty all the way down" posture — spore-311 hardened the filesystem while
  the write path to the actual memory stayed open. Evidence cited: MPBench ASR 50.46% (daemon 07-08
  adversarial ore). **DECIDE: scope a write-policy/validation review, or explicitly accept-and-document
  the risk.** Either is an answer; 46 days of neither is not. `spore-325`

- **AM-PATTERN-ALIAS** — a pattern-NAME rename forks history: the omission audit false-flags both old
  and new name, and the new name restarts at 1x, **losing earned max_level**. Same family as AM-PRESERVE.
  Real (`verify_or_surface_before_claiming` → `before_acting` was an actual rename), LOW priority
  (self-resolved via bedrock crystallization). FIX = `[renamed-from:]` / `[alias-of:]` / `[supersedes:]`
  markers threaded into omission + history + contradiction-scan + crystal routing. **BATCH with the next
  anneal touch, not its own session.** NOT the dropped AM-IDALIAS (that was the episode-id namespace).
  ✅ **PREMISE VERIFIED 2026-08-27.** The only `alias` hits in `anneal_memory/*.py` are CLI *command*
  aliases (`search`/`recall`, `cli.py:2763-2771`). No pattern-name aliasing exists. `spore-047`

- **AM-RECALL-IDF deferred enhancements** (from the 2026-06-21 apparatus; NONE blocking; AM-RECALL-IDF
  shipped in anneal `57511eb`/0.9.3 + flow `3ffaae7`). Three real-but-deferred items from L1/L2/L3:
  **(1)** PER-CALL OVERRIDE of the IDF constants (`IDF_SCORE_THRESHOLD` / `IDF_MIN_CORPUS` / `IDF_FLOOR` /
  `IDF_ANCHOR_WEIGHT`) as `retrieve_relevant` kwargs defaulting to the module constants — kills the
  monkeypatch-in-tests smell and lets an adopter tune without monkeypatching the module.
  **(2)** DISTRIBUTION-RELATIVE precision bar — `IDF_SCORE_THRESHOLD=1.6` is **flow-CALIBRATED** for a
  conceptual/partnership regime; an entity-dense adopter (code identifiers, proper nouns → terms
  naturally rare) UNDER-tightens at 1.6 and should re-sweep. A bar set as a percentile of the observed
  query-weight distribution (or k × sum-of-floored-weights) auto-adapts per corpus. The √N anchor already
  de-risks this.
  **(3)** SINGLE-SNAPSHOT Store stats read — `corpus_n` and `doc_freq` come from separate `Store.recall()`
  calls, not one SQLite snapshot; a concurrent writer between them can skew df/n near the anchor (codex
  L3 MED). Benign today (df clamped to `corpus_n`, and the writer-adds race is direction-safe toward
  recall), but a Store-level stats API returning candidates + counts + corpus-count in ONE read txn
  erases the residual.
  TRIGGER: a 2nd real adopter on a different corpus regime (dogfood discriminator — don't build for the
  hypothetical), OR Levain v2 needing config, OR the snapshot race measurably biting. `spore-146`

- **weekly_audit / episodic deferred hardening** (2026-06-07 window-fix apparatus;
  complement + kimi + codex convergent). **(1)** PER-STORE full-window guarantee — raising the final limit
  to 5000 (`per_store=10000`) covers the current ~625/7d volume, but **a single store >10k/7d would
  truncate its oldest days BEFORE the federated merge, undetectable by the final-count cap-warning.** Fix
  = propagate per-store truncation signals into `query_episodes()` warnings, OR add an explicit federate
  "full-window" mode. Touches the shared `episodic.py` API → own pass. **(2)** STRATIFIED per-day sampling —
  newest-`MAX_EPISODES_PER_DAY`(50)-per-day is recency-biased WITHIN a high-volume day and can suppress
  early-day or less-chatty-agent patterns from a 130-episode day. Both reviewers called it non-blocking (a
  meta-pattern audit is robust to time-of-day skew); the sharper version stratifies by (agent, type, store)
  quotas with recency fill. A quality refinement, NOT a regression. `spore-040`

- **Episodic stores need a git backup** — currently only local `state/episodic.db`.
  ✅ **PREMISE VERIFIED 2026-08-27:** the file is **gitignored, 23 MB**, and no backup task exists in
  `scheduled_tasks.json`. The store is genuinely single-copy on one laptop. `spore-125`

- **AM-GAUNTLET — a published adversarial immune-benchmark** (false-demotion-rate / bad-promotion-blocked /
  sycophancy-drift / poisoning-graduation / rotated-citation-gaming / contradiction-surfacing). **Does NOT
  fix a live issue** — this is POSITIONING (Bold Stand empirical teeth) plus regression-safety (catches the
  next autoimmune/poisoning sibling). Axis = precision-vs-recall of the immune classifier;
  autoimmune-resistance is ROW ONE, AM-PRESERVE the worked example. TRIGGER: before the next public
  methodology push, OR if a 3rd autoimmune sibling surfaces — whichever comes first.
  ⚠ `positioning_ahead_of_product_kills_credibility` → **no rush, and that is the point.** `spore-048`

- **RECEIPT_VERSION=3: `exposed[].producer` (str|null) + `source` enum value `'afferent'`** (ratified
  2026-06-23, contract §2), landing with the other additive-nullable action-face fields
  (`gate`/`authority_scope`/`actor_first_estimate`). Full 4-layer apparatus — a shipped-contract change.
  Originally gated on the vagus efferent gate (Phase 2).
  ⚠⚠ **PREMISE QUESTIONABLE — CHECK BEFORE BUILDING (2026-08-27).** The gate it waited on has SHIPPED
  (`projects/vagus/next.md`: Slice 4a + 4b + 4c ✅), **but that same record shows the vagus compiler now
  enforcing `RECEIPT_VERSION=4` invariants** — so this item may be specifying fields against a superseded
  contract version. Settle *which version the contract is actually at* before implementing anything here.
  Migrated WITH the question attached rather than silently, because a stale premise written into project
  memory is exactly what the premise-check exists to prevent. `spore-166`

---

## ✅ SHIPPED history → `COMPLETED_SESSIONS_ARCHIVE.md`

The full reverse-chron SHIPPED LEDGER (0.4.x → **0.9.6**, the current public PyPI release) was ARCHIVED 2026-07-14 — this plan now carries only what is NOT done. Detail lives in CHANGELOG + git + `COMPLETED_SESSIONS_ARCHIVE.md`: the 0.9.6 first-public-0.9.x release · AM-SPORE-CAS · AM-RECALL-IDF · the projection-checkpoint/hwm primitive · §9 step-0/step-1 · AM-PYTYPED · AM-CONTLOCK hardening · AM-SNAPSHOT ① · Slice B (shadow) · Slice A · the 0.6–0.8 crystallized-pattern tier · 0.7.x solo-safety + sycophancy gate · the 0.4.x/0.5.x foundations · v0.3.x ship blocks · the Bold Stand fixes arc · the WRAP_PROTOCOL retirement (2026-06-01) · the Phase-1 adversarial stress-test.

## Positioning frame (daemon 2026-06-10 — not a build)

Drop "notes + search," adopt **"memory lifecycle / control plane"** (MemOS/MemoryOS/Memori converge on memory-as-system-resource; anneal is the sovereign version — vocabulary lags architecture). Candidate eval-harness scoping: grade whether a wrap chose the *right operation under mutation* (Memory-R1 trains ADD/UPDATE/DELETE/NOOP; anneal's human-judged compression is the manual high-fidelity version). Research grounding (single-paper each, don't fearmonger): Memora/FAMA (2604.20006) penalizes obsolete-memory use → validates capture/judged-compression/staleness/contradiction/demotion as the measured surface. Poisoning (2606.04329) → routed to augmentation_harness. Multi-party memory ceiling (2605.14498): speaker/source/agent identity must be first-class at ingestion AND retrieval AND consolidation. The forward-facing claim is **governed memory transformations improve specific regimes**, never "memory always improves agents" (EvoMemBench — memory can hurt by injecting irrelevant evidence / stripping execution detail / transferring mismatched procedures).
