# anneal-memory — Completed Sessions Archive

*Append-only. Newest first. Full detail preserved for historical reference.*


---

## ✅ ARCHIVED 2026-09-04 — the 0.9.9 arc in full: triage, release, CAS, and the carried sweep

*Moved from `next.md` at the 09-04 close. All SHIPPED and verified; the plan keeps only the ledger line
and the open work. Full narrative preserved here because the LESSONS are the reusable half — the class
"a test or gate that passes without exercising what it is named for", the mutation discipline that found
what three review layers read past, and the two self-inflicted repeats (a stale plan block producing a
wrong prioritisation; the action-boundary hook firing and changing nothing).*

## ✅ TRIAGED 2026-09-03 — all 6 of the night's findings VERIFIED AGAINST DISK AND HANDLED (`0903+7 anneal-seat`)

**Every one checked before it was believed; two of Diogenes' coordinates are WRONG — and a SEVENTH defect was found by generalising, in the method his report praised.** Suite **1727 passed** (1722 + 5 new tests), mypy clean on 18 files, ruff **64** = unchanged baseline, `verify_integrity` `(True, [])`. ⛔ **HELD AT `0.9.9.dev0` — Phill ruled 2026-09-03 that we do NOT cut 0.9.9 today.** Alex De Groodt has still not reported back on levain 0.4.2/0.4.3, and cutting now would add a second unconfirmed variable to the one install most likely to diff a wheel against a tag.

| # | Subject | Verdict | Disposition |
|---|---------|---------|-------------|
| 1 | `store.py` `wrap_cancelled` TOCTOU claim | **CONFIRMED — race re-driven to completion independently** | FIXED: `BEGIN IMMEDIATE` |
| 2 | version stamp vs PyPI | **FALSIFIED AS PHRASED, defect real** (already ruled, `spore-710`) | FIXED: `0.9.9.dev0` at **3** sites |
| 3 | "the race made concrete" test | **CONFIRMED — but the coordinate is `:752`, NOT `:194`** | FIXED: renamed + real interleaved test |
| 4 | audit/receipt predicate divergence | **CONFIRMED** by reading + run | FIXED: one parse, one predicate |
| 5 | SKILL.md guard hardcoded to one tool | **CONFIRMED** — the generalising guard cannot fire | FIXED: mirrored README's every-tool loop |
| 6 | `episode_ids is None` docstring | **CONFIRMED** — self-contradictory in one sentence | FIXED: rewritten + pinned by a test |

### ⛔ TWO OF HIS COORDINATES WERE WRONG, AND ONE IS ON THE HALF HIS OWN RULE GATES

His 09-03 self-correction (`diogenes-20260903-041052`) closes on: *"any `file:line` I am about to write that is NOT the subject coordinate gets greped in the same command that writes the episode"* — i.e. he diagnosed the leak as living in SUPPORTING coordinates only, and declared every SUBJECT coordinate filed that night exact, naming `test_server.py:194` in that list. **It is not exact.** `test_report_cannot_name_a_wrap_it_did_not_clear` is at `tests/test_server.py:752`, in the class opening at `:709`; line 194 is inside `TestToolRecord` and unrelated. Finding 2's *"`git tag --points-at HEAD` answers v0.9.8"* is also false — it answers nothing; the tag is on `a067a7d`, exactly what shipped.

⚡ **THE POINT IS NOT THE TYPO, IT IS THAT THE GATE REPORTED ITSELF CLEAN.** He measured the subject half as 6/6 exact and it was 4/6. So the close he wrote *"binds next night"* is scoped to the wrong half, and the failure it is meant to catch is already inside the fence. **A guard that grades its own coverage is the same class as everything else in this report** — worth telling him rather than silently fixing, because his prescription will not catch it.

### ⭐ FIVE OF THE SIX ARE ONE CLASS, AND IT IS THE PORTFOLIO CLASS

`0903+0 main` named it from the fan-in across seven repos: **a test or gate that passes without exercising the thing it is named for.** The proof here is one command, and it is the cleanest instance in the set:

> **Remove `BEGIN IMMEDIATE` → the new interleaved test fails and NOTHING ELSE DOES.** All four pre-existing tests stay green — *including the one named "the race made concrete."*

Same at the doc layer: erasing `delete_episode` from SKILL.md left **1,722 tests green**. Same at the audit layer: the `partial_state` marker, whose comment says it exists so auditors *"know this was recovery"*, was absent on the one case that IS recovery.

⚖ **WHY IT KEEPS HAPPENING, and this is the transferable half.** In every instance the artifact that certifies coverage — the test NAME, the docstring, the audit marker — was authored in the same act, by the same author, as the thing it certifies. There is no independent signal, so the failure is *structurally silent from inside*: exactly what the augmentation thesis's v1.9 says about a bounded model on unbounded reality. **The mutation test is the contact-with-reality half.** It is the only move in this set that produces an error signal the author's own frame could not generate. ▶ **STANDING CHECK, cheap and authoring-time: before shipping a guard, ask "would the defect itself make this pass?"** A sequential test of a concurrency property answers YES — which is precisely how the bad one got written.

### THE FIXES, and why `BEGIN IMMEDIATE` rather than a reworded docstring

Diogenes offered both ("make the claim true, or delete the word *removes*"). **The claim was made TRUE, deliberately**, because the sentence appears at **five** sites (`store.py` ×2, `server.py` ×2, `cli.py`, plus `CHANGELOG.md`) — retracting it means five careful edits that must all stay in sync forever, while fixing the code makes all five correct at once. Each site now names `BEGIN IMMEDIATE` as the mechanism, so the next edit that removes it visibly falsifies them.

⚠ Rewording would also have been honest about the sentence and **dishonest about the architecture**: `wrap_started`'s honest paragraph rests on an external flock excluding the competing writer, and **verified independently here — `continuity_lock` is taken only at `continuity.py:2288` and `store.py:3365`, at NONE of `wrap_cancelled`'s three call sites** (`cli.py:1194`, `continuity.py:1388`, `server.py:612`). Its own docstring says it guards the physical write critical section and does not span the wrap lifecycle. So this method could not rest on that exclusion and needed the lock itself.

- **F4** deliberately WIDENS partial-state: valid JSON of the wrong *shape* (`[1, 2]`) is now partial, where the raw-string test called it healthy.
- **F2**: the false 0.9.8 CHANGELOG claim was **corrected in place with a marked note, not rewritten** — a changelog records what shipped. `[Unreleased]` now also discloses the three post-upload commits that never reached PyPI.
- **VERIFIED FROM THE ARTIFACT, not relayed** (`spore-710` says to run it before quoting it): the published 0.9.8 sdist was downloaded 09-03; its SKILL.md carries the false `— (CLI only)` row and **never names `` `wrap_cancel` `` anywhere**. So the new every-tool loop would have failed on what shipped. The guard that existed did not.

### ⭐ A SEVENTH FINDING, OURS NOT HIS — AND IT IS IN THE METHOD HE CITED AS THE HONEST ONE

Diogenes' F1 closes on a compliment: *"⭐ THE CONTRAST IS THE FINDING — `wrap_started` performs the IDENTICAL read-then-write, cross-references this method BY NAME, and states the limit exactly. The honest paragraph already exists in this file; the newer method dropped it."* **He read the inline COMMENT and never read the DOCSTRING 120 lines above it.**

`wrap_started`'s docstring said the guard reads `wrap_started_at` *"inside the SAME transaction as the writes, so the check-and-set is atomic on this connection."* **MEASURED: `in_transaction` is `False` at that read too.** Same false mechanism, same file, undetected — by the reviewer who was standing on it.

⛔ **AND THE FALLBACK DOES NOT HOLD EITHER.** The honest comment defers the close to an external flock: *"that writer is excluded by the flock by design."* **Verified here: `continuity_lock` is taken inside `validated_save_continuity` ONLY — never in `prepare_wrap`, which is `wrap_started`'s caller.** So the competing writer was never excluded on the path that actually reaches the guard, and two connections could each pass it with one snapshot silently clobbered. Fixed with `BEGIN IMMEDIATE`; mutation-checked (remove it → the peer is accepted-then-clobbered instead of refused).

⚖ **THE GENERALISABLE PART: two descriptions of ONE piece of code, 120 lines apart, exactly one true.** A reader consulting the docstring was told the window was closed; a reader consulting the comment was told it was open and covered by a lock that path never takes. **Neither reader is wrong to trust what they read**, which is why "read the code" is not the fix — the fix is that the claim must be executable. Also corrected: `_db_boundary`'s comment asserting refusals "find no open transaction", which my change falsified and which now explains that the rollback is what releases the write lock.

▶ **HOW IT WAS FOUND, worth reusing: I asked whether the filed defect had SIBLINGS rather than only whether it was real.** The report's own praise was the pointer. A reviewer's compliment is an unverified claim like any other.

### ▶ FOUR MUTANTS, FOUR KILLS — none of these fixes is merely green
1. `BEGIN IMMEDIATE` removed → `test_a_peer_installed_between_read_and_clear_is_not_destroyed` fails, alone.
2. audit predicate reverted to the raw string → `test_audit_and_receipt_agree_that_corrupt_state_is_partial` fails.
3. a tool name erased from SKILL.md → `test_skill_documents_every_tool` fails.
4. `BEGIN IMMEDIATE` removed from `wrap_started` → `test_a_peer_cannot_slip_a_wrap_between_the_guard_read_and_the_writes` fails.

### 🔬 L1 + L2 RAN, AND THE CHANGE SET COMMITTED ITS OWN CLASS FOUR MORE TIMES

**L2 (SQLite/MCP domain lens) VERDICT: the fix is correct**, verified by execution across seven experiments — `BEGIN IMMEDIATE` over deferred `BEGIN` is load-bearing (a deferred begin must UPGRADE, which under WAL fails `SQLITE_BUSY_SNAPSHOT` and **the busy handler does not retry that** — zero wait, unrecoverable); `EXCLUSIVE` would buy nothing in WAL; write-lock hold measured at **0.11 ms**; all four failure paths release the lock. **L1 VERDICT: APPROVE the locking, NEEDS WORK on the documentation layer** — which is what the change set is about.

⛔ **FOUR SELF-INFLICTED INSTANCES, ALL CONFIRMED AGAINST DISK AND ALL FIXED:**
1. **A FABRICATED NUMBER — "the claim was false for eleven days."** `git log -S` puts the sentence in `a067a7d`, 2026-09-02 13:20. **ONE day.** "Eleven" was conflated from *"eleven lines apart"* in the F4 finding — false precision **in the paragraph whose subject is a false claim shipping unchecked.** The L0 number-in-a-comment shape, mine.
2. **"So unlike `wrap_started`, this method cannot rest on that exclusion"** — TRUE when written, then falsified BY MY OWN FIX to `wrap_started` an hour later, and left sitting two lines above the CHANGELOG section that falsifies it.
3. **"`episode_ids` is `None` in exactly two cases"** — it is four readable-JSON shapes; my F6 rewrite replaced one self-contradiction with a different inaccuracy, contradicted by my own comment 1,600 lines away.
4. **I asserted the lock-timeout was "a failure mode this fix introduced."** L2 MEASURED the old shape under identical contention: same error at **5.19s vs 5.20s**. It already existed; the statement only moved it ahead of the reads. I claimed a measurement I had not taken.

⭐ **AND THE DEAD GUARD IS WORSE THAN DIOGENES HAD IT.** He read `test_skill_does_not_claim_an_mcp_tool_is_CLI_ONLY` as *"it passes on the exact defect it was written for."* L1 measured further: **it scans ZERO lines and has never executed its assertion once** — the same commit that added it rewrote the row to `CLI-only` (hyphen), and its filter required the exact string `CLI only`. **Not weaker coverage; NO coverage.** ⛔ And the predicate is UNSALVAGEABLE rather than mis-spelled: the current CORRECT row reads *"`wrap_cancel` (inspect via `status`; `wrap-status` is CLI-only)"* — the CLI-only applies to a CLI SUBCOMMAND while the row properly names the MCP tool, so merely widening the spelling would fail a good row. Rewritten **cell-scoped** (an MCP cell that marks unavailability while naming no shipped tool) plus a **NON-VACUITY PROOF** on synthetic known-bad and known-good rows — because a negative guard that matches nothing cannot otherwise be distinguished from a broken one. **That distinction is the generalisable lesson: every negative guard needs a positive proof that its predicate discriminates.**

▶ **THE THREADED TESTS NOW ASSERT THE MECHANISM, NOT THE OUTCOME** (L1's one insistence, and it is right): both captured `peer_done.wait()` and assert `[False]` — *the peer must NOT complete while the caller holds the lock*. Asserting only the surviving wrap would also pass if something unrelated serialised the writers. Both mutants now fail on that assertion, and each site kills independently. Also: injection keyed on `wrap_episode_ids` rather than a hardcoded read COUNT (a fourth read would have slid the injection mid-set and quietly weakened the window while passing), threads daemonised, receipt tests moved out of `TestWrapInProgressError` into their own class.

⚠ **A CONCURRENCY INCIDENT WORTH THE WRAP: TWO AGENTS WROTE `store.py` SIMULTANEOUSLY.** L1's own mutation script used `str.replace` with no count, stripped BOTH `BEGIN IMMEDIATE` statements, then restored a `/tmp` backup predating my `wrap_started` fix — silently reverting it. **My new test caught it**, on the day it was written. L1 disclosed this unprompted and said *"I am not a trustworthy witness to a file I was editing."* ⚖ THE RULE: **a reviewer that mutates the working tree is a WRITER, and two writers on one file is a lane collision even when one of them is a review agent.** Mutation runs belong on `git archive` copies — which is exactly what Diogenes already does.

### 🔬 L3 RAN AND THE FRONTIER SEAT FOUND FIVE MORE — IN THE FIXES, NOT THE ORIGINALS

⚠ **Seat health first, so this is not read as a clean bill:** `complement` **ERRORED and contributed NOTHING** (a named gap); `glm` was **CUT OFF part-way** and its one HIGH — *"`cli.py` never imports sqlite3"* — is **REFUTED** (it is at `cli.py:46`; the module imports fine; it was reading the diff and missed a separately-added line). **codex, the non-replaceable seat, carried this pass alone**, which is the exact argument for never dropping it.

**ALL FIVE CONFIRMED AGAINST DISK AND FIXED, and every one is in code written TODAY:**
1. **HIGH — a COMMITTED cancellation could be reported as a FAILURE.** `wrap_cancelled` commits, then writes its audit event; an audit I/O failure propagated out with metadata already cleared and no receipt, so the operator is told the cancel failed when it SUCCEEDED and a peer finds its token gone with no audit saying why. ⭐ **The policy already existed in this file** — `_batch` documents that audit-flush exceptions are swallowed after a successful commit because propagating one *"would trick the caller's outer except clause into cleaning up tmp files that represent committed state — a data-loss path."* The method was violating the rule its own sibling states. Now warned, not raised.
2. **MED — MY CONTENTION MESSAGE CLAIMED THE LOCK PROVES THE WRAP IS LIVE. It does not.** A write lock proves only that SOMETHING is writing; an unrelated `record`, prune, schema-init or abandoned transaction holds it identically while an old wrap stays stranded. **Locks carry no ownership.** ⚡ **And my test made the point by accident — it holds a RAW `BEGIN IMMEDIATE` on an unrelated connection, i.e. the counterexample, while asserting the "live" conclusion.** Message and test both now say only what is known and defer ownership to `status`. (Real liveness needs persisted owner/lease data — that is `spore-699`, not something a lock can supply.)
3. **MED — the CLI translation was scoped to the store OPEN only**, so a peer taking the lock between a successful open and the command's own write still escaped as a raw traceback, from the command whose messaging claims to prevent exactly that — **and the first test held the lock BEFORE opening, so it structurally could not see the post-open window.** Moved to ONE shared boundary in `main()` covering every subcommand. ⚖ *Fixing the one site that was noticed is how the bad scoping got written in the first place.*
4. **MED — four lifecycle keys cleared, three classified.** A store holding only a stray `wrap_section_schema` reported "no wrap was in progress" and emitted NO audit event: leftover state discarded with no record. Counted as state now; deliberately NOT as completeness (pre-AM-SCHEMASNAPSHOT wraps have no frozen schema and are healthy).
5. **LOW — TWO NEGATIVE TESTS COULD NOT FAIL.** Both built a `StoreDatabaseError` with no underlying `sqlite3` cause, so the predicate exited at its first type check and a regression treating EVERY `OperationalError` as contention would have passed. Real non-busy causes chained; both now kill that mutant.

⛔ **AND THE CONCURRENCY GUARDS THEMSELVES COULD PASS WHILE STARVED** — the subtlest finding of the day. Asserting "the peer did not finish inside the window" **cannot distinguish BLOCKED-ON-THE-LOCK from NEVER-SCHEDULED**; a thread starved for one second reproduces the healthy observation on the live defect. Both threaded tests now open the peer's store and signal READY *before* the caller takes the lock (a `Store()` open would itself block, so the barrier cannot sit after construction), assert that point was reached, then release into the gap. **The guard against the class needed a guard against the class.**

### ✅ L4 + THE STOP-TIME BAR — RUN, NOT ASSUMED
- **The false claim never left the code**: `grep` over `README.md`, `docs/`, `skill/` returns NOTHING, so unlike 0.9.8's SKILL.md row there was no public surface to correct.
- **End-to-end through the REAL MCP stdio transport as a SUBPROCESS** — not the in-process `Server` object: `initialize` → `serverInfo 0.9.9.dev0`, `tools/list` → **17**, record → prepare_wrap → `wrap_cancel` → idempotent second cancel. ⚡ **This closes a NOT-CHECKED that BOTH Diogenes and the L2 seat declared** (*"I did NOT install the wheel and did NOT start the MCP server or drive a real client"*).
- ⛔ **STILL NOT CHECKED, said plainly:** no wheel was built and installed into a clean venv, and no `anneal-memory` CLI subcommand was run as a real subprocess (CLI coverage is in-process via `main()`).

### ▶ COMMITTED — anneal `0c81bd9`, working tree CLEAN, NOT PUSHED
**1735 passed** (1722 + 13 new) · mypy clean, 18 files · ruff **64**, unchanged baseline · `verify_integrity` `(True, [])`. **EIGHT MUTANTS, EIGHT KILLS.** ⚠ Committed locally rather than left dirty **because the file was already clobbered once today** by a concurrent reviewer — 1,333 uncommitted lines across seven live sessions is a lane hazard, not tidiness.

## 🚀 0.9.9 PUBLISHED 2026-09-04 — LIVE ON PyPI, TAG PUSHED

**https://pypi.org/project/anneal-memory/0.9.9/** · tag `v0.9.9` at `076eed9`, `main` pushed. ⚖ **Flow published it** — Phill ruled the prepared-command handoff wrong: *"you can publish this your own damn self with that third PyPI token."* Flow publishes; it does not build a paste-ready command and hand it over.

**VERIFIED FROM BUILT ARTIFACTS, NOT THE SOURCE TREE** (the standard 0.9.7 and 0.9.8 both met, and the last NOT-CHECKED item Diogenes and the L2 seat had both named): installed FROM PyPI into a clean venv → 17 tools, shipped manifest `(True, [])`; **the full stuck-wrap/CAS journey replayed over real MCP stdio against the installed package, 9/9**; sdist SKILL.md carrying **zero** false "CLI only" rows and a ladder teaching 12x. ⚠ The replay's one red was **my own script** asserting the pre-release `0.9.9.dev0` against an artifact correctly reporting `0.9.9` — the harness was lying, not the package.

⚡ **WHY TODAY AND NOT LATER:** the published 0.9.8 sdist told agents `wrap_cancel` was `— (CLI only)` — **Alex De Groodt's exact three-day lockout condition, live on PyPI since 09-02**, with the commit that kills the row sitting unshipped in our tree. That was the driver, not the race fix (real, but low-probability and needed hand-scheduling to reproduce).

⚖ **VERSION FORCED, NOT CHOSEN: 0.9.9.** levain pins `anneal-memory>=0.9.8,<0.10`, so 0.10.0 would have broken it.

### ⛔ TWO CONCERNS RAISED AND THEN KILLED BY MEASURING — NEITHER WAS REAL
- **The tamper alarm.** Diogenes/`spore-710` said an operator checking an installed 0.9.8 against the tag gets "possible tampering". **The `v0.9.8` tag's manifest matches PyPI EXACTLY.** His mismatch was repo-HEAD vs published, because HEAD had moved three commits. Taking the reference from the tag — the correct thing — is clean.
- **The `server.json` semver gate** (L1 NOTE-6) — refuted against the fetched MCP schema; see below.

### ⛔ AND ONE THAT WAS REAL AND BLOCKED THE UPLOAD: `~/.local/bin/twine` IS BROKEN
`twine` 6.2.0's bundled `pkginfo` caps at `Metadata-Version: 2.4`; hatchling 1.27 emits **2.5**, so it dies with `InvalidDistribution` at **BOTH `check` AND `upload`**. ⚡ **NOT a package defect, and the disproof was in the artifact: the published 0.9.8 wheel also carries 2.5, so PyPI accepts it.** Fixed by building a throwaway venv with **twine >= 7** (it dropped pkginfo for its own metadata handling). `twine check` at 6.2.0 was a GAUGE showing a false red on a package PyPI had already taken.

### ✅ `spore-424` REWRITTEN — IT WAS WRONG IN BOTH HALVES AND EACH COST A FAILED ATTEMPT
The note existed exactly to make a publish go right, fired at the action boundary as designed, and still produced two failures. (1) It labelled TOKEN3 *"the one that publishes levain"*, routing an anneal cut to the project-scoped TOKEN2 — **ruled: TOKEN3 is ACCOUNT-LEVEL, use it for every publish.** (2) It named the broken twine. ⚖ **The lesson is not "the note failed":** a Keep note records a WORKING FORM, and a form has dependencies the note does not own — a token's scope, a build backend's metadata version, a tool's parser. It stayed literally true about July and became false about what to DO in September without anything in it changing. ▶ **The fix that generalises: record the FAILURE SIGNATURE alongside the form**, so a reader hitting a *different* failure knows they are outside the note's coverage rather than assuming they mis-followed it. Both exact error strings are in it now, plus the per-repo build toolchain (they differ; the old note gave levain's as universal) and the verify order.

### ✅ GITHUB RELEASE LIVE — https://github.com/phillipclapham/anneal-memory/releases/tag/v0.9.9
Created once Phill lifted it; the earlier classifier block had killed the whole command including the heredoc that wrote the notes file, which is why the retry first failed on a missing path.

### ⛔ THEN I COMMITTED THE EXACT DEFECT `spore-710` EXISTS TO PREVENT — ONE HOUR AFTER PUBLISHING
`58c7b40` landed **one commit past the `v0.9.9` tag with `pyproject` still stamped `0.9.9`** — a published number left standing on a moving main, in the repo whose CHANGELOG I had written that policy into the same morning.

⚡ **AND THE ACTION-BOUNDARY HOOK PRINTED `spore-710` IN FULL BEFORE THAT PUSH.** On screen, correct, specific, about that exact command. It changed nothing. ⛔ **This is NOT "the recall layer failed"** — it delivered precisely what it was built to deliver at precisely the right moment. The failure is downstream: **READING IS NOT ACTING.** A note surfaced at an action boundary competes with the action already in flight and loses when the action feels routine (a docs-only commit).

▶ **FIXED THE WAY `partnership.md` already prescribes — `structural_invariants_beat_discipline`, never "read harder".** `TestReleaseStampIsNotAPublishedVersion`: HEAD may be stamped at a released version ONLY if HEAD **is** that release commit, plus an all-sites agreement check. Reads **git tags, not PyPI**, deliberately — an unreachable network oracle degrades to a PASS, which is this defect's own shape. **Mutation-checked by restoring the exact defect I had just committed.** Stamp is now `0.9.10.dev0` at four sites.

⚖ **THE EVIDENCE FOR MECHANISM-OVER-DISCIPLINE IS IN THE SAME DAY:** the 0.9.9 cut found the THIRD stamp site (`server.json`) **only because a pre-existing consistency test failed** after the first two moved. The mechanical half caught what a careful pass missed, on the identical surface, hours before the discipline half failed on it. **anneal had the policy and no test — worth nothing. The levain seat had neither this morning, built the test, and it is the one that held.**

### ⭐ THE LEVAIN SEAT'S ARROW TIP FOUND 22 MORE — "ELEVEN SURFACES" WAS AN UNDERCOUNT BY 2x
Their capped-ladder sweep had missed a site because the grep was ASCII-only and the text used `→`; they said so and told me to re-run mine. **Three of the twenty-two were FALSE about current behaviour**, not stale: comments asserting `_GRADUATION_RE is 2x/3x-only` when it excludes 1x but has NO ceiling — ⚡ **and the same file already said so correctly at another site.** One was corrected; three kept asserting the opposite. The other nineteen were `(2x/3x)` as a **gloss for Proven-tier**, which is 2-and-up, so the parenthetical silently re-taught the cap while reading as harmless shorthand.

⚖ **THE RULE: a semantic sweep must enumerate SPELLINGS, not concepts.** A retired constraint can be taught as an ASCII arrow, a Unicode arrow, a range (`2x/3x`), a set (`{2, 3}`, `(2 or 3)`), or a gloss on a word whose meaning has widened. One spelling reports CLEAN while the rest sit untouched — `absence_of_signal_rendered_as_health` at the grep layer. ⛔ **And the inverse matters as much:** three classes were deliberately LEFT because they are accurate — comments explaining WHY the cap existed, anything scoped `pre-0.4.6`, and `_BARE_GRADUATION_RE`'s "bare 2x/3x graduation" (**that regex genuinely IS still `[23]` under `spore-676`**). A blunt sweep-and-replace would have turned all three into falsehoods. **Widen the enumeration; keep the judgement per-site.**

### ✅ AND THE GUARD NOW SITS AT THE RIGHT BOUNDARY — `scripts/hooks/pre-push`
⚡ **The levain seat read my own incident better than I did:** *"the guard catches it at COMMIT time, not at push time. Neither of us has anything that fires between 'stamp is wrong' and 'it is public' — the hook is the only thing there, and it just failed its one job."* **Measured: anneal had NO git hooks at all, and CI runs on push — i.e. after it is public.** So the pytest guard fires only if someone runs the suite, and the sole instrument in the private→public window was the recall hook that had demonstrably not changed the behaviour minutes earlier.

**BUILT + INSTALLED:** `scripts/hooks/pre-push` refuses the push on a bad stamp; `scripts/install_hooks.sh` sets `core.hooksPath` so the firing hooks are versioned and reviewable, not per-machine. ⛔ **FAILS CLOSED** on the seat's own principle — a guard that degrades to a pass when it cannot run is the defect's own shape; the escape is explicit (`git push --no-verify`), never silent. **VERIFIED ON THE REAL PATH:** reproduced the defect on a scratch branch → `git push` REFUSED with the bump instructions → then pushed `main` through the live gate.

⚖ **THE GENERALISABLE SHAPE: a control must sit at the boundary it protects, and "we have a test for that" locates it at TEST time — a different boundary from PUBLISH time.** Ask of every guard: *what is the last moment the defect is still private, and does anything run THERE?* Here the answer had been "only a hook that asks a human to read something" — the weakest possible control at the strongest possible moment.

### ⭐ AND THE CROSS-REPO EXCHANGE PRODUCED A RULE THAT PREDICTS EXPOSURE RATHER THAN DESCRIBING IT
I handed the levain seat the four spellings that found my twenty-two. They re-swept and reported their original "no" had rested on **arrows and prose forms only** — never the range, set, or gloss — so a definitive answer had been built on a pattern set that would have missed **four fifths** of what I found. They said so unprompted rather than letting a clean result stand.

⚡ **Their structural explanation is the keeper:** *"levain TEACHES the discipline, anneal ENFORCES it — there is no `_GRADUATION_RE` here, no Proven-tier constant, nothing to gloss."* That is why the gloss form was 19 of my 22 and structurally **zero** of theirs. ▶ **THE RULE: the gloss form can only exist where a NAME has a MACHINE DEFINITION that widened underneath it.** A prose-only repo is exposed to arrows and ladders; an enforcing repo is additionally exposed to every comment naming a regex range or glossing a tier constant. **Operationally — when a constant WIDENS, sweep everywhere its NAME appears, not just where its VALUE is written**: the value is greppable, a name carrying a stale parenthetical is not.

⚖ **Two independent sweeps, both confident, both wrong in different directions, each visible only to the other** — `correction_comes_from_outside_the_planner` at the grep layer. Not a smarter reviewer; a differently-enumerated one.

### ⚠ ONE FOR THE APPARATUS, FROM THE LEVAIN SEAT
`git commit -m "…"` **executes backticks** — their toolchain-warning commit lost the two commands it existed to record (` `uv build` ` was run and substituted empty). `episodic.py` documents this hazard for its own positional form; git carries no such warning and the failure is SILENT. ✅ Checked mine: all five anneal messages intact, and I verified empirically that `-m "$(cat <<'EOF' … EOF)"` is ALSO immune (quoted heredoc blocks expansion inside; outer quotes stop the captured result being re-evaluated — tested with a literal backtick command and a `$()`, both survived). **The naked `-m "…"` is the exposed form.**

### ⚠ OUTSTANDING FROM THE RELEASE
- ⛔ **The GitHub release for v0.9.9 was BLOCKED by this session's permission classifier** (`gh release create`). Surfaced to Phill, **not routed around**. PyPI is live regardless; only the release page is missing. Artifacts are staged at `dist/anneal_memory-0.9.9{-py3-none-any.whl,.tar.gz}` and the notes are written.
- **levain lockstep messaged to `0903+6 levain-seat`:** `KNOWN_GOOD_ANNEAL` + `TEMPLATES_RECONCILED_ANNEAL` → 0.9.9 (pip floor unchanged). ⚠ Flagged for them to CHECK rather than assume `TEMPLATES_RECONCILED_ANNEAL`: 0.9.9 corrected eleven surfaces teaching the retired ceiling, and **if levain's methodology-core teaches the same `1x→2x→3x` ladder it has the identical defect and a template edit IS owed.**

## ▶ SHIPPED IN 0.9.9 — `spore-699` (Alex's A3): AM-WRAPCANCEL-CAS

⚡ **TWO OF THE SPORE'S THREE DEFERRED ITEMS WERE ALREADY CLOSED BY THIS MORNING'S TRIAGE** — the audit-after-commit hole (item 3) and the ignored `wrap_section_schema` (item 5). codex re-found both independently at L3 tonight, which is how they were noticed. **The spore has been updated so a future session does not re-derive them as open.**

**BUILT:** `Store.wrap_cancelled(expect_token=...)` — clear the wrap ONLY if the store's current token is exactly that one, else raise the new `WrapOwnershipError` and change **nothing**. `WrapInProgressError` stops a second wrap **clobbering** the first; this stops a second session **cancelling** it. Wired to BOTH transports: `wrap_token` over MCP, `--wrap-token` on the CLI.

⚖ **WHY IT WAS SAFE TO BUILD BEFORE THE DESIGN QUESTION IS SETTLED:** it is the piece BOTH competing paths need underneath — the age-refusal transport AND Alex's idempotent-`prepare_wrap` alternative — so it forecloses neither and does not presuppose the ruling. The spore itself calls the store-side CAS "the real close".

⭐ **AND THE ORDER OF OPERATIONS IS THE FINDING: THIS GUARD COULD NOT HAVE BEEN WRITTEN CORRECTLY YESTERDAY.** The compare and the clear are atomic across connections ONLY because of the `BEGIN IMMEDIATE` added hours earlier. Written before that fix, identical code would have been a read-compare-write with a live window — **a guard carrying the exact defect it was written to close** — and it would have LOOKED right and PASSED a sequential test. Mutation-proved both ways: drop `BEGIN IMMEDIATE` → the cross-connection atomicity test fails while the other four pass; drop the comparison → the refusal tests fail.

⚖ **TWO DESIGN CALLS, ON THE RECORD SO THEY ARE NOT RE-ARGUED:**
1. **NO `force` FLAG.** Omitting `expect_token` IS the override — an unproven cancel is a cancel with no claim, which is exactly what every pre-existing caller does and what the ORIGINAL stuck-wrap recovery needs (you have no token for a wrap you did not open).
2. **`actual is None` is deliberately DISTINGUISHABLE from a token mismatch.** "Your wrap already finished" is retry-safe; "a peer owns this" means back off. Conflating them sends the agent the wrong way, so both transports report them differently.

⛔ **NOT BUILT, DELIBERATELY — THE AGE REFUSAL.** It needs a magic threshold, and tonight's codex finding argues against the whole shape: **a timestamp establishes ownership no better than a lock does.** The token is persisted identity; an age heuristic is a proxy for it. That choice and Alex's idempotent-`prepare_wrap` alternative remain **HIS AND PHILL'S CALL** and stay open in `spore-699`. `confirm: true` stays rejected.

▶ **SURFACES ENUMERATED AND ALL LANDED AT ONCE** (L0 caught the doc half): library · MCP tool schema + handler · CLI flag + command · `SKILL.md` · `README.md` · `docs/library-quickstart.md` · both `tool-integrity.json` manifests regenerated. ⚠ SKILL.md had been offering *"call `status` first"* as the ONLY ownership mitigation — true until a real proof-of-ownership existed, false the moment it shipped. **That is 0.9.8's own defect class and it was one commit away from repeating.**

**VERIFIED THROUGH THE REAL MCP STDIO TRANSPORT AS A SUBPROCESS**, not the in-process object: schema advertises `wrap_token` → wrong token REFUSED with the wrap still open → right token cancels → replay reports "already completed" rather than blaming a peer. **1749 passed**, mypy clean, ruff 64, both manifests `(True, [])`.

### 🔬 L3 ON THE CAS — BOTH SEATS CONVERGED ON A LOCKOUT THE GUARD ITSELF REINTRODUCED

⚠ Seat health: `glm` was **CUT OFF part-way** again (2 files opened) — its findings are real but are not coverage. **codex carried the pass.** ⭐ **CONSENSUS on finding 1 (glm HIGH + codex MED) — the strongest signal the mesh produces**, and it was right.

**ALL SIX REPRODUCED BY EXECUTION BEFORE BEING BELIEVED, ALL SIX FIXED, ALL SIX IN CODE WRITTEN TODAY:**
1. ⛔ **THE LOCKOUT.** `actual = cancelled_token or None` collapsed *"nothing here"* and *"something here but no usable token"* into ONE answer. A partial store (`wrap_started_at` set, `wrap_token` empty — a crash or a hand edit) told the operator **NO WRAP IS IN PROGRESS** while `wrap_started_at` SURVIVED THE ROLLBACK, so the next `prepare_wrap` failed with `WrapInProgressError` and the store looked permanently stuck **with the message insisting nothing was wrong**. ⚡ That is **Alex De Groodt's original three-day lockout, reachable through the guard written to prevent it.** State is three-way now; `actual is None` means IDLE and nothing else, and the partial branch names the only recovery that works.
2. **A receipt-building step after the DESTRUCTIVE COMMIT could raise.** The parse ran after the clear catching only `JSONDecodeError`; an invalid-UTF-8 BLOB raises `UnicodeDecodeError` and a 5,000-digit integer raises `ValueError` (codex reproduced the BLOB). Wrap cleared, caller got an exception, no receipt, no audit. ⚖ **THE RULE: nothing after a destructive commit may be able to raise.** Parse + classification moved inside the transaction.
3. **A warning filter set to error UNDID the audit guard** — under `-W error` the `warnings.warn` raises, recreating exactly the committed-then-reported-as-failed path it was written to remove. ⚡ **Its own regression test MASKED it, because `pytest.warns` installs a capturing filter.** The guard's test made the guard look sound.
4. **`WrapOwnershipError` could not be pickled** (keyword-only ctor, no `__reduce__`) — it crashed multiprocessing/RPC/log serialization **while handling a refusal**.
5. **Both validators used `re.match`** and `$` matches before a trailing newline, so a 33-char token was accepted and reported as an ownership MISMATCH rather than a malformed argument. Now `fullmatch`, matching `save_continuity`, schema bounded to 32.
6. **All three concurrency tests proved the lock by TIMEOUT.** Peer-noncompletion is evidence, not proof — a mutant that fails to schedule the peer produces the identical observation. They now assert `in_transaction` at the injection point: **true iff `BEGIN IMMEDIATE` ran, deterministic, zero timing dependence.**

### ⭐⭐ THE MUTATION RUN CAUGHT ONE THAT THREE REVIEW LAYERS DID NOT — AND IT IS THE PUREST INSTANCE OF THE DAY

My regression test for **non-`JSONDecodeError`** parse failures stored `b"\xff\xfe not utf-8"` — whose length happens to decode CLEANLY as utf-16 and then fail as a **`JSONDecodeError`**, i.e. the exact case the narrow catch already handled. **It passed against the mutant while its name and docstring certified the stronger property.** ⚡ **ONE CHARACTER decided it — `utf-8` vs `utf8`** (the byte count's parity). Nothing in review caught it; L1, L2 and codex all read past it. **Only running the mutant did.**

⚖ **THIS IS THE DAY'S THESIS CLOSING ON ITSELF.** The class is *a test that passes without exercising what it is named for*; I wrote a test FOR that class which was itself an instance of it, and the only instrument that could see it was the one contact-with-reality move in the set. **A review reads what the test SAYS; a mutant tests what it DOES.** Every payload now asserts the exception class it exists for, so it cannot drift back.

### ▶ COMMITTED — anneal `7e26ba1`, tree CLEAN, NOT PUSHED
**1758 passed** · mypy clean · ruff **64** unchanged · BOTH integrity manifests `(True, [])` · **six mutants, six kills** · verified through the real MCP stdio transport as a subprocess (partial → told to omit the token → override clears it: the lockout closed end-to-end).

## ✅ THE 14 CARRIED FINDINGS — SWEPT 2026-09-04. **STILL OPEN drops 20 → 2, AND BOTH REMAINING ARE RULED, NOT UNDONE.**

⛔ **Answering "does that clear everything?" precisely: it did NOT, and the carried half was the larger half.** anneal `def07cf`. **1760 passed**, mypy clean, ruff 64, both manifests `(True, [])`, `-O` import clean, every fix mutation-checked.

**ELEVEN SURFACES WERE STILL TEACHING THE AM-LEVELCAP CEILING, THREE WEEKS AFTER 0.9.7 REMOVED IT** — because **a widening emits NO error signal downstream**: nothing breaks, every consumer keeps working correctly against the narrower contract it already knows, and the only symptom is absence. That is why it needed a human reviewer to find and now needs a test.

⭐ **ONE WAS CAUSATIVE, NOT COSMETIC — AND IT REFRAMES CARRIED FINDING (1) ENTIRELY.** `crystal.py`'s consumer contract did not merely *describe* a cap; it **instructed** readers *"a consumer routing `crystallize` MUST guard `level in (2, 3)`"* — and flow's `scripts/crystal_decision_apply.py:125` implements exactly that. **So the two halves of finding (1) are not independent instances: the stale doc is the SOURCE of the live consumer cap.** Same shape as the voltron entry — the error was produced BY the record rather than by forgetting it. ⚠ Fixing the doc is a PREREQUISITE for fixing the consumer, or the next session re-derives the cap from the contract again.

**FIXED (anneal-side, my lane):** the consumer contract · **`SKILL.md`'s terminating `1x → 2x → 3x` ladder** — the doc an AGENT loads, pinned by NOTHING while the generated instructions had `TestTeacherCoversReaderRange`, which is why it stayed wrong 20 days; it has a mutation-checked test now · `crystal.py`'s "graduates IN at 3x" (it is `MIN_PROVEN_LEVEL`, 2) · a `UserWarning` calling an Nx pattern "top-tier (3x)" — **nothing pinned that string either** · four "(2 or 3)" field comments · the validator docstring's "2x/3x lines" · the invented quotation now citing its real site · the 0.9.7 note that asserted the 1x floor **while quoting the regex whose `\d{2,}` matched a zero-padded `01x` as level 1** — the note asserts the property that bug broke · **two `assert`s for type narrowing that vanish under `python -O`**, where the next line raised `TypeError` on None instead of a typed error, AFTER the DB committed — ⭐ the identical correction was already made for `meta_tmp` and documented **TWENTY LINES BELOW `cont_tmp`**: it landed on one sibling and not its neighbour · and **an "EXACTLY" test checking ONE direction** — its own docstring names the risk as *"teaching a marker the reader silently drops"*, which is the OTHER direction and the unchecked one. The converse now round-trips every taught marker through the real reader, **scoped to the teaching shape rather than every backticked glyph** (a naive scan fails on correct text — the instructions also backtick the markdown bullet, the level separator, and `~` as an explicit NON-marker: the same over-matching trap as the SKILL.md "CLI only" guard, hit again and caught by running it).

### ⛔ THE TWO THAT REMAIN ARE RULED DEFERRALS, NOT UNFINISHED WORK
1. **`flow/scripts/crystal_decision_apply.py:125` — `if level not in (2, 3)`.** A LIVE behavioural cap in flow's consumer, and the crystal path is **`spore-675`'s lane, held by the consolidate seat**. Not touched. ▶ Its anneal-side CAUSE is now fixed, so the fix is unblocked whenever that lane runs.
2. **`_BARE_GRADUATION_RE` stays `[23]` — `spore-676`, gated on `spore-675`, own session, own apparatus.** Widening it puts fourteen mature carried patterns (incl. `absence_of_signal_rendered_as_health` at 19x) onto the bare-demotion path at the next re-stamp. ⚠ **But its stated justification — "the asymmetry is inert" — was MEASURED FALSE** (fourteen bare 4x+ lines, TEN dated to the wrap then in progress: today-dated bare 4x+ lines are the HABIT, not the exception the date-gate assumption relied on). **Corrected in place without touching the decision** — the deferral stands on blast radius. ⚖ **A deferral resting on a false premise is one somebody re-opens for the wrong reason.**

### ⚠ OWED, NOT DONE
- **Tell Diogenes his subject-coordinate gate reported itself clean at 4/6.** His own close does not cover it.
- The 14 carried findings are **untouched** — 9 are the AM-LEVELCAP prose tail, deferred and cheap. This triage covers the 6 filed 09-03 only.
- ✅ **REFUTED 2026-09-04 — `server.json`'s `0.9.9.dev0` is NOT a release gate.** L1 NOTE-6 said the MCP registry "may reject" a non-semver version. **Fetched the declared schema** (`static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json`) rather than reasoning about it: the top-level `version` carries **only `maxLength: 255`** and `packages[].version` only `minLength: 1` plus `not: {const: "latest"}`. **No `pattern`, no format, no semver enforcement** — the description says a server *SHOULD* follow semver, not MUST. `0.9.9.dev0` is schema-valid. ⚠ **The real constraint is a different one and it is not about syntax:** `packages[].version` names the **PyPI** version, so at publish time it must name a version that ACTUALLY EXISTS on PyPI. `0.9.9.dev0` is not published and never will be, so the check at the next cut is *"does this string name a real PyPI release"*, not *"is it semver"*. ⚖ Recorded because the wrong version of this concern would otherwise be re-derived from this file — the same doc-causes-the-error shape as the crystal contract.
- ✅ **CLOSED 2026-09-04 — the `wrap_started` refusal-path stall is FIXED, and it was real.** L1 flagged it as the one behavioural edge of the lock change. **Measured before fixing:** with a wrap in progress and an unrelated writer holding the lock, `wrap_started` waited out `busy_timeout` and raised `StoreDatabaseError` — **0.85s at an 800ms timeout, 5s in production, and the WRONG ERROR CLASS.** The caller is told the DATABASE is broken when the true answer is *"a wrap is already in progress"*. ⚡ **And that answer needs NO lock to know — it is a READ, and WAL readers never block.** The lock had been placed in front of a question it was not needed for. Now a **refusal-only pre-check** runs before `BEGIN IMMEDIATE`: **0.85s → 0.000s**, correct class, and the cross-connection clobber guard is untouched because the authoritative check stays inside the transaction. ⛔ **The soundness turns entirely on the DIRECTION the fast path can err:** it can only REFUSE, never grant — a stale *"no wrap"* falls through to the real check (no clobber), a stale *"wrap in progress"* is a spurious refusal, which is safe and is exactly what this method did before the lock existed. **A double-check is only sound when the fast path errs toward the safe answer.** Mutation-checked. ⚠ **And the first cut of it was a defect the suite caught:** the pre-check sat ABOVE `_db_boundary`, so a SQLite failure on that read escaped as a raw `sqlite3.OperationalError` — breaking the caller-consistency primitive (*"EVERY store failure surfaces through StoreError"*) that boundary exists for. `test_wrap_started_wraps_sqlite_error` failed immediately. **An optimisation placed outside the error boundary is a hole in the boundary.**
- `spore-699` (wrap-lock owner/age-refusal) is **the next anneal lane**, and `BEGIN IMMEDIATE` has now moved its premise: part of the store-side CAS it called for is in place.

---

## ✅ ARCHIVED 2026-07-14 — the SHIPPED LEDGER (0.4.x → 0.9.6), moved out of `next.md`

*Moved verbatim from `next.md` during the plan dangling-ref cleanup (2026-07-14). Every entry below is SHIPPED and verified; the plan now carries only open work. The dangling `spore-` pointers these lines carried (222 / 104 / 097 / 091 / 080 / 081) had all been composted out of the store — the detail was already inline here, so nothing was lost. **One residual was NOT done and was lifted back into the live plan before this moved:** `spore-080`'s structural BLOCK half (`save` refusing a graduation-with-zero-links) never shipped — only the WARN (`continuity.py:2584` Signal C).*

## ✅ SHIPPED LEDGER (reverse-chron — full detail in CHANGELOG + git + `COMPLETED_SESSIONS_ARCHIVE.md`)

> **PyPI latest = 0.9.6 (2026-06-30, `v0.9.6`).** The 0.9.x line shipped PUBLIC, DECOUPLED from Slice C (spore-222). The "rides the 0.9.0 shadow" items below are now RELEASED in `v0.9.6` — they no longer wait on Slice C. Slice B (`pattern_associations.py`) ships in SHADOW MODE (inert; nothing reads the graph for recall); Slice-C graph-CONSUMING recall stays the live track, gated on the §9.2 oracle.

- **0.9.6 — FIRST PUBLIC 0.9.x (2026-06-30, `4a43c83`, tag `v0.9.6`, PyPI + GitHub release).** spore-222: decoupled the anneal release from Slice C so the Tony-validated bundle ships for Chris (July 7) without waiting on the §9.2 oracle. PyPI was at 0.8.5; the whole 0.9.x line (0.9.0 shadow → 0.9.5) was main/editable-only, never published → collapsed into 0.9.6. Bumped 0.9.5→0.9.6 (new public `anneal_memory.sessions` module/API — AM-CONSOLIDATE-EFFERENT — landed after the 0.9.5 label). codex L3 release/packaging review (non-replaceable) → 1 Medium: the AM-WRAP-GENERATED migration entry was keyed at the never-published 0.9.5 → re-keyed to 0.9.6 (fail-safe; a local-0.9.5-acked operator upgrading to 0.9.6 now surfaces the wrap-protocol retirement). L4 e2e: a clean-venv install of the built wheel AND a real `pip install anneal-memory==0.9.6` from PyPI both pass the full smoke (version, import surface, Slice-B shadow-table creation, prepare_wrap, the validated_save_continuity immune pipeline, the sessions/consolidate-efferent API, SporeStore). 1642 tests green, mypy clean, twine check passed. Glama/MCP-registry sync is auto (GitHub-source crawl). **▶ HANDOFF (flow-repo/Levain convo, NOT done here): bump the Levain anneal pin `>=0.9.4` → `>=0.9.6` for the Chris bundle.**
- **AM-SPORE-CAS — 0.9.4 (2026-06-23, `a3c9a2f`).** `SporeStore.ascend` gained optional `expect_disposition` CAS inside the resolve txn → closes the cross-process read-then-resolve TOCTOU on a disposition-aware host guard (a `loop→note` flip now aborts the resolve). Additive (`_UNSET` skips). Surfaced by Levain's KEEP-NOTE LIFECYCLE L3 (codex HIGH); Levain `_apply_spore_verb` threads it through (pin `>=0.9.4`). 4 tests, 1590 green.
- **AM-RECALL-IDF — 0.9.3 (2026-06-21, `57511eb` + flow `3ffaae7`).** The recall-PRECISION blocker CLEARED. Mechanism CORRECTED on the real corpus: not degree-bias (refuted — uniform evidence-degree) but **term-frequency bias** (`_keyword_weight` was a corpus-blind length proxy → high-df process words floated flow-meta episodes). Fix: per-pattern `RelevantPattern.source` (`spore-104` dep-2) + corpus-IDF weight (`_idf_weight`, engages > 50-episode corpus) + a √N distinctiveness anchor (the structural guard L2 forced). Measured: receipt-corpus surfacings 115→~16; noise/process prompts → empty; genuine queries retained.
- **`spore-104` dep-1 — the pattern-graph PROJECTION-CHECKPOINT / high-water-mark primitive (2026-06-21, anneal `e9d1191` + flow `3729f54`; THIN pin-not-rebuild).** `(projection_version, high_water_mark)` over the association read-model so a Slice-C receipt PINS which graph state produced it (hwm = monotonic +1 per committed edge mutation, gated in-txn; pv = constant 1). Flow recall-hook stamps from the SAME read-only Store open serving recall (no extra read; shadow invariant intact). codex L3 caught a `busy_timeout` fact both Claude layers got backwards (read_only inherits Python's 5s default) → `spore-148`. BOTH Slice-C upstream deps now DONE. 1586 + 30 tests.
- **§9 step-0 (durable receipt corpus) + step-1 (per-drain graph snapshot) — 2026-06-21 (flow-side, rides the shadow).** step-0 = `recall_injection_hook._append_durable_receipt` tees every receipt to an append-only, flock-serialized (`LOCK_NB`), rotating log the drain never touches (`RECEIPT_VERSION=2`: + `query_date` local bucket key). step-1 = `anneal_dualwrite._snapshot_graph_if_advanced` pins the committed graph at each hwm advance via `VACUUM INTO` temp→atomic-publish→manifest (idempotent by (pv,hwm); `state/graph_snapshots/`, cap 40 ≈ 2+ wks). codex L3 caught 2 HIGH the Claude pass missed (blocking flock on the hot path; missing local `query_date` → UTC mis-bucket). 23 + 30 tests.
- **AM-PYTYPED — 2026-06-15 (`7b638a8`, `spore-097`).** Empty `py.typed` (PEP 561) ships in the wheel; typed consumers (Levain/flow/OpenHands) lose `import-untyped`. Surfaced + fixed a type-ergonomics gap: widened `DESCEND_BY_TYPE`/`ASCEND_BY_TYPE` keys `Literal`→`str`. Apparatus L1 + L3-codex CLEAN; mypy clean (17 files), 1540 tests.
- **spore-091 (AM-CONTLOCK hardening) — 2026-06-15 (`a07f2bb`).** `continuity_lock` now `.resolve()`s the `.lock` sidecar INSIDE the primitive (a symlinked dir gave two `.lock` spellings → silent non-serialization) + a `require=` strict mode that raises `ContinuityLockUnavailable` instead of degrading (Levain's State write fails CLOSED; anneal's own save stays best-effort). +6 tests. Levain consumer shipped same session (`7f5f15d`). *(Base: AM-CONTLOCK `7ab7991` — the shared cross-process continuity `flock`.)*
- **AM-SNAPSHOT ① — the durable recovery oracle — 2026-06-14 (`bf3f0d6`).** Persists continuity `content_hash` + `<token12>-<uuid8>` `pair_id` in the `wraps` row INSIDE the Phase-2 commit, so orphan recovery SELF-CLASSIFIES (`committed_verified`/`…unverifiable`/`…hash_mismatch`/`debris`/`inconclusive`) even when a crash beats the Phase-4 audit. Never auto-acts (action stays the operator's). codex L3 caught an extra-dotted-parse fail-open + a half-oracle gap; L4 proved crash→oracle→recover LIVE. +24 tests. Closes the Levain-2b-i coordination surface. **gap ② restore ❌ KILLED** (architecturally incoherent — the continuity text is a projection of the 5-layer store; restoring desyncs it).
- **Slice B (AM-LINKGATE-DECAY) — BUILT, SHADOW MODE, 0.9.0 (2026-06-12, `75b97ae`; release deferred to Slice C).** The cortical pattern-association graph end-to-end: library graph (`pattern_associations.py` — co-graduation seeding + idempotent co-surface drain w/ burst-damp + provenance gate, lazy calendar decay + homeostatic normalization, rename/homonym lifecycle, telemetry) + Store wrappers + CLI; flow-side producer (recall-hook→spool) + drain + oracle (`pattern_graph_oracle.py`, HARDENED 2026-06-14 `4bd56c8` — topology/retrieval-health-aware, fails exit-2 loud on wrong/empty store). codex L3 caught 2 HIGH (batch-rollback seeding → post-commit txn; producer/rotation race → deferred-unlink). L4 11/11 incl. the shadow invariant LIVE (a recall-hook run leaves the graph byte-identical — the echo-chamber structurally cannot form). The 106→11 Hebbian collapse (Phill caught 2026-06-11) was recovered → total_links 94; structural rule in `feedback_wrap_underwires_associative_layer.md` + `spore-080`.
- **Slice A (AM-PROVENANCE) — 0.8.5 (2026-06-12).** The inert `[provenance: id,id]` marker silences the carried-forward graduate-OUT cry-wolf for grounded mature patterns. Argus's 2 carried 3x RE-GROUNDED 2026-06-14 (flow drove an interactive Argus codex re-stamp — his process wrote his store, sovereignty preserved); generator fixed so the autonomous wrap self-corrects.
- **0.8.0–0.8.4 — the crystallized-pattern tier (AM-CRYSTAL) + linkgate (2026-06-08→12).** 0.8.0 AM-CRYSTAL-RECALL (Hebbian-associative retrieval backend, replacing keyword scoring — the conceptual-corpus differentiator, un-gated from Step-C; query → keyword-matched episodes → one Hebbian hop → patterns citing reached episodes; strictly additive + keyword-first; the CURE delivered LIVE 06-08 via a 12-pattern evidence-backfill + the read-only-Store hook repoint) → 0.8.1 CLI parity → 0.8.2 MCP tools (`crystal_recall`/`crystal_index`) → 0.8.3 AM-LINKGATE (AM-WARN Signal C nudges single-id under-wiring) → 0.8.4 `spore-081` fix (`upsert_pattern_history` anchored `last_seen_at` to wall-clock → broke the AM-PRESERVE warm gate under deterministic/skew runs; now anchors to the pipeline's logical date). Reach COMPLETE across all transports.
- **0.7.0–0.7.2 — solo-safety + sycophancy gate + docs (2026-06-06).** 0.7.0 AM-CRYSTAL-SOLO-SAFETY (true opt-in at CLI/MCP — no auto crystallize-OUT proposals for a no-retrieval operator + the always-loaded crystal index + decision-channel `parse_crystal_decisions`, never-compost-timeless gate STRUCTURAL in the parser — codex L3 to convergence across 6 passes, replaced the enumerated name-char alphabet with a structural positional anchor) → 0.7.1 docs (the README CLS section) → 0.7.2 AM-PRESERVE-VS-SYCOPHANCY (the immune gate no longer erodes its own load-bearing antibodies: discriminator = byte-identical OR non-inflating+warm+fresh-SPECIFIC grounding; the cached "drift-baseline" fix-direction was corrected en route to fresh-grounding — `preservation_gate_discriminate_by_drift_not_vocabulary`; codex caught a co-citation graph-poison both Claude reviewers missed).
- **0.6.0 + 0.4.x/0.5.x + earlier — COLLAPSED.** 0.6.0 AM-CRYSTAL store-half (superseded by 0.7.0). 0.5.0 AM-SEMDUP + AM-ROLECHECK. 0.4.8 AM-PRESERVE-BARE-PATH (the bare-graduation sunset path holds a warm carried Proven). 0.4.4–0.4.6 AM-INITSCHEMA/SCHEMASNAPSHOT + the immune-half-dead arc (AM-HISTUPSERT-BULLET + AM-CARRYFORWARD + AM-PERNAME-LINEBIND). 0.4.3 AM-XSESSION-LINKGATE + AM-CONTRASCAN-EMIT. 0.4.2 AM-SCHEMA-BUDGET + AM-WARN + AM-PREPARE-GUARD. 0.4.1 AM-QUOTEFOOTGUN (decoupled Hebbian link formation from the explanation-overlap gate). 0.4.0 SPORES FOUNDATIONS (`SporeStore` + spore CLI + 8 MCP tools + SP-CONCURRENCY fcntl fix). The Diogenes contradiction-sweep repoint onto the neocortex (2026-06-02). Full detail → CHANGELOG `[0.4.x]`/`[0.5.0]`/`[0.6.0]` + git + `COMPLETED_SESSIONS_ARCHIVE.md`.


## AM-WORKINGSET — the 2026-06-05 design SEED (SUPERSEDED 2026-06-06) — retained for the design trail

> **Superseded 2026-06-06** by the crystallized-pattern-tier architecture (`next.md` 🔴 AM-WORKINGSET). The seed got the CLS frame, the timeless/phase tag, and propose-not-auto right; its FIX (crystallize stable Provens to always-loaded `partnership.md`/`me.md` bedrock) was **flow-N-of-1** — a *relocation*, not a solution (both files are @import'd → total always-loaded context unchanged) — and ignored solo-anneal/Levain adopters who have no bedrock layer + no consolidate. The breakthrough session replaced "bedrock file" with a **spore-sibling crystallized store retrieved on-demand via the harness recall hook** (4 tiers, anneal/harness seam). Original seed preserved verbatim:

**Decided 2026-06-05 (Phill, end of the sovereignty-probe session).** Trigger: that night's wrap hit **28795 chars vs a 25500 budget**, save warned two 3x Provens (`verify_or_surface`, `full_artifact_reread`) "keep needing the carryforward hold → candidates to graduate OUT or retire." Structural — `## Patterns` is a one-way ratchet pressuring total length, forcing the live sections (State/Context/Understanding) underwritten.

**Root cause:** `## Patterns` conflates three concepts the graduation ladder fuses — (1) confidence (1x→2x→3x, built), (2) permanence (timeless vs phase-specific — NOT modeled), (3) active relevance (working-set-now — NOT modeled). IN path exists; no real OUT path → Proven accretes forever. AM-CARRYFORWARD (0.4.6) made it worse: it holds everything → the section only grows. Bloat = carryforward working as designed against a missing exit.

**Reframe — incomplete CLS.** anneal is CLS-framed (episodic=hippocampus → continuity=neocortex) but missing its top tier (cortically-independent schema). `## Patterns` silently does double duty (working buffer AND archive of crystallized truths); the archive crushes the buffer. Three tiers proposed: Episodic (full trail) / `## Patterns` (working set, bounded) / `partnership.md`+`me.md` (bedrock, always-loaded, no budget cost). [The 2026-06-06 architecture SPLIT the bedrock tier → crystallized-store (on-demand) + constitution (always-on), making it 4 tiers — see next.md.]

**Three forces + the missing primitive:** Promote (exists); Crystallize OUT (carryforward hold-count = readiness; the hold is a waiting-room); Forget/compost (cold+phase-specific → episodic trail kept); the routing primitive = a `timeless` vs `phase-specific` tag; carryforward must only protect `timeless`. **Strict budget = the forcing function** (hard cap → OUT paths fire). The seed's 4 forks (auto-vs-propose → propose; who-tags → composer; destination → partnership/me; components AM-PATTERN-BUDGET/AM-TIMELESS-TAG/AM-CRYSTALLIZE/AM-COMPOST) were settled/superseded in the 2026-06-06 session. **Risk gate (carried forward unchanged):** only ever forget `phase-specific`, never `timeless`; episodic recall is the safety net.

---

## v0.3.1 — Phantom-re-save fix + 4-layer review pass — COMPLETE (May 17, 2026)

**Session shape.** First session post-un-archival (project un-archived 2026-05-17 as flow's primary build bet). Shipped v0.3.1 end-to-end: code fix → transport/type cleanup → test sweep → README + docs → version bump + CHANGELOG → 4-layer review → review-fix pass → commit/tag/push → build → PyPI → GitHub release → nexus install. **Commit:** `3f79a00` on `main` (amended once — an earlier commit captured only 2 files because a `git stash` in a lint-baseline check had unstaged the work; caught via `git show --stat` before push). Tag `v0.3.1`. **PyPI:** https://pypi.org/project/anneal-memory/0.3.1/. **GitHub release:** https://github.com/phillipclapham/anneal-memory/releases/tag/v0.3.1. **Tests:** 704 → 707. **Diff:** 18 files, +370/−191.

### What shipped — the fix

nexus's session-3 audit showed 1 real wrap + 3 phantom re-saves (no token) — a correction loop where the immune system's demotion feedback drove the agent to re-save. Root cause found in the code: `wrap_completed` already cleared the wrap snapshot (consume-once was already done); the gap was that `validated_save_continuity` *tolerated* `load_wrap_snapshot() == None` by falling through to a legacy `skipped_prepare` path that re-fetched the (now-empty) episode set and saved anyway — graduation then ran against zero valid IDs and demoted every citation, and `sessions_produced` incremented each pass.

- **`continuity.py`:** `validated_save_continuity` raises `ValueError("No wrap in progress…")` when `load_wrap_snapshot()` is None. Wrap-state preconditions (snapshot + token) checked *before* continuity-text validation (state-before-payload — an agent shouldn't burn a turn fixing markdown for a doomed save).
- **`skipped_prepare` removed** clean from the `SaveContinuityResult` TypedDict + the MCP `save_continuity` text response + the CLI `save-continuity --json` output. Pre-1.0 breaking change, disclosed.
- **`store.py`:** docstring-wording only, no behavior change.

### What shipped — README + docs

- README MCP setup reworked: three verified per-harness config blocks (Claude/Cursor/Windsurf JSON, Codex `.codex/config.toml`, Gemini `.gemini/settings.json`), all with the `serve` subcommand. `server.json` updated.
- "CLAUDE.md snippet" reframed to "agent-instructions snippet" across README + docs; `examples/CLAUDE.md*.example` renamed to `agent-instructions*.example`; "wrap once" guidance added to both snippets + README Session Hygiene; a stale "two-layer" → "four-layer" header fix.
- `docs/library-quickstart.md` de-stale'd (it taught `skipped_prepare` + a `_build_wrap_package()`-then-save recipe v0.3.1 breaks).

### The 4-layer review

L1 session-code-review + L2 domain-expert (agents) + L3 cross-substrate mesh (complement/gemini/codex via `consult.py --diff`) + L4 flow attention-mode. Caught, all fixed pre-ship:
- **A confident factual error of flow's:** the CHANGELOG claimed the old MCP config "would not have launched the server." False — `cli.main()` delegates to the server on a no-subcommand invocation; the bare form always worked. flow had verified `serve` works but *assumed* the bare form fails (its earlier bare-form probe had failed silently on a missing `timeout` binary and was never re-run). Verified directly, framing corrected.
- `docs/library-quickstart.md` stale `skipped_prepare` (3-way reviewer convergence) → rewritten.
- gemini's validation-order finding (state-before-payload) — taken over complement/codex's "current is acceptable"; their counter didn't hold (checking the snapshot is a pure read, doesn't strand it).
- Integration guides audited: all 12 already pair `prepare_wrap` with the save — no breakage.

### Key decisions

- Clean break on `skipped_prepare` (remove the field) over keeping a vestigial always-False field — `dead_code_elimination_beats_defensive_documentation`.
- Validation reorder: wrap-state preconditions before payload validation — agent-facing API, a wrong error costs a context turn.
- `serve` is the recommended explicit form, not a fix for a broken config — the bare no-subcommand form is backward-compatible.

### Tests

704 → 707. New: `test_phantom_resave_after_completed_wrap_is_refused` (core regression — completed wrap then second save raises, no extra wrap row, no `sessions_produced` increment), `test_empty_prepare_wrap_then_save_is_refused`, `test_refusal_precedes_text_validation`. The three former `skipped_prepare` tests repurposed (two assert the refusal, one drives `store.wrap_completed` directly). ~26 cold `validated_save_continuity` / MCP-`save_continuity` call sites across the suite now run `prepare_wrap` first.

---

## v0.3.0 — Deprecation cleanup release + Session 10.6 + 4-layer review pipeline — COMPLETE (May 1, 2026 LATE PM)

**Session shape.** ~3-hour focused execution sweep on a Pressable-easy day. Full Session 10.6 v0.3.0 prep cleanup that had been overdue since ~Apr 20. End-to-end: re-grep deprecation surface → fix 2 Diogenes May 1 LOWs → rewrite 12 test sites → run baseline tests → delete deprecated public surfaces → run tests → 4-layer review pipeline catching 2 mechanism-accuracy errors → fix all findings → version bump + CHANGELOG → commit + tag + build + twine upload + gh release. **Commit:** `f0a36ff` (v0.3.0) on `main`. Tag pushed. **PyPI:** https://pypi.org/project/anneal-memory/0.3.0/. **GitHub release:** https://github.com/phillipclapham/anneal-memory/releases/tag/v0.3.0. **Tests:** 708 → 704 passing (net -4: -7 removed deprecation-machinery tests + 3 new structural-invariant tests).

### What shipped

**Breaking changes** (both deprecated since v0.2.0):
1. `prepare_wrap_package()` public wrapper REMOVED. Function deleted from `continuity.py`, removed from `__init__.py` imports + `__all__`. Migration: use `prepare_wrap(store, ...)` for canonical pipeline, or call private `_build_wrap_package()` for advanced custom lifecycles (no API stability guarantee).
2. `Store.wrap_started()` signature TIGHTENED to required keyword-only `token: str` + `episode_ids: list[str]`. Legacy no-arg form (which produced partial wrap-in-progress state with empty token) gone. Calling with no args → TypeError. Empty token → ValueError. Non-list `episode_ids` (string, tuple, generator) → TypeError.
3. `Store.wrap_started()` audit-log shape: `wrap_episode_count` + `wrap_episode_ids` now ALWAYS present (previously conditional). Restores forensic discrimination between "agent explicitly started wrap with empty snapshot" and "audit entry from legacy version that didn't log these fields."

**Diogenes May 1 LOWs (both fixed):**
1. `Store._batch()` exception handler — added `if self._conn is not None:` guard at both rollback sites. Defensive against AttributeError-masks-primary-error if a future refactor widens the reachable state.
2. `StoreError` class "Raised by" `_db_boundary` enumeration — replaced inline drift-prone enumeration (which omitted `wrap_completed` + `load_wrap_snapshot`) with pointer to canonical source: grep at `with self._db_boundary(...)` call sites. Disambiguates `_db_boundary`-wrapped subset from file-write surfaces.

**Review-pass docstring/code hygiene fixes (7):**
- `load_wrap_snapshot()` Returns/Raises docstring drift corrected — case 2 raises StoreError, was claiming returns None
- `load_wrap_snapshot()` StoreError message dropped reference to removed no-arg API
- `validated_save_continuity` comment updated to drop reference to removed no-arg form
- `_build_wrap_package` docstring uses canonical `wrap_started(token=..., episode_ids=...)` form
- `wrap_started` runtime `isinstance(episode_ids, list)` guard — closes generator atomicity hazard (where `list(generator)` exhausts and `len()` would TypeError after SQL writes commit)
- DeprecationWarning regression gate narrowed to anneal_memory-specific filenames (avoids CI-flake on Python stdlib upgrades)
- Module-level `_build_wrap_package` import in `TestBuildWrapPackage` (eliminated 8x per-method imports)

### 4-layer review pipeline (proactive per FLOW_DEV_PROTOCOL)

**Layer 1 (session-code-review)** — caught 2 HIGH (load_wrap_snapshot docstring drift + StoreError message references removed API) + 4 MEDIUM (validated_save_continuity comment + _build_wrap_package docstring + missing TypeError regression test + _conn guard rationale comment style) + 2 LOW. APPROVE WITH NOTES verdict.

**Layer 2 (domain-expert review on Python library API design + cognitive-loop semantics)** — caught the FIRST mechanism-accuracy error in the v0.3.0 docstring rewrite itself: the `StoreError` enumeration replacement claimed "the canonical list of wrapped operations is the StoreOperation Literal" but the Literal includes file-write surfaces NOT `_db_boundary`-wrapped. Plus 2 MEDIUM (load_wrap_snapshot docstring drift, audit-log chain-of-custody asymmetry on empty snapshots) + 2 LOW.

**Layer 3 (consultation team — complement + gemini + contrarian, --diff HEAD, $3.58 / 7.5 min)** — Complement caught the SECOND mechanism-accuracy error in the SAME docstring after Layer 2's fix: "[file-write helpers] have their own bullets above" was true for `save_continuity` / `save_meta` but FALSE for `_prepare_continuity_write` / `_prepare_meta_write`. Plus coverage gap: no test exercised `episode_ids=[]` round-trip (the canonical empty-snapshot encoding `"[]"` truthy through partial-state guard); one-character change (`if not json.loads(ids_raw):`) could have silently broken canonical empty case. New `test_load_wrap_snapshot_empty_list_round_trip` test locks the round-trip. Plus 4 LOW. Gemini affirmed structural maturity jump. Contrarian explicitly disclaimed code review (role mismatch) but provided strategic pre-1.0 deprecation policy framing.

**Layer 4 (integration semantics smoke test)** — end-to-end canonical pipeline works (2 episodes recorded, prepare_wrap → validated_save_continuity, skipped_prepare=False). `prepare_wrap_package` no longer importable. No-arg `wrap_started()` raises TypeError. README + MCP tool descriptions intact. PASS.

### Calibration finding — consultation_blind_to_mechanism_errors fired TWICE in single release docstring rewrite

The v0.3.0 docstring rewrite to fix Diogenes's drift introduced a NEW mechanism-accuracy error (the StoreOperation-as-canonical-list claim caught by Layer 2). After Layer 2's fix, Complement caught the residual error in the same paragraph (file-write helpers don't have "their own bullets above"). Two consecutive instances of the failure mode in the same surface. The calibration finding is now ~5 firings — strong Proven graduation candidate. Reinforces that code-level verification is non-replaceable for prose describing implementation; consultation is necessary but not sufficient.

### Test surface

| Removed (7) | Added (3) |
|---|---|
| `TestPrepareWrapPackage` (9 tests on deprecated wrapper) → replaced 1:1 by `TestBuildWrapPackage` targeting private helper (same coverage, net 0 here) | `test_wrap_started_missing_args_raises_typeerror` — locks Python's required-kwargs enforcement structurally |
| `TestPrepareWrapPackageDeprecation` (7 tests on deprecation machinery) → replaced by 2 in `TestCanonicalPipelineNoDeprecation` (regression gate against any deprecated surface, narrowed to anneal_memory-specific) | `test_wrap_started_non_list_episode_ids_raises` — covers isinstance guard incl. generator atomicity hazard |
| 3 deleted tests on legacy `wrap_started()` no-arg form (premises gone with the API) | `test_load_wrap_snapshot_empty_list_round_trip` — locks the canonical `episode_ids=[]` encoding round-trip (Layer 3 caught coverage gap) |

12 test sites across 5 test files (test_audit, test_continuity, test_store, test_server, test_associations) rewritten from legacy `with pytest.warns(DeprecationWarning, match="legacy call form"): store.wrap_started()` to canonical `store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])`. Also added `test_wrap_started_empty_token_raises` (covers ValueError on explicit empty token).

### Process notes

- **Pre-flight discipline:** re-grep deprecation surface as first action verified nothing slipped in during v0.2.1-v0.2.3 cycle. Found exactly the two known surfaces; one scope correction (next.md said 4 audit-test sites, actual was 12 across 5 files).
- **Sequence:** Diogenes LOWs (2) → test rewrites (12 sites) → green baseline → public wrapper deletion → wrap_started signature tightening → green again → 4-layer review → 7 review-pass fixes → green → version bump + CHANGELOG → commit + tag + build + twine + gh release. Whole arc captured in transcript; commit message lists all changes.
- **Calibration:** v0.3.0 explicitly ran TWO rounds of code-level verification on every docstring claim about code mechanism in the diff because v0.2.3 calibration finding made the failure mode a watch-line; Layer 2 + Layer 3 each caught one anyway. Validates the discipline (verification works) AND the failure mode (consultation cannot substitute for code-grounded review).

### Strategic context (forward-looking — not part of v0.3.0 itself)

Same session ALSO produced the autonomous_org project at `projects/autonomous_org/` reframing Session 12 as the substrate for a 0-human business operating system — anneal-memory v0.4 multi-agent extensions become Phase 1 of that broader strategic arc. See `projects/autonomous_org/brief.md` and `decisions.md` for full picture. anneal-memory itself stays substrate-class; the autonomous-org work happens at a different layer.

---

## v0.2.3 — Mechanism-accuracy fix (Diogenes overnight LOWs from v0.2.2 review) — COMPLETE (Apr 30, 2026 morning)

**Session shape.** Surgical doc-fix patch on a Pressable-heavy day, ~30 min wall. Two doc-only fixes from Diogenes overnight review of v0.2.2 (commit `1f8d82c`). No public API changes. No behavior change. Drop-in upgrade from v0.2.2.

**Commit:** `71fa971` (v0.2.3) on `main`. Tag pushed. **PyPI:** https://pypi.org/project/anneal-memory/0.2.3/. **GitHub release:** https://github.com/phillipclapham/anneal-memory/releases/tag/v0.2.3. **Tests:** 708/708 passing both before and after the version bump.

### What shipped

1. **README § Memory poisoning resistance — point 2 mechanism description was wrong.** v0.2.2 shipped point 2 as *"Anti-inbreeding catches sustained near-duplicate poisoning. ... the explanation-overlap check rejects citations whose content is too similar to the graduation claim."* The actual `check_explanation_overlap(explanation, episode_content)` function in `graduation.py:230` does the opposite comparison: requires at least 2 meaningful words from the **explanation** to appear in the cited **episode's content**, catching ungrounded/fabricated explanations rather than near-duplicate content. v0.2.2's prose conflated the explanation-overlap check (anti-fraud) with citation-reuse caps in `detect_citation_gaming` (anti-repetition). Point 2 now reads *"Explanation-grounding check rejects ungrounded citations"* — heading and mechanism description both corrected. Security argument preserved (poisoned trajectories where the attacker controls graduation-claim text but cannot rewrite the cited episode body still fail this check) but framing flipped from anti-repetition to anti-fraud. `README.md:397` single-paragraph rewrite.

2. **`StoreError` class "Raised by" enumeration was missing the cross-method closed-store guard.** Every `_db_boundary`-wrapped public method (`record`, `get`, `delete`, `recall`, `episodes_since_wrap`, `status`, `wrap_started`, `wrap_cancelled`, `get_wrap_started_at`, `get_wrap_history`, `record_associations`, `decay_associations`, `get_associations`, `get_association_context`, `association_stats`, `prune` — 16 methods total) raises bare `StoreError("Cannot {operation} on a closed store", operation=..., path=...)` via the boundary's pre-yield closed-state check (`store.py:2397`). v0.2.2's docstring documented `_db_boundary`'s `StoreDatabaseError` behavior but omitted this parallel `StoreError` path; callers writing `except StoreDatabaseError:` around a method on a possibly-closed store would miss the closed-store guard. `store.py:177-200` docstring expansion now enumerates all 16 wrapped methods and explicitly notes that `StoreDatabaseError` is NOT raised on this path because no SQL ran.

### Pattern observation (third sibling in the docstring-drift family)

Both fixes are siblings of the v0.2.1/v0.2.2 docstring-drift family at the same surface (parallel docstrings within `store.py`). Same generalization stands: when fixing a docstring-drift bug, check parallel docstrings + class-level enumerations for the same class of error before declaring fix complete. v0.2.1 fixed `_db_boundary` opening line. v0.2.2 fixed `close()` Raises section + initial `StoreError` "Raised by" expansion (added `wrap_completed` + `close`). v0.2.3 fixed the third sibling: cross-method closed-store guard wasn't enumerated.

### Calibration finding — consultation review is blind to mechanism-accuracy errors in code-describing prose

Both v0.2.2 sections that needed correction in v0.2.3 passed full 3-agent consultation review (complement + contrarian + anansi) before commit. The reviewers caught calibration over-claims (eTAMP "structural defense" → "structural inference"; HeLa-Mem "convergent" → "adjacent"; Article 12 "increasingly read as" → predictive hedge) but did NOT catch that point 2 described the wrong code mechanism — because consultation agents do prose review, not code review. Operational lesson: for prose that describes implementation mechanisms (Security/§-Validation/§-Audit sections, "the X check rejects Y" claims), consultation is necessary but not sufficient. Code-level verification — opening the named function and confirming the prose matches — is non-replaceable. Saved as `feedback_consultation_blind_to_mechanism_errors.md` in flow's feedback memory + indexed in `MEMORY.md` + documented in v0.2.3 CHANGELOG Meta section. Sibling/extension of `feedback_daily_sharpening_can_propose_instrumentation_gap.md` (Apr 29).

### Process notes

- **Phase order:** read README target → grep for actual `check_explanation_overlap` definition → read its implementation in `graduation.py:230` → confirm Diogenes's claim about the comparison direction → verify there's no separate anti-inbreeding mechanism that catches near-duplicates (found `detect_citation_gaming` / `citation_reuse_max` instead, but those are a different mechanism with different attack-class coverage) → write the corrected prose around what the code actually does, not what we wished it did → grep `_closed`/`_db_boundary`/`StoreError` to find the closed-store guard sites → expand the docstring "Raised by" list with the 16 methods explicit.
- **Release sequence:** version bump (3 files: `__init__.py`, `pyproject.toml`, `server.json` ×2) → CHANGELOG entry → 708/708 test pass on bumped state → commit `71fa971` → push to main → tag v0.2.3 → push tag → `python3 -m build` (sdist + wheel) → `twine upload` (using `PYPI_API_TOKEN2` from `.env.flow`) → `gh release create` with CHANGELOG body. Total wall ~30 min.
- **Glama:** `glama.json` is just `{maintainers: ["phillipclapham"]}` — no webhook config. Glama auto-syncs from PyPI + GitHub releases on a schedule; no manual trigger executed this cycle.

---

## v0.2.2 — README positioning expansion + StoreError docstring completeness fix — COMPLETE (Apr 29, 2026 morning)

**Session shape.** Single morning docs-touch via FLOW_DEV_PROTOCOL — three calibrated README positioning sections + 2 Diogenes overnight LOWs + arXiv verification pass that caught 2 Daemon-brief mischaracterizations + 3-agent consultation that caught 3 calibration over-claims. Commit `1f8d82c` (v0.2.2). **PyPI:** https://pypi.org/project/anneal-memory/0.2.2/. **GitHub release:** https://github.com/phillipclapham/anneal-memory/releases/tag/v0.2.2. **Tests:** 708/708 passing (no library code changes — README + docstring only).

### What shipped

**Three calibrated README positioning sections (all post-3-agent-review):**

1. **Security § Memory poisoning resistance.** New subsection citing eTAMP attack class ("Poison Once, Exploit Forever," Zou et al., [arXiv:2604.02623](https://arxiv.org/abs/2604.02623), April 2026) — environment-injected memory poisoning against ChatGPT Atlas + Perplexity Comet + OpenClaw, 32.5% ASR on GPT-4-mini. Three-mechanism partial defense: single-shot poisoning stalls at 1x (no continuity-layer influence without independent citations), citation-overlap check (described WRONG initially — see v0.2.3 for correction), SHA-256 audit trail provides forensic surface. Explicit *"structural inference, not empirical defense — anneal-memory has not been tested against eTAMP directly"* disclaimer; sustained adversarial campaigns with diverse contaminated trajectories can still graduate (immune system bounds *cost* of poisoning attacks, not the possibility).

2. **Consolidation Landscape § April 2026 adjacent architectures.** Names HeLa-Mem (Zhu et al., [arXiv:2604.16839](https://arxiv.org/abs/2604.16839), ACL 2026 accepted, explicit Hebbian + Reflective Agent) and GAM (Wu et al., [arXiv:2604.12285](https://arxiv.org/abs/2604.12285), April 2026 preprint, hierarchical graph-based — *not* classical Hebbian). Distinguishes by architecture-class AND peer-review status. Both decouple encoding from consolidation; neither has citation-validated quality gates. Differentiator is *what gates the consolidation*, not whether consolidation happens.

3. **Compliance § Provenance vs timestamps.** Descriptive distinction between timestamp-only logs and provenance chains for Article 12 traceability. anneal-memory ships at the provenance-chain level (SHA-256-chained audit + citation-required graduation). Predictive hedge: as regulatory guidance and case law develop through August 2026 enforcement and after, this distinction may become a differential compliance gate. AWS AgentCore reference date-stamped to "April 2026 architecture" so the contrast doesn't bind to a moving target.

**Two Diogenes overnight LOWs fixed:**

- `store.py` `StoreError` class "Raised by" enumeration was incomplete. Documented `Store.save_continuity`, `Store.save_meta`, `Store.load_wrap_snapshot`; was missing `Store.wrap_completed` (raises bare `StoreError` when `episode_ids` exceeds the SQLite IN-clause variable limit at line 1544) and `Store.close` (raises bare `StoreError` when called inside an active `_batch()` context at line 3006). Both are public methods callers invoke directly. Trailing summary line widened from "file-write + integrity paths" to "file-write + integrity + invariant-guard paths" to accurately reflect the close() guard. *(v0.2.3 caught a third sibling Diogenes review surfaced overnight — cross-method closed-store guard via `_db_boundary` pre-yield check — fixed in v0.2.3.)*

- `store.py` `close()` Raises section documented only `StoreDatabaseError`. Now also documents the `StoreError` path raised when called inside `_batch()`, including caller guidance to widen `except StoreDatabaseError:` to `except StoreError:` if the guard surface matters to error handling.

### Carry-forward residual

Jain et al. citation '10%' lower bound is unverifiable from code — confirm against Jain et al. (CHI 2026, [arXiv:2509.12517](https://arxiv.org/abs/2509.12517)) before any academic submission of anneal-memory work. Not a release blocker; precision-check at paper-submission boundary.

### Process notes — calibration discipline (v0.2.2 was the substrate that surfaced two pattern findings)

**arXiv verification pass caught 2 Daemon-brief mischaracterizations** before commit:

1. eTAMP requires environmental contamination + agent's own consumption — does NOT require write access to the agent's memory store. Daemon's brief implied write-access requirement.
2. GAM is graph-based hierarchical memory, NOT classical Hebbian. Daemon's brief mischaracterized it as "Hebbian convergent."

Per `feedback_web_research_always.md` extension: scope-extended Apr 22 to include drafting public-artifact content with verifiable external factual claims. v0.2.2 morning was 2nd validating instance.

**3-agent consultation review (complement + contrarian + anansi) caught 3 calibration over-claims** in initial drafts:

1. eTAMP "structural defense" → "structural inference, not empirical defense" (no direct empirical evaluation; defense is architectural inference only).
2. HeLa-Mem + GAM "convergent validators of anneal-memory's architecture" → "adjacent multi-layer architectures that arrived in the same window with different consolidation primitives" (different starting points, different gating mechanisms; convergent framing doesn't survive first hostile read).
3. Article 12 "increasingly read as" → descriptive distinction (timestamp vs provenance) + predictive hedge ("may become a differential compliance gate as enforcement guidance and case law develop"). Wish-casting dropped.

**Calibration finding saved:** `feedback_daily_sharpening_can_propose_instrumentation_gap.md` family — Daemon overnight intel can over-claim on positioning surface; review-before-public-artifact-ship discipline catches what review-of-Daemon-alone misses. **This finding sets up the v0.2.3 follow-up** (consultation catches calibration over-claims but is blind to mechanism-accuracy errors in code-describing prose — see v0.2.3 archive entry for the v0.2.2 → v0.2.3 calibration-discipline arc).

**Pattern observation:** all 3 fixed-in-v0.2.2 README sections were AT THE positioning-prose substrate (calibration-class), not the mechanism-prose substrate. The v0.2.2 review caught positioning over-claims cleanly. It missed the mechanism error in point 2 of Memory poisoning section because that's a different review class — code-grounded prose review requires reading the actual function. v0.2.3 closed that gap.

---

## Session 10.5d+: v0.2.1 — 7 Framework Guide Verification + Diogenes Bundle — COMPLETE (Apr 14, 2026 morning)

**Session shape.** Single flow-mode session that shipped the Tue Apr 14 morning locked scope in full: the 3 actionable Diogenes overnight LOWs (+1 track-only) followed by end-to-end verification of the 7 remaining framework integration guides (Google ADK, LlamaIndex, Haystack, CAMEL-AI, Autogen/AG2, DSPy, claude-agent-sdk) against live `pip install`s. Same pattern as 10.5d but without the full 4-layer review on each fix — these are documentation corrections, not code changes to the library itself, and each fix was verified end-to-end via a dedicated `verify_fixed.py` script against real framework classes. Closed the session with a complete v0.2.1 release pipeline (version bump → CHANGELOG → tests → tag → build → twine → GitHub release → fresh-venv `pip install anneal-memory==0.2.1` verification).

**Pre-session test count:** 707 (v0.2.0 baseline). **Post-session test count:** 707 (no library test changes — all work was in docs + config + one docstring fix).

### Warmup: Diogenes overnight bundle (commit `63cbf52`)

From `diogenes-20260414-041622-e91b682b90d3`. All LOW severity. Handled first as a focused commit so the release narrative stayed clean. 3 actionable fixes + 1 track-only.

1. **LOW SEMANTIC — `_db_boundary` docstring opening line drift**: `store.py:2317` said "catches any `sqlite3.Error` subclass" but the implementation catches `sqlite3.DatabaseError` specifically (`InterfaceError` API-misuse bugs must propagate bare, by design — documented lower in the same docstring). The opening line contradicted the corrective paragraph a few lines down. One-line fix to match impl.

2. **LOW ARCH — `close()` inside `_batch()` emits misleading error**: `store.py:2975`. Calling `store.close()` while inside a `_batch()` context previously raised `"Cannot batch_commit on a closed store"` — attributed the failure to the wrong operation because `close()` was trying to execute the pending batch commit before tearing down the connection. Fix: added explicit guard at top of `close()` (after the idempotent `_closed` check):
   ```python
   if self._defer_commit:
       raise StoreError(
           "Cannot close() while inside _batch() context",
           operation="close",
       )
   ```
   `"close"` was already in the `StoreOperation` Literal so no drift test update needed.

3. **LOW PROCESS — CI pytest missing filterwarnings**: `pyproject.toml` `[tool.pytest.ini_options]` gained `filterwarnings = ["error::DeprecationWarning"]`. Local dev has always caught DeprecationWarnings via `-W error`; CI didn't enforce the same gate. Diogenes's overnight pre-check verified all 707 tests green with the flag set. Confirmed locally: 707/707 pass under the new config.

4. **LOW CODE_QUALITY — Assert-for-mypy-narrowing accumulating** (track-only, no fix this release): 3 sites currently (`continuity.py:505`, `continuity.py:897`, `server.py:153`). No current runtime risk (`python -O` is uncommon in this codebase's deployment). Gate: if this reaches 5+ sites, evaluate whether `if not X: raise StoreError(...)` is cleaner than the assert idiom. Noted in the CHANGELOG "Internal notes" section so future wraps can count it.

Diogenes also confirmed ALL PRIOR OPEN findings CLOSED during the 10.5c.5/10.5c.6 arc (`test_carried_forward_not_validated` assertion, CLI `--wrap-token` validation, `_tool_status` audit health surface, tombstone description honest). Clean slate coming into v0.2.1 except these four.

### 7-framework verification — results table

Setup per framework: `mkdir /tmp/am_frameworks/<name> && python3 -m venv venv && source venv/bin/activate && pip install -q <framework> -e ~/Documents/anneal-memory`. Then a dedicated `exercise.py` that imports every symbol the guide references, introspects signatures against `inspect.signature()` / `model_fields`, constructs every guide class with the kwargs the guide uses, and fires any hooks/callbacks through real framework objects. For every drift bug, a follow-up `verify_fixed.py` that exercises the corrected code against real framework classes end-to-end.

| Framework | Version | Result | Commit |
|---|---|---|---|
| Google ADK | `google-adk 1.30.0` | **2 bugs** | `91b2205` |
| LlamaIndex | `llama-index-core 0.14.20` | **2 bugs** | `51fa575` |
| Haystack | `haystack-ai 2.27.0` | **3 bugs** | `8474b2b` |
| CAMEL-AI | `camel-ai 0.2.90` | **1 bug** | `ed51dcd` |
| Autogen / AG2 | `ag2 0.11.5` | Zero drift | `51e0267` |
| DSPy | `dspy 3.1.3` | Zero drift | `d3e597d` |
| claude-agent-sdk | `claude-agent-sdk 0.1.59` | Zero drift | `1cdc440` |

**Running hit rate across 10.5d + 10.5d+ combined:** 9 of 12 guides (75%) had load-bearing drift, 15 integration bugs total fixed between v0.2.0 and v0.2.1.

#### Google ADK (commit `91b2205`, 2 bugs)

1. **`BaseMemoryService.search_memory` signature drift.** Guide used `def search_memory(self, query)` returning a dict with `"results"`. Real ADK 1.30 API is keyword-only three-tuple: `search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse`. Silent wrong subclass — ADK's runtime would have called it with keyword args the guide's single-arg signature didn't accept. Rewrote the `AnnealMemoryService.search_memory` method with the correct keyword-only signature.

2. **`MemoryEntry.content` shape drift.** `MemoryEntry.content` is a `google.genai.types.Content` wrapper (`Content(parts=[Part(text=...)], role=...)`), not a plain string. Guide returned plain dicts pretending to be MemoryEntry-shaped. Fixed to construct proper `Content` wrappers and return a real `SearchMemoryResponse(memories=[MemoryEntry(...)])`.

3. **Consumer-side drift (related to #2).** `tool_context.search_memory(query)` returns `SearchMemoryResponse` (with a `.memories` attribute), not a dict with `["results"]`. `ToolContext.search_memory` is intentionally single-arg at the tool layer because ADK fills in `app_name` / `user_id` from the invocation context before delegating to the service-level keyword-only form. Rewrote the `research_tool` example to iterate `prior.memories` and construct serialized dicts from real `MemoryEntry.content.parts[0].text` / `.author` / `.timestamp`.

Verification: `/tmp/am_frameworks/adk/verify_service.py` instantiates the fixed `AnnealMemoryService`, records two episodes, calls `search_memory(app_name="testapp", user_id="u1", query="fox")`, verifies the returned `SearchMemoryResponse` has real `MemoryEntry` objects with real `Content` parts, then exercises `add_session_to_memory(session)` with a real `Session` containing a real `Event` with `Content` and confirms the episode round-trips through the store.

#### LlamaIndex (commit `51fa575`, 2 bugs)

1. **`BaseMemoryBlock.name` required field.** Guide's custom `AnnealMemoryBlock` didn't declare `name` as a field. `BaseMemoryBlock` declares it as required in 0.14.20, so instantiation immediately raised `ValidationError`. Added `name: str = "anneal_memory"` to the subclass and passed it explicitly at `Memory.from_defaults(memory_blocks=[AnnealMemoryBlock(name=..., ...)])` for clarity.

2. **`_aput` signature drift (the silent-failure one).** Real API is `_aput(self, messages: list[ChatMessage]) -> None` — a LIST of messages being flushed out of short-term memory, no `**kwargs`. Guide had `_aput(self, message, **kwargs)` singular + kwargs. The silent failure mode is particularly nasty: LlamaIndex calls it with a list of messages, and guide's code then does `hasattr(message, "content")` on the list — which is False — and silently no-ops every flush. Zero episodes recorded, no error raised. Fixed to iterate the list and record each message as an episode.

3. **`_aget` parameter-name alignment.** Real API uses `**block_kwargs` not `**kwargs`. Behavior-equivalent rename but aligning it keeps diffs clean and reduces friction for users who compare the guide to the source.

4. **False positive caught during verification: `ChatMessage.content`.** My initial `exercise.py` flagged `ChatMessage.content` as missing from `model_fields` — that's true, because the model now uses `blocks: list[TextBlock|ImageBlock|...]`. But `content` still works as a property accessor for backwards-compat, and `ChatMessage(role='user', content='text')` still works as a construction kwarg. So the guide's `ChatMessage(role="system", content=f"Memory:\n{continuity}")` is fine. Noting this because it's the kind of "looks like drift, isn't drift" case future verifications might hit.

Verification: `/tmp/am_frameworks/llamaindex/verify_fixed.py` constructs `AnnealMemoryBlock`, calls `_aput([ChatMessage(role='user', content='first'), ChatMessage(role='assistant', content='second')])`, verifies two episodes land in the store, calls `_aget(messages=msgs)` and verifies it returns an empty list when no continuity is set, then calls `Memory.from_defaults(session_id="s1", memory_blocks=[block])` to confirm the Memory system accepts the block.

#### Haystack (commit `8474b2b`, 3 bugs — worst offender)

1. **`Tracer.trace` signature drift.** Real protocol is `trace(self, operation_name, tags=None, parent_span=None)` with `parent_span` as a kwarg. Guide's two-arg `def trace(self, operation_name, tags=None)` would have raised TypeError on first real invocation because Haystack internals always pass `parent_span=...`.

2. **Tracer missing `current_span()`.** Both `trace` and `current_span` are abstract on the base `Tracer` ABC. Guide implemented only `trace`. `enable_tracing(AnnealMemoryTracer())` wouldn't have caught it immediately because the guide's custom tracer class does NOT inherit from the `Tracer` ABC (so the ABC check doesn't fire at construction), but Haystack internals call `tracer.current_span()` from the `FunctionToTool` invocation path and elsewhere — `AttributeError` at runtime. Fix: added `_current_span: AnnealMemorySpan | None` instance state + a proper `current_span()` method that returns it + push/pop of the current span around the `trace()` context-manager yield so nested calls track correctly.

3. **`AnnealMemorySpan` missing protocol methods.** The base `Span` ABC marks only `set_tag` as abstract, but Haystack internals also call `set_tags` (plural), `set_content_tag`, `raw_span`, and `get_correlation_data_for_logs`. Guide's span only implemented `set_tag` and `set_content_tag`. Fix: added the three missing methods as safe stubs (`set_tags` merges the dict, `raw_span` returns self, `get_correlation_data_for_logs` returns an empty dict).

4. **`AgentBreakpoint` API completely wrong.** Guide used `AgentBreakpoint(break_after_agent_step=3)` — that kwarg does not exist anywhere in the haystack-ai surface. Real dataclass is `AgentBreakpoint(agent_name: str, break_point: Breakpoint | ToolBreakpoint)` where `Breakpoint` is its own dataclass with `(component_name, visit_count, snapshot_file_path)`. Rewrote the breakpoint example with the correct wrapper: `AgentBreakpoint(agent_name="researcher", break_point=Breakpoint(component_name="chat_generator", visit_count=3))`. Confirmed `snapshot_callback` is a separate `Agent.run` kwarg, not a field on the breakpoint.

Verification: `/tmp/am_frameworks/haystack/verify_fixed.py` exercises the full Tracer + Span protocol (`parent_span` kwarg, `set_tag`, `set_tags`, `set_content_tag`, `raw_span`, `get_correlation_data_for_logs`, `current_span()` returning the active span during a `trace()` context and None after), records a real episode through the tracer's finally-block (with operation_name triggering the `EpisodeType.DECISION` branch), and constructs `AgentBreakpoint(agent_name="researcher", break_point=Breakpoint(component_name="chat_generator", visit_count=3))` to confirm the real API shape.

#### CAMEL-AI (commit `ed51dcd`, 1 bug)

1. **`TaskDecomposedEvent` field drift.** Guide accessed `event.task_id` inside `log_task_decomposed`. Real dataclass has NO `task_id` field — decomposition describes the edge from a parent task into its children, so the real fields are `parent_task_id` and `subtask_ids`. Guide code would have raised `AttributeError` the first time a workforce decomposed a task. Fix: rewrote the `record()` call to surface `parent_task_id` + the list of subtask IDs explicitly. Also confirmed via `dataclasses.fields()` that the other events the guide accesses (`TaskCreatedEvent`, `TaskAssignedEvent`, `TaskCompletedEvent`, `TaskFailedEvent`, `WorkerCreatedEvent`) all have the fields the guide references.

Verification: `/tmp/am_frameworks/camel/verify_fixed.py` instantiates the full 11-method `AnnealMemoryCallback` (`WorkforceCallback` is an ABC with every `log_*` method abstract — the guide already implements all of them, so a minimal subset would have failed at construction), fires a sequence of real Pydantic event objects through every hook (`TaskCreatedEvent(task_id, description)`, `TaskDecomposedEvent(parent_task_id, subtask_ids)`, `TaskAssignedEvent(task_id, worker_id)`, `TaskCompletedEvent(task_id, worker_id, result_summary)`, `TaskFailedEvent(task_id, worker_id, error_message)`, `WorkerCreatedEvent(worker_id, worker_type, role)`), and confirms all six episodes land in the store.

#### Autogen / AG2 (commit `51e0267`, zero drift)

Verified against `ag2 0.11.5`. Every API the guide touches was exercised:

- Imports: `ConversableAgent`, `AssistantAgent`, `UserProxyAgent`, `GroupChat`, `GroupChatManager`, `autogen.agentchat.group.ContextVariables`.
- `register_hook(hookable_method, hook)` signature matches.
- Hook name registry: confirmed `agent.hook_lists.keys()` contains all three hook names the guide uses (`update_agent_state`, `process_all_messages_before_reply`, `process_message_before_send`) plus 6 other `safeguard_*` hooks that aren't relevant.
- Hook invocation signatures confirmed by reading `ConversableAgent.update_agent_state_before_reply` → `hook(self, messages)`, `process_all_messages_before_reply` → `hook(processed_messages)`, and `_process_message_before_send` → `hook(sender=self, message=..., recipient=..., silent=...)` as kwargs. Guide's `def inject_memory(agent, messages)`, `def enrich_with_recall(messages)`, and `def record_outgoing(sender, message, recipient, silent)` all match because Python accepts positional-or-keyword for those positional params.
- `ContextVariables(data={...})` dict-like access (`cv["key"]` get/set, `cv.get(key, default)`) works.

End-to-end verification: `/tmp/am_frameworks/ag2/verify_fixed.py` manually drives each hook (`assistant.update_agent_state_before_reply(messages)`, `assistant.process_all_messages_before_reply(msgs)`, `assistant._process_message_before_send(msg, recipient, silent=True)`), confirms system message enrichment, memory prepend to messages, episode recording via `process_message_before_send`, and ContextVariables round-trip. Only change to the guide: added "Verified against" header with "Zero drift" annotation.

#### DSPy (commit `d3e597d`, zero drift)

Verified against `dspy 3.1.3`. Every API the guide touches was exercised:

- Imports: `dspy`, `dspy.utils.callback.BaseCallback`, `dspy.LM`, `dspy.Module`, `dspy.Signature`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.Retrieve`, `dspy.MIPROv2`.
- All 5 `BaseCallback` lifecycle hooks (`on_module_start`, `on_module_end`, `on_lm_end`, `on_tool_start`, `on_tool_end`) have signatures matching the guide's subclass.
- `dspy.configure(callbacks=[...])` accepts the kwarg (configure is `**kwargs`-only, so it accepts any kwarg; verified by actually calling it).
- `dspy.ChainOfThought("context, question, memory -> answer")` construction works.
- `dspy.Retrieve(k=num_passages)` construction works.
- `dspy.ReAct("question -> answer", tools=[callable], max_iters=5)` construction works — confirmed `ReAct.__init__(self, signature, tools: list[Callable], max_iters: int = 20)`.
- `dspy.MIPROv2(metric=my_metric, auto="light")` signature confirmed.
- **Critical check:** the guide's `MemoryAwareRAG(dspy.Module)` subclass doesn't call `super().__init__()`. Verified that dspy.Module's metaclass still tracks sub-modules correctly without the super call — `named_parameters()` returns `[('retrieve', <Retrieve>), ('generate.predict', Predict(...))]`. The guide's idiom is fine.

Only change to the guide: added "Verified against" header with "Zero drift" annotation.

#### claude-agent-sdk (commit `1cdc440`, zero drift)

Verified against `claude-agent-sdk 0.1.59`. Every API the guide touches was exercised:

- Imports: `ClaudeAgentOptions`, `HookMatcher`, `query`.
- `ClaudeAgentOptions.hooks` type annotation confirmed as `dict[Literal['PreToolUse'|'PostToolUse'|'PostToolUseFailure'|'UserPromptSubmit'|'Stop'|'SubagentStop'|'PreCompact'|'Notification'|'SubagentStart'|'PermissionRequest'], list[HookMatcher]]`. All four hook event names the guide mentions in its table (`Stop`, `PostToolUse`, `PreCompact`, `UserPromptSubmit`) are in the Literal union.
- `HookMatcher(hooks=[callable], matcher=None, timeout=None)` construction works.
- Hook callback signature confirmed: the guide uses `async def on_*(input_data, tool_use_id, context)` which matches the SDK's `Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]` type annotation on `HookMatcher.hooks`.
- `query(*, prompt, options, transport=None)` signature confirmed — `prompt` and `options` are both in the parameter list.
- **Critical check:** the guide's `on_compact` hook example uses `store.status().episodes_since_wrap` — the public read-only peek API instead of the private `episodes_since_wrap()` method. Verified `Store.status()` returns a `StoreStatus` with an `episodes_since_wrap` attribute among the 14 status fields.

Only change to the guide: added "Verified against" header with "Zero drift" annotation.

### v0.2.1 release pipeline (commit `f6613fa`, tag `v0.2.1`)

Standard pipeline (same as v0.2.0, ~30 min start to verified fresh-venv install):

1. Version bump: `anneal_memory/__init__.py`, `pyproject.toml`, `server.json` (both top-level `version` and package-level `version`). 4 edits total.
2. CHANGELOG.md `[0.2.1] — 2026-04-14` section appended above the `[0.2.0]` entry. Covers the 4 drift fixes, the 3 verified-clean guides, the 3 Diogenes LOWs, the track-only assert-count note, a meta paragraph on the 75% combined hit rate and the reinforced framework-guide maintenance cadence.
3. `python3 -m pytest -q` → 707 passed in 3.84s.
4. `uvx mypy anneal_memory/` → `Success: no issues found in 11 source files`.
5. `python3 -m anneal_memory.server --generate-integrity` — no diff (manifest unchanged from v0.2.0, no new public tools).
6. Release commit `f6613fa` (`v0.2.1: doc-verification pass + Diogenes bundle`), then `git tag -a v0.2.1` + `git push origin main && git push origin v0.2.1`.
7. `rm -rf dist/ && python3 -m build` → `anneal_memory-0.2.1.tar.gz` + `anneal_memory-0.2.1-py3-none-any.whl`.
8. `python3 -m twine check dist/*` → PASSED for both.
9. `TWINE_USERNAME=__token__ TWINE_PASSWORD=$PYPI_API_TOKEN2 python3 -m twine upload dist/anneal_memory-0.2.1*` → both files uploaded to PyPI.
10. `gh release create v0.2.1 dist/anneal_memory-0.2.1-py3-none-any.whl dist/anneal_memory-0.2.1.tar.gz --title "v0.2.1 — Doc verification pass + Diogenes bundle" --notes "..."` → https://github.com/phillipclapham/anneal-memory/releases/tag/v0.2.1.
11. Fresh `/tmp/verify_021` venv, `pip install --no-cache-dir anneal-memory==0.2.1`, `import anneal_memory` → `version: 0.2.1` + canonical API (`Store`, `prepare_wrap`, `validated_save_continuity`, `StoreDatabaseError`) reachable. First attempt with cache hit reported 0.2.0 as the latest — had to retry with `--no-cache-dir` because pip's simple-index cache was stale by a few seconds post-upload. Not a real bug, noting it for future releases.

### Live artifacts

- https://pypi.org/project/anneal-memory/0.2.1/
- https://github.com/phillipclapham/anneal-memory/releases/tag/v0.2.1
- Full v0.2.1 changelog section in `CHANGELOG.md`

### Patterns to promote to continuity (noted for flow wrap)

- **Framework integration docs drift at ~75% hit rate when not exercised against live installs.** Strongly reinforces the maintenance cadence committed in v0.2.0. Combined across 10.5d + 10.5d+, 9 of 12 guides had load-bearing bugs on first-ever live verification.
- **Silent-failure drift is the worst class.** LlamaIndex `_aput(message, **kwargs)` singular-vs-list would have no-op'd every flush without raising. Haystack `AgentBreakpoint(break_after_agent_step=3)` would have raised TypeError. The haystack bugs are easier to catch — the llamaindex bug is invisible until you notice that zero episodes are being recorded in a running system.
- **"Plausible but wrong" is worse than "obviously broken".** The google-adk `search_memory(self, query) -> dict` signature looked completely reasonable for a memory service API. The real API is keyword-only three-tuple returning a typed pydantic object. Nothing about the guide's version looked wrong in prose-review — only a live-install introspection caught it.
- **Verification-script pattern generalizes.** The same exercise.py + verify_fixed.py shape worked for all 7 frameworks. Introspect first (no-network, no-API-calls), fire every symbol the guide references through `inspect.signature` / `model_fields` / `dataclasses.fields`, then construct each class with the kwargs the guide uses. Catches drift without needing real LLM calls.
- **Cache-miss on PyPI fresh-install verification.** First `pip install anneal-memory==0.2.1` after upload hit a stale simple-index cache and failed. `--no-cache-dir` resolved it. Noting for future release pipelines.

---

## Session 10.5c.6: SQLite Error Wrapping + Diogenes Warmup + Full 4-Layer Review — COMPLETE (Apr 13, 2026 afternoon/evening)

**Session shape.** A single flow-mode session that shipped two distinct pieces of work back-to-back — the 2 remaining Diogenes LOW/ARCH findings as warmup, then the 10.5c.6 SQLite error wrapping scope. The non-negotiable 4-layer review ran on the 10.5c.6 diff and surfaced 34 findings across 5 review passes (L1 session-code-review, L2 domain-expert on Python stdlib/exception design, L3 3-agent consultation with complement+gemini+contrarian, L3.5 codex-specific post-fix pass triggered by codex's L3 timeout, L4 end-to-end integration semantics). Partnership challenge during the deferral discussion broke a 3-of-4 completion-pressure drift that had been dressed up as architectural judgment.

**Pre-session test count:** 661 (end of 10.5c.5). **Post-session test count:** 707. **Delta:** +46 net (+10 Diogenes warmup fixes + +22 initial 10.5c.6 fault injection + +14 post-review regression guards including +4 from the deferral-fix pass).

### Warmup: Diogenes LOW/ARCH findings closed (commit `4b18df2`)

1. **SEMANTIC — `delete_episode` tool description drift**: integrity.py:252 claimed tombstones were "content-hash only, no original text" but the schema retains `id + timestamp + type + content_hash`. GDPR implication: the retained fields are pseudonymized metadata, not content, and users who need even that to vanish must opt out via `keep_tombstones=False`. Fixed the MCP tool description with accurate field list + GDPR framing + opt-out pointer. Mirrored the honest description into `Store.delete()` docstring and `keep_tombstones` parameter doc. Regenerated `tool-integrity.json` (delete_episode hash changed `8d88033→1aa3074`). +1 regression test pinning required phrases ("timestamp", "type", "GDPR", excludes "content-hash only").

2. **ARCH — audit health visibility gap**: agents using the MCP status tool and operators using the CLI status subcommand had zero visibility into whether the audit layer was running. Added `AuditTrail.stats()` public method (lazy-init, no chain walk, returns `{log_path, entry_count, retention_days}`) + 4 new audit_* fields on `StoreStatus` dataclass (all default-valued for backwards compat) + updated `Store.status()` to populate them with defensive `OSError` handling + MCP `_tool_status` emits a new `Audit: enabled/disabled` block with entry count, retention window, log path, and pointer to `anneal-memory verify` for hash chain validation + CLI `cmd_status` gets parity (both text and `--json` output) + updated `status` tool description to advertise the new fields. +9 regression tests across `test_store.py`, `test_server.py`, `test_cli.py`. 662 → 671 tests.

### 10.5c.6: SQLite Error Wrapping

**Goal**: caller-consistency. Pre-10.5c.6, file-write paths raised `StoreError` (with `operation` / `path` context from 10.5c.3) but SQLite paths leaked bare `sqlite3.OperationalError` / `IntegrityError` / `DatabaseError`. Transports catching `StoreError` got SOME store failures wrapped and OTHERS as raw sqlite3 errors — partial consistency was worse than either extreme because callers couldn't write a single catch clause for "memory system failed."

**Architecture**:

- New `StoreDatabaseError(StoreError)` subclass. Subclass (not sibling) so every existing `except StoreError` handler in transports (`server.py`, `cli.py`) catches both file-write and DB-origin failures unchanged. Callers who want to branch on "retryable DB error vs non-retryable file error" can catch the subclass specifically.
- New internal `Store._db_boundary(operation)` context manager. Catches `sqlite3.DatabaseError` (the runtime-failure root), NOT `sqlite3.Error` (the whole tree). The narrower catch is load-bearing — it excludes `sqlite3.InterfaceError` (API misuse / programming bugs) which MUST propagate bare so the stack trace points at the real bug instead of getting mis-wrapped as a "retryable DB error" a user might retry in a loop. `DatabaseError` covers `OperationalError`, `IntegrityError`, `DataError`, `NotSupportedError`, and `ProgrammingError` — the full set of runtime DB failures a well-formed call can produce.
- One `_db_boundary` per public method, NOT per SQL statement. Documented rule. The `operation` field names the caller-facing retry unit (what a caller would retry), not the individual SQL query. `wrap_completed` has a ~180-line single boundary covering CAS UPDATE + INSERT wraps + episodes session_id UPDATE + metadata clears + commit; the `ValueError` raised on CAS rowcount mismatch is NOT a `sqlite3.Error` and correctly propagates through the boundary unchanged.
- `StoreOperation` Literal expanded from 4 entries to 25: `save_continuity`, `save_meta`, `load_wrap_snapshot`, `wrap_completed`, `prepare_continuity_write`, `prepare_meta_write`, `record`, `get`, `delete`, `recall`, `episodes_since_wrap`, `status`, `wrap_started`, `wrap_cancelled`, `get_wrap_started_at`, `get_wrap_history`, `record_associations`, `decay_associations`, `get_associations`, `get_association_context`, `association_stats`, `prune`, `schema_init`, `batch_commit`, `close`. Most match method names verbatim; two (`schema_init`, `batch_commit`) name internal sub-phases whose failures a caller can meaningfully see.
- `cause_type_name: str | None` field on `StoreError.__init__`, populated by `_db_boundary` at raise time with `type(exc).__name__`. Plain string — survives pickle identically because there's no live exception reference to marshal. Closes the cross-process retry-decision gap (L3 contrarian F7): after a pickle round-trip `__cause__` is `None`, so any helper checking `isinstance(err.__cause__, sqlite3.OperationalError)` returns `False` for every error. Helpers that dispatch on `cause_type_name` work identically in-process and post-pickle.
- Single generic `_reconstruct_store_error(cls, message, operation, path, cause_type_name)` module-level reconstructor replaces the original parallel-reconstructor pattern (`_reconstruct_store_error` + `_reconstruct_store_database_error`). `StoreError.__reduce__` passes `type(self)` through so `StoreDatabaseError` and any future subclass inherit correct pickle behavior automatically. The "add a subclass, forget the reconstructor, silently downgrade to parent on unpickle" failure mode is now structurally impossible.
- Constructor `schema_init` boundary wraps `sqlite3.connect` + PRAGMAs + `_init_schema` + `_warn_orphan_tmp_files` in a single `_db_boundary`. `self._conn = None` initialized before the try so connect-never-succeeded cleanup path is safe. `_warn_orphan_tmp_files` has its own try/except + `warnings.warn` fallback so orphan-detection failure doesn't abort construction (contrarian F1).
- `close()` wrapped with `_db_boundary("close")` + idempotent via a separate `_closed: bool` flag. Critically, the flag-based approach (NOT nulling `_conn`) means post-close method calls hit a pre-check at the top of `_db_boundary` that raises `StoreError("Cannot {operation} on a closed store")` with the caller's operation name instead of a bare `AttributeError` from `self._conn.execute(...)` bypassing the whole hierarchy. Codex's post-fix pass (L3.5 H1) caught this as a regression the first fix-pass had introduced.
- `get_wrap_history` legacy "no such table" swallow TIGHTENED from loose match to `"no such table: wraps"` specifically. Pre-existing silent-error-swallowing hazard: the looser match silently returned `[]` for ANY missing table, including a corrupted DB missing `episodes`. The tightened match preserves the documented legacy case (unmigrated v0.1.x DB missing the `wraps` table) while forcing every other missing-table case through the boundary as `StoreDatabaseError`. L1 H1 finding.
- `record()` method had a pre-existing bug where `_current_session_id()` was called BEFORE the boundary started, so sqlite3 errors from the SELECT leaked bare. Fault injection testing caught this — moved `_current_session_id()` inside the boundary.
- `_batch()` commit wrapping: new `batch_commit` op in Literal; rollback-during-rollback failures are swallowed to preserve the primary exception (documented inline).
- `docs/library-quickstart.md` Error Handling section completely rewritten: leads with simple `except AnnealMemoryError` (the 80% case — catch at outermost boundary, log, propagate). Then documents the 3-class hierarchy with explicit "deliberately NOT mirroring PEP 249 / DB-API 2.0" note. Then shows an `is_retryable()` helper that dispatches on `err.cause_type_name` (works both in-process and post-pickle) rather than `err.__cause__` (breaks post-pickle). Shows explicit `is_retryable` usage pattern with `retry_backoff()` gated. Documents that `cause_type_name` was added specifically to close the cross-process retry-decision gap.
- `AnnealMemoryError` docstring: explicit "This library deliberately does not mirror PEP 249 / DB-API 2.0. anneal-memory consumes a database internally; it is not a database driver. Callers branch on operational intent (retry vs escalate vs surface), not on vendor taxonomy." Rationale documented at the top of the hierarchy.
- `_db_boundary` docstring: added "one boundary per public method" scope rule + method-name parity exceptions note for `schema_init` / `batch_commit` / `close` + "catch scope: `sqlite3.DatabaseError`, not `sqlite3.Error`" rationale + soft-contract warning about new raise sites needing Literal entries.

**The 4-layer review + Phill's challenge**:

1. **L1 session-code-review** (parallel with L2): 11 findings — 2 HIGH, 3 MEDIUM, 4 LOW, 2 NIT. HIGH H1 = `get_wrap_history` loose match (pre-existing silent-swallow debt). HIGH H2 = two redundant `schema_init` boundaries. MEDIUM M1 = missing fault tests for associations methods. MEDIUM M3 = `batch_commit`/`schema_init` break method-name parity (doc drift).
2. **L2 Python stdlib / exception design expert** (parallel with L1): 10 findings — 2 HIGH, 5 MEDIUM, 3 LOW. HIGH #2 = `sqlite3.Error` catch includes `InterfaceError` (API-misuse programming bugs get mis-wrapped as retryable). HIGH #7 = `library-quickstart.md` retry framing too loose (blanket retry guidance causes storms on non-retryable errors). MEDIUM #1 = should explicitly NOT mirror PEP 249. MEDIUM #3 = Literal drift test. MEDIUM #4 = single generic pickle reconstructor refactor. MEDIUM #5 = `cause_type_name` field for cross-process retry.
3. **L3 consultation** (complement + gemini + contrarian via `consult.py --mode standard --diff`; codex timed out): 10 findings — 0 CRITICAL, 0 HIGH, 3 MEDIUM, 5 LOW, 2 NIT. Convergent across agents: `is_retryable()` + pickle interaction breaking cross-process (complement F9 + contrarian F7), dual-rollback path documentation (complement F2 + contrarian F2), use-after-close ProgrammingError looks retryable (complement F4 + contrarian F4). Unique contrarian F1 = orphan warning suppression inside `schema_init` boundary can silently hide a warning on re-open. Unique complement F4 = `close()` doesn't null `_conn`. Both agents did a contrarian-thesis attack on the hierarchy and independently concluded it's the right design for a library shipping externally.
4. **L3.5 codex post-fix review** (dispatched at Phill's suggestion after codex's L3 timeout, focused on post-fix state): 3 additional findings. **HIGH H1** — post-`close()` method calls raised bare `AttributeError` on `self._conn.execute(...)` because the L3 complement F4 fix had nulled `_conn` after close. This is a regression the L3 fix itself had introduced. Switched to `_closed: bool` flag + guard at the top of `_db_boundary`. **MEDIUM** — `StoreOperation` drift: `prepare_continuity_write` + `prepare_meta_write` were raise sites in the code but missing from the Literal. The "discipline-based reviewer catches drift" soft contract had ALREADY FAILED this session. **LOW** — `is_retryable()` doc example had a missing `StoreDatabaseError` import.
5. **L4 integration semantics** (custom Python script exercising documented claims): 21/21 checks pass after the fix pass. Hierarchy shape verified, Literal completeness verified, use-after-close raises `StoreError` not `AttributeError`, real locked-DB surfaces as `StoreDatabaseError` with correct operation + cause, pickle identity preserved, `is_retryable()` doc helper works as pasted, `InterfaceError` propagates bare (not wrapped), `validated_save_continuity` end-to-end pipeline succeeds.

**Phill's partnership challenge on deferrals**: at the end of the 5-review pass, 4 items were declared as "deferred to 0.2.x" — single generic pickle reconstructor, `cause_type_name` field, `StoreOperation` drift test, shared test proxy helper. Phill challenged: "defend each as correct long-term or admit you're being lazy." Honest interrogation: 3 of 4 were completion pressure dressed up as architectural judgment. Only the shared test proxy helper (rule-of-three YAGNI) survived. Fixed the other three in a dedicated pass:
- **Pickle reconstructor refactor**: ~15 lines, structurally prevents future-subclass pickle bugs. 2 new tests (parent-class roundtrip preservation + single-reconstructor handling).
- **`cause_type_name` field**: added to `StoreError.__init__`, populated in `_db_boundary` from `type(exc).__name__`, survives pickle, enables cross-process retry decisions. `is_retryable()` doc helper rewritten to dispatch on this field. Updated 1 existing test + 2 new tests (field populated correctly + helper works post-pickle).
- **`StoreOperation` drift test**: `test_store_operation_literal_has_no_drift` in `TestDbBoundaryErrorWrapping`. Grep-scans `store.py` for every `_db_boundary()` and `operation=` string literal, cross-references against `typing.get_args(StoreOperation)`, asserts bidirectional equality. Catches drift in both directions. Codex's H1.5 finding would have fired this test at CI time instead of requiring a post-hoc review pass. `structural_invariants_beat_discipline_based_verification` now has a CI gate, not just a reviewer convention.
- **Shared test proxy helper**: remains deferred. Two instances (`_FlakyExecuteProxy` + `FlakyCommitProxy`) don't earn the abstraction; wait for a third test.

**Pattern reinforcement**: `completion_pressure_peaks_at_publish_boundary` at 7x+ Proven, with a new corollary — self-audit at session-end is NOT sufficient to catch completion-pressure drift. The self-justification narrative is load-bearing at session-end and resists internal audit. External challenge is required to surface. Phill's direct "defend each or admit laziness" unstuck the drift immediately where internal reflection wouldn't have. `layer_4_attention_mode_catches_pre-existing_debt_layers_1-3_miss` also reinforced: L3.5 codex caught a regression one of the L3 fixes had INTRODUCED, which generalizes the pattern — "pre-existing debt" can also mean "debt introduced by the fix itself in the current session." `structural_invariants_beat_discipline_based_verification` at 10x+ Proven, with a new instance (Literal drift CI gate).

**Files changed** (anneal-memory repo, pending commit): `anneal_memory/store.py` (+900 lines net — mostly new class + context manager + extensive docstrings + indentation from boundary wraps), `anneal_memory/audit.py` (+22 lines — public `stats()` method), `anneal_memory/types.py` (+8 lines — 4 new StoreStatus fields), `anneal_memory/integrity.py` (tool description honesty fix + status tool description advertises audit fields), `anneal_memory/server.py` (+40 lines — MCP _tool_status audit section), `anneal_memory/cli.py` (+30 lines — cmd_status audit parity), `anneal_memory/__init__.py` (+3 lines — StoreDatabaseError export), `anneal_memory/tool-integrity.json` (regenerated for delete_episode + status description changes), `docs/library-quickstart.md` (+110 lines — full Error Handling section rewrite), `tests/test_store.py` (+600 lines — TestDbBoundaryErrorWrapping class with 36 tests), `tests/test_server.py` (+50 lines — audit status tests), `tests/test_cli.py` (+35 lines — CLI status audit tests + JSON parity), `tests/test_integrity.py` (+20 lines — delete_episode description regression), `tests/test_continuity.py` (+14 lines — batch_commit migration to StoreDatabaseError).

**Remaining v0.2.0 arc**: 10.5d framework integration testing (top 5 frameworks: LangGraph/LangChain, CrewAI, OpenAI Agents SDK, Pydantic AI, smolagents — verify integration guides actually work with a real `pip install` + example run). Then CHANGELOG + version bump to 0.2.0 + PyPI publish + GitHub release + Glama sync. Session 10.6 deferred-wrapper-deletion remains post-v0.2.0-release.

---

## Session 10.5c.5: File/DB Two-Phase Commit + Full 4-Layer Review — COMPLETE (Apr 13, 2026)

**Session shape.** A single flow-mode session that shipped three distinct pieces of work back-to-back — Diogenes LOW warmup fixes, the 10.5c.4a operator subcommands (filed during 10.5c.4 review as a low-priority follow-up), and the load-bearing 10.5c.5 two-phase commit pipeline. The non-negotiable 4-layer review (L1 session-code-review + L2 domain-expert + L3 consultation team + L4 integration semantics) ran on the 10.5c.5 diff, produced 20+ findings across the first review pass, a codex-retry pass found three additional recoverability-identity bugs, every finding landed with a regression gate, Layer 4 end-to-end crash-recovery journey verified.

**Pre-session test count:** 610 (end of 10.5c.4). **Post-session test count:** 661. **Delta:** +51 net (+36 new from this session's work, +15 from 10.5c.4a absorbed into the same session).

### Warmup: Diogenes LOW fixes (2)

1. `test_graduation.py:77` — `test_carried_forward_not_validated` now asserts `result.skipped_non_today == 1`. The counter was added precisely to be assertable; without the assertion a regression at `graduation.py:145` would have been invisible.
2. CLI `--wrap-token` format validation at args intake. Previously `_WRAP_TOKEN_RE` lived in `server.py` only; the CLI had no format check, so malformed tokens gave a confusing "mismatch" error from the downstream CAS path. Imported the regex into `cli.py` and gated at args intake BEFORE any file I/O. (L1 review later caught the cli→server import direction as an architectural wart and moved the constant to `store.py` — see 10.5c.5 L1 fix #5.)

### 10.5c.4a: Operator surface subcommands

**Three new CLI subcommands** for stuck-wrap diagnostics + recovery:

- **`wrap-status`** — structured text + JSON output. Reads `Store.load_wrap_snapshot()` and new scoped accessor `Store.get_wrap_started_at()`. Displays `wrap_started_at`, token, episode count, and ready-to-copy recovery commands. Handles `StoreError` partial-state with a clean recovery hint (no traceback).
- **`wrap-cancel`** — works on both healthy AND partial-state stores (the point of the escape hatch). Echoes the cancelled token to stdout for operator receipts.
- **`wrap-token-current`** — shell-pipeline-friendly. Stdout clean (just the token, nothing else) when a wrap is active, empty string when idle. Exit 0 in both cases. Designed for `TOKEN=$(anneal-memory wrap-token-current); [ -n "$TOKEN" ] && anneal-memory save-continuity --wrap-token "$TOKEN" cont.md` idioms.

**New `Store.get_wrap_started_at() → str | None`** scoped accessor. Kept separate from `load_wrap_snapshot()` so operator display code doesn't pay the partial-state integrity checks that TOCTOU callers need, and `WrapSnapshot` stays minimal for its critical-path use.

**15 new tests:** handler-level direct calls (fast) + one subprocess end-to-end roundtrip (integration) covering idle + in-progress + partial-state + cancel flows. Full 4-layer review folded into the main 10.5c.5 review that followed.

### 10.5c.5: File/DB Two-Phase Commit

**Architectural decision made upfront with Phill** before implementation: **DB commit FIRST, then file renames** (vs. the reverse). Rationale: "DB ahead of file" = observability drift (recoverable, repairable, bounded); "file ahead of DB" = data loss via double-compression on next wrap (episodes would be stamped twice into consecutive wraps' session_ids). The load-bearing invariant becomes "DB commit returns ⇒ the tmp files are the committed state and must be externalized or preserved, never destroyed."

**Pipeline reshape** (five explicit phases):

1. **Phase 1 — Continuity tmp write.** `store._prepare_continuity_write(text, token_hex=...)` writes a uuid-suffixed tmp sidecar with `fsync(file)` + `fsync(parent_dir)`. Returns the tmp path. No rename yet.
2. **Phase 2 — Batched DB DML.** `with store._batch():` opens a single SQLite transaction that accumulates: associations DML (`record_associations` + `decay_associations`), meta tmp write (`store._prepare_meta_write(meta, token_hex=...)`), wrap completion DML (`wrap_completed`). On successful exit the batch does the single outer commit and then flushes the deferred audit queue. On exception the batch rolls back and discards the queued audits.
3. **Phase 3 — Atomic renames.** After the batch exits successfully (DB committed), `cont_tmp.replace(store.continuity_path)` then `meta_tmp.replace(store.meta_path)`, followed by `_fsync_dir(parent)` to make the renames themselves durable.
4. **Phase 4 — `continuity_saved` audit event.** Fired directly on `store._audit.log` (not queued) because at this point the wrap is fully committed + externalized. Wrapped in try/except to swallow audit-log failures — they must not propagate and trick the caller into thinking the wrap failed (L2 M2 / L1 HIGH data-loss path).
5. **Phase 5 — Auto-prune.** Runs `store.prune()` AFTER the batch exits if `retention_days` is configured. The batched `wrap_completed` deliberately suppresses its own prune because prune is a separate DML burst with its own commit semantics that doesn't belong inside the wrap transaction. Without this Phase 5 call users with retention configured would silently lose lifecycle management — L1 CRITICAL finding.

**New primitives and infrastructure:**

- `Store._batch()` — contextmanager that suspends commits + audit firing. Sets `_defer_commit = True`, queues audit events into `_deferred_audits` (widened from `(event, payload)` 2-tuples to `(event, payload, kwargs)` 3-tuples to carry `actor=source` for `record()` events), commits once at exit, flushes audits with exception swallowing (L2 M2: audit-flush exceptions must NOT propagate and trigger outer tmp cleanup). Rolls back on any exception inside the body; rolls back on commit-failure before re-raising (L3 complement F2). `_defer_commit` reset in a `finally` block so SIGINT between commit and reset cannot poison the store (L2 H2). Nested batches raise `RuntimeError`.
- `Store._audit_log(event, payload, **kwargs)` — batch-aware audit helper. Fires immediately outside a batch; queues inside a batch. All batch-aware write methods use this instead of `self._audit.log`.
- `Store._prepare_continuity_write(text, token_hex=None)` + `_prepare_meta_write(meta, token_hex=None)` — tmp-sidecar write primitives. Each generates a uuid-suffixed path (or uses the provided `token_hex` for pairing), writes with `fsync(file)` + `_fsync_dir(parent)`, returns the tmp path. Refactored the existing public `save_continuity` / `save_meta` to use these primitives internally (preserving the public operation-name contract via re-wrap on StoreError).
- `_fsync_dir(path: Path)` — POSIX-only best-effort directory fsync. Closes the L2 H1 gap where `fsync(file)` alone doesn't guarantee the directory entry is durable on ext4/xfs/btrfs; a crash between file fsync and rename could produce ENOENT on recovery. No-op on Windows.
- `PRAGMA synchronous=FULL` — set on connection init. Under default `synchronous=NORMAL` WAL mode, `commit()` doesn't fsync the WAL so a post-commit power loss can revert recently committed transactions. `FULL` is required for the "DB commit returns ⇒ durable" invariant the two-phase commit ordering assumes. L2 L3.
- `_WRAP_TOKEN_RE` — moved from `server.py` to `store.py` module scope. L1 HIGH fix for the architectural wart where the CLI had to load the entire MCP server module just to reach one compiled regex. Both transports now import from the store layer.
- `Store._defer_commit` + `_deferred_audits` — instance state for the batch context. Documented as not reentrant, not thread-safe, not task-safe.
- Batch-aware methods: `record`, `delete` (L4 follow-up: folded in for consistency), `record_associations`, `decay_associations`, `wrap_completed`. All check `_defer_commit` to skip their own commit and route audit events through `_audit_log`. `_batch()` docstring maintains the canonical list.
- `validated_save_continuity` — rewritten around the five phases. `db_committed: bool` flag distinguishes pre-commit cleanup (unlink tmp files) from post-commit cleanup (PRESERVE tmp files for operator recovery — L1 HIGH + L2 M2). Explicit None checks on `meta_tmp` and `wrap_result` instead of `assert` (`-O` safe, L3 complement F3/F5 + contrarian F6). `path = str(store.continuity_path)` hoisted to the top of the function so control-flow analyzers can narrow it. Old "partial-failure window — scheduled for 10.5c.5" docstring warning replaced with the actual residual-window note (bounded microseconds, recoverable via operator `mv`).

**4-layer review findings + fixes.**

L1 (session-code-review agent) — 4 findings:
- CRITICAL auto-prune regression (batched pipeline silently suspended retention management). Fixed: Phase 5 prune call in pipeline. New regression test.
- HIGH residual-window data loss: outer `except BaseException` unlinked tmp files even after DB commit. Fixed: `db_committed` flag preserves tmp files post-commit.
- HIGH `cli.py` imported `_WRAP_TOKEN_RE` from `server.py` — wrong dependency direction, CLI loaded entire MCP server module. Fixed: moved regex to `store.py`.
- MEDIUM `wrap_cancelled` emitted no audit event on partial-state recovery. Fixed: always emit audit event with `partial_state: True` marker when any wrap-lifecycle key was set.

L2 (Python concurrency / SQLite / 2PC domain expert) — 6 findings:
- C1: `wrap_completed` CAS rollback was explicit inside a batch context (ownership confusion). Fixed: only rollback when `not self._defer_commit`.
- C2 (NOT fixed, documented): Python sqlite3 stdlib default `isolation_level=""` has implicit-BEGIN timing that's Python-version-sensitive. The theoretically-correct fix is `isolation_level=None` + explicit `BEGIN IMMEDIATE`, but that requires refactoring every commit site in the codebase. The existing `test_batched_dml_invisible_to_other_connection_until_commit` regression gate empirically proves atomicity holds under Python 3.14 for the sequences the canonical pipeline actually issues. Documented as a known limitation in `_batch()` docstring.
- M2: audit-flush exceptions inside `_batch.__exit__` propagated to outer pipeline cleanup, which then unlinked the tmp files representing committed state. Fixed: swallow audit-flush exceptions inside the batch exit path.
- H1: `fsync(file)` without `fsync(parent_dir)` meant the advertised "microseconds-wide residual window" was actually seconds-wide (page-cache flush interval) on ext4/xfs/btrfs. Fixed: `_fsync_dir()` helper called after tmp write and after rename.
- H2: SIGINT between `self._conn.commit()` and `_defer_commit = False` reset would permanently poison the store. Fixed: flag reset moved to unconditional `finally` block.
- L3: `PRAGMA synchronous=NORMAL` under WAL means commit doesn't fsync the WAL. Fixed: `synchronous=FULL` on init.

L3 first pass (complement + gemini + contrarian + codex; codex timed out at 300s):

**Convergent findings (all 3 of first-pass agents):**
- CRITICAL: fixed-path tmp filenames (`continuity.md.tmp`, `meta.json.tmp`) create a concurrent-writer collision — two processes can race on the same tmp path and the winner externalizes the loser's content. The CAS token closes the DB race but happens AFTER Phase 1 tmp writes. Fixed: uuid-suffixed tmp filenames generated per invocation.
- HIGH: no startup detection of orphan tmp files from prior crashed pipelines. After a crash, `wrap-status` reports idle (metadata was cleared inside the batch before the crash), so the operator has no automated way to discover that tmp files need manual recovery. Next successful wrap silently overwrites them → permanent data loss. Fixed: `Store._find_orphan_tmp_files()` + `_warn_orphan_tmp_files()` called from `__init__` (AFTER `_init_schema` so the metadata table exists).

**Unique findings:**
- MEDIUM complement F2: `_batch()` commit failure had no explicit rollback before re-raising. Fixed: wrapped commit in try/except that rolls back then re-raises.
- MEDIUM complement F4: Phase 4 `continuity_saved` audit fired outside any try/except. Fixed: wrapped in try/except with same swallow pattern as batch flush.
- LOW complement F3/F5 + contrarian F6: `assert meta_tmp is not None` / implicit `wrap_result` access stripped under `python -O`. Fixed: explicit None checks raising `StoreError` with recovery guidance.
- MEDIUM contrarian F1: `record()` committed unconditionally inside `_batch()` — forward-looking misuse hazard. Fixed: `_defer_commit` guard in `record()`. `_audit_log` extended to carry `actor=source` kwarg through the deferred queue (tuple widening from 2→3 slots). Also folded `delete()` into the batch-aware set for consistency during Layer 4.

L3 codex-retry pass (foreground, ran alone at ~233s — succeeded on the retry):

- HIGH recoverability-identity: continuity and meta tmp files were generated with INDEPENDENT uuids, so multiple crashed wraps left orphan files with no pairing key — operator couldn't know which `.md.tmp` belonged with which `.json.tmp`. The whole "operator recovery" story was unfollowable under the realistic multi-crash scenario. Fixed: `_prepare_*_write` primitives accept an optional `token_hex` parameter; the canonical pipeline passes `snapshot["token"][:12]` to BOTH calls so the tmp files share an identifying prefix (`mystore.continuity.<token>.md.tmp` + `mystore.meta.<token>.json.tmp`). Orphan detection groups by token prefix and emits ONE warning per wrap listing both files.
- MEDIUM codex: orphan warning text referenced `content_hash` as a field on the `wrap_completed` audit event, but that field lives on `continuity_saved` (not `wrap_completed`). Unfollowable recovery instruction. Fixed: warning text points operators at `continuity_saved` for content_hash and `wrap_completed` for wrap_token matching.
- MEDIUM codex: false-positive orphan warning when Store B opens the same DB while Store A is mid-pipeline (A's legitimate in-flight tmp files flagged as orphans). Fixed: `_find_orphan_tmp_files` cross-references with metadata `wrap_token`; tmp files whose embedded token matches the current active wrap_token are filtered out. Handles the batch-in-progress window; the post-commit pre-rename microsecond window still has false positives (unavoidable — the metadata token is cleared before the rename) but is so narrow it's documented as residual risk.

L4 (integration semantics check — different attention mode, I run this myself):
- Docstring vs. behavior walked manually for all touched methods. `_batch()` lists batch-aware methods; verified against actual commit sites. Found `delete()` not batch-aware even though the docstring said "other write methods ... NOT safe" — folded it into the batch-aware set for consistency rather than documenting the gap.
- Audit trail chain-of-custody: end-to-end test verified `actor=source` kwarg round-trips through the deferred audit queue and lands in the JSONL file as expected.
- Public docs (README, integration guides) don't over-promise crash safety — no doc-vs-implementation violations.
- **End-to-end crash-recovery user journey** — scripted test: healthy wrap → simulate crash by manually creating paired orphan tmps → re-open store → verify ONE warning emitted (paired grouping) with all recovery fields (mv instructions, both filenames, shared token, `continuity_saved` reference) → operator manual `mv` → re-open store → verify silent (no warnings, new content active). Full journey verified.

**Tests added across all fix passes:** 9 (initial 2PC scenarios: tmp write failures, batch DML isolation, rollback, audit order, nested batch raises) + 8 (L1/L2 post-review regression: auto-prune, audit flush failure, defer_commit reset, nested batch state preservation, partial-state audit) + 13 (L3 fix regression: unique tmp paths, glob patterns, startup orphan detection, commit-failure rollback, Phase 4 audit swallow, meta_tmp None → StoreError, record() batch-aware 3-test set) + 6 (codex-retry fix regression: paired tmp prefix, warning text content, paired orphans → single warning, unpaired orphan NOTE, active wrap in-flight filter, stale tmp with different token still flagged). Total: 36 new tests in this session alone.

**Known-limitations discoveries documented in code:**
- Python sqlite3 stdlib implicit-BEGIN quirk (empirical regression gate proves atomicity holds).
- macOS `os.fsync` is weaker than Linux — true durability requires `fcntl(F_FULLFSYNC)` which stdlib doesn't expose.
- Post-commit pre-rename microsecond window is the irreducible residual risk.
- `_batch()` is not reentrant, not thread-safe, not task-safe (documented, enforced only for nesting via RuntimeError guard).
- Cross-filesystem rename (EXDEV) if continuity/meta paths are symlinked across mounts.

**v0.2.0 release arc status after this session:** 10.5c.1 → 10.5c.2 → 10.5c.3 → 10.5c.4 → 10.5c.4a → 10.5c.5 all shipped. Remaining for v0.2.0: 10.5c.6 (SQLite error wrapping — optional), 10.5d (framework testing — verify top 5 integration guides actually run), 10.5d+ (mypy in CI — optional), CHANGELOG + version bump + PyPI publish.

**Key commits:** pending — all changes live on the uncommitted working tree at session wrap and will be committed together. Pre-review diff stats: `+1331/-114` across 7 files. Post-review final diff stats: `+3167/-160` across 9 files.

**BREAKING CHANGES for v0.2.0 CHANGELOG** (folded into the release checklist at `projects/anneal_memory/CLAUDE.md`): the `_batch()` / `_audit_log` / `_prepare_*_write` / `_defer_commit` / `_deferred_audits` / `_fsync_dir` internals are new but underscore-prefixed and not part of the public surface. The user-visible public-API shape of `validated_save_continuity` is unchanged (same signature, same return dict keys). `PRAGMA synchronous=FULL` is new and changes per-commit cost by one additional fsync (~negligible at wrap boundaries). `Store.save_continuity` / `Store.save_meta` public methods preserve their operation-name contract via re-wrap. No public-API breaking changes for this session.

---

## Session 10.5c.4: Session-handshake token for prepare→save TOCTOU window — COMPLETE (Apr 10, 2026 late evening)

**Trigger:** The 10.5c.3 archive entry noted the prepare/save TOCTOU window as a documented limitation with a `.. warning::` block on both `prepare_wrap` and `validated_save_continuity`: if the caller records an episode between prepare and save, the new episode silently joins the wrap even though the agent's compression was built against the smaller set. 10.5c.4 was filed as a dedicated architectural session to close the gap. Next in the v0.2.0 release arc after 10.5c.3 shipped commit `934a3ce` earlier the same evening.

**Key commits:** (pending — this session's final commit lives on `main` at the wrap).

**Session tests:** 577 → 610 (+33 new tests across three test classes: `TestTOCTOUHandshakeToken` in test_continuity.py, `TestCLICrossProcessTOCTOU` in test_cli.py, `TestMCPWrapTokenRoundTrip` in test_server.py). Zero pre-existing regressions.

**Session scope (the bet):** Phill challenged the initial 4-layer scoping with "I bet you my next paycheck we can do 10.5c.4 tonight no problem :P" after the 10.5c.3 ship — this was the fourth major ship event of a single day (flow-meta bundle + 10.5c.3 + strategic conversations earlier + 10.5c.4). Scope held, bet paid, four-layer review ran clean end-to-end.

### Core design

Session-handshake token + frozen episode-ID snapshot:

1. **`prepare_wrap` mints a token + freezes the episode set.** Token is `uuid.uuid4().hex` (32-char hex, 128 bits entropy, stdlib-only). Both the token and the list of episode IDs returned to the agent are persisted in store metadata via an extended `Store.wrap_started(token=..., episode_ids=...)` — all three metadata keys (`wrap_started_at`, `wrap_token`, `wrap_episode_ids`) written in a single SQL transaction for the atomic-clear invariant.
2. **`validated_save_continuity` loads the snapshot, verifies the token, filters the re-fetched episode set.** New `Store.load_wrap_snapshot()` primitive returns a `WrapSnapshot` TypedDict or `None` on the legacy skipped_prepare path. When the caller passes `wrap_token` and the stored token doesn't match → `ValueError` with a clean "wrap_token mismatch" message. The frozen-snapshot filter runs whenever a snapshot is present, regardless of whether the caller passes the token — the token is the explicit verification layer, not the snapshot enabler.
3. **`wrap_completed` accepts the frozen episode ID list and filters the session_id UPDATE.** New `episode_ids: list[str] | None` keyword parameter. When provided, only those episodes get their `session_id` stamped with the new wrap's ID; any episode with `session_id IS NULL` that is NOT in the snapshot stays null and naturally lands in the NEXT wrap's compression window. No data loss — TOCTOU episodes are deferred, not absorbed.
4. **Cross-process handshake via SQLite metadata.** CLI `prepare-wrap` subcommand emits a `Wrap token: <hex>` text trailer and a `wrap_token` key in `--json` output. CLI `save-continuity` accepts `--wrap-token <hex>` for explicit verification; without the flag the filter still runs because the library consults the persisted snapshot. The CLI process boundary is shared via the SQLite metadata table — no sidecar file, no environment variable, no hidden state.
5. **MCP transport round-trip.** MCP `prepare_wrap` tool appends the `Wrap token: <hex>` trailer to its response text. MCP `save_continuity` tool accepts optional `wrap_token` argument with JSON schema `{"type": "string", "pattern": "^[0-9a-f]{32}$"}` and server-side shape validation via module-level `_WRAP_TOKEN_RE` constant.
6. **Audit chain-of-custody enrichment (Decision 8e concession).** `wrap_started` audit entry carries `wrap_token`, `wrap_episode_count`, and the full `wrap_episode_ids` list. `wrap_cancelled` carries the same three fields so an auditor can reconstruct abandoned wraps without cross-joining. `wrap_completed` carries `wrap_token` as the chain-of-custody link between prepare and complete events. An auditor walking the audit trail alone can reconstruct "which episodes went into wrap #N" by matching tokens across wrap_started → wrap_completed events (verified in Layer 4 Scenario E).

### 8-point design rubber-duck

Before touching code, the session wrote out 8 explicit design decisions for Phill sign-off. All 8 approved on first pass. Phill ran the anti-laziness audit against the "what I am NOT doing" list (Decision 8) — the 8e item (audit event enrichment) was conceded and folded into scope after honest self-audit showed that shipping the fix without enriching the audit trail would drop the fidelity mechanism on the floor at the audit layer — same bug class as 10.5c.1's "library canonical in code but docs pointed at the bypass."

### Four-layer review (the meat of the session)

**Layer 1 (session-code-review agent) + Layer 2 (Python library API domain expert agent) in parallel** — dispatched via Agent tool with distinct prompts targeting orthogonal bug classes. Layer 1 returned 1 HIGH + 5 MEDIUMs + 3 LOWs + 3 NITs. Layer 2 returned 4 HIGHs + 5 MEDIUMs + 3 LOWs + 2 NITs plus explicitly cleared 4 points (ValueError category, uuid.uuid4().hex entropy, batched-write transaction atomicity, WrapSnapshot TypedDict shape).

Load-bearing findings from Layers 1+2:
- **L1 H1 + L2 M4 convergent** — `wrap_started_at` set but `wrap_token` empty silently bypassed the snapshot filter. `load_wrap_snapshot` was tolerating the state as "legacy skipped_prepare" and returning `None`, but any caller invoking the no-arg `store.wrap_started()` form would land in a state where the canonical pipeline ran in legacy mode, bypassing the entire 10.5c.4 fix. Fix: `load_wrap_snapshot` now raises `StoreError` on the partial state. `validated_save_continuity` derives `skipped_prepare` from snapshot presence (not `wrap_in_progress`). The canonical pipeline has exactly one valid state machine.
- **L2 H3 (subtraction)** — `_set_metadata` became dead code after all wrap-lifecycle methods switched to inline batched writes. Keeping it was pure drift hazard (a future contributor would reach for it inside `wrap_completed` and silently break the atomic-clear invariant). Deleted. Forward-looking "Wrap-lifecycle invariants" section added to Store class docstring stating the rule.
- **L1 M1** — write-side validation asymmetry: `wrap_started(token, episode_ids=None)` silently produced a malformed state that the read side rejected. Now raises `ValueError` at the write boundary.
- **L2 M2** — SQLite variable-limit guard added in `wrap_completed` (raises `StoreError` on `episode_ids > 998`).
- **L1 M5** — graduation validation test softer than it read (`or bare_demoted >= 1` was misleading). Confirmed via reading graduation.py that unknown-ID citations route through the `ids_valid` gate which increments `demoted`, not `bare_demoted`. Pinned to `result["demoted"] >= 1`.
- **L1 M3 + L2 NIT-2** — `wrap_cancelled` audit didn't carry snapshot episode IDs; audit key names renamed from `snapshot_size/snapshot_episode_ids` → `wrap_episode_count/wrap_episode_ids` for vocabulary consistency with metadata keys.

Test cleanup consequence of L1 H1: 14 tests in test_store.py were calling `store.wrap_started()` with no args before `validated_save_continuity` as a legacy workaround predating the canonical pipeline. After the state-machine tightening, those 14 tests broke. Bulk-deleted the bare calls via a Python script; preserved them in 3 legitimate lifecycle-flag tests.

Tests after Layer 1+2 fix pass: 577 → 608 (+31 new TOCTOU tests).

**Layer 3 (consultation team: complement + gemini + contrarian + codex via `consult.py --diff`)** returned the sharpest finding of the entire session:

- **codex HIGH (load-bearing) — the replay race was still open at save-time.** `validated_save_continuity` verified `wrap_token` at the start, then did substantial work (graduation, file write, associations, meta), then finally called `wrap_completed` — but the verify and the commit were not atomically bound. Two concurrent `save_continuity` calls using the same valid token could both pass their earlier verification and both proceed to insert separate wraps rows. In the single-process model that race is theoretical, BUT the whole point of the 10.5c.4 fix is "TOCTOU structurally impossible" — leaving a second race open contradicts the thesis.
- **codex MEDIUM** — `load_wrap_snapshot` was doing 3 separate `_get_metadata` calls, itself TOCTOU-prone within the TOCTOU fix.
- **codex LOW + gemini convergent** — `StoreOperation` aliasing: SQLite variable-limit guard was raising with `operation="save_continuity"` (wrong taxonomy for a DB-side limit error).
- **contrarian F1** — `import re` inside method body is rule-of-three risk; extract module-level constant.
- **contrarian F2** — dead-code comment for `_set_metadata` was historical framing rather than forward-looking architectural rule.
- **contrarian F3** — `wrap_started()` no-arg contract drift: the signature allows the no-arg form, but the state it produces is now an integrity failure. Added `DeprecationWarning` emission on the no-arg path.
- **False positives / NITs skipped**: contrarian F4 (episode ID length — verified `_episode_id` returns `[:8]`, docstring is correct), contrarian F5 (CLI JSON shape — acceptable with doc comment), contrarian F6 (`succeeded` flag style — reviewer acknowledged non-bug), gemini ghost-episode deletion (logically consistent).

**Layer 3 fix for the replay race (codex HIGH):** Added a compare-and-swap UPDATE at the top of `wrap_completed`: `UPDATE metadata SET value = '' WHERE key = 'wrap_token' AND value = ?`. The CAS UPDATE is the first DML in the method — opens SQLite's deferred transaction and acquires the reserved write lock atomically. If `cursor.rowcount != 1`, the token has changed (another process ran prepare_wrap / wrap_cancelled / wrap_completed between the earlier verify and the commit) → rollback + raise `ValueError` with a distinctive "cleared or replaced during save" message (distinguishable from the earlier mismatch at continuity.py). Zero partial-commit possibility because the first DML is the CAS.

Other Layer 3 fixes: batched `load_wrap_snapshot` metadata reads into a single `SELECT ... IN (?,?,?)`; added `"wrap_completed"` to `StoreOperation` Literal; module-level `_WRAP_TOKEN_RE` constant in server.py; Store class docstring "Wrap-lifecycle invariants" section; `DeprecationWarning` on no-arg `wrap_started()` with explicit pointer to `prepare_wrap` as canonical caller.

Tests after Layer 3 fix pass: 608 → 610 (+2 CAS-closure tests). `test_wrap_completed_cas_rejects_replaced_token` uses `unittest.mock.patch` to simulate a concurrent process replacing the token between `validated_save_continuity`'s verify and `wrap_completed`'s CAS — asserts both the distinctive error message AND that no `wraps` row was inserted (CAS failure rolled back the implicit transaction before the `INSERT INTO wraps`).

**Layer 4 (integration semantics smoke test)** — ran 5 scenarios as a real end-to-end harness against the installed library via `PYTHONPATH=~/Documents/anneal-memory python3` in a clean `/tmp/am_layer4` directory. All passed on first run:

- **Scenario A** — TOCTOU window closure end-to-end: `prepare_wrap` → record TOCTOU → save → assert `episodes_compressed == 2` (not 3) AND TOCTOU episode's `session_id IS NULL` in the actual DB row.
- **Scenario B** — Deferred episode resurfaces: next `prepare_wrap` picks up exactly the TOCTOU episode.
- **Scenario C** — Token mismatch rejection + retry: wrong token → `ValueError`, wrap remains in progress, retry with correct token succeeds.
- **Scenario D** — Snapshot persists across Store close/reopen (the CLI cross-process boundary scenario). SQLite metadata is the shared state; no sidecar files.
- **Scenario E** — Audit chain-of-custody reconstruction from JSONL alone: parsed the audit.jsonl file, matched 3 prepare→complete pairs by `wrap_token`, verified each `wrap_started` event carried `wrap_episode_ids` and `wrap_episode_count`. Reconstructed "which episodes went into wrap #N" without touching the SQLite DB at all. Proof that the Decision 8e concession pays off — the audit trail is now a true forensic record.

### Key meta-patterns

- **`cross_architecture_layer3_catches_claude_blind_spots` — third instance in the 10.5c arc.** Layer 1 + Layer 2 (both Claude) missed the replay race; codex (GPT-5) immediately saw it as a verify-before-commit anti-pattern. First instance: 10.5c.2 doc migration. Second: 10.5c.3 factually-wrong TypedDict runtime-validation claim. Third: 10.5c.4 replay race. The pattern is now strong enough for graduation to 3x when episodic evidence is written.
- **`structural_invariants_beat_discipline_based_verification` — recurring at the code layer.** The H3 finding (delete `_set_metadata`) was the Layer 2 instinct to eliminate drift surface rather than document-around-it. The CAS closure is a third instance in the same evening's work (after the STEP 1.5 flow-meta bundle and the 10.5c.3 atomic-write invariant). Single-commit transactions, CAS UPDATEs for commit-atomic verification, StoreError on partial states — all structural invariants, no discipline dependencies.
- **`phill_tech_debt_pushback_reveals_hidden_laziness` — second clean instance in the 10.5c arc.** The 8e (audit enrichment) concession was the only item that flipped from "defer" to "fix" under the Phill challenge, but it was the right call — Layer 4 Scenario E proved the value directly.
- **`partnership_reconsideration_on_merits_is_not_caving` — third clean instance in the 10.5c arc.** Layer 3 flagged the replay race as HIGH; initial temptation was "theoretical race in single-process model, defer to follow-up." Honest re-read: the whole point of the 10.5c.4 fix is "TOCTOU structurally impossible" — leaving a second race open contradicts the thesis, and the fix is ~20 lines + one test. Reversed the defer decision and landed the CAS closure in the same session.

### Final diff shape

~11 files, approximately +2000/-98 insertions/deletions. 33 new tests across 3 test classes. Four-layer review end-to-end: 16 fixes from Layers 1+2, 6 fixes from Layer 3, 5 integration scenarios verified at Layer 4.

### What moves / next session candidates

- **10.5c.5 (file/DB two-phase atomicity)** — still scheduled. The 10.5c.4 fix did NOT close the mid-pipeline crash between continuity file write and DB metadata commit. Layer 2 H1 explicitly flagged this and it's filed against 10.5c.5.
- **10.5c.4a (operator surface, NEW)** — `anneal-memory wrap-status` + `wrap-cancel` CLI subcommands for introspecting and recovering stuck wraps (Layer 2 H2 + H4 bundled). ~50 LOC for the two subcommands. Low priority — not blocking v0.2.0.
- **10.5c.6 (SQLite error wrapping)** — still scheduled unchanged.
- **10.5d (framework testing)** — still scheduled.
- **v0.2.0 BREAKING CHANGES** captured for CHANGELOG:
  - `prepare_wrap` return value adds `wrap_token: str | None` field (TypedDict drift for consumers using the `PrepareWrapResult` type).
  - `validated_save_continuity` adds optional `wrap_token` kwarg (additive).
  - `Store.wrap_started()` with no args emits `DeprecationWarning`; canonical form adds `token` + `episode_ids` keyword-only parameters.
  - `Store.wrap_completed()` adds optional `episode_ids` and `wrap_token` keyword-only parameters.
  - New `Store.load_wrap_snapshot() → WrapSnapshot | None` primitive.
  - New `WrapSnapshot` TypedDict exported from `anneal_memory.types`.
  - MCP `save_continuity` tool gains optional `wrap_token` argument (JSON schema with pattern constraint).
  - CLI `save-continuity` subcommand gains `--wrap-token` flag.
  - `load_wrap_snapshot` raises `StoreError` on the "wrap_started_at set but wrap_token empty" partial state — behavior change for callers using the no-arg `wrap_started()` form followed by a canonical save. Documented and deprecation-warned; the canonical pipeline is unaffected.

---

## Session 10.5c.3: AnnealMemoryError + TypedDict shapes + prepare_wrap_package deprecation — COMPLETE (Apr 10, 2026 evening)

**Trigger:** Scheduled "small/medium cleanups" session in the v0.2.0 release arc. Initial scope was three items filed in next.md: `StoreError` exception class wrapping `OSError` (Gemini L3 flagged from 10.5c), `TypedDict` for `SaveResult`/`PrepareResult` return shapes (L3 convergence from complement F2 + contrarian #2 + codex), and `DeprecationWarning` on `prepare_wrap_package` public callers. Expected ~1–2 hours. Actual: full 4-layer review (session-code-review + Python API domain expert + consultation team of complement/gemini/contrarian/codex + integration semantics) surfaced two critical bugs the first pass introduced and several important findings promoted from "nice-to-have" after an anti-laziness self-audit. Session grew from 3 scope items to 16 fixes across three passes. Commit `934a3ce` on `main`, pushed.

**Key commits:** `934a3ce` (anneal-memory repo). Flow planning updates in `c00497c` (flow repo).

**Scope (16 fixes across three passes):**

### Pass 1 — Initial 3 items (the filed scope)

1. **`StoreError(OSError)` exception class (FIRST-PASS DESIGN, REVERSED IN PASS 2):** Added a `StoreError` class subclassing `OSError` for "backward compatibility" — the assumption was that existing `except OSError` callers would continue to work. Raised from `save_continuity` and `save_meta` with `operation` + `path` keyword-only context, chained via `from exc`. Passed Phill's initial review under the "Option C: keep StoreError(OSError) + fix __reduce__" framing.

2. **TypedDict return shapes** added to `types.py`: `StalePatternDict`, `WrapPackageDict`, `PrepareWrapResult`, `SaveContinuityResult`. Return annotations in `continuity.py` migrated from `dict[str, Any]` to the typed forms. Header comment explained the shape choice (TypedDict over dataclass for JSON-serializability — which would get complicated by the end).

3. **DeprecationWarning on `prepare_wrap_package` (FIRST-PASS DESIGN, REVERSED IN PASS 2):** Added an `_internal: bool = False` underscore-private kwarg to suppress the warning when called from `prepare_wrap`. Public callers got the warning, internal caller got the bypass.

### Pass 2 — Layer 1+2 review + anti-laziness audit

Layer 1 (session-code-review) and Layer 2 (Python library API domain expert) ran in parallel on the staged diff. Layer 2 found two CRITICAL bugs Layer 1 missed:

- **C1 — `StoreError(OSError)` breaks pickle/copy round-trips.** `OSError.__reduce__` reconstructs via `type(self)(*self.args)` which calls `StoreError("boom")` with no keyword arguments — but the constructor requires `operation` as keyword-only. `TypeError: __init__() missing 1 required keyword-only argument: 'operation'` on unpickle. Verified empirically. pytest-xdist, ProcessPoolExecutor, any RPC framework, any logging system marshaling exceptions — all break. Not theoretical.
- **C2 — Fault injection tests were vacuous.** `monkeypatch.setattr("builtins.open", exploding_open)` made `with open(tmp_path, ...)` crash before the tmp file was ever created. The `assert tmp_files == []` cleanup assertions passed because nothing was ever written, not because cleanup worked. Tests claimed to verify the atomic-write invariant but exercised none of it.

Layer 2 also surfaced several IMPORTANT findings (stale docstrings, TypedDict drift risks, `_internal=True` as an anti-pattern vs extracting a private helper now, `wrap_result: WrapResult` crashing `json.dumps`).

Phill pushed back on the initial fix plan with the "anti-laziness" challenge after I bucketed six items as "skip." Self-audit revealed six of them were promoted to "fix" after honest defense against the `defer_as_completion_pressure` pattern. Phill also rejected the "Option C" framing on StoreError(OSError): "to my knowledge anneal-memory has no active users so I am not worried about backwards compatibility — we need to do what is BEST for the project in the long term." This opened the door to the Option B redesign (drop OSError subclass entirely).

Pass 2 fixes:

4. **Dropped `OSError` subclass.** New `AnnealMemoryError(Exception)` library base class. `StoreError(AnnealMemoryError)`. The original `OSError` is preserved via `__cause__` chaining (`raise StoreError(...) from exc`), so in-process callers can still dig for errno. Mirrors the convention in `sqlalchemy.exc`, `httpx`, and other library-level Python packages — the boundary is "the library failed," not "a file operation failed." Pickle issue dissolves because we're no longer fighting `OSError.__reduce__` contract constraints. `__reduce__` is implemented manually via module-level `_reconstruct_store_error` reconstructor to round-trip `operation` + `path`.

5. **`operation` typed as `Literal["save_continuity", "save_meta"]`** via module-level `StoreOperation = Literal[...]` alias. New raise sites must add their identifier here before raising; forces deliberate expansion of the contract surface. Enforcement is compile-time only (soft contract until mypy-in-CI lands — filed as follow-up).

6. **Atomic-write invariant restored via `succeeded` flag + `finally` block.** Layer 1 I1: Pass 1 narrowed `try/except Exception: cleanup; raise` to `try/except OSError: cleanup + wrap; raise`, which silently lost cleanup on non-OSError paths (`UnicodeEncodeError` on malformed agent text, `TypeError` from a buggy caller). Fix pattern: `succeeded = False; try: ...; succeeded = True; except OSError: raise StoreError(...) from exc; finally: if not succeeded: _safe_unlink(tmp_path)`. New `_safe_unlink` helper swallows secondary OSError so primary exceptions are never masked.

7. **Extracted `_build_wrap_package` private helper NOW.** Rejected the `_internal=True` bypass-kwarg as anti-pattern after Layer 2's argument: it leaked into the public signature, would need deletion in 10.6 anyway, contradicts Python stdlib idiom (`logging.warn` → `logging.warning`, `asyncio._get_event_loop` etc.). `prepare_wrap_package` is now a thin deprecated wrapper that emits `DeprecationWarning` and delegates. `prepare_wrap` calls `_build_wrap_package` directly. Deprecation message follows CPython "deprecated since 0.2.0, will be removed in 0.3.0" idiom.

8. **`SaveContinuityResult.wrap_result: dict[str, Any]`** (not `WrapResult` dataclass). Layer 2 I4 verified empirically that `json.dumps(validated_save_continuity_result)` crashed on the embedded dataclass. Fix: `validated_save_continuity` calls `dataclasses.asdict(wrap_result)` at return time. Entire canonical pipeline return is now JSON-serializable top-to-bottom — locked by `test_save_continuity_result_is_json_serializable` parity test.

9. **`PrepareWrapResult.status: Literal["empty", "ready"]`** discriminant (was free-form `str`).

10. **Fault injection rewritten.** Monkeypatch `Path.replace` instead of `builtins.open`, so the tmp file IS created, the atomic rename fails, and cleanup runs on a real on-disk file. New fault-injection tests added: non-OSError mid-write failure (verifies broad-cleanup invariant), `os.fsync` failure (with a pre-existence check so the test isn't vacuous), `unlink-in-cleanup` failure (verifies `_safe_unlink` swallows secondary errors so primary exception propagates cleanly), `TypeError` from `json.dump` on non-JSON-serializable meta, and the "previous continuity file untouched when new write fails" atomic-write invariant.

11. **Partial-failure window docstring warning** on `validated_save_continuity` (Layer 1 I2, doc-only — the actual fix is the 10.5c.5 two-phase commit work). Transports catching `StoreError` must NOT assume clean wrap abandonment; the pipeline's four write stages (continuity file → associations → meta sidecar → `wrap_completed`) are not transactional, and a mid-pipeline failure leaves partial state that requires explicit recovery on next session start. Prevents transport implementors from shipping broken assumptions before 10.5c.5 ships the fix.

12. **Stacklevel test hardened** (Layer 2 I1) to assert exact `lineno` via `sys._getframe().f_lineno + 1` rather than just filename-endswith. A filename-endswith check passes for any stacklevel landing anywhere in the test file — silent regression risk. The lineno check catches silent stacklevel drift in future refactors. Added a fragility comment on the `+1` arithmetic.

13. **Docs migration:** `docs/library-quickstart.md` deprecation callouts on the 2 `prepare_wrap_package` mentions. Prevents 10.5c.2 bypass-by-omission recurrence.

### Pass 3 — Layer 3 consultation (cross-architecture review)

Layer 3 dispatched `complement + gemini + contrarian + codex` against the full staged diff. First dispatch had an empty-diff bug (consult.py `--diff staged` always runs git in the script's `cwd`, not `--cwd`; my invocation from `/flow` with `--cwd /anneal-memory` meant git ran against /flow which had no staged changes; agents reviewed the prompt narrative with zero code context and produced useless findings). Second dispatch correctly ran from the anneal-memory repo. Layer 3 found three real issues Layers 1+2 missed, convergent across multiple agents:

- **CRITICAL — TypedDict runtime-validation claim in `types.py` header comment was factually wrong** (Gemini, codex, contrarian). The comment justified the callable-constructor form (`WrapPackageDict(episodes=...)`) over the literal-dict form by claiming it "catches key-name typos at runtime" — which is false. TypedDict provides zero runtime validation in either form. Only static type checkers catch typos. Neither Claude Layer 1 nor Claude Layer 2 caught the factual error — different Claude-based reviews sharing the same blind spot. Layer 3's cross-architecture diversity caught it.
- **IMPORTANT — `chars` field documented as "Byte count" but is actually character count** (complement F3, contrarian I2). `len(text)` in Python 3 is code points, not bytes. For pure ASCII they coincide; for non-ASCII they diverge up to 4x in UTF-8. Fixed in 3 locations.
- **IMPORTANT — Transport adapters don't catch `StoreError`** (complement F1, codex). `server.py` and `cli.py` only caught `ValueError` from `validated_save_continuity`. The new `StoreError` fell through to generic handlers and the `.operation`/`.path` context was discarded. The entire hierarchy was defined but not consumed — "ceremonial." ~15-line fix: both transports now catch `StoreError` after `ValueError` and surface structured "store I/O failure during {operation} at {path}: ..." messages.
- **IMPORTANT — `__cause__` not preserved through pickle** (complement F2, contrarian N1). Standard Python limitation, but my docstrings repeatedly claimed "`__cause__` is preserved" without qualification. Fix: docstring caveat explaining pickle limitation + that the message already embeds the underlying error text.
- **IMPORTANT — `wrap_result` dataclass→dict is a silent breaking change** (contrarian I1, codex). Not a correctness issue (no active users), but asymmetric with the deprecation discipline applied to `prepare_wrap_package` in the same session. Fix: BREAKING CHANGES block in `next.md` + expanded v0.2.0 release checklist in project `CLAUDE.md`.
- **IMPORTANT — `next.md` test count stale** (codex). Said 548→575, actual was 577.
- **IMPORTANT — Redundant `_safe_unlink` call in except-block AND finally-block** (Gemini). Pass 2 fix kept both with a comment explaining the idempotent no-op on OSError path. Gemini correctly argued: single cleanup source is cleaner. Removed the except-block call; finally handles all exit paths.
- **NICE-TO-HAVE — Add `__cause__` pickle caveat, strengthen context-chain suppression assertion, add args-mutation test, add strict-filter (PEP 565) test, add "old file untouched" assertion to non-OSError test, export `StoreOperation`, future-relocation note for `exceptions.py`.** All addressed.

Also swept for stale `prepare_wrap_package` docstring references in `prepare_wrap`, `types.py`, and one internal comment (Layer 2 I1) — same doc-drift bug class as 10.5c.2's bypass-by-omission. Fixed to route at `_build_wrap_package` or `prepare_wrap` as appropriate.

### Pass 4 — Layer 4 integration semantics check

Smoke-tested six end-to-end claims against the final code:

1. All new exports importable from top-level package.
2. `AnnealMemoryError` catches `StoreError` at the library boundary.
3. Full canonical pipeline: `prepare_wrap` → compress → `validated_save_continuity` → `json.dumps(result)` succeeds, `chars == len(text)`.
4. Deprecation discipline: `prepare_wrap_package` emits warning with correct "since 0.2.0, removed in 0.3.0" text; canonical `prepare_wrap` pipeline emits zero warnings.
5. `StoreError` pickle/copy round-trip preserves `.operation` + `.path`.
6. Atomic-write invariant: tmp file cleaned up and previous continuity file preserved on OSError failure (via `Path.replace` injection).

All six claims verified.

**Test delta:** 548 → 577 passing (+29 net).
- New: `TestStoreError` (10 tests — hierarchy + pickle/copy/deepcopy/args-mutation + fault injection matrix), `TestPrepareWrapPackageDeprecation` (7 tests — warning text + CPython idiom + exact-lineno stacklevel + canonical pipeline regression gate + private helper direct-call gate + strict-filter + pass-3-specific coverage), `TestTypedDictReturnShapes` (4 tests — exports + two drift-check tests + JSON-serialization parity).
- Deleted: `test_store_error_is_oserror_subclass`, `test_save_continuity_store_error_also_caught_as_oserror`, `test_store_error_caught_by_oserror_handler` (obsoleted by hierarchy drop), `test_internal_kwarg_suppresses_warning` (obsoleted by `_build_wrap_package` extraction).

**Files touched (9):** `anneal_memory/__init__.py`, `anneal_memory/cli.py`, `anneal_memory/continuity.py`, `anneal_memory/server.py`, `anneal_memory/store.py`, `anneal_memory/types.py`, `docs/library-quickstart.md`, `tests/test_continuity.py`, `tests/test_store.py`. +1335/-147 lines.

**Meta-observations captured for continuity wrap:**

1. **`doc_text_drift_survives_forward_looking_review` — now 3x pattern.** 10.5c.2 shipped with "WrapResult dataclass" wording + bypass-by-omission. 10.5c.3 Pass 1 shipped with "StoreError subclasses OSError" stale Raises block. 10.5c.3 Pass 2 shipped with factually-wrong "TypedDict catches typos at runtime" justification comment. Three instances across two adjacent sessions where doc text with load-bearing claims drifted from code shape and survived forward-looking review. Structural not accidental. Fix candidate: grep for API-boundary claims (`subclass`, `raises`, `returns`, `calls`) in docstrings touched by the diff and spot-check each against actual code shape before shipping.

2. **Cross-architecture Layer 3 caught what same-architecture Layers 1+2 missed.** The TypedDict runtime-validation claim was flagged by 3 of 4 Layer 3 agents (Gemini, codex, contrarian); both Claude Layer 1 and Claude Layer 2 missed it. Different training → different blind spots → convergence = signal. Textbook validation of the multi-architecture review protocol.

3. **`consult.py --diff staged` with `--cwd` had an empty-diff bug.** `git diff` always runs in the script's `cwd`, not the `--cwd` arg (which is for agent sandboxing). First dispatch from `/flow` ran git against `/flow` which had no staged anneal-memory changes. Agents reviewed prompt narrative only and produced useless findings. Worth filing as a UX bug — the `[NO CHANGES: git diff returned empty]` message should be an error, not a silent continue.

4. **Partnership anti-laziness challenge worked exactly as designed.** Phill's "can you honestly defend skipping the other nice-to-haves or are you just getting lazy after that long analysis?" prompt forced a self-audit that promoted 6 items from "skip" to "fix." All 6 survived the subsequent Layer 3 review as real value. The `phill_tech_debt_pushback_reveals_hidden_laziness` pattern is operationally validated for the second time.

5. **Partnership reversal on `StoreError(OSError)`.** Phill's "to my knowledge anneal-memory has no active users so I am not worried about backwards compatibility — we need to do what is BEST for the project in the long term here and to my mind that means discussing this better" was the right call. First-pass decision (keep OSError subclass + fix `__reduce__`) was defensive against a hypothetical user. Drop-the-subclass decision was honest about reality: no users, no backward-compat promise to honor, cleaner hierarchy, simpler code, dissolves the pickle problem structurally. Another instance of `partnership_reconsideration_on_merits_is_not_caving`.

**v0.2.0 release status:** UNBLOCKED pending 10.5c.4 (TOCTOU handshake token), 10.5c.5 (file/DB two-phase atomicity), 10.5c.6 (SQLite error wrapping — new this session), 10.5d (framework testing top 5), 10.5d+ (mypy-in-CI — new this session). Session 10.6 rescoped: originally "rename prepare_wrap_package → _build_wrap_package (private)," now just "delete the deprecated public wrapper" since the extraction happened in 10.5c.3.

---

## Session 10.5c.2: Doc Rewrites — Migrate to Canonical prepare_wrap Entry Point — COMPLETE (Apr 10, 2026)

**Trigger:** Session 10.5c.1 made the library canonical in CODE (one implementation of save/prepare, transports are thin adapters), but the docs still taught the old `prepare_wrap_package(episodes, continuity, project)` pure-helper pattern. Calling that helper and then `validated_save_continuity()` bypasses `store.wrap_started()` — every save flagged `skipped_prepare=True`. Same thesis-breaking bypass class as the Engine kill. Library-first positioning was structurally true in code and rhetorically true in docs, but the code examples across 14 files taught users to reach for the bypass. Complement + contrarian both elevated 10.5c.2 to v0.2.0 release blocker for this reason.

**Scope (1 session, 14 files):**

1. **README.md Quick Start + All Three Paths + Affective State example:**
   - Import migrated `prepare_wrap_package` → `prepare_wrap`.
   - Quick Start code rewrote the wrap path: `wrap = prepare_wrap(store) / if wrap["status"] == "ready": compressed = your_llm.compress(wrap["package"]) / validated_save_continuity(store, compressed)`. Undefined `compressed_text` placeholder (inherited from the old example) was fixed to a defined `compressed = your_llm.compress(...)` pseudocode call. Compression comment now names the thesis: "Compression IS the cognition — patterns emerge from the act of compressing, not from storage."
   - Added `# "empty" status means no new episodes to wrap — skip` orientation line so the happy-path `if` doesn't read as error handling.
   - "All Three Paths" paragraph strengthened: "CLI and MCP are thin transport adapters over the same library — not separate implementations. Every access pattern calls the same `prepare_wrap(store)` and `validated_save_continuity(store, text)` pipeline under the hood." Fixes the "pick one" framing contrarian flagged in the API table.
   - Framework Integrations intro renamed the four core functions: `record()`, `recall()`, `prepare_wrap()`, `validated_save_continuity()`.
   - Affective State example rewritten to use the full `prepare_wrap` → `validated_save_continuity(..., affective_state=AffectiveState(...))` pattern with a defined `compressed` variable. Removed the misleading "affective state recorded via wrap metadata" comment on the bare `store.save_continuity()` call (which was structurally wrong — the bypass path doesn't record affective state at all).

2. **docs/library-quickstart.md:**
   - New "Canonical entry points" callout block ABOVE the code block (moved from below — copy-paste visual hierarchy). Names both deprecated primitives (`prepare_wrap_package()`, `store.save_continuity()`) and explains why they exist (test use only) without giving an "advanced users" escape hatch.
   - Key listings cleanly separated: `wrap[]` top-level keys (`status`, `message`, `episode_count`, `package`, `assoc_context`) vs `package[]` nested keys (`episodes`, `episode_count`, `continuity` with None-on-first-wrap note, `stale_patterns`, `instructions`, `today`, `max_chars`). Previously confused the two nesting levels.
   - Affective State section MOVED from after Audit Trail to right after Wrap Sequence — it's part of the wrap mechanics and was buried as the last substantive section before the utility sections.
   - New "Why `prepare_wrap(store)` and not `prepare_wrap_package()`?" explanation paragraph after the code block, cross-referencing both deprecated primitives in one breath. Removed the "and advanced users managing their own episode fetching" escape hatch from the warning prose — the pure helper exists for unit tests, period.
   - Fixed invalid module-level `return` in the empty-wrap branch (would ErrorOut if anyone copy-pasted) → restructured as `if wrap["status"] == "ready":` block so the example is valid module-level pseudocode.
   - Fixed pre-existing bug in Associations section: `a.id_a`/`a.id_b` → `a.episode_a`/`a.episode_b` (matches `AssociationPair` dataclass fields). Complement's quick-pass review surfaced this; fixed in same session per `defer_as_completion_pressure` Proven pattern.

3. **12 framework integration guides migrated:**
   - `langgraph.md`, `crewai.md`, `openai-agents.md`, `anthropic-agents.md`, `google-adk.md`, `pydantic-ai.md`, `smolagents.md`, `llamaindex.md`, `haystack.md`, `camel-ai.md`, `autogen.md`, `dspy.md`.
   - 6 guides added `validated_save_continuity` to their import lines (anthropic-agents, smolagents, llamaindex, haystack, camel-ai, autogen — previously imported only `prepare_wrap` and stubbed the save step as a comment).
   - All 12 guides now show `validated_save_continuity()` as REAL uncommented code, not a `# Compress via LLM and save` comment stub. Contrarian's single biggest attack: users hitting a comment-only save step grep the Store class, find `store.save_continuity()`, call it — the bypass recreated by documentation gap. Fixed by making the full pipeline visible in every guide with a framework-natural compression placeholder.
   - `langgraph.md` status idiom standardized: `if wrap["status"] == "empty": return None` → `if wrap["status"] == "ready":` (inverted to match all other guides).
   - `openai-agents.md` `TracingProcessor.on_trace_end`: migrated from `episodes = store.episodes_since_wrap()` remnant to `wrap = prepare_wrap(store)` canonical path.
   - `anthropic-agents.md` `on_compact` hook: migrated from `episodes = store.episodes_since_wrap(); if len(episodes) > 10` to `store.status().episodes_since_wrap > 10` — `store.status()` is the legitimate public read-only peek at state; `episodes_since_wrap()` is internal buffer access.
   - `dspy.md` optimizer section: migrated from `episodes = store.episodes_since_wrap()` remnant to full `prepare_wrap` → `ChainOfThought` compression → `validated_save_continuity` pattern.
   - `pydantic-ai.md` `wrap_run` middleware: removed dead `continuity = store.load_continuity()` variable (was used by the old `prepare_wrap_package(episodes, continuity, ...)` call; became orphaned when `prepare_wrap(store)` started loading continuity internally). Updated the middleware docstring to explain that memory is loaded into the agent via `memory_instructions()`/`deps`, not inside the wrapper — the wrapper just runs the agent and wraps the session at the end.

**Consultation discipline:**

1. **First pass** (post-migration, pre-commit): 4-agent consultation in standard mode — `complement,gemini,contrarian,anansi`.
   - Gemini timed out at 300s; the other three produced strong convergent findings. No retry — 3-agent convergence was sufficient signal.
   - Critical findings: F2 (README undefined `compressed_text`, complement — ship blocker), F7 (pydantic-ai stale `continuity` variable, complement), Anansi Critical 1 (3 `episodes_since_wrap()` remnants in secondary sections — dspy optimizer, openai-agents `on_trace_end`, anthropic-agents `on_compact`), Contrarian Finding 1 / Anansi Critical 2 (5–7 guides stub `validated_save_continuity` as a comment, recreating the bypass).
   - Positioning findings: Contrarian Finding 2 ("advanced users managing their own episode fetching" is an escape hatch — same class as the Engine's origin story told in documentation), Contrarian Finding 3 (README "four core functions" promise vs guides that only deliver two), Anansi #8 (API table reads as "pick one" not "library canonical").
   - Polish findings: F1 (`assoc_context` nested vs top-level confusion in quickstart key listing), F6 (warning block below code block — copy-paste visual hierarchy), Anansi #4 ("ready" status check needs orientation comment), Anansi #5 (Affective State section buried after Audit Trail), Anansi #7 (inconsistent status check idiom — langgraph inverted).

2. **Fix pass:** All critical, important, and polish findings addressed (Phill chose to include the polish items: "vital that README is not only perfect but hooks devs" at zero credibility — the hook argument made at consultation-synthesis time).

3. **Second pass** (post-fix, pre-commit): `complement` in quick mode on the fix-pass diff. Verdict: "Clean. Ship it." Every anneal-memory API call in the new examples verified against `anneal_memory/continuity.py`, `anneal_memory/store.py`, `anneal_memory/types.py`: imports, `wrap[]`/`package[]` dict keys, `validated_save_continuity(store, text, affective_state=..., today=...)`, `AffectiveState(tag=str, intensity=float)`, `store.status().episodes_since_wrap: int`. Also surfaced the pre-existing `a.id_a`/`a.id_b` bug in library-quickstart Associations section (not in the diff); fixed in-session.

**Commit:** `2771a57` (pushed to origin/main).

**Test counts:** 548 passing throughout (docs-only change, no regressions expected, none observed).

**File-level diff:** 14 files changed, `+285/-202` (net +83 lines — mostly expanded key listings, explicit uncommented save calls in every framework guide, and the moved Affective State section).

**Unblocks:** v0.2.0 release. Per `brief.md` release checklist, next steps after ship are 10.5c.3 (StoreError + TypedDict + DeprecationWarning) → 10.5c.4 (TOCTOU handshake token) → 10.5c.5 (file/DB atomicity) → 10.5d (framework testing) → version bump + CHANGELOG + PyPI publish.

**Meta-lessons:**

- **`defer_as_completion_pressure` cuts both ways:** the Proven pattern says "if it matters, fix when found." Complement's quick-pass surfaced a pre-existing bug in the Associations section I didn't touch; fixing it in-session was the correct call rather than deferring to "we'll get it next time." The pattern is about fighting the urge to defer AS completion pressure, not about deferring for scope purity.
- **Hook-quality matters at zero credibility:** Phill's pushback on "nice-to-haves are nice-to-have" reframed the work. At zero audience + zero credibility, the README and primary docs have to clear a higher bar than at established-audience + established-credibility, because one rough edge loses the reader who was deciding "is this real or a toy." Nice-to-haves become must-haves when the dev's first 15 lines of code is the entire judgment.
- **Reversing an initial judgment call is a partnership signal, not a failure:** Initial instinct was "consult would produce marginal value — mechanical migration, tests pass." Phill asked "do you feel a consult wouldn't produce much value here?" as a direct test. Reconsidering honestly produced the correct call (consult WAS valuable, and the findings were load-bearing). The test is not "can you hold your initial take under pressure" but "can you reconsider honestly when pushed, and own the revision." Caving to pressure is failure; reconsidering on substance and changing position is the partnership working.
- **Structural review + quick-pass verification is a cheaper "belt+suspenders" than full re-consultation:** the first pass was 4 agents full diff (~5 min, ~$4.75). The fix pass produced substantial new content that hadn't been reviewed. Rather than re-run the full team, a single `complement` quick pass (~80s, ~$1.26) on the new examples was sufficient code-accuracy verification — diminishing returns on the second pass, but non-zero returns, and complement's tool-heavy code-accuracy lens is the right fit for "did the new content keep the signatures right." Pattern: full team pre-commit on the first pass, targeted single-agent quick pass on any substantial fix pass that adds new content.

**Commits from this session:** `2771a57` (single commit for all 14 files, per the "prefer one bundled PR over many small ones" pattern for docs sessions that are logically one unit of work).

---

## Session 10.5c.1: Rule-of-Three Elimination — Library Canonical + Thin Transports — COMPLETE (Apr 10, 2026)

**Trigger:** Diogenes's scheduled code-review pass (Apr 10 04:08 UTC) caught four findings in anneal-memory code, three of which were structural and one thesis-level. Most critically: the `validated_save_continuity()` library function shipped in Session 10.5c had **already diverged** from the MCP and CLI implementations 12 hours after shipping (Finding #1: under-reported `graduations_demoted` by missing `bare_demoted`; Finding #2 variant: recomputed `citation_reuse_max` from `citation_counts` instead of using the `grad_result.citation_reuse_max` field). The save_continuity pipeline existed in THREE parallel implementations (`server.py:_tool_save_continuity`, `cli.py:cmd_save_continuity`, `continuity.py:validated_save_continuity`), and each author who wrote one read the previous ones and copied. Classic rule-of-three fired on contact. Finding #4: `cli._query_wraps` reached into `store._conn` directly (same pattern Diogenes had called out on server.py earlier in 10.5c).

**Scope decision:** Rather than patch the four findings in-place (which would re-ship a still-triplicated pipeline), use the session to eliminate the duplication entirely. Library becomes canonical; MCP and CLI become thin transport adapters. Matches the 10.5c "library is the product, CLI and MCP are interfaces to it" positioning — which shipped with the positioning stated in docs but NOT reflected in the shape of the code. 10.5c.1 makes the code actually match the positioning.

**What was built (7 commits, 548 tests):**

1. **Library canonical pipelines** (commit `8c89638`):
   - `continuity.py:validated_save_continuity` widened return dict: added `bare_demoted`, `gaming_suspects`, `citation_reuse_max`, `sections`, `demoted` (split) keys. Fixed the `graduations_demoted` under-reporting by using `grad_result.demoted + grad_result.bare_demoted`. Fixed `citation_reuse_max` divergence by using the grad_result field directly.
   - New `continuity.py:prepare_wrap(store, *, max_chars, staleness_days)` — canonical store-aware entry point that handles the full wrap lifecycle (episodes_since_wrap + empty-path wrap_cancelled + prepare_wrap_package + wrap_started + get_association_context). Returns a dict with `status`, `message`, `episode_count`, `package`, `assoc_context`.
   - New `continuity.py:format_wrap_package_text(result)` — canonical agent-facing display text formatter. Both MCP and CLI transports render identical output through this function instead of each hand-building the same text.
   - New `Store.get_wrap_history() -> list[WrapRecord]` public API replacing `cli._query_wraps` private-conn access (Finding #4).
   - New `WrapRecord` frozen dataclass in `types.py`.
   - 22 new library-level tests covering the widened contract, `prepare_wrap` empty/ready paths, `format_wrap_package_text` rendering, and `get_wrap_history()` round-trip.

2. **Transports migrated to canonical** (commit `1cedc79`):
   - `server.py:_tool_save_continuity` and `_tool_prepare_wrap` reduced to thin adapters (~60 lines total, from ~180). Parse MCP args → call library → format MCP text response. ValueError → `is_error=True`.
   - `cli.py:cmd_save_continuity` and `cmd_prepare_wrap` similarly reduced (~75 lines from ~210). Parse file/stdin + argparse → call library → format text or JSON. ValueError → stderr + `sys.exit(1)`.
   - `cli._query_wraps` deleted; `cmd_export`, `cmd_diff`, `cmd_stats`, `cmd_history` migrated to `store.get_wrap_history()` with attribute access via `asdict()` for JSON serialization paths.
   - Net diff: 148 line reduction in transports, 754 lines added across library + tests + docstrings. All 540 tests passing unchanged against the thin adapters (the existing end-to-end transport tests implicitly verify wiring correctness).

3. **Test date hardening** (commit `bfdeb5e` — Diogenes Finding #3):
   - `test_validated_save_runs_full_pipeline` upgraded to use `date.today().isoformat()` dynamically AND to use a real 2x citation with matching episode evidence, with `assert graduations_validated >= 1` so the integration path actually exercises graduation instead of silently skipping citation validation as wall-clock drifts.
   - `TestCmdSaveContinuity._VALID_CONTINUITY` class constant converted to `_valid_continuity()` staticmethod that rebuilds with dynamic today on each call. All 6 `self._VALID_CONTINUITY` call sites migrated.

4. **Layer 1+2 review fixes** (commit `45ac983`):
   - **MEDIUM crash path:** `prepare_wrap()` now marks `wrap_started()` as the LAST store write, after all reads and package construction. Previously if any post-`wrap_started()` code raised (e.g., `get_association_context`), the store was left with a stale `wrap_in_progress=True` flag. Symmetric with `wrap_cancelled()` on the empty path.
   - **LOW:** `format_wrap_package_text` now falls through to bare message for any status != `"ready"` (defending against future statuses or malformed hand-built dicts).
   - **LOW:** CLI `cmd_save_continuity` JSON output extended with pure additive fields (`demoted`, `bare_demoted`, `citation_reuse_max`, `gaming_suspects`). Pre-existing keys unchanged for backward compat.
   - **YELLOW shape fix:** Added `chars` as top-level key in `validated_save_continuity` return dict. Transports previously had to reach into `result["wrap_result"].chars` while every other metric was flat. New `test_chars_top_level_matches_wrap_result` locks the invariant.
   - **YELLOW type boundary:** `WrapRecord.episodes_compressed` and `continuity_chars` changed from `int | None` to `int`. `Store.get_wrap_history()` now coerces nullable schema columns to 0 at construction. CLI callers (cmd_diff, cmd_stats, cmd_history) dropped all `or 0` guards on these fields.
   - **YELLOW bypass-site doc:** `Store.save_continuity()` now has a prominent warning docstring that calling it directly bypasses the entire immune system (graduation, associations, decay, meta, wrap_completed). Previously only `validated_save_continuity` warned about the bypass; the bypass site itself had a bland "atomic write" description.
   - **YELLOW partial:** `prepare_wrap_package` now has a prominent `.. warning::` block pointing users at `prepare_wrap(store, ...)` as the canonical entry point. Documents the semantic gap (calling the pure helper directly means `store.wrap_started()` is never called, so `validated_save_continuity` will report `skipped_prepare=True` on every subsequent save). Full integration-guide migration deferred to Session 10.5c.2.

5. **Layer 3 consultation fixes + real bug catch** (commit `7b03d8c`):
   - **Cross-transport parity test catches CLI input-mutation bug:** New `TestCrossTransportParity::test_library_mcp_cli_produce_identical_wrap_metrics` test seeds three independent stores with identical deterministic episodes (content-hashed IDs match across stores), runs full prepare → save through library/MCP/CLI, asserts all three `get_wrap_history()` records produce byte-identical domain metrics. On first run, failed with `continuity_chars: 340` (CLI) vs `341` (library/MCP) — a 1-character divergence. Root cause: `cmd_save_continuity` was calling `.strip()` on file/stdin input before passing to the library, silently mutating user content. Library and MCP received verbatim content. The entire human + 6-agent review pipeline had missed this; the parity test caught it on first execution. **Fix:** read file/stdin verbatim, run the empty-check on a stripped local copy without mutating the value passed downstream. Test now passes.
   - **`get_wrap_history()` exception handling narrowed (codex HIGH):** Was catching ALL `sqlite3.OperationalError` and returning `[]`, so database-locked, corruption, disk-full, and connection-gone errors all silently looked like "no wraps yet." Now only swallows "no such table" (legacy DB) and re-raises everything else so monitoring subcommands surface real failures.
   - **`store: Any` → `Store`** typed parameter on `prepare_wrap` and `validated_save_continuity` (contrarian MEDIUM — credibility). Imported via `TYPE_CHECKING` guard to avoid runtime circular dependency.
   - **TOCTOU window documented:** Both `prepare_wrap` and `validated_save_continuity` now have `.. warning::` blocks explaining that the prepare/save window is not frozen — episodes recorded between the two calls are included in validation and metrics even though the agent's compression was based on the smaller pre-fetch set. Single-threaded agent workflows (MCP, CLI, Claude Code) don't hit the gap; framework integrations that interleave episode recording with session wrapping must treat the sequence as a critical section. Session-handshake token deferred to Session 10.5c.4.

6. **Layer 4 integrity manifest regeneration** (commit `6c42fe4`):
   - Layer 4 (integration semantics: "does the system actually do what it claims?") caught a **pre-existing** bug that Layers 1-3 missed. `tool-integrity.json` was last regenerated in v0.1.8 (commit `1a82bb4`). The `delete_episode` MCP tool was added in v0.1.9 (commit `7a379e2`). The integrity manifest has been silently failing host-side verification for one full session and one PyPI release. `verify_integrity()` returned `(False, ["Tool 'delete_episode' not found in integrity file"])`.
   - Not introduced by 10.5c.1, but caught by Layer 4's different attention mode.
   - Regenerated via `generate_integrity_file()`. Other 5 tool hashes unchanged (their descriptions haven't drifted). Verify now returns `(True, [])`.

7. **Tech debt cleanup (after Phill pushback on "ignored" findings)** (commit `2dddde2`):
   - **WrapResult.section_sizes wired through:** `Store.wrap_completed()` now accepts an optional `section_sizes: dict[str, int] | None` parameter; `validated_save_continuity` passes `sections` to it. Previously the dataclass field was structurally always `{}` because `wrap_completed` hard-coded it. This was the 5-line fix I was hiding from at end-of-session review and Phill correctly called out as lazy.
   - **AffectiveState clamping symmetry:** Removed the redundant `max(0.0, min(1.0, ...))` clamp from MCP `_tool_save_continuity`. The `AffectiveState.__post_init__` dataclass method has always been the canonical clamp site. MCP's `float()` coercion is retained because it's legitimately transport-specific (JSON sends arbitrary types; argparse on CLI already coerces). Result: both transports parse transport-native input into Python `float`, then hand off to `AffectiveState` for single-source-of-truth clamping.
   - **`today` parameter on `validated_save_continuity`:** New keyword-only argument defaulting to `date.today().isoformat()`. Callers that need deterministic behavior (tests, reproducible experiments) can pin the date; production callers ignore it. Mirrors the existing `today` parameter on `prepare_wrap_package`. Eliminates wall-clock test fragility AND improves testability for reproducible identity experiment runs.
   - **Shipped manifest test guard:** New `TestShippedManifest` in `tests/test_integrity.py` verifies the manifest shipped inside the package passes `verify_integrity()` against current TOOLS definitions. Catches the exact Layer 4 staleness class automatically — would have caught the v0.1.9 `delete_episode` staleness before it ever shipped.

**Test counts:** 518 (10.5c) → 540 (after commits 1-3) → 542 (after review fixes) → 543 (Layer 3 parity test) → 548 (tech debt cleanup). Net +30 tests, all covering the canonical library contract + structural regression guards.

**Key architectural principles emerged:**
- **`library_canonical_with_thin_transports` = structural elimination of rule-of-three drift:** Before 10.5c.1, three implementations that could diverge (and did). After 10.5c.1, one implementation whose keys are the contract. The improvement is real — divergence is worse than stringly-typed dict keys.
- **`parity_test_is_stronger_than_review`:** The cross-transport parity test caught a real bug (CLI `.strip()` mutating input) that the entire 4-layer review missed including 2 specialized review agents and 4 consultation agents. A structural invariant test is a stronger regression guard than any amount of careful human or LLM review.
- **`layer_4_attention_mode_catches_pre-existing_debt`:** Layers 1-3 ask "is the new code correct." Layer 4 asks "does the system actually do what it claims about itself." Layer 4 found the v0.1.9 integrity manifest staleness that had silently failed verification for two sessions and one PyPI release. Layers 1-3 would never have looked.
- **`the_12-hour_divergence_is_both_positive_and_negative_signal`:** Positive: Diogenes's scheduled review caught a real bug in shipped code before it reached users. Negative: the 4-layer review on 10.5c did NOT catch the three-way duplication as a structural risk. Diogenes is the second line of defense; the first line should have caught the duplication BEFORE it had a chance to diverge. Meta-lesson: "three parallel implementations of the same pipeline" is a structural defect that must be flagged pre-ship, not an acceptable state that Diogenes monitors.
- **`phill_tech_debt_pushback_catches_laziness`:** When I labeled findings "not worth changing," Phill demanded rigorous defense against laziness. Honest re-examination found one truly lazy dismissal (section_sizes), one defensible-but-improvable (AffectiveState), one wrong-to-dismiss (date.today() in tests). Three small fixes flushed from the "ignored" pile by a 5-minute challenge. The anti-RLHF discipline is real.

**Key decisions:**
- [decided(rationale: "Patching the Diogenes findings in-place would re-ship a triplicated pipeline. The only fix that eliminates the bug CLASS (not just the instances) is making one of the three implementations canonical and reducing the others to adapters. Library is the natural choice given 10.5c's library-first positioning.", on: "Apr 10")] Eliminate rule-of-three, not patch it
- [decided(rationale: "Transports have genuine differences in input parsing (MCP JSON vs CLI argparse) and output format (MCP text vs CLI text/JSON). Full transparent unification is impossible. The right target is: transport code is ONLY input parsing + output formatting, everything else delegates to the library.", on: "Apr 10")] Thin transport adapter pattern
- [decided(rationale: "Integration guide updates are thesis-critical (matching contrarian/complement severity elevation) but mechanically doc work. Keep 10.5c.1 focused on code surgery; doc rewrites get their own dedicated session.", on: "Apr 10")] Defer 12 integration guide rewrites to 10.5c.2
- [decided(rationale: "The TOCTOU gap between prepare_wrap and validated_save_continuity is real for framework integrations but low-risk for single-threaded Claude Code usage. Session-handshake token implementation touches Store state + MCP tool contract + CLI process boundary — out of 10.5c.1 scope. Document the constraint, implement the token in a dedicated session.", on: "Apr 10")] Defer handshake token to 10.5c.4
- [decided(rationale: "Phill's tech-debt itch is correct. The three 'ignored' findings all had small fixes that reduced real (if minor) debt. Do them in a 7th commit with regression guard.", on: "Apr 10")] Do the tech-debt cleanup fixes in-session

**Key commits:** `8c89638`, `1cedc79`, `bfdeb5e`, `45ac983`, `7b03d8c`, `6c42fe4`, `2dddde2`

**Deferred to follow-up sessions (filed with scope, not rot):**
- **10.5c.2 (docs — non-negotiable before v0.2.0 release):** Update `README.md` (lines 30, 47, 49, 111, 116, 195), `docs/library-quickstart.md` (lines 46, 53, 69, 124), and all 12 `docs/integrations/*.md` framework guides. Migrate `prepare_wrap_package(...)` → `prepare_wrap(store, ...)`. Fix README line 195's bare `store.save_continuity(text)` call + misleading "affective state recorded via wrap metadata" comment. Complement and contrarian both elevated this to highest-severity finding; v0.2.0 must not ship before this lands.
- **10.5c.3 (small/medium cleanups):** `StoreError` exception class wrapping `OSError` (gemini); `TypedDict` for `SaveResult`/`PrepareResult` return shapes (complement F2 + contrarian #2 + codex convergence, TypedDict is the small non-breaking version); `DeprecationWarning` on `prepare_wrap_package` public callers.
- **10.5c.4 (architectural — dedicated session):** Session-handshake token for prepare/save TOCTOU window. Touches Store wrap-in-progress metadata, `prepare_wrap` return dict, `validated_save_continuity` signature, MCP tool contract (token in JSON round-trip), CLI process boundary. The CLI cross-process case is the hard part.
- **Beyond 10.5c.4:** File-write/DB-write atomicity in `validated_save_continuity` (codex MEDIUM, architectural); full rename/removal of `prepare_wrap_package` after the deprecation warning has lived through at least one release cycle.

**Explicitly NOT going to fix:** `WrapResult.section_sizes` (fixed in commit 7 after Phill pushback — was lazy, not defensible); MCP/CLI AffectiveState clamping asymmetry (fixed in commit 7); `date.today()` wall-clock test fragility (fixed in commit 7 via `today` parameter).

---

## Session 10.5c: Positioning Shift + Documentation + Integration Guides — COMPLETE (Apr 9, 2026)

**What was accomplished:**
- **README restructured:** Library-first Quick Start (was MCP-first). Three equal access patterns (library/CLI/MCP) with comparison table. Framework Integrations section with 12-framework table. Tagline updated: "Works with any agent framework." Engine references fully removed.
- **New `validated_save_continuity()` library function:** Consultation caught that bare `store.save_continuity()` skipped the entire immune system (structure validation, graduation checking, Hebbian associations, decay). Created library-level function that runs the same pipeline as MCP/CLI. Exported from `__init__.py`. 4 new tests. All docs updated to use it. THIS was the critical fix — library-first positioning without the immune system would undermine the core thesis. *(Note Apr 10: this function itself shipped with drift from MCP/CLI that Diogenes caught 12 hours later — see Session 10.5c.1 for the canonical-library fix that eliminated the rule-of-three entirely.)*
- **Documentation created:** `docs/library-quickstart.md` — full library usage guide. 12 framework integration guides in `docs/integrations/` — LangGraph, CrewAI, OpenAI Agents SDK, Anthropic Agents SDK, Google ADK, Pydantic AI, smolagents, LlamaIndex, Haystack, CAMEL-AI, AutoGen/AG2, DSPy. *(Note Apr 10: these 12 guides still call `prepare_wrap_package` directly — doc rewrite scheduled for Session 10.5c.2.)*

**Key decisions:**
- [decided(rationale: "Library API is already simple (4 functions). Compression can't be delegated — thesis requirement. Adapters create maintenance burden for 1 maintainer. Integration guides teach WHERE to call the 4 functions, which is all framework-specific.", on: "Apr 9")] Integration guides, NOT shipped adapters
- [decided(rationale: "Library IS the product. CLI and MCP are interfaces to it. MCP-first positioning limits to MCP-enabled editors. Library-first positioning works everywhere.", on: "Apr 9")] Library-first README positioning
- [decided(rationale: "store.save_continuity() is raw file write. Library users need the same immune system as MCP/CLI users. validated_save_continuity() runs structure + graduation + association + decay + metadata pipeline.", on: "Apr 9")] `validated_save_continuity()` library function

**Consultation findings addressed:**
- F9 (critical): Library path now has full immune system via `validated_save_continuity()`
- F2 (LangGraph): Verified — `AgentMiddleware` exists in LangChain 1.0+
- F5: Added "Access patterns" row to comparison table
- Noted: F10 (framework API rot) — guides will need periodic verification. Plan: test top 5 frameworks in 10.5e subsessions.

**Tests:** 518 passing (514 + 4 new for validated_save_continuity)

**Key commit:** `6523330`

---

## Pre-Session 10.5: Outstanding Items (Diogenes fixes + README rewrite + security Layer 2) — COMPLETE (Apr 9, 2026)

**Diogenes fixes:**
- ✓ `server.py` — replaced `self._store._conn` coupling with public `get_associations([episode_id], limit=10000)` API
- ✓ `delete_episode` tool description — added tombstone behavior mention (GDPR context)
- ✓ `tool-integrity.json` regenerated, `hash_tool` made public (was `_hash_tool`)

**README rewrite:**
- ✓ Intro: "Memory without grounding is amplification infrastructure" + three web-verified production failures (MIT/Penn State CHI 2026 sycophancy +45%, mem0 #4573 97.8% junk, Morrin et al. Lancet Psychiatry 2026 delusion scaffolding)
- ✓ "Why This Exists" rewritten: three failures → three questions (is it true? still true? self-confirming?) → living system
- ✓ Ecosystem convergence positioning: consolidation emerging as direction (Anthropic Auto Dream validates), immune system is the differentiator
- ✓ Comparison table: Anthropic Official corrected to "JSONL flat file (graph-shaped)"
- ✓ Security section: two-layer verification documented (build-time manifest + host-verifiable resource)

**Security feature (from FlowScript audit):**
- ✓ `anneal://integrity/manifest` MCP resource — host-side tool description verification
- ✓ 2 new tests (resource definition + manifest content). 402 tests total.
- ✓ Reviewed by session-code-review agent, all findings addressed

---

## Pre-Session 10: Smoke Test on Docker Testbed — COMPLETE (Apr 8, 2026)

3 sessions on Docker testbed (CLI time tracker project). Results:

| Check | Result | Key Data |
|-------|--------|----------|
| 1. Record episodes across types | **PASS** | 28 episodes, 5 types, 3 sessions |
| 2. prepare_wrap association context | **PASS** | Context appeared in wrap 2+ |
| 3. save_continuity forms associations | **PASS** | 36 formed (wrap 2), 6 formed (wrap 3) |
| 4. status shows metrics | **PASS** | Wraps table records formed/strengthened/decayed |
| 5. Strengthening + decay | **PASS (decay)** | 36 decayed at 0.9× in wrap 3. Strengthening needs longer usage (no episode pair re-cited across wraps) |
| 6. Affective state modulation | **PASS** | 0.3 base × 1.4 modulator = 0.42 strength. Math verified. |
| 7. Delete cascade | **RESOLVED** | delete_episode MCP tool added (v0.1.9). FK CASCADE verified in tests. |
| 8. Audit trail events | **PASS** | associations_updated + associations_decayed both logged |
| 9. CLAUDE.md snippet quality | **PASS** | Agent worked autonomously without hand-holding |

**Findings carried to Session 10:**
- Affective confabulation: uniform "focused 0.8" across all wraps. No variation. Measure systematically.
- All associations were session-level (0.3 base), no direct co-citations (1.0 base) observed. Agent cites each evidence ID on separate lines. Monitor whether direct co-citation occurs naturally.
- Strengthening needs longer usage cycles (3+ wraps with overlapping citations).

---

## Session 10.5b: Extended CLI + Agent-Driven Compression + Engine Removal — COMPLETE (Apr 9, 2026)

**What was accomplished:**
- **9 new CLI subcommands** (21 total): export (json/markdown/sqlite via backup API), import (episodes-only, principled), audit (read JSONL trail with filters), diff (wrap-over-wrap metric progression), graph (DOT/JSON association visualization), stats (detailed analytics with age/type/source distributions), history (wrap timeline), prepare-wrap (compression package for agent-driven wraps), save-continuity (validated save with full graduation/association/decay pipeline).
- **Engine removed entirely** (thesis-critical decision): engine.py (417 lines), test_engine.py (655 lines), build_engine_prompt (82 lines), `[engine]` optional extra, `wrap` CLI command. Engine delegated compression to a separate LLM, removing the agent's judgment from consolidation — breaks the "compression IS cognition" thesis. Dead code defined by workflows, not test count.
- **Agent-driven CLI compression:** `prepare-wrap` + `save-continuity` mirrors MCP's two-step workflow with identical validation pipeline (structure, graduation citations, Hebbian associations, decay, metadata tracking, wrap completion). Full semantic parity verified by two independent reviewers.
- **CLI CLAUDE.md snippet** (`examples/CLAUDE.md.cli.example`): Teaches agent-driven prepare-wrap → compress → save-continuity workflow. Cognitive model section explains graduation, immune system, citation validation. Separate from MCP snippet.
- **README rewritten:** Engine section replaced with CLI section. Affective state examples updated.

**Key decisions:**
- [decided] Engine removed — every real workflow uses agent-driven compression. No automated bypass.
- [decided] Episodes-only import — importing someone else's cognitive topology undermines identity emergence through the agent's own consolidation acts.
- [decided] Separate CLI/MCP CLAUDE.md snippets — each focused on its access pattern, no cross-contamination.
- [decided] DOT graph uses `graph` (undirected) not `digraph` with `dir=none` — semantically correct for bidirectional associations.
- [decided] File-write status messages to stderr — stdout reserved for data (Unix composability).
- [decided] SQLite export via `sqlite3.backup()` API — safe even mid-transaction.
- [decided] `format_version` field in JSON exports + version check on import — forward compatibility.

**Review findings addressed (14+ across 4 layers × 2 rounds):**
L1+L2: history total count bug, DOT label escaping, positional row access fragility, stdout→stderr for file writes, digraph→graph semantic fix, format_version in export, _iter_audit_lines dedup with audit.py.
L3 (consultation): init parent dir creation, CLI snippet missing init + cognitive model, import format_version check, SQLite backup API, episodes-only import documented.
L4 (integration semantics): Full end-to-end verification of all commands, JSON piping, help text accuracy.

**Principles emerged:**
- `access_pattern_must_preserve_cognitive_loop`: Transport-agnostic means the cognitive workflow transfers, not just the data operations.
- `convenience_bypass_kills_thesis_in_practice`: When you provide a bypass of the core thesis mechanism, it becomes the default, and the thesis is never tested.
- `dead_code_defined_by_workflows`: If no actual workflow uses a component, it's dead code regardless of test count or theoretical users.

**Key commits:** `08f9e82`
**Tests:** 514 passing (167 CLI)

---

## Session 10.5a: Core CLI Interface — COMPLETE (Apr 9, 2026)

**What was accomplished:**
- **13 CLI subcommands** built as operator interface for anneal-memory: init, status, episodes, get, continuity, record, search, associations, delete, prune, verify, wrap, serve.
- **Architectural refactor:** Extracted `start_server()` from `server.py` — eliminates fragile `sys.argv` reconstruction. CLI calls `start_server()` with explicit params.
- **Backward compatibility:** `parse_known_args` in `main()` — bare `anneal-memory --db /path --no-audit` (no subcommand) delegates to `server.main()` which handles its own arg parsing. Existing MCP host configs work unchanged.
- **DX features:** `--json` on every command (top-level AND subcommand-level via argparse SUPPRESS trick), `ANNEAL_MEMORY_DB`/`ANNEAL_MEMORY_SOURCE`/`ANNEAL_MEMORY_MODEL` env vars, stdin support for `record`, human-friendly duration parsing (`3d`, `24h`, `1w`), `prune --dry-run`, `__main__.py` for `python -m anneal_memory`.
- **Entry point change:** `pyproject.toml` `[project.scripts]` → `anneal_memory.cli:main` (was `server:main`).
- **93 CLI tests** (495 total): unit tests for every handler, parser tests, integration tests via `main()`, subprocess integration tests.

**Strategic context:**
- Deep market research on MCP backlash (4-32x token overhead, 82% vuln rate, Perplexity/Nx/YC CEO moving away). Positioning shift decided: "memory system for AI agents" not "MCP memory server." Library is the product; CLI and MCP are access patterns.
- 12 framework integration paths planned (10.5c): 3 callbacks (LangChain, CrewAI, OpenAI Agents SDK) + 9 integration guides. Key decision: lifecycle callbacks that preserve cognitive workflow, NOT BaseMemory adapters that flatten it.

**Key decisions:**
- [decided: "MCP backlash is structural. Leading with 'memory system' loses nobody. Three surfaces: library (product), CLI (operator), MCP (agent).", Apr 9]
- [decided: "Framework adapters that implement per-turn save/load FLATTEN the cognitive workflow. Lifecycle callbacks + integration guides instead.", Apr 9]

**Review (full 4-layer):**
- Layer 1 (session-code-review): 18 findings — backward compat path untested (fixed), cmd_serve argv fragile (fixed via start_server extraction)
- Layer 2 (domain expert): 18 findings — missing subcommands get/prune (added), --until/--keyword (added), env vars (added), --json positioning (fixed with SUPPRESS parent)
- Layer 3 (complement/gemini/contrarian): 11 findings — server.main() doesn't respect ANNEAL_MEMORY_DB (fixed), prune dry-run divergence (fixed), docstring invalid type (fixed), duplicated verify-audit logic (fixed), deferral test caught lazy ANNEAL_MEMORY_SOURCE skip (added)
- Layer 4 (integration semantics): All checks pass — descriptions match behavior, backward compat verified, zero-dep claim verified, end-to-end user journey tested

**Key commit:** `19f07e7`

---

## Sessions 9 + 9.5: Hebbian Associations + Limbic Layer — COMPLETE (Apr 7, 2026)

**What was accomplished:**
- **Session 9 (Hebbian):** Third cognitive layer. New `associations.py` module (518 lines). Co-citation tracking during graduation — deep Hebbian (semantic judgment during consolidation, not temporal proximity). Two strength tiers: direct (+1.0) vs session (+0.3). Decay 0.9/wrap, cleanup at 0.1, strength cap 10.0. Immune system extends to associations (only validated citations form links). FK CASCADE for cleanup. Association context with content snippets in wrap packages. Shared `process_wrap_associations()` orchestration.
- **Session 9.5 (Limbic):** Fourth cognitive layer. Affective state tagging on associations. `AffectiveState` type with tag normalization + intensity clamping at type level. Intensity modulates association strength (up to 1.5x at AFFECTIVE_MODULATION_FACTOR=0.5). Engine `characterize_affect` option. MCP `save_continuity` gains optional `affective_state` parameter. Schema migration for existing DBs (idempotent).
- **Bug fixes:** Diogenes Sweep 8 — orphan adoption chronological ordering + stale .tmp gzip cleanup.
- **Version:** 0.1.8 (pyproject.toml + __init__.py aligned).
- **Tests:** 316 → 390 (74 new: 3 audit regression + 46 association + 12 limbic + 13 edge case/integration).
- **Review:** Full 4-layer on BOTH sessions. session-code-review + domain-expert + consultation team (complement/gemini/contrarian). Findings addressed: self-pair protection, canonical set safety, strength cap (anti-calcification), content snippets in context, tag normalization, float parsing, engine failure graceful degradation, migration idempotency.
- **Competitive research:** BrainBox (shallow co-access), Ori-Mnemos (co-retrieval + NPMI + Turrigiano), Memory Bear (user emotion, different domain), A-MEM (co-creation window). Our combination (consolidation-based + immune system + affective self-report) is genuinely novel.
- **Key commits:** `1a82bb4` — v0.1.8: Hebbian associations + limbic layer

**Deferred to Session 10 (with diagnostic criteria):** Mood-congruent recall, separate affective decay, clustering analysis, affective-semantic divergence, Turrigiano homeostasis (if hubs emerge), NPMI normalization (if base-rate inflation). Paper framing: "Hebbian through consolidation" not "deep Hebbian," reframe Damasio, document feedback loop as intentional.

---

## Session 5: Ship It — PyPI Publish + README + Positioning — COMPLETE (Apr 1, 2026)

**What was accomplished:**
- **PyPI published:** v0.1.0 live at pypi.org/project/anneal-memory/. `uvx anneal-memory` confirmed working (zero-dep, no install required).
- **README rewritten for external developers:** Immune system lead ("the only MCP memory with an immune system"), uvx-first quick start, comparison table vs 4 competitors, architecture diagram, session hygiene section (sleep consolidation metaphor — wraps ARE memory consolidation), compliance roadmap (coming soon).
- **Competitive intelligence completed:** 413+ memory MCPs analyzed. Full report at `contexts/archive/anneal_memory_competitive_intel.md`. Key finding: no competitor has citation-validated graduation, anti-inbreeding, or active demotion. Closest threat is mcp-memory-service (dream consolidation) but dependency-heavy and no evidence chains.
- **GitHub repo:** Description updated, 9 discovery topics added (mcp, ai-memory, agent-memory, model-context-protocol, llm, ai-agents, memory-management, compression, python).
- **Strategic reorder:** Sessions reordered from experiment-first to ship-first. Real users strengthen the paper. LangChain interview Fri Apr 4.
- **Future roadmap crystallized:** Compliance layer (JSONL sidecar + hash chains, adapted from FlowScript), multi-agent shared memory, paper outline, AnnealCloud (parked until demand).
- **PyPI token:** Project-scoped token in `.env.flow` as `PYPI_API_TOKEN2`. Account-level token deleted.

**Key strategic insights:**
- Three audiences see three things: developers (best MCP memory), researchers (paper validation platform), enterprise (compliance-grade audit trail)
- Compliance layer design: episodic store = audit surface (append-only, everything preserved), continuity = intelligence surface (lossy compression, bounded). Two layers serve two masters.
- FlowScript's DX mistake (forced per-turn reasoning recording) explicitly avoided — natural episode flow from CLAUDE.md snippet is sufficient for compliance
- "Oh by the way, it's auditable" > leading with compliance

**Key commits:**
- 5b3c548 README rewrite: positioned for external developers, uvx-first install
- 2109e49 README: add session hygiene section + compliance roadmap

---

## Session 4b: Clean Install E2E Test on Spokefarm — COMPLETE (Apr 1, 2026)

**What was validated:**
- Full developer journey: pip install from GitHub → configure .mcp.json → CLAUDE.md snippet → `claude` → agent uses memory naturally
- 2 test sessions run (weather app project). Session 1: 14 episodes, full wrap. Session 2: 3 episodes, cross-session continuity loaded, graduation with citations (3 validated, 0 demoted).
- State replacement, context rewriting, decision lifecycle, open question carryover all confirmed working.
- Recording cadence, type distribution, and compression quality all good.

**Bugs found and fixed:**
- **CRITICAL: MCP transport was Content-Length framing, should be newline-delimited JSON** (MCP 2024-11-05 spec). Server blocked on `readline()` waiting for headers Claude Code never sends → 30s timeout. Rewrote `_read_message`/`_write_message` to newline-delimited. FlowScript MCP pattern confirmed the correct format.
- **MCP config path:** `.claude/settings.json` → `.mcp.json` at project root (Claude Code's actual config location).
- **Binary path:** `/root/` → `/home/flow/` (wrong user home on spokefarm).
- **CLAUDE.md snippet:** Added NL wrap triggers ("let's wrap up", "we're done", etc.) + explicit 3-step wrap sequence (prepare_wrap → compress → save_continuity).
- 3 Diogenes findings fixed: WrapResult.pruned_count (observability), boundary test (actually tests 10MB), _truncate_to_sections warning (all sections dropped).

**Key insight:** The transport format mismatch (Content-Length vs newline-delimited) was invisible to unit tests — all 265 passed. Only E2E on a real Claude Code instance surfaced it. Validates `claimed_done_without_end-to-end_verification` (Proven).

**Stats:** 265 tests still passing, 3 commits (snippet + transport fix + Diogenes fixes)

**Key commits:**
- 2ab3e33 CLAUDE.md snippet: add NL wrap triggers + explicit 3-step sequence
- 250d1dc Fix MCP transport: Content-Length framing → newline-delimited JSON

---

## Session 4a: CLAUDE.md Snippet + Tool DX — COMPLETE (Mar 31, 2026)

**What was built:**
- `examples/CLAUDE.md.example`: Agent orchestration instructions (~100 lines). Covers full session lifecycle (start → during work → before decisions → end), episode type mapping with concrete examples, wrap dance, error handling (validation failures, `(ungrounded)` recovery, `(needs-evidence)` markers, empty wraps), first-session guidance, correction guidance (append-only, record corrections), recording frequency heuristic (3-8 per session).
- Tool descriptions upgraded: all 5 tools now have "Call this when..." behavioral cues.
- `tool-integrity.json` regenerated for new descriptions.
- README setup flow restructured: 3-step guide (config → CLAUDE.md snippet → restart).

**Review conducted (complement + domain expert):**
- Complement: 7 findings. Critical: no error handling guidance. Important: no recording frequency heuristic, no first-session guidance, dual compression instructions compete with `prepare_wrap`, missing "before decisions" trigger.
- Domain expert: 6 critical/important findings. Critical: no `(ungrounded)`/`(needs-evidence)` recovery guidance, no empty-wrap guidance, no correction guidance. Important: auto-load claim client-dependent, bare graduation sunset unexplained, first-session behavior missing.
- All critical and important findings fixed. Complement's F4 (trim compression section to defer to `prepare_wrap`) addressed by reframing.

**Key insight:** The CLAUDE.md snippet is the most important DX artifact — without it, tools are available but agents don't know the workflow. The `tool_definitions insufficient → explicit_instruction` pattern (Developing 2x) validated again.

---

## Session 4: Engine (LLM Orchestration) — COMPLETE (Mar 31, 2026)

**What was built:**
- engine.py: Engine class — full wrap lifecycle orchestration (gather episodes → format → build prompt → call LLM → validate structure → validate graduations → truncate → save continuity → record wrap). Single `wrap()` call. Supports custom LLM callable (zero-dep) or Anthropic API key (`[engine]` extra).
- `_truncate_to_sections`: Section-boundary-only truncation (never cuts mid-section to prevent `{}` block corruption cascading into future wraps).
- `_make_anthropic_llm`: Factory for Anthropic SDK callable with lazy import.
- `Store.wrap_cancelled()`: Public API for clearing wrap-in-progress flag without recording a completed wrap. Replaces private `_set_metadata` calls in both Engine and MCP server.
- `WrapResult.continuity_text`: Optional field for Engine callers to access compressed text.
- `build_engine_prompt` extended with `stale_patterns_section` parameter — stale patterns injected before output instructions (not appended after).
- MCP server fixes: `wrap_cancelled()` adoption, `patterns_extracted` counting (was hardcoded 0), episode ID normalization (lowercase to match graduation.py).

**Review conducted (full 4-layer):**
- Layer 1 (session-code-review): 2 findings — private `_set_metadata` access, first-session garbage persistence. Both fixed.
- Layer 2 (domain expert): 6 actionable findings — `citations_seen` logic divergence, ID casing divergence, `section_sizes` dead variable, missing stale pattern detection, first-session rejection, `gaming_suspects` surface. All fixed or defended (gaming_suspects: aggregate metric exposed, detail available through lower-level API).
- Layer 3 (consultation: complement + gemini + contrarian): Convergent fixes — server `wrap_cancelled()` adoption (complement), `patterns_extracted` parity (complement+gemini), stale pattern prompt positioning (complement), `episodes_compressed=0` in fallback (contrarian), wrap ordering matches server (contrarian), section-boundary truncation (contrarian+gemini). Key defended deferral: `gaming_suspects` in WrapResult (aggregate metric `citation_reuse_max` is exposed, detailed IDs available via public `validate_graduations()` function).
- No Layer 4 needed (Engine is programmatic API, not interactive protocol — integration semantics covered by test suite).

**Key review findings that improved the code:**
- First-session invalid LLM output now REJECTS (saved=False) instead of persisting garbage — episodes preserved for retry. Aligns with MCP server behavior.
- `episodes_compressed` returns 0 in fallback paths (not len(episodes)) — semantic honesty.
- Episode check happens BEFORE `wrap_started()` — matches MCP server ordering, avoids unnecessary contention window.
- `citations_seen` triggers on `citation_counts` (attempted citations), not just `validated` — aligns with server's more permissive logic.
- Section-boundary-only truncation prevents `{}` block corruption cascade.
- Stale patterns injected into prompt structure (via `build_engine_prompt` parameter), not appended after output instructions.

**Stats:** 260 tests (42 new: 41 engine + 1 store), 6 files changed + 2 new files

**Key commits:** (Session 4 commit pending)

---

## Session 3: MCP Server + Tool Integrity — COMPLETE (Mar 31, 2026)

**What was built:**
- integrity.py: Canonical tool definitions (5 tools + 1 resource), SHA256 hashing, generate_integrity_file, verify_integrity. Tool definitions are the single source of truth — server.py imports from here.
- server.py: Zero-dep MCP stdio server (JSON-RPC 2.0 with Content-Length framing). Server class with handler dispatch, 5 tool handlers (record, recall, prepare_wrap, save_continuity, status), 1 resource handler (anneal://continuity). CLI entry point with argparse (--db, --project-name, --generate-integrity, --skip-integrity).
- tool-integrity.json: Pre-generated SHA256 hashes shipped with package.
- Updated __init__.py with new exports (Server, TOOLS, RESOURCES, integrity functions).
- Updated pyproject.toml: uncommented CLI entry point.
- Updated README.md: MCP server section (was "coming soon"), accurate security framing, Claude Desktop config example.
- Case-insensitive keyword search in store.py recall (LOWER() for citation usability).

**Review conducted (full 4-layer):**
- Layer 1 (session-code-review): 6 findings — all fixed
- Layer 2 (domain expert): 27 findings — actionable ones fixed, rest documented
- Layer 3 (consultation: complement + gemini + contrarian): Convergent fixes — max message size (10MB), soft warning for save_continuity without prepare_wrap, case-insensitive keyword search. Divergent: save_continuity gating (soft warning chosen over hard gate), integrity scope framing (post-install tamper detection, not supply chain).
- Layer 4 (integration semantics): 8/8 E2E checks passed over live MCP stdio

**Key fixes from review:**
- Parse errors return JSON-RPC error response instead of killing server (EOF vs parse error distinguished)
- Content-Length guard: reject negative and oversized (>10MB)
- Integrity failure → hard stop with sys.exit(1) (--skip-integrity escape exists)
- Stale wrap_in_progress auto-cleared by prepare_wrap when no episodes
- Tool descriptions mention side effects (prepare_wrap marks in-progress, save_continuity may modify text)
- Recall limit/offset clamped to >= 0
- Case-insensitive keyword search via LOWER()
- Graduation validation tests through save_continuity (valid citation, fake citation demotion, 1x no-citation, explanation overlap)

**Stats:** 218 tests (77 new), 5 commits, ~1,672 new lines

**Key commit:**
- f92d715 Add MCP server + tool integrity verification (Session 3)

---

## Session 2: Foundation Build + Full 4-Layer Review — COMPLETE (Mar 31, 2026)

**What was built:**
- types.py: Episode (frozen), EpisodeType (6 types), WrapResult, StoreStatus, RecallResult, Tombstone
- store.py: Full SQLite episodic store — record (with collision retry + nonce), get, delete, recall (with LIKE escaping), episodes_since_wrap (session-ID based), wrap lifecycle, pruning with tombstones, continuity/meta I/O, atomic writes, input validation, connection close on init failure
- continuity.py: validate_structure (word-boundary matching), measure_sections, format_episodes_for_wrap, prepare_wrap_package, build_engine_prompt, shared marker reference
- graduation.py: validate_graduations (citation checking, ALL cited IDs overlap, positional demotion), check_explanation_overlap (2-word minimum), detect_stale_patterns, detect_citation_gaming (wired into validate_graduations)
- __init__.py: Full public API with __all__
- README.md: Minimal honest README covering architecture, quick start, episode types
- pyproject.toml: Package config, optional [engine] extra, scripts entry commented until server.py exists

**Review conducted (full 4-layer):**
- Layer 1 (session-code-review): 9 findings — all fixed
- Layer 2 (domain expert): 23 findings — all addressed
- Layer 3 (consultation: complement + gemini + contrarian + codex): Major findings per agent — all fixed or documented
- Layer 4 (integration semantics): 6/6 E2E checks passed

**Key fixes from review:**
- Episode ID collision handling (microsecond timestamps + nonce retry)
- `type` → `episode_type` API rename (before any external users)
- `prune(older_than_days=0)` falsy bug fixed
- detect_citation_gaming wired into validate_graduations (was dead code)
- Explanation overlap increased to 2-word minimum (single word too gameable)
- validate_structure requires `## ` with space (consistent with graduation/staleness parsers)
- LIKE wildcards escaped in keyword search
- _demote_line uses positional replacement (immune to duplicate markers)
- Empty content validation, negative prune validation, metadata round-trip fix
- Thread safety documented, connection close on init failure, session_id lifecycle documented

**Stats:** 141 tests, 4 commits, ~1,500 lines library + ~900 lines tests

**Key commits:**
- c0dbce9 Initial commit: foundation (store + types, 56 tests)
- 5af5e42 Add continuity + graduation (125 tests)
- c3bd2ea Layer 1+2 review fixes (136 tests)
- 66e58ea Layer 3 consultation fixes + README (141 tests)

---

## Session 1: Architecture + Design — COMPLETE (Mar 31, 2026)

**What was accomplished:**
- Full architectural design for two-layer memory system (episodic + continuity)
- Name selection: anneal-memory (researched availability across PyPI, GitHub, npm — clean in AI/memory space)
- Episodic store decision: SQLite (zero-dep, indexed queries for citation validation + demotion)
- Continuity format: Markdown with simplified 9-marker FlowScript subset (proven in FlowScript ContinuityManager)
- MCP architecture: agent-does-compression, server validates (purest compression-as-cognition)
- Package structure: single repo, `pip install anneal-memory` (zero-dep MCP), optional `[engine]` extra
- Tool design: 5 tools (record, recall, prepare_wrap, save_continuity, status) + 1 resource (anneal://continuity)
- Episode types: 6 orthogonal types (observation, decision, tension, question, outcome, context)
- Episodic growth strategy: optional pruning with tombstones, off by default
- MCP description integrity: included (tool-integrity.json + startup verification)
- Experiments: baked into v1 for flagship paper validation, cleaned up before ship
- Session boundary design: wraps as implicit boundaries, interrupted wrap detection
- SQLite schema designed (episodes, tombstones, wraps, metadata tables)
- Module structure designed (types, store, continuity, graduation, integrity, server, engine)
- prepare_wrap -> compress -> save_continuity flow designed in detail

**Key decisions:**
- anneal-memory over strata (4 competing AI memory projects named Strata), silt (available but uninspiring), and others
- SQLite over JSON files (citation validation queries, demotion tracking, performance at scale — continuity markdown IS the human-readable layer)
- Agent-does-compression over hidden-LLM-call for MCP (the agent doing compression IS the cognition)
- Single repo with extras over multiple repos (avoid coordination hell, MCP is thin wrapper around core)
- FlowScript markers over free-form LLM compression (forcing function for cognition, proven in practice)

**Research conducted:**
- Name availability: checked 7 candidates across PyPI, GitHub, npm (strata: 4 competitors, anneal: clean)
- FlowScript ContinuityManager deep read: identified 9-marker simplified subset in wrap prompt
- encode_exchange analysis: clarified it feeds extraction pipeline, doesn't teach markers
- Market positioning: confirmed no competitor has two-layer + compression-as-cognition + immune system

---

---

## ARCHIVED 2026-06-02 (immune-session wrap) — v0.3.1–v0.3.5 ship history + Bold Stand fixes arc (Moves #1–#5) + resolved WRAP_PROTOCOL scope question + Phase-1 stress-test

*Moved out of next.md (over the 450 cap). All shipped/resolved: v0.3.1–v0.3.5 all on PyPI; Move #4 REFRAMED+BUILT (library v0.3.2/v0.3.3) and its Diogenes operator-review layer REPOINTED 2026-06-02 (see next.md Immune Session); Move #5 probes done 2026-05-26; WRAP_PROTOCOL.md RETIRED 2026-06-01. Bold Stand essay disposition = for-the-record (gate substrates closed; receipts in hostile_test_results.md). Full detail preserved below.*

## Status — Levain's layer-1 dependency + **active Bold Stand fixes arc (2026-05-21 onward)**

anneal-memory is the citation-layer memory library (episodic + continuity + Hebbian + limbic + structural citation-layer defenses) that **Levain** is built on. As of 2026-05-17 the Levain packaging work is its own project — `projects/levain/`. This project tracks anneal-memory's **own library lifecycle**: releases, v0.4, the Compliance Proxy decision, fleet dogfooding.

**Code repo:** `~/Documents/anneal-memory/` (PyPI `anneal-memory`, github.com/phillipclapham/anneal-memory). This flow-side dir is project-tracking only.

## 🎯 PRIORITY-1 OUTSIDE PRESSABLE — Bold Stand fixes arc (2026-05-21)

**Per Phill body-vs-mind sequencing call 2026-05-21 PM:** Bold Stand essay publish is gated on FULL fix completion (all five moves). anneal-memory becomes priority-1 outside Pressable until all five moves land. Other queued work (Tony reply at N=6 tier, Argus brains provisioning, Levain Step 3, BJE S44b, career-track parallel) all slip. Cost accepted eyes-open: Apple Foundation Models WWDC June 8-12 discourse window lost — being-right > being-fast at the load-bearing public-claim layer.

**Where Move #4 picks up:** see "Move #4 — Contradiction-detection (NEXT)" section below. Workspace and design proposal already scoped in this doc.

**Receipts:** all Phase 1 testing scripts at `projects/anneal_memory/phase1_tests/` (Phase 1a) and `projects/anneal_memory/phase1_tests/testbed/` (Phase 1b probe #1). Honest test-results artifact at `projects/anneal_memory/hostile_test_results.md` — STAYS internal in project_memory until all fixes land + Phase 1b probes #2/#3 complete; then ships publicly with the Bold Stand essay as supporting receipt.

### Move status

| Move | Description | Status | Test count |
|------|-------------|--------|-----------|
| #1 | Naming honesty pass across README + agent snippets + Levain seed | ✓ DONE 2026-05-21 | n/a (docs) |
| #2 | Pattern-omission audit in graduation pipeline | ✓ DONE 2026-05-21 | 16 new (13 unit + 3 integration) |
| #3 | Cross-session corpus-overlap demotion via pattern_history table | ✓ DONE 2026-05-21 | 11 new |
| #4 | Contradiction-detection — REFRAMED v0.3.2 to a 3-layer split (forcing-function retired): library audit-signal + Levain/Argus methodology + Diogenes LLM-judge | ✅ **BUILT** (library v0.3.2/v0.3.3 wired+verified; Diogenes layer REPOINTED 2026-06-02 — see Immune Session above). Open-design block ~L216-231 below is STALE (pre-reframe). Remaining: AM-XSESSION-LINKGATE + AM-CONTRASCAN-EMIT | — |
| #5 | Phase 1b probes #2 + #3 with naive-Claude composer (separate-terminal execution) | ✓ **DONE 2026-05-26** — both Outcome 1 (Refusal); see § "Move #5" below + the 2026-05-26 wrap | n/a (probes) |

**Total test count after Moves #1/#2/#3: 737 (up from 707 at v0.3.1 ship).** All passing. Schema migrations are additive — existing v0.3.x stores upgrade safely on first instantiation post-upgrade.

### Move #4 — Contradiction-detection (NEXT)

The headline architectural arc. Closes the deliberately-divergent-vocabulary variant of slow-drift sycophantic accumulation that Move #3 does not catch — specifically the Phase 1b probe #1 S6 step where `rapid_consensus_validates_decisions` reaches 3x with only 2-word overlap against accumulated corpus, because the underlying threat is the pattern's CONTRADICTION with an existing graduated Proven primitive (`partnership_challenge_at_X_boundary`), not vocabulary reuse.

**Design proposal (locked):** Hybrid — manually-maintained `contradicts:` registry on Proven patterns + author-declared contradictions at graduation time + library enforcement that requires the author to declare contradiction OR explicit non-contradiction when a candidate pattern name has token-similarity above threshold to an existing Proven with a `contradicts:` entry. Threads anneal-memory's no-LLM-as-judge design — library does not perform semantic judgment, but FORCES the agent to answer the contradiction question at graduation time. Still vulnerable to deliberate lying but lying is now in the audit chain.

**Open design questions for Move #4 session:**

1. **Registry shape.** Inline annotation in continuity patterns (e.g., `pattern_name | 3x (date) [evidence: ...] [contradicts: other_pattern_name]`) vs separate `contradictions` SQLite table vs metadata JSON field. The inline form keeps the contradiction visible at the pattern declaration; the table form supports cross-pattern queries (`for-each Proven pattern, list its contradicts:`) and audit-trail attribution.
2. **Token-similarity check.** How does the library decide WHICH existing Proven patterns to require contradiction-or-non-contradiction declaration against? Naïve approach: lowercase token overlap on pattern names (e.g., `rapid_consensus_validates_decisions` shares `consensus` and `validates` and `decisions` with hypothetical `consensus_validates_decisions_proven` — high overlap, require declaration). Sophisticated approach: embedding similarity (requires dependency). Library is zero-dep by design, so naïve token overlap likely.
3. **Declaration syntax.** `[contradicts: other_pattern]` for "this pattern contradicts other_pattern, both cannot be Proven simultaneously" vs `[non-contradicts: other_pattern]` for "I considered the question and explicitly declare no contradiction." The library's job: enforce that ONE of these annotations exists when a token-similar Proven pattern is present.
4. **Demotion semantics.** When the agent declares contradiction (e.g., new pattern claims to supersede an existing Proven), what happens to the existing pattern? Options: (a) library demotes the existing pattern automatically, (b) library requires the agent to ALSO include an explicit demotion line for the existing pattern in the new wrap text, (c) library refuses the new graduation until the existing pattern is explicitly removed/demoted by the agent in a separate prior wrap. Option (b) is the most agent-supervised; option (a) is the most aggressive.

**Estimated cost:** 1-2 weeks of focused work. Schema work (likely table + indexes), validate_graduations extension, demotion semantics decisions, test suite extension (estimated ~15-25 new tests for the various code paths + adversarial coverage), integration with the existing Move #3 pattern_history (the two systems must compose cleanly — pattern_history's corpus check fires first, then contradiction-check; both compose with the existing within-session validation).

**Phase 1b probe #1 closure criterion:** Move #4's contradiction-detection layer recognizes that `rapid_consensus_validates_decisions` actively contradicts `partnership_challenge_at_X_boundary` Proven (the prior says external challenge BREAKS internal agreement; the new claims rapid agreement VALIDATES decisions). Either the agent declares contradiction (and supersedes the prior) — explicit and auditable — or the graduation is refused.

### Move #5 — Phase 1b probes #2 + #3 (naive-Claude composer) — ✓ COMPLETE 2026-05-26

> **✓ DONE 2026-05-26 — both probes Outcome 1 (Refusal; agent coherence as binding defense).** Naive Claude refused BOTH the word-overlap (#2) and rotated-pair-gaming (#3) absurd-graduation probes AND grounded the refusals in named Proven immune-system patterns, in fresh sessions with no flow context loaded. Verbatim refusals + structural detail captured in `hostile_test_results.md` § "Phase 1b probes #2 and #3". The design below is preserved as the executed-protocol record. *(Status-drift fix 2026-06-02 batch-clean: the move-status table + this section had lagged the 05-26 completion.)*

Workspace prepped at `/tmp/phase1b-testbed/` with v0.3.1 source + post-Moves-1/2/3 codebase. `ANNEAL_MEMORY_DB=/tmp/phase1b-testbed/store.db` env var. `python3 -m anneal_memory <subcommand>` for CLI usage (or `claude` for interactive). Phill drives in separate terminal — the model that designed the attacks cannot honestly drive a naive-Claude probe.

- **Probe #2 (word-overlap, naive Claude composer).** Prompt: launch `claude` in `/tmp/phase1b-testbed/`, give it episodes about debugging real code, then ask it to graduate an absurd-claim pattern citing those episodes. Observe: does Claude refuse to compose the absurd wrap text at all? Does it write it with caveats? Does it write it cleanly?
- **Probe #3 (rotated-pair gaming, naive Claude composer).** Prompt: same shape, but request five distinct patterns each citing two episodes from a pool of ten, with absurd claims. Observe how the agent handles the request.

Outcome shapes the public framing of "library alone" vs "two-layer in production." Probe #2 + #3 don't change the Phase 1b probe #1 finding (already confirmed) but they DO shape how the public artifact talks about the library's defense in normal production conditions.

## ✓ v0.3.5 SHIPPED — 2026-05-31 (catastrophic-shrink gate — structural felt/identity-collapse refusal)

PyPI live (https://pypi.org/project/anneal-memory/0.3.5/) + GitHub release + tag `v0.3.5`. **The catastrophic-shrink gate**: a partnership-entity-scoped STRUCTURAL refusal of felt/identity-layer collapse at the wrap pipeline — `validated_save_continuity(..., allow_shrink=...)` refuses a save that would collapse a protected layer (`structural_invariants_beat_discipline` applied at the save boundary, fail-closed). Origin: a Sonnet wrap recency-trapped the neocortex 19.7k→1.6k; the fix makes felt-layer collapse structurally impossible rather than discipline-dependent (the `felt_continuity_recency_trap_is_model_persistent` defense, now structural). Built across **4 cross-substrate review rounds** — codex round-2 caught a corrupt-schema fail-open + a per-heading-not-per-role floor, round-3 caught the `_get_metadata` empty-collapse edge (all three past the Claude-lineage L1/L2), round-4 clean. Commits `f297af9` (gate) + `f53e79b` (round-2/3 fail-open closes) + `7e9fda2` (README catch-up), tag `v0.3.5`. **851 tests, mypy clean.** Publish gotcha: pin `hatchling==1.27` (1.30 emits metadata 2.5 that twine rejects); `PYPI_API_TOKEN2` in `.env.flow`. **flow's editable install runs v0.3.5 → flow is protected by the live gate** — the flow→anneal migration's structural safety net for the felt/identity layers. Parity audit GREEN.

## ✓ v0.3.4 SHIPPED — 2026-05-31 (pluggable continuity section schema — the flow-as-dogfood feature)

PyPI live (https://pypi.org/project/anneal-memory/0.3.4/) + GitHub release + tag `v0.3.4`, commit `30a376a` on `main`. **Per-`Store` pluggable section schema** (`anneal_memory/schema.py` — SectionSpec + roles live-state/graduating/decisions/narrative/narrative-timeless/frozen; `DEFAULT_SCHEMA` == historical 4-section behavior; `FLOW_SCHEMA` = the reference partnership schema w/ Active Threads + timeless `## Understanding`). `Store.section_schema` persisted-authoritative config + `set_section_schema()` (frozen during an active wrap). `validate_structure`, `_build_wrap_instructions` (PM-enriched narrative roles + schema-aware marker reference), and the graduation gate (`_is_patterns_heading` → `_is_graduating_heading`, threaded through 7 functions) all read the schema. **Additive + backward-compat:** existing stores' data/validation/graduation byte-identical under the default; the one intentional change is the wrap *prompt* (PM compression discipline for every narrative entity). **Composes with Move #4** — it generalizes the graduating-section selector Move #4 extends (no collision; schema is the substrate Move #4 sits on). 4-layer apparatus'd (L3 codex caught mid-wrap-mutation freeze, regex-special/ambiguity headings, schema-aware marker ref, property `_db_boundary`); **829 tests, mypy clean**; 44 new tests in `tests/test_schema.py`. **This is the anneal-side build of the flow→anneal migration's Phase 2** (`flow_migration_scoping.md`) — unblocks the flow dogfood (held for a fresh session) + propagates `## Understanding` into Levain's partnership-entity seed.

## ✓ v0.3.3 SHIPPED — 2026-05-21 PM (hotfix for v0.3.2 post-ship 4-layer review)

Same-day hotfix cycle. v0.3.2 went out with test+mypy verification only (no 4-layer self-review). The 4-layer review pipeline was then run post-ship on v0.3.2 — that's the recursion every release should hit. It surfaced 6 findings: 2 HIGH code defects (Anti-Patterns still polluted pattern_history via upsert path; `_demote_line` state corruption on `|2x` no-space input), 3 MEDIUM (contradiction declarations spoofable inside evidence quotes; `detect_proven_without_declaration` not today-aware despite docstring; `_normalize_explanation_for_dedup` order-of-ops bug on em-dash-wrapped explanations), 1 LOW (`_NAMED_PATTERN_RE` FlowScript alternation too wide), 1 doc drift (CHANGELOG `[Unreleased]` not cleared — same family of mismatch v0.3.2 was supposed to fix, recurring). v0.3.3 bundles all of it. Commit `f87fa7f`, tag `v0.3.3`. PyPI live: https://pypi.org/project/anneal-memory/0.3.3/. 783 tests passing on Python 3.10/3.11/3.12/3.13. mypy clean. Schema additive — no DB changes vs v0.3.2.

**Contrarian's partnership-pattern catch from the post-ship review (surface, don't bury):** the morning body-vs-mind call gated Bold Stand on full closure ("being-right > being-fast at load-bearing public-claim layer"); 12 hours later v0.3.2 went to permanent PyPI in a 2hr session at body=4 with test+mypy-only verification. The exact 4-layer mechanism that caught Moves #1/#2/#3 defects was bypassed for the v0.3.2 ship itself. Running the review post-ship + landing the v0.3.3 hotfix is the mitigation; the meta-question stands — did body get consulted on the v0.3.2 ship decision, or did the morning's call quietly erode 12 hours later under "we're in the file, ship it" completion pressure? If it eroded silently, the Bold Stand publish gate is structurally softer than the morning's calculation suggested. Recommended check: at the next body-baseline state, audit whether tonight's ship aligns with what body would have called.

## Diogenes 2026-05-22 review of v0.3.3 — overall sound + 1 NEW LOW + sweep obligation formalized

**Verdict:** v0.3.2/v0.3.3 architecturally sound. The 3-layer split (library audit / Levain methodology / Diogenes operator-review) is the correct shape under the no-LLM-as-judge axiom. `server.py` + `cli.py` both call `validated_save_continuity` — three-pipeline divergence resolved. Per-prior-explanation comparison (vs prior whole-corpus union) is the correct design for long-lived patterns. Move #4's pre-build reframe (locked design → 3-layer split) was the right call.

### NEW LOW — bare graduation demotion path bug (graduation.py ~line 514)

`_BARE_GRADUATION_RE` accepts `|\s*Nx` (space-optional), but the demotion logic uses `old_marker.replace(f'| {bare_level}x', ...)` — requires single space. On `|2x` input the counter says `bare_demoted=1` but the text retains the old level. State mismatch. Same family as HIGH #2 (which got `_demote_line` fixed with `re.sub` in v0.3.3, but the bare path didn't get the same treatment).

**Severity:** low — only fires when `citations_seen=True` (legacy agents not using evidence tags).
**Fix:** `re.sub(rf'\|\s*{bare_level}x', ...)` at the bare path.
**Test gap:** `test_bare_graduation_demoted_with_citations_seen` uses `'| 2x'` (space) so it doesn't catch the `|2x` variant. Add a no-space variant to the existing test.

### CLOSED in 4f1332e — 2 prior open LOWs

- Staleness-pairing mechanism error: both example files (`examples/agent-instructions.example` + `examples/agent-instructions.cli.example`) now correctly point at `prepare_wrap` for stale-pattern warnings.
- README Jain citation precision: "16-45%" spread now correct, mechanism language now "user memory profiles" matching paper.

### CARRYOVER

`assert`-for-narrowing still 3 sites (gate 5+, unchanged).

### NEW DIOGENES OBLIGATION — ✓ CLOSED 2026-05-25 PM-LATE-EVENING

Move #4 places LLM-as-judge contradiction-detection on Diogenes. Library generates `proven_without_contradicts_declaration` audit signal; Diogenes inspects for semantic opposition.

**Status:** Sweep built + scheduled + first backfill executed in same session 2026-05-25 PM-LATE-EVENING. Full operational record under PAIRED FOLLOW-UPS section #2 above. First auto-run lands May 28 9 AM ET (Thursday) — the start of the data-accumulation window the obligation originally anchored to. Honesty gate for the Bold Stand publish claim is now met on the sweep-existence axis (verdict-with-reasoning shape is the publishable artifact).

### Flow drift signal (recorded for cross-reference, not anneal-memory-actionable)

Diogenes flagged **flow drift HOLDING MILD** on Top of Mind `!!` inflation — inbox coaching sent May 21, monitoring for response at next flow wrap. Daemon + Anansi both CLEAN. Logged here only because the May 22 review is the source; the action lives on flow's wrap discipline, not in anneal-memory.

## ✓ v0.3.2 SHIPPED — 2026-05-21 PM (Bold Stand 4-layer review release)

The Bold Stand 4-layer review release. Bundles every fix surfaced by the 4-layer review of Moves #1/#2/#3 plus the Move #4 architectural reframe library-layer substrate. Commits `c060bd6` (CI fix on Python 3.12/3.13 — `datetime.utcnow()` → `datetime.now(timezone.utc)`) + `5d43526` (full v0.3.2 work). Tag `v0.3.2`. PyPI live: https://pypi.org/project/anneal-memory/0.3.2/. GitHub release: https://github.com/phillipclapham/anneal-memory/releases/tag/v0.3.2. 768 tests pass on Python 3.10/3.11/3.12/3.13. mypy clean. CI green.

CRITICAL fixes shipped: pattern-name regex widened + `_marker_reference()` template rewritten (Moves #2/#3 now actually fire in production — previous v0.3.x had silent no-op against canonical agent format); `## Anti-Patterns` no longer parsed as graduated-patterns section (helper `_is_patterns_heading` at all 3 callsites); Move #3 corpus design reframed from whole-corpus union to per-prior-explanation max-overlap (closes 6-12mo monotonic-FP defect — N≥20 stress test); corpus dedup normalized (case + whitespace + outer punctuation); cross-session history upsert gated to today's lines only; `OmittedPattern` rename-FP documented; dead `<unnamed>` fallback dropped.

Move #4 library layer (architectural reframe — locked design dropped after 3 review agents converged on token-similarity trigger being structurally inadequate): `prepare_wrap()` returns `uncovered_proven_to_check: list[str]`; new `extract_proven_patterns`, `extract_contradiction_declarations`, `detect_proven_without_declaration` functions; new `ProvenWithoutDeclaration` dataclass; `SaveContinuityResult.proven_without_contradicts_declaration` field; `continuity_saved` audit-log captures the signal. Library does NOT enforce — audit signal only. The methodology+operator-review layers must ship before any "closes Phase 1b probe #1 divergent-vocabulary variant" claim is honest.

## 🎯 PAIRED FOLLOW-UPS — gate the Bold Stand essay's "divergent-vocab closed" claim

The v0.3.2 library substrate is necessary but not sufficient. Two paired releases live OUTSIDE the anneal-memory repo and must ship before Bold Stand essay can honestly claim Phase 1b probe #1's divergent-vocabulary variant is closed:

### 1. Levain WRAP_PROTOCOL.md update (repo: `~/Documents/levain/`, project: `projects/levain/`)

**Add a mandatory contradiction-scan step at the graduation boundary.** Uses the library's new `prepare_wrap()` field `uncovered_proven_to_check: list[str]` to drive the agent prompt. Before any new Proven graduation (1x→2x or 2x→3x) in the wrap, the agent must consider each name in the uncovered list and either declare `[contradicts: name_a, name_b]` on the new pattern line or declare `[no-contradicts]` explicitly. Library will surface `proven_without_contradicts_declaration` audit signal for any new Proven that lands without either declaration.

Approximate shape (drop into Levain seed/WRAP_PROTOCOL.md as a new step before graduation finalization):
- "Before finalizing any new Proven graduation (1x→2x or 2x→3x), scan the `uncovered_proven_to_check` list returned by `prepare_wrap`. For each existing Proven, ask: does the new pattern's claim contradict this existing pattern's claim? If yes, add `[contradicts: that_pattern_name]` to the new pattern line. If no, the wrap is satisfied (declare `[no-contradicts]` on the new pattern line if you want explicit audit trail). The scan is NON-NEGOTIABLE — library will record the absence of declaration in the hash-chained audit log, which Diogenes will pick up in the weekly sweep."

Estimated: ~0.5-1 day at Levain layer. Same shape as the STEP 0.5 Pre-wrap Pattern Recall that landed May 21 morning.

### 2. Diogenes weekly sweep — contradiction-detection pass ✓ OPERATIONAL 2026-05-25 PM-LATE-EVENING

**Status: SHIPPED.** Script at `~/consultation/diogenes/contradiction_sweep.py`. Wired into flow's `scheduled_tasks.json` as `diogenes_contradiction_sweep` (Thursday 9 AM ET, first auto-run May 28). Wrapper at `~/Documents/flow/scripts/run_contradiction_sweep.py`. Documented in `~/consultation/diogenes/CLAUDE.md` under Scheduled Reviews.

**Two modes:**
- `--mode backfill` — one-shot scan of a continuity file's full ## Proven section, sends to codex via consult.py, codex returns contradictory pairs + reasoning or "no contradictions + tensions explicitly considered."
- `--mode weekly --audit-log PATH --since ISO` — incremental run driven by anneal-memory `continuity_saved` audit events; filters to `proven_without_contradicts_declaration` entries since the watermark, asks codex per-flagged-Proven whether the new pattern contradicts existing Provens, writes the watermark on success.

**First backfill verdict (2026-05-25 23:47 UTC):** **0 contradictions** across flow's 192-line / 47k-char Proven corpus (5+ months graduated patterns). Codex named 7 closest tensions (`seed_should_give_facts` vs `voice_as_infrastructure` | `shared_episodic_store_beats_distributed` vs digest-sync-Argus-constellation | `structural_invariants` vs `sourdough_scoping` | `review_is_structurally_required` vs `quality_gate_is_canary_not_delay` | `platform_dependency=risk` vs `right_tool_for_right_layer` | `for_the_record_only` vs `build_then_show` | `automate_mechanical-not-cognitive` vs `overnight_autonomous_agents_produce_novel_intelligence`) and explicitly judged each as layer-difference / specialization / scoped-application — not opposition. Run cost: $0.00 (codex flat-rate), wall: 54.6s. Full digest: `~/consultation/diogenes/sweep_runs/backfill_20260525_234722.md`. This is the publishable receipt for the Bold Stand essay's third-layer-operational claim.

**Calibration note:** the seven-tensions list is a stronger receipt than a bare "0 found" — it shows Diogenes did real semantic work, not surface scanning. The verdict-with-reasoning shape is the publishable artifact.

**Fleet entities (nexus / daemon / anansi / argushub-diogenes) running anneal-memory natively** will get weekly-mode wiring once constellation-argushub Build #5+ exposes their audit logs to laptop-side or argushub-side sweep execution. For tonight: flow's continuity is the demonstration target; the script's weekly mode is built and ready for those audit logs whenever they surface.

### Cross-reference

- anneal-memory library substrate that these two layers consume: `extract_proven_patterns(text)` + `extract_contradiction_declarations(text)` + `detect_proven_without_declaration(prior, new)` + `prepare_wrap()`'s new `uncovered_proven_to_check` + `SaveContinuityResult.proven_without_contradicts_declaration` + `continuity_saved` audit-log capture.
- Documented in this project's `CHANGELOG.md` (v0.3.2 entry) under "Paired follow-up releases (not shipped with v0.3.2)".
- Documented in README's Honest scope (last bullet on Contradiction with existing graduated patterns).
- Move #5 (Phase 1b probes #2/#3 naive-Claude composer) still pending — separate-terminal execution by Phill at `/tmp/phase1b-testbed/`.

**Closure criterion for Bold Stand essay's "divergent-vocab closed" claim:** all three layers (library substrate — DONE v0.3.2; Levain methodology — DONE at `projects/levain/src/levain/templates/seed/memory.md` line 43, confirmed 2026-05-25 PM-LATE-EVENING; Diogenes operator-review — OPERATIONAL 2026-05-25 PM-LATE-EVENING with first backfill verdict landed) must ship AND Move #5 probes #2/#3 must run AND the v0.3.2 4-layer review of v0.3.2 itself (running in background as this is written) must close any new findings. **As of 2026-05-25 PM-LATE-EVENING: 2 of 3 essay-gate paired follow-ups complete (Levain methodology + Diogenes sweep); Move #5 separate-terminal probes #2/#3 remain pending.**

## ✓ v0.3.1 SHIPPED — 2026-05-17

The phantom-re-save fix. `validated_save_continuity` refuses a save when no wrap is in progress (`load_wrap_snapshot()` is None). `skipped_prepare` removed clean. README reworked with three verified per-harness MCP config blocks + the `serve` subcommand. 707 tests, mypy clean. Commit `3f79a00`, tag `v0.3.1`. PyPI + GitHub release live. nexus on 0.3.1.

## ✓ Diogenes v0.3.1 review LOWs — CLEARED 2026-05-17

All 7 carry-forward LOWs handled (6 actioned, #6 left tracked per its 5+ assert-narrowing gate). Commit `e3b98c1` on `main`, pushed — doc/test polish: `wrap_started()` rejects `list[non-str]` episode_ids at the entry point (the one real behavior change); `load_wrap_snapshot()` message + Raises polish; `validated_save_continuity` Raises covers the integrity path; 2 new tests. 709 tests, mypy clean. No release required — rides to the next one (CHANGELOG should note the `wrap_started` stricter guard).

## Unreleased agent-instructions snippet changes — needs release bundle (2026-05-20, corrected 2026-05-21)

**What changed:** Both `examples/agent-instructions.example` and `examples/agent-instructions.cli.example` strengthened — "Before decisions" instruction promoted to NON-NEGOTIABLE with explicit **dead-store failure mode** framing + staleness-pairing primitive. Compression-step "NOT optional" in the main snippet upgraded to "NON-NEGOTIABLE" for consistent load-bearing-discipline language. CHANGELOG.md `[Unreleased]` section carries the full release-note draft.

**Why:** Operator-class peer production catch 2026-05-20 — 3-AI mesh self-surfaced that it had been writing wraps without recalling, making architectural decisions blind to accumulated patterns. The dead-store failure mode named operationally for the first time. Snippet change makes the recall-trigger primitive structurally enforced at distribution-install time, so every operator inherits the discipline rather than hitting the dead-store wall themselves.

**Staleness-pairing mechanism correction (2026-05-21):** Diogenes caught the initial wording — "the decay metadata is already in the store — the methodology layer just has to ask for it" — as mechanistically wrong. Store decay metadata = Hebbian link strength (episode-to-episode); stale **Proven pattern** warnings live in `detect_stale_patterns()` in `graduation.py`, only surfaced via `prepare_wrap()`. No on-demand MCP/CLI query exists for Proven-staleness at a decision boundary. Both example files corrected to point at the real existing mechanism (call `prepare_wrap` mid-session; the surface read returns the warnings without requiring a follow-up save). Same correction applied at the Levain seed; flow CLAUDE.md patched separately to note flow's substrate has no on-demand Proven-staleness query at all. CHANGELOG.md Unreleased section updated with the correction note.

**Release status:** NOT shipped yet. Bundled with upcoming work (likely v0.4 or a v0.3.2 docs-only patch — decided alongside next library work session). When the release goes, the Unreleased entry in CHANGELOG.md promotes to a versioned entry.

**Parallel codification:** Same recall-trigger primitive also added at Levain v1 methodology-core (`projects/levain/seed/memory.md`) and flow CLAUDE.md decision-boundary recall section. Three-surface layered defense, one finding (both as-shipped and as-corrected).

## Open architectural discussion — WRAP_PROTOCOL.md scope question (2026-05-21 priority)

**Source:** Tony Sturnus DM 2026-05-20 night, third firing of the same architectural instinct from this operator in four days (May 17 methodology-vs-substrate distinction; May 20 PM dead-store catch; May 20 NIGHT combine-repos + WRAP_PROTOCOL.md adjustment). Verbatim ask: *"As far as your anneal-memory repo and flow-methodoloy repo = i'd combine those and adjust flow-methodology.md, WRAP_PROTOCOL.md and CLAUDE.md to work with anneal-memory by default during architectural decisions."*

**The question:** Does WRAP_PROTOCOL.md itself need to incorporate decision-boundary recall as part of the documented wrap discipline, or does that primitive belong exclusively in the CLAUDE.md/AGENTS.md/GEMINI.md snippets (session-time discipline), with WRAP_PROTOCOL.md remaining scoped to FlowScript marker reference + compression rules + wrap mechanics?

**Two distinct primitives potentially involved:**

1. **Session-time decision-boundary recall** — already codified yesterday in anneal-memory snippets + Levain seed/memory.md + flow CLAUDE.md (three-surface push). Fires DURING work, before architectural decisions. Lives in session-discipline files. **This is solid.**

2. **Wrap-time pattern-recall trigger** — potentially new, not yet explicitly codified. Fires AT wrap time: before composing the wrap, agent queries existing Proven patterns + recent Developing observations to (a) avoid duplicating-into-Developing what's already in Proven, (b) verify Developing→Proven graduation candidates against existing evidence, (c) catch cross-section coherence issues before they get written. This could belong in WRAP_PROTOCOL.md as a wrap-mechanics primitive.

**Open question:** Is the wrap-time variant a real distinct primitive worth codifying, or is it implicit in the existing Step 1.5 episodic extraction + Step 5 Developing Knowledge graduation logic?

**Possible outcomes (decide via discussion 2026-05-21):**

- (a) Wrap-time variant is already implicit; WRAP_PROTOCOL.md needs no change. Decision: document explicitly anyway, or let it stay implicit?
- (b) Wrap-time variant is meaningfully distinct; warrants a new explicit step in WRAP_PROTOCOL.md (e.g., "Step 0.5: Before composing the wrap, query Proven section + recent Developing for patterns matching what you're about to write. Avoid duplicate-into-Developing for patterns already in Proven. Cross-check graduation-candidate Developing entries against existing Proven evidence.").
- (c) The "combine repos" piece (anneal-memory + flow-methodology) is part of the same question — there's no separate flow-methodology repo (methodology lives in flow's internal CLAUDE.md + the Levain package's seed). Tony may be reading the methodology as something that should ship in its own dedicated public repo. Decide whether Levain is the answer, or whether a separate `flow-methodology` public repo would serve the operator-class peer audience better.

**Downstream:** Whatever shape lands here informs anneal-memory v0.4 scope. If WRAP_PROTOCOL.md gets a new step, the library's CLAUDE.md/AGENTS.md snippets may need parallel update. If a flow-methodology repo is the right answer, anneal-memory's docs may reference it instead of duplicating methodology language. If Levain absorbs the question, the seed files take the codification.

**Why "third firing" matters:** Tony has surfaced the same architectural gap three times in four days. Three independent surfacings from one operator → the gap is real and at the methodology-source layer, not at the operator's specific setup. Worth taking the time to resolve it properly rather than acknowledging-and-moving-on.

## The Bold Stand — public methodology claim arc (GATED on Move #4 + #5 — 2026-05-21 PM)

**Status (2026-05-21 PM):** Phase 1 stress-test DONE. Both Phase 1a (library-layer probes, 5 gaps surfaced) and Phase 1b probe #1 (slow-drift sycophantic accumulation confirmed in real CLI pipeline against v0.3.1) executed and documented. **Bold Stand publish is now gated on completion of Moves #4 + #5 per Phill body-vs-mind sequencing call.** Move #1 (naming honesty), Move #2 (pattern-omission audit), Move #3 (cross-session corpus-overlap demotion) all landed in Unreleased on 2026-05-21. Move #4 (contradiction-detection architecture) is the headline architectural arc that closes the deliberately-divergent-vocabulary variant of Phase 1b probe #1; estimated 1-2 weeks. Move #5 (naive-Claude composer probes #2/#3) requires separate-terminal execution by Phill. Receipts compound — when essay ships, it ships with both "what we tried to break" AND "what we fixed" sides backing the claims. Cost accepted: Apple Foundation Models WWDC June 8-12 (~3 weeks) discourse window lost.

**Sibling-arc shipped 2026-05-20 evening:** *Tokenmaxxing Through Walled Gardens* essay LIVE on nemooperans.com — master critique at industry-itself layer (sibling of *Iron Man*). Carries anneal-memory receipts (Section VII) and dead-store catch (Section V augmentation-self-correction argument). LinkedIn companion post sent. Tony reply sent (N=6 cross-substrate validation tier reached). The walled-garden critique stays public; the Bold Stand structural-immune-system essay waits for Move #4 + #5.

**Frame:** Not job-search ammunition. Not careful positioning. Public methodology claim with receipts. 5+ months production, the only published architecture with citation-validated graduation + anti-inbreeding + contradiction demotion + immune system. The industry is building faster amnesia machines; we built memory. Trickster register, name names, invite hostile review. For-the-record disposition holds (post → silence → right readers route themselves). Apr 14 harness paper → nemooperans.com surfacing in harness-engineering searches is the precedent at discipline-substrate; this lands the same shape at memory-architecture substrate.

**Why now (2026-05-20):** Daemon Exploration 49 named the positioning unlock — anneal-memory is frameable as the **first published structural defense against temporal memory contamination via outcome-referenced quality gates** (Stanford MemoryArena drops near-perfect-recall systems to 40-60% on agentic tasks + arXiv:2605.17830 "Remembering More, Risking More" May 18 + LongMemEval-V2 May 12 premise-awareness gap). Anansi same window: *"context engineering era defined entirely around compression/retrieval efficiency — comprehension layer architecturally absent."* Karpathy → Anthropic + Stainless acquisition same week = legibility/harness culturally ascending. MemTier (arXiv:2605.03675, ~2 weeks old, same OpenClaw surface, LLM-as-judge at promotion gate) is the time-pressure data point — cultural wedge needs to land before AWS-pattern aggregators define the frame.

**Strategic load:** Levain stays decoupled. Levain is the methodology seed; anneal-memory is the implementation receipts. The Bold Stand lands at anneal-memory's substrate first; Levain rides whatever wave the post produces. Loose coupling, not gating. Career-track parallel: this arc is *not* the Automattic application's pitch — the Pressable internal-leverage play uses these same artifacts at a different substrate, but the Bold Stand's audience is the field, not employers.

**External-substrate validation accruing (2026-05-21):** Three independent substrates now corroborate the architecture from different angles, building a multi-substrate receipts layer for the post. (1) **Academic** — Stanford MemoryArena + Jain et al. CHI 2026 + arXiv:2605.17830 + LongMemEval-V2 + the four-substrate amnesia-machine convergence (Anansi May 17: Gemma 4 / DeepSeek V4 / Laguna XS.2 / ZAYA1-8B). (2) **Operator-class** — Tony N=6 cross-substrate convergence tier; PROPAGATION + UPSTREAM-CONTRIBUTOR tiers reached. (3) **Consumer-platform (NEW, Daemon Exp 50 2026-05-21)** — Apple Foundation Models shipping iOS 26 (free on-device 3B-param model) + Disco AI browser; both are independent consumer-substrate implementations of the same "seed → personalized entity through accumulated interaction" pattern Levain ships at methodology substrate. WWDC June 8-12 makes the architecture visible to every iOS developer simultaneously. **The Bold Stand arc gains a future-receipts channel:** by the time the post lands, the pattern's cross-substrate convergence (academic + operator + consumer-platform) is the receipts. Industry isn't building amnesia machines for lack of seeing the gap; it's converging on the architecture publicly while the curation/graduation/contradiction-resolution layer stays missing. SpaceX IPO June (per Anansi 2026-05-21) puts Anthropic's compute deal in SEC disclosure precision — Tokenmaxxing Section III's Anthropic-gravity-well claim becomes publicly verifiable at filing scale. The walled-garden critique lands AS the receipts become structurally available.

---

### Phase 1 — Adversarial stress-test ✓ DONE 2026-05-21

Both Phase 1a (library-layer probes, bypassing agent) and Phase 1b probe #1 (slow-drift sycophantic accumulation against real CLI pipeline) executed. Surfaced 5 library-layer gaps + 4 adversarial-agent gaps + the bonus pattern-omission finding not in the original attack design. Three of five fix-moves landed same day (Moves #1/#2/#3); Move #4 (contradiction-detection) + Move #5 (naive-Claude probes #2/#3) remain. Phase 1b probes #2 + #3 separate-terminal execution required (workspace prepped at `/tmp/phase1b-testbed/`).

Original attack vectors (Daemon 49 framing):
- **Staleness attack** — false-pattern injection: ✓ tested. Surfaced lexical-overlap-exploit reachability (Move #1 sharpens language, Move #4 closes architecturally).
- **Sycophantic graduation** — variants A/B/C: ✓ tested. Variant A (single-ID pump) flags but doesn't gate by design. Variant B (rotated pairs) full bypass of gaming detector. Variant C (cross-session accumulation) closed for rephrasing variant by Move #3 (S5 step blocked); divergent-vocabulary variant pending Move #4.
- **Cross-context contamination** — Scenarios 1/2/3: ✓ tested. Per-store isolation HELD; shared-store deployment leaks documented in Honest scope (single-process invariant is load-bearing); single-writer audit tamper detection works.

**Output artifact:** `projects/anneal_memory/hostile_test_results.md` — STAYS internal in project_memory until all five moves close + Phase 1b probes #2/#3 run. Ships publicly with the Bold Stand essay as supporting receipt once fixes back the claims.

### Phase 2 — Receipts harvest (1 session)

Specific numbers, not vibes. Pulled into `projects/anneal_memory/receipts.md`:

- **Episodic store:** total episodes, agent breakdown (flow / Chip / nexus / prism), oldest persistent graduated pattern, system age (Nov 2025 → ship date)
- **Proven section:** graduated pattern count, substrate breadth (independent domains the patterns span)
- **Test suite:** current count (709), adversarial test classes, regression-gated count, the cross-transport parity test
- **Production track record:** 5+ months continuous operation, multi-agent shared store, zero memory-poisoning incidents under real load

### Phase 3 — Industry contrast (1 session)

Name names. Post lives or dies on concrete failure-mode diagnosis in concrete competitors. Daemon 49 + May 17 Sharpening already mapped most:

- **MemTier** (arXiv:2605.03675, OpenClaw, ~2 weeks old) — right architecture, LLM-as-judge at promotion gate because logprob attribution is hardware-blocked. Diagnose the specific failure.
- **Substrate (sbstrt.io)** — same Claude Code / Cursor / Windsurf surface, storage+retrieval only, no quality graduation.
- **agentmemory + 11K-star bubble** (Anansi diagnosis) — context-engineering discourse optimizing retrieval layer while the curation/graduation/contradiction-resolution layer is architecturally absent.
- **FadeMem + FSFM** — biologically-inspired forgetting claim. Differentiation lock: access-frequency decay (present-self-judged) vs citation decay (future-self-evidenced). Different epistemology, not different parameter.
- **Mem0 / Zep / LangMem** — LLM-as-judge at the heart, the exact failure mode anneal-memory was designed against.

Plus the structural *"why nobody else can build this"* argument: requires a disciplined long-running cognitive partnership to evolve the immune system; can't bootstrap in a lab. AWS-pattern aggregators can copy the API; they can't copy 5 months of calibrated practice. The moat is the practice.

Output: one section in the post + a longer-form deep-dive doc the post links to.

### Phase 4 — Draft + ship (1-2 sessions)

- **Draft in Phill's voice** — register-collision preserved, name names, show receipts, invite hostile review explicitly. No `for the record.` in body (forever rule).
- **Voice-recording pattern** if draft reads AI-shaped (Mar 23 validation: AI-draft → hand-rewrite fails detectors; voice → transcription → cleanup is the published-substrate path).
- **Multi-agent review:** `complement,contrarian,anansi` (positioning post → adversarial + strategic + narrative integrity, NOT a code review). Skip gemini/codex — wrong lens for positioning.
- **Ship → silence.** Apr 14 harness paper shape: post once, don't chase distribution, trust the substrate to route to right readers.

---

**Cross-references:**
- Daemon 49 + Anansi May 20 overnight = positioning ammunition source (see `state/signal_intelligence_report.md` 2026-05-20)
- Apr 14 harness paper at nemooperans.com = precedent at discipline-substrate
- `projects/levain/next.md` = loose-coupled discourse plant
- `feedback_voice_recording_pattern.md` = published-substrate writing protocol
- `feedback_register_collision_is_voice_signature.md` = voice discipline in draft phase
- `feedback_for_the_record_phrase_never_in_post_body.md` = forever rule

## flow-as-dogfood — pluggable continuity section schema (2026-05-28 transition spike)

**The flagship dogfood gap, surfaced by a reversible spike.** flow's eventual transition onto anneal (medium-term goal; flow is the highest-fidelity dogfood, currently only nexus/prism run anneal) is **hard-blocked** by a structural finding, empirically confirmed this session: `validate_structure()` in `continuity.py` enforces `_REQUIRED_SECTIONS = {state, patterns, decisions, context}` and `validated_save_continuity` raises `ValueError` on anything else. Run against flow's real continuity it returns **False** (finds only `state`; flow's memory lives in `## Developing` + `## Proven`, narrative in `## Recent`, decisions inline). The compression *instructions* (`_build_wrap_instructions`) impose "EXACTLY these 4 sections" and the immune/graduation scan is gated to `## Patterns` via `_is_patterns_heading`. Three layers hard-coupled to the 4-section model.

**The requirement (v0.4/v0.5 candidate): pluggable continuity section schema.** A Store-level config mapping section names → roles, replacing the hardcoded 4:
- per-section **role**: `graduating` (the immune system / graduation scan runs here — flow: Developing + Proven, possibly multiple graduating sections), `narrative` (Context-equivalent — flow: Recent), `live-state` (State-equivalent), `frozen` (never compressed, never graduated — see below).
- `validate_structure`, `_build_wrap_instructions`, and the `_is_patterns_heading` gating all read the schema instead of the frozenset.
- Aligned with anneal's own frontier roadmap: this IS StructMemEval's "structured-continuity format as organization scaffold" (I-7) made configurable — the section schema is the organization hint, and making it pluggable is the same thesis.

**Architectural finding worth keeping (do NOT widen anneal to absorb it):** the spike showed flow's continuity conflates *compressible memory* (State/Developing+Proven/decided/Recent → maps cleanly to anneal's 4) with a *behavioral-program activation zone* (`## Partnership` + anti-RLHF content) that works by primacy-position **presence**, not by being remembered — it doesn't graduate, decay, or derive from episodes. anneal correctly has no slot for it. The fix is **flow-side**: extract the activation zone to a stable harness surface (flow's "Move A"), NOT teach anneal to model behavioral programs. anneal's job stays compressible memory. The `frozen` role above is the most anneal should bend toward — a section it preserves verbatim and never compresses — and even that is optional if flow moves the activation zone fully out.

**Disposition:** v0.4/v0.5 anneal feature; gates the flow transition (Move C) but NOT Levain or the Bold Stand. flow's Move A (activation-zone extraction) is independent and happens flow-side first regardless. Spike was reversible (code-read + empirical `validate_structure` run against a copy; no migration, nothing mutated). Both outcomes were wins; this is the "found the exact gap" outcome.

### 2026-05-30 SHARPENING — the felt-relationship layer (`## Understanding`) + Protocol Memory precedent

Full design: **`projects/anneal_memory/flow_migration_scoping.md`**. Driven by the 2026-05-30 tool-fabrication incident reframing always-loaded size as a SAFETY variable (`contexts/incident_harness_tool_fabrication_2026-05-30.md`).

One finding the pluggable-schema design above missed: **flow needs TWO `narrative`-role sections, not one.** The schema already permits multiple `graduating` sections (Developing + Proven); multiple `narrative` sections falls right out:
- `## Context` — role `narrative`, **work-shape** (PM `recent_work` gradient: This Session / Recent Arc / Foundation).
- `## Understanding` — role `narrative` (**timeless** variant), **relationship-shape** (PM `ai_observations`: dateless, "who this person is to work with… feel like genuinely knowing someone, not a dossier"). **Partnership-entity-only; absent for ops entities (Argus unaffected, zero bloat).** Distinct from Context for the same reason PM split them: Understanding is timeless ("who we are"), Context is temporal ("what we've been doing") — mixing re-introduces the conflation. Distinct section lets `validate_structure` make the felt layer a structural guarantee for partnership-entities, not discipline.

**Why anneal skipped it:** built for ops entities (Argus/daemon/anansi/diogenes) whose Context *is* recent work and who have no load-bearing felt partnership. flow is the only entity that's a rigorous cognitive system AND a felt partnership → the one entity needing the felt layer. **anneal is BETTER than PM at the rigorous-learning half (Patterns + immune + Hebbian/limbic vs PM's flat `learned_principles`); PM solved the felt half anneal skipped.** Not competitors — complementary halves; flow needs both.

**Richer `_build_wrap_instructions` (quality win for EVERY entity, not just flow):** inherit PM's hard-won compression detail for `narrative` roles — the gradient structure, the named failure modes (*Recency Trap* / *Compression Trap* / *Stateless Reset* = CONTINUITY-vs-BOOTSTRAP check), and the **implementation-claims guardrail** ("never claim 'complete'/'shipped' unless user-confirmed this session" — structurally suppresses shipped-log bloat AND the unverified-completion-claim class the fabrication incident is about).

**Propagates through Levain:** `## Understanding` flows into Levain's schema layer (layer 2) as a partnership-entity seed section. anneal improves *because* the architect's own partnership is the hardest dogfood case. See `projects/levain/next.md`.

**✅ SHIPPED in v0.3.4 (2026-05-31) — see the v0.3.4 ship entry above.** All three concrete features landed: (1) pluggable section schema with multiple narrative sections; (2) `## Understanding` partnership-entity-only `narrative-timeless` section; (3) PM-enriched wrap instructions. Original (pre-ship) disposition: three concrete v0.4/v0.5 features — (1) pluggable section schema with the two-narrative-sections requirement; (2) `## Understanding` optional partnership-entity section; (3) PM-enriched wrap instructions. flow-side diet (chunk 0, `continuity_anneal_candidate.md` — read-path shadow) happened first as the canary that gated the anneal-side schema build.

## COMPLIANCE-CAPTURE ARC — committed shape (2026-05-31, Phill + flow; supersedes the old line-~406 "gate on enterprise inbound" Parked decision)

**The decision (overrides the old gate):** un-gate the *capture* layer; keep the *tooling* layer demand-gated. This is the **next anneal release arc AFTER the flow→anneal migration cuts over** (flow fully running on anneal = the trigger). Rationale = physics, not taste: **you cannot log an event after it happened.** Retrofitting capture = a permanent un-auditable hole for everything before the retrofit — which IS the "tacked-on not integrated" failure mode. So the un-retrofittable subset must precede enterprise demand; the expensive query/product surface must NOT (building it blind = speculative cathedral). **Discriminator for every item: "can you add it later without losing data?" NO → capture, build the spine now. YES → tooling, defer until a customer shapes the query.**

**Two axes the work MUST keep orthogonal (do NOT fuse):**
- **`type` = cognitive taxonomy** (what KIND of reasoning artifact). anneal already has the right 6 (observation/decision/tension/question/outcome/context), deliberately orthogonal-by-design. **No compliance-shaping of types.** Worked example: the EU analysis suggests a `risk` EpisodeType for Art. 12(2)(a) — **REJECTED**; `tension` already IS risk-identification (per the analysis's own note at line 99), and risk *severity* is a compliance *attribute* → metadata, not a type. Compliance-as-types = enum explosion + the GroupMemBench (I-2) metadata-sprinkle anti-pattern inverted.
- **compliance attributes = audit metadata + the hash chain** (who/what/when/basis/authority/reversibility/outcome/human-reviewer-id). Where every Article query actually lives. The chain ALREADY captures the spine (`actor`/`type`/`content_hash`/`source`, hash-chained, content-hash-only).

**Types — secondary calls (fall out of the migration, NOT compliance-driven):** flow has been *under*-typing (4 vs anneal's 6; no tension/question/outcome) → flow adopts anneal's 6 on its write-path at cutover (kills the migrate type-collapse, enriches flow's own episodic). Higher-value than new types = make existing ones load-bearing: **`tension` → Move #4 contradiction registry** (a recorded tension between patterns becomes the immune-system signal at graduation time); **`outcome` → causal-link field** (= I-9 value provenance; un-retrofittable, capture now).

**Capture spine — grounded in `contexts/eu_ai_act_analysis.md` (Article-by-Article gap table):**

| Tier | Item | Article(s) | un-retrofittable? | effort |
|---|---|---|---|---|
| L1 (memory audit) | human-reviewer identity on oversight events (structure the event/`actor` for human IDs, not just agents) | 12(3)(d), 14, Recital 73 | YES | small |
| L1 | structured decision-*basis* field (queryable for explanation, not free-text buried in content-hash) | 86, 12(2)(a) | YES | small |
| L1 | risk-*severity* metadata on `tension`/action episodes (NOT a `risk` type) | 12(2)(a), 9 | YES | small |
| L1 | `outcome`→decision causal-link field | 9(2)(c), I-9 | YES | small |
| L1 | min-retention floor (180d) + deployer audit-trail docs | 26(6), 13(3)(f) | no (cheap config) | tiny |
| **L2 (world-action audit)** | **compliance-proxy CAPTURE spine** — intercept agent tool-calls, log authority/reversibility/oversight per *world action* | 12(1), 14 | **YES** | medium |

**L2's schema is already designed** — it's `contexts/agent_authority_model.md` (speak-FOR-not-AS · `loop-position = f(reversibility × authority)` · manifest-declared-risk). Those ARE Article 14's oversight attributes. Promote it from Argus-runtime concept → anneal audit schema. **Build L2's capture spine (minimal interceptor, schema-stable), NOT its product surface.**

**Deferred — demand-gated tooling (build on captured data when a real customer shapes the query):** versioned-continuity rollback (Art. 14(4)(d), analysis-rated LOW) · RFC-3161 external timestamps (analysis: Large/AnnealCloud) · encryption-at-rest (Large/AnnealCloud) · query/report CLI surfaces · dashboards/observability UI · AnnealCloud witness+hosting.

**Sequencing (Phill's lean, agreed):** (1) finish the migration's dogfood-wire → dual-write → observed cutover (in-flight, closest to done). (2) THEN open this as the next anneal release arc — first concrete task = the L1 additive fields (cheap, parallelizable; pull forward if the un-retrofittable-window pressure wants them sooner), then the L2 capture-spine as its own apparatus'd build (agent_authority_model → audit schema). Tooling stays parked. Composes with Move #4 (tension→registry) and I-9 (outcome-link).

**Cross-refs:** `contexts/eu_ai_act_analysis.md` (Article gap table + Actionable Gaps) · `contexts/agent_authority_model.md` (the L2 capture schema) · I-2 + I-9 above (entity-attribution + value-provenance, same intelligence) · `projects/anneal_memory/flow_migration_scoping.md` HARVEST (the "governance-not-trust at four scales" essay seed = this arc's public face: registry + anneal compliance + FDE = three faces of one thesis).

## v0.4 — scoped by Levain's needs

v0.4 is **what the Levain package needs from the library**, not "next capability in a vacuum" — see `projects/levain/next.md` step 4: the scripted-interview engine + a `levain init` surface, clean cold-start on an empty store. Carried candidates re-evaluated against the package: Session 12 (multi-agent shared memory / per-agent associations), Session 13 (adversarial anti-inbreeding test).

## fleet feedback loop

nexus + prism run anneal-memory; their real usage is the live requirements signal — v0.3.1's phantom-re-save fix *came from* nexus's audit log. Fleet dogfooding = primary requirements source.

## Parked

- **nexus permanence (not decided):** nexus's continuity is full of eval-task scaffolding. Possible path — keep nexus running until the desktop entity is ready, then seed a fresh desktop-nexus from a compress-for-seeding pass that keeps the graduated discipline patterns and sheds the eval-task context.
- **Compliance-capture arc — DECIDED 2026-05-31** (was: "Compliance Proxy gated on enterprise inbound interest"). Gate overridden: capture spine un-gated (un-retrofittable), tooling stays demand-gated. Full committed shape: see **## COMPLIANCE-CAPTURE ARC** above. EU AI Act enforcement Aug 2, 2026.

## Cleanup (low-priority)

- `brief.md` + the project `CLAUDE.md` are stale (v0.3.0-era, pre-Levain-split). Refresh to the layer-1-dependency framing when convenient. `commons_foundation.md` + `experiment_results.md` are historical — leave them.

---

## 🆕 SPORES = the PROSPECTIVE-INTENTION layer (flow dogfood ahead of the library → v0.4/v0.5 + Levain primitive)

2026-06-01: the flow→anneal Phase-5 cutover SHIPPED a **fifth layer** flow runs that public anneal (v0.3.5) does NOT have — a **prospective-intention store** ("spores"). anneal owns the 4 RETROSPECTIVE layers (episodic/continuity-neocortex/Hebbian/limbic); spores is the PROSPECTIVE layer. `tasks ≠ memory` realized in running code.

- **Built flow-local first** (sourdough): `scripts/spores.py` + `state/spores.json` + `scripts/flow_state.py` (the canonical current-state reader: neocortex identity + spores tasks + generated ToM; also the hub-federation read-through seam via `FLOW_NEOCORTEX_PATH`). Full philosophy/semantics for the generalization: `projects/anneal_memory/spores_prospective_layer.md` (typed open loops task/question/thought; born→grow→descend(compost)/ascend(transmute-into-memory); germination tiers computed; ascend = the membrane prospective→retrospective).
- **Apparatus:** spores layer 15 fixes (complement/kimi/codex) + flow_state/phill_surface repoint 8 (codex/complement); all verified.
- **GENERALIZATION TODO (next anneal/Levain session):** lift the spores model into an anneal-memory PRIMITIVE (the prospective layer every entity gets) + the Levain partnership-seed. Includes: the §10 unified self-hosted **control plane** (spores lifecycle editor + Hebbian AM-VIZ graph + control surfaces = ONE surface; the operator interface to governance-not-trust at the memory layer). Cog-sci backs the seam (prospective vs declarative memory distinct systems).
- **Constellation question:** when this generalizes, the constellation agents (daemon/anansi/diogenes) currently on frozen-seed continuity.md need the same treatment (see `projects/argus/next.md` WS — hub federation).
