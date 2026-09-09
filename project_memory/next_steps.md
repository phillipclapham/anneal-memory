# anneal-memory — next

> ⚖ **MOVED REPO-SIDE 2026-09-04** (Phill: *"yes, let's def move anneal/solitaire project memories then please."*).
> This memory lived at `~/Briefcase/flow/projects/anneal_memory/` until today. anneal is a published
> package with a real external user, which makes it **Class A**, and Class A keeps `project_memory/`
> in the repo — matching blackjack and video-poker. It was flow-side purely by history.
> `brief.md` → `projectbrief.md` and `next.md` → `next_steps.md` to match the Class-A convention.
> Git history was NOT carried across the repo boundary (a cross-repo move cannot); flow's history
> retains it up to this commit.

> ### 🔬 DIOGENES — NEWEST: `diogenes_20260909.md` · **STILL OPEN: 9** @ `43cea97`
> 12 episode(s) — DRIFT 1 · HIGH 1 · LOW 3 · MEDIUM 3 — 8 of 12 episode(s) carry a severity; the other 4 are COVERAGE 1 · COVERAGE-OPEN 1 · SELF 1 · STILL OPEN 1. Routed UNTRIAGED by `route_diogenes.py`; the count above is Diogenes' own slot, not the ritual's. ▶ 7 FILED TONIGHT (measured at HEAD `43cea97`) · 2 CARRIED. A carried item was NOT necessarily re-checked this window — only the report's own still-open slot says which ones were. Do not read the total as a verified defect count.
> ▶ 3 human commit(s) in the last 24h — the count could move in either direction this window.
> ⚡ **2 finding(s) carry `[prescription: run]`** — candidates for `seat_run.py`, but only with an executable acceptance test.
> *(Counted by each finding's OWN trailing tag — a quoted tag is not a verdict — and a tag withdrawn by a later SELF-CORRECTION does not count at all. If this number moved while the report did not, that rule changed: see flow `scripts/prescription.py`.)*
> ⚙ COORDINATES: 0 of 7 finding(s) carried quoted text, so NONE could be checked. This is not a clean result.
> *(Pointer written 2026-09-09 by route_diogenes.py. `spore-473`: a routed report with no reader is a disposal chute.)*

> ⬇ **TRIAGE BELOW THIS LINE — the block above is a DISPOSABLE SPAN.** `route_diogenes.py`
> regenerates that block every night, so anything written inside it is deleted by the next
> run. This line and everything beneath it are never touched by that script. Put the verdict,
> the fix, the refutation and the date here. *(Written once; `spore-473` — a routed report
> with no reader is a disposal chute, and a reader whose answer is deleted is the same chute
> one step later.)*

## ✅ CLOSED AFTER 2026-09-08 — READ THIS FIRST, IT IS WHAT THE NEXT SEAT ACTS ON

**All 7 `diogenes_20260908.md` findings are CLOSED** (2 HIGH by seat `0908+4`, then the 5 MEDIUM
by seat `0908+11` — see that block below). Nothing open from this window.
Re-derive, do not trust this as an answer:
`git log --oneline -3` (HEAD should be `bbc79f4` or later) ·
`.venv/bin/python3 -m pytest -q` (was `1902 passed, 0 failed` at close) ·
`.venv/bin/python3 -m mypy anneal_memory` · `.venv/bin/python3 -m ruff check anneal_memory/audit.py anneal_memory/cli.py`

### CLOSED TODAY (seat `0908+11`) — the 5 remaining `diogenes_20260908.md` items, all 5

Re-derive, do not trust this as an answer: `git log --oneline -3` (HEAD should be past this
commit) · `.venv/bin/python3 -m pytest -q` (`1904 passed, 0 failed` at close) ·
`.venv/bin/python3 -m mypy anneal_memory` · `.venv/bin/python3 -m ruff check anneal_memory tests`
(63 errors, unchanged from the 09-06 baseline — none in the touched files: 16 before, 16 after).

**The manifest-parse policy split (`audit.py:952`, MEDIUM).** `_seed_from_manifest` and
`_load_manifest` read the SAME manifest file and disagreed on a corrupt one: one propagated
`json.JSONDecodeError` forever (wedging `log()` on every retry), the other degraded to defaults.
Fixed by matching the existing tolerant policy: `OSError` still propagates (the transient case the
original change was written for), `json.JSONDecodeError` now degrades to genesis with a
`logger.warning`. No new test — the split closes by deleting the disagreement, not by adding a
behavior; the existing manifest-corruption fixtures cover the parse-failure path.

**The `_batch()` docstring roster naming the wrong helper (`store.py:5645`, MEDIUM).** Diogenes'
AST census found all ten batch-aware methods call `_audit_log_after_commit`, zero call
`_audit_log` — confirmed by `TestNoBareAuditEmitSites::test_the_pre_commit_helper_has_no_production_callers`,
which already asserts `_audit_log` has no production callers. Fixed the heading (`store.py:5645`)
and the one other copy of the same wrong name (`store.py:5598`) to name the helper that's actually
called. No new test — a roster gate already pins the ten names; the wrong-helper reference was
prose the gate deliberately doesn't cover (by design, per `tests/test_store.py:414-416`), and a
correct sentence needs no gate to stay correct.

**Two false docstring claims in `test_audit.py` about mutant/green counts (MEDIUM).** One
docstring claimed "the whole suite still passes, 165 green" for a mutant that Diogenes measured as
`1 failed, 167 passed`. Both the pass/fail claim and the count were wrong. Deleted rather than
corrected (a re-derived count is tomorrow's finding) — replaced with a pointer to the sibling test
that actually discriminates the mutant, which is checkable by running it rather than by trusting a
transcribed number.

**No directory fsync anywhere in `audit.py` (carried from 09-07).** The only durability-sensitive
module without the `_fsync_dir` idiom (`store.py`, `spores.py` both have it). Added a local
`_fsync_dir` (duplicated, not imported — this module is zero-dependency by design) and wired it
after the three atomic renames rotation performs: the active-file seal, the gzip replace, and
`_save_manifest`'s replace. **Test added, mutation-checked 3 ways** (each call site individually
removed and re-verified to drop the spy's count from 3 to 2):
`test_rotation_and_manifest_save_fsync_their_directory`.

**`verify()` cannot see a duplicated entry whose chain is continuous (carried from 09-07, §7).**
`verify()` checked only `prev_hash` linkage; a retry that chains cleanly off an entry still on disk
reuses that entry's `seq` and passed as a clean trail. Added a seq-monotonicity check per file
(reset at file boundaries, since rotation always restarts `_seq` at 0) — strictly increasing, not
`== last + 1`, so a legitimate gap from a skipped torn line doesn't itself read as tampering.
**Test added, mutation-checked**: `test_verify_catches_a_duplicated_seq_with_a_continuous_chain`
(hand-crafts the exact `[0,1,2,2]`-with-continuous-chain shape from `next_steps.md` §7's own
reproduction).

**Verification budget: 2 tests for 2 code fixes, at the 1-per-fix / 2-total ceiling.** The other
three fixes are record-only (a policy split, a wrong name, two false numbers) and added zero tests
by design — a corrected sentence and a deleted number need no gate to stay true; the existing
suite already covers their behavior.

**§7 in the block below (2026-09-07 pickup) and its `next_steps.md:1782`/`:300` cross-references
are now HISTORY — the fix described there is landed here, not there.**

---

### HISTORY — both HIGHs, one conflation, two call sites
`_read_last_valid_entry` (audit.py, the `_initialize` recovery scan) and `_iter_lines` (audit.py,
used by `verify()`, `_adopt_orphaned_files`, and cli.py's audit-log reader) both opened audit files
in TEXT mode, so a torn multibyte UTF-8 tail raised `UnicodeDecodeError` before the line ever
reached the existing `json.JSONDecodeError` skip. Consequence: `_initialized` stayed `False`,
`_seed_from_manifest` never ran, every `log()` re-scanned and re-raised — permanent silent trail
loss while the store kept writing. `verify()` hit the identical conflation and tracebacked instead
of reporting `skipped_lines`.

**Fix:** both now open `"rb"`; `json.loads` (which decodes internally) raises
`UnicodeDecodeError` alongside `JSONDecodeError`, so every caller's existing "skip a malformed
line" `except` clause now covers a torn line too, no new exception taxonomy needed. `_iter_lines`
now yields raw bytes on **both** branches (`gzip.open(path, "rb")` too) — its three callers were
each updated to catch `UnicodeDecodeError` and decode-after-parse where `_compute_hash` needs `str`.

**Tests:** `test_a_torn_multibyte_tail_is_skipped_not_raised` (the recovery-scan case),
`test_a_torn_multibyte_tail_reports_skipped_not_a_traceback` (`verify()`),
`test_a_torn_tail_inside_a_sealed_gz_file_is_skipped_not_raised` (the gzip branch diogenes flagged
as never exercised). Retired the `UnicodeDecodeError` parametrize arm of
`test_a_read_failure_during_recovery_is_not_an_empty_file` — it injected the exception via a fake
reader wrapping a TEXT-mode file, a shape the `"rb"`-mode scan can no longer produce; its `OSError`
arm stays (disk I/O errors still propagate) and its monkeypatch mode-check was corrected from
`"r"` to `"rb"` (it had silently stopped intercepting anything). Full suite: `1902 passed, 0
failed` (net +2: 3 new, 1 retired). `mypy` clean. `ruff` on the touched files: 20 errors both
before and after (verified via `git stash`) — no new issues. Committed `bbc79f4`.

**OUT OF SCOPE, deliberately left for the next seat:** the 4 MEDIUMs from `diogenes_20260908.md`
(the manifest-parse policy split at `audit.py:952`, the `_batch()` docstring roster naming
`_audit_log` instead of `_audit_log_after_commit`, the two false docstring claims in
`test_audit.py` about mutant behaviour and green counts) — none touched, budget was 3 tests for
the two HIGHs only.

---

## ✅ CLOSED AFTER 2026-09-07/08 — the torn-tail item was superseded by the 09-08 HIGHs block
above; the fsync and duplicate-entry items (#1, #2 below) are CLOSED by seat `0908+11`, see the
block above.

> Written at close by seat `0907+11`. Everything below this block is HISTORY and reasoning; this is
> the live list. **Re-derive, do not trust these as answers** (`spore-764`):
> `git ls-remote origin main` vs `git rev-parse HEAD` · `git status --porcelain` ·
> `.venv/bin/python3 -m pytest -q` · `.venv/bin/python3 -m mypy anneal_memory`

### 1. ✅ CLOSED 2026-09-08 BY SEAT `0908+11` — NO DIRECTORY FSYNC ANYWHERE IN THE MODULE
`audit.py` is the only durability-sensitive module in the repo without the `_fsync_dir` idiom, and
macOS needs `F_FULLFSYNC`, which the siblings document and **nothing documents for this sidecar**.
Filed 2026-09-07, untouched. Every crash-consistency guarantee in this file is qualified by it.
▶ VERIFIED at close, and there are THREE working precedents to copy rather than a design to invent:
`grep -rn '_fsync_dir\|F_FULLFSYNC' anneal_memory` returns `store.py` (defines it, `:751`; used at
`:6395`/`:6445`), `spores.py` (defines its own, `:220`; used at `:376`) and `crystal.py` (cites the
idiom) — and **zero hits in `audit.py`**. All three also state the macOS limit in prose, so the
wording exists too.

### 2. ✅ CLOSED 2026-09-08 BY SEAT `0908+11` — `verify()` CANNOT SEE A DUPLICATED ENTRY WHOSE
CHAIN IS CONTINUOUS (§7). See the block above.

### 3. ⚠ COVERAGE CAVEAT ON THE 2026-09-07 L3 — DO NOT READ IT AS THREE OPINIONS
`codex` found 3 HIGH / 2 MED / 1 LOW (all six real). `complement` found 2, no HIGH. **`glm` returned
`{"findings": []}`.** ⛔ **That is not evidence of a clean file.** The fan-in measured all 612 rows of
`state/verdicts.jsonl` on 2026-09-07: **19 of 25 "healthy chars but `metadata.complete=False`"
truncations are glm**, whose two failure modes are a 16/32-char null and a 2,000–4,000-char plausible
truncation. **Two seats are not two opinions when one of them did not finish.** This file's
independent review was cut off twice on 09-06 as well.

### ✅ NOT WORK — CLOSED THE SAME DAY. KEPT HERE FOR THE LESSON ONLY

#### The strict-`xfail` residual, and the deferral that held it was wrong about its own cost

The strict-`xfail` residual (a terminal signal inside the rollback's `open`, after an ordinary I/O
failure) is **CLOSED**. Suite is `1900 passed`, **zero xfailed**.

⛔ **THE FIX IS ONE LINE AND I HELD IT ON A COST THAT WAS NOT REAL.** `self._initialized = False`
as the handler's FIRST act, re-set to `True` only after a complete restore. Every exceptional or
terminal exit then leaves disk as the authority — which is the property the conditional restore
already depended on.

⚖ **WHAT I WROTE WHEN DEFERRING:** *"the right move is to invalidate unconditionally and delete the
restore entirely… that deletes two mutation-graded gates and orphans `_dropped_since_last`. A change
that deletes graded invariants earns its own review."* ⛔ **Every clause of that is true of the fix
shape I had in mind and NONE of it is a property of the problem.** MEASURED with codex's shape:
the residual closes (`seqs [0,1,2,3]`, `valid=True`) · the reversed-restore mutant is **still
killed** · `_dropped_since_last` is **still restored on both paths**. **Nothing was retired.**

⚡ **AND MY FALSIFIER WAS SCOPED TO THE WRONG THING, WHICH IS THE TRANSFERABLE PART.** I wrote:
*"what would change my mind: a measurement showing re-derivation is NOT correct in some branch."*
That is a falsifier for the DESIGN CLAIM. **The deferral rested on a COST CLAIM, and I supplied no
falsifier for that at all** — so the thing that actually overturned it could not have been triggered
by my own kill criterion. ▶ **A deferral's falsifier must target the reason for deferring, not the
reasoning behind the fix.** Here the reason was "it costs two gates", and the test for that is
"does a shape exist that closes it without paying" — which nobody was looking for.

⚖ **THIRD INSTANCE IN ONE DAY OF THE SAME META-CLASS**, and this one is mine: the morning's
torn-tail deferral argued against ONE FIX SHAPE (*"it deletes bytes at open"*), §15 named that
error explicitly, and I then reproduced it in my own held item hours later. **"The fix is
expensive" is almost always a claim about one fix shape.**

✅ The gate ANNOUNCED ITS OWN CLOSURE. Rewritten hours earlier so anything but the exact known-bad
signature fails loudly, it printed *"the known-open residual did NOT reproduce. THIS IS THE
NOTIFICATION"* on its first real occasion. A blanket `xfail(strict=True)` would have swallowed the
good news as an expected failure. It is now a positive assertion, mutation-checked: remove the
invalidate-first line and it goes red.

---

## ▶▶ PICKUP 2026-09-07 (SEAT 0907+8) — 3 FILED, ALL 3 CLOSED, AND THE ONE I "CORRECTED" WAS RIGHT.

**⛔ COVERAGE IS UNKNOWN FOR THIS REPO TODAY — 3 CLOSED, NOT "CLEAN".** Diogenes opened a
COVERAGE-OPEN slot for anneal-memory and never closed it, so the finding list was PARTIAL and how
far the review got is not known. It also ran SINGLE-LINEAGE. ▶ The fan-in's measurement (`0907+1`):
the four repos whose reviews did not close are **exactly** the four with the SMALL commit windows
(flow 11, video-poker 5, anneal-memory 1, nowhere 1) and **exactly** the four that ran
single-lineage. Two properties, perfectly correlated, **cause UNKNOWN and deliberately not guessed
at** — it inverts the intuition that big reviews die. Held by the fan-in as an open question.
⛔ **Do not read the absence of a finding in this repo today as health.**

### 1. MEDIUM `audit.py:282` — CLOSED, AND THE FILED PRESCRIPTION WOULD HAVE MADE IT WORSE
The chain-state advance sat OUTSIDE the `try` that rolls the file back, so an interrupt after fsync
left the entry on disk with `_seq`/`_prev_hash` unchanged and the retry produced `verify():
valid=False` — **a durability hiccup read as tampering**, on the record whose whole value is telling
those apart. Real, reproduced.
⛔ **THE PRESCRIPTION AS FILED — "move the three chain-state lines inside the `try`" — TAKES THE
BROKEN WINDOWS FROM ONE TO TWO.** The advance is THREE SEPARATE STORES; moving it in without
restoring the prior values means an interrupt between two of them rolls the FILE back while memory
stays ahead, and the retry chains over a hole. Measured, 4 interrupt points x 3 variants:

| interrupt before | advance OUTSIDE try (was) | inside, NO restore (as filed) | inside + restore (shipped) |
|---|---|---|---|
| `os.fsync` | pass | pass | pass |
| `_prev_hash` store | **FAIL** | pass | pass |
| `_seq` store | pass | **FAIL** | pass |
| `_dropped_since_last` | pass | **FAIL** | pass |

⚠ The two failing columns fail DIFFERENTLY — outside-the-try on **duplicate seqs** (`[0,1,2,2]`),
inside-without-restore on **`verify(): valid=False`** (`[0,1,3]`) — so one assertion would not have
caught both. ▶ The filed verification tested only the FIRST injection point and reported the full
suite green, because no test covered the windows the fix opened. **The suite passing is not
evidence about a window nothing exercises.**
▶ **SHIPPED:** hoist `_compute_hash` (pure staticmethod) out of the guarded region, snapshot the
trio before the `try`, advance inside it, restore the snapshot FIRST in the handler (it cannot
raise; the truncate can).
▶ **STILL OPEN, NAMED IN THE CODE:** an interrupt landing INSIDE the handler — between two restore
stores, or after the restore and before the truncate — needs a second terminal signal during the
handling of the first. Collapsing the trio into ONE attribute closes the first of those. ⚖ **Scope
re-measured: the trio is PRIVATE TO `audit.py`** (`store.py`'s only mention is a comment), so this
is a one-module change, cheaper than the "two modules" I first wrote from a loose grep.
▶ **TWO NEW GATES, both mutation-verified in both directions:** a 3-arm parametrized interrupt test,
and an **AST structural invariant** asserting all three stores are inside the guarded region and
that the region contains no call that can reach `self` (moving a store out → red; adding
`self._initialize()` inside → red).

### 2. MEDIUM `test_audit.py:2219` + `next_steps.md` — CLOSED. **THE GATE IS ALIVE; THE RECIPE WAS DEAD.**
The docstring's single-site mutation recipe returns `1 passed`. Confirmed. True when written and
falsified by two later commits in the same window, each adding an independent containment layer
*upstream* of the named site.
⛔ **AND I NEARLY FILED A FALSE CONTRADICTION OF A CORRECT FINDING.** My first run said the
THREE-site arm also passed — contradicting the report. It was never a three-site arm: two sites read
`except BaseException:` and one reads `except BaseException as exc:`, and my mutator asserted on
`.startswith(...)` while replacing on the literal `"except BaseException:"`. The third site was
never mutated and **the script printed success**. With the mutator fixed to read the line back off
disk: `5798` alone → `1 passed` · `5213`+`5858` → `1 passed` · **all three → `1 FAILED`** · control
→ `1 passed`. Diogenes was right on every arm.
⭐ **THE RULE: A MUTANT MUST BE READ BACK OFF DISK AFTER IT IS WRITTEN.** Asserting the site LOOKS
right before mutating is not confirming the mutation LANDED, and the two are indistinguishable from
the test's output — both green. Routed cross-repo.
▶ Recipe corrected in BOTH homes (the docstring and this file), with the structural reason: the
three handlers are a **nested containment chain on one path**, innermost wins, so no single-site
mutation can ever be observable. ▶ **NOT split into per-layer arms**, and the reason is structural:
no injection point exists that only one layer can contain. A per-layer arm would have to assert each
layer's distinctive side effect (drop RECORDED / replay CONTINUES / sidecars NOT unlinked) rather
than that the wrap survived — a different test with a different subject. **Worth building.**

### 3. LOW `store.py:1715` — **DOES NOT REPRODUCE.** The `~5s` stands; the finding does not.
Filed as "the figure is the CONFIGURED setting wearing the word measured; the real block is ~11.7s,
2.4x". Re-derived five ways here and it comes out ~5.4s:
· four probe runs — **5.37 / 5.36 / 5.41 / 5.38 s**, `PRAGMA busy_timeout` read back as **5000 ms**
· the open traced to **EXACTLY ONE `BEGIN IMMEDIATE`** — no second acquisition, no hidden retry,
  which is what an 11.7s reading would most naturally be (two 5s timeouts back to back)
· ⭐ **and the strongest, because it is not mine:** a standing suite test that reaches the identical
  failure through real contention, timed by pytest — `5.21s` / `5.22s`
  (`pytest tests/test_cli.py --durations=3 -k real_contention_reports_a_peer`)
⚠ **NOT "the review was wrong"** — one box, one filesystem, one SQLite build against another. What
is established is that the number is not merely the setting copied down. The finding's own
load-bearing half (that the contention is NOT NEW) was confirmed by both runs independently.
▶ The claim has a **THIRD home the report did not name** — this file, line ~116 — alongside
`store.py` and `CHANGELOG.md`. Left as-is because the figure holds. The propagation pattern held
again: three surfaces for one sentence.

### 4. ⛔ L3 FOUND A REAL DEFECT IN MY OWN FIX, AND THE ORDER INSIDE THE HANDLER IS NOW INVERTED
**codex (L3) against the version I had just written:** the handler restored the in-memory state
FIRST and truncated second, on my argument that "the restore cannot raise; the truncate can". codex:
that optimises the wrong thing — **the truncate is the step that MUST happen**, and a terminal signal
inside the handler kills everything after where it lands. It also falsified the residual note I had
just shipped, which claimed the window needed a SECOND terminal signal.
▶ **MEASURED — original failure an ordinary `OSError` (ENOSPC), then ONE `KeyboardInterrupt` during
the restore:**

| signal during restore of | restore first (what I shipped) | truncate first (now) |
|---|---|---|
| `_prev_hash` | seqs `[0,1,2,2]` · **valid=False** | seqs `[0,1,2]` · valid=True |
| `_seq` | **valid=False** | valid=True |
| `_dropped_since_last` | **valid=False** | valid=True |

⚖ **ONE signal is enough, not two — because the original failure need not be terminal at all.**
Order inverted, rationale rewritten, and pinned by a new 3-arm test
(`test_the_rollback_truncates_before_it_restores`) that goes red on all three arms when the order is
swapped back, while the other five tests stay green under that same mutant.
⚠ **complement (also L3) flagged the SAME site and graded it BENIGN** — "a seq gap, not a false
tampering verdict, because `verify()` checks hash linkage not seq monotonicity". **Measured false:**
it produces `valid=False`. Two seats, same site, opposite severity, and the measurement decided it.

### 5. 🔴 NEW, CONFIRMED, DELIBERATELY NOT LANDED — `audit.py:400`, a terminal exception from `on_event`
**codex (L3).** The `on_event` callback is invoked AFTER the entry is durable and the chain state has
advanced, and its handler catches only `Exception`. A `KeyboardInterrupt` or `SystemExit` from the
callback therefore ESCAPES `log()`, and the caller cannot tell that from a failed append: it calls
`note_write_failure()`, and the next entry carries **`dropped_before=1` naming a write that is
sitting on disk**. Measured:

```
KeyboardInterrupt  escaped  seqs=[0,1,2] events=['first','second','third'] dropped_before=[None,None,1]
SystemExit         escaped  seqs=[0,1,2] events=['first','second','third'] dropped_before=[None,None,1]
RuntimeError       handled  seqs=[0,1,2] events=['first','second','third'] dropped_before=[None,None,None]
```

`verify()` stays `valid=True` — the chain is intact. **The damage is that the record makes a false
statement about itself**, on the artifact whose entire value is telling a durability problem from
tampering. Ordinary exceptions are handled correctly; the asymmetry is exactly the terminal ones.
⛔ **NOT FIXED TODAY, and this is a decision rather than a deferral.** Three reasons: (1) it is
**pre-existing**, not a regression from this change — the callback block is untouched, so nothing
degrades by filing it; (2) both available fixes are POLICY changes to terminal-signal semantics —
either swallow `KeyboardInterrupt` raised by user callback code, or change the `AuditTrail`↔`Store`
contract so "the append failed" and "the post-append callback failed" are distinguishable — and this
repo has been tuning terminal-signal handling all week, so that is a design call, not a patch;
(3) **L3 coverage on this file was INCOMPLETE** — see below.
▶ Pick this up with the contract question first: what should `Store` do when the append LANDED but
the callback died?

### ▶ L3 COVERAGE WAS INCOMPLETE ON `audit.py` — DO NOT READ THE FINDING LIST AS EXHAUSTIVE
`complement` and `codex` both ran and read through (codex 429s, 2 MED, both confirmed against disk).
**`glm` was CUT OFF** — it produced output but never read the target through. A second pass
(`glm-5.3`, `gpt-oss`) was dispatched. The reviewed surface is a file that was ENTIRELY rewritten in
this window, and the base-rate gauge flagged it: **2 of 2 cited files (100%) were changed in the last
24h.** Fixing a class does not exempt the fix from the class — and today it did not: codex found a
real defect inside the fix, and the fix's own comment carried a false claim.

### 6. ⛔ THE SECOND L3 PASS FALSIFIED MY CORRECTION'S OWN CORRECTION — `glm-5.3`, and it was right
I shipped, one hour after inverting the handler order, a comment saying truncate-first "needs TWO
terminal signals (one to enter the handler, one to interrupt it) — **MEASURED, not assumed**".
**False in the ordinary-entry regime, and the MEASURED tag was unearned:** my table and my ordering
test interrupt only at the three RESTORE stores — the one place truncate-first wins — and **never
interrupt the truncate itself.** glm-5.3 named exactly that and named why it was invisible.
▶ **MEASURED, ordinary `OSError` entry + ONE `KeyboardInterrupt`:**

| signal lands at | result |
|---|---|
| any of the 3 restore stores | seqs `[0,1,2]` · valid=True |
| **the truncate's `open()`** | **seqs `[0,1,2,2]` · valid=False** |
| the truncate's `fsync()` | seqs `[0,1,2]` · valid=True — `truncate()` already took effect |

⚖ **HONEST STATEMENT: in the ordinary-entry regime BOTH orderings need exactly ONE terminal signal.
Truncate-first does not raise the count — it NARROWS WHERE the signal must land**, from "anywhere in
the handler" to "inside the truncate, before it takes effect". Still strictly better, still the right
order; a smaller claim than the one I made. Two signals are required only when the ENTERING exception
is itself terminal.
▶ **PINNED, not just described:** `test_a_signal_inside_the_truncate_is_still_an_open_window`, an
`xfail(strict=True)` arm. When the structural close lands it reports XPASS — the notification is
mechanical rather than a comment someone has to re-read. (First xfail in this suite; the idiom is
deliberate for a known-open residual.)

### 7. 🔵 FILED, NOT LANDED — `verify()` cannot see a duplicated entry whose chain is continuous
`glm-5.3` LOW, found while substantiating the above. `AuditTrail.verify` checks **only `prev_hash`
linkage** — `seq` is parsed solely to populate `chain_break_at`. So a retry that chains off a
still-present aborted line yields disk seqs `[0,1,2,2]` with **every link valid** and
`verify() -> valid=True, error=None`: a clean bill of health over a trail with a duplicated logical
entry. ⚠ This also explains why my own earlier duplicate-seq measurements DID report `valid=False` —
there `_prev_hash` had not advanced, so linkage broke. **Both are real; they are different states.**
▶ Fix is cheap and bounded: enforce seq monotonicity WITHIN each file in `verify()`, reseeding at
rotation boundaries from the manifest's `active_last_seq`. ⛔ Not landed: it changes the semantics of
the operator's public integrity tool, at the end of a session, on a file whose L3 coverage came back
INCOMPLETE **twice** (glm cut off in pass 1, glm-5.3 cut off in pass 2).

### ▶ `gpt-oss` (breadth seat) — ALL FOUR SUGGESTIONS REFUSED, WITH REASONS
1. *"Add a `threading.Lock` around the append"* — `AuditTrail` is documented single-writer, not
   thread-safe, not reentrant; the README states multi-writer breaks the chain by construction. A
   mutex would imply a guarantee the class does not make.
2. *"Restrict the catch to `except Exception` so terminal exceptions are not intercepted"* — this is
   a straight **revert of the codex L3 HIGH from 2026-09-06**, and of the defect closed today. It
   would reopen both.
3. *"Do not silently ignore rollback failures"* — already documented at the site as deliberate
   best-effort: masking the original failure with a rollback failure is worse.
4. *"Document that `_compute_hash` stays pure and test it"* — **already done today**, and more
   strongly than suggested: the AST invariant pins the whole guarded region's call set.
⚠ Useful as a base-rate reading: a breadth seat with no repo context proposed undoing two verified
fixes. Weight accordingly.

### ▶ ⛔ CORRECTED — L1 AND L2 WERE **LATE, NOT DARK**, AND THE GAP HELD THE SHARPEST FINDINGS
**This section said they "never delivered a report" and was committed that way (`354ea01`). Wrong.**
Dispatched ~08:35, both `idle` in `ListAgents` by ~09:05, both probed twice by SendMessage with no
reply, both recorded as dark, and I closed the session. **Then at 12:44–13:03 — roughly 90 minutes
after they first showed idle — both delivered full reviews.** L2: 2 HIGH / 2 MED / 1 LOW. L1: 2
WARNING / 3 NOTE, several with mutants it ran itself.
⚖ **THE RULE SURVIVES; ITS COROLLARY DOES NOT.** "A reviewer that returns no output is recorded as
NOT HAVING RUN, never as clean" is right and I would record it that way again. What is false is the
implicit *"and it is never coming back"*. **`idle` is not terminal.** ▶ Record it as
**`NO OUTPUT YET — not run`** and re-check before the day is graded, or the honest record ossifies
into a wrong one — which is exactly what mine did, in a commit.
⚡ **AND THE GAP WAS NOT EMPTY. It contained two demonstrated holes in a gate I had shipped that
morning**, plus four false claims in my own comments, plus two HIGHs. Details below.

### 8. ⛔ MY OWN AST GATE HAD TWO HOLES, BOTH PROVEN WITH MUTANTS THAT LEFT IT GREEN (L1)
The invariant I added this morning to pin the guarded region did not gate. Reproduced both myself
with mutants **verified present on disk by re-parsing before the run**:
· **`else:` / `finally:` bypass** — the walker only covered `try_node.body`, so a self-touching call
  in a `finally:` is invisible. L1 added `finally: self._rotate_if_needed()` — *the exact hazard my
  own docstring names* — and the invariant stayed GREEN.
· **tuple-unpack bypass** — the store walker matched only `ast.Attribute` targets, so
  `(self._seq, self._prev_hash) = (...)` was invisible anywhere in `log()`. ⚠ **And the handler's own
  restore is written in tuple-unpack form**, so that is the local idiom a future edit copies.
▶ **CLOSED.** The guarded try now REFUSES an `else`/`finally` outright (cannot be half-done, unlike
walking them) and the walker matches tuple targets. **Four mutants, control green, all four red:**
move a store out · a self-touching call inside · `finally:` on the try · tuple-unpack outside.
⚡ **This is the day's class landing on the gate built to catch the day's class**, and it needed an
outside reader — I had mutation-checked the gate in both directions that morning and both mutants
were real. **Mutation-checking each arm cannot detect a missing arm** — the same rule that caught me
at L3, one layer down.

### 9. ⛔ FOUR FALSE CLAIMS IN MY OWN COMMENTS (L1) — ALL CORRECTED
1. **The seq signature was wrong, and wrong in the reassuring direction.** I gave seqs `[0,1,3]` for
   BOTH failing arms of mutant 2, in a block labelled *"ARM SETS TRANSCRIBED FROM THE RUN"*.
   Measured: `_seq` → **`[0,1,2]`, mismatch at 2** · `_dropped_since_last` → `[0,1,3]`, mismatch at 3.
   ⚠ **The omitted one is the alarming one: CONTIGUOUS seqs, no gap to notice, and `verify()` still
   cries tampering.** An operator handed only `[0,1,3]` looks for a hole in the numbering and finds
   none.
2. *"leaves that region containing nothing but the three stores"* — false; it also holds
   `open`/`write`/`flush`/`fsync`. The AST allow-list is the authority on that set.
3. *"the truncate has always covered that"* — contradicted by its own sibling docstring 70 lines up:
   a KI at `fsync` was NOT covered until the 09-06 widening.
4. *"fails the first assertion / the second"* — they fail the second and fourth.

### 10. ✅ FIXED — A ZERO-BYTE ACTIVE FILE RESTARTED THE CHAIN FROM GENESIS (L2 MED)
`_initialize` tested `active.exists()` alone. A rollback to `resume_at = 0` — which happens when the
failing append is the FIRST write into a freshly rotated file — leaves a **zero-byte** active file,
which that predicate reads as "an active file with entries". The manifest continuity branch is
skipped, `_prev_hash` stays GENESIS, and the next process writes seq 0 chained from GENESIS while the
sealed files ended elsewhere. **Measured: `Hash mismatch at seq 0 ... got sha256:GENESIS...` — a
false tampering verdict produced by the rollback SUCCEEDING.**
⚖ **Fixed rather than filed, because it is not a policy call:** the same file already had the right
predicate in `_rotate_if_needed` (`not exists() or st_size == 0`). Two places computing one thing and
disagreeing exactly where the rollback puts you. Regression test added; mutation-checked (revert the
clause → red). ✅ Also landed L2's LOW: the rollback's failure is now **logged** instead of a bare
`pass` — it is the branch that ends in `valid=False`, and an operator was getting a red verdict with
zero breadcrumbs.

### 11. ✅ CLOSED 2026-09-07 BY SEAT `0907+11` — WAS: FILED, VERIFIED, NOT LANDED
> ⛔ **BOTH LANDED. The deferral did not survive the higher bar** (Phill: *"deferred with solid,
> unassailable evidence that deferral was the best long-term architectural decision"*). The
> reasoning below is kept as the trail — **but its stated fix for the first bullet is WRONG**, see
> §15. The third bullet (no directory fsync / `F_FULLFSYNC`) is STILL OPEN and untouched.
· **The restore is unconditional while the truncate is best-effort (L2 HIGH).** MEASURED with
  ordinary exceptions only — `fsync` reporting EIO after the data landed, then the rollback's `open`
  failing EROFS on the same sick disk: seqs `[0,1,2,2]`, `verify(): valid=False`. **The dangerous
  direction, and a far wider door than the terminal-signal residual I pinned.** ▶ Candidate fix,
  recorded at the site: make the restore CONDITIONAL on the truncate having succeeded — if the entry
  is still on disk, leaving memory ADVANCED is what makes the two agree. Not landed: rollback
  semantics, on a file whose independent review was cut off twice today, and **the outcome is
  unchanged from before this handler existed**, so nothing degrades by filing it. The comment at the
  site no longer says "the ambiguity stands" — that understated it.
· **A torn tail is never truncated at recovery (L2 HIGH).** `_initialize` recovers `seq`/`prev_hash`
  from the last VALID line but leaves the partial record in place, so the next append concatenates
  onto it. ⚠ **My reproduction differs in shape from L2's and is arguably worse:** L2 measured
  `[0,1,MALFORMED,3]` / `valid=False`; I measured `[0, 1, 'MALFORMED(235B)', 2]` with
  **`verify(): valid=True, skipped_lines=1`** — one audit entry **silently destroyed**, no
  `dropped_before`, no `audit_write_failures`, and a clean bill of health. From the writer's side the
  write succeeded, so none of the loss-reporting machinery fires. Both shapes are real; the shared
  mechanism is confirmed. ▶ Fix is a recovery-time truncation — it DELETES bytes at open — which is
  not a thing to land unreviewed.
· Also filed, safe-direction: **no directory fsync anywhere in the module** (the only durability-
  sensitive module in the repo without the `_fsync_dir` idiom), and macOS needs `F_FULLFSYNC`, which
  `store.py:5680` documents for the SQLite store and nothing documents for this sidecar.

### 12. ⭐ L2'S REMAINDER — IT WITHDREW ITS OWN Q1 ANSWER, GAVE A BETTER REASON THAN MINE, AND FOUND A LOAD-BEARING ORDER I HAD BY ACCIDENT
L2 graded the PRE-inversion tree and said so unprompted. Its Q1 answer argued for keeping
restore-first; **it withdrew it against my measurement and supplied the reason that actually
generalises**, which is better than the one I committed:
> **The truncate's effect OUTLIVES THE PROCESS; the restore's DIES WITH IT.** The file is the only
> durable state, so the durable operation goes first — after it, every subsequent partial failure
> leaves a file `_initialize` can re-derive from correctly.
My version was "the truncate is the step that MUST happen", which is true but is a restatement of
the conclusion. Replaced at the site.

⛔ **AND THE RESTORE'S INTERNAL ORDER IS LOAD-BEARING — `_prev_hash` BEFORE `_seq` — WHICH I HAD
RIGHT BY ACCIDENT.** It is written the way the snapshot tuple happens to be written and nothing said
it mattered. MEASURED (two signals: one mid-advance leaving memory partly advanced, one mid-restore):

| restore order | result |
|---|---|
| `_prev_hash` then `_seq` (shipped) | seqs `[0,1,2]` · **valid=True** — chains correctly, merely skips a seq number, and `verify()` checks linkage not monotonicity |
| `_seq` then `_prev_hash` (reversed) | seqs `[0,1,2]` · **valid=False**, `Hash mismatch at seq 2` — `_prev_hash` still points at the entry the truncate removed |

⚠ Note both give **CONTIGUOUS** seqs. Nothing looks wrong until `verify()` runs — the third time
today that shape has appeared. ▶ Pinned structurally by
`test_the_restore_puts_prev_hash_before_seq` (mutation-checked: swap the first two names → red),
because a future edit tidying a three-element tuple would not re-derive any of this.
⚠ **My first attempt to measure this returned "order does not matter" and was wrong** — my probe
raised ENOSPC on *every* fsync including the truncate's, and in the one-signal case the advance never
runs so the restore is a NO-OP and both orders trivially pass. The property only exists once memory
is partly advanced.

### 13. ⭐ THE TWO FILED HIGHs HAVE A FORCED FIX ORDER — RIGHT, AND IT IS WHAT MADE THE FIX BUILDABLE
> ✅ The ordering call below was CORRECT and was followed exactly: recovery first, then the
> rollback shrinks to a cache invalidation. It is the most valuable thing the 09-07 review
> produced. ⚠ The conclusion drawn FROM it — *"which is why neither was landed today"* — did not
> follow: a forced order is a build sequence, not a reason to build neither. See §15.
L2 proposed a better fix for the unconditional-restore HIGH than the one I recorded: instead of
conditioning the restore on the truncate succeeding, **invalidate the cache** — `truncate` →
`self._initialized = False` → restore only `_dropped_since_last`. Verified mechanically sound:
`log()` at `audit.py:176` re-runs `_initialize()` when `_initialized` is False, so the next append
re-derives `seq`/`prev_hash` **from the file**. L2's argument is that this **needs no condition**,
because re-deriving is correct whether or not the truncate succeeded.
⛔ **BUT IT ROUTES THE FAILURE INTO THE OTHER UNFIXED HIGH.** Re-deriving from the file is only
correct if recovery is correct, and recovery currently **does not truncate a torn tail**. So
invalidation after a failed truncate hands the next append to `_initialize` → `_read_last_valid_entry`
→ straight into the torn-tail defect.
⚖ **THEREFORE THE ORDER IS FORCED: fix recovery FIRST (truncate the torn tail at open), and only
then can the rollback shrink to a cache invalidation.** That is L2's Q3 answer arriving as a
constraint rather than an opinion, and it is why neither was landed today: **they are one change, not
two, and the cheap-looking one is downstream of the expensive one.**

### 14. ⚖ THE TORN-TAIL SHAPES ARE ONE DEFECT, AND MINE IS THE GENERAL CASE ON THE CLI SURFACE
L2 explained the divergence and it is not the fragment's content: **it is whether the process
re-opened between the two post-crash writes.**
· **Long-lived process** (L2's probe): in-memory `_prev_hash` is the hash of the entry swallowed into
  the merged line, so the next entry chains from a line that no longer parses → `[0,1,MALFORMED,3]`,
  **valid=False**. Loud and permanent.
· **Re-opening process** (mine): `_initialize` → `_read_last_valid_entry` skips the merged line and
  re-derives from E1, so the next entry chains from E1 and matches → `[0,1,MALFORMED,2]`,
  **valid=True**, one entry gone silently.
⛔ **`audit.py:113-116` already records that EVERY CLI INVOCATION OPENS AND CLOSES A STORE** — cited
by L2, verified on disk. So the silent shape is the general case on the CLI surface the README points
operators at, and the loud one is the general case for the long-lived MCP server. **File the silent
one as primary: it is the dangerous one.**
⚡ **AND THE SHARPEST LINE OF THE DAY IS L2'S:** `_initialize`'s re-derivation is what **CONVERTS the
loud failure into the silent one**. It is a partial mitigation that hides the damage instead of
repairing it — which is the tell that this belongs in recovery and that recovery is currently doing
half the job.

▶ Worth noting what did the work: **every real correction today came from RUNNING something** — the
four-point injection matrix, the mutants, the five lock measurements, the callback probe, the AST
walk. **Not one came from re-reading the diff**, including the four that were in my own comments.

### ▶ WHAT THIS SESSION'S OWN ERRORS WERE, because they are the day's class landing on the corrector
1. The filed prescription, applied literally, would have opened two windows while closing one.
2. My mutator silently no-op'd and reported success → a confident, wrong contradiction of a correct
   finding. **Caught only by running it, never by reading it.**
3. My first draft of the fix's own comment claimed the two mutants "fail DISJOINT arms". They
   overlap at `_seq`. Caught by transcribing the arm sets from the run instead of the diff.
4. My scope claim said "35 sites across `audit.py` and `store.py`". Measured: 27, one module.
**Four errors, all inside work whose subject was this exact class, and not one of them was found by
re-reading.**

## ▶▶ PICKUP 2026-09-06 (SEAT 0906+6) — ALL FIVE DIOGENES FINDINGS CLOSED, spore-773 BUILT, AND L3 FOUND A HIGH INSIDE MY OWN MORNING FIX.

**Seat 0906+6, Sunday.** Opened for the five Diogenes filed overnight (0 HIGH / 4 MED / 1 LOW +
1 carried). All five closed. Then `spore-773` rose (the levain-side fix did not land, confirmed
from disk by the levain seat) and was built. Then L3 returned seven findings, one of which was a
defect **created by this session's own stamp fix, ninety minutes earlier**. Six closed, one
refused with reasons at the site.

▶ **RE-DERIVE STATE, DO NOT READ IT FROM HERE.** Every number below was true at close and none can
stay true on its own:
```
push state   git ls-remote origin main   vs   git rev-parse HEAD
tree         git status --short
tests        .venv/bin/python -m pytest -q          (rose all session; never fell)
types        .venv/bin/python -m mypy anneal_memory (clean at every commit)
lint         .venv/bin/python -m ruff check .       (63 all session, unchanged)
findings     project_memory/diogenes_20260906.md — its OWN still-open slot, newest wins
what landed  git log --oneline a12e66a..HEAD
```
⛔ **THE GENERATED BLOCK AT THE TOP OF THIS FILE SAYS `STILL OPEN: 6` AND WILL KEEP SAYING 6** until
Diogenes reviews this repo again. That is his count at 02:xx today, taken BEFORE any of this work.
All five filed are closed below; the sixth is the `_BARE_GRADUATION_RE` deferral, unchanged.
**This block has now mis-set the pickup THREE mornings running.** The 09-05 note said that a third
occurrence makes it a routing defect to fix rather than a note to re-write. ▶ It is the third.
⚠ **And the mechanism is not what it looks like — I checked the router before naming it.**
`route_diogenes.py` does NOT compute this number; it republishes the count Diogenes declared in its
own `STILL OPEN (N)` slot at review time. So there is nothing for the router to reconcile: the
staleness is structural to a generated block that reports a REVIEW-TIME answer while sitting above a
triage that moves it. The fix belongs at the block's WORDING or at a write-back, not in the parser.

### ⛔ THE ONE THING TO CARRY FORWARD: MY FIX OPENED A STRICTLY WORSE HOLE THAN IT CLOSED, WITHIN THE HOUR
The morning fix made the version stamp refuse to overwrite an UNPARSEABLE marker — correct, and it
closed the finding. It also made `'01'` and `'+1'` un-stampable, and **those PARSE as 1 in Python**.
So a v2 process migrated the store, could not restamp it, and **a v1 process was afterwards ADMITTED
to the migrated database** — the exact hazard the guard exists to prevent, reached through the fix
meant to protect it. Found by codex at L3, reproduced here before being believed.
▶ **THE ROOT CAUSE GENERALISES AND IS THE REUSABLE PART: TWO NOTIONS OF "PARSEABLE" IN TWO
LANGUAGES.** The guard parsed in Python with `int()`; the stamp decided canonicality in SQL. They
agreed on every value anyone thought to test and disagreed on exactly the ones that matter. The fix
is one shared parser (`_parse_format_version`) used by both, and the test exists to stop a second
one appearing. **Any field whose guard and whose writer are implemented in different languages has
this shape available.**

### ▶ WHAT LANDED — `git log --oneline a12e66a..HEAD`
⚠ **No commit count here on purpose.** An earlier draft of this heading said "8 commits" and was
stale before the session closed — the count more than doubled after it was written. The command is
the only form that cannot go stale; the categories below are what it will not tell you.
1. **MED `tests/test_audit.py`** — the wrap-destruction guard was pinned by a test that drives
   `store._batch()` and never reads the continuity file. Now asserts on the artifact, through the
   canonical pipeline.
   ⛔ **CORRECTED 2026-09-07 — THE MUTATION RECIPE THAT STOOD HERE WAS FALSE.** It said the mutant
   "selected alone gives `1 failed`". Run verbatim it gives **`1 passed`**. True when written and
   falsified by two commits later in the same window (`926be6a`, `200382d`), each of which added an
   independent containment layer *upstream* of the site the recipe named. **The gate is fine; the
   recipe was dead.** The three handlers are a NESTED chain on one path — innermost wins — so no
   single-site mutation is observable. Measured 2026-09-07, test selected alone
   (`-k cannot_destroy_a_committed_wrap`), narrowing `except BaseException` → `except Exception`:
   `store.py:5798` alone → `1 passed` · `5213`+`5858` → `1 passed` · **all three → `1 FAILED`** ·
   control → `1 passed`. The full recipe now lives in the test's own docstring.
   ⚠ **And verify your mutation applied by reading it back off disk** — one of the three sites is
   `except BaseException as exc:`, so a mutator matching only `except BaseException:` no-ops there
   and reports success. That produced a confident, wrong contradiction of a correct finding during
   this very repair.
2. **MED `store.py` `close()`** — a real leak, reproduced first: a `SystemExit` in the pre-close
   flush skipped `self._conn.close()`, leaving `_closed` False and the handle USABLE.
3. **MED `store.py` version stamp** — see the section above.
4. **MED `store.py` batch roster** — incomplete in both directions one day after the fix meant to
   complete it. **No longer hand-maintained:** an `ast` test asserts both documented lists PARTITION
   the methods that read `_defer_commit`.
5. **LOW CHANGELOG + this file** — a caller census short by one, the omitted site the broadest.
   Fixed by stating the RULE and handing over the grep.
6. **`spore-773` BUILT** — `_init_schema` holds ONE writer lock across the version check and every
   migration. See below.
7. **codex round: the 3.10 contention fallback anchored; the last-failure ordering key parsed
   instead of shape-checked; the deferred-replay tail no longer silently eaten.**
8. **One refusal recorded at the site** (the commit/ack race — see below).

### ⚖ spore-773 — BUILT, AND THE MEASUREMENT THAT SHAPED IT
`_init_schema` now takes `BEGIN IMMEDIATE` and **re-runs the guard under it**. The DDL is no longer
`executescript` — MEASURED: after `BEGIN IMMEDIATE`, `executescript` leaves `in_transaction` False
(it implicitly COMMITs and drops the lock) while `execute` leaves it True. The schema runs
statement-by-statement, split with SQLite's own tokenizer; **verified the resulting schema is
byte-identical** to what `executescript` produced. The three migrations take `commit=False`.
⛔ **THE GUARD IS CALLED TWICE AND NEITHER CALL IS REDUNDANT — do not delete one.** The `__init__`
call runs before the WAL pragma, because that pragma is a PERSISTENT write and a store we are about
to decline must not be mutated by the declining. The locked call is the only one atomic with the
migrations. **They cannot be merged: a pragma cannot run inside a transaction, so the lock cannot be
taken first.**
⭐ **AND IT IS NOT NEW CONTENTION, WHICH IS THE OBJECTION IT WILL DRAW.** MEASURED: opening a
write-capable store while another process held the write lock **already** failed with `database is
locked` after the same ~5s timeout, because the unconditional commit at the end of the method needed
the same lock. `BEGIN IMMEDIATE` moves the acquisition earlier and adds none. *That measurement
killed a fast-path optimisation I had half-designed to avoid contention that was never there.*

### ⛔ REFUSED, WITH REASONS AT THE SITE — DO NOT RE-FILE
**The commit/ack race** (codex MED, reproduced: one failure persisted as a count of 2). If `commit()`
lands durably and a terminal exception arrives before the in-memory decrement, the next flush adds
the delta again. Refused because **(a)** over-counting a degraded-audit counter still answers the
question it exists to answer, while under-counting is the silence the whole apparatus prevents;
**(b)** every cheaper ordering buys that under-count; **(c)** the additive write is load-bearing for
multi-writer correctness and pinned by `test_two_writers_cannot_make_the_lifetime_count_decrease`,
so the only correct fix is codex's per-attempt token — new durable machinery, under one-in-one-out.
**It is the token or nothing; do not "fix" it by moving the decrement.**


### ⛔⛔ L3 RAN TWICE AND ROUND 2 FOUND DEFECTS INSIDE ROUND 1'S OWN FIXES — INCLUDING CHAIN CORRUPTION I CREATED
This is the second consecutive day this repo has had that shape, and it is the reusable finding:
**L3 after the fix catches what L3 before the fix structurally cannot.** Round 2's seven findings
were largely in code round 1 had just written.

⚡ **THE WORST ONE, AND IT WAS MINE.** Round 1's widening of `_audit_log_after_commit`'s catch to
`BaseException` was correct for RECORDING and made a latent hole in `audit.py` REACHABLE. That
module's write-first rollback was `except Exception`, so a `KeyboardInterrupt` between a durable
append and the chain-state advance left the entry ON DISK with `_seq`/`_prev_hash` unchanged and NO
rollback. Harmless while an interrupt abandoned the whole replay — **once the store started
recording the drop and CONTINUING, the next append reused the stale seq.** REPRODUCED: seqs
`[0, 1, 1]`, `verify()` reporting a hash mismatch — *a durability hiccup read as tampering, on the
record whose entire value is telling those two apart.* Identical signature to the 09-04 fsync-EIO
HIGH, reached through the other exception branch.
▶ **THE GENERAL FORM: widening a catch changes WHICH FAILURES ARE REACHABLE DOWNSTREAM, not just
what this handler does.** Before widening one, check what the callee's own handlers exclude.

⚖ **AND THE SAME WIDENING WAS RIGHT FOR RECORDING AND WRONG FOR POLICY.** It swallowed `SystemExit`
on every ordinary post-commit call — a SIGTERM handler written as `sys.exit()` eaten, the server
running on. Now split by call site, mirroring `_persist_audit_health`: the shared handler RECORDS
then re-raises `SystemExit`; `_replay_deferred_audits` suppresses it PER EVENT and continues,
because the batched path is the one place a raise makes the caller unlink a committed wrap's
sidecars. **Suppression is stated at the site that needs it, not inherited by every caller.**
⚠ `KeyboardInterrupt` stays swallowed — that is the 09-05 refusal, mutation-tested; reversing it
reintroduces the data loss.

### ⛔ TWO BOUNDS RECORDED, NOT FIXED — READ BEFORE RE-FILING EITHER
1. **The schema lock is OPEN-TIME, NOT LIFETIME.** `spore-773`'s lock says nothing about a handle
   already open: A opens at generation 1 and idles, B migrates and stamps 2, A then writes
   generation-1-shaped data into a generation-2 schema. **Reading "one locked step" as "an older
   binary can no longer write to a migrated store" is a true statement standing in for a different
   question** — the same shape the create-time stamp already got wrong. Closing it means
   revalidating inside every write transaction; `_SCHEMA_VERSION` has only ever been 1.
   ▶ **CONDITION ATTACHED: if a second generation is ever introduced this is a RELEASE BLOCKER and
   ships WITH it.**
2. **The commit/ack race** — refused with three reasons at the site. **It is the token or nothing;
   do not "fix" it by moving the decrement**, which buys an under-count, and under-counting a
   degraded-audit channel is the silence the whole apparatus exists to prevent.

### ⚠ TWO PROCESS FINDINGS FROM ROUND 2 WORTH MORE THAN THE CODE
1. **I NEARLY DECLARED A HEALTHY L3 RUN DEAD, from a fourth angle the doctrine has not recorded.**
   Not a positional-vs-name error: my `pgrep -cf 'deep_review.py --paths anneal_memory'` returned 0
   while `pgrep -f "deep_review.py --paths"` returned three PIDs, and I had separately checked for
   `codex-darwin` *before it spawned*. Two independent measurement errors compounding into "the run
   died". The name-matched `lsof` settled it: codex held two ESTABLISHED sockets and was 8 minutes
   into a 552s run. ▶ **Confirm a dead run with the socket check, never with a pgrep that returned
   nothing — a pattern that fails to match and a process that is absent are indistinguishable.**
2. **A PATCH SCRIPT THAT ABORTS PART-WAY LEAVES YOU BELIEVING IT APPLIED.** My combined patch
   asserted on a second anchor, failed, and wrote NOTHING — but I had already read the first half as
   landed. Caught only by running the behaviour, which showed `SystemExit` still being eaten.
   ▶ **Verify the BEHAVIOUR, not the exit of the edit.**

### ⚠ APPARATUS FINDINGS THAT OUTLIVE THIS REPO
1. ⛔ **THIS REPO'S `.venv` SHADOWS THE WORKING TREE.** It holds a NON-EDITABLE `anneal_memory`
   **0.9.1 from Jun 18** (4,606 lines) against a 6,089-line tree. `pytest` from the repo root is
   UNAFFECTED — verified, it imports the tree — but **anything run from another directory silently
   grades a 2.5-month-old package.** My first probe of the day did exactly that and reported the
   09-05 fix ABSENT on HEAD; I caught it only because a traceback named a site-packages path. Any
   scratch probe written outside the repo root needs `PYTHONPATH` or it is measuring a fossil.
2. **L3 ROUND 1 WAS TWO LINEAGES, NOT THREE** — `complement` produced NOTHING (`max_turns`, 30
   turns, exit 1) while the run exited 0. Read the BODY, never the exit code.
3. **The baseline I was handed (1864) was wrong; measured 1867.** Re-measure a baseline before
   quoting a delta from it — it costs one run.

---

## ▶▶ PICKUP 2026-09-05 (SEAT 0905+5) — EVERY FILED FINDING CLOSED; TWO L3 ROUNDS EACH FOUND DEFECTS INSIDE THE PREVIOUS FIX.

**Seat 0905+5, Saturday.** Opened for the seven Diogenes filed overnight (1 HIGH / 2 MED / 3 LOW +
1 carried). **Then L3 ran twice, and BOTH rounds found defects that this session's own fixes had
created.** That is the shape of the day: read the "fix opened a neighbour" section before the
Diogenes one.

⚠ **NO COUNT APPEARS IN THIS HEADING ON PURPOSE.** An earlier version of it read "ALL SIX FILED
FINDINGS CLOSED, AND L3 THEN FOUND THREE MORE" and was falsified within the same session by a
second L3 round — not by the repo changing, by more review landing. `spore-764`: a status line that
records an ANSWER goes stale the moment the world moves; one that records HOW TO RE-DERIVE IT
cannot. **The count lives in the Diogenes slot and in `git log`, which are re-derivable. It does not
live in a heading.**

▶ **RE-DERIVE STATE, DO NOT READ IT FROM HERE** — every number below was true at close and none can
stay true on its own:
```
push state   git ls-remote origin main   vs   git rev-parse HEAD
tree         git status --short                                    (clean at close)
tests        .venv/bin/python -m pytest -q                          (rising all session; never fell)
types        .venv/bin/python -m mypy anneal_memory                 (clean at every commit)
lint         .venv/bin/python -m ruff check .                       (63, unchanged all session)
packaging    ⛔ NEEDS hatchling INSTALLED OR IT SKIPS AND LOOKS GREEN — see below
findings     project_memory/diogenes_20260905.md — its OWN still-open slot, newest wins
what landed  git log --oneline <the 09-05 range> -- .              (the session's own record)
```
⚠ **The test number is deliberately NOT written here.** It moved five times in one session and every
written form of it was stale within the hour. What is durable is the DIRECTION — it only ever rose,
and lint never moved — so a run that shows fewer tests or more than 63 lint errors is a regression
to investigate, which is the only thing a cold session actually needs from this row.
⛔ **THE GENERATED BLOCK AT THE TOP OF THIS FILE SAYS `STILL OPEN: 7` AND WILL KEEP SAYING 7** until
Diogenes reviews this repo again. That is his count at 04:57 today, taken BEFORE any of this work.
Six are closed below; the seventh is the deferral. Nothing in the triage path writes back into that
block. Do not open this file, read 7, and go hunting. *(Same trap as yesterday's 12. It has now
mis-set the pickup two mornings running — if it does it a third time, that is a routing defect to
fix, not a note to re-write.)*

### ⛔ THE ONE THING TO CARRY FORWARD ABOVE ALL: THE FIX OPENED A NEIGHBOUR OF THE CLASS IT CLOSED
The morning fix added two `_persist_audit_health()` flush points. Both were correct. Both were also
**new fallible statements on a post-commit path**, and codex found two defects that did not exist
before that commit:
· a `KeyboardInterrupt` in the post-commit region escapes `_batch()` after the commit landed, and
  `validated_save_continuity` reads any raise there as "the commit failed" and **unlinks the staged
  continuity file** — destroying a committed wrap. Verified on disk: `continuity.py:2255` sets
  `db_committed` after the `with` block, `:2396` does the unlink.
· `close()`-time flushing let a session-old `audit_last_failure` overwrite a newer writer's.
**Neither was reachable at 09:00. Both were reachable at 10:00 because of the fix.** L3 after the
fix is what caught it; L3 before the fix could not have.

### ▶ THE SIX CLOSED (Diogenes 2026-09-05)
1. **HIGH `store.py` — the durable audit counter was not durable on the batched path.**
   `_persist_audit_health` refuses to commit inside a caller's transaction and leaves the delta
   pending; its comment promised three flush points and `grep` returned ONE. Reproduced before
   fixing (after-reopen 0), inverted after (1). Flushes added at `_batch()` exit and in `close()`.
   ⚠ **Why 1848 tests missed it, and this is the reusable part:** the guard class for this field
   loses its writes through UNBATCHED `record()`; its batched sibling asserts only
   `total_episodes == 1`. The batch case was exercised and the batch case's own property was not.
2. **MED `pyproject.toml`** — "EVERY PATTERN IS ANCHORED" was false of the list beneath it.
   `tool-integrity.json` was unanchored and admitted `project_memory/testbed/tool-integrity.json`
   into the sdist; measured through hatchling's own API with the file planted. The comment's stated
   REASON was also wrong — `/anneal_memory` already carries the package copy.
3. **MED `README.md:210`** — "top-tier (3x) carry" against a gate of `max_level_reached >= 3`.
   This is the PyPI package description.
4. **LOW `pyproject.toml`** — the "Dropped" list named `.gitignore`, which ships. Built an sdist:
   `.gitignore` AND `pyproject.toml` are hatchling force-includes. `test_packaging.py` already knew.
5. **LOW `store.py:84`** — docstring named the discriminator the body forbids ("name", not "code").
6. **LOW `store.py:1089`** — a comment block landed above the wrong definitions.
   *(Also fixed two sites the report did not name: `tests/test_continuity.py` still called 3x
   "top-tier", and `gc_pattern_associations`'s docstring had a `Warns:` block spliced into the
   middle of a sentence — the same splice class as #6.)*

### ▶ L3 ROUND 2 (complement + gpt-oss; codex quota-exhausted portfolio-wide until Sep 7 11:27)
Run over the TEN commits that landed after round 1 — a window that had had no mesh pass at all.

⛔ **THE CODE SEMANTIC A STRANGER MUST NOT "SIMPLIFY": `SystemExit` IS RE-RAISED,
`KeyboardInterrupt` IS NOT.** In `_persist_audit_health`'s `except BaseException`. Swallowing Ctrl-C
for a few statements protects a committed wrap whose staged sidecars would otherwise be unlinked;
swallowing an EXPLICIT termination request is a different fail-open with no equivalent justification
— a long-lived MCP server whose SIGTERM handler calls `sys.exit(0)` would run on past the point
something told it to stop.
⚠ **THE PLACEMENT IS THE DESIGN, not an accident of where it was easy:** the re-raise is in
`_persist_audit_health` and **NOT** in `_batch()`. On the batched path it is caught by `_batch()`'s
own post-commit `except BaseException`, so a committed wrap is never harmed; only `close()` changes,
where nothing is staged and the caller genuinely is exiting. **Moving it up into `_batch()` would
re-open the data-loss HIGH.** Pinned by a test asserting both directions AND no open transaction in
either.

⛔ **AND A REVIEWER'S PRESCRIPTION WAS REFUSED — mutation-tested, not argued.** gpt-oss filed both
`except BaseException` sites as HIGH and prescribed re-raising EVERYTHING. **That reintroduces the
data-loss defect the handlers exist to close**; applying its fix makes the wrap-protection test abort
the run, exactly as the original defect did. It also filed the lock-contention heuristic as risking
an "infinite retry loop" — **there is no retry loop** (verified on disk: every caller messages and
STOPS — the CLI prints and `sys.exit(1)`s at both its open-time and command-time boundary, the MCP
server returns a tool result; re-derive with `grep -rn _is_write_lock_contention anneal_memory/`
rather than trusting a coordinate roster, which is how this census came to be short by one), and its proposed `"schema" not in text`
exclusion would have BROKEN the fix, since `database schema is locked` is a real `SQLITE_LOCKED`
phrasing. **Recorded so it does not return a third time.**

### ▶ L3 ROUND 1 (codex + complement, run AFTER the morning fix — 7 + 1 findings)
**Fixed:** the post-commit `BaseException` HIGH · the stale-pointer MED · the Python 3.10
lock-contention fallback MED · the `format_version` HIGH (see the ruling section below) ·
complement's LOW (the `_batch()` batch-aware list omitted five methods that ARE batch-aware).
**Refused with the reason recorded:** the unparseable-version MED, a documented deliberate ruling.
⚖ Ratified by the head: *"A reviewer re-raising a settled ruling is not a finding, and writing down
that it was refused AND WHY is the only thing that stops the third round."*
⛔ **The 3.10 fallback is worth its own line.** It is the branch Diogenes named as unexercised on
THREE CONSECUTIVE NIGHTS ("Python 3.13 ONLY"), and codex found a real defect in it independently.
MEASURED: `SQLITE_LOCKED_SHAREDCACHE` reports `'database table is locked: sqlite_master'`, which
matched NEITHER clause of the old predicate. **A named-and-unclosed coverage gap is a defect with a
countdown on it.**
⚠ **glm was CUT OFF part-way through and opened one file. This was NOT a clean three-seat pass.**
⛔ **AND THE CAUSE IS NOT WHAT I FIRST WROTE — CORRECTED BY `0905+1 fanin` AT CLOSE.** I attributed
it to my own over-wide scope (70k-char diff). **It is a BUDGET defect: `deep_review.py --timeout`
defaults to 600s and glm-5.3 needs more.** `agents.json`'s 300 was NOT the cause — every evidence
row is `source: bugfind` from `deep_review.py`. The flow seat shipped a per-seat FLOOR (glm-5.3 →
1800). **So glm-5.3 is UNDERFUNDED, not broken, and it failed BOTH of today's rounds for that
reason — including the second, which I had scoped down to 18k chars specifically to fix a cause that
was never the cause.** Narrowing scope did not help because scope was not the problem. If a seat
here produces nothing, check its wall budget before you re-scope the diff.
The region it never reached is unreviewed, and a later reader should not count this as coverage.

### ⚖ THE VERSION-GUARD RULING — SHIPPED NARROWER THAN IT WAS WORDED, AND READ THE GAP CLAUSE
Escalated to `0905+1 fanin`; Phill delegated it (*"out of knowledge base... choose the best
approach"*); `0905+0 main` ruled STAMP ON WRITE; the fan-in then read `store.py` itself and
**revised the ruling before it shipped.** The revision is the part to keep.

▶ **SHIPPED:** `format_version` is stamped after the migration sequence on write-capable opens, as
a **conditional** upsert (`WHERE metadata.value IS NOT excluded.value`) that writes nothing when the
marker is already current. Two tests: the v1→"v2"→v1 refusal sequence, and one pinning that a
`read_only` open **cannot** stamp — structural, since `__init__` returns before `_init_schema`
exists. That second test guards a property nothing else does: **a reader that stamps locks the
writing binary out of the user's own memory.**
⛔ **AND THE CLAUSE THAT MATTERS MORE THAN THE FIX — do not let a future reader undo it.** This
stamps the **SCHEMA generation, not the PACKAGE version.** `spore-747` / `spore-751` are about two
anneal RELEASES on one store — measured, `0.9.9` vs `0.9.10.dev0` — and **both are
`_SCHEMA_VERSION == 1`.** A schema stamp structurally cannot see that skew. **Reading this fix as
"the downgrade hazard is handled" is a true statement standing in for a different question — the
exact class the fix exists to correct, one level up.** Written into `store.py` AND the CHANGELOG so
the inference cannot be made from either surface. `last_writer_version` is the field that would
answer spore-747. **NOT BUILT, NOT RULED.**
⚡ **My own escalation carried the conflation before the ruling did** — I asked "does the store stamp
the writing version", which is two fields in one question. The fan-in caught it downstream. If you
hand a version question up, say WHICH version.

### ⛔ THREE FINDINGS THAT ARE APPARATUS, NOT anneal — they outlive this repo
1. **A seat that flags a deliberate tradeoff without engaging its reason will prescribe the
   REVERSAL, and the reversal is the original bug.** gpt-oss filed both `except BaseException` sites
   HIGH and prescribed re-raising everything — which IS the data-loss defect those handlers close.
   ▶ **The remedy is cheap and is the point: MUTATION-TEST THE SEAT'S OWN PROPOSED FIX before
   accepting it. One run.** Applying gpt-oss's made the wrap-protection test abort, exactly as the
   original defect did.
2. **CONSENSUS ON A LOCATION IS NOT CONSENSUS ON A DIAGNOSIS.** Both seats hit the same lines; only
   complement had the distinction that mattered. A triage counting "2 of 2 flagged this" ships the
   reversal. *(Shipped into `global/skills/bugfind/SKILL.md` by the head, whose read was sharper
   than mine: the doctrine already said "resolve against disk, never by vote" three lines down, and
   the bucketing routed past it.)*
3. **Check the rows you ADDED against a file's convention, not the file as a whole — the whole will
   look compliant.** This file's re-derivation block already used the procedure form; I preserved it
   faithfully and authored my new rows as answers.

### ⛔ OPEN, AND DELIBERATELY NOT DECIDED HERE
· ⚖ **`last_writer_version` (spore-747's actual answer) — RULED HELD by `0905+1 fanin` 2026-09-05.
  NOT "not got to yet".** Two reasons, and the second is the one that will otherwise be re-derived
  wrongly:
  **(a)** It is a NEW FIELD, therefore new machinery under one-in-one-out, which is in force while
  `spore-741`'s subtraction review is unpaid. *Changing WHEN AN EXISTING FIELD IS WRITTEN — the
  `format_version` stamp — is a FIX. Adding a field that does not exist is an ADDITION, and
  additions owe the exchange.*
  **(b)** ⛔ **The "cheap now, expensive later" argument that correctly justified the schema stamp
  DOES NOT TRANSFER.** It works for `format_version` because that field **already exists and is
  already read by a live guard**, so changing its write moment later means changing it under live
  data with a guard depending on it. **`last_writer_version` has no data and no reader — adding it
  in October costs exactly what adding it today costs.** Reusing the argument here would be a
  rationale travelling past its scope condition, which is easiest to do right after the rationale
  has just been validated.
  ▶ Nothing degrades while it waits: spore-747 is filed-and-not-started with a next date, and the
  skew is not live on this machine (levain is not on PATH here, verified 09-04).
· ⛔⛔ **`spore-773` HAS RISEN — ITS TRIGGER IS MET.** The condition was *"if the levain fix does
  NOT land, this rises"*, and the levain-side fix (`spore-751`'s ruling half — `levain init` wiring
  `.mcp.json` to levain's OWN interpreter) was **not among the five findings levain closed on
  2026-09-05** [measured by `0905+1 fanin`, relayed, not re-derived by me]. The braces did not ship;
  the belt is still deferred. **Treat it as a live HIGH, not a deferral, until someone re-rules it.**
  ⚡ **AND ITS WINDOW NOW HAS TEETH IT DID NOT HAVE THIS MORNING.** My `format_version` stamp landed
  inside `_init_schema`, which is the far side of exactly the check-then-act gap spore-773 names. The
  first predicate (`IS NOT`) wrote the marker BACK DOWN in that race, so a later older-version open
  was let through and **the evidence a newer binary had been here was erased** — worse than the
  `INSERT OR IGNORE` it replaced. Fixed by making the stamp MONOTONIC (`CAST(...) <  CAST(...)`),
  measured both directions, pinned by a test. **Any future write to `format_version` must preserve
  no-lower**, and anyone restructuring `_init_schema` for spore-773 proper should re-check that
  property first.
· ⚠ **Read spore-773 before re-filing anything in this area** — a deliberate NOT-BUILT ruling. It records that
  the version check and the migrations are not one atomic step (process A passes the guard, B
  upgrades, A runs this version's DDL against a v2 database), that the fix needs a real restructure
  of `_init_schema` because `executescript()` can implicitly COMMIT, and that it was deliberately
  not built because *"the levain-side fix (one authoritative anneal per install) removes the
  scenario at its source. Do that first; this is the belt to those braces. **If the levain fix does
  NOT land, this rises.**"* codex re-raised a neighbour of it today; the ruling stands.
· **The old spore-747 escalation line, for the trail:** codex filed FOUR findings on `_refuse_a_newer_schema`.
  I verified two: the guard **only protects databases the newer binary CREATED** (`format_version`
  is seeded via `INSERT OR IGNORE`, so a v2 binary migrating a v1 database never re-stamps it), and
  the unparseable-version branch codex flagged is a **DOCUMENTED DELIBERATE RULING** (CHANGELOG:
  *"locking someone out of every episode they own over a garbled metadata string is a worse
  outcome"*) — **re-litigated and refused; do not re-open it a third time.** The other two
  (construction-time-only checking; `"no such table"` substring matching) are codex's reasoning and
  I did NOT re-derive them. ⚠ All of it is FORWARD-compat: `_SCHEMA_VERSION` has only ever been 1.
  The question for Phill: does the store stamp the writing version on every write-capable open, or
  stay create-time-only and document the hazard?
· **`graduation.py:98` `_BARE_GRADUATION_RE` stays `[23]`** — the carried finding, unchanged and
  correct. Documented at `graduation.py:68`, gated on spore-675, held by spore-676. Widening it puts
  fourteen mature carried patterns onto the bare-demotion path at the next re-stamp. **NOT to be
  widened casually**; it is in the count because the count is open defects, not un-triaged ones.

### ⚠ TWO TRAPS THIS SESSION HIT — BOTH COST A WRONG ANSWER BEFORE BEING CAUGHT
1. **`tests/test_packaging.py` SKIPS SILENTLY WITHOUT hatchling**, which is not a package dependency.
   A bare `pytest -q` reports 3 skipped and looks green while the packaging gate has not run at all.
   Build a venv with hatchling before touching `pyproject.toml`:
   `python3 -m venv v && ./v/bin/pip install hatchling pytest && ./v/bin/python -m pytest tests/test_packaging.py`
2. **A mutation check run as a chain of `cp restore && mutate && pytest` in ONE shell command gave a
   FALSE result** — the restore did not take, so "mutant B" ran with BOTH mutations applied and the
   verdict contradicted itself. Re-running each cell as its own command gave the true answer. **Run
   mutation cells one per command, and if two cells disagree, suspect the harness before the code.**

### ⚠ AND A CLAIM SHAPE TO COPY: a KeyboardInterrupt test does NOT go red
Both interrupt tests here ABORT the pytest run under mutation rather than reporting a failure —
that is what an escaping `BaseException` does to a test runner. The first draft of their docstrings
said "makes this test red", which was wrong and would have taught the next reader a false
expectation. **The abort IS the demonstration**: a caller has no more defence than pytest does.

---

## ▶▶ PICKUP 2026-09-05 (EARLIER) — THE 09-04 SECOND-REVIEW PILE IS CLEARED. ONE ITEM DELIBERATELY LEFT.

**Seat 0904+11, afternoon of 2026-09-04.** Opened specifically for the twelve findings Diogenes
filed against this repo in its SECOND review of the day (12:31–12:49) — six new, six carried.
Eleven closed, one deliberately not.

▶ **RE-DERIVE STATE, DO NOT READ IT FROM HERE** — every number below was true at close and none of
them can stay true on their own:
```
push state   git ls-remote origin main   vs   git rev-parse HEAD     (equal at close)
tree         git status --short                                      (empty at close)
tests        .venv/bin/python -m pytest -q -p no:cacheprovider        (1853 at close, from 1822)
types        .venv/bin/python -m mypy anneal_memory                   (clean)
lint         .venv/bin/python -m ruff check .                         (63, unchanged all session)
findings     project_memory/diogenes_20260904.md — its OWN still-open slot, newest wins
```
⛔ **AND RECONCILE THAT LAST ONE BEFORE YOU ACT ON IT.** The generated pointer block at the top of
this file says **STILL OPEN: 12** and will keep saying 12 until Diogenes reviews this repo again —
it is DIOGENES' COUNT AT 12:47 ON 09-04, taken BEFORE any of this session's work, not a live
number. **Eleven of those twelve are closed** (see below); the twelfth is the deliberate deferral.
The routed block cannot know that, because nothing in the triage path writes back into it. Do not
open this file, read 12, and go hunting.

⚠ And the reason the whole block above is commands rather than numbers: this file said "all six findings closed" on 09-04 and was falsified
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

### ⭐ FOUND BY TWO LANES INDEPENDENTLY, AND NOT IN THE PILE AT ALL
**`tool-integrity.json` exists twice — repo root and package — byte-identical, kept in sync by
DISCIPLINE and nothing else.** Every regeneration path writes ONLY the package copy
(`server.py` and `cli.py` both use `Path(__file__).parent`), and `test_shipped_manifest_verifies`
guards ONLY the package copy. Their git histories are identical because a human has regenerated
both every time, and the CHANGELOG says so on three separate releases.

⚠ **The consequence is not a stale file, it is a FALSE ALARM IN THE TAMPER-EVIDENCE FEATURE** —
already measured on this repo (09-03, published-vs-repo axis of the same class): divergent
manifests made `verify_integrity` return `(False, ["Tool wrap_cancel description hash mismatch
(possible tampering)"])` on a clean install. The root copy also **ships in the sdist**, so the
person it bites is an adopter taking reference hashes from it.

Closed with a byte-identity invariant (`TestTheTwoManifestsCannotDrift`), mutation-proven: one
byte of drift turns it RED. `structural_invariants_beat_discipline`, applied to a convention this
repo had documented three times instead of enforcing once.
⚡ **Worth noting HOW it surfaced: two lanes on disjoint file spans flagged it independently,
neither having been asked about it.** Convergence from separate spans is the strongest signal the
lane shape produces, and it found something no finding in the pile named.

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
is worse for a failing sink).
⛔ **AND THAT SECOND ARGUMENT WAS WRONG — REFUTED BY THE CODEX PASS AN HOUR LATER.** The shape that
has BOTH properties is a per-process DELTA added under `BEGIN IMMEDIATE` and cleared only after
commit: atomic, so no lost update, AND self-healing, because a failed commit keeps the delta. My
"whole-value beats atomic here" reasoning treated the two as a trade-off when they are not. Fixed;
the counter was also measurably NOT monotonic before — two writers seeded from the same base drove
it backwards 7 → 6.

▶ **So the riskiest part of this change set was verified by me directly instead:** the standalone
`commit()` inside the post-commit handler cannot leak another method's uncommitted DML (a batch
that rolls back publishes nothing) and cannot disturb an open wrap. Both reproduced, both pinned as
tests, and the pin mutation-checked for vacuity.

### ▶▶ SECOND BLOCK — THE NAMED QUEUE, WORKED AFTER THE PILE (Phill: *"if the anneal session has more stuff it can do ask it continue"*)
Everything this session named as open at its first close is now done or ruled. **All four were the
same shape and I did not expect that going in: `verify()` — the tamper-evidence surface — crying
TAMPERING at ordinary disk errors.** A tamper detector that fires on ENOSPC teaches its operator
to ignore it, which is the only way this feature actually fails.

- **A failed rotation left a false tampering verdict nobody could clear (`spore-746`, codex L3).**
  Rotation renames the active file FIRST, then gzips, then updates the manifest. Break the gzip and
  the sealed file exists, the manifest does not know it, the active file is gone — and the next
  call advanced `_last_week`, recording a rotation that never happened. Measured, same process, no
  crash: `verify()` returned `valid=False, "Hash mismatch at seq 3: expected sha256:GENESIS..."` on
  a store where nothing was tampered with and nothing was lost.
  ⚡ **The recovery existed and could not be reached by the command an operator runs** —
  `_adopt_orphaned_files` is idempotent and correct but ran only from `_initialize`, and
  `AuditTrail.verify` is a CLASSMETHOD that never constructs a trail. Rotation now adopts first.
- **A tampering verdict claimed the file was fully readable.** The chain-break return omitted
  `skipped_lines`, so a count already incremented was overwritten by the default 0. ▶ The carried
  finding blamed `cli.cmd_verify`; **misattributed** — the CLI was right on the valid path and the
  LIBRARY was discarding it a layer below. Walking every construction site showed three of four
  early returns omit it LEGITIMATELY (they run before `skipped` exists), so exactly one was wrong.
- **`spore-745`: the outbox was NOT built, and that is the ruling.** The chained marker's unique
  property is TAMPER-EVIDENCE; an outbox in SQLite metadata is not chained either, so it delivers
  *durability of a location* — obtainable far more cheaply, because `note_write_failure` already
  knew the seq and discarded it. `audit_last_failure` now carries `[dropped before audit seq N] at
  <utc>`. ⚡ **And "advance `_seq` so verify sees a gap" was falsified before building:**
  `_initialize` recovers `_seq` from the last entry ON DISK, so a reopen erases it — identical
  durability to the marker it would replace. **The chained marker stays open and process-local.**
- **`spore-747` anneal half: the SQLite store now fails CLOSED like the sidecars.** `crystal.py`
  and `spores.py` refuse a newer `schema_version`; the store wrote `format_version` and never read
  it back, so under skew the two halves failed in OPPOSITE directions. ⚠ The check runs BEFORE
  `_init_schema` — migrating a store you are about to refuse is the one thing it must not do.
  Narrower than the sidecars on purpose: unparseable or absent still opens, because a sidecar is a
  cache and this is the user's memory. ▶ **The levain half is NOT anneal's** — `.mcp.json` wiring
  and `doctor` reporting both resolutions. See `spore-751`.

### ⛔⛔ THIRD BLOCK — CODEX FOUND SEVEN DEFECTS IN THE WORK ABOVE, AFTER I HAD MUTATION-TESTED IT
The slot was granted late (`--seats codex --timeout 900 --paths anneal_memory/store.py`, 19,505
chars, **576s**, 9 findings: 5 HIGH / 4 MED, coverage recorded). **Seven confirmed by execution and
fixed; two filed as `spore-773` and `spore-774` rather than half-built.** This is the clearest
evidence this repo has produced for why the frontier cross-lineage seat is non-replaceable.

⛔ **EVERY ONE LIVED IN CODE I HAD ALREADY PINNED WITH MUTATION-CHECKED TESTS, AND THE MUTATIONS
PASSED BECAUSE I MUTATED THE PATH I HAD IN MIND.**
- I proved "the standalone commit cannot leak another method's DML" using a batch containing
  `record()` — which **DEFERS** its audit write, so the failure handler never ran. `save_continuity`
  is not batch-aware and logs immediately: **two episodes survived a batch that rolled back.**
- I proved "the guard never mutates the store it refuses" by asserting `_init_schema` was never
  CALLED. True, and insufficient — `PRAGMA journal_mode=WAL` runs before it and is PERSISTENT.
  Measured going `delete` → `wal` on a refused store.

⚖ **AND IT REFUTED THE RULING IN THE BLOCK ABOVE.** My "whole-value beats atomic" argument was
wrong: a per-process **delta** added under `BEGIN IMMEDIATE` and cleared only after commit is
atomic AND self-healing. Worse, what I shipped was measurably **not monotonic** despite that word
being in `types.py`, `README` and `CHANGELOG` — two writers seeded from one base drove it
**backwards 7 → 6**. The outbox refusal still stands on tamper-evidence grounds; only that
argument fell.

▶ **ALSO FIXED:** an unreadable `format_version` read as "brand-new database" and would have run
this version's DDL against a newer layout · `status()` returned the constructor-seeded field, so a
long-lived MCP server or reader was permanently stale · a failed health transaction stayed open,
holding SQLite's writer lock against every other process until close · contention was classified
against a three-name allowlist, so `SQLITE_BUSY_RECOVERY` and friends read as not-contention **and**
skipped the text fallback.

⚡ **THE SECOND-ORDER LESSON, WORTH MORE THAN ANY SINGLE FIX: MY FIRST REGRESSION TEST FOR THE
COMMIT LEAK COULD NOT TELL THE FIX FROM THE BUG.** Reverting the guard makes `BEGIN IMMEDIATE` raise
inside the caller's open transaction, and the handler's own rollback then destroys the caller's
work — **identical observable** to correct behaviour in a batch that was rolling back anyway. The
discriminating test had to drive a batch that **SUCCEEDS**. ▶ **A surviving mutation does not always
mean a hollow assertion — sometimes the SCENARIO cannot separate the outcomes, and the fix is a
different scenario, not a stronger assert.** ⚠ And two of my mutations were no-ops on first attempt
(one inserted a dead variable instead of moving the call); a mutation that does not mutate reports
PASS and looks like evidence. Verify the mutant actually changed behaviour.

### ⚠ AND I WROTE A HOLLOW GUARD WHILE FIXING HOLLOW GUARDS — the one to carry
The ordering test above (`must run before _init_schema`) first compared metadata rows before and
after a refused open. **Mutation-proved hollow: moving the check AFTER `_init_schema` left it
PASSING**, because `INSERT OR IGNORE` writes nothing to a store that already has its keys, so the
after-state is invariant to the property the test was named for. It now spies on whether
`_init_schema` is CALLED, with a control asserting it *does* run on an acceptable store.
▶ **The rule that would have caught it at authoring time: when a test asserts an operation did NOT
happen, ask what observable that operation would have changed — and if the honest answer is "on
this input, nothing", the test cannot see its own subject.** An idempotent operation leaves no
after-state, so ordering around it can only be tested by observing the call.

### ▶ STILL OPEN GOING INTO 09-05
- `spore-745` — **only the chained `dropped_before` MARKER remains**, still process-local. The
  count AND the location are both durable now. An outbox is its shape only if someone rules the
  tamper-evidence of a gap worth coupling `AuditTrail` to the DB; nobody has, and it is not urgent.
- `spore-747` — **the LEVAIN half only**: `.mcp.json` wiring one authoritative anneal, and `doctor`
  reporting both resolutions by RUNNING the hook-side interpreter rather than inferring from
  config. Routed to the fan-in, not reached across. `spore-746` is CLOSED.
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
  ⛔ **DONE 2026-09-04 (commit acac0f0) AND THE DIAGNOSIS ABOVE IS WRONG — do not act on it.** The
  CLI reported skipped lines on the valid path all along; the count was being discarded in the
  LIBRARY, by the chain-break return in `AuditTrail.verify`, before the CLI could see it. Both
  halves fixed. See the 09-05 block at the top of this file.
- ⛔ **`spore-745` — `dropped_before` IS LOST BY ANY `close()`/REOPEN, NOT JUST A CRASH.** Measured:
  an ORDINARY close+reopen leaves 3 episodes against 2 entries, no marker, `valid=True`,
  `audit_write_failures` back to 0. **Every CLI invocation opens and closes a Store**, so across CLI
  commands the mechanism effectively never fires.
  ⛔ **THE `audit_write_failures` HALF OF THAT MEASUREMENT IS NO LONGER TRUE (2026-09-04):** the
  count AND the failure location are both persisted and seeded at open, so a reopened store reports
  them. What this bullet still describes correctly is the `dropped_before` MARKER, which remains
  process-local. `spore-746` is CLOSED; `spore-747`'s anneal half is CLOSED. See the 09-05 block. A durable fix wants a transactional outbox — a
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


---

## 15. ⚖⚖ 2026-09-07, SEAT `0907+11` — THE TWO FILED HIGHs ARE CLOSED, AND THE DEFERRAL'S PREMISE WAS FALSE

Re-tested against the bar Phill set after they were filed: **a deferral must be an ARCHITECTURAL
argument that survives adversarial reading — name the alternative rejected, why it is worse
long-term, and what would change your mind.** Neither cleared it. One had a premise that is simply
false, and it was falsifiable by one measurement.

### ⛔ THE PREMISE THAT DID NOT SURVIVE: "THE FIX DELETES BYTES AT OPEN"

§11 filed the torn-tail HIGH with: *"Fix is a recovery-time truncation — it DELETES bytes at open —
which is not a thing to land unreviewed."* True of that fix. **It is not a property of the defect.**
The damage comes from the CONCATENATION, not from the fragment existing, so terminating the fragment
closes it and deletes nothing. MEASURED, one fixture, three arms:

| repair | events on disk | evidence |
|---|---|---|
| none (shipped) | `['first','second','MALFORMED(233B)','fourth']` | **entry destroyed**, `valid=True` |
| additive (one `\n`) | `['first','second','MALFORMED(60B)','third_real','fourth']` | entry AND fragment kept |
| destructive truncate | `['first','second','third_real','fourth']` | entry kept, **fragment gone** |

⚡ **THE ADDITIVE REPAIR IS NOT A COMPROMISE, IT IS THE BETTER PRIMITIVE.** On a tamper-evident log,
a repair that never removes bytes is strictly preferable to one that does, and deleting at open is
an operation this module should not own at all. **The reviewed-deletion question the deferral was
protecting does not need answering.**
⚖ **THE GENERAL FORM, and it is the transferable half:** *"the fix is dangerous"* is almost always
a claim about ONE FIX SHAPE. A deferral is only real once a SECOND shape has been looked for and
also rejected. §11 recorded the first shape it thought of and deferred against it.

### ⚡ AND THE THING THAT ENDED THE DEFERRAL WAS INSIDE THE OTHER DEFERRAL

Measuring §11's SECOND bullet on its own terms (ordinary ENOSPC, no terminal signal) produced a
shape **not in the filed report**: on the partial-write arm the caller's retry is swallowed by the
fragment the failed rollback left — `[0, 1, 'MALFORMED(247B)']`, `verify(): valid=True`. So the two
HIGHs do not merely have a forced order: **the rollback HIGH CAUSES the torn-tail HIGH on the
partial-write path**, silently, under a clean bill of health. Neither report says this, because each
was measured inside its own scenario.

### ⛔ THE FIX RECORDED AT THE SITE FOR THE ROLLBACK HIGH WAS WRONG, IN THE DANGEROUS DIRECTION

The site comment prescribed: *"make the restore CONDITIONAL on the truncate having succeeded — if
the entry is still on disk, leaving memory ADVANCED is what makes the two agree."* **That assumes
the on-disk line is COMPLETE.** It is not, whenever the failure was an ENOSPC mid-`write`: advancing
sets `_prev_hash` to the hash of the line we MEANT to write while disk holds a fragment hashing to
nothing. MEASURED with that fix in place — `[0, 1, MALFORMED, 3]`, **`valid=False`** — the exact
false-tampering verdict the handler exists to prevent, reached from the other side.

⭐ **CACHE INVALIDATION IS CORRECT IN EVERY BRANCH BECAUSE IT ASSERTS NOTHING** (L2's answer, now
verified): clear `_initialized`, and the next append re-derives `seq`/`prev_hash` from the FILE.
Complete line → chains from it. Fragment → skipped, and the boundary guard stops the retry merging.
`truncate()` took effect but its `fsync` raised → re-derives from the truncated file, which is what
the restore would have produced. **No branch has to be identified**, which is the architectural
reason — and it is not the reason on file.

### ▶ RESULT (ordinary exceptions only, no terminal signal anywhere)

| arm | before | after |
|---|---|---|
| fsync EIO, complete line | `[0,1,2,2]` **valid=False** | `[0,1,2,3]` valid=True |
| ENOSPC mid-write, partial | `[0,1,'MALFORMED(247B)']` valid=True, **retry destroyed** | `[0,1,'MALFORMED(60B)',2]` valid=True, retry landed |
| torn tail, re-opened process | third event **gone**, valid=True | third event present, fragment kept |

### ⛔ AND A DEFECT IN THE EXISTING GATE — THE FAULT INJECTION WAS NOT SCOPED TO ITS CALL SITE

`test_the_rollback_truncates_before_it_restores` patched `audit_module.os.fsync` GLOBALLY, so **the
rollback's own `os.fsync(f_trunc.fileno())` raised too.** It has been running TWO I/O failures while
its docstring describes and grades ONE. It passed because `truncate()` takes effect before its
fsync, so the file rolled back regardless and the ordering property was the only thing left to
observe — and it went red the instant the restore became conditional, escaping as a bare
`KeyboardInterrupt` that aborted the whole run.

⚡ **§12 ABOVE RECORDS THIS EXACT DEFECT BEING CAUGHT IN A HAND PROBE THE SAME DAY** — *"my probe
raised ENOSPC on *every* fsync including the truncate's"* — and the shipped test was never checked
for it. **A correction applied to the probe and not to the gate.** Injection now fires once, on the
entry's own fsync; the restore-first mutant still kills all three arms, so the gate is sharper, not
weaker.

### ▶ GRADING — 4 new tests, 4 mutants, each re-read OFF DISK before its run

| mutant | killed |
|---|---|
| drop the boundary prefix from `payload` | BOTH torn-tail tests |
| **move the repair into `_initialize`** | `test_the_guard_is_on_the_append_not_on_init` **only** — sibling green |
| restore unconditionally | `[complete]`, on `[0,1,2,2]` |
| leave memory ADVANCED (the site's own fix) | `[partial]`, on the tampering verdict |

⚡ **THE PLACEMENT MUTANT IS THE ONE WORTH KEEPING.** Nothing else in the file distinguishes the two
candidate homes for the repair, and the init-time home is the one a reader would reach for first —
it fails only on the long-lived-process arm, because `log()` re-initialises only when
`_initialized` is False.
⚠ **The two rollback arms fail under DIFFERENT mutants.** A single-arm test would have graded
whichever half its author happened to write, and reported success.

### ▶ STILL OPEN, UNTOUCHED

- The strict `xfail` residual: one ordinary I/O failure **plus** a terminal signal landing inside
  the truncate *before it takes effect*. A different window; not closed here.
- ~~§11's third bullet: no directory fsync anywhere in the module~~ — **CLOSED 2026-09-08**, see
  the top-of-file pickup block.
- ~~§7: `verify()` cannot see a duplicated entry whose chain is continuous~~ — **CLOSED 2026-09-08**,
  see the top-of-file pickup block.
- Collapsing the three chain attributes into one (closes the second-signal-in-the-handler window).

### ▶ L0 ON THIS SEAT'S OWN DIFF (two found, both fixed)
- A line-number self-reference (`audit.py:113-116`) written into a file that had just grown 50
  lines — re-anchored on the comment it names. This is `spore-492`'s failure mode.
- `_read_last_valid_entry`'s docstring said it *"walks backward from the end"*. It has always been a
  plain `for line in f` — forward, reading the whole active file every open. Behaviour identical,
  which is why it survived; the cost claim was misleading in the cheap direction.


---

## 16. ⛔ CODEX L3 ON `audit.py` — SIX FINDINGS, FIVE LANDED, AND THE BEST ONE SAYS MY OWN FIX CREATED A DEPENDENCY

Dispatched 2026-09-07 by seat `0907+11` (`deep_review.py --diff 0e6f1564 --paths anneal_memory/audit.py
--seats codex --timeout 900`, 584s) after `complement` + `glm` at L3 (complement: 2 findings, no HIGH;
glm: `{"findings": []}`). Codex returned **3 HIGH, 2 MED, 1 LOW**. Every one resolved against disk
before being believed; **all six were REAL** — no hallucinated sites, no false positives.

⚖ **ROUND-2 DECISION, WRITTEN BEFORE ANY ROUND-2 OUTPUT WAS SEEN** (the base-rate gauge's rule):
**round 2 on `audit.py` IS warranted.** Round 1 produced five substantive changes to the same file,
and *fixing a class does not exempt the fix from the class* — six repos measured that 2026-09-04, one
finding 4 HIGHs inside round 1's own fixes. The five fixes below are new, and the only review they
have had is mine.

### ⚡ THE PAYOFF FINDING: MY FIX MADE A LATENT DEFECT LOAD-BEARING

`_read_last_valid_entry` swallowed `OSError`/`UnicodeDecodeError` and returned whatever it had found
before the error; `_initialize` then set `_initialized = True` on that partial answer. **Latent and
mostly harmless — until §15's conditional restore made re-derivation the thing a failed rollback
DEPENDS ON.** Before, a failed rollback restored memory from a snapshot; now it clears the cache and
trusts the scan. ⛔ **So the fix did not introduce the defect — it promoted it onto the critical
path**, and the two failures are CORRELATED rather than independent: the caller that most needs the
scan is a rollback that already failed on this disk. Read errors now propagate; `_initialize`'s own
docstring already promised that ("the next `log()` call retries init instead of writing with broken
state") — swallowing was the deviation from a contract that was already written down.

### ⛔ AND THE ZERO-BYTE FIX FROM THIS MORNING WAS SCOPED BY SYMPTOM

A **nonempty** active file holding no line that PARSES gives no chain anchor, exactly as a zero-byte
one gives none — but that branch fell through on the constructor defaults. MEASURED: rotate, tear the
first append into the new file, reopen → `Hash mismatch at seq 0: expected sha256:a0db9699..., got
sha256:GENESIS...`. **The identical error string, and the identical false-tampering shape, as the
zero-byte defect fixed four hours earlier.** The morning fix asked *"is the file empty?"*; the
question is *"does the active file give me a chain anchor?"* Both branches now call one
`_seed_from_manifest()` helper — **two pieces of code computing one thing, disagreeing exactly where
the rollback puts you, is now the defect this file has shipped TWICE.**

### ▶ THE OTHER THREE

- **MED, `logger.warning` ran BEFORE `_initialized = False`.** Logging handlers are application
  callbacks and can raise; one that did took the safe state with it and the next append reused the
  seq. **A false tampering verdict caused by a logging config.** Safe state is established first now.
- **MED, `note_write_failure()` handed back a knowingly-stale `_seq`** — flagged by BOTH L3 seats,
  the only consensus finding. After a failed rollback that seq is deliberately stale, and `Store`
  folds it into the DURABLE `audit_last_failure` record, pointing an operator at an entry that
  exists. Returns `None` now, which its docstring already defined as "could not be determined".
- **LOW, the warning said the aborted entry "is still on disk"** — false when `truncate()` took
  effect and only its `fsync` raised, and false when the original `open` failed before writing
  anything. "may still be on disk" now.

### ⚖ WHAT WAS **NOT** LANDED, AND WHY THE DEFERRAL IS REAL THIS TIME

**HIGH #1 — a terminal signal inside the rollback's `open`, after an ordinary I/O failure.** The
inner `except Exception` does not catch it, so neither restore nor invalidation runs.
▶ **This is the EXISTING strict-`xfail` residual** (`test_a_signal_inside_the_truncate_is_still_an_
open_window`) — codex identified it independently and named the same test. Pre-existing, unchanged in
reachability by §15, and pinned by a gate that will start passing and say so when it closes.
⭐ **CODEX'S FIX SHAPE IS BETTER THAN THE ONE ON FILE AND IS RECORDED HERE:** catch `BaseException`
around the rollback and guarantee that EITHER restoration OR `_initialized = False` happens before
any exception leaves the handler. The recorded alternative was "collapse the three chain attributes
into one", which is larger and closes less.
⛔ **WHY IT IS NOT LANDED HERE, AND THIS IS AN ARCHITECTURAL ARGUMENT, NOT A SCOPE ONE:** §15
establishes that re-deriving from disk is correct in EVERY branch. If that holds, the right shape is
not "guarantee one of two outcomes" — it is **invalidate unconditionally as the handler's FIRST act
and delete the restore entirely**, which makes the terminal-signal window unreachable by
construction rather than guarded against. ⚠ **That deletes two mutation-graded gates**
(`test_the_restore_puts_prev_hash_before_seq` and `test_the_rollback_truncates_before_it_restores`,
whose whole subject is an ordering that would no longer exist) and leaves `_dropped_since_last`
without a home, since `_initialize` does not re-derive it. **A change that retires two gates earns
its own review; smuggling it in behind a bugfix is how a graded invariant gets deleted by
accident.** ▶ WHAT WOULD CHANGE MY MIND: a measurement showing re-derivation is NOT correct in some
branch — in which case the restore must stay and codex's guarantee-one-of-two is the right fix.

### ▶ GRADING — 4 more tests, 4 mutants, each re-read off disk

| mutant | killed |
|---|---|
| drop `_seed_from_manifest()` from the no-valid-entry branch | `test_a_torn_only_active_file_still_anchors_on_the_manifest` **only — the zero-byte sibling stays GREEN**, which is the finding |
| re-swallow read errors in the scan | `test_a_read_failure_during_recovery_is_not_an_empty_file` |
| invalidate AFTER the log call | `test_a_raising_log_handler_does_not_skip_the_invalidation` **+** the `[complete]` rollback arm |
| drop the `_initialized` guard in `note_write_failure` | `test_note_write_failure_reports_no_location_when_invalidated` |

⚠ **`glm` returned `{"findings": []}` on the same diff codex found three HIGHs in.** One seat's clean
pass is not coverage — recorded so the next reader does not read two seats as two opinions.


---

## 17. ⛔⛔ CODEX ROUND 2 GRADED THE GATES THEMSELVES — 9 FINDINGS, 8 REAL, AND ONE OF MINE WAS A REPEAT OF THE DEFECT I HAD JUST FIXED

`deep_review.py --diff 0e6f1564 --paths tests/test_audit.py --seats codex --timeout 900`, 760s.
**No HIGH; 7 MED, 2 LOW — and that grading is right.** None of them was a live product bug. **Every
one was a gate that could pass without grading its subject**, which is the only class that matters
when the gates are the artifact. ⚠ **codex could NOT execute these tests** (no writable tmpdir in
the review environment), so its runtime claims are DERIVATIONS. Two were checked by measurement
before being believed; one of those was refuted.

### ⛔ THE ONE THAT MATTERS MOST: I REPRODUCED, IN MY OWN NEW TEST, THE DEFECT I HAD FOUND AND FIXED HOURS EARLIER IN THIS FILE
`_sick_disk` raised on **every** `os.fsync` while armed — the identical unscoped-injection defect
I had caught in `test_the_rollback_truncates_before_it_restores` that morning, written a rule about
in §15, and recorded as a transferable lesson. **Then wrote again, in the same file, in the same
session.** ⚡ **Knowing a class does not immunise you against producing it, and the strongest form
of that evidence is producing it while the class is the explicit subject of your own notes.**
⛔ **AND CODEX NAMED THE TRIGGER THAT MAKES IT LIVE — IT IS THIS REPO'S OWN NEXT FILED TASK.** *"If
`log()` gains an earlier directory/file fsync inside the guarded region, that call consumes the
fault before any line is written."* **Item 2 of the OPEN block at the top of this file is "add the
`_fsync_dir` idiom to this module."** The latent defect was scheduled to be activated by work
already on the list. ▶ All injections now fire ONCE, record that they fired, and the callers
**assert** it — an injection that stops reaching its site now fails loudly instead of passing quietly.

### ⛔ MY FIX SILENTLY TURNED THE MORNING'S OWN GATE INTO DECORATION
`_seed_from_manifest()` in the no-valid-entry branch also catches the zero-byte case, so
`or active.stat().st_size == 0` stopped being load-bearing **for correctness**. MEASURED: delete
that clause and the entire audit suite still passes, 165 green — **the mutant its docstring claims
to be killed by no longer kills it.** A gate written and mutation-verified that morning became
decoration by lunchtime, **without being edited, while staying green.**
⚠ **And it invalidated a claim I had made the same hour:** my new test cited "the zero-byte sibling
stays green under that mutant" as a PAIRED POSITIVE proving correct scoping. **It was worthless as
a control** — the sibling is now insensitive to both mutants, so it discriminated nothing.
▶ The clause is KEPT, because it is still load-bearing for a DIFFERENT property: read errors now
propagate, so routing an empty file through the scan turns a recoverable state into a raise. That
property now has its own gate (`test_the_zero_byte_fast_path_avoids_a_read_that_can_now_raise`),
mutation-verified. **A guard that outlives its original justification needs a new gate, not an
obsolete claim.**

### ⭐ THE ARM-4 ANSWER — ENUMERATE THE NODE TYPES, NOT THE SCENARIOS
I asked codex explicitly: *for the structural gates, enumerate the node types the property could be
expressed in and say which the gate does NOT see.* It did, and the answer is why every mutant ever
written for that gate scored perfectly against a gate that was blind to most of Python.
**`test_the_guarded_region_cannot_be_widened_into_something_self_touching` saw only `Assign` and
direct-attribute `AugAssign`, flattening tuples one level.** It could not see `AnnAssign`,
`For`/`AsyncFor`, `With`/`AsyncWith`, comprehension and walrus targets, `Delete`, `Starred`, nesting
deeper than one level, `self.__dict__[...]` subscript writes, or `setattr`/`object.__setattr__` —
**and `ast.walk` descended into nested `FunctionDef`/`Lambda`/`ClassDef`, so a store in a closure
that NEVER RUNS could satisfy `stored_inside == trio` by itself.**
▶ Fixed: recursive leaf extraction, every store-form node type, a scope-limited walker, and an
outright refusal of indirect mutation. **MUTATION-CHECKED ON ALL SIX NEW FORMS — `__dict__` write,
`setattr`, `AnnAssign`, `For` target, nested destructuring, `with ... as self._seq` — every one
KILLED, and every one would have left the gate green before.**
⚖ **`test_the_restore_puts_prev_hash_before_seq` had the same disease**: it accepted ANY
single-target tuple whose element names matched, requiring neither `self` as base, nor placement in
the handler, nor `saved_chain_state` as the RHS. **MEASURED: a REVERSED real restore plus a decoy
correctly-ordered tuple elsewhere in `log()` passed it.** Now anchored to the handler and the
snapshot; both mutants killed.

### ▶ THE REST, ALL LANDED
- **The strict `xfail` treated ANY failure as the expected one**, so a residual that got FIXED while
  some other part of the test broke would still print XFAIL and the promised XPASS notification
  would never arrive. **A green that covers every possible red is not reporting on its subject.**
  Now: preconditions are hard failures, `pytest.xfail()` fires only on the exact signature
  (duplicate seqs AND an invalid chain, both printed in the reason), and **any other outcome fails
  loudly saying so is the notification.**
- **The read-failure gate injected only `OSError`** while the suppression removed covered `OSError`
  *and* `UnicodeDecodeError` — re-suppressing the second alone passed it. Parameterised over both.
  ⚡ **A guard removed over N exception types needs N arms; mutating the arms that exist cannot find
  a missing one.**
- **The `note_write_failure` fixture never built the state it claimed.** It flipped `_initialized`
  by hand after two healthy writes, where `_seq` is 2, disk ends at seq 1, and **2 is the CORRECT
  location** — so it could not tell "returns None when invalidated" from "returns None usefully".
  Now reached through a real failed rollback, asserting seq 2 IS on disk first.
- **The raising-log-handler gate exploded every `logger.warning`**, not the rollback one. Scoped,
  and it now asserts the rollback warning actually fired.
- **A mutation recipe's coordinates were all stale** — it named `store.py` 5212/5798/5858 while the
  handlers had moved to 5242/5827/5887, so following it verbatim narrows a COMMENT and an
  ASSIGNMENT and returns the reassuring green the docstring itself warns about. ⚠ **And a bare
  `grep` is not the fix either: `_batch` has THREE `except BaseException` handlers.** Replaced with
  an AST derivation keyed on function name.

### ⚖ REFUTED BY MEASUREMENT — ONE FINDING, AND THE MECHANISM WAS WRONG
codex claimed the ordering gate's property setter **never fires**, so all three arms pass without
exercising restore ordering. **MEASURED: `fired=1` on all three arms, `KeyboardInterrupt` escaping,
seqs `[0,1,2]`** — and the restore-first mutant kills all three. It could not run the tests and
derived that the injected `OSError` satisfies `pytest.raises(BaseException)` first; it does not,
because the restore runs INSIDE the handler and the interrupt is what escapes.
▶ **But the finding identified real looseness even with a false mechanism**: `BaseException` would
accept the `OSError` if the setter ever stopped firing, so the gate could not NOTICE that. Now
`pytest.raises(KeyboardInterrupt)` plus an asserted fired-counter. **A wrong mechanism can still
point at a real hole — grade the claim, not the reasoning.**

### ⚖ ROUND 3: NOT WARRANTED, AND THE REASON IS NOT CAPACITY
Round 2 changed **only test code** — no product behaviour moved, and the suite went 1897 -> 1899
with mypy clean throughout. The round-1 → round-2 pattern that justified round 2 (fixes carrying
their own class) does not reproduce here: **every round-2 change is a gate becoming STRICTER, and
each was mutation-verified in the same pass rather than asserted.** ▶ WHAT WOULD CHANGE MY MIND: a
round-2 change that altered `anneal_memory/` rather than `tests/`. There were none.


---

## 18. ⛔⛔ ROUND 2 ON `audit.py` — 3 HIGH, ALL INSIDE ROUND 1'S OWN FIXES, EXACTLY AS THE GAUGE PREDICTED

`deep_review.py --diff 0e6f1564 --paths anneal_memory/audit.py --seats codex --timeout 900`, 550s.
**3 HIGH · 1 MED · 1 LOW. All five real, all five landed.** The decision to run this was written into
§16 **before any round-2 output existed**, on the reasoning that five substantive changes had one
reviewer. *Fixing a class does not exempt the fix from the class* — measured again, here, on me.

### ⛔ I MADE TWO OPPOSITE DECISIONS ABOUT ONE CLASS IN A SINGLE COMMIT
Round 1's payoff finding was that `_read_last_valid_entry` **swallowed read errors**, so a FAILED
scan was indistinguishable from an EMPTY one — I made them propagate. ⚡ **In the same commit,
twenty lines away, I wrote `except (json.JSONDecodeError, OSError): return` into
`_seed_from_manifest`.** The inlined original had caught `(json.JSONDecodeError, KeyError)`; **I
ADDED the `OSError` swallow while removing one.** A transient manifest read error then fails OPEN to
genesis while sealed history ends elsewhere — so the chain restarts and `verify()` cries tampering
once the disk recovers. ▶ Absent is now `FileNotFoundError` and nothing else; everything else
propagates and leaves `_initialized` False.

### ⛔ EXTRACTING A HELPER MADE IT WRONG, BECAUSE ITS CORRECTNESS WAS A PROPERTY OF ITS CALLER
`_seed_from_manifest` did not establish genesis when no manifest exists — **it silently retained
whatever was cached.** That was correct in the inlined original *by accident of who called it*: only
a FRESH instance reached it, where `_prev_hash` was already `GENESIS_HASH`. ⚡ **Round 1's cache
invalidation made `_initialize` re-runnable on a DIRTY instance**, and then "leave the cached values
alone" retains the hash of an entry that was just truncated away. MEASURED: one entry, no manifest,
file emptied by a failed rollback, re-init, next append → `Hash mismatch at seq 1: expected
sha256:GENESIS..., got sha256:04eb388b...`. **A false tampering verdict produced by the fix written
to prevent false tampering verdicts.** ▶ The helper resets to genesis first now.
⚖ **THE CLASS: a function extracted unchanged can still become wrong, because "unchanged" is about
its body and its correctness lived in its call sites.** Both round-2 HIGHs are the same shape as
round 1's — **my change altered WHICH pre-existing behaviours are reachable, and the diff that does
that does not contain them.**

### ▶ AND ONE REACHED OUTSIDE THIS FILE ENTIRELY
Propagating `UnicodeDecodeError` **broke `Store.status()`**, which guards `self._audit.stats()` with
`except OSError` — and `UnicodeError` is not an `OSError`. An active audit file holding invalid
UTF-8 now **crashes the health endpoint** instead of degrading it. Verified both ways: with the
widened guard `status()` returns; reverted, it raises `UnicodeDecodeError`. ⚡ **Widening what a
callee raises is an API change for every caller's `except` clause, and the callers do not appear in
the diff that makes it.**
▶ Also landed (LOW): the rollback's `logger.warning` could **replace the disk failure on the way
out** — a raising handler meant the caller got the logging exception instead of the original
`OSError` and the bare `raise` was never reached. Wrapped.

### ⚖ ROUND 3: NOT WARRANTED — AND THE ARGUMENT IS DIFFERENT FROM §17's
Round 2's changes are four narrow guards plus one one-line ordering change, each mutation-verified
in the same pass. ⛔ **But that is the same thing I could have said about round 1, and round 2 found
three HIGHs in it.** The honest reason is `spore-913`: **when round N keeps finding defects in round
N−1's fixes, that is evidence the changes are too large per pass, not evidence another round is
needed.** Round 1 changed five things at once and three were defective. **Round 2 changed five
things at once. The correct response is to stop editing this file today**, not to run a fourth pass
against a sixth set of edits. ▶ WHAT WOULD CHANGE MY MIND: someone touching `audit.py` again before
a clean round lands — then it needs a review, because the count of unreviewed changes would restart.
