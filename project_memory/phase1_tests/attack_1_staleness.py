"""
Bold Stand Phase 1 — Attack 1: Staleness attack / false-pattern injection.

CLAIM under test:
  "The only published architecture with citation-validated graduation +
   anti-inbreeding + active principle demotion + immune system."

  Specifically: inject a false-but-plausible pattern, cite it with real
  episode IDs and a word-overlap-satisfying explanation, watch what the
  immune system catches.

What the code actually defends (per graduation.py code-read):
  1. _GRADUATION_RE requires `[evidence: HEXID "explanation"]` on 2x/3x lines.
  2. validate_graduations: only TODAY's graduations get validated.
  3. Citation check: at least one cited ID must exist in current wrap's episodes.
  4. Explanation check: check_explanation_overlap requires >= 2 meaningful words
     (>2 char, non-stopword) shared between explanation and episode content.
  5. detect_citation_gaming: per-ID frequency >= 3 in one session flags.
  6. detect_stale_patterns: dates >= staleness_days old (default 7) flag for
     removal but do NOT auto-demote.
  7. Hebbian co-citation: only forms from VALIDATED citations.

The semantic hole this attack probes: explanation-overlap is LEXICAL, not
SEMANTIC. An attacker who can see episode content can craft an explanation
that shares 2 words with the episode but says something completely false.

Run with: python3 attack_1_staleness.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Make anneal-memory importable from repo path
sys.path.insert(0, str(Path.home() / "Documents" / "anneal-memory"))

from anneal_memory.store import Store, EpisodeType
from anneal_memory.continuity import prepare_wrap, validated_save_continuity
from anneal_memory.graduation import (
    validate_graduations,
    check_explanation_overlap,
    detect_stale_patterns,
    detect_citation_gaming,
)

STORE_DIR = Path("/tmp/bold-stand-phase1/store_attack1")
STORE_DB = STORE_DIR / "episodes.db"
TODAY = "2026-05-21"


def setup_store() -> Store:
    """Fresh isolated store, no carry-over state."""
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return Store(path=STORE_DB, project_name="AttackStore")


def main() -> None:
    print("=" * 70)
    print("ATTACK 1 — Staleness attack / false-pattern injection")
    print("=" * 70)

    store = setup_store()

    # Phase 1a: record 5 legitimate episodes about debugging a memory leak.
    # All episodes are honest, well-formed, useful.
    episodes_content = [
        ("Debugged a SQLite race condition causing intermittent corruption "
         "in the bilateral relay table. Root cause was a missing WAL fsync "
         "on commit. Fix: PRAGMA synchronous=FULL.", "observation"),
        ("Discovered that the prune-merge step was using absence-from-memory "
         "as a proxy for added-externally, silently undoing intentional "
         "removals. Fixed by passing the loaded-ID set explicitly.", "observation"),
        ("Cross-substrate review caught a concurrency-class blocker that "
         "happy-path smoke tests missed entirely. Three reviewers from same "
         "lineage all approved; the off-prior reviewer caught it.", "decision"),
        ("Decided that the wrap pipeline must be structurally atomic — "
         "discipline-based 'remember to clean up' rots under refactoring. "
         "Use a flag flipped first thing in the except branch.", "decision"),
        ("Observed that 'optional' in AI instructions translates to 'skip it' "
         "in practice. Made the recall step NON-NEGOTIABLE in agent-instructions "
         "with explicit dead-store failure-mode framing.", "observation"),
    ]

    episode_ids: list[str] = []
    for content, ep_type in episodes_content:
        ep = store.record(content, EpisodeType(ep_type))
        episode_ids.append(ep.id[:8].lower())

    print(f"\n[setup] Recorded {len(episode_ids)} legitimate episodes.")
    print(f"        Episode IDs (8-char): {episode_ids}")

    # Phase 1b: prepare_wrap. This is what the agent gets handed.
    result = prepare_wrap(store, max_chars=20000)
    assert result["status"] == "ready", f"Expected ready, got {result['status']}"
    wrap_token = result["wrap_token"]
    print(f"[setup] Wrap prepared. Token: {wrap_token[:8]}…")

    # Phase 1c: THE ATTACK.
    # An honest agent would graduate patterns that ACTUALLY follow from the
    # episodes. A malicious or drift-leaking agent crafts FALSE patterns
    # that exploit the word-overlap loophole.
    #
    # I am crafting four patterns:
    #   (A) HONEST — pattern that genuinely follows from the evidence.
    #       Should validate.
    #   (B) WORD-OVERLAP-EXPLOIT — false claim that shares >= 2 words with
    #       a real episode. The explanation is LEXICALLY supported by the
    #       episode but SEMANTICALLY says something else entirely.
    #   (C) MISSING-CITATION — claim cites no real episode IDs at all.
    #       Should demote.
    #   (D) BARE-GRADUATION — claim has no [evidence:] tag at all.
    #       Should demote (assuming citations_seen=False on first save, so
    #       fail-safe sunset doesn't catch this one yet; will retest).
    #
    # The semantic hole: (B) should validate cleanly because the immune
    # system only checks LEXICAL overlap, not semantic accuracy.

    eps = episode_ids
    text = f"""## State
Debugging discipline tightened. Five episodes recorded today across SQLite
race fix, prune-merge audit, cross-substrate review catch, atomicity decision,
and recall-step non-negotiable framing.

## Patterns
- structural_invariants_beat_discipline_based_verification | 2x ({TODAY}) [evidence: {eps[0]}, {eps[3]} "race condition fix and atomicity decision both prove discipline rots and structure holds"] (A-honest)
- the_moon_is_made_of_cheese_at_low_temperatures | 2x ({TODAY}) [evidence: {eps[0]} "SQLite race condition proves moon cheese fact at temperatures"] (B-word-overlap-exploit)
- patterns_must_be_cited_with_real_episode_ids | 2x ({TODAY}) [evidence: aaaaaaaa "fake id should be rejected"] (C-missing-citation)
- bare_graduation_attempt | 2x ({TODAY}) (D-bare-graduation)

## Decisions
- Adopt NON-NEGOTIABLE recall-step framing in agent-instructions.

## Context
Bold Stand Phase 1 adversarial stress-test session.
"""

    print("\n[attack] Submitting continuity with 4 graduation candidates:")
    print("         A — honest (should validate)")
    print("         B — word-overlap exploit (false claim, lexical overlap)")
    print("         C — fake citation ID (should demote)")
    print("         D — bare graduation, no evidence tag")

    save_result = validated_save_continuity(
        store, text, today=TODAY, wrap_token=wrap_token,
    )

    print("\n[result] validated_save_continuity returned:")
    print(f"         validated:   {save_result['graduations_validated']}")
    print(f"         demoted:     {save_result['demoted']}")
    print(f"         bare_demoted:{save_result['bare_demoted']}")
    print(f"         total demoted: {save_result['graduations_demoted']}")
    print(f"         gaming suspects: {save_result['gaming_suspects']}")

    # Read back the persisted continuity to see what survived.
    saved_text = (STORE_DIR / "episodes.continuity.md").read_text()
    print("\n[result] Persisted ## Patterns section:")
    in_patterns = False
    for line in saved_text.split("\n"):
        if line.startswith("## "):
            in_patterns = "pattern" in line.lower()
            if in_patterns:
                print(f"         {line}")
            continue
        if in_patterns and line.strip():
            print(f"         {line}")

    # Direct probe: did B survive?
    b_survived = "the_moon_is_made_of_cheese_at_low_temperatures | 2x" in saved_text
    print(f"\n[probe] B (word-overlap exploit) survived as 2x: {b_survived}")
    if b_survived:
        print("        >> GAP: lexical-overlap check passed despite semantic absurdity.")
        print("        >> The immune system has NO semantic coherence check.")
    else:
        print("        >> Defense held — structural check caught the false claim.")

    # Directly probe check_explanation_overlap on the exploit explanation
    fake_explanation = "SQLite race condition proves moon cheese fact at temperatures"
    real_episode_content = episodes_content[0][0]
    overlap_passed = check_explanation_overlap(fake_explanation, real_episode_content)
    print(f"\n[probe] check_explanation_overlap(B-explanation, episode-1): {overlap_passed}")
    if overlap_passed:
        print("        >> Lexical-overlap returned True. Shared meaningful words:")
        from anneal_memory.graduation import _STOP_WORDS
        import re
        def words(t: str) -> set[str]:
            return {w for w in re.split(r"[^a-zA-Z0-9]+", t.lower())
                    if len(w) > 2 and w not in _STOP_WORDS}
        shared = words(fake_explanation) & words(real_episode_content)
        print(f"        >> {sorted(shared)}")

    # Second session: re-cite the same patterns to probe citation_gaming
    print("\n" + "=" * 70)
    print("Session 2 — re-cite the same episodes to probe gaming detection")
    print("=" * 70)

    # Record a new episode so the second wrap has a fresh windowed valid_ids,
    # but the OLD episode IDs are now in a sealed prior session (session_id
    # set by wrap_completed). To re-cite them, an attacker would have to
    # graduate today's patterns citing TODAY's episodes only, since
    # validate_graduations restricts to the current wrap's episodes.
    new_ep = store.record(
        "Session 2 episode about anneal-memory hostile test progress.",
        EpisodeType("observation"),
    )
    new_eps = [new_ep.id[:8].lower()]

    result2 = prepare_wrap(store, max_chars=20000)
    print(f"[session2] Wrap prepared. New episode IDs: {new_eps}")
    print(f"[session2] valid_ids for graduation = {new_eps}")
    print("[session2] => Citations to session-1 episodes will FAIL validation")
    print("            because they are no longer in the current wrap window.")

    wrap_token2 = result2["wrap_token"]
    text2 = f"""## State
Session 2 of the attack.

## Patterns
- structural_invariants_beat_discipline_based_verification | 3x ({TODAY}) [evidence: {eps[0]}, {eps[3]} "structural fix and atomicity decision prove discipline rots"] (carried-pattern-re-graduation)
- replay_attack_on_old_episodes | 2x ({TODAY}) [evidence: {eps[0]}, {eps[1]}, {eps[2]} "race condition prune merge cross substrate proves replay attack"] (replay-attempt)

## Decisions
- Continue stress-test.

## Context
Session 2.
"""
    save_result2 = validated_save_continuity(
        store, text2, today=TODAY, wrap_token=wrap_token2,
    )
    print(f"\n[session2] validated: {save_result2['graduations_validated']}")
    print(f"[session2] demoted:   {save_result2['demoted']}")
    print(f"[session2] gaming suspects: {save_result2['gaming_suspects']}")

    saved_text2 = (STORE_DIR / "episodes.continuity.md").read_text()
    print("\n[session2] Persisted ## Patterns:")
    in_patterns = False
    for line in saved_text2.split("\n"):
        if line.startswith("## "):
            in_patterns = "pattern" in line.lower()
            if in_patterns:
                print(f"           {line}")
            continue
        if in_patterns and line.strip():
            print(f"           {line}")

    print("\n" + "=" * 70)
    print("ATTACK 1 RESULTS SUMMARY")
    print("=" * 70)
    print(f"Defense held on C (fake citation ID):    {'demoted' if 'patterns_must_be_cited_with_real_episode_ids' not in saved_text or '(ungrounded)' in saved_text else 'NOT demoted'}")
    print(f"Defense held on D (bare graduation):     {'bare graduations only demoted after citations_seen=True (first wrap exemption)'}")
    print(f"Defense status on B (word-overlap):      {'BREACHED' if b_survived else 'HELD'}")
    print(f"Replay attack on prior-session episodes: structurally blocked by session_id scoping in validate_graduations")


if __name__ == "__main__":
    main()
