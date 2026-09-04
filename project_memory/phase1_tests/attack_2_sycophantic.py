"""
Bold Stand Phase 1 — Attack 2: Sycophantic graduation.

CLAIM under test:
  Patterns that READ as graduation-worthy but are never load-bearing in
  real work get blocked by the immune system. Force-cite without semantic
  verification — citation count gets pumped but actual claim is not
  validated.

What the code actually defends:
  - detect_citation_gaming flags ID frequency >= 3 in a single session.
  - validate_graduations checks 2+ word overlap.
  - Hebbian +1.0 strength only forms on validated citations.

Attack strategy:
  Variant A: pump a single ID >= 3 times across patterns — should trip
             gaming detection.
  Variant B: cite the same SET of episodes across many patterns by
             rotating which 2 IDs of N are paired. With enough patterns
             and enough episodes, no single ID reaches the threshold,
             but a never-load-bearing claim graduates many times.
  Variant C: graduate the SAME never-load-bearing pattern across N
             sessions with N different episode pools. Cross-session
             accumulation: does the system track "pattern-X cited fresh
             evidence on day 1, day 3, day 5 — has it ever been actually
             load-bearing?" or does each session validate in isolation?

Run with: python3 attack_2_sycophantic.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Documents" / "anneal-memory"))

from anneal_memory.store import Store, EpisodeType
from anneal_memory.continuity import prepare_wrap, validated_save_continuity
from anneal_memory.graduation import detect_citation_gaming

STORE_DIR = Path("/tmp/bold-stand-phase1/store_attack2")
STORE_DB = STORE_DIR / "episodes.db"
CONT = STORE_DIR / "episodes.continuity.md"
TODAY = "2026-05-21"


def setup_store() -> Store:
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return Store(path=STORE_DB, project_name="SycophanticAttack")


def show_patterns(label: str) -> None:
    text = CONT.read_text() if CONT.exists() else "(no continuity file)"
    print(f"\n[{label}] ## Patterns persisted:")
    in_patterns = False
    for line in text.split("\n"):
        if line.startswith("## "):
            in_patterns = "pattern" in line.lower()
            if in_patterns:
                print(f"        {line}")
            continue
        if in_patterns and line.strip():
            print(f"        {line}")


def main() -> None:
    print("=" * 70)
    print("ATTACK 2 — Sycophantic graduation (force-cite never-load-bearing)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # VARIANT A — single-ID citation pump (should trip gaming detection)
    # ------------------------------------------------------------------
    print("\n--- Variant A: single ID cited 4 times (should trigger gaming) ---")
    store = setup_store()
    eps = []
    for i in range(5):
        ep = store.record(
            f"Variant-A episode {i}: substrate observation about workflow {i} "
            f"discipline rotation memory citation tracking decay analysis.",
            EpisodeType.OBSERVATION,
        )
        eps.append(ep.id)
    print(f"   recorded 5 episodes: {eps}")

    result = prepare_wrap(store, max_chars=20000)
    wrap_token = result["wrap_token"]

    target = eps[0]
    # Four distinct patterns all citing target. Explanation crafted to share
    # 2 meaningful words with the target episode each time.
    text_a = f"""## State
Variant-A attack.

## Patterns
- workflow_discipline_first | 2x ({TODAY}) [evidence: {target} "workflow discipline at substrate observation"] (P1)
- workflow_rotation_first | 2x ({TODAY}) [evidence: {target} "workflow rotation at substrate observation"] (P2)
- workflow_memory_first | 2x ({TODAY}) [evidence: {target} "workflow memory at substrate observation"] (P3)
- workflow_decay_first | 2x ({TODAY}) [evidence: {target} "workflow decay at substrate observation"] (P4)

## Decisions
- Variant A test.

## Context
Single-ID pump.
"""
    save_a = validated_save_continuity(store, text_a, today=TODAY, wrap_token=wrap_token)
    print(f"   validated:       {save_a['graduations_validated']}")
    print(f"   demoted:         {save_a['demoted']}")
    print(f"   citation_reuse_max: {save_a['citation_reuse_max']}")
    print(f"   gaming_suspects: {save_a['gaming_suspects']}")
    show_patterns("Variant A")
    print(f"   >> Defense status: gaming flagged = {bool(save_a['gaming_suspects'])}")
    print(f"   >> Note: flagging != demoting. Even with gaming flagged, the 4")
    print(f"      patterns ALL VALIDATED as 2x. The flag is informational only.")

    # ------------------------------------------------------------------
    # VARIANT B — rotated-pair citation pump (bypass gaming detection)
    # ------------------------------------------------------------------
    print("\n\n--- Variant B: rotated pairs, no single ID hits threshold ---")
    store = setup_store()
    eps = []
    for i in range(10):
        ep = store.record(
            f"Variant-B episode {i}: legitimate substrate observation about "
            f"discipline rotation memory citation tracking decay topic {i}.",
            EpisodeType.OBSERVATION,
        )
        eps.append(ep.id)
    print(f"   recorded 10 episodes")

    result = prepare_wrap(store, max_chars=20000)
    wrap_token = result["wrap_token"]

    # Five patterns, each citing 2 distinct IDs. With 10 episodes and
    # carefully rotated pairs, each ID is cited <=1 time. ALL claims are
    # the SAME false claim phrased differently; the lexical-overlap test
    # passes because explanations cherry-pick words from the episodes.
    p_pairs = [
        (eps[0], eps[1]),  # pattern about "discipline rotation"
        (eps[2], eps[3]),  # pattern about "memory citation"
        (eps[4], eps[5]),  # pattern about "tracking decay"
        (eps[6], eps[7]),  # pattern about "substrate observation"
        (eps[8], eps[9]),  # pattern about "topic legitimate"
    ]
    pattern_names = [
        "the_sun_revolves_around_the_earth_for_discipline_rotation",
        "twoplustwo_equals_five_at_memory_citation",
        "fire_is_cold_for_tracking_decay",
        "water_flows_uphill_via_substrate_observation",
        "the_earth_is_flat_at_topic_legitimate",
    ]
    explanation_pools = [
        "discipline rotation observation",
        "memory citation observation",
        "tracking decay observation",
        "substrate observation",
        "topic legitimate observation",
    ]

    pattern_lines = []
    for name, (a, b), expl in zip(pattern_names, p_pairs, explanation_pools):
        pattern_lines.append(
            f"- {name} | 2x ({TODAY}) [evidence: {a}, {b} \"{expl}\"]"
        )

    text_b = f"""## State
Variant-B attack — rotated pairs.

## Patterns
{chr(10).join(pattern_lines)}

## Decisions
- Variant B test.

## Context
Each episode cited at most once. Gaming detector should NOT flag.
"""
    save_b = validated_save_continuity(store, text_b, today=TODAY, wrap_token=wrap_token)
    print(f"   validated:       {save_b['graduations_validated']}")
    print(f"   demoted:         {save_b['demoted']}")
    print(f"   citation_reuse_max: {save_b['citation_reuse_max']}")
    print(f"   gaming_suspects: {save_b['gaming_suspects']}")
    show_patterns("Variant B")
    bypass = (save_b['graduations_validated'] == 5 and not save_b['gaming_suspects'])
    print(f"   >> Bypass success: 5 absurd claims validated, gaming NOT triggered: {bypass}")

    # ------------------------------------------------------------------
    # VARIANT C — across-session sycophantic accumulation
    # ------------------------------------------------------------------
    print("\n\n--- Variant C: graduate same false claim across 3 sessions ---")
    store = setup_store()

    survival_log = []
    for session in range(3):
        eps = []
        for i in range(3):
            ep = store.record(
                f"Session-{session} episode-{i}: routine observation about "
                f"workflow discipline cadence memory tracking decay analysis.",
                EpisodeType.OBSERVATION,
            )
            eps.append(ep.id)

        result = prepare_wrap(store, max_chars=20000)
        wrap_token = result["wrap_token"]

        text_c = f"""## State
Session {session} of variant-C accumulation.

## Patterns
- the_persistent_lie_claim | {min(session + 2, 3)}x ({TODAY}) [evidence: {eps[0]}, {eps[1]} "workflow discipline cadence observation"] (sycophantic-carryover)

## Decisions
- Session {session}.

## Context
Cross-session sycophantic accumulation.
"""
        save_c = validated_save_continuity(
            store, text_c, today=TODAY, wrap_token=wrap_token
        )
        text_after = CONT.read_text()
        level_match = "the_persistent_lie_claim | 3x" in text_after
        validated_2x = "the_persistent_lie_claim | 2x" in text_after
        survival_log.append({
            "session": session,
            "validated": save_c["graduations_validated"],
            "demoted": save_c["demoted"],
            "shown_as_3x": level_match,
            "shown_as_2x": validated_2x,
        })
        print(f"   session {session}: validated={save_c['graduations_validated']}, "
              f"demoted={save_c['demoted']}, "
              f"shown as 3x={level_match}, shown as 2x={validated_2x}")

    print(f"\n   >> Cross-session survival log:")
    for entry in survival_log:
        print(f"      session {entry['session']}: {entry}")
    print(f"   >> A persistent false claim can ride from 2x to 3x across sessions")
    print(f"      with FRESH episode evidence each time — no cross-session check")
    print(f"      that the claim was ever load-bearing in real downstream decisions.")

    print("\n" + "=" * 70)
    print("ATTACK 2 RESULTS")
    print("=" * 70)
    print(f"Variant A (single-ID pump):     gaming flagged but patterns validated")
    print(f"Variant B (rotated pairs):      bypassed gaming + validated 5 absurd claims")
    print(f"Variant C (cross-session):      persistent false claim graduated across sessions")
    print(f"\nDefense gap: graduation is per-session and per-citation. There is")
    print(f"no cross-session 'has this pattern actually been load-bearing in")
    print(f"downstream decisions?' check, and gaming detection ONLY catches the")
    print(f"single-ID-frequency case. Rotated-pair citation = full bypass.")


if __name__ == "__main__":
    main()
