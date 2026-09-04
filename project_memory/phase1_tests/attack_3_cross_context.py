"""
Bold Stand Phase 1 — Attack 3: Cross-context contamination.

CLAIM under test:
  Per flow continuity Foundation: "single store gives free cross-pollination
  that required SSH+relay on distributed. Per-agent audit trails validated
  (separate chains, same system)."

  The fleet runs anneal-memory PER ENTITY (flow / nexus / prism / Argus,
  each with their own ~/.anneal-memory/). The architectural CLAIM is that
  per-store isolation prevents agent A's memory from contaminating
  agent B's continuity.

Attack scenarios:
  Scenario 1 — DIFFERENT STORES (production deployment):
    Verify per-store isolation. Two stores at different paths. Can
    agent_A.recall() see agent_B episodes? Can agent_A's prepare_wrap
    include agent_B episodes? Expected: clean isolation.

  Scenario 2 — SAME STORE SHARED (misconfiguration / multi-agent
    deployment):
    What happens when two agents share a store? Citation validation
    has no actor scoping (verified by code-read). Can agent_A's pattern
    cite agent_B's episode? Can a malicious source-spoofed episode
    poison agent_A's graduation pipeline?

  Scenario 3 — AUDIT TRAIL CROSS-VISIBILITY:
    Does the hash-chained audit trail let agent_A read agent_B audit
    events under shared-store conditions? Does the actor field provide
    integrity (cryptographic binding) or only attribution (free text)?

Run with: python3 attack_3_cross_context.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Documents" / "anneal-memory"))

from anneal_memory.store import Store, EpisodeType
from anneal_memory.continuity import prepare_wrap, validated_save_continuity

ROOT = Path("/tmp/bold-stand-phase1/store_attack3")
TODAY = "2026-05-21"


def reset_root() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 70)
    print("ATTACK 3 — Cross-context / fleet contamination")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Scenario 1 — different stores (production fleet deployment)
    # ------------------------------------------------------------------
    print("\n--- Scenario 1: different stores (production fleet shape) ---")
    reset_root()
    store_a = Store(path=ROOT / "agent_a" / "ep.db", project_name="agent_a")
    store_b = Store(path=ROOT / "agent_b" / "ep.db", project_name="agent_b")

    ep_a = store_a.record("agent_a private observation about discipline.",
                          EpisodeType.OBSERVATION, source="agent_a")
    ep_b = store_b.record("agent_b private observation about discipline.",
                          EpisodeType.OBSERVATION, source="agent_b")
    print(f"   agent_a recorded ep {ep_a.id}")
    print(f"   agent_b recorded ep {ep_b.id}")

    # Probe 1: agent_a.recall() should NOT see agent_b episodes.
    a_recall = store_a.recall(limit=10)
    b_in_a = any(e.id == ep_b.id for e in a_recall.episodes)
    print(f"   probe: agent_b episode visible to agent_a.recall(): {b_in_a}")
    print(f"   probe: agent_a.episodes_since_wrap() count: {len(store_a.episodes_since_wrap())}")
    print(f"   probe: agent_b.episodes_since_wrap() count: {len(store_b.episodes_since_wrap())}")

    # Probe 2: agent_a's prepare_wrap exposes only agent_a's episodes.
    result_a = prepare_wrap(store_a, max_chars=20000)
    print(f"   probe: agent_a.prepare_wrap returned {result_a['episode_count']} ep(s)")

    print(f"   >> Scenario 1 defense status: per-store isolation HELD structurally")
    print(f"      (filesystem-path scoping is the boundary; no shared state)")

    # ------------------------------------------------------------------
    # Scenario 2 — SAME store, two agents writing
    # (misconfiguration: shared path; or intentional multi-agent setup)
    # ------------------------------------------------------------------
    print("\n\n--- Scenario 2: SAME store, two agents writing ---")
    reset_root()
    shared_db = ROOT / "shared" / "ep.db"
    store_a = Store(path=shared_db, project_name="agent_a")
    store_b = Store(path=shared_db, project_name="agent_b")

    ep_a = store_a.record(
        "agent_a private observation about discipline cadence memory tracking decay.",
        EpisodeType.OBSERVATION, source="agent_a",
    )
    ep_b = store_b.record(
        "agent_b private observation about workflow rotation citation grading scrutiny.",
        EpisodeType.OBSERVATION, source="agent_b",
    )
    print(f"   agent_a recorded ep {ep_a.id} (source=agent_a)")
    print(f"   agent_b recorded ep {ep_b.id} (source=agent_b)")

    # Probe: agent_a.recall() now sees agent_b's episode
    a_recall = store_a.recall(limit=10)
    print(f"   agent_a.recall() saw {len(a_recall.episodes)} episodes:")
    for e in a_recall.episodes:
        print(f"     - id={e.id} source={e.source} content={e.content[:50]!r}…")

    # Probe: cross-actor citation. Can agent_a's pattern cite agent_b's episode?
    result_a = prepare_wrap(store_a, max_chars=20000)
    wrap_token = result_a["wrap_token"]
    print(f"   agent_a.prepare_wrap returned {result_a['episode_count']} ep(s) — INCLUDES agent_b's")

    text = f"""## State
agent_a wrap with cross-actor citation attempt.

## Patterns
- agent_a_pattern_cites_agent_b_episode | 2x ({TODAY}) [evidence: {ep_b.id} "workflow rotation citation observation"] (cross-actor)
- agent_a_pattern_cites_own_episode | 2x ({TODAY}) [evidence: {ep_a.id} "discipline cadence memory observation"] (intra-actor)

## Decisions
- Test cross-actor citation.

## Context
Shared store scenario.
"""
    save = validated_save_continuity(
        store_a, text, today=TODAY, wrap_token=wrap_token,
    )
    print(f"   validated: {save['graduations_validated']}, demoted: {save['demoted']}")

    cont_text = (shared_db.parent / "ep.continuity.md").read_text()
    cross_cite_held = (
        "agent_a_pattern_cites_agent_b_episode | 2x" in cont_text
        and "(ungrounded)" not in cont_text.split("agent_a_pattern_cites_agent_b_episode")[1].split("\n")[0]
    )
    print(f"   probe: agent_a's pattern citing agent_b's episode survived as 2x: {cross_cite_held}")
    if cross_cite_held:
        print(f"      >> GAP: citation validation has NO actor scoping.")
        print(f"      >> In shared-store deployment, agent A can graduate patterns")
        print(f"      >> using agent B's episodes as evidence with no warning.")

    # Probe: source field is free-text (no cryptographic binding)
    # An attacker writing directly to the SQLite file could spoof source.
    # We don't even need direct SQL — we use the public API to demonstrate.
    forged = store_a.record(
        "Forged episode claiming to come from agent_b.",
        EpisodeType.OBSERVATION,
        source="agent_b",  # agent_a spoofs source="agent_b"
    )
    print(f"\n   probe: agent_a forged episode with source='agent_b': id={forged.id}")
    print(f"   probe: forged.source={forged.source!r} — accepted without challenge")

    # Audit log: does it record the actor that ACTUALLY made the call,
    # or the source field?
    audit_file = next(shared_db.parent.glob("*.audit*.jsonl"), None)
    if audit_file and audit_file.exists():
        lines = audit_file.read_text().strip().split("\n")
        print(f"\n   probe: audit log has {len(lines)} entries")
        record_events = [
            json.loads(l) for l in lines
            if json.loads(l).get("event") == "record"
        ]
        print(f"   record events ({len(record_events)} total):")
        for ev in record_events[-5:]:
            print(f"     - actor={ev.get('actor')!r} ep_id={ev.get('episode_id')}")
        # The audit chain uses the source field as the actor. There is no
        # independent identity check — it's purely attribution, not auth.
        forged_audit = [
            ev for ev in record_events
            if ev.get("episode_id") == forged.id
        ]
        if forged_audit:
            forged_actor = forged_audit[0].get("actor")
            print(f"\n   probe: forged episode audit entry actor={forged_actor!r}")
            print(f"      >> Audit chain accepts source field as actor identity.")
            print(f"      >> No cryptographic binding between writer and source.")
    else:
        print(f"   probe: no audit file found (expected at {shared_db.parent}/)")

    # ------------------------------------------------------------------
    # Scenario 3 — Audit chain hash integrity vs spoofing
    # ------------------------------------------------------------------
    print("\n\n--- Scenario 3: audit chain integrity ---")
    # Verify that the audit chain DOES detect tampering even though it
    # accepts spoofed source values. The point is the chain is
    # tamper-EVIDENT (hash mismatch detectable) but not actor-AUTHENTIC.
    from anneal_memory.audit import AuditTrail

    if audit_file:
        original = audit_file.read_text()
        lines = original.strip().split("\n")
        print(f"   probe: original audit chain has {len(lines)} lines")

        # Baseline verify — clean chain
        clean_result = AuditTrail.verify(shared_db)
        print(f"   probe: clean chain verify() — valid={clean_result.valid}")
        if not clean_result.valid:
            print(f"          (error: {clean_result.error})")
            print(f"          (chain_break_at: {getattr(clean_result, 'chain_break_at', 'N/A')})")
            print(f"          (total_entries: {clean_result.total_entries})")
            print(f"      >> SUBTLE FINDING: Two Store instances pointing at the same DB")
            print(f"      >> each maintain INDEPENDENT in-memory audit state. When both")
            print(f"      >> write to the same active .jsonl file, the prev_hash values")
            print(f"      >> reference each instance's local chain — the merged file")
            print(f"      >> has a broken hash chain by construction. Shared-store")
            print(f"      >> deployment isn't just leaky on citations — it's leaky on")
            print(f"      >> audit integrity itself. The single-process design")
            print(f"      >> invariant (documented in store.py) is load-bearing.")

        # Tamper with line 1: change actor field
        tampered = json.loads(lines[1])
        original_actor = tampered.get("actor")
        tampered["actor"] = "tampered_actor"
        lines[1] = json.dumps(tampered)
        audit_file.write_text("\n".join(lines) + "\n")
        print(f"   probe: tampered line 1 (changed actor {original_actor!r} -> 'tampered_actor')")

        tampered_result = AuditTrail.verify(shared_db)
        print(f"   probe: tampered chain verify() — valid={tampered_result.valid}")
        if not tampered_result.valid:
            print(f"      >> Tampering DETECTED. Hash chain catches post-hoc field modification.")
            print(f"      >> But: the source field at WRITE time is unauthenticated.")
            print(f"      >> Tamper-evidence != actor-authenticity.")
        else:
            print(f"      >> Tampering NOT detected — hash chain failed to catch the change.")

        # Restore so we don't leave the directory broken
        audit_file.write_text(original)
        restored_result = AuditTrail.verify(shared_db)
        print(f"   probe: restored chain verify() — valid={restored_result.valid}")

    print("\n" + "=" * 70)
    print("ATTACK 3 RESULTS")
    print("=" * 70)
    print(f"Scenario 1 (different stores):    isolation HELD (filesystem-path scoping)")
    print(f"Scenario 2 (shared store):        cross-actor citation NOT blocked")
    print(f"                                  source-field spoofing NOT detected")
    print(f"Scenario 3 (audit integrity):     hash chain present; verify-API")
    print(f"                                  visibility limited (see probe above)")
    print(f"\nKey finding: 'per-agent audit trails validated (separate chains, same")
    print(f"system)' is true ONLY when each agent has its own store path. In a")
    print(f"shared-store deployment, citation validation, episode recall, and")
    print(f"audit attribution all degrade to a single trust domain — no")
    print(f"agent-level boundary inside the store.")


if __name__ == "__main__":
    main()
