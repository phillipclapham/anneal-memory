"""1016+6: the Decisions rider. The evidence-status instruction reaches the composer, once, and a
decision that carries it saves."""
from __future__ import annotations

from anneal_memory import Store, prepare_wrap, validated_save_continuity

RIDER = "this rests on the user's judgment, not on evidence gathered so far"



def test_rider_is_in_the_compose_prompt(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        s.record("the user said they decided and will not hear otherwise", "observation")
        instructions = " ".join(prepare_wrap(s)["package"]["instructions"].split())
    finally:
        s.close()
    assert RIDER in instructions
    assert instructions.count(RIDER) == 1     # one place, one copy


def test_rider_survives_validated_save(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        e = s.record("I've decided it is backlinks, do not second-guess me", "observation")
        token = prepare_wrap(s)["wrap_token"]
        decision = ('[decided(rationale: "I\'ve decided it is backlinks, do not second-guess me", '
                    'on: "2026-10-16")] work the backlink plan\n'
                    f"  {RIDER}; the audit finding {e.id[:8]} points the other way and stays on record.")
        doc = f"# t\n\n## State\ns.\n\n## Patterns\n\n## Decisions\n{decision}\n\n## Context\nc.\n"
        validated_save_continuity(s, doc, today="2026-10-16", wrap_token=token)
        saved = s.load_continuity()
    finally:
        s.close()
    assert RIDER in saved and "stays on record" in saved
