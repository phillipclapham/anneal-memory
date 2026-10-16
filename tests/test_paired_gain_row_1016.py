"""1016+6: the paired_gain gauntlet row: skip without a key, report shape."""
from __future__ import annotations

from bench.gauntlet import paired_gain


def test_paired_gain_skips_without_key_and_without_opt_in(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANNEAL_BENCH_PAID", raising=False)
    both = " ".join(paired_gain.requires())
    assert "ANTHROPIC_API_KEY" in both and "ANNEAL_BENCH_PAID" in both
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")           # a key alone never spends
    assert [m for m in paired_gain.requires() if "ANNEAL_BENCH_PAID" in m]
    monkeypatch.setenv("ANNEAL_BENCH_PAID", "1")
    assert paired_gain.requires() == []


def _unit(i: int, none: int, cont: int) -> dict:
    arms = {a: {"score": 1, "context_chars": 100} for a in
            ("oracle", "full-dump", "anneal-recall", "paper-probe")}
    arms["none"] = {"score": none, "context_chars": 0}
    arms["anneal-continuity"] = {"score": cont, "context_chars": 300}
    arms["full-dump"]["context_chars"] = 3300
    return {"uid": f"b{i}", "arms": arms, "diag": {"recall": {"fact_retrieved": i == 0}},
            "wrap": {"anneal": {"saved": 27, "cancelled": 1 if i == 0 else 0}}}


def test_paired_gain_report_shape():
    out = paired_gain.summarize([_unit(0, 0, 1), _unit(1, 0, 1), _unit(2, 1, 1), _unit(3, 0, 0)])
    assert out["units"] == 4
    assert out["application.none"] == 0.25 and out["application.anneal-continuity"] == 0.75
    assert out["gain.continuity_minus_none"] == 0.5
    assert out["context_chars.anneal-continuity"] == 300 and out["context_chars.full-dump"] == 3300
    assert out["recall_fact_retrieved"] == 0.25
    assert out["wraps_saved"] == 108 and out["wraps_cancelled"] == 1 and out["unscored_answers"] == 0
    assert paired_gain.EXPECTED is None      # model output is reported, never pass/failed on a slice
