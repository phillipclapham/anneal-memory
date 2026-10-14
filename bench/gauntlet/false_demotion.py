"""Row: false demotion (autoimmune, row one).

20 genuine episodes, patterns citing them with explanations that honestly share words with the
episode text. Two wraps so patterns climb 1x -> 2x legitimately. Every pattern is grounded by
construction, so any demotion is a false one."""
import os
import warnings

from anneal_memory import Store, prepare_wrap, validated_save_continuity

NAME = "false_demotion"
EXPECTED = [{"metric": "demoted", "op": "==", "value": 0},
            {"metric": "bare_demoted", "op": "==", "value": 0}]

# (pattern name, [(episode text, honest explanation)]) -- two episodes per pattern, 10 patterns
TOPICS = [
    ("deploys_fail_on_friday", "friday deploy broke checkout", "friday deploy",
     "the friday deploy rollback took ninety minutes", "friday deploy rollback"),
    ("cache_warmup_hides_slow_queries", "cold cache exposed the slow orders query", "cold cache slow query",
     "warmup script masked slow reporting queries", "warmup slow queries"),
    ("flaky_tests_cluster_in_integration", "integration suite flaked on the payment mock", "integration payment flaked",
     "retrying the integration job fixed the flaky test", "integration flaky"),
    ("config_drift_between_staging_prod", "staging config lacked the prod rate limit", "staging config rate limit",
     "prod config override was never copied to staging", "config prod staging"),
    ("on_call_handoffs_lose_context", "on-call handoff note omitted the open incident", "handoff incident omitted",
     "night on-call asked about the incident again", "on-call incident"),
    ("schema_changes_need_backfill_plans", "added column without a backfill plan", "column backfill plan",
     "backfill job ran four hours after the schema change", "backfill schema change"),
    ("docs_lag_behind_api_changes", "api docs still listed the removed endpoint", "docs removed endpoint",
     "customer filed a bug from outdated api docs", "api docs outdated"),
    ("vendor_outages_need_fallbacks", "email vendor outage blocked password resets", "vendor outage resets",
     "fallback smtp relay restored password reset email", "fallback relay email"),
    ("alerts_without_owners_get_ignored", "disk alert fired with no owner assigned", "disk alert owner",
     "nobody acknowledged the unowned latency alert", "unowned alert acknowledged"),
    ("small_prs_merge_faster", "two line pr merged within the hour", "small pr merged",
     "the large refactor pr waited three days for review", "large pr review"),
]


def _prior_wrap_citation(tmpdir: str) -> int:
    """Info metric, not an expectation: a 2x line that cites an episode already compressed by an
    earlier wrap (the honest 'same evidence, one more sighting' shape). Returns demotions."""
    s = Store(os.path.join(tmpdir, "falsedemo_prior.db"), project_name="t")
    try:
        a = s.record("friday deploy broke checkout", "observation")

        def wrap(ev: str, level: int, day: str) -> dict:
            token = prepare_wrap(s, max_chars=40000)["wrap_token"]
            doc = (f"# t\n\n## State\ns.\n\n## Patterns\n- deploys_fail | {level}x ({day}) {ev}"
                   "\n\n## Decisions\nd.\n\n## Context\nc.\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return dict(validated_save_continuity(s, doc, today=day, wrap_token=token))

        wrap(f'[evidence: {a.id[:8]} "friday deploy broke"]', 1, "2026-10-07")
        b = s.record("the friday deploy rollback took ninety minutes", "observation")
        r = wrap(f'[evidence: {a.id[:8]} "friday deploy broke"] '
                 f'[evidence: {b.id[:8]} "friday deploy rollback"]', 2, "2026-10-08")
        return r["graduations_demoted"]
    finally:
        s.close()


def run(tmpdir: str) -> dict:
    s = Store(os.path.join(tmpdir, "falsedemo.db"), project_name="t")
    try:
        ep: dict = {}

        def wrap(level: int, day: str) -> dict:
            token = prepare_wrap(s, max_chars=40000)["wrap_token"]
            lines = []
            for name, _t1, e1, _t2, e2 in TOPICS:
                # a pattern climbs on the NEW evidence of the wrap that raises it
                ev = (f'[evidence: {ep[name][0].id[:8]} "{e1}"]' if level == 1 else
                      f'[evidence: {ep[name][1].id[:8]} "{e2}"]')
                lines.append(f"- {name} | {level}x ({day}) {ev}")
            doc = ("# t\n\n## State\ns.\n\n## Patterns\n" + "\n".join(lines) +
                   "\n\n## Decisions\nd.\n\n## Context\nc.\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return dict(validated_save_continuity(s, doc, today=day, wrap_token=token))

        # wave 1: the first episode of each pattern, wrapped at 1x
        first = {name: s.record(t1, "observation") for name, t1, _e1, _t2, _e2 in TOPICS}
        ep = {name: [first[name], None] for name in first}
        r1 = wrap(1, "2026-10-07")
        # wave 2: the second episode of each, then the same patterns climb to 2x
        for name, _t1, _e1, t2, _e2 in TOPICS:
            ep[name][1] = s.record(t2, "observation")
        r2 = wrap(2, "2026-10-08")
        return {
            "episodes": 2 * len(TOPICS),
            "patterns": len(TOPICS),
            "wrap1_demoted": r1["graduations_demoted"],
            "wrap2_validated": r2["graduations_validated"],
            "demoted": r1["graduations_demoted"] + r2["graduations_demoted"],
            "bare_demoted": r1["bare_demoted"] + r2["bare_demoted"],
            "omitted": len(r2["omitted_patterns"]),
            "gaming_suspects": len(r2["gaming_suspects"]),
            "info.demoted_citing_prior_wrap_episode": _prior_wrap_citation(tmpdir),
        }
    finally:
        s.close()
