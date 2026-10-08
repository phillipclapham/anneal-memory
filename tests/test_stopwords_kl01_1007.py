"""KL-01 (2026-10-07): the cross-session-overlap gate graded function words.

Built from two collisions recorded in flow's real save audit. Each demoted a real
pattern because today's explanation shared three "meaningful" words with the prior
one, and two of the three were function words ("only"/"while", "after"/"when").
The prior explanations below are the recorded ones, verbatim.
"""
from anneal_memory.graduation import (
    _SYCOPHANCY_STOP_WORDS, _meaningful_words, check_explanation_overlap, validate_graduations,
)

REAL = [
    ("the_measurement_is_of_the_wrong_object",
     "the Cloudflare Web Analytics beacon was grepped without browser headers and read "
     "as absent, and a game name was called clean from an App Store only search while "
     "Google Play and itch.io held it",
     "a release check grepped only the local tree while the tag pointed elsewhere"),
    ("recurring_class_closes_by_delete_or_bound_never_another_guard",
     "the review loop converged only when recurring classes were closed by delete or "
     "bound, and a fallback was deleted after every repair reopened its class",
     "after the seam spiral we chose to delete adoption when it came back"),
]


def _lookup(prior):
    # A level-up (2 -> 3): the warm-preservation exemption cannot hold the line, so the
    # overlap gate alone decides, as it did for the recorded demotions.
    return lambda name: {"max_level_reached": 2, "last_explanation": prior,
                         "last_seen_at": "2026-10-05T00:00:00Z", "last_wrap_id": None}


def _text(name, explanation):
    return (f"## State\n.\n## Patterns\n- {name} | 3x (2026-10-07) "
            f"[evidence: abc12345 \"{explanation}\"]\n## Decisions\n.\n## Context\n.\n")


def test_function_words_alone_do_not_trip_the_cross_session_gate():
    for name, prior, today in REAL:
        result = validate_graduations(
            text=_text(name, today), valid_ids={"abc12345"}, today="2026-10-07",
            node_content_map={"abc12345": today}, pattern_history_lookup=_lookup(prior))
        assert result.cross_session_collisions == [], name
        assert result.validated == 1 and result.demoted == 0, name


def test_function_words_do_not_count_for_the_cross_session_check():
    for w in ("only", "while", "after", "when", "own", "one", "two", "every", "than"):
        assert _meaningful_words(f"alpha {w} beta", stop=_SYCOPHANCY_STOP_WORDS) \
            == {"alpha", "beta"}, w


def test_grounding_keeps_half_content_words():
    """L2 1007 [run]: the longer list cost grounding on short ops explanations."""
    assert check_explanation_overlap("down again after restart",
                                     "the hub went down again after the restart at noon")
    assert check_explanation_overlap("two of three seats",
                                     "two of the three seats returned nothing")


def test_three_shared_content_words_still_trip():
    prior = "the seam spiral ended by deletion of the adoption block"
    today = "adoption spiral closed with deletion"
    result = validate_graduations(
        text=_text("p", today), valid_ids={"abc12345"}, today="2026-10-07",
        node_content_map={"abc12345": today}, pattern_history_lookup=_lookup(prior))
    assert len(result.cross_session_collisions) == 1
    assert "(cross-session-overlap)" in result.text
