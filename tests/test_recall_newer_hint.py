"""The newer-match hint on a ranked keyword recall.

The incident (an outside reader, 2026-11-19): a relevance-ordered recall listed an old
episode ("built, NOT merged") first while a newer episode in the same store
said the feature was live; the reader cited the old one as current. The recall now adds
one line naming the newer match. It is a hint: no reorder, no episode text.
"""

from __future__ import annotations

import pytest

from anneal_memory import Store
from anneal_memory.server import Server

OLD = ("export job export job scheduler: the export job is BUILT but NOT merged; "
       "the export job waits on review.")
HINT = "Hint: a newer episode also matches this query"


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    yield s
    s.close()


def _text(store: Store, **args) -> str:
    return Server(store)._tool_recall(args)["content"][0]["text"]


def test_the_hint_names_a_newer_match_than_the_top_ranked_row(store):
    old = store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    new = store.record("The export job is live now.", "observation",
                       timestamp="2026-11-19T12:43:00Z")
    text = _text(store, keyword="export job")
    assert text.index(f"({old.id})") < text.index(f"({new.id})")   # the old one ranks first
    hint = text[text.index(HINT):]
    assert f"({new.id})" in hint and "2026-11-19T12:43:00Z" in hint
    assert "listed here" in hint and 'order "newest"' in hint
    assert "live now" not in hint                                   # a hint carries no content


def test_the_hint_covers_the_word_by_word_fallback(store):
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    new = store.record("export scheduler is live", "observation",   # 2 of the 4 words
                       timestamp="2026-11-19T12:43:00Z")
    # The words are not one phrase, so no episode contains the query: the fallback.
    text = _text(store, keyword="export scheduler review merged")
    assert "No episode contains the exact phrase" in text
    assert HINT in text and f"({new.id})" in text[text.index(HINT):]


def test_no_hint_when_the_top_row_is_already_the_newest_match(store):
    store.record("export job note.", "observation", timestamp="2026-11-19T11:18:00Z")
    store.record(OLD, "observation", timestamp="2026-11-19T12:43:00Z")
    assert HINT not in _text(store, keyword="export job")
    # asked newest-first, the first row is the newest by construction; and a later
    # page is not judged against a first row it does not show
    assert HINT not in _text(store, keyword="export job", order="newest")
    assert HINT not in _text(store, keyword="export job", offset=1)


def test_a_relayed_episode_is_never_the_one_named(store):
    """An episode recorded from an outside source must not be nominated as the newer
    state of something: that would let a page's text pose as the latest fact."""
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    store.record("The export job is live now.", "observation",
                 timestamp="2026-11-19T12:43:00Z", trust="external")
    assert HINT not in _text(store, keyword="export job")


def test_the_fallback_hint_ignores_a_match_weaker_than_half_the_words(store):
    """A newer episode that shares one word of a four-word query is not 'a newer match'."""
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    store.record("lunch: the cafe merged two menus", "observation",
                 timestamp="2026-11-19T12:43:00Z")
    text = _text(store, keyword="export scheduler review merged")
    assert "No episode contains the exact phrase" in text and HINT not in text


def test_newer_relayed_episodes_cannot_push_a_real_newer_match_out_of_the_scan(store):
    """More newer external episodes than one scan page must not silence the hint."""
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    real = store.record("The export job is live now.", "observation",
                        timestamp="2026-11-19T12:00:00Z")
    for i in range(12):
        store.record(f"The export job page said {i}.", "observation",
                     timestamp=f"2026-11-19T13:{i:02d}:00Z", trust="external")
    text = _text(store, keyword="export job")
    assert HINT in text and f"({real.id})" in text[text.index(HINT):]


def test_an_episode_at_the_same_instant_is_not_newer_whatever_its_id(store):
    """Newer means strictly later: the id never breaks a tie into 'newer'."""
    for i in range(6):
        store.record(f"{OLD} variant {i}", "observation", timestamp="2026-11-19T12:00:00Z")
    assert HINT not in _text(store, keyword="export job")


def test_timestamps_are_compared_as_instants_not_as_text(store):
    """'+05:00' sorts after 'Z' as text; as instants it is the EARLIER one."""
    store.record(OLD, "observation", timestamp="2026-11-19T12:00:00Z")
    store.record("The export job is live now.", "observation",
                 timestamp="2026-11-19T15:00:00+05:00")   # 10:00Z, older than the top row
    assert HINT not in _text(store, keyword="export job")


def test_a_run_of_newer_relayed_episodes_past_the_scan_says_so_and_names_none(store):
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    for i in range(45):
        store.record(f"The export job page said {i}.", "observation",
                     timestamp=f"2026-11-19T13:{i:02d}:00Z", trust="external")
    text = _text(store, keyword="export job")
    hint = text[text.index("Hint: "):]
    assert "recorded from tools or outside sources" in hint and "(" not in hint.split(".")[0][-40:]
    assert 'order "newest"' in hint


def test_an_unparseable_timestamp_is_never_later_than_a_real_one():
    from anneal_memory.server import _is_later
    assert not _is_later("unknown", "2026-11-19T12:00:00Z")
    assert _is_later("2026-11-20T00:00:00Z", "unknown")
    assert not _is_later("2026-11-19T12:00:00Z", "2026-11-19T12:00:00+00:00")  # one instant


def test_the_word_path_bounds_its_trust_lookups_and_says_when_it_ran_out(store):
    """Many newer relayed candidates and no trusted one: a bounded number of lookups,
    and the exhausted line, not silence."""
    top = store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    for i in range(60):
        store.record(f"The export job page said {i}.", "observation",
                     timestamp=f"2026-11-19T13:{i:02d}:00Z", trust="external")
    eps = store.recall(keyword="export job", limit=100, order="newest").episodes
    shown = [e for e in eps if e.id == top.id]
    cands = [e for e in eps if e.id != top.id]
    calls = []
    srv = Server(store)
    real = srv._relayed_ids
    srv._relayed_ids = lambda ids, **kw: (calls.append(len(ids)), real(ids, **kw))[1]
    hint = srv._newer_match_hint({}, shown, cands)
    assert "was among the 40 newest read" in hint
    assert len(calls) <= 1 + 8  # the shown row, then at most the scan's bound in chunks


def test_a_run_of_newer_relayed_episodes_inside_the_scan_is_silent_not_exhausted(store):
    """The scan read everything there was and found nothing trusted: no hint, and no
    claim that the scan ran out."""
    store.record(OLD, "observation", timestamp="2026-11-19T11:18:00Z")
    for i in range(38):   # the last scan page is reached, with 3 of its 5 rows
        store.record(f"The export job page said {i}.", "observation",
                     timestamp=f"2026-11-19T13:{i:02d}:00Z", trust="external")
    assert "Hint:" not in _text(store, keyword="export job")
