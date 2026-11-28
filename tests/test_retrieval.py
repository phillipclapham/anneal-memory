"""Tests for the keyword and corpus-IDF half of the search path (retrieval.py).

Covers keyword extraction and the corpus-IDF weights ``search_episodes`` ranks by.
The search itself is covered in ``tests/test_typed_query.py``.
"""

from __future__ import annotations

import pytest

from anneal_memory import Store, extract_keywords
from anneal_memory.types import EpisodeType
from anneal_memory import retrieval as _retrieval


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mem.db")
    yield s
    s.close()


# -- keyword extraction ----------------------------------------------------


class TestExtractKeywords:
    def test_drops_stopwords_and_short(self):
        kws = extract_keywords("the apparatus is a routing drift")
        assert "apparatus" in kws and "routing" in kws and "drift" in kws
        assert "the" not in kws and "is" not in kws and "a" not in kws

    def test_preserves_snake_case_and_hyphen(self):
        kws = extract_keywords("the cross_substrate review and non-replaceable codex")
        assert "cross_substrate" in kws
        assert "non-replaceable" in kws

    def test_dedup_order_preserving(self):
        kws = extract_keywords("codex codex apparatus codex")
        assert kws == ["codex", "apparatus"]

    def test_caps_breadth(self):
        many = " ".join(f"term{i}word" for i in range(40))
        assert len(extract_keywords(many)) <= _retrieval.MAX_KEYWORDS


# -- corpus-IDF keyword weighting (the precision fix, spore-141/spore-104 dep-2) ----


class TestCorpusIDF:
    """On a real-sized corpus, a high-document-frequency PROCESS word is deflated by
    corpus-IDF so it cannot rank an episode it merely brushed; below ``IDF_MIN_CORPUS``
    the length-proxy is used (so small fixtures stay byte-unchanged)."""

    def _seed_big_corpus(self, store, n=60):
        # Two unique per-episode tokens (`qa{i}z`/`qb{i}z`) — the trailing `z` makes them
        # collision-free under substring match (`qa3z` is NOT inside `qa30z`), so their
        # DF is exactly 1, unlike a bare `widget_3` which substrings `widget_30..39`.
        ids = []
        for i in range(n):
            e = store.record(
                f"A session convo note number {i} about qa{i}z and qb{i}z in the ongoing "
                f"build, recorded with enough body to clear the episode length floor.",
                EpisodeType.DECISION, source="flow",
                timestamp=f"2026-06-05T{i // 60:02d}:{i % 60:02d}:00Z",
            )
            ids.append(e.id)
        return ids

    def test_idf_weight_floors_ubiquitous_lifts_rare(self):
        # A term in (nearly) every episode floors; a unique term tops out near 1.0.
        assert _retrieval._idf_weight(1000, 1000) == _retrieval.IDF_FLOOR  # in ALL → floor
        assert _retrieval._idf_weight(1, 1000) > 0.8                       # unique → high
        assert _retrieval._idf_weight(500, 1000) < _retrieval._idf_weight(5, 1000)

    def test_query_weights_uses_idf_above_min_corpus(self, store):
        self._seed_big_corpus(store, n=60)
        assert store.recall(limit=0).total_matching >= _retrieval.IDF_MIN_CORPUS
        weights, used_idf = _retrieval._query_weights(
            store, ["session", "qa3z"], {"session": 60, "qa3z": 1}
        )
        assert used_idf is True
        # the ubiquitous word is floored; the rare one outweighs it decisively.
        assert weights["session"] == pytest.approx(_retrieval.IDF_FLOOR, abs=0.01)
        assert weights["qa3z"] > weights["session"]

    def test_query_weights_falls_back_below_min_corpus(self, store):
        store.record(
            "One apparatus-and-verification episode, comfortably past the length floor "
            "so it is a valid candidate but the corpus stays tiny.",
            EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-05T09:00:00Z",
        )
        weights, used_idf = _retrieval._query_weights(
            store, ["apparatus", "verification"], {"apparatus": 1, "verification": 1}
        )
        assert used_idf is False  # below IDF_MIN_CORPUS → length-proxy
        assert weights["apparatus"] == _retrieval._keyword_weight("apparatus")
