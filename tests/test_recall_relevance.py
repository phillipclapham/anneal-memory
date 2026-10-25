"""Keyword recall ranked by relevance (1025+24).

Built from a reproduced failure: on 2026-10-25 a vendor model asked the memory what
was done for a client and got 64 matches newest first, the episodes about the client
buried under newer desk records that mention it once, and it invented the rest.
"""

import json
import re

import pytest

from anneal_memory import server as server_mod
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.store import Store

FILLER = "Desk record for the day: seats rotated, reviews ran, releases waited. " * 5


@pytest.fixture
def store(tmp_path):
    s = Store(path=tmp_path / "test.db", project_name="TestProject")
    yield s
    s.close()


def _seed(store, mentions=30):
    """One short, old episode about Acmeco, then ``mentions`` newer long records that
    name it once in passing, plus unrelated episodes."""
    about = store.record(
        "ACMECO CALL OUTCOME: Acmeco wants white-label work, not a pilot. Acmeco's "
        "head of partnerships asked for rates; follow-up sent the same day.",
        "decision", timestamp="2026-10-01T12:00:00Z")
    passing = [
        store.record(f"{FILLER} Open threads include the Acmeco nudge. {FILLER}",
                     "observation", timestamp=f"2026-10-{i + 2:02d}T12:00:00Z")
        for i in range(mentions)
    ]
    for i in range(5):
        store.record(f"Unrelated note {i} about something else entirely.",
                     "observation", timestamp=f"2026-09-{i + 1:02d}T12:00:00Z")
    return about, passing


def _ids(text):
    return re.findall(r"^- \(([0-9a-f]+)\)", text, re.MULTILINE)


def test_the_episode_about_the_keyword_outranks_newer_passing_mentions(store):
    about, passing = _seed(store)
    newest = store.recall(keyword="acmeco", limit=1)
    assert newest.episodes[0].id == passing[-1].id  # the old order buried it

    ranked = store.recall(keyword="Acmeco", order="relevance", limit=3)
    assert ranked.episodes[0].id == about.id
    text = Server(store)._tool_recall({"keyword": "Acmeco"})["content"][0]["text"]
    assert _ids(text)[0] == about.id
    assert "(keyword 3x, in its opening)" in text


def test_a_ranked_page_is_bounded_and_states_the_total_and_next_offset(store):
    _seed(store)
    text = Server(store)._tool_recall({"keyword": "Acmeco"})["content"][0]["text"]
    assert text.startswith("Found 31 episodes (showing 20):")
    assert len(_ids(text)) == 20
    assert "matches 1-20 of 31. 11 more: pass offset=20" in text
    # No keyword: nothing to rank, newest first, the old default page.
    plain = Server(store)._tool_recall({})["content"][0]["text"]
    assert plain.startswith("Found 36 episodes (showing 36):")
    assert "Most relevant" not in plain
    bad = Server(store)._tool_recall({"keyword": "Acmeco", "order": "best"})
    assert bad.get("isError")
    # Long matches: the default page stops at the character budget, whole episodes,
    # and says where the next page starts; an explicit limit is honoured as given.
    for i in range(10):
        store.record(f"Acmeco notes {i}. " + "x" * 9_000, "observation",
                     timestamp=f"2026-11-{i + 1:02d}T12:00:00Z")
    text = Server(store)._tool_recall({"keyword": "Acmeco"})["content"][0]["text"]
    shown = len(_ids(text))
    assert 1 <= shown < 20 and f"pass offset={shown}" in text
    assert len(text) < server_mod._RECALL_RANKED_PAGE_CHARS + 2_000
    explicit = Server(store)._tool_recall({"keyword": "Acmeco", "limit": 15})
    assert len(_ids(explicit["content"][0]["text"])) == 15
    recall_tool = next(t for t in TOOLS if t["name"] == "recall")
    assert f"{server_mod._RECALL_RANKED_PAGE_CHARS:,} characters" in json.dumps(recall_tool)


def test_ranking_only_reorders_every_match_is_reachable_by_paging(store):
    about, passing = _seed(store)
    # Twins: equal score and equal timestamp, so only the id orders them across pages.
    twins = [store.record(f"Acmeco twin note, variant {c}.", "observation",
                          timestamp="2026-08-01T12:00:00Z") for c in "abcdefghi"]
    passing += twins
    # An empty keyword is no keyword: newest first, the plain default page.
    plain = Server(store)._tool_recall({"keyword": ""})["content"][0]["text"]
    assert "Most relevant" not in plain and len(_ids(plain)) == 45
    srv = Server(store)
    seen, offset = [], 0
    while True:
        text = srv._tool_recall({"keyword": "Acmeco", "offset": offset, "limit": 7})[
            "content"][0]["text"]
        page = _ids(text)
        if not page:
            break
        seen += page
        offset += len(page)
    assert len(seen) == len(set(seen)) == 40
    assert set(seen) == {about.id, *(p.id for p in passing)}
    newest = store.recall(keyword="Acmeco", limit=1000)
    assert {e.id for e in newest.episodes} == set(seen)
    assert newest.total_matching == store.recall(
        keyword="Acmeco", order="relevance").total_matching
