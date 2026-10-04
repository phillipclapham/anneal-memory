"""0.9.33: a save above the schema's hard maximum is refused, loudly.

The bound is ``ceil(1.25 * default_max_chars(schema))``: 31,875 for flow's schema,
whose largest save on record (30,425, 2026-10-03) must still pass.
"""

from __future__ import annotations

import pickle

import pytest

from anneal_memory import (
    ContinuityValidationError, FLOW_SCHEMA, Store, prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.schema import DEFAULT_SCHEMA, default_max_chars, hard_max_chars
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

_FLOW_TEXT = """# Hard — Memory (v1)

## State
s

{durable}## Active Threads
t

## Patterns
- p | 1x (2026-10-04)

## Decisions
[decided(rationale: "x", on: "2026-10-04")] y

## Context
{pad}

## Understanding
u
"""


def _text(total, durable=""):
    """A FLOW_SCHEMA text whose size, the durable section excluded, is ``total``."""
    base = len(_FLOW_TEXT.format(pad="", durable=""))
    return _FLOW_TEXT.format(pad="c" * (total - base), durable=durable)


def _flow_store(tmp_path, events):
    s = Store(str(tmp_path / "m.db"), project_name="Hard", section_schema=FLOW_SCHEMA,
              on_audit_event=events.append)
    s.record("hard max episode", EpisodeType.OBSERVATION)
    return s


def test_bound_comes_from_the_schema_and_flows_largest_save_passes():
    assert hard_max_chars(DEFAULT_SCHEMA) == 25_000
    assert hard_max_chars(FLOW_SCHEMA) == 31_875 == -(-5 * default_max_chars(FLOW_SCHEMA) // 4)
    assert 30_425 < hard_max_chars(FLOW_SCHEMA)  # flow's largest save on record


def test_refusal_names_size_bound_and_what_to_cut_and_audits(tmp_path):
    events: list = []
    bound = hard_max_chars(FLOW_SCHEMA)
    with _flow_store(tmp_path, events) as s:
        # max_chars on prepare_wrap moves the target, never the bound.
        assert prepare_wrap(s, max_chars=999_999)["status"] == "ready"
        with pytest.raises(ContinuityValidationError) as e:
            validated_save_continuity(s, _text(bound + 1), allow_shrink=True, today="2026-10-04")
        err = e.value
        assert (err.chars, err.bound, err.target) == (bound + 1, bound, 25_500)
        msg = str(err)
        assert str(bound + 1) in msg and str(bound) in msg and "25500" in msg
        # FACT-shaped sections are named to cut from; identity layers are not.
        assert "State (" in msg and "Active Threads (" in msg and "Context (" in msg
        assert "Do NOT cut Patterns (" in msg and "Understanding (" in msg
        assert pickle.loads(pickle.dumps(err)).chars == bound + 1
        refused = [x for x in events if x["event"] == "continuity_refused"]
        assert len(refused) == 1 and refused[0]["data"]["over_by"] == 1
        # The wrap is still open: the same save, now at the bound, goes through,
        # and a large Durable Facts section does not count toward it.
        durable = "## Durable Facts\n" + "".join(f"- fact {i}\n" for i in range(400)) + "\n"
        assert validated_save_continuity(s, _text(bound, durable), today="2026-10-04")["chars"] > bound


def test_refusal_reaches_the_agent_over_mcp(tmp_path):
    events: list = []
    with _flow_store(tmp_path, events) as s:
        prepare_wrap(s)
        result = Server(s)._tool_save_continuity({"text": _text(hard_max_chars(FLOW_SCHEMA) + 50)})
        text = result["content"][0]["text"]
        assert result["isError"] and "hard maximum" in text and "Do NOT cut" in text
        assert any(x["event"] == "continuity_refused" for x in events)


_DEFAULT_TEXT = """# Grad — Memory (v1)

## State
s

## Patterns
- p | 2x (2026-10-04)

## Decisions
[decided(rationale: "x", on: "2026-10-04")] y

## Context
{pad}
"""


def test_the_bound_measures_the_text_that_is_written(tmp_path):
    """Graduation can rewrite a line longer (a bare ``2x`` becomes ``1x`` plus a
    note once the store has seen citations): a text at the bound going in was
    25,017 coming out and was saved (codex L3 HIGH, reproduced)."""
    bound = hard_max_chars(DEFAULT_SCHEMA)
    with Store(str(tmp_path / "m.db"), project_name="Grad") as s:
        s.save_meta({**s.load_meta(), "citations_seen": True})
        s.record("grad episode", EpisodeType.OBSERVATION)
        assert prepare_wrap(s, max_chars=bound + 5000)["status"] == "ready"
        base = len(_DEFAULT_TEXT.format(pad=""))
        with pytest.raises(ContinuityValidationError) as e:
            validated_save_continuity(
                s, _DEFAULT_TEXT.format(pad="c" * (bound - base)), today="2026-10-04")
        assert e.value.chars > bound  # the written text, not the input
        assert "you submitted" in str(e.value)  # the written size is not the submitted size
