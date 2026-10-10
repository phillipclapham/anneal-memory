"""MCP save_continuity carries ``durable_warnings`` in its result text."""

from __future__ import annotations

import pytest

from anneal_memory import Store
from anneal_memory import server as srv
from anneal_memory.server import Server

TEXT = (
    "# TestProject — Memory (v1)\n"
    "## State\nActive on test\n"
    "## Patterns\n\n"
    "## Decisions\n\n"
    "## Context\nFirst session.\n"
)


@pytest.fixture
def server(tmp_path):
    s = Store(tmp_path / "m.db", project_name="TestProject", audit=False)
    yield Server(s)
    s.close()


def _save(server, monkeypatch, **extra):
    real = srv._lib_validated_save_continuity

    def stubbed(*a, **k):
        res = real(*a, **k)
        res.update(extra)
        return res

    monkeypatch.setattr(srv, "_lib_validated_save_continuity", stubbed)
    server._tool_record({"content": "Test obs", "episode_type": "observation"})
    server._tool_prepare_wrap({})
    result = server._tool_save_continuity({"text": TEXT})
    assert not result.get("isError")
    return result["content"][0]["text"]


def test_durable_warnings_are_listed_under_durable_facts(server, monkeypatch):
    out = _save(server, monkeypatch, durable_warnings=[
        "Durable facts: `## Durable Facts` in this wrap left out 1 line(s) of the prior "
        "continuity, re-inserted verbatim: - tree nut allergy",
        "a warning without the prefix",
    ])
    head, _, rest = out.partition("\nDurable facts:\n")
    assert rest.startswith(
        "  - `## Durable Facts` in this wrap left out 1 line(s) of the prior continuity")
    assert "\n  - a warning without the prefix\n" in rest
    assert "Durable facts: Durable facts:" not in out
    assert out.index("Durable facts:") < out.index("Section sizes:")


@pytest.mark.parametrize("extra", [{}, {"durable_warnings": []}, {"durable_warnings": None}])
def test_no_warnings_adds_nothing(server, monkeypatch, extra):
    assert "Durable facts" not in _save(server, monkeypatch, **extra)
