"""Reader scopes (anneal_memory.scope): a build-time projection, gated by a post-check."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from anneal_memory import scope
from anneal_memory.scope import ScopeError, parse_policy, project
from anneal_memory.store import Store

REPO = Path(__file__).resolve().parent.parent


def _policy(window: str = "line", ids: list[str] | None = None, **extra) -> dict:
    return {
        "version": 1, "name": "reader-b", "allow": ["upper acme"],
        "rules": [
            {"name": "client-security", "all_of": [["acme", "zorblax"],
                                                    ["unencrypted", "password"]],
             "window": window},
            {"name": "widgetco", "any_of": ["widgetco", "bramble"]},
        ],
        "deny_episode_ids": ids or [],
        **extra,
    }


def _write_policy(path: Path, doc: dict) -> Path:
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture
def src(tmp_path: Path):
    d = tmp_path / "full"
    d.mkdir()
    ids = {}
    with Store(d / "memory.db") as s:
        ids["sec"] = s.record("ZORBLAX disk unencrypted, flagged to Pat", "observation").id
        ids["split"] = s.record(
            "Acme cleanup unbilled\nrotate the levain password next week", "observation").id
        ids["widgetco"] = s.record("Widgetco care plan call scheduled", "observation").id
        ids["levain"] = s.record("levain 0.11.23 shipped", "observation").id
        ids["allow"] = s.record("drove through Upper Acme, unencrypted wifi", "observation").id
        ids["byid"] = s.record("a plain note the host's tags deny", "observation").id
        s.save_continuity(
            "# flow — Memory (v1)\n\n## State\n"
            "levain shipped · Widgetco call Tuesday · ZORBLAX disk unencrypted · rent paid\n"
            "- **▶▶ CLIENTS**: Acme cleanup unbilled · rotate the password\n"
            "- **▶ GAMES**: Shatterbeat live\n")
    (d / "memory.crystal.json").write_text(json.dumps({
        "schema_version": 1, "retired": [], "surfaced_fold": {},
        "crystal": [{"name": "harness_before_model", "explanation": "generic"},
                    {"name": "read_the_terms", "explanation": "the Bramble terms first"},
                    {"name": "disk_lesson", "explanation": "zorblax shipped unencrypted"}]}))
    return d, ids


def _contents(db: Path) -> dict[str, str]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return dict(c.execute("SELECT id, content FROM episodes"))
    finally:
        c.close()


def test_denied_text_absent_after_present_before(src, tmp_path):
    full, ids = src
    pol = parse_policy(json.dumps(_policy(ids=[ids["byid"]])).encode())
    # positive control: every seeded fact is in the source
    before = _contents(full / "memory.db")
    assert "ZORBLAX disk unencrypted" in before[ids["sec"]] and ids["byid"] in before
    assert "Widgetco" in (full / "memory.continuity.md").read_text()
    assert "Bramble" in (full / "memory.crystal.json").read_text()

    root = tmp_path / "proj"
    rep = project(full, root, pol)
    out = root / "current"
    eps = _contents(out / "memory.db")
    # line window: the client and "password" on different lines of one episode survive
    assert set(eps) == {ids["split"], ids["levain"], ids["allow"]}
    assert rep["episodes_dropped"] == {"total": 3, "by_rule": {"client-security": 1, "widgetco": 1},
                                       "by_id": 1}
    assert rep["crystals_dropped"] == 2 and rep["scope"] == "reader-b"
    assert rep["continuity"] == "lexical-segment (interim)" and rep["policy_sha256"] == pol.sha256
    cont = (out / "memory.continuity.md").read_text()
    assert "Widgetco" not in cont and "ZORBLAX" not in cont and "Acme" not in cont
    assert "levain shipped · rent paid" in cont and "Shatterbeat live" in cont
    # client and security split across two segments of one line: the whole line goes, its label stays
    assert "- **▶▶ CLIENTS** (scoped item removed)" in cont
    names = [c["name"] for c in json.loads((out / "memory.crystal.json").read_text())["crystal"]]
    assert names == ["harness_before_model"]
    raw = (out / "memory.db").read_bytes().lower()
    assert b"widgetco" not in raw and b"zorblax" not in raw  # VACUUM INTO left no free-page copy
    assert json.loads((out / "scope.json").read_text()) == {
        "name": "reader-b", "policy_sha256": pol.sha256, "continuity": "lexical-segment (interim)"}
    with Store(out / "memory.db", read_only=True, audit=False) as s:  # a served open works
        assert s.load_continuity() is not None


def test_document_window_drops_the_split_episode(src, tmp_path):
    full, ids = src
    project(full, tmp_path / "proj", parse_policy(json.dumps(_policy("document")).encode()))
    eps = _contents(tmp_path / "proj" / "current" / "memory.db")
    assert ids["split"] not in eps and ids["levain"] in eps and ids["allow"] in eps


def test_post_check_refuses_and_unpublishes(src, tmp_path, monkeypatch):
    full, _ = src
    pol = parse_policy(json.dumps(_policy()).encode())
    root = tmp_path / "proj"
    project(full, root, pol)
    assert (root / "current" / "memory.db").exists()
    real = scope.Policy.hits
    # the filter misses one row; the post-check (which does not go through hits) must not
    monkeypatch.setattr(scope.Policy, "hits",
                        lambda self, text: [] if "Widgetco" in text else real(self, text))
    with pytest.raises(ScopeError, match="episodes.content"):
        project(full, root, pol)
    assert not (root / "current").exists()
    assert not list(root.glob("gen-*")) and not list(root.glob(".stage.*"))


@pytest.mark.parametrize("doc, msg", [
    ({"version": 1, "name": "x", "rules": []}, "no rules and no deny_episode_ids"),
    ({"version": 1, "name": "x", "rules": [{"name": "r", "all_of": [["a"], []], "window": "line"}]},
     "group 1 is empty"),
    ({"version": 1, "name": "x", "rules": [{"name": "r", "all_of": [], "window": "line"}]},
     "non-empty list of groups"),
    ({"version": 1, "name": "x", "allow": ["upper acme"],
      "rules": [{"name": "r", "any_of": ["upper acme"]}]}, "allow phrase hides"),
    ({"version": 2, "name": "x", "deny_episode_ids": ["a"]}, "version"),
    ({"version": 1, "name": "x", "deny_episode_ids": ["a"], "deny": []}, "unknown key"),
    ({"version": 1, "name": "x", "rules": [{"name": "r", "any_of": ["a"], "all_of": [["b"]]}]},
     "exactly one"),
])
def test_bad_policy_refuses(doc, msg):
    with pytest.raises(ScopeError, match=msg):
        parse_policy(json.dumps(doc).encode())


def test_cli_publishes_then_refuses_with_exit_3(src, tmp_path):
    full, _ = src
    root = tmp_path / "proj"
    env = {**os.environ, "PYTHONPATH": str(REPO)}

    def run(policy: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "anneal_memory", "scope-project", "--src", str(full),
             "--out", str(root), "--policy", str(policy)],
            capture_output=True, text=True, env=env, timeout=120)

    ok = run(_write_policy(tmp_path / "p.json", _policy()))
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["episodes_dropped"]["total"] == 2
    assert (root / "current" / "scope.json").exists()
    bad = run(_write_policy(tmp_path / "empty.json", {"version": 1, "name": "x", "rules": []}))
    assert bad.returncode == 3 and "REFUSED (projection unpublished)" in bad.stderr
    assert not (root / "current").exists()
