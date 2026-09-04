"""Tests for anneal_memory.integrity — tool description integrity verification."""

import json
import pytest
from pathlib import Path

from anneal_memory.integrity import (
    TOOLS,
    RESOURCES,
    hash_tool,
    generate_integrity_file,
    verify_integrity,
)


class TestToolDefinitions:
    """Verify canonical tool definitions are well-formed."""

    def test_tool_count(self):
        assert len(TOOLS) == 17  # 7 core + 2 crystal + 8 spore (prospective layer)

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "record", "recall", "prepare_wrap", "save_continuity", "wrap_cancel",
            "delete_episode", "status",
            "crystal_recall", "crystal_index",
            "spore_add", "spore_get", "spore_list", "spore_touch",
            "spore_update", "spore_descend", "spore_ascend", "spore_surface",
        }

    def test_all_tools_have_required_fields(self):
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"
            assert "properties" in tool["inputSchema"]

    def test_record_requires_content_and_type(self):
        record = next(t for t in TOOLS if t["name"] == "record")
        assert "required" in record["inputSchema"]
        assert set(record["inputSchema"]["required"]) == {"content", "episode_type"}

    def test_save_continuity_requires_text(self):
        save = next(t for t in TOOLS if t["name"] == "save_continuity")
        assert "required" in save["inputSchema"]
        assert "text" in save["inputSchema"]["required"]

    def test_status_has_no_required_params(self):
        status = next(t for t in TOOLS if t["name"] == "status")
        assert "required" not in status["inputSchema"]

    def test_episode_type_enum_consistent(self):
        """All tools that reference episode_type should have the same enum values."""
        expected = {"observation", "decision", "tension", "question", "outcome", "context"}
        for tool in TOOLS:
            props = tool["inputSchema"]["properties"]
            if "episode_type" in props:
                assert set(props["episode_type"]["enum"]) == expected

    def test_resources_defined(self):
        assert len(RESOURCES) == 2

    def test_continuity_resource(self):
        assert RESOURCES[0]["uri"] == "anneal://continuity"
        assert RESOURCES[0]["mimeType"] == "text/markdown"

    def test_integrity_manifest_resource(self):
        assert RESOURCES[1]["uri"] == "anneal://integrity/manifest"
        assert RESOURCES[1]["mimeType"] == "application/json"

    def test_delete_episode_description_honest_about_tombstone_fields(self):
        """Regression guard for Diogenes SEMANTIC finding (Apr 13 2026).

        The delete_episode tool description previously claimed tombstones
        were "content-hash only, no original text," but the schema stores
        id + timestamp + type + content_hash. GDPR implications hinge on
        what survives deletion — the description must match reality.
        """
        delete = next(t for t in TOOLS if t["name"] == "delete_episode")
        desc = delete["description"]
        assert "content-hash only" not in desc
        assert "timestamp" in desc
        assert "type" in desc
        assert "hash" in desc
        assert "GDPR" in desc


class TestHashTool:
    """Verify hash generation is deterministic and sensitive."""

    def test_deterministic(self):
        tool = TOOLS[0]
        assert hash_tool(tool) == hash_tool(tool)

    def test_different_descriptions_different_hashes(self):
        tool_a = {"description": "Do A", "inputSchema": {"type": "object", "properties": {}}}
        tool_b = {"description": "Do B", "inputSchema": {"type": "object", "properties": {}}}
        assert hash_tool(tool_a) != hash_tool(tool_b)

    def test_different_schemas_different_hashes(self):
        tool_a = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        tool_b = {
            "description": "Same",
            "inputSchema": {"type": "object", "properties": {"y": {"type": "integer"}}},
        }
        assert hash_tool(tool_a) != hash_tool(tool_b)

    def test_hash_is_sha256_hex(self):
        h = hash_tool(TOOLS[0])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestGenerateIntegrityFile:
    """Test integrity file generation."""

    def test_generates_valid_json(self, tmp_path):
        path = tmp_path / "integrity.json"
        result = generate_integrity_file(path)
        assert result == path
        data = json.loads(path.read_text())
        assert "version" in data
        assert "tools" in data
        assert data["version"] == 1

    def test_contains_all_tool_hashes(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        assert set(data["tools"].keys()) == {t["name"] for t in TOOLS}

    def test_hashes_match_current_definitions(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        for tool in TOOLS:
            assert data["tools"][tool["name"]] == hash_tool(tool)


class TestVerifyIntegrity:
    """Test integrity verification."""

    def test_valid_file_passes(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        valid, issues = verify_integrity(path)
        assert valid is True
        assert issues == []

    def test_missing_file_fails(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 1
        assert "not found" in issues[0]

    def test_corrupt_json_fails(self, tmp_path):
        path = tmp_path / "integrity.json"
        path.write_text("not json{{{")
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 1
        assert "Failed to read" in issues[0]

    def test_tampered_hash_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        # Tamper with the record tool's hash
        data["tools"]["record"] = "0" * 64
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("record" in i and "mismatch" in i for i in issues)

    def test_missing_tool_in_file_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        del data["tools"]["status"]
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("status" in i and "not found" in i for i in issues)

    def test_extra_tool_in_file_detected(self, tmp_path):
        path = tmp_path / "integrity.json"
        generate_integrity_file(path)
        data = json.loads(path.read_text())
        data["tools"]["evil_tool"] = "0" * 64
        path.write_text(json.dumps(data))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert any("evil_tool" in i and "Unknown" in i for i in issues)

    def test_empty_tools_dict_fails(self, tmp_path):
        path = tmp_path / "integrity.json"
        path.write_text(json.dumps({"version": 1, "tools": {}}))
        valid, issues = verify_integrity(path)
        assert valid is False
        assert len(issues) == 17  # All 17 tools missing


class TestShippedManifest:
    """Verify the manifest shipped inside the package matches current TOOLS.

    This test would have caught the v0.1.9 staleness that Session 10.5c.1
    Layer 4 found: ``delete_episode`` was added in v0.1.9 but
    ``tool-integrity.json`` was last regenerated in v0.1.8, so the shipped
    manifest had been silently failing host-side verification for one
    full session and one PyPI release. Without this test, any future tool
    addition can ship with the same staleness.

    If this test fails after adding, renaming, or editing a tool, run::

        python3 -c "from anneal_memory.integrity import generate_integrity_file; from pathlib import Path; generate_integrity_file(Path('anneal_memory/tool-integrity.json'))"

    and commit the regenerated manifest.
    """

    def test_shipped_manifest_verifies(self):
        """The manifest shipped inside the anneal_memory package must
        pass verify_integrity() against the current TOOLS definitions.
        """
        import anneal_memory
        pkg_root = Path(anneal_memory.__file__).parent
        manifest = pkg_root / "tool-integrity.json"
        assert manifest.exists(), (
            f"Shipped integrity manifest missing at {manifest} — "
            "regenerate with generate_integrity_file()"
        )
        valid, issues = verify_integrity(manifest)
        assert valid, (
            f"Shipped tool-integrity.json is out of sync with TOOLS. "
            f"Issues: {issues}. Regenerate via "
            f"generate_integrity_file(Path('anneal_memory/tool-integrity.json'))"
        )

    def test_shipped_manifest_covers_all_current_tools(self):
        """Every tool in TOOLS must have an entry in the shipped manifest.

        Complements test_shipped_manifest_verifies by failing loudly and
        specifically when a new tool is added without regenerating, even
        if verify_integrity's behavior changes in the future.
        """
        import anneal_memory
        pkg_root = Path(anneal_memory.__file__).parent
        manifest = pkg_root / "tool-integrity.json"
        data = json.loads(manifest.read_text())
        manifest_tool_names = set(data["tools"].keys())
        current_tool_names = {t["name"] for t in TOOLS}
        missing = current_tool_names - manifest_tool_names
        extra = manifest_tool_names - current_tool_names
        assert not missing, (
            f"Tools in TOOLS but missing from shipped manifest: {missing}. "
            f"Run generate_integrity_file() and commit."
        )
        assert not extra, (
            f"Tools in shipped manifest but removed from TOOLS: {extra}. "
            f"Run generate_integrity_file() and commit."
        )


class TestSourceCompilesCleanlyOnEveryTargetPython:
    """No module may contain an invalid escape sequence.

    ⛔ THIS BROKE CI ON 2026-09-04 AND COULD NOT BREAK LOCALLY. A docstring
    written with ``\\`` escapes is a *SyntaxWarning* on Python 3.12+ and a hard
    *SyntaxError* under pytest's assertion rewriter on 3.10/3.11. The dev
    machine runs 3.13; CI runs 3.10-3.13. The full suite passed locally, twice,
    against a file that could not even be COLLECTED on half the support matrix.

    ⚖ THE GENERAL POINT, which is why this test exists rather than a note:
    "tests pass locally" is a claim about ONE interpreter. With a four-version
    matrix it is evidence about a quarter of it, and the missing three quarters
    are invisible rather than red. Same shape as every other defect this suite
    has grown a guard for — an instrument reporting health on an axis it does
    not measure.

    ⚡ It is catchable locally: the warning EXISTS on 3.13, it simply is not
    fatal there. Promoting it to an error reproduces the 3.10 failure on any
    interpreter, so this closes the version gap for this class without needing
    the other interpreters installed.
    """

    def test_no_module_has_an_invalid_escape_sequence(self):
        import warnings

        root = Path(__file__).resolve().parent.parent
        offenders = []
        for path in sorted(
            list((root / "anneal_memory").rglob("*.py"))
            + list((root / "tests").rglob("*.py"))
        ):
            src = path.read_text(encoding="utf-8")
            with warnings.catch_warnings():
                warnings.simplefilter("error", SyntaxWarning)
                try:
                    compile(src, str(path), "exec")
                except SyntaxWarning as exc:
                    offenders.append(f"{path.relative_to(root)}: {exc}")
                except SyntaxError as exc:
                    offenders.append(f"{path.relative_to(root)}: SyntaxError: {exc}")
        assert not offenders, (
            "invalid escape sequence(s) — a SyntaxWarning here is a hard "
            "SyntaxError on Python 3.10/3.11 under pytest's assertion "
            "rewriter, so the file cannot even be COLLECTED there:\n  "
            + "\n  ".join(offenders)
        )


class TestReleaseStampIsNotAPublishedVersion:
    """HEAD must not be stamped at a version that is already released.

    ⛔ THE POLICY (spore-710, 2026-09-03): the commit after a release bumps to
    the next ``.devN``, or the release commit is the last one on that number.
    Otherwise the version names TWO trees and what a developer clones is not
    what an adopter installs — including a `tool-integrity.json` that no longer
    matches the published one.

    ⚠ WHY THIS IS A TEST AND NOT A RULE. It was already a written policy, in
    this repo's own CHANGELOG, when the very next commit after the v0.9.9 tag
    landed still stamped 0.9.9. The action-boundary hook surfaced spore-710
    before that push and the push happened anyway: the note was on screen,
    correct and specific, and it did not change the behaviour. Reading is not
    acting, and structural invariants beat discipline.

    Reads GIT TAGS, not PyPI, deliberately: an unreachable network oracle
    degrades to a PASS, which is this defect's whole shape — an instrument that
    reports health because it could not look.
    """

    def _run(self, *args):
        import subprocess
        root = self._root()
        try:
            r = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    def _root(self):
        import anneal_memory
        return Path(anneal_memory.__file__).resolve().parent.parent

    def test_head_stamp_is_not_an_already_released_version(self):
        import re
        from anneal_memory import __version__

        if self._run("rev-parse", "--git-dir") is None:
            pytest.skip("not a git checkout")
        tags = self._run("tag", "--list", "v*")
        if not tags:
            pytest.skip("no release tags to compare against")

        released = {t.lstrip("v") for t in tags.splitlines() if t.strip()}
        if __version__ not in released:
            return  # a .devN or an unreleased number — correct by construction

        # The stamp equals a released version. That is legal ONLY if HEAD is
        # that exact release commit.
        at_head = (self._run("tag", "--points-at", "HEAD") or "").splitlines()
        assert f"v{__version__}" in at_head, (
            f"HEAD is stamped {__version__!r}, which is already released, but "
            f"HEAD is not the v{__version__} commit. Bump to the next .devN — "
            f"a published number on a moving main makes the version name two "
            f"trees (spore-710)."
        )

    def test_committed_head_stamp_is_not_an_already_released_version(self):
        """The twin above, with the subject a PUSH actually publishes.

        ⛔ THE TEST ABOVE READS THE WORKING TREE AND A PUSH SHIPS COMMITS.
        ``from anneal_memory import __version__`` imports from disk, while
        ``git tag --points-at HEAD`` reads the commit graph — two subjects in
        one assertion. REPRODUCED 2026-09-04 in a scratch clone: with HEAD
        committed at ``0.9.9`` on a non-tag commit and an UNCOMMITTED bump to
        ``0.9.10.dev0`` in the working tree, the gate went from red to GREEN
        and the pre-push hook exited 0 — publishing the bad commit. The fix
        that was supposed to prevent the defect made the guard report health.

        Both subjects are legitimate: in CI the checkout IS the commit, so the
        working-tree test is the right one there. They are two tests because
        they are two claims, not because one is a better version of the other.

        ⚠ SCOPE, stated rather than implied: this checks HEAD. A push of a
        non-HEAD branch or an older range is not covered. HEAD is what the
        pre-push hook's pytest run is scoped to anyway, and widening this to
        walk the pushed range would make a GATE big — deliberately not done
        (spore-551: the gate list is small and fixed).

        MUTATION-CHECKED: reproduced red on the scratch clone above, green
        after committing the bump.
        """
        import re

        if self._run("rev-parse", "--git-dir") is None:
            pytest.skip("not a git checkout")
        tags = self._run("tag", "--list", "v*")
        if not tags:
            pytest.skip("no release tags to compare against")

        blob = self._run("show", "HEAD:anneal_memory/__init__.py")
        if blob is None:
            pytest.skip("HEAD carries no anneal_memory/__init__.py")
        m = re.search(r'__version__ = "([^"]+)"', blob)
        assert m, "HEAD's __init__.py no longer declares __version__"
        committed = m.group(1)

        released = {t.lstrip("v") for t in tags.splitlines() if t.strip()}
        if committed not in released:
            return  # a .devN or an unreleased number — correct by construction

        at_head = (self._run("tag", "--points-at", "HEAD") or "").splitlines()
        assert f"v{committed}" in at_head, (
            f"THE COMMIT AT HEAD is stamped {committed!r}, which is already "
            f"released, but HEAD is not the v{committed} commit. A clean "
            f"working tree is not the question — pushing publishes this "
            f"commit. Bump to the next .devN and COMMIT it (spore-710)."
        )

    def test_every_stamp_site_agrees(self):
        """The 0.9.9 cut found a THIRD stamp site (server.json) only because a
        consistency test failed. Pin all of them together."""
        import json
        import re
        from anneal_memory import __version__

        root = self._root()
        pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'^version = "([^"]+)"', pyproject, re.M)
        assert m, "pyproject.toml no longer declares a version"
        assert m.group(1) == __version__, (
            f"pyproject {m.group(1)!r} != __version__ {__version__!r}"
        )

        sj = json.loads((root / "server.json").read_text(encoding="utf-8"))
        assert sj["version"] == __version__, "server.json top-level version drifted"
        for pkg in sj.get("packages", []):
            assert pkg["version"] == __version__, (
                "server.json packages[].version drifted — and this one names the "
                "PyPI version, so at publish it must name a release that exists"
            )


class TestVersionConsistency:
    """Every shipped version string must agree with ``__version__``.

    The 0.4.0 ship bumped only ``pyproject.toml`` and left ``__version__``
    and both ``server.json`` fields at 0.3.5 — so ``anneal-memory==0.4.0``
    reported ``__version__ == "0.3.5"`` and nothing caught it (the existing
    version tests only assert the server/CLI *report* ``__version__`, not
    that the manifests agree with it). This makes that drift structurally
    impossible instead of discipline-dependent.
    """

    def _repo_root(self) -> Path:
        import anneal_memory
        return Path(anneal_memory.__file__).parent.parent

    def test_pyproject_version_matches_dunder(self):
        import re
        import anneal_memory
        text = (self._repo_root() / "pyproject.toml").read_text()
        # Scope to the [project] table before matching ``version`` so a
        # future ``[tool.*]`` table introducing its own column-0
        # ``version = "..."`` cannot bind the wrong line (L1 hardening).
        # Split on TOML table headers; take the [project] body only.
        project_body = ""
        for chunk in re.split(r'(?m)^(\[[^\]]+\])\s*$', text):
            if chunk == "[project]":
                project_body = "__MARK__"
            elif project_body == "__MARK__":
                project_body = chunk
                break
        assert project_body and project_body != "__MARK__", (
            "no [project] table found in pyproject.toml"
        )
        m = re.search(r'(?m)^version = "([^"]+)"', project_body)
        assert m is not None, "no version in pyproject.toml [project] table"
        assert m.group(1) == anneal_memory.__version__, (
            f"pyproject.toml [project] version {m.group(1)!r} != "
            f"__version__ {anneal_memory.__version__!r}"
        )

    def test_server_json_versions_match_dunder(self):
        import anneal_memory
        data = json.loads((self._repo_root() / "server.json").read_text())
        assert data["version"] == anneal_memory.__version__, (
            f"server.json top-level version {data['version']!r} != "
            f"__version__ {anneal_memory.__version__!r}"
        )
        for pkg in data["packages"]:
            assert pkg["version"] == anneal_memory.__version__, (
                f"server.json package version {pkg['version']!r} != "
                f"__version__ {anneal_memory.__version__!r}"
            )


class TestSkillManifest:
    """The Claude Code Skill must keep a valid, routable frontmatter.

    The Skill + lean snippets are **repository artifacts** — distributed via
    this repo (and the sdist), deliberately NOT bundled in the wheel (Claude
    Code skills ship via repos/marketplaces, not PyPI). So resolving them
    through the source tree (``__file__.parent.parent``) is the correct guard:
    it validates the copy adopters actually fetch. It does NOT — and is not
    meant to — assert wheel delivery.

    A SKILL.md with a missing/empty ``name`` or ``description`` silently
    fails to register or auto-activate — an invisible-infrastructure failure
    the adopter only discovers when memory work doesn't happen. The
    ``description`` is the routing field (it decides *when* the skill loads),
    so it must be present, descriptive, and name the wrap so the skill routes
    at session end as well as start. Stdlib-only parse (the library is
    zero-dep; the test stays so too).
    """

    def _repo_root(self) -> Path:
        import anneal_memory
        return Path(anneal_memory.__file__).parent.parent

    def _frontmatter(self, text: str) -> dict[str, str]:
        assert text.startswith("---\n"), (
            "SKILL.md must open with a '---' YAML frontmatter fence"
        )
        end = text.find("\n---", 4)
        assert end != -1, "SKILL.md frontmatter is not closed by a '---' fence"
        fm: dict[str, str] = {}
        for line in text[4:end].splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, sep, val = line.partition(":")
            assert sep, f"malformed frontmatter line: {line!r}"
            fm[key.strip()] = val.strip()
        return fm

    def test_skill_frontmatter_valid(self):
        skill = self._repo_root() / "skill" / "anneal-memory" / "SKILL.md"
        assert skill.exists(), "shipped skill/anneal-memory/SKILL.md missing"
        fm = self._frontmatter(skill.read_text())
        assert fm.get("name") == "anneal-memory", (
            f"SKILL.md name must be 'anneal-memory', got {fm.get('name')!r}"
        )
        desc = fm.get("description", "")
        assert len(desc) >= 40, (
            "SKILL.md description is the routing field — it must be present "
            f"and descriptive (got {len(desc)} chars)"
        )
        assert "wrap" in desc.lower(), (
            "SKILL.md description should mention the wrap so the skill routes "
            "at session end, not only at session start"
        )

    def test_lean_snippets_point_to_skill(self):
        """Regression guard: each lean snippet file references the Skill.

        Scope note: this checks the whole file, so the human-facing comment
        header (which tells the adopter to install the Skill) satisfies it —
        it does not assert the agent-pasted ``## Memory`` body names the Skill
        (the body delegates depth generically). It catches removal of the
        Skill pointer from the file, which is the regression that matters.
        """
        examples = self._repo_root() / "examples"
        for name in (
            "agent-instructions.lean.example",
            "agent-instructions.lean.cli.example",
        ):
            p = examples / name
            assert p.exists(), f"lean snippet examples/{name} missing"
            assert "SKILL" in p.read_text(), (
                f"{name} should point adopters to the SKILL for depth"
            )



def _mcp_cell_marks_unavailable(row: str, tool_names: set[str]) -> bool:
    """True when a SKILL.md table row's MCP cell declares the tool unavailable
    over MCP — i.e. it says CLI-only (either spelling) and names no shipped MCP
    tool in that same cell.

    Cell-scoped on purpose: a row may legitimately say a CLI *subcommand* is
    CLI-only while correctly naming the MCP tool alongside it, and a row-scoped
    test cannot tell that apart from the defect.
    """
    if not row.startswith("|"):
        return False
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    if len(cells) < 2:
        return False
    mcp_cell = cells[-1]
    lowered = mcp_cell.lower()
    if "cli only" not in lowered and "cli-only" not in lowered:
        return False
    return not any(f"`{name}`" in mcp_cell for name in tool_names)


class TestDocumentedToolCount:
    """The README and server.py both state the tool count in prose. Nothing
    checked either, and both were stale the moment a tool was added — the same
    class as the finding this release fixes (a description that disagrees with
    correct code), so the fix is an assertion rather than a corrected number."""

    def _root(self):
        import anneal_memory
        return Path(anneal_memory.__file__).resolve().parent.parent

    def test_readme_states_the_real_count(self):
        import re
        readme = (self._root() / "README.md").read_text(encoding="utf-8")
        m = re.search(r"\*\*(\d+) tools total\*\*", readme)
        assert m, "README no longer states a tool total — update this test if deliberate"
        assert int(m.group(1)) == len(TOOLS)

    def test_readme_documents_every_tool(self):
        """A tool absent from the table is a tool an adopter never learns exists —
        which is exactly how `wrap_cancel` stayed invisible."""
        readme = (self._root() / "README.md").read_text(encoding="utf-8")
        for tool in TOOLS:
            name = tool["name"]
            if name.startswith("spore_"):
                continue  # documented collectively as `spore_*`
            assert f"`{name}`" in readme, f"MCP tool {name!r} is not documented in README.md"

    def test_no_skill_row_marks_an_mcp_tool_as_having_no_mcp_surface(self):
        """SKILL.md must never present a shipped MCP tool as unavailable to MCP.

        ⚠ REWRITTEN 2026-09-03 — the previous guard,
        ``test_skill_does_not_claim_an_mcp_tool_is_CLI_ONLY``, SCANNED ZERO
        LINES and had therefore never executed its assertion once. Its filter
        required a row to start with ``|`` AND contain the exact string
        ``"CLI only"``; the same commit that added the guard rewrote the only
        qualifying row to say ``CLI-only`` (hyphen), so the loop body never
        ran. Not weaker coverage — NO coverage. Found by the L1 seat, which
        sharpened Diogenes' reading (he had it as "passes on the defect it was
        written for"; it cannot fail at all).

        ⛔ AND THE OLD PREDICATE WAS UNSALVAGEABLE, not merely mis-spelled.
        "row says CLI-only AND names a tool" false-positives on the CURRENT,
        CORRECT row, which reads ``| Recover a stuck wrap | ... | `wrap_cancel`
        (inspect via `status`; `wrap-status` is CLI-only) |`` — the CLI-only
        applies to the CLI subcommand ``wrap-status``, while the row correctly
        names the MCP tool. Widening the spelling would have made a good row
        fail.

        The real defect shape was the MCP CELL marking unavailability while
        naming no tool (``— (CLI only)``), so the predicate is cell-scoped.
        Its discrimination is proved on synthetic input by
        :meth:`test_the_unavailable_marker_predicate_actually_discriminates`,
        so this assertion cannot go quietly vacuous again.
        """
        skill = self._root() / "skill" / "anneal-memory" / "SKILL.md"
        if not skill.is_file():
            pytest.skip("skill/ not present in this checkout")
        tool_names = {t["name"] for t in TOOLS}
        offenders = [
            line
            for line in skill.read_text(encoding="utf-8").splitlines()
            if _mcp_cell_marks_unavailable(line, tool_names)
        ]
        assert not offenders, (
            "SKILL.md row(s) mark the MCP surface unavailable without naming "
            f"any shipped MCP tool in that cell: {offenders}"
        )

    def test_the_unavailable_marker_predicate_actually_discriminates(self):
        """The non-vacuity proof for the guard above.

        A NEGATIVE guard is healthy when it matches nothing, which makes
        "matches nothing because the file is clean" indistinguishable from
        "matches nothing because the filter is broken" — exactly how its
        predecessor sat dead in the suite. So the predicate is exercised here
        on inputs of BOTH classes, independent of what SKILL.md happens to say.
        """
        tool_names = {t["name"] for t in TOOLS}

        # The real defect, verbatim from the published 0.9.8 sdist.
        bad = (
            "| Recover a stuck wrap | `anneal-memory wrap-status` · "
            "`wrap-cancel` | — (CLI only) |"
        )
        assert _mcp_cell_marks_unavailable(bad, tool_names)
        # Hyphenated spelling — the one that killed the old filter.
        assert _mcp_cell_marks_unavailable(
            "| Recover a stuck wrap | `wrap-cancel` | — (CLI-only) |", tool_names
        )
        # The CURRENT, CORRECT row: says CLI-only about a CLI subcommand while
        # naming the MCP tool. Must NOT fire.
        good = (
            "| Recover a stuck wrap | `anneal-memory wrap-status` · "
            "`wrap-cancel` | `wrap_cancel` (inspect via `status`; "
            "`wrap-status` is CLI-only) |"
        )
        assert not _mcp_cell_marks_unavailable(good, tool_names)
        # Prose is not a table row.
        assert not _mcp_cell_marks_unavailable(
            '*(Before 0.9.8 this row read "— (CLI only)".)*', tool_names
        )

    def test_skill_documents_the_wrap_recovery_mcp_tool(self):
        """The positive half — the row must actually name the tool. A row that
        merely stopped saying 'CLI only' would still leave the agent unaware
        the tool exists, which is the condition Alex De Groodt was in."""
        skill = self._root() / "skill" / "anneal-memory" / "SKILL.md"
        if not skill.is_file():
            pytest.skip("skill/ not present in this checkout")
        text = skill.read_text(encoding="utf-8")
        assert "`wrap_cancel`" in text, (
            "SKILL.md never names the wrap_cancel MCP tool — an agent reading it "
            "still has no in-band way out of a stuck wrap"
        )

    def test_skill_does_not_teach_a_terminating_graduation_ladder(self):
        """SKILL.md must not teach a ceiling the library removed in 0.9.7.

        ⚠ THIS ROTTED FOR TWENTY DAYS AND NOTHING FIRED. AM-LEVELCAP (0.9.7,
        2026-08-14) replaced the `level in (2, 3)` gate with `MIN_PROVEN_LEVEL`
        and no ceiling, because a pattern earned 4+ times could neither validate
        nor crystallize out. The generated wrap instructions were pinned to the
        reader's range by ``TestTeacherCoversReaderRange``; **SKILL.md — the doc
        an AGENT loads — was pinned by nothing**, and went on teaching
        `1x → 2x → 3x` full stop until 2026-09-04.

        ⛔ A WIDENING EMITS NO ERROR SIGNAL ANYWHERE DOWNSTREAM. Nothing breaks:
        every consumer keeps working correctly against the narrower contract it
        already knows, and the only symptom is absence — which is precisely why
        this needs a test rather than review. Same shape as the `wrap_cancel`
        row in this same file, and as `crystal.py`'s consumer contract, which
        kept instructing readers to guard `level in (2, 3)` and so RE-CREATED
        the cap in at least one consumer that trusted it.

        Positive assertion on purpose: an absence-scan would trip on the
        historical note that quotes the old ladder, and a negative guard that
        matches nothing cannot be told from a broken one.

        ⛔ THIS GATE ASSERTED PER-FILE UNTIL 2026-09-04 AND THAT MADE IT BLIND
        TO ITS OWN SUBJECT. It took ``max()`` over the UNION of levels across
        every ladder line, so ONE corrected line certified EVERY uncorrected
        one. Measured at the time: line 85 contributed ``12x`` and passed the
        file, while line 35 went on teaching ``1x→2x→3x`` — the exact ceiling
        this test is named for, in the same document, unseen. The assertion's
        subject was the union; the claim's subject is every ladder the file
        teaches. An agent does not read the union, it reads a line and stops.

        MUTATION-CHECKED: restore `1x → 2x → 3x` on EITHER ladder line and this
        fails. (Under the old union form only the last one did.)
        """
        import re

        skill = self._root() / "skill" / "anneal-memory" / "SKILL.md"
        if not skill.is_file():
            pytest.skip("skill/ not present in this checkout")

        ladder_lines = [
            (num, line)
            for num, line in enumerate(
                skill.read_text(encoding="utf-8").splitlines(), 1
            )
            if "graduate" in line.lower() and re.search(r"\d+x", line)
        ]
        assert ladder_lines, "SKILL.md no longer teaches graduation at all"

        # ⚠ THE SUBJECT IS THE LADDER, NOT THE LINE. A per-line max over every
        # ``Nx`` on the line is still a proxy: line 85 also mentions 12x/18x in
        # its DEMOTION sentence, so a truncated ladder on that line would pass
        # on the strength of text that is not a ladder at all. Measured
        # 2026-09-04 by mutation — the per-line form did not kill that mutant.
        # So extract the ladder RUNS themselves and judge each one.
        #
        # A run is a taught ladder when it ASCENDS (``4x``→``3x`` is a demotion
        # example, not a ladder) and is not a whole-span quote of a retired
        # ladder. CONVENTION, enforced here: quote a retired ladder as ONE
        # backticked span (`` `1x → 2x → 3x` ``); teach a live one unquoted or
        # with per-token backticks. This fails in the SAFE direction — forget
        # the backticks around a historical quote and the gate fires loudly;
        # it cannot silently pass a real ceiling.
        ladder_run = re.compile(r"(?:`?\d+x`?\s*(?:→|->)\s*)+(?:`?\d+x`?|…|\.\.\.)")
        capped = []
        for num, line in ladder_lines:
            for match in ladder_run.finditer(line):
                run = match.group(0)
                levels = [int(x) for x in re.findall(r"(\d+)x", run)]
                if not all(b > a for a, b in zip(levels, levels[1:])):
                    continue  # descending: a demotion example, not a ladder
                if run.startswith("`") and run.endswith("`") and "`" not in run[1:-1]:
                    continue  # a whole-span quote of a retired ladder
                if run.rstrip().endswith(("…", "...")) or max(levels) > 3:
                    continue  # open-ended, or demonstrably past the old cap
                capped.append((num, max(levels), run))
        assert not capped, (
            "SKILL.md line(s) teach a ladder topping out at 3x or below. The "
            "library has had NO ceiling since 0.9.7 (MIN_PROVEN_LEVEL, no upper "
            "bound) — an agent following such a line flattens mature patterns "
            "back to 3x and loses the high-water mark. A sibling line elsewhere "
            "in the file teaching the open ladder does NOT fix this: an agent "
            "reads a line, not the union. Offending line(s): "
            + "; ".join(f"L{n} (ladder tops out at {lv}x): {t[:90]}" for n, lv, t in capped)
        )

    def test_skill_documents_every_tool(self):
        """The mirror of :meth:`test_readme_documents_every_tool`, over the doc
        that actually rotted.

        ⚠ ADDED 2026-09-03. The asymmetry was the defect: README.md, which did
        NOT rot, carried an every-tool loop, while SKILL.md — the depth doc an
        AGENT loads, and the file that DID rot — was guarded by a single
        hardcoded ``wrap_cancel`` string literal. Measured before writing this:
        erasing ``delete_episode`` from SKILL.md left the FULL suite green, so
        a tool could vanish from the agent-facing doc and nothing fired. That
        reproduces Alex De Groodt's condition — a tool the reader never learns
        exists — with no red test anywhere.

        The sibling negative guard (``..._is_CLI_ONLY``) cannot cover this: in
        SKILL.md the MCP column is where the backticked tool names live, so a
        row whose MCP cell reads "CLI only" does not name the tool BY
        CONSTRUCTION and the scan has nothing to match. It passes on the exact
        defect it was written for; this loop is what generalises.

        MUTATION-CHECKED: remove any non-``spore_*`` tool name from SKILL.md
        and this test fails. And checked against the DEFECT ITSELF rather than
        a mutant: the published 0.9.8 sdist was downloaded from PyPI on
        2026-09-03 and its SKILL.md does not contain the string
        the backticked tool name anywhere, so this loop would have failed on
        artifact that shipped. The guard that was there did not.
        """
        skill = self._root() / "skill" / "anneal-memory" / "SKILL.md"
        if not skill.is_file():
            pytest.skip("skill/ not present in this checkout")
        text = skill.read_text(encoding="utf-8")
        for tool in TOOLS:
            name = tool["name"]
            if name.startswith("spore_"):
                continue  # documented collectively as `spore_*`
            assert f"`{name}`" in text, (
                f"MCP tool {name!r} is not documented in SKILL.md — the depth "
                f"doc an agent loads. It cannot use a tool it never learns of."
            )

    def test_server_docstring_states_the_real_count(self):
        import re
        from anneal_memory import server
        m = re.search(r"(\d+) tools \+ \d+ resources", server.__doc__ or "")
        assert m, "server module docstring no longer states a tool count"
        assert int(m.group(1)) == len(TOOLS)
