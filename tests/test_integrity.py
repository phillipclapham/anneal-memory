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
        assert len(TOOLS) == 16  # 6 core + 2 crystal + 8 spore (prospective layer)

    def test_tool_names(self):
        names = {t["name"] for t in TOOLS}
        assert names == {
            "record", "recall", "prepare_wrap", "save_continuity", "delete_episode", "status",
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
        assert len(issues) == 16  # All 16 tools missing


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
