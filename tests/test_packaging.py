"""What the published artifacts are allowed to contain.

⛔ WHY THIS FILE EXISTS. Until 2026-09-04 there was no
``[tool.hatch.build.targets.sdist]`` section, so the sdist took hatchling's
default — the entire VCS tree. Measured at 06e5858 by building it: 1,436,684
bytes carrying 39 ``project_memory/`` files, including six overnight review
reports that enumerate OPEN, UNFIXED defects in the shipped product by file and
line, plus three files holding the operator's absolute home path. The wheel was
never affected, which is exactly why it survived nine releases: the artifact
almost everyone consumes was clean, and the one that was not is the one nobody
opens. A PyPI sdist cannot be reliably unpublished, so this is catchable once.

⚠ THIS TEST ASSERTS ON HATCHLING'S REAL FILE-SELECTION API — the same code path
that decides the tarball's contents — and NOT on the text of pyproject.toml.
Grepping the config for the string ``project_memory`` would be a guard that
cannot see its own subject: it would pass on a config that names the directory
in a comment and ships it anyway, and it would fail on a correct config that
excludes it by some other rule. The oracle was validated against the artifact
rather than assumed: at 06e5858 the API reproduced the real tarball's file list
exactly, 112 of 112.

⚠ SCOPE, STATED PLAINLY: this gate is LOCAL-ONLY. CI installs the package with
build isolation, so ``hatchling`` is not importable on the matrix and these
tests skip there. That is not ideal, but it is not idle either — releases are
cut on the dev machine, which is where this runs. Adding ``pip install
hatchling`` to the CI workflow would make it fire on the matrix too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import anneal_memory

REPO_ROOT = Path(anneal_memory.__file__).resolve().parent.parent


def _requires_source_tree() -> None:
    """Skip when running against an installed package rather than the repo."""
    if not (REPO_ROOT / "pyproject.toml").exists():
        pytest.skip("not a source checkout — no pyproject.toml to select from")


def _sdist_distribution_paths() -> set[str]:
    """Every path the sdist will contain, via hatchling's own selection API.

    Two sources, because hatchling has two and using only the first would
    under-report. ``recurse_included_files()`` applies the configured
    ``include`` allowlist. ``get_default_build_data()['force_include']`` is the
    set hatchling adds unconditionally at build time — pyproject.toml, the
    ``readme =`` target, the license, and every VCS ignore file — which bypasses
    ``include`` entirely so that an sdist can reproduce the same VCS-exclusion
    behaviour when rebuilt. Derived from the API, never hardcoded: if hatchling
    changes what it forces, this follows it instead of going stale.

    PKG-INFO is generated into the tarball by the backend and is not a selected
    file, so it does not appear here.
    """
    _requires_source_tree()
    sdist = pytest.importorskip(
        "hatchling.builders.sdist",
        reason="hatchling is a build dependency; not present under CI's build isolation",
    )
    builder = sdist.SdistBuilder(str(REPO_ROOT))
    paths = {f.distribution_path for f in builder.recurse_included_files()}
    paths |= set(builder.get_default_build_data()["force_include"].values())
    return paths


# The declared publishable surface, graded one entry at a time against the
# property recorded in pyproject.toml: an sdist carries what a consumer needs to
# build, install, test and legally redistribute this package from source, plus
# whatever the shipped README's relative links require.
#
# ⚠ THIS IS DEFAULT-DENY AND THE FAILURE IS THE FEATURE. Add a top-level entry
# to the repo and this test goes red until someone states whether it ships. That
# is the whole difference between this and an exclusion aimed at one directory
# name: a denylist only ever catches the leak you already found, and
# ``project_memory/`` was not predictable before it existed.
EXPECTED_SDIST_TOP_LEVEL = {
    # the package, and the tests that prove it
    "anneal_memory",
    "tests",
    # README links every one of these; the suite hard-asserts the last two
    "docs",
    "examples",
    "skill",
    # the description, the record, the licence
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    # asserted on by tests/test_integrity.py
    "server.json",
    # README.md:4 embeds it by relative path
    "logo-400.png",
    # the repo-root copy; the package's own copy lives under anneal_memory/
    "tool-integrity.json",
    # forced by hatchling regardless of `include` — see _sdist_distribution_paths
    "pyproject.toml",
    ".gitignore",
}


class TestSdistShipsOnlyThePublishableSurface:
    def test_top_level_entries_are_exactly_the_declared_set(self):
        top_level = {p.split("/")[0] for p in _sdist_distribution_paths()}

        leaked = top_level - EXPECTED_SDIST_TOP_LEVEL
        assert not leaked, (
            f"The sdist would publish undeclared top-level entries: {sorted(leaked)}. "
            "Nothing enters the published sdist without a decision. Either add it to "
            "`include` in [tool.hatch.build.targets.sdist] AND to "
            "EXPECTED_SDIST_TOP_LEVEL with the reason it must ship, or leave it out. "
            "Remember what the sdist is: a public, effectively permanent artifact."
        )

        missing = EXPECTED_SDIST_TOP_LEVEL - top_level
        assert not missing, (
            f"The sdist no longer ships declared entries: {sorted(missing)}. "
            "An sdist that omits something the build, the test suite or the README "
            "needs is a worse defect than the leak this allowlist was written to "
            "close. Check the `include` patterns in pyproject.toml."
        )

    def test_no_internal_project_tracking_is_published(self):
        """The reported site, checked as a consequence rather than a premise.

        This assertion is deliberately redundant with the exactness test above —
        it names ``project_memory/`` only so that the failure message explains
        the stakes to whoever trips it. The test that does the real work is the
        one that knows nothing about this directory's name.
        """
        published = _sdist_distribution_paths()
        tracked = sorted(p for p in published if p.startswith("project_memory/"))
        assert not tracked, (
            f"{len(tracked)} project-tracking files would be published to PyPI, e.g. "
            f"{tracked[:3]}. These are review receipts naming unfixed defects and "
            "absolute operator paths. An sdist cannot be reliably unpublished."
        )


class TestWheelSurfaceIsUnchanged:
    """The wheel was always safe; pin it so a future sdist edit cannot widen it.

    ``packages = ["anneal_memory"]`` is what kept nine releases' wheels clean
    while the sdist leaked. This asserts that invariant directly instead of
    trusting that the two targets stay independent.
    """

    def test_wheel_contains_only_the_package(self):
        _requires_source_tree()
        wheel = pytest.importorskip(
            "hatchling.builders.wheel",
            reason="hatchling is a build dependency; not present under CI's build isolation",
        )
        builder = wheel.WheelBuilder(str(REPO_ROOT))
        top_level = {
            f.distribution_path.split("/")[0]
            for f in builder.recurse_included_files()
        }
        assert top_level == {"anneal_memory"}, (
            f"The wheel's top-level surface changed: {sorted(top_level)}. It must "
            "contain the package and nothing else."
        )
