"""AM-CRYSTAL-DECISION-CHANNEL — the structured routing-decision return.

Tests for ``parse_crystal_decisions`` (the fenced-pipe ``crystal-decisions`` block
parser + the structural never-compost-a-timeless gate) and for the prepare-wrap
instruction that teaches the format. The channel is the OUT half of node-3: anneal
PARSES, the harness ROUTES.
"""

from __future__ import annotations

import warnings

import pytest

from anneal_memory import (
    VALID_ACTIVATION_MODES,
    VALID_PERMANENCE,
    VALID_ROUTES,
    CrystalDecision,
    CrystalStore,
    parse_crystal_decisions,
)
from anneal_memory.continuity import _crystallization_block


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #

PATTERNS = """\
## Patterns

- verify_or_surface | 3x (2026-06-06) [evidence: a1b2c3d4, e5f6g7h8 "verify before \
acting on cached state"] — verify or surface before acting on cached state.
- verify_or_surface_before_acting | 2x (2026-06-05) [evidence: ffff0000 "the longer \
sibling"] — a distinct longer name that must not be matched by the shorter one.
- phase_thing | 2x (2026-06-01) — a phase-specific cold pattern with no evidence tag.
- bedrock_miss | 3x (2026-06-02) [evidence: 11112222 "catastrophic if missed"] — a \
miss corrupts the substrate.
"""


def _wrap(block_body: str, patterns: str = PATTERNS) -> str:
    return f"{patterns}\nSome narrative prose.\n\n```crystal-decisions\n{block_body}\n```\n"


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #

def test_happy_path_all_three_routes_in_document_order():
    wrap = _wrap(
        "verify_or_surface | crystallize | timeless | just-in-time\n"
        "bedrock_miss | constitution | timeless | catastrophic\n"
        "phase_thing | compost | phase-specific | just-in-time"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == [
        "verify_or_surface",
        "bedrock_miss",
        "phase_thing",
    ]
    assert [d.route for d in decisions] == ["crystallize", "constitution", "compost"]


def test_returns_crystaldecision_namedtuples():
    wrap = _wrap("verify_or_surface | crystallize | timeless | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert isinstance(d, CrystalDecision)
    assert d.name == "verify_or_surface"
    assert d.route == "crystallize"
    assert d.permanence == "timeless"
    assert d.activation_mode == "just-in-time"


def test_no_block_returns_empty():
    assert parse_crystal_decisions("just prose, no fenced block at all") == []


def test_empty_block_returns_empty():
    assert parse_crystal_decisions("```crystal-decisions\n```") == []


# --------------------------------------------------------------------------- #
# metadata extraction from the pattern line
# --------------------------------------------------------------------------- #

def test_metadata_pulled_from_matched_pattern_line():
    wrap = _wrap("verify_or_surface | crystallize | timeless | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 3
    assert d.evidence_ids == ["a1b2c3d4", "e5f6g7h8"]
    assert d.explanation == "verify or surface before acting on cached state."


def test_pattern_line_without_evidence_tag():
    wrap = _wrap("phase_thing | crystallize | phase-specific | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == []
    assert d.explanation == "a phase-specific cold pattern with no evidence tag."


def test_missing_pattern_line_yields_empty_metadata():
    # A constitution/compost decision needs no grounding, so a missing line is fine.
    wrap = _wrap("not_in_patterns | constitution | timeless | catastrophic")
    (d,) = parse_crystal_decisions(wrap)
    assert d.level is None
    assert d.explanation == ""
    assert d.evidence_ids == []


def test_substring_name_does_not_false_match_longer_sibling():
    # 'verify_or_surface' must pull ITS line (3x), never the longer
    # 'verify_or_surface_before_acting' (2x) whose name it is a prefix of.
    wrap = _wrap("verify_or_surface | crystallize | timeless | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 3
    assert d.evidence_ids == ["a1b2c3d4", "e5f6g7h8"]


def test_longer_sibling_pulls_its_own_line():
    wrap = _wrap(
        "verify_or_surface_before_acting | crystallize | timeless | just-in-time"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == ["ffff0000"]


# --------------------------------------------------------------------------- #
# the structural forget-gate (the non-negotiable risk gate)
# --------------------------------------------------------------------------- #

def test_compost_timeless_is_refused_and_warns():
    wrap = _wrap("verify_or_surface | compost | timeless | just-in-time")
    with pytest.warns(UserWarning, match="refused to compost the TIMELESS"):
        decisions = parse_crystal_decisions(wrap)
    assert decisions == []  # excluded — a timeless pattern is never dropped


def test_compost_timeless_excluded_but_other_rows_survive():
    wrap = _wrap(
        "verify_or_surface | compost | timeless | just-in-time\n"
        "phase_thing | compost | phase-specific | just-in-time"
    )
    with pytest.warns(UserWarning, match="verify_or_surface"):
        decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["phase_thing"]


def test_compost_phase_specific_is_allowed():
    wrap = _wrap("phase_thing | compost | phase-specific | just-in-time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected
        decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["phase_thing"]


def test_crystallize_timeless_is_allowed():
    wrap = _wrap("verify_or_surface | crystallize | timeless | just-in-time")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decisions = parse_crystal_decisions(wrap)
    assert decisions[0].route == "crystallize"


def test_constitution_timeless_catastrophic_is_allowed():
    wrap = _wrap("bedrock_miss | constitution | timeless | catastrophic")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decisions = parse_crystal_decisions(wrap)
    assert decisions[0].route == "constitution"


# --------------------------------------------------------------------------- #
# tolerance: malformed rows are skipped, never fatal
# --------------------------------------------------------------------------- #

def test_header_and_separator_rows_skipped():
    wrap = _wrap(
        "name | route | permanence | activation_mode\n"
        "--- | --- | --- | ---\n"
        "verify_or_surface | crystallize | timeless | just-in-time"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface"]


def test_unknown_enum_values_skipped():
    wrap = _wrap(
        "a | bogus_route | timeless | just-in-time\n"
        "b | crystallize | sometimes | just-in-time\n"
        "c | crystallize | timeless | whenever\n"
        "verify_or_surface | crystallize | timeless | just-in-time"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface"]


def test_wrong_cell_count_skipped():
    wrap = _wrap(
        "too | few\n"
        "way | too | many | cells | here | now\n"
        "no pipes at all just garbage\n"
        "verify_or_surface | crystallize | timeless | just-in-time"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface"]


def test_empty_name_skipped():
    wrap = _wrap(" | crystallize | timeless | just-in-time")
    assert parse_crystal_decisions(wrap) == []


def test_markdown_border_pipes_tolerated():
    wrap = _wrap("| verify_or_surface | crystallize | timeless | just-in-time |")
    (d,) = parse_crystal_decisions(wrap)
    assert d.name == "verify_or_surface"
    assert d.route == "crystallize"


def test_malformed_row_does_not_abort_the_set():
    # A fat-fingered row mid-block must not sink the rows around it.
    wrap = _wrap(
        "verify_or_surface | crystallize | timeless | just-in-time\n"
        "@@@ utterly broken @@@\n"
        "phase_thing | compost | phase-specific | just-in-time"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface", "phase_thing"]


# --------------------------------------------------------------------------- #
# multiple blocks
# --------------------------------------------------------------------------- #

def test_multiple_blocks_concatenated():
    wrap = (
        PATTERNS
        + "\n```crystal-decisions\nverify_or_surface | crystallize | timeless | "
        "just-in-time\n```\n\nmore prose\n\n```crystal-decisions\nphase_thing | "
        "compost | phase-specific | just-in-time\n```\n"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface", "phase_thing"]


def test_indented_fence_is_parsed():
    wrap = (
        PATTERNS
        + "\n    ```crystal-decisions\n    verify_or_surface | crystallize | "
        "timeless | just-in-time\n    ```\n"
    )
    decisions = parse_crystal_decisions(wrap)
    assert [d.name for d in decisions] == ["verify_or_surface"]


# --------------------------------------------------------------------------- #
# end-to-end: a crystallize decision feeds CrystalStore.crystallize
# --------------------------------------------------------------------------- #

def test_crystallize_decision_round_trips_into_the_store(tmp_path):
    wrap = _wrap("verify_or_surface | crystallize | timeless | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.route == "crystallize" and d.level == 3

    store = CrystalStore(tmp_path / "x.crystal.json")
    store.crystallize(
        name=d.name,
        level=d.level,
        explanation=d.explanation,
        evidence=d.evidence_ids,
        permanence=d.permanence,
        activation_mode=d.activation_mode,
        today=__import__("datetime").date(2026, 6, 6),
    )
    (row,) = store.active()
    assert row["name"] == "verify_or_surface"
    assert row["level"] == 3
    assert row["evidence"] == ["a1b2c3d4", "e5f6g7h8"]
    assert row["permanence"] == "timeless"
    assert row["activation_mode"] == "just-in-time"


# --------------------------------------------------------------------------- #
# instruction ↔ parser consistency (no enum drift)
# --------------------------------------------------------------------------- #

def _candidate(content: str, days_stale: int = 9):
    return {
        "line": 1,
        "content": content,
        "level": 3,
        "last_date": "2026-05-20",
        "days_stale": days_stale,
    }


def test_instruction_teaches_the_fenced_block():
    block = _crystallization_block([_candidate("- foo | 3x (2026-05-20) — bar")], None)
    assert "```crystal-decisions" in block
    assert "name | route | permanence | activation_mode" in block


def test_instruction_enum_vocab_matches_parser_constants():
    # Every enum value the parser accepts must appear in the instruction the agent
    # reads — else the channel teaches a vocabulary the parser rejects (drift).
    block = _crystallization_block([_candidate("- foo | 3x (2026-05-20) — bar")], None)
    for value in VALID_ROUTES + VALID_PERMANENCE + VALID_ACTIVATION_MODES:
        assert value in block, f"instruction missing enum value {value!r}"


def test_instruction_states_the_forget_gate():
    block = _crystallization_block([_candidate("- foo | 3x (2026-05-20) — bar")], None)
    low = block.lower()
    assert "compost" in low and "timeless" in low and "refused" in low


def test_no_decision_channel_when_no_out_candidates():
    # Only re-warm candidates (nothing routing OUT) → no decision-channel teaching.
    block = _crystallization_block(None, ["some_hot_pattern"])
    assert "```crystal-decisions" not in block


def test_instruction_header_matches_crystaldecision_field_order():
    # Pin the taught column order to the tuple structurally (L2) — reordering
    # CrystalDecision's fields without updating the instruction would fail here.
    block = _crystallization_block([_candidate("- foo | 3x (2026-05-20) — bar")], None)
    assert " | ".join(CrystalDecision._fields[:4]) in block


# --------------------------------------------------------------------------- #
# regression: grounding extraction against REALISTIC pattern lines
# (the clean PATTERNS fixture hid these — apparatus L1/L2)
# --------------------------------------------------------------------------- #

def test_explanation_not_hijacked_by_emdash_inside_evidence_quote():
    # L1 HIGH: the canonical WRITE-PATH evidence "why" routinely contains an
    # em-dash; a naive split('—') would fire INSIDE the quote and corrupt the
    # explanation fed to crystallize. The evidence span must be stripped first.
    wrap = (
        '- p | 3x (2026-06-06) [evidence: aa11bb22 "the harness fires hooks '
        'interactively — but skips them headless"] — the real felt prose.\n'
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.explanation == "the real felt prose."
    assert d.evidence_ids == ["aa11bb22"]


def test_quoted_only_why_falls_back_to_the_quote():
    # L1 MEDIUM: a line whose only explanation is the quoted "why" (no felt-prose
    # tail) must not yield "" (which ValueErrors in crystallize) — fall back to it.
    wrap = (
        '- p | 2x (2026-06-06) [evidence: aa11bb22 "the only explanation here"]\n'
        "\n```crystal-decisions\np | crystallize | phase-specific | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.explanation == "the only explanation here"
    assert d.level == 2


def test_regraduated_pattern_grounds_on_the_highest_level_line():
    # L2 MEDIUM: a sharpened-in-place pattern leaves a stale 2x line ABOVE the
    # fresh 3x line; first-match would ground on the stale one. Best-line wins.
    wrap = (
        '- foo | 2x (2026-05-01) [evidence: STALE001 "old"] — old graduation.\n'
        '- foo | 3x (2026-06-06) [evidence: FRESH001 "new"] — new graduation.\n'
        "\n```crystal-decisions\nfoo | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 3
    assert d.evidence_ids == ["FRESH001"]
    assert d.explanation == "new graduation."


def test_undated_prose_decoy_does_not_win_over_dated_graduation_line():
    # L1 LOW: an undated `name | Nx` mention in prose before the real graduation
    # line must not out-rank it. A dated line beats an undated one regardless of level.
    wrap = (
        "Earlier I mentioned bar | 9x in passing, with no date.\n"
        '- bar | 2x (2026-06-06) [evidence: real0001 "grounded"] — the real line.\n'
        "\n```crystal-decisions\nbar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2  # the dated 2x graduation line, not the undated 9x prose decoy
    assert d.evidence_ids == ["real0001"]


def test_markdown_emphasis_on_cells_is_tolerated():
    # L2 LOW: the instruction backtick-wraps enum names in its prose, inviting an
    # agent to bold a cell. Strip surrounding */backtick so it still validates.
    wrap = _wrap("verify_or_surface | **crystallize** | `timeless` | just-in-time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.route == "crystallize"
    assert d.permanence == "timeless"


def test_endash_in_hyphenated_enum_is_normalized():
    # L2 LOW: an agent typing en-dashes in 'phase-specific' / 'just-in-time'.
    wrap = _wrap("phase_thing | compost | phase–specific | just–in–time")
    (d,) = parse_crystal_decisions(wrap)
    assert d.route == "compost"
    assert d.permanence == "phase-specific"
    assert d.activation_mode == "just-in-time"


# --------------------------------------------------------------------------- #
# regression: codex L3 (regex / return-contract class)
# --------------------------------------------------------------------------- #

def test_hyphenated_longer_name_does_not_false_match_short_name():
    # codex L3 MEDIUM: '_' is a word char but '-'/'.' are not, so the old [^\w]
    # boundary let 'bar' grab 'foo-bar | 3x'. The name-alphabet lookbehind fixes it.
    wrap = (
        '- foo-bar | 3x (2026-06-06) [evidence: wrong1 "wrong"] — wrong longer pattern.\n'
        '- bar | 2x (2026-06-06) [evidence: right1 "right"] — right short pattern.\n'
        "\n```crystal-decisions\nbar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == ["right1"]
    assert d.explanation == "right short pattern."


def test_dotted_longer_name_does_not_false_match_short_name():
    wrap = (
        '- a.bar | 3x (2026-06-06) [evidence: wrong1 "wrong"] — dotted longer.\n'
        '- bar | 2x (2026-06-06) [evidence: right1 "right"] — short.\n'
        "\n```crystal-decisions\nbar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.evidence_ids == ["right1"]


def test_dated_prose_decoy_does_not_fake_has_date():
    # codex L3 MEDIUM: a date ANYWHERE on a line wrongly proved the marker dated.
    # The marker date is now captured at the marker, so a 'Reminder (date): p | 9x'
    # prose decoy reads as undated and loses to the real dated graduation line.
    wrap = (
        "Reminder (2026-06-06): p | 9x was a joke in prose, not a pattern line.\n"
        '- p | 2x (2026-06-06) [evidence: real1 "real"] — real pattern line.\n'
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == ["real1"]
    assert d.explanation == "real pattern line."


def test_unterminated_evidence_body_does_not_hang():
    # codex L3 pass #4: `[evidence:` repeated with NO closing `]` forced an O(n²)
    # full-scan (112s). The `]`-presence guard + bounded body keep it linear; this
    # returns promptly (a hang would stall the whole suite — that IS the assertion).
    text = (
        "## Patterns\n- p | 3x (2026-06-06) [evidence: " + ("[evidence:" * 30000) + "\n"
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(text)
    assert d.name == "p"
    assert d.evidence_ids == []  # the malformed unterminated tag yields no evidence


def test_far_close_evidence_body_is_fast_and_not_wrong_captured():
    # codex L3 pass #5: `[evidence:`×30000 then ONE far `]` was linear-but-slow (3s)
    # with `search` AND wrong-captured a 4096-char chunk as an id. find-first +
    # match-anchored does ONE bounded attempt at the first opener → fast, no capture.
    text = (
        "## Patterns\n- p | 3x (2026-06-06) [evidence: " + ("[evidence:" * 30000) + "]\n"
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(text)
    assert d.name == "p"
    assert d.evidence_ids == []  # the malformed repeated-opener body is NOT captured


def test_huge_level_digits_do_not_raise():
    # codex L3 MEDIUM: int() on a 5000-digit Nx raises ValueError (3.11+), breaking
    # the no-raise-on-str contract. The marker no longer matches a >3-digit level,
    # so the giant marker is simply ungrounded — never a raise.
    huge = "9" * 5000
    wrap = (
        f"- p | {huge}x (2026-06-06) — huge level.\n"
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)  # must not raise
    assert d.name == "p"
    assert d.level is None  # the pathological marker did not ground


def test_bracket_inside_evidence_quote_does_not_close_the_tag():
    # codex L3 MEDIUM: ']' inside the quoted why must not close [evidence: …] early
    # (the same explanation-corruption class as the em-dash HIGH, via a bracket).
    wrap = (
        '- p | 3x (2026-06-06) [evidence: aa11bb22 "why has ] — quoted tail"] — '
        "real felt prose.\n"
        "\n```crystal-decisions\np | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.explanation == "real felt prose."
    assert d.evidence_ids == ["aa11bb22"]


def test_balanced_brackets_inside_evidence_quote():
    # complement L3 FINDING 1: a '[edge cases]' bracket pair inside the quoted why
    # (no em-dash, evidence-terminated line) — the quote-aware regex consumes both.
    wrap = (
        '- foo | 2x (2026-06-06) [evidence: abc123 "handles [edge cases] properly"]\n'
        "\n```crystal-decisions\nfoo | crystallize | phase-specific | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.evidence_ids == ["abc123"]
    assert d.explanation == "handles [edge cases] properly"  # quoted-why fallback


def test_crlf_line_endings_parse():
    # complement "might be wrong about" #2: a CRLF wrap (Windows transport) must
    # still parse — row.strip() eats the \r, the fence anchors survive.
    wrap = (
        "- verify_or_surface | 3x (2026-06-06) [evidence: aa11 \"x\"] — prose.\r\n"
        "\r\n```crystal-decisions\r\n"
        "verify_or_surface | crystallize | timeless | just-in-time\r\n"
        "```\r\n"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.name == "verify_or_surface"
    assert d.route == "crystallize"
    assert d.level == 3


def test_unicode_dash_longer_name_does_not_false_match_short_name():
    # codex L3 convergence #2 MEDIUM: the suffix-boundary lookbehind must treat the
    # unicode en/em-dashes as name chars too (names preserve them verbatim), else
    # 'bar' grounds on the longer 'foo—bar | 3x' sibling (the '—' before 'bar' isn't
    # in the boundary alphabet, so the lookbehind wrongly passes).
    wrap = (
        '- foo—bar | 3x (2026-06-02) [evidence: wrong "w"] — wrong longer line.\n'
        '- bar | 2x (2026-06-01) [evidence: right "r"] — right short line.\n'
        "\n```crystal-decisions\nbar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == ["right"]
    assert d.explanation == "right short line."


@pytest.mark.parametrize("longer", ["foo/bar", "foo bar", "fooébar", "foo.bar", "foo~bar"])
def test_suffix_name_never_false_matches_longer_sibling_structural(longer):
    # codex L3 ×3: an enumerated boundary alphabet is always one char short (/, space,
    # accents, ~ all leaked). The STRUCTURAL anchor (name must be the first content
    # token after the bullet/marker prefix) ends the class — 'bar' can't ground on any
    # 'foo<sep>bar' longer sibling, whatever the separator char.
    wrap = (
        f'- {longer} | 3x (2026-06-02) [evidence: wrong "w"] — wrong longer line.\n'
        '- bar | 2x (2026-06-01) [evidence: right "r"] — right short line.\n'
        "\n```crystal-decisions\nbar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.level == 2
    assert d.evidence_ids == ["right"]
    assert d.explanation == "right short line."


def test_name_with_internal_slash_grounds_to_its_own_line():
    # the structural anchor must still GROUND a legitimate exotic-char name to its line.
    wrap = (
        '- a/b | 3x (2026-06-02) [evidence: real "r"] — the slash-named pattern.\n'
        "\n```crystal-decisions\na/b | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.name == "a/b"
    assert d.level == 3
    assert d.evidence_ids == ["real"]


def test_marker_prefixed_graduation_line_grounds():
    # FlowScript priority/done markers (! !! ✓) precede the name on a graduation line;
    # the prefix run must still let the name ground.
    wrap = (
        '  ! clean_room | 3x (2026-06-02) [evidence: ev1 "x"] — felt prose.\n'
        '✓ done_thing | 2x (2026-06-01) [evidence: ev2 "y"] — done prose.\n'
        "\n```crystal-decisions\n"
        "clean_room | crystallize | timeless | just-in-time\n"
        "done_thing | crystallize | phase-specific | just-in-time\n```"
    )
    decisions = parse_crystal_decisions(wrap)
    by_name = {d.name: d for d in decisions}
    assert by_name["clean_room"].level == 3
    assert by_name["clean_room"].evidence_ids == ["ev1"]
    assert by_name["done_thing"].level == 2
    assert by_name["done_thing"].evidence_ids == ["ev2"]


def test_emdash_name_does_not_fold_into_ascii_hyphen_sibling():
    # codex L3 CONVERGENCE MEDIUM: dash-folding the NAME cell made an em-dash name
    # ground onto a DIFFERENT ascii-hyphen pattern line (wrong-line match, silent
    # mis-grounding — NOT a safe miss). The name cell now preserves exact dashes.
    wrap = (
        '- foo-bar | 2x (2026-06-01) [evidence: ascii1 "a"] — ascii line.\n'
        '- foo—bar | 3x (2026-06-02) [evidence: uni1 "u"] — unicode line.\n'
        "\n```crystal-decisions\nfoo—bar | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.name == "foo—bar"  # exact em-dash name preserved, not folded to hyphen
    assert d.level == 3  # grounded on the em-dash line, not the ascii-hyphen sibling
    assert d.evidence_ids == ["uni1"]
    assert d.explanation == "unicode line."


def test_emphasis_on_pattern_line_name_is_a_safe_miss():
    # complement L3 FINDING 2 / kimi: a bolded name in the PATTERN LINE (not the
    # decision row) won't ground — documented as a SAFE miss (level None), not a
    # corruption. The instruction directs plain names; this locks the safe behavior.
    wrap = (
        '- **foo_pattern** | 3x (2026-06-06) [evidence: abc "why"] — prose.\n'
        "\n```crystal-decisions\nfoo_pattern | crystallize | timeless | just-in-time\n```"
    )
    (d,) = parse_crystal_decisions(wrap)
    assert d.route == "crystallize"  # the decision still parses
    assert d.level is None  # grounding safely failed (no corruption)
    assert d.explanation == ""
