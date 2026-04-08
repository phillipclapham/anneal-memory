"""Engine — LLM orchestration for programmatic compression.

For applications that want automated session compression without
an interactive agent. The Engine calls the LLM, validates the output,
and saves the result — all in one step.

Usage with Anthropic API key::

    from anneal_memory import Engine, Store

    store = Store("./memory.db")
    engine = Engine(store, api_key="sk-ant-...")
    result = engine.wrap()

Usage with custom LLM callable::

    engine = Engine(store, llm=lambda prompt: my_llm(prompt))
    result = engine.wrap()

The Engine is the ``[engine]`` optional extra::

    pip install anneal-memory[engine]

Zero dependencies beyond Python stdlib when using a custom LLM callable.
The ``anthropic`` package is only required when using ``api_key``.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import date
from typing import Callable

from .associations import process_wrap_associations
from .continuity import (
    build_engine_prompt,
    format_episodes_for_wrap,
    measure_sections,
    validate_structure,
)
from .graduation import detect_stale_patterns, validate_graduations
from .store import Store
from .types import AffectiveState, Episode, WrapResult


def _log(msg: str) -> None:
    """Log to stderr with engine prefix."""
    sys.stderr.write(f"[anneal-engine] {msg}\n")
    sys.stderr.flush()


def _make_anthropic_llm(
    api_key: str, model: str, max_tokens: int
) -> Callable[[str], str]:
    """Create an LLM callable using the Anthropic SDK.

    Raises ImportError with install instructions if anthropic is not available.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "The anthropic package is required for Engine with api_key. "
            "Install it with: pip install anneal-memory[engine]"
        ) from None

    client = anthropic.Anthropic(api_key=api_key)

    def call(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return call


class Engine:
    """LLM-driven session compression engine.

    Orchestrates the full wrap cycle: gather episodes from the store,
    build a compression prompt, call the LLM, validate the output
    (structure + graduation citations), and save the continuity file.

    Provide either ``api_key`` (uses Anthropic SDK) or ``llm``
    (any callable that takes a prompt string and returns a string).

    Args:
        store: The Store instance to compress from/to.
        llm: Custom LLM callable ``(prompt: str) -> str``.
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY``
                 env var if not provided.
        model: Model name for the Anthropic API.
        max_tokens: Max output tokens for the LLM call.
        max_chars: Max continuity file size in characters.
    """

    def __init__(
        self,
        store: Store,
        *,
        llm: Callable[[str], str] | None = None,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        max_chars: int = 20000,
        characterize_affect: bool = False,
    ) -> None:
        self._store = store
        self._max_chars = max_chars
        self._characterize_affect = characterize_affect

        if llm is not None:
            self._llm = llm
        else:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                raise ValueError(
                    "Either llm callable or api_key "
                    "(or ANTHROPIC_API_KEY env var) is required"
                )
            self._llm = _make_anthropic_llm(key, model, max_tokens)

    # -- Public API --

    def wrap(self) -> WrapResult:
        """Run the full wrap cycle.

        1. Mark wrap started (for stale-wrap detection)
        2. Gather episodes since last wrap
        3. Skip if no new episodes
        4. Format episodes + build compression prompt
        5. Call LLM
        6. Validate structure (4 required sections)
        7. Validate graduation citations
        8. Truncate if over max_chars
        9. Save continuity file + update metadata
        10. Record wrap completion in store

        Returns:
            WrapResult with metrics and continuity_text.
            ``saved=False`` if no episodes to compress or if
            structure validation failed with existing continuity as fallback.
        """
        # Check for episodes BEFORE marking wrap started —
        # matches MCP server ordering, avoids unnecessary contention
        episodes = self._store.episodes_since_wrap()
        if not episodes:
            _log("No episodes since last wrap — skipping")
            return WrapResult(
                saved=False,
                chars=0,
                section_sizes={},
                episodes_compressed=0,
            )

        self._store.wrap_started()

        try:
            existing = self._store.load_continuity()
            return self._compress_validate_save(episodes, existing)

        except Exception:
            # Clear wrap-in-progress on failure so the store
            # doesn't think a wrap is still running
            self._store.wrap_cancelled()
            raise

    # -- Properties --

    @property
    def store(self) -> Store:
        """The underlying Store instance."""
        return self._store

    @property
    def max_chars(self) -> int:
        """Maximum continuity file size in characters."""
        return self._max_chars

    # -- Internal pipeline --

    def _characterize_affective_state(self) -> AffectiveState | None:
        """Ask the LLM to characterize its functional state after compression.

        Makes a short second LLM call. Returns None if parsing fails
        (graceful degradation — associations still form at base strength).
        """
        prompt = (
            "In exactly one line, characterize your functional state during "
            "that compression. Format: TAG INTENSITY\n"
            "TAG = one word (engaged, curious, uncertain, frustrated, calm, focused, etc.)\n"
            "INTENSITY = number 0.0-1.0 (how strongly you felt it)\n"
            "Example: engaged 0.8\n"
            "Respond with ONLY the tag and number, nothing else."
        )
        try:
            raw = self._llm(prompt).strip().lower()
            parts = raw.split()
            if len(parts) >= 2:
                tag = parts[0]
                intensity = float(parts[1])
                intensity = max(0.0, min(1.0, intensity))
                _log(f"Affective state: {tag} {intensity:.1f}")
                return AffectiveState(tag=tag, intensity=intensity)
        except (ValueError, IndexError, Exception) as e:
            _log(f"Affective characterization failed (non-fatal): {e}")
        return None

    def _compress_validate_save(
        self,
        episodes: list[Episode],
        existing_continuity: str | None,
    ) -> WrapResult:
        """Core compression pipeline: prompt -> LLM -> validate -> save."""
        today = date.today().isoformat()

        # Format episodes for the prompt
        formatted = format_episodes_for_wrap(episodes)

        # Detect stale patterns and build section for the prompt
        stale_section = ""
        if existing_continuity:
            stale = detect_stale_patterns(existing_continuity, today)
            if stale:
                stale_lines = [
                    "\n## Stale Patterns (remove or re-validate these)",
                    "The following patterns have not been validated recently:\n",
                ]
                for s in stale:
                    stale_lines.append(
                        f"- Line {s.line_number}: {s.content} "
                        f"({s.days_stale} days stale)"
                    )
                stale_section = "\n".join(stale_lines) + "\n"

        # Build association context for episodes in this wrap window
        episode_ids = [ep.id for ep in episodes]
        assoc_context = self._store.get_association_context(episode_ids)
        assoc_section = ""
        if assoc_context:
            assoc_section = "\n" + assoc_context + "\n"

        # Build compression prompt (stale patterns + associations injected)
        prompt = build_engine_prompt(
            session_summary=formatted,
            existing_continuity=existing_continuity,
            project_name=self._store.project_name,
            max_chars=self._max_chars,
            today=today,
            stale_patterns_section=stale_section,
            association_section=assoc_section,
        )

        _log(f"Compressing {len(episodes)} episodes (max {self._max_chars} chars)")

        # Call LLM
        raw_output = self._llm(prompt)
        text = raw_output.strip()

        # Validate structure (4 required sections)
        if not validate_structure(text):
            _log("WARNING: LLM output missing required sections")
            self._store.wrap_cancelled()
            if existing_continuity:
                _log("Falling back to existing continuity")
                return WrapResult(
                    saved=False,
                    chars=len(existing_continuity),
                    section_sizes=measure_sections(existing_continuity),
                    episodes_compressed=0,
                    continuity_text=existing_continuity,
                )
            # First session, no fallback — reject rather than persist garbage.
            # Episodes remain in the store for the next wrap attempt.
            _log("No fallback available — rejecting invalid output (first session)")
            return WrapResult(
                saved=False,
                chars=0,
                section_sizes={},
                episodes_compressed=0,
            )

        # Validate graduation citations
        valid_ids = {ep.id[:8].lower() for ep in episodes}
        content_map = {ep.id[:8].lower(): ep.content for ep in episodes}
        meta = self._store.load_meta()

        grad_result = validate_graduations(
            text=text,
            valid_ids=valid_ids,
            today=today,
            node_content_map=content_map,
            citations_seen=meta.get("citations_seen", False),
        )
        text = grad_result.text

        if grad_result.demoted or grad_result.bare_demoted:
            _log(
                f"Graduations: {grad_result.validated} validated, "
                f"{grad_result.demoted + grad_result.bare_demoted} demoted"
            )
        if grad_result.citation_reuse_max > 2:
            _log(
                f"Warning: single episode cited "
                f"{grad_result.citation_reuse_max} times"
            )

        # Optionally characterize affective state after compression
        affective_state = None
        if self._characterize_affect:
            affective_state = self._characterize_affective_state()

        # Record Hebbian associations from validated co-citations + decay
        assoc_formed, assoc_strengthened, assoc_decayed = \
            process_wrap_associations(self._store, grad_result, affective_state)

        # Truncate if over budget
        if len(text) > self._max_chars:
            _log(
                f"Output {len(text)} chars exceeds "
                f"max {self._max_chars} — truncating"
            )
            text = _truncate_to_sections(text, self._max_chars)

        # Measure sections
        section_sizes = measure_sections(text)
        patterns = len(re.findall(r"\|\s*\d+x", text))

        # Save continuity file
        self._store.save_continuity(text)

        # Update metadata — match server logic: citations_seen triggers
        # when LLM demonstrates any citation ability (valid or attempted)
        if grad_result.validated > 0 or grad_result.citation_counts:
            meta["citations_seen"] = True
        meta["sessions_produced"] = meta.get("sessions_produced", 0) + 1
        self._store.save_meta(meta)

        # Record wrap completion in store
        total_demoted = grad_result.demoted + grad_result.bare_demoted
        result = self._store.wrap_completed(
            episodes_compressed=len(episodes),
            continuity_chars=len(text),
            graduations_validated=grad_result.validated,
            graduations_demoted=total_demoted,
            citation_reuse_max=grad_result.citation_reuse_max,
            patterns_extracted=patterns,
            associations_formed=assoc_formed,
            associations_strengthened=assoc_strengthened,
            associations_decayed=assoc_decayed,
        )
        # Attach fields that wrap_completed doesn't know about
        result.continuity_text = text
        result.section_sizes = section_sizes
        return result


# -- Module-level helpers --


def _truncate_to_sections(text: str, max_chars: int) -> str:
    """Truncate to fit max_chars by dropping entire sections from the end.

    Only includes COMPLETE sections that fit within the budget. Never cuts
    mid-section — a partial section can leave unclosed ``{}`` pattern blocks
    or malformed markers that corrupt the next wrap's input.

    Losing a section is recoverable (episodes remain in the store for the
    next wrap). Corrupting a section cascades through all future wraps.
    """
    if len(text) <= max_chars:
        return text

    lines = text.split("\n")
    section_starts: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            section_starts.append(i)

    if not section_starts:
        # No sections found — hard truncate (shouldn't happen with valid output)
        return text[:max_chars]

    # Include everything before first section (title line, etc.)
    result_lines: list[str] = list(lines[: section_starts[0]])

    # Add complete sections until the next one would exceed the limit
    sections_included = 0
    for idx, start in enumerate(section_starts):
        end = (
            section_starts[idx + 1]
            if idx + 1 < len(section_starts)
            else len(lines)
        )
        section_lines = lines[start:end]
        candidate = "\n".join(result_lines + section_lines)

        if len(candidate) > max_chars:
            # This section won't fit — stop here, don't include it
            break

        result_lines.extend(section_lines)
        sections_included += 1

    if sections_included == 0:
        _log(
            f"WARNING: max_chars ({max_chars}) too small to fit even the "
            f"first section — all sections dropped. Result will fail "
            f"validate_structure on next wrap."
        )

    return "\n".join(result_lines)
