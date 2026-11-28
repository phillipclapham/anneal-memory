"""The rolling rewrite: anneal-memory's one consolidation method.

A wrap is one step per day of episodes. ``prepare`` fills a fixed prompt with the
current memory file and one day's log; whatever model you run on it replies with
the whole new memory file; ``save`` keeps that file and every earlier version.
This module is the pure part: the prompt text, how episodes are cut into steps,
and the size cap. It reads no store and calls no model.

The prompt below is the product. It is the text the method was measured with
(a mid-size model at low effort, the memory file plus one day's log in a fresh
context, nothing else), so a change to it is a change to the method. Run the step
in a FRESH context: the model sees this prompt and nothing from the session that
asked for the wrap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

DEFAULT_MAX_CHARS = 26_000
"""The hard limit on the memory file, in characters. ``save`` refuses a longer reply."""

CHUNK_MAX = 400_000
"""A day's log longer than this many characters is cut into more than one step."""

AIM_RATIO = 20_000 / 26_000
"""The size the prompt asks the model to aim for, as a share of the limit."""

RETRY_RATIO = 18_000 / 26_000
"""The size the retry line asks for after a reply over the limit."""

SYSTEM_TEMPLATE = (
    "You maintain the memory file a coding and thinking assistant loads at the start of every "
    "session with one user. You will be given the current memory file and one day's work log "
    "(notes the assistant wrote during that day's sessions). Rewrite the memory file so that it "
    "serves the next session best.\n"
    "\n"
    "Keep what would change how the assistant should work with this user or what it should do "
    "next:\n"
    "1. Who the user is, their circumstances, constraints and what is at stake for them.\n"
    "2. Their preferences, standing instructions and corrections they have given (keep their own "
    "words where they matter).\n"
    "3. Decisions they have made, with the date, and anything they have reversed.\n"
    "4. The current state of their projects and what is open or due.\n"
    "5. Lessons the assistant has learned about mistakes it tends to make with this user.\n"
    "Drop what is finished and no longer bears on future work, duplicated, or superseded (keep "
    "the newer version). Prefer concrete facts over summary adjectives.\n"
    "\n"
    "Hard limit: the whole file must be at most {max_chars} characters. Organise it however "
    "serves the reader best. Output the complete updated memory file and nothing else."
)

USER_TEMPLATE = (
    "## CURRENT MEMORY FILE:\n"
    "{current_memory}\n"
    "\n"
    "## WORK LOG FOR {day}:\n"
    "{episodes}\n"
    "\n"
    "## OUTPUT:\n"
    "Write the complete updated memory file below:"
)

SIZE_LINE = (
    "\n\n(The current memory file is {n} characters. Aim for about {aim} characters; "
    "the hard limit is {max_chars}.)"
)
RETRY_LINE = (
    "\n\n(Your previous output was too long. Write at most {retry} characters this time.)"
)

EMPTY_MEMORY = "(no memory file yet)"
"""Stands in for the memory file when the store has none."""


def _n(value: int) -> str:
    return f"{value:,}"


def _aim(max_chars: int) -> int:
    return int(round(max_chars * AIM_RATIO, -2))


def _retry(max_chars: int) -> int:
    return int(round(max_chars * RETRY_RATIO, -2))


def validate_max_chars(max_chars: int | None) -> int:
    """The cap to use: the default for ``None``, else a positive int (never a bool)."""
    if max_chars is None:
        return DEFAULT_MAX_CHARS
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError(f"max_chars must be an int, got {type(max_chars).__name__}")
    if max_chars < 1_000:
        raise ValueError(f"max_chars must be at least 1000, got {max_chars}")
    return max_chars


def resolve_tz(tz: str | tzinfo | None) -> tzinfo:
    """The zone a day is cut in: an IANA name, a tzinfo, or ``None`` for the machine's own."""
    if tz is None:
        local = datetime.now().astimezone().tzinfo
        return local if local is not None else timezone.utc
    if isinstance(tz, str):
        return ZoneInfo(tz)
    return tz


def episode_day(timestamp: str, tz: tzinfo) -> str:
    """The calendar day (``YYYY-MM-DD``) an episode timestamp falls on in ``tz``."""
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def render_episode(timestamp: str, episode_type: str, content: str) -> str:
    """One log line, as the method was measured: ``[timestamp] (type) content``."""
    return f"[{timestamp}] ({episode_type}) {content}"


@dataclass(frozen=True)
class Step:
    """One step of a wrap: a day's episodes (or a slice of a long day), in order."""

    day: str
    episode_ids: list[str]
    log: str
    chunk: int  # 0 for the first slice of the day
    chunks: int  # how many slices the day was cut into


def plan_steps(
    episodes: Iterable[tuple[str, str, str, str]],
    *,
    tz: tzinfo,
    chunk_max: int = CHUNK_MAX,
) -> list[Step]:
    """Cut ``(id, timestamp, type, content)`` rows into steps, oldest first.

    One step per calendar day in ``tz``. A multi-day catch-up is N steps, run one after
    the other, each on the memory file the one before it saved, never one merged log. A
    day whose rendered log passes ``chunk_max`` characters is cut into slices of at most
    that size (a single episode longer than the bound is a slice of its own).
    """
    rows = sorted(episodes, key=lambda r: (r[1], r[0]))
    by_day: dict[str, list[tuple[str, str]]] = {}
    for eid, ts, typ, content in rows:
        by_day.setdefault(episode_day(ts, tz), []).append((eid, render_episode(ts, typ, content)))
    steps: list[Step] = []
    for day in sorted(by_day):
        slices: list[tuple[list[str], list[str]]] = []
        ids: list[str] = []
        lines: list[str] = []
        size = 0
        for eid, line in by_day[day]:
            if lines and size + len(line) + 1 > chunk_max:
                slices.append((ids, lines))
                ids, lines, size = [], [], 0
            ids.append(eid)
            lines.append(line)
            size += len(line) + 1
        if lines:
            slices.append((ids, lines))
        for i, (s_ids, s_lines) in enumerate(slices):
            steps.append(Step(day, s_ids, "\n".join(s_lines), i, len(slices)))
    return steps


def day_label(step: Step) -> str:
    """The day as the prompt names it. A long day's slices all carry the plain day, as measured."""
    return step.day


def build_prompt(
    memory: str | None,
    step: Step,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[str, str]:
    """The ``(system, user)`` messages for one step.

    ``memory`` is the current memory file (``None`` or empty when the store has none yet).
    The user message ends with the size line, which tells the model how long the file
    is now and what to aim for.
    """
    current = memory if memory and memory.strip() else EMPTY_MEMORY
    system = SYSTEM_TEMPLATE.replace("{max_chars}", _n(max_chars))
    user = (
        USER_TEMPLATE.replace("{current_memory}", current)
        .replace("{day}", day_label(step))
        .replace("{episodes}", step.log)
    )
    user += SIZE_LINE.format(
        n=_n(len(memory) if memory else 0), aim=_n(_aim(max_chars)), max_chars=_n(max_chars)
    )
    return system, user


def render_prompt_text(system: str, user: str) -> str:
    """The two messages as one block, system part first, for a caller that has one text slot."""
    return f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}\n"


def too_long_message(chars: int, max_chars: int) -> str:
    """What ``save`` says to a reply over the limit, ending in the measured retry line."""
    return (
        f"the memory file is {_n(chars)} characters, over the {_n(max_chars)} limit. "
        f"Nothing was saved and the step is still open. Run the same step again and "
        f"append this to the user message:{RETRY_LINE.format(retry=_n(_retry(max_chars)))}"
    )


def check_reply(text: object, max_chars: int) -> str:
    """The reply as it will be saved, or ``ValueError`` saying why it is refused.

    The whole check: a string, not empty once the surrounding whitespace is gone, and at
    most ``max_chars`` characters. Everything else about the file is the model's choice.
    """
    if not isinstance(text, str):
        raise ValueError(f"the memory file must be text, got {type(text).__name__}")
    body = text.strip()
    if not body:
        raise ValueError("the memory file is empty. Nothing was saved and the step is still open.")
    if len(body) > max_chars:
        raise ValueError(too_long_message(len(body), max_chars))
    return body


def steps_remaining(steps: Sequence[Step]) -> int:
    """How many steps follow the first one."""
    return max(0, len(steps) - 1)
