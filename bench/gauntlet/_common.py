"""Shared helpers. main refuses a fresh pattern line at 2x ("claimed a level the prior
continuity does not support"), so a row that wants a 2x graduation climbs honestly: a 1x wrap
on one episode, then a 2x wrap on episodes recorded after it (a citation of an episode an
earlier wrap already compressed is demoted; see false_demotion's info metric)."""
import warnings
from typing import Any

from anneal_memory import Store, prepare_wrap, validated_save_continuity

WRAP1_DAY = "2026-10-07"
WRAP2_DAY = "2026-10-08"


def wrap(store: Store, pattern_lines: list[str], day: str, *, record_warnings: bool = False):
    """prepare_wrap + validated_save_continuity on `pattern_lines`.
    Returns the result dict, plus the caught warnings when `record_warnings`."""
    token = prepare_wrap(store, max_chars=40000)["wrap_token"]
    doc = ("# t\n\n## State\ns.\n\n## Patterns\n" + "\n".join(pattern_lines) +
           "\n\n## Decisions\nd.\n\n## Context\nc.\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = dict(validated_save_continuity(store, doc, today=day, wrap_token=token))
    return (r, [str(x.message) for x in w]) if record_warnings else r


def cite(episode: Any, explanation: str) -> str:
    return f'[evidence: {episode.id[:8]} "{explanation}"]'
