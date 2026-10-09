"""Seed a raw prior continuity for fixtures that graduate on their first save.

The prior-state bound (1007+29) cuts every pattern line to
``max(1, prior level + 1 if it validated this wrap)``, so a fixture whose first
save graduates a pattern to Nx needs that pattern at (N-1)x in a prior first.
A key with a space or a colon is written as a freeform line (``thought: ...``).
"""
from __future__ import annotations


def seed_prior_levels(store, levels, on="2026-01-01"):
    body = "".join(f"- {n} | {lv}x ({on})\n" for n, lv in levels.items())
    store.save_continuity(
        "## State\nseed.\n\n## Patterns\n" + body
        + "\n## Decisions\n- d.\n\n## Context\n- c.\n"
    )
