"""Regenerate tests/data/public_api_snapshot.json, the frozen public surface.

    python scripts/gen_public_api_snapshot.py            # rewrite the snapshot
    python scripts/gen_public_api_snapshot.py --check    # exit 1 if it differs

The snapshot is what ``tests/test_public_api.py`` compares against. Regenerate
it only for a change that docs/stability.md allows: an addition (MINOR), or a
deliberate removal or incompatible change (MAJOR), with a CHANGELOG line.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "tests" / "data" / "public_api_snapshot.json"

_ADDR_RE = re.compile(r" object at 0x[0-9a-f]+>")
_PRIMITIVES = (str, int, float, bool, type(None))


def _signature(obj: Any) -> str | None:
    try:
        # a sentinel default reprs with its address; the address is not the API
        return _ADDR_RE.sub(">", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return None


def _describe_class(cls: type) -> dict[str, Any]:
    methods: dict[str, str | None] = {}
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        raw = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
        if isinstance(raw, property):
            methods[name] = "property"
        elif callable(raw):
            methods[name] = _signature(raw)
    return {
        "kind": "class",
        "module": cls.__module__,
        "init": _signature(cls),
        "methods": methods,
    }


def _describe(name: str, obj: Any) -> dict[str, Any]:
    if inspect.isclass(obj):
        return _describe_class(obj)
    if inspect.isroutine(obj):
        return {"kind": "function", "module": obj.__module__, "signature": _signature(obj)}
    if inspect.ismodule(obj):
        return {"kind": "module", "name": obj.__name__}
    entry: dict[str, Any] = {"kind": "value", "type": type(obj).__name__}
    if isinstance(obj, _PRIMITIVES):
        entry["value"] = obj
    elif isinstance(obj, re.Pattern):
        entry["value"] = obj.pattern
    elif isinstance(obj, (tuple, frozenset, set)) and all(isinstance(x, _PRIMITIVES) for x in obj):
        entry["value"] = sorted(obj, key=repr) if not isinstance(obj, tuple) else list(obj)
    return entry


def _parser_tree(parser: argparse.ArgumentParser) -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    subcommands: dict[str, Any] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub in sorted(action.choices.items()):
                subcommands[sub_name] = _parser_tree(sub)
            continue
        if isinstance(action, argparse._HelpAction):
            continue
        choices = list(action.choices) if action.choices is not None else None
        default = action.default
        if default is not None and not isinstance(default, _PRIMITIVES):
            default = repr(default)
        options.append(
            {
                "dest": action.dest,
                "flags": list(action.option_strings),
                "nargs": action.nargs,
                "required": bool(getattr(action, "required", False)),
                "default": default,
                "choices": choices,
                "type": getattr(action.type, "__name__", None) if action.type else None,
                "action": type(action).__name__,
            }
        )
    options.sort(key=lambda o: (o["flags"] or [""], o["dest"]))
    return {"options": options, "subcommands": subcommands}


def collect() -> dict[str, Any]:
    import anneal_memory
    from anneal_memory import cli
    from anneal_memory.integrity import TOOLS

    exports = {
        name: _describe(name, getattr(anneal_memory, name))
        for name in sorted(anneal_memory.__all__)
    }
    return {
        "all": sorted(anneal_memory.__all__),
        "exports": exports,
        "cli": _parser_tree(cli.build_parser()),
        "mcp_tools": {t["name"]: t["inputSchema"] for t in sorted(TOOLS, key=lambda t: t["name"])},
    }


def render(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare, do not write")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    text = render(collect())
    if args.check:
        current = SNAPSHOT_PATH.read_text(encoding="utf-8") if SNAPSHOT_PATH.exists() else ""
        if current != text:
            print("public API snapshot differs; run scripts/gen_public_api_snapshot.py", file=sys.stderr)
            return 1
        return 0
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
