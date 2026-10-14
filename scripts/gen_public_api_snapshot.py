"""Regenerate tests/data/public_api_snapshot.json, the frozen public surface.

    python scripts/gen_public_api_snapshot.py            # rewrite the snapshot
    python scripts/gen_public_api_snapshot.py --check    # exit 1 if it differs

The snapshot is what ``tests/test_public_api.py`` compares against. Regenerate
it only for a change that docs/stability.md allows: an addition (MINOR), or a
deliberate removal or incompatible change (MAJOR), with a CHANGELOG line.

The snapshot is generated on one interpreter (its major.minor is recorded under
"python") and is independent of the environment: the CLI parser is built with
ANNEAL_MEMORY_DB and ANNEAL_MEMORY_SOURCE removed, and a default under the home
directory is written as "~/...". A value it cannot represent raises, naming the
export, rather than being recorded as a bare type.
"""

from __future__ import annotations

import argparse
import enum
import functools
import inspect
import json
import os
import re
import sys
import typing
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT / "tests" / "data" / "public_api_snapshot.json"

# only the address inside a repr (it ends at the repr's ">"), never text in a string
_ADDR_RE = re.compile(r" at 0x[0-9a-fA-F]+>")
_PRIMITIVES = (str, int, float, bool, type(None))
_ENV_VARS = ("ANNEAL_MEMORY_DB", "ANNEAL_MEMORY_SOURCE")
_PKG = "anneal_memory"


def _signature(obj: Any) -> str | None:
    try:
        # a sentinel default reprs with its address; the address is not the API
        return _ADDR_RE.sub(">", str(inspect.signature(obj)))
    except (TypeError, ValueError):
        return None


def _home_canon(text: str) -> str:
    home = str(Path.home())
    if home and (text == home or text.startswith(home + os.sep) or text.startswith(home + "/")):
        return "~" + text[len(home):].replace("\\", "/")
    return text


def _canon(value: Any, where: str) -> Any:
    """A JSON-safe, order-stable form of ``value``; unknown shapes raise."""
    if isinstance(value, str):
        return _home_canon(value)
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, enum.Enum):
        return f"{type(value).__qualname__}.{value.name}"
    if isinstance(value, dict):
        return {str(k): _canon(v, where) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v, where) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canon(v, where) for v in value), key=repr)
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": int(value.flags)}
    if typing.get_origin(value) is not None or type(value).__module__ == "typing":
        return _ADDR_RE.sub(">", str(value))
    raise TypeError(
        f"public API snapshot cannot represent {where}: {type(value).__module__}.{type(value).__qualname__}; "
        "add a serializer to _canon"
    )


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _member(name: str, owner: type, cls: type, raw: Any) -> dict[str, Any] | None:
    where = f"{_qualified(cls)}.{name}"
    if isinstance(raw, staticmethod):
        entry: dict[str, Any] = {"kind": "staticmethod", "signature": _signature(raw.__func__)}
    elif isinstance(raw, classmethod):
        entry = {"kind": "classmethod", "signature": _signature(raw.__func__)}
    elif isinstance(raw, (property, functools.cached_property)):
        settable = isinstance(raw, property) and raw.fset is not None
        entry = {"kind": "property", "settable": settable}
    elif type(raw).__name__ == "_tuplegetter":  # a NamedTuple field
        entry = {"kind": "field", "index": list(getattr(cls, "_fields", ())).index(name)}
    elif inspect.isclass(raw):
        entry = {"kind": "class", "signature": _signature(raw)}
    elif callable(raw):
        entry = {"kind": "method", "signature": _signature(raw)}
    else:
        entry = {"kind": "attribute", "value": _canon(raw, where)}
    if owner is not cls:
        entry["defined_in"] = _qualified(owner)
    return entry


def _describe_class(cls: type) -> dict[str, Any]:
    members: dict[str, Any] = {}
    enum_members: dict[str, str] | None = None
    if issubclass(cls, enum.Enum):
        enum_members = {n: repr(m.value) for n, m in cls.__members__.items()}
    for owner in cls.__mro__:
        if owner is object or not owner.__module__.startswith(_PKG):
            continue
        for name, raw in vars(owner).items():
            if name.startswith("_") or name in members:
                continue
            if enum_members is not None and name in enum_members:
                continue
            members[name] = _member(name, owner, cls, raw)
    out: dict[str, Any] = {
        "kind": "class",
        "module": cls.__module__,
        "bases": [_qualified(b) for b in cls.__bases__ if b is not object],
        "init": _signature(cls),
        "members": dict(sorted(members.items())),
    }
    if enum_members is not None:
        out["enum_members"] = enum_members
    if typing.is_typeddict(cls):
        out["typed_dict"] = {
            "annotations": {k: _ADDR_RE.sub(">", str(v)) for k, v in sorted(cls.__annotations__.items())},
            "required": sorted(cls.__required_keys__),
            "optional": sorted(cls.__optional_keys__),
        }
    return out


def _describe(name: str, obj: Any) -> dict[str, Any]:
    if inspect.isclass(obj):
        return _describe_class(obj)
    if inspect.isroutine(obj):
        return {"kind": "function", "module": obj.__module__, "signature": _signature(obj)}
    if inspect.ismodule(obj):
        return {"kind": "module", "name": obj.__name__}
    return {"kind": "value", "type": type(obj).__name__, "value": _canon(obj, name)}


def _type_name(t: Any) -> str | None:
    if t is None:
        return None
    qual = getattr(t, "__qualname__", None)
    if qual:
        mod = getattr(t, "__module__", "builtins")
        return qual if mod == "builtins" else f"{mod}.{qual}"
    return _ADDR_RE.sub(">", repr(t))


def _parser_tree(parser: argparse.ArgumentParser, path: str = "cli") -> dict[str, Any]:
    options: list[dict[str, Any]] = []
    subcommands: dict[str, Any] = {}
    subparsers: dict[str, Any] | None = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers = {"dest": action.dest, "required": bool(action.required)}
            for sub_name, sub in sorted(action.choices.items()):
                subcommands[sub_name] = _parser_tree(sub, f"{path} {sub_name}")
            continue
        if isinstance(action, argparse._HelpAction):
            continue
        where = f"{path} {action.dest}"
        options.append(
            {
                "dest": action.dest,
                "flags": list(action.option_strings),
                "nargs": action.nargs,
                "required": bool(getattr(action, "required", False)),
                "default": _canon(action.default, where),
                "const": _canon(action.const, where),
                "metavar": _canon(action.metavar, where),
                "choices": _canon(list(action.choices), where) if action.choices is not None else None,
                "type": _type_name(action.type),
                "action": type(action).__name__,
            }
        )
    options.sort(key=lambda o: (o["flags"] or [""], o["dest"]))
    exclusive = sorted(
        (
            {
                "options": sorted(a.option_strings[0] if a.option_strings else a.dest for a in g._group_actions),
                "required": bool(g.required),
            }
            for g in parser._mutually_exclusive_groups
        ),
        key=lambda g: g["options"],
    )
    # set_defaults: which handler a command dispatches to (L3 r2 codex)
    defaults = {k: (f"{v.__module__}.{v.__qualname__}" if callable(v) else _canon(v, f"{path} {k}"))
                for k, v in sorted(parser._defaults.items())}
    return {
        "options": options,
        "defaults": defaults,
        "subparsers": subparsers,
        "exclusive_groups": exclusive,
        "subcommands": subcommands,
    }


def _tools(tools: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in tools:
        if t["name"] in out:
            raise ValueError(f"duplicate MCP tool name: {t['name']}")
        out[t["name"]] = t["inputSchema"]
    return dict(sorted(out.items()))


def collect() -> dict[str, Any]:
    saved = {k: os.environ.pop(k) for k in _ENV_VARS if k in os.environ}
    try:
        import anneal_memory
        from anneal_memory import cli
        from anneal_memory.integrity import TOOLS

        exports = {
            name: _describe(name, getattr(anneal_memory, name))
            for name in sorted(anneal_memory.__all__)
        }
        return {
            "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
            "all": sorted(anneal_memory.__all__),
            "exports": exports,
            "cli": _parser_tree(cli.build_parser()),
            "mcp_tools": _tools(TOOLS),
        }
    finally:
        os.environ.update(saved)


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
