"""Bind a caller-sized list of keys to a statement without the statement
growing with the list.

An inline ``IN (?, ?, ...)`` binds one variable per key, and SQLite caps the
variable count per statement (999 on old builds, 32766 on stock ones). Where
the SQLite library has JSON1, the keys travel as ONE JSON-array parameter
read through ``json_each``; where it does not, the inline list is kept, so
such a build behaves exactly as before.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

_json_each: bool | None = None


def json_each_available() -> bool:
    """Whether this process's SQLite library has ``json_each`` (JSON1).

    Probed once: JSON1 is a property of the library, shared by every
    connection in the process.
    """
    global _json_each
    if _json_each is None:
        try:
            probe = sqlite3.connect(":memory:")
            try:
                probe.execute("SELECT value FROM json_each('[]')").fetchall()
            finally:
                probe.close()
            _json_each = True
        except sqlite3.Error:
            _json_each = False
    return _json_each


def key_list(keys: Iterable[str]) -> tuple[str, list[str]]:
    """``(fragment, params)`` for ``<expr> IN <fragment>`` over ``keys``.

    Keys are bound as values, never formatted into the SQL. Use the
    fragment once per bound copy of ``params``.
    """
    keys = list(keys)
    if json_each_available():
        return "(SELECT value FROM json_each(?))", [json.dumps(keys)]
    return "(" + ",".join("?" * len(keys)) + ")", keys
