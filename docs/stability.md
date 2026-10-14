# Stability and versioning

anneal-memory follows semantic versioning from 1.0. This page says what the version number covers, how a change gets announced, and which environments are supported.

## What is public

Four things are public, and only these:

1. **The Python API.** Every name in `anneal_memory.__all__`. Each name is also frozen at the module path that defines it, so `from anneal_memory.durable import section_chars` and `from anneal_memory import section_chars` both stay valid. Names that start with an underscore, and public-looking names that are not in `__all__`, are internal and carry no promise, even when a module exposes them.
2. **The command line.** The subcommands, their flags, and the keys of the `--json` output of each command.
3. **The MCP server.** The tool names and each tool's input schema. The tool descriptions are guarded separately by the integrity manifest ([tests/test_integrity.py](../tests/test_integrity.py)); the schemas are guarded by the public API snapshot test.
4. **The files on disk.** See "On-disk format" below.

Anything not listed here can change in any release.

## Semantic versioning

- **MAJOR** (`2.0.0`): a removal of, or an incompatible change to, anything public.
- **MINOR** (`1.1.0`): an addition. New names, new flags, new subcommands, new MCP tools, new optional fields in output.
- **PATCH** (`1.0.1`): a fix. A fix may move public behaviour toward the behaviour the documentation already describes. It is still a PATCH, and it gets a line in the [CHANGELOG](../CHANGELOG.md).

## Deprecation

A public name, flag or field is deprecated before it is removed:

1. It stays working for at least one MINOR release after the deprecation is announced.
2. Using it raises a `DeprecationWarning`, and the [CHANGELOG](../CHANGELOG.md) has a line that names the replacement.
3. It is removed no earlier than the next MAJOR release.

The first deprecation is `allow_unlinked` (the `allow_unlinked` argument of `validated_save_continuity`, the CLI flag `--allow-unlinked`, and the MCP argument of the same name). It is accepted and does nothing today. The MCP tool's result already says it is deprecated and did nothing. The plan is to emit a `DeprecationWarning` from the library and the CLI through 1.x and remove the argument in 2.0.

## Supported environments

- **Python 3.10 or newer.** `pyproject.toml` sets `requires-python = ">=3.10"`.
- **CI** ([`.github/workflows/test.yml`](../.github/workflows/test.yml)) runs the full test suite on Linux under Python 3.10, 3.11, 3.12 and 3.13. It runs a reduced subset on Windows under Python 3.13, and runs `mypy anneal_memory/` on Python 3.13.
- **Typing.** The package ships `py.typed`. The mypy gate is part of CI.
- **Windows is single-writer.** Write locking is POSIX `fcntl` and there is no lock on Windows, so concurrent writers to one store can race or lose updates there. One process and one thread at a time is supported. The [README](../README.md) section "Windows" has the details. This is a known edge, not a bug that a release will fix silently.

## On-disk format

This section states what 1.0 will promise. It takes effect when the schema generation 2 migration lands, and it is not true of earlier releases.

- A store written by any release from 0.9.x on opens under 1.x without operator action.
- The migration is forward-only, additive and idempotent, and it copies the database before changing it.
- A store written by 1.x is refused by an older writer. The refusal is deliberate: it keeps an older release from writing into a store whose format it does not know.
- The sidecar files and the per-record formats (the audit log, the outcome log) are versioned. A record written by one release stays readable by every later release.
- A program that writes to a store with plain `sqlite3` calls `anneal_memory.register_writer_schema(conn)` on its connection first; it also sets `PRAGMA trusted_schema=ON` on that connection, and raises `sqlite3.NotSupportedError` if the connection will not take it. Reading needs nothing.

The mechanism of the migration is described in the CHANGELOG entry for the release that ships it.

## MCP surface

The MCP tool names and input schemas are frozen on the same terms as the Python API: a new optional argument or a new tool is a MINOR change, and removing or retyping an argument, or removing a tool, is a MAJOR change.
