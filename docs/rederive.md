# Re-derive at load: STATE lines that prove themselves

The `project` section schema carries a `## State` section with the role
`derived-state`. Every non-blank line in that section ends with one of two
annotations:

```markdown
## State
- The working tree is at 0.9.16.dev0 [derive: grep -c "0.9.16.dev0" anneal_memory/__init__.py => 1]
- The v0.9.15 tag is on main [derive: git merge-base --is-ancestor v0.9.15 @REF]
- Release notes read well [judged: Phill, 2026-09-30, against the CHANGELOG draft]
```

- `[derive: COMMAND => EXPECTED]` is a **value claim**. It agrees when the
  command's stdout, with whitespace collapsed, equals `EXPECTED`. Exit 0 and
  exit 1 both compare the output, because exit 1 means "no match" for every
  allowed program (`grep -c` prints `0` and exits 1). Any other exit is an
  error.
- `[derive: COMMAND]` is a **truth claim**. Exit 0 agrees, exit 1 disagrees,
  and any other exit is an error.
- `[judged: WHO, WHEN, AGAINST WHAT]` marks a judgement that no command can
  check. It is accepted as written and never executed.

When the continuity is loaded with re-derive (`anneal-memory continuity
--rederive`, or `rederive_continuity()` in the library), each derive command
runs, and the line is flagged inline: `✓` when it agrees, `⚠ STALE` when it
disagrees, `⚠ DERIVE ERROR` when it errors, and `⛔ REFUSED` when its command
is outside the allowlist below. A save (`validated_save_continuity`) refuses a
State section in which a line carries no annotation or a refused command, and,
on a store that is enabled for re-derive, refuses one whose command errors.

## Why this needs containment

Re-derive executes commands that are stored in a memory file. That file is
written by an agent during a wrap, from episodes, and episodes can carry text
from anywhere: review findings, commit messages, pasted web content. A STATE
line is therefore attacker-influenceable text, and running it without
containment would turn a prompt injection into code execution on the machine
that loads the memory.

The containment below was designed before the code. Each rule names the
attack it closes.

## The containment rules

### 1. Nothing runs unless the user opted this store in, on this machine

Re-derive is off for every store until the user runs:

```sh
anneal-memory --db PATH derive allow --root REPO_DIR
```

The opt-in is recorded **outside the store**, in a per-user trust file
(`~/.anneal-memory/derive-trust.json`, or the path in
`ANNEAL_MEMORY_DERIVE_TRUST`). It binds the store's resolved database path to
the resolved root directory the commands run in. This follows the `direnv
allow` model:

- A store that arrives from somewhere else (a clone, a copied file, a shared
  drive) is untrusted until its user allows it locally, because the opt-in does
  not travel with the store.
- The root the commands run in comes only from the trust file, never from the
  store or the continuity text, so a crafted memory cannot point execution at
  another directory.
- No MCP tool reads or writes the trust file, so an agent that only holds the
  MCP tools cannot enable execution for itself.
- `derive revoke` removes the binding.

On an untrusted store, a re-derive load returns the text with a one-line
header saying re-derive is not enabled and runs nothing. A save still performs
the static checks (annotation present, command inside the allowlist), which
never execute anything.

### 2. No shell, ever

Commands are split with `shlex` and run with `shell=False`, so the shell never
interprets anything. As defence in depth, a command containing any of
`` ; & | $ ` < > \ `` or a control character is refused before it is split:
none of them is needed by an allowed form, and their presence signals an
attempt.

### 3. An allowlist of read-only command forms, checked argument by argument

The program must be one of those below, and every argument must fit its
form. The table describes intent; the authority is the code
(`_GIT_FORMS` and the `_validate_*` functions in `anneal_memory/rederive.py`),
and a refused command's message names the allowed set. Anything else is refused, including flags that are harmless today: the
allowlist is what is known to be read-only, never what is not yet known to be
dangerous.

| Program | Allowed form |
|---|---|
| `git` | the subcommand must be the first argument (so `-c`, `-C`, `--exec-path`, `--git-dir` and every other global option are impossible), drawn from `rev-parse`, `describe`, `rev-list`, `merge-base`, `log`, `cat-file`, `ls-files`, `ls-tree`, `for-each-ref`, each with its own flag allowlist |
| `grep` | `grep [-cFEiwxqlLshHrvn...] PATTERN PATH...`, short flags only, at least one path |
| `wc` | `wc [-lcwm] PATH...`, at least one path |
| `test` | `test -e|-f|-d|-s PATH` |

Flags that take a value are accepted only in `--flag=value` form, so a value
can never be mistaken for a flag or a flag for a value. A positional argument
may not begin with `-`.

Git subcommands that are **excluded on purpose**, with the reason:

- `status`, `diff`, `show`, and anything producing a patch (`log -p`,
  `--stat`, `-S`, `-G`): they can run textconv drivers, external diff
  programs or the fsmonitor hook from repository configuration.
- `grep`: its `-O` / `--open-files-in-pager` flag runs a program.
- `tag`, `branch`: their bare forms create refs. `for-each-ref` covers
  listing.
- `diff --no-index`, `grep --no-index`: they read files outside the repository.
- `--output`, `--textconv`, `--filters`, `--ext-diff`, `--batch*`: they write
  files, run filters or read stdin.

Git is also launched with `--no-pager` and with configuration overrides that
switch off the fsmonitor hook, point hooks at an empty path and forbid every
network protocol, and with environment variables that disable lazy fetching in
partial clones, terminal prompts and optional index locks. A read cannot fetch,
prompt or write.

### 4. File arguments stay inside the root

For `grep`, `wc` and `test`, every path argument must be relative, and its
resolved real path (after symlinks) must lie inside the resolved root. This
stops a crafted line from reading `~/.ssh/id_rsa` or `/etc/passwd` and pasting
the result into the loaded context. `grep -R` (which follows symlinks) and
`grep -f` / `-e` are refused. Git keeps its own reads inside the repository.

### 5. The environment, the working directory and the ref are pinned

- The working directory is the resolved root from the trust file.
- The child environment is built from scratch: `PATH`, `HOME`, `LC_ALL=C` and
  the git settings above. Nothing else is inherited, so `GIT_DIR`,
  `GIT_WORK_TREE`, `LD_PRELOAD` and similar never reach the child.
- stdin is closed.
- The ref is pinned once per load: `@REF` in a git argument is replaced with
  the full commit id that `HEAD` (or the caller's `--ref`) resolved to at the
  start of the load, so every line in one load is judged against the same
  commit, and the report names it. `grep`, `wc` and `test` read the working
  tree, and the report says so.

### 6. Every run is bounded

Each command has a timeout and an output cap, and each load has a total time
budget and a cap on the number of lines it derives. A command that exceeds its
timeout or cap has its process group killed and counts as an error. Lines left
when the load budget runs out are flagged `⚠ NOT DERIVED` rather than silently
passed. The exact limits are the `DEFAULT_*` constants in
`anneal_memory/rederive.py`.

### 7. Output cannot rewrite the document

Command output is shown in a flag only as a single line: control characters
are removed, whitespace is collapsed, and the value is truncated and quoted.
A command that prints `## Decisions` cannot inject a section into the loaded
continuity.

## What this does not cover

- **A trusted user's own PATH.** The program is found through `PATH`; a user
  whose `PATH` is hostile is already compromised.
- **Repository contents the user trusts.** The allowed git forms do not run
  repository hooks, filters or diff drivers, but they do read the repository;
  opting a store in is a statement that the root is the user's own.
- **The truth of a `[judged: ...]` line.** It is accepted as written. It exists
  so a judgement is labelled as one rather than passed off as a fact.
- **The MCP resource.** `anneal://continuity` returns the stored text without
  re-deriving; re-derive is reached through the CLI and the library call only.
