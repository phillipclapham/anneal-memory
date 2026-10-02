# Re-derive at load: STATE lines that prove themselves

The `project` section schema carries a `## State` section with the role
`derived-state`. Every non-blank line in that section ends with one of two
annotations:

```markdown
## State
- The working tree is at 0.9.16.dev0 [derive: grep -c "0.9.16.dev0" anneal_memory/__init__.py => 1]
- The v0.9.15 tag is on main [derive: git merge-base --is-ancestor v0.9.15 @REF]
- Release notes read well [judged: maintainer, 2026-09-30, against the CHANGELOG draft]
```

- `[derive: COMMAND => EXPECTED]` is a **value claim**. It agrees when the
  command's stdout, with whitespace collapsed, equals `EXPECTED`. Exit 0
  compares the output. Exit 1 compares it only for `grep` (`grep -c` prints
  `0` and exits 1 when nothing matches); for any other program exit 1 can
  follow partial output (`wc` after the files it could read,
  `git ls-files --error-unmatch` after the tracked names), so it is an error.
  Any other exit is an error.
- `[derive: COMMAND]` is a **truth claim**. Exit 0 agrees, exit 1 disagrees,
  and any other exit is an error. Write existence claims in a form that exits 1
  when false: `test -e PATH`, `grep -q`, `git merge-base --is-ancestor A B`,
  `git rev-parse --verify -q REF`. Git exits 128 on a missing ref or object,
  which is an error, and an error refuses the save.
- `[judged: WHO, WHEN, AGAINST WHAT]` marks a judgement that no command can
  check. It is accepted as written and never executed.
- `[derive@LABEL: …]` is either form, run in the root bound to `LABEL` for
  this store instead of the default root (see "Several roots for one store"
  below).

When the continuity is loaded with re-derive (`anneal-memory continuity
--rederive`, or `rederive_continuity()` in the library), each derive command
runs, and the line is flagged inline: `✓` when it agrees, `⚠ STALE` when it
disagrees, `⚠ DERIVE ERROR` when it errors, `⛔ REFUSED` when its command is
outside the allowlist below, `⚠ UNBOUND` when its label names no root bound
for this store, and `⚠ NOT DERIVED` when the load ran out of budget before
reaching it. Text loaded this way starts with a
`> [anneal re-derive]` line and can be saved back as it is. The save removes
that header only when it is the first line and the whole line has the shape
re-derive writes (its fixed opening, a count of State lines and its fixed
closing sentence), so an authored note is not deleted, and it always removes a flag after a State
line's closing `]`, so a forged `✓` is never persisted.
Re-deriving already re-derived text gives the same result as re-deriving the
original. `continuity --rederive` exits 3, not 0, when no State check ran and
answered: the store is not opted in, its schema has no derived-state section,
its State holds only `[judged: ...]` lines, every command was refused or failed
before it ran (a repository shape the rules below reject, say), or the load
budget ran out first. It says so on stderr, and the report's `clean` is false. Stale lines do not change the exit status; their flags are
the signal.

A save (`validated_save_continuity`) refuses a State section in which a line
carries no annotation or a refused command or label, and, on a store that is
enabled for re-derive, one whose command errors or whose label is unbound. Stale and not-derived lines do not
refuse the save: they are returned in the result's `stale_state` list (and in
the MCP tool's reply), with a warning after the save commits. On a store that
is not opted in, a save runs the static checks alone; pass `require_rederive=True`
(CLI `save-continuity --require-rederive`) to refuse such a save instead, and
also a save in which no State command ran. It is for a caller that checked the
opt-in before opening the wrap and must not commit unchecked lines if the trust
was revoked meanwhile. The MCP `save_continuity` tool does not take it.

`prepare_wrap` shows the composer the current continuity with each State line
flagged on an opted-in store; its instructions say the flags are stripped at
save, or that the store is not opted in. The header line is never handed to
the composer: moved anywhere but the first line, it would outlive the save. The
package's optional `unconfirmed_state` lists every State line that was not
confirmed (wider than the save's `stale_state`, which lists only the lines that
do not refuse a save). These are the same commands under the same containment as a load;
nothing runs on a store that is not opted in.

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
- No MCP tool writes the trust file, so an agent that only holds the MCP
  tools cannot enable execution for itself. (Once the user has opted a store
  in, an MCP `prepare_wrap` and an MCP save do run the State commands; see the
  last section.)
- The trust file is ignored unless the current user owns it and nobody else
  can write it, and a binding is ignored when the trust file sits inside the
  bound root (a trust file that could have arrived with the repository).
  `allow` and `revoke` refuse to rewrite a trust file they cannot read, rather
  than drop the bindings in it.
- The trust file is opened once, without following a symlink, and ownership
  and permissions are checked on that same descriptor before it is read, so it
  cannot be swapped between the check and the read. `allow` and `revoke` hold
  an exclusive lock on a sibling `.lock` file across their read-modify-write,
  so concurrent calls cannot lose an update (a revoke cannot be undone by a
  racing allow). `allow` refuses to write a binding the reader would ignore.
- `revoke` and `status` work on a store whose database has gone, so a stale
  binding can always be removed.
- The binding is by **path**. A different store copied over a trusted path
  inherits the trust; treat an allowed path as the user's own.
- Re-derive runs on POSIX systems only. Elsewhere the ownership checks, the
  no-follow open and the lock cannot be made, so nothing is ever trusted.
- `derive revoke` removes the binding.

#### Several roots for one store

A project can span repositories. A store binds one **default root** with
`derive allow --root DIR`, and then any number of **labelled roots**:

```sh
anneal-memory --db PATH derive allow --root REPO_DIR --visibility public
anneal-memory --db PATH derive allow --root OTHER_DIR --label other --visibility public
```

A `[derive@other: …]` line runs in `OTHER_DIR`; a `[derive: …]` line still
runs in the default root, so a store with one root is unchanged.

- A label is a name, never a path: lowercase letters, digits and `-`, at most
  32 characters. The text chooses among roots the user bound and cannot name
  one; a malformed label is refused, and a label with no bound root is
  `⚠ UNBOUND`, which is not clean and refuses an opted-in save (a claim that
  could not be checked never reads as one that held).
- **One visibility class per store.** A store with labelled roots declares
  `--visibility public` or `private` on every root, the same on all of them,
  and `allow` refuses a mismatch. A count oracle over a private repository's
  tracked files is thereby never reachable from a store whose roots are public.
  The declaration is the user's: anneal cannot see where a repository is
  published.
- A labelled root needs a default root first, may not be nested in (or hold)
  another root of the same store, and passes the same checks as the default
  root (the repository shape, the trust file outside it).
- The read side applies the same rules again, so a hand-edited trust file
  cannot widen what a store reaches: a labelled root whose visibility does not
  match, which is nested, or which is not a usable directory is dropped, and
  its lines read `⚠ UNBOUND`.
- `derive revoke --label NAME` drops one labelled root; `derive revoke` drops
  the store's whole binding. `derive status` lists every root that is honoured.
- A wrap freezes the root map its `prepare_wrap` read, with each root's
  directory identity (the device and inode of the root and of its `.git`): the
  flags the composer sees and the save's checks use the same roots. A root a
  saved `[derive…]` line runs in that is not the one frozen for its label (a
  label rebound, or bound for the first time, or the directory at its path
  replaced, while the composer worked) refuses the save; cancelling the wrap by its token and a new
  `prepare_wrap` recover. A root revoked meanwhile is left to the rules above;
  when every root was revoked the save goes through unchecked and warns that
  nothing was checked (`require_rederive` refuses it instead). A wrap prepared
  by a version without the freeze refuses its save while any root is bound
  and a derive line would run.
- The trust file keeps each store's default root where earlier versions read
  it, so a version without labels still finds the right default root. Such a
  version reads a `[derive@LABEL: …]` line as unannotated and refuses to save
  it; it never runs one. Write labelled lines only once every process that
  wraps the store understands them.

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
(`_GIT_FORMS`, `_VALIDATORS` and the `_validate_*` functions in
`anneal_memory/rederive.py`), and a refused command's message names the
allowed set. Anything else is refused, including flags that are harmless
today: the allowlist is what is known to be read-only, never what is not yet
known to be dangerous.

| Program | Allowed form |
|---|---|
| `git` | the subcommand must be the first argument (so `-c`, `-C`, `--exec-path`, `--git-dir` and every other global option are impossible), drawn from `rev-parse`, `describe`, `rev-list`, `merge-base --is-ancestor`, `cat-file -e`, `ls-files`, each with its own flag allowlist. Every one answers with an exit status, a commit id, a count, a tag name or a tracked path, never with file or commit content |
| `grep` | `grep FLAGS PATTERN PATH...` with short flags only, at least one of `-c` `-q` `-l` `-L` (it answers with a count or a yes/no and never prints file content), no `-r` / `-R` / `-e` / `-f`, and at least one path |
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
- `tag`, `branch`: their bare forms create refs.
- `log`, `cat-file -p` / `-t` / `-s`, `ls-tree`, `for-each-ref`: they print
  content, and review found repository configuration and metadata reaching
  through them three times (a `format.pretty` that brings back
  signature-verifying placeholders, a `mailmap.file` outside the root, an
  object store outside the root through `alternates`). They were removed
  rather than filtered: a guard defeated a new way each round has no bound.
- `diff --no-index`, `grep --no-index`: they read files outside the repository.
- `--output`, `--textconv`, `--filters`, `--ext-diff`, `--batch*`: they write
  files, run filters or read stdin.
- **No format strings at all** (`--format`, `--pretty`, `--date`), for the
  same reason.

Git is also launched with `--no-pager` and with configuration overrides that
switch off the fsmonitor hook, point hooks at an empty path, replace every gpg
program with `false`, switch off signature display and every mailmap source
(`log.mailmap`, `mailmap.file`, `mailmap.blob`), and with environment
variables that forbid every transport protocol (`GIT_ALLOW_PROTOCOL=none`,
which a repository's own `protocol.*.allow` cannot override), disable lazy
fetching in partial clones, terminal prompts and optional index locks. A read
cannot fetch, prompt, verify a signature or write.

### 4. File arguments stay inside the root

For `grep`, `wc` and `test`, every path argument must be relative, must not
contain `..` or pass through `.git` (in any letter case), must not pass
through a symlink anywhere along it, and its resolved real path must lie
inside the resolved root. A command is at most a few hundred characters and a
`grep` / `wc` line names at most a handful of paths (`_MAX_COMMAND_CHARS`,
`_MAX_PATHS`). This stops a crafted line from
reading `~/.ssh/id_rsa` or `/etc/passwd` and pasting the result into the
loaded context.

Inside the root, `grep` and `wc` read only files git tracks as regular files,
matched by exact name (`GIT_LITERAL_PATHSPECS`, index mode `100644` or
`100755`, so not a glob, a submodule or a tracked symlink): for a line that
names a directory or anything but a regular file (a FIFO, a device) is
refused; a line that
names an untracked or ignored file (a `.env`, a key) nothing is read, and the
claim is judged `⚠ STALE` (a file that left git is the most common real
drift, and it should not block a save). A crafted line cannot turn a count
into an oracle on a secret the repository never held; a tracked symlink to
one is refused by the symlink rule above. `test` may check that any path inside the root exists.

Git runs only in a plain repository: the root's `.git` must be a real
directory (not a gitfile or symlink, so not a linked worktree), `objects` and
`refs` must not be symlinks, there must be no `commondir` and no object
`alternates`, and no symlink anywhere under `.git` (a loose ref, `HEAD`,
`packed-refs`) may point out of the root. The repository's config may not
`include` another file (git's own parser decides, so git 2.25 or newer is
needed), there may be no `.git/config.worktree`, and git reads no global or system config
(`GIT_CONFIG_GLOBAL`, `GIT_CONFIG_NOSYSTEM`). Otherwise git would follow its own metadata out of the root, and
the line is a `⚠ DERIVE ERROR`.

Git cannot look above the root: `GIT_CEILING_DIRECTORIES` is set to the
root's parent, so a root that is a subdirectory of a larger repository (a
monorepo, a dotfiles repository in `$HOME`) gets no repository at all rather
than the parent's. Bind the root to a repository's top level.

### 5. The environment, the working directory and the ref are pinned

- The working directory is the resolved root from the trust file.
- The child environment is built from scratch (`_child_env` in
  `anneal_memory/rederive.py` is the list). Nothing else is inherited, so
  `GIT_DIR`, `GIT_WORK_TREE`, `LD_PRELOAD` and similar never reach the child.
- `PATH` keeps only absolute entries outside the root, and a program that
  resolves to a file inside the root is refused, so neither git nor a helper
  git starts (a gpg program, say) can be a repository file, and so a `grep` committed to the repository never
  runs in place of the system one. On Windows only a `.exe` runs (a `.bat` or
  `.cmd` wrapper would go through `cmd.exe`, a shell).
- stdin is closed.
- The ref is pinned once per root per load: `@REF` in a git argument is
  replaced with the full commit id that root's `HEAD` resolved to at the start
  of the load (the caller's `--ref` pins the default root only), so every line
  in one load is judged against the same commit of its root, and the report
  names each. `grep`, `wc` and `test` read the working
  tree, and the report says so.

### 6. Every run is bounded

Each command has a timeout and an output cap, and each load has a total time
budget and a cap on the number of lines it derives. A command that exceeds its
timeout or cap has its process group killed and counts as an error. A line
spends at most one timeout on the tracked-file check and one on its command,
and the load budget includes, for every root a load uses, the repository
shape check (the walk of `.git` and the config read) and resolving the pinned
ref. Every derive line counts toward the
line cap, refused ones too. A line is started only while at least one
line's worth (two command timeouts) of the load budget is left, so running out of budget never shows up as a command's own timeout;
lines not reached are flagged `⚠ NOT DERIVED` rather than silently passed. The exact limits are the `DEFAULT_*` constants in
`anneal_memory/rederive.py`.

### 7. Output cannot rewrite the document

Command output is shown in a flag only as a single line: control characters
are removed, whitespace is collapsed, and the value is truncated and quoted.
A command that prints `## Decisions` cannot inject a section into the loaded
continuity.

## What this does not cover

- **Reach across a store's own roots.** Any text that reaches a State section
  can point a count at any root bound to that store. The visibility class keeps
  public and private repositories apart; inside one class, binding a root is a
  statement that the store's text may count over its tracked files.
- **A false visibility declaration.** `--visibility` is taken as declared.
- **A root's commit moving during a wrap.** The freeze holds which directory
  each label names, not the commit it is at: a save re-checks every line at
  the commit each root has then.
- **A root replaced without a new identity.** A directory's identity is its
  device and inode, plus its creation time where the platform reports one.
  Linux reports none to Python, and ext4, xfs and tmpfs can give a freed inode
  to the next directory created, so there a replacement can match the frozen
  identity. The save takes the identities again after its commands ran, which
  catches a root replaced while they ran but not one swapped away and back.

- **A trusted user's own PATH.** The program is found through the absolute
  entries of `PATH`; a user whose `PATH` is hostile is already compromised.
- **A local actor who can already write inside the root.** Paths are checked,
  then the command runs: a concurrent swap to a symlink in between, or a hard
  link to a file outside the root, is not detected.
- **What the allowed git forms do print.** Commit ids, counts, tag names and
  tracked paths from the repository, including history that has been
  rewritten but not yet garbage collected, can reach the loaded text.
- **A child stuck in uninterruptible I/O.** A killed process that cannot be
  reaped within a couple of seconds (a hung network filesystem) is left
  behind with its reader threads; the line reports an error.
- **Repository contents the user trusts.** The allowed git forms do not run
  repository hooks, filters or diff drivers, but they do read the repository;
  opting a store in is a statement that the root is the user's own.
- **The truth of a `[judged: ...]` line.** It is accepted as written. It exists
  so a judgement is labelled as one rather than passed off as a fact.
- **The MCP surface.** The `anneal://continuity` resource returns the stored
  text without re-deriving. The `prepare_wrap` tool re-derives the stored
  continuity for the composer and returns the flags, including the values a
  stale or erroring line reports. The `save_continuity` tool runs the same save
  gate as the library. So on an opted-in store both execute the State
  commands, under the same containment, and a refused save's error message
  carries up to one flag's worth of a command's stderr per line. An agent can
  repeat refused saves, so treat an opted-in root as readable by the agent
  within the rules above. What MCP cannot do is opt a store in.
