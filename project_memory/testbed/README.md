# anneal-memory testbed — clean-container experimental infrastructure

Infrastructure for running anneal-memory Session 10 (and beyond) experiments in containerized environments **deliberately isolated from Phill's global `~/.claude/CLAUDE.md` + flow system**. Clean substrate = clean measurement.

Moved here from `~/Documents/anneal-memory/testbed/` on 2026-04-23 to keep the anneal-memory repo focused on library code + tests + docs. Experimental infrastructure lives with flow's project management for anneal_memory.

---

## Why containerize experiments

Session 10+ experiments test anneal-memory's behavior under specific conditions (graduation gates, citation decay, immune system, sycophancy resistance under persistent-memory pressure). If those experiments run inside Phill's normal dev environment, the global `~/.claude/CLAUDE.md` + continuity.md + me.md context leaks into every Claude invocation — contaminating results because the model-under-test has flow system behavioral calibration built in.

Clean container isolates:
- Fresh Claude Code CLI install (no cached identity, no memory bootstrapping from Phill's filesystem)
- Fresh anneal-memory install from PyPI (no local dev tree, no test DB cross-contamination)
- No flow system context leakage
- Reproducible starting state across runs

Volumes preserve Claude auth + anneal-memory data across container restarts so you don't re-authenticate every run, but the container itself is reset-clean between experiment sessions.

---

## Files

- `Dockerfile` — Python 3.12-slim base + Claude Code CLI (native installer) + anneal-memory from PyPI. Mounts `/root/.claude` and `/root/.anneal-memory` as volumes for persistence.
- Future: experimental data files (session runs, JSON outputs, result logs) will accumulate here as Session 10+ experiments execute. Naming convention TBD at session kickoff.

---

## Build + run

```bash
cd ~/Documents/flow/projects/anneal_memory/testbed
docker build -t anneal-testbed .
docker run -it --rm \
    -v anneal-claude-auth:/root/.claude \
    -v anneal-data:/root/.anneal-memory \
    -v $(pwd):/workspace \
    anneal-testbed
```

First run: authenticate Claude Code via the in-container CLI. Auth persists in the `anneal-claude-auth` volume for subsequent runs.

---

## Session 10 plan placeholder

Session 10 is the first scheduled set of clean-env experiments. Planning lives in `../next.md` when scoped. Data artifacts produced during Session 10 (episode dumps, graduation logs, sycophancy benchmark outputs, citation-decay curves) will be captured in this directory under `session_10/` and committed to the flow repo as part of the experimental record.

Keep data artifacts structured:
- `session_10/raw/` — unprocessed experiment output (JSON, logs)
- `session_10/analysis/` — Python notebooks or analysis scripts
- `session_10/results/` — synthesized findings + markdown writeups

---

## Why this location

- Not in `anneal-memory` repo: Phill explicitly wants anneal-memory focused on library + tests + docs only. Infrastructure-for-running-experiments is not library code.
- Not in its own repo: too much ceremony for a Dockerfile + future data files. Co-location with project management is cleaner.
- Under `flow/projects/anneal_memory/`: this is where brief.md, next.md, commons_foundation.md, experiment_results.md already live. Infrastructure and planning co-located, one directory for anything you need to work on anneal-memory from a partnership-management perspective.
