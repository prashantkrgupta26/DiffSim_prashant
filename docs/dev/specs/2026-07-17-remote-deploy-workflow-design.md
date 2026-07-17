# Remote-deploy workflow — office Mac thinks, gpubox/Nova compute

**Design spec. 2026-07-17.** Companion runbook (to be written during
implementation): `docs/dev/remote-workflow.md`.

## Problem

The office Mac (`A43N6YJ9`) is the driving/thinking seat but has **no CUDA
GPU**. All production compute is GPU-only by program rule. Two GPU resources
exist: **gpubox** (2× RTX 6000 Ada, WSL2, ssh-key reachable) and **Nova** (Iowa
State SLURM cluster, interactive Duo auth). We need a workflow where the Mac
owns thinking + orchestration and dispatches actual GPU work to those machines
without the Mac ever needing a local GPU.

## Decisions (ratified in brainstorm 2026-07-17)

1. **Two brains.** A separate Claude Code runs on gpubox as a compute executor;
   this Mac session is the thinking + orchestration brain. (gpubox has Claude
   Code v2.1.211 installed at `/home/bglab/.local/bin/claude`.)
2. **Headless dispatch.** The Mac hands work to the box via
   `ssh gpubox 'claude -p "<task>"'` — non-interactive, autonomous, per scoped
   task. Long solves are launched **detached in tmux** by the box and polled
   from the Mac; the box returns a handle, not a blocked session.
3. **Sync model B for gpubox + git for Nova.** For gpubox, code moves over the
   direct ssh channel (rsync out, `git fetch` over ssh back); **the box never
   talks to GitHub origin**. The Mac is the sole origin gatekeeper. Nova stays
   git-mediated through origin because it is a cluster.
4. **Nova scripts are stubbed for now** — structure + usage + a "not yet wired"
   guard — and completed the next time a Nova kit is actually submitted.

## Architecture

| Component | Role |
|---|---|
| **Mac brain** (this session) | Thinking/authoring/papers/planning **+ orchestrator**. The only thing that talks to GitHub origin. Dispatches GPU tasks to gpubox; drives Nova via git+sbatch. |
| **gpubox box-Claude** (headless) | Pure compute executor. `claude -p`, autonomous, launches long solves detached in tmux and returns a handle, tees logs. Never touches origin. |
| **Nova** | Batch cluster. git-based (`fetch && reset --hard origin/master`) + `sbatch`, over the reused Duo `ControlMaster` socket. |

### Environment facts (measured 2026-07-16/17)

- **Mac:** repo `~/DiffSim`, remote `git@github.com:BaskarGS/diffsim.git` (ssh),
  on `master`.
- **gpubox** ssh alias reachable **non-interactively** via key (`Dell-Tower`,
  WSL2). Repo at `~/Baskar/DiffSim`; venv python at
  `~/Baskar/DiffSim/.venv/bin/python`; tmux present; git remote is **https +
  `~/.git-credentials`** (the pending-rotation PAT — avoided by sync model B).
- **Nova** ssh alias requires **interactive Duo auth** (BatchMode denied), but
  `~/.ssh/config` already has `ControlMaster auto` / `ControlPersist 7d`, so one
  user-established master connection is reusable for a week.

### Data flow

- **Mac → gpubox:** `rsync` (excludes `.git`, `.venv`, data/renders/papers/
  `__pycache__`), **gated on the run-lock** (see Hazards).
- **gpubox → Mac:** `git fetch gpubox` over ssh (gpubox added as an ssh git
  remote on the Mac). The Mac reviews the box's local commits, then pushes to
  origin as supervisor. No PAT involved in the gpubox loop.
- **Mac ↔ Nova:** through origin. Mac pushes → Nova fetch/reset → `sbatch` →
  poll `squeue`.

This keeps the program rules intact: **agents never push; the supervisor
verifies and pushes.** The box commits locally; the Mac (with the user)
reviews and is the sole pusher to origin.

## Components — `scripts/remote/`

A thin, documented set of POSIX shell scripts. No framework.

| Script | Purpose |
|---|---|
| `config.sh` | Sourced by all. Host aliases, remote repo paths, **venv python path** (hardcoded — never bare `python`), rsync excludes, model + permission flags, lock path. |
| `remote-doctor.sh` | Preflight: gpubox reachable+key, Claude version, venv python, tmux, **GPU health probe** (`gpu_burn.py`), Mac has the `gpubox` git remote, Nova master socket alive. |
| `gpubox-sync.sh` | rsync Mac→box working tree. **Hard-refuses if the run-lock is held.** Supports `--dry-run`. |
| `gpubox-dispatch.sh "<task>"` | Headless `claude -p` on the box, in the repo dir, with permission + model flags from `config.sh`. Returns the box's text to the Mac. |
| `gpubox-run.sh "<cmd>"` | Long-run primitive: detached `tmux new-session -d`, `tee` to timestamped `logs/…`, **writes the run-lock**, returns log path + tmux session name. |
| `gpubox-poll.sh <log>` | Reads the log **file** (`tail` on a file, never piping a producer → no SIGPIPE), reports tmux liveness, clears the lock when the run exits. |
| `gpubox-fetch.sh` | `git fetch gpubox` + show incoming commits/diff for supervisor review before the Mac pushes to origin. |
| `nova-sync-submit.sh <kit>` | **STUB.** Intended: ensure origin current, ssh nova (reuse master) fetch+reset, fill `FIXME_PARTITION`, `sbatch` the named kit, return job id. Guarded "not yet wired." |
| `nova-poll.sh <jobid>` | **STUB.** Intended: `squeue`/`sacct` status + slurm-log tail. Guarded "not yet wired." |

## Hazards encoded mechanically

These are hard-won ops rules (see `docs/dev/2026-07-15-mac-brainstorm-handoff.md`
Sec 5) turned into mechanism rather than memory:

- **Never edit a `.py` mid-run (Warp lazy kernel compile → phantom errors).**
  The **run-lock** (`~/Baskar/DiffSim/.remote-run.lock`) is written when a solve
  launches and cleared when it exits; `gpubox-sync.sh` hard-refuses while held.
- **Never pipe a producer through `head`/`tail` (SIGPIPE under `pipefail`).**
  Long runs `tee` to a file; polling reads the *file*. The producer is never in
  a pipe.
- **Never bare `python` on gpubox** (resolves to base conda, no warp). Every
  remote invocation uses the venv python from `config.sh`.

## Headless permissions (the one security trade-off)

The box Claude runs with `--dangerously-skip-permissions`. Justification: it
needs bash + write + GPU access autonomously, and gpubox is a single-user
trusted box the user owns. This is an explicit, accepted trade-off, isolated to
the box; the Mac brain runs under normal permissions. Model is a `config.sh`
knob (default: box's configured default, overridable per dispatch).

## Error handling

- Reachability check before every dispatch; on failure, a clear message.
- Nova master socket dead → explicit "re-auth Duo" prompt (the Mac cannot
  initiate Duo; the user must establish the master once).
- GPU-pathology probe before big campaigns — surfaces the two measured modes
  (clock-governor pinning at 210–450 MHz; fresh processes 10–100× slow
  regardless of reported clocks). Healthy baseline: 2685 MHz sustained,
  ~31.2 burn-launches/s.
- Run-lock prevents concurrent clobber of an in-flight solve.

## Testing / verification

- `remote-doctor.sh` returns all-green.
- A `nvidia-smi` dispatch round-trips text back to the Mac.
- A tiny detached run via `gpubox-run.sh` produces a pollable log and clears its
  lock on exit.
- `gpubox-fetch.sh` shows a clean no-op on an unchanged tree.
- `gpubox-sync.sh` refuses while a lock is held (negative test).

## Explicitly out of scope (YAGNI)

- Git-mediated task inbox (headless dispatch chosen instead).
- Any box→origin path (Mac is the sole origin gatekeeper).
- Fully-wired Nova scripts (stubbed until the next real kit submission).
- Two-way real-time messaging between the Mac and box Claudes (dispatch is
  request/response; long runs are poll-based).
