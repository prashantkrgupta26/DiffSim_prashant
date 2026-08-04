# Remote-deploy workflow (office Mac → gpubox / Nova)

The Mac thinks and orchestrates; a headless Claude on gpubox executes GPU work.
The Mac is the sole GitHub gatekeeper. Spec: `docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md`.

**Source of truth:** the office Mac (this Dropbox-synced checkout) is the BRAIN and
its DiffSim repo is the canonical MASTER. gpubox and Nova are compute workers:
code flows Mac→worker, results+commits flow worker→Mac and are reconciled into
master. Never treat a worker's tree as authority; if a worker is ahead/uncommitted,
fetch and reconcile it back here rather than developing against the worker copy.

## One-time
- `bash scripts/remote/remote-doctor.sh --fix` — verifies the box and adds the `gpubox` git remote.
- Nova: `ssh nova true` once to establish the Duo `ControlMaster` (reused 7 days).
- **Box-Claude auth**: `ssh gpubox` then `claude auth login` — the OAuth session expires periodically; dispatch skips until this is done (`test_dispatch_roundtrip` will SKIP until authed). Dispatch references the box Claude by absolute path via the `CLAUDE_BIN` config var (default `/home/bglab/.local/bin/claude`), so it does not depend on a login-shell PATH.

## Daily loop (gpubox)
1. **Author** on the Mac (this repo).
2. **Sync**: `bash scripts/remote/gpubox-sync.sh` (refuses if a solve is running or the box has uncommitted work).
3. **Dispatch** a scoped task: `bash scripts/remote/gpubox-dispatch.sh "…task…"`.
   - The box-Claude should **commit its work locally** (it cannot push).
   - For a long solve it should launch via `gpubox-run.sh` and return the handle.
4. **Poll**: `bash scripts/remote/gpubox-poll.sh <log-path>`.
5. **Retrieve**: `bash scripts/remote/gpubox-fetch.sh`, review, then **you** push to origin.

## Hazards (encoded, not remembered)
- Run-lock blocks sync mid-solve (Warp lazy compile). Poll it out first.
- Logs are `tee`'d to files; polling reads files (no SIGPIPE).
- All remote python is `.venv/bin/python` — never bare `python`.

## gpubox flakiness
Intermittent campus-link / clock-governor pathology (handoff §6). If ssh drops
or fresh processes run 10–100× slow, retry; a host reboot is the cure if persistent.

## Nova
Scripts are stubbed. Wire `nova-sync-submit.sh` / `nova-poll.sh` on the next kit.

## gpubox-test.sh — the pre-merge suite gate (added 2026-07-20)

`scripts/remote/gpubox-test.sh [pytest-args]` runs the pytest suite on
gpubox with `pytest -n ${JOBS:-12}` (xdist): FORCE-syncs the CURRENT
working tree (works from a worktree too — the `.git` exclude covers the
worktree pointer file), ensures pytest-xdist in the box venv, launches
under the run-lock, polls to completion, prints the summary + FAILED
lines, exits nonzero on failure. Full suite ≈ minutes for the parallel
bulk vs 80+ min single-process on the Mac, and it covers CUDA-only
behavior the Mac cannot see (GPU atomic scatter is few-ULP, not
bit-equal — caught 2026-07-19).

Known-environmental: the two `test_stepper_solvers` amgx parity tests
need `libamgxsh.so` on the loader path; the script probes
`$HOME/AMGX/build` and `/usr/local/lib` and proceeds without it (tests
fail-not-skip until AMGX is rebuilt on the box).

Keep the box venv pinned to the repo: `.venv/bin/pip install -e ".[dev]"`
plus an explicit `warp-lang==<pinned>` when the floor moves (2026-07-20:
upgraded 1.14→1.15 to match Mac/Nova).
