# Remote-deploy workflow (office Mac → gpubox / Nova)

The Mac thinks and orchestrates; a headless Claude on gpubox executes GPU work.
The Mac is the sole GitHub gatekeeper. Spec: `docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md`.

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
