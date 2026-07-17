# Remote-deploy Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the office Mac (no local GPU) a small shell toolkit that dispatches GPU work to gpubox (headless box-Claude) and, later, Nova, with the Mac as sole GitHub gatekeeper.

**Architecture:** Two brains — this Mac thinks/orchestrates; a headless Claude on gpubox executes. Code goes Mac→box by `rsync` (gated on a run-lock) and box→Mac by `git fetch` over ssh; the box never touches GitHub origin. Nova stays git-mediated but its scripts are stubbed for now. See spec: `docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md`.

**Tech Stack:** POSIX/bash scripts under `scripts/remote/`, ssh + rsync + tmux, pytest (subprocess) for gates — consistent with the repo's existing pytest suite.

## Global Constraints

- **Never bare `python` on gpubox** — always the venv python `~/Baskar/DiffSim/.venv/bin/python` (bare `python` = base conda, no warp). All remote python goes through `$GPUBOX_VENV_PY`.
- **Never edit a `.py` while a run imports it** (Warp lazy kernel compile → phantom errors) — enforced by the run-lock: `gpubox-sync.sh` hard-refuses while `~/Baskar/DiffSim/.remote-run.lock` exists.
- **Never pipe a producer through `head`/`tail`** (SIGPIPE under `pipefail`) — long runs `tee` to a file; polling reads the *file*.
- **Agents never push; the supervisor verifies and pushes** — the box commits locally; only the Mac pushes to origin.
- **The box never authenticates to GitHub** — Mac↔box code flows over the direct ssh channel only (removes the pending-rotation PAT from the gpubox loop).
- Remote host aliases already exist in `~/.ssh/config`: `gpubox` (ssh-key, non-interactive), `nova` (interactive Duo; `ControlMaster`/`ControlPersist 7d`).
- Box paths: repo `/home/bglab/Baskar/DiffSim`; Claude at `/home/bglab/.local/bin/claude` (v2.1.211).
- **Security trade-off (accepted):** the box-Claude runs `--dangerously-skip-permissions` (single-user trusted box). Isolated to the box; the Mac runs normal permissions.

---

### Task 1: Shared config (`config.sh`)

**Files:**
- Create: `scripts/remote/config.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- Produces (sourced by every other script): env vars `GPUBOX_HOST`, `GPUBOX_REPO`, `GPUBOX_REPO_ABS`, `GPUBOX_VENV_PY`, `GPUBOX_LOCK`, `GPUBOX_LOGDIR`, `GPUBOX_GIT_REMOTE`, `GPUBOX_GIT_URL`, `BOX_CLAUDE_FLAGS`, `BOX_MODEL`, `NOVA_HOST`, `REPO_ROOT`; and shell array `RSYNC_EXCLUDES`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
import pathlib, subprocess
REPO = pathlib.Path(__file__).resolve().parents[1]
RD = REPO / "scripts" / "remote"

def _source_var(var):
    out = subprocess.run(
        ["bash", "-c", f'source "{RD}/config.sh"; printf "%s" "${{{var}}}"'],
        capture_output=True, text=True)
    return out.stdout.strip()

def _sh(script, *args):
    # Shared helper: run a toolkit script, capture output. Used by many tasks.
    return subprocess.run(["bash", str(RD / script), *args],
                          capture_output=True, text=True)

def test_config_exports_core_vars():
    assert _source_var("GPUBOX_HOST") == "gpubox"
    assert _source_var("GPUBOX_REPO_ABS").endswith("Baskar/DiffSim")
    assert _source_var("GPUBOX_VENV_PY").endswith(".venv/bin/python")
    assert _source_var("GPUBOX_LOCK").endswith(".remote-run.lock")
    assert _source_var("GPUBOX_GIT_URL") == "gpubox:Baskar/DiffSim"
    assert _source_var("BOX_CLAUDE_FLAGS") == "--dangerously-skip-permissions"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && python -m pytest tests/test_remote_toolkit.py::test_config_exports_core_vars -v`
Expected: FAIL — `config.sh` does not exist (bash source error / empty vars).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/config.sh`:

```bash
#!/usr/bin/env bash
# scripts/remote/config.sh — shared config for the remote-deploy toolkit.
# SOURCE this (do not execute). All remote scripts source it.

REMOTE_TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_ROOT="$(cd "$REMOTE_TOOLKIT_DIR/../.." && pwd)"

# --- gpubox ---
export GPUBOX_HOST="${GPUBOX_HOST:-gpubox}"
export GPUBOX_REPO="${GPUBOX_REPO:-Baskar/DiffSim}"                 # relative to remote $HOME
export GPUBOX_REPO_ABS="${GPUBOX_REPO_ABS:-/home/bglab/Baskar/DiffSim}"
export GPUBOX_VENV_PY="${GPUBOX_VENV_PY:-$GPUBOX_REPO_ABS/.venv/bin/python}"
export GPUBOX_LOCK="${GPUBOX_LOCK:-$GPUBOX_REPO_ABS/.remote-run.lock}"
export GPUBOX_LOGDIR="${GPUBOX_LOGDIR:-$GPUBOX_REPO_ABS/logs}"
export GPUBOX_GIT_REMOTE="${GPUBOX_GIT_REMOTE:-gpubox}"             # name of ssh git remote on the Mac
export GPUBOX_GIT_URL="${GPUBOX_GIT_URL:-$GPUBOX_HOST:$GPUBOX_REPO}"

# Headless box-Claude flags (accepted security trade-off: single-user box).
export BOX_CLAUDE_FLAGS="${BOX_CLAUDE_FLAGS:---dangerously-skip-permissions}"
export BOX_MODEL="${BOX_MODEL:-}"   # empty = box default; set to override

# rsync excludes (Mac -> box): only source syncs; data/artifacts stay put.
# NOTE: bash cannot export arrays; scripts SOURCE this file so it stays in scope.
RSYNC_EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'logs/'
  --exclude 'data/'
  --exclude 'papers/'
  --exclude 'CriticalEvaluations/'
  --exclude '*.npz'
  --exclude '*.pt'
)

# --- nova (scripts stubbed for now) ---
export NOVA_HOST="${NOVA_HOST:-nova}"
export NOVA_REPO_ABS="${NOVA_REPO_ABS:-}"   # set when Nova scripts are wired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_config_exports_core_vars -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/config.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): shared config.sh for remote-deploy toolkit"
```

---

### Task 2: Pure helpers (`lib.sh`)

**Files:**
- Create: `scripts/remote/lib.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- Consumes: nothing (pure, no ssh).
- Produces: `rlog <msg>` / `rerr <msg>` (stderr loggers); `remote_log_path <logdir> <run_id> [tag]` → `<logdir>/<tag>-<run_id>.log`; `remote_tmux_name <run_id>` → `diffsim-<sanitized run_id>`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
def test_lib_log_path_and_tmux_name():
    out = subprocess.run(
        ["bash", "-c",
         f'source "{RD}/lib.sh"; '
         f'remote_log_path /logs 20260717-1 solve; echo; '
         f'remote_tmux_name "20260717/1"'],
        capture_output=True, text=True).stdout.split("\n")
    assert out[0] == "/logs/solve-20260717-1.log"
    assert out[1] == "diffsim-20260717-1"   # '/' sanitized to '-'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_lib_log_path_and_tmux_name -v`
Expected: FAIL — `lib.sh` not found.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/lib.sh`:

```bash
#!/usr/bin/env bash
# scripts/remote/lib.sh — pure helpers, no ssh side effects. Unit-tested.

rlog() { printf '\033[1;34m[remote]\033[0m %s\n' "$*" >&2; }
rerr() { printf '\033[1;31m[remote:err]\033[0m %s\n' "$*" >&2; }

# remote_log_path <logdir> <run_id> [tag]
remote_log_path() {
  local logdir="$1" run_id="$2" tag="${3:-run}"
  printf '%s/%s-%s.log' "$logdir" "$tag" "$run_id"
}

# remote_tmux_name <run_id>  (sanitize to [A-Za-z0-9_-])
remote_tmux_name() {
  printf 'diffsim-%s' "$(printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '-')"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_lib_log_path_and_tmux_name -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/lib.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): pure helpers lib.sh (log path, tmux name, loggers)"
```

---

### Task 3: Preflight doctor (`remote-doctor.sh`)

**Files:**
- Create: `scripts/remote/remote-doctor.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- Consumes: `config.sh`, `lib.sh`.
- Produces: exit 0 iff all gpubox checks pass; `--fix` adds the `gpubox` git remote on the Mac if missing; Nova check is informational (never fails the doctor).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
import subprocess as _sp   # for _sp.run(["ssh", ...]) in integration tests
def _gpubox_up():
    return _sp.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    "gpubox", "true"]).returncode == 0

# NOTE: _sh() is defined once in Task 1's test additions — do not redefine here.

import pytest
needs_box = pytest.mark.skipif(not _gpubox_up(), reason="gpubox unreachable")

@needs_box
def test_doctor_green():
    r = _sh("remote-doctor.sh", "--fix")
    assert r.returncode == 0, f"doctor failed:\n{r.stdout}\n{r.stderr}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_doctor_green -v`
Expected: FAIL (script missing) — or SKIP if gpubox is down. If SKIPPED, bring the box back before proceeding (this task's deliverable is not verifiable while the box is unreachable).

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/remote-doctor.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

if [ "${1:-}" = "--fix" ]; then
  git -C "$REPO_ROOT" remote get-url "$GPUBOX_GIT_REMOTE" >/dev/null 2>&1 || {
    rlog "adding git remote '$GPUBOX_GIT_REMOTE' -> $GPUBOX_GIT_URL"
    git -C "$REPO_ROOT" remote add "$GPUBOX_GIT_REMOTE" "$GPUBOX_GIT_URL"
  }
fi

fail=0
check() { if eval "$2" >/dev/null 2>&1; then rlog "OK   $1"; else rerr "FAIL $1"; fail=1; fi; }

rlog "gpubox preflight ($GPUBOX_HOST)"
check "ssh key reachable"          "ssh -o BatchMode=yes -o ConnectTimeout=8 $GPUBOX_HOST true"
check "claude on box"              "ssh $GPUBOX_HOST 'bash -lc \"command -v claude\"'"
check "venv python present"        "ssh $GPUBOX_HOST test -x $GPUBOX_VENV_PY"
check "tmux present"               "ssh $GPUBOX_HOST 'command -v tmux'"
check "rsync present on box"       "ssh $GPUBOX_HOST 'command -v rsync'"
check "repo present"               "ssh $GPUBOX_HOST test -d $GPUBOX_REPO_ABS"
check "mac has gpubox git remote"  "git -C $REPO_ROOT remote get-url $GPUBOX_GIT_REMOTE"

if ssh -o BatchMode=yes -o ConnectTimeout=6 "$NOVA_HOST" true 2>/dev/null; then
  rlog "OK   nova master socket alive"
else
  rlog "INFO nova needs interactive Duo auth (run: ssh $NOVA_HOST true)"
fi

[ "$fail" -eq 0 ] && rlog "doctor: all gpubox checks passed" || rerr "doctor: failures above"
exit "$fail"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_doctor_green -v`
Expected: PASS (gpubox reachable). Also run `bash scripts/remote/remote-doctor.sh --fix` and eyeball the OK lines.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/remote-doctor.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): remote-doctor.sh preflight + gpubox git remote setup"
```

---

### Task 4: Sync Mac→box, guarded (`gpubox-sync.sh`)

**Files:**
- Create: `scripts/remote/gpubox-sync.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- Consumes: `config.sh`, `lib.sh`.
- Produces: rsync of `$REPO_ROOT/` → `$GPUBOX_HOST:$GPUBOX_REPO_ABS/` with `RSYNC_EXCLUDES` and `--delete`. Exit 3 if the run-lock is present; exit 4 if the box has uncommitted changes (unless `FORCE=1`); `--dry-run` supported.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
@needs_box
def test_sync_refuses_under_lock():
    lock = _source_var("GPUBOX_LOCK")
    _sp.run(["ssh", "gpubox", f"touch '{lock}'"], check=True)
    try:
        r = _sh("gpubox-sync.sh", "--dry-run")
        assert r.returncode == 3, f"expected refusal (3), got {r.returncode}\n{r.stderr}"
        assert "run-lock" in r.stderr
    finally:
        _sp.run(["ssh", "gpubox", f"rm -f '{lock}'"], check=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_sync_refuses_under_lock -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/gpubox-sync.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

# 1) Refuse if a solve is in flight (Warp lazy-compile hazard).
if ssh -o BatchMode=yes "$GPUBOX_HOST" test -e "$GPUBOX_LOCK"; then
  rerr "run-lock present ($GPUBOX_LOCK) — a solve is in flight."
  rerr "refusing to sync: editing a .py mid-run breaks Warp. Poll it out first."
  exit 3
fi

# 2) Refuse if the box has uncommitted work we'd clobber (unless FORCE=1).
if [ -z "${FORCE:-}" ] && \
   ssh "$GPUBOX_HOST" "git -C '$GPUBOX_REPO_ABS' status --porcelain | grep -q ."; then
  rerr "box has uncommitted changes — fetch them first (gpubox-fetch.sh) or set FORCE=1."
  exit 4
fi

dry=""; [ "${1:-}" = "--dry-run" ] && dry="--dry-run"
rlog "rsync $REPO_ROOT/ -> $GPUBOX_HOST:$GPUBOX_REPO_ABS/ $dry"
rsync -az --delete $dry "${RSYNC_EXCLUDES[@]}" \
  "$REPO_ROOT"/ "$GPUBOX_HOST:$GPUBOX_REPO_ABS"/
rlog "sync complete"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_sync_refuses_under_lock -v`
Expected: PASS. Then sanity-check a real dry-run: `bash scripts/remote/gpubox-sync.sh --dry-run` (should list files, not error).

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/gpubox-sync.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): gpubox-sync.sh (rsync, lock + dirty-tree guards)"
```

---

### Task 5: Detached run + poll, lock lifecycle (`gpubox-run.sh`, `gpubox-poll.sh`)

**Files:**
- Create: `scripts/remote/gpubox-run.sh`, `scripts/remote/gpubox-poll.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- `gpubox-run.sh "<command>" [tag]` → launches `<command>` detached in `tmux` on the box, writes `$GPUBOX_LOCK`, tees output to a timestamped log, clears the lock on exit (trap). Prints one line: `<tmux-session>\t<log-path>`.
- `gpubox-poll.sh <log-path> [tail-lines]` → prints `status=RUNNING|DONE` and the last N lines of the log file; clears a stale lock if nothing is running.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
import time
@needs_box
def test_run_poll_lock_lifecycle():
    r = _sh("gpubox-run.sh", "echo hello-from-box; sleep 3", "smoke")
    assert r.returncode == 0, r.stderr
    sess, log = r.stdout.strip().split("\t")
    assert sess.startswith("diffsim-")
    lock = _source_var("GPUBOX_LOCK")
    # lock exists while running
    assert _sp.run(["ssh", "gpubox", f"test -e '{lock}'"]).returncode == 0
    # poll until DONE (cap ~30s)
    done = False
    for _ in range(15):
        p = _sh("gpubox-poll.sh", log)
        if "status=DONE" in p.stdout + p.stderr:
            done = True; break
        time.sleep(2)
    assert done, "run never reported DONE"
    # log captured output, lock cleared
    p = _sh("gpubox-poll.sh", log)
    assert "hello-from-box" in p.stdout
    assert _sp.run(["ssh", "gpubox", f"test -e '{lock}'"]).returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_run_poll_lock_lifecycle -v`
Expected: FAIL — scripts missing.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/gpubox-run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

cmd="${1:?usage: gpubox-run.sh \"<command>\" [tag]}"
tag="${2:-run}"
run_id="$(date +%Y%m%d-%H%M%S)-$$"
log="$(remote_log_path "$GPUBOX_LOGDIR" "$run_id" "$tag")"
sess="$(remote_tmux_name "$run_id")"

# Launch detached: write lock, cd repo, trap-clear lock on exit, tee to log.
ssh "$GPUBOX_HOST" bash -s <<EOF
set -e
mkdir -p "$GPUBOX_LOGDIR"
: > "$GPUBOX_LOCK"
tmux new-session -d -s "$sess" \
  "cd '$GPUBOX_REPO_ABS'; trap 'rm -f \"$GPUBOX_LOCK\"' EXIT; { $cmd ; } 2>&1 | tee '$log'"
EOF
rlog "launched tmux=$sess"
rlog "log=$log"
printf '%s\t%s\n' "$sess" "$log"
```

Create `scripts/remote/gpubox-poll.sh`:

```bash
#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

log="${1:?usage: gpubox-poll.sh <remote-log-path> [tail-lines]}"
n="${2:-40}"

if ssh "$GPUBOX_HOST" 'tmux ls 2>/dev/null | grep -q "^diffsim-"'; then
  rlog "status=RUNNING"
else
  rlog "status=DONE"
  ssh "$GPUBOX_HOST" "rm -f '$GPUBOX_LOCK'" || true   # defensive: clear stale lock
fi
# Read the log FILE (never pipe a producer through tail).
ssh "$GPUBOX_HOST" "tail -n $n '$log' 2>/dev/null || echo '(log not found yet)'"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_run_poll_lock_lifecycle -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/gpubox-run.sh scripts/remote/gpubox-poll.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): gpubox-run.sh + gpubox-poll.sh (detached tmux, lock lifecycle, tee logs)"
```

---

### Task 6: Headless dispatch (`gpubox-dispatch.sh`)

**Files:**
- Create: `scripts/remote/gpubox-dispatch.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- `gpubox-dispatch.sh "<task>"` → runs `claude -p "<task>" $BOX_CLAUDE_FLAGS [--model $BOX_MODEL]` in the box repo dir; box-Claude's final text goes to stdout.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
@needs_box
def test_dispatch_roundtrip():
    r = _sh("gpubox-dispatch.sh", "Reply with exactly the token PONG and nothing else.")
    assert "PONG" in r.stdout.upper(), f"stdout={r.stdout!r} stderr={r.stderr!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_dispatch_roundtrip -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/gpubox-dispatch.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

task="${1:?usage: gpubox-dispatch.sh \"<task for box Claude>\"}"
model_flag=""; [ -n "$BOX_MODEL" ] && model_flag="--model $BOX_MODEL"

if ! ssh -o BatchMode=yes -o ConnectTimeout=8 "$GPUBOX_HOST" true 2>/dev/null; then
  rerr "gpubox unreachable — cannot dispatch."
  exit 2
fi

rlog "dispatching headless box-Claude…"
ssh "$GPUBOX_HOST" bash -lc \
  "cd '$GPUBOX_REPO_ABS' && claude -p \"$task\" $BOX_CLAUDE_FLAGS $model_flag"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_dispatch_roundtrip -v`
Expected: PASS. (If the box-Claude needs first-run trust for the repo dir, `--dangerously-skip-permissions` bypasses it; if it still prompts, add `cd` dir to trusted config once — note in runbook.)

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/gpubox-dispatch.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): gpubox-dispatch.sh (headless box-Claude round-trip)"
```

---

### Task 7: Retrieve box commits (`gpubox-fetch.sh`)

**Files:**
- Create: `scripts/remote/gpubox-fetch.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- `gpubox-fetch.sh` → ensures the `gpubox` ssh git remote exists, `git fetch`es it, and prints the box/master commits + diffstat not yet in the Mac's HEAD (for supervisor review before pushing to origin). Exit 0 on a clean no-op.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
@needs_box
def test_fetch_noop_clean():
    r = _sh("gpubox-fetch.sh")
    assert r.returncode == 0, f"stderr={r.stderr}"
    assert _sp.run(["git", "-C", str(REPO), "remote", "get-url", "gpubox"]).returncode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_fetch_noop_clean -v`
Expected: FAIL — script missing.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/gpubox-fetch.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

git -C "$REPO_ROOT" remote get-url "$GPUBOX_GIT_REMOTE" >/dev/null 2>&1 || {
  rlog "adding git remote '$GPUBOX_GIT_REMOTE' -> $GPUBOX_GIT_URL"
  git -C "$REPO_ROOT" remote add "$GPUBOX_GIT_REMOTE" "$GPUBOX_GIT_URL"
}
rlog "fetching from box…"
git -C "$REPO_ROOT" fetch "$GPUBOX_GIT_REMOTE"
rlog "commits on $GPUBOX_GIT_REMOTE/master not in HEAD:"
git -C "$REPO_ROOT" log --oneline "HEAD..$GPUBOX_GIT_REMOTE/master" || true
rlog "diffstat:"
git -C "$REPO_ROOT" diff --stat "HEAD..$GPUBOX_GIT_REMOTE/master" || true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_fetch_noop_clean -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/gpubox-fetch.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): gpubox-fetch.sh (retrieve box commits over ssh for review)"
```

---

### Task 8: Nova stubs (`nova-sync-submit.sh`, `nova-poll.sh`)

**Files:**
- Create: `scripts/remote/nova-sync-submit.sh`, `scripts/remote/nova-poll.sh`
- Test: `tests/test_remote_toolkit.py`

**Interfaces:**
- Both print their intended flow to stderr and exit 64 (`not yet wired`). No ssh, no side effects — safe to run anywhere.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_remote_toolkit.py`:

```python
def test_nova_stubs_are_guarded():
    for s in ("nova-sync-submit.sh", "nova-poll.sh"):
        r = _sh(s)
        assert r.returncode == 64, f"{s} rc={r.returncode}"
        assert "not yet wired" in r.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_remote_toolkit.py::test_nova_stubs_are_guarded -v`
Expected: FAIL — scripts missing.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/remote/nova-sync-submit.sh`:

```bash
#!/usr/bin/env bash
# STUB — wired on next real Nova kit submission (see design spec §Decisions).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cat >&2 <<'MSG'
[nova] not yet wired.
Intended flow (see docs/dev/2026-07-15-mac-brainstorm-handoff.md §4):
  1. Mac: ensure origin/master current (git push).
  2. ssh nova (reuse Duo ControlMaster): git fetch && git reset --hard origin/master
  3. fill FIXME_PARTITION from: sinfo -o "%P %G %N"
  4. sbatch cluster/wodo_campaign/<kit>.sbatch   -> return job id
MSG
exit 64
```

Create `scripts/remote/nova-poll.sh`:

```bash
#!/usr/bin/env bash
# STUB — wired on next real Nova kit submission.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cat >&2 <<'MSG'
[nova] not yet wired.
Intended: squeue -j <jobid> / sacct -j <jobid> + tail the slurm-<jobid>.out log
over the reused Duo ControlMaster socket.
MSG
exit 64
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_remote_toolkit.py::test_nova_stubs_are_guarded -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/remote/nova-sync-submit.sh scripts/remote/nova-poll.sh tests/test_remote_toolkit.py
git commit -m "feat(remote): nova-*.sh stubs (guarded, not yet wired)"
```

---

### Task 9: Runbook + full-suite verification

**Files:**
- Create: `docs/dev/remote-workflow.md`
- Modify: (make all scripts executable)
- Test: full `tests/test_remote_toolkit.py`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: the operator runbook and a green test suite.

- [ ] **Step 1: Make scripts executable**

Run: `chmod +x scripts/remote/*.sh`

- [ ] **Step 2: Write the runbook**

Create `docs/dev/remote-workflow.md`:

```markdown
# Remote-deploy workflow (office Mac → gpubox / Nova)

The Mac thinks and orchestrates; a headless Claude on gpubox executes GPU work.
The Mac is the sole GitHub gatekeeper. Spec: `docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md`.

## One-time
- `bash scripts/remote/remote-doctor.sh --fix` — verifies the box and adds the `gpubox` git remote.
- Nova: `ssh nova true` once to establish the Duo `ControlMaster` (reused 7 days).

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
```

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/test_remote_toolkit.py -v`
Expected: unit tests PASS; integration tests PASS if gpubox is up, else SKIP with "gpubox unreachable". No FAIL.

- [ ] **Step 4: Commit**

```bash
git add scripts/remote docs/dev/remote-workflow.md
git commit -m "docs(remote): operator runbook + executable bits; green toolkit suite"
```

- [ ] **Step 5: Land the design + handoff docs**

```bash
git add docs/dev/specs/2026-07-17-remote-deploy-workflow-design.md \
        docs/dev/plans/2026-07-17-remote-deploy-workflow-plan.md \
        docs/dev/2026-07-15-mac-brainstorm-handoff.md
git commit -m "docs(remote): design spec, plan, and brainstorm handoff"
```

---

## Self-Review

**Spec coverage:**
- Architecture (two brains, headless dispatch, sync model B) → Tasks 4/6/7 + runbook. ✓
- `config.sh`, `lib.sh`, `remote-doctor.sh`, `gpubox-sync.sh`, `gpubox-dispatch.sh`, `gpubox-run.sh`, `gpubox-poll.sh`, `gpubox-fetch.sh`, Nova stubs → Tasks 1–8. ✓
- Hazards (run-lock, no-pipe-tail, venv guard) → Tasks 4/5 + Global Constraints. ✓
- Headless permission trade-off → Task 6 + Global Constraints. ✓
- Error handling (reachability, Nova re-auth, lock) → Tasks 3/4/6. ✓
- Testing (doctor green, dispatch round-trip, run/poll/lock, sync-refuses, fetch no-op) → Tasks 3–7. ✓
- Runbook `docs/dev/remote-workflow.md` → Task 9. ✓
- GPU-pathology probe: covered informally (doctor reachability + runbook note). Deeper `gpu_burn.py` probe deferred to campaign time — not a blocker for the toolkit.

**Placeholder scan:** none — every script and test is shown in full.

**Type/name consistency:** `GPUBOX_*` vars, `remote_log_path`/`remote_tmux_name`, `diffsim-` tmux prefix, exit codes (3 lock / 4 dirty / 64 stub) are consistent across tasks and tests.
