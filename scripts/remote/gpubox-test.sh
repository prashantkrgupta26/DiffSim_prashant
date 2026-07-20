#!/usr/bin/env bash
# gpubox-test.sh — run the pytest suite on gpubox with xdist parallelism.
#
#   gpubox-test.sh                  # full suite, -n 12
#   gpubox-test.sh tests/foo.py     # any extra args go straight to pytest
#   JOBS=8 gpubox-test.sh           # override worker count
#
# Why: the Mac's CPU-only warp build grinds the full suite for 80+ minutes
# single-process and cannot see GPU-only failures at all (measured
# 2026-07-19: byte-identity assertions that hold on serial CPU scatter
# fail at the ULP level under GPU atomics). The box runs the same suite
# in ~2h -> minutes for the parallel bulk, WITH cuda coverage. This is
# the standard pre-merge gate.
#
# Flow: FORCE-sync the working tree out -> ensure venv deps -> launch the
# run-lock-guarded tmux run -> poll until the pytest summary appears ->
# print summary + FAILED lines -> exit 0 iff no failures.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

JOBS="${JOBS:-12}"   # 12 of 40 cores: BLAS-threaded direct solves oversubscribe past this
pytest_args=("$@")

rlog "sync (FORCE=1) ..."
FORCE=1 "$HERE/gpubox-sync.sh" >/dev/null

rlog "ensure venv deps (pytest-xdist) ..."
ssh "$GPUBOX_HOST" "cd '$GPUBOX_REPO_ABS' && .venv/bin/python -c 'import xdist' 2>/dev/null \
  || .venv/bin/pip install -q pytest-xdist"

# WSL cuda needs /usr/lib/wsl/lib; AMGX (built from source, no wheel) needs
# its lib dir when present — probe for it rather than hardcoding.
remote_env='LD_LIBRARY_PATH=/usr/lib/wsl/lib'
amgx_lib="$(ssh "$GPUBOX_HOST" 'ls "$HOME"/AMGX/build/libamgxsh.so /usr/local/lib/libamgxsh.so 2>/dev/null | head -1' || true)"
[ -n "$amgx_lib" ] && remote_env="LD_LIBRARY_PATH=/usr/lib/wsl/lib:${amgx_lib%/*}"

cmd="$remote_env .venv/bin/pytest -q -n $JOBS -p no:cacheprovider ${pytest_args[*]:-}"
rlog "launch: $cmd"
out="$("$HERE/gpubox-run.sh" "$cmd" pytest-gate)"
log="$(printf '%s\n' "$out" | tail -1 | cut -f2)"
rlog "log=$log"

# Poll for the final summary line ("N passed ... in Ns"). The tmux session
# dying without one means the run crashed -> report the tail and fail.
for _ in $(seq 1 240); do
  summary="$(ssh "$GPUBOX_HOST" "tail -3 '$log' 2>/dev/null" | grep -E '(passed|failed|error).* in [0-9.]+s' | tail -1 || true)"
  [ -n "$summary" ] && break
  if ! ssh "$GPUBOX_HOST" 'tmux ls 2>/dev/null | grep -q "^diffsim-"'; then
    sleep 2   # let tee flush, then re-check once
    summary="$(ssh "$GPUBOX_HOST" "tail -3 '$log' 2>/dev/null" | grep -E '(passed|failed|error).* in [0-9.]+s' | tail -1 || true)"
    if [ -z "$summary" ]; then
      rerr "run ended with no pytest summary — crashed? log tail:"
      ssh "$GPUBOX_HOST" "tail -20 '$log' 2>/dev/null" >&2
      exit 4
    fi
    break
  fi
  sleep 30
done
if [ -z "${summary:-}" ]; then
  rerr "timed out (2h) waiting for the suite — session left running; poll with gpubox-poll.sh '$log'"
  exit 5
fi

echo "$summary"
ssh "$GPUBOX_HOST" "grep '^FAILED' '$log'" || true
case "$summary" in
  *" failed"*|*error*) exit 1 ;;
  *) exit 0 ;;
esac
