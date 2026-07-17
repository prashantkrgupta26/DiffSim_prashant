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
