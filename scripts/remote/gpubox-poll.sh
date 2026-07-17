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
