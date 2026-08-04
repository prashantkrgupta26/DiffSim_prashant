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
# NOTE: the tmux command below relies on $GPUBOX_REPO_ABS / $GPUBOX_LOCK / $log
# containing no spaces or shell-special chars (they are fixed config paths).
# The multi-layer quoting (local heredoc -> ssh -> tmux -> shell) is not
# space-safe; keep these paths simple.
ssh "$GPUBOX_HOST" bash -s <<EOF
set -e
mkdir -p "$GPUBOX_LOGDIR"
if [ -e "$GPUBOX_LOCK" ]; then
  echo "run-lock present ($GPUBOX_LOCK) — a solve is already in flight; refusing to launch a second run." >&2
  exit 3
fi
: > "$GPUBOX_LOCK"
if ! tmux new-session -d -s "$sess" \
  "trap 'rm -f \"$GPUBOX_LOCK\"' EXIT; cd '$GPUBOX_REPO_ABS' && { env $GPUBOX_GPU_ENV $cmd ; } 2>&1 | tee '$log'"; then
  rm -f "$GPUBOX_LOCK"
  echo "tmux failed to launch — cleared lock." >&2
  exit 5
fi
EOF
rlog "launched tmux=$sess"
rlog "log=$log"
printf '%s\t%s\n' "$sess" "$log"
