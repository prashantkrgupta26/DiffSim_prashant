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
# Reference the box's Claude by absolute path ($CLAUDE_BIN) so we neither depend
# on a login shell sourcing PATH nor need a bash -lc wrapper — keeping the remote
# quoting to a single level.
# NOTE: $task is single-double-quoted on the remote side; avoid literal double
# quotes in the task text (a fixed limitation of shipping a prompt over ssh).
# shellcheck disable=SC2029  # deliberate: $task/$BOX_CLAUDE_FLAGS/$model_flag expand on the Mac side
ssh "$GPUBOX_HOST" "cd '$GPUBOX_REPO_ABS' && $CLAUDE_BIN -p \"$task\" $BOX_CLAUDE_FLAGS $model_flag"
