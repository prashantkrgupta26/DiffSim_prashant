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
# Pass the entire remote command as a single string so the login shell
# sources the full profile (which adds ~/.local/bin/claude to PATH).
# shellcheck disable=SC2029  # deliberate: $task/$BOX_CLAUDE_FLAGS/$model_flag expand on the Mac side
ssh "$GPUBOX_HOST" "bash -lc \"cd '$GPUBOX_REPO_ABS' && claude -p \\\"$task\\\" $BOX_CLAUDE_FLAGS $model_flag\""
