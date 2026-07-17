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
