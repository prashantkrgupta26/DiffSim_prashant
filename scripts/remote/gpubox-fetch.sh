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
