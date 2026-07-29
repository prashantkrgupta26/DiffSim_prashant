#!/usr/bin/env bash
# cluster/a3-probes/ship-bundle.sh — bundle a3-strengthen-probes branch and
# deliver it to the Nova /work clone WITHOUT submitting a job (hold 11777138
# is already running; srun legs drive the measurements).
# Usage: bash cluster/a3-probes/ship-bundle.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../scripts/remote/config.sh"

BRANCH="a3-strengthen-probes"
SHA="$(git -C "$REPO_ROOT" rev-parse --short "$BRANCH")"
BUNDLE="/tmp/diffsim-${BRANCH//\//-}-${SHA}.bundle"

echo "[a3-ship] bundling $BRANCH ($SHA) -> $BUNDLE"
git -C "$REPO_ROOT" bundle create "$BUNDLE" "$BRANCH"

REMOTE_BUNDLE="$NOVA_BUNDLE_DIR/$(basename "$BUNDLE")"
echo "[a3-ship] shipping bundle -> $NOVA_HOST:$REMOTE_BUNDLE"
ssh "$NOVA_HOST" "mkdir -p '$NOVA_BUNDLE_DIR'"
scp -q "$BUNDLE" "$NOVA_HOST:$REMOTE_BUNDLE"

echo "[a3-ship] updating /work clone"
ssh "$NOVA_HOST" bash -s <<EOF
set -euo pipefail
cd "$NOVA_REPO_ABS"
git fetch "$REMOTE_BUNDLE" "$BRANCH":refs/remotes/mac-bundle/"$BRANCH"
git checkout -B "$BRANCH" refs/remotes/mac-bundle/"$BRANCH"
git --no-pager log --oneline -1
echo "[a3-ship] /work clone ready"
EOF
echo "[a3-ship] done — run legs via srun --jobid=11777138 --overlap"
