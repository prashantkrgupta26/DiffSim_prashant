#!/usr/bin/env bash
# Register THIS GPU box as a self-hosted GitHub Actions runner labelled `gpu`,
# and install it as a persistent service — so the gated `gpu-smoke` job in
# .github/workflows/ci.yml can run here.
#
# WHY A SCRIPT: the only piece I (Claude) cannot produce is the runner
# REGISTRATION TOKEN — GitHub mints it per-repo in the UI and it is short-lived
# (~1 h). Everything else (download the right runner build, label it `gpu`,
# unattended config, systemd service install) is mechanical, so this script does
# it in one shot. You paste the token; the script does the rest.
#
# ── STEP 1: get a registration token ──────────────────────────────────────────
#   GitHub → repo Settings → Actions → Runners → "New self-hosted runner" →
#   Linux/x64. Copy the token from the `./config.sh --token <TOKEN>` line shown
#   there (starts with A..., NOT a PAT — do not use your git PAT here).
#
# ── STEP 2: run this script on the GPU box, from the repo root ────────────────
#   bash .github/scripts/setup-gpu-runner.sh --token <REGISTRATION_TOKEN>
#
# ── STEP 3: enable the job ────────────────────────────────────────────────────
#   GitHub → Settings → Secrets and variables → Actions → Variables →
#   New repository variable:  ENABLE_GPU_CI = true
#   (Until this is set, gpu-smoke is skipped, not failed.)
#
# Optional flags:
#   --url  <repo-url>   default: https://github.com/BaskarGS/diffsim
#   --name <runner>     default: <hostname>-gpu
#   --labels <csv>      default: gpu            (must include `gpu` to match ci.yml)
#   --dir  <path>       default: ~/actions-runner
#   --version <x.y.z>   default: latest release (queried from the GitHub API)
#   --no-service        configure only; run interactively with ./run.sh yourself
#
# Re-runnable: if the target dir already holds a configured runner it is removed
# (svc uninstall + config remove) with the same token before reconfiguring.
set -euo pipefail

URL="https://github.com/BaskarGS/diffsim"
NAME="$(hostname)-gpu"
LABELS="gpu"
DIR="$HOME/actions-runner"
VERSION=""
TOKEN=""
INSTALL_SERVICE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token) TOKEN="$2"; shift 2;;
    --url) URL="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --labels) LABELS="$2"; shift 2;;
    --dir) DIR="$2"; shift 2;;
    --version) VERSION="$2"; shift 2;;
    --no-service) INSTALL_SERVICE=0; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown flag: $1" >&2; exit 2;;
  esac
done

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: --token <REGISTRATION_TOKEN> is required (see STEP 1 in this file)." >&2
  exit 2
fi
if [[ ",$LABELS," != *",gpu,"* ]]; then
  echo "ERROR: --labels must include 'gpu' (ci.yml uses runs-on: [self-hosted, gpu])." >&2
  exit 2
fi

# Resolve the latest runner version if not pinned (strip the leading 'v').
if [[ -z "$VERSION" ]]; then
  echo "Resolving latest actions/runner release ..."
  VERSION="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest \
    | grep -m1 '"tag_name"' | sed -E 's/.*"v?([0-9.]+)".*/\1/')"
  [[ -n "$VERSION" ]] || { echo "Could not resolve latest version; pass --version x.y.z" >&2; exit 1; }
fi
echo "Runner version: $VERSION   labels: $LABELS   name: $NAME"
echo "Repo: $URL   dir: $DIR"

TARBALL="actions-runner-linux-x64-${VERSION}.tar.gz"
DL="https://github.com/actions/runner/releases/download/v${VERSION}/${TARBALL}"

# If a runner is already configured here, tear it down cleanly first.
if [[ -f "$DIR/.runner" ]]; then
  echo "Existing runner found in $DIR — removing before reconfigure ..."
  ( cd "$DIR"
    if [[ -f svc.sh ]]; then sudo ./svc.sh stop || true; sudo ./svc.sh uninstall || true; fi
    ./config.sh remove --token "$TOKEN" || true )
fi

mkdir -p "$DIR"; cd "$DIR"
if [[ ! -f "./config.sh" ]]; then
  echo "Downloading $DL ..."
  curl -fsSL -o "$TARBALL" "$DL"
  tar xzf "$TARBALL"
  rm -f "$TARBALL"
fi

echo "Configuring runner (unattended) ..."
./config.sh --unattended --replace \
  --url "$URL" --token "$TOKEN" \
  --name "$NAME" --labels "$LABELS" \
  --work "_work"

if [[ "$INSTALL_SERVICE" == "1" ]]; then
  echo "Installing as a systemd service (persists across reboots) ..."
  sudo ./svc.sh install
  sudo ./svc.sh start
  sudo ./svc.sh status || true
  echo
  echo "Runner is live as a service. Manage it with:"
  echo "  sudo $DIR/svc.sh {status|stop|start|uninstall}"
else
  echo
  echo "Configured (no service). Start it interactively with:  cd $DIR && ./run.sh"
fi

cat <<EOF

NEXT: enable the job in GitHub —
  Settings → Secrets and variables → Actions → Variables → New repository variable
    ENABLE_GPU_CI = true

Then push (or re-run) any workflow: the 'gpu-smoke' job will pick up on this box.
Verify the runner shows Idle/Active under Settings → Actions → Runners.
EOF
