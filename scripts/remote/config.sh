#!/usr/bin/env bash
# scripts/remote/config.sh — shared config for the remote-deploy toolkit.
# SOURCE this (do not execute). All remote scripts source it.

REMOTE_TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_ROOT="$(cd "$REMOTE_TOOLKIT_DIR/../.." && pwd)"

# --- gpubox ---
export GPUBOX_HOST="${GPUBOX_HOST:-gpubox}"
export GPUBOX_REPO="${GPUBOX_REPO:-Baskar/DiffSim}"                 # relative to remote $HOME
export GPUBOX_REPO_ABS="${GPUBOX_REPO_ABS:-/home/bglab/Baskar/DiffSim}"
export GPUBOX_VENV_PY="${GPUBOX_VENV_PY:-$GPUBOX_REPO_ABS/.venv/bin/python}"
export GPUBOX_LOCK="${GPUBOX_LOCK:-$GPUBOX_REPO_ABS/.remote-run.lock}"
export GPUBOX_LOGDIR="${GPUBOX_LOGDIR:-$GPUBOX_REPO_ABS/logs}"
export GPUBOX_GIT_REMOTE="${GPUBOX_GIT_REMOTE:-gpubox}"             # name of ssh git remote on the Mac
export GPUBOX_GIT_URL="${GPUBOX_GIT_URL:-$GPUBOX_HOST:$GPUBOX_REPO}"

# gpubox is WSL2: the native libcuda stub shadows the WSL GPU-passthrough
# driver. EVERY GPU process must see /usr/lib/wsl/lib first or Warp/torch
# report "no CUDA-capable device" while nvidia-smi works fine.
export GPUBOX_GPU_ENV="LD_LIBRARY_PATH=/usr/lib/wsl/lib"

# Absolute path to the box's Claude CLI — referenced directly so dispatch does
# not depend on a login shell sourcing PATH (and to keep remote quoting shallow).
export CLAUDE_BIN="${CLAUDE_BIN:-/home/bglab/.local/bin/claude}"
# Headless box-Claude flags (accepted security trade-off: single-user box).
export BOX_CLAUDE_FLAGS="${BOX_CLAUDE_FLAGS:---dangerously-skip-permissions}"
export BOX_MODEL="${BOX_MODEL:-}"   # empty = box default; set to override

# rsync excludes (Mac -> box): only source syncs; data/artifacts stay put.
# NOTE: bash cannot export arrays; scripts SOURCE this file so it stays in scope.
RSYNC_EXCLUDES=(
  --exclude '.git'
  --exclude '.claude/'
  --exclude '.superpowers/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude 'logs/'
  --exclude 'data/'
  --exclude 'papers/'
  --exclude 'CriticalEvaluations/'
  --exclude '*.npz'
  --exclude '*.pt'
)

# --- nova (Iowa State SLURM) ---
# Nova has NO GitHub auth: ship code via `git bundle` over the Duo ssh channel,
# then fetch+checkout into /work (2PB scratch). HOME is over-quota — NEVER run
# jobs from HOME; everything lives under NOVA_REPO_ABS on /work.
export NOVA_HOST="${NOVA_HOST:-nova}"
export NOVA_REPO_ABS="${NOVA_REPO_ABS:-/work/mech-ai/baskarg/DiffSim}"
export NOVA_ACCOUNT="${NOVA_ACCOUNT:-mech-ai}"
export NOVA_QOS="${NOVA_QOS:-normal}"
export NOVA_CPUS_PER_TASK="${NOVA_CPUS_PER_TASK:-6}"   # Nova caps <=6 cpu/GPU
# HARD CAP: <=4 concurrent Nova GPU jobs (account gres/gpu quota is 19, pooled).
export NOVA_MAX_GPU_JOBS="${NOVA_MAX_GPU_JOBS:-4}"
export NOVA_BUNDLE_DIR="${NOVA_BUNDLE_DIR:-/work/mech-ai/baskarg/bundles}"
