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

# Absolute path to the box's Claude CLI — referenced directly so dispatch does
# not depend on a login shell sourcing PATH (and to keep remote quoting shallow).
export CLAUDE_BIN="${CLAUDE_BIN:-/home/bglab/.local/bin/claude}"
# Headless box-Claude flags (accepted security trade-off: single-user box).
export BOX_CLAUDE_FLAGS="${BOX_CLAUDE_FLAGS:---dangerously-skip-permissions}"
export BOX_MODEL="${BOX_MODEL:-}"   # empty = box default; set to override

# rsync excludes (Mac -> box): only source syncs; data/artifacts stay put.
# NOTE: bash cannot export arrays; scripts SOURCE this file so it stays in scope.
RSYNC_EXCLUDES=(
  --exclude '.git/'
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

# --- nova (scripts stubbed for now) ---
export NOVA_HOST="${NOVA_HOST:-nova}"
export NOVA_REPO_ABS="${NOVA_REPO_ABS:-}"   # set when Nova scripts are wired
