#!/usr/bin/env bash
# scripts/remote/nova-sync-submit.sh — ship the current branch to Nova via
# `git bundle` (Nova has NO GitHub auth) and submit a SLURM GPU job, enforcing
# the <=4-concurrent-GPU-jobs hard cap.
#
# Usage:
#   nova-sync-submit.sh <sbatch-kit> [--card a100|gh200] [--time HH:MM:SS]
#                        [--mem MEM] [--dry-run]
#   nova-sync-submit.sh cluster/a100-smoke.sh --card a100 --time 00:30:00 --mem 32G
#   nova-sync-submit.sh cluster/gh200-smoke.sh --card gh200 --time 00:30:00 --mem 32G
#
# The <sbatch-kit> is a path (relative to the repo root) to either a .sbatch
# template or a plain script; we submit it with sbatch and inject the
# partition/account/qos/gres/cpus flags on the command line so one script serves
# many cards. Returns the numeric job ID on stdout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
source "$HERE/lib.sh"

KIT="${1:-}"
if [ -z "$KIT" ]; then
  rerr "usage: nova-sync-submit.sh <sbatch-kit> [--card a100|gh200] [--time ..] [--mem ..] [--dry-run]"
  exit 64
fi
shift

CARD="a100"
WALLTIME="00:30:00"
MEM="32G"
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --card) CARD="$2"; shift 2 ;;
    --time) WALLTIME="$2"; shift 2 ;;
    --mem)  MEM="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) rerr "unknown arg: $1"; exit 64 ;;
  esac
done

# --- card -> partition/gres --------------------------------------------------
case "$CARD" in
  a100)  PARTITION="nova";     GRES="gpu:a100:1" ;;
  gh200) PARTITION="nova-arm"; GRES="gpu:gh200:1" ;;
  h200)  PARTITION="nova";     GRES="gpu:h200:1" ;;
  *) rerr "unknown --card '$CARD' (want a100|gh200|h200)"; exit 64 ;;
esac

if [ ! -f "$REPO_ROOT/$KIT" ]; then
  rerr "kit not found in repo: $REPO_ROOT/$KIT"
  exit 64
fi

# --- 1. bundle the current branch and ship it over the Duo ssh channel --------
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
BUNDLE="/tmp/diffsim-${BRANCH//\//-}-${SHA}.bundle"
rlog "bundling $BRANCH ($SHA) -> $BUNDLE"
git -C "$REPO_ROOT" bundle create "$BUNDLE" "$BRANCH" >/dev/null

REMOTE_BUNDLE="$NOVA_BUNDLE_DIR/$(basename "$BUNDLE")"
rlog "shipping bundle to $NOVA_HOST:$REMOTE_BUNDLE"
ssh "$NOVA_HOST" "mkdir -p '$NOVA_BUNDLE_DIR'"
scp -q "$BUNDLE" "$NOVA_HOST:$REMOTE_BUNDLE"

# fetch+checkout into /work clone (NEVER reset-hard: Nova clone may carry local
# branches, e.g. chunked-csr). Fetch the branch from the bundle, then update the
# working tree to that ref.
rlog "updating /work clone from bundle (fetch + checkout, no reset)"
ssh "$NOVA_HOST" bash -s <<EOF
set -euo pipefail
cd "$NOVA_REPO_ABS"
git fetch "$REMOTE_BUNDLE" "$BRANCH":refs/remotes/mac-bundle/"$BRANCH"
git checkout -B "$BRANCH" refs/remotes/mac-bundle/"$BRANCH"
git --no-pager log --oneline -1
EOF

# --- 2. enforce the <=4 concurrent GPU-job hard cap ---------------------------
rlog "checking Nova GPU-job count (cap = $NOVA_MAX_GPU_JOBS)"
N=$(ssh "$NOVA_HOST" "squeue -u baskarg --noheader -o '%b' 2>/dev/null | grep -ci gpu" || echo 0)
N=${N//[^0-9]/}; N=${N:-0}
rlog "current Nova GPU jobs: $N"
if [ "$N" -ge "$NOVA_MAX_GPU_JOBS" ]; then
  rerr "REFUSING to submit: $N concurrent Nova GPU jobs >= cap $NOVA_MAX_GPU_JOBS."
  rerr "Wait for jobs to finish (squeue -u baskarg) before submitting."
  exit 75   # EX_TEMPFAIL
fi

# --- 3. submit ----------------------------------------------------------------
SUBMIT=(
  sbatch --parsable
  --account="$NOVA_ACCOUNT"
  --partition="$PARTITION"
  --qos="$NOVA_QOS"
  --cpus-per-task="$NOVA_CPUS_PER_TASK"
  --gres="$GRES"
  --time="$WALLTIME"
  --mem="$MEM"
  --job-name="diffsim-$CARD-smoke"
  --output="cluster/results/%x-%j.out"
  "$KIT"
)
rlog "submit: ${SUBMIT[*]} (in $NOVA_REPO_ABS)"
if [ "$DRY_RUN" -eq 1 ]; then
  rlog "DRY-RUN: not submitting"
  exit 0
fi

JOBID=$(ssh "$NOVA_HOST" "cd '$NOVA_REPO_ABS' && ${SUBMIT[*]}")
JOBID=${JOBID//[^0-9]/}
if [ -z "$JOBID" ]; then
  rerr "sbatch did not return a job id"
  exit 1
fi
rlog "submitted Nova job $JOBID ($CARD / $PARTITION / $GRES)"
printf '%s\n' "$JOBID"
