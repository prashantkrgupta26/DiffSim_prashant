#!/usr/bin/env bash
# cluster/a3-probes/run-legs.sh — A3 strengthening probe measurement legs.
# Run ONE leg per invocation inside hold job 11777138:
#   srun --jobid=11777138 --overlap bash -c 'bash /work/mech-ai/baskarg/DiffSim/cluster/a3-probes/run-legs.sh <LEG>'
#
# Available legs: p1-restart30  p2-restart120  p3-warmstart  p4-fused  p5-combo
#
# All legs run 3d-L7 uniform, device assembly, fgmres_bdiag (except p4-fused),
# 5 steps — comparing vs baseline: 1300.2 iters/step, 54.8 s/step.
# Logs: cluster/results/a3-<leg>.log (+ -smi.log for GPU trace).
# Each leg wrapped in timeout 1200 (20 min).

set -uo pipefail
LEG="${1:-}"
if [ -z "$LEG" ]; then
  echo "Usage: $0 <leg>  (p1-restart30 | p2-restart120 | p3-warmstart | p4-fused | p5-combo)"
  exit 1
fi

# ---- Standard GH200 environment ----
WORK=/work/mech-ai/baskarg
RPM_DIR="$WORK/python311-arm"
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH="$WORK/.warp-cache-arm"
mkdir -p "$WARP_CACHE_PATH"
cd "$WORK/DiffSim"
. .venv-nova-arm/bin/activate

LOGDIR="$WORK/DiffSim/cluster/results"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/a3-${LEG}.log"
SMILOG="$LOGDIR/a3-${LEG}-smi.log"

echo "[a3-probes] leg=$LEG  branch=$(git log --oneline -1)  $(date)" | tee "$LOG"
echo "[a3-probes] node=$(hostname)  gpu=$(nvidia-smi -L 2>/dev/null | head -1)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null | tee -a "$LOG"

# GPU SMI side-car
( while true; do
    echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)";
    sleep 5; done ) > "$SMILOG" 2>&1 &
SMI_PID=$!
trap "kill $SMI_PID 2>/dev/null; true" EXIT

# ---- Base env for all legs ----
export SADDLE_POINTS=3d-L7
export SADDLE_SOLVERS=fgmres_bdiag
export SADDLE_ASSEMBLY=device
export SADDLE_NSTEPS=5
export SADDLE_DEVICE=cuda:0

run_leg() {
  echo "[a3-probes] START leg=$LEG  env: SADDLE_RESTART=${SADDLE_RESTART:-<default>} SADDLE_X0=${SADDLE_X0:-<default>} SADDLE_SOLVERS=$SADDLE_SOLVERS" | tee -a "$LOG"
  RSS_BEFORE=$(awk '/VmRSS/{print $2}' /proc/$$/status 2>/dev/null || echo 0)

  timeout 1200 python tests/gpu_saddle_ladder.py 2>&1 | tee -a "$LOG"
  LEG_RC=${PIPESTATUS[0]}

  RSS_AFTER=$(awk '/VmRSS/{print $2}' /proc/$$/status 2>/dev/null || echo 0)
  SMI_PEAK=$(awk -F"[ ,]" '{if($3+0>m) m=$3+0}END{print m" MiB"}' "$SMILOG" 2>/dev/null || echo "?")

  echo "LEG_RC=$LEG_RC" | tee -a "$LOG"
  echo "RSS_AFTER=${RSS_AFTER} kB" | tee -a "$LOG"
  echo "SMI_PEAK=$SMI_PEAK" | tee -a "$LOG"
  echo "[a3-probes] END leg=$LEG  $(date)" | tee -a "$LOG"
}

case "$LEG" in
  p1-restart30)
    export SADDLE_RESTART=30
    run_leg ;;

  p2-restart120)
    export SADDLE_RESTART=120
    run_leg ;;

  p3-warmstart)
    export SADDLE_X0=extrap
    run_leg ;;

  p4-fused)
    export SADDLE_SOLVERS=fused
    # raw fused: no preconditioner hook (see §11 note C); divergence is a valid result
    run_leg ;;

  p5-combo)
    # best restart from p1/p2 + warm start — determined after p1+p2 land;
    # default to restart=30 as the smaller subspace (adjust once p1/p2 are measured)
    export SADDLE_RESTART=30
    export SADDLE_X0=extrap
    run_leg ;;

  *)
    echo "Unknown leg '$LEG'; valid: p1-restart30 p2-restart120 p3-warmstart p4-fused p5-combo"
    exit 1 ;;
esac
