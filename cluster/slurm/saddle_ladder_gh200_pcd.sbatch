#!/bin/bash
#SBATCH --job-name=saddle-l7r9-pcd
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
# T4 gate leg (pcd-amgx): 9.24M-DOF 3d-L7r9 with device assembly + AMGX inner.
# Device assembly (SADDLE_ASSEMBLY=device) removes the ~243-245 GB host CSR
# transient.  SADDLE_PCD_INNER=amgx routes the velocity F-block through AMGX
# BiCGStab+classical-AMG (hierarchy reused within the same sparsity; per-apply
# is solve-only after first step setup).  NSTEPS=5 is feasible with device
# assembly (setup time expected minutes, not 1.5 h).
#
# PREREQUISITE: pyamgx must be installed in .venv-nova-arm before submission.
# As of 2026-07-28 pyamgx is ABSENT on nova — see T4 report for install path.
# The controller must verify pyamgx before submitting this kit.
# Without pyamgx the job will fail at the first fgmres_pcd step.
#
# Reference bdiag leg: run saddle_ladder_gh200.sbatch first to confirm device
# assembly works (setup time + RSS) before submitting the pcd-amgx leg.
# Partition/account/qos/gres/cpus/mem/time are injected by nova-sync-submit.sh.
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64:${AMGX_LIB_DIR:-/work/mech-ai/baskarg/AMGX/build}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
nvidia-smi -L

SMI_LOG="/work/mech-ai/baskarg/DiffSim/cluster/results/${SLURM_JOB_ID}-smi.log"
( while true; do
    echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)";
    sleep 10; done ) > "$SMI_LOG" 2>&1 &
SMI_PID=$!

# One solver per PROCESS (cross-leg-accumulation lesson: never share a process
# between ladder legs).  device assembly + AMGX inner: the headline gate leg
# for the pcd-amgx verdict (outer iters, s/step split, SMI peak).
RC_PCD=1
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0 \
  SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device SADDLE_PCD_INNER=amgx \
  PYTHONPATH=src:tests python tests/gpu_saddle_ladder.py
RC_PCD=$?
echo "LEG_RC pcd $RC_PCD"

kill $SMI_PID 2>/dev/null
echo "SMI_PEAK $(awk '{if(NF>=3 && $3+0>m)m=$3+0}END{print m" MiB"}' "$SMI_LOG" 2>/dev/null)"
echo "SADDLE_L7R9_PCD_RC pcd=$RC_PCD"
