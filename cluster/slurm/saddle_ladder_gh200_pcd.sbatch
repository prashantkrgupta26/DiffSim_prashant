#!/bin/bash
#SBATCH --job-name=saddle-l7r9-pcd
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
# T4 Phase 1.5 gate leg (pcd-jacobi): 9.24M-DOF 3d-L7r9 with device assembly +
# Jacobi-CG F-inner (SADDLE_PCD_INNER unset = jacobi default).
# Device assembly (SADDLE_ASSEMBLY=device) removes the ~243-245 GB host CSR
# transient.  NSTEPS=5 is feasible with device assembly (setup time expected
# minutes, not 1.5 h).
#
# pyamgx is NOT required for this leg — Jacobi-CG is the F-inner solver and
# has no external dependencies.
#
# NOTE (future amgx-config retry only): if a future probe retries AMGX with a
# different config (aggregation AMG / stronger smoother: GS/ILU-type,
# Krylov-smoothed), the controller must first install pyamgx in .venv-nova-arm
# on nova before submitting with SADDLE_PCD_INNER=amgx.  The Phase 1.5
# measurement refuted maxiter-tuning as the fix — see §8.1 of the campaign doc.
#
# Expected wall based on gpubox Phase 1.5 pcd-jacobi+device measurement:
# see docs/dev/2026-07-28-track-a2-campaign.md §4.2 table for s/step reading.
# GH200 (sm_90a) will likely be faster than gpubox RTX 6000 Ada (sm_89).
#
# Reference bdiag leg: run saddle_ladder_gh200.sbatch first to confirm device
# assembly works (setup time + RSS) before submitting this pcd leg.
# Partition/account/qos/gres/cpus/mem/time are injected by nova-sync-submit.sh.
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
nvidia-smi -L

SMI_LOG="/work/mech-ai/baskarg/DiffSim/cluster/results/${SLURM_JOB_ID}-smi.log"
( while true; do
    echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)";
    sleep 10; done ) > "$SMI_LOG" 2>&1 &
SMI_PID=$!

# One solver per PROCESS (cross-leg-accumulation lesson: never share a process
# between ladder legs).  device assembly + jacobi inner: the Phase 1.5 gate leg
# for the pcd-jacobi verdict (outer iters, s/step split, SMI peak).
RC_PCD=1
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0 \
  SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device \
  PYTHONPATH=src:tests python tests/gpu_saddle_ladder.py
RC_PCD=$?
echo "LEG_RC pcd $RC_PCD"

kill $SMI_PID 2>/dev/null
echo "SMI_PEAK $(awk '{if(NF>=3 && $3+0>m)m=$3+0}END{print m" MiB"}' "$SMI_LOG" 2>/dev/null)"
echo "SADDLE_L7R9_PCD_RC pcd=$RC_PCD"
