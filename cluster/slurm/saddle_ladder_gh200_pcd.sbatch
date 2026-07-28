#!/bin/bash
#SBATCH --job-name=saddle-l7r9-pcd
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
# A5 verdict leg: the 9.24M-DOF 3d-L7r9 ladder point (both saddle solvers).
# gpubox (62 GB host) OOM'd this point; WP0 (job 11771926) measured the r7b9
# mesh build itself at 3.71 GB peak on Grace, so 400G gives ample headroom for
# build + host CSR (~16 GB) + solver working set while we capture the true peak.
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
# between ladder legs). Honest rows: a non-zero RC on one leg must not mask the
# other, so RCs are captured per leg.
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_pcd SADDLE_DEVICE=cuda:0 SADDLE_NSTEPS=5 \
  PYTHONPATH=src:tests python tests/gpu_saddle_ladder.py
RC_PCD=$?
echo "LEG_RC pcd $RC_PCD"

kill $SMI_PID 2>/dev/null
echo "SMI_PEAK $(awk '{if(NF>=3 && $3+0>m)m=$3+0}END{print m" MiB"}' "$SMI_LOG" 2>/dev/null)"
echo "SADDLE_L7R9_RC pcd=$RC_PCD"
