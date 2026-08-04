#!/bin/bash
#SBATCH --job-name=saddle-ladder-l7r9
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
# T4 gate leg (bdiag): 9.24M-DOF 3d-L7r9 with device assembly.
# Device assembly (SADDLE_ASSEMBLY=device) removes the 243-245 GB host CSR
# transient observed in the Track A campaign; expected setup time <10 min vs
# ~1.5 h host-path.  Reference: host-path bdiag 3d-L7r9 1196.8 iters /
# 242.2 s/step (9.08M), host assembly setup ~1.5 h, RSS 243-245 GB.
# The SMI sampler captures the GPU memory peak to confirm device-assembly
# savings.  Partition/account/qos/gres/cpus/mem/time are injected by
# nova-sync-submit.sh.
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
# between ladder legs).  SADDLE_ASSEMBLY=device skips the host-side CSR build;
# device assembly expected to cut setup time from ~1.5 h to minutes.
RC_BDIAG=1
SADDLE_POINTS=3d-L7r9 SADDLE_SOLVERS=fgmres_bdiag SADDLE_DEVICE=cuda:0 \
  SADDLE_NSTEPS=5 SADDLE_ASSEMBLY=device \
  PYTHONPATH=src:tests python tests/gpu_saddle_ladder.py
RC_BDIAG=$?
echo "LEG_RC bdiag $RC_BDIAG"

kill $SMI_PID 2>/dev/null
echo "SMI_PEAK $(awk '{if(NF>=3 && $3+0>m)m=$3+0}END{print m" MiB"}' "$SMI_LOG" 2>/dev/null)"
echo "SADDLE_L7R9_BDIAG_RC bdiag=$RC_BDIAG"
