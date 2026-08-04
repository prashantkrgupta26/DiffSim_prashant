#!/bin/bash
#SBATCH --job-name=thinshell-gh200-ladder
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm
#SBATCH --qos=normal
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
nvidia-smi -L

# Background nvidia-smi sampling loop (pattern from cluster/gh200_osub.sh)
SMI_LOG="/work/mech-ai/baskarg/DiffSim/cluster/results/${SLURM_JOB_ID}-smi.log"
( while true; do
    echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)";
    sleep 10; done ) > "$SMI_LOG" 2>&1 &
SMI_PID=$!

PYTHONPATH=src:tests python tests/gpu_gh200_ladder.py

RC=$?
kill $SMI_PID 2>/dev/null
echo "SMI_PEAK $(awk '{if(NF>=3 && $3+0>m)m=$3+0}END{print m" MiB"}' "$SMI_LOG" 2>/dev/null)"
echo "LADDER_EXIT_RC $RC"
