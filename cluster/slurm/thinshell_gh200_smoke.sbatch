#!/bin/bash
#SBATCH --job-name=thinshell-gh200-smoke
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm
#SBATCH --qos=normal
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
nvidia-smi -L
python tests/gpu_smoke_thinshell.py
