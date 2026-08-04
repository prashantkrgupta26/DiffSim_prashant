#!/bin/bash
#SBATCH --job-name=thinshell-proj-triage
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100-pcie:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
cd /work/mech-ai/baskarg/DiffSim
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache
. .venv-nova/bin/activate
nvidia-smi -L
python tests/gpu_proj_triage_re250.py
