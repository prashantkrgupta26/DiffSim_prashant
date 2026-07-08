#!/bin/bash
#SBATCH --job-name=radius_sweep_L7
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/baskarg/band_sweep/radius_sweep_L7_%j.log

set -u

cd ~/DiffSim
module load python/3.11.11-dnlm5qp
source .venv-nova/bin/activate

# Run from a scratch dir so band_study.py's cwd-relative output file
# (band_study_results.txt) lands outside the repo.
WORKDIR=/home/baskarg/band_sweep
mkdir -p "$WORKDIR"
cd "$WORKDIR"
RESULTS="$WORKDIR/radius_sweep_L7_results.txt"

echo "=== L7 sweep job $SLURM_JOB_ID started $(date) on $(hostname) ===" >> "$RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$RESULTS" 2>&1

for r in 0.15 0.17 0.19 0.21 0.23 0.25 0.27 0.29 0.31 0.33 0.35; do
    echo "--- radius $r start $(date) ---" >> "$RESULTS"
    python ~/DiffSim/benchmarks/band_study.py 3 7 --radius "$r" >> "$RESULTS" 2>&1
    echo "--- radius $r exit=$? $(date) ---" >> "$RESULTS"
done

echo "=== L7 sweep job $SLURM_JOB_ID finished $(date) ===" >> "$RESULTS"
