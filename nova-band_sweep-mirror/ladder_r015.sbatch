#!/bin/bash
#SBATCH --job-name=ladder_r015
#SBATCH --partition=nova
#SBATCH --account=mech-ai
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/home/baskarg/band_sweep/ladder_r015_%j.log

set -u

cd ~/DiffSim
module load python/3.11.11-dnlm5qp
source .venv-nova/bin/activate

# Run from a scratch dir so band_study.py's cwd-relative output file
# (band_study_results.txt) lands outside the repo.
WORKDIR=/home/baskarg/band_sweep
mkdir -p "$WORKDIR"
cd "$WORKDIR"
RESULTS="$WORKDIR/ladder_r015_results.txt"

echo "=== r=0.15 ladder job $SLURM_JOB_ID started $(date) on $(hostname) ===" >> "$RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$RESULTS" 2>&1

python ~/DiffSim/benchmarks/band_study.py 3 4 5 6 7 --radius 0.15 >> "$RESULTS" 2>&1
echo "--- ladder exit=$? $(date) ---" >> "$RESULTS"

echo "=== r=0.15 ladder job $SLURM_JOB_ID finished $(date) ===" >> "$RESULTS"
