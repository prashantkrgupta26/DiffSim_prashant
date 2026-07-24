#!/bin/bash
#SBATCH --job-name=diffsim-a100
#SBATCH --account=mech-ai
#SBATCH --partition=nova                  # A100-80 lives on the x86 `nova` partition
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6                 # Nova caps <=6 cpu/GPU
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
# NOTE: run from /work/mech-ai/baskarg/DiffSim — HOME is over-quota.
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
.venv-nova/bin/python cluster/campaign.py --card a100 --hours 48
