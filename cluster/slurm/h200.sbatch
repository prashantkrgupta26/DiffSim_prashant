#!/bin/bash
#SBATCH --job-name=diffsim-h200
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:h200:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
.venv-nova/bin/python cluster/campaign.py --card h200 --hours 48
