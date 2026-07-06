#!/bin/bash
#SBATCH --job-name=diffsim-a100
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
.venv-nova/bin/python cluster/campaign.py --card a100 --hours 48
