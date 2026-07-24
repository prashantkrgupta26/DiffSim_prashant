#!/bin/bash
#SBATCH --job-name=diffsim-gh200
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm              # GH200 (aarch64) lives on `nova-arm`
#SBATCH --qos=normal
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6                 # Nova caps <=6 cpu/GPU
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
# NOTE: run from /work/mech-ai/baskarg/DiffSim — HOME is over-quota.
# ARM Python 3.11 was RPM-extracted (no root); its libs are NOT on the system
# path, so LD_LIBRARY_PATH must point at the extracted lib64 for every process.
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/DiffSim/.warp-cache-arm
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap_arm.sh
.venv-nova-arm/bin/python cluster/campaign.py --card gh200 --hours 48
