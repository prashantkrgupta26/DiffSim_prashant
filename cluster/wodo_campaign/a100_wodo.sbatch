#!/bin/bash
# M4 Wodo CMS-2012 replication campaign — SEQUENTIAL full-resolution
# runs (figs 3-7, 14 cases; benchmarks/wodo_nova.py --list for the
# table). 2-D full = 250x100 (the paper's own mesh); fig3 1-D at
# 4x256. Device-bound marches (slot-map scatter + zero-copy cuDSS;
# measured 56x vs host-splu, 8x vs host-cudss at 96x48). Order:
# gate-critical regimes first (fig6 N-ladder, fig7 chi-asymmetry),
# then fig4/5, then the cheap 1-D fig3 sanity triple.
#SBATCH --job-name=wodo-a100
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/wodo-a100-$SLURM_JOB_ID
mkdir -p $OUT
RES=$OUT/results.txt
echo "== Wodo Nova campaign (full res, device-bound) $(date) ==" | tee $RES
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $RES

for spec in "6 n5" "6 n100" "7 chifs" "7 chips" "6 n20" "7 sym" \
            "4 bi0.03" "4 bi0.3" "4 bi3" \
            "5 blend11" "5 blend108" \
            "3 bi0.1" "3 bi1" "3 bi10"; do
  set -- $spec
  echo "===== fig$1 / $2 (full) $(date) =====" | tee -a $RES
  $PY benchmarks/wodo_nova.py --fig "$1" --case "$2" \
      --resolution full --to-phis 0.05 --device-bound \
      --wall-cap 10800 --results-file $RES --outdir $OUT \
      > $OUT/fig$1_$2.log 2>&1 \
      || echo "RESULT fig=$1 case=$2 OUTCOME=CRASHED (see log)" \
         | tee -a $RES
  tail -n 4 $OUT/fig$1_$2.log
done
echo "== campaign done $(date) ==" | tee -a $RES
