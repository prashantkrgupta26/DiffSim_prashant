#!/bin/bash
# Negi 2018 spin-coating campaign in FULL 3-D — the four rpm cases
# (Bi = 0.0073/0.0051/0.0033/0.0019) at 125x125x250 cells over the
# paper's 500x500x1000 nm domain (15.94M dofs, 1.70B nnz = 79% of the
# int32 slot range), blockch_dev device-resident marches (G5 rung-c
# mechanics: node-graph pattern + batched Ae + device pair setup).
#
# SINGLE SOURCE OF TRUTH: parameters come from the film front-end
# configs (src/diffsim/film/configs/negi2018_*rpm.yaml). This script
# only overrides the DOMAIN to 3-D and the solver to blockch_dev —
# never the physics.
#
# PREFLIGHT NOTE (--no-strict, recorded deviation): at 125 lateral cells
# over the paper's 500 nm box (dx = 4 nm) the forming interface sits AT
# ~2.2 elements (xi_early 0.0088 h0-units) — below the validated 2.5-elem
# floor, so the interface-resolution check FAILs for all four rpm cases.
# The 4-element rule would need >= ~227 lateral cells (~52M dofs), past
# even the 80 GB card, so refinement is not an option on a single GPU;
# we run --no-strict (preflight: warn) and monitor phi min/max in flight,
# recording the check verdicts in each runlog — exactly the deviation the
# sibling a100_wodo3d_hero kit takes for its N=100 rungs. WITHOUT this
# flag the default strict preflight ABORTS every case before it marches.
#
# EXPECTED WALLS (honest basis): the G5 ladder measured 236 s/step
# steady at 15.15M dofs on RTX 6000 Ada 48 GB (33.4 GB GPU / 37.7 GB
# host peak; docs/dev/2026-07-09-blockch-preconditioner.md). A100-80
# is an ESTIMATE at ~1.5-2x faster (HBM-bandwidth-bound spmv), i.e.
# ~120-160 s/step at this 15.94M-dof size. The dh_cap ladder needs
# >= ~225 accepted steps to dryness -> plan >= 1 day per case; the
# 10.5 h per-case wall cap below (4 cases fit the 48 h job) yields a
# PARTIAL march per case on the first pass (snapshots at the h
# milestones still land) — extend --set stop.wall_cap and resubmit
# per-case for production dryness runs.
# GPU memory forecast (G5 model): 35.0 GB GPU / 39.7 GB host — 2.3x
# headroom on A100-80.
#SBATCH --job-name=negi3d-a100
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/negi3d-a100-$SLURM_JOB_ID
mkdir -p $OUT
RES=$OUT/results.txt
echo "== Negi 3-D campaign (125x125x250, blockch_dev) $(date) ==" | tee $RES
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $RES

for rpm in 6000 3000 1500 500; do
  echo "===== negi2018_${rpm}rpm 3-D $(date) =====" | tee -a $RES
  $PY -m diffsim.film src/diffsim/film/configs/negi2018_${rpm}rpm.yaml \
      --outdir $OUT/${rpm}rpm \
      --set domain.dim=3 \
      --set domain.Ly=0.5 \
      --set "domain.resolution=[125,125,250]" \
      --set numerics.linsolver=blockch_dev \
      --set stop.wall_cap=37800 \
      --no-strict \
      > $OUT/negi_${rpm}rpm.log 2>&1 \
      || echo "RESULT case=${rpm}rpm OUTCOME=CRASHED-OR-AUTOPSY (see log)" \
         | tee -a $RES
  grep "^RESULT" $OUT/negi_${rpm}rpm.log | tee -a $RES
  tail -n 4 $OUT/negi_${rpm}rpm.log
done
echo "== campaign done $(date) ==" | tee -a $RES
