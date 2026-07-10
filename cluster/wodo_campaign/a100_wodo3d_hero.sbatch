#!/bin/bash
# Wodo CMS-2012 3-D HERO — the Fig 6 chain-length ladder and Fig 7
# solvent-selectivity regimes extended to FULL 3-D at the paper's own
# 230x230x70 mesh (15.15M dofs, 1.61B nnz = 75% of the int32 slot
# range; the paper ran this size on 256 CPUs). Reuses the G5 rung-c
# mechanics exactly: blockch_dev device-resident setup, node-graph
# CSR pattern, batched Ae fill.
#
# SINGLE SOURCE OF TRUTH: physics comes from the film front-end
# configs (src/diffsim/film/configs/wodo2012_fig{6,7}_*.yaml). The
# --set overrides below are the documented 3-D EXTENSION only:
#   * domain 3.3 x 3.3 x 1 at 230x230x70 (the paper's 3-D box; the
#     metric needs Lx/nx == Ly/ny, and 3.3/230 = 0.014348 both ways);
#   * phi_s0 = 0.66 (the paper's 3-D initial solvent fraction; the
#     2-D configs carry their Sec. 7 value 0.75) — blend stays 1:1
#     via the ratio form;
#   * blockch_dev (cuDSS direct ALLOC-fails 8x below this size).
#
# PREFLIGHT NOTE (--no-strict, recorded deviation): at 230 cells over
# Lx = 3.3 the N=100 cases sit AT/below the validated 2.5-element
# interface floor (xi_early = 2.5 elems, xi_final = 1.8 — the paper
# itself ran 3-D only for N=5, which preflights fine at 6.8 elems).
# The interface-resolution check therefore FAILs for the N=100 rungs;
# we run them anyway with phi excursions monitored in flight and
# record the check verdicts in each runlog.
#
# EXPECTED WALLS (honest basis): 236 s/step steady MEASURED at this
# exact size on RTX 6000 Ada 48 GB (G5 rung c, docs/dev/2026-07-09-
# blockch-preconditioner.md; first step +~500 s for the symbolic pair
# map). A100-80 is an ESTIMATE at ~1.5-2x faster => ~120-160 s/step.
# dh_cap needs >= ~185 accepted steps to h=0.27 -> >= ~6-8 h/case
# BEST case; the 15 h per-case cap gives the two gate-critical cases
# (n5, n100) a full-dryness shot and the rest partial marches.
# Memory forecast (G5 model): 33.4 GB GPU / 37.8 GB host (measured
# 33.4/37.7 at rung c) — 2.4x GPU headroom on A100-80.
#SBATCH --job-name=wodo3d-hero
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=96:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/wodo3d-hero-$SLURM_JOB_ID
mkdir -p $OUT
RES=$OUT/results.txt
echo "== Wodo 3-D hero (230x230x70, blockch_dev) $(date) ==" | tee $RES
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $RES

# gate-critical first: the N-ladder endpoints, then selectivity pair,
# then the middle rungs
for case in fig6_n5 fig6_n100 fig7_chifs fig7_chips fig6_n20 fig7_sym; do
  echo "===== wodo2012_${case} 3-D $(date) =====" | tee -a $RES
  $PY -m diffsim.film src/diffsim/film/configs/wodo2012_${case}.yaml \
      --outdir $OUT/${case} \
      --set domain.dim=3 \
      --set domain.Lx=3.3 \
      --set domain.Ly=3.3 \
      --set "domain.resolution=[230,230,70]" \
      --set "physics.blend={ratio: '1:1', phi_s0: 0.66}" \
      --set numerics.linsolver=blockch_dev \
      --set stop.wall_cap=54000 \
      --no-strict \
      > $OUT/wodo3d_${case}.log 2>&1 \
      || echo "RESULT case=${case} OUTCOME=CRASHED-OR-AUTOPSY (see log)" \
         | tee -a $RES
  grep "^RESULT" $OUT/wodo3d_${case}.log | tee -a $RES
  tail -n 4 $OUT/wodo3d_${case}.log
done
echo "== hero done $(date) ==" | tee -a $RES
