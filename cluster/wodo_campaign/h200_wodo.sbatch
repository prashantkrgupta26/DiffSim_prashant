#!/bin/bash
# M4 Wodo reduced-3D STRETCH — ONE case: 128x128x48 (~3.4M dofs,
# phi_s0=0.66, Lx=Ly=3.3, Np=Nf=5, Bi=0.3), device-bound cuDSS.
# The runner wraps the whole case in try/except: an ALLOC failure is
# a RECORDED outcome (it locates the direct-solver wall on 141 GB —
# the measured 48-GB wall was ~2.8M dofs 3-D Poisson; this coupled
# 4-dof CH system has a denser factor, so either outcome is
# informative).
#
# NOTE (recorded M4 item): FULL-resolution 3-D (the paper's
# 230x230x70, ~15M dofs) is beyond ANY direct solver here — it needs
# the CH BLOCK PRECONDITIONER (Krylov on the (phi,mu) saddle
# structure) before a 3-D replication campaign is possible.
#SBATCH --job-name=wodo-h200
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:h200:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/wodo-h200-$SLURM_JOB_ID
mkdir -p $OUT
RES=$OUT/results.txt
echo "== Wodo 3-D stretch (128x128x48) $(date) ==" | tee $RES
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $RES

$PY benchmarks/wodo_nova.py --fig 3d --case stretch \
    --resolution full --to-phis 0.05 --device-bound \
    --wall-cap 72000 --results-file $RES --outdir $OUT \
    > $OUT/fig3d_stretch.log 2>&1 \
    || echo "RESULT fig=3d case=stretch OUTCOME=CRASHED (see log)" \
       | tee -a $RES
tail -n 6 $OUT/fig3d_stretch.log
echo "== done $(date) ==" | tee -a $RES
