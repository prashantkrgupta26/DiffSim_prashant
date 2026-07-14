#!/bin/bash
# S3-3D HERO (blockch_dev) — the M5 evaporation-quench hero case at
# 64x64x32 (811,008 dofs), the size that CEILINGed cuDSS (>16 min
# factorization crawl at 48.2 GB; dev note
# docs/dev/2026-07-13-blockch-mpf.md Sec 3).
#
# WHAT CHANGED (2026-07-14): this kit was a measure-first cuDSS PILOT
# (solver-bound, expected minutes/step).  blockch-(M,K) is now merged
# (src/diffsim/solvers/linsolve.py blockch_dev; MultiPhaseStepper
# linsolver=blockch_dev) and MEASURED 8.4x faster than cuDSS at slab64
# (2.10 vs 17.7 s/call) — it marches the 811k cuDSS-CEILING case at
# 28.6 s/step.  So the kit now runs linsolver=blockch_dev
# (assembly="device"): a real campaign, not a ceiling probe.
#
# EXPECTED WALLS (honest basis, MEASURED RTX 6000 Ada 48 GB — dev note
# 2026-07-13-blockch-mpf.md, NOT invented):
#   * 3d_slab64z32 (811,008 dofs, M2/K1): 28.6 s/step, 2.16 s/call,
#     ~22 GB GPU (blockch_dev; cuDSS CEILINGed here).       [Sec 3]
# A100-80 estimate (HBM2e ~2x GDDR6 on the spmv-bound solve): ~15 s/step
# with ~3.6x GPU headroom (dev note; extrapolation, NOT a measurement).
# The 6 h wall cap bounds the spend; step JSON streams to the log, so
# even a few-step run is a complete measurement.
#
# Physics/config single source of truth: benchmarks/performance/
# m5_device_assembly.py CASES["3d_slab64z32"] + make_stepper (S3b
# production energetics, assembly="device", march ladder).  The larger
# (M=3,K=2) 128x128x64 rung is the OPTIONAL variant a100_s3d_hero_mk32
# .sbatch (23.9 GB / 160.8 s/step measured).
#SBATCH --job-name=s3d-hero
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/s3d-hero-$SLURM_JOB_ID
mkdir -p $OUT
echo "== S3-3D hero (slab64z32, device assembly, blockch_dev) $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY cluster/wodo_campaign/s3d_pilot_driver.py \
    --case 3d_slab64z32 --assembly device --linsolver blockch_dev \
    --noise 5e-3 --steps 60 --wall-cap 21000 --outdir $OUT \
    > $OUT/hero.log 2>&1 \
    || echo "HERO CRASHED-OR-CAPPED (see log)"
grep -E '"event"' $OUT/hero.log | tail -5
echo "== hero done $(date) =="
