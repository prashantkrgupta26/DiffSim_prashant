#!/bin/bash
# S3-3D HERO mk32 (OPTIONAL larger rung) — the (M=3, K=2) capacity
# case at 128x128x64: 1,064,960 nodes, 10,649,600 dofs (ndof = 10),
# block-masked nnz 1.537B.  This is the largest single-card rung the
# blockch preconditioner reaches for this (M,K) — cuDSS cannot touch
# it (its measured ceiling was 812k dofs, 13x fewer) and the SUPERSET
# pattern overflows int32 here, so --block-sparse is REQUIRED.
#
# EXPECTED WALLS (honest basis, MEASURED RTX 6000 Ada 48 GB — dev note
# docs/dev/2026-07-13-blockch-mpf.md Sec 6, NOT invented):
#   * 3d_slab128z64_mk32 (10.6M dofs, M3/K2, block-masked 54/100 live):
#     160.8 s/step, 9.06 s/call, asm 4.08 s/call, 4 Newton its/step,
#     zero fallbacks, 23.9 GB GPU / 18.9 GB host; setup+first step
#     612.1 s.                                                [Sec 6]
# A100-80 estimate (HBM2e ~2x GDDR6 on the spmv-bound solve): ~80-90
# s/step with 80 GB clearing ~3.3x the 23.9 GB footprint (dev note,
# extrapolation, NOT a measurement).
#
# NOISE CAVEAT (measured, dev note Sec 5 B4): the full noisy-quench
# march at 128x128x64 is the RECORDED FRONTIER — the 2-D-calibrated FDT
# amplitude (noise_psi 5e-3) is ~30x too large per gauss point at
# h = 1/128 3-D and the onset attempt diverges in the linear solve; a
# 3-D noise recalibration is the open item.  The DETERMINISTIC quench
# marches fine (the measured basis above: accepted fixed-dt Newton
# solves, zero rejects).  This rung therefore runs noise=0 — a capacity
# + wall demonstration of blockch_dev stepping the 10.6M-dof (M=3,K=2)
# system on one card, not a noisy-morphology production run.
#
# Physics/config single source of truth: benchmarks/performance/
# m5_device_assembly.py CASES["3d_slab128z64_mk32"] + make_stepper
# (mk=(3,2); the S3b energetics extended by a third polymer + second
# crystallizable species).
#SBATCH --job-name=s3d-hero-mk32
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/s3d-hero-mk32-$SLURM_JOB_ID
mkdir -p $OUT
echo "== S3-3D hero mk32 (128x128x64, M3/K2, blockch_dev + block-mask) $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY cluster/wodo_campaign/s3d_pilot_driver.py \
    --case 3d_slab128z64_mk32 --assembly device --linsolver blockch_dev \
    --block-sparse --noise 0.0 --steps 40 --wall-cap 39600 \
    --outdir $OUT \
    > $OUT/hero_mk32.log 2>&1 \
    || echo "HERO-MK32 CRASHED-OR-CAPPED (see log)"
grep -E '"event"' $OUT/hero_mk32.log | tail -5
echo "== hero mk32 done $(date) =="
