#!/bin/bash
# S3-3D PILOT (measure-first, NOT a campaign) — the M5 evaporation-
# quench hero candidate at 64x64x32 (811,776 dofs), the size that
# exceeds the 48 GB workstation ceiling (cuDSS hybrid crawl at
# 48.2 GB; docs/dev/2026-07-13-m5-device-assembly.md Sec 4-5).
#
# PURPOSE: get the real A100-80 numbers — setup wall, s/step, memory —
# for ~60 steps of the production noisy quench, with field snapshots.
# EXPECTATION (honest basis): SOLVER-BOUND under cuDSS direct.
# Measured on RTX 6000 Ada 48 GB at HALF this size (64x64x16, 417,792
# dofs): 71.7 s/step, cuDSS 17.7 s/call, 24.5 GB GPU, setup 124 s.
# At 2x dofs expect minutes/step and ~50-65 GB — if the factorization
# enters a hybrid/managed-memory crawl even on 80 GB, the pilot's
# verdict is "wait for blockch-(M,K)" (the recorded follow-up that
# replaces the direct solve; task ledger).  The 6 h wall cap bounds
# the spend either way; step JSON prints stream to the log so even a
# 3-step run is a complete measurement.
#
# Physics/config single source of truth: benchmarks/performance/
# m5_device_assembly.py CASES["3d_slab64z32"] + make_stepper (S3b
# production energetics, assembly="device", march ladder).
#SBATCH --job-name=s3d-pilot
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if typed
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=07:00:00
#SBATCH --output=cluster/results/%x-%j.out
cd $SLURM_SUBMIT_DIR
bash cluster/bootstrap.sh
PY=.venv-nova/bin/python
OUT=cluster/results/s3d-pilot-$SLURM_JOB_ID
mkdir -p $OUT
echo "== S3-3D pilot (slab64z32, device assembly) $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY cluster/wodo_campaign/s3d_pilot_driver.py \
    --case 3d_slab64z32 --assembly device --noise 5e-3 \
    --steps 60 --wall-cap 21000 --outdir $OUT \
    > $OUT/pilot.log 2>&1 \
    || echo "PILOT CRASHED-OR-CAPPED (see log)"
grep -E '"event"' $OUT/pilot.log | tail -5
echo "== pilot done $(date) =="
