#!/usr/bin/env bash
set -euo pipefail
VENV=/work/mech-ai/baskarg/DiffSim/.venv-nova-arm
RPM_DIR=/work/mech-ai/baskarg/python311-arm
LIBDIR="$RPM_DIR/usr/lib64"
export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== GH200 Smoke Test ==="
hostname; uname -m
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

. "$VENV/bin/activate"
python - << PYEOF
import torch, warp as wp
wp.init()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", wp.__version__)
print("warp cuda device:", wp.get_cuda_device())
PYEOF

echo "=== running diffsim smoke (5 steps) ==="
mkdir -p /work/mech-ai/baskarg/DiffSim/cluster/results/gh200-smoke
time python -m diffsim.film \
  /work/mech-ai/baskarg/DiffSim/src/diffsim/film/configs/wodo2012_fig6_n5.yaml \
  --outdir /work/mech-ai/baskarg/DiffSim/cluster/results/gh200-smoke \
  --set stop.max_steps=5 \
  --no-strict \
  2>&1
EXIT=$?
echo "diffsim.film exited with $EXIT"
echo "=== smoke done ==="
