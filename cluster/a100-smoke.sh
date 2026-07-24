#!/usr/bin/env bash
# DiffSim A100 (x86_64) smoke test — Nova `nova` partition.
# Uses the x86 venv `.venv-nova` (built by cluster/bootstrap.sh).
set -euo pipefail
VENV=/work/mech-ai/baskarg/DiffSim/.venv-nova

echo "=== A100 Smoke Test ==="
hostname; uname -m
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

. "$VENV/bin/activate"
python - << 'PYEOF'
import torch, warp as wp
wp.init()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", wp.__version__)
print("warp cuda device:", wp.get_cuda_device())
PYEOF

echo "=== running diffsim smoke (5 steps) ==="
mkdir -p /work/mech-ai/baskarg/DiffSim/cluster/results/a100-smoke
time python -m diffsim.film \
  /work/mech-ai/baskarg/DiffSim/src/diffsim/film/configs/wodo2012_fig6_n5.yaml \
  --outdir /work/mech-ai/baskarg/DiffSim/cluster/results/a100-smoke \
  --set stop.max_steps=5 \
  --no-strict \
  2>&1
EXIT=$?
echo "diffsim.film exited with $EXIT"
echo "=== smoke done ==="
