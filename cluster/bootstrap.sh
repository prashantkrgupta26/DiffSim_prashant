#!/usr/bin/env bash
# DiffSim Nova bootstrap: x86_64 (A100/H200) and aarch64 (GH200).
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=$(uname -m)
PY=${PYTHON:-python3}
echo "== DiffSim bootstrap on $ARCH =="
$PY -m venv .venv-nova
. .venv-nova/bin/activate
pip install -q --upgrade pip
pip install -q numpy scipy warp-lang pytest
if [ "$ARCH" = "aarch64" ]; then
  # GH200: ARM wheels exist for torch (cu12) and warp; AMGX skipped
  pip install -q torch --index-url https://download.pytorch.org/whl/cu126 \
    || pip install -q torch
else
  pip install -q torch --index-url https://download.pytorch.org/whl/cu126 \
    || pip install -q torch
fi
pip install -q "nvmath-python[cu12]" || echo "WARN: cuDSS unavailable"
pip install -q -e . --no-deps
python - << 'PYEOF'
import torch, warp as wp
wp.init()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", wp.__version__)
try:
    import nvmath
    print("nvmath OK (cuDSS available)")
except ImportError:
    print("nvmath MISSING - solver-table/capacity stages will skip cudss")
PYEOF
echo "== bootstrap done: .venv-nova =="
