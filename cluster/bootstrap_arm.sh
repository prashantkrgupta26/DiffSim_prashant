#!/usr/bin/env bash
# DiffSim ARM bootstrap for GH200 (aarch64) — Nova nova-arm partition
# Uses Python 3.11 extracted from EPEL RPM (no root needed).
# Run inside a nova-arm job (with or without GPU gres).
set -euo pipefail
cd /work/mech-ai/baskarg/DiffSim

VENV=/work/mech-ai/baskarg/DiffSim/.venv-nova-arm
RPM_DIR=/work/mech-ai/baskarg/python311-arm
PY311="$RPM_DIR/usr/bin/python3.11"
LIBDIR="$RPM_DIR/usr/lib64"

# Prepend RPM libs to LD_LIBRARY_PATH for this and all child processes
export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== DiffSim ARM bootstrap: arch=$(uname -m) python=$($PY311 --version 2>&1) =="

if [ -x "$VENV/bin/python" ] && LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" "$VENV/bin/python" -c "import diffsim" 2>/dev/null; then
  echo "== .venv-nova-arm already provisioned (diffsim importable); skipping =="
  exit 0
fi

# Recreate venv without ensurepip (pip missing in RPM extraction)
"$PY311" -m venv --without-pip "$VENV"
. "$VENV/bin/activate"

# Bootstrap pip via get-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python
pip install -q --upgrade pip

pip install -q numpy scipy warp-lang pytest pyyaml
# ARM cu126 wheels: PyTorch publishes aarch64+cu12 wheels  
pip install -q torch --index-url https://download.pytorch.org/whl/cu126 \
  || pip install -q torch
pip install -q "nvmath-python[cu12]" || echo "WARN: cuDSS unavailable on ARM"
pip install -q -e . --no-deps

python - << PYEOF
import sys
print("Python:", sys.version, sys.executable)
import torch, warp as wp
wp.init()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", wp.__version__)
try:
    import nvmath
    print("nvmath OK (cuDSS available)")
except ImportError:
    print("nvmath MISSING - cudss stages will skip")
import diffsim
print("diffsim importable OK")
PYEOF
echo "== bootstrap_arm done: $VENV =="
