#!/usr/bin/env bash
set -uo pipefail
cd /home/bglab/Baskar/DiffSim-l8
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$HOME/AMGX/build
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/bglab/Baskar/DiffSim-l8/src
PY=/home/bglab/Baskar/DiffSim/.venv/bin/python
mkdir -p logs
echo "===== A) PPE PARITY L4-5 (AMGX vs splu, device K_p) $(date) ====="
$PY tests/ppe_e2e_device.py --parity --levels 4 5
echo "A_EXIT=$?"
echo "===== B) DEVICE-vs-HOST K_p PARITY L7 (structural+numeric) $(date) ====="
$PY tests/ppe_amgx_scaling.py --assembly-parity --levels 7
echo "B_EXIT=$?"
echo "===== C) E2E DEVICE PATH L6 L7 L8 (AMGX solve) $(date) ====="
$PY tests/ppe_e2e_device.py --levels 6 7 8 --steps 4
echo "C_EXIT=$?"
echo "===== ALL_DONE $(date) ====="
