#!/bin/bash
set -uo pipefail
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/DiffSim/.warp-cache-hdr
. .venv-nova-arm/bin/activate
export PYTHONPATH=src:tests
export SADDLE_DEVICE_CSR=1
export TRUCK_ASSEMBLY=device
RES=cluster/results; mkdir -p "$RES"
LOG="$RES/t5-gate.log"
echo "[t5-gate] $(date -u +%FT%TZ) start" | tee "$LOG"
nvidia-smi -L | tee -a "$LOG"
python -c "import warp; print('WARP',warp.__version__)" | tee -a "$LOG"
python cluster/t5_gate.py 2>&1 | tee -a "$LOG"
echo "[t5-gate] $(date -u +%FT%TZ) done rc=${PIPESTATUS[0]}" | tee -a "$LOG"
