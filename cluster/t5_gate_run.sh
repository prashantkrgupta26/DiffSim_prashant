#!/bin/bash
# T5 gate-leg runner.
#
# LOG HYGIENE (post-incident, TU5R): the log filename carries a UTC timestamp
# suffix so a relaunch can NEVER overwrite a prior leg's log, and every write
# uses `tee -a` (append-only) so nothing already in the file is truncated.
# A stable symlink `t5-gate-latest.log` always points at the newest run.
# The old behaviour (`tee "$LOG"` to a fixed `t5-gate.log`) destroyed leg 1's
# record when leg 2 was launched; that must not recur.
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
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RES/t5-gate-${TS}.log"
ln -sf "$(basename "$LOG")" "$RES/t5-gate-latest.log"
echo "[t5-gate] $(date -u +%FT%TZ) start (log=$LOG)" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"
python -c "import warp; print('WARP',warp.__version__)" | tee -a "$LOG"
python cluster/t5_gate.py 2>&1 | tee -a "$LOG"
echo "[t5-gate] $(date -u +%FT%TZ) done rc=${PIPESTATUS[0]}" | tee -a "$LOG"
