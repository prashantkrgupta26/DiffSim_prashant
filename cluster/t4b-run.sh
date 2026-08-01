#!/bin/bash
# T4b leg driver — run inside srun --overlap on the hold node.
# Usage: bash cluster/t4b-run.sh <tag> <python-invocation...>
#   e.g. bash cluster/t4b-run.sh t4b-diag python cluster/t4b_diag.py
set -uo pipefail
cd /work/mech-ai/baskarg/DiffSim
TAG="$1"; shift
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/DiffSim/.warp-cache-arm
. .venv-nova-arm/bin/activate
export PYTHONPATH=src:tests
RES=cluster/results
mkdir -p "$RES"
LOG="$RES/${TAG}.log"
echo "[t4b] $(date -u +%FT%TZ) TAG=$TAG cmd: $*" | tee "$LOG"
nvidia-smi -L | tee -a "$LOG"
"$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[t4b] $(date -u +%FT%TZ) done $TAG rc=$RC" | tee -a "$LOG"
exit $RC
