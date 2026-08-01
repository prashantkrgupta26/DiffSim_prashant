#!/bin/bash
# T5 header-graft probe wrapper: runs "$@" under the warp-native-header graft
# environment (WARP_CACHE_PATH=.warp-cache-hdr).
#
# LOG HYGIENE (post-incident, TU5R): this wrapper now ALWAYS logs. Previously
# it printed to stdout only, so a SUCCESSFUL probe left no cluster log and an
# audit could only find an EARLIER failed probe's log — the successful probe
# was invisible. Every invocation tees (append-only) to a UTC-timestamped log
# under cluster/results/, with a stable `t5-hdr-latest.log` symlink, and exits
# with the wrapped command's return code.
set -uo pipefail
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/DiffSim/.warp-cache-hdr
. .venv-nova-arm/bin/activate
export PYTHONPATH=src:tests
export SADDLE_DEVICE_CSR=1
RES=cluster/results; mkdir -p "$RES"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RES/t5-hdr-${TS}.log"
ln -sf "$(basename "$LOG")" "$RES/t5-hdr-latest.log"
{
  echo "[t5-hdr] $(date -u +%FT%TZ) start (log=$LOG) cmd: $*"
  python -c "import warp; print('WARP', warp.__version__)"
  "$@"
} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[t5-hdr] $(date -u +%FT%TZ) done rc=$rc" | tee -a "$LOG"
exit "$rc"
