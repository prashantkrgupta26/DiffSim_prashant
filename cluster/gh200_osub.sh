#!/usr/bin/env bash
# GH200 oversubscription law (M1d): hero-size film under managed memory,
# 4 rungs: balloon 0 (baseline), 62.7, 67.7, 71.3 GB -> 110/130/150% of
# effective HBM for the ~35 GB working set. One GPU, sequential.
set -uo pipefail
VENV=/work/mech-ai/baskarg/DiffSim/.venv-nova-arm
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache

echo "=== GH200 oversubscription probe $(date) ==="
hostname; uname -m
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
. "$VENV/bin/activate"

BASE=/work/mech-ai/baskarg/DiffSim/cluster/results/gh200-probe2
for B in 0 62.7 67.7 71.3; do
  TAG=osub-b${B}
  OUT=$BASE/${TAG}-${SLURM_JOB_ID}
  mkdir -p "$OUT"
  echo "##### RUN $TAG $(date) #####"
  ( while true; do
      echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)";
      sleep 2; done ) > "$OUT/smi.log" 2>&1 &
  SMI_PID=$!
  EXTRA="--managed"
  if [ "$B" != "0" ]; then EXTRA="--managed --balloon-gb $B"; fi
  python /work/mech-ai/baskarg/DiffSim/benchmarks/gh200_capacity_probe.py \
    --res 230 230 70 --max-steps 4 --outdir "$OUT" $EXTRA 2>&1
  echo "RUN_RC $TAG $?"
  kill $SMI_PID 2>/dev/null
  echo "RUN_PEAK_SMI $TAG $(awk "{if(\$3>m)m=\$3}END{print m\" MiB\"}" "$OUT/smi.log" 2>/dev/null)"
done
echo "=== oversubscription probe done $(date) ==="
