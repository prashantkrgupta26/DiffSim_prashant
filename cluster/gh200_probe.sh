#!/usr/bin/env bash
# GH200 coherent-capacity probe launcher (M1d #34 / idx-widening #33).
# One size per job. Args: NX NY NZ MAXSTEPS [extra probe args verbatim]
set -uo pipefail
NX=$1; NY=$2; NZ=$3; MAXSTEPS=$4; shift 4

VENV=/work/mech-ai/baskarg/DiffSim/.venv-nova-arm
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache
mkdir -p "$WARP_CACHE_PATH"

TAG=n${NX}x${NY}x${NZ}$(echo "$@" | tr -cd "a-z0-9" | head -c 12)
OUT=/work/mech-ai/baskarg/DiffSim/cluster/results/gh200-probe2/${TAG}-${SLURM_JOB_ID}
mkdir -p "$OUT"

echo "=== GH200 capacity probe $TAG extra=[$*] $(date) ==="
hostname; uname -m
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git -C /work/mech-ai/baskarg/DiffSim log --oneline -1

. "$VENV/bin/activate"
python -c "import diffsim, warp, torch, nvmath, yaml; print(\"imports ok warp\", warp.__version__)" || { echo IMPORT_FAIL; exit 3; }

( while true; do
    echo "SMI $(date +%s) $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)";
    sleep 2; done ) > "$OUT/smi.log" 2>&1 &
SMI_PID=$!

python /work/mech-ai/baskarg/DiffSim/benchmarks/gh200_capacity_probe.py \
  --res $NX $NY $NZ --max-steps $MAXSTEPS --outdir "$OUT" "$@" 2>&1
RC=$?
kill $SMI_PID 2>/dev/null
echo "PROBE_PEAK_SMI $(awk -F"[ ,]" "{if(\$3>m)m=\$3}END{print m\" MiB\"}" "$OUT/smi.log" 2>/dev/null)"
echo "=== probe $TAG done rc=$RC $(date) ==="
