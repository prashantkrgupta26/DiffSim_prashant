#!/usr/bin/env bash
# Task #40 G3: size-ladder off/auto A/B — measures where launch overhead
# (size-independent) crosses over against GPU arithmetic (size-dependent).
# Run on gpubox via gpubox-run.sh; prints per-config stage tables.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
for cfg in "64 24" "48 16" "32 12"; do
  set -- $cfg
  for m in off auto; do
    echo "=== nx=$1 nz=$2 mode=$m ==="
    DIFFSIM_KRYLOV_GRAPH=$m .venv/bin/python \
      benchmarks/phase-field/wodo_profile_host.py \
      --nx "$1" --nz "$2" --device cuda:1 --warmup 2 --steps 5 \
      --no-launch-audit 2>&1 | grep -E "STAGE TABLE|  solve |profile mesh"
  done
done
