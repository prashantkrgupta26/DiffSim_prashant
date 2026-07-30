#!/bin/bash
set -uo pipefail
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/DiffSim/.warp-cache-hdr
. .venv-nova-arm/bin/activate
export PYTHONPATH=src:tests
export SADDLE_DEVICE_CSR=1
python -c "import warp; print('WARP', warp.__version__)"
"$@"
