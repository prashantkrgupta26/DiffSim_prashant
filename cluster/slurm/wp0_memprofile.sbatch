#!/bin/bash
#SBATCH --job-name=wp0-memprofile
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm
#SBATCH --qos=normal
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mech-ai/baskarg/DiffSim/cluster/results/%x-%j.out
# WP0: reproduce the r7b9 adaptive mesh-build memory anomaly on Grace
# (209.7GB OOM at the 200G cgroup during the GH200 ladder) with headroom
# to SURVIVE it and measure the true peak. CASES=7:9 = the exact anomaly
# case; 6:9 rides along as the on-node small-scale reference point.
cd /work/mech-ai/baskarg/DiffSim
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WARP_CACHE_PATH=/work/mech-ai/baskarg/.warp-cache-arm
. .venv-nova-arm/bin/activate
python -c "import psutil" 2>/dev/null || python -m pip install -q psutil
echo "=== node: $(hostname) $(uname -m); mem cgroup: ${SLURM_MEM_PER_NODE}M ==="
# background whole-process RSS log (2s cadence) as the coarse cross-check
( while true; do
    echo "RSS $(date +%s) $(grep VmRSS /proc/self/../$$/status 2>/dev/null | awk '{print $2}')" ;
    ps -o rss= -p $(pgrep -f mesh_build_memprofile | head -1) 2>/dev/null ;
    sleep 2 ; done ) > cluster/results/wp0-rss-${SLURM_JOB_ID}.log 2>&1 &
SMI_PID=$!
CASES="6:9,7:9" python tests/mesh_build_memprofile.py
RC=$?
kill $SMI_PID 2>/dev/null
echo "WP0_RC=$RC"
