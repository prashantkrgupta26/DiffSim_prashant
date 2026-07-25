#!/bin/bash
#SBATCH --job-name=nccl-cg-cap-gh200
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/results/%x-%j.out
# NCCL-CG SINGLE-GPU CAPACITY on the Nova GH200 (nova-arm, aarch64, 1 GPU).
# The GH200 partition has ONE node with ONE GPU -> no multi-GPU scaling here;
# this measures how large a PPE fits + solve time on the GH200's coherent
# memory (Grace LPDDR + HBM).  bootstrap.sh auto-detects aarch64; AMGX is
# skipped on ARM, so we use --precond jacobi.  world_size=1 (gloo avoids any
# 1-rank NCCL quirk; comm is trivial for a single rank).
#
# NB the host assemble_csr path (scipy) is the build bottleneck at large L
# (~minutes at L7); L8+ needs the device assembler (follow-up).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "== host $(hostname)  $(uname -m)  $(date)"
bash cluster/bootstrap.sh
PY=.venv-nova/bin
mkdir -p cluster/results
$PY/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "### GH200 single-GPU capacity (world_size=1, jacobi) ###"
for L in 5 6 7; do
  echo "===== L=$L  (nodes=(2^L+1)^3) ====="
  timeout 1200 $PY/torchrun --standalone --nproc_per_node=1 \
    scripts/nccl_cg_proof.py --stage real --level $L --backend gloo \
    --precond jacobi --rhs random --no-verify 2>&1 \
    | grep -E "SCALE|PCG:|CUDA out of memory|Error|Traceback" | tail -4 \
    || echo "  (L=$L failed/OOM)"
done

echo "== nccl-cg gh200 capacity done  $(date) =="
