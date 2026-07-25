#!/bin/bash
#SBATCH --job-name=nccl-cg-scale-a100
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=cluster/results/%x-%j.out
# NCCL-CG multi-GPU STRONG SCALING on a 4-GPU A100 node (real NVLink).
# Run from /work/mech-ai/baskarg/DiffSim (HOME is over-quota).
#
# Real NVLink node => NCCL runs with DEFAULTS. Do NOT set the gpubox/WSL2
# workarounds (NCCL_P2P_DISABLE / SHM_DISABLE / CUMEM_ENABLE=0) here — they
# would cripple NVLink.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "== host $(hostname)  $(date)"
bash cluster/bootstrap.sh
PY=.venv-nova/bin
mkdir -p cluster/results
$PY/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "### STRONG SCALING: fixed level, vary GPU count (jacobi; AMGX not built on Nova) ###"
for L in 5 6 7; do
  for N in 1 2 4; do
    echo "===== L=$L  N=$N ====="
    timeout 900 $PY/torchrun --standalone --nproc_per_node=$N \
      scripts/nccl_cg_proof.py --stage real --level $L --backend nccl \
      --precond jacobi --rhs random --no-verify 2>&1 \
      | grep -E "SCALE|PCG:|CUDA out of memory|Error|Traceback" | tail -4 \
      || echo "  (L=$L N=$N failed/OOM)"
  done
done

echo "### VERIFIED correctness point (L5, N=2, with the serial reference) ###"
timeout 900 $PY/torchrun --standalone --nproc_per_node=2 \
  scripts/nccl_cg_proof.py --stage real --level 5 --backend nccl \
  --precond jacobi --rhs random 2>&1 | grep -E "PASS|FAIL|rel_err|iters" | tail -3

echo "== nccl-cg a100 scaling done  $(date) =="
