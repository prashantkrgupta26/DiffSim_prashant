#!/bin/bash
#SBATCH --job-name=nccl-cg-cap-gh200
#SBATCH --account=mech-ai
#SBATCH --partition=nova-arm
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/results/%x-%j.out
# NCCL-CG SINGLE-GPU CAPACITY on the Nova GH200 (nova-arm, aarch64, 1 GPU).
# GPU-RESIDENT path: --assembler device (warp GPU K_p assembly, kills the ~1000s
# host assemble_csr wall) + on-device torch.sparse SpMV. This is what pushes a
# single GH200 past L7 toward L8/L9 (135M nodes) on its 480GB coherent memory.
# jacobi precond (AMGX skipped on ARM). world_size=1.
#
# ARM: use bootstrap_arm.sh + .venv-nova-arm + the RPM python3.11 lib path.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "== host $(hostname)  $(uname -m)  $(date)"
RPM_DIR=/work/mech-ai/baskarg/python311-arm
export LD_LIBRARY_PATH="$RPM_DIR/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash cluster/bootstrap_arm.sh
PY=.venv-nova-arm/bin
mkdir -p cluster/results
$PY/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
# Does warp see the GH200 GPU? (device assembler needs warp CUDA; else it falls
# back to the host scipy path and L8/L9 will hit the assembly wall.)
$PY/python -c "import warp as wp; wp.init(); print('warp cuda devices:', wp.get_cuda_device_count())" 2>&1 | grep -iE "cuda devices|error 100" | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "### GH200 single-GPU capacity, GPU-resident (device assembler + torch.sparse SpMV) ###"
# --backend nccl so the driver selects device=cuda (gloo => device=cpu, which
# defeats the GPU-resident path). world_size=1 NCCL is trivial (no real comm).
for L in 5 6 7 8 9; do
  nodes=$(python3 -c "print((2**$L+1)**3)" 2>/dev/null)
  LOG=cluster/results/gh200_L${L}_${SLURM_JOB_ID}.log
  echo "===== L=$L  (nodes=$nodes) -> $LOG ====="
  timeout 3200 $PY/torchrun --standalone --nproc_per_node=1 \
    scripts/nccl_cg_proof.py --stage real --level $L --backend nccl \
    --precond jacobi --rhs random --assembler device --no-verify > "$LOG" 2>&1
  rc=$?
  grep -E "SCALE|PCG:" "$LOG" | tail -2
  if [ $rc -ne 0 ]; then
    echo "  L=$L exited rc=$rc — tail of error:"
    grep -iE "Error|Traceback|CUDA out of memory|Warp|nvrtc|assert|RuntimeError|killed" "$LOG" | tail -8
  fi
done

echo "== nccl-cg gh200 capacity (GPU-resident) done  $(date) =="
