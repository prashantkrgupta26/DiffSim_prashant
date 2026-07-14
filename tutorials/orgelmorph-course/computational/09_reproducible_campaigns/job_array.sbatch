#!/bin/bash
# OrgElMorph course - Computational C9: SLURM JOB-ARRAY campaign sketch.
#
# The local demo (campaign.py) runs the whole tiny sweep in one process on one
# GPU.  At research scale each sweep cell is one GPU-hours-to-days job, so the
# campaign becomes a SLURM job ARRAY: one array task per cell, each task
# resolving its (chi, seed) from $SLURM_ARRAY_TASK_ID.  This mirrors the real
# cluster kits in cluster/wodo_campaign/ (a100_wodo.sbatch, a100_s3d_pilot
# .sbatch): the same #SBATCH header shape, the same cluster/bootstrap.sh +
# .venv-nova python, the same "stream JSON to a per-task log so even a capped
# task is a complete measurement" discipline.  FIXME markers are the values you
# fill from `sinfo -o "%P %G %N"` on your cluster (as those kits document).
#
# Submit:   sbatch --array=0-7 job_array.sbatch
#           (8 cells = 4 chi x 2 seeds; adjust --array to match your manifest)
# Restart:  re-submitting is SAFE -- campaign.py skips already-completed cells
#           (checkpoint/restart), so a preempted array requeues only the
#           unfinished tasks.  Aggregate once all tasks land with aggregate.py.
#
#SBATCH --job-name=c9-campaign
#SBATCH --partition=FIXME_PARTITION      # fill from: sinfo -o "%P %G %N"
#SBATCH --gres=gpu:1                     # FIXME: gpu:a100:1 if the GRES is typed
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4                    # 8 cells, at most 4 running at once
#SBATCH --output=cluster/results/%x-%A_%a.out   # %A=array job, %a=task index

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash cluster/bootstrap.sh                # same bootstrap the wodo kits use
PY=.venv-nova/bin/python
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"

CH=tutorials/orgelmorph-course/computational/09_reproducible_campaigns
OUT=cluster/results/c9-$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT"

echo "== C9 campaign task $SLURM_ARRAY_TASK_ID $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Each array task runs exactly ONE cell of the manifest, selected by the task
# index (campaign.py --only picks the k-th enumerated run).  The immutable
# resolved manifest is shared; the per-cell output dir is keyed by run_id, so
# tasks never collide and a rerun is idempotent.
$PY "$CH/campaign.py" \
    --manifest "$CH/manifest.yaml" \
    --output   "$OUT/demo" \
    --only     "$SLURM_ARRAY_TASK_ID" \
    > "$OUT/task-$SLURM_ARRAY_TASK_ID.log" 2>&1 \
    || echo "TASK $SLURM_ARRAY_TASK_ID CRASHED-OR-CAPPED (surfaced, not dropped)"

grep -E '\[(OK|FAIL)\]' "$OUT/task-$SLURM_ARRAY_TASK_ID.log" | tail -3
echo "== task $SLURM_ARRAY_TASK_ID done $(date) =="

# After ALL tasks finish, aggregate once (a dependent single-task job):
#   sbatch --dependency=afterany:$SLURM_ARRAY_JOB_ID <<'EOF'
#   #!/bin/bash
#   #SBATCH --partition=FIXME_PARTITION
#   $PY $CH/campaign.py --manifest $CH/manifest.yaml --output $OUT/demo
#   EOF
# The aggregate step re-reads every cell's status.json (completed OR failed),
# so any task that crashed is SURFACED in summary.json, never silently dropped.
