#!/usr/bin/env bash
# scripts/remote/nova-poll.sh — report the state of a Nova SLURM job over the
# reused Duo ControlMaster socket: live squeue, sacct accounting, and a tail of
# the job log. Optionally waits until the job leaves the queue.
#
# Usage:
#   nova-poll.sh <jobid> [--tail N] [--wait] [--interval SEC]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
source "$HERE/lib.sh"

JOBID="${1:-}"
if [ -z "$JOBID" ] || ! [[ "$JOBID" =~ ^[0-9]+$ ]]; then
  rerr "usage: nova-poll.sh <jobid> [--tail N] [--wait] [--interval SEC]"
  exit 64
fi
shift

TAIL_N=40
WAIT=0
INTERVAL=15
while [ $# -gt 0 ]; do
  case "$1" in
    --tail) TAIL_N="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    *) rerr "unknown arg: $1"; exit 64 ;;
  esac
done

# The sbatch --output pattern is cluster/results/%x-%j.out; %x expands per job.
# Resolve the actual log by globbing on the job id.
LOGGLOB="$NOVA_REPO_ABS/cluster/results/*-$JOBID.out"

poll_once() {
  rlog "=== Nova job $JOBID ==="
  ssh "$NOVA_HOST" bash -s "$JOBID" "$TAIL_N" "$LOGGLOB" <<'EOF'
set -uo pipefail
JOBID="$1"; TAIL_N="$2"; LOGGLOB="$3"
echo "--- squeue (live) ---"
squeue -j "$JOBID" 2>/dev/null || true
echo "--- sacct (accounting) ---"
sacct -j "$JOBID" --format=JobID,State,ExitCode,Elapsed,NodeList%20 2>/dev/null || true
echo "--- log tail ($TAIL_N) ---"
LOG=$(ls -1t $LOGGLOB 2>/dev/null | head -1 || true)
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  echo "(log: $LOG)"
  tail -n "$TAIL_N" "$LOG"
else
  echo "(no log yet at $LOGGLOB)"
fi
EOF
}

job_active() {
  # returns 0 while the job is still PENDING/RUNNING/etc in the queue
  local st
  st=$(ssh "$NOVA_HOST" "squeue -j '$JOBID' --noheader -o '%T' 2>/dev/null" || true)
  [ -n "$st" ]
}

if [ "$WAIT" -eq 1 ]; then
  while job_active; do
    poll_once
    rlog "job still in queue; sleeping ${INTERVAL}s"
    sleep "$INTERVAL"
  done
  rlog "job left the queue; final report:"
fi
poll_once
