#!/usr/bin/env bash
# STUB — wired on next real Nova kit submission.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cat >&2 <<'MSG'
[nova] not yet wired.
Intended: squeue -j <jobid> / sacct -j <jobid> + tail the slurm-<jobid>.out log
over the reused Duo ControlMaster socket.
MSG
exit 64
