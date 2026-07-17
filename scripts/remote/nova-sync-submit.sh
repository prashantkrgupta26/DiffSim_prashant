#!/usr/bin/env bash
# STUB — wired on next real Nova kit submission (see design spec §Decisions).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cat >&2 <<'MSG'
[nova] not yet wired.
Intended flow (see docs/dev/2026-07-15-mac-brainstorm-handoff.md §4):
  1. Mac: ensure origin/master current (git push).
  2. ssh nova (reuse Duo ControlMaster): git fetch && git reset --hard origin/master
  3. fill FIXME_PARTITION from: sinfo -o "%P %G %N"
  4. sbatch cluster/wodo_campaign/<kit>.sbatch   -> return job id
MSG
exit 64
