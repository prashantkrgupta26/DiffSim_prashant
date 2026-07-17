#!/usr/bin/env bash
# fetch-artifacts.sh — pull large run artifacts (npz/renders/checkpoints) from a
# remote compute host back to the Mac. These are excluded from both rsync-out and
# git, so this is their only return path. Opt-in by subdir; NEVER --delete (local
# files are never removed). Serves gpubox now and Nova once NOVA_REPO_ABS is set.
#
# Usage: fetch-artifacts.sh [--host gpubox|nova] [--dry-run] <remote-subdir> [dest]
#   <remote-subdir>  path relative to the host's repo root (e.g. results/negi3d)
#   dest             local destination (default: $REPO_ROOT/<remote-subdir>)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"; source "$HERE/lib.sh"

host="$GPUBOX_HOST"; remote_root="$GPUBOX_REPO_ABS"; dry=""
while [ $# -gt 0 ]; do
  case "$1" in
    --host)
      shift
      case "${1:-}" in
        gpubox|"$GPUBOX_HOST") host="$GPUBOX_HOST"; remote_root="$GPUBOX_REPO_ABS" ;;
        nova|"$NOVA_HOST")     host="$NOVA_HOST";   remote_root="$NOVA_REPO_ABS" ;;
        *) rerr "unknown --host: ${1:-}"; exit 2 ;;
      esac
      shift ;;
    --dry-run) dry="--dry-run"; shift ;;
    --) shift; break ;;
    -*) rerr "unknown flag: $1"; exit 2 ;;
    *) break ;;
  esac
done

sub="${1:?usage: fetch-artifacts.sh [--host gpubox|nova] [--dry-run] <remote-subdir> [dest]}"
dest="${2:-$REPO_ROOT/$sub}"

if [ -z "$remote_root" ]; then
  rerr "no repo path configured for host '$host' (set NOVA_REPO_ABS in config.sh to use --host nova)."
  exit 2
fi

mkdir -p "$dest"
rlog "pull $host:$remote_root/$sub/ -> $dest/ ${dry:+(dry-run)}"
# -a preserves; no --delete so we never remove local artifacts. Trailing slashes
# copy the subdir's CONTENTS into dest.
rsync -az $dry "$host:$remote_root/$sub"/ "$dest"/
rlog "artifacts fetched -> $dest"
