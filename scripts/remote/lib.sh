#!/usr/bin/env bash
# scripts/remote/lib.sh — pure helpers, no ssh side effects. Unit-tested.

rlog() { printf '\033[1;34m[remote]\033[0m %s\n' "$*" >&2; }
rerr() { printf '\033[1;31m[remote:err]\033[0m %s\n' "$*" >&2; }

# remote_log_path <logdir> <run_id> [tag]
remote_log_path() {
  local logdir="$1" run_id="$2" tag="${3:-run}"
  printf '%s/%s-%s.log' "$logdir" "$tag" "$run_id"
}

# remote_tmux_name <run_id>  (sanitize to [A-Za-z0-9_-])
remote_tmux_name() {
  printf 'diffsim-%s' "$(printf '%s' "$1" | tr -c 'A-Za-z0-9_-' '-')"
}
