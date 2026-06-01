#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="s2s_pc6_t2m_ridge_jfm"
RUNNER_SCRIPT="$REPO_ROOT/scripts/run_s2s_pc6_t2m_top20_regions_ridge_target_sensitivity_jfm_interactive.sh"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "attach with: tmux attach-session -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
tmux send-keys -t "$SESSION_NAME" "cd \"$REPO_ROOT\"" C-m
tmux send-keys -t "$SESSION_NAME" "salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m2637 bash \"$RUNNER_SCRIPT\"" C-m

echo "Started tmux session on login node: $SESSION_NAME"
echo "Attach with: tmux attach-session -t $SESSION_NAME"
