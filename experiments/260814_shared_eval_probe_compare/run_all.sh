#!/usr/bin/env bash
# Full shared-eval 3-probe comparison: build eval sets, train probe C, score, summarize.
# Extra args are forwarded to every stage (e.g. --trial 6 --variant original, --force).
#
# Launch under tmux (see CLAUDE.md), single GPU:
#   tmux new-session -d -s shared_eval \
#     'bash /workspace/experiments/260814_shared_eval_probe_compare/run_all.sh'
set -euo pipefail
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace
export PYTHONUNBUFFERED=1

DIR=/workspace/experiments/260814_shared_eval_probe_compare
LOG="$DIR/run_$(hostname).log"

{
  echo "=== build_shared_eval.py $* ==="
  python -u "$DIR/build_shared_eval.py" "$@"
  echo "=== train_probe_c.py $* ==="
  python -u "$DIR/train_probe_c.py" "$@"
  echo "=== run_compare.py $* ==="
  python -u "$DIR/run_compare.py" "$@"
  echo "=== summarize.py ==="
  python -u "$DIR/summarize.py"
} >> "$LOG" 2>&1

echo "Done. Log: $LOG"
