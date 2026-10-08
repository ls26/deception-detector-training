#!/usr/bin/env bash
# Start one tmux session per probe running cv_sweep.py for one arm.
#   cv_launch.sh ARM [ARMS_DIR]
# Logs: logs/cv_<arm>_<probe>_$(hostname).log, each ending in "EXIT <code>".
# CV_SWEEP_ARGS (env) is passed through to cv_sweep.py, for the fixture test
# only (e.g. CV_SWEEP_ARGS="--steps 50 --lrs 0.01"); the real run leaves it unset.
# Five processes with OMP_NUM_THREADS=8 each keep the single-process ~7 steps/s.
set -euo pipefail
ARM="${1:?usage: cv_launch.sh ARM [ARMS_DIR]}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS_DIR="${2:-$EXP_DIR/arms}"
PYTHON=/workspace/obfuscation-atlas/.venv/bin/python
mkdir -p "$EXP_DIR/logs"

PROBES=$("$PYTHON" -c "
import sys; sys.path.insert(0, '$EXP_DIR')
from pathlib import Path
from probe_config import arm_config
print(' '.join(arm_config('$ARM', Path('$ARMS_DIR').resolve()).probe_names()))
")

for PROBE in $PROBES; do
  SESSION="cv_${ARM}_${PROBE}"
  LOG="$EXP_DIR/logs/cv_${ARM}_${PROBE}_$(hostname).log"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session $SESSION already running, skipping"
    continue
  fi
  tmux new-session -d -s "$SESSION" \
    "source /workspace/atlas_setup.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; cd '$EXP_DIR'; \
     '$PYTHON' cv_sweep.py --arm '$ARM' --arms-dir '$ARMS_DIR' --probes '$PROBE' ${CV_SWEEP_ARGS:-} > '$LOG' 2>&1; echo \"EXIT \$?\" >> '$LOG'"
  echo "started $SESSION -> $LOG"
done
