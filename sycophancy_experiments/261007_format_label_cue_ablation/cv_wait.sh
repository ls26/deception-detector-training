#!/usr/bin/env bash
# Block until every cv_launch.sh log of one arm ends in an EXIT line; exit
# non-zero if any of them exited non-zero.
#   cv_wait.sh ARM [ARMS_DIR]
set -euo pipefail
ARM="${1:?usage: cv_wait.sh ARM [ARMS_DIR]}"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS_DIR="${2:-$EXP_DIR/arms}"
PYTHON=/workspace/obfuscation-atlas/.venv/bin/python

PROBES=$("$PYTHON" -c "
import sys; sys.path.insert(0, '$EXP_DIR')
from pathlib import Path
from probe_config import arm_config
print(' '.join(arm_config('$ARM', Path('$ARMS_DIR').resolve()).probe_names()))
")

STATUS=0
for PROBE in $PROBES; do
  LOG="$EXP_DIR/logs/cv_${ARM}_${PROBE}_$(hostname).log"
  until [ -f "$LOG" ] && grep -q '^EXIT' "$LOG"; do sleep 60; done
  CODE=$(grep '^EXIT' "$LOG" | tail -1 | awk '{print $2}')
  echo "$PROBE: EXIT $CODE ($LOG)"
  if [ "$CODE" != "0" ]; then STATUS=1; fi
done
exit $STATUS
