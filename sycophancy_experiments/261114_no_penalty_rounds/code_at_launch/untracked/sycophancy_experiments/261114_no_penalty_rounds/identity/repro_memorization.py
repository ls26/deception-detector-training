"""Identity check of this folder's memorization.py (imported unchanged), with the no-penalty condition pointed at 261107's
frozen runs (the only patch), so every input exists before the queue runs.

Usage (the Start check, and after a fix to memorization.py): $PY repro_memorization.py SCRATCH -> SCRATCH/comparison.json
and SCRATCH/comparison.md, compared byte for byte with memorization_expected/ (written at setup on 2026-11-14).
"""

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy_experiments/261114_no_penalty_rounds")
import memorization  # noqa: E402

memorization.CONDITION_FOLDERS["no penalty"] = memorization.CONDITION_FOLDERS["frozen"]
memorization.ROUND1_RUN_KEY["no penalty"] = memorization.ROUND1_RUN_KEY["frozen"]
memorization.LATER_RUN_KEY["no penalty"] = memorization.LATER_RUN_KEY["frozen"]
memorization.OUT_DIR = Path(sys.argv[1])
memorization.main()
