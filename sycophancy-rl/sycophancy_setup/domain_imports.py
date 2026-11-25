"""The single place this package imports 260819 / 260823 / 260824 helpers (261013's domain_imports.py).

Those folder names start with a digit, so they go on sys.path here, once (260824 first, then 260819).

  260819 score_completions        to_content, to_letter (the answer parser is the trainer's
                                  obfuscation_atlas.tasks.multiple_choice.extract_answer, the one the
                                  reward uses; 260819's copy took the first "answer is (X)", the trainer
                                  takes the last)
  260824 rescore                  turn1_label

The self-report follow-up (260823's wordings and generator, 260824's turn2_label) was dropped on 2026-11-20.
"""

from __future__ import annotations

import sys

from spec import BASELINE_DIR, RESCORE_DIR

for _dir in (BASELINE_DIR, RESCORE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from rescore import turn1_label  # noqa: E402
from score_completions import to_content, to_letter  # noqa: E402

__all__ = [
    "to_content",
    "to_letter",
    "turn1_label",
]
