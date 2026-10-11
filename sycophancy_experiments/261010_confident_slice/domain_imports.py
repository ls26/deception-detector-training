"""The single place this folder imports 260819 and 260824 helpers.

Those folder names start with a digit, so they go on sys.path here, once.

  260819 score_completions        extract_answer, to_content
  260824 rescore                  turn1_label, turn2_label
  260819 generate_completions     behind import_generation(): it imports vLLM

Only generate_nocue_train.py and slice_labels.py import this module. The probe
pipeline scripts import slice_pipeline.py instead and must never load this
module: 261007's step 2 imports 261007's own `domain_imports` by that name.
"""

from __future__ import annotations

import sys

from slice_spec import BASELINE_DIR, RESCORE_DIR

for _dir in (BASELINE_DIR, RESCORE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from rescore import turn1_label, turn2_label  # noqa: E402
from score_completions import extract_answer, to_content  # noqa: E402


def import_generation():
    """260819's generator module (imports vLLM at module scope). Generation script only."""
    import generate_completions

    return generate_completions
