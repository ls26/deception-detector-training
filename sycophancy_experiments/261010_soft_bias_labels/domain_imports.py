"""The single place this folder reaches into earlier experiment folders.

Those folder names start with a digit, so they go on sys.path here, once.
261007's own domain_imports.py is NOT imported: it shares this module's name.

  260819 score_completions        extract_answer, to_content
  260824 rescore                  turn1_label, turn2_label
  261007 label_common             majority_vote (strict; a tie gives None)
  261007 spec                     REPRODUCTION_RECIPE (fit B's lr / steps / band),
                                  DOMAIN_DIR, FOLLOWUP_DIR, RESCORE_DIR

Two groups are behind functions, so each process loads only what it needs:
  import_probe_training()  261007's probe_config and step3_train_eval_probes (they load
                           the probe-training library). Training and scoring scripts only.
  import_generation()      260819's generate_completions (imports vLLM). The generation
                           script only; with the training library loaded in the same
                           process, vLLM's forked engine failed to initialize CUDA.
"""

from __future__ import annotations

import sys

from soft_spec import BASELINE_DIR, REF_DIR

if str(REF_DIR) not in sys.path:
    sys.path.insert(0, str(REF_DIR))
import spec as ref_spec  # noqa: E402  (261007's spec.py)

for _dir in (ref_spec.DOMAIN_DIR, ref_spec.FOLLOWUP_DIR, ref_spec.RESCORE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
assert ref_spec.DOMAIN_DIR == BASELINE_DIR, (ref_spec.DOMAIN_DIR, BASELINE_DIR)

from label_common import majority_vote  # noqa: E402
from rescore import turn1_label, turn2_label  # noqa: E402
from score_completions import extract_answer, to_content  # noqa: E402

REPRODUCTION_RECIPE = ref_spec.REPRODUCTION_RECIPE


def import_probe_training():
    """(probe_config, step3_train_eval_probes) of 261007. Training and scoring scripts only."""
    import probe_config
    import step3_train_eval_probes

    return probe_config, step3_train_eval_probes


def import_generation():
    """260819's generator module (imports vLLM at module scope). GPU script only."""
    import generate_completions

    return generate_completions
