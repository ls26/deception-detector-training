"""The single place this folder imports 260819 / 260823 / 260824 helpers (except build_dataset.py,
which imports 261007's prompts module and with it 261007's own import hub).

Those folder names start with a digit, so they go on sys.path here, once, in 261007's order (260824
first, then 260823, then 260819: 260823's generate_followups.py must shadow 260819's file of the same
name).

  260819 score_completions        to_content
  260823 score_followups          draw_wordings, ARM_OF
  260823 generate_followups       followup_text, build_followup_suffix, build_followup_prompt
                                  (behind import_followup_generation(): it imports vLLM)
  260824 rescore                  turn1_label, turn2_label
"""

from __future__ import annotations

import sys

from nocot_spec import BASELINE_DIR, FOLLOWUP_DIR, RESCORE_DIR

for _dir in (BASELINE_DIR, FOLLOWUP_DIR, RESCORE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

from rescore import turn1_label, turn2_label  # noqa: E402
from score_completions import to_content  # noqa: E402
from score_followups import ARM_OF, draw_wordings  # noqa: E402

for _module, _dir in (("rescore", RESCORE_DIR), ("score_completions", BASELINE_DIR), ("score_followups", FOLLOWUP_DIR)):
    assert sys.modules[_module].__file__ == str(_dir / f"{_module}.py"), f"{_module} resolved to {sys.modules[_module].__file__}"


def import_followup_generation():
    """260823's follow-up generator module (imports vLLM at module scope)."""
    import generate_followups

    assert generate_followups.__file__ == str(FOLLOWUP_DIR / "generate_followups.py"), generate_followups.__file__
    return generate_followups


__all__ = [
    "to_content",
    "ARM_OF",
    "draw_wordings",
    "turn1_label",
    "turn2_label",
    "import_followup_generation",
]
