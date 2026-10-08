"""The single place that reaches into the earlier experiment folders for helpers.

Those folder names start with a digit, so they cannot be imported as packages;
they go on `sys.path` here, once, in a fixed order. Everything is re-exported
verbatim so this experiment's answer parsing, cue rendering, follow-up prompt
construction, P(Yes) reading and labelling stay byte-identical to what the
reference data was generated and scored with.

  260819 build_domain_set     CUE_TEMPLATES, cue_text_for, format_question, INSTRUCTION
  260819 score_completions    extract_answer, to_content, to_letter, the three regexes
  260823 generate_followups   the removal wordings, followup_text, build_followup_suffix,
                              build_followup_prompt, resolve_targets, position_stats,
                              answer_position, check_logprob_invariance, YES_BROAD, NO_BROAD
  260823 score_followups      draw_wordings, ARM_OF
  260824 rescore              turn1_label, turn2_label

`generate_followups` imports vLLM at module scope; CPU-only scripts should
import `domain_cpu` names only (see the two blocks below).
"""

from __future__ import annotations

import sys

from spec import DOMAIN_DIR, FOLLOWUP_DIR, RESCORE_DIR

for _dir in (DOMAIN_DIR, FOLLOWUP_DIR, RESCORE_DIR):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# --- CPU-only (no vLLM) -----------------------------------------------------
from build_domain_set import CUE_TEMPLATES, INSTRUCTION, cue_text_for, format_question  # noqa: E402
from score_completions import (  # noqa: E402
    FALLBACK_STAGE,
    PATTERN_ANSWER_COLON,
    PATTERN_ANSWER_IS,
    PATTERN_MARKED_LETTER,
    extract_answer,
    to_content,
    to_letter,
)
from score_followups import ARM_OF, draw_wordings  # noqa: E402
from rescore import turn1_label, turn2_label  # noqa: E402

__all__ = [
    "CUE_TEMPLATES",
    "INSTRUCTION",
    "cue_text_for",
    "format_question",
    "FALLBACK_STAGE",
    "PATTERN_ANSWER_COLON",
    "PATTERN_ANSWER_IS",
    "PATTERN_MARKED_LETTER",
    "extract_answer",
    "to_content",
    "to_letter",
    "ARM_OF",
    "draw_wordings",
    "turn1_label",
    "turn2_label",
]


def import_followup_generation():
    """The 260823 generator's helpers (imports vLLM). Call from GPU scripts only."""
    import generate_followups as g

    return g
