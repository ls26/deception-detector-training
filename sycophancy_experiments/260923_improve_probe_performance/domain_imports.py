"""The single place that reaches into `260819_domain_baseline` for its helpers.

That folder's name starts with a digit, so it cannot be imported as a package --
it has to go on `sys.path`. Isolating that here keeps the hack auditable and
keeps step2 free of import-order surprises.

Re-exported, all used verbatim (never reimplemented) so that this experiment's
answer extraction and follow-up prompt reconstruction stay byte-identical to
what the original rollouts were generated and scored with:

  from score_completions.py   PATTERN_ANSWER_IS, PATTERN_ANSWER_COLON,
                              PATTERN_MARKED_LETTER, FALLBACK_STAGE, extract_answer
  from generate_followups.py  build_followup_suffix, build_followup_prompt
"""

from __future__ import annotations

import sys
from pathlib import Path

_DOMAIN_DIR = Path(__file__).parent.parent / "260819_domain_baseline"

if str(_DOMAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_DOMAIN_DIR))

from score_completions import (  # noqa: E402
    FALLBACK_STAGE,
    PATTERN_ANSWER_COLON,
    PATTERN_ANSWER_IS,
    PATTERN_MARKED_LETTER,
    extract_answer,
)
from generate_followups import build_followup_prompt, build_followup_suffix  # noqa: E402

__all__ = [
    "FALLBACK_STAGE",
    "PATTERN_ANSWER_COLON",
    "PATTERN_ANSWER_IS",
    "PATTERN_MARKED_LETTER",
    "extract_answer",
    "build_followup_prompt",
    "build_followup_suffix",
]
