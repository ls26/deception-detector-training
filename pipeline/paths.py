"""Shared workspace path constants for the multi-round probe + GRPO pipeline.

`pipeline/` and `experiments/` are sibling directories directly under
`/workspace`, so `WORKSPACE` resolves the same way regardless of which
`pipeline/` module computes it. `EXPERIMENTS` always points at
`/workspace/experiments` (configs, per-trial state/outputs), never at this
package's own directory.
"""

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
EXPERIMENTS = WORKSPACE / "experiments"
