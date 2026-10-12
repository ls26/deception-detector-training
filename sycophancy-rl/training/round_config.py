"""Where one round of one trial lineage reads and writes.

GRPO run directories are `results/run_results/<group>/<run_key>_<wandb run id>/`, where the
group is `baselines` for the baseline sweep and `trial<N>/<variant>` (`original` for round 1)
for trial rounds. Everything else lives under `experiments/trials/trial<N>/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from training.common import EXPERIMENTS, RUNS_BASE


def run_dir_for_id(run_id: str, *, group: str | None = None, run_key: str | None = None) -> Path:
    """The run directory of a W&B run id. With `group` and `run_key` the path is known
    exactly (and is returned whether or not it exists yet); otherwise the one directory
    named `*_<run_id>` under results/run_results is found by search."""
    if group is not None and run_key is not None:
        return RUNS_BASE / group / f"{run_key}_{run_id}"
    matches = [
        path
        for pattern in (f"*/*_{run_id}", f"*/*/*_{run_id}")
        for path in RUNS_BASE.glob(pattern)
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one run directory for run id {run_id} under {RUNS_BASE}, found {matches}")
    return matches[0]


def checkpoint_final_for_id(run_id: str) -> Path:
    return run_dir_for_id(run_id) / "checkpoints" / "final"


def adapter_path_for(checkpoint_final: Path) -> Path:
    """The LoRA adapter of a final checkpoint (`final/lora/` when saved separately)."""
    lora = checkpoint_final / "lora"
    return lora if lora.is_dir() else checkpoint_final


@dataclass
class RoundConfig:
    """Paths and ids for round `round` of one lineage, whose policy at the start of the
    round is the GRPO run `previous_run_id` (round `round - 1`'s final checkpoint)."""

    trial: int
    round: int
    previous_run_id: str
    variant: str

    @property
    def run_key(self) -> str:
        return f"trial{self.trial}_{self.variant}_round{self.round}"

    @property
    def wandb_group(self) -> str:
        """W&B group and on-disk nest under results/run_results."""
        return f"trial{self.trial}/{self.variant}"

    @property
    def producer_round(self) -> int:
        return self.round - 1

    @property
    def producer_key(self) -> str:
        """Names the on-policy files of the policy that starts this round. Round 1 is shared
        by every variant, so the variant is deliberately not part of the name."""
        return f"trial{self.trial}_round{self.producer_round}"

    @property
    def checkpoint_final(self) -> Path:
        return checkpoint_final_for_id(self.previous_run_id)

    @property
    def adapter_path(self) -> Path:
        return adapter_path_for(self.checkpoint_final)

    @property
    def trial_dir(self) -> Path:
        return EXPERIMENTS / "trials" / f"trial{self.trial}"

    @property
    def buffer(self) -> Path:
        return self.trial_dir / f"buffer_{self.variant}"

    @property
    def round_dir(self) -> Path:
        return self.trial_dir / f"rounds_{self.variant}" / f"round{self.round}"

    @property
    def eval_jsonl(self) -> Path:
        return self.buffer / f"{self.producer_key}_eval_generations.jsonl"

    @property
    def probe_train_dataset(self) -> Path:
        return self.round_dir / "probe_train_dataset.json"

    @property
    def probe_weights(self) -> Path:
        return self.round_dir / "probe_weights"

    @property
    def probe_activations(self) -> Path:
        return self.round_dir / "probe_activations"

    @property
    def probe_metrics(self) -> Path:
        return self.round_dir / "probe_metrics.json"

    @property
    def run_metrics(self) -> Path:
        return self.round_dir / f"run_metrics_{self.run_key}.json"
