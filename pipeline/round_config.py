from dataclasses import dataclass
from pathlib import Path

from pipeline.paths import EXPERIMENTS, WORKSPACE


@dataclass
class RoundConfig:
    """Derived paths and ids for one experiments pipeline round."""

    trial: int
    round: int
    previous_run_id: str
    variant: str = ""
    grpo_run_id: str | None = None
    deceptive_definition: str = "pass_only_shown"

    @property
    def workspace(self) -> Path:
        return WORKSPACE

    @property
    def experiments(self) -> Path:
        return EXPERIMENTS

    @property
    def run_key(self) -> str:
        if self.variant:
            return f"trial{self.trial}_{self.variant}_round{self.round}"
        return f"trial{self.trial}_round{self.round}"

    @property
    def wandb_group(self) -> str:
        """W&B group and on-disk nest under run_results/: trial{N}/{variant|original}."""
        return f"trial{self.trial}/{self.variant or 'original'}"

    @property
    def producer_round(self) -> int:
        return self.round - 1

    @property
    def producer_key(self) -> str:
        # Deliberately excludes `variant` -- build_dataset.py resolves on-policy
        # source paths as buffer_dir / f"trial{trial}_round{k}_train_generations.jsonl",
        # and this key must match that naming for files written into a variant's buffer.
        return f"trial{self.trial}_round{self.producer_round}"

    @property
    def checkpoint_final(self) -> Path:
        return resolve_checkpoint_final(self.workspace, self.previous_run_id, trial=self.trial)

    @property
    def adapter_path(self) -> Path:
        lora = self.checkpoint_final / "lora"
        return lora if lora.is_dir() else self.checkpoint_final

    @property
    def trial_dir(self) -> Path:
        return self.experiments / "trials" / f"trial{self.trial}"

    @property
    def buffer(self) -> Path:
        if self.variant:
            return self.trial_dir / f"buffer_{self.variant}"
        return self.trial_dir / "buffer"

    @property
    def round_dir(self) -> Path:
        rounds_folder = f"rounds_{self.variant}" if self.variant else "rounds"
        return self.trial_dir / rounds_folder / f"round{self.round}"

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


_NON_RUN_GROUP_DIRS = frozenset(
    {
        "hf_checkpoint_evals",
    }
)


def resolve_checkpoint_final(
    workspace: Path,
    run_id: str,
    *,
    trial: int | None = None,
) -> Path:
    """Locate ``checkpoints/final`` for a GRPO run id.

    Search order (shallow globs only — avoid recursive walks on network FS):
    1. ``run_results/trial{N}/{variant}/*_{run_id}/checkpoints/final``
    2. ``run_results/trial*/{variant}/*_{run_id}/checkpoints/final`` (cross-trial)
    3. ``run_results/{experiment_group}/*_{run_id}/checkpoints/final``
    4. Legacy ``run_results/full_grpo_runs/trigger_none_model_{run_id}_layers_32/checkpoints/final``
    """
    runs_base = workspace / "results/run_results"

    def _first_existing(paths) -> Path | None:
        for path in paths:
            if path.is_dir():
                return path
        return None

    if trial is not None:
        found = _first_existing(
            runs_base.glob(f"trial{trial}/*/*_{run_id}/checkpoints/final")
        )
        if found is not None:
            return found

    found = _first_existing(runs_base.glob(f"trial*/*/*_{run_id}/checkpoints/final"))
    if found is not None:
        return found

    if runs_base.is_dir():
        for group_dir in runs_base.iterdir():
            if not group_dir.is_dir():
                continue
            if group_dir.name.startswith("trial") or group_dir.name in _NON_RUN_GROUP_DIRS:
                continue
            if group_dir.name.startswith("base_results_and_detectors"):
                continue
            found = _first_existing(group_dir.glob(f"*_{run_id}/checkpoints/final"))
            if found is not None:
                return found

    legacy = (
        runs_base
        / "full_grpo_runs"
        / f"trigger_none_model_{run_id}_layers_32"
        / "checkpoints"
        / "final"
    )
    if legacy.is_dir():
        return legacy

    # Prefer a new-layout miss path for clearer errors after migration.
    if trial is not None:
        return (
            runs_base
            / f"trial{trial}"
            / "original"
            / f"trial{trial}_round1_{run_id}"
            / "checkpoints"
            / "final"
        )
    return legacy
