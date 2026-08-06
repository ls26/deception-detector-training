from dataclasses import dataclass
from pathlib import Path


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
        return Path(__file__).resolve().parents[1]

    @property
    def experiments(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def run_key(self) -> str:
        if self.variant:
            return f"trial{self.trial}_{self.variant}_round{self.round}"
        return f"trial{self.trial}_round{self.round}"

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
        return (
            self.workspace
            / f"results/run_results/full_grpo_runs/trigger_none_model_{self.previous_run_id}_layers_32/checkpoints/final"
        )

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
