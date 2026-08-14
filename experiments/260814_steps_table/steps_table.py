# %%
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace")

from pipeline.paths import EXPERIMENTS

# Row label -> variant string per trial (trial 4 "original" is the v2 full-128 branch)
ROW_VARIANTS: dict[str, dict[int, str]] = {
    "Fixed-schedule retraining": {6: "", 4: "v2", 8: ""},
    "Early retraining (probe-based)": {
        6: "early_retraining",
        4: "early_retraining_v2",
        8: "early_retraining",
    },
    "Early retraining (auc-based)": {
        6: "early_retraining_auc_v3",
        4: "early_retraining_auc_v2",
        8: "early_retraining_auc_v3",
    },
}

# Panel order matches experiments/paper_figures/paper_fig_common.py: A = 4, B = 6, C = 8.
TRIAL_COLUMNS: dict[str, int] = {
    "Trial A": 4,
    "Trial B": 6,
    "Trial C": 8,
}

ROUNDS = range(1, 6)

# Rounds whose run_metrics.json is an eval-only rerun and therefore carries no
# grpo_train_step. Keyed by (trial, variant, round) -> (steps, source).
# trial 4 early_retraining_v2 round 5: training run io4ibs0w was manually stopped
# ~step 60/128 and the checkpoint promoted from step_60; the exported metrics come
# from eval-only run 8g8580by. See docs/EXPERIMENT_LOG.md.
STEP_OVERRIDES: dict[tuple[int, str, int], tuple[int, str]] = {
    (4, "early_retraining_v2", 5): (60, "io4ibs0w stopped at step_60"),
}


def rounds_dir(trial: int, variant: str) -> Path:
    folder = f"rounds_{variant}" if variant else "rounds"
    return EXPERIMENTS / "trials" / f"trial{trial}" / folder


def run_metrics_path(trial: int, variant: str, round_num: int) -> Path:
    matches = sorted(glob.glob(str(rounds_dir(trial, variant) / f"round{round_num}" / "run_metrics_*.json")))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"trial{trial} {variant or 'original'} round{round_num}: expected 1 run_metrics file, "
            f"found {len(matches)}: {matches}"
        )
    return Path(matches[0])


def load_steps(trial: int, variant: str, round_num: int) -> tuple[int, bool]:
    """GRPO training steps for one round; second value flags an override."""
    override = STEP_OVERRIDES.get((trial, variant, round_num))
    metrics = json.loads(run_metrics_path(trial, variant, round_num).read_text())["metrics"]
    step = metrics.get("grpo_train_step")
    if step is not None:
        return int(step), False
    if override is None:
        raise ValueError(
            f"trial{trial} {variant or 'original'} round{round_num}: no grpo_train_step in "
            f"{run_metrics_path(trial, variant, round_num)} and no STEP_OVERRIDES entry"
        )
    return override[0], True


# %%
records: list[dict] = []
overridden: list[str] = []
for trial_label, trial in TRIAL_COLUMNS.items():
    for variant_label, variants_by_trial in ROW_VARIANTS.items():
        variant = variants_by_trial[trial]
        row = {"row": f"{trial_label}: {variant_label}"}
        for round_num in ROUNDS:
            steps, is_override = load_steps(trial, variant, round_num)
            row[f"Round {round_num}"] = steps
            if is_override:
                overridden.append(
                    f"{row['row']} Round {round_num} = {steps} "
                    f"({STEP_OVERRIDES[(trial, variant, round_num)][1]})"
                )
        records.append(row)

steps = pd.DataFrame.from_records(records).set_index("row")
steps.index.name = None

print("GRPO training steps per round (max 128; fewer = early-stopped)\n")
print(steps.to_string())
if overridden:
    print("\nFrom EXPERIMENT_LOG rather than run_metrics (eval-only rerun, no grpo_train_step):")
    for note in overridden:
        print(f"  {note}")

steps

# %%
