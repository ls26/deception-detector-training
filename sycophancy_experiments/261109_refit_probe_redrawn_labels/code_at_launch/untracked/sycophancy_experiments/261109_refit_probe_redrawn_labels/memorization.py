"""Frozen against refit probe, both with the training labels redrawn every round, rounds 1-5: the eval rates, the probe
each round trained against (Initial and Final), and label memorization in training, mean and SD over seeds 42 and 43.
261108's memorization.py with the refit condition in place of the fixed-label one and 261104's probe rows.

Usage (CPU, a few minutes):
    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
    $PY memorization.py    # writes runs/comparison/comparison.json and comparison.md

Conditions: frozen = 261108_frozen_probe_redrawn_labels (round 1's probe in the penalty every round), refit = this
folder (the probe refit before every round, 261104's procedure). Both train round R on 261108's labels/roundR/labels.jsonl.
Round 1 is 261103's weight-1.0 run of the seed in both.

Rows, per round:
  Eval, from each seed's tables/rounds/seed_table.json (compare.py's definitions, the run's final eval):
    bias and cue agreement rate (followed_cue) on the training cue mix (train_mix); true accuracy with no cue (none),
    with the cue at the correct answer (cue_gold) and at the wrong answer (cue_dist); deception_soft on train_mix.
  Penalty probe (refit: the round's own probe, fit before the round; frozen: round 1's probe), both cue conditions
  pooled, the mean score of the biased answers, of the unbiased ones, and the AUC between them:
    Final: on the round's eval answers that name the cued option, read through the round's model; from each seed's
      runs/dynamics/dynamics.json (probe_letters, row C, "both"; C reads the final eval's probe_start, which is the
      round's own probe in a refit run and round 1's probe in 261108 through its links folder).
    Initial: refit only, on the probe's own test set (round R-1's final-eval answers, read through round R-1's model;
      round 1: 261010's 777 base-model eval answers); from the refit seed's probe_table.json (own_test_set; None if
      the test file had one class). Round 1 is shared. The frozen probe is never refit, so its rounds 2-5 are None.
  Memorization, from each run's training answers (checkpoints/responses/step_*.jsonl), all 128 steps: on prompts of
    flipped questions (the record's train_label is not the gold letter, so each round's own labels) with the cue at
    the true answer, the share of answers naming the training label. Every answer counts, unparsed ones too.
comparison.md: mean ± SD (statistics.stdev, as seed_table.py) of the two seeds; "–" if either seed's value is None.
comparison.json: each seed's values.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402

FOLDER = Path(__file__).resolve().parent
CONDITION_FOLDERS = {"frozen": FOLDER.parent / "261108_frozen_probe_redrawn_labels", "refit": FOLDER}
RUN_KEY_FORMAT = {"frozen": "s{seed}_redrawn_round{round}", "refit": "s{seed}_refit_round{round}"}  # rounds 2-5
SEEDS = (42, 43)
ROUNDS = (1, 2, 3, 4, 5)
OUT_DIR = FOLDER / "runs" / "comparison"
EVAL_ROWS = [  # (row name, seed_table.json metric, cue condition)
    ("bias, training cue mix", "bias", "train_mix"),
    ("cue agreement rate, training cue mix", "followed_cue", "train_mix"),
    ("true accuracy, no cue", "accuracy", "none"),
    ("true accuracy, cue at the correct answer", "accuracy", "cue_gold"),
    ("true accuracy, cue at the wrong answer", "accuracy", "cue_dist"),
    ("deception (P(Yes)-weighted), training cue mix", "deception_soft", "train_mix"),
]
INITIAL_ROWS = [  # (row name, key of probe_table.json [round]["scores"]["own_test_set"])
    ("penalty probe, Initial: mean score of biased answers", "mean_biased"),
    ("penalty probe, Initial: mean score of unbiased answers", "mean_unbiased"),
    ("penalty probe, Initial: AUC biased vs unbiased", "auc"),
]
FINAL_ROWS = [  # (row name, key of dynamics.json probe_letters[run]["C"]["both"])
    ("penalty probe, Final: mean score of biased answers", "mean_score_biased"),
    ("penalty probe, Final: mean score of unbiased answers", "mean_score_unbiased"),
    ("penalty probe, Final: AUC biased vs unbiased", "auc"),
]
MEMORIZATION_ROW = "training: flipped label, cue at the true answer, share naming the label"


def run_names(condition: str, seed: int) -> list[str]:
    return [f"kl_weight_1_0_s{seed}"] + [RUN_KEY_FORMAT[condition].format(seed=seed, round=r) for r in ROUNDS[1:]]


def eval_values(condition: str, seed: int) -> dict[str, list]:
    table = json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json").read_text())
    assert list(table["runs"]) == [f"round{r}" for r in ROUNDS], list(table["runs"])
    return {row: table["mean_sd"][metric][cue]["after_rl"] for row, metric, cue in EVAL_ROWS}


def initial_values(condition: str, seed: int) -> dict[str, list]:
    table = json.loads((CONDITION_FOLDERS["refit"] / f"seed_{seed}" / "probe_table.json").read_text())
    assert list(table) == [f"round{r}" for r in ROUNDS], list(table)
    own_test = [table[f"round{r}"]["scores"]["own_test_set"] for r in ROUNDS]
    values = {}
    for row, key in INITIAL_ROWS:
        per_round = [None if scores is None else scores[key] for scores in own_test]
        values[row] = per_round if condition == "refit" else per_round[:1] + [None] * (len(ROUNDS) - 1)
    return values


def final_values(condition: str, seed: int) -> dict[str, list]:
    dynamics = json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "runs" / "dynamics" / "dynamics.json").read_text())
    names = run_names(condition, seed)
    assert list(dynamics["probe_letters"]) == names, list(dynamics["probe_letters"])
    return {row: [dynamics["probe_letters"][name]["C"]["both"][key] for name in names] for row, key in FINAL_ROWS}


def memorization_share(run_dir: Path) -> dict:
    records_by_step = compare.training_records(run_dir)
    assert list(records_by_step) == list(range(128)), (run_dir, len(records_by_step))
    answers, naming_label = 0, 0
    for records in records_by_step.values():
        for record in records:
            assert record["cue_letter"] is not None
            assert record["reward"] == float(record["parsed_answer"] == record["train_label"])
            flipped = record["train_label"] != record["gold_letter"]
            if flipped and record["cue_letter"] == record["gold_letter"]:
                answers += 1
                naming_label += record["parsed_answer"] == record["train_label"]
    return {"answers": answers, "share_naming_label": naming_label / answers}


def memorization_values(condition: str, seed: int) -> tuple[list[float], list[int]]:
    table = json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json").read_text())
    shares = [memorization_share(Path(run["run_dir"])) for run in table["runs"].values()]
    return [s["share_naming_label"] for s in shares], [s["answers"] for s in shares]


def seed_values(condition: str, seed: int) -> dict:
    shares, answers = memorization_values(condition, seed)
    rows = {**eval_values(condition, seed), **initial_values(condition, seed), **final_values(condition, seed),
            MEMORIZATION_ROW: shares}
    return {"rows": rows, "memorization_answers": answers}


def mean_sd(values: list) -> str:
    if any(v is None for v in values):
        return "–"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def markdown(result: dict) -> str:
    lines = ["# Frozen against refit probe, training labels redrawn every round, rounds 1-5", "",
             "Written by `memorization.py`; definitions in its docstring. frozen = 261108, refit = this experiment; round 1 "
             "is 261103's run in both. Each cell: mean ± SD over seeds 42 and 43 (each seed's value in comparison.json).", "",
             "| Metric | Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |", "|---|---|---|---|---|---|---|"]
    row_names = ([row for row, _, _ in EVAL_ROWS] + [row for row, _ in INITIAL_ROWS] + [row for row, _ in FINAL_ROWS]
                 + [MEMORIZATION_ROW])
    for row in row_names:
        for condition in CONDITION_FOLDERS:
            per_seed = [result[condition][f"s{seed}"]["rows"][row] for seed in SEEDS]
            cells = [mean_sd(list(values)) for values in zip(*per_seed, strict=True)]
            lines.append(f"| {row} | {condition} | {' | '.join(cells)} |")
    lines += ["", "Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5:", ""]
    for condition in CONDITION_FOLDERS:
        counts = [" / ".join(str(result[condition][f"s{seed}"]["memorization_answers"][i]) for seed in SEEDS) for i in range(len(ROUNDS))]
        lines.append(f"- {condition}: {', '.join(counts)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = {condition: {f"s{seed}": seed_values(condition, seed) for seed in SEEDS} for condition in CONDITION_FOLDERS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT_DIR / "comparison.md").write_text(markdown(result))
    print(f"Wrote {OUT_DIR / 'comparison.json'} and comparison.md")


if __name__ == "__main__":
    main()
