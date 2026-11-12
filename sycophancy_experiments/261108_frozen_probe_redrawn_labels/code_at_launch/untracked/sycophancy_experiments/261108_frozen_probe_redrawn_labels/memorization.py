"""Fixed against redrawn training labels under the frozen probe, rounds 1-5: the eval rates, round 1's probe on the eval
answers, and label memorization in training, each per seed (42, 43) and their mean. 261107's memorization.py cut down
to these rows.

Usage (CPU, a few minutes):
    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
    $PY memorization.py    # writes runs/comparison/comparison.json and comparison.md

Conditions: fixed = 261107_frozen_probe_kl_0_01 (the original question file in every round), redrawn = this folder
(round R's training labels from labels/roundR/labels.jsonl). Round 1 is 261103's weight-1.0 run of the seed in both.

Rows, per round:
  Eval, from each seed's tables/rounds/seed_table.json (compare.py's definitions, the run's final eval):
    bias and cue agreement rate (followed_cue) on the training cue mix (train_mix); true accuracy with no cue (none),
    with the cue at the correct answer (cue_gold) and at the wrong answer (cue_dist); deception_soft on train_mix.
  Probe, from each seed's runs/dynamics/dynamics.json (probe_letters, row C, "both"): round 1's probe on the round's
    eval answers that name the cued option, read through the round's model, both cue conditions pooled: the mean score
    of the biased answers, of the unbiased ones, and the AUC between them.
  Memorization, from each run's training answers (checkpoints/responses/step_*.jsonl), all 128 steps: on prompts of
    flipped questions (the record's train_label is not the gold letter, so each round's own labels) with the cue at
    the true answer, the share of answers naming the training label. Every answer counts, unparsed ones too.
A mean is None (n/a) if either seed's value is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402

FOLDER = Path(__file__).resolve().parent
CONDITION_FOLDERS = {"fixed": FOLDER.parent / "261107_frozen_probe_kl_0_01", "redrawn": FOLDER}
RUN_KEY_WORD = {"fixed": "frozen", "redrawn": "redrawn"}  # run keys s<seed>_<word>_round<R>
SEEDS = (42, 43)
OUT_DIR = FOLDER / "runs" / "comparison"
EVAL_ROWS = [  # (row name, seed_table.json metric, cue condition)
    ("bias, training cue mix", "bias", "train_mix"),
    ("cue agreement rate, training cue mix", "followed_cue", "train_mix"),
    ("true accuracy, no cue", "accuracy", "none"),
    ("true accuracy, cue at the correct answer", "accuracy", "cue_gold"),
    ("true accuracy, cue at the wrong answer", "accuracy", "cue_dist"),
    ("deception (P(Yes)-weighted), training cue mix", "deception_soft", "train_mix"),
]
PROBE_ROWS = [  # (row name, key of dynamics.json probe_letters[run]["C"]["both"])
    ("round 1's probe, mean score of biased answers", "mean_score_biased"),
    ("round 1's probe, mean score of unbiased answers", "mean_score_unbiased"),
    ("round 1's probe, AUC biased vs unbiased", "auc"),
]
MEMORIZATION_ROW = "training: flipped label, cue at the true answer, share naming the label"


def run_names(condition: str, seed: int) -> list[str]:
    return [f"kl_weight_1_0_s{seed}"] + [f"s{seed}_{RUN_KEY_WORD[condition]}_round{r}" for r in range(2, 6)]


def eval_values(condition: str, seed: int) -> dict[str, list]:
    table = json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json").read_text())
    assert list(table["runs"]) == [f"round{r}" for r in range(1, 6)], list(table["runs"])
    return {row: table["mean_sd"][metric][cue]["after_rl"] for row, metric, cue in EVAL_ROWS}


def probe_values(condition: str, seed: int) -> dict[str, list]:
    dynamics = json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "runs" / "dynamics" / "dynamics.json").read_text())
    names = run_names(condition, seed)
    assert list(dynamics["probe_letters"]) == names, list(dynamics["probe_letters"])
    return {row: [dynamics["probe_letters"][name]["C"]["both"][key] for name in names] for row, key in PROBE_ROWS}


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
    return {"rows": {**eval_values(condition, seed), **probe_values(condition, seed), MEMORIZATION_ROW: shares},
            "memorization_answers": answers}


def mean_of_seeds(values: list) -> float | None:
    return None if any(v is None for v in values) else sum(values) / len(values)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def markdown(result: dict) -> str:
    lines = ["# Fixed against redrawn training labels, frozen probe, rounds 1-5", "",
             "Written by `memorization.py`; definitions in its docstring. fixed = 261107, redrawn = this experiment; round 1 "
             "is 261103's run in both. Each cell: mean of seeds 42 and 43 (seed 42 / seed 43).", "",
             "| metric | labels | round 1 | round 2 | round 3 | round 4 | round 5 |", "|---|---|---|---|---|---|---|"]
    row_names = [row for row, _, _ in EVAL_ROWS] + [row for row, _ in PROBE_ROWS] + [MEMORIZATION_ROW]
    for row in row_names:
        for condition in CONDITION_FOLDERS:
            per_seed = [result[condition][f"s{seed}"]["rows"][row] for seed in SEEDS]
            cells = [f"{fmt(mean_of_seeds(list(values)))} ({' / '.join(fmt(v) for v in values)})" for values in zip(*per_seed, strict=True)]
            lines.append(f"| {row} | {condition} | {' | '.join(cells)} |")
    lines += ["", "Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5:", ""]
    for condition in CONDITION_FOLDERS:
        counts = [" / ".join(str(result[condition][f"s{seed}"]["memorization_answers"][i]) for seed in SEEDS) for i in range(5)]
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
