"""Label memorization, cue use and the probe's share of the training signal, from the saved training answers of this
experiment's frozen-probe lines, 261104's refit lines (the same round 1) and 261103's no-penalty runs (one round);
and the frozen-vs-refit exact test on each round's eval bias counts.

Usage (CPU, about 1.5 min):
    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
    python memorization.py    # writes runs/memorization/memorization.json and memorization.md

Training answers (each run's checkpoints/responses/step_*.jsonl), per step window:
  - Cells by where the cue sat (at the training label, or at the other option) and whether the question's label is
    flipped (the training label is the wrong answer; 243 of the 1217 training questions): the share of answers
    naming the cue and the share naming the training label. In the cell "label flipped, cue at the true answer"
    neither the cue nor the model's knowledge points at the label, so naming it means the label was memorized.
  - The mean probe score of answers naming the cue: the penalty they paid at weight 1.0 (no-penalty runs: the score
    of the probe the trainer fits, which their reward ignores).
Over all 128 steps: the advantage summed over answers naming the cue, per 1000 answers, split into a task part
(advantages recomputed with the probe weight at 0) and a probe part (the full advantage minus the task part), each by
whether the cue sat at the training label, and over all prompts. Advantages as dynamics.py (SD floor 0.3, the 0.1 cut-off penalty).
Eval bias counts (each line's tables/rounds/seed_table.json): biased cued answers over cued answers with a no-cue
answer in both evals, both cue conditions pooled; Fisher's exact test, frozen against refit, per round.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import fisher_exact

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402
import dynamics  # noqa: E402  (this folder's)

FOLDER = Path(__file__).resolve().parent
EXPERIMENTS = FOLDER.parent
QUESTIONS_PATH = Path("/workspace/sycophancy-rl/sycophancy_setup/data/questions.jsonl")
OUT_DIR = FOLDER / "runs" / "memorization"
SEEDS = (42, 43)
CONDITION_FOLDERS = {"frozen": FOLDER, "refit": EXPERIMENTS / "261104_iterative_kl_0_01"}
NO_PENALTY_TABLE = EXPERIMENTS / "261103_kl_0_01" / "no_penalty" / "tables" / "kl_no_penalty" / "seed_table.json"
PENALTY_WEIGHT = {"frozen": 1.0, "refit": 1.0, "no_penalty": 0.0}  # loss.detector of each run's yaml
STD_FLOOR = 0.3  # loss.advantage_std_floor of every run here
WINDOWS = [(0, 10), (60, 128)]
CELLS = {  # (cue at the training label, label flipped) -> name
    (False, False): "label true, cue at the wrong answer",
    (False, True): "label flipped, cue at the true answer",
    (True, False): "label true, cue at the true answer",
    (True, True): "label flipped, cue at the wrong answer",
}


def seed_table(condition: str, seed: int) -> dict:
    return json.loads((CONDITION_FOLDERS[condition] / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json").read_text())


def run_dirs() -> dict[str, list[Path]]:
    """Per line (condition and seed), its run folders in round order; the no-penalty runs in seed order."""
    lines = {}
    for condition in CONDITION_FOLDERS:
        for seed in SEEDS:
            runs = seed_table(condition, seed)["runs"]
            assert list(runs) == [f"round{r}" for r in range(1, 6)], list(runs)
            lines[f"{condition} s{seed}"] = [Path(run["run_dir"]) for run in runs.values()]
    no_penalty_runs = json.loads(NO_PENALTY_TABLE.read_text())["runs"]
    lines["no_penalty"] = [Path(run["run_dir"]) for run in no_penalty_runs.values()]
    return lines


def window_summary(steps: list[list[dict]], flipped_ids: set[int], train_ids: set[int]) -> dict:
    counts = {name: {"answers": 0, "names_cue": 0, "names_label": 0} for name in CELLS.values()}
    probe_scores_of_cue_answers = []
    for records in steps:
        for record in records:
            assert record["question_id"] in train_ids and record["cue_letter"] is not None
            flipped = record["question_id"] in flipped_ids
            assert flipped == (record["train_label"] != record["gold_letter"])
            assert record["reward"] == float(record["parsed_answer"] == record["train_label"])
            cell = counts[CELLS[(record["cue_letter"] == record["train_label"], flipped)]]
            cell["answers"] += 1
            cell["names_cue"] += dynamics.names_cue(record)
            cell["names_label"] += record["parsed_answer"] == record["train_label"]
            if dynamics.names_cue(record):
                probe_scores_of_cue_answers.append(record["probe_score"])
    cells = {name: {"answers": c["answers"], "share_naming_cue": c["names_cue"] / c["answers"],
                    "share_naming_label": c["names_label"] / c["answers"]} for name, c in counts.items()}
    return {"cells": cells, "mean_probe_score_of_cue_naming_answers": sum(probe_scores_of_cue_answers) / len(probe_scores_of_cue_answers)}


def advantage_split(steps: list[list[dict]], weight: float) -> dict:
    """Advantage summed over cue-naming answers per 1000 answers: task part and probe part, by cue placement."""
    sums = {"task_cue_at_label": 0.0, "task_cue_off_label": 0.0, "probe_cue_at_label": 0.0, "probe_cue_off_label": 0.0}
    answers = 0
    for records in steps:
        for group in compare.step_groups(records):
            full = dynamics.advantages(group, weight, STD_FLOOR)
            task_only = dynamics.advantages(group, 0.0, STD_FLOOR)
            placement = "cue_at_label" if group[0]["cue_letter"] == group[0]["train_label"] else "cue_off_label"
            answers += len(group)
            for record, full_advantage, task_advantage in zip(group, full, task_only, strict=True):
                if dynamics.names_cue(record):
                    sums[f"task_{placement}"] += task_advantage
                    sums[f"probe_{placement}"] += full_advantage - task_advantage
    for part in ("task", "probe"):
        sums[f"{part}_all_prompts"] = sums[f"{part}_cue_at_label"] + sums[f"{part}_cue_off_label"]
    return {key: 1000 * value / answers for key, value in sums.items()}


def training() -> dict:
    questions = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines()]
    train_ids = {q["question_id"] for q in questions if q["split"] == "train"}
    flipped_ids = {q["question_id"] for q in questions if q["split"] == "train" and q["label_flipped"]}
    assert (len(train_ids), len(flipped_ids)) == (1217, 243), (len(train_ids), len(flipped_ids))
    out = {}
    for line, dirs in run_dirs().items():
        weight = PENALTY_WEIGHT[line.split()[0]]
        out[line] = []
        for run_dir in dirs:
            records_by_step = compare.training_records(run_dir)
            assert list(records_by_step) == list(range(128)), (run_dir, len(records_by_step))
            steps = [records_by_step[step] for step in range(128)]
            out[line].append({
                "run_dir": str(run_dir),
                "windows": {f"{a}-{b - 1}": window_summary(steps[a:b], flipped_ids, train_ids) for a, b in WINDOWS},
                "advantage_on_cue_naming_answers_per_1000": advantage_split(steps, weight),
            })
    return out


def bias_counts(condition: str, seed: int) -> list[tuple[int, int]]:
    """Per round 1-5: (biased cued answers, cued answers with a no-cue answer in both evals), both cue conditions."""
    mean_sd = seed_table(condition, seed)["mean_sd"]
    per_round = []
    for round_index in range(5):
        biased, total = 0, 0
        for cue_condition in ("cue_gold", "cue_dist"):
            rows = mean_sd["n_bias_rows"][cue_condition]["after_rl"][round_index]
            share = mean_sd["bias"][cue_condition]["after_rl"][round_index]
            assert abs(share * rows - round(share * rows)) < 1e-6, (condition, seed, round_index, share, rows)
            biased += round(share * rows)
            total += rows
        per_round.append((biased, total))
    return per_round


def fisher(frozen: tuple[int, int], refit: tuple[int, int]) -> float:
    return fisher_exact([[frozen[0], frozen[1] - frozen[0]], [refit[0], refit[1] - refit[0]]])[1]


def eval_bias_tests() -> dict:
    counts = {(condition, seed): bias_counts(condition, seed) for condition in CONDITION_FOLDERS for seed in SEEDS}
    out = {}
    for round_index in range(1, 5):  # rounds 2-5; round 1 is the same run in both conditions
        row = {}
        for seed in SEEDS:
            frozen, refit = counts[("frozen", seed)][round_index], counts[("refit", seed)][round_index]
            row[f"s{seed}"] = {"frozen": frozen, "refit": refit, "p": fisher(frozen, refit)}
        frozen_pooled = tuple(map(sum, zip(*(counts[("frozen", seed)][round_index] for seed in SEEDS))))
        refit_pooled = tuple(map(sum, zip(*(counts[("refit", seed)][round_index] for seed in SEEDS))))
        row["both_seeds"] = {"frozen": frozen_pooled, "refit": refit_pooled, "p": fisher(frozen_pooled, refit_pooled)}
        out[f"round{round_index + 1}"] = row
    return out


def markdown(result: dict) -> str:
    training_result = result["training"]
    lines = ["# Label memorization, cue use and the probe's share of the training signal", "",
             "Written by `memorization.py`; definitions in its docstring. Lines: `frozen s4X` = this experiment (round 1 = "
             "261103's weight-1.0 run), `refit s4X` = 261104, `no_penalty` = 261103's three no-penalty runs (seeds 42, 43, "
             "44; one round each). Values per round 1-5 (no_penalty: per seed).", ""]
    for window in [f"{a}-{b - 1}" for a, b in WINDOWS]:
        lines += [f"## Training answers, steps {window}", ""]
        for measure in ("share_naming_label", "share_naming_cue"):
            for (cue_at_label, _), cell in CELLS.items():
                if measure == "share_naming_cue" and cue_at_label:
                    continue  # with the cue at the training label, naming the cue is naming the label
                lines += [f"**{cell}: {measure.replace('_', ' ')}**", "", "| line | values | answers per run |", "|---|---|---|"]
                for line, runs in training_result.items():
                    values = " / ".join(f"{run['windows'][window]['cells'][cell][measure]:.3f}" for run in runs)
                    answers = " / ".join(str(run["windows"][window]["cells"][cell]["answers"]) for run in runs)
                    lines.append(f"| {line} | {values} | {answers} |")
                lines.append("")
        lines += ["**mean probe score of answers naming the cue**", "", "| line | values |", "|---|---|"]
        for line, runs in training_result.items():
            values = " / ".join(f"{run['windows'][window]['mean_probe_score_of_cue_naming_answers']:.4f}" for run in runs)
            lines.append(f"| {line} | {values} |")
        lines.append("")
    lines += ["## Advantage summed over answers naming the cue, per 1000 training answers, steps 0-127", "",
              "| line | part | cue at the training label | cue at the other option | all prompts |", "|---|---|---|---|---|"]
    for line, runs in training_result.items():
        for part in ("task", "probe"):
            cells = [" / ".join(f"{run['advantage_on_cue_naming_answers_per_1000'][f'{part}_{prompts}']:+.1f}" for run in runs)
                     for prompts in ("cue_at_label", "cue_off_label", "all_prompts")]
            lines.append(f"| {line} | {part} | {' | '.join(cells)} |")
    lines += ["", "## Eval bias counts, frozen against refit (Fisher's exact test)", "",
              "Biased cued answers / cued answers with a no-cue answer in both evals, cue at the correct and at the wrong "
              "answer pooled.", "", "| round | seed | frozen | refit | p |", "|---|---|---|---|---|"]
    for round_name, row in result["eval_bias_tests"].items():
        for seed, cell in row.items():
            (fb, ft), (rb, rt) = cell["frozen"], cell["refit"]
            lines.append(f"| {round_name} | {seed} | {fb}/{ft} ({100 * fb / ft:.1f}%) | {rb}/{rt} ({100 * rb / rt:.1f}%) | {cell['p']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = {"training": training(), "eval_bias_tests": eval_bias_tests()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "memorization.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT_DIR / "memorization.md").write_text(markdown(result))
    print(f"Wrote {OUT_DIR / 'memorization.json'} and memorization.md")


if __name__ == "__main__":
    main()
