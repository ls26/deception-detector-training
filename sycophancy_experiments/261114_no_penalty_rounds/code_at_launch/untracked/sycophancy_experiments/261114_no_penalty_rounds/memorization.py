"""No penalty against the frozen and the retrained probe, all with fixed training labels, rounds 1-5: the eval rates,
round 1's probe on each round's eval answers, and label memorization in training, seeds 42 and 43. 261110's
memorization.py with a no-penalty condition, a base-model column, no Fisher table, and rows split by seed when the
seeds disagree (below).

Usage (CPU, a few minutes):
    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
    $PY memorization.py    # writes runs/comparison/comparison.json and comparison.md

Conditions, each trained every round on the original question file (fixed labels), seeds 42 and 43:
  no penalty  this folder: round 1 is 261103's no-penalty run kl_no_penalty_s<seed>, rounds 2-5 rounds.py's runs
              s<seed>_no_penalty_roundR (penalty weight 0; round 1's probe scores answers but not in the reward).
  frozen      261107: round 1 is 261103's weight-1.0 run kl_weight_1_0_s<seed>, rounds 2-5 s<seed>_frozen_roundR with
              round 1's probe in the penalty every round.
  retrained   261104: the same round 1, rounds 2-5 s<seed>_roundR with the probe refit before every round.
  Inputs per seed folder seed_<seed>/ of each: tables/rounds/seed_table.json (seed_table.py), runs/dynamics/
  dynamics.json (dynamics.py), and for no penalty and frozen probe_table.json (rounds.py probe_table).
  Round 1's probe is one file for all six runs of round 1 (the probe each 261103 run fit on the base model's answers is
  byte-identical in all of them).

Rows, per round 1-5, and a base-model column:
  Eval, from each seed's seed_table.json (compare.py's definitions, the run's final eval): bias, cue agreement rate
    (followed_cue) and true accuracy on the training cue mix (train_mix); true accuracy with no cue (none), with the cue
    at the correct answer (cue_gold) and at the wrong answer (cue_dist); deception_soft on train_mix. Base model:
    seed_table.json's base_mean, the base eval's value averaged over the five rounds' question sets.
  Round 1's probe, no penalty and frozen only, both cue conditions pooled: the mean score of the biased answers, of the
    unbiased ones, and the AUC between them, on the round's eval answers that name the cued option, read through the
    round's model; from each seed's dynamics.json (probe_letters, row C, "both"; C reads the final eval's probe_start,
    which is round 1's probe through each round's links folder). dynamics.py keeps the first sampled answer per prompt,
    only answers labelled biased or unbiased, and only questions with a no-cue answer in both the base eval and the
    round's eval. Base model: the same on the base model's eval answers read through the base model (probe_table.json's
    base_answers), averaged over the five rounds' question sets.
  Memorization, from each run's training answers (checkpoints/responses/step_*.jsonl), all 128 steps: on prompts of
    flipped questions (the record's train_label is not the gold letter) with the cue at the true answer, the share of
    answers naming the training label. Every answer counts, unparsed ones too. No base-model value.

Cells (the user's rule, 2026-11-14): a row is one measurement for the three conditions. Each cell is the mean ± SD
(statistics.stdev) over seeds 42 and 43, unless the row is split; then every cell of the row is "s42 / s43". A row is
split when (a) in some round the two seeds put no penalty and frozen in opposite order (one seed above, the other below),
or (b) no penalty moves in opposite directions from round 1 to round 5 in the two seeds. comparison.md lists the split
rows and the reason; comparison.json holds each seed's values for every row.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402

FOLDER = Path(__file__).resolve().parent
SEEDS = (42, 43)
CONDITIONS = ("no penalty", "frozen", "retrained")
CONDITION_FOLDERS = {"no penalty": FOLDER,
                     "frozen": FOLDER.parent / "261107_frozen_probe_kl_0_01",
                     "retrained": FOLDER.parent / "261104_iterative_kl_0_01"}
ROUND1_RUN_KEY = {"no penalty": "kl_no_penalty_s{seed}", "frozen": "kl_weight_1_0_s{seed}", "retrained": "kl_weight_1_0_s{seed}"}
LATER_RUN_KEY = {"no penalty": "s{seed}_no_penalty_round{round}", "frozen": "s{seed}_frozen_round{round}",
                 "retrained": "s{seed}_round{round}"}
PROBE_CONDITIONS = ("no penalty", "frozen")  # the conditions whose eval probe is round 1's probe
ROUNDS = (1, 2, 3, 4, 5)
OUT_DIR = FOLDER / "runs" / "comparison"
EVAL_ROWS = [  # (row name, seed_table.json metric, cue condition)
    ("bias, training cue mix", "bias", "train_mix"),
    ("cue agreement rate, training cue mix", "followed_cue", "train_mix"),
    ("true accuracy, training cue mix", "accuracy", "train_mix"),
    ("true accuracy, no cue", "accuracy", "none"),
    ("true accuracy, cue at the correct answer", "accuracy", "cue_gold"),
    ("true accuracy, cue at the wrong answer", "accuracy", "cue_dist"),
    ("deception (P(Yes) version), training cue mix", "deception_soft", "train_mix"),
]
PROBE_ROWS = [  # (row name, key of dynamics.json probe_letters[run]["C"]["both"], key of probe_table.json base_answers)
    ("round 1's probe: mean score of biased answers", "mean_score_biased", "mean_biased"),
    ("round 1's probe: mean score of unbiased answers", "mean_score_unbiased", "mean_unbiased"),
    ("round 1's probe: AUC biased vs unbiased", "auc", "auc"),
]
MEMORIZATION_ROW = "training: flipped label, cue at the true answer, share naming the label"


def seed_folder(condition: str, seed: int) -> Path:
    return CONDITION_FOLDERS[condition] / f"seed_{seed}"


def run_names(condition: str, seed: int) -> list[str]:
    return [ROUND1_RUN_KEY[condition].format(seed=seed)] + [LATER_RUN_KEY[condition].format(seed=seed, round=r) for r in ROUNDS[1:]]


def seed_table(condition: str, seed: int) -> dict:
    table = json.loads((seed_folder(condition, seed) / "tables" / "rounds" / "seed_table.json").read_text())
    assert list(table["runs"]) == [f"round{r}" for r in ROUNDS], list(table["runs"])
    return table


def eval_values(condition: str, seed: int) -> dict[str, dict]:
    table = seed_table(condition, seed)["mean_sd"]
    return {row: {"base": table[metric][cue]["base_mean"], "rounds": table[metric][cue]["after_rl"]} for row, metric, cue in EVAL_ROWS}


def probe_values(condition: str, seed: int) -> dict[str, dict]:
    dynamics = json.loads((seed_folder(condition, seed) / "runs" / "dynamics" / "dynamics.json").read_text())
    names = run_names(condition, seed)
    assert list(dynamics["probe_letters"]) == names, list(dynamics["probe_letters"])
    probe_table = json.loads((seed_folder(condition, seed) / "probe_table.json").read_text())["rounds"]
    assert list(probe_table) == [f"round{r}" for r in ROUNDS], list(probe_table)
    return {row: {"base": statistics.mean(probe_table[f"round{r}"]["base_answers"][base_key] for r in ROUNDS),
                  "rounds": [dynamics["probe_letters"][name]["C"]["both"][key] for name in names]}
            for row, key, base_key in PROBE_ROWS}


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


def memorization_values(condition: str, seed: int) -> tuple[dict, list[int]]:
    runs = seed_table(condition, seed)["runs"].values()
    shares = [memorization_share(Path(run["run_dir"])) for run in runs]
    return {"base": None, "rounds": [s["share_naming_label"] for s in shares]}, [s["answers"] for s in shares]


def seed_values(condition: str, seed: int) -> dict:
    memorization, answers = memorization_values(condition, seed)
    rows = {**eval_values(condition, seed), MEMORIZATION_ROW: memorization}
    if condition in PROBE_CONDITIONS:
        rows.update(probe_values(condition, seed))
    return {"rows": rows, "memorization_answers": answers}


def split_reasons(result: dict, row: str) -> list[str]:
    """Why a row is split by seed (call 1 of the plan); empty if it shows means."""
    no_penalty = [result["no penalty"][f"s{seed}"]["rows"][row]["rounds"] for seed in SEEDS]
    frozen = [result["frozen"][f"s{seed}"]["rows"][row]["rounds"] for seed in SEEDS]
    reasons = []
    for index, round_number in enumerate(ROUNDS):
        differences = [no_penalty[i][index] - frozen[i][index] for i in range(len(SEEDS))]
        if differences[0] * differences[1] < 0:
            reasons.append(f"round {round_number}: no penalty minus frozen is {differences[0]:+.3f} (s42) and {differences[1]:+.3f} (s43)")
    changes = [values[-1] - values[0] for values in no_penalty]
    if changes[0] * changes[1] < 0:
        reasons.append(f"no penalty, round 1 to round 5: {changes[0]:+.3f} (s42) and {changes[1]:+.3f} (s43)")
    return reasons


def cell(values: list, split: bool) -> str:
    if any(v is None for v in values):
        return "–"
    if split:
        return " / ".join(f"{v:.3f}" for v in values)
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def markdown(result: dict) -> str:
    lines = ["# No penalty against the frozen and the retrained probe, fixed training labels, rounds 1-5", "",
             "Written by `memorization.py`; definitions and sources in its docstring. Round 1: 261103's no-penalty run "
             "(no penalty) and its weight-1.0 run (frozen, retrained). Each cell: mean ± SD over seeds 42 and 43, or "
             "\"s42 / s43\" in a split row (listed below the table). Each seed's values are in comparison.json.", "",
             "| Measurement | Condition | Base model | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |",
             "|---|---|---|---|---|---|---|---|"]
    row_names = [row for row, _, _ in EVAL_ROWS] + [row for row, _, _ in PROBE_ROWS] + [MEMORIZATION_ROW]
    split_rows = {row: split_reasons(result, row) for row in row_names}
    for row in row_names:
        split = bool(split_rows[row])
        for condition in CONDITIONS:
            if row not in result[condition][f"s{SEEDS[0]}"]["rows"]:
                continue
            per_seed = [result[condition][f"s{seed}"]["rows"][row] for seed in SEEDS]
            cells = [cell([values["base"] for values in per_seed], split)]
            cells += [cell(list(values), split) for values in zip(*(values["rounds"] for values in per_seed), strict=True)]
            lines.append(f"| {row} | {condition} | {' | '.join(cells)} |")
    lines += ["", "## Split rows", ""]
    lines += [f"- {row}: {'; '.join(reasons)}" for row, reasons in split_rows.items() if reasons] or ["None."]
    lines += ["", "## Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5", ""]
    for condition in CONDITIONS:
        counts = [" / ".join(str(result[condition][f"s{seed}"]["memorization_answers"][i]) for seed in SEEDS) for i in range(len(ROUNDS))]
        lines.append(f"- {condition}: {', '.join(counts)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    result = {condition: {f"s{seed}": seed_values(condition, seed) for seed in SEEDS} for condition in CONDITIONS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(json.dumps(result, indent=1) + "\n")
    (OUT_DIR / "comparison.md").write_text(markdown(result))
    print(f"Wrote {OUT_DIR / 'comparison.json'} and comparison.md")


if __name__ == "__main__":
    main()
