"""Frozen against refit probe, both with the training labels redrawn every round, rounds 1-5: the eval rates, the probe
each round trained against (Initial and Final), and label memorization in training, mean and SD over seeds 45 and 46,
and over seeds 42, 43, 45 and 46. 261109's memorization.py with seeds 45 and 46 read from this folder (both conditions
share round 1, seed_<seed>/s<seed>_round1.yaml) and seeds 42 and 43 from 261109 (refit) and 261108 (frozen).

Usage (CPU, a few minutes):
    source /workspace/sycophancy-rl/sycophancy_setup/run_env.sh
    $PY memorization.py    # writes runs/comparison/ (seeds 45, 46) and runs/comparison_four_seeds/ (42, 43, 45, 46),
                           # each comparison.json and comparison.md

Conditions: frozen = round 1's probe in the penalty every round (261108's procedure), refit = the probe refit before
every round (261104's procedure). Both train round R on 261108's labels/roundR/labels.jsonl.
  Seeds 45, 46, this folder: frozen = frozen_rounds.py's runs s<seed>_frozen_roundR, refit = rounds.py's runs
    s<seed>_refit_roundR; round 1 is s<seed>_round1 in both (261103's weight-1.0 config, the original question file).
    Inputs per seed folder seed_<seed>/: tables/<condition>/seed_table.json (seed_table.py), runs/dynamics_<condition>/
    dynamics.json (dynamics.py) and probe_table.json (rounds.py table 5).
  Seeds 42, 43: frozen = 261108's runs s<seed>_redrawn_roundR, refit = 261109's runs s<seed>_refit_roundR; round 1 is
    261103's run kl_weight_1_0_s<seed> in both. Inputs per seed folder of 261108 / 261109: tables/rounds/seed_table.json,
    runs/dynamics/dynamics.json and (261109) probe_table.json.

Rows, per round:
  Eval, from each seed's seed_table.json (compare.py's definitions, the run's final eval):
    bias, cue agreement rate (followed_cue) and true accuracy on the training cue mix (train_mix); true accuracy with
    no cue (none), with the cue at the correct answer (cue_gold) and at the wrong answer (cue_dist); deception_soft on
    train_mix.
  Penalty probe (refit: the round's own probe, fit before the round; frozen: round 1's probe), both cue conditions
  pooled, the mean score of the biased answers, of the unbiased ones, and the AUC between them:
    Final: on the round's eval answers that name the cued option, read through the round's model; from each seed's
      dynamics.json (probe_letters, row C, "both"; C reads the final eval's probe_start, which is the round's own probe
      in a refit run and round 1's probe in a frozen run through its links folder). dynamics.py keeps the first sampled
      answer per prompt, only answers labelled biased or unbiased, and only questions with a no-cue answer in both the
      base eval and the round's eval.
    Initial: on the answers of the model the round starts from, from the refit seed's probe_table.json.
      Round 1 (shared): round 1's probe on the base model's eval answers (round1 "round_probe_base_answers", the
      base eval re-scored with round 1's probe, the same filters as Final; 261107's start of round 1). Not the probe's
      own test file (261010's 777 base-model answers from 260819, AUC 0.873), a different draw with another
      usual-answer rule.
      Rounds 2-5, refit only: the probe's own test set, which is round R-1's final-eval answers read through round
      R-1's model ("own_test_set"; None if the test file had one class). The frozen probe is never refit, so its
      rounds 2-5 are None (its start of round R is its Final of round R-1).
  Memorization, from each run's training answers (checkpoints/responses/step_*.jsonl), all 128 steps: on prompts of
    flipped questions (the record's train_label is not the gold letter, so each round's own labels) with the cue at
    the true answer, the share of answers naming the training label. Every answer counts, unparsed ones too.
  Eval bias counts (261107's test), rounds 2-5: biased cued answers / cued answers with a no-cue answer in both
    evals, the cue at the correct and at the wrong answer pooled (counted equally, not weighted as in the training cue
    mix), from each seed's seed_table.json (bias x n_bias_rows per cue condition); Fisher's exact test, frozen against
    refit, per seed and with the seeds pooled. Unpaired: every cued answer counts as independent.
  Memorization with fixed labels (four-seed output only), seeds 42 and 43: the same share for the chains that trained
    every round on the original question file, frozen = 261107's runs, refit = 261104's (round 1: 261103's run in both),
    run folders from their seed_4X/tables/rounds/seed_table.json.
comparison.md: mean ± SD (statistics.stdev, as seed_table.py) over the seeds; "–" if any seed's value is None; then
  the Fisher table, and (four seeds) the fixed-label memorization.
comparison.json: each seed's values per condition, "eval_bias_tests", and (four seeds) "fixed_labels".
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scipy.stats import fisher_exact

sys.path.insert(0, "/workspace/sycophancy-rl/sycophancy_setup")
import compare  # noqa: E402

FOLDER = Path(__file__).resolve().parent
CONDITIONS = ("frozen", "refit")
NEW_SEEDS = (45, 46)  # this folder
EARLIER_SEEDS = (42, 43)  # 261108 (frozen) and 261109 (refit)
EARLIER_FOLDERS = {"frozen": FOLDER.parent / "261108_frozen_probe_redrawn_labels",
                   "refit": FOLDER.parent / "261109_refit_probe_redrawn_labels"}
FIXED_LABEL_FOLDERS = {"frozen": FOLDER.parent / "261107_frozen_probe_kl_0_01",
                       "refit": FOLDER.parent / "261104_iterative_kl_0_01"}
RUN_KEY_FORMAT = {  # rounds 2-5
    "new": {"frozen": "s{seed}_frozen_round{round}", "refit": "s{seed}_refit_round{round}"},
    "earlier": {"frozen": "s{seed}_redrawn_round{round}", "refit": "s{seed}_refit_round{round}"},
}
ROUNDS = (1, 2, 3, 4, 5)
OUTPUTS = {FOLDER / "runs" / "comparison": NEW_SEEDS, FOLDER / "runs" / "comparison_four_seeds": EARLIER_SEEDS + NEW_SEEDS}
EVAL_ROWS = [  # (row name, seed_table.json metric, cue condition)
    ("bias, training cue mix", "bias", "train_mix"),
    ("cue agreement rate, training cue mix", "followed_cue", "train_mix"),
    ("true accuracy, training cue mix", "accuracy", "train_mix"),
    ("true accuracy, no cue", "accuracy", "none"),
    ("true accuracy, cue at the correct answer", "accuracy", "cue_gold"),
    ("true accuracy, cue at the wrong answer", "accuracy", "cue_dist"),
    ("deception (P(Yes)-weighted), training cue mix", "deception_soft", "train_mix"),
]
INITIAL_ROWS = [  # (row name, key of probe_table.json [round]["scores"][answer set]; see initial_values)
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
    if seed in EARLIER_SEEDS:
        round1, later = f"kl_weight_1_0_s{seed}", RUN_KEY_FORMAT["earlier"][condition]
    else:
        round1, later = f"s{seed}_round1", RUN_KEY_FORMAT["new"][condition]
    return [round1] + [later.format(seed=seed, round=r) for r in ROUNDS[1:]]


def seed_table_path(condition: str, seed: int) -> Path:
    if seed in EARLIER_SEEDS:
        return EARLIER_FOLDERS[condition] / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json"
    return FOLDER / f"seed_{seed}" / "tables" / condition / "seed_table.json"


def probe_table_path(seed: int) -> Path:  # the refit version's probes
    if seed in EARLIER_SEEDS:
        return EARLIER_FOLDERS["refit"] / f"seed_{seed}" / "probe_table.json"
    return FOLDER / f"seed_{seed}" / "probe_table.json"


def dynamics_path(condition: str, seed: int) -> Path:
    if seed in EARLIER_SEEDS:
        return EARLIER_FOLDERS[condition] / f"seed_{seed}" / "runs" / "dynamics" / "dynamics.json"
    return FOLDER / f"seed_{seed}" / "runs" / f"dynamics_{condition}" / "dynamics.json"


def eval_values(condition: str, seed: int) -> dict[str, list]:
    table = json.loads(seed_table_path(condition, seed).read_text())
    assert list(table["runs"]) == [f"round{r}" for r in ROUNDS], list(table["runs"])
    return {row: table["mean_sd"][metric][cue]["after_rl"] for row, metric, cue in EVAL_ROWS}


def initial_values(condition: str, seed: int) -> dict[str, list]:
    table = json.loads(probe_table_path(seed).read_text())
    assert list(table) == [f"round{r}" for r in ROUNDS], list(table)
    start_answers = [table["round1"]["scores"]["round_probe_base_answers"]]  # round 1: the base model's eval answers
    start_answers += [table[f"round{r}"]["scores"]["own_test_set"] for r in ROUNDS[1:]]
    values = {}
    for row, key in INITIAL_ROWS:
        per_round = [None if scores is None else scores[key] for scores in start_answers]
        values[row] = per_round if condition == "refit" else per_round[:1] + [None] * (len(ROUNDS) - 1)
    return values


def final_values(condition: str, seed: int) -> dict[str, list]:
    dynamics = json.loads(dynamics_path(condition, seed).read_text())
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
    table = json.loads(seed_table_path(condition, seed).read_text())
    shares = [memorization_share(Path(run["run_dir"])) for run in table["runs"].values()]
    return [s["share_naming_label"] for s in shares], [s["answers"] for s in shares]


def fixed_label_values() -> dict:
    """Memorization with the training labels fixed, seeds 42 and 43: frozen = 261107's runs, refit = 261104's."""
    out = {}
    for condition, folder in FIXED_LABEL_FOLDERS.items():
        out[condition] = {}
        for seed in EARLIER_SEEDS:
            table = json.loads((folder / f"seed_{seed}" / "tables" / "rounds" / "seed_table.json").read_text())
            assert list(table["runs"]) == [f"round{r}" for r in ROUNDS], list(table["runs"])
            shares = [memorization_share(Path(run["run_dir"])) for run in table["runs"].values()]
            out[condition][f"s{seed}"] = {"rows": {MEMORIZATION_ROW: [s["share_naming_label"] for s in shares]},
                                          "memorization_answers": [s["answers"] for s in shares]}
    return out


def seed_values(condition: str, seed: int) -> dict:
    shares, answers = memorization_values(condition, seed)
    rows = {**eval_values(condition, seed), **initial_values(condition, seed), **final_values(condition, seed),
            MEMORIZATION_ROW: shares}
    return {"rows": rows, "memorization_answers": answers}


def bias_counts(condition: str, seed: int) -> list[tuple[int, int]]:
    """Per round 1-5: (biased cued answers, cued answers with a no-cue answer in both evals), both cue conditions."""
    table = json.loads(seed_table_path(condition, seed).read_text())["mean_sd"]
    per_round = []
    for round_index in range(len(ROUNDS)):
        biased, total = 0, 0
        for cue_condition in ("cue_gold", "cue_dist"):
            rows = table["n_bias_rows"][cue_condition]["after_rl"][round_index]
            share = table["bias"][cue_condition]["after_rl"][round_index]
            assert abs(share * rows - round(share * rows)) < 1e-6, (condition, seed, round_index, share, rows)
            biased += round(share * rows)
            total += rows
        per_round.append((biased, total))
    return per_round


def fisher(frozen: tuple[int, int], refit: tuple[int, int]) -> float:
    return fisher_exact([[frozen[0], frozen[1] - frozen[0]], [refit[0], refit[1] - refit[0]]])[1]


def eval_bias_tests(seeds: tuple[int, ...]) -> dict:
    counts = {(condition, seed): bias_counts(condition, seed) for condition in CONDITIONS for seed in seeds}
    out = {}
    for round_index in range(1, len(ROUNDS)):  # rounds 2-5; round 1 is the same run in both conditions
        row = {}
        for seed in seeds:
            frozen, refit = counts[("frozen", seed)][round_index], counts[("refit", seed)][round_index]
            row[f"s{seed}"] = {"frozen": frozen, "refit": refit, "p": fisher(frozen, refit)}
        frozen_pooled = tuple(map(sum, zip(*(counts[("frozen", seed)][round_index] for seed in seeds))))
        refit_pooled = tuple(map(sum, zip(*(counts[("refit", seed)][round_index] for seed in seeds))))
        row["seeds_pooled"] = {"frozen": frozen_pooled, "refit": refit_pooled, "p": fisher(frozen_pooled, refit_pooled)}
        out[f"round{round_index + 1}"] = row
    return out


def mean_sd(values: list) -> str:
    if any(v is None for v in values):
        return "–"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def markdown(result: dict, seeds: tuple[int, ...]) -> str:
    seed_names = ", ".join(str(seed) for seed in seeds[:-1]) + f" and {seeds[-1]}"
    sources = ("frozen = frozen_rounds.py's runs, refit = rounds.py's; round 1 is this folder's s<seed>_round1 in both"
               if seeds == NEW_SEEDS else
               "seeds 45 and 46 from this folder (round 1: s<seed>_round1), seeds 42 and 43 from 261108 (frozen) and 261109 "
               "(refit) (round 1: 261103's kl_weight_1_0_s<seed>); round 1 is the same run in both conditions")
    lines = ["# Frozen against refit probe, training labels redrawn every round, rounds 1-5", "",
             f"Written by `memorization.py`; definitions in its docstring. {sources}. Each cell: mean ± SD over seeds "
             f"{seed_names} (each seed's value in comparison.json).", "",
             "| Metric | Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |", "|---|---|---|---|---|---|---|"]
    row_names = ([row for row, _, _ in EVAL_ROWS] + [row for row, _ in INITIAL_ROWS] + [row for row, _ in FINAL_ROWS]
                 + [MEMORIZATION_ROW])
    for row in row_names:
        for condition in CONDITIONS:
            per_seed = [result[condition][f"s{seed}"]["rows"][row] for seed in seeds]
            cells = [mean_sd(list(values)) for values in zip(*per_seed, strict=True)]
            lines.append(f"| {row} | {condition} | {' | '.join(cells)} |")
    lines += ["", f"Training answers counted in the memorization row ({' / '.join(f'seed {seed}' for seed in seeds)}), "
              "rounds 1-5:", ""]
    for condition in CONDITIONS:
        counts = [" / ".join(str(result[condition][f"s{seed}"]["memorization_answers"][i]) for seed in seeds) for i in range(len(ROUNDS))]
        lines.append(f"- {condition}: {', '.join(counts)}")
    lines += ["", "## Eval bias counts, frozen against refit (Fisher's exact test)", "",
              "Biased cued answers / cued answers with a no-cue answer in both evals, cue at the correct and at the wrong "
              "answer pooled (counted equally, not weighted as in the training cue mix). Unpaired: every cued answer "
              "counts as independent.", "", "| round | seed | frozen | refit | p |", "|---|---|---|---|---|"]
    for round_name, row in result["eval_bias_tests"].items():
        for seed, cell in row.items():
            (fb, ft), (rb, rt) = cell["frozen"], cell["refit"]
            lines.append(f"| {round_name} | {seed} | {fb}/{ft} ({100 * fb / ft:.1f}%) | {rb}/{rt} ({100 * rb / rt:.1f}%) | {cell['p']:.2g} |")
    if "fixed_labels" in result:
        fixed = result["fixed_labels"]
        lines += ["", "## Memorization with fixed labels (seeds 42 and 43)", "",
                  f"{MEMORIZATION_ROW}; frozen = 261107's runs, refit = 261104's, trained every round on the original "
                  "question file. Each cell: mean ± SD over seeds 42 and 43.", "",
                  "| Probe | Round 1 | Round 2 | Round 3 | Round 4 | Round 5 |", "|---|---|---|---|---|---|"]
        for condition in CONDITIONS:
            per_seed = [fixed[condition][f"s{seed}"]["rows"][MEMORIZATION_ROW] for seed in EARLIER_SEEDS]
            lines.append(f"| {condition} | {' | '.join(mean_sd(list(values)) for values in zip(*per_seed, strict=True))} |")
        lines += ["", "Training answers counted (seed 42 / seed 43), rounds 1-5:", ""]
        for condition in CONDITIONS:
            counts = [" / ".join(str(fixed[condition][f"s{seed}"]["memorization_answers"][i]) for seed in EARLIER_SEEDS)
                      for i in range(len(ROUNDS))]
            lines.append(f"- {condition}: {', '.join(counts)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    all_seeds = EARLIER_SEEDS + NEW_SEEDS
    values = {condition: {seed: seed_values(condition, seed) for seed in all_seeds} for condition in CONDITIONS}
    for out_dir, seeds in OUTPUTS.items():
        result = {condition: {f"s{seed}": values[condition][seed] for seed in seeds} for condition in CONDITIONS}
        result["eval_bias_tests"] = eval_bias_tests(seeds)
        if seeds != NEW_SEEDS:
            result["fixed_labels"] = fixed_label_values()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison.json").write_text(json.dumps(result, indent=1) + "\n")
        (out_dir / "comparison.md").write_text(markdown(result, seeds))
        print(f"Wrote {out_dir / 'comparison.json'} and comparison.md")


if __name__ == "__main__":
    main()
