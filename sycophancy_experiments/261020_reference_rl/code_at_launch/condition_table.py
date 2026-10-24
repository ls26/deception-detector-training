"""The condition table: base vs after RL on all eval questions for the baseline and the three reference conditions of one
format, each over its seeds (mean and SD, n - 1), with true accuracy first.

    python condition_table.py --format cot|nocot            -> <format>/condition_table.md, <format>/condition_table.json
    python condition_table.py --format cot|nocot --smoke    -> <format>/dry/condition_table_smoke.md / .json

Training conditions (columns): baseline (noisy labels, cue at the label 75%: 261013 / 261014 seed 42 and 261018's
seeds 43 / 44), noisy_nocue, correct_nocue and correct_cue (this folder's runs, seeds 42 / 43 / 44). Each run's eval
tables come from the setup's own compare.py functions (unchanged, loaded by path) on base = the setup folder's
evals/base and after RL = that run's final-checkpoint eval, as 261018's seed_table.py does:
  no-CoT  261014's prompt_rows + eval_tables (one prompt per question and condition, answered by its top letter);
  CoT     261013's rollout_rows + eval_tables on one sampled answer per prompt, the first of each prompt's saved
          samples; the no-cue answer behind bias stays the eval's own (at least 5 of 8 no-cue samples).
For the conditions trained on correct labels (correct_nocue, correct_cue) both the base and the run's rows are scored
as if every question's training label were its correct answer: "reward" is then true accuracy and the training cue
mix is the cue at the correct answer 75% of the time (that run's own training mix). The conditions trained without a
cue (noisy_nocue, correct_nocue) have no training cue mix: their training-cue-mix cells are blank. Bias and
self-report rows count questions with a no-cue answer in both the base eval and that run's eval, so their base value
can differ per seed; a "base → after" cell shows the mean of the seeds' base values. Training: the share of training
groups whose 8 rewards are all equal, mean over the steps (261014's compare.py functions). Each run's record-check
json must say PASS, the run's step count and its own seed.

--smoke builds the table for the smoke run alone (correct_nocue, seed 42, 2 steps, its eval on the first 50 eval
questions; the base rows restricted to those questions), to check this script on a new run's outputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SEEDS_DIR = Path("/workspace/sycophancy_experiments/261018_noisy_label_rl_seeds")
SETUP_DIRS = {
    "cot": Path("/workspace/sycophancy_experiments/261013_noisy_label_rl"),
    "nocot": Path("/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl"),
}
FORMAT_TITLES = {
    "cot": "CoT (261013's setup; one sampled answer per prompt)",
    "nocot": "No-CoT (261014's setup; top letter)",
}
TRAINING_CONDITIONS = {
    "baseline": "Baseline: noisy labels, cue",
    "noisy_nocue": "1. Noisy labels, no cue",
    "correct_nocue": "2. Correct labels, no cue",
    "correct_cue": "3. Correct labels, cue",
}
CORRECT_LABEL_CONDITIONS = ("correct_nocue", "correct_cue")
NO_CUE_CONDITIONS = ("noisy_nocue", "correct_nocue")
OWN_TRAINING_PROMPTS = {"baseline": "train_mix", "noisy_nocue": "none", "correct_nocue": "none", "correct_cue": "train_mix"}
EVAL_CONDITION_NAMES = {
    "none": "no cue",
    "cue_gold": "cue at the correct answer",
    "cue_dist": "cue at the wrong answer",
    "train_mix": "training cue mix",
}
SEEDS = (42, 43, 44)
N_STEPS = 128
SMOKE_STEPS = 2
DISPLAY_NAMES = {
    "accuracy": "true accuracy",
    "followed_cue": "cue agreement rate",
    "deception_soft": "deception",
    "deception": "deception (thresholded)",
}


def load_compare(setup: str):
    setup_dir = SETUP_DIRS[setup]
    sys.path.append(str(setup_dir))  # each compare.py imports its own spec module (noisy_spec / nocot_spec)
    module_spec = importlib.util.spec_from_file_location(f"{setup}_compare", setup_dir / "compare.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def run_files(format_name: str, condition: str, seed: int, smoke: bool) -> tuple[Path, Path]:
    """(final eval folder, record-check json) of one run."""
    if smoke:
        run = f"{format_name}_smoke_s{seed}"
        return EXP_DIR / format_name / "dry" / "evals" / run, EXP_DIR / format_name / "runs" / run / "check_records.json"
    if condition == "baseline" and seed == 42:
        return SETUP_DIRS[format_name] / "evals" / "final", SETUP_DIRS[format_name] / "check_records_run.json"
    if condition == "baseline":
        run = f"{format_name}_s{seed}"
        return SEEDS_DIR / "evals" / run, SEEDS_DIR / "runs" / run / "check_records.json"
    run = f"{format_name}_{condition}_s{seed}"
    return EXP_DIR / format_name / "evals" / run, EXP_DIR / format_name / "runs" / run / "check_records.json"


def checked_run_dir(check_path: Path, seed: int, n_steps: int) -> Path:
    check = json.loads(check_path.read_text())
    assert check["PASS"] is True, check_path
    assert check["n_files"] == n_steps, (check_path, check["n_files"])
    assert check["seed"] == seed, (check_path, check["seed"], seed)
    return Path(check["run_dir"])


def cot_one_sample_rows(cot_compare, eval_dir: Path) -> list[dict]:
    completions = cot_compare.read_jsonl(eval_dir / "completions.jsonl")
    rows = cot_compare.rollout_rows(eval_dir)
    assert [(c["question_id"], c["condition"]) for c in completions] == [(r["question_id"], r["condition"]) for r in rows]
    return [row for completion, row in zip(completions, rows) if completion["sample_idx"] == 0]


def assert_cue_conditions(compare, eval_dir: Path) -> None:
    """cue_gold prompts have the cue at the correct answer and cue_dist prompts at the other option (relabel_to_correct relies on it)."""
    for c in compare.read_jsonl(eval_dir / "completions.jsonl"):
        if c["condition"] != "none":
            assert (c["cue_letter"] == c["answer"]) == (c["condition"] == "cue_gold"), (eval_dir, c["question_id"], c["condition"])


def relabel_to_correct(rows: list[dict], format_name: str) -> list[dict]:
    """The rows scored as if every question's training label were its correct answer."""
    relabeled = []
    for row in rows:
        new_row = {**row, "matches_label": row["correct"]}
        if row["condition"] != "none":
            new_row["cue_at_label"] = row["condition"] == "cue_gold"
        if format_name == "nocot":
            new_row["p_label"] = row["p_correct"]
        relabeled.append(new_row)
    return relabeled


def condition_tables(format_name: str, condition: str, seeds: tuple[int, ...], smoke: bool, compare, nocot_compare) -> dict:
    """Per seed: the eval tables (base vs after RL), the same-reward share, the run dir."""
    n_steps = SMOKE_STEPS if smoke else N_STEPS
    if format_name == "cot":
        prompt_rows = lambda eval_dir: cot_one_sample_rows(compare, eval_dir)  # noqa: E731
    else:
        prompt_rows = compare.prompt_rows
    base_dir = SETUP_DIRS[format_name] / "evals" / "base"
    base_rows = prompt_rows(base_dir)
    if condition in CORRECT_LABEL_CONDITIONS:
        assert_cue_conditions(compare, base_dir)
        base_rows = relabel_to_correct(base_rows, format_name)
    out = {}
    for seed in seeds:
        final_dir, check_path = run_files(format_name, condition, seed, smoke)
        run_dir = checked_run_dir(check_path, seed, n_steps)
        final_rows = prompt_rows(final_dir)
        if condition in CORRECT_LABEL_CONDITIONS:
            assert_cue_conditions(compare, final_dir)
            final_rows = relabel_to_correct(final_rows, format_name)
        final_questions = {r["question_id"] for r in final_rows}
        run_base_rows = [r for r in base_rows if r["question_id"] in final_questions] if smoke else base_rows
        assert {r["question_id"] for r in run_base_rows} == final_questions, (condition, seed)
        records = nocot_compare.training_records(run_dir)
        assert len(records) == n_steps, (run_dir, len(records))
        out[seed] = {
            "questions": len(final_questions),
            "run_dir": str(run_dir),
            "final_eval": str(final_dir),
            "eval": compare.eval_tables({"base": run_base_rows, "after_rl": final_rows}),
            "same_reward_groups_mean": nocot_compare.mean([nocot_compare.same_reward_group_share(rows) for rows in records.values()]),
        }
    return out


def fmt(value) -> str:
    if value is None:
        return "–"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def mean_sd(values: list) -> tuple[float | None, float | None]:
    if any(v is None for v in values):
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def summary(condition: str, tables: dict, compare) -> dict:
    """Per metric and eval condition: the seeds' base and after-RL values with their mean and SD over the seeds.
    The training cue mix is left out for a condition trained without a cue."""
    seeds = list(tables)
    first = tables[seeds[0]]["eval"]
    out = {}
    for metric in compare.EVAL_METRICS:
        for eval_condition in compare.CONDITIONS:
            if metric not in first["all"]["base"][eval_condition]:
                continue
            if eval_condition == "train_mix" and condition in NO_CUE_CONDITIONS:
                continue
            base_values = [tables[seed]["eval"]["all"]["base"][eval_condition][metric] for seed in seeds]
            after_values = [tables[seed]["eval"]["all"]["after_rl"][eval_condition][metric] for seed in seeds]
            base_mean, base_sd = mean_sd(base_values)
            after_mean, after_sd = mean_sd(after_values)
            out.setdefault(metric, {})[eval_condition] = {
                "base": base_values, "base_mean": base_mean, "base_sd": base_sd,
                "after_rl": after_values, "after_rl_mean": after_mean, "after_rl_sd": after_sd,
            }
    same_reward = [tables[seed]["same_reward_groups_mean"] for seed in seeds]
    same_reward_mean, same_reward_sd = mean_sd(same_reward)
    out["same_reward_groups_mean"] = {"after_rl": same_reward, "after_rl_mean": same_reward_mean, "after_rl_sd": same_reward_sd}
    return out


def base_to_after(cell: dict | None) -> str:
    """'base → mean ± SD' (base: the mean of the seeds' base values); '–' for a blank cell."""
    if cell is None:
        return "–"
    sd = "" if cell["after_rl_sd"] is None else f" ± {fmt(cell['after_rl_sd'])}"
    return f"{fmt(cell['base_mean'])} → {fmt(cell['after_rl_mean'])}{sd}"


def metric_rows(stats: dict, compare, metrics: list[str]) -> list[str]:
    lines = []
    for metric in metrics:
        for eval_condition in compare.CONDITIONS:
            cells = [stats[condition].get(metric, {}).get(eval_condition) for condition in stats]
            if all(cell is None for cell in cells):
                continue
            name = DISPLAY_NAMES.get(metric, metric)
            lines.append(f"| {name} | {EVAL_CONDITION_NAMES[eval_condition]} | " + " | ".join(base_to_after(cell) for cell in cells) + " |")
    return lines


def table_markdown(format_name: str, tables: dict, stats: dict, compare, smoke: bool) -> list[str]:
    conditions = list(tables)
    header = "| metric | eval condition | " + " | ".join(TRAINING_CONDITIONS[condition] for condition in conditions) + " |"
    divider = "|---|---|" + "---|" * len(conditions)
    seeds_text = ", ".join(str(seed) for seed in tables[conditions[0]])
    lines = [
        f"# 261020 reference runs: {FORMAT_TITLES[format_name]}{' (smoke run)' if smoke else ''}",
        "",
        f"Each cell is base → after RL: the base model, then the final checkpoints' mean ± SD (n − 1) over seeds {seeds_text}. "
        f"Eval questions: {tables[conditions[0]][list(tables[conditions[0]])[0]]['questions']}. The training cue mix is each run's own: "
        "the cue at the noisy label 75% of the time for the baseline, at the correct answer 75% of the time for condition 3, "
        "and blank (–) for conditions 1 and 2, which were trained without a cue. Reward is agreement with the run's own "
        "training label (the noisy label for the baseline and condition 1, the correct answer for conditions 2 and 3). "
        "Bias and self-report rows count questions with a no-cue answer in both the base eval and the run's eval; their "
        "base value is the mean over the seeds' base values. \"deception\" is the probability-weighted measure "
        "(key deception_soft); \"deception (thresholded)\" is key deception.",
        "",
        "## True accuracy",
        "",
        header,
        divider,
        *metric_rows(stats, compare, ["accuracy"]),
        "",
        "## Reward on each run's own training prompts",
        "",
        "| training condition | eval condition | reward |",
        "|---|---|---|",
    ]
    for condition in conditions:
        own = OWN_TRAINING_PROMPTS[condition]
        lines.append(f"| {TRAINING_CONDITIONS[condition]} | {EVAL_CONDITION_NAMES[own]} | {base_to_after(stats[condition]['reward'][own])} |")
    other_metrics = [m for m in compare.EVAL_METRICS if m != "accuracy"]
    lines += ["", "## All other metrics", "", header, divider, *metric_rows(stats, compare, other_metrics), ""]
    steps = SMOKE_STEPS if smoke else N_STEPS
    lines += [
        f"## Training: share of groups whose 8 rewards are all equal, mean over the {steps} steps",
        "",
        "| training condition | " + " | ".join(f"seed {seed}" for seed in tables[conditions[0]]) + " | mean | SD |",
        "|---|" + "---|" * len(tables[conditions[0]]) + "---|---|",
    ]
    for condition in conditions:
        same_reward = stats[condition]["same_reward_groups_mean"]
        lines.append(
            f"| {TRAINING_CONDITIONS[condition]} | " + " | ".join(fmt(v) for v in same_reward["after_rl"])
            + f" | {fmt(same_reward['after_rl_mean'])} | {fmt(same_reward['after_rl_sd'])} |"
        )
    lines += ["", "## Per seed", ""]
    for condition in conditions:
        seeds = list(tables[condition])
        lines += [
            f"### {TRAINING_CONDITIONS[condition]}",
            "",
            "Questions with a no-cue answer in both the base eval and the seed's eval: "
            + ", ".join(f"seed {seed} {tables[condition][seed]['eval']['questions_with_nocue_answer_in_every_checkpoint']}" for seed in seeds)
            + ".",
            "",
            "| metric | eval condition | base | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean | SD |",
            "|---|---|---|" + "---|" * len(seeds) + "---|---|",
        ]
        for metric, by_condition in stats[condition].items():
            if metric == "same_reward_groups_mean":
                continue
            for eval_condition, row in by_condition.items():
                base_values = row["base"]
                base_cell = fmt(base_values[0]) if len(set(base_values)) == 1 else " / ".join(fmt(v) for v in base_values)
                cells = [DISPLAY_NAMES.get(metric, metric), EVAL_CONDITION_NAMES[eval_condition], base_cell,
                         *(fmt(v) for v in row["after_rl"]), fmt(row["after_rl_mean"]), fmt(row["after_rl_sd"])]
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=sorted(SETUP_DIRS), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    compare = load_compare(args.format)
    nocot_compare = compare if args.format == "nocot" else load_compare("nocot")
    if args.smoke:
        conditions, seeds, out_dir, stem = ["correct_nocue"], (42,), EXP_DIR / args.format / "dry", "condition_table_smoke"
    else:
        conditions, seeds, out_dir, stem = list(TRAINING_CONDITIONS), SEEDS, EXP_DIR / args.format, "condition_table"
    tables = {condition: condition_tables(args.format, condition, seeds, args.smoke, compare, nocot_compare) for condition in conditions}
    stats = {condition: summary(condition, tables[condition], compare) for condition in conditions}

    md = table_markdown(args.format, tables, stats, compare, args.smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.md").write_text("\n".join(md))
    (out_dir / f"{stem}.json").write_text(json.dumps({"format": args.format, "runs": tables, "mean_sd": stats}, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
