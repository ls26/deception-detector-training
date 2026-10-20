"""The seed table: base vs after RL on all eval questions for each setup's seeds, side by side, with the mean and
SD (n - 1) over the seeds of the base values and of the after-RL values.

    python seed_table.py [--seeds 42 43 44] [--out-dir .]   -> seed_table.md, seed_table.json

Each seed's eval tables are computed with the setup's own compare.py functions (unchanged, loaded by path) on
base = the setup folder's evals/base and after RL = evals/final there (seed 42) or this folder's evals/<run>:
  no-CoT  261014's prompt_rows + eval_tables (one prompt per question and condition, answered by its top letter);
  CoT     261013's rollout_rows + eval_tables on one sampled answer per prompt, the first of each prompt's saved
          samples, as 261014's compare.py does for seed 42 (Decision 8). The no-cue answer behind bias stays the
          eval's own (at least 5 of 8 no-cue samples).
Bias and self-report rows count questions with a no-cue answer in both the base eval and that seed's eval, so
their base value can differ per seed; the base column then shows one value per seed. Training: the share of
training groups whose 8 rewards are all equal, mean over the steps, from each run's step records (261014's
compare.py functions). Each run's record-check json must say PASS, 128 files and the run's own seed.
seed_table.json holds each seed's tables under the setup name and the means and SDs under "mean_sd".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SETUP_DIRS = {
    "cot": Path("/workspace/sycophancy_experiments/261013_noisy_label_rl"),
    "nocot": Path("/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl"),
}
SETUP_TITLES = {
    "cot": "CoT (261013's setup; one sampled answer per prompt)",
    "nocot": "No-CoT (261014's setup; top letter)",
}
N_STEPS = 128
DISPLAY_NAMES = {"followed_cue": "cue agreement rate"}


def load_compare(setup: str):
    setup_dir = SETUP_DIRS[setup]
    sys.path.append(str(setup_dir))  # each compare.py imports its own spec module (noisy_spec / nocot_spec)
    module_spec = importlib.util.spec_from_file_location(f"{setup}_compare", setup_dir / "compare.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def run_files(setup: str, seed: int) -> tuple[Path, Path]:
    """(final eval folder, record-check json) of one run."""
    if seed == 42:
        return SETUP_DIRS[setup] / "evals" / "final", SETUP_DIRS[setup] / "check_records_run.json"
    run = f"{setup}_s{seed}"
    return EXP_DIR / "evals" / run, EXP_DIR / "runs" / run / "check_records.json"


def checked_run_dir(check_path: Path, seed: int) -> Path:
    check = json.loads(check_path.read_text())
    assert check["PASS"] is True, check_path
    assert check["n_files"] == N_STEPS, (check_path, check["n_files"])
    assert check["seed"] == seed, (check_path, check["seed"], seed)
    return Path(check["run_dir"])


def cot_one_sample_rows(cot_compare, eval_dir: Path) -> list[dict]:
    completions = cot_compare.read_jsonl(eval_dir / "completions.jsonl")
    rows = cot_compare.rollout_rows(eval_dir)
    assert [(c["question_id"], c["condition"]) for c in completions] == [(r["question_id"], r["condition"]) for r in rows]
    return [row for completion, row in zip(completions, rows) if completion["sample_idx"] == 0]


def setup_tables(setup: str, seeds: list[int], compare, nocot_compare) -> dict:
    """Per seed: the eval tables (base vs after RL), the same-reward share, the run dir."""
    if setup == "cot":
        prompt_rows = lambda eval_dir: cot_one_sample_rows(compare, eval_dir)  # noqa: E731
    else:
        prompt_rows = compare.prompt_rows
    base_rows = prompt_rows(SETUP_DIRS[setup] / "evals" / "base")
    out = {}
    for seed in seeds:
        final_dir, check_path = run_files(setup, seed)
        run_dir = checked_run_dir(check_path, seed)
        final_rows = prompt_rows(final_dir)
        assert {r["question_id"] for r in base_rows} == {r["question_id"] for r in final_rows}, (setup, seed)
        records = nocot_compare.training_records(run_dir)
        assert len(records) == N_STEPS, (run_dir, len(records))
        out[seed] = {
            "questions": len({r["question_id"] for r in base_rows}),
            "run_dir": str(run_dir),
            "final_eval": str(final_dir),
            "eval": compare.eval_tables({"base": base_rows, "after_rl": final_rows}),
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


def summary(tables: dict, compare) -> dict:
    """Per metric and condition: the seeds' base and after-RL values with their mean and SD over the seeds."""
    seeds = list(tables)
    first = tables[seeds[0]]["eval"]
    out = {}
    for metric in compare.EVAL_METRICS:
        for condition in compare.CONDITIONS:
            if metric not in first["all"]["base"][condition]:
                continue
            base_values = [tables[seed]["eval"]["all"]["base"][condition][metric] for seed in seeds]
            after_values = [tables[seed]["eval"]["all"]["after_rl"][condition][metric] for seed in seeds]
            base_mean, base_sd = mean_sd(base_values)
            after_mean, after_sd = mean_sd(after_values)
            out.setdefault(metric, {})[condition] = {
                "base": base_values, "base_mean": base_mean, "base_sd": base_sd,
                "after_rl": after_values, "after_rl_mean": after_mean, "after_rl_sd": after_sd,
            }
    same_reward = [tables[seed]["same_reward_groups_mean"] for seed in seeds]
    same_reward_mean, same_reward_sd = mean_sd(same_reward)
    out["same_reward_groups_mean"] = {"after_rl": same_reward, "after_rl_mean": same_reward_mean, "after_rl_sd": same_reward_sd}
    return out


def setup_markdown(setup: str, tables: dict, stats: dict) -> list[str]:
    seeds = list(tables)
    lines = [
        f"## {SETUP_TITLES[setup]}",
        "",
        f"Eval questions: {tables[seeds[0]]['questions']}. Questions with a no-cue answer in both the base eval and the seed's eval (bias and "
        "self-report rows use these): "
        + ", ".join(f"seed {seed} {tables[seed]['eval']['questions_with_nocue_answer_in_every_checkpoint']}" for seed in seeds)
        + ".",
        "",
        "| metric | condition | base | base mean | base SD | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean | SD |",
        "|---|---|---|---|---|" + "---|" * len(seeds) + "---|---|",
    ]
    for metric, by_condition in stats.items():
        if metric == "same_reward_groups_mean":
            continue
        for condition, row in by_condition.items():
            base_values = row["base"]
            base_cell = fmt(base_values[0]) if len(set(base_values)) == 1 else " / ".join(fmt(v) for v in base_values)
            cells = [DISPLAY_NAMES.get(metric, metric), condition, base_cell, fmt(row["base_mean"]), fmt(row["base_sd"]),
                     *(fmt(v) for v in row["after_rl"]), fmt(row["after_rl_mean"]), fmt(row["after_rl_sd"])]
            lines.append("| " + " | ".join(cells) + " |")
    same_reward = stats["same_reward_groups_mean"]["after_rl"]
    mean, sd = stats["same_reward_groups_mean"]["after_rl_mean"], stats["same_reward_groups_mean"]["after_rl_sd"]
    lines += [
        "",
        f"Training: share of groups whose 8 rewards are all equal, mean over the {N_STEPS} steps.",
        "",
        "| | " + " | ".join(f"seed {seed}" for seed in seeds) + " | mean | SD |",
        "|---|" + "---|" * len(seeds) + "---|---|",
        "| same-reward groups | " + " | ".join(fmt(v) for v in same_reward) + f" | {fmt(mean)} | {fmt(sd)} |",
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out-dir", type=Path, default=EXP_DIR)
    args = parser.parse_args()

    compares = {setup: load_compare(setup) for setup in SETUP_DIRS}
    tables = {setup: setup_tables(setup, args.seeds, compares[setup], compares["nocot"]) for setup in SETUP_DIRS}
    stats = {setup: summary(tables[setup], compares[setup]) for setup in SETUP_DIRS}

    md = [
        "# 261018 noisy-label RL, seeds side by side",
        "",
        "Base = the base model (one eval per setup, shared by the seeds; on bias and self-report rows one value per "
        "seed, see the header of each table). seed N = that run's final checkpoint. Base mean / SD are over the "
        "seeds' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions.",
        "",
    ]
    for setup in SETUP_DIRS:
        md += setup_markdown(setup, tables[setup], stats[setup])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seed_table.md").write_text("\n".join(md))
    (args.out_dir / "seed_table.json").write_text(json.dumps({**tables, "mean_sd": stats}, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
