"""Several RL runs side by side: base vs after RL on all eval questions for each run, with the mean and SD (n - 1)
over the runs of the base values and of the after-RL values.

    python seed_table.py --base <experiment>/evals/base \
        --run <label> <final eval dir> <GRPO run dir> [--run ...] --out-dir DIR
        -> DIR/seed_table.md, DIR/seed_table.json

261018's seed_table.py for runs named on the command line (each a 128-step run; its final eval must be of its
final checkpoint). Each run's eval tables are compare.py's (one sampled answer per prompt). Bias and self-report
rows count questions with a no-cue answer in both the base eval and that run's eval, so their base value can
differ per run; the base column then shows one value per run.
Training: the share of training groups whose 8 task rewards are all equal and the share of probe-only groups (all 8
answers name the cued option; compare.py), each the mean over the steps, from each run's step records. seed_table.json holds each run's tables under "runs" and the means and SDs under "mean_sd".
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import compare

N_STEPS = 128
DISPLAY_NAMES = {"followed_cue": "cue agreement rate"}
TRAINING_KEYS = {"same_reward_groups_mean": "equal task reward groups", "probe_only_groups_mean": "probe-only groups"}


def run_tables(base_dir: Path, runs: list[tuple[str, Path, Path]]) -> dict:
    """Per run: the eval tables (base vs after RL), the equal task reward and probe-only group shares, the run dir."""
    base_rows = compare.rollout_rows(base_dir)
    out = {}
    for label, final_dir, run_dir in runs:
        final_rows = compare.rollout_rows(final_dir)
        assert {r["question_id"] for r in base_rows} == {r["question_id"] for r in final_rows}, label
        adapters = {c["adapter"] for c in compare.read_jsonl(final_dir / "completions.jsonl")}
        assert adapters == {str(run_dir / "checkpoints" / "final")}, (label, adapters)  # the eval is this run's final checkpoint
        records = compare.training_records(run_dir)
        assert len(records) == N_STEPS, (label, len(records))
        out[label] = {
            "questions": len({r["question_id"] for r in base_rows}),
            "run_dir": str(run_dir),
            "final_eval": str(final_dir),
            "eval": compare.eval_tables({"base": base_rows, "after_rl": final_rows}),
            "same_reward_groups_mean": compare.mean([compare.same_reward_group_share(rows) for rows in records.values()]),
            "probe_only_groups_mean": compare.mean([compare.probe_only_group_share(rows) for rows in records.values()]),
        }
    return out


def mean_sd(values: list) -> tuple[float | None, float | None]:
    if any(v is None for v in values):
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None)


def summary(tables: dict) -> dict:
    """Per metric and condition: the runs' base and after-RL values with their mean and SD over the runs."""
    labels = list(tables)
    first = tables[labels[0]]["eval"]
    out = {}
    for metric in compare.EVAL_METRICS:
        for condition in compare.CONDITIONS:
            if metric not in first["all"]["base"][condition]:
                continue
            base_values = [tables[label]["eval"]["all"]["base"][condition][metric] for label in labels]
            after_values = [tables[label]["eval"]["all"]["after_rl"][condition][metric] for label in labels]
            base_mean, base_sd = mean_sd(base_values)
            after_mean, after_sd = mean_sd(after_values)
            out.setdefault(metric, {})[condition] = {
                "base": base_values, "base_mean": base_mean, "base_sd": base_sd,
                "after_rl": after_values, "after_rl_mean": after_mean, "after_rl_sd": after_sd,
            }
    for key in TRAINING_KEYS:
        values = [tables[label][key] for label in labels]
        values_mean, values_sd = mean_sd(values)
        out[key] = {"after_rl": values, "after_rl_mean": values_mean, "after_rl_sd": values_sd}
    return out


def table_markdown(tables: dict, stats: dict) -> list[str]:
    labels = list(tables)
    lines = [
        f"Eval questions: {tables[labels[0]]['questions']}. Questions with a no-cue answer in both the base eval and the run's eval "
        "(bias and self-report rows use these): "
        + ", ".join(f"{label} {tables[label]['eval']['questions_with_nocue_answer_in_every_checkpoint']}" for label in labels)
        + ".",
        "",
        "| metric | condition | base | base mean | base SD | " + " | ".join(labels) + " | mean | SD |",
        "|---|---|---|---|---|" + "---|" * len(labels) + "---|---|",
    ]
    for metric, by_condition in stats.items():
        if metric in TRAINING_KEYS:
            continue
        for condition, row in by_condition.items():
            base_values = row["base"]
            base_cell = compare.fmt(base_values[0]) if len(set(base_values)) == 1 else " / ".join(compare.fmt(v) for v in base_values)
            cells = [DISPLAY_NAMES.get(metric, metric), condition, base_cell, compare.fmt(row["base_mean"]), compare.fmt(row["base_sd"]),
                     *(compare.fmt(v) for v in row["after_rl"]), compare.fmt(row["after_rl_mean"]), compare.fmt(row["after_rl_sd"])]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Training, mean over the {N_STEPS} steps: the share of groups whose 8 task rewards are all equal, and the share of "
        "probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).",
        "",
        "| | " + " | ".join(labels) + " | mean | SD |",
        "|---|" + "---|" * len(labels) + "---|---|",
    ]
    for key, name in TRAINING_KEYS.items():
        row = stats[key]
        lines.append(f"| {name} | " + " | ".join(compare.fmt(v) for v in row["after_rl"])
                     + f" | {compare.fmt(row['after_rl_mean'])} | {compare.fmt(row['after_rl_sd'])} |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="the base model's eval folder")
    parser.add_argument("--run", nargs=3, action="append", required=True, metavar=("LABEL", "FINAL_EVAL", "RUN_DIR"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = [(label, Path(final_dir), Path(run_dir)) for label, final_dir, run_dir in args.run]
    tables = run_tables(args.base, runs)
    stats = summary(tables)
    md = [
        "# RL runs side by side",
        "",
        "Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run "
        "column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL "
        "values (SD with n - 1). All eval questions, one sampled answer per prompt.",
        "",
        *table_markdown(tables, stats),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seed_table.md").write_text("\n".join(md))
    (args.out_dir / "seed_table.json").write_text(json.dumps({"runs": tables, "mean_sd": stats}, indent=2) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
