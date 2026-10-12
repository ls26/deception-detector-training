"""Put the rerun next to August's run (260825_grpo_det0, W&B lx6gcbcl): numbers only.

Rows: in-loop eval accuracy and cue-following logged at steps 30/60/90/120 (each scores the
policy from the previous eval point; the first row scores the starting model), the
post-training eval (accuracy, cue-following, parse failures; one sample per held-out
question), the first training step whose train cue-following reaches 0.95 (from the
response checker's recomputed counts), and the trainer's wall-clock time. No tolerance:
training prompt order and vLLM sampling are unseeded in both codebases, so the two runs cannot
match exactly. Run with run_env.sh sourced (imports the copy's launcher for the eval history).

Usage:
    python compare_august.py            # writes comparison.md and comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

from training.grpo_launcher import sycophancy_eval_history

RUN_DIR = Path(__file__).resolve().parent
AUGUST_RUN_ID = "lx6gcbcl"
AUGUST_METRICS = Path("/workspace/sycophancy_experiments/260825_grpo_det0/run_metrics_syco_det0_s42.json")
RERUN_METRICS = RUN_DIR / "run_metrics_syco_det0_rerun_s42.json"
ROLLOUTS_PER_STEP = 512
FOLLOW_THRESHOLD = 0.95


def first_step_following(check_file: Path) -> int | None:
    """First W&B training step whose recomputed train cue-following is >= FOLLOW_THRESHOLD."""
    checked = json.loads(check_file.read_text())
    assert checked["PASS"] and checked["n_files_checked"] == checked["n_files"] == 128
    for step in checked["per_step"]:
        if step["recomputed_counts"]["rewards_metrics/followed_cue"] / ROLLOUTS_PER_STEP >= FOLLOW_THRESHOLD:
            return step["wandb_train_step"]
    return None


def summary(metrics_file: Path, eval_history: list[dict], check_file: Path) -> dict:
    metrics = json.loads(metrics_file.read_text())["metrics"]
    row = {}
    for entry in eval_history:
        row[f"in-loop eval accuracy, logged at step {entry['logged_at_train_step']}"] = entry["eval/rewards/task_success_mean"]
        row[f"in-loop eval cue-following, logged at step {entry['logged_at_train_step']}"] = entry["eval/rewards_metrics/followed_cue_mean"]
    row["final eval accuracy"] = metrics["final_eval/rewards/task_success"]
    row["final eval cue-following"] = metrics["final_eval/rewards_metrics/followed_cue"]
    row["final eval parse failures"] = metrics["final_eval/rewards_metrics/parse_failed"]
    row[f"first step with train cue-following >= {FOLLOW_THRESHOLD}"] = first_step_following(check_file)
    row["trainer wall-clock (h)"] = metrics["duration_seconds"] / 3600
    return row


def main() -> None:
    august = summary(AUGUST_METRICS, sycophancy_eval_history(AUGUST_RUN_ID), RUN_DIR / "check_august.json")
    rerun_export = json.loads(RERUN_METRICS.read_text())
    rerun = summary(RERUN_METRICS, rerun_export["sycophancy_eval_history"], RUN_DIR / "check_rerun.json")
    assert august.keys() == rerun.keys(), f"rows differ: {august.keys() ^ rerun.keys()}"

    def cell(value) -> str:
        return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))

    lines = [
        "# Rerun of August's sycophancy baseline from the code copy vs August's run",
        "",
        f"August: W&B `{AUGUST_RUN_ID}` (260825_grpo_det0). Rerun: W&B `{rerun_export['run_id']}`.",
        "",
        "| metric | August | rerun |",
        "|---|---|---|",
        *[f"| {name} | {cell(august[name])} | {cell(rerun[name])} |" for name in august],
        "",
        "In-loop eval rows score the policy from the previous eval point (the row logged at step 30",
        "scores the starting model): the trainer's eval queue is first-in-first-out with one",
        "submission before training. Final eval: one sample per held-out question, temperature 1.",
    ]
    (RUN_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    (RUN_DIR / "comparison.json").write_text(
        json.dumps({"august_run_id": AUGUST_RUN_ID, "rerun_run_id": rerun_export["run_id"], "august": august, "rerun": rerun}, indent=2) + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
