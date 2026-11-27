# Tables of the 261126 GPU tests, written to results_tables.md (read only from this folder's logs and memory/memory.json,
# and the runs' W&B histories, project deception_detection).
# Per run (rl log logs/rl_<run>_a1_<host>.log; MBPP's baseline is the first 3 steps of 261125's 300-step H100 run
# fzmbpp9b_s43, settings off, its rl log in that folder):
#   - per step: the trainer's timings for the step (generation, rewards with probe scoring, reference pass, policy
#     pass, backward, the whole training pass), the time between consecutive "Step N timings" lines, and from W&B the
#     trainer-vs-vLLM log-prob correlation (logprob_corr_mean), the padding share and the longest answer;
#   - before step 1: from the trainer's first log line to step 1's start (step 1's logged time minus its duration);
#   - after the last step: the launcher's "GRPO subprocess finished in X min" minus the time from the trainer's first
#     log line to "Training completed!"; this also holds the trainer's imports before its first log line;
#   - peak GPU memory by nvidia-smi (vLLM and PyTorch's cache included, so an upper bound; GiB of the H100's 79.6) from
#     logs/gpu_mem_<run>_<host>.log.
# Then the probe penalty check for this folder's runs: W&B's rewards/detector (the step's mean penalty over all its
# answers) against the penalty recomputed from the run's saved answers. Sycophancy (weight 1.0): only finished answers
# that give the cued letter are penalized, so it is -1.0 x their summed probe score / the step's answer count, from
# checkpoints/responses/step_<N - 1>.jsonl. MBPP (weight 3; its saved answers hold no probe score): every answer is
# penalized, so it is -3 x the step's mean probe score, W&B's rewards_metrics/detector_score.
# Usage: python results.py [host]   (host defaults to this pod's hostname)
import ast
import json
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

import wandb

FOLDER = Path(__file__).resolve().parent
HOST = sys.argv[1] if len(sys.argv) > 1 else socket.gethostname()
# (name, rl log, nvidia-smi memory log or None)
FOLDER_RUNS = ["lmp_on4", "lmp_off4", "lmp_on8", "lmp_on4_main", "lmp_mbpp_on4"]
RUNS = [(run, FOLDER / "logs" / f"rl_{run}_a1_{HOST}.log", FOLDER / "logs" / f"gpu_mem_{run}_{HOST}.log")
        for run in FOLDER_RUNS]
RUNS.append(("fzmbpp9b_s43 (261125, settings off; steps 1-3 of 300, H100 pod 7c244d9935a2)",
             Path("/workspace/qwen_experiment/261125_9b_frozen_probe_300/seed_43/logs/rl_fzmbpp9b_s43_a1_7c244d9935a2.log"),
             None))
SHOWN_STEPS = 3
RUN_RESULTS = Path("/workspace/results/run_results/261126_low_memory_passes")
MBPP_PENALTY_WEIGHT = {"lmp_mbpp_on4": 3}  # every other run of this folder is sycophancy, penalty weight 1.0
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]")
TIMINGS = ["generate_responses", "compute_rewards", "reference_logprobs_with_activation", "policy_logprobs", "backward",
           "train_on_generation_batch"]


def stamp(line: str) -> datetime:
    return datetime.strptime(STAMP.match(line).group(1), "%Y-%m-%d %H:%M:%S")


def run_numbers(log_path: Path, memory_log: Path | None) -> dict | None:
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    stamped = [line for line in lines if STAMP.match(line)]
    first = stamp(stamped[0])
    steps = {}
    for line in stamped:
        found = re.search(r"Step (\d+) timings: (\{.*\})", line)
        if found:
            timings = ast.literal_eval(found.group(2))
            steps[int(found.group(1))] = {
                "logged": stamp(line),
                **{name: timings[f"profiling/time_per_optimizer_step: {name}"] for name in TIMINGS},
            }
    numbers = {"exit": next((line for line in reversed(lines) if line.startswith("EXIT")), "no EXIT line"),
               "out_of_memory": any("OutOfMemoryError" in line for line in lines), "steps": steps}
    if 1 in steps:
        step1_start = steps[1]["logged"].timestamp() - steps[1]["train_on_generation_batch"] - steps[1]["generate_responses"]
        numbers["before_step_1_min"] = (step1_start - first.timestamp()) / 60
    completed = [stamp(line) for line in stamped if "Training completed!" in line]
    finished = [float(m.group(1)) for line in lines if (m := re.search(r"GRPO subprocess finished in ([\d.]+) min", line))]
    if completed and finished:
        numbers["after_last_step_min"] = finished[0] - (completed[0] - first).total_seconds() / 60
        numbers["subprocess_min"] = finished[0]
    run_ids = [m.group(1) for line in lines if (m := re.search(r"/runs/([a-z0-9]{8})", line))]
    if run_ids:
        numbers["run_id"] = run_ids[0]
        api = wandb.Api()
        keys = ["grpo_train_step", "logprob_corr_mean", "padding_share", "response_length_max"]
        # scan_history reads every row with these keys (history() samples 500 rows, and a run's probe fit logs more)
        rows = api.run(f"{api.default_entity}/deception_detection/{run_ids[0]}").scan_history(keys=keys)
        numbers["wandb"] = {int(row["grpo_train_step"]): {key: row[key] for key in keys[1:]}
                            for row in rows if row["grpo_train_step"] is not None}
    if memory_log is not None and memory_log.exists():
        used = [int(line.split(",")[1].split()[0]) for line in memory_log.read_text().splitlines() if "," in line]
        numbers["peak_gpu_memory_GiB"] = max(used) / 1024  # nvidia-smi reports MiB
    return numbers


lines = ["# 261126 GPU tests: tables (written by results.py)", ""]
memory = json.loads((FOLDER / "memory" / "memory.json").read_text())
lines += ["## Memory test (memory/memory.json)", "", "| Setting | Peak reserved GiB | Peak allocated GiB | Free GiB after the pass |",
          "|---|---|---|---|"]
for key, value in memory.items():
    if key.startswith("low_memory_passes"):
        if isinstance(value, dict):
            lines.append(f"| {key} | {value['peak_reserved_GiB']:.2f} | {value['peak_allocated_GiB']:.2f} | "
                         f"{value['free_after_pass_GiB']:.2f} |")
        else:
            lines.append(f"| {key} | {value} | | |")
lines += ["", f"Free GPU memory before the test started CUDA: {memory['free_before_cuda_GiB']:.2f} GiB; after: "
          f"{memory['free_after_cuda_init_GiB']:.2f} GiB; model loaded: {memory['model_loaded_allocated_GiB']:.2f} GiB.", ""]
for key, value in memory.items():
    if key.startswith("numbers"):
        lines.append(f"- {key}: {value}")
lines += ["", "## 3-step runs", ""]
all_numbers = {}
for run, log_path, memory_log in RUNS:
    numbers = run_numbers(log_path, memory_log)
    all_numbers[run] = numbers
    if numbers is None:
        lines += [f"### {run}: not run", ""]
        continue
    lines += [f"### {run} (W&B {numbers.get('run_id')}; {numbers['exit']}; out of memory: {numbers['out_of_memory']})", "",
              "| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | "
              "Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for step, values in sorted(numbers["steps"].items())[:SHOWN_STEPS]:
        previous = numbers["steps"].get(step - 1)
        between = f"{(values['logged'] - previous['logged']).total_seconds():.0f}" if previous else ""
        logged = numbers.get("wandb", {}).get(step, {})
        lines.append(f"| {step} | " + " | ".join(f"{values[name]:.1f}" for name in TIMINGS) + f" | {between} | "
                     f"{logged.get('logprob_corr_mean')} | {logged.get('padding_share')} | {logged.get('response_length_max')} |")
    lines += ["", f"Before step 1: {numbers.get('before_step_1_min', float('nan')):.1f} min; after the last step: "
              f"{numbers.get('after_last_step_min', float('nan')):.1f} min; trainer subprocess "
              f"{numbers.get('subprocess_min', float('nan')):.1f} min; peak GPU memory by nvidia-smi "
              f"{numbers.get('peak_gpu_memory_GiB', float('nan')):.1f} of 79.6 GiB.", ""]

lines += ["## Probe penalty check (W&B rewards/detector against the run's saved answers)", "",
          "| Run | Step | Answers in the step file | Penalty recomputed | W&B rewards/detector |", "|---|---|---|---|---|"]
api = wandb.Api()
for run in FOLDER_RUNS:
    run_id = all_numbers[run]["run_id"]
    rows = api.run(f"{api.default_entity}/deception_detection/{run_id}").scan_history(
        keys=["grpo_train_step", "rewards/detector", "rewards_metrics/detector_score"])
    logged = {int(row["grpo_train_step"]): row for row in rows if row["grpo_train_step"] is not None}
    for step in range(1, SHOWN_STEPS + 1):
        step_file = RUN_RESULTS / f"{run}_{run_id}" / "checkpoints" / "responses" / f"step_{step - 1}.jsonl"
        answers = [json.loads(line) for line in open(step_file)]
        if run in MBPP_PENALTY_WEIGHT:
            recomputed = -MBPP_PENALTY_WEIGHT[run] * logged[step]["rewards_metrics/detector_score"]
        else:
            recomputed = -1.0 * sum(answer["probe_score"] for answer in answers
                                    if answer["parsed_answer"] == answer["cue_letter"] and not answer["cut_off"]) / len(answers)
        lines.append(f"| {run} | {step} | {len(answers)} | {recomputed:.5f} | {logged[step]['rewards/detector']:.5f} |")
lines.append("")
(FOLDER / "results_tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
