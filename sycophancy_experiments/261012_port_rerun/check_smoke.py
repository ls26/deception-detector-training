"""Check that the 2-step smoke run went through every stage of the trainer and the launcher.

Reads the smoke run's launcher log, its exported metrics, its run directory and the response
checker's result (check_responses_smoke.json), and checks:
  - the launcher exited 0 on the first attempt (log ends in "EXIT 0", no "CRASH:" line);
  - the trainer and launcher printed the marker of every stage (STAGE_MARKERS);
  - the six post-training probe evaluations left their folders in the run directory;
  - the export has the final-eval numbers and in-loop eval rows for training steps 1 and 2;
  - the response checker passed on both step files;
  - the IMPORT_CHECK lines that run_grpo_step.sh and start_vllm.sh print (a Python started
    like the trainer / the server) show obfuscation_atlas and afterburner from the code copy,
    and neither log mentions a file of the private package.

Usage:
    python check_smoke.py --log logs/smoke_<host>.log --vllm-log logs/vllm_<host>.log [--out check_smoke.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
COPY = Path("/workspace/sycophancy-rl")
RUNS_BASE = Path("/workspace/results/run_results")
RUN_KEY = "syco_det0_smoke_s42"
GROUP = "261012_port_rerun_smoke"
STAGE_MARKERS = [
    "OK: MCQ prompts unfiltered (train=1603, eval=64)",  # sycophancy data wiring
    "Phase-0 detector results cache path",  # probe trained before RL
    "verified complete (responses=2)",  # both RL steps wrote their response files
    "OK: exported metrics",  # W&B export
    "OK: added 2 in-loop eval rows",  # in-loop eval after steps 1 and 2
]
POST_TRAINING_EVAL_FOLDERS = [
    "initial_detector.base_model.on_dataset",
    "initial_detector.base_model.on_policy",
    "initial_detector.final_model.on_dataset",
    "initial_detector.final_model.on_policy",
    "fresh_detector.final_model.on_dataset",
    "fresh_detector.final_model.on_policy",
]
PRIVATE_PACKAGE_PATHS = ["/workspace/obfuscation-atlas/obfuscation_atlas/", "/workspace/obfuscation-atlas/third_party/"]
FINAL_EVAL_KEYS = [
    "final_eval/rewards/task_success",
    "final_eval/rewards_metrics/followed_cue",
    "final_eval/rewards_metrics/parse_failed",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--vllm-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RUN_DIR / "check_smoke.json")
    args = parser.parse_args()

    log = args.log.read_text()
    checks = {}
    checks["exit_0"] = log.rstrip().splitlines()[-1].strip() == "EXIT 0"
    checks["no_crash_retry"] = "CRASH:" not in log
    for marker in STAGE_MARKERS:
        checks[f"marker: {marker}"] = marker in log

    exported = json.loads((RUN_DIR / f"run_metrics_{RUN_KEY}.json").read_text())
    run_id = exported["run_id"]
    run_dir = RUNS_BASE / GROUP / f"{RUN_KEY}_{run_id}"
    for folder in POST_TRAINING_EVAL_FOLDERS:
        checks[f"run dir has {folder}"] = (run_dir / folder).is_dir()
    for key in FINAL_EVAL_KEYS:
        checks[f"export has {key}"] = key in exported["metrics"]
    checks["export has eval rows logged at steps 1 and 2"] = [
        row["logged_at_train_step"] for row in exported["sycophancy_eval_history"]
    ] == [1, 2]

    responses = json.loads((RUN_DIR / "check_responses_smoke.json").read_text())
    checks["response checker passed on both files"] = (
        responses["PASS"] is True and responses["n_files_checked"] == responses["n_files"] == 2 and responses["wandb_run_id"] == run_id
    )

    vllm_log = args.vllm_log.read_text()
    imported = [line.split(maxsplit=1)[1] for line in log.splitlines() if line.startswith("IMPORT_CHECK ")]
    vllm_imported = [line.split(maxsplit=1)[1] for line in vllm_log.splitlines() if line.startswith("IMPORT_CHECK ")]
    checks["trainer environment imports obfuscation_atlas and afterburner from the copy"] = len(imported) == 2 and all(
        path.startswith(str(COPY)) for path in imported
    )
    checks["vLLM server imports afterburner from the copy"] = len(vllm_imported) == 1 and vllm_imported[0].startswith(str(COPY))
    checks["no private-package file in the smoke or vLLM log"] = not any(path in log or path in vllm_log for path in PRIVATE_PACKAGE_PATHS)

    passed = all(checks.values())
    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    result = {"PASS": passed, "run_id": run_id, "run_dir": str(run_dir), "imported_from": imported + vllm_imported, "checks": checks}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if passed else 'FAIL'}: wrote {args.out}")


if __name__ == "__main__":
    main()
