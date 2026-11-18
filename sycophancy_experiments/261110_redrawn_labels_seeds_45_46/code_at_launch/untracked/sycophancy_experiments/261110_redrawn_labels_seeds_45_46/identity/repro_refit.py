"""Reproduction test of 261110's rounds.py against 261109's seed 43: build 2-5, config 2, table 5.

The new rounds.py is imported unchanged; only run_key (old run names) and run_dir(1) (261103's round-1 folder) are
patched, and a scratch line folder links in 261103's / 261104's round-1 files and 261109's rounds 2-5.

Usage (the unattended queue's Start check): $PY repro_refit.py SCRATCH -> SCRATCH/result.json, with SCRATCH written as
<scratch>, so it can be compared byte for byte with refit_expected.json (written at setup on 2026-11-10).
"""

import filecmp
import json
import sys
from pathlib import Path

import yaml

NEW = Path("/workspace/sycophancy_experiments/261110_redrawn_labels_seeds_45_46")
E9 = Path("/workspace/sycophancy_experiments/261109_refit_probe_redrawn_labels")
E4 = Path("/workspace/sycophancy_experiments/261104_iterative_kl_0_01")
E3 = Path("/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0")
R1_RUN = Path("/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle")
SCRATCH = Path(sys.argv[1])
LINE = SCRATCH / "seed_43"

sys.path.insert(0, str(NEW))
import rounds  # noqa: E402

old_run_dir = rounds.run_dir


def run_key(round_number):
    return "kl_weight_1_0_s43" if round_number == 1 else f"s43_refit_round{round_number}"


def run_dir(round_number):
    return R1_RUN if round_number == 1 else old_run_dir(round_number)


rounds.run_key = run_key
rounds.run_dir = run_dir
rounds.GROUP = E9.name
rounds.TEMPLATE_PATH = E9 / "round_template.yaml"
rounds.EXP_DIR = LINE
rounds.SEED = 43

# scratch line folder
for sub in ("evals", "generations", "probe_sets"):
    (LINE / sub).mkdir(parents=True)
(LINE / "evals" / "kl_weight_1_0_s43").symlink_to(E3 / "evals" / "kl_weight_1_0_s43")
(LINE / "evals" / "base_rescored_kl_weight_1_0_s43").symlink_to(E3 / "evals" / "base_rescored_kl_weight_1_0_s43")
(LINE / "evals" / "first_probe_round1").symlink_to(E4 / "seed_43" / "evals" / "first_probe_round1")
(LINE / "generations" / "round1").symlink_to(E4 / "seed_43" / "generations" / "round1")
(LINE / "probe_sets" / "round1").symlink_to(E4 / "seed_43" / "probe_sets" / "round1")
for r in (2, 3, 4, 5):
    key = f"s43_refit_round{r}"
    (LINE / "evals" / key).symlink_to(E9 / "seed_43" / "evals" / key)
    (LINE / "evals" / f"base_rescored_{key}").symlink_to(E9 / "seed_43" / "evals" / f"base_rescored_{key}")
    (LINE / "evals" / f"first_probe_round{r}").symlink_to(E9 / "seed_43" / "evals" / f"first_probe_round{r}")
    (LINE / f"run_metrics_{key}.json").symlink_to(E9 / "seed_43" / f"run_metrics_{key}.json")
    if r < 5:
        (LINE / "generations" / f"round{r}").symlink_to(E9 / "seed_43" / "generations" / f"round{r}")

results = {}
for r in (2, 3, 4, 5):
    out = SCRATCH / "build" / f"round{r}"
    rounds.build(r, out)
    saved = E9 / "seed_43" / "probe_sets" / f"round{r}"
    new_info, saved_info = json.loads((out / "build.json").read_text()), json.loads((saved / "build.json").read_text())
    new_test_answers, saved_test_answers = new_info.pop("test_answers"), saved_info.pop("test_answers")
    results[f"build {r}"] = {
        "train_identical": filecmp.cmp(out / "train.jsonl", saved / "train.jsonl", shallow=False),
        "test_identical": filecmp.cmp(out / "test.jsonl", saved / "test.jsonl", shallow=False),
        "build_json_equal_except_test_answers": new_info == saved_info,
        "test_answers": [new_test_answers, saved_test_answers],
    }
    # config r and table need the round's own set where 261109 has it
    (LINE / "probe_sets" / f"round{r}").symlink_to(saved)

rounds.config(2)
new_yaml, saved_yaml = LINE / "s43_refit_round2.yaml", E9 / "seed_43" / "s43_refit_round2.yaml"
new_config, saved_config = yaml.safe_load(new_yaml.read_text()), yaml.safe_load(saved_yaml.read_text())
results["config 2"] = {
    "differing_top_keys": sorted(k for k in set(new_config) | set(saved_config) if new_config.get(k) != saved_config.get(k)),
    "differing_override_keys": sorted(k for k in set(new_config["overrides"]) | set(saved_config["overrides"])
                                      if new_config["overrides"].get(k) != saved_config["overrides"].get(k)),
    "output_dir": [new_config["output_dir"], saved_config["output_dir"]],
    "smoke_yaml_written": (LINE / "smoke_s43_refit_round2.yaml").exists(),
}

rounds.table(5, SCRATCH / "table")
new_table = json.loads((SCRATCH / "table" / "probe_table.json").read_text())
saved_table = json.loads((E9 / "seed_43" / "probe_table.json").read_text())
for table in (new_table, saved_table):
    for value in table.values():
        value.pop("test_answers")
results["table 5"] = {"probe_table_json_equal_except_test_answers": new_table == saved_table}
text = json.dumps(results, indent=2).replace(str(SCRATCH), "<scratch>")
(SCRATCH / "result.json").write_text(text + "\n")
print(text)
