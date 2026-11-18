"""Reproduction test of 261110's frozen_rounds.py against 261108's seed 43: config 2 and links 2.

The new frozen_rounds.py is imported unchanged; only run_key (261108's run names) and run_dir(1) (261103's round-1
folder) are patched, in a scratch line folder that links 261108's round-2 run_metrics.

Usage (the unattended queue's Start check): $PY repro_frozen.py SCRATCH -> SCRATCH/result.json, with SCRATCH written
as <scratch>, so it can be compared byte for byte with frozen_expected.json (written at setup on 2026-11-10).
"""

import json
import sys
from pathlib import Path

import yaml

NEW = Path("/workspace/sycophancy_experiments/261110_redrawn_labels_seeds_45_46")
E8 = Path("/workspace/sycophancy_experiments/261108_frozen_probe_redrawn_labels")
R1_RUN = Path("/workspace/results/run_results/261103_kl_0_01/kl_weight_1_0_s43_hxpczsle")
SCRATCH = Path(sys.argv[1])
LINE = SCRATCH / "seed_43"

sys.path.insert(0, str(NEW))
import frozen_rounds  # noqa: E402

old_run_dir = frozen_rounds.run_dir


def run_key(round_number):
    return "kl_weight_1_0_s43" if round_number == 1 else f"s43_redrawn_round{round_number}"


def run_dir(round_number):
    return R1_RUN if round_number == 1 else old_run_dir(round_number)


frozen_rounds.run_key = run_key
frozen_rounds.run_dir = run_dir
frozen_rounds.GROUP = E8.name
frozen_rounds.TEMPLATE_PATH = E8 / "round_template.yaml"
frozen_rounds.EXP_DIR = LINE
frozen_rounds.SEED = 43
LINE.mkdir(parents=True)
(LINE / "run_metrics_s43_redrawn_round2.json").symlink_to(E8 / "seed_43" / "run_metrics_s43_redrawn_round2.json")

frozen_rounds.config(2)
new_config = yaml.safe_load((LINE / "s43_redrawn_round2.yaml").read_text())
saved_config = yaml.safe_load((E8 / "seed_43" / "s43_redrawn_round2.yaml").read_text())
frozen_rounds.links(2)
links = {}
for subdir in sorted(p.name for p in (LINE / "links" / "round2").iterdir()):
    links[subdir] = [str((LINE / "links" / "round2" / subdir).resolve()), str((E8 / "seed_43" / "links" / "round2" / subdir).resolve())]
text = json.dumps({
    "config 2": {
        "differing_top_keys": sorted(k for k in set(new_config) | set(saved_config) if new_config.get(k) != saved_config.get(k)),
        "differing_override_keys": sorted(k for k in set(new_config["overrides"]) | set(saved_config["overrides"])
                                          if new_config["overrides"].get(k) != saved_config["overrides"].get(k)),
        "output_dir": [new_config["output_dir"], saved_config["output_dir"]],
        "smoke_yaml_written": (LINE / "smoke_s43_redrawn_round2.yaml").exists(),
    },
    "links 2": {subdir: {"new_and_saved": pair, "same_target": pair[0] == pair[1]} for subdir, pair in links.items()},
}, indent=2).replace(str(SCRATCH), "<scratch>")
(SCRATCH / "result.json").write_text(text + "\n")
print(text)
