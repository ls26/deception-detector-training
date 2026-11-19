"""Identity check of this folder's rounds.py (imported unchanged): config 2 of seed 42 as the queue writes it, then
links 2 and probe_table pointed at 261107's seed 42, whose rounds.py has the same links and probe_table code. Only the
run names, the W&B group and round 1 (261103's weight-1.0 runs) are patched, in a scratch line folder that links
261107's saved evals, tables and round-2 run_metrics.

Usage (the Start check, and after a fix to rounds.py): $PY repro_rounds.py SCRATCH -> SCRATCH/result.json, with SCRATCH
written as <scratch>, compared byte for byte with rounds_expected.json (written at setup on 2026-11-14).
"""

import json
import sys
from pathlib import Path

NEW = Path("/workspace/sycophancy_experiments/261114_no_penalty_rounds")
OLD = Path("/workspace/sycophancy_experiments/261107_frozen_probe_kl_0_01")
SCRATCH = Path(sys.argv[1])

sys.path.insert(0, str(NEW))
import rounds  # noqa: E402

# config 2 of seed 42, exactly as the queue writes it, into a scratch line folder
rounds.SEED = 42
rounds.EXP_DIR = SCRATCH / "config_line"
rounds.EXP_DIR.mkdir(parents=True)
rounds.config(2)
config_text = (rounds.EXP_DIR / "s42_no_penalty_round2.yaml").read_text()

# links 2 and probe_table on 261107's seed 42
line = SCRATCH / "seed_42"
line.mkdir()
for name in ("evals", "tables", "run_metrics_s42_frozen_round2.json"):
    (line / name).symlink_to(OLD / "seed_42" / name)
rounds.EXP_DIR = line
rounds.GROUP = OLD.name
rounds.ROUND1_FOLDER = Path("/workspace/sycophancy_experiments/261103_kl_0_01/weight_1_0")
rounds.ROUND1_RUNS = {42: ("kl_weight_1_0_s42", "am1g59lu"), 43: ("kl_weight_1_0_s43", "hxpczsle")}
rounds.run_key = lambda round_number: f"s42_frozen_round{round_number}"
rounds.links(2)
rounds.probe_table()

links = {subdir.name: str(subdir.resolve()) for subdir in sorted((line / "links" / "round2").iterdir())}
text = json.dumps({
    "config 2": config_text,
    "links 2": links,
    "probe_table.json": json.loads((line / "probe_table.json").read_text()),
    "probe_table.md": (line / "probe_table.md").read_text(),
}, indent=2).replace(str(SCRATCH), "<scratch>")
(SCRATCH / "result.json").write_text(text + "\n")
print(text)
