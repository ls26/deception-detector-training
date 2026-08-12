# %%
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")

from common import results_path

# Row label -> variant string per trial (trial 4 "original" is the v2 full-128 branch)
ROW_VARIANTS: dict[str, dict[int, str]] = {
    "Original": {6: "", 4: "v2", 8: ""},
    "Early retraining (probe-based)": {
        6: "early_retraining",
        4: "early_retraining_v2",
        8: "early_retraining",
    },
    "Early retraining (auc-based)": {
        6: "early_retraining_auc_v3",
        4: "early_retraining_auc_v2",
        8: "early_retraining_auc_v3",
    },
}

TRIAL_COLUMNS: dict[str, int] = {
    "Trial A": 6,
    "Trial B": 4,
    "Trial C": 8,
}


def load_supplemented_auc(trial: int, variant: str) -> float:
    path = results_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    data = json.loads(path.read_text())
    return float(data["supplemented_on_policy"]["auc"])

# %%
rows = list(ROW_VARIANTS.keys())
cols = list(TRIAL_COLUMNS.keys())

table = pd.DataFrame(index=rows, columns=cols, dtype=float)

for row_label, variants_by_trial in ROW_VARIANTS.items():
    for col_label, trial in TRIAL_COLUMNS.items():
        variant = variants_by_trial[trial]
        table.loc[row_label, col_label] = load_supplemented_auc(trial, variant)

table.round(3)

# %%
# Optional: color scale (green = high AUC, red = low)
styled = table.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.5, vmax=1.0)
styled

# %%
