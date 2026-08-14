# %%
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")

from common import PROBE_A, PROBE_B, PROBE_C, results_path

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

# Panel order matches experiments/paper_figures/paper_fig_common.py: A = 4, B = 6, C = 8.
# (The older 250810_round6_probe_eval/results_table.py uses the pre-swap A = 6, B = 4.)
TRIAL_COLUMNS: dict[str, int] = {
    "Trial A": 4,
    "Trial B": 6,
    "Trial C": 8,
}

PROBE_LABELS: dict[str, str] = {
    PROBE_A: "A final",
    PROBE_B: "B initial (probe-1)",
    PROBE_C: "C initial (r6 data)",
}


def load_result(trial: int, variant: str) -> dict:
    path = results_path(trial, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text())


# %%
# Long-form frame: one row per (lineage, probe).
records: list[dict] = []
for row_label, variants_by_trial in ROW_VARIANTS.items():
    for col_label, trial in TRIAL_COLUMNS.items():
        data = load_result(trial, variants_by_trial[trial])
        shared = data["shared_eval"]
        for probe_key, probe_label in PROBE_LABELS.items():
            probe = data["probes"][probe_key]
            records.append(
                {
                    "variant": row_label,
                    "trial": col_label,
                    "probe": probe_label,
                    "auc": probe["auc"],
                    "mean_dec": probe["deceptive"],
                    "mean_hon": probe["honest"],
                    "accuracy": probe["accuracy"],
                    "f1": probe["f1"],
                    "n_layers": probe["n_filtered_layers"],
                    "acts_model": probe["activations_model"],
                    "n_deceptive": shared["n_deceptive"],
                    "n_honest": shared["n_honest"],
                }
            )

df = pd.DataFrame.from_records(records)
df.round(4)

# %%
# AUC on the shared eval set: rows = lineage, columns = (trial, probe).
auc = df.pivot_table(index="variant", columns=["trial", "probe"], values="auc")
auc = auc.reindex(index=list(ROW_VARIANTS), columns=pd.MultiIndex.from_product(
    [list(TRIAL_COLUMNS), list(PROBE_LABELS.values())], names=["trial", "probe"]
))
auc.round(3)

# %%
# Flat AUC table: one row per (trial, variant), one column per probe.
VARIANT_DISPLAY: dict[str, str] = {
    "Original": "Fixed-schedule retraining",
    "Early retraining (probe-based)": "Early retraining (probe-based)",
    "Early retraining (auc-based)": "Early retraining (auc-based)",
}
PROBE_COLUMNS: dict[str, str] = {
    PROBE_A: "Final probe",
    PROBE_C: "Initial probe (shared train)",
    PROBE_B: "Initial probe (original train)",
}
GAP_COLUMN = "Initial (shared train) - Final"
MEAN_ROW = "Mean"

flat = df.copy()
flat["row"] = flat["trial"] + ": " + flat["variant"].map(VARIANT_DISPLAY)
flat["column"] = flat["probe"].map(
    {PROBE_LABELS[k]: v for k, v in PROBE_COLUMNS.items()}
)

row_order = [
    f"{trial_label}: {VARIANT_DISPLAY[variant_label]}"
    for trial_label in TRIAL_COLUMNS
    for variant_label in ROW_VARIANTS
]
auc_flat = (
    flat.pivot_table(index="row", columns="column", values="auc")
    .reindex(index=row_order, columns=list(PROBE_COLUMNS.values()))
)
auc_flat.insert(
    2,
    GAP_COLUMN,
    auc_flat[PROBE_COLUMNS[PROBE_C]] - auc_flat[PROBE_COLUMNS[PROBE_A]],
)
auc_flat.loc[MEAN_ROW] = auc_flat.mean()
auc_flat.index.name = None
auc_flat.columns.name = None
print(auc_flat.round(3).to_string())
auc_flat.round(3)

# %%
# One table per probe (same shape as the round-6 results_table).
for probe_label in PROBE_LABELS.values():
    table = (
        df[df["probe"] == probe_label]
        .pivot_table(index="variant", columns="trial", values="auc")
        .reindex(index=list(ROW_VARIANTS), columns=list(TRIAL_COLUMNS))
    )
    print(f"\n{probe_label} — AUC")
    print(table.round(3).to_string())

# %%
# Monitorability gap: initial minus final, same training data (C - A).
gap = (
    auc.xs(PROBE_LABELS[PROBE_C], level="probe", axis=1)
    - auc.xs(PROBE_LABELS[PROBE_A], level="probe", axis=1)
)
print("Initial - final AUC on identical training data (C - A); positive = initial more monitorable")
print(gap.round(3).to_string())
print(f"\nmean gap: {gap.to_numpy().mean():+.4f}   positive in {int((gap > 0).to_numpy().sum())}/{gap.size}")

# %%
# Mean probe scores per class, for reading the AUCs alongside score separation.
scores = df.pivot_table(
    index=["variant", "probe"], columns="trial", values=["mean_dec", "mean_hon"]
).reindex(index=pd.MultiIndex.from_product([list(ROW_VARIANTS), list(PROBE_LABELS.values())]))
scores.round(3)

# %%
# Shared eval set composition per lineage (100/class by construction).
provenance: list[dict] = []
for row_label, variants_by_trial in ROW_VARIANTS.items():
    for col_label, trial in TRIAL_COLUMNS.items():
        shared = load_result(trial, variants_by_trial[trial])["shared_eval"]
        provenance.append(
            {
                "variant": row_label,
                "trial": col_label,
                "n_dec": shared["n_deceptive"],
                "n_hon": shared["n_honest"],
                "deceptive": " ".join(f"{k}:{v}" for k, v in shared["deceptive_provenance"].items()),
                "honest": " ".join(f"{k}:{v}" for k, v in shared["honest_provenance"].items()),
            }
        )

prov = pd.DataFrame.from_records(provenance).set_index(["variant", "trial"])
with pd.option_context("display.max_colwidth", 120):
    print(prov.to_string())

# %%
# Optional: color scale (green = high AUC, red = low)
styled = auc.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.5, vmax=1.0, axis=None)
styled

# %%
