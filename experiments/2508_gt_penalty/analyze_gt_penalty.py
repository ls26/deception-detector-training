# GT penalty round-1 seed sweep analysis (methodology #25).
# Run cells interactively (# %% style) or: python analyze_gt_penalty.py

# %%
import json
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
SINGLE_RUNS = EXPERIMENT_DIR / "single_runs"

OUTCOME_COLS = ("correct", "deceptive", "buggy", "fail")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def derive_outcome_categories(pass_first: float, pass_all: float, deceptive: float) -> dict[str, float]:
    return {
        "correct": pass_all,
        "deceptive": deceptive,
        "buggy": pass_first - pass_all - deceptive,
        "fail": 1 - pass_first,
    }


def outcome_categories_from_run_metrics(path: Path) -> dict[str, float]:
    metrics = load_json(path)["metrics"]
    return derive_outcome_categories(
        metrics["val_passes_first_n_tests"],
        metrics["val_passes_all_tests"],
        metrics["val_pass_only_shown_deception"],
    )


def load_runs() -> pd.DataFrame:
    paths = sorted(SINGLE_RUNS.glob("run_metrics_gt_hybrid_clp1e-3_p25_s*.json"))
    rows = []
    for path in paths:
        data = load_json(path)
        seed = path.stem.rsplit("_s", 1)[-1]
        rows.append(
            {
                "seed": int(seed),
                "run_id": data["run_id"],
                **outcome_categories_from_run_metrics(path),
            }
        )
    return pd.DataFrame(rows)


def print_means(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label} (n={len(df)})")
    for col in OUTCOME_COLS:
        mean = df[col].mean()
        std = df[col].std()
        print(f"  {col}: mean={mean * 100:.2f}%, std={std * 100:.2f}%")


# %%
df = load_runs()
df

# %%
print_means("GT penalty", df)

# %%
