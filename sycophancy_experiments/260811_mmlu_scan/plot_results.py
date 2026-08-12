"""Plot per-category results from the MMLU-Pro difficulty scan.

Draws majority-vote accuracy and mean single-sample accuracy as grouped bars,
overlays the self-consistency rate (fraction of questions where all samples
agree), and marks the random-guess baseline implied by the option counts.

Usage:
    python plot_results.py
    python plot_results.py --summary category_summary.csv --output category_accuracy.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent


def random_baseline(scan_set: Path) -> float | None:
    """Expected accuracy from guessing, averaged over the scan set's option counts."""
    if not scan_set.exists():
        return None
    chances = []
    with scan_set.open() as f:
        for line in f:
            if line.strip():
                chances.append(1.0 / len(json.loads(line)["options"]))
    return sum(chances) / len(chances) if chances else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-category MMLU-Pro scan results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--summary", type=Path, default=EXP_DIR / "category_summary.csv")
    parser.add_argument("--scan-set", type=Path, default=EXP_DIR / "scan_set.jsonl")
    parser.add_argument("--output", type=Path, default=EXP_DIR / "category_accuracy.png")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.summary)
    df = df.sort_values("majority_accuracy", ascending=False).reset_index(drop=True)
    print(f"=== Loaded {len(df)} categories from {args.summary} ===")

    n_questions = int(df["n_questions"].sum())
    n_samples = int(df["n_samples"].sum())
    overall_majority = (df["majority_accuracy"] * df["n_questions"]).sum() / n_questions
    overall_sample = (df["sample_accuracy"] * df["n_samples"]).sum() / n_samples
    overall_agree = (df["all_agree_rate"] * df["n_questions"]).sum() / n_questions
    chance = random_baseline(args.scan_set)

    x = range(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5))

    bars_majority = ax.bar(
        [i - width / 2 for i in x],
        df["majority_accuracy"],
        width,
        label="Majority-vote accuracy (3 samples)",
        color="#3b74b8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        df["sample_accuracy"],
        width,
        label="Mean single-sample accuracy",
        color="#9dc3e6",
    )

    ax.plot(
        list(x),
        df["all_agree_rate"],
        marker="D",
        linestyle="none",
        markersize=7,
        color="#d1495b",
        label="Self-consistency (all 3 samples agree)",
    )

    ax.axhline(
        overall_majority,
        linestyle="--",
        linewidth=1.2,
        color="#333333",
        label=f"Overall majority accuracy ({overall_majority:.3f})",
    )
    if chance is not None:
        ax.axhline(
            chance,
            linestyle=":",
            linewidth=1.2,
            color="#777777",
            label=f"Random guess ({chance:.3f})",
        )

    ax.bar_label(bars_majority, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["category"], rotation=35, ha="right")
    ax.set_ylabel("Accuracy / rate")
    ax.set_ylim(0, max(1.0, df["majority_accuracy"].max() + 0.12))
    ax.set_title(
        "MMLU-Pro per-category difficulty scan - Meta-Llama-3-8B-Instruct\n"
        f"{n_questions} questions ({n_questions // len(df)}/category), "
        f"{n_samples} samples, zero-shot CoT, temperature 1.0  |  "
        f"mean single-sample {overall_sample:.3f}, self-consistency {overall_agree:.3f}"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"  Saved chart to {args.output}")

    parse_fail = (df["parse_fail_rate"] * df["n_samples"]).sum() / n_samples
    if parse_fail > 0.01:
        print(f"  NOTE: overall parse failure rate is {parse_fail:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
