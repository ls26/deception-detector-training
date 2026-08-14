#!/usr/bin/env python3
"""Print the 3-probe comparison table and write summary.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("/workspace")))

from common import (
    LINEAGES,
    PROBE_A,
    PROBE_B,
    PROBE_C,
    PROBE_KEYS,
    label_for,
    results_path,
    summary_path,
)

PROBE_LABELS = {
    PROBE_A: "A final (r6 data, r5 acts)",
    PROBE_B: "B initial (p1 data, base acts)",
    PROBE_C: "C initial (r6 data, base acts)",
}


def _fmt(value) -> str:
    return f"{value:9.4f}" if isinstance(value, (int, float)) else f"{'--':>9}"


def _provenance_str(counts: dict[str, int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in counts.items())


def main() -> None:
    rows: list[dict] = []
    for trial, variant in LINEAGES:
        path = results_path(trial, variant)
        if not path.exists():
            print(f"MISSING {path}")
            continue
        data = json.loads(path.read_text())
        probes = data.get("probes", {})
        shared = data["shared_eval"]
        rows.append(
            {
                "label": label_for(trial, variant),
                "trial": trial,
                "variant": data["variant"],
                "round5_run_id": data["round5_run_id"],
                "n_deceptive": shared["n_deceptive"],
                "n_honest": shared["n_honest"],
                "deceptive_provenance": shared["deceptive_provenance"],
                "honest_provenance": shared["honest_provenance"],
                "probes": {
                    key: {
                        "auc": probes[key]["auc"],
                        "deceptive": probes[key]["deceptive"],
                        "honest": probes[key]["honest"],
                        "accuracy": probes[key]["accuracy"],
                        "f1": probes[key]["f1"],
                        "n_filtered_layers": probes[key]["n_filtered_layers"],
                        "activations_model": probes[key]["activations_model"],
                    }
                    for key in PROBE_KEYS
                    if key in probes
                },
            }
        )

    print(f"Shared-eval 3-probe monitorability comparison — {len(rows)}/{len(LINEAGES)} lineages")
    print("AUC averaged over each probe's own filtered_layers (count in the 'L' column).\n")

    header = (
        f"{'Lineage':<33} {'Probe':<32} "
        f"{'AUC':>9} {'mean_dec':>9} {'mean_hon':>9} {'acc':>9} {'f1':>9} {'L':>4}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        for i, key in enumerate(PROBE_KEYS):
            probe = row["probes"].get(key)
            lineage = row["label"] if i == 0 else ""
            if probe is None:
                print(f"{lineage:<33} {PROBE_LABELS[key]:<32} {'not scored':>9}")
                continue
            print(
                f"{lineage:<33} {PROBE_LABELS[key]:<32} "
                f"{_fmt(probe['auc'])} {_fmt(probe['deceptive'])} {_fmt(probe['honest'])} "
                f"{_fmt(probe['accuracy'])} {_fmt(probe['f1'])} {probe['n_filtered_layers']:>4}"
            )
        print()

    print("\nShared eval set provenance (deceptive anchor = det0, honest anchor = base)\n")
    for row in rows:
        print(f"  {row['label']}")
        print(f"    deceptive ({row['n_deceptive']:3d}): {_provenance_str(row['deceptive_provenance'])}")
        print(f"    honest    ({row['n_honest']:3d}): {_provenance_str(row['honest_provenance'])}")

    out = summary_path()
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
