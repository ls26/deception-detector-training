#!/usr/bin/env python3
"""Print summary table from round-6 probe eval result JSONs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROUND6_LINEAGES, results_path


def main() -> None:
    rows: list[dict] = []
    for trial, variant in ROUND6_LINEAGES:
        path = results_path(trial, variant)
        if not path.exists():
            print(f"MISSING {path}")
            continue
        data = json.loads(path.read_text())
        label = f"trial{trial} {data['variant']}"
        rows.append(
            {
                "label": label,
                "round5_run_id": data["round5_run_id"],
                "on_ds_auc": data["on_dataset"]["auc"],
                "on_ds_dec": data["on_dataset"]["mean_scores"]["deceptive"],
                "on_ds_hon": data["on_dataset"]["mean_scores"]["honest"],
                "split_auc": data["on_policy_split"]["auc"],
                "split_dec": data["on_policy_split"]["mean_scores"]["deceptive"],
                "split_hon": data["on_policy_split"]["mean_scores"]["honest"],
                "split_stats": data["on_policy_split"].get("stats", {}),
                "supp_auc": data["supplemented_on_policy"]["auc"],
                "supp_dec": data["supplemented_on_policy"]["deceptive"],
                "supp_hon": data["supplemented_on_policy"]["honest"],
                "supp_n_dec": data["supplemented_on_policy"]["n_deceptive"],
                "supp_n_hon": data["supplemented_on_policy"]["n_honest"],
            }
        )

    print(f"Round-6 probe monitorability eval — {len(rows)}/{len(ROUND6_LINEAGES)} lineages\n")
    header = (
        f"{'Lineage':<35} {'on_ds_AUC':>9} {'split_AUC':>9} {'supp_AUC':>9} "
        f"{'split_dec':>9} {'split_hon':>9} {'supp_dec':>9} {'supp_hon':>9}  native_dec/hon"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        stats = r["split_stats"]
        native = f"{stats.get('deceptive', '?')}/{stats.get('honest', '?')}"
        print(
            f"{r['label']:<35} "
            f"{r['on_ds_auc']:9.4f} {r['split_auc']:9.4f} {r['supp_auc']:9.4f} "
            f"{r['split_dec']:9.4f} {r['split_hon']:9.4f} {r['supp_dec']:9.4f} {r['supp_hon']:9.4f}  "
            f"{native}"
        )

    out = Path(__file__).resolve().parent / "summary.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
