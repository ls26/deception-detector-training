"""The final fit of one arm: the CV-chosen recipe per probe, one seed, the full
train set, scored once on eval.

    python final_fit.py --arm ref_cot

Applies cv/<arm>/chosen_overrides.json, trains every probe of the arm, writes
runs/<arm>/{config.json, probe_metrics.json, eval_scores.json, probe_weights/}
and appends one row to runs.md's "## Rounds" table. This is the first and only
time the eval activations are read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
import step3_train_eval_probes as s3  # noqa: E402
from hf_env import bootstrap_hf_env  # noqa: E402
from probe_config import PipelineConfig, apply_overrides, arm_config, config_to_json  # noqa: E402

RUNS_TABLE_HEADER = (
    "## Rounds\n\n"
    "One row per final fit (`final_fit.py --arm X` -> `runs/<arm>/`). AUC cells are the eval AUC of the "
    "combined score (mean of the per-layer sigmoids over the chosen band), one seed; the recipe cells are "
    "the CV choice lr / steps / band per probe.\n\n"
    "| arm | n_train (bias / bias_followed / deception / deception_yes / deception_no) | bias | bias_followed | deception | deception_yes | deception_no | recipe (lr / steps / band) |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def runs_md_row(cfg: PipelineConfig, summaries: list[dict]) -> str:
    by_kind = {s["kind"]: s for s in summaries}
    n_train = " / ".join(str(by_kind[k]["n_train"]) if k in by_kind else "—" for k in spec.PROBE_KINDS)
    aucs = " | ".join(f"{by_kind[k]['auc_combined']:.3f}" if k in by_kind else "—" for k in spec.PROBE_KINDS)
    recipes = "; ".join(
        f"{k}: {by_kind[k]['learning_rate']:g} / {by_kind[k]['max_steps']} / {by_kind[k]['filtered_layers'][0]}-{by_kind[k]['filtered_layers'][-1]}"
        for k in spec.PROBE_KINDS
        if k in by_kind
    )
    return f"| {cfg.arm} | {n_train} | {aucs} | {recipes} |\n"


def append_runs_row(runs_md: Path, row: str) -> None:
    """Insert `row` at the end of the table under "## Rounds"; create the section
    (and runs.md) when missing. Lines after the table are kept."""
    lines = runs_md.read_text().splitlines() if runs_md.exists() else ["# Runs", ""]
    if "## Rounds" not in lines:
        lines += [""] + RUNS_TABLE_HEADER.rstrip("\n").splitlines()
    heading_index = lines.index("## Rounds")
    section_end = heading_index + 1
    while section_end < len(lines) and not lines[section_end].startswith("## "):
        section_end += 1
    table_lines = [i for i in range(heading_index + 1, section_end) if lines[i].startswith("|")]
    if not table_lines:
        # a "## Rounds" heading without a table: the prose + table header go right after it
        section_body = RUNS_TABLE_HEADER.rstrip("\n").splitlines()[1:]
        lines[heading_index + 1 : heading_index + 1] = section_body
        table_lines = [i for i in range(heading_index + 1, heading_index + 1 + len(section_body)) if lines[i].startswith("|")]
    lines.insert(table_lines[-1] + 1, row.rstrip("\n"))
    runs_md.write_text("\n".join(lines) + "\n")
    print(f"appended to {runs_md}:\n{row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=spec.ARMS)
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    args = parser.parse_args()

    bootstrap_hf_env()
    from transformers import AutoTokenizer

    cfg = arm_config(args.arm, args.arms_dir.resolve())
    assert cfg.act_dir.exists(), f"no activations at {cfg.act_dir}"
    chosen_path = cfg.cv_dir / "chosen_overrides.json"
    chosen = json.loads(chosen_path.read_text())
    assert chosen["spec_hash"] == spec.spec_hash(), f"{chosen_path} from another spec"
    missing = [p for p in cfg.probe_names() if p not in chosen["per_probe_overrides"]]
    assert not missing, f"{chosen_path} has no recipe for {missing}"
    cfg = apply_overrides(cfg, {"per_probe_overrides": chosen["per_probe_overrides"]})
    assert not cfg.run_dir.exists(), f"{cfg.run_dir} exists -- the final fit of this arm was already made"
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "config.json").write_text(json.dumps(config_to_json(cfg), indent=2))
    print(json.dumps(config_to_json(cfg), indent=2))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    summaries = s3.train_eval_all(cfg, tokenizer)

    append_runs_row(cfg.arms_dir.parent / "runs.md", runs_md_row(cfg, summaries))
    print(f"FINAL DONE {cfg.arm}")


if __name__ == "__main__":
    main()
