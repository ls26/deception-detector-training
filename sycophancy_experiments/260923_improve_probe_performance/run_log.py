"""Round bookkeeping for the autonomous probe-improvement run.

One training round = one call of `step3.train_eval_all_seeds` on one config.
Each round gets `runs/{NN}_{slug}/` holding `config.json`, `probe_metrics.json`
and the per-seed weights, and one row in `runs.md`. `runs.md` is the record the
agent re-reads after a context reset, so every round appends to it before the
next one starts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from probe_config import EXP_DIR, PipelineConfig, config_to_json

ROUND_CAP = 40
RUNS_MD = EXP_DIR / "runs.md"

#: the five reported probe kinds, bias family first. A round's row is keyed by
#: kind, not by full probe name, so a positions round (e.g. bias_rm_lastcompl)
#: still fills the same five columns; the `activations` / `change` cells say
#: which position it was.
REPORTED_KINDS: tuple[str, ...] = ("bias", "bias_followed", "deception", "deception_yes", "deception_no")


def probes_by_kind(metrics: dict) -> dict[str, dict]:
    by_kind: dict[str, dict] = {}
    for p in metrics["probes"]:
        assert p["kind"] not in by_kind, f"two probes of kind {p['kind']} in one round: report one position per round"
        by_kind[p["kind"]] = p
    return by_kind


def round_dirs(cfg: PipelineConfig) -> list[Path]:
    if not cfg.runs_dir.exists():
        return []
    return sorted(p for p in cfg.runs_dir.iterdir() if p.is_dir())


def start_round(cfg: PipelineConfig, slug: str) -> PipelineConfig:
    """Create the next round folder and return a config that writes into it."""
    assert slug and "/" not in slug and " " not in slug, f"slug must be a short path-safe name, got {slug!r}"
    round_number = len(round_dirs(cfg)) + 1
    assert round_number <= ROUND_CAP, f"round cap of {ROUND_CAP} reached -- stop and write SUMMARY.md"
    round_dir = cfg.runs_dir / f"{round_number:02d}_{slug}"
    round_dir.mkdir(parents=True)
    round_cfg = replace(cfg, weights_dir=round_dir / "probe_weights", metrics_path=round_dir / "probe_metrics.json")
    (round_dir / "config.json").write_text(json.dumps(config_to_json(round_cfg), indent=2))
    print(f"round {round_number}/{ROUND_CAP}: {round_dir}")
    return round_cfg


def _auc_cell(probe_entry: dict, key: str = "auc_filtered") -> str:
    """'mean ± sd' when the metrics came from train_eval_all_seeds, else the single value."""
    if f"{key}_mean" in probe_entry:
        return f"{probe_entry[f'{key}_mean']:.3f} ± {probe_entry[f'{key}_sd']:.3f}"
    return f"{probe_entry[key]:.3f}"


def five_probe_table(metrics_path: Path) -> str:
    """The reported probes from one metrics file, bias family first."""
    metrics = json.loads(Path(metrics_path).read_text())
    by_kind = probes_by_kind(metrics)
    seeds = metrics.get("seeds", [metrics["config"]["probe_seed"]])
    header = (
        f"{'probe':28s} {'eval AUC (filtered)':>20s} {'layer-mean logit':>18s} "
        f"{'n_train':>8s} {'n_eval':>7s} {'layers':>7s}"
    )
    lines = [f"seeds: {seeds}", header, "-" * len(header)]
    for kind in REPORTED_KINDS:
        if kind not in by_kind:
            lines.append(f"{kind:28s} {'not trained':>20s}")
            continue
        p = by_kind[kind]
        lines.append(
            f"{p['probe']:28s} {_auc_cell(p):>20s} {_auc_cell(p, 'auc_layer_mean_logit'):>18s} "
            f"{p['n_train']:8d} {p['n_eval']:7d} {len(p['filtered_layers']):7d}"
        )
    lines.append("-" * len(header))
    return "\n".join(lines)


RUNS_TABLE_HEADER = (
    "| round | change | activations | bias | bias_followed | deception | deception_yes | deception_no | decision / next |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def append_runs_table(round_cfg: PipelineConfig, change: str, decision: str) -> None:
    """Append this round's row to runs.md (eval AUC, filtered layers, mean ± sd over seeds)."""
    metrics = json.loads(round_cfg.metrics_path.read_text())
    by_kind = probes_by_kind(metrics)
    cells = [_auc_cell(by_kind[kind]) if kind in by_kind else "—" for kind in REPORTED_KINDS]
    row = (
        f"| {round_cfg.metrics_path.parent.name} | {change} | {round_cfg.act_dir.name} | "
        + " | ".join(cells)
        + f" | {decision} |\n"
    )
    text = RUNS_MD.read_text()
    assert "## Rounds" in text, "runs.md has no '## Rounds' section"
    RUNS_MD.write_text(text.rstrip("\n") + "\n" + row)
    print(f"appended to {RUNS_MD}:\n{row}")
