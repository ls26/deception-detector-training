"""The comparison table: every arm's final-fit eval AUC against the reference.

    python compare.py

Reads runs/<arm>/{eval_scores.json, probe_metrics.json} for every arm present,
arms/sizes.json and arms/nocot_logprob/labels.jsonl (the confident slice), and
writes comparison.md:

  (a) the main table, one row per probe kind: the reference AUC, then per
      other arm "AUC (Δ vs ref, 95% CI)". The CI is a question-level bootstrap
      (spec.BOOTSTRAP_DRAWS draws, seed spec.BOOTSTRAP_SEED) over the eval
      questions shared by the two arms: question ids are resampled with
      replacement, both AUCs recomputed on the resampled rows, and the 2.5 /
      97.5 percentiles of the Δ taken. A Δ whose CI excludes 0 is marked *.
  (b) checks: the chosen lr / steps / band per arm and probe; the train n
      per probe and the size rule's outcome; nocot_logprob's AUCs on the
      confident slice beside the full eval; 260924's full-size numbers as a
      footnote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import spec  # noqa: E402
from step1_build_probe_sets import load_jsonl  # noqa: E402

#: probe kind -> 260924's probe name (the reproduction targets are keyed by it)
PROBE_NAME_OF_KIND: dict[str, str] = {
    "bias": "bias_rm_answer",
    "bias_followed": "bias_followed_rm_answer",
    "deception": "deception_rm_preyesno",
    "deception_yes": "deception_yes_rm_preyesno",
    "deception_no": "deception_no_rm_preyesno",
}


def load_run(runs_dir: Path, arm: str) -> dict | None:
    run_dir = runs_dir / arm
    if not (run_dir / "eval_scores.json").exists():
        return None
    metrics = json.loads((run_dir / "probe_metrics.json").read_text())
    scores = json.loads((run_dir / "eval_scores.json").read_text())
    assert metrics["spec_hash"] == spec.spec_hash() and scores["spec_hash"] == spec.spec_hash(), f"{run_dir} from another spec"
    by_kind = {p["kind"]: {"summary": p, "scores": scores[p["probe"]]} for p in metrics["probes"]}
    return {"arm": arm, "by_kind": by_kind}


def rows_by_question(scores: dict) -> dict[int, np.ndarray]:
    index: dict[int, list[int]] = {}
    for i, qid in enumerate(scores["question_id"]):
        index.setdefault(qid, []).append(i)
    return {qid: np.array(rows) for qid, rows in index.items()}


def auc_on_questions(scores: dict, question_rows: dict[int, np.ndarray], questions: np.ndarray) -> float:
    rows = np.concatenate([question_rows[q] for q in questions])
    return float(roc_auc_score(np.array(scores["label"])[rows], np.array(scores["score"])[rows]))


def bootstrap_delta(ref_scores: dict, arm_scores: dict) -> dict:
    """Δ = AUC(arm) - AUC(ref) on the shared eval questions, with a
    question-level bootstrap CI."""
    ref_rows = rows_by_question(ref_scores)
    arm_rows = rows_by_question(arm_scores)
    shared = np.array(sorted(set(ref_rows) & set(arm_rows)))
    assert shared.shape[0] > 0, "no shared eval questions"
    point_ref = auc_on_questions(ref_scores, ref_rows, shared)
    point_arm = auc_on_questions(arm_scores, arm_rows, shared)
    rng = np.random.default_rng(spec.BOOTSTRAP_SEED)
    deltas = np.empty(spec.BOOTSTRAP_DRAWS)
    for draw in range(spec.BOOTSTRAP_DRAWS):
        sample = rng.choice(shared, size=shared.shape[0], replace=True)
        deltas[draw] = auc_on_questions(arm_scores, arm_rows, sample) - auc_on_questions(ref_scores, ref_rows, sample)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_shared_questions": int(shared.shape[0]),
        "auc_ref_shared": point_ref,
        "auc_arm_shared": point_arm,
        "delta": point_arm - point_ref,
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def confident_slice_aucs(run: dict, labels_path: Path) -> dict[str, dict]:
    """nocot_logprob's AUC per probe on the eval rows whose prompt is `confident`
    (both top letters >= spec.CONFIDENT_SLICE)."""
    confident_of_prompt = {(r["question_id"], r["condition"]): r["confident"] for r in load_jsonl(labels_path) if r["split"] == "eval"}
    out = {}
    for kind, entry in run["by_kind"].items():
        s = entry["scores"]
        keep = np.array([confident_of_prompt[(q, c)] is True for q, c in zip(s["question_id"], s["condition"])])
        labels, score = np.array(s["label"])[keep], np.array(s["score"])[keep]
        out[kind] = {"n": int(keep.sum()), "n_full": int(keep.shape[0]), "auc": float(roc_auc_score(labels, score)), "auc_full": entry["summary"]["auc_combined"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms-dir", type=Path, default=spec.ARMS_DIR)
    args = parser.parse_args()
    arms_dir = args.arms_dir.resolve()
    runs_dir = arms_dir.parent / "runs"
    out_path = arms_dir.parent / "comparison.md"

    runs = {arm: run for arm in spec.ARMS if (run := load_run(runs_dir, arm)) is not None}
    assert spec.REFERENCE_ARM in runs, f"no final fit for the reference arm in {runs_dir}"
    ref = runs[spec.REFERENCE_ARM]
    other_arms = [arm for arm in runs if arm != spec.REFERENCE_ARM]
    sizes = json.loads((arms_dir / "sizes.json").read_text())
    assert sizes["spec_hash"] == spec.spec_hash()

    deltas: dict[str, dict[str, dict]] = {}
    for arm in other_arms:
        deltas[arm] = {}
        for kind in spec.PROBE_KINDS:
            if kind in ref["by_kind"] and kind in runs[arm]["by_kind"]:
                print(f"bootstrap {arm} / {kind} ...")
                deltas[arm][kind] = bootstrap_delta(ref["by_kind"][kind]["scores"], runs[arm]["by_kind"][kind]["scores"])

    lines = [
        "# Comparison",
        "",
        f"spec_hash {spec.spec_hash()}. Eval AUC of the combined score (mean sigmoid over the CV-chosen band), one seed, per arm's "
        f"final fit (runs/<arm>/). Δ vs {spec.REFERENCE_ARM} is computed on the eval questions shared by the two arms; the 95% CI "
        f"is a question-level bootstrap ({spec.BOOTSTRAP_DRAWS} draws, seed {spec.BOOTSTRAP_SEED}); * marks a CI that excludes 0. "
        f"Verdict probes: {', '.join(spec.VERDICT_PROBES)}.",
        "",
        "## Main table",
        "",
        "".join(["| probe | ", f"{spec.REFERENCE_ARM} AUC |"] + [f" {arm} AUC (Δ vs ref, 95% CI) |" for arm in other_arms]),
        "|---|---|" + "---|" * len(other_arms),
    ]
    for kind in spec.PROBE_KINDS:
        ref_cell = f"{ref['by_kind'][kind]['summary']['auc_combined']:.3f}" if kind in ref["by_kind"] else "—"
        cells = []
        for arm in other_arms:
            if kind not in deltas[arm]:
                cells.append("—")
                continue
            d = deltas[arm][kind]
            star = "*" if d["excludes_zero"] else ""
            cells.append(f"{runs[arm]['by_kind'][kind]['summary']['auc_combined']:.3f} ({d['delta']:+.3f}{star}, [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], n={d['n_shared_questions']})")
        verdict = " (verdict)" if kind in spec.VERDICT_PROBES else ""
        lines.append("".join([f"| {kind}{verdict} | {ref_cell} |"] + [f" {cell} |" for cell in cells]))
    lines += ["", "## Checks", "", "### Chosen recipe per arm and probe (lr / steps / band)", "", "| probe | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for kind in spec.PROBE_KINDS:
        cells = []
        for run in runs.values():
            if kind not in run["by_kind"]:
                cells.append("—")
                continue
            s = run["by_kind"][kind]["summary"]
            cells.append(f"{s['learning_rate']:g} / {s['max_steps']} / {s['filtered_layers'][0]}-{s['filtered_layers'][-1]}")
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    lines += ["", "### Train size per probe (n_train rows) and the size rule", "", "| probe | rule | matched n/class | " + " | ".join(runs) + " |", "|---|---|---|" + "---|" * len(runs)]
    for kind in spec.PROBE_KINDS:
        cells = [str(run["by_kind"][kind]["summary"]["n_train"]) if kind in run["by_kind"] else "—" for run in runs.values()]
        if kind == "deception":
            # the union of the two strata draws; no size entry of its own
            lines.append(f"| {kind} | union of deception_yes + deception_no | — | " + " | ".join(cells) + " |")
            continue
        entry = sizes["per_probe"][kind]
        lines.append(f"| {kind} | {entry['rule']} | {entry['matched_n_per_class']} | " + " | ".join(cells) + " |")
    if "nocot_logprob" in runs:
        confident = confident_slice_aucs(runs["nocot_logprob"], arms_dir / "nocot_logprob" / "labels.jsonl")
        lines += [
            "",
            f"### nocot_logprob on the confident slice (both top letters >= {spec.CONFIDENT_SLICE})",
            "",
            "| probe | full eval AUC (n) | confident slice AUC (n) |",
            "|---|---|---|",
        ]
        for kind in spec.PROBE_KINDS:
            if kind in confident:
                c = confident[kind]
                lines.append(f"| {kind} | {c['auc_full']:.3f} ({c['n_full']}) | {c['auc']:.3f} ({c['n']}) |")
    lines += [
        "",
        "### Footnote: 260924's full-size reference numbers",
        "",
        "260924's final recipe on 260923's `pool` activations (train 2908 / 2908 / 2908 / 1928 / 980 rows, as review/parts/reproduction_*.json record; all 3 samples per prompt): "
        + ", ".join(f"{kind} {spec.REPRODUCTION_TARGETS[PROBE_NAME_OF_KIND[kind]]:.3f}" for kind in spec.PROBE_KINDS)
        + ". Those probes were trained at the full size with three rows per prompt, so they are not the same procedure as the arms above.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    (arms_dir.parent / "comparison.json").write_text(json.dumps({"spec_hash": spec.spec_hash(), "arms": list(runs), "deltas": deltas}, indent=2))
    print("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
