"""Hard-trained vs soft-trained probes on three eval label sets, per probe.

    python compare_soft.py      # runs/{int,hard01,soft}, labels.jsonl -> comparison.md, comparison.json
    python compare_soft.py --runs-dir dry/runs --labels dry/labels.jsonl --out-dir dry   # dry run

Eval label sets, on fit B's eval rows of each probe:
  old hard    fit B's label (strict majority of the 3 original no-cue samples)
  fresh hard  the same rule on the 8 fresh samples; rows the rule drops are counted
  fresh soft  soft_8: the soft label from the 8 fresh samples
Hard-trained = runs/hard01, soft-trained = runs/soft.

Metric: pairwise AUC with each row a positive of weight q and a negative of weight
1 - q, same-row pairs excluded, score ties counting half. On 0/1 labels this is the
usual AUC. CI: question-level bootstrap of soft-trained minus hard-trained, the same
draws for both, a row drawn k times weighted k, BOOTSTRAP_DRAWS draws, BOOTSTRAP_SEED.
Also reported: the fresh-soft ceiling (q itself as the score), Spearman(score, q),
and the library check (int refit, hard01 refit, fit B's saved scores).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import soft_spec as S

SAID_OF = {"bias_followed_rm_answer": None, "deception_yes_rm_preyesno": "Yes", "deception_no_rm_preyesno": "No"}
COLUMNS = ("old hard", "fresh hard", "fresh soft")


def pair_matrix(scores: np.ndarray) -> np.ndarray:
    """K[i, j] = 1 if score i > score j, 0.5 if equal, 0 otherwise."""
    return (scores[:, None] > scores[None, :]) + 0.5 * (scores[:, None] == scores[None, :])


def weighted_auc(pairs: np.ndarray, q: np.ndarray, w: np.ndarray) -> float:
    """Sum over pairs of different rows of w_i w_j q_i (1 - q_j) K_ij, over the same sum without K.
    The diagonal (a row with itself, K_ii = 0.5) is removed from both sums."""
    pos, neg = w * q, w * (1.0 - q)
    same_row = np.sum(w * w * q * (1.0 - q))
    denominator = pos.sum() * neg.sum() - same_row
    assert denominator > 0, "no positive-negative weight across rows (one class only)"
    return float((pos @ pairs @ neg - 0.5 * same_row) / denominator)


def bootstrap_delta(question_ids: np.ndarray, q: np.ndarray, pairs_hard: np.ndarray, pairs_soft: np.ndarray) -> dict:
    """Soft-trained minus hard-trained, question-level bootstrap with the same draws for both."""
    questions, row_question = np.unique(question_ids, return_inverse=True)
    rng = np.random.default_rng(S.BOOTSTRAP_SEED)
    deltas = np.empty(S.BOOTSTRAP_DRAWS)
    for draw in range(S.BOOTSTRAP_DRAWS):
        times_drawn = np.bincount(rng.integers(0, len(questions), len(questions)), minlength=len(questions))
        w = times_drawn[row_question].astype(float)
        deltas[draw] = weighted_auc(pairs_soft, q, w) - weighted_auc(pairs_hard, q, w)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {"ci_low": float(low), "ci_high": float(high), "excludes_zero": bool(low > 0 or high < 0)}


def load_eval_labels(labels_path: Path, probe: str) -> dict[tuple, dict]:
    probe_set = S.PROBE_SET_OF[probe]
    out = {}
    for line in labels_path.read_text().splitlines():
        rec = json.loads(line)
        if rec["probe_set"] == probe_set and rec["split"] == "eval" and rec["said"] == SAID_OF[probe]:
            out[(rec["question_id"], rec["condition"], rec["sample_idx"])] = rec
    return out


def compare_probe(probe: str, runs: dict[str, dict], fit_b_saved: dict, labels_path: Path, full_run: bool) -> dict:
    hard, soft = runs["hard01"][probe], runs["soft"][probe]
    keys = list(zip(hard["question_id"], hard["condition"], hard["sample_idx"]))
    assert keys == list(zip(soft["question_id"], soft["condition"], soft["sample_idx"])), "arms scored different rows"
    labels = load_eval_labels(labels_path, probe)
    assert set(keys) == set(labels), "eval rows differ between eval_scores.json and labels.jsonl"
    rows = [labels[k] for k in keys]
    assert [r["fit_b_label"] for r in rows] == hard["fit_b_label"], "fit B labels differ"

    score_hard, score_soft = np.array(hard["score"]), np.array(soft["score"])
    question_ids = np.array([k[0] for k in keys])
    has_fresh = np.array(["soft_8" in r for r in rows])
    if full_run:
        assert has_fresh.all(), "eval rows without fresh labels"

    column_rows = {
        "old hard": (np.ones(len(rows), dtype=bool), np.array([r["fit_b_label"] for r in rows], dtype=float)),
        "fresh hard": (has_fresh & np.array([r.get("hard_8") is not None for r in rows]), np.array([r.get("hard_8") or 0 for r in rows], dtype=float)),
        "fresh soft": (has_fresh, np.array([r.get("soft_8", 0.0) for r in rows], dtype=float)),
    }
    result: dict = {"n_eval_rows": len(rows), "n_fresh_hard_dropped": int((has_fresh & np.array([r.get("hard_8") is None for r in rows])).sum())}
    for column, (keep, q_all) in column_rows.items():
        q = q_all[keep]
        pairs_hard, pairs_soft = pair_matrix(score_hard[keep]), pair_matrix(score_soft[keep])
        ones = np.ones(int(keep.sum()))
        entry = {
            "n_rows": int(keep.sum()),
            "n_questions": int(len(np.unique(question_ids[keep]))),
            "auc_hard_trained": weighted_auc(pairs_hard, q, ones),
            "auc_soft_trained": weighted_auc(pairs_soft, q, ones),
        }
        entry["delta"] = entry["auc_soft_trained"] - entry["auc_hard_trained"]
        entry.update(bootstrap_delta(question_ids[keep], q, pairs_hard, pairs_soft))
        if column == "fresh soft":
            entry["ceiling"] = weighted_auc(pair_matrix(q), q, ones)
            entry["spearman_hard_trained"] = float(spearmanr(score_hard[keep], q).statistic)
            entry["spearman_soft_trained"] = float(spearmanr(score_soft[keep], q).statistic)
        else:
            # The weighted AUC on 0/1 labels against sklearn's: reported, not asserted.
            entry["sklearn_auc_hard_trained"] = float(roc_auc_score(q, score_hard[keep]))
        result[column] = entry

    int_scores = np.array(runs["int"][probe]["score"])
    result["library_check"] = {
        "auc_fit_b_saved": float(roc_auc_score(fit_b_saved[probe]["label"], fit_b_saved[probe]["score"])),
        "auc_int_refit": float(roc_auc_score(hard["fit_b_label"], int_scores)),
        "auc_hard01_refit": float(roc_auc_score(hard["fit_b_label"], score_hard)),
        "max_abs_diff_int_vs_fit_b": float(np.max(np.abs(int_scores - np.array(fit_b_saved[probe]["score"])))),
        "max_abs_diff_hard01_vs_int": float(np.max(np.abs(score_hard - int_scores))),
    }
    return result


def fmt_delta(e: dict) -> str:
    star = "*" if e["excludes_zero"] else ""
    return f"{e['delta']:+.3f}{star} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}]"


def write_markdown(results: dict, path: Path) -> None:
    lines = [
        "# Comparison: hard-trained vs soft-trained probes",
        "",
        "Eval metric: pairwise AUC with soft weights (the usual AUC on 0/1 labels). Δ = soft-trained minus",
        f"hard-trained; 95% CI from a question-level bootstrap ({S.BOOTSTRAP_DRAWS} draws, seed {S.BOOTSTRAP_SEED});",
        "`*` marks a CI that excludes 0. One training seed (0).",
        "",
    ]
    for probe, r in results.items():
        lines += [f"## {probe}", "", "| eval labels | rows | hard-trained | soft-trained | Δ [95% CI] |", "|---|---|---|---|---|"]
        for column in COLUMNS:
            e = r[column]
            rows = f"{e['n_rows']}"
            if column == "fresh hard":
                rows += f" ({r['n_fresh_hard_dropped']} dropped)"
            lines.append(f"| {column} | {rows} | {e['auc_hard_trained']:.3f} | {e['auc_soft_trained']:.3f} | {fmt_delta(e)} |")
        fs = r["fresh soft"]
        lines += [
            "",
            f"Fresh-soft ceiling (q as the score): {fs['ceiling']:.3f}. "
            f"Spearman(score, q): hard-trained {fs['spearman_hard_trained']:.3f}, soft-trained {fs['spearman_soft_trained']:.3f}.",
            "",
        ]
    lines += ["## Library check", "", "| probe | fit B saved AUC | int refit AUC | hard01 refit AUC | max diff int vs fit B | max diff hard01 vs int |", "|---|---|---|---|---|---|"]
    for probe, r in results.items():
        c = r["library_check"]
        lines.append(
            f"| {probe} | {c['auc_fit_b_saved']:.6f} | {c['auc_int_refit']:.6f} | {c['auc_hard01_refit']:.6f} | "
            f"{c['max_abs_diff_int_vs_fit_b']:.2e} | {c['max_abs_diff_hard01_vs_int']:.2e} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=S.RUNS_DIR)
    parser.add_argument("--labels", type=Path, default=S.LABELS)
    parser.add_argument("--out-dir", type=Path, default=S.EXP_DIR)
    args = parser.parse_args()
    full_run = args.labels == S.LABELS

    runs = {mode: json.loads((args.runs_dir / mode / "eval_scores.json").read_text()) for mode in S.TARGET_MODES}
    fit_b_saved = json.loads((S.FIT_B_RUN_DIR / "eval_scores.json").read_text())
    results = {probe: compare_probe(probe, runs, fit_b_saved, args.labels, full_run) for probe in S.PROBES}

    (args.out_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    write_markdown(results, args.out_dir / "comparison.md")
    print((args.out_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
