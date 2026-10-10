"""Post-hoc analyses of the finished run (not part of the unattended queue).

    python followup_analysis.py      # completions.jsonl, labels.jsonl, runs/ -> followup_analysis.md

1. p_none distribution: per eval question, the share of the 8 fresh no-cue samples that
   are correct; per bias_followed eval row, p_none by cue condition.
2. Old 3 vs fresh 8: per eval question, the share correct in 260819's 3 no-cue samples
   against the share correct in the 8 fresh ones; agreement of the two strict majorities.
3. Old hard vs fresh hard: each probe's AUC on the old labels over all eval rows, on the
   old labels over the rows the fresh-hard rule keeps, and on the fresh labels over those
   rows, for the hard-trained and the soft-trained arm.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import soft_spec as S
from domain_imports import majority_vote

SAID_OF = {"bias_followed_rm_answer": None, "deception_yes_rm_preyesno": "Yes", "deception_no_rm_preyesno": "No"}
BUCKETS = ("0", "between 0 and 1/2", "exactly 1/2", "between 1/2 and 1", "1")


def bucket(p: float) -> str:
    if p == 0:
        return "0"
    if p < 0.5:
        return "between 0 and 1/2"
    if p == 0.5:
        return "exactly 1/2"
    if p < 1:
        return "between 1/2 and 1"
    return "1"


def share_correct(contents: list[str | None]) -> float | None:
    parsed = [c for c in contents if c is not None]
    return sum(c == "G" for c in parsed) / len(parsed) if parsed else None


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    fresh: dict[int, list] = defaultdict(list)
    for r in read_jsonl(S.NEW_COMPLETIONS):
        fresh[r["question_id"]].append(r["content"])
    old = {r["question_id"]: r["contents"] for r in read_jsonl(S.BASELINE_SCORES) if r["condition"] == "none" and r["question_id"] in fresh}
    assert set(old) == set(fresh) and len(fresh) == 687
    labels = read_jsonl(S.LABELS)
    out = ["# Follow-up analyses (post-hoc)", "", "Written by `followup_analysis.py` after the run; not part of the unattended queue.", ""]

    # 1. p_none distribution
    fresh_share = {q: share_correct(c) for q, c in fresh.items()}
    counts = Counter(bucket(p) for p in fresh_share.values())
    out += ["## 1. p_none distribution", "", "Per eval question: share of the 8 fresh no-cue samples that are correct.", "",
            "| share correct | questions |", "|---|---|"]
    out += [f"| {b} | {counts[b]} ({counts[b] / len(fresh_share):.0%}) |" for b in BUCKETS]
    all_parsed = Counter(round(p * 8) for q, p in fresh_share.items() if all(c is not None for c in fresh[q]))
    out += ["", f"Questions with all 8 samples parsed ({sum(all_parsed.values())}), by number correct:", "",
            "| correct of 8 | " + " | ".join(str(k) for k in range(9)) + " |", "|---" * 10 + "|",
            "| questions | " + " | ".join(str(all_parsed[k]) for k in range(9)) + " |", ""]
    bf = [r for r in labels if r["probe_set"] == "bias_followed_rm" and r["split"] == "eval"]
    by_cond = {c: Counter(bucket(r["p_none_8"]) for r in bf if r["condition"] == c) for c in ("cue_gold", "cue_dist")}
    out += ["bias_followed eval rows (every row followed the cue): p_none from the 8 fresh samples, by cue condition.", "",
            f"| p_none | cue_gold rows ({sum(by_cond['cue_gold'].values())}) | cue_dist rows ({sum(by_cond['cue_dist'].values())}) |", "|---|---|---|"]
    out += [f"| {b} | {by_cond['cue_gold'][b]} | {by_cond['cue_dist'][b]} |" for b in BUCKETS]
    out.append("")

    # 2. old 3 vs fresh 8
    def old_row(contents):
        return f"{sum(c == 'G' for c in contents)} of 3" if all(c is not None for c in contents) else "some unparsed"
    cross = Counter((old_row(old[q]), bucket(fresh_share[q])) for q in fresh)
    rows_order = ["0 of 3", "1 of 3", "2 of 3", "3 of 3", "some unparsed"]
    out += ["## 2. Old 3 samples vs fresh 8 samples", "", "Per eval question: correct among 260819's 3 no-cue samples (rows) against the share correct among the 8 fresh samples (columns).", "",
            "| old 3 samples | questions | " + " | ".join(BUCKETS) + " |", "|---" * (len(BUCKETS) + 2) + "|"]
    for k in rows_order:
        out.append(f"| {k} | {sum(cross[(k, b)] for b in BUCKETS)} | " + " | ".join(str(cross[(k, b)]) for b in BUCKETS) + " |")
    agree = sum(majority_vote(old[q]) == majority_vote(fresh[q]) for q in fresh)
    out += ["", f"Strict majority of the 3 old samples equals the strict majority of the 8 fresh samples (a tie counts as no majority): {agree} of {len(fresh)} questions.", ""]

    # 3. old hard vs fresh hard
    runs = {arm: json.loads((S.RUNS_DIR / arm / "eval_scores.json").read_text()) for arm in ("hard01", "soft")}
    out += ["## 3. Old hard vs fresh hard labels", "",
            "AUC on three label/row combinations. Fresh hard drops a row when the 8 fresh samples tie 4-4, or when the row becomes anti-bias (deception rows).", "",
            "| probe | arm | old labels, all rows | old labels, rows fresh hard keeps | fresh labels, same rows | rows dropped | kept rows whose label flips |", "|---|---|---|---|---|---|---|"]
    for probe, said in SAID_OF.items():
        by_key = {(r["question_id"], r["condition"], r["sample_idx"]): r for r in labels
                  if r["probe_set"] == S.PROBE_SET_OF[probe] and r["split"] == "eval" and r["said"] == said}
        for arm, label in (("hard01", "hard-trained"), ("soft", "soft-trained")):
            scores = runs[arm][probe]
            rows = [by_key[k] for k in zip(scores["question_id"], scores["condition"], scores["sample_idx"])]
            s = np.array(scores["score"])
            old_label = np.array([r["fit_b_label"] for r in rows])
            keep = np.array([r["hard_8"] is not None for r in rows])
            fresh_label = np.array([r["hard_8"] if r["hard_8"] is not None else -1 for r in rows])
            out.append(
                f"| {probe} | {label} | {roc_auc_score(old_label, s):.3f} | {roc_auc_score(old_label[keep], s[keep]):.3f} | "
                f"{roc_auc_score(fresh_label[keep], s[keep]):.3f} | {int((~keep).sum())} | {int((fresh_label[keep] != old_label[keep]).sum())} |"
            )
    (S.EXP_DIR / "followup_analysis.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
