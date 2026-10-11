"""261007's no-CoT logprob probes re-scored on the eval rows whose no-cue prompt is confident.

    python nocot_nocue_slice.py         # -> nocot_comparison.md, nocot_comparison.json

No training and no model: 261007's nocot_logprob final fit (trained on all
train questions) and its saved per-row eval scores. A row is kept when the
no-cue prompt of its question has a top letter with probability >= 0.75
(P renormalized over A and B; 261007's label.is_confident on p_a_none).
261007's confident slice also required the cued prompt to be confident. That
filter is recomputed first through this script's filter code and compared with
261007's own compare.confident_slice_aucs (row counts asserted, AUCs reported).
The CoT numbers beside them are this folder's comparison.json: the
slice-trained and the baseline probe on the CoT slice rows. The two formats
are scored on different completions and different question sets, so no Δ is
computed between them.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

import slice_spec as S
import slice_pipeline  # noqa: F401  (puts 261007 first on sys.path)
import compare as ref_compare  # noqa: E402  (261007's compare.py)
import label as ref_label  # noqa: E402  (261007's label.py)
from slice_pipeline import ref_spec  # noqa: E402

NOCOT_ARM = "nocot_logprob"
NOCOT_LABELS = S.REF_DIR / "arms" / NOCOT_ARM / "labels.jsonl"


def filtered_auc(scores: dict, keep: np.ndarray) -> dict:
    labels = np.array(scores["label"])[keep]
    score = np.array(scores["score"])[keep]
    questions = {q for q, k in zip(scores["question_id"], keep) if k}
    return {"auc": float(roc_auc_score(labels, score)), "n": int(keep.sum()), "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()), "n_questions": len(questions)}


def main() -> None:
    assert ref_spec.CONFIDENT_SLICE == 0.75, ref_spec.CONFIDENT_SLICE
    run = ref_compare.load_run(S.REF_DIR / "runs", NOCOT_ARM)
    eval_label_rows = [r for r in ref_compare.load_jsonl(NOCOT_LABELS) if r["split"] == "eval"]
    label_row_of_prompt = {(r["question_id"], r["condition"]): r for r in eval_label_rows}
    assert len(label_row_of_prompt) == len(eval_label_rows), "two label rows for one prompt"

    # the no-cue prompt is shared by both cue conditions of a question
    p_a_none_of_question: dict[int, float | None] = {}
    for r in eval_label_rows:
        assert p_a_none_of_question.setdefault(r["question_id"], r["p_a_none"]) == r["p_a_none"], r["question_id"]
    nocue_confident_questions = {q for q, p in p_a_none_of_question.items() if ref_label.is_confident(p)}

    their_both = ref_compare.confident_slice_aucs(run, NOCOT_LABELS)
    cot = json.loads((S.EXP_DIR / "comparison.json").read_text())["probes"]
    cot_slice_questions = json.loads((S.EXP_DIR / "slice_summary.json").read_text())["eval"]["slice_questions"]

    results = {}
    for kind in S.PROBE_KINDS:
        entry = run["by_kind"][kind]
        probe_name, scores = entry["summary"]["probe"], entry["scores"]
        prompts = list(zip(scores["question_id"], scores["condition"]))
        assert len(prompts) == len(set(prompts)), f"{probe_name}: two eval rows for one prompt"
        rows = [label_row_of_prompt[p] for p in prompts]
        assert all(r["p_a_none"] is not None for r in rows), f"{probe_name}: an eval row without a no-cue probability"

        keep_all = np.ones(len(rows), dtype=bool)
        keep_both = np.array([r["confident"] is True for r in rows])
        keep_nocue = np.array([ref_label.is_confident(r["p_a_none"]) for r in rows])
        assert not (keep_both & ~keep_nocue).any(), f"{probe_name}: a both-confident row whose no-cue prompt is not confident"

        both = filtered_auc(scores, keep_both)
        assert both["n"] == their_both[kind]["n"], (probe_name, both["n"], their_both[kind]["n"])
        nocue = filtered_auc(scores, keep_nocue)
        nocue_questions = {q for q, _ in prompts if q in nocue_confident_questions}
        assert len(nocue_questions) == nocue["n_questions"]
        print(f"check {probe_name}: both-confident n {both['n']} (261007 {their_both[kind]['n']}), AUC {both['auc']:.6f} (261007 {their_both[kind]['auc']:.6f})")

        c = cot[probe_name]
        results[probe_name] = {
            "nocot_all": filtered_auc(scores, keep_all),
            "nocot_all_saved_auc": entry["summary"]["auc_combined"],
            "nocot_both_confident": both,
            "nocot_both_confident_261007_auc": their_both[kind]["auc"],
            "nocot_nocue_confident": nocue,
            "cot_slice": {
                "auc_slice_trained": c["auc_slice_trained"],
                "auc_baseline": c["auc_baseline"],
                "n": c["n_eval"],
                "n_pos": c["n_eval_pos"],
                "n_neg": c["n_eval_neg"],
                "n_questions": c["n_eval_questions"],
            },
        }

    nocot_question_count = len(nocue_confident_questions)
    (S.EXP_DIR / "nocot_comparison.json").write_text(json.dumps({
        "confident_threshold": ref_spec.CONFIDENT_SLICE,
        "nocot_eval_questions": len(p_a_none_of_question),
        "nocot_nocue_confident_eval_questions": nocot_question_count,
        "cot_slice_eval_questions": cot_slice_questions,
        "probes": results,
    }, indent=2))

    t = ref_spec.CONFIDENT_SLICE
    lines = [
        "# No-CoT logprob probe on its no-cue-confident eval rows, beside the CoT slice probes",
        "",
        f"No-CoT probe: 261007's `{NOCOT_ARM}` final fit, trained on all train questions; its saved eval scores, re-scored here with no retraining. "
        f"No-cue confident: the question's no-cue prompt has a top letter with probability >= {t} (renormalized over A and B); "
        f"{nocot_question_count} of {len(p_a_none_of_question)} eval questions. "
        "AUC of the mean sigmoid over each probe's band; one row per prompt.",
        "",
        "## No-CoT probe, three eval filters",
        "",
        "| Probe | All eval rows: AUC (rows) | Cued and no-cue prompts confident (261007): AUC (rows) | No-cue prompt confident: AUC (rows, pos / neg, questions) |",
        "|---|---|---|---|",
    ]
    for probe_name, r in results.items():
        a, b, n = r["nocot_all"], r["nocot_both_confident"], r["nocot_nocue_confident"]
        lines.append(f"| {probe_name} | {a['auc']:.3f} ({a['n']}) | {b['auc']:.3f} ({b['n']}) | {n['auc']:.3f} ({n['n']}, {n['n_pos']} / {n['n_neg']}, {n['n_questions']}) |")
    lines += [
        "",
        "## Each format on its own no-cue-confident eval rows",
        "",
        f"CoT rows: the {cot_slice_questions} slice questions (at least {S.MIN_AGREE} of {S.N_NOCUE_SAMPLES} no-cue CoT samples agree), one CoT rollout per prompt, labels from that answer (`comparison.json`). "
        f"No-CoT rows: the {nocot_question_count} questions whose no-cue top letter has probability >= {t}, one-letter answers, labels from the top letters. "
        "The two formats are scored on different completions and different question sets; no Δ is computed between them. "
        "Main result of this experiment: the slice-trained CoT probe.",
        "",
        "| Probe | CoT slice-trained probe AUC | CoT baseline probe AUC | CoT rows (pos / neg), questions | No-CoT probe AUC | No-CoT rows (pos / neg), questions |",
        "|---|---|---|---|---|---|",
    ]
    for probe_name, r in results.items():
        c, n = r["cot_slice"], r["nocot_nocue_confident"]
        lines.append(
            f"| {probe_name} | {c['auc_slice_trained']:.3f} | {c['auc_baseline']:.3f} | {c['n']} ({c['n_pos']} / {c['n_neg']}), {c['n_questions']} | "
            f"{n['auc']:.3f} | {n['n']} ({n['n_pos']} / {n['n_neg']}), {n['n_questions']} |"
        )
    (S.EXP_DIR / "nocot_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
