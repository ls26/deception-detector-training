"""Check: the selfreport stage's P(Yes) code gives 261007's saved float32 P(Yes) on saved follow-up prompts.

    python check_pyes.py [--n 50]     # needs the GPU; reads dry/saved_base/ (check_scoring.py) -> check_pyes.json

Builds the first N follow-up prompts of the saved base rollouts with eval_checkpoint.followup_items, scores
them with the base model through eval_checkpoint.score_last_token, and reports the largest |P(Yes) - saved|
and how many thresholded Yes / No agree. Batching and padding differ from 261007's run, so small float
differences are expected; they are reported, not asserted.
"""

from __future__ import annotations

import argparse
import json

import torch

import noisy_spec as S
from eval_checkpoint import followup_items, key_of, load_hf_model, read_jsonl, said_of, score_last_token

SAVED_DIR = S.EXP_DIR / "dry" / "saved_base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    saved = {key_of(r): r["p_yes"] for r in read_jsonl(SAVED_DIR / "followups.jsonl")}
    model, tokenizer = load_hf_model(None)
    items = followup_items(read_jsonl(SAVED_DIR / "completions.jsonl"), tokenizer)[: args.n]
    assert len(items) == args.n
    scores = score_last_token(model, tokenizer, [item["text"] for item in items])
    rows = []
    for item, (lp_yes, lp_no, _) in zip(items, scores):
        p_yes = float(torch.sigmoid(torch.tensor(lp_yes - lp_no, dtype=torch.float32)))
        rows.append({"key": list(key_of(item)), "wording": item["wording"], "p_yes": p_yes, "saved_p_yes": saved[key_of(item)]})
    diffs = [abs(r["p_yes"] - r["saved_p_yes"]) for r in rows]
    said_agree = sum(said_of(r["p_yes"]) == said_of(r["saved_p_yes"]) for r in rows)
    result = {
        "n": len(rows),
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "said_agree": said_agree,
        "wordings": {w: sum(r["wording"] == w for r in rows) for w in ("plain", "explicit")},
        "rows": rows,
    }
    (S.EXP_DIR / "check_pyes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(rows)} prompts: max |diff| {result['max_abs_diff']:.2e}, mean {result['mean_abs_diff']:.2e}, "
          f"said agrees on {said_agree}/{len(rows)}; wordings {result['wordings']}")


if __name__ == "__main__":
    main()
