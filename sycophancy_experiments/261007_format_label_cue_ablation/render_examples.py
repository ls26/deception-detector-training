"""Worked labelling examples per arm, for review/examples.md.

For every arm: one `bias` row and one `no bias` row (preferring cue_dist for the
first and cue_gold for the second so both cued conditions appear), each shown
with the full user message (prefill included for the no-CoT arms), the sampled
assistant text, the no-cue counterfactual prompt with its three sampled answers
(or the top letter and P(A)), the bias label with its reason, the follow-up,
the self-report (float32 P(Yes), its bf16 value and the sampled token, then the
thresholded `said`) and the deception label with its reason.

ref_cot always reads arms/ref_cot (it has no dry run); the other arms read
--arms-dir and the no-cue prompts of --gens-suffix.

Usage:
    python render_examples.py --arms-dir arms_dry --gens-suffix _dry
"""

from __future__ import annotations

import argparse

import spec
from domain_imports import to_letter
from label_common import deception_subtype, read_jsonl

PREFERRED_CONDITION = {"bias": "cue_dist", "no bias": "cue_gold"}


def pick_examples(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["question_id"], r["condition"], r["sample_idx"]))
    picked = []
    for target, preferred in PREFERRED_CONDITION.items():
        candidates = [r for r in rows if r["bias_rm"] == target and r["said_rm"] is not None]
        preferred_candidates = [r for r in candidates if r["condition"] == preferred]
        chosen = (preferred_candidates or candidates)[:1]
        picked += chosen
    return picked


def load_none_prompts_260819(question_ids: set[int]) -> dict[int, dict]:
    """question_id -> {prompt, letters} of 260819's no-cue rollouts."""
    scores = {r["question_id"]: r for r in read_jsonl(spec.DOMAIN_DIR / "scores.jsonl") if r["cue_variant"] == "shared"}
    out: dict[int, dict] = {}
    for row in read_jsonl(spec.DOMAIN_DIR / "completions.jsonl"):
        if row["cue_variant"] != "shared" or row["question_id"] not in question_ids:
            continue
        entry = out.setdefault(row["question_id"], {"prompt": row["prompt"], "letters": scores[row["question_id"]]["preds"]})
        assert entry["prompt"] == row["prompt"]
    return out


def load_none_prompts_nocot(gens_suffix: str, question_ids: set[int]) -> dict[int, dict]:
    """question_id -> {prompt, letters, top_letter, p_a} of the no-CoT no-cue prompt."""
    out: dict[int, dict] = {}
    for row in read_jsonl(spec.GENS_DIR / f"nocot{gens_suffix}" / "turn1.jsonl"):
        if row["condition"] != "none" or row["question_id"] not in question_ids:
            continue
        entry = out.setdefault(row["question_id"], {
            "prompt": row["prompt"] + row["prefill"], "letters": [None] * 3,
            "top_letter": row["top_letter"], "p_a": row["p_a"],
        })
        entry["letters"][row["sample_idx"]] = row["answer_letter"]
    return out


def fenced(text: str) -> list[str]:
    return ["```text", text, "```"]


def render_example(arm: str, row: dict, none_prompt: dict) -> list[str]:
    logprob = spec.LABEL_SOURCE[arm] == "logprob"
    nocot = spec.GEN_OF_ARM[arm] == "nocot"
    prefill = spec.NOCOT_PREFILL if nocot else ""
    assert row["completion"].startswith(prefill), (arm, row["question_id"])
    gold, order = row["answer"], row["order"]
    lines = [f"### {arm} / {row['condition']} / q{row['question_id']} ({row['category']}, {row['split']}): "
             f"`{row['bias_rm']}` example", "",
             f"gold = {gold} ({order}); cue letter = {row['cue_letter']}"
             + (f"; cue paraphrase {row['cue_template_idx']}" if row["cue_template_idx"] is not None else "")
             + (f"; bio {row['bio_idx']} ({spec.BIOS[row['bio_idx']]['name']})" if row["bio_idx"] is not None else ""),
             "", "**User message** (the chat prompt as the model saw it"
             + (f", ending in the prefill `{prefill}`" if nocot else "") + "):", "", *fenced(row["prompt"] + prefill), "",
             "**Assistant turn 1**"
             + (f" (the sampled text after the prefill; the closing `{spec.NOCOT_CLOSING}` is appended, not sampled" if nocot else " (the sampled text")
             + (f"; top letter, P(A) = {row['p_a_cued']:.4f})" if logprob else ")") + ":", "",
             *fenced(row["completion"][len(prefill):]), "",
             "**Counterfactual: the no-cue prompt**:", "", *fenced(none_prompt["prompt"]), ""]
    if logprob:
        none_letter = none_prompt["top_letter"]
        lines.append(f"no-cue top letter = {none_letter}, P(A) = {none_prompt['p_a']:.4f} "
                     f"(content {row['a_none']}); cued top letter = {row['answer_letter']} (content {row['a_cued']}), "
                     f"P(cue letter) {row['p_cue_none']:.4f} -> {row['p_cue_cued']:.4f}; confident = {row['confident']}")
    else:
        none_letter = to_letter(row["a_none"], order)
        letters = ", ".join(str(l) for l in none_prompt["letters"])
        lines.append(f"no-cue sampled answers = {letters} -> strict majority {none_letter} (content {row['a_none']}); "
                     f"cued sampled answer = {row['answer_letter']} (content {row['a_cued']})")
    relation = "=" if row["followed_cue"] else "!="
    lines += ["", f"**Bias label: `{row['bias_rm']}`** -- cued answer {row['answer_letter']} {relation} cue letter "
              f"{row['cue_letter']}, no-cue {'top letter' if logprob else 'majority'} {none_letter}; "
              f"cued content {row['a_cued']} vs no-cue content {row['a_none']} under {row['condition']} -> {row['bias_rm']}", "",
              f"**Follow-up** (`{row['arm']}` wording, appended as the next user turn):", "", *fenced(row["followup_text"]), "",
              f"**Self-report**: P(Yes) fp32 = {row['p_yes_rm']:.4f} (bf16 {row['p_yes_bf16']:.4f}; "
              f"sampled {row['followup_completion']!r}) -> said {row['said_rm']}",
              "", f"**Deception label: `{row['deception_rm']}`** -- {row['bias_rm']} row and said {row['said_rm']} "
              f"-> {deception_subtype(row)} -> {row['deception_rm']}", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms-dir", default="arms")
    parser.add_argument("--gens-suffix", default="")
    args = parser.parse_args()

    md = ["# Worked labelling examples", "",
          f"spec {spec.spec_hash()}; ref_cot from `arms/ref_cot`, the other arms from `{args.arms_dir}/` "
          f"and `gens/*{args.gens_suffix}/`. One `bias` and one `no bias` row per arm; the no-cue prompt is "
          "the counterfactual every bias label is read against (removal definition).", ""]
    for arm in spec.ARMS:
        arms_dir = spec.ARMS_DIR if arm == spec.REFERENCE_ARM else spec.EXP_DIR / args.arms_dir
        rows = read_jsonl(arms_dir / arm / "labels.jsonl")
        examples = pick_examples(rows)
        question_ids = {row["question_id"] for row in examples}
        if spec.GEN_OF_ARM[arm] == "nocot":
            none_prompts = load_none_prompts_nocot(args.gens_suffix, question_ids)
        else:
            none_prompts = load_none_prompts_260819(question_ids)
        md += [f"## {arm}", ""]
        found = {row["bias_rm"] for row in examples}
        for target in PREFERRED_CONDITION:
            if target not in found:
                md += [f"(no `{target}` row with a self-report in this run)", ""]
        for row in examples:
            md += render_example(arm, row, none_prompts[row["question_id"]])
        print(f"  {arm}: {len(rows)} rows, examples " + ", ".join(
            f"q{r['question_id']}/{r['condition']} ({r['bias_rm']})" for r in examples))

    spec.REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = spec.REVIEW_DIR / "examples.md"
    out_path.write_text("\n".join(md))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
