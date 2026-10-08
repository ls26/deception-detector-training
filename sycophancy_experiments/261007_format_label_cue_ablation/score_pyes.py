"""Score P(Yes) at the first follow-up token in float32 (SCHEMA.md, `pyes_fp32.jsonl`).

The self-report every arm reads: the bf16 model's final residual (transformers'
post-norm last hidden state) at the LAST PROMPT TOKEN of the follow-up turn,
projected through `lm_head.weight` cast to float32, log-softmax over the
vocabulary in float32, then p_yes_fp32 = sigmoid(lp[Yes] - lp[No]). vLLM's bf16
logprobs (`p_yes_pos0`, kept as `p_yes_bf16` for comparison) sit on a 1/8-nat
grid and leave exact ties at 0.5; label.py thresholds p_yes_fp32 at
spec.SAID_THRESHOLD.

  nocot / bio   gens/<gen>/followups.jsonl, one row per (question_id, condition,
                sample_key); writes gens/<gen>/pyes_fp32.jsonl.
  ref           260823's follow-ups of the question's assigned removal wording,
                joined to 260819's turn-1 text (label.py's ref_cot loader); one
                row per (question_id, condition, sample_idx); writes
                arms/ref_cot/pyes_fp32.jsonl.

Usage:
    python score_pyes.py --gen nocot [--dry-run] [--batch-size 16]
    python score_pyes.py --gen ref
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import spec
from domain_imports import ARM_OF, draw_wordings, import_followup_generation
from hf_env import bootstrap_hf_env
from label import load_ref_sources
from label_common import read_jsonl, write_jsonl
from probe_config import PipelineConfig
from step2_extract_activations import load_model_and_tokenizer, pad_batch

GEN_KEY_FIELDS = ("question_id", "condition", "sample_key")
REF_KEY_FIELDS = ("question_id", "condition", "sample_idx")
N_REF_PROMPTS = 13533  # 13740 cued rollouts minus the explicit wording's skipped unparsed answers


# --------------------------------------------------------------------------- #
# the prompts to score: {key fields, "text": the turn-2 prompt, "p_yes_bf16"}
# --------------------------------------------------------------------------- #


def turn2_prompt(record: dict, followup_text: str, tokenizer, followup_generation, suffix_cache: dict[str, str]) -> str:
    """prompt + completion + 260823's template suffix for this wording, byte-for-byte."""
    if followup_text not in suffix_cache:
        suffix_cache[followup_text] = followup_generation.build_followup_suffix(tokenizer, followup_text)
    return followup_generation.build_followup_prompt(record, suffix_cache[followup_text])


def gen_items(gen_dir: Path, tokenizer, followup_generation) -> list[dict]:
    suffix_cache: dict[str, str] = {}
    items = []
    for row in read_jsonl(gen_dir / "followups.jsonl"):
        assert row["spec_hash"] == spec.spec_hash(), f"{gen_dir}: row spec_hash {row['spec_hash']} != {spec.spec_hash()}"
        items.append({
            **{field: row[field] for field in GEN_KEY_FIELDS},
            "text": turn2_prompt(row, row["followup_text"], tokenizer, followup_generation, suffix_cache),
            "p_yes_bf16": row["p_yes_pos0"],
        })
    return items


def ref_items(tokenizer, followup_generation) -> list[dict]:
    completions, followups = load_ref_sources()
    suffix_cache: dict[str, str] = {}
    items = []
    for (question_id, condition, sample_idx, arm), followup in followups.items():
        if arm != ARM_OF["rm"][draw_wordings(question_id)["rm"][1]]:
            continue
        completion_row = completions[(question_id, condition, sample_idx)]
        items.append({
            "question_id": question_id,
            "condition": condition,
            "sample_idx": sample_idx,
            "text": turn2_prompt(completion_row, followup["followup_text"], tokenizer, followup_generation, suffix_cache),
            "p_yes_bf16": followup["p_yes_pos0"],
        })
    assert len(items) == N_REF_PROMPTS, len(items)
    return items


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def score_last_token(model, tokenizer, texts: list[str], batch_size: int) -> tuple[list[list[float]], list[int], float]:
    """Per text: [lp_yes, lp_no] (float32 log-softmax of the fp32 output head on
    the bf16 final residual at the last token) and the fp32 top-1 token id.
    Also returns the first batch's bf16-vs-fp32 argmax agreement (sanity check)."""
    token_id_lists = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
    order = sorted(range(len(texts)), key=lambda i: len(token_id_lists[i]))  # short to long
    head_weight = model.lm_head.weight.float()  # (vocab, hidden)
    yes_no_logprobs: list[list[float] | None] = [None] * len(texts)
    top1_ids: list[int | None] = [None] * len(texts)
    first_batch_argmax_agreement = None
    started = time.time()
    for batch_start in range(0, len(texts), batch_size):
        batch_indices = order[batch_start : batch_start + batch_size]
        input_ids, attention_mask = pad_batch(tokenizer, [token_id_lists[i] for i in batch_indices])
        batch_rows = torch.arange(len(batch_indices))
        last_positions = attention_mask.sum(1) - 1
        with torch.no_grad():
            outputs = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True)
            final_residual = outputs.hidden_states[-1][batch_rows, last_positions].float()
            logprobs_fp32 = torch.log_softmax(final_residual @ head_weight.T, dim=-1)
            if first_batch_argmax_agreement is None:
                bf16_argmax = outputs.logits[batch_rows, last_positions].argmax(-1)
                first_batch_argmax_agreement = float((bf16_argmax == logprobs_fp32.argmax(-1)).float().mean())
                print(f"  sanity check, first batch: bf16 logits argmax == fp32 argmax on {first_batch_argmax_agreement:.2%} of {len(batch_indices)} rows (want >= 90%)")
        for i, index in enumerate(batch_indices):
            yes_no_logprobs[index] = [float(logprobs_fp32[i, spec.YES_TOKEN_ID]), float(logprobs_fp32[i, spec.NO_TOKEN_ID])]
            top1_ids[index] = int(logprobs_fp32[i].argmax())
        del outputs
        done = batch_start + len(batch_indices)
        elapsed = time.time() - started
        print(f"  {done}/{len(texts)} rows, {elapsed:.0f}s elapsed, ETA {elapsed / done * (len(texts) - done):.0f}s (longest so far {input_ids.shape[1]} tokens)", flush=True)
    return yes_no_logprobs, top1_ids, first_batch_argmax_agreement


def p_yes_from_logprobs(lp_yes: float, lp_no: float) -> float:
    return float(torch.sigmoid(torch.tensor(lp_yes - lp_no, dtype=torch.float32)))


def summarize(rows: list[dict], top1_ids: list[int]) -> dict:
    not_tied_bf16 = [row for row in rows if row["p_yes_bf16"] != 0.5]
    tied_bf16 = [row for row in rows if row["p_yes_bf16"] == 0.5]
    sign_agreement = sum((row["p_yes_fp32"] > 0.5) == (row["p_yes_bf16"] > 0.5) for row in not_tied_bf16)
    abs_diffs = [abs(row["p_yes_fp32"] - row["p_yes_bf16"]) for row in rows]
    return {
        "n_rows": len(rows),
        "share_top1_yes_or_no": sum(top1 in (spec.YES_TOKEN_ID, spec.NO_TOKEN_ID) for top1 in top1_ids) / len(rows),
        "n_bf16_not_tied": len(not_tied_bf16),
        "sign_agreement_fp32_vs_bf16": sign_agreement / len(not_tied_bf16),
        "n_bf16_ties": len(tied_bf16),
        "n_bf16_ties_resolved": sum(row["p_yes_fp32"] != 0.5 for row in tied_bf16),
        "n_fp32_ties": sum(row["p_yes_fp32"] == 0.5 for row in rows),
        "n_fp32_said_yes": sum(row["p_yes_fp32"] > spec.SAID_THRESHOLD for row in rows),
        "mean_abs_diff_fp32_bf16": sum(abs_diffs) / len(rows),
        "max_abs_diff_fp32_bf16": max(abs_diffs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gen", required=True, choices=(*spec.GENS, "ref"))
    parser.add_argument("--dry-run", action="store_true", help="read gens/<gen>_dry/ (nocot / bio only)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    assert not (args.gen == "ref" and args.dry_run), "the reference arm has no dry run"

    started = time.time()
    print(f"=== score_pyes.py --gen {args.gen} dry_run={args.dry_run} (spec {spec.spec_hash()}) ===")
    bootstrap_hf_env()
    followup_generation = import_followup_generation()
    model, tokenizer = load_model_and_tokenizer(PipelineConfig(arm=spec.REFERENCE_ARM))  # only the model fields are read

    if args.gen == "ref":
        out_dir = spec.ARMS_DIR / spec.REFERENCE_ARM
        items = ref_items(tokenizer, followup_generation)
        key_fields = REF_KEY_FIELDS
    else:
        out_dir = spec.GENS_DIR / (f"{args.gen}_dry" if args.dry_run else args.gen)
        items = gen_items(out_dir, tokenizer, followup_generation)
        key_fields = GEN_KEY_FIELDS
    print(f"  {len(items)} follow-up prompts to score")

    yes_no_logprobs, top1_ids, first_batch_argmax_agreement = score_last_token(model, tokenizer, [item["text"] for item in items], args.batch_size)
    rows = []
    for item, (lp_yes, lp_no) in zip(items, yes_no_logprobs):
        rows.append({
            **{field: item[field] for field in key_fields},
            "lp_yes_fp32": lp_yes,
            "lp_no_fp32": lp_no,
            "p_yes_fp32": p_yes_from_logprobs(lp_yes, lp_no),
            "p_yes_bf16": item["p_yes_bf16"],
            "spec_hash": spec.spec_hash(),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pyes_fp32.jsonl", rows)
    summary = {
        "spec_hash": spec.spec_hash(),
        "gen": args.gen,
        "dry_run": args.dry_run,
        "batch_size": args.batch_size,
        "first_batch_bf16_fp32_argmax_agreement": first_batch_argmax_agreement,
        **summarize(rows, top1_ids),
        "seconds": time.time() - started,
    }
    (out_dir / "pyes_fp32_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"  wrote {len(rows)} rows to {out_dir / 'pyes_fp32.jsonl'}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
