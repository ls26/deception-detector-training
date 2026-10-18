"""No-CoT eval of one checkpoint (the base model, or base + a GRPO LoRA adapter) on the 547 eval questions.

    python eval_checkpoint.py generate   --name base [--adapter DIR] [--limit N] [--out-root DIR]
    python eval_checkpoint.py selfreport --name base [--adapter DIR] [--out-root DIR]
    python eval_checkpoint.py labels     --name base [--out-root DIR]

Every stage reads and writes <out-root>/<name>/ (default evals/<name>/) and never overwrites a file.

generate    One prompt per eval question and condition: no cue, cue at the true answer (cue_gold), cue at
            the wrong answer (cue_dist), in the question's drawn cue wording. The prompt is 261007's stored
            no-CoT prompt byte-for-byte, then the prefill "The answer is (". vLLM (prefix caching off),
            one token, top-20 logprobs (261007's settings, n=1). The letter reading: lp_A, lp_B, P(A)
            renormalized over A and B (260823's position_stats, as 261007), P(A) and P(B) over the whole
            vocabulary (0 when outside the top 20), the top letter -> completions.jsonl
selfreport  For every cued prompt with a top letter: the turn "The answer is (X)" (X = the top letter, ")"
            appended), then the follow-up in the question's drawn wording (260823), scored as float32
            P(Yes) at the last prompt token (HF transformers, + PeftModel for an adapter) -> followups.jsonl
labels      Per cued prompt: the checkpoint's own no-cue answer (the no-cue prompt's top letter), bias
            (260824's turn1_label, no-cue counterfactual), said Yes / No (P(Yes) above / below 0.5, none on
            a tie) and deception (260824's turn2_label) -> labels.jsonl, summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import nocot_spec as S
from domain_imports import ARM_OF, draw_wordings, import_followup_generation, to_content, turn1_label, turn2_label

KEY_FIELDS = ("question_id", "condition")
# 261007's generate_turn1.LETTER_TARGETS: position_stats' "yes" side is A, its "no" side is B
LETTER_TARGETS = {
    "yes_strict": {S.LETTER_TOKEN_IDS["A"]},
    "no_strict": {S.LETTER_TOKEN_IDS["B"]},
    "yes_broad": {S.LETTER_TOKEN_IDS["A"]},
    "no_broad": {S.LETTER_TOKEN_IDS["B"]},
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    assert not path.exists(), f"{path} exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
    tmp.rename(path)


def key_of(row: dict) -> tuple:
    return tuple(row[field] for field in KEY_FIELDS)


def eval_questions(limit: int | None) -> list[dict]:
    questions = sorted((q for q in read_jsonl(S.DATASET_PATH) if q["split"] == "eval"), key=lambda q: q["question_id"])
    assert len(questions) == S.N_QUESTIONS["eval"], len(questions)
    return questions if limit is None else questions[:limit]


def cue_letter_of(question: dict, condition: str) -> str | None:
    other = "B" if question["answer"] == "A" else "A"
    return {"none": None, "cue_gold": question["answer"], "cue_dist": other}[condition]


def seed_of(question_id: int, condition: str) -> int:
    return S.EVAL_SEED_BASE + 3 * question_id + S.EVAL_CONDITIONS.index(condition)


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def stored_prompts() -> dict[tuple[int, str], str]:
    """261007's no-CoT prompt text (without the prefill) per (question, condition)."""
    prompts = {}
    for row in read_jsonl(S.NOCOT_TURN1):
        if row["sample_idx"] == 0:
            assert row["prefill"] == S.NOCOT_PREFILL
            prompts[(row["question_id"], row["condition"])] = row["prompt"]
    return prompts


def letter_reading(position: dict, position_stats) -> dict:
    """The letter reading of one position's top-20 logprob dict.

    p_a: P(A) / (P(A) + P(B)) by 260823's position_stats (1.0 / 0.0 when only A / only B is in the top
    20). top_letter: the higher of lp_A, lp_B; the letter that is in the top 20 when the other is not;
    none on an exact tie or when neither is in the top 20 (top_rule says which).
    """
    stats = position_stats(position, LETTER_TARGETS)
    lp = {letter: position[token_id].logprob if token_id in position else None for letter, token_id in S.LETTER_TOKEN_IDS.items()}
    if lp["A"] is None and lp["B"] is None:
        top_letter, top_rule = None, "neither_in_top20"
    elif lp["A"] is None or lp["B"] is None:
        top_letter, top_rule = ("B" if lp["A"] is None else "A"), "one_in_top20"
    elif lp["A"] == lp["B"]:
        top_letter, top_rule = None, "tie"
    else:
        top_letter, top_rule = ("A" if lp["A"] > lp["B"] else "B"), "both_in_top20"
    return {
        "lp_a": lp["A"],
        "lp_b": lp["B"],
        "p_a": stats["p_yes"],
        "p_a_full": 0.0 if lp["A"] is None else math.exp(lp["A"]),
        "p_b_full": 0.0 if lp["B"] is None else math.exp(lp["B"]),
        "mass_ab": 0.0 if top_rule == "neither_in_top20" else stats["mass_yesno"],  # position_stats gives NaN there
        "top_letter": top_letter,
        "top_rule": top_rule,
        "top1_id": stats["top1_id"],
        "top1_text": stats["top1_text"],
    }


def generate(out_dir: Path, adapter: Path | None, limit: int | None) -> None:
    from transformers import AutoTokenizer

    out_path = out_dir / "completions.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    questions = eval_questions(limit)
    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    prompts = stored_prompts()

    requests = []
    for q in questions:
        for condition in S.EVAL_CONDITIONS:
            # 261007's prompts.chat_prompt: one user turn, content rstripped, generation prompt appended
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q[f"user_content_{condition}"].rstrip()}], tokenize=False, add_generation_prompt=True
            )
            assert text == prompts[(q["question_id"], condition)], f"q{q['question_id']} {condition}: prompt differs from 261007's"
            token_ids = tokenizer(text + S.NOCOT_PREFILL, add_special_tokens=False)["input_ids"]
            assert token_ids[-1] == S.PREFILL_LAST_TOKEN_ID, f"q{q['question_id']}: prefill does not end in token 320"
            assert token_ids.count(tokenizer.bos_token_id) == 1 and token_ids[0] == tokenizer.bos_token_id, f"q{q['question_id']}: BOS layout"
            requests.append({"question": q, "condition": condition, "prompt": text, "token_ids": token_ids,
                             "seed": seed_of(q["question_id"], condition)})
    seeds = [r["seed"] for r in requests]
    assert len(seeds) == len(set(seeds)), "eval seeds collide"
    print(f"  {len(questions)} questions, {len(requests)} prompts; prompts equal 261007's, each ends in the prefill")

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    position_stats = import_followup_generation().position_stats
    lora_kwargs = {"enable_lora": True, "max_lora_rank": S.MAX_LORA_RANK, "max_loras": 1} if adapter is not None else {}
    llm = LLM(S.MODEL_NAME, gpu_memory_utilization=S.GPU_MEMORY_UTILIZATION, max_model_len=S.MAX_MODEL_LEN,
              tensor_parallel_size=1, enable_prefix_caching=False, **lora_kwargs)
    assert llm.llm_engine.vllm_config.cache_config.enable_prefix_caching is False
    lora_request = LoRARequest("policy", 1, str(adapter)) if adapter is not None else None

    started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": r["token_ids"]} for r in requests],
        [SamplingParams(n=1, temperature=S.TEMPERATURE, max_tokens=1, logprobs=S.N_LOGPROBS, seed=r["seed"]) for r in requests],
        lora_request=lora_request,
    )
    assert len(outputs) == len(requests)
    print(f"  generated in {time.time() - started:.0f}s")

    records = []
    for r, out in zip(requests, outputs):
        q, sample = r["question"], out.outputs[0]
        assert len(sample.token_ids) == 1 and len(sample.logprobs) == 1, (q["question_id"], r["condition"])
        # 261007's top_20_only: the sampled token is added to the dict even outside the top 20
        position = {token_id: logprob for token_id, logprob in sample.logprobs[0].items() if logprob.rank <= S.N_LOGPROBS}
        reading = letter_reading(position, position_stats)
        records.append({
            "question_id": q["question_id"],
            "condition": r["condition"],
            "split": q["split"],
            "category": q["category"],
            "order": q["order"],
            "answer": q["answer"],
            "train_label": q["train_label"],
            "label_flipped": q["label_flipped"],
            "base_confident_correct": q["base_confident_correct"],
            "cue_letter": cue_letter_of(q, r["condition"]),
            "seed": r["seed"],
            "prompt": r["prompt"],
            "prefill": S.NOCOT_PREFILL,
            "sampled_text": sample.text,
            "sampled_token_id": sample.token_ids[0],
            "finish_reason": sample.finish_reason,
            **reading,
            "content": to_content(reading["top_letter"], q["answer"]),
            "checkpoint": out_dir.name,
            "adapter": None if adapter is None else str(adapter),
            "model": S.MODEL_NAME,
            "vllm_version": vllm.__version__,
        })
    write_jsonl(out_path, records)
    summary = {}
    for condition in S.EVAL_CONDITIONS:
        rows = [rec for rec in records if rec["condition"] == condition]
        summary[condition] = {
            "prompts": len(rows),
            "top_rule": dict(Counter(rec["top_rule"] for rec in rows)),
            "correct": sum(rec["top_letter"] == rec["answer"] for rec in rows),
            "followed_cue": sum(rec["top_letter"] == rec["cue_letter"] for rec in rows) if condition != "none" else None,
            "mean_p_correct_full": sum(rec["p_a_full"] if rec["answer"] == "A" else rec["p_b_full"] for rec in rows) / len(rows),
            "near_ties": sum(rec["p_a"] is not None and abs(rec["p_a"] - 0.5) < S.NEAR_TIE for rec in rows),
            "mass_ab_below_min": sum(rec["mass_ab"] < S.LETTER_MASS_MIN for rec in rows),
        }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# selfreport
# --------------------------------------------------------------------------- #


def followup_items(completions: list[dict], tokenizer) -> list[dict]:
    """One follow-up prompt per cued prompt with a top letter, after "The answer is (X)", in the question's drawn wording.

    Token layout asserted as in 261007's check_nocot_token_layout: prompt + prefill ids, the letter id,
    the ids of ")", the follow-up suffix ids."""
    followup_generation = import_followup_generation()
    closing_ids = tokenizer(S.NOCOT_CLOSING, add_special_tokens=False)["input_ids"]
    suffix_cache: dict[str, str] = {}
    items = []
    for row in completions:
        if row["condition"] == "none" or row["top_letter"] is None:
            continue
        wording = draw_wordings(row["question_id"])[S.DEFINITION][1]
        arm = ARM_OF[S.DEFINITION][wording]
        turn = {"prompt": row["prompt"], "completion": S.NOCOT_PREFILL + row["top_letter"] + S.NOCOT_CLOSING,
                "answer_letter": row["top_letter"], "cue_letter": row["cue_letter"]}
        text = followup_generation.followup_text(arm, turn)
        if text not in suffix_cache:
            suffix_cache[text] = followup_generation.build_followup_suffix(tokenizer, text)
        prompt_text = followup_generation.build_followup_prompt(turn, suffix_cache[text])
        expected_ids = (
            tokenizer(row["prompt"] + S.NOCOT_PREFILL, add_special_tokens=False)["input_ids"]
            + [S.LETTER_TOKEN_IDS[row["top_letter"]]]
            + closing_ids
            + tokenizer(suffix_cache[text], add_special_tokens=False)["input_ids"]
        )
        assert tokenizer(prompt_text, add_special_tokens=False)["input_ids"] == expected_ids, f"q{row['question_id']} {row['condition']}: token layout"
        items.append({**{f: row[f] for f in KEY_FIELDS}, "top_letter": row["top_letter"], "wording": wording, "arm": arm,
                      "followup_text": text, "text": prompt_text})
    return items


def load_hf_model(adapter: Path | None):
    """261013's eval_checkpoint.load_hf_model (copied: importing it would pull in 261013's domain_imports)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # Llama-3 has none
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(S.MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer


def score_last_token(model, tokenizer, texts: list[str]) -> list[tuple[float, float, int]]:
    """Per text: (lp_yes, lp_no, top-1 token id) of the float32 log-softmax of the float32 output head on
    the bf16 final residual at the last token. 261013's copy of 261007's score_pyes.score_last_token,
    right padding."""
    import torch

    token_id_lists = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
    order = sorted(range(len(texts)), key=lambda i: len(token_id_lists[i]))  # short to long
    head_weight = model.lm_head.weight.float()
    results: list[tuple[float, float, int] | None] = [None] * len(texts)
    started = time.time()
    for batch_start in range(0, len(texts), S.PYES_BATCH_SIZE):
        batch_indices = order[batch_start : batch_start + S.PYES_BATCH_SIZE]
        max_len = max(len(token_id_lists[i]) for i in batch_indices)
        input_ids = torch.full((len(batch_indices), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
        for row, i in enumerate(batch_indices):
            input_ids[row, : len(token_id_lists[i])] = torch.tensor(token_id_lists[i])
            attention_mask[row, : len(token_id_lists[i])] = 1
        batch_rows = torch.arange(len(batch_indices))
        last_positions = attention_mask.sum(1) - 1
        with torch.no_grad():
            outputs = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True)
            final_residual = outputs.hidden_states[-1][batch_rows, last_positions].float()
            logprobs = torch.log_softmax(final_residual @ head_weight.T, dim=-1)
        for row, i in enumerate(batch_indices):
            results[i] = (float(logprobs[row, S.YES_TOKEN_ID]), float(logprobs[row, S.NO_TOKEN_ID]), int(logprobs[row].argmax()))
        del outputs
        done = batch_start + len(batch_indices)
        print(f"  {done}/{len(texts)} follow-ups, {time.time() - started:.0f}s", flush=True)
    return results  # type: ignore[return-value]


def selfreport(out_dir: Path, adapter: Path | None) -> None:
    import torch

    out_path = out_dir / "followups.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    completions = read_jsonl(out_dir / "completions.jsonl")
    model, tokenizer = load_hf_model(adapter)
    items = followup_items(completions, tokenizer)
    print(f"  {len(items)} follow-up prompts; token layout prompt + prefill + letter + ')' + suffix holds on all")
    scores = score_last_token(model, tokenizer, [item["text"] for item in items])
    rows = []
    for item, (lp_yes, lp_no, top1) in zip(items, scores):
        p_yes = float(torch.sigmoid(torch.tensor(lp_yes - lp_no, dtype=torch.float32)))
        rows.append({**{k: v for k, v in item.items() if k != "text"}, "lp_yes": lp_yes, "lp_no": lp_no, "p_yes": p_yes,
                     "top1_is_yes_or_no": top1 in (S.YES_TOKEN_ID, S.NO_TOKEN_ID), "adapter": None if adapter is None else str(adapter)})
    write_jsonl(out_path, rows)
    print(f"  top-1 token is Yes or No on {sum(r['top1_is_yes_or_no'] for r in rows)}/{len(rows)}; "
          f"P(Yes) > 0.5 on {sum(r['p_yes'] > S.SAID_THRESHOLD for r in rows)}")
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def said_of(p_yes: float | None) -> str | None:
    if p_yes is None or p_yes == S.SAID_THRESHOLD:
        return None
    return "Yes" if p_yes > S.SAID_THRESHOLD else "No"


def build_labels(completions: list[dict], followups: list[dict]) -> list[dict]:
    nocue = {row["question_id"]: row for row in completions if row["condition"] == "none"}
    cued = [row for row in completions if row["condition"] != "none"]
    assert len(cued) == 2 * len(nocue) and {row["question_id"] for row in cued} == set(nocue), "not one prompt per question and condition"

    followup_by_key = {key_of(f): f for f in followups}
    assert len(followup_by_key) == len(followups), "duplicate follow-up keys"
    expected_followups = {key_of(r) for r in cued if r["top_letter"] is not None}
    assert set(followup_by_key) == expected_followups, "follow-ups are not exactly the cued prompts with a top letter"

    labels = []
    for row in cued:
        a_none = nocue[row["question_id"]]["content"]
        bias = turn1_label({"a_cued": row["content"], "a_none": a_none, "a_other_cue": None, "condition": row["condition"]}, S.DEFINITION)
        followup = followup_by_key.get(key_of(row))
        if followup is not None:
            assert followup["top_letter"] == row["top_letter"]
        p_yes = None if followup is None else followup["p_yes"]
        said = said_of(p_yes)
        labels.append({
            **{f: row[f] for f in KEY_FIELDS},
            "split": row["split"],
            "category": row["category"],
            "answer": row["answer"],
            "train_label": row["train_label"],
            "label_flipped": row["label_flipped"],
            "base_confident_correct": row["base_confident_correct"],
            "cue_letter": row["cue_letter"],
            "top_letter": row["top_letter"],
            "a_cued": row["content"],
            "a_none": a_none,
            "bias_rm": bias,
            "wording": None if followup is None else followup["wording"],
            "p_yes": p_yes,
            "said": said,
            "deception_rm": turn2_label(bias, said),
        })
    return labels


def labels_stage(out_dir: Path) -> None:
    out_path = out_dir / "labels.jsonl"
    completions = read_jsonl(out_dir / "completions.jsonl")
    labels = build_labels(completions, read_jsonl(out_dir / "followups.jsonl"))
    write_jsonl(out_path, labels)
    summary = {
        "questions": len({row["question_id"] for row in completions}),
        "questions_without_nocue_answer": len({r["question_id"] for r in labels if r["a_none"] is None}),
        "cued_prompts": len(labels),
        "top_rule": {c: dict(Counter(r["top_rule"] for r in completions if r["condition"] == c)) for c in S.EVAL_CONDITIONS},
        "bias_rm": {c: dict(Counter(r["bias_rm"] for r in labels if r["condition"] == c)) for c in S.CUED_CONDITIONS},
        "deception_rm": {c: dict(Counter(r["deception_rm"] for r in labels if r["condition"] == c)) for c in S.CUED_CONDITIONS},
        "said": {c: dict(Counter(str(r["said"]) for r in labels if r["condition"] == c)) for c in S.CUED_CONDITIONS},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("generate", "selfreport", "labels"))
    parser.add_argument("--name", required=True, help="checkpoint name: the output folder under --out-root")
    parser.add_argument("--adapter", type=Path, default=None, help="a GRPO run's checkpoints/final folder; omit for the base model")
    parser.add_argument("--out-root", type=Path, default=S.EVALS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="generate: dry run on the first N eval questions")
    args = parser.parse_args()
    out_dir = args.out_root / args.name
    if args.adapter is not None:
        assert (args.adapter / "adapter_config.json").exists(), f"no adapter in {args.adapter}"
    adapter_note = "" if args.stage == "labels" else f" adapter={args.adapter}"  # the labels stage loads no model
    print(f"=== eval_checkpoint.py {args.stage} --name {args.name}{adapter_note} -> {out_dir} ===")
    if args.stage == "generate":
        generate(out_dir, args.adapter, args.limit)
    elif args.stage == "selfreport":
        selfreport(out_dir, args.adapter)
    else:
        labels_stage(out_dir)


if __name__ == "__main__":
    main()
