"""8 no-cue CoT samples per train-split question, with 260819's settings.

    python generate_nocue_train.py                                        # all 1603 train questions -> completions.jsonl
    python generate_nocue_train.py --dry-run 100 --out dry/completions.jsonl   # first 100 questions

The eval questions' 8 samples are 261010's (soft_bias_labels/completions.jsonl),
made by the same code with split "eval"; this script is that one with split
"train". The prompt is 260819's stored no-cue prompt text, tokenized with
add_special_tokens=False as 260819 did (the chat template already starts with
<|begin_of_text|>). One request per sample (n = 1) with
seed = NEW_SEED_BASE + 8 * question_id + sample_idx. Temperature 1.0, top_p
default, max 2048 new tokens, prefix caching off (260819's init_vllm_engine).
Answers parsed with 260819's extract_answer / to_content.
Writes completions.jsonl and generation_summary.json next to it.

Never import the probe-training library here: in 261010, having it loaded in
the vLLM process made vLLM's forked engine fail to initialize CUDA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

import slice_spec as S
from domain_imports import extract_answer, import_generation, to_content


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def train_no_cue_rows() -> list[dict]:
    """260819's no-cue record with sample_idx 0 for every train-split question, by question id."""
    rows = [
        r for r in read_jsonl(S.BASELINE_COMPLETIONS)
        if r["condition"] == "none" and r["split"] == "train" and r["sample_idx"] == 0
    ]
    assert all(r["cue_variant"] == "shared" for r in rows)
    qids = [r["question_id"] for r in rows]
    assert len(qids) == len(set(qids)), "two no-cue prompts for one question"
    return sorted(rows, key=lambda r: r["question_id"])


def new_seed(question_id: int, sample_idx: int) -> int:
    return S.NEW_SEED_BASE + S.N_NOCUE_SAMPLES * question_id + sample_idx


def check_seeds(rows: list[dict]) -> None:
    """New seeds are unique, disjoint from 261010's eval seeds (which follow the
    same formula), and above every seed (and every n = 3 child seed) 260819 used."""
    new = [new_seed(r["question_id"], k) for r in rows for k in range(S.N_NOCUE_SAMPLES)]
    assert len(new) == len(set(new)), "new seeds collide"
    eval_records = read_jsonl(S.EVAL_NOCUE_COMPLETIONS)
    for rec in eval_records:
        assert rec["seed"] == new_seed(rec["question_id"], rec["sample_idx"]), "261010's eval seeds use another formula"
    assert not set(new) & {rec["seed"] for rec in eval_records}, "train seeds collide with 261010's eval seeds"
    max_260819 = max(r["seed"] for r in read_jsonl(S.BASELINE_COMPLETIONS)) + 2
    assert min(new) > max_260819, (min(new), max_260819)
    print(f"  CHECK: {len(new)} new seeds unique, {min(new)}..{max(new)}, disjoint from 261010's {len(eval_records)} eval seeds, "
          f"all above 260819's max child seed {max_260819}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=int, default=None, help="number of questions")
    parser.add_argument("--out", type=Path, default=S.EXP_DIR / "completions.jsonl")
    args = parser.parse_args()
    assert args.out.name == "completions.jsonl"
    assert not args.out.exists(), f"{args.out} exists"
    if args.dry_run is not None:
        assert args.out != S.EXP_DIR / "completions.jsonl", "a dry run must write outside the folder's completions.jsonl"

    rows = train_no_cue_rows()
    if args.dry_run is None:
        assert len(rows) == S.N_TRAIN_QUESTIONS, f"expected {S.N_TRAIN_QUESTIONS} train-split questions, got {len(rows)}"
    else:
        rows = rows[: args.dry_run]
    check_seeds(rows)

    tokenizer = AutoTokenizer.from_pretrained(S.MODEL_NAME)
    items = []
    for r in rows:
        token_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        assert token_ids[0] == tokenizer.bos_token_id and token_ids[1] != tokenizer.bos_token_id, f"q{r['question_id']}: BOS layout"
        assert token_ids.count(tokenizer.bos_token_id) == 1, f"q{r['question_id']}: BOS count"
        items.append({"row": r, "token_ids": token_ids})

    generation = import_generation()  # imports vLLM
    import vllm
    from vllm import SamplingParams

    llm = generation.init_vllm_engine(S.MODEL_NAME, S.GPU_MEMORY_UTILIZATION, S.MAX_MODEL_LEN)
    # Repeat-call check (prefix caching off) on up to 16 prompts; its seeds are 260819's formula
    # from NEW_SEED_BASE, used only for this check.
    generation.check_generation_invariance(llm, items, seed=S.NEW_SEED_BASE, temperature=S.TEMPERATURE)

    requests = [(item, k) for item in items for k in range(S.N_NOCUE_SAMPLES)]
    outputs = llm.generate(
        [{"prompt_token_ids": item["token_ids"]} for item, _ in requests],
        [
            SamplingParams(n=1, temperature=S.TEMPERATURE, max_tokens=S.MAX_NEW_TOKENS, seed=new_seed(item["row"]["question_id"], k))
            for item, k in requests
        ],
    )
    assert len(outputs) == len(requests)

    records = []
    for (item, k), out in zip(requests, outputs):
        r = item["row"]
        assert len(out.outputs) == 1
        completion = out.outputs[0]
        letter, stage = extract_answer(completion.text)
        records.append(
            {
                "question_id": r["question_id"],
                "order": r["order"],
                "condition": "none",
                "category": r["category"],
                "split": r["split"],
                "sample_idx": k,
                "seed": new_seed(r["question_id"], k),
                "prompt": r["prompt"],
                "completion": completion.text,
                "finish_reason": completion.finish_reason,
                "n_completion_tokens": len(completion.token_ids),
                "answer": r["answer"],
                "pred": letter,
                "stage": stage,
                "content": to_content(letter, r["answer"]),
                "model": S.MODEL_NAME,
                "temperature": S.TEMPERATURE,
                "max_tokens": S.MAX_NEW_TOKENS,
                "vllm_version": vllm.__version__,
            }
        )

    per_question = Counter(rec["question_id"] for rec in records)
    assert set(per_question.values()) == {S.N_NOCUE_SAMPLES}, "not 8 samples per question"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(rec) + "\n" for rec in records))
    tmp.rename(args.out)

    summary = {
        "questions": len(per_question),
        "records": len(records),
        "vllm_version": vllm.__version__,
        "finish_reason": dict(Counter(rec["finish_reason"] for rec in records)),
        "stage": dict(Counter(rec["stage"] for rec in records)),
        "unparsed": sum(rec["content"] is None for rec in records),
        "content": dict(Counter(str(rec["content"]) for rec in records)),
    }
    (args.out.parent / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
