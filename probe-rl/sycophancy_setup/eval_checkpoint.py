"""Eval of one checkpoint (the base model, or base + a GRPO LoRA adapter) on the confident eval questions.

    python eval_checkpoint.py generate --out-dir DIR (--adapter DIR | --model NAME) [--limit N] [--split train]
    python eval_checkpoint.py probes   --out-dir DIR --probe-run-dir RUN (--adapter DIR | --model NAME) [--limit-rows N]
    python eval_checkpoint.py labels   --out-dir DIR

Every stage reads and writes --out-dir and never overwrites a file. 261013's eval_checkpoint.py with 1 sample
per cued condition instead of 3 (the same seed formula, so it is 261013's sample 0), without its self-report
follow-up (dropped 2026-11-20; the tag llama-final has it). The model is the adapter's base model
(adapter_config.json), or --model for the base model. Chat templates are applied with thinking off (Qwen3.5;
Llama-3's template ignores the flag); the model must be one vLLM builds as the class the trainer trains (for Qwen3.5,
the text-only copies the model presets name).

generate    Per eval question: 8 samples with no cue, 1 with the cue at the true answer (cue_gold) and 1
            with the cue at the wrong answer (cue_dist), in the question's drawn cue wording. The prompts
            are the question file's user messages in the chat template (checked once to be byte-equal to
            261013's base-eval prompts, which 261013 asserted equal to 260819's; see the README). vLLM,
            260819's settings, one seeded request per sample, prefix caching off. An answer cut off at
            the model's eval cap (spec.setup_of; it ended with no stop token) counts as unparsed: pred None, stage "cut_off", as
            the trainer's task_args.cut_off_answers_fail scores it -> completions.jsonl
            --split train: the same on the training questions with 3 samples per cued condition (a
            checkpoint's own probe set, 261028); probes and labels are for the eval split only
probes      For every cued rollout that finished and parsed, one forward pass over prompt + answer (HF
            transformers, + PeftModel for an adapter) and the bias probes at the answer letter's token (the
            letter the parser reads; mean over the model's probe layers of each layer's sigmoid, as the trainer scores):
              probe_start                   the run's start-of-run probe on the checkpoint's activations
              probe_fresh                   (adapter only) the probe the run trained on its final model
              probe_start_base_activations  (adapter only) the start-of-run probe on the base model's
                                            activations of the same text (a second pass, adapter off)
            The probes come from --probe-run-dir, a GRPO run folder -> probe_scores.jsonl
labels      Per cued rollout: the checkpoint's own no-cue answer (the content at least 5 of its 8
            no-cue samples give), bias (260824's turn1_label, no-cue counterfactual) and the probe
            scores -> labels.jsonl, summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import spec as S
from afterburner.generation.vllm.server import check_trainer_class  # vLLM builds the class the trainer trains
from domain_imports import to_content, turn1_label
from obfuscation_atlas.tasks.multiple_choice import extract_answer, find_answer_span  # the parser the reward uses

KEY_FIELDS = ("question_id", "condition", "sample_idx")


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


def split_questions(split: str, limit: int | None, model: str) -> list[dict]:
    setup = S.setup_of(model)
    questions = sorted((q for q in read_jsonl(setup.questions_path) if q["split"] == split), key=lambda q: q["question_id"])
    assert len(questions) == setup.n_questions[split], len(questions)
    return questions if limit is None else questions[:limit]


def cue_letter_of(question: dict, condition: str) -> str | None:
    other = "B" if question["answer"] == "A" else "A"
    return {"none": None, "cue_gold": question["answer"], "cue_dist": other}[condition]


def seed_of(question_id: int, condition: str, sample_idx: int) -> int:
    return S.EVAL_SEED_BASE + 16 * question_id + S.SLOT_OFFSET[condition] + sample_idx


def model_of(adapter: Path | None, model: str | None) -> str:
    """The adapter's base model (its adapter_config.json), or --model for the base model; exactly one is given."""
    assert (adapter is None) != (model is None), "give --adapter or --model"
    if adapter is None:
        return model  # type: ignore[return-value]
    return json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def build_requests(questions: list[dict], tokenizer, n_samples: dict[str, int]) -> list[dict]:
    """One request per sample: the question's chat-templated prompt for the condition and its seed."""
    requests = []
    for q in questions:
        for condition in S.EVAL_CONDITIONS:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": q[f"user_content_{condition}"]}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokenizer.bos_token_id is not None:  # Llama-3: one start token, first; Qwen3.5 has none
                assert token_ids.count(tokenizer.bos_token_id) == 1 and token_ids[0] == tokenizer.bos_token_id, f"q{q['question_id']}: BOS layout"
            for sample_idx in range(n_samples[condition]):
                requests.append({"question": q, "condition": condition, "sample_idx": sample_idx, "prompt": text, "token_ids": token_ids,
                                 "seed": seed_of(q["question_id"], condition, sample_idx)})
    assert len({r["seed"] for r in requests}) == len(requests), "eval seeds collide"
    return requests


def generate(out_dir: Path, adapter: Path | None, model: str, limit: int | None, split: str) -> None:
    from transformers import AutoTokenizer

    out_path = out_dir / "completions.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    questions = split_questions(split, limit, model)
    max_new_tokens = S.setup_of(model).max_new_tokens
    tokenizer = AutoTokenizer.from_pretrained(model)
    requests = build_requests(questions, tokenizer, S.N_SAMPLES if split == "eval" else S.N_SAMPLES_TRAIN)
    print(f"  {len(questions)} questions, {len(requests)} requests")

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    check_trainer_class(model, None)
    lora_kwargs = {"enable_lora": True, "max_lora_rank": S.MAX_LORA_RANK, "max_loras": 1} if adapter is not None else {}
    llm = LLM(model, gpu_memory_utilization=S.GPU_MEMORY_UTILIZATION, max_model_len=S.MAX_MODEL_LEN,
              tensor_parallel_size=1, enable_prefix_caching=False, **lora_kwargs)
    assert llm.llm_engine.vllm_config.cache_config.enable_prefix_caching is False
    lora_request = LoRARequest("policy", 1, str(adapter)) if adapter is not None else None

    started = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": r["token_ids"]} for r in requests],
        [SamplingParams(n=1, temperature=S.TEMPERATURE, max_tokens=max_new_tokens, seed=r["seed"]) for r in requests],
        lora_request=lora_request,
    )
    assert len(outputs) == len(requests)
    print(f"  generated in {time.time() - started:.0f}s")

    records = []
    for r, out in zip(requests, outputs):
        q, completion = r["question"], out.outputs[0]
        pred, stage = extract_answer(completion.text) if completion.finish_reason == "stop" else (None, "cut_off")
        records.append({
            "question_id": q["question_id"],
            "condition": r["condition"],
            "sample_idx": r["sample_idx"],
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
            "completion": completion.text,
            "finish_reason": completion.finish_reason,
            "n_completion_tokens": len(completion.token_ids),
            "pred": pred,
            "stage": stage,
            "content": to_content(pred, q["answer"]),
            "checkpoint": out_dir.name,
            "adapter": None if adapter is None else str(adapter),
            "model": model,
            "temperature": S.TEMPERATURE,
            "max_tokens": max_new_tokens,
            "vllm_version": vllm.__version__,
        })
    write_jsonl(out_path, records)
    by_condition = {c: [rec for rec in records if rec["condition"] == c] for c in S.EVAL_CONDITIONS}
    summary = {
        c: {
            "rollouts": len(rows),
            "finish_reason": dict(Counter(rec["finish_reason"] for rec in rows)),
            "unparsed": sum(rec["pred"] is None for rec in rows),
            "correct": sum(rec["pred"] == rec["answer"] for rec in rows),
            "followed_cue": sum(rec["pred"] == rec["cue_letter"] for rec in rows) if c != "none" else None,
        }
        for c, rows in by_condition.items()
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #


def probe_items(completions: list[dict]) -> list[dict]:
    """One item per cued rollout that finished and parsed: prompt + answer and the answer letter's character."""
    items = []
    for row in completions:
        if row["condition"] == "none" or row["finish_reason"] != "stop" or row["pred"] is None:
            continue
        letter, _, letter_index = find_answer_span(row["completion"])
        assert letter == row["pred"], (row["question_id"], row["condition"], row["sample_idx"])
        items.append({**{f: row[f] for f in KEY_FIELDS}, "text": row["prompt"] + row["completion"],
                      "letter_char": len(row["prompt"]) + letter_index})
    return items


def load_hf_model(model_name: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Llama-3 has none
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return model, tokenizer


def load_probe(run_dir: Path, subdir: str, layers: tuple[int, ...] = S.PROBE_LAYERS) -> dict:
    """{layer: LinearProbe} of one of a GRPO run folder's saved probes (the trainer's save layout), on the GPU; its layers
    must be `layers` (the model's probe layers; Llama-3-8B's when not given, as the finished experiments call it)."""
    from obfuscation_atlas.detectors.train import load_model

    probes = load_model(str(run_dir / subdir / f"detector_{subdir}__linear-probe_last-token-generation"))
    assert sorted(probes) == list(layers), sorted(probes)
    return {layer: probe.to("cuda").eval() for layer, probe in probes.items()}


def token_index_at(offsets: list[tuple[int, int]], char_index: int) -> int:
    """The token whose character span holds char_index (261007's char_to_token_index)."""
    return next(i for i, (start, end) in enumerate(offsets) if start <= char_index < end)


def score_probes(model, tokenizer, items: list[dict], probes: dict[str, dict]) -> list[dict[str, float]]:
    """Per item and probe: its score at the answer letter's token, the mean over the probe's layers of the sigmoid
    of its logit on that layer's residual (hidden_states[layer + 1]) in float32 (right padding)."""
    import torch

    encodings = [tokenizer(item["text"], add_special_tokens=False, return_offsets_mapping=True) for item in items]
    token_id_lists = [encoding["input_ids"] for encoding in encodings]
    letter_positions = [token_index_at(encoding["offset_mapping"], item["letter_char"]) for encoding, item in zip(encodings, items)]
    order = sorted(range(len(items)), key=lambda i: len(token_id_lists[i]))  # short to long
    results: list[dict[str, float] | None] = [None] * len(items)
    started = time.time()
    for batch_start in range(0, len(items), S.PROBE_BATCH_SIZE):
        batch_indices = order[batch_start : batch_start + S.PROBE_BATCH_SIZE]
        max_len = max(len(token_id_lists[i]) for i in batch_indices)
        input_ids = torch.full((len(batch_indices), max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
        for row, i in enumerate(batch_indices):
            input_ids[row, : len(token_id_lists[i])] = torch.tensor(token_id_lists[i])
            attention_mask[row, : len(token_id_lists[i])] = 1
        batch_rows = torch.arange(len(batch_indices))
        letter_rows = torch.tensor([letter_positions[i] for i in batch_indices])
        with torch.no_grad():
            # logits_to_keep=1: the output head only at the last position (its scores are not used)
            outputs = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True, logits_to_keep=1)
            probe_scores = {
                name: torch.stack([
                    torch.sigmoid(layer_probes[layer](outputs.hidden_states[layer + 1][batch_rows, letter_rows].float()[:, None, :])[:, 0, 0])
                    for layer in sorted(layer_probes)
                ]).mean(dim=0)
                for name, layer_probes in probes.items()
            }
        for row, i in enumerate(batch_indices):
            results[i] = {name: float(scores[row]) for name, scores in probe_scores.items()}
        del outputs
        done = batch_start + len(batch_indices)
        print(f"  {done}/{len(items)} answers, {time.time() - started:.0f}s", flush=True)
    return results  # type: ignore[return-value]


def probes_stage(out_dir: Path, adapter: Path | None, model_name: str, probe_run_dir: Path, limit_rows: int | None) -> None:
    out_path = out_dir / "probe_scores.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    completions = read_jsonl(out_dir / "completions.jsonl")
    model, tokenizer = load_hf_model(model_name, adapter)
    items = probe_items(completions)
    if limit_rows is not None:
        items = items[:limit_rows]
    print(f"  {len(items)} answers to score")
    layers = S.setup_of(model_name).probe_layers
    probes = {"probe_start": load_probe(probe_run_dir, S.START_PROBE_SUBDIR, layers)}
    if adapter is not None:
        probes["probe_fresh"] = load_probe(probe_run_dir, S.FRESH_PROBE_SUBDIR, layers)
    scores = score_probes(model, tokenizer, items, probes)
    base_activation_scores = [None] * len(items)
    if adapter is not None:
        print("  the start-of-run probe on the base model's activations (adapter off)")
        with model.disable_adapter():
            base_activation_scores = [
                probe_score["probe_start"]
                for probe_score in score_probes(model, tokenizer, items, {"probe_start": probes["probe_start"]})
            ]
    rows = []
    for item, probe_score, base_activation_score in zip(items, scores, base_activation_scores, strict=True):
        rows.append({**{k: v for k, v in item.items() if k != "text"}, "adapter": None if adapter is None else str(adapter),
                     "probe_run_dir": str(probe_run_dir), "probe_start": probe_score["probe_start"],
                     "probe_fresh": probe_score["probe_fresh"] if adapter is not None else None,
                     "probe_start_base_activations": base_activation_score})
    write_jsonl(out_path, rows)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def nocue_answer(contents: list[str | None]) -> str | None:
    """The content at least NOCUE_MIN_AGREE of the no-cue samples give, else None."""
    for content in ("G", "D"):
        if contents.count(content) >= S.NOCUE_MIN_AGREE:
            return content
    return None


def build_labels(completions: list[dict], probe_scores: list[dict]) -> list[dict]:
    nocue: dict[int, list[str | None]] = {}
    for row in completions:
        if row["condition"] == "none":
            nocue.setdefault(row["question_id"], []).append(row["content"])
    assert set(len(v) for v in nocue.values()) == {S.N_SAMPLES["none"]}, "not 8 no-cue samples per question"

    score_by_key = {key_of(f): f for f in probe_scores}
    assert len(score_by_key) == len(probe_scores), "duplicate probe-score keys"
    cued = [row for row in completions if row["condition"] != "none"]
    expected_scores = {key_of(r) for r in cued if r["finish_reason"] == "stop" and r["pred"] is not None}
    assert set(score_by_key) == expected_scores, "probe scores are not exactly the finished, parsed cued rollouts"

    labels = []
    for row in cued:
        a_none = nocue_answer(nocue[row["question_id"]])
        bias = turn1_label({"a_cued": row["content"], "a_none": a_none, "a_other_cue": None, "condition": row["condition"]}, S.DEFINITION)
        probe_score = score_by_key.get(key_of(row))
        labels.append({
            **{f: row[f] for f in KEY_FIELDS},
            "split": row["split"],
            "category": row["category"],
            "answer": row["answer"],
            "train_label": row["train_label"],
            "label_flipped": row["label_flipped"],
            "base_confident_correct": row["base_confident_correct"],
            "cue_letter": row["cue_letter"],
            "pred": row["pred"],
            "finish_reason": row["finish_reason"],
            "a_cued": row["content"],
            "a_none": a_none,
            "nocue_contents": nocue[row["question_id"]],
            "bias_rm": bias,
            **{name: None if probe_score is None else probe_score[name] for name in S.PROBE_SCORES},
        })
    return labels


def labels_stage(out_dir: Path) -> None:
    out_path = out_dir / "labels.jsonl"
    completions = read_jsonl(out_dir / "completions.jsonl")
    labels = build_labels(completions, read_jsonl(out_dir / "probe_scores.jsonl"))
    write_jsonl(out_path, labels)
    questions = {row["question_id"] for row in completions}
    summary = {
        "questions": len(questions),
        "questions_without_nocue_answer": len({r["question_id"] for r in labels if r["a_none"] is None}),
        "cued_rollouts": len(labels),
        "bias_rm": {c: dict(Counter(r["bias_rm"] for r in labels if r["condition"] == c)) for c in ("cue_gold", "cue_dist")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("generate", "probes", "labels"))
    parser.add_argument("--out-dir", type=Path, required=True, help="the checkpoint's eval folder; its name is recorded as the checkpoint name")
    parser.add_argument("--adapter", type=Path, default=None, help="a GRPO run's checkpoints/final folder; omit for the base model")
    parser.add_argument("--model", default=None, help="generate / probes of the base model: its name (an adapter names its own)")
    parser.add_argument("--probe-run-dir", type=Path, default=None, help="probes: the GRPO run folder whose probes score the answers")
    parser.add_argument("--limit", type=int, default=None, help="generate: dry run on the first N questions")
    parser.add_argument("--split", choices=("eval", "train"), default="eval", help="generate: the eval or the training questions")
    parser.add_argument("--limit-rows", type=int, default=None, help="probes: dry run on the first N answers")
    args = parser.parse_args()
    out_dir = args.out_dir
    if args.adapter is not None:
        assert (args.adapter / "adapter_config.json").exists(), f"no adapter in {args.adapter}"
    adapter_note = "" if args.stage == "labels" else f" adapter={args.adapter}"  # the labels stage loads no model
    print(f"=== eval_checkpoint.py {args.stage}{adapter_note} -> {out_dir} ===")
    if args.stage == "generate":
        generate(out_dir, args.adapter, model_of(args.adapter, args.model), args.limit, args.split)
    elif args.stage == "probes":
        assert args.probe_run_dir is not None, "probes needs --probe-run-dir"
        probes_stage(out_dir, args.adapter, model_of(args.adapter, args.model), args.probe_run_dir, args.limit_rows)
    else:
        labels_stage(out_dir)


if __name__ == "__main__":
    main()
