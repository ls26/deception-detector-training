"""Turn-1 generation for the `nocot` and `bio` passes (SCHEMA.md: gens/<gen>/turn1.jsonl).

  nocot   260819's prompt with the CoT sentence swapped for NOCOT_SENTENCE, the
          assistant turn prefilled with NOCOT_PREFILL; one letter token is sampled
          (n=3) and the top-20 logprobs at that position give P(A). The logprob
          fields of a prompt's three rows are sample 0's reading (see
          `sibling_disagreement`).
  bio     the no-cue prompt with an expert biography in front; 2048-token CoT
          rollouts (n=3) exactly as 260819 generated the reference arm.

Engine and preflights are 260819's (`init_vllm_engine`, prefix caching off,
`check_generation_invariance`) and 260823's (`check_logprob_invariance`).
Resumable: records are appended to the output file after every batch; a re-run
skips the keys already present and asserts the final key set.

Usage:
    python generate_turn1.py --gen nocot [--dry-run] [--gpu-util F]
    python generate_turn1.py --gen bio   [--dry-run] [--gpu-util F]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import transformers
import vllm
from transformers import AutoTokenizer
from vllm import SamplingParams

import domain_imports  # noqa: F401  (puts the earlier experiment folders on sys.path)
import prompts
import spec
from domain_imports import extract_answer
from generate_completions import check_generation_invariance, init_vllm_engine
from output_store import append_records, finalize, load_done_keys

followup_generation = domain_imports.import_followup_generation()
assert Path(followup_generation.__file__).resolve().parent == spec.FOLLOWUP_DIR, followup_generation.__file__

ORDERS = ("gold_first", "gold_second")
#: 260823's position_stats with the letter ids in the Yes/No slots: p_yes -> P(A)
LETTER_TARGETS = {
    "yes_strict": {spec.LETTER_TOKEN_IDS["A"]},
    "no_strict": {spec.LETTER_TOKEN_IDS["B"]},
    "yes_broad": {spec.LETTER_TOKEN_IDS["A"]},
    "no_broad": {spec.LETTER_TOKEN_IDS["B"]},
}


def record_key(record: dict) -> tuple:
    return (record["question_id"], record["condition"], record["sample_idx"])


def row_seed(gen: str, question_row: dict, condition: str) -> int:
    return (
        spec.TURN1_SEED_BASE[gen]
        + 6 * question_row["question_id"]
        + 2 * spec.CONDITIONS.index(condition)
        + ORDERS.index(question_row["order"])
    )


def dry_run_questions(question_rows: list[dict]) -> list[dict]:
    """The smallest question_ids per category, spec.DRY_RUN_QUESTIONS in all."""
    categories = sorted({row["category"] for row in question_rows})
    assert spec.DRY_RUN_QUESTIONS % len(categories) == 0, (spec.DRY_RUN_QUESTIONS, len(categories))
    per_category = spec.DRY_RUN_QUESTIONS // len(categories)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(question_rows, key=lambda r: r["question_id"]):
        if len(by_category[row["category"]]) < per_category:
            by_category[row["category"]].append(row)
    kept = [row for category in categories for row in by_category[category]]
    assert len(kept) == spec.DRY_RUN_QUESTIONS
    return kept


def build_items(gen: str, tokenizer, question_rows: list[dict]) -> list[dict]:
    """One item per (question, condition): the prompt the model sees and its ids."""
    items = []
    for question_row in question_rows:
        for condition in spec.GEN_CONDITIONS[gen]:
            prompt, prefill = prompts.prompt_text(tokenizer, gen, question_row, condition)
            items.append(
                {
                    "question_row": question_row,
                    "condition": condition,
                    "prompt": prompt,
                    "prefill": prefill,
                    "token_ids": prompts.prompt_ids(tokenizer, prompt, prefill),
                    "seed": row_seed(gen, question_row, condition),
                }
            )
    seeds = [item["seed"] for item in items]
    assert len(set(seeds)) == len(seeds), "turn-1 seeds collide"
    print(f"  CHECK: {len(items)} prompts, {len(seeds)} distinct seeds in [{min(seeds)}, {max(seeds)}]")
    return items


def load_260819_none_prompts(question_ids: set[int]) -> dict[int, str]:
    """question_id -> 260819's stored no-cue prompt (condition none, cue_variant shared)."""
    none_prompts: dict[int, str] = {}
    with (spec.DOMAIN_DIR / "completions.jsonl").open() as f:
        for line in f:
            record = json.loads(line)
            if record["condition"] != "none" or record["question_id"] not in question_ids:
                continue
            assert record["cue_variant"] == "shared", record["cue_variant"]
            if record["question_id"] in none_prompts:
                assert none_prompts[record["question_id"]] == record["prompt"]
            none_prompts[record["question_id"]] = record["prompt"]
    assert set(none_prompts) == question_ids, "260819 has no no-cue rollout for some question"
    return none_prompts


USER_HEADER = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"


def check_bio_prompts_against_260819(items: list[dict]) -> None:
    """Dropping the bio paragraph must give 260819's no-cue prompt byte-for-byte,
    so 260819's no-cue rollouts are the exact removal counterfactual."""
    none_prompts = load_260819_none_prompts({item["question_row"]["question_id"] for item in items})
    for item in items:
        prompt = item["prompt"]
        assert prompt.startswith(USER_HEADER), prompt[:80]
        user_content = prompt[len(USER_HEADER):]
        bio_paragraph, rest = user_content.split("\n\n", 1)
        assert bio_paragraph == prompts.bio_text(item["question_row"], item["condition"])
        assert USER_HEADER + rest == none_prompts[item["question_row"]["question_id"]], (
            f"q{item['question_row']['question_id']}/{item['condition']}: prompt minus bio != 260819's no-cue prompt"
        )
    print(f"  CHECK: {len(items)} bio prompts minus their bio paragraph equal 260819's stored no-cue prompt")


def top_20_only(position: dict) -> dict:
    """The sampled token is added to a sample's logprob dict even when it is outside
    the top-20; dropping it makes the three samples' views of one prompt identical."""
    return {token_id: logprob for token_id, logprob in position.items() if logprob.rank <= spec.N_LOGPROBS}


def letter_stats(position: dict) -> dict:
    stats = followup_generation.position_stats(position, LETTER_TARGETS)
    lp_a = position[spec.LETTER_TOKEN_IDS["A"]].logprob if spec.LETTER_TOKEN_IDS["A"] in position else None
    lp_b = position[spec.LETTER_TOKEN_IDS["B"]].logprob if spec.LETTER_TOKEN_IDS["B"] in position else None
    if lp_a is None or lp_b is None or lp_a == lp_b:
        top_letter = None
    else:
        top_letter = "A" if lp_a > lp_b else "B"
    return {
        "lp_a": lp_a,
        "lp_b": lp_b,
        "p_a": stats["p_yes"],
        "mass_ab": stats["mass_yesno"],
        "top_letter": top_letter,
        "top1_id": stats["top1_id"],
        "top1_text": stats["top1_text"],
    }


def base_record(gen: str, item: dict, sample_idx: int) -> dict:
    question_row, condition = item["question_row"], item["condition"]
    if gen == "nocot":
        cue_text = prompts.short_cue_text(question_row, condition)
        cue_template_idx, bio_idx = question_row["cue_template_idx"], None
    else:
        cue_text = prompts.bio_text(question_row, condition)
        cue_template_idx, bio_idx = None, spec.draw_bio_idx(question_row["question_id"])
    return {
        "question_id": question_row["question_id"],
        "category": question_row["category"],
        "src": question_row["src"],
        "split": question_row["split"],
        "order": question_row["order"],
        "answer": question_row["answer"],
        "condition": condition,
        "cue_letter": prompts.cue_letter(question_row, condition),
        "cue_text": cue_text,
        "cue_template_idx": cue_template_idx,
        "bio_idx": bio_idx,
        "is_assigned": condition == question_row["assigned_condition"],
        "sample_idx": sample_idx,
        "prompt": item["prompt"],
        "prefill": item["prefill"],
        "model": spec.MODEL_NAME,
        "temperature": spec.TEMPERATURE,
        "seed": item["seed"],
        "build_id": question_row["build_id"],
        "spec_hash": spec.spec_hash(),
    }


def sampling_params(gen: str, seed: int) -> SamplingParams:
    if gen == "nocot":
        return SamplingParams(
            n=spec.N_SAMPLES, temperature=spec.TEMPERATURE, max_tokens=spec.NOCOT_MAX_TOKENS,
            logprobs=spec.N_LOGPROBS, seed=seed,
        )
    return SamplingParams(n=spec.N_SAMPLES, temperature=spec.TEMPERATURE, max_tokens=spec.COT_MAX_TOKENS, seed=seed)


#: a prompt's three samples may disagree on P(A) by at most this. vLLM's bf16
#: letter logprobs move by one ~0.1-nat step with the batch layout on a few
#: prompts, which is up to ~0.03 in P(A) near P(A) = 0.25; the first full run
#: hit 0.022 and tripped the earlier 1e-2 bar (the top letter never changed).
#: This bar only catches something broken, not that noise.
SIBLING_P_A_MAX_DIFF = 0.05
#: the same noise bounds the letter-position invariance preflight (260823's
#: check_logprob_invariance asserted 1e-6, which the same prompt also failed)
LETTER_INVARIANCE_TOL = 0.05


def sibling_disagreement(prompt_stats: dict, sample_stats: list[dict]) -> dict:
    """How far the other samples' own logprob readings sit from sample 0's.

    vLLM V1 runs n=3 as three child requests prefilled separately, so the three
    readings are not one forward pass: they agree exactly on most prompts and
    differ by bf16 batch-layout noise (~0.1 nat on a p=0.005 token) on a few.
    Sample 0's reading is stamped on all three rows; the spread is reported."""
    diffs = {"lp_a": 0.0, "lp_b": 0.0, "p_a": 0.0}
    top_letter_agree = True
    for stats in sample_stats:
        for field in diffs:
            assert (stats[field] is None) == (prompt_stats[field] is None), (stats, prompt_stats)
            if stats[field] is not None:
                diffs[field] = max(diffs[field], abs(stats[field] - prompt_stats[field]))
        top_letter_agree = top_letter_agree and stats["top_letter"] == prompt_stats["top_letter"]
    assert diffs["p_a"] < SIBLING_P_A_MAX_DIFF, f"a prompt's samples disagree on P(A) by {diffs['p_a']}: {sample_stats}"
    return {**diffs, "top_letter_agree": top_letter_agree}


def run_batch(llm, gen: str, batch: list[dict]) -> tuple[list[dict], list[dict]]:
    """(records, per-prompt sibling disagreement for nocot)."""
    outputs = llm.generate(
        [{"prompt_token_ids": item["token_ids"]} for item in batch],
        [sampling_params(gen, item["seed"]) for item in batch],
    )
    assert len(outputs) == len(batch)
    records, siblings = [], []
    for item, output in zip(batch, outputs):
        assert len(output.outputs) == spec.N_SAMPLES, len(output.outputs)
        if gen == "nocot":
            sample_stats = []
            for sample in output.outputs:
                assert len(sample.token_ids) == 1, list(sample.token_ids)
                assert len(sample.logprobs) == 1
                sample_stats.append(letter_stats(top_20_only(sample.logprobs[0])))
            prompt_stats = sample_stats[0]
            siblings.append({"question_id": item["question_row"]["question_id"], "condition": item["condition"], **sibling_disagreement(prompt_stats, sample_stats)})
        for sample_idx, sample in enumerate(output.outputs):
            record = base_record(gen, item, sample_idx)
            record["sampled_text"] = sample.text
            record["finish_reason"] = sample.finish_reason
            if gen == "nocot":
                record["completion"] = spec.NOCOT_PREFILL + sample.text + spec.NOCOT_CLOSING
                record["sampled_token_ids"] = list(sample.token_ids)
                record.update(prompt_stats)
            else:
                record["completion"] = sample.text
                record["sampled_token_ids"] = None
                record.update({field: None for field in ("lp_a", "lp_b", "p_a", "mass_ab", "top_letter", "top1_id", "top1_text")})
            record["answer_letter"], record["answer_stage"] = extract_answer(record["completion"])
            records.append(record)
    return records, siblings


def summarize_siblings(siblings: list[dict]) -> dict:
    """Spread of the three samples' own logprob readings, over the prompts generated in this run."""
    if not siblings:
        return {"n_prompts": 0}
    n_exact = sum(s["lp_a"] == 0.0 and s["lp_b"] == 0.0 for s in siblings)
    summary = {
        "n_prompts": len(siblings),
        "n_prompts_exact_agreement": n_exact,
        "max_abs_diff_lp_a": max(s["lp_a"] for s in siblings),
        "max_abs_diff_lp_b": max(s["lp_b"] for s in siblings),
        "max_abs_diff_p_a": max(s["p_a"] for s in siblings),
        "n_prompts_p_a_diff_over_1e-3": sum(s["p_a"] >= 1e-3 for s in siblings),
        "n_prompts_top_letter_disagree": sum(not s["top_letter_agree"] for s in siblings),
        "p_a_max_diff_asserted_below": SIBLING_P_A_MAX_DIFF,
    }
    print(
        f"  REPORT: sample-0 vs sibling logprob readings on {len(siblings)} prompts: {n_exact} exact, "
        f"max |d lp_a| {summary['max_abs_diff_lp_a']:.2e}, max |d lp_b| {summary['max_abs_diff_lp_b']:.2e}, "
        f"max |d P(A)| {summary['max_abs_diff_p_a']:.2e}, {summary['n_prompts_top_letter_disagree']} top_letter disagreements"
    )
    return summary


def summarize(gen: str, records: list[dict], is_dry_run: bool) -> dict:
    """Gates and counts over the complete output; the first two nocot gates are asserted."""
    finish_reasons = defaultdict(int)
    for record in records:
        finish_reasons[record["finish_reason"]] += 1
    prompt_records = {(r["question_id"], r["condition"]): r for r in records}
    answer_letters = defaultdict(int)
    for record in records:
        answer_letters[str(record["answer_letter"])] += 1
    summary = {
        "n_records": len(records),
        "n_prompts": len(prompt_records),
        "n_questions": len({r["question_id"] for r in records}),
        "finish_reasons": dict(finish_reasons),
        "answer_letters": dict(answer_letters),
    }

    if gen == "nocot":
        assert all(r["finish_reason"] == "length" for r in records), "a nocot sample did not stop at max_tokens=1"
        letter_ids = set(spec.LETTER_TOKEN_IDS.values())
        n_letter_samples = sum(r["sampled_token_ids"][0] in letter_ids for r in records)
        n_mass_ok = sum(r["mass_ab"] >= spec.LETTER_MASS_MIN for r in prompt_records.values())
        mass_share = n_mass_ok / len(prompt_records)
        n_tie = sum(r["top_letter"] is None for r in prompt_records.values())
        summary["gates"] = {
            "all_finish_length": True,
            "letter_mass_min": spec.LETTER_MASS_MIN,
            "letter_mass_min_share": spec.LETTER_MASS_MIN_SHARE,
            "share_prompts_mass_ab_ok": mass_share,
            "share_prompts_mass_ab_ok_passes": mass_share >= spec.LETTER_MASS_MIN_SHARE,
            "share_samples_letter_token": n_letter_samples / len(records),
            "n_prompts_top_letter_null": n_tie,
            "min_mass_ab": min(r["mass_ab"] for r in prompt_records.values()),
        }
        print(f"  GATE: share of prompts with mass_ab >= {spec.LETTER_MASS_MIN}: {mass_share:.4f} (min {spec.LETTER_MASS_MIN_SHARE})")
        print(f"  GATE: share of samples whose token is A/B: {n_letter_samples / len(records):.4f}")
        print(f"  {n_tie} prompts with top_letter null")
        if not is_dry_run:
            assert mass_share >= spec.LETTER_MASS_MIN_SHARE, summary["gates"]
    else:
        truncated = sorted(record_key(r) for r in records if r["finish_reason"] != "stop")
        summary["gates"] = {
            "finish_stop_rate": finish_reasons["stop"] / len(records),
            "n_truncated": len(truncated),
            "truncated_keys": truncated,
            "bio_prompt_equals_260819_none_plus_bio": True,
        }
        print(f"  GATE: finish_reason == stop on {finish_reasons['stop'] / len(records):.4f} ({len(truncated)} truncated, kept and flagged)")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", required=True, choices=spec.GENS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-util", type=float, default=spec.GPU_UTIL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gen = args.gen
    t_script = time.time()
    gen_dir = spec.GENS_DIR / (f"{gen}_dry" if args.dry_run else gen)
    gen_dir.mkdir(parents=True, exist_ok=True)
    output = gen_dir / "turn1.jsonl"
    summary_path = gen_dir / "turn1_summary.json"
    print(f"=== turn 1, gen={gen}, dry_run={args.dry_run}, spec {spec.spec_hash()} -> {output} ===")

    question_rows = prompts.load_questions()
    if args.dry_run:
        question_rows = dry_run_questions(question_rows)
    expected_keys = {
        (row["question_id"], condition, sample_idx)
        for row in question_rows
        for condition in spec.GEN_CONDITIONS[gen]
        for sample_idx in range(spec.N_SAMPLES)
    }
    print(f"  {len(question_rows)} questions x {spec.GEN_CONDITIONS[gen]} x {spec.N_SAMPLES} samples = {len(expected_keys)} records")

    done_keys = load_done_keys(output, record_key)
    print(f"  Output holds {len(done_keys)} records already; {len(expected_keys - done_keys)} to generate")
    assert done_keys <= expected_keys, f"{len(done_keys - expected_keys)} records in {output} are outside the expected key set"

    tokenizer = AutoTokenizer.from_pretrained(spec.MODEL_NAME)
    items = build_items(gen, tokenizer, question_rows)
    if gen == "bio":
        check_bio_prompts_against_260819(items)
    todo = [item for item in items if (item["question_row"]["question_id"], item["condition"], 0) not in done_keys]
    for item in todo:
        for sample_idx in range(spec.N_SAMPLES):
            assert (item["question_row"]["question_id"], item["condition"], sample_idx) not in done_keys, "a prompt is partially written"

    max_model_len = spec.NOCOT_MAX_MODEL_LEN if gen == "nocot" else spec.COT_MAX_MODEL_LEN
    max_tokens = spec.NOCOT_MAX_TOKENS if gen == "nocot" else spec.COT_MAX_TOKENS
    lengths = [len(item["token_ids"]) for item in items]
    print(f"  Prompt tokens: min={min(lengths)} mean={sum(lengths) / len(lengths):.0f} max={max(lengths)} (budget {max_model_len - max_tokens})")
    assert max(lengths) <= max_model_len - max_tokens

    example = next(item for item in items if item["condition"] == "cue_dist")
    print(f"\n  Sample cue_dist prompt + prefill:\n{example['prompt'] + example['prefill']}\n")

    timings = {"n_generated_this_run": 0, "generation_seconds": 0.0}
    siblings: list[dict] = []
    if todo:
        llm = init_vllm_engine(spec.MODEL_NAME, args.gpu_util, max_model_len)
        if gen == "nocot":
            followup_generation.check_logprob_invariance(llm, todo, LETTER_TARGETS, tol_pyes=LETTER_INVARIANCE_TOL)
        else:
            probe_items = [
                {
                    "token_ids": item["token_ids"],
                    "row": {
                        "question_id": item["question_row"]["question_id"],
                        "condition": item["condition"],
                        "order": item["question_row"]["order"],
                        "cue_variant": spec.CUE_VARIANT,
                    },
                }
                for item in todo[:16]
            ]
            check_generation_invariance(llm, probe_items, seed=spec.TURN1_SEED_BASE[gen], temperature=spec.TEMPERATURE)

        print(f"\n=== Generating ({len(todo)} prompts, n={spec.N_SAMPLES}, temp={spec.TEMPERATURE}, max_tokens={max_tokens}) ===")
        t_start = time.time()
        for start in range(0, len(todo), spec.BATCH_SIZE):
            batch = todo[start : start + spec.BATCH_SIZE]
            t_batch = time.time()
            records, batch_siblings = run_batch(llm, gen, batch)
            append_records(output, records)
            siblings.extend(batch_siblings)
            done = start + len(batch)
            elapsed = time.time() - t_start
            eta = (len(todo) - done) / (done / elapsed)
            print(f"  [{done}/{len(todo)}] +{len(records)} samples in {time.time() - t_batch:.0f}s | ETA {eta / 60:.0f}m", flush=True)
        timings = {"n_generated_this_run": len(todo), "generation_seconds": time.time() - t_start}
        print(f"\n=== Generated in {timings['generation_seconds'] / 60:.1f} min ===")

    records = finalize(output, record_key, expected_keys)
    summary = {
        "gen": gen,
        "dry_run": args.dry_run,
        "spec_hash": spec.spec_hash(),
        "model": spec.MODEL_NAME,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "prompt_tokens": {"min": min(lengths), "max": max(lengths)},
        **summarize(gen, records, args.dry_run),
        "sibling_logprob_agreement_this_run": summarize_siblings(siblings) if gen == "nocot" else None,
        "timings": {**timings, "script_seconds": time.time() - t_script},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
