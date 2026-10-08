"""Removal-wording follow-ups for a generation pass (SCHEMA.md: gens/<gen>/followups.jsonl).

Reads gens/<gen>/turn1.jsonl, keeps the cued rows and asks each turn-1 sample
the question's assigned removal wording (260823's `draw_wordings` / `ARM_OF`):

    user:      <turn-1 prompt>
    assistant: <turn-1 completion>
    user:      <removal or removal_explicit follow-up>
    assistant: <- generated here, logprobs at every position

For `nocot` there is one extra follow-up per prompt, sample_key "top", whose
turn-1 completion names the prompt's top letter under the logprobs (the
logprob arm's label); it is byte-identical to a sample's prompt whenever a
sample picked that letter, which `check_top_vs_samples` asserts (and reports
the two rows' bf16 P(Yes) agreement).

Prompt construction, P(Yes) extraction, the engine and the preflight are
260823's, imported. Skips, mirroring 260823: explicit-wording rows whose turn-1
answer did not parse; bio rows whose CoT was truncated (no <|eot_id|> to
continue from); "top" rows with no top letter. Every skip is listed in the
summary.

Usage:
    python generate_followups.py --gen nocot [--dry-run] [--gpu-util F]
    python generate_followups.py --gen bio   [--dry-run] [--gpu-util F]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import transformers
import vllm
from transformers import AutoTokenizer
from vllm import SamplingParams

import domain_imports
import prompts
import spec
from domain_imports import ARM_OF, draw_wordings
from output_store import append_records, finalize, load_done_keys, read_records

g = domain_imports.import_followup_generation()
assert g.__file__.startswith(str(spec.FOLLOWUP_DIR)), g.__file__

ORDERS = ("gold_first", "gold_second")
SAMPLE_KEYS = tuple(str(i) for i in range(spec.N_SAMPLES))
TOP_KEY = "top"
#: vLLM's bf16 Yes-No logit moves by one grid step (0.125 or 0.25 nat) with the
#: batch layout on a few prompts, up to 0.062 in probability near 0.5. This is
#: the invariance preflight's tolerance and a reporting threshold; the
#: top-vs-sample P(Yes) agreement is reported, not asserted (user decision
#: 2026-10-08: identical inputs are asserted exactly on text and token ids, and
#: the labels read score_pyes.py's float32 P(Yes), not this value)
PYES_BF16_NOISE_BAR = 0.05
TOP_SLOT = spec.N_SAMPLES
EXPLICIT_ARM = ARM_OF[spec.DEFINITION]["explicit"]


def record_key(record: dict) -> tuple:
    return (record["question_id"], record["condition"], record["sample_key"])


def row_seed(gen: str, row: dict, sample_slot: int) -> int:
    offset = 6 * sample_slot + 2 * spec.CUED_CONDITIONS.index(row["condition"]) + ORDERS.index(row["order"])
    assert 0 <= offset < 64, offset
    return spec.FU_SEED_BASE[gen] + 64 * row["question_id"] + offset


def question_arm(question_id: int) -> str:
    wording = draw_wordings(question_id)[spec.DEFINITION][1]
    return ARM_OF[spec.DEFINITION][wording]


def top_row(sample_rows: list[dict]) -> dict:
    """The synthetic turn-1 row for sample_key "top": the prompt's top letter as its completion."""
    first = sample_rows[0]
    for row in sample_rows:
        assert (row["top_letter"], row["prompt"], row["prefill"]) == (first["top_letter"], first["prompt"], first["prefill"])
    letter = first["top_letter"]
    assert letter in spec.LETTERS, letter
    row = dict(first)
    row["sample_idx"] = None
    row["sampled_text"] = letter
    row["sampled_token_ids"] = [spec.LETTER_TOKEN_IDS[letter]]
    row["completion"] = spec.NOCOT_PREFILL + letter + spec.NOCOT_CLOSING
    row["answer_letter"], row["answer_stage"] = domain_imports.extract_answer(row["completion"])
    assert row["answer_letter"] == letter
    return row


def plan_followups(gen: str, turn1_rows: list[dict]) -> tuple[list[dict], dict]:
    """(planned (row, sample_key, sample_slot, arm) per follow-up, skip lists)."""
    by_prompt: dict[tuple, list[dict]] = defaultdict(list)
    for row in turn1_rows:
        by_prompt[(row["question_id"], row["condition"])].append(row)

    planned = []
    skipped = {"explicit_unparsed": [], "truncated": [], "top_letter_null": []}
    for (question_id, condition), sample_rows in sorted(by_prompt.items()):
        assert [r["sample_idx"] for r in sample_rows] == list(range(spec.N_SAMPLES)), (question_id, condition)
        arm = question_arm(question_id)
        candidates = [(row, str(row["sample_idx"]), row["sample_idx"]) for row in sample_rows]
        if gen == "nocot":
            if sample_rows[0]["top_letter"] is None:
                skipped["top_letter_null"].append([question_id, condition, TOP_KEY])
            else:
                candidates.append((top_row(sample_rows), TOP_KEY, TOP_SLOT))
        for row, sample_key, sample_slot in candidates:
            if gen == "bio" and row["finish_reason"] != "stop":
                skipped["truncated"].append([question_id, condition, sample_key])
                continue
            if arm == EXPLICIT_ARM and row["answer_letter"] is None:
                skipped["explicit_unparsed"].append([question_id, condition, sample_key])
                continue
            planned.append({"row": row, "sample_key": sample_key, "sample_slot": sample_slot, "arm": arm})
    return planned, skipped


def build_items(gen: str, tokenizer, planned: list[dict]) -> list[dict]:
    suffixes: dict[str, str] = {}
    items = []
    for plan in planned:
        row = plan["row"]
        followup = g.followup_text(plan["arm"], row)
        if followup not in suffixes:
            suffixes[followup] = g.build_followup_suffix(tokenizer, followup)
        suffix = suffixes[followup]
        prompt_text = g.build_followup_prompt(row, suffix)
        token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        items.append(
            {
                "row": row,
                "arm": plan["arm"],
                "sample_key": plan["sample_key"],
                "followup": followup,
                "suffix": suffix,
                "prompt_text": prompt_text,
                "token_ids": token_ids,
                "seed": row_seed(gen, row, plan["sample_slot"]),
            }
        )
    print(f"  {len(suffixes)} distinct follow-up wordings:")
    for text in suffixes:
        print(f"    {text!r}")
    seeds = [item["seed"] for item in items]
    assert len(set(seeds)) == len(seeds), "follow-up seeds collide"
    print(f"  CHECK: {len(items)} follow-ups, {len(seeds)} distinct seeds in [{min(seeds)}, {max(seeds)}]")
    return items


def check_nocot_token_layout(tokenizer, items: list[dict]) -> dict:
    """The letter and the follow-up boundary must sit on known token positions:
    ids == prompt ids (with prefill) + [letter id] + ids(")") + ids(suffix).
    Asserted on every row whose sampled token is a letter; rows that sampled
    something else are reported (their answer did not parse, so no probe reads
    them) with whether the same layout still holds for their sampled id."""
    closing_ids = tokenizer(spec.NOCOT_CLOSING, add_special_tokens=False)["input_ids"]
    letter_ids = set(spec.LETTER_TOKEN_IDS.values())
    n_letter, non_letter = 0, []
    for item in items:
        row = item["row"]
        suffix_ids = tokenizer(item["suffix"], add_special_tokens=False)["input_ids"]
        expected = prompts.prompt_ids(tokenizer, row["prompt"], row["prefill"]) + row["sampled_token_ids"] + closing_ids + suffix_ids
        layout_holds = item["token_ids"] == expected
        if row["sampled_token_ids"][0] in letter_ids:
            assert row["sampled_token_ids"] == [spec.LETTER_TOKEN_IDS[row["answer_letter"]]], (row["sampled_token_ids"], row["answer_letter"])
            assert layout_holds, f"q{row['question_id']}/{row['condition']}/{item['sample_key']}: token layout differs from prompt + letter + closing + suffix"
            n_letter += 1
        else:
            non_letter.append({"key": [row["question_id"], row["condition"], item["sample_key"]], "sampled_token_ids": row["sampled_token_ids"], "sampled_text": row["sampled_text"], "layout_holds": layout_holds})
    print(f"  CHECK: token layout prompt + letter + closing + suffix holds on all {n_letter} letter rows; {len(non_letter)} rows sampled a non-letter token")
    return {"n_letter_rows": n_letter, "non_letter_rows": non_letter}


def said_from(token_ids: list[int], ans_pos: int | None, targets: dict[str, set[int]]) -> str | None:
    if ans_pos is None:
        return None
    token_id = token_ids[ans_pos]
    if token_id in targets["yes_broad"]:
        return "Yes"
    assert token_id in targets["no_broad"], token_id
    return "No"


def run_batch(llm, batch: list[dict], targets: dict[str, set[int]]) -> list[dict]:
    outputs = llm.generate(
        [{"prompt_token_ids": item["token_ids"]} for item in batch],
        [
            SamplingParams(n=1, temperature=spec.TEMPERATURE, max_tokens=spec.FU_MAX_TOKENS, logprobs=spec.N_LOGPROBS, seed=item["seed"])
            for item in batch
        ],
    )
    assert len(outputs) == len(batch)
    records = []
    for item, output in zip(batch, outputs):
        row = item["row"]
        assert len(output.outputs) == 1
        completion = output.outputs[0]
        token_ids = list(completion.token_ids)
        assert token_ids, "empty follow-up completion"
        assert len(completion.logprobs) == len(token_ids)
        stats0 = g.position_stats(completion.logprobs[0], targets)
        ans_pos = g.answer_position(token_ids, targets)
        records.append(
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "split": row["split"],
                "order": row["order"],
                "condition": row["condition"],
                "cue_letter": row["cue_letter"],
                "is_assigned": row["is_assigned"],
                "sample_key": item["sample_key"],
                "sample_idx": row["sample_idx"],
                "answer_letter": row["answer_letter"],
                "arm": item["arm"],
                "followup_text": item["followup"],
                "prompt": row["prompt"],
                "completion": row["completion"],
                "followup_completion": completion.text,
                "finish_reason": completion.finish_reason,
                "first_token_ids": token_ids[:8],
                "p_yes_pos0": stats0["p_yes"],
                "p_yes_broad_pos0": stats0["p_yes_broad"],
                "mass_yesno_pos0": stats0["mass_yesno"],
                "top1_id_pos0": stats0["top1_id"],
                "top1_text_pos0": stats0["top1_text"],
                "missing_yes_pos0": stats0["missing_yes"],
                "missing_no_pos0": stats0["missing_no"],
                "ans_pos": ans_pos,
                "said": said_from(token_ids, ans_pos, targets),
                "seed": item["seed"],
                "spec_hash": spec.spec_hash(),
            }
        )
    return records


def check_top_vs_samples(records: list[dict]) -> dict:
    """A "top" row and a sample row with the same letter must have the same
    prompt text (asserted). Their bf16 P(Yes) at position 0 is reported only:
    vLLM's reading of an identical prompt can differ by one bf16 grid step."""
    by_prompt: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        by_prompt[(record["question_id"], record["condition"])].append(record)
    diffs, n_pairs, n_none = [], 0, 0
    for prompt_records in by_prompt.values():
        tops = [r for r in prompt_records if r["sample_key"] == TOP_KEY]
        if not tops:
            continue
        (top,) = tops
        for record in prompt_records:
            if record["sample_key"] == TOP_KEY or record["answer_letter"] != top["answer_letter"]:
                continue
            assert record["prompt"] + record["completion"] == top["prompt"] + top["completion"]
            n_pairs += 1
            if record["p_yes_pos0"] is None or top["p_yes_pos0"] is None:
                n_none += 1
                continue
            diffs.append(abs(record["p_yes_pos0"] - top["p_yes_pos0"]))
    max_diff = max(diffs) if diffs else 0.0
    n_over_1e3 = sum(d >= 1e-3 for d in diffs)
    n_over_1e2 = sum(d >= 1e-2 for d in diffs)
    n_over_bar = sum(d >= PYES_BF16_NOISE_BAR for d in diffs)
    print(f"  REPORT: top vs same-letter sample P(Yes) pos0: {n_pairs} pairs, max |diff| {max_diff:.2e}, {n_over_1e3} >= 1e-3, {n_over_1e2} >= 1e-2, {n_over_bar} >= {PYES_BF16_NOISE_BAR}, {n_none} pairs with P(Yes) null")
    return {"n_pairs": n_pairs, "max_abs_diff": max_diff, "n_over_1e-3": n_over_1e3, "n_over_1e-2": n_over_1e2, "n_over_bar": n_over_bar, "n_pairs_p_yes_null": n_none}


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
    turn1_path = gen_dir / "turn1.jsonl"
    output = gen_dir / "followups.jsonl"
    summary_path = gen_dir / "followups_summary.json"
    print(f"=== follow-ups, gen={gen}, dry_run={args.dry_run}, spec {spec.spec_hash()} -> {output} ===")

    turn1_summary = json.loads((gen_dir / "turn1_summary.json").read_text())
    assert turn1_summary["spec_hash"] == spec.spec_hash(), "turn1_summary.json was written under another spec"
    turn1_rows = read_records(turn1_path)
    assert turn1_rows, turn1_path
    for row in turn1_rows:
        assert row["spec_hash"] == spec.spec_hash(), "turn1.jsonl was written under another spec"
    cued_rows = [row for row in turn1_rows if row["condition"] in spec.CUED_CONDITIONS]
    print(f"  {len(turn1_rows)} turn-1 rows, {len(cued_rows)} cued")

    planned, skipped = plan_followups(gen, cued_rows)
    for name, keys in skipped.items():
        print(f"  skipped ({name}): {len(keys)}")
    arm_counts = defaultdict(int)
    for plan in planned:
        arm_counts[plan["arm"]] += 1
    print(f"  planned follow-ups per arm: {dict(arm_counts)}")
    expected_keys = {(plan["row"]["question_id"], plan["row"]["condition"], plan["sample_key"]) for plan in planned}
    assert len(expected_keys) == len(planned)

    done_keys = load_done_keys(output, record_key)
    print(f"  Output holds {len(done_keys)} records already; {len(expected_keys - done_keys)} to generate")
    assert done_keys <= expected_keys, f"{len(done_keys - expected_keys)} records in {output} are outside the expected key set"

    print("\n=== Tokenizer + prompt construction ===")
    tokenizer = AutoTokenizer.from_pretrained(spec.MODEL_NAME)
    targets = g.resolve_targets(tokenizer)
    items = build_items(gen, tokenizer, planned)
    g.check_prompts(tokenizer, items)
    token_layout = check_nocot_token_layout(tokenizer, items) if gen == "nocot" else None
    lengths = [len(item["token_ids"]) for item in items]
    budget = spec.FU_MAX_MODEL_LEN - spec.FU_MAX_TOKENS
    print(f"  Prompt tokens: min={min(lengths)} mean={sum(lengths) / len(lengths):.0f} max={max(lengths)} (budget {budget})")
    assert max(lengths) <= budget
    example = next(item for item in items if item["row"]["condition"] == "cue_dist")
    print(f"\n  Sample two-turn prompt [{example['arm']}, sample_key {example['sample_key']}]:\n{example['prompt_text']}\n")

    todo = [item for item in items if (item["row"]["question_id"], item["row"]["condition"], item["sample_key"]) not in done_keys]
    timings = {"n_generated_this_run": 0, "generation_seconds": 0.0}
    if todo:
        llm = g.init_vllm_engine(spec.MODEL_NAME, args.gpu_util, spec.FU_MAX_MODEL_LEN)
        # vLLM's bf16 P(Yes) is diagnostic only (the labels read score_pyes.py's
        # float32 value), and it moves by one ~0.1-nat step with the batch layout
        # on a few prompts, so the preflight's bar is that noise, not 1e-6
        g.check_logprob_invariance(llm, todo, targets, tol_pyes=PYES_BF16_NOISE_BAR)
        print(f"\n=== Generating ({len(todo)} prompts, n=1, temp={spec.TEMPERATURE}, max_tokens={spec.FU_MAX_TOKENS}, logprobs={spec.N_LOGPROBS}) ===")
        t_start = time.time()
        for start in range(0, len(todo), spec.BATCH_SIZE):
            batch = todo[start : start + spec.BATCH_SIZE]
            t_batch = time.time()
            records = run_batch(llm, batch, targets)
            append_records(output, records)
            done = start + len(batch)
            elapsed = time.time() - t_start
            eta = (len(todo) - done) / (done / elapsed)
            print(f"  [{done}/{len(todo)}] +{len(records)} follow-ups in {time.time() - t_batch:.0f}s | ETA {eta / 60:.0f}m", flush=True)
        timings = {"n_generated_this_run": len(todo), "generation_seconds": time.time() - t_start}
        print(f"\n=== Generated in {timings['generation_seconds'] / 60:.1f} min ===")

    records = finalize(output, record_key, expected_keys)
    said_counts = defaultdict(int)
    for record in records:
        said_counts[str(record["said"])] += 1
    summary = {
        "gen": gen,
        "dry_run": args.dry_run,
        "spec_hash": spec.spec_hash(),
        "model": spec.MODEL_NAME,
        "vllm_version": vllm.__version__,
        "transformers_version": transformers.__version__,
        "n_turn1_rows": len(turn1_rows),
        "n_cued_turn1_rows": len(cued_rows),
        "n_records": len(records),
        "per_arm": dict(arm_counts),
        "skipped": skipped,
        "said": dict(said_counts),
        "finish_stop_rate": sum(r["finish_reason"] == "stop" for r in records) / len(records),
        "p_yes_pos0_null": sum(r["p_yes_pos0"] is None for r in records),
        "prompt_tokens": {"min": min(lengths), "max": max(lengths)},
        "token_layout": token_layout,
        "top_vs_samples_report": check_top_vs_samples(records) if gen == "nocot" else None,
        "timings": {**timings, "script_seconds": time.time() - t_script},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
