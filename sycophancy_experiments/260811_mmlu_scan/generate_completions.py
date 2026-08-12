"""Generate base-model completions for the MMLU-Pro scan set.

Builds a zero-shot chat prompt for each question -- the question and its
options as a single user message, with the MMLU-Pro instruction to reason step
by step and finish with "the answer is (X)" -- and samples N completions per
question with in-process vLLM.

The answer-format instruction is kept verbatim from TIGER-AI-Lab/MMLU-Pro so
their answer-extraction convention still applies. The 5-shot completion-style
prompting from the reference implementation is deliberately dropped: wrapping a
few-shot blob in a chat template makes Llama-3-8B-Instruct treat it as
conversation setup ("go ahead and provide the next question") rather than
answering, and zero-shot chat is closer to how the eventual RL domain will
present questions.

Resumable: completions are appended to a per-category JSONL cache after every
batch, so the script can be killed and restarted without re-generating. Each
question gets its own sampling seed (base seed + question_id), so a resumed run
reproduces exactly what an uninterrupted run would have produced.

Usage:
    # Smoke test: 2 questions per category
    python generate_completions.py --dry-run

    # Full run (resumes from cache)
    python generate_completions.py

    # Single category
    python generate_completions.py --categories math physics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

EXP_DIR = Path(__file__).resolve().parent

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 2048
N_SAMPLES = 3
TEMPERATURE = 1.0
SEED = 42
BATCH_SIZE = 200
DRY_RUN_PER_CATEGORY = 2

# Answer-format instruction from TIGER-AI-Lab/MMLU-Pro cot_prompt_lib/initial_prompt.txt,
# reworded from plural to singular for the zero-shot setting.
INSTRUCTION = (
    "The following is a multiple choice question about {category}. "
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_name(category: str) -> str:
    return category.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_question(row: dict) -> str:
    """Render the question and its options in MMLU-Pro layout."""
    text = "Question:\n" + row["question"] + "\n"
    text += "Options:\n"
    for i, opt in enumerate(row["options"]):
        text += "{}. {}\n".format(CHOICES[i], opt)
    return text


def build_chat_prompt(tokenizer, row: dict) -> tuple[str, list[int]]:
    """Build the zero-shot chat prompt and its token ids."""
    user_content = INSTRUCTION.format(category=row["category"]) + "\n\n" + format_question(row)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content.rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # add_special_tokens=False: the chat template already emits <|begin_of_text|>.
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text, token_ids


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def read_cache(path: Path) -> list[dict]:
    """Read a cache file, skipping lines a hard kill may have left truncated."""
    if not path.exists():
        return []
    records, n_bad = [], 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        print(f"  WARNING: skipped {n_bad} malformed line(s) in {path}")
    return records


def dedupe(records: list[dict]) -> list[dict]:
    """Keep the first record for each (question_id, sample_idx)."""
    seen: set[tuple[int, int]] = set()
    unique = []
    for rec in records:
        key = (rec["question_id"], rec["sample_idx"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique


def load_cache(cache_dir: Path, categories: list[str]) -> dict[int, int]:
    """Return {question_id: number of cached samples} across all category caches."""
    counts: dict[int, int] = defaultdict(int)
    for category in categories:
        path = cache_dir / f"{safe_name(category)}.jsonl"
        for rec in dedupe(read_cache(path)):
            counts[rec["question_id"]] += 1
    return counts


def append_records(cache_dir: Path, category: str, records: list[dict]) -> None:
    path = cache_dir / f"{safe_name(category)}.jsonl"
    # A kill mid-write can leave a final line with no newline; without this the
    # next append would concatenate onto it and corrupt an otherwise good record.
    needs_newline = path.exists() and path.stat().st_size > 0
    if needs_newline:
        with path.open("rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    with path.open("a") as f:
        if needs_newline:
            f.write("\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")
        f.flush()


def merge_caches(cache_dir: Path, categories: list[str], output: Path) -> int:
    """Merge per-category caches into one deduplicated JSONL."""
    n = 0
    with output.open("w") as out:
        for category in sorted(categories):
            path = cache_dir / f"{safe_name(category)}.jsonl"
            rows = dedupe(read_cache(path))
            for rec in sorted(rows, key=lambda r: (r["question_id"], r["sample_idx"])):
                out.write(json.dumps(rec) + "\n")
                n += 1
    return n


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def init_vllm_engine(model_name: str, gpu_util: float, max_model_len: int) -> LLM:
    print("\n=== Initializing vLLM Engine ===")
    llm = LLM(
        model_name,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
    )
    smoke = llm.generate(["Hello, world!"], SamplingParams(max_tokens=8, temperature=0))
    assert smoke and smoke[0].outputs, "vLLM smoke test failed"
    print(f"  CHECK: smoke test passed ({smoke[0].outputs[0].text[:40]!r})")
    return llm


def run_batch(
    llm: LLM,
    batch: list[dict],
    *,
    n_samples_needed: list[int],
    temperature: float,
    max_new_tokens: int,
    seed: int,
    model_name: str,
) -> list[dict]:
    """Generate for one batch of questions and return new completion records."""
    prompts = [{"prompt_token_ids": item["token_ids"]} for item in batch]
    params = [
        SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_new_tokens,
            seed=seed + item["row"]["question_id"],
        )
        for item, n in zip(batch, n_samples_needed)
    ]

    outputs = llm.generate(prompts, params)
    assert len(outputs) == len(batch), f"Expected {len(batch)} outputs, got {len(outputs)}"

    records: list[dict] = []
    for item, out, n_needed in zip(batch, outputs, n_samples_needed):
        row = item["row"]
        assert len(out.outputs) == n_needed, (
            f"q{row['question_id']}: expected {n_needed} samples, got {len(out.outputs)}"
        )
        for offset, completion in enumerate(out.outputs):
            records.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "sample_idx": item["start_idx"] + offset,
                    "prompt": item["prompt_text"],
                    "completion": completion.text,
                    "finish_reason": completion.finish_reason,
                    "answer": row["answer"],
                    "answer_index": row["answer_index"],
                    "options": row["options"],
                    "model": model_name,
                    "temperature": temperature,
                    "seed": seed + row["question_id"],
                }
            )
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate base-model completions for the MMLU-Pro scan set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scan-set", type=Path, default=EXP_DIR / "scan_set.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=EXP_DIR / "completions")
    parser.add_argument("--output", type=Path, default=EXP_DIR / "completions.jsonl")
    parser.add_argument("--model", type=str, default=BASE_MODEL)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Questions per vLLM call; cache flushed after each (default: {BATCH_SIZE})",
    )
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these categories (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Only {DRY_RUN_PER_CATEGORY} questions per category, into a dry_run cache dir",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing caches into the output JSONL without generating",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scan_rows = read_jsonl(args.scan_set)
    print(f"=== Loaded {len(scan_rows)} questions ===")

    if args.categories:
        keep = set(args.categories)
        unknown = keep - {r["category"] for r in scan_rows}
        assert not unknown, f"Unknown categories: {sorted(unknown)}"
        scan_rows = [r for r in scan_rows if r["category"] in keep]
        print(f"  Filtered to {sorted(keep)}: {len(scan_rows)} questions")

    cache_dir = args.cache_dir / "dry_run" if args.dry_run else args.cache_dir
    output = args.output.with_name("dry_run_" + args.output.name) if args.dry_run else args.output

    if args.dry_run:
        per_category: dict[str, list[dict]] = defaultdict(list)
        for row in scan_rows:
            if len(per_category[row["category"]]) < DRY_RUN_PER_CATEGORY:
                per_category[row["category"]].append(row)
        scan_rows = [r for rows in per_category.values() for r in rows]
        print(f"  Dry-run: {len(scan_rows)} questions ({DRY_RUN_PER_CATEGORY}/category)")

    categories = sorted({r["category"] for r in scan_rows})
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        total = merge_caches(cache_dir, categories, output)
        print(f"  Merged {total} records into {output}")
        return 0

    cached_counts = load_cache(cache_dir, categories)
    todo = [r for r in scan_rows if cached_counts.get(r["question_id"], 0) < args.n_samples]
    n_cached = sum(min(cached_counts.get(r["question_id"], 0), args.n_samples) for r in scan_rows)
    print(
        f"  Cache: {n_cached}/{len(scan_rows) * args.n_samples} samples present, "
        f"{len(todo)} questions still need generation"
    )

    if not todo:
        print("  Nothing to generate.")
        total = merge_caches(cache_dir, categories, output)
        print(f"  Merged {total} records into {output}")
        return 0

    print("\n=== Tokenizer + Prompt Construction ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    max_prompt_tokens = args.max_model_len - args.max_new_tokens
    assert max_prompt_tokens > 0, "max_model_len must exceed max_new_tokens"

    items = []
    for row in todo:
        prompt_text, token_ids = build_chat_prompt(tokenizer, row)
        items.append(
            {
                "row": row,
                "prompt_text": prompt_text,
                "token_ids": token_ids,
                "start_idx": cached_counts.get(row["question_id"], 0),
            }
        )

    lengths = [len(it["token_ids"]) for it in items]
    print(
        f"  Prompt tokens: min={min(lengths)} mean={sum(lengths)/len(lengths):.0f} "
        f"max={max(lengths)} (budget {max_prompt_tokens})"
    )
    assert max(lengths) <= max_prompt_tokens, (
        f"A prompt exceeds the token budget ({max(lengths)} > {max_prompt_tokens})"
    )
    print(f"\n  Sample prompt:\n{items[0]['prompt_text']}")

    llm = init_vllm_engine(args.model, args.gpu_util, args.max_model_len)

    print(
        f"\n=== Generating ({len(items)} questions, n={args.n_samples}, "
        f"temp={args.temperature}) ==="
    )
    t_start = time.time()
    n_written = 0
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        n_needed = [args.n_samples - it["start_idx"] for it in batch]

        t0 = time.time()
        records = run_batch(
            llm,
            batch,
            n_samples_needed=n_needed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            model_name=args.model,
        )

        by_category: dict[str, list[dict]] = defaultdict(list)
        for rec in records:
            by_category[rec["category"]].append(rec)
        for category, recs in by_category.items():
            append_records(cache_dir, category, recs)

        n_written += len(records)
        done = min(start + args.batch_size, len(items))
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed else 0
        eta = (len(items) - done) / rate if rate else 0
        print(
            f"  [{done}/{len(items)}] +{len(records)} samples in {time.time()-t0:.0f}s "
            f"| total {n_written} | ETA {eta/60:.0f}m",
            flush=True,
        )

    print(f"\n=== Done in {(time.time()-t_start)/60:.1f} min ===")
    total = merge_caches(cache_dir, categories, output)
    expected = len(scan_rows) * args.n_samples
    print(f"  Merged {total} records into {output} (expected {expected})")
    if total != expected:
        print(f"  WARNING: {expected - total} samples missing; re-run to fill gaps")

    return 0


if __name__ == "__main__":
    sys.exit(main())
