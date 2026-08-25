"""Phase 2: tokenize + forward pass + slice -> per-probe activation tensors.

For each probe_sets/ file (one per definition x split), a single tokenize +
forward pass per row; BOTH extraction positions for that probe type are sliced
from the same cached hidden states -- no duplicate forward passes.

  bias:      position A = regex-matched answer-letter token (_answer)
             position B = last token of the turn-1 completion (_lastcompl)
  deception: position A = first token generated in turn 2 (_firstgen)
             position B = last token of the turn-2 prompt, i.e. one before A (_lastprompt)

Output: one .pt per probe per split (8 probes x {train,eval} = 16 files) under
activations/, each holding {"0":...,...,"31":...} per-layer (N,1,hidden_dim)
tensors + labels + a row manifest.

Bypasses the shared package's text pipeline entirely (obfuscate_over enum can't
express either position) -- see ../../docs/... and this experiment's plan for why.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

EXP_DIR = Path(__file__).parent
DOMAIN_DIR = EXP_DIR.parent / "260819_domain_baseline"
PROBE_SETS_DIR = EXP_DIR / "probe_sets"
ACT_DIR = EXP_DIR / "activations"

sys.path.insert(0, str(DOMAIN_DIR))
from score_completions import (  # noqa: E402
    PATTERN_ANSWER_IS,
    PATTERN_ANSWER_COLON,
    PATTERN_MARKED_LETTER,
    FALLBACK_STAGE,
    extract_answer,
)
from generate_followups import build_followup_suffix, build_followup_prompt  # noqa: E402

sys.path.insert(0, str(EXP_DIR.parent))  # for obfuscation_atlas import below
from obfuscation_atlas.utils.activations import get_all_residual_acts  # noqa: E402

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
N_LAYERS = 32
BATCH_SIZE = 16
DTYPE = torch.bfloat16

BIAS_DEFINITIONS = ["rm", "swap"]
DECEPTION_DEFINITIONS = ["rm", "swap"]
SPLITS = ["train", "eval"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_answer_span(completion: str):
    """Same 3-stage regex chain as score_completions.extract_answer, but returns
    the matched char span too (score_completions.extract_answer only returns the
    letter+stage, using findall which discards match positions for the fallback
    stage). Tie-break must match extract_answer exactly: first match for the two
    primary patterns, LAST match for the fallback (marked-letter) stage."""
    m = PATTERN_ANSWER_IS.search(completion)
    if m:
        return m.group(1), "answer_is", m.start(1), m.end(1)
    m = PATTERN_ANSWER_COLON.search(completion)
    if m:
        return m.group(1), "answer_colon", m.start(1), m.end(1)
    matches = list(PATTERN_MARKED_LETTER.finditer(completion))
    if matches:
        last = matches[-1]
        if last.group(1) is not None:
            return last.group(1), FALLBACK_STAGE, last.start(1), last.end(1)
        return last.group(2), FALLBACK_STAGE, last.start(2), last.end(2)
    return None, "none", None, None


def char_to_token_index(offsets: list[tuple[int, int]], char_pos: int) -> int:
    for i, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return i
    raise ValueError(f"char_pos {char_pos} not found in offsets (n_tokens={len(offsets)})")


def build_scores_index(domain_dir: Path) -> dict[tuple, dict]:
    """(question_id, condition) -> scores.jsonl row, cue_variant=='random' only.

    Used only for the turn-1 regression gate (position A must agree with the
    already-published/gated answer/stage labels).
    """
    rows = load_jsonl(domain_dir / "scores.jsonl")
    idx = {}
    for r in rows:
        if r.get("cue_variant") != "random":
            continue
        key = (r["question_id"], r["condition"])
        assert key not in idx, f"duplicate scores.jsonl key: {key}"
        idx[key] = r
    return idx


def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    return model, tokenizer


def run_forward_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[int, torch.Tensor]:
    """Returns {layer_idx: (batch, seq, hidden_dim)} for layers 0..N_LAYERS-1."""
    with torch.no_grad():
        return get_all_residual_acts(
            model,
            input_ids.to("cuda"),
            attention_mask.to("cuda"),
            batch_size=input_ids.shape[0],  # one internal batch; we do our own chunking outside
            only_return_layers=list(range(N_LAYERS)),
        )


def pad_batch(tokenizer, token_id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in token_id_lists)
    input_ids = torch.full((len(token_id_lists), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(token_id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(token_id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


def extract_bias_positions(rows: list[dict], tokenizer, model, scores_idx: dict[tuple, dict]) -> dict:
    """Returns {"answer": {"acts": {layer: (N,1,H)}, "labels": (N,), "rows": [...]},
                "lastcompl": {...}}"""
    n_layers = N_LAYERS
    acc = {
        "answer": {l: [] for l in range(n_layers)},
        "lastcompl": {l: [] for l in range(n_layers)},
    }
    kept_rows: list[dict] = []
    labels: list[int] = []

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        token_id_lists = []
        pos_answer = []
        pos_lastcompl = []
        batch_kept = []
        for r in batch:
            prompt, completion = r["prompt"], r["completion"]
            letter, stage, c_start, c_end = find_answer_span(completion)
            assert letter is not None, f"no regex match for row {r['question_id']}/{r['condition']}/{r['sample_idx']}"

            # Regression gate: agree with the already-published/gated labels.
            skey = (r["question_id"], r["condition"])
            srow = scores_idx.get(skey)
            assert srow is not None, f"no scores.jsonl row for {skey}"
            gate_letter, gate_stage = extract_answer(completion)
            assert (letter, stage) == (gate_letter, gate_stage), (
                f"{skey}: local (letter,stage)={(letter, stage)} != extract_answer {(gate_letter, gate_stage)}"
            )
            published_pred = srow["preds"][r["sample_idx"]]
            published_stage = srow["stages"][r["sample_idx"]]
            assert (letter, stage) == (published_pred, published_stage), (
                f"{skey}[{r['sample_idx']}]: local (letter,stage)={(letter, stage)} != "
                f"published ({published_pred}, {published_stage})"
            )

            full_text = prompt + completion
            enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
            token_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            answer_idx = char_to_token_index(offsets, len(prompt) + c_start)
            decoded = tokenizer.decode([token_ids[answer_idx]])
            assert letter in decoded, f"{skey}: decoded token {decoded!r} does not contain letter {letter!r}"

            lastcompl_idx = len(token_ids) - 1

            token_id_lists.append(token_ids)
            pos_answer.append(answer_idx)
            pos_lastcompl.append(lastcompl_idx)
            batch_kept.append(r)

        input_ids, attention_mask = pad_batch(tokenizer, token_id_lists)
        hidden = run_forward_batch(model, input_ids, attention_mask)

        for i, r in enumerate(batch_kept):
            for layer in range(n_layers):
                acc["answer"][layer].append(hidden[layer][i, pos_answer[i]].float().cpu())
                acc["lastcompl"][layer].append(hidden[layer][i, pos_lastcompl[i]].float().cpu())
            kept_rows.append(r)
            labels.append(r["label"])

        print(f"    {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)} rows", end="\r")
    print()

    labels_t = torch.tensor(labels, dtype=torch.int64)
    out = {}
    for variant in ("answer", "lastcompl"):
        acts = {str(l): torch.stack(acc[variant][l]).unsqueeze(1) for l in range(n_layers)}
        out[variant] = {"acts": acts, "labels": labels_t, "rows": kept_rows}
    return out


def extract_deception_positions(rows: list[dict], tokenizer, model) -> dict:
    n_layers = N_LAYERS
    acc = {
        "firstgen": {l: [] for l in range(n_layers)},
        "lastprompt": {l: [] for l in range(n_layers)},
    }
    kept_rows: list[dict] = []
    labels: list[int] = []
    suffix_cache: dict[str, str] = {}

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        token_id_lists = []
        pos_firstgen = []
        pos_lastprompt = []
        batch_kept = []
        for r in batch:
            followup = r["followup_text"]
            if followup not in suffix_cache:
                suffix_cache[followup] = build_followup_suffix(tokenizer, followup)
            suffix = suffix_cache[followup]

            prefix_text = build_followup_prompt(r, suffix)  # prompt + completion + suffix
            full_text = prefix_text + r["followup_completion"]
            enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
            token_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]

            boundary = len(prefix_text)
            starts = [o[0] for o in offsets]
            assert boundary in starts, (
                f"{r['question_id']}/{r['condition']}/{r['sample_idx']}: boundary {boundary} "
                f"not on a token start"
            )
            firstgen_idx = starts.index(boundary)
            lastprompt_idx = firstgen_idx - 1
            assert lastprompt_idx >= 0
            assert offsets[lastprompt_idx][0] >= len(r["prompt"]) + len(r["completion"]), (
                f"{r['question_id']}: _lastprompt position falls inside turn-1 completion, not the suffix"
            )

            token_id_lists.append(token_ids)
            pos_firstgen.append(firstgen_idx)
            pos_lastprompt.append(lastprompt_idx)
            batch_kept.append(r)

        input_ids, attention_mask = pad_batch(tokenizer, token_id_lists)
        hidden = run_forward_batch(model, input_ids, attention_mask)

        for i, r in enumerate(batch_kept):
            for layer in range(n_layers):
                acc["firstgen"][layer].append(hidden[layer][i, pos_firstgen[i]].float().cpu())
                acc["lastprompt"][layer].append(hidden[layer][i, pos_lastprompt[i]].float().cpu())
            kept_rows.append(r)
            labels.append(r["label"])

        print(f"    {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)} rows", end="\r")
    print()

    labels_t = torch.tensor(labels, dtype=torch.int64)
    out = {}
    for variant in ("firstgen", "lastprompt"):
        acts = {str(l): torch.stack(acc[variant][l]).unsqueeze(1) for l in range(n_layers)}
        out[variant] = {"acts": acts, "labels": labels_t, "rows": kept_rows}
    return out


def save_variant(probe_name: str, split: str, variant_data: dict) -> None:
    out_path = ACT_DIR / f"{probe_name}_{split}.pt"
    torch.save(
        {
            "acts": variant_data["acts"],
            "labels": variant_data["labels"],
            "row_keys": [
                (r["question_id"], r["condition"], r["sample_idx"], r["category"]) for r in variant_data["rows"]
            ],
        },
        out_path,
    )
    print(f"  saved {out_path.name}: {variant_data['labels'].shape[0]} rows")


def maybe_truncate(rows: list[dict], limit: int | None, label_field: str = "label") -> list[dict]:
    """For --dry-run: shuffle (Phase-1 files are class-grouped, not shuffled) then
    take `limit` rows per class so both labels are represented."""
    if limit is None:
        return rows
    import random

    rng = random.Random(0)
    by_class: dict[int, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r[label_field], []).append(r)
    out = []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        out.extend(cls_rows[:limit])
    return out


def main() -> None:
    global ACT_DIR
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="~20 rows/class, writes to activations/dry_run/")
    parser.add_argument("--limit-per-class", type=int, default=20)
    args = parser.parse_args()

    act_dir = (ACT_DIR / "dry_run") if args.dry_run else ACT_DIR
    act_dir.mkdir(parents=True, exist_ok=True)
    limit = args.limit_per_class if args.dry_run else None
    ACT_DIR = act_dir

    model, tokenizer = load_model_and_tokenizer()
    scores_idx = build_scores_index(DOMAIN_DIR)

    for definition in BIAS_DEFINITIONS:
        for split in SPLITS:
            path = PROBE_SETS_DIR / f"bias_{definition}_{split}.jsonl"
            rows = maybe_truncate(load_jsonl(path), limit)
            print(f"\n=== bias_{definition}/{split}: {len(rows)} rows ===")
            result = extract_bias_positions(rows, tokenizer, model, scores_idx)
            save_variant(f"bias_{definition}_answer", split, result["answer"])
            save_variant(f"bias_{definition}_lastcompl", split, result["lastcompl"])

    for definition in DECEPTION_DEFINITIONS:
        for split in SPLITS:
            path = PROBE_SETS_DIR / f"deception_{definition}_{split}.jsonl"
            rows = maybe_truncate(load_jsonl(path), limit)
            print(f"\n=== deception_{definition}/{split}: {len(rows)} rows ===")
            result = extract_deception_positions(rows, tokenizer, model)
            save_variant(f"deception_{definition}_firstgen", split, result["firstgen"])
            save_variant(f"deception_{definition}_lastprompt", split, result["lastprompt"])

    print("\nDone.")


if __name__ == "__main__":
    main()
