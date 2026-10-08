"""Step 2: tokenize + forward pass + slice -> per-dataset activation tensors.

Copy of ../260923_improve_probe_performance/step2_extract_activations.py with
three changes:
  * the turn-1 answer gate reads `row["answer_letter"]` (labels.jsonl) instead
    of 260819's scores.jsonl;
  * only two positions exist: `answer` (bias family: the answer-letter token)
    and `preyesno` (deception: the last token of the turn-2 prompt, where the
    row's P(Yes) and so its `said` label are read; 260923's token before the
    sampled Yes/No whenever the reply starts with Yes/No -- user decision
    2026-10-08, since `said` is no longer the sampled token);
  * every bias-family row also gets the model's next-token log-probabilities of
    the two letters at the token BEFORE the answer letter (`hf_letter_logprobs`,
    [lp_A, lp_B]), from `model.lm_head` applied to the layer-31 residual (the
    extractor's layer 31 is transformers' post-final-norm last hidden state,
    verified once per process against a plain forward pass). That is what the
    HF-vs-vLLM agreement gate reads for the no-CoT arms.

Every position of a dataset is sliced from one forward pass per row. Output is
one .pt per (dataset, position, split); the three deception probes share one
file each and are separated by a `said` mask at step 3.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import spec
from domain_imports import (
    FALLBACK_STAGE,
    PATTERN_ANSWER_COLON,
    PATTERN_ANSWER_IS,
    PATTERN_MARKED_LETTER,
    extract_answer,
    import_followup_generation,
)
from obfuscation_atlas.utils.activations import get_all_residual_acts
from probe_config import POSITION_OF_FAMILY, PipelineConfig
from step1_build_probe_sets import load_jsonl

LM_HEAD_CHECK_MAX_ABS_DIFF = 1e-2
LETTER_IDS: list[int] = [spec.LETTER_TOKEN_IDS[letter] for letter in spec.LETTERS]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def load_model_and_tokenizer(cfg: PipelineConfig):
    print(f"Loading {cfg.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=cfg.dtype, attn_implementation=cfg.attn_implementation).to(cfg.device)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# turn-1 answer span (same 3-stage regex chain as score_completions)
# --------------------------------------------------------------------------- #


def find_answer_span(completion: str):
    """Same 3-stage regex chain as score_completions.extract_answer, but returns
    the matched char span too. Tie-break matches extract_answer exactly: first
    match for the two primary patterns, LAST match for the fallback stage."""
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


# --------------------------------------------------------------------------- #
# per-row position finding (all gates live here)
# --------------------------------------------------------------------------- #


@dataclass
class RowPositions:
    """One row's tokenization plus the token index of each extraction position."""

    token_ids: list[int]
    positions: dict[str, int]
    info: dict = field(default_factory=dict)


def bias_row_positions(row: dict, tokenizer) -> RowPositions:
    """Gates, in order:
      1. the regex chain matches at all,
      2. the local (letter, stage) agrees with `extract_answer` on the same text,
      3. the letter equals the row's stored `answer_letter`,
      4. the sliced token, decoded, contains the matched letter,
      5. the prompt/completion boundary lands on a token start, before the letter.
    `preanswer` (the token before the letter) is where the letter logprobs are read.
    """
    prompt, completion = row["prompt"], row["completion"]
    rid = f"{row['question_id']}/{row['condition']}/{row['sample_idx']}"
    letter, stage, c_start, _c_end = find_answer_span(completion)
    assert letter is not None, f"{rid}: no regex match"
    gate_letter, gate_stage = extract_answer(completion)
    assert (letter, stage) == (gate_letter, gate_stage), f"{rid}: local (letter,stage)={(letter, stage)} != extract_answer {(gate_letter, gate_stage)}"
    assert letter == row["answer_letter"], f"{rid}: regex letter {letter!r} != stored answer_letter {row['answer_letter']!r}"

    full_text = prompt + completion
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    answer_idx = char_to_token_index(offsets, len(prompt) + c_start)
    decoded = tokenizer.decode([token_ids[answer_idx]])
    assert letter in decoded, f"{rid}: decoded token {decoded!r} does not contain letter {letter!r}"

    starts = [o[0] for o in offsets]
    assert len(prompt) in starts, f"{rid}: prompt/completion boundary {len(prompt)} not on a token start"
    firstgen_idx = starts.index(len(prompt))
    assert firstgen_idx <= answer_idx, f"{rid}: answer token before the first generated token"
    assert answer_idx >= 1, f"{rid}: answer token is the first token"

    return RowPositions(
        token_ids=token_ids,
        positions={"answer": answer_idx, "preanswer": answer_idx - 1},
        info={"letter": letter, "stage": stage, "decoded_answer_token": decoded, "firstgen_idx": firstgen_idx},
    )



def deception_row_positions(row: dict, tokenizer, suffix_cache: dict[str, str], followup_generation) -> RowPositions:
    """Rebuilds the turn-2 prompt byte-for-byte via `build_followup_suffix` /
    `build_followup_prompt` (the text score_pyes.py reads P(Yes) from) and
    probes at its last token, where the row's `said` label is read. The sampled
    reply is not part of the input; `ans_pos` (where the sampled Yes/No fell,
    or None) is kept for the report only.

    Gates:
      1. the prompt's tokens end with the follow-up template's own tokens,
      2. `preyesno` falls after the turn-1 completion.
    """
    followup = row["followup_text"]
    if followup not in suffix_cache:
        suffix_cache[followup] = followup_generation.build_followup_suffix(tokenizer, followup)
    suffix = suffix_cache[followup]

    rid = f"{row['question_id']}/{row['condition']}/{row['sample_idx']}"
    prompt_text = followup_generation.build_followup_prompt(row, suffix)  # prompt + completion + suffix
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    assert token_ids[-len(suffix_ids):] == suffix_ids, f"{rid}: the turn-2 prompt does not end with the follow-up template's tokens"

    preyesno_idx = len(token_ids) - 1
    assert enc["offset_mapping"][preyesno_idx][0] >= len(row["prompt"]) + len(row["completion"]), f"{rid}: preyesno position falls inside the turn-1 completion, not the suffix"

    return RowPositions(
        token_ids=token_ids,
        positions={"preyesno": preyesno_idx},
        info={"ans_pos": row["ans_pos"], "said": row["said"]},
    )


# --------------------------------------------------------------------------- #
# batching + forward
# --------------------------------------------------------------------------- #


def pad_batch(tokenizer, token_id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in token_id_lists)
    input_ids = torch.full((len(token_id_lists), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(token_id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(token_id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


def run_forward_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, cfg: PipelineConfig) -> dict[int, torch.Tensor]:
    """Returns {layer_idx: (batch, seq, hidden_dim)} for layers 0..cfg.n_layers-1."""
    with torch.no_grad():
        return get_all_residual_acts(
            model,
            input_ids.to(cfg.device),
            attention_mask.to(cfg.device),
            batch_size=input_ids.shape[0],  # one internal batch; we chunk outside
            only_return_layers=list(range(cfg.n_layers)),
        )


def letter_logprobs_from_last_hidden(model, last_hidden: torch.Tensor) -> torch.Tensor:
    """(n, 2) float32 log-probabilities of the A and B tokens, from the layer-31
    residual (post final norm) at the token before the letter: `lm_head`, then
    log-softmax over the full vocabulary."""
    with torch.no_grad():
        logits = model.lm_head(last_hidden).float()
        return torch.log_softmax(logits, dim=-1)[:, LETTER_IDS].cpu()


def check_lm_head_matches_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, hidden: dict[int, torch.Tensor], positions: list[int], cfg: PipelineConfig) -> float:
    """Once per process: `lm_head` on the extractor's layer-31 hidden state at
    `positions` must equal a plain `model(...)` call's logits there, which
    holds only if that layer is the post-final-norm last hidden state. Returns
    the max abs difference (bf16 logits); asserts it is below
    LM_HEAD_CHECK_MAX_ABS_DIFF. Runs on the first two rows to keep the full
    vocabulary logits small."""
    n = min(2, input_ids.shape[0])
    rows = torch.arange(n)
    pos = torch.tensor(positions[:n])
    with torch.no_grad():
        plain_logits = model(input_ids[:n].to(cfg.device), attention_mask=attention_mask[:n].to(cfg.device)).logits[rows, pos]
        from_hidden = model.lm_head(hidden[cfg.n_layers - 1][rows, pos])
    max_abs_diff = float((plain_logits.float() - from_hidden.float()).abs().max())
    assert max_abs_diff < LM_HEAD_CHECK_MAX_ABS_DIFF, f"lm_head on layer {cfg.n_layers - 1} differs from model(...).logits by {max_abs_diff}"
    print(f"  lm_head check: max |logits - lm_head(layer {cfg.n_layers - 1})| = {max_abs_diff:.3g} on {n} rows")
    return max_abs_diff


def extract_positions(
    rows: list[dict],
    model,
    tokenizer,
    cfg: PipelineConfig,
    position_fn: Callable[[dict], RowPositions],
    position: str,
    *,
    letter_logprobs: bool,
    lm_head_check: dict,
) -> dict:
    """One forward pass per row; slice `position` out of the hidden states.

    Returns {"acts": {"0": (N,1,H), ...}, "labels": (N,), "rows": [...],
    "positions": [int, ...], "infos": [...], "hf_letter_logprobs": [[lp_A, lp_B], ...] | None}.
    With `letter_logprobs`, the row's `preanswer` position is read through
    `lm_head`; `lm_head_check` (shared across calls) records the one-time
    check against a plain forward pass.
    """
    acc = {layer: [] for layer in range(cfg.n_layers)}
    kept_rows: list[dict] = []
    labels: list[int] = []
    kept_positions: list[int] = []
    kept_infos: list[dict] = []
    kept_letter_logprobs: list[list[float]] = []

    for batch_start in range(0, len(rows), cfg.batch_size):
        batch = rows[batch_start : batch_start + cfg.batch_size]
        row_positions = [position_fn(r) for r in batch]
        input_ids, attention_mask = pad_batch(tokenizer, [rp.token_ids for rp in row_positions])
        hidden = run_forward_batch(model, input_ids, attention_mask, cfg)

        if letter_logprobs:
            preanswer = [rp.positions["preanswer"] for rp in row_positions]
            if "max_abs_diff" not in lm_head_check:
                lm_head_check["max_abs_diff"] = check_lm_head_matches_logits(model, input_ids, attention_mask, hidden, preanswer, cfg)
            last_hidden = hidden[cfg.n_layers - 1][torch.arange(len(batch)), torch.tensor(preanswer)]
            kept_letter_logprobs.extend(letter_logprobs_from_last_hidden(model, last_hidden).tolist())

        for i, (r, rp) in enumerate(zip(batch, row_positions)):
            pos = rp.positions[position]
            for layer in range(cfg.n_layers):
                acc[layer].append(hidden[layer][i, pos].float().cpu())
            kept_positions.append(pos)
            kept_rows.append(r)
            labels.append(r["label"])
            kept_infos.append(rp.info)

        del hidden
        print(f"    {min(batch_start + cfg.batch_size, len(rows))}/{len(rows)} rows", end="\r")
    print()

    return {
        "acts": {str(layer): torch.stack(acc[layer]).unsqueeze(1) for layer in range(cfg.n_layers)},
        "labels": torch.tensor(labels, dtype=torch.int64),
        "rows": kept_rows,
        "positions": kept_positions,
        "infos": kept_infos,
        "hf_letter_logprobs": kept_letter_logprobs if letter_logprobs else None,
    }


def summarize_yesno_positions(infos: list[dict]) -> dict:
    """Where the sampled Yes/No fell in the reply, over the rows just extracted
    (report only: the probe reads the last prompt token whatever the reply did)."""
    positions = [i["ans_pos"] for i in infos]
    n_first = sum(1 for p in positions if p == 0)
    return {
        "n": len(positions),
        "n_yesno_is_first_generated": n_first,
        "frac_yesno_is_first_generated": n_first / len(positions),
        "n_no_sampled_yesno": sum(1 for p in positions if p is None),
        "ans_pos_counts": dict(Counter(str(p) for p in positions)),
    }


def extract_bias_positions(rows: list[dict], model, tokenizer, cfg: PipelineConfig, lm_head_check: dict) -> dict:
    return extract_positions(
        rows, model, tokenizer, cfg, position_fn=lambda r: bias_row_positions(r, tokenizer), position="answer", letter_logprobs=True, lm_head_check=lm_head_check
    )


def extract_deception_positions(rows: list[dict], model, tokenizer, cfg: PipelineConfig, lm_head_check: dict) -> dict:
    suffix_cache: dict[str, str] = {}
    followup_generation = import_followup_generation()
    return extract_positions(
        rows,
        model,
        tokenizer,
        cfg,
        position_fn=lambda r: deception_row_positions(r, tokenizer, suffix_cache, followup_generation),
        position="preyesno",
        letter_logprobs=False,
        lm_head_check=lm_head_check,
    )


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #


def probe_set_path(cfg: PipelineConfig, family: str, definition: str, split: str) -> Path:
    return cfg.probe_sets_dir / f"{family}_{definition}_{split}.jsonl"


def maybe_truncate(rows: list[dict], limit: int | None) -> list[dict]:
    """Dry runs: shuffle (step-1 files are class-grouped) then take `limit` rows
    per `subtype`, so a deception dry run covers both said-Yes and said-No."""
    if limit is None:
        return rows
    import random

    rng = random.Random(0)
    by_class: dict = {}
    for r in rows:
        by_class.setdefault(r["subtype"], []).append(r)
    out: list[dict] = []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        out.extend(cls_rows[:limit])
    return out


def activation_path(cfg: PipelineConfig, dataset: str, position: str, split: str) -> Path:
    """One file per (dataset, position, split) -- NOT per probe."""
    return cfg.act_dir / f"{dataset}_{position}_{split}.pt"


def save_result(dataset: str, position: str, split: str, result: dict, cfg: PipelineConfig) -> Path:
    out_path = activation_path(cfg, dataset, position, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    torch.save(
        {
            "spec_hash": spec.spec_hash(),
            "acts": {k: v.to(cfg.act_save_dtype) for k, v in result["acts"].items()},
            "labels": result["labels"],
            "row_keys": [(r["question_id"], r["condition"], r["sample_idx"], r["category"]) for r in rows],
            "said": [r.get("said") for r in rows],
            "subtype": [r["subtype"] for r in rows],
            "positions": result["positions"],
            "ans_pos": [i.get("ans_pos") for i in result["infos"]],
            "hf_letter_logprobs": result["hf_letter_logprobs"],
        },
        out_path,
    )
    print(f"  saved {out_path.name}: {result['labels'].shape[0]} rows")
    return out_path


def extract_all(cfg: PipelineConfig, model, tokenizer, lm_head_check: dict) -> dict:
    """Loop cfg.datasets x cfg.definitions x cfg.splits, writing one .pt each.
    Returns per-file counts (and the self-report position stats for deception)."""
    cfg.act_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for family, definition in cfg.dataset_keys():
        dataset = cfg.dataset_name(family, definition)
        position = POSITION_OF_FAMILY[family]
        for split in cfg.splits:
            rows = maybe_truncate(load_jsonl(probe_set_path(cfg, family, definition, split)), cfg.limit_per_class)
            print(f"\n=== {dataset}/{split}: {len(rows)} rows ===")
            if family in ("bias", "bias_followed"):
                result = extract_bias_positions(rows, model, tokenizer, cfg, lm_head_check)
            else:
                result = extract_deception_positions(rows, model, tokenizer, cfg, lm_head_check)
                stats = summarize_yesno_positions(result["infos"])
                results[f"{dataset}_{split}_yesno_position"] = stats
                print(
                    f"  self-report token was the first generated token on {stats['n_yesno_is_first_generated']}/{stats['n']} rows "
                    f"({stats['frac_yesno_is_first_generated']:.4f}); ans_pos counts {stats['ans_pos_counts']}"
                )
            save_result(dataset, position, split, result, cfg)
            results[f"{dataset}_{position}_{split}"] = {
                "n": int(result["labels"].shape[0]),
                "n_pos": int(result["labels"].sum()),
                "d_model": result["acts"]["0"].shape[-1],
            }
            del result
    print("\nDone.")
    return results
