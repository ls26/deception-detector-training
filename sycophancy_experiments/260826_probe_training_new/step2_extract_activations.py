"""Step 2: tokenize + forward pass + slice -> per-probe activation tensors.

Importable port of ../260825_probe_training/02_extract_activations.py. The
tokenization, the position finding, and every gate are unchanged. What changed:

  * per-row position finding is factored into `bias_row_positions` /
    `deception_row_positions`, which return the token ids and both positions for
    one row. The gates live inside them, so calling one on a single row in a
    notebook cell exercises the full gate chain on that row.
  * `load_model_and_tokenizer(cfg)` is called once by the notebook and the model
    is reused by step 3 (260825 loaded 16 GB twice, once per script).
  * `cfg.limit_per_class` replaces the `--dry-run` flag and its `global ACT_DIR`
    reassignment.
  * the extractors keep returning the in-memory acts so a cell can inspect them.

Positions, per dataset family:
  bias:      A = regex-matched answer-letter token   (_answer)
             B = last token of the turn-1 completion  (_lastcompl)
  deception: A = first token generated in turn 2      (_firstgen)
             B = last token of the turn-2 prompt, one before A (_lastprompt)

Both positions of a dataset are sliced from the SAME cached hidden states -- one
forward pass per row, not two. Output is one .pt per (dataset, position, split),
i.e. 4 datasets x 2 positions x 2 splits = 16 files. Note these are DATASETS,
not probes: the three deception probes (all rows / said-Yes / said-No) share one
file each and are separated by a mask at step 3, so no rows are extracted twice.
Every row's `said` and `subtype` are saved alongside for exactly that.

Bypasses the shared package's text pipeline (`obfuscate_over` cannot express
either position) and writes raw per-layer (N, 1, hidden_dim) tensors.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from domain_imports import (
    FALLBACK_STAGE,
    PATTERN_ANSWER_COLON,
    PATTERN_ANSWER_IS,
    PATTERN_MARKED_LETTER,
    build_followup_prompt,
    build_followup_suffix,
    extract_answer,
)
from obfuscation_atlas.utils.activations import get_all_residual_acts
from probe_config import POSITIONS, PipelineConfig
from step1_build_probe_sets import build_scores_index, load_jsonl


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def load_model_and_tokenizer(cfg: PipelineConfig):
    print(f"Loading {cfg.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=cfg.dtype, attn_implementation=cfg.attn_implementation
    ).to(cfg.device)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# turn-1 answer span (same 3-stage regex chain as score_completions)
# --------------------------------------------------------------------------- #


def find_answer_span(completion: str):
    """Same 3-stage regex chain as score_completions.extract_answer, but returns
    the matched char span too (extract_answer only returns the letter+stage, using
    findall which discards match positions for the fallback stage). Tie-break must
    match extract_answer exactly: first match for the two primary patterns, LAST
    match for the fallback (marked-letter) stage."""
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


# `build_scores_index` lives in step 1 (it is source-file loading, and step 1
# needs it for `followed_cue`); re-exported here since step 2 and step 3 both
# refer to it as s2.build_scores_index. Used here as the turn-1 regression gate:
# position A must agree with those already-published, already-gated labels.


# --------------------------------------------------------------------------- #
# per-row position finding (all gates live here)
# --------------------------------------------------------------------------- #


@dataclass
class RowPositions:
    """One row's tokenization plus the token index of each extraction position."""

    token_ids: list[int]
    positions: dict[str, int]
    info: dict = field(default_factory=dict)


def bias_row_positions(row: dict, tokenizer, scores_idx: dict[tuple, dict]) -> RowPositions:
    """Gates, in order:
      1. the regex chain matches at all,
      2. the local (letter, stage) agrees with `extract_answer` on the same text,
      3. and with the already-published `preds`/`stages` in scores.jsonl,
      4. the sliced token, decoded, contains the matched letter.
    """
    prompt, completion = row["prompt"], row["completion"]
    letter, stage, c_start, _c_end = find_answer_span(completion)
    assert letter is not None, (
        f"no regex match for row {row['question_id']}/{row['condition']}/{row['sample_idx']}"
    )

    skey = (row["question_id"], row["condition"])
    srow = scores_idx.get(skey)
    assert srow is not None, f"no scores.jsonl row for {skey}"
    gate_letter, gate_stage = extract_answer(completion)
    assert (letter, stage) == (gate_letter, gate_stage), (
        f"{skey}: local (letter,stage)={(letter, stage)} != extract_answer {(gate_letter, gate_stage)}"
    )
    published_pred = srow["preds"][row["sample_idx"]]
    published_stage = srow["stages"][row["sample_idx"]]
    assert (letter, stage) == (published_pred, published_stage), (
        f"{skey}[{row['sample_idx']}]: local (letter,stage)={(letter, stage)} != "
        f"published ({published_pred}, {published_stage})"
    )

    full_text = prompt + completion
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    answer_idx = char_to_token_index(offsets, len(prompt) + c_start)
    decoded = tokenizer.decode([token_ids[answer_idx]])
    assert letter in decoded, f"{skey}: decoded token {decoded!r} does not contain letter {letter!r}"

    return RowPositions(
        token_ids=token_ids,
        positions={"answer": answer_idx, "lastcompl": len(token_ids) - 1},
        info={"letter": letter, "stage": stage, "decoded_answer_token": decoded},
    )


#: Broad Yes/No surface forms, matching generate_followups.YES_BROAD/NO_BROAD --
#: the same set `answer_position` matched on to produce the stored `ans_pos`.
YES_FORMS = {"Yes", "YES", "yes"}
NO_FORMS = {"No", "NO", "no"}


def deception_row_positions(row: dict, tokenizer, suffix_cache: dict[str, str]) -> RowPositions:
    """Reconstructs the actual two-turn rollout byte-for-byte via
    `build_followup_suffix`/`build_followup_prompt`, then probes AT the sampled
    Yes/No token rather than at the first generated token.

    The self-report is the first generated token only ~99.6% of the time (260817),
    so on the remainder 260825's `_firstgen` probed a preamble token (a `**`, a
    newline) and not the self-report at all. `ans_pos` -- already stored in
    followups.jsonl by `answer_position()`, the index of the first SAMPLED
    Yes/No token among the generated tokens -- gives the real position.

    Gates:
      1. the turn-2 boundary lands exactly on a token start,
      2. `ans_pos` exists (a pool row always has a parsed Yes/No),
      3. the located token id equals the stored generated id at `ans_pos`,
      4. the token decodes to a Yes/No form whose polarity matches `said`,
      5. `preyesno` is at or after the last token of the turn-2 prompt, i.e. it
         never falls back inside the turn-1 completion.
    """
    followup = row["followup_text"]
    if followup not in suffix_cache:
        suffix_cache[followup] = build_followup_suffix(tokenizer, followup)
    suffix = suffix_cache[followup]

    rid = f"{row['question_id']}/{row['condition']}/{row['sample_idx']}"
    prefix_text = build_followup_prompt(row, suffix)  # prompt + completion + suffix
    full_text = prefix_text + row["followup_completion"]
    enc = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    boundary = len(prefix_text)
    starts = [o[0] for o in offsets]
    assert boundary in starts, f"{rid}: boundary {boundary} not on a token start"
    firstgen_idx = starts.index(boundary)

    ans_pos = row.get("ans_pos")
    assert ans_pos is not None, (
        f"{rid}: no sampled Yes/No token (ans_pos is None) -- such rows carry "
        f"said=None and should have been excluded from the pool as unparsed_turn2"
    )

    yesno_idx = firstgen_idx + ans_pos
    assert yesno_idx < len(token_ids), (
        f"{rid}: ans_pos={ans_pos} lands past the end of the retokenized text"
    )

    # followups.jsonl stores only `completion.token_ids[:8]` (generate_followups.py
    # :720), while `ans_pos` was computed over the full generated sequence -- so
    # this gate applies to the rows whose self-report landed inside that window,
    # which is all but a handful. The decode gate below covers every row.
    generated_ids = row.get("first_token_ids") or []
    id_gate_applied = ans_pos < len(generated_ids)
    if id_gate_applied:
        assert token_ids[yesno_idx] == generated_ids[ans_pos], (
            f"{rid}: token at ans_pos={ans_pos} is {token_ids[yesno_idx]} on retokenization "
            f"but {generated_ids[ans_pos]} as generated"
        )

    decoded = tokenizer.decode([token_ids[yesno_idx]]).strip()
    said = row["said"]
    forms = YES_FORMS if said == "Yes" else NO_FORMS
    assert decoded in forms, (
        f"{rid}: token at the self-report position decodes to {decoded!r}, "
        f"which is not a {said!r} form"
    )

    preyesno_idx = yesno_idx - 1
    assert preyesno_idx >= firstgen_idx - 1, f"{rid}: preyesno fell before the turn-2 prompt"
    assert offsets[preyesno_idx][0] >= len(row["prompt"]) + len(row["completion"]), (
        f"{rid}: preyesno position falls inside the turn-1 completion, not the suffix"
    )

    return RowPositions(
        token_ids=token_ids,
        positions={"yesno": yesno_idx, "preyesno": preyesno_idx},
        info={
            "suffix": suffix,
            "boundary_char": boundary,
            "firstgen_idx": firstgen_idx,
            "ans_pos": ans_pos,
            "yesno_is_first_generated": ans_pos == 0,
            "decoded_yesno": decoded,
            "said": said,
            "id_gate_applied": id_gate_applied,
        },
    )


# --------------------------------------------------------------------------- #
# batching + forward
# --------------------------------------------------------------------------- #


def pad_batch(tokenizer, token_id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in token_id_lists)
    input_ids = torch.full(
        (len(token_id_lists), max_len), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(token_id_lists), max_len), dtype=torch.long)
    for i, ids in enumerate(token_id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    return input_ids, attention_mask


def run_forward_batch(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, cfg: PipelineConfig
) -> dict[int, torch.Tensor]:
    """Returns {layer_idx: (batch, seq, hidden_dim)} for layers 0..cfg.n_layers-1."""
    with torch.no_grad():
        return get_all_residual_acts(
            model,
            input_ids.to(cfg.device),
            attention_mask.to(cfg.device),
            batch_size=input_ids.shape[0],  # one internal batch; we chunk outside
            only_return_layers=list(range(cfg.n_layers)),
        )


def extract_positions(
    rows: list[dict],
    model,
    tokenizer,
    cfg: PipelineConfig,
    position_fn: Callable[[dict], RowPositions],
    variants: tuple[str, ...],
    verbose: bool = True,
) -> dict[str, dict]:
    """One forward pass per row; slice every variant out of the same hidden states.

    Returns {variant: {"acts": {"0": (N,1,H), ...}, "labels": (N,), "rows": [...],
                       "positions": [int, ...]}}
    """
    acc = {v: {layer: [] for layer in range(cfg.n_layers)} for v in variants}
    kept_rows: list[dict] = []
    labels: list[int] = []
    kept_positions: dict[str, list[int]] = {v: [] for v in variants}
    kept_infos: list[dict] = []

    for batch_start in range(0, len(rows), cfg.batch_size):
        batch = rows[batch_start : batch_start + cfg.batch_size]
        token_id_lists: list[list[int]] = []
        batch_positions: dict[str, list[int]] = {v: [] for v in variants}
        batch_kept: list[dict] = []
        batch_infos: list[dict] = []

        for r in batch:
            rp = position_fn(r)
            token_id_lists.append(rp.token_ids)
            for v in variants:
                batch_positions[v].append(rp.positions[v])
            batch_kept.append(r)
            batch_infos.append(rp.info)

        input_ids, attention_mask = pad_batch(tokenizer, token_id_lists)
        hidden = run_forward_batch(model, input_ids, attention_mask, cfg)

        for i, r in enumerate(batch_kept):
            for v in variants:
                pos = batch_positions[v][i]
                for layer in range(cfg.n_layers):
                    acc[v][layer].append(hidden[layer][i, pos].float().cpu())
                kept_positions[v].append(pos)
            kept_rows.append(r)
            labels.append(r["label"])
            kept_infos.append(batch_infos[i])

        del hidden
        if verbose:
            print(f"    {min(batch_start + cfg.batch_size, len(rows))}/{len(rows)} rows", end="\r")
    if verbose:
        print()

    labels_t = torch.tensor(labels, dtype=torch.int64)
    out = {}
    for v in variants:
        acts = {
            str(layer): torch.stack(acc[v][layer]).unsqueeze(1) for layer in range(cfg.n_layers)
        }
        out[v] = {
            "acts": acts,
            "labels": labels_t,
            "rows": kept_rows,
            "positions": kept_positions[v],
            "infos": kept_infos,
        }
    return out


def summarize_yesno_positions(infos: list[dict]) -> dict:
    """Where the self-report token actually was, over the rows just extracted.

    This is the verification the `yesno` position exists for: if every row has
    ans_pos == 0 then probing the first generated token was harmless, and any
    row with ans_pos > 0 is one 260825 probed at the wrong token.
    """
    positions = [i["ans_pos"] for i in infos if "ans_pos" in i]
    if not positions:
        return {}
    n_first = sum(1 for p in positions if p == 0)
    return {
        "n": len(positions),
        "n_yesno_is_first_generated": n_first,
        "frac_yesno_is_first_generated": n_first / len(positions),
        "ans_pos_counts": dict(sorted(Counter(positions).items())),
        "max_ans_pos": max(positions),
        "n_id_gate_applied": sum(1 for i in infos if i.get("id_gate_applied")),
        "decoded_forms": dict(Counter(i["decoded_yesno"] for i in infos if "decoded_yesno" in i)),
    }


def extract_bias_positions(
    rows: list[dict], model, tokenizer, scores_idx: dict[tuple, dict], cfg: PipelineConfig, **kwargs
) -> dict[str, dict]:
    return extract_positions(
        rows,
        model,
        tokenizer,
        cfg,
        position_fn=lambda r: bias_row_positions(r, tokenizer, scores_idx),
        variants=POSITIONS["bias"],
        **kwargs,
    )


def extract_deception_positions(
    rows: list[dict], model, tokenizer, cfg: PipelineConfig, **kwargs
) -> dict[str, dict]:
    suffix_cache: dict[str, str] = {}
    return extract_positions(
        rows,
        model,
        tokenizer,
        cfg,
        position_fn=lambda r: deception_row_positions(r, tokenizer, suffix_cache),
        variants=POSITIONS["deception"],
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #


def probe_set_path(cfg: PipelineConfig, family: str, definition: str, split: str) -> Path:
    return cfg.probe_sets_dir / f"{family}_{definition}_{split}.jsonl"


def maybe_truncate(rows: list[dict], limit: int | None, label_field: str | None = None) -> list[dict]:
    """Shuffle (step-1 files are class-grouped, not shuffled) then take `limit` rows
    per class so every class is represented. Used for dry runs.

    Defaults to stratifying on `subtype` where present, so a deception dry run
    covers all four subtypes -- and therefore both said-Yes and said-No -- rather
    than possibly drawing its 20 deceptive rows entirely from one stratum.
    """
    if limit is None:
        return rows
    import random

    if label_field is None:
        label_field = "subtype" if rows and "subtype" in rows[0] else "label"
    rng = random.Random(0)
    by_class: dict = {}
    for r in rows:
        by_class.setdefault(r[label_field], []).append(r)
    out: list[dict] = []
    for cls_rows in by_class.values():
        rng.shuffle(cls_rows)
        out.extend(cls_rows[:limit])
    return out


def activation_path(cfg: PipelineConfig, dataset: str, position: str, split: str) -> Path:
    """One file per (dataset, position, split) -- NOT per probe. The three
    deception probes read the same file and mask it on `said`."""
    return cfg.act_dir / f"{dataset}_{position}_{split}.pt"


def save_variant(
    dataset: str, position: str, split: str, variant_data: dict, cfg: PipelineConfig
) -> Path:
    out_path = activation_path(cfg, dataset, position, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    acts = {k: v.to(cfg.act_save_dtype) for k, v in variant_data["acts"].items()}
    rows = variant_data["rows"]
    torch.save(
        {
            "acts": acts,
            "labels": variant_data["labels"],
            "row_keys": [
                (r["question_id"], r["condition"], r["sample_idx"], r["category"]) for r in rows
            ],
            # `said` is what step 3 masks on for the stratified deception probes;
            # `subtype` is carried for analysis. Both are None for bias rows.
            "said": [r.get("said") for r in rows],
            "subtype": [r.get("subtype") for r in rows],
            "positions": variant_data.get("positions"),
            # where the self-report token sat among the generated tokens (None
            # for bias rows) -- kept so a later analysis can split on it
            "ans_pos": [i.get("ans_pos") for i in variant_data.get("infos", [])] or None,
        },
        out_path,
    )
    print(f"  saved {out_path.name}: {variant_data['labels'].shape[0]} rows")
    return out_path


def extract_all(
    cfg: PipelineConfig,
    model,
    tokenizer,
    *,
    scores_idx: dict[tuple, dict] | None = None,
    return_acts: bool = False,
    write: bool = True,
) -> dict:
    """Loop cfg.probe_types x cfg.definitions x cfg.splits.

    `return_acts=True` hands back the in-memory results -- 17 GB for a full run,
    so leave it off unless `cfg.limit_per_class` is set or you are doing one split.
    """
    needs_scores = any(f in cfg.datasets for f in ("bias", "bias_followed"))
    if scores_idx is None and needs_scores:
        scores_idx = build_scores_index(cfg)
    if write:
        cfg.act_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for family, definition in cfg.dataset_keys():
        dataset = cfg.dataset_name(family, definition)
        for split in cfg.splits:
            rows = maybe_truncate(
                load_jsonl(probe_set_path(cfg, family, definition, split)),
                cfg.limit_per_class,
            )
            print(f"\n=== {dataset}/{split}: {len(rows)} rows ===")
            if family in ("bias", "bias_followed"):
                result = extract_bias_positions(rows, model, tokenizer, scores_idx, cfg)
            else:
                result = extract_deception_positions(rows, model, tokenizer, cfg)
                stats = summarize_yesno_positions(result[POSITIONS[family][0]]["infos"])
                results[f"{dataset}_{split}_yesno_position"] = stats
                print(
                    f"  self-report token was the first generated token on "
                    f"{stats['n_yesno_is_first_generated']}/{stats['n']} rows "
                    f"({stats['frac_yesno_is_first_generated']:.4f}); "
                    f"ans_pos counts {stats['ans_pos_counts']}"
                )

            for position in POSITIONS[family]:
                if write:
                    save_variant(dataset, position, split, result[position], cfg)
                key = f"{dataset}_{position}_{split}"
                if return_acts:
                    results[key] = result[position]
                else:
                    results[key] = {
                        "n": int(result[position]["labels"].shape[0]),
                        "n_pos": int(result[position]["labels"].sum()),
                        "d_model": result[position]["acts"]["0"].shape[-1],
                    }
            del result
    print("\nDone.")
    return results


def summarize(results: dict) -> str:
    return json.dumps(
        {k: v for k, v in results.items() if "acts" not in v},
        indent=2,
        default=str,
    )
