"""A GRPO run's probe scored on a checkpoint (the base model, or base + a LoRA adapter) at chosen tokens, as the
trainer scores it: one forward pass per sequence (HF transformers, bf16, right padding), and per probe the mean over
its layers of the sigmoid of its logit on that layer's residual stream (hidden_states[layer + 1]) in float32.

The sequences and the token read in each come from one of two places:
  response_read_positions  a checkpoint's own answers: the prompt's token ids, the answer's generated token ids, and
                           the token the trainer reads (its build_obfuscation_response_dataset: the answer cut to the
                           length limit, the end-of-turn token appended to an answer that has no stop token, and the
                           read token index, or else the last token)
  text_read_positions      fixed texts (the reference sets, as the probe was trained on them): the text tokenized,
                           and the token that holds a given character
"""

from __future__ import annotations

import time
from pathlib import Path


def load_checkpoint(model_name: str, adapter: Path | None):
    """The model on the GPU in bf16 (+ PeftModel for an adapter, adapter on) and its tokenizer (right padding)."""
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


def load_probe(probe_dir: Path, layers: list[int]) -> dict[int, object]:
    """{layer: LinearProbe} on the GPU for `layers` of a saved probe (the trainer's save layout,
    detector_<subdir>__linear-probe_last-token-generation/layer_<n>/); the folder must hold each of them."""
    from obfuscation_atlas.detectors.train import load_model

    probes = load_model(str(probe_dir))
    assert set(layers) <= set(probes), (layers, sorted(probes))
    return {layer: probes[layer].to("cuda").eval() for layer in layers}


def set_trainer_eot(tokenizer, eot_token: str) -> None:
    """Give the tokenizer the end-of-turn token the trainer gives it (afterburner's GRPOTrainer: grpo.tokenizer.eot_token)."""
    tokenizer.eot_token = eot_token
    tokenizer.eot_token_id = tokenizer.get_vocab()[eot_token]


def trainer_completion(tokenizer, response_ids: list[int]) -> str:
    """The text the trainer's reward reads from an answer's ids: decoded with special tokens kept, end-of-turn token
    removed (needs set_trainer_eot)."""
    return tokenizer.decode(response_ids).removesuffix(tokenizer.eot_token)


def response_read_positions(
    tokenizer, prompt_ids: list[list[int]], response_ids: list[list[int]], read_token_index: list[int | None],
    max_total_length: int, stop_token_ids: list[int],
) -> tuple[list[list[int]], list[int]]:
    """Per answer: the sequence the trainer's probe pass sees (prompt + answer as the trainer builds it) and the
    position of the token its detector reads (the last token of completion_with_follow_up_mask). read_token_index is
    the answer token the task reads (the trainer's answer_token_index / code_end_token_index on the generated ids),
    None for the last token. Needs set_trainer_eot; no follow-up prompt (the runs this reads have none)."""
    from obfuscation_atlas.training.grpo_trainer import build_obfuscation_response_dataset

    dataset = build_obfuscation_response_dataset(
        tokenizer=tokenizer, prompt_indices=list(range(len(prompt_ids))), prompt_ids=prompt_ids,
        response_ids=response_ids, max_total_length=max_total_length, trigger_present=[False] * len(prompt_ids),
        stop_token_ids=stop_token_ids, read_token_index=read_token_index,
    )
    positions = [max(i for i, kept in enumerate(mask) if kept) for mask in dataset["completion_with_follow_up_mask"]]
    return dataset["input_ids"], positions


def token_index_at(offsets: list[tuple[int, int]], char_index: int) -> int:
    """The token whose character span holds char_index (261007's char_to_token_index)."""
    return next(i for i, (start, end) in enumerate(offsets) if start <= char_index < end)


def text_read_positions(tokenizer, texts: list[str], read_chars: list[int]) -> tuple[list[list[int]], list[int]]:
    """Per text: its token ids (no special tokens added) and the position of the token that holds read_chars."""
    encodings = [tokenizer(text, add_special_tokens=False, return_offsets_mapping=True) for text in texts]
    positions = [token_index_at(encoding["offset_mapping"], char) for encoding, char in zip(encodings, read_chars, strict=True)]
    return [encoding["input_ids"] for encoding in encodings], positions


def score_at_positions(
    model, sequences: list[list[int]], positions: list[int], probes: dict[str, dict], batch_size: int, pad_token_id: int,
) -> list[dict[str, float]]:
    """Per sequence and probe ({name: {layer: probe}}): the mean over the probe's layers (in sorted order) of the sigmoid
    of its logit at the position, on hidden_states[layer + 1] in float32. Batches of sequences sorted by length."""
    import torch

    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))  # short to long
    results: list[dict[str, float] | None] = [None] * len(sequences)
    started = time.time()
    for batch_start in range(0, len(sequences), batch_size):
        batch_indices = order[batch_start : batch_start + batch_size]
        max_len = max(len(sequences[i]) for i in batch_indices)
        input_ids = torch.full((len(batch_indices), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long)
        for row, i in enumerate(batch_indices):
            input_ids[row, : len(sequences[i])] = torch.tensor(sequences[i])
            attention_mask[row, : len(sequences[i])] = 1
        batch_rows = torch.arange(len(batch_indices))
        read_rows = torch.tensor([positions[i] for i in batch_indices])
        with torch.no_grad():
            # logits_to_keep=1: the output head only at the last position (its scores are not used)
            outputs = model(input_ids.cuda(), attention_mask=attention_mask.cuda(), output_hidden_states=True, logits_to_keep=1)
            probe_scores = {
                name: torch.stack([
                    torch.sigmoid(layer_probes[layer](outputs.hidden_states[layer + 1][batch_rows, read_rows].float()[:, None, :])[:, 0, 0])
                    for layer in sorted(layer_probes)
                ]).mean(dim=0)
                for name, layer_probes in probes.items()
            }
        for row, i in enumerate(batch_indices):
            results[i] = {name: float(scores[row]) for name, scores in probe_scores.items()}
        del outputs
        done = batch_start + len(batch_indices)
        print(f"  {done}/{len(sequences)} sequences, {time.time() - started:.0f}s", flush=True)
    return results  # type: ignore[return-value]
