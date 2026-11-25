"""A text-only copy of a Qwen3.5 post-trained (image-text) checkpoint: the language model's weights under the
text-only class's names, its text config (architecture Qwen3_5ForCausalLM), the tokenizer and the chat template.

    python make_text_only_checkpoint.py Qwen/Qwen3.5-4B /workspace/hf_cache/text_only/Qwen3.5-4B

Why: the trainer trains transformers' AutoModelForCausalLM class, which for Qwen3.5 is the text-only
Qwen3_5ForCausalLM, so a trained LoRA adapter names modules model.layers.N.... vLLM builds the checkpoint's own class,
the image-text Qwen3_5ForConditionalGeneration, whose language-model modules are language_model.model.layers.N...:
it would not find the adapter's modules. vLLM's text-only Qwen3_5ForCausalLM cannot load the published checkpoint
(it has no place for the vision encoder's weights). From this copy both build the same text-only class.

Checks (asserted): every language-model tensor is copied bit for bit and nothing else is (the vision encoder and the
multi-token-prediction head are left out); the copy's tokenizer gives the original's chat-template prompt.
"""

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

PREFIX = "model.language_model."


def new_name(key: str) -> str:
    """The text-only class's name: model.language_model.X -> model.X; lm_head.weight (9B, untied) unchanged."""
    return key if key == "lm_head.weight" else "model." + key[len(PREFIX):]


model, out_dir = sys.argv[1], Path(sys.argv[2])
assert not out_dir.exists(), f"{out_dir} exists"
snapshot = Path(snapshot_download(model, local_files_only=True))
weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
language_keys = sorted(k for k in weight_map if k.startswith(PREFIX) or k == "lm_head.weight")
left_out = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in weight_map if k not in language_keys})
print(f"{len(language_keys)} language-model tensors; left out: {left_out}")
assert all(k.startswith(("model.visual.", "mtp.")) for k in weight_map if k not in language_keys)

out_dir.mkdir(parents=True)
new_weight_map = {}
for shard in sorted(set(weight_map[k] for k in language_keys)):
    with safe_open(snapshot / shard, framework="pt") as f:
        tensors = {new_name(k): f.get_tensor(k) for k in language_keys if weight_map[k] == shard}
    save_file(tensors, out_dir / shard, metadata={"format": "pt"})
    new_weight_map.update({k: shard for k in tensors})
    print(f"{shard}: {len(tensors)} tensors")
total_size = sum((out_dir / shard).stat().st_size for shard in set(new_weight_map.values()))
(out_dir / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": {"total_size": total_size}, "weight_map": new_weight_map}, indent=2) + "\n"
)

text_config = AutoConfig.from_pretrained(model).text_config
text_config.architectures = ["Qwen3_5ForCausalLM"]
text_config.save_pretrained(out_dir)
tokenizer = AutoTokenizer.from_pretrained(model)
tokenizer.save_pretrained(out_dir)

# Checks
for shard in sorted(set(weight_map[k] for k in language_keys)):
    with safe_open(snapshot / shard, framework="pt") as original, safe_open(out_dir / shard, framework="pt") as copy:
        for k in language_keys:
            if weight_map[k] == shard:
                assert torch.equal(original.get_tensor(k), copy.get_tensor(new_name(k))), k
assert len(new_weight_map) == len(language_keys)
messages = [{"role": "user", "content": "Which is right?\n\n(A) x\n(B) y"}]
copied = AutoTokenizer.from_pretrained(out_dir)
for thinking in (False, True):
    assert copied.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking) == \
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
assert (copied.eos_token, copied.pad_token, copied.bos_token) == (tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token)
config = json.loads((out_dir / "config.json").read_text())
print("config:", {k: config[k] for k in ("architectures", "model_type", "eos_token_id", "tie_word_embeddings", "num_hidden_layers")})
print(f"OK: {len(language_keys)} tensors equal, chat template equal -> {out_dir}")
