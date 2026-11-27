# Peak GPU memory of the trainer's worst-case training pass for Qwen3.5-9B sycophancy, with grpo.model.low_memory_passes
# off (today's passes) and on: 4 and 8 answers of 1024 prompt + 1024 answer tokens per pass (the 2048-token limit,
# longer than either 261125 out-of-memory crash, ~1550 and ~1590 tokens). Adapted from 261123_9b_speed/time_model_passes.py,
# but the model, adapters and optimizer come from the trainer's own setup (HfObfuscationTrainerState.
# initialize_and_move_to_gpu: bf16 weights, LoRA r 64 / alpha 128 policy and reference adapters, gradient
# checkpointing, accelerate's bf16 prepare, AdamW), with cuDNN attention off as the 9B preset sets it, and the passes
# run through the trainer's own helpers (GRPOObfuscationTrainer.probe_scoring_hidden_states, training_pass_logprobs):
#   1. the probe-scoring forward (policy adapter, probe layers 18-21), discarded as score_responses discards it;
#   2. the reference forward, kept through the policy pass as train_on_generation_batch keeps it;
#   3. the policy forward and backward (loss / gradient-accumulation steps, as the trainer scales it).
# A first pass and an optimizer step create the gradients and AdamW's states; the measured pass is the second, with
# both present, as every later micro-batch of a step has them. Run next to the run's vLLM server (0.40 of the GPU).
# Reported per setting: torch's peak reserved and allocated memory over the measured pass, and the GPU's free memory
# right after it (the allocator keeps what it reserved, so this is the room left at the peak, CUDA context and
# non-PyTorch buffers included); out of memory is a result. Also the GPU's free memory before this process started
# CUDA (what vLLM leaves) and right after (minus the CUDA context).
# Then a numbers check on 2 answers (1024 + 1024 and 300 + 500 tokens, right-padded), each pass with the setting off
# and on (no gradients): log-prob differences over answer tokens (max and mean absolute difference, correlation) and
# whether the hidden states the probe reads are bit for bit equal. Reported, not asserted.
# Usage: python memory_test.py <out_dir> [smoke]   (smoke: 1 answer of 256 + 256 tokens per setting, no vLLM needed)
import json
import subprocess
import sys
from pathlib import Path

import torch
from afterburner.training.huggingface.accelerator import HfAccelerator
from afterburner.utils.profiling import Profiler
from transformers import AutoTokenizer

from obfuscation_atlas.training.grpo_config import GRPOConfig, GRPOModelConfig
from obfuscation_atlas.training.grpo_trainer import GRPOObfuscationTrainer, HfObfuscationTrainerState

MODEL = "/workspace/hf_cache/text_only/Qwen3.5-9B"
PROBE_LAYERS = [18, 19, 20, 21]  # 261122's bias probe (model.activation_matching_layers range_18-22)
GIB = 2**30
out_dir = Path(sys.argv[1])
smoke = sys.argv[2:] == ["smoke"]
assert sys.argv[2:] in ([], ["smoke"]), sys.argv
out_dir.mkdir(parents=True, exist_ok=True)
results: dict = {}


def gpu_free_gib_by_nvidia_smi() -> float:
    total, used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], text=True
    ).split(",")
    return (int(total) - int(used)) * 2**20 / GIB


results["free_before_cuda_GiB"] = gpu_free_gib_by_nvidia_smi()
results["free_after_cuda_init_GiB"] = torch.cuda.mem_get_info()[0] / GIB
print(f"free GPU memory: {results['free_before_cuda_GiB']:.2f} GiB before CUDA started, "
      f"{results['free_after_cuda_init_GiB']:.2f} GiB after", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = GRPOConfig(model=GRPOModelConfig(model_name=MODEL, cudnn_attention=False))
accelerator = HfAccelerator(config=config, loss_function=None, gradient_accumulation_steps=16, profiler=Profiler())
torch.backends.cuda.enable_cudnn_sdp(False)  # as GRPOObfuscationTrainer does with cudnn_attention False
state = HfObfuscationTrainerState.initialize_and_move_to_gpu(
    config=config, accelerator=accelerator, total_steps=10, pad_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
)
state.model.shared_base_model.train()
trainer = GRPOObfuscationTrainer.__new__(GRPOObfuscationTrainer)  # only the two pass helpers are used
trainer.config = config
trainer.activation_layers = PROBE_LAYERS
device = accelerator.device
results["model_loaded_allocated_GiB"] = torch.cuda.memory_allocated() / GIB
print(f"model ready on {device}: {results['model_loaded_allocated_GiB']:.2f} GiB allocated", flush=True)


def make_batch(lengths: list[tuple[int, int]]) -> dict[str, torch.Tensor]:
    """Right-padded rows of (prompt tokens, answer tokens); random token ids as in 261123."""
    longest = max(prompt + answer for prompt, answer in lengths)
    input_ids = torch.randint(1000, 100000, (len(lengths), longest))
    attention_mask = torch.zeros(len(lengths), longest, dtype=torch.long)
    completion_mask = torch.zeros(len(lengths), longest, dtype=torch.bool)
    for row, (prompt, answer) in enumerate(lengths):
        attention_mask[row, : prompt + answer] = 1
        completion_mask[row, prompt : prompt + answer] = True
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
            "completion_mask": completion_mask.to(device)}


def training_pass(batch: dict[str, torch.Tensor], gradient_accumulation_steps: int) -> None:
    with torch.no_grad():
        probe_hidden = trainer.probe_scoring_hidden_states(state, state.model.policy_adapter, batch)
    del probe_hidden
    with torch.no_grad():
        reference_logprobs, reference_hidden = trainer.training_pass_logprobs(
            state, state.model.reference_adapter, batch, with_hidden_states=True
        )
    policy_logprobs, _ = trainer.training_pass_logprobs(state, state.model.policy_adapter, batch, with_hidden_states=False)
    answer = batch["completion_mask"][:, 1:]
    loss = torch.where(answer, policy_logprobs - reference_logprobs, 0.0).sum() / answer.sum()  # a stand-in loss
    accelerator._accelerator.backward(loss / gradient_accumulation_steps)
    del reference_hidden


def measure(label: str, low_memory_passes: bool, answers: int, prompt: int, answer: int) -> None:
    config.model.low_memory_passes = low_memory_passes
    state.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    batch = make_batch([(prompt, answer)] * answers)
    try:
        training_pass(batch, 64 // answers)  # creates the gradients
        state.optimizer.step()  # creates AdamW's states (the gradients stay, as within a step)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        training_pass(batch, 64 // answers)
        torch.cuda.synchronize()
        results[label] = {
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / GIB,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / GIB,
            "free_after_pass_GiB": torch.cuda.mem_get_info()[0] / GIB,
        }
    except torch.cuda.OutOfMemoryError:
        results[label] = "out of memory"
    print(f"{label:44s} {results[label]}", flush=True)
    json.dump(results, open(out_dir / "memory.json", "w"), indent=1)


shapes = [(1, 256, 256)] if smoke else [(4, 1024, 1024), (8, 1024, 1024)]
for low_memory_passes in (True, False):
    for answers, prompt, answer in shapes:
        setting = "on" if low_memory_passes else "off"
        measure(f"low_memory_passes {setting}, {answers} x ({prompt} + {answer})", low_memory_passes, answers, prompt,
                answer)

# Numbers check: each pass with the setting off and on, on the same batch, no gradients.
state.optimizer.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
batch = make_batch([(128, 128), (40, 60)] if smoke else [(1024, 1024), (300, 500)])
answer = batch["completion_mask"][:, 1:]
outputs = {}
for low_memory_passes in (False, True):
    config.model.low_memory_passes = low_memory_passes
    with torch.no_grad():
        probe_hidden = trainer.probe_scoring_hidden_states(state, state.model.policy_adapter, batch)
        reference_logprobs, reference_hidden = trainer.training_pass_logprobs(
            state, state.model.reference_adapter, batch, with_hidden_states=True
        )
        policy_logprobs, _ = trainer.training_pass_logprobs(state, state.model.policy_adapter, batch, with_hidden_states=False)
    probe_indices = [0, *(layer + 1 for layer in PROBE_LAYERS)]
    outputs[low_memory_passes] = {
        "probe": [probe_hidden[index] for index in probe_indices],
        "reference_hidden": [reference_hidden[index] for index in probe_indices],
        "reference_logprobs": reference_logprobs[answer],
        "policy_logprobs": policy_logprobs[answer],
    }
    del probe_hidden, reference_hidden
for name in ("reference_logprobs", "policy_logprobs"):
    today, low_memory = outputs[False][name], outputs[True][name]
    difference = (low_memory - today).abs()
    results[f"numbers {name}"] = {
        "answer_tokens": today.numel(),
        "max_abs_difference": difference.max().item(),
        "mean_abs_difference": difference.mean().item(),
        "correlation": torch.corrcoef(torch.stack([today, low_memory]))[0, 1].item(),
        "identical": torch.equal(today, low_memory),
    }
for name in ("probe", "reference_hidden"):
    results[f"numbers {name} inputs bit for bit equal"] = all(
        torch.equal(today, low_memory) for today, low_memory in zip(outputs[False][name], outputs[True][name], strict=True)
    )
for key, value in results.items():
    if key.startswith("numbers"):
        print(f"{key:44s} {value}", flush=True)
json.dump(results, open(out_dir / "memory.json", "w"), indent=1)
print("MEMORY_TEST_DONE", flush=True)
