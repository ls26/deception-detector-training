# Times Qwen3.5-9B's training-pass work on its own (no vLLM, no trainer), to find the ~0.4-0.5 s fixed cost per
# forward/backward pass seen in the 261122 runs. The model is built the way afterburner's HFModel builds it: bf16
# weights, sdpa attention, LoRA r 64 / alpha 128 on all linear layers as two adapters ("policy", "reference"),
# gradient checkpointing, then accelerate's prepare with bf16 mixed precision. Each pass is run the way
# grpo_trainer.py runs it:
#   probe     - probe-scoring forward: policy adapter, no gradients, all hidden states (score_responses)
#   reference - KL forward: reference adapter, no gradients, all hidden states
#   policy    - policy forward (no hidden states) + backward of a stand-in loss (the real loss is ~0.1 s per step)
# Each switches the adapter first, as HFModel.set_adapter does (PEFT set_adapter + a requires_grad loop).
# Part 1: wall time (GPU waited for) and CPU time to issue the work (before waiting), at 1 answer of 150 / 500 /
#         1000 tokens and 4 answers of up to 500 (right-padded, 18.5% padding), same length every repeat.
# Part 2: 1 answer whose length changes on every pass (as in training), to see any per-length compile cost.
# Part 3: one PyTorch profile of each pass at 1 x 500 tokens: GPU kernel time, kernels launched, waits, the
#         operations with the most CPU time, and CPU time inside linear-attention vs full-attention layers.
# Mode "fixes" instead times whole training passes (probe + reference + policy, as the trainer runs one micro-batch)
# on random lengths (question 200 + answer 50-1024 tokens, new lengths for every setting) under candidate fixes:
# cuDNN attention off, a cheaper adapter switch, LoRA weights in bf16, gradient checkpointing off, more answers per
# pass, and torch.compile on each decoder layer; plus peak memory at the longest answers (1224 tokens).
# Mode "cudnn_cache": does cuDNN attention keep its per-length setup? The reference forward and the policy pass run
# through the same set of distinct lengths twice, with cuDNN attention on and then off; if the second round is as
# slow as the first, training (which revisits lengths) pays the per-length cost every time.
# Mode "profile_sorted": cuDNN attention off and the cheap adapter switch on; at 1, 4 and 8 answers of 500 tokens each
# (equal lengths, as sorting gives), each pass's wall time (median of 5) and its GPU kernel time from one profile
# (summed over GPU events only, as the profiler's own "Self CUDA time total"), so the share of the time the GPU
# computes shows whether bigger passes are still CPU-bound.
# Usage: python time_model_passes.py <out_dir> profile|fixes|cudnn_cache|profile_sorted
import json
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from afterburner.training.interface import compute_logprobs_from_logits
from obfuscation_atlas.training.grpo_config import LoraParams
from peft import get_peft_model
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM

MODEL_PATH = "/workspace/hf_cache/text_only/Qwen3.5-9B"
REPEATS = 5
WARMUPS = 2
out_dir = Path(sys.argv[1])
mode = sys.argv[2]
assert mode in ("profile", "fixes", "cudnn_cache", "profile_sorted"), mode
out_dir.mkdir(parents=True, exist_ok=True)

accelerator = Accelerator(mixed_precision="bf16")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, attn_implementation="sdpa", dtype=torch.bfloat16, device_map={"": "cpu"}
)
model.config.use_cache = False
lora_config = LoraParams().to_lora_config()
model = get_peft_model(model, lora_config, adapter_name="policy")
model.base_model.add_adapter(adapter_name="reference", adapter_config=lora_config)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = accelerator.prepare(model)
model.train()
device = accelerator.device
print(f"model ready on {device}; adapters {list(model.peft_config)}", flush=True)


FAST_ADAPTER_SWITCH = False  # fixes mode: reset requires_grad on the LoRA parameters only (PEFT changes only those)
lora_parameter_flags = [(param, "policy" in name) for name, param in model.named_parameters() if "lora_" in name]


def set_adapter(adapter_name):
    model.base_model.set_adapter([adapter_name])
    if FAST_ADAPTER_SWITCH:
        for param, trainable in lora_parameter_flags:
            param.requires_grad = trainable
        return
    for name, param in model.named_parameters():
        param.requires_grad = "policy" in name


def forward_logprobs(adapter_name, batch, output_hidden_states):
    set_adapter(adapter_name)
    grad_context = nullcontext() if adapter_name == "policy" else torch.no_grad()
    with grad_context:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=output_hidden_states,
        )
    logprobs = compute_logprobs_from_logits(outputs.logits, batch["input_ids"], batch["completion_mask"])
    return logprobs, outputs.hidden_states


def probe_pass(batch):
    with torch.no_grad():
        forward_logprobs("policy", batch, output_hidden_states=True)


def reference_pass(batch):
    with torch.no_grad():
        forward_logprobs("reference", batch, output_hidden_states=True)


def policy_pass(batch):
    logprobs, _ = forward_logprobs("policy", batch, output_hidden_states=False)
    completion = batch["completion_mask"][:, 1:]
    loss = -(logprobs * completion).sum() / completion.sum()
    accelerator.backward(loss / 16)


PASSES = {"probe": probe_pass, "reference": reference_pass, "policy": policy_pass}


def make_batch(lengths, prompt_length):
    longest = max(lengths)
    input_ids = torch.randint(1000, 100000, (len(lengths), longest))
    attention_mask = torch.zeros(len(lengths), longest, dtype=torch.long)
    completion_mask = torch.zeros(len(lengths), longest, dtype=torch.bool)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
        completion_mask[row, prompt_length:length] = True
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
            "completion_mask": completion_mask.to(device)}


def timed(pass_function, batch):
    torch.cuda.synchronize()
    start = time.perf_counter()
    pass_function(batch)
    issued = time.perf_counter()
    torch.cuda.synchronize()
    finished = time.perf_counter()
    model.zero_grad(set_to_none=True)
    return issued - start, finished - start


results = {}


def training_pass(batch):
    torch.cuda.synchronize()
    start = time.perf_counter()
    probe_pass(batch)
    reference_pass(batch)
    policy_pass(batch)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return time.perf_counter() - start


def time_training_passes(label, answers_per_pass, seed, passes=10, similar_lengths=False):
    """Median seconds per training pass on random lengths; similar_lengths mimics answers sorted by length."""
    rng = random.Random(seed)

    def random_lengths():
        if similar_lengths:
            center = rng.randint(250, 1224)
            return [max(250, center - rng.randint(0, center // 20)) for _ in range(answers_per_pass)]
        return [rng.randint(250, 1224) for _ in range(answers_per_pass)]

    for _ in range(WARMUPS):
        training_pass(make_batch(random_lengths(), 200))
    times = [training_pass(make_batch(random_lengths(), 200)) for _ in range(passes)]
    per_pass = statistics.median(times)
    results[label] = {"per_pass_s": per_pass, "per_answer_s": per_pass / answers_per_pass, "all_s": times}
    print(f"{label:58s} per pass {per_pass * 1e3:7.0f} ms   per answer {per_pass / answers_per_pass * 1e3:6.0f} ms",
          flush=True)


def peak_memory_one_pass(label, answers_per_pass):
    """Peak memory of one training pass at the longest answers; running out of memory is a result here."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        training_pass(make_batch([1224] * answers_per_pass, 200))
        results[label] = torch.cuda.max_memory_allocated() / 1e9
    except torch.cuda.OutOfMemoryError:
        results[label] = "out of memory (80 GB)"
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    print(f"{label:58s} peak memory allocated {results[label]} GB (vLLM's ~32 GB not included)", flush=True)


if mode == "profile_sorted":
    from torch.autograd import DeviceType

    torch.backends.cuda.enable_cudnn_sdp(False)
    FAST_ADAPTER_SWITCH = True
    for answers in (1, 4, 8):
        batch = make_batch([500] * answers, 200)
        for pass_name, pass_function in PASSES.items():
            for _ in range(WARMUPS):
                timed(pass_function, batch)
            wall = statistics.median(timed(pass_function, batch)[1] for _ in range(REPEATS))
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
                pass_function(batch)
                torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            events = profiler.key_averages()
            gpu_ms = sum(e.self_device_time_total for e in events if e.device_type == DeviceType.CUDA) / 1e3
            label = f"{answers}x500 {pass_name}"
            results[label] = {"wall_ms": wall * 1e3, "gpu_kernel_ms": gpu_ms, "gpu_share": gpu_ms / (wall * 1e3)}
            print(f"{label:16s} wall {wall * 1e3:7.1f} ms   GPU computing {gpu_ms:7.1f} ms   share "
                  f"{gpu_ms / (wall * 1e3):.0%}", flush=True)
            with open(out_dir / f"profile_sorted_{answers}x500_{pass_name}.txt", "w") as table_file:
                table_file.write(events.table(sort_by="self_cpu_time_total", row_limit=30, max_name_column_width=70))
    json.dump(results, open(out_dir / "profile_sorted.json", "w"), indent=1)
    print("PROFILE_SORTED_DONE", flush=True)
    sys.exit(0)

if mode == "cudnn_cache":
    for cudnn_on in (True, False):
        torch.backends.cuda.enable_cudnn_sdp(cudnn_on)
        for pass_name, pass_function, distinct_lengths in (("reference", reference_pass, 300),
                                                           ("policy", policy_pass, 100)):
            offset = 0 if cudnn_on else 1  # odd lengths with cuDNN off, so neither setting reuses the other's
            lengths = [250 + 2 * index + offset for index in range(distinct_lengths)]
            for round_index in (1, 2):
                walls = [timed(pass_function, make_batch([length], 200))[1] for length in lengths]
                label = f"cuDNN {'on' if cudnn_on else 'off'} {pass_name} round {round_index}"
                results[label] = walls
                print(f"{label:34s} ({distinct_lengths} distinct lengths): median {statistics.median(walls) * 1e3:.0f}"
                      f" ms, mean {statistics.mean(walls) * 1e3:.0f} ms", flush=True)
    json.dump(results, open(out_dir / "cudnn_cache.json", "w"), indent=1)
    print("CACHE_DONE", flush=True)
    sys.exit(0)

if mode == "fixes":
    dtypes ={(("policy" if "policy" in name else "reference"), str(param.dtype)) for name, param in
              model.named_parameters() if "lora_" in name}
    print(f"LoRA weight dtypes: {sorted(dtypes)}; base weights {next(model.parameters()).dtype}", flush=True)
    start = time.perf_counter()
    for _ in range(20):
        model.base_model.set_adapter(["policy"])
    print(f"PEFT set_adapter alone: {(time.perf_counter() - start) / 20 * 1e3:.1f} ms per call", flush=True)

    seed = 0
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass baseline", answers, seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass cuDNN attention off", answers, seed)
    torch.backends.cuda.enable_cudnn_sdp(True)
    FAST_ADAPTER_SWITCH = True
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass fast adapter switch", answers, seed)
    torch.backends.cuda.enable_cudnn_sdp(False)
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass cuDNN off + fast switch", answers, seed)

    fp32_lora = [param for name, param in model.named_parameters() if "lora_" in name and param.dtype == torch.float32]
    for param in fp32_lora:
        param.data = param.data.to(torch.bfloat16)
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass cuDNN off + fast switch + bf16 LoRA weights", answers, seed)
    for param in fp32_lora:
        param.data = param.data.float()

    seed += 1
    time_training_passes("8/pass cuDNN off + fast switch", 8, seed, passes=6)
    for answers in (4, 8):
        seed += 1
        time_training_passes(f"{answers}/pass cuDNN off + fast switch, similar lengths", answers, seed, passes=6,
                             similar_lengths=True)

    # Without checkpointing, 4 answers per pass ran out of the 80 GB in the first fixes run (fixes_run1_<host>.log,
    # 2026-11-24 15:55 UTC), so only 1 per pass is timed.
    model.gradient_checkpointing_disable()
    seed += 1
    time_training_passes("1/pass cuDNN off + fast switch + no checkpointing", 1, seed)
    peak_memory_one_pass("1 x 1224 tokens, no checkpointing", 1)
    model.gradient_checkpointing_enable()
    json.dump(results, open(out_dir / "fixes.json", "w"), indent=1)

    for answers in (1, 4, 8, 16):
        peak_memory_one_pass(f"{answers} x 1224 tokens, checkpointing", answers)
    json.dump(results, open(out_dir / "fixes.json", "w"), indent=1)

    compile_start = time.perf_counter()
    for layer in model.base_model.model.model.layers:
        layer.compile(dynamic=True)
    training_pass(make_batch([700], 200))
    print(f"torch.compile: first pass (compiling) took {time.perf_counter() - compile_start:.0f} s", flush=True)
    for answers in (1, 4):
        seed += 1
        time_training_passes(f"{answers}/pass cuDNN off + fast switch + compiled layers", answers, seed)
    json.dump(results, open(out_dir / "fixes.json", "w"), indent=1)
    print("FIXES_DONE", flush=True)
    sys.exit(0)

# Part 1: same shape every repeat.
SHAPES = {"1x150": ([150], 100), "1x500": ([500], 200), "1x1000": ([1000], 200), "4x500": ([500, 450, 380, 300], 200)}
for shape_name, (lengths, prompt_length) in SHAPES.items():
    batch = make_batch(lengths, prompt_length)
    torch.cuda.reset_peak_memory_stats()
    for pass_name, pass_function in PASSES.items():
        for _ in range(WARMUPS):
            timed(pass_function, batch)
        runs = [timed(pass_function, batch) for _ in range(REPEATS)]
        issue = statistics.median(r[0] for r in runs)
        wall = statistics.median(r[1] for r in runs)
        results[f"{shape_name} {pass_name}"] = {"issue_s": issue, "wall_s": wall}
        print(f"{shape_name:7s} {pass_name:9s} wall {wall * 1e3:7.1f} ms   CPU issue {issue * 1e3:7.1f} ms", flush=True)
    results[f"{shape_name} peak_memory_GB"] = torch.cuda.max_memory_allocated() / 1e9
    print(f"{shape_name:7s} peak memory allocated {results[f'{shape_name} peak_memory_GB']:.1f} GB", flush=True)

# Part 2: a new length on every pass (as in training); compare with Part 1's same-length times.
for pass_name, pass_function in PASSES.items():
    runs = []
    for step in range(8):
        length = 611 + 37 * step + {"probe": 0, "reference": 7, "policy": 13}[pass_name]
        runs.append(timed(pass_function, make_batch([length], 200)))
    walls = [r[1] for r in runs]
    results[f"new_length {pass_name}"] = {"wall_s_each": walls}
    print(f"new length each pass, {pass_name:9s}: wall ms " + " ".join(f"{w * 1e3:.0f}" for w in walls), flush=True)

# set_adapter alone (the trainer calls it once per pass type).
start = time.perf_counter()
for _ in range(20):
    set_adapter("policy")
results["set_adapter_s"] = (time.perf_counter() - start) / 20
print(f"set_adapter + requires_grad loop: {results['set_adapter_s'] * 1e3:.1f} ms per call", flush=True)
json.dump(results, open(out_dir / "timings.json", "w"), indent=1)

# Part 3: profile one pass of each kind at 1 x 500 tokens, with CPU ranges around each decoder layer by type.
peft_inner = model.base_model.model  # Qwen3_5ForCausalLM
layer_types = peft_inner.config.layer_types
open_ranges = {}


def enter_layer(layer_index):
    def hook(module, args):
        open_ranges.setdefault(layer_index, []).append(record_function(f"layer: {layer_types[layer_index]}"))
        open_ranges[layer_index][-1].__enter__()
    return hook


def exit_layer(layer_index):
    def hook(module, args, output):
        open_ranges[layer_index].pop().__exit__(None, None, None)
    return hook


for index, layer in enumerate(peft_inner.model.layers):
    layer.register_forward_pre_hook(enter_layer(index))
    layer.register_forward_hook(exit_layer(index))

batch = make_batch([500], 200)
for pass_name, pass_function in PASSES.items():
    for _ in range(WARMUPS):
        timed(pass_function, batch)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        start = time.perf_counter()
        pass_function(batch)
        torch.cuda.synchronize()
        profiled_wall = time.perf_counter() - start
    model.zero_grad(set_to_none=True)
    events = profiler.key_averages()
    gpu_kernel_ms = sum(e.self_device_time_total for e in events) / 1e3
    gpu_kernel_count = sum(e.count for e in events if e.self_device_time_total > 0)
    by_name = {e.key: e for e in events}
    summary = {
        "profiled_wall_ms": profiled_wall * 1e3,
        "gpu_kernel_time_ms": gpu_kernel_ms,
        "gpu_kernels": gpu_kernel_count,
        "cudaLaunchKernel_calls": by_name["cudaLaunchKernel"].count if "cudaLaunchKernel" in by_name else 0,
        "cuLaunchKernel(Ex)_calls (Triton)": sum(by_name[k].count for k in by_name if k.startswith("cuLaunchKernel")),
        "aten_ops_called": sum(e.count for e in events if e.key.startswith("aten::")),
        "sync_waits(cudaStreamSynchronize+cudaDeviceSynchronize+item)": sum(
            by_name[k].count for k in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "aten::item") if k in by_name
        ),
        "cpu_ms_in_linear_attention_layers": by_name["layer: linear_attention"].cpu_time_total / 1e3
        if "layer: linear_attention" in by_name else None,
        "cpu_ms_in_full_attention_layers": by_name["layer: full_attention"].cpu_time_total / 1e3
        if "layer: full_attention" in by_name else None,
        "gpu_ms_in_linear_attention_layers": by_name["layer: linear_attention"].device_time_total / 1e3
        if "layer: linear_attention" in by_name else None,
        "gpu_ms_in_full_attention_layers": by_name["layer: full_attention"].device_time_total / 1e3
        if "layer: full_attention" in by_name else None,
    }
    results[f"profile 1x500 {pass_name}"] = summary
    print(f"profile 1x500 {pass_name}: " + json.dumps({k: round(v, 1) if isinstance(v, float) else v
                                                     for k, v in summary.items()}), flush=True)
    with open(out_dir / f"profile_1x500_{pass_name}.txt", "w") as table_file:
        table_file.write(events.table(sort_by="self_cpu_time_total", row_limit=40, max_name_column_width=70))
        table_file.write("\n\n")
        table_file.write(events.table(sort_by="self_device_time_total", row_limit=25, max_name_column_width=70))
    profiler.export_chrome_trace(str(out_dir / f"trace_1x500_{pass_name}.json"))

json.dump(results, open(out_dir / "timings.json", "w"), indent=1)
print("TIMING_DONE", flush=True)
