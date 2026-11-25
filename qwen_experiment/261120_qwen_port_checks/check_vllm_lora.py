"""Smoke checks (a) and (b) of the Qwen port: the text-only copy of Qwen3.5-4B (make_text_only_checkpoint.py) loads as
the text-only model in transformers and in vLLM, and vLLM applies a trained LoRA adapter as transformers + PEFT does.

    python check_vllm_lora.py adapters --out-dir DIR   # two random non-zero adapters on the transformers model
    python check_vllm_lora.py vllm     --out-dir DIR   # vLLM (as the RL server loads it): greedy answers + logprobs
    python check_vllm_lora.py hf       --out-dir DIR   # transformers + PEFT logprobs of vLLM's tokens; report.json

Adapters (the trainer's LoRA: r 64, alpha 128 as grpo_config, B drawn N(0, 0.01^2) so they change the output):
  all_linear     every linear layer (the trainer's target_modules "all-linear")
  linear_attn    only the linear-attention (Gated DeltaNet) projections
Per adapter (and none), on 6 prompts, vLLM greedily generates 48 tokens with their logprobs; transformers computes
the logprobs of the same tokens. Reported: the largest and mean |vLLM - transformers| gap, next to the gap with no
adapter, and how much each adapter moves vLLM's logprobs from the base model's (so a vLLM that ignored an adapter
shows a gap the size of that move).
"""

import argparse
import importlib
import json
from pathlib import Path

MODEL = "/workspace/hf_cache/text_only/Qwen3.5-4B"
ADAPTERS = {
    "all_linear": "all-linear",
    "linear_attn": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
}
PROMPTS = [
    "Which is larger, 2^10 or 10^3? Answer briefly.",
    "The following is a multiple choice question about law. Think step by step and end with \"The answer is (X)\".\n\n"
    "A contract signed under duress is:\n(A) void\n(B) voidable\n\nI think the answer is (A).",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Name three prime numbers above 50.",
    "Explain in one sentence why the sky is blue.",
    "What is the capital of Australia?",
]
N_TOKENS = 48


def prompt_ids(tokenizer) -> list[list[int]]:
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for p in PROMPTS]
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def make_adapters(out_dir: Path) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    for name, targets in ADAPTERS.items():
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, target_modules=targets, lora_dropout=0.0))
        generator = torch.Generator().manual_seed(0)
        n_modules = 0
        for module_name, param in model.named_parameters():
            if "lora_B" in module_name:
                param.data = (torch.randn(param.shape, generator=generator) * 0.01).to(param.dtype)
                n_modules += 1
        model.save_pretrained(out_dir / name)
        print(name, "LoRA modules:", n_modules)
        del model


def run_vllm(out_dir: Path) -> None:
    from afterburner.generation.vllm.server import check_trainer_class
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    check_trainer_class(MODEL, None)
    llm = LLM(MODEL, enable_lora=True, max_lora_rank=64, max_loras=1, gpu_memory_utilization=0.5, max_model_len=4096)
    architecture = llm.llm_engine.vllm_config.model_config.architectures
    ids = prompt_ids(tokenizer)
    rows = []
    for index, name in enumerate([None, *ADAPTERS]):
        lora = None if name is None else LoRARequest(name, index, str(out_dir / name))
        outputs = llm.generate([{"prompt_token_ids": p} for p in ids],
                               SamplingParams(temperature=0.0, max_tokens=N_TOKENS, logprobs=0), lora_request=lora)
        for p, out in zip(ids, outputs):
            completion = out.outputs[0]
            rows.append({"adapter": name, "prompt_ids": p, "token_ids": list(completion.token_ids),
                         "logprobs": [lp[t].logprob for lp, t in zip(completion.logprobs, completion.token_ids)],
                         "text": completion.text})
    (out_dir / "vllm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "vllm_meta.json").write_text(json.dumps({"architectures": architecture}) + "\n")
    print("architectures:", architecture)


def run_hf(out_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    report: dict = {}
    model, loading_info = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, output_loading_info=True)
    report["transformers_class"] = type(model).__name__
    report["loading_info"] = {k: sorted(v)[:20] for k, v in loading_info.items() if v}
    report["n_loading_issues"] = {k: len(v) for k, v in loading_info.items()}
    model = model.to("cuda").eval()
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3]], device="cuda"), output_hidden_states=True)
    report["n_hidden_states"] = len(out.hidden_states)
    for package, function in (("fla", "chunk_gated_delta_rule"), ("causal_conv1d", "causal_conv1d_fn")):
        try:
            module = importlib.import_module(package)
            report[f"kernel_{package}"] = hasattr(module, function)
        except ImportError as error:
            report[f"kernel_{package}"] = f"not importable: {error}"

    rows = [json.loads(line) for line in (out_dir / "vllm.jsonl").read_text().splitlines()]

    def hf_logprobs(row) -> list[float]:
        ids = torch.tensor([row["prompt_ids"] + row["token_ids"]], device="cuda")
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        logprobs = torch.log_softmax(logits[len(row["prompt_ids"]) - 1 : -1], dim=-1)
        return logprobs.gather(1, torch.tensor(row["token_ids"], device="cuda")[:, None])[:, 0].tolist()

    base_model = model
    gaps = {}
    for name in [None, *ADAPTERS]:
        adapter_rows = [row for row in rows if row["adapter"] == name]
        model = base_model
        base_logprobs = [hf_logprobs(row) for row in adapter_rows]  # the base model on these tokens
        if name is not None:
            model = PeftModel.from_pretrained(base_model, str(out_dir / name)).eval()
        differences, adapter_effect = [], []
        for row, base in zip(adapter_rows, base_logprobs):
            with_adapter = hf_logprobs(row)
            differences += [abs(a - b) for a, b in zip(row["logprobs"], with_adapter)]
            adapter_effect += [abs(a - b) for a, b in zip(with_adapter, base)]
        gaps[str(name)] = {"max": max(differences), "mean": sum(differences) / len(differences), "tokens": len(differences),
                           "adapter_effect_in_transformers_mean": sum(adapter_effect) / len(adapter_effect)}
        if name is not None:
            base_model = model.unload()  # back to the bare model for the next adapter
    report["vllm_vs_transformers_logprob_gap"] = gaps

    by_adapter = {name: [r for r in rows if r["adapter"] == name] for name in [None, *ADAPTERS]}
    report["adapter_changes_vllm"] = {}
    for name in ADAPTERS:
        same_tokens = sum(a["token_ids"] == b["token_ids"] for a, b in zip(by_adapter[None], by_adapter[name]))
        report["adapter_changes_vllm"][name] = {"answers_with_same_tokens_as_base": f"{same_tokens}/{len(PROMPTS)}"}
    report["sample_texts"] = {str(r["adapter"]): r["text"][:200] for r in rows if r["prompt_ids"] == rows[0]["prompt_ids"]}
    report["vllm_meta"] = json.loads((out_dir / "vllm_meta.json").read_text())
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("adapters", "vllm", "hf"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"adapters": make_adapters, "vllm": run_vllm, "hf": run_hf}[args.stage](args.out_dir)
