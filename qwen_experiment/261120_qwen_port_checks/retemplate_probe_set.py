"""Llama's bias probe set (sycophancy-rl/sycophancy_setup/data/bias_probe_{train,eval}.jsonl) with each prompt put
in Qwen3.5's chat template (thinking off); completions, labels and every other field unchanged. For the smoke runs
only: it tests that the trainer trains and reads a probe on Qwen, not a probe of Qwen's own answers.

    python retemplate_probe_set.py --out-dir data
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

SOURCE = Path("/workspace/sycophancy-rl/sycophancy_setup/data")
LLAMA_START = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
LLAMA_END = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path, required=True)
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
args.out_dir.mkdir(parents=True, exist_ok=True)
for split in ("train", "eval"):
    rows = [json.loads(line) for line in (SOURCE / f"bias_probe_{split}.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["prompt"].startswith(LLAMA_START) and row["prompt"].endswith(LLAMA_END)
        user_content = row["prompt"][len(LLAMA_START) : -len(LLAMA_END)]
        row["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False,
                                                      add_generation_prompt=True, enable_thinking=False)
    out_path = args.out_dir / f"bias_probe_{split}_qwen_template.jsonl"
    out_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{out_path}: {len(rows)} rows; first prompt ends {rows[0]['prompt'][-60:]!r}")
