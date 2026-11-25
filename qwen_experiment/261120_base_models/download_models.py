"""Download the three Qwen3.5 checkpoints into the shared HF cache (HF_HOME=/workspace/hf_cache)."""
import time

from huggingface_hub import snapshot_download

for model in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"]:
    start = time.time()
    path = snapshot_download(model)
    print(f"{model} -> {path} ({time.time() - start:.0f} s)", flush=True)
print("DONE", flush=True)
