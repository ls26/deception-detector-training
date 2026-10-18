# Source (do not execute) before every step: `source run_env.sh`.
# Runs everything from the code copy at /workspace/sycophancy-rl: its training package and its
# vendored GRPO framework shadow the private copies installed in the shared venv.
COPY=/workspace/sycophancy-rl
export PYTHONPATH="$COPY:$COPY/obfuscation-atlas:$COPY/obfuscation-atlas/third_party/afterburner/src"
# The venv's bin first: on a B200, vLLM picks the FlashInfer attention backend, which JIT-builds
# kernels with the venv's `ninja`. (On the H200 it picks FlashAttention; the line is harmless.)
export PATH="/workspace/obfuscation-atlas/.venv/bin:$PATH"
export HF_HOME=/workspace/hf_cache
export WANDB_PROJECT=deception_detection
set -a
source /workspace/.secrets.env
set +a
PY=/workspace/obfuscation-atlas/.venv/bin/python
RUN_DIR=/workspace/sycophancy_experiments/261014_nocot_noisy_label_rl
HOST=$(hostname)
