# Source (do not execute) before every step of any task in this copy (sycophancy, MBPP):
# `source /workspace/sycophancy-rl/env.sh`. Sycophancy steps source sycophancy_setup/run_env.sh, which sources this.
# Runs everything from the code copy at /workspace/sycophancy-rl: its training package and its vendored GRPO framework
# (afterburner) shadow the older copies the shared venv installs from /workspace/obfuscation-atlas. Without this, the
# copy's trainer imports the older afterburner, whose compute_advantages_from_rewards has no std_floor argument, and
# crashes at the first GRPO step.
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
HOST=$(hostname)
