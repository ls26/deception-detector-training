# Source (do not execute) before every step of any task in this copy (sycophancy, MBPP):
# `source /workspace/probe-rl/env.sh`. Sycophancy steps source sycophancy_setup/run_env.sh, which sources this.
# COPY is the folder this file is in, so another checkout of the code (a git worktree) runs its own files.
# Runs everything from that copy: its launcher, its trainer and its vendored GRPO framework (afterburner) shadow the
# older copies the shared venv at /workspace/obfuscation-atlas/.venv installs. Without this, the trainer imports the
# older afterburner, whose compute_advantages_from_rewards has no std_floor argument, and crashes at the first GRPO
# step.
COPY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$COPY:$COPY/obfuscation-atlas:$COPY/obfuscation-atlas/third_party/afterburner/src"
# The copy's own venv (.venv, pinned in requirements.txt: transformers 5 and vLLM 0.33, which Qwen3.5 needs; Llama-3
# runs in it too). The shared venv at /workspace/obfuscation-atlas/.venv (transformers 4.56, vLLM 0.13) runs the code
# at the git tag llama-final, for exact reruns of the Llama experiments.
# The venv's bin first: on a B200, vLLM picks the FlashInfer attention backend, which JIT-builds
# kernels with the venv's `ninja`. (On the H200 it picks FlashAttention; the line is harmless.)
export PATH="$COPY/.venv/bin:$PATH"
export HF_HOME=/workspace/hf_cache
export WANDB_PROJECT=deception_detection
set -a
source /workspace/.secrets.env
SECRETS_READ=$?
set +a
[[ $SECRETS_READ -eq 0 ]] || { echo "env.sh: could not read /workspace/.secrets.env" >&2; return 1; }
PY=$COPY/.venv/bin/python
HOST=$(hostname)
