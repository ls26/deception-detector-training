# Source before every step of this folder: the new venv (/workspace/sycophancy-rl/.venv) with the sycophancy-rl code
# copy first on PYTHONPATH, as sycophancy-rl/env.sh does for the old venv (env.sh is switched to the new venv only in
# the shared-code phase). Offline: every model is in the HF cache.
COPY=/workspace/sycophancy-rl
export PYTHONPATH="$COPY:$COPY/obfuscation-atlas:$COPY/obfuscation-atlas/third_party/afterburner/src"
export PATH="$COPY/.venv/bin:$PATH"
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=0
PY=$COPY/.venv/bin/python
HOST=$(hostname)
