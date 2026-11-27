# Sourced by this folder's scripts: the live copy's environment (probe-rl/env.sh: its venv, HF_HOME, W&B, the secrets,
# $PY, $HOST) with the import paths switched to this checkout's code (/workspace/eff-wip, branch `efficiency`), whose
# trainer has the two new settings. The checkout has no venv of its own, so $PY stays the live copy's.
# Phase 2 (before main's reorganization, 2026-11-26) used sycophancy-rl/ for both paths; phase 3 uses probe-rl/.
source /workspace/probe-rl/env.sh
CODE=/workspace/eff-wip/probe-rl
export PYTHONPATH="$CODE:$CODE/obfuscation-atlas:$CODE/obfuscation-atlas/third_party/afterburner/src"
F=/workspace/eff-wip/qwen_experiment/261126_low_memory_passes
