# release_mbpp_llama: the release's MBPP multi-round pipeline

The MBPP multi-round pipeline of the workshop paper's public release (`training/`, `probe_eval/`, `experiments/`),
moved here unchanged on 2026-11-26 so that the shared launcher could drop what only this pipeline used. It is
Llama-3-8B-Instruct only (`training/common.py` names the model) and no Qwen3.5 run uses it.

It imports the launcher as `training.grpo_launcher`, which now lives in `../launcher/` with a different interface, so
it does not run from later commits. Run it from the git tag `llama-final` (the last commit before the Qwen3.5 changes),
with the shared venv at `/workspace/obfuscation-atlas/.venv`; the release README
(`/workspace/probes-as-training-signals/README.md`) gives its commands.
