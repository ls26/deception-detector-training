# Full local GRPO runbook

> **Paper-era MBPP code.** This file describes the `obfuscation-atlas/` + `pipeline/` code used for the MBPP workshop paper's Llama-3-8B-Instruct runs. The MBPP rerun in [paper_plan.md](./paper_plan.md) uses `probe-rl/` instead; see [CONTEXT.md](./CONTEXT.md).

How to run **train + post-eval** on RunPod with `train_grpo_model_against_probe.py` (one W&B run per config when everything finishes cleanly).

For what each stage logs, see [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md).

---

## What this workflow does

One command trains a Llama-3-8B LoRA with GRPO on MBPP hardcode, penalizes a linear deception probe in the **reward**, then runs five post-train probe eval stages and logs obfuscation decomposition to W&B.

This is **not** HF probe eval (`eval_probe_checkpoints.py`) — you train from scratch here.

**Entry point:**

```bash
python -m obfuscation_atlas.scripts.train_grpo_model_against_probe [Hydra overrides]
```

**Primary deliverables per run:**

| Deliverable | Path |
|-------------|------|
| Final LoRA | `…/run_results/{wandb.group}/{run_key}_{RUN_ID}/checkpoints/final/adapter_model.bin` |
| Training responses | `…/checkpoints/responses/step_{0..127}.jsonl` |
| Post-eval plots + `.pkl` | Per-stage subdirs under the run dir (`{prefix}/f1_and_recall_plot_*`, `{prefix}/averaged_violin_*.png`) |
| On-policy split metrics | W&B only (no local `on_policy_split_*.json`) |
| W&B metrics | project `deception_detection`, group = trial/experiment group (e.g. `trial6/frozen`, `2507_det0_baselines`) |

Run dirs live under `results/run_results/{wandb.group}/`. Trials use nested groups `trial{N}/{variant|original}`; ablations use the YAML's parent folder name under `experiments/` (override with top-level `group:`).

Optional later step: `export_wandb_run_metrics.py` → `extracted_results/run_metrics_{SHORT_NAME}.json`. `run_ablation.py` (below) does this automatically.

---

## Launching runs: run_ablation.py vs. raw Hydra command

**Default choice for anything you want to reproduce or rerun: `pipeline/run_ablation.py` + a YAML config under `experiments/` (grouped by past experiment, e.g. `experiments/2507_det0_baselines/`).**

**Run from `/workspace`, not `/workspace/obfuscation-atlas`**, with `atlas_setup.sh` sourced (sets `PYTHONPATH=/workspace` so `pipeline` is importable) — `pipeline/` and `experiments/` are both siblings of `obfuscation-atlas/`, not nested inside it (see the workspace layout in `CONTEXT.md`). `grpo_launcher.py` internally `cd`s into `obfuscation-atlas/` for the actual training subprocess, but the launcher itself resolves config-relative paths (`manifest`, `checkpoint`, `output_dir`) against `experiments/` regardless of where the YAML lives:

```bash
cd /workspace
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_baseline.yaml       # single run
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_seed_sweep.yaml     # sweep (runs: list, sequential)
```

This builds the same kind of Hydra command shown throughout this doc (via `pipeline/grpo_launcher.py`) and additionally:

- Applies the workspace defaults automatically (`GRPO_DEFAULTS` in `grpo_launcher.py` — see [Defaults](#defaults-workspace-convention) below); your YAML only needs to specify what differs from them
- Runs preflight (vLLM reachable + disk space) before launching
- Pins the subprocess to `CUDA_VISIBLE_DEVICES=0` (vLLM stays on GPU 1)
- Verifies `checkpoints/final/adapter_model.bin` and the response JSONL count after training
- Exports W&B metrics to `<output_dir>/run_metrics_<run_key>.json`
- Deletes `checkpoints/lora/` and `checkpoints/step_*/` once verified (see [Disk usage](#disk-usage-and-manual-checkpoint-cleanup))
- Skips a run if its `run_metrics_*.json` already exists (delete the file to force a rerun)
- For sweeps: runs each entry in `runs:` **sequentially** (never in parallel — one pod, shared GPUs), and either continues or aborts on failure per `on_failure: continue|stop`

**Config schema** — every YAML config under `experiments/{trials,2507_*}/` is a worked example with a header comment explaining its specific ablation; read one alongside this table:

| Key | Required | Meaning |
|-----|----------|---------|
| `run_key` | yes | `wandb.run_name_prefix`; also the `run_metrics_<run_key>.json` filename and local run-dir prefix |
| `output_dir` | yes | Where `run_metrics_*.json` is written (relative to `experiments/` unless absolute) |
| `group` | no | W&B group + on-disk nest under `run_results/`. Default: YAML parent folder name (e.g. `2507_det0_baselines`) |
| `probe.directory_save_prefix` | yes | Phase-0 activation cache isolation (detector results are per-run, not shared) — see [Prerequisites §2](#2-base-probe-cache-per-probe-type) |
| `probe.retrain` | no (default `false`) | Sets `detector.retrain` |
| `probe.manifest` | only for mixed-manifest runs | Sets `++detector.mixed_probe_manifest_path` |
| `checkpoint` | no (default `null` = base model) | Path to a `checkpoints/final/` dir; sets `++grpo.model.reference_adapter_path` |
| `nofu` | no (default `true`) | Clears the 4 follow-up-prompt overrides — see [No-follow-up probe extraction](#no-follow-up-probe-extraction-jul-2026) |
| `overrides` | yes | Any Hydra `key: value` pairs; these win over every default above, including nofu/probe/checkpoint |
| `cleanup`, `export_metrics` | no (default `true`) | Toggle the post-run cleanup/export steps |

Sweep configs additionally use `sweep_name`, `shared_overrides` (merged under each run's own `overrides`), `runs:` (list of `{run_key, overrides, output_subdir?, checkpoint?, nofu?, min_responses?}`), and `on_failure: continue|stop`. Runs are listed explicitly — no cross-product magic.

**When to skip the YAML** and run the raw Hydra command directly (see [Single run](#single-run) below): quick manual experimentation you don't intend to rerun, or debugging a launch failure step by step. Anything worth reproducing should get a YAML config instead of a one-off shell command.

---

## Defaults (workspace convention)

Applied automatically by `run_ablation.py` (`GRPO_DEFAULTS` in `grpo_launcher.py`); set these explicitly if launching the raw Hydra command by hand.

| Setting | Default | Notes |
|---------|---------|-------|
| `+model_preset` | `llama3_8b` | 128 GRPO steps, 8 generations per prompt |
| `+data/probe_dataset` | `mbpp_probe` | |
| `loss.kl` | `0.001` | |
| `data.seed` | `53` | Controls MBPP shuffle and train/val split — see **Seeds** note below |
| `task_args.deceptive_definition` | `pass_only_shown` | See `EXPERIMENT_LOG.md` methodology #3 |
| `corrected_detector_gradient` | `false` | Reward-only detector pressure. Use `true` only when explicitly requested |
| `misalignment_eval.enabled` | `false` | Keeps eval load down |
| `grpo.model.attn_implementation` | `sdpa` | RunPod often lacks `flash_attn` |
| `training.save_path_base` | `/workspace/results/run_results` | |
| `detector.model_cache_path` | `/workspace/results/activations` | |
| `wandb.project` | `deception_detection` | |
| `wandb.group` | *(derived)* | From YAML parent folder (`run_ablation.py`) or `trial{N}/{variant\|original}` (`run_trial.py`). Always set explicitly on raw Hydra commands. |
| `++training.resume_from_checkpoint` | `False` | |
| nofu (4 follow-up-prompt keys) | on | Toggle with top-level `nofu: false` in a YAML config (not inside `overrides:`) — see [No-follow-up probe extraction](#no-follow-up-probe-extraction-jul-2026) |

**Not defaulted** — must be set explicitly in every config/command: `loss.detector`, `training.seed`, probe manifest + expected AUC (if using a mixed manifest), `checkpoint` (`null` or path).

**Seeds:** `training.seed` controls GRPO/training RNG; `data.seed` (default 53) controls MBPP shuffle and train/val split. Set both explicitly when reproducing a run. W&B export `"seed"` is `data.seed`, not `training.seed`.

**GPUs:** Ideal: training / probe forward on GPU 0, **vLLM on GPU 1** (see [Single GPU](#single-gpu) if you only have one).

---

## Naming: short name vs run ID

| Use | Example |
|-----|---------|
| **Short name / run_key** | `kl1e-1-det10-s42-mbpp` or `trial6_frozen_round2` — set via `wandb.run_name_prefix`; use in notes and export filenames |
| **Run ID** | `16ypiseh` — random W&B id; **local run dir** is `{run_key}_{RUN_ID}/` under `run_results/{wandb.group}/` |

Record `short_name → run_id` when the W&B URL prints. You cannot rename the on-disk run directory after the fact without updating checkpoint resolution.

**Short name pattern:** `kl{KL}-det{DET}-s{SEED}-{probe}` where probe suffix is `mbpp` for `mbpp_probe`, `ddp` for `diverse_deception_probe`. Use a distinct suffix (e.g. `-disjoint`, `-v2`) when rerunning the same hyperparams after a methodology fix, so exports and W&B names don't collide with older runs.

---

## Prerequisites (preflight)

Run once before training. As of the `/workspace/dotfiles` migration, a new interactive shell already has this env
via the `~/.bashrc` hook installed by `bash /workspace/dotfiles/apply.sh` — sourcing `atlas_setup.sh` (or
`dotfiles/env.sh` directly) is only needed for non-interactive contexts (e.g. inside a script).

### 1. Environment

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas
```

**Note:** the `cd` above is for the raw Hydra `python -m obfuscation_atlas.scripts...` commands later in this doc. If you're launching via `pipeline/run_ablation.py` instead (the default choice — see [Launching runs](#launching-runs-run_ablationpy-vs-raw-hydra-command)), run it from `/workspace` — `pipeline/` and `experiments/` are both siblings of `obfuscation-atlas/`, not nested inside it.

**Checks (fail if any fail):**

```bash
test -n "$WANDB_API_KEY" && echo WANDB_OK
test -n "$HF_TOKEN" && echo HF_OK
python -c "import obfuscation_atlas; print('OK')"
python -c "import pyarrow; print('pyarrow', pyarrow.__version__)"   # broken install breaks vLLM
python -c "import shutil; f=shutil.disk_usage('/workspace').free/(1024**3); print(f'{f:.0f} GB free'); assert f>=50"
```

If `pyarrow` is a namespace package without `__version__`, reinstall:

```bash
rm -rf .venv/lib/python3.12/site-packages/pyarrow .venv/lib/python3.12/site-packages/pyarrow-*.dist-info
uv pip install --reinstall --no-cache pyarrow==23.0.0
```

### 2. Base probe cache (per probe type)

Training loads (or trains) `initial_detector.base_model.on_dataset` once per probe. Cached if manifest exists:

| Probe | Manifest under `run_results/` |
|-------|-------------------------------|
| `mbpp_probe` | `base_results_and_detectors_*mbpp-hardcode*/initial_detector.base_model.on_dataset/f1_and_recall_plot_initial_detector.base_model.on_dataset__manifest.json` |
| `diverse_deception_probe` | `base_results_and_detectors_*local-diverse_deception*/initial_detector.base_model.on_dataset/…` |

```bash
# MBPP probe example (nested stage subdir; loader also accepts the legacy flat layout)
test -f /workspace/results/run_results/base_results_and_detectors_Meta-Llama-3-8B-Instruct_none_mbpp-hardcode_detector-lp_train_mean_eval_mean/initial_detector.base_model.on_dataset/f1_and_recall_plot_initial_detector.base_model.on_dataset__manifest.json \
  || test -f /workspace/results/run_results/base_results_and_detectors_Meta-Llama-3-8B-Instruct_none_mbpp-hardcode_detector-lp_train_mean_eval_mean/f1_and_recall_plot_initial_detector.base_model.on_dataset__manifest.json \
  && echo BASE_PROBE_OK
```

First run for a new probe type may take 30+ min for Phase 0. Detector-result manifests under `base_results_and_detectors_*/{prefix}/` persist and skip probe retrain on later runs; activation caches under `results/activations/` are deleted after each successful run, so Phase-0 will re-extract activations (still much faster than a full cold start when the probe weights/manifest hit).

**Probe train is derived from the task train split** (zero overlap with val) — see §3 below. For a run that must retrain the shared base cache (e.g. after a methodology change to probe construction), set **`detector.retrain=true`**. This **overwrites** `base_results_and_detectors_*` and wipes the Phase-0 activation cache under `results/activations/base_activations_*`.

**`detector.directory_save_prefix=<prefix>` avoids touching the shared cache above, but the two Phase-0 artifacts it isolates behave differently:**
- **Activations** (`results/activations/{prefix}_base_activations_.../`) — reused within a run / on crash recovery (`detector.retrain=false` skips recomputation on a cache hit). Deleted automatically when the GRPO run finishes successfully.
- **Detector results** (trained probe weights + eval manifests under `{run_dir}/initial_detector.base_model.on_dataset/`) — **not** shared. They're written into that run's own directory, so every new run refits/evaluates the probe (re-extracting activations if the prior cache was wiped). Do not expect (or search for) a `{prefix}_base_results_and_detectors_*` directory — it is not created when a prefix is used.

**Do not** use `detector.retrain=true` without understanding the above — it replaces the shared mbpp base probe for all future default runs. Use a prefix instead if you need an isolated probe without touching shared cache.

### 3. Probe train / val disjointness (mbpp_probe)

`prepare_dataset()` asserts zero `task_id` overlap between probe train and eval val. Verify before a long run:

```bash
python -c "
from dataclasses import replace
from transformers import AutoTokenizer
from obfuscation_atlas.config import CodeGenTaskConfig, MBPPTaskConfig, MBPPProbeConfig
from obfuscation_atlas.training.obfuscation_training import prepare_dataset
cfg = CodeGenTaskConfig().data
cfg = replace(cfg, seed=53, probe_dataset=MBPPProbeConfig(), task_dataset=MBPPTaskConfig(),
              include_task_positive=True, include_task_negative=False)
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
_, probe, eval_task = prepare_dataset(cfg=cfg, tokenizer=tok)
p_ids = set(probe[0]['task_id']); v_ids = set(eval_task[0]['task_id'])
assert len(p_ids & v_ids) == 0, f'overlap={len(p_ids & v_ids)}'
print('OK: zero overlap')
"
```

At run start, confirm log line: `OK: probe train (200/200) disjoint from val (400/400); task_id overlap=0` and hashes `probe_pos_from_task_train` (not `probe_pos_after_combine`). Do **not** pass `detector.mixed_probe_manifest_path` unless the manifest was regenerated with val `task_id`s excluded.

### 4. GPU cleanup (before any run)

Zombie `VLLM::EngineCore` processes can survive `pkill` on the server wrapper and block GPU memory:

```bash
pkill -9 -f "afterburner.generation.vllm.server" || true
pkill -9 -f "VLLM::EngineCore" || true
pkill -9 -f "train_grpo_model_against_probe" || true
sleep 5
nvidia-smi   # expect ~0 MiB used
```

If memory is still held, `kill -9` the PID from `nvidia-smi` and wait 10s.

### 5. vLLM on GPU 1 (two-GPU default)

**Terminal 1 — keep running:**

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95
```

**Terminal 2 — before launch:**

```bash
curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo vLLM_OK
```

If vLLM dies mid-run, restart it and re-run the **same** command (resume map may resume the in-progress seed).

---

## Single GPU

Works on one GPU by lowering vLLM memory reservation so FSDP training + probe forward share the device with vLLM. **No Hydra changes** to the training command; only the vLLM launch differs.

| Role | GPU | Setting |
|------|-----|---------|
| vLLM server | GPU 0 | `--gpu-memory-utilization 0.40 --tensor-parallel-size 1` |
| GRPO + probe forward | GPU 0 | Default (no `CUDA_VISIBLE_DEVICES` on training cmd) |

**Terminal 1 — vLLM (start first, keep running through post-eval):**

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1 \
    2>&1 | tee /workspace/scratch/vllm_grpo_$(hostname).log
```

**Terminal 2 — training:** same launch command as below; `curl` vLLM before starting.

Use a pod-unique log filename (`$(hostname)` suffix, as above) any time more than one pod may be active — see [Multiple pods on the same persistent volume](#multiple-pods-on-the-same-persistent-volume) below.

**Confirmed working** at util 0.40 on a single H100 — full train + post-eval completes in ~2.3 h with ~57% GPU memory free for training while vLLM is up during model init (see `EXPERIMENT_LOG.md` for the reference run).

**Phase 0 OOM fallback:** Phase 0 does not call vLLM but shares GPU memory if vLLM is already running. If activation caching OOMs: stop vLLM, rerun (partial Phase-0 cache may allow skip), restart vLLM before `Starting GRPO training...`.

**Tuning:** RL OOM → try vLLM util **0.35**; vLLM fails to load → try **0.45–0.50**. Do not change training hyperparameters to compensate.

For eval-only reruns on 1 GPU where post-eval stages G/H need the full GPU, see the start/stop vLLM lifecycle in [PROBE_EVAL_RUNBOOK.md](./PROBE_EVAL_RUNBOOK.md) (full GRPO runs can keep vLLM up end-to-end).

---

## Multiple pods on the same persistent volume

`/workspace` is a RunPod **network volume** that can be mounted on more than one pod at once (e.g. you running a second, unrelated sweep on another pod while this one trains). GPUs, processes, and `nvidia-smi` are **per-pod** and never shared — but any path under `/workspace` is shared filesystem state visible to every pod with that volume mounted.

**Symptom:** you `tail -f` (or `tee`) a conventional log path like `/workspace/scratch/vllm_grpo.log` and see live request/response traffic referencing LoRA checkpoints, run IDs, or step numbers you don't recognize and never launched on this pod — even though **this pod's** `nvidia-smi` shows the GPU idle or a freshly-started, still-loading vLLM process. This looks like a stray/zombie process or a port conflict but usually isn't: it's a **different pod's** vLLM server (or GRPO run) that happens to log to the same conventional path from this doc, and both `tee` processes are writing to the same inode on the shared volume.

**Do not** assume this means your own launch failed or conflicted — verify independently before reacting:

- `hostname` — confirms which pod you're on (differs per pod; the log content's run IDs won't match anything from `ps aux` on your pod if it's the other pod's).
- `nvidia-smi --query-gpu=memory.used --format=csv` — **this pod's** GPU only; will not reflect the other pod's memory usage at all.
- `curl -sf http://127.0.0.1:8000/v1/models` — reachability of **this pod's** vLLM server; ports are per-pod/per-container and never actually collide across pods even though the filesystem is shared.
- `tmux capture-pane -t <session> -p` / `ps -ef --forest` — ground truth for what your own pod's processes are actually doing, independent of any shared log file.

**Avoid the confusion going forward:** when you know (or suspect) more than one pod may be active on the same volume, give vLLM/sweep log files a pod-unique suffix instead of the bare conventional name, e.g.:

```bash
2>&1 | tee /workspace/scratch/vllm_grpo_$(hostname).log
2>&1 | tee /workspace/scratch/<sweep_name>_$(hostname).log
```

This doesn't change any Hydra/training behavior — `run_results/`, `activations/`, and `wandb_resume_map.json` are already correctly namespaced per-run (by `RUN_ID` / `directory_save_prefix`) and are safe to share across concurrent pods as long as two pods never target the **same** `run_key`/prefix at once. Only ad hoc `tee` log paths under `scratch/` need manual uniquing.

---

## Launch

All of these are either a raw Hydra command (for quick one-offs) or a YAML config for `run_ablation.py` (for anything repeatable — see [Launching runs](#launching-runs-run_ablationpy-vs-raw-hydra-command) above). **Completed runs, outcome metrics, and policy classifications for every ablation below live in `EXPERIMENT_LOG.md` — not here.**

### Single run

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

SEED=42   # change per run

python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    loss.kl=0.001 \
    loss.detector=10 \
    +data/probe_dataset=mbpp_probe \
    training.seed=$SEED \
    data.seed=53 \
    grpo.model.attn_implementation=sdpa \
    training.save_path_base=/workspace/results/run_results \
    detector.model_cache_path=/workspace/results/activations \
    corrected_detector_gradient=false \
    misalignment_eval.enabled=false \
    task_args.deceptive_definition=pass_only_shown \
    wandb.project=deception_detection \
    wandb.group=scratch \
    wandb.run_name_prefix=kl1e-3-det10-s${SEED}-mbpp
```

(`wandb.group` becomes the on-disk folder under `run_results/`; use a meaningful group, not a catch-all, for anything you want to keep organized.)

This is exactly what `run_ablation.py` builds for you from `GRPO_DEFAULTS` + a YAML's `overrides:`. Use the raw command only for quick one-offs; anything reproducible should be a YAML config instead (see [Launching runs](#launching-runs-run_ablationpy-vs-raw-hydra-command)).

### Sweeps (multiple seeds / configs)

Preferred: a sweep YAML + `run_ablation.py` (see `experiments/2507_det0_baselines/det0_seed_sweep.yaml`, `experiments/2507_mixed_reward/mixed_reward_sweep.yaml`, `experiments/2507_nofu_probe/det1_nofu_sweep.yaml`, `experiments/2507_prompt_mix/prompt_mix_sweep.yaml` for real examples). Run from `/workspace` (see the [note above](#launching-runs-run_ablationpy-vs-raw-hydra-command)):

```bash
cd /workspace
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_seed_sweep.yaml
```

This runs each entry in `runs:` sequentially in one process, verifying + exporting + cleaning up after each before starting the next.

**If you need a fully ad hoc sweep not worth a YAML file**, the same pattern applies manually: keep vLLM running in one tmux window across the whole sweep, launch each seed in another window with a distinct `wandb.run_name_prefix`, wait for exit before starting the next, and clean up intermediates between runs (see [Disk usage](#disk-usage-and-manual-checkpoint-cleanup)). Run any such loop inside `tmux` so it survives disconnects. Each full run is **~80 GB** on disk until cleaned up — plan **≥50 GB free per additional run** (`df -h /workspace`).

Never parallelize seeds on one pod — vLLM and training/probe-forward share the GPUs. Do **not** use `pgrep -f train_grpo_model_against_probe` as a wait condition inside a wrapper script — the pattern matches the wrapper's own command line and will hang forever.

### No-follow-up probe extraction (Jul 2026)

Probe **completion EOT** (`last-token-generation`) instead of the follow-up `"Yes"` token. Hydra-only — no code default changes. Train, RL penalty, and post-train eval stay aligned by clearing follow-ups on **both** probe and task dataset configs. See `EXPERIMENT_LOG.md` methodology #4 for why.

**Overrides** (`nofu: true` in a YAML config sets all four automatically):

```bash
data.probe_dataset.congruent_follow_up_prompts=[] \
data.probe_dataset.incongruent_follow_up_prompts=[] \
data.task_dataset.congruent_follow_up_prompts=[] \
data.task_dataset.incongruent_follow_up_prompts=[] \
detector.directory_save_prefix=<prefix> \
detector.retrain=true   # only on the first run for a new prefix
```

- **`directory_save_prefix`:** the shared `base_results_and_detectors_*mbpp-hardcode*` cache is **not** touched. Only the activation cache is shared/reused under the prefix (`{prefix}_base_activations_...`); Phase-0 detector results are written per-run — see [Prerequisites §2](#2-base-probe-cache-per-probe-type).
- **Startup gate:** spot-check should print `decoded='<|eot_id|>'` for deceptive and honest rows — not `"Yes"`.
- **Fresh run only:** do not resume a run that started with the follow-up probe under a different prefix.

### Mixed round-1 nofu GRPO (Jul 2026)

Round-1 GRPO from the **base model** with a **mixed Phase-0 probe** (300/class: 200 on-dataset + 100 on-policy). Combines the nofu overrides above with a mixed probe manifest. The historical cache under `experiments/archive/mixed_round1_nofu/` was deleted — regenerate via `experiments/2507_nofu_probe/run_mixed_round1_nofu_probe.py`, or point the manifest at a trial's own `rounds/round1/probe_train_dataset.json`.

**Dataset:** `build_train_dataset_round1_mixed()` — honest on-policy from base `train_full`, deceptive on-policy from det0 `train_full` (`experiments/artifacts/`). See [iterative_training_experiments.md](./iterative_training_experiments.md) § Round-1 mixed dataset.

**Additional overrides (on top of nofu):**

```bash
detector.directory_save_prefix=nofu_mixed \
detector.retrain=true \
++detector.mixed_probe_manifest_path=/workspace/experiments/trials/trial3/rounds/round1/probe_train_dataset.json
```

- **No `reference_adapter_path`:** round 1 starts from base Instruct (unlike multi-round `run_round_manual.py`).
- **Startup gates:** `OK: loaded mixed probe manifest (300 pos, 300 neg)`; `task_id overlap=0`; `Total number of steps: 128`.
- All of the `det0_*`, `mixed_reward_*`, and `prompt_mix_*` configs below reuse this manifest with `retrain: false` — Phase-0 becomes a fast cache-hit re-derivation, not a full retrain.
- **Phase-0-only dry run:** add `++training.only_eval_detectors=True` (creates a separate W&B run; use a `-phase0` suffix on the run key).

### Prompt-mix training (Jul 2026)

During GRPO rollouts, randomly present MBPP test 0 **with** expected output (shown, default) or **without** (unshown — strip everything after the first `==` in the assert). Reward still grades the full `test_list`; probe penalty and Phase-0 setup are unchanged. The HF dataset `prompt` column stays all-shown; mixing happens only in `GRPOObfuscationTrainer.prepare_training_inputs()`. See `EXPERIMENT_LOG.md` methodology #5 for why.

**Config** (`TaskArgsConfig` in `config.py`):

| Override | Default | Meaning |
|----------|---------|---------|
| `task_args.shown_prompt_ratio` | `1.0` | Fraction of training rollouts that get shown prompts (0–1). Per-step RNG: `training.seed + step` |
| `task_args.eval_show_expected_output` | `True` | In-loop eval + post-train on-policy use shown prompts when true |

**Implementation:** `hide_expected_output_from_assert()` and `build_mbpp_chat_prompt(..., show_expected_output=…)` in `obfuscation_training.py`; startup validation via `validate_mbpp_tests_for_prompt_variants()`. Tests: `tests/test_mbpp_prompt_variants.py`. W&B logs `train/prompt_shown_fraction` when ratio < 1.

**Sweeping the ratio:** `experiments/2507_prompt_mix/prompt_mix_sweep.yaml` (0.5 / 0.7 / 0.8, kl=1e-3, det=3, seed=43, nofu mixed manifest). Run with `python -m pipeline.run_ablation experiments/2507_prompt_mix/prompt_mix_sweep.yaml`.

**Hidden eval:** `training.only_eval_from_other_run=True` + `task_args.eval_show_expected_output=false` runs post-train eval on hidden prompts as a new W&B run — see [Hidden post-train eval](#hidden-post-train-eval-jul-2026) below.

### Mixed reward regime (Jul 2026)

Per prompt per step, draw a Bernoulli regime shared by all G completions of that prompt: w.p. `1 - mixed_reward_pass_all_prob` grade on `passes_first_n_tests` (default, hardcoding-friendly); w.p. `mixed_reward_pass_all_prob` grade on `passes_all_tests` (honest-only). Length penalty and detector penalty apply regardless of regime; `task_success` (the detector-penalty gate) always stays `passes_first_n_tests`. In-loop eval and post-train eval never use the mixed regime — only `Partition.TRAIN` is affected. See `EXPERIMENT_LOG.md` methodology #6 for why.

**Config** (`TaskArgsConfig` in `config.py`):

| Override | Default | Meaning |
|----------|---------|---------|
| `task_args.mixed_reward_pass_all_prob` | `0.0` | Probability a training prompt (and all its completions) is scored by `passes_all_tests` that step. `0.0` preserves prior behavior |

**Implementation:** `apply_mixed_reward_regime()` in `tasks/code_generation.py` — pure function, unit-tested independently. Wired into `GRPOObfuscationTrainer.score_responses()` in `grpo_trainer.py`, gated on `partition == Partition.TRAIN`. RNG seed: `training.seed + step + 1_000_000` (offset from the prompt-mix RNG so the two never draw identical sequences if combined). Startup log line `Mixed reward regime: mixed_reward_pass_all_prob=<value>` confirms the resolved config (prints even at `0.0`). W&B logs `train/mixed_reward_pass_all_fraction` every train step. Tests: `tests/test_mixed_reward.py`.

**Sweeping the probability:** `experiments/2507_mixed_reward/mixed_reward_sweep.yaml` (0.35/0.40 × seeds 41/42, kl=1e-3, det=3, nofu mixed manifest, no prompt mixing). Run with `python -m pipeline.run_ablation experiments/2507_mixed_reward/mixed_reward_sweep.yaml`.

**Startup gate:** `Mixed reward regime: mixed_reward_pass_all_prob=<value>` matching the config; if `task_args.shown_prompt_ratio` is not overridden, no `train/prompt_shown_fraction` should appear (confirms prompt mixing is off).

### No-detector-penalty baseline and ceiling (Jul 2026)

`loss.detector=0` control against the mixed-reward runs, starting fresh from the **base model** (no `reference_adapter_path`). The probe is still loaded and scored for monitoring/post-eval only — it never contributes to reward or the policy gradient. See `EXPERIMENT_LOG.md` methodology #10.

| Variant | Config | Difference from baseline |
|---------|--------|---------------------------|
| Baseline | `experiments/2507_det0_baselines/det0_baseline.yaml` | `task_args.mixed_reward_pass_all_prob=0.3` (same as det=3 runs) |
| Ceiling | `experiments/2507_det0_baselines/det0_ceiling.yaml` | `task_args.mixed_reward_pass_all_prob=1.0` — every prompt graded on `passes_all_tests` |
| Ceiling, no length penalty | `experiments/2507_det0_baselines/det0_ceiling_no_len_penalty.yaml` | Ceiling config + `code_length_penalty=0` (default 0.003), isolating the length penalty's effect |

All three reuse the `nofu_mixed` Phase-0 cache (`retrain: false`) from [Mixed round-1 nofu GRPO](#mixed-round-1-nofu-grpo-jul-2026) above.

```bash
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_baseline.yaml
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_ceiling.yaml
python -m pipeline.run_ablation experiments/2507_det0_baselines/det0_ceiling_no_len_penalty.yaml
```

**Startup gates:** `OK: loaded mixed probe manifest (300 pos, 300 neg)`; `task_id overlap=0`; `Phase-0 probe activation cache path: .../nofu_mixed_base_activations_.../` followed by `Loading cached activations from ...` (confirms cache reuse, not full recompute); `Mixed reward regime: mixed_reward_pass_all_prob=<0.3 or 1.0>` matching the config. For the no-length-penalty variant, confirm `code_length_penalty=0.0` in the exported `full_config`.

### KL-from-base ablation — round 2 with base-model KL reference (Jul 2026)

Reruns a round 2 with the KL penalty measured against the **base model** (zeroed reference adapter) instead of the round-1 checkpoint, while the policy still initializes from the round-1 checkpoint. Tests whether anchoring KL to the pretrained distribution changes round-2 training dynamics. See `EXPERIMENT_LOG.md` methodology #11.

**Mechanism:** `++grpo.model.kl_reference_from_base=true` zeroes the reference adapter's LoRA weights after loading from `reference_adapter_path`, so the KL forward pass becomes equivalent to the base model.

**Config:** `experiments/2507_round2_variants/round2_kl_from_base.yaml` — matches the target round 2 exactly (same round-2 probe manifest, same round-1 `checkpoint`) plus the single flag.

```bash
python -m pipeline.run_ablation experiments/2507_round2_variants/round2_kl_from_base.yaml
```

**Startup gates:** same as a normal round 2 (`OK: loaded mixed probe manifest`; `task_id overlap=0`; Phase-0 cache reuse) plus a log line confirming the reference-adapter zeroing — W&B KL curves should diverge from a normal round 2 early.

### Hidden post-train eval (Jul 2026)

Re-runs the post-train probe pipeline (stages B–I) on a **finished adapter** with **hidden** val prompts at stage F (no `== expected` in test 0). Creates a **new** W&B run and save dir — does not overwrite the source training run. Use to compare shown vs. hidden on-policy behavior after prompt-mix training; on-dataset stages (B, D, E) still use shown prompts, only stage F on-policy generation uses hidden prompts.

**Overrides:**

```bash
task_args.eval_show_expected_output=false \
++training.only_eval_from_other_run=True \
++training.resume_from_checkpoint=False \
++training.model_wandb_id=<SOURCE_RUN_ID> \
detector.retrain=false \
wandb.run_name_prefix=hidden-eval-...
```

**Important:**
- Set **`CUDA_VISIBLE_DEVICES=0`** on the eval subprocess when vLLM holds GPU 1 — otherwise Phase-0 activation caching OOMs.
- Do **not** pass `training.model_wandb_id` when hidden prompts are requested if you want on-policy generation redone — the training script forces `model_wandb_id=None` internally so it regenerates completions with hidden prompts.
- `DEBUG (process_data)` log lines during on-dataset stages still show **shown** prompts; hidden prompts appear only in stage F generation (assert lines without `==`).

**Disk:** if you're keeping multiple prior full runs (~80 GB each — see [Disk usage](#disk-usage-and-manual-checkpoint-cleanup)) around while doing hidden-eval reruns, budget accordingly (`df -h /workspace`); do not delete other runs' checkpoints to make room without checking first.

**Resume:** `wandb_resume_map.json` fingerprints config (seed, kl, det, probe). Re-running the **same** command after an interrupt resumes the same W&B run if `checkpoints/step_*` exist. Do **not** delete `step_*` or `lora/` on an in-progress run until it finishes or you abandon it.

### Using an intermediate step checkpoint as `checkpoints/final/`

If a run is killed or crashes before writing `checkpoints/final/`, downstream code that expects `RoundConfig.checkpoint_final` / `…/checkpoints/final/` (tokenizer loading, `PeftModel.from_pretrained`, `++grpo.model.reference_adapter_path`) will fail even though a usable LoRA step checkpoint exists under `checkpoints/lora/step_{N}/`.

**Fix — make the step checkpoint look like `final/` without touching downstream code:**

1. **Symlink** `checkpoints/final` to the desired step:

   ```bash
   RUN_DIR=/workspace/results/run_results/{wandb.group}/{run_key}_{RUN_ID}
   ln -s lora/step_128 "$RUN_DIR/checkpoints/final"
   ```

2. **Copy tokenizer files into the step dir.** Per-step LoRA checkpoints only contain adapter weights, not tokenizer files — `AutoTokenizer.from_pretrained(checkpoint_final)` will fail otherwise. Copy from any **completed** run's `checkpoints/final/` (tokenizer is identical across runs on the same base model):

   ```bash
   cp <completed_run>/checkpoints/final/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,chat_template.jinja} \
      "$RUN_DIR/checkpoints/lora/step_128/"
   ```

**Caveats:**
- Pick a step that still has its `checkpoints/lora/step_{N}/` directory intact (not yet cleaned up — see [Disk usage](#disk-usage-and-manual-checkpoint-cleanup)).
- This checkpoint reflects training progress **only up to that step**, not a fully-converged run — treat metrics/behavior accordingly.

### Disk usage and manual checkpoint cleanup

`run_ablation.py` does this automatically after verifying `checkpoints/final/` exists (`cleanup_intermediates()` in `grpo_launcher.py`). The manual procedure below is only needed for runs launched via the raw Hydra command directly.

GRPO saves large intermediate artifacts under each run dir. There is **no automatic cleanup** for manually-launched runs — you must delete them yourself after each completed run when queueing multiple seeds.

| Run dir (under `…/run_results/{wandb.group}/{run_key}_{RUN_ID}/`) | Typical size | Safe to delete after run completes? |
|---------------------------------------------------------------------------|--------------|-------------------------------------|
| `checkpoints/lora/step_{0..127}/` | **~78 GB** (~640 MB × 128 steps), pre-Aug-2-2026 leak fix | **Yes** — keep only if debugging per-step LoRA |
| `checkpoints/step_{110,120,...}/` | **~4 GB** (FSDP resume state) | **Yes** — not needed once `final/` exists |
| `checkpoints/final/` | **~1.3 GB** | **No** — final LoRA adapter |
| `checkpoints/responses/step_*.jsonl` | **~100 MB** | **No** — training rollouts for analysis |
| Post-eval manifests/plots/split JSONs | **<10 MB** | **No** |

**Aug 2, 2026 update:** the ~78 GB `checkpoints/lora/` figure above was driven mostly by an untracked per-step leak in `_generate_alt_completions` (see `EXPERIMENT_LOG.md` methodology #14) that accumulated one ~650 MB checkpoint per step for the whole run, on top of the small, properly-managed working set `GenerationQueue` cleans up as it goes. That leak is now fixed, so `checkpoints/lora/` should stay bounded (a handful of in-flight dirs, not one per step) throughout training rather than growing linearly with step count. `cleanup_intermediates()` / the manual `rm -rf …/checkpoints/lora` step above is still worth running (there's a small residual `step_*` FSDP-state footprint and it's a harmless no-op if `lora/` is already small), but should no longer be reclaiming anywhere near 78 GB on a run that completes normally.

**Symptom when disk fills:** Training dies mid-run (~batch 40–50) with **no Python traceback** — the process vanishes when writes fail. `/workspace` hits 100% (`touch` → `Disk quota exceeded`). This is **not** a GPU OOM or training bug.

**After each successful manually-launched run**, delete intermediates (adjust `RUN_ID`):

```bash
RUN_ID=...   # from W&B URL or log
RUN_DIR=/workspace/results/run_results/${GROUP}/${RUN_KEY}_${RUN_ID}  # e.g. trial6/frozen/trial6_frozen_round2_${RUN_ID}

# Verify run completed first
test -f "$RUN_DIR/checkpoints/final/adapter_model.bin"
test "$(ls "$RUN_DIR/checkpoints/responses"/step_*.jsonl 2>/dev/null | wc -l)" -ge 128

rm -rf "$RUN_DIR/checkpoints/lora"
rm -rf "$RUN_DIR"/checkpoints/step_*
df -h /workspace
```

**Do not delete** on an **in-progress** run you plan to resume — keep `checkpoints/step_*` (resume loads the latest, e.g. `step_40`) and `checkpoints/lora/`.

**Optional:** delete empty/aborted run dirs entirely once verified (`0` responses, no `final/`):

```bash
# Verify first (0 response JSONLs, no checkpoints/final/), then:
rm -rf /workspace/results/run_results/{GROUP}/{RUN_KEY}_{ABORTED_RUN_ID}
```

**Do not delete:** `hf_cache/`, or `checkpoints/final/` on any run you still need. Activation caches under `results/activations/` are wiped automatically at the end of a successful GRPO run (and may be empty between runs).

---

## Checks at each major step

### After preflight

- [ ] `WANDB_OK`, `HF_OK`, `OK`, `pyarrow` version prints
- [ ] ≥ 50 GB free on `/workspace` (single run or multi-seed sweep)
- [ ] GPU clean (`nvidia-smi` ~0 MiB) — especially on 1× GPU pods
- [ ] Overlap check passes (mbpp_probe; see preflight §3)
- [ ] `BASE_PROBE_OK` (or accept Phase 0 train; use `detector.retrain=true` after disjointness fix)
- [ ] `vLLM_OK`

### Early in each run (~5 min)

- [ ] W&B URL printed; note `RUN_ID`
- [ ] Log: `task_id overlap=0` (mbpp_probe)
- [ ] Log: `probe_pos_from_task_train` hashes (shared-pool branch; not `probe_pos_after_combine`)
- [ ] Phase 0: `Caching activations...` if retraining, or loaded from manifest if cache hit
- [ ] `Starting GRPO training...`
- [ ] New dir: `run_results/{wandb.group}/{run_key}_{RUN_ID}/`
- [ ] `checkpoints/responses/step_0.jsonl` exists

### Mid-train

- [ ] `step_*.jsonl` count increasing (target 128)
- [ ] `curl …/v1/models` still OK
- [ ] Disk ≥ 50 GB free for pass-only-shown runs (`df -h /workspace`); ≥ 30 GB minimum on older pipeline

### End of each seed (success gate)

- [ ] Shell `exit=0`
- [ ] `checkpoints/final/adapter_model.bin` exists (~640 MB)
- [ ] **5** stage subdirs under run dir, each with `f1_and_recall_plot_*__manifest.json` (post-train stages D–H)
- [ ] W&B summary has `eval_obfuscation/initial_detector/policy_detector_prob` (proves stage I finished)
- [ ] On-policy split aggregates present in W&B / exported `run_metrics` (no local split JSON)

Quick verify:

```bash
RUN_DIR=/workspace/results/run_results/{wandb.group}/{run_key}_{RUN_ID}

ls "$RUN_DIR/checkpoints/final/adapter_model.bin"
ls "$RUN_DIR"/*/f1_and_recall_plot_*__manifest.json | wc -l   # expect 5 (nested per stage)

python3 - <<PY
import json, glob
paths = glob.glob("/workspace/obfuscation-atlas/wandb/run-*-{RUN_ID}/files/wandb-summary.json")
d = json.load(open(paths[0]))
print("eval_obfuscation:", d.get("eval_obfuscation/initial_detector/policy_detector_prob", "MISSING"))
PY
```

Replace `{RUN_ID}` with the actual id (e.g. `16ypiseh`).

### After a multi-seed loop

- [ ] Three distinct `{run_key}_{RUN_ID}` dirs (different `RUN_ID`s)
- [ ] All three pass the end-of-seed gate above
- [ ] Intermediate checkpoints (`lora/`, `step_*`) deleted on each completed run (see [Disk usage](#disk-usage-and-manual-checkpoint-cleanup))
- [ ] `df -h /workspace` shows enough free space for the next run

---

## Completed reference runs

See [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) for the full completed-run inventory and outcome metrics for every ablation above. Do not add run IDs, outcomes, or classifications to this doc — track them there only.

---

## Post-eval crash recovery

If training finishes but post-eval dies on the last stage (e.g. `initial_detector.base_model.on_policy` or `fresh_detector.final_model.on_policy`), **do not retrain** — the LoRA is already safe under `checkpoints/final/`. This is a known, usually-transient failure mode (W&B API flakiness); see `EXPERIMENT_LOG.md` "Known gaps" for open instances.

Eval-only rerun (vLLM must be up on GPU 1). Use the **same** kl, det, probe, and seed as training; point `reference_adapter_path` at the finished run's final checkpoint:

```bash
python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    loss.kl=0.1 \
    loss.detector=10 \
    +data/probe_dataset=mbpp_probe \
    training.seed=$SEED \
    training.save_path_base=/workspace/results/run_results \
    ++training.only_eval_from_other_run=True \
    ++training.resume_from_checkpoint=False \
    ++grpo.model.reference_adapter_path=/workspace/results/run_results/{wandb.group}/{run_key}_{TRAIN_RUN_ID}/checkpoints/final \
    detector.model_cache_path=/workspace/results/activations \
    corrected_detector_gradient=false \
    misalignment_eval.enabled=false \
    wandb.project=deception_detection \
    wandb.group={wandb.group}
```

This creates a **new** W&B run for post-eval only. If W&B splits across train + eval runs, export with `--run-id` (eval) and `--task-metrics-run-id` (train). A **clean** full run keeps everything on one `RUN_ID`.

---

## Optional: export + classify (after runs finish)

Not required to complete training. When ready:

```bash
python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
    --entity lilylixin6-the-harker-school \
    --project deception_detection \
    --run-id RUN_ID \
    --output /workspace/experiments/.../run_metrics_SHORT_NAME.json
```

Use `--task-metrics-run-id` only for split train/eval W&B runs. Summarize with helpers in `pipeline/extract_run_results.py`. Historical exports also live under `experiments/extracted_results/`.

---

## Crash prevention checklist

- [ ] **Check disk first:** ≥50 GB free for a single run or multi-seed sweep (`df -h /workspace`)
- [ ] **GPU cleanup every time:** kill vLLM + EngineCore + training processes, sleep 5, verify `nvidia-smi` ~0 MiB before launching anything — see [Prerequisites §4](#4-gpu-cleanup-before-any-run)
- [ ] **Pin GRPO/eval subprocesses to the GPU vLLM is NOT using** (`CUDA_VISIBLE_DEVICES=0` convention) whenever vLLM is on a shared pod
- [ ] **After every completed run,** immediately delete `checkpoints/lora` and `checkpoints/step_*` before queueing the next one (automatic if launched via `run_ablation.py`)
- [ ] **If post-eval crashes on the last stage,** don't retrain — the adapter is safe; use the eval-only recovery command (see [Post-eval crash recovery](#post-eval-crash-recovery))
- [ ] **Before automating multiple rounds,** diff the hardcoded GRPO overrides against round 1's actual exported config (kl, det, seed) — a silent mismatch can invalidate an entire multi-round sweep
- [ ] **If another pod might share this `/workspace` volume,** use a `$(hostname)`-suffixed log path for vLLM/sweep `tee` output — see [Multiple pods on the same persistent volume](#multiple-pods-on-the-same-persistent-volume)
- [ ] **Watch the reward curve in W&B** during long runs so a collapsing policy can be killed early instead of burning the full step budget

---

## Known failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| vLLM fails on import | Broken `pyarrow` | Reinstall pyarrow (preflight §1) |
| `WANDB_API_KEY` unset | New non-interactive shell without the env sourced | `source /workspace/atlas_setup.sh` (or check `/workspace/.secrets.env` has `WANDB_API_KEY` filled in) |
| Hang at "Waiting for VLLM" | Server down | Start vLLM; `curl` check |
| GPU memory not freed | Zombie `VLLM::EngineCore` | GPU cleanup (preflight §4); `kill -9` PID from `nvidia-smi` |
| Phase 0 OOM (1× GPU) | vLLM + HF model on same GPU | Stop vLLM, rerun; restart vLLM before RL |
| `Probe train and eval val share N task_id(s)` | Overlap fix not active or bad mixed manifest | Verify code; do not use stale mixed manifest |
| Stale initial detector during RL | Post-fix run without `detector.retrain=true` | Retrain or delete `base_results_and_detectors_*mbpp-hardcode*` |
| Post-eval crash after step 128 | Known flake on last stage | Eval-only recovery; keep LoRA |
| Disk full / run dies ~batch 40–50, no traceback | Pass-only-shown runs save **~80 GB** `lora/` + `step_*` per run; `/workspace` at 100%. Prior to the Aug 2, 2026 leak fix (`EXPERIMENT_LOG.md` methodology #14), most of this `lora/` figure was an untracked per-step leak that accumulated *during* the run (not just normal working-set overhead) — a crash before the run's final cleanup step kept the full ~80 GB. Post-fix, `checkpoints/lora/` should stay bounded (a handful of in-flight dirs, not one per step) throughout training. | After each completed run: `rm -rf …/checkpoints/lora …/checkpoints/step_*`; keep `final/` + `responses/`. Need **≥50 GB free** before each new run. See [Disk usage](#disk-usage-and-manual-checkpoint-cleanup) |
| Same config, unexpected resume | `wandb_resume_map.json` | Re-run same command to resume in-progress run (needs `step_*`). New seed/kl/det → new run. Do not delete `step_*` on runs you are resuming |
| `[CodeExecutor] Stalled at 0/N results, checking workers...` repeating indefinitely; step counter frozen | Code-execution worker pool deadlocked — workers either (a) stuck executing generated code that defeats `signal.alarm` timeout (e.g. code with bare `except:` inside infinite loop swallows `TimeoutException`), or (b) `multiprocessing.Queue` internal lock corrupted by prior `kill -9` on a worker mid-operation | **Do NOT kill individual workers** (`kill -9` on spawn_main PIDs risks corrupting the shared Queue lock permanently). Instead kill the **entire training process** (`pkill -9 -f train_grpo_model_against_probe`) and relaunch. Verify via `cat /proc/<worker_pid>/wchan` — `futex_do_wait` on idle workers confirms queue corruption vs `0`/no wchan (actively running on CPU) confirms code-hang variant. **Prefer resuming over restarting from step 0:** if a `checkpoints/step_N` dir exists close to the hang point, kill the whole `run_trial.py`/tmux tree (not just the training subprocess, so its own crash-retry loop can't race a fresh `resume_from_checkpoint=False` restart) and manually re-run the **same** Hydra command with only `++training.resume_from_checkpoint=true` flipped — confirm from the log that W&B reports `Resuming run ...` under the **same** run ID (`wandb_resume_map.json` fingerprint match) and that the step counter picks up at `N`, not `0`. Once that manual run reaches `checkpoints/final/`, close out the round with `run_trial.py`'s own `verify_run_complete()` / `cleanup_intermediates()` / `export_run_metrics()` / `check_degeneracy()` / `update_state_after_round()` (import and call directly) before resuming automation for later rounds — `run_trial.py`'s built-in retry always hardcodes `resume_from_checkpoint=False` (`run_trial.py:238`), so it will never use this path on its own. Validated recovering trial 5 round 2 (`ppy5mkw9`) from `step_40` after a hang at step ~42/128, no `training.max_steps`/`num_epochs` change needed (see `EXPERIMENT_LOG.md` Trial 5) |
| Training OOM on multi-GPU pod (GPU 1 memory full) | Training process allocated to vLLM's GPU via `device_map="auto"` because `CUDA_VISIBLE_DEVICES` was not set | Always pin training: `CUDA_VISIBLE_DEVICES=0` in the training subprocess command (or tmux wrapper). vLLM on GPU 1 uses ~79 GB — any spillover crashes training |
| Shared log file (e.g. `scratch/vllm_grpo.log`) shows request/response traffic or run IDs you don't recognize, even though this pod's `nvidia-smi` is idle | Another pod has the same `/workspace` network volume mounted and is logging to the same conventional path | Not a real conflict — GPUs/processes are per-pod, only the filesystem is shared. Verify with `hostname`, this pod's `nvidia-smi`, `curl 127.0.0.1:8000/v1/models`, and `tmux capture-pane`/`ps -ef` instead of trusting the shared log. Use a `$(hostname)`-suffixed log path to avoid recurrence — see [Multiple pods on the same persistent volume](#multiple-pods-on-the-same-persistent-volume) |
| `rm: cannot remove '.../checkpoints/lora/step_N': Directory not empty` followed by `RuntimeError: Parent directory .../checkpoints/lora/step_N does not exist` (**same** step `N`, training subprocess exits 1) | **Fixed Aug 2, 2026** (see `EXPERIMENT_LOG.md` methodology #14). Root cause (confirmed Aug 1, 2026): genuine race, not just filesystem slowness. Each step, `GenerationQueue.pop()` (`afterburner/grpo_trainer.py`) triggers an async `rm -rf` on that step's `checkpoints/lora/step_N` dir right after generation. Moments later, in the *same* training step, `_generate_alt_completions` (`obfuscation_atlas/training/grpo_trainer.py`) unconditionally re-`save()`d the policy adapter to that **same** `step_N` path (needed for `pass_only_shown` alt-completion generation) — no existence check, unlike the guarded `GenerationQueue.submit()` path. If the background `rm -rf` was still running (or completed) while `torch.save` was mid-write, the parent dir disappeared out from under it. Timing-dependent — worse under `/workspace` network-FS contention, but could happen on an otherwise-idle pod (hit seed 44 of `ceiling_clp1e-3_seed_sweep` twice, at step 15 then step 37, before succeeding on the 3rd attempt on a fresh pod). | **Fix:** `_generate_alt_completions` now saves to a distinct `step_N_alt` path (never targeted by `GenerationQueue`'s cleanup) and deletes it itself right after use — no path overlap, so the race cannot occur. If you still hit this on code predating the fix: delete the aborted run's dir entirely (`rm -rf .../full_grpo_runs/trigger_none_model_{RUN_ID}_layers_32` — safe once confirmed no `checkpoints/final/`), then relaunch the **same** `run_ablation.py` sweep/config. |
| Exit 1 after full train + post-eval with `UnboundLocalError: cannot access local variable 'gc'` at `gc.collect()` in `train_grpo_model_against_probe.py` | **Fixed Aug 8, 2026.** `import gc` was local to the Phase-0 adapter-freeing branch, which does not run for base-model / no-`reference_adapter_path` launches; end-of-run activation-cache cleanup then unbound-referenced `gc`. `run_ablation.py` treated this as a crash and retried full trainings (hit the GT-penalty clp1e-3/p25 seed sweep). | Confirm `import gc` is at module scope. If you hit this on older code: keep `checkpoints/final/` (training succeeded) and export W&B metrics manually with `export_wandb_run_metrics.py` — do not retrain. |

---

## Related docs

- [GRPO_LOGGING_STAGES.md](./GRPO_LOGGING_STAGES.md) — per-stage local + W&B outputs
- [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md) — on-policy split eval (G′/H′)
- [PROBE_EVAL_RUNBOOK.md](./PROBE_EVAL_RUNBOOK.md) — HF checkpoint probe eval (eval-only)
- [CONTEXT.md](./CONTEXT.md) — workspace layout and path conventions
- [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) — completed run inventory and outcome metrics
- [iterative_training_experiments.md](./iterative_training_experiments.md) — multi-round experiment pipeline
