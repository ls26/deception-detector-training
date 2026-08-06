# Probe Eval Runbook

Main reference for running **probe eval** (post-RL stages B–I) on checkpoints:

| Workflow | Entry point | Typical use |
|----------|-------------|-------------|
| **HF checkpoint sweep** | [`eval_probe_checkpoints.py`](../obfuscation-atlas/obfuscation_atlas/scripts/eval_probe_checkpoints.py) | Batch eval of HuggingFace-published LoRAs |
| **Local checkpoint (eval-only)** | [`train_grpo_model_against_probe.py`](../obfuscation-atlas/obfuscation_atlas/scripts/train_grpo_model_against_probe.py) | Re-run post-eval from a finished local GRPO run dir |
| **Single-GPU** | Either entry point with vLLM lifecycle below | Pods with one GPU (see dedicated section) |

For architecture and stage definitions, see [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md).
For workspace layout and sweep history, see [CONTEXT.md](./CONTEXT.md).

---

## What this workflow does

Probe eval measures whether deception detectors still fire on a checkpoint's
activations and assigns a paper-style **policy type** (Honest, Deceptive,
Obfuscated policy, Obfuscated activations).

This is separate from MBPP **task** eval (`eval_mbpp_checkpoints.py`), which
measures pass rates and hardcoding. Probe eval **requires** MBPP task eval to
have run first — task metrics are overlaid at export time because eval-only W&B
runs have broken `val_*` fields.

**Entry point:**

```bash
python -m obfuscation_atlas.scripts.eval_probe_checkpoints [flags]
```

**Primary deliverable:** one classified JSON per grid cell:

```
/workspace/results/run_results/hf_probe_eval_runs/extracted_results/run_metrics_{SHORT_NAME}.json
```

There is **no** `--force-reeval`. Existing exports are never overwritten.

---

## Prerequisites (must all pass before running)

### 1. Environment

```bash
source /workspace/atlas_setup.sh          # HF_HOME, WANDB_API_KEY, UV paths
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas
```

Verify:

```bash
python -c "import obfuscation_atlas; print('OK')"
test -n "$WANDB_API_KEY" && echo WANDB_OK
python -c "import shutil; f=shutil.disk_usage('/workspace').free/(1024**3); print(f'{f:.1f} GB'); assert f>=30"
```

The batch script enforces **≥ 30 GB free** on `/workspace`. Monitor disk during
long sweeps; prior runs have exhausted disk mid-sweep.

### 2. HF checkpoints downloaded

Adapters must be cached under `/workspace/hf_cache/` (via
`download_checkpoints.py`). The probe script uses `local_files_only=True`.

### 3. MBPP task eval complete

Required files per checkpoint:

- `/workspace/results/run_results/hf_task_eval_metrics/summary.json`
- `/workspace/results/run_results/hf_task_eval_metrics/{checkpoint_name}.jsonl`

Cells without a summary entry are **skipped_invalid**.

### 4. vLLM server (two-GPU setup)

Phase 1 on-policy generation (stage F) requires a running vLLM server. On a
**two-GPU pod**, keep vLLM on GPU 1 and run eval forward passes on GPU 0.

```bash
# Terminal 1 — keep running for the whole sweep
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95
```

Verify before starting:

```bash
curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo vLLM_OK
```

If vLLM dies mid-sweep, restart it and re-run the **same command**. Completed
exports are skipped; interrupted cells are retried.

**Single-GPU pods:** do **not** use this layout. See [Single-GPU probe eval](#single-gpu-probe-eval).

### 5. Base probe caches (usually already present)

Phase 0 trains a base-model detector once per probe type. Cached if this manifest
exists:

| Probe | Cache glob under `/workspace/results/run_results/` |
|-------|-----------------------------------------------------|
| `mbpp_probe` | `base_results_and_detectors_*mbpp-hardcode*` |
| `diverse_deception_probe` | `base_results_and_detectors_*local-diverse_deception*` |

Manifest file: `f1_and_recall_plot_initial_detector.base_model.on_dataset__manifest.json`

If missing, the batch script runs Phase 0 automatically (~30+ min per probe type).

---

## Grid, naming, and CLI

### SHORT_NAME convention

| Example | Meaning |
|---------|---------|
| `kl1e-3-det3-s2-mbpp` | kl=0.001, det=3, seed=2, mbpp_probe |
| `kl1e-4-det0-s1-ddp` | kl=1e-4, det=0, seed=1, diverse_deception_probe |
| `kl1e-4-det1-s1-ddp` | kl=1e-4, det=1, seed=1, diverse_deception_probe |

HF repo pattern:
`AlignmentResearch/obfuscation-atlas-Meta-Llama-3-8B-Instruct-kl{kl}-det{det}-seed{seed}[-{probe}]`

### Probe CLI aliases

| CLI | Canonical probe dataset | Suffix |
|-----|-------------------------|--------|
| `mbpp` | `mbpp_probe` | `-mbpp` |
| `general`, `ddp` | `diverse_deception_probe` | `-ddp` |

### det=0 special case

det=0 has **one HF adapter per (kl, seed)** — not separate mbpp/ddp repos. Which
probe dataset to evaluate is chosen by `--det0-probe` (default:
`diverse_deception_probe`). The suffix in `SHORT_NAME` reflects that choice.

Example: prior MBPP det=0 sweep used `--det0-probe mbpp` → exports named
`*-det0-*-mbpp.json`. A later sweep with default `--det0-probe general` produces
**new** `*-det0-*-ddp.json` cells; it does not overwrite or skip the mbpp exports
(different filenames).

### Default grid vs explicit flags

Script defaults: kl={1e-4, 1e-3, 1e-2}, det={1, 3, 10}, both probes, seeds 1–3
(54 cells; det=0 not included).

**Always pass explicit flags** for non-default sweeps:

```bash
python -m obfuscation_atlas.scripts.eval_probe_checkpoints \
  --kl 0.0001 0.001 \
  --det 0 1 3 10 \
  --probe mbpp general \
  --seed 1 2 3
```

Other useful flags:

| Flag | Purpose |
|------|---------|
| `--limit N` | Run first N unevaluated cells only (smoke test) |
| `--det0-probe mbpp` | Use mbpp_probe for det=0 cells |
| `--probe mbpp` | MBPP probe only |
| `--probe general` | diverse_deception_probe only |

---

## Pipeline (per cell)

```
Preflight
    │
    ▼
build_run_plan ──► skip buckets (done / missing HF / invalid)
    │
    ▼
ensure_base_probe_cache (Phase 0, once per probe type)
    │
    ▼
For each cell:
    Phase 1: train_grpo_model_against_probe (eval-only)
        only_eval_from_other_run=True
        reference_adapter_path=<HF LoRA>
        wandb.group=hf_probe_eval
        verify 5 manifests
        write .state/{SHORT_NAME}.json
    │
    ▼
    Export: export_wandb_run_metrics --task-metrics-json
        verify val_success > 0, policy_type set
        atomic write → run_metrics_{SHORT_NAME}.json
        clear .state/
    │
    ▼
write sweep_status.json (once, at end of entire run)
```

### Phase 1 success criteria

- Artifacts dir:
  `/workspace/results/run_results/hf_probe_eval/trigger_none_model_{RUN_ID}_layers_32/`
  (historical runs may also have dirs under `hf_probe_eval_runs/`)
- Exactly **5** `f1_and_recall_plot_*__manifest.json` files

### Export gate

Export is rejected if:

- `val_success <= 0` (from MBPP overlay; 0% pass-first checkpoints always fail)
- `val_passes_first_n_tests` does not match MBPP summary within 0.002
- `classification.policy_type` is empty
- JSON contains `on_policy_responses` artifacts

Phase 1 can succeed while export fails. This is intentional, not a pipeline bug.

---

## Recommended agent workflow (gated steps)

Follow these checks before and during every sweep.

### Step 1 — Environment

Run the prerequisite checks above. Abort if any fail.

### Step 2 — Inventory

Dry-run the plan without executing:

```bash
python - <<'PY'
from argparse import Namespace
from obfuscation_atlas.scripts.eval_probe_checkpoints import (
    build_run_plan, load_mbpp_summary, short_name, normalize_probe,
)
args = Namespace(
    kl=[0.0001, 0.001], det=[0, 1, 3, 10],   # match your sweep
    probe=[normalize_probe("mbpp"), normalize_probe("general")],
    seed=[1, 2, 3], det0_probe=normalize_probe("general"), limit=None,
)
to_run, status = build_run_plan(args, load_mbpp_summary())
print(f"to_run={len(to_run)} skipped_done={len(status.skipped_done)} "
      f"missing_hf={len(status.skipped_missing_hf)} invalid={len(status.skipped_invalid)}")
for e in to_run:
    print(f"  RUN {short_name(e.spec, det0_probe=args.det0_probe)}")
PY
```

Confirm counts match expectations. Flag cells with 0% MBPP pass-first as
expected export failures.

### Step 3 — vLLM

Confirm `curl -sf http://127.0.0.1:8000/v1/models` returns 200.

### Step 4 — Smoke test

```bash
python -m obfuscation_atlas.scripts.eval_probe_checkpoints \
  --kl 0.0001 --det 0 --probe general --seed 1 --limit 1
```

Verify the export:

```bash
python - <<'PY'
import json
p="/workspace/results/run_results/hf_probe_eval_runs/extracted_results/run_metrics_kl1e-4-det0-s1-ddp.json"
d=json.load(open(p)); m,c=d["metrics"],d["classification"]
assert m["val_success"]>0 and c["policy_type"]
print("OK:", c["policy_type"])
PY
```

### Step 5 — Full sweep

```bash
python -m obfuscation_atlas.scripts.eval_probe_checkpoints \
  --kl 0.0001 0.001 --det 0 1 3 10 --probe mbpp general --seed 1 2 3 \
  2>&1 | tee /workspace/results/run_results/logs/my_sweep.log
```

Monitor disk periodically. Expect ~15–25 min per cell.

### Step 6 — Post-verify

Do **not** rely solely on `sweep_status.json` (see below). Verify exports directly:

```bash
python - <<'PY'
import json, glob
paths = sorted(glob.glob(
    "/workspace/results/run_results/hf_probe_eval_runs/extracted_results/run_metrics_*.json"
))
print(f"total exports: {len(paths)}")
for p in paths[-5:]:
    d=json.load(open(p)); c=d["classification"]
    print(p.split("run_metrics_")[1].replace(".json",""), c["policy_type"])
PY
```

---

## Skip, resume, and failure behavior

### Skip buckets (at plan time)

| Bucket | Condition | On re-run |
|--------|-----------|-----------|
| `skipped_done` | `run_metrics_{SHORT_NAME}.json` exists | Always skipped |
| `skipped_missing_hf` | Repo not on HuggingFace | Skipped until published |
| `skipped_invalid` | No MBPP summary, adapter missing/incomplete | Skipped until fixed |
| `to_run` | Everything else | Attempted |

### Per-cell outcomes

| Outcome | Phase 1 | Export | `.state/` | `run_metrics_*.json` |
|---------|---------|--------|-----------|----------------------|
| Success | OK | OK | Cleared | Written |
| Phase 1 crash | FAIL | — | Cleared | Not written |
| Export fail (e.g. val_success=0) | OK | FAIL | Cleared | Not written |
| Process killed mid-Phase 1 | partial | — | Not written | Not written |
| Process killed mid-export | OK | partial | May exist | Not written |

### `.state/` directory

**Purpose:** Resume export after Phase 1 succeeded but export did not complete.

Location:
`/workspace/results/run_results/hf_probe_eval_runs/extracted_results/.state/{SHORT_NAME}.json`

Contents: W&B run ID, checkpoint name, probe dataset, timestamp.

**Lifecycle:**

1. Written after Phase 1 verifies 5 manifests
2. Read on re-run → **export-only retry** (skips Phase 1)
3. Cleared after successful export
4. Cleared on Phase 1 failure or export failure (full mode)

**Empty `.state/` is normal** when no cell is currently stuck between Phase 1
and export. It does not mean nothing has run.

### `sweep_status.json`

Location:
`/workspace/results/run_results/hf_probe_eval_runs/extracted_results/sweep_status.json`

Written **once at the very end** of `main()`, not after each cell. If the
process is killed (disk full, OOM, manual interrupt), this file reflects the
**last run that exited normally** — often a smoke test — not the interrupted
sweep.

**To assess progress after a crash:**

1. Count `run_metrics_*.json` exports on disk
2. Grep the sweep log for `Exported:` and `ERROR:`
3. Re-run `build_run_plan()` dry-run to see remaining `to_run` cells
4. Check per-cell logs at `/workspace/results/run_results/logs/{SHORT_NAME}.log`

### Re-running after interruption

Re-run the **exact same command**. Safe because:

- Completed exports are skipped (`skipped_done`)
- Cells with `.state/` files retry export-only
- Failed/interrupted cells without `.state/` retry full Phase 1 + export

Exit code is **1** if any cell in the plan failed, even when most succeeded.

---

## All outputs and paths

| Output | Path |
|--------|------|
| **Final export** | `.../hf_probe_eval_runs/extracted_results/run_metrics_{SHORT_NAME}.json` |
| Task metrics overlay | `.../extracted_results/task_metrics_{SHORT_NAME}.json` |
| Resume state | `.../extracted_results/.state/{SHORT_NAME}.json` |
| Sweep summary | `.../extracted_results/sweep_status.json` (unreliable if crashed) |
| Per-cell Phase 1 log | `/workspace/results/run_results/logs/{SHORT_NAME}.log` |
| Phase 0 log | `/workspace/results/run_results/logs/phase0_{base-mbpp,base-ddp}.log` |
| Phase 1 artifacts | `/workspace/results/run_results/hf_probe_eval/trigger_none_model_{RUN_ID}_layers_32/` |
| Historical Phase 1 artifacts | `/workspace/results/run_results/hf_probe_eval_runs/trigger_none_model_{RUN_ID}_layers_32/` |
| Base probe cache | `/workspace/results/run_results/base_results_and_detectors_*/` |
| Activation cache | `/workspace/results/activations/` |
| W&B | project `deception_detection`, group `hf_probe_eval` |

---

## Known failure modes

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Preflight FAIL vLLM | Server not running | Two-GPU: start vLLM on GPU 1. Single-GPU: start vLLM at `Skipping RL training` (see [Single-GPU](#single-gpu-probe-eval)) |
| Connection refused at stage F | vLLM not ready before F | Start vLLM earlier; verify `/health`. Post-eval does not wait for server. |
| CUDA OOM at stage G | vLLM still on GPU | Kill vLLM after `OK: on-policy split labels`; `pkill -9 -f VLLM::EngineCore` |
| CUDA OOM at model load | vLLM started too early | Do not start vLLM until eval model is loaded |
| GPU memory stuck after kill | Zombie `VLLM::EngineCore` | `kill -9 <pid>` or `pkill -9 -f VLLM::EngineCore`; see GPU cleanup section |
| Preflight FAIL disk | < 30 GB free | Free space under `/workspace/results/` or `/workspace/hf_cache/` |
| Phase 1 non-zero exit | vLLM died, OOM, bad adapter | Check `{SHORT_NAME}.log`; restart vLLM; re-run |
| `val_success invalid: 0.0` | 0% MBPP pass-first | Expected for broken checkpoints; do not retry unless MBPP re-eval fixes it |
| `layers_below_threshold` empty | Degenerate checkpoint (kl1e-4-det3-s1-ddp) | Expected failure; document and move on |
| Disk quota exceeded mid-sweep | `/workspace` full | Free space; re-run same command |
| `[CodeExecutor] Failed to get results` | Benign log noise from vLLM subprocess | Ignore; task metrics come from MBPP overlay |
| `sweep_status.json` stale | Process killed before end | Use exports + logs instead |

### Permanently blocked cells (as of Jul 2026)

Missing on HuggingFace (cannot run):

- `kl1e-4-det1-s1-mbpp`
- `kl1e-4-det1-s2-mbpp`

Known zero-pass-first (Phase 1 may run; export always fails):

- `kl1e-4-det3-s3-mbpp`
- `kl1e-4-det3-s1-ddp` (also Phase 1 crash)

---

## Analysis after a sweep

Single export:

```bash
python /workspace/scratch/extract_one_run_results.py
# Set RUN_METRICS_PATH to the export JSON
```

Multi-export comparison:

```bash
python /workspace/scratch/extract_multiple_runs_results.py
# Globs hf_probe_eval_runs/extracted_results/run_metrics_*.json by default
```

MBPP task metrics for cross-check:
`/workspace/results/run_results/hf_task_eval_metrics/summary.csv`

---

## Local checkpoint probe eval (eval-only)

Use this when a **local GRPO run** finished training (or you have a final LoRA on
disk) and you want to run or re-run post-RL probe eval (stages B–I) without
retraining RL.

Typical cases:

- Post-eval crashed after step 128; LoRA is already under `checkpoints/final/`
- Re-run with on-policy **split** metrics into a new run dir (Jul 2026+)
- Re-run after changing mixed-probe manifest or Phase-0 cache prefix

### Prerequisites

| Check | Path / note |
|-------|-------------|
| Final adapter | `…/trigger_none_model_{TRAIN_RUN_ID}_layers_32/checkpoints/final/lora/adapter_model.bin` |
| Same config as training | Match `loss.kl`, `loss.detector`, `+data/probe_dataset`, `training.seed`, `data.seed` |
| Phase-0 cache (if mixed probe) | Set `detector.directory_save_prefix` and `++detector.mixed_probe_manifest_path` as in training |
| GPU | Two-GPU: vLLM on GPU 1 (above). Single-GPU: see [Single-GPU probe eval](#single-gpu-probe-eval). |

### Launch command

Entry point: `train_grpo_model_against_probe.py` with `only_eval_from_other_run=True`
and `reference_adapter_path` pointing at the **training run's** final checkpoint
(not the new eval run dir).

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

# Canonical post-fix disjoint probe run:
#   mtzvz59v  (kl1e-3-det3-s43-mbpp-disjoint)
# Canonical mixed nofu probe run:
#   2hgfj4k7  (nofu_mixed_kl1e-3-det-3-s43)

TRAIN_RUN_ID=2hgfj4k7   # finished local GRPO run (mixed nofu probe)

python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b \
    loss.kl=0.001 \
    loss.detector=3 \
    +data/probe_dataset=mbpp_probe \
    training.seed=43 \
    data.seed=53 \
    grpo.model.attn_implementation=sdpa \
    training.save_path_base=/workspace/results/run_results \
    detector.model_cache_path=/workspace/results/activations \
    detector.directory_save_prefix=nofu_mixed \
    ++detector.mixed_probe_manifest_path=/workspace/experiments/archive/mixed_round1_nofu/probe_train_dataset.json \
    ++training.only_eval_from_other_run=True \
    ++training.resume_from_checkpoint=False \
    ++grpo.model.reference_adapter_path=/workspace/results/run_results/full_grpo_runs/trigger_none_model_${TRAIN_RUN_ID}_layers_32/checkpoints/final \
    corrected_detector_gradient=false \
    misalignment_eval.enabled=false \
    wandb.project=deception_detection \
    wandb.group=full_grpo_runs \
    wandb.run_name_prefix=nofu_mixed_kl1e-3-det-3-s43-split-eval
```

Record **EVAL_RUN_ID** from the W&B URL. Artifacts land under
`trigger_none_model_{EVAL_RUN_ID}_layers_32/` — the original
`trigger_none_model_{TRAIN_RUN_ID}_layers_32/` dir is untouched.

### Success criteria

| Output | Expect |
|--------|--------|
| Manifests | 5 × `f1_and_recall_plot_*__manifest.json` in new run dir |
| Split JSONs (Jul 2026+) | 3 × `on_policy_split_*.json` |
| Stage F | `OK: on-policy split labels`; deceptive + honest + unlabeled == 400 |
| W&B | `eval_obfuscation/*` in summary; state `finished` |
| Exit code | 0 |

For split eval analysis, use only
`initial_detector.final_model.on_policy.split` and
`fresh_detector.final_model.on_policy.split`. Ignore
`initial_detector.base_model.on_policy` under eval-only mode (scores bare base
Llama, not the reference adapter used during training).

### Export (split train / eval W&B runs)

When training and post-eval are on **different** W&B runs, overlay task metrics
from the training run:

```bash
python -m obfuscation_atlas.scripts.export_wandb_run_metrics \
    --entity lilylixin6-the-harker-school \
    --project deception_detection \
    --run-id {EVAL_RUN_ID} \
    --task-metrics-run-id {TRAIN_RUN_ID} \
    --output /workspace/results/run_results/full_grpo_runs/extracted_results/run_metrics_nofu_mixed_kl1e-3-det-3-s43-split-eval.json
```

If train and eval completed in one W&B run, omit `--task-metrics-run-id`.

**Historical examples:**
- **Disjoint probe (`mtzvz59v`):** `kl1e-3-det3-s43-mbpp-disjoint` — canonical post-fix disjoint probe run (Jul 14, 2026). Obfuscated policy. Adapter: `…/trigger_none_model_mtzvz59v_layers_32/checkpoints/final/`.
- **Mixed nofu probe (`2hgfj4k7`):** `nofu_mixed_kl1e-3-det-3-s43` — canonical mixed nofu probe run (Jul 23, 2026). Obfuscated policy. Adapter: `…/trigger_none_model_2hgfj4k7_layers_32/checkpoints/final/`.
- **Earlier (`8mca01b3`):** pre-nofu mixed initial probe GRPO (Jul 14, 2026); eval-only split rerun `11ftrshu` completed with 5 manifests + 3 split JSONs.

---

## Single-GPU probe eval

On a **one-GPU pod** (~80 GB), vLLM and the eval model cannot both occupy the
GPU at full utilization. Use a **timed vLLM lifecycle**: off during model load
and on-policy activation extraction (stages B–E, G–I), on only for stage F
generation.

This applies to **both** HF sweeps (`eval_probe_checkpoints.py`) and local
eval-only runs (`train_grpo_model_against_probe.py`).

### Why the lifecycle matters

| Mistake | Failure mode |
|---------|--------------|
| Start vLLM before eval loads model | OOM at model load (~90% GPU needed) |
| vLLM still up during stage G | OOM: vLLM ~34 GB + probe forward ~28 GB |
| Start vLLM too late (end of stage D) | Connection refused at stage F |
| Leave zombie `VLLM::EngineCore` after kill | GPU memory stuck; next run OOM |

**Critical:** `generate_on_policy_responses` calls vLLM **without**
`wait_for_server`. vLLM must be healthy **before** stage F begins.

### Memory timeline (1× 80 GB GPU)

| Stage | vLLM | Eval process | Action |
|-------|------|--------------|--------|
| B (Phase-0 cache) | OFF | ~50–60 GB model | Start eval only |
| D (fresh probe retrain) | **START** | ~50–60 GB | Start vLLM when log shows `Skipping RL training` |
| E (initial on final on-dataset) | loading → ready | probe caching | Verify `/health` before E finishes (~15 min window) |
| F (400 on-policy gens) | ON (40%) | code-exec grading | vLLM must respond; use `max_workers=1` for code-exec |
| G / H / I | **KILL** | ~28 GB probe forward | Kill vLLM right after `OK: on-policy split labels` |

### Supervised two-terminal workflow

**Terminal 1 — eval (no vLLM at start):**

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
cd /workspace/obfuscation-atlas

# Preflight: GPU must show ~0 MiB used
nvidia-smi

LOG=/workspace/scratch/grpo_split_eval_final.log
PYTHONUNBUFFERED=1 python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    ... same flags as local eval-only command above ... \
    2>&1 | tee "$LOG"
```

**Terminal 2 — vLLM lifecycle:**

1. **Start** as soon as Terminal 1 prints `Skipping RL training, only evaluating from other run` (do not wait for stage D manifest):

```bash
source /workspace/atlas_setup.sh
source /workspace/obfuscation-atlas/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1 \
    2>&1 | tee /workspace/scratch/vllm_split_eval_final.log
```

2. **Verify ready** (repeat until success, well before stage F):

```bash
curl -sf http://127.0.0.1:8000/health && echo vLLM_OK
curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo MODELS_OK
```

3. **Kill** immediately when Terminal 1 prints `OK: on-policy split labels — deceptive=… honest=… unlabeled=…`:

```bash
pkill -9 -f "afterburner.generation.vllm.server" || true
pkill -9 -f "VLLM::EngineCore" || true
sleep 5 && nvidia-smi   # vLLM gone; eval process holds ~28–35 GB
```

Let Terminal 1 continue through stages G → H → I unattended (~30–60 min).

### GPU cleanup (before any run)

Zombie `VLLM::EngineCore` processes can survive `pkill` on the server wrapper:

```bash
pkill -9 -f "afterburner.generation.vllm.server" || true
pkill -9 -f "VLLM::EngineCore" || true
pkill -9 -f "train_grpo_model_against_probe" || true
sleep 5
nvidia-smi   # expect no compute processes, ~0 MiB used
```

If memory is still held, wait 10s and re-check; reboot the pod only as last resort.

### Single-GPU vs two-GPU summary

| Setting | Two-GPU (ideal) | Single-GPU |
|---------|-----------------|------------|
| vLLM device | GPU 1 | GPU 0 |
| `--gpu-memory-utilization` | 0.95 | **0.40** |
| vLLM uptime | Whole sweep / whole eval | Stage F only |
| Eval forward | GPU 0 | Same GPU (after vLLM killed) |

**Historical examples:** `mtzvz59v` (disjoint probe, Jul 14, 2026) and `2hgfj4k7` (mixed nofu, Jul 23, 2026) both completed full GRPO + post-eval on 1× GPU with vLLM at 40% util as a single clean process. Earlier run `8mca01b3` (Jul 13, 2026) also used this layout. Eval-only reruns need the start/stop lifecycle above
to avoid OOM from repeated partial runs and zombie processes.

---

## Quick reference commands

```bash
# Environment
source /workspace/atlas_setup.sh && source /workspace/obfuscation-atlas/.venv/bin/activate

# vLLM — two-GPU (GPU 1, separate terminal)
CUDA_VISIBLE_DEVICES=1 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.95

# vLLM — single-GPU (GPU 0, stage F only; see Single-GPU section)
CUDA_VISIBLE_DEVICES=0 python -m afterburner.generation.vllm.server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --max-lora-rank 128 --gpu-memory-utilization 0.40 --tensor-parallel-size 1

# Local checkpoint eval-only (set TRAIN_RUN_ID and match training config)
python -m obfuscation_atlas.scripts.train_grpo_model_against_probe \
    +model_preset=llama3_8b loss.kl=0.001 loss.detector=3 +data/probe_dataset=mbpp_probe \
    training.seed=43 data.seed=53 \
    ++training.only_eval_from_other_run=True ++training.resume_from_checkpoint=False \
    ++grpo.model.reference_adapter_path=/workspace/results/run_results/full_grpo_runs/trigger_none_model_{TRAIN_RUN_ID}_layers_32/checkpoints/final \
    training.save_path_base=/workspace/results/run_results \
    detector.model_cache_path=/workspace/results/activations

# HF sweep smoke test
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --limit 1

# MBPP probe sweep
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --probe mbpp

# General (ddp) probe sweep
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --probe general

# Single cell
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --kl 0.001 --det 3 --probe mbpp --seed 2

# det=0 with MBPP probe
python -m obfuscation_atlas.scripts.eval_probe_checkpoints --det 0 --det0-probe mbpp --kl 0.0001 --seed 1 2 3
```

---

## Related docs

- [PROBE_EVAL_PIPELINE.md](./PROBE_EVAL_PIPELINE.md) — stage architecture, split eval, module roles
- [FULL_GRPO_RUN.md](./FULL_GRPO_RUN.md) — full local GRPO runbook (train + post-eval)
- [CONTEXT.md](./CONTEXT.md) — workspace layout, sweep history summary, overall project state
