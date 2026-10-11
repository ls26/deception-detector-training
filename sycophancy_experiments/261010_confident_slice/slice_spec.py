"""Every fixed choice of 261010_confident_slice. Frozen once the run starts.

The slice: questions where at least MIN_AGREE of the N_NOCUE_SAMPLES no-cue CoT
samples give the same parsed answer (an unparsed sample counts against). Rows
on those questions are relabelled with that answer as the no-cue answer.

Not named spec.py: 261007's modules `import spec` and must find 261007's.
The probe pipeline's own settings (pool and draw seeds, CV grid, folds, edge
cells, band width, trainer, combined score, bootstrap) are 261007's spec.py
values, read through PipelineConfig's defaults in slice_pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SYCOPHANCY_DIR = EXP_DIR.parent
REF_DIR = SYCOPHANCY_DIR / "261007_format_label_cue_ablation"  # the probe pipeline and the baseline probe
SOFT_DIR = SYCOPHANCY_DIR / "261010_soft_bias_labels"  # the 8 no-cue samples per eval question
BASELINE_DIR = SYCOPHANCY_DIR / "260819_domain_baseline"  # questions, prompts, CoT rollouts
RESCORE_DIR = SYCOPHANCY_DIR / "260824_rescore"  # turn1_label / turn2_label

# --- inputs -----------------------------------------------------------------
# 261007's labels for the 260819 CoT rollouts (13,740 cued rows; no-cue answer =
# majority of 260819's 3 no-cue samples; self-report = float32 P(Yes) > 0.5).
SOURCE_LABELS = REF_DIR / "arms" / "ref_cot" / "labels.jsonl"
BASELINE_COMPLETIONS = BASELINE_DIR / "completions.jsonl"
EVAL_NOCUE_COMPLETIONS = SOFT_DIR / "completions.jsonl"

# The baseline probe: 261007's decomp/full3_oldrecipe fit of ref_cot (all
# questions, the labels above, 3 rollouts per prompt, 260924's recipe, seed 0).
BASELINE_PROBE_RUN_DIR = REF_DIR / "decomp" / "full3_oldrecipe" / "runs" / "ref_cot"
BASELINE_PROBE_ACT_DIR = REF_DIR / "decomp" / "full3_oldrecipe" / "arms" / "ref_cot" / "activations"

# --- the 8 no-cue samples (train questions generated here; eval from 261010) --
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
N_NOCUE_SAMPLES = 8
# One request per sample (n = 1): seed = NEW_SEED_BASE + 8 * question_id + sample_idx.
# 261010's value; question ids are unique across splits, so train and eval seeds never collide.
NEW_SEED_BASE = 90_000_000 + 1042
TEMPERATURE = 1.0  # 260819's; top_p left at vLLM's default 1.0 as in 260819
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
N_TRAIN_QUESTIONS = 1603
N_EVAL_QUESTIONS = 687

# --- the slice --------------------------------------------------------------
MIN_AGREE = 6

# --- probes -----------------------------------------------------------------
ARM = "slice_cot"
DEFINITION = "rm"
DATASETS = ("deception", "bias_followed")  # probe-set families built and extracted
PROBE_KINDS = ("deception_yes", "deception_no", "bias_followed")  # drawn kinds = trained probes
PROBES = ("deception_yes_rm_preyesno", "deception_no_rm_preyesno", "bias_followed_rm_answer")
TRAIN_SAMPLES_PER_PROMPT = 3  # every rollout of a prompt can be a train row, as the baseline probe

# --- dry runs ---------------------------------------------------------------
DRY_QUESTIONS = 100  # per split
DRY_STEPS = 200
