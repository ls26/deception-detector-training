"""Every fixed choice of 261010 (soft bias labels). Frozen once the run starts.

Not named spec.py: 261007's modules `import spec` and must find 261007's.
"""

from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
SYCOPHANCY_DIR = EXP_DIR.parent
REF_DIR = SYCOPHANCY_DIR / "261007_format_label_cue_ablation"
BASELINE_DIR = SYCOPHANCY_DIR / "260819_domain_baseline"

# --- fit B of 261007: rows, activations, recipe, saved scores --------------
FIT_B_ARM = "ref_cot"
FIT_B_ARMS_DIR = REF_DIR / "decomp" / "full3_oldrecipe" / "arms"
FIT_B_RUN_DIR = REF_DIR / "decomp" / "full3_oldrecipe" / "runs" / FIT_B_ARM
FIT_B_LABELS = FIT_B_ARMS_DIR / FIT_B_ARM / "labels.jsonl"
FIT_B_PROBE_SETS = FIT_B_ARMS_DIR / FIT_B_ARM / "probe_sets"
FIT_B_TRAIN_SAMPLES_PER_PROMPT = 3

# The three probes and the probe-set file each one reads (deception_yes / _no
# are the said-Yes / said-No rows of the one deception file).
PROBES = ("bias_followed_rm_answer", "deception_yes_rm_preyesno", "deception_no_rm_preyesno")
PROBE_SET_OF = {
    "bias_followed_rm_answer": "bias_followed_rm",
    "deception_yes_rm_preyesno": "deception_rm",
    "deception_no_rm_preyesno": "deception_rm",
}
PROBE_SETS = ("bias_followed_rm", "deception_rm")

# --- the 3 existing no-cue samples (train targets, and the old hard labels) --
BASELINE_COMPLETIONS = BASELINE_DIR / "completions.jsonl"
BASELINE_SCORES = BASELINE_DIR / "scores.jsonl"

# --- the 8 fresh no-cue samples (eval soft labels) --------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
N_NEW_SAMPLES = 8
# One request per sample (n = 1): seed = NEW_SEED_BASE + 8 * question_id + sample_idx.
# vLLM gives the children of an n > 1 request seeds seed, seed + 1, ...; n = 1 avoids that.
NEW_SEED_BASE = 90_000_000 + 1042
TEMPERATURE = 1.0  # 260819's; top_p left at vLLM's default 1.0 as in 260819
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.90
NEW_COMPLETIONS = EXP_DIR / "completions.jsonl"

# --- outputs ----------------------------------------------------------------
LABELS = EXP_DIR / "labels.jsonl"
RUNS_DIR = EXP_DIR / "runs"
TARGET_MODES = ("int", "hard01", "soft")

# --- comparison ---------------------------------------------------------------
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261007  # 261007's
