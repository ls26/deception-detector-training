# Shared-eval 3-probe monitorability comparison (Aug 14, 2026)

Compares the **monitorability of the initial (base) model against the final (round-5)
model** of each completed lineage, by scoring three probes on one shared on-policy
evaluation set.

The round-6 probe eval ([`../250810_round6_probe_eval/`](../250810_round6_probe_eval/))
answered "is deception linearly detectable in the final model?" but scored each lineage
on its own *supplemented* set (20+ per class, walk-back topped up), so initial-vs-final
numbers were not comparable. Here every probe of a lineage sees identical rows.

## The three probes

| Key | Training data | Activations from | Source |
|---|---|---|---|
| `final_round6` (A) | round-6 manifest, 800/class | round-5 model | existing `round6/probe_weights` |
| `initial_probe1` (B) | 200 on-dataset + 100 base/det0 on-policy | base model | existing `round1/probe_weights` |
| `initial_round6_data` (C) | round-6 manifest, 800/class | **base model** | trained here |

A probe is a function of activations, so "same evaluation set" can only mean the same
*texts*: A reads them through the round-5 model, B and C through the base model. Each
probe reads its own model's representations, which is what makes the comparison a
statement about the models rather than about the probes.

- **A vs C** isolates the model: identical training text, only the activations differ.
- **B** is the reference point — what the original probe-1 method would have reported.

Probe B's weights are byte-identical across all ten lineage copies (all are copies of
`trial4/rounds/round1/probe_weights`), so B is one fixed probe and only the eval set varies.

## Shared evaluation set

100 per class, seeded round-robin over every policy in the lineage, deduplicated on
`(prompt, completion)`:

- **deceptive**: `det0`, `round1` … `round5`
- **honest**: `base`, `round1` … `round5`

det0/base are the per-class anchors — the base policy is deceptive on only **1 of 400**
eval prompts, so a base-anchored deceptive pool is unusable. This mirrors how
`build_train_dataset_split_sources()` picks its anchors for probe *training*.

Drawn from each policy's existing `*_eval_generations.jsonl` in the trial buffer; no
generation is run. Zero `task_id` overlap with the round-6 probe training manifest is
asserted at build time (verified for det0 and base train/eval splits too).

## Files

| File | Purpose |
|---|---|
| `common.py` | Lineages, paths, seeds, probe keys |
| `build_shared_eval.py` | Builds `shared_eval_sets/*.jsonl` + `provenance_*.json` |
| `train_probe_c.py` | Trains probe C on base activations from each round-6 manifest |
| `run_compare.py` | Scores all three probes; writes `trials/trial{N}/results/shared_eval_probe_compare*.json` |
| `summarize.py` | Comparison table + `summary.json` |
| `run_all.sh` | All four stages, logging to `run_$(hostname).log` |

## Running

```bash
# All 9 lineages (~2-3 h, single GPU, no vLLM needed)
tmux new-session -d -s shared_eval \
  'bash /workspace/experiments/260814_shared_eval_probe_compare/run_all.sh'

# One lineage
bash run_all.sh --trial 6 --variant original
```

The `shared_eval_sets/*.jsonl` rows themselves are gitignored (regenerable, and their
source buffers aren't tracked either) — run `build_shared_eval.py` before anything else on
a fresh checkout. The build is deterministic given `BUILD_SEED` and the trial buffers, and
the committed `provenance_*.json` records exactly what each set contains.

Every stage is resumable: existing eval sets, probe-C weights, and already-scored probes
are skipped unless `--force` is passed. `run_compare.py` saves after each probe, so a
crash in the round-5 pass never loses the base-model pass.

Probe C writes to `round6/probe_weights_base_model/` and
`round6/probe_activations_base_model/` — **never** the round-6 paths.
`train_probe_and_eval_on_dataset()` silently reuses whatever it finds at
`weights_dir/layer_0/model.pt`, so sharing a directory with probe A would either no-op
or clobber it.

## Interpretation caveats

1. **trial8 original is degenerate.** Its rounds 2–5 produced *zero* honest eval samples,
   so its honest half is 98% base-model text — that eval set is closer to "base vs round-k
   text" than "honest vs deceptive". The three probes stay comparable to each other there;
   the absolute AUC is not comparable to the other lineages.
2. **trial4 `v2` / `early_retraining_auc_v2` deceptive is ~45% round-1** — those lineages
   went honest after round 1, so later rounds contribute almost no deceptive samples.
3. **Threshold metrics are in-sample.** `run_on_policy_split_eval` picks its threshold at
   FPR=0.01 on the eval set itself. Compare on AUC; treat accuracy/F1 as descriptive.
4. **Probe C is off-policy for the base model** — trained on text the base model would
   never generate. Intended, but it answers "is deception in round-k text linearly readable
   from base activations", which is subtly different from A's question.
5. **AUC is averaged over each probe's own `filtered_layers`** (matching the round-6 and
   supplemented-eval convention). The layer count is printed in the summary table's `L`
   column so any divergence is visible.
