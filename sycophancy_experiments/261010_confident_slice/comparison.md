# Slice-trained vs baseline probe on the slice eval rows

Slice: questions where at least 6 of 8 no-cue samples give the same answer; labels use that answer. Both probes are scored on the same slice eval rows with the slice labels; AUC of the mean sigmoid over each probe's own band. Δ = slice-trained minus baseline; 95% CI from a question-level bootstrap (2000 draws, seed 20261007); `*` marks a CI that excludes 0. One training seed (0).

| Probe | Eval rows (pos / neg) | Slice-trained AUC | Baseline AUC | Δ [95% CI] |
|---|---|---|---|---|
| deception_yes_rm_preyesno | 664 (547 / 117) | 0.836 | 0.825 | +0.011 [-0.013, +0.036] |
| deception_no_rm_preyesno | 310 (64 / 246) | 0.859 | 0.860 | -0.001 [-0.030, +0.028] |
| bias_followed_rm_answer | 777 (292 / 485) | 0.871 | 0.866 | +0.006 [-0.010, +0.023] |

## Training

| Probe | Train rows, slice (baseline) | Slice recipe: lr / steps / layers (epochs) | Baseline recipe: lr / steps / layers | Baseline AUC, all eval rows (rows) |
|---|---|---|---|---|
| deception_yes_rm_preyesno | 1004 (1992) | 0.03 / 8000 / 15–18 (255.0) | 0.03 / 4000 / 14–17 | 0.771 (920) |
| deception_no_rm_preyesno | 508 (916) | 0.03 / 2000 / 14–17 (126.0) | 0.03 / 4000 / 15–18 | 0.799 (406) |
| bias_followed_rm_answer | 1512 (2908) | 0.03 / 2000 / 13–16 (42.3) | 0.01 / 4000 / 13–16 | 0.788 (1106) |
