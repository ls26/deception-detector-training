# Runs

One row per recorded round (`round.py` → `runs/NN_slug/`, row added by
`append_row.py`). Cells are the eval AUC of the **combined score** (mean of the
per-layer sigmoids over the selected layers, the RL pipeline's averaged score),
mean ± sd over seeds. Each round's `probe_metrics.json` also holds the
mean-logit AUC, 260923's per-layer-mean AUC, and the per-probe recipe.

## Rounds

| round | change | activations | bias | bias_followed | deception | deception_yes | deception_no | decision / next |
|---|---|---|---|---|---|---|---|---|
| 01_cv_chosen_pool_starred | CV-chosen recipe (lr 0.03, 1000 steps, band 13-16 / 14-17 / 14-17) on pool; starred probes only | pool | — | 0.782 ± 0.003 | — | 0.778 ± 0.002 | 0.735 ± 0.002 | first CV-selected round; predicted OOF 0.783 / 0.770 / 0.747 vs eval 0.782 / 0.778 / 0.735; next: same for bias + deception, then loss-rule and pool_nu comparisons |
| 02_cv_chosen_pool_rest | same recipe (lr 0.03, 1000 steps, band 13-16 / 14-17), bias and deception probes | pool | 0.837 ± 0.003 | — | 0.580 ± 0.003 | — | — | completes round 01's table; bias 0.837 vs round 31's 0.818 under the same headline |
| 03_cv_chosen_lossrule_pool | same lr 0.03 / 1000 steps as rounds 01-02, but the pipeline's train-loss<0.6 layer rule instead of the CV band | pool | 0.824 ± 0.004 | 0.756 ± 0.007 | 0.580 ± 0.007 | 0.766 ± 0.011 | 0.712 ± 0.005 | band vs loss rule at equal lr: +0.013 / +0.026 / 0.000 / +0.012 / +0.023; lr 0.03 vs round 31's 0.01 under the loss rule: +0.006 / +0.011 / +0.004 / +0.006 / +0.006. The band is the larger part of the gain. Loss rule kept 31 / 31 / 32 / 3 / 19 layers (seed 0) |
| 04_cv_chosen_pool_nu_evalnu | high-confidence variant: train on pool_nu (1152 / 1152 / 1152 / 812 / 340 rows), eval only on no-cue-unanimous questions (783 / 612 / 783 / 490 / 293 rows); rounds 01-02's lr and bands, steps rescaled to equal epochs (396 / 396 / 396 / 421 / 347) | pool_nu | 0.889 ± 0.001 | 0.850 ± 0.002 | 0.617 ± 0.001 | 0.831 ± 0.003 | 0.767 ± 0.001 | vs rounds 01-02's weights re-scored on the same filtered eval (0.903 / 0.871 / 0.601 / 0.832 / 0.796): -0.014 / -0.021 / +0.016 / -0.001 / -0.029. Filtering the train set costs rows and does not help, as in 260923; the filtered eval itself is 0.05-0.09 easier for every probe |
| 05_cv_chosen2000_pool_starred | CV re-selected with the 2000-step cells: 2000 steps (lr 0.01 for bias_followed, 0.03 otherwise), same bands; starred probes | pool | — | 0.782 ± 0.002 | — | 0.786 ± 0.001 | 0.746 ± 0.003 | vs rounds 01-02 (1000 steps): 0.000 / +0.008 / +0.011 on bias_followed / deception_yes / deception_no, matching the OOF prediction (+0.004 to +0.012). Not yet on the steps plateau; 4000-step cells pending |
| 06_cv_chosen2000_pool_rest | same 2000-step recipe, bias and deception probes | pool | 0.839 ± 0.002 | — | 0.601 ± 0.003 | — | — | bias +0.002, deception +0.021 vs round 02 |
| 07_cv_final_pool_starred | final CV selection over the full 3x3 grid: 4000 steps (lr 0.01 bias family, 0.03 deception), bands 13-16 / 14-17 / 15-18; starred probes | pool | — | 0.788 ± 0.002 | — | 0.793 ± 0.001 | 0.749 ± 0.000 | vs rounds 05-06 (2000 steps): +0.006 / +0.007 / +0.003. OOF still rising with lr x steps for the deception strata at the grid's corner; bias family flat. A grid extending to 8000 steps or lr 0.1 is the natural follow-up |
| 08_cv_final_pool_rest | same full-grid recipe, bias and deception probes | pool | 0.840 ± 0.001 | — | 0.590 ± 0.004 | — | — | bias +0.001, deception -0.011 vs round 06 (deception is near chance by construction) |
