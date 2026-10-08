# CV selection, arm nocot_sampled

5-fold GroupKFold over train question_ids (seed 0); grid lr x steps from cv_sweep.py (plus cv_edge.py cells); bands of width [4]; combined score = mean_sigmoid. The eval split was not used. spec_hash 10b44b32b4c2.

| probe | lr | steps | epochs (full train) | band | OOF ± fold sd | loss rule | all 32 |
|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.01 | 4000 | 141.0 | 13-16 | 0.887 ± 0.008 | 0.869 | 0.867 |
| deception_rm_preyesno | 0.1 | 8000 | 268.9 | 27-30 | 0.643 ± 0.023 | 0.642 | 0.635 |
| deception_yes_rm_preyesno | 0.1 | 8000 | 376.5 | 28-31 | 0.775 ± 0.030 | 0.756 | 0.755 |
| deception_no_rm_preyesno | 0.03 | 2000 | 235.3 | 15-18 | 0.739 ± 0.033 | 0.705 | 0.703 |
| bias_followed_rm_answer | 0.01 | 8000 | 204.5 | 24-27 | 0.842 ± 0.033 | 0.838 | 0.836 |

## Per probe

### bias_rm_answer  (908 train rows, 787 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 89.1 | 13-16 | 0.886 ± 0.008 | 0.866 ± 0.023 [31/32/32/32/30] | 0.866 ± 0.023 | 174 |
| 0.01 | 4000 | 178.2 | 13-16 | 0.887 ± 0.008 **←** | 0.869 ± 0.019 [30/29/31/31/32] | 0.867 ± 0.019 | 328 |
| 0.01 | 8000 | 356.4 | 13-16 | 0.883 ± 0.011 | 0.869 ± 0.016 [30/32/31/31/31] | 0.868 ± 0.016 | 322 |
| 0.03 | 2000 | 89.1 | 14-17 | 0.886 ± 0.009 | 0.871 ± 0.016 [31/32/31/32/31] | 0.870 ± 0.016 | 158 |
| 0.03 | 4000 | 178.2 | 14-17 | 0.882 ± 0.008 | 0.868 ± 0.016 [27/29/31/31/32] | 0.865 ± 0.016 | 330 |

Chosen: lr 0.01, 4000 steps (141.0 epochs on the full train set), layers 13-16; OOF 0.887 ± 0.008 (pooled 0.882).

### deception_rm_preyesno  (952 train rows, 818 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 85.0 | 18-21 | 0.622 ± 0.006 | 0.604 ± 0.017 [4/19/32/32/32] | 0.603 ± 0.017 | 174 |
| 0.01 | 4000 | 170.1 | 19-22 | 0.629 ± 0.009 | 0.612 ± 0.024 [11/3/13/32/32] | 0.608 ± 0.020 | 327 |
| 0.03 | 2000 | 85.0 | 19-22 | 0.630 ± 0.012 | 0.617 ± 0.016 [4/15/24/32/18] | 0.610 ± 0.020 | 160 |
| 0.03 | 4000 | 170.1 | 19-22 | 0.636 ± 0.016 | 0.620 ± 0.028 [17/1/29/14/32] | 0.618 ± 0.026 | 330 |
| 0.03 | 8000 | 340.1 | 19-22 | 0.639 ± 0.017 | 0.627 ± 0.025 [27/18/24/28/30] | 0.626 ± 0.024 | 269 |
| 0.1 | 4000 | 170.1 | 19-22 | 0.638 ± 0.022 | 0.627 ± 0.032 [20/19/1/19/28] | 0.629 ± 0.032 | 144 |
| 0.1 | 8000 | 340.1 | 27-30 | 0.643 ± 0.023 **←** | 0.642 ± 0.025 [32/25/30/27/5] | 0.635 ± 0.029 | 257 |

Chosen: lr 0.1, 8000 steps (268.9 epochs on the full train set), layers 27-30; OOF 0.643 ± 0.023 (pooled 0.633).

### deception_yes_rm_preyesno  (680 train rows, 609 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 119.4 | 15-18 | 0.748 ± 0.030 | 0.713 ± 0.036 [30/26/3/26/21] | 0.726 ± 0.021 | 173 |
| 0.01 | 4000 | 238.8 | 15-18 | 0.757 ± 0.024 | 0.739 ± 0.029 [31/25/22/24/29] | 0.737 ± 0.027 | 326 |
| 0.03 | 2000 | 119.4 | 15-18 | 0.757 ± 0.022 | 0.736 ± 0.028 [27/30/8/27/22] | 0.739 ± 0.028 | 157 |
| 0.03 | 4000 | 238.8 | 15-18 | 0.758 ± 0.019 | 0.747 ± 0.026 [32/32/18/25/31] | 0.748 ± 0.027 | 331 |
| 0.03 | 8000 | 477.6 | 28-31 | 0.765 ± 0.031 | 0.754 ± 0.023 [32/28/32/30/28] | 0.751 ± 0.021 | 257 |
| 0.1 | 4000 | 238.8 | 28-31 | 0.769 ± 0.031 | 0.756 ± 0.026 [32/32/26/25/32] | 0.754 ± 0.025 | 128 |
| 0.1 | 8000 | 477.6 | 28-31 | 0.775 ± 0.030 **←** | 0.756 ± 0.021 [32/27/32/32/32] | 0.755 ± 0.020 | 254 |

Chosen: lr 0.1, 8000 steps (376.5 epochs on the full train set), layers 28-31; OOF 0.775 ± 0.030 (pooled 0.768).

### deception_no_rm_preyesno  (272 train rows, 236 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 300.3 | 15-18 | 0.732 ± 0.036 | 0.694 ± 0.037 [32/32/21/23/22] | 0.698 ± 0.035 | 176 |
| 0.01 | 4000 | 600.6 | 15-18 | 0.739 ± 0.030 | 0.708 ± 0.037 [32/28/20/28/32] | 0.707 ± 0.030 | 333 |
| 0.03 | 2000 | 300.3 | 15-18 | 0.739 ± 0.033 **←** | 0.705 ± 0.033 [32/32/31/27/32] | 0.703 ± 0.032 | 161 |
| 0.03 | 4000 | 600.6 | 15-18 | 0.738 ± 0.039 | 0.700 ± 0.042 [32/31/27/32/32] | 0.698 ± 0.038 | 333 |
| 0.1 | 2000 | 300.3 | 18-21 | 0.734 ± 0.037 | 0.701 ± 0.040 [29/32/27/32/32] | 0.698 ± 0.036 | 68 |

Chosen: lr 0.03, 2000 steps (235.3 epochs on the full train set), layers 15-18; OOF 0.739 ± 0.033 (pooled 0.713).

### bias_followed_rm_answer  (1252 train rows, 949 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 64.5 | 13-16 | 0.840 ± 0.028 | 0.842 ± 0.032 [23/22/23/20/23] | 0.832 ± 0.034 | 175 |
| 0.01 | 4000 | 128.9 | 14-17 | 0.842 ± 0.029 | 0.842 ± 0.033 [27/23/30/25/25] | 0.838 ± 0.033 | 328 |
| 0.01 | 8000 | 257.9 | 24-27 | 0.842 ± 0.033 **←** | 0.838 ± 0.033 [30/32/32/31/29] | 0.836 ± 0.032 | 247 |
| 0.03 | 2000 | 64.5 | 21-24 | 0.840 ± 0.035 | 0.844 ± 0.032 [24/24/25/23/25] | 0.835 ± 0.035 | 160 |
| 0.03 | 4000 | 128.9 | 24-27 | 0.841 ± 0.034 | 0.838 ± 0.031 [30/24/29/30/25] | 0.835 ± 0.032 | 331 |

Chosen: lr 0.01, 8000 steps (204.5 epochs on the full train set), layers 24-27; OOF 0.842 ± 0.033 (pooled 0.841).
