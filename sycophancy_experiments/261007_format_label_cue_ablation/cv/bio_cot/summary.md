# CV selection, arm bio_cot

5-fold GroupKFold over train question_ids (seed 0); grid lr x steps from cv_sweep.py (plus cv_edge.py cells); bands of width [4]; combined score = mean_sigmoid. The eval split was not used. spec_hash 10b44b32b4c2.

| probe | lr | steps | epochs (full train) | band | OOF ± fold sd | loss rule | all 32 |
|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.01 | 2000 | 70.5 | 13-16 | 0.826 ± 0.028 | 0.804 | 0.804 |
| deception_rm_preyesno | 0.01 | 2000 | 78.8 | 14-17 | 0.661 ± 0.031 | 0.650 | 0.647 |
| deception_yes_rm_preyesno | 0.03 | 2000 | 94.1 | 14-17 | 0.724 ± 0.032 | 0.712 | 0.711 |
| deception_no_rm_preyesno | 0.1 | 2000 | 484.8 | 11-14 | 0.639 ± 0.127 | 0.592 | 0.592 |
| bias_followed_rm_answer | 0.01 | 2000 | 51.1 | 13-16 | 0.769 ± 0.050 | 0.726 | 0.722 |

## Per probe

### bias_rm_answer  (908 train rows, 797 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 88.7 | 13-16 | 0.826 ± 0.028 **←** | 0.804 ± 0.032 [16/32/32/30/31] | 0.804 ± 0.031 | 174 |
| 0.01 | 4000 | 177.5 | 13-16 | 0.819 ± 0.024 | 0.802 ± 0.030 [30/32/30/30/32] | 0.802 ± 0.030 | 318 |
| 0.03 | 2000 | 88.7 | 13-16 | 0.817 ± 0.022 | 0.803 ± 0.027 [29/31/32/31/31] | 0.802 ± 0.026 | 159 |
| 0.03 | 4000 | 177.5 | 13-16 | 0.807 ± 0.019 | 0.797 ± 0.029 [30/32/31/31/30] | 0.796 ± 0.029 | 309 |

Chosen: lr 0.01, 2000 steps (70.5 epochs on the full train set), layers 13-16; OOF 0.826 ± 0.028 (pooled 0.824).

### deception_rm_preyesno  (812 train rows, 724 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 99.7 | 14-17 | 0.661 ± 0.031 **←** | 0.650 ± 0.036 [7/14/9/32/23] | 0.647 ± 0.048 | 174 |
| 0.01 | 4000 | 199.4 | 14-17 | 0.658 ± 0.027 | 0.656 ± 0.036 [32/30/31/19/7] | 0.649 ± 0.042 | 317 |
| 0.03 | 2000 | 99.7 | 14-17 | 0.656 ± 0.025 | 0.650 ± 0.038 [21/24/31/30/31] | 0.650 ± 0.038 | 158 |
| 0.03 | 4000 | 199.4 | 23-26 | 0.653 ± 0.042 | 0.652 ± 0.046 [1/30/32/29/23] | 0.644 ± 0.035 | 308 |

Chosen: lr 0.01, 2000 steps (78.8 epochs on the full train set), layers 14-17; OOF 0.661 ± 0.031 (pooled 0.660).

### deception_yes_rm_preyesno  (680 train rows, 610 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 120.1 | 14-17 | 0.713 ± 0.037 | 0.695 ± 0.032 [14/32/28/27/22] | 0.692 ± 0.036 | 174 |
| 0.01 | 4000 | 240.1 | 14-17 | 0.719 ± 0.031 | 0.709 ± 0.020 [32/30/27/24/18] | 0.708 ± 0.029 | 317 |
| 0.03 | 2000 | 120.1 | 14-17 | 0.724 ± 0.032 **←** | 0.712 ± 0.033 [27/32/32/31/29] | 0.711 ± 0.033 | 158 |
| 0.03 | 4000 | 240.1 | 14-17 | 0.723 ± 0.031 | 0.718 ± 0.038 [27/31/32/32/24] | 0.716 ± 0.038 | 310 |
| 0.1 | 2000 | 120.1 | 26-29 | 0.723 ± 0.025 | 0.715 ± 0.041 [23/31/32/32/30] | 0.713 ± 0.039 | 86 |

Chosen: lr 0.03, 2000 steps (94.1 epochs on the full train set), layers 14-17; OOF 0.724 ± 0.032 (pooled 0.725).

### deception_no_rm_preyesno  (132 train rows, 127 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 638.2 | 10-13 | 0.630 ± 0.155 | 0.630 ± 0.129 [32/32/31/27/32] | 0.630 ± 0.129 | 177 |
| 0.01 | 4000 | 1276.4 | 10-13 | 0.628 ± 0.139 | 0.615 ± 0.127 [32/32/32/32/32] | 0.615 ± 0.127 | 348 |
| 0.03 | 2000 | 638.2 | 11-14 | 0.633 ± 0.139 | 0.608 ± 0.135 [32/32/32/32/32] | 0.608 ± 0.135 | 170 |
| 0.03 | 4000 | 1276.4 | 11-14 | 0.629 ± 0.121 | 0.595 ± 0.115 [32/32/32/32/32] | 0.595 ± 0.115 | 316 |
| 0.1 | 2000 | 638.2 | 11-14 | 0.639 ± 0.127 **←** | 0.592 ± 0.119 [32/32/32/32/32] | 0.592 ± 0.119 | 98 |

Chosen: lr 0.1, 2000 steps (484.8 epochs on the full train set), layers 11-14; OOF 0.639 ± 0.127 (pooled 0.637).

### bias_followed_rm_answer  (1252 train rows, 934 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 64.2 | 13-16 | 0.769 ± 0.050 **←** | 0.726 ± 0.043 [32/30/24/29/29] | 0.722 ± 0.043 | 174 |
| 0.01 | 4000 | 128.4 | 13-16 | 0.762 ± 0.051 | 0.726 ± 0.045 [32/32/32/31/29] | 0.726 ± 0.045 | 317 |
| 0.03 | 2000 | 64.2 | 13-16 | 0.758 ± 0.054 | 0.726 ± 0.048 [32/32/26/32/31] | 0.726 ± 0.048 | 158 |
| 0.03 | 4000 | 128.4 | 13-16 | 0.749 ± 0.054 | 0.722 ± 0.049 [32/32/32/31/30] | 0.722 ± 0.049 | 309 |

Chosen: lr 0.01, 2000 steps (51.1 epochs on the full train set), layers 13-16; OOF 0.769 ± 0.050 (pooled 0.764).
