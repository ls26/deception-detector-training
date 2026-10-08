# CV selection, arm ref_cot

5-fold GroupKFold over train question_ids (seed 0); grid lr x steps from cv_sweep.py (plus cv_edge.py cells); bands of width [4]; combined score = mean_sigmoid. The eval split was not used. spec_hash 10b44b32b4c2.

| probe | lr | steps | epochs (full train) | band | OOF ± fold sd | loss rule | all 32 |
|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.01 | 4000 | 141.0 | 13-16 | 0.863 ± 0.024 | 0.831 | 0.831 |
| deception_rm_preyesno | 0.1 | 8000 | 262.3 | 17-20 | 0.603 ± 0.043 | 0.590 | 0.592 |
| deception_yes_rm_preyesno | 0.1 | 4000 | 188.2 | 15-18 | 0.724 ± 0.032 | 0.709 | 0.711 |
| deception_no_rm_preyesno | 0.1 | 8000 | 864.9 | 22-25 | 0.792 ± 0.012 | 0.744 | 0.744 |
| bias_followed_rm_answer | 0.01 | 4000 | 102.2 | 13-16 | 0.764 ± 0.018 | 0.735 | 0.734 |

## Per probe

### bias_rm_answer  (908 train rows, 782 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 89.4 | 13-16 | 0.862 ± 0.023 | 0.827 ± 0.027 [32/31/30/32/32] | 0.826 ± 0.027 | 175 |
| 0.01 | 4000 | 178.8 | 13-16 | 0.863 ± 0.024 **←** | 0.831 ± 0.028 [32/32/32/32/32] | 0.831 ± 0.028 | 342 |
| 0.01 | 8000 | 357.6 | 13-16 | 0.859 ± 0.027 | 0.836 ± 0.030 [32/32/32/31/32] | 0.835 ± 0.030 | 322 |
| 0.03 | 2000 | 89.4 | 13-16 | 0.860 ± 0.025 | 0.833 ± 0.029 [31/31/30/32/32] | 0.833 ± 0.028 | 169 |
| 0.03 | 4000 | 178.8 | 13-16 | 0.857 ± 0.027 | 0.833 ± 0.030 [32/32/32/32/32] | 0.833 ± 0.030 | 377 |

Chosen: lr 0.01, 4000 steps (141.0 epochs on the full train set), layers 13-16; OOF 0.863 ± 0.024 (pooled 0.858).

### deception_rm_preyesno  (976 train rows, 833 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 82.1 | 14-17 | 0.586 ± 0.035 | 0.589 ± 0.029 [3/32/32/6/32] | 0.584 ± 0.033 | 180 |
| 0.01 | 4000 | 164.3 | 15-18 | 0.590 ± 0.041 | 0.587 ± 0.044 [2/2/32/32/9] | 0.588 ± 0.041 | 341 |
| 0.03 | 2000 | 82.1 | 15-18 | 0.590 ± 0.044 | 0.582 ± 0.051 [14/4/20/30/32] | 0.592 ± 0.046 | 170 |
| 0.03 | 4000 | 164.3 | 17-20 | 0.592 ± 0.043 | 0.593 ± 0.041 [15/32/18/13/32] | 0.593 ± 0.050 | 376 |
| 0.03 | 8000 | 328.6 | 17-20 | 0.599 ± 0.046 | 0.587 ± 0.055 [32/28/32/5/15] | 0.593 ± 0.052 | 267 |
| 0.1 | 4000 | 164.3 | 17-20 | 0.599 ± 0.049 | 0.597 ± 0.049 [29/32/22/3/32] | 0.592 ± 0.052 | 155 |
| 0.1 | 8000 | 328.6 | 17-20 | 0.603 ± 0.043 **←** | 0.590 ± 0.053 [32/32/32/15/27] | 0.592 ± 0.051 | 256 |

Chosen: lr 0.1, 8000 steps (262.3 epochs on the full train set), layers 17-20; OOF 0.603 ± 0.043 (pooled 0.602).

### deception_yes_rm_preyesno  (680 train rows, 604 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 119.8 | 14-17 | 0.710 ± 0.057 | 0.693 ± 0.060 [32/12/20/30/29] | 0.681 ± 0.062 | 179 |
| 0.01 | 4000 | 239.5 | 14-17 | 0.718 ± 0.051 | 0.698 ± 0.056 [25/20/32/23/31] | 0.691 ± 0.059 | 339 |
| 0.03 | 2000 | 119.8 | 14-17 | 0.720 ± 0.046 | 0.704 ± 0.051 [32/22/22/31/30] | 0.696 ± 0.054 | 168 |
| 0.03 | 4000 | 239.5 | 15-18 | 0.723 ± 0.040 | 0.709 ± 0.046 [31/24/32/27/32] | 0.707 ± 0.049 | 377 |
| 0.03 | 8000 | 479.1 | 15-18 | 0.723 ± 0.034 | 0.709 ± 0.042 [32/32/32/31/32] | 0.709 ± 0.042 | 263 |
| 0.1 | 4000 | 239.5 | 15-18 | 0.724 ± 0.032 **←** | 0.709 ± 0.036 [31/26/32/23/32] | 0.711 ± 0.038 | 131 |
| 0.1 | 8000 | 479.1 | 17-20 | 0.723 ± 0.027 | 0.707 ± 0.037 [32/32/32/32/31] | 0.707 ± 0.037 | 261 |

Chosen: lr 0.1, 4000 steps (188.2 epochs on the full train set), layers 15-18; OOF 0.724 ± 0.032 (pooled 0.703).

### deception_no_rm_preyesno  (296 train rows, 270 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 274.6 | 14-17 | 0.771 ± 0.042 | 0.728 ± 0.045 [31/27/26/32/32] | 0.723 ± 0.045 | 176 |
| 0.01 | 4000 | 549.1 | 14-17 | 0.783 ± 0.038 | 0.740 ± 0.036 [32/32/32/32/32] | 0.740 ± 0.036 | 346 |
| 0.03 | 2000 | 274.6 | 20-23 | 0.783 ± 0.026 | 0.742 ± 0.036 [32/32/32/32/32] | 0.742 ± 0.036 | 173 |
| 0.03 | 4000 | 549.1 | 20-23 | 0.786 ± 0.021 | 0.746 ± 0.031 [32/32/32/32/32] | 0.746 ± 0.031 | 382 |
| 0.03 | 8000 | 1098.2 | 23-26 | 0.785 ± 0.014 | 0.748 ± 0.020 [32/32/32/32/32] | 0.748 ± 0.020 | 266 |
| 0.1 | 4000 | 549.1 | 23-26 | 0.791 ± 0.017 | 0.745 ± 0.018 [32/32/32/32/32] | 0.745 ± 0.018 | 135 |
| 0.1 | 8000 | 1098.2 | 22-25 | 0.792 ± 0.012 **←** | 0.744 ± 0.013 [32/32/32/32/32] | 0.744 ± 0.013 | 250 |

Chosen: lr 0.1, 8000 steps (864.9 epochs on the full train set), layers 22-25; OOF 0.792 ± 0.012 (pooled 0.785).

### bias_followed_rm_answer  (1252 train rows, 930 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 64.6 | 13-16 | 0.759 ± 0.019 | 0.723 ± 0.027 [27/27/32/32/32] | 0.721 ± 0.029 | 178 |
| 0.01 | 4000 | 129.1 | 13-16 | 0.764 ± 0.018 **←** | 0.735 ± 0.027 [30/32/32/29/32] | 0.734 ± 0.028 | 338 |
| 0.01 | 8000 | 258.3 | 13-16 | 0.763 ± 0.020 | 0.738 ± 0.027 [31/32/31/32/32] | 0.737 ± 0.027 | 211 |
| 0.03 | 2000 | 64.6 | 13-16 | 0.760 ± 0.016 | 0.735 ± 0.022 [26/28/32/32/32] | 0.732 ± 0.025 | 169 |
| 0.03 | 4000 | 129.1 | 13-16 | 0.760 ± 0.020 | 0.740 ± 0.029 [31/32/32/32/32] | 0.740 ± 0.029 | 375 |

Chosen: lr 0.01, 4000 steps (102.2 epochs on the full train set), layers 13-16; OOF 0.764 ± 0.018 (pooled 0.759).
