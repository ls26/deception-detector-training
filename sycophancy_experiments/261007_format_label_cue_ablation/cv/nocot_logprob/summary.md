# CV selection, arm nocot_logprob

5-fold GroupKFold over train question_ids (seed 0); grid lr x steps from cv_sweep.py (plus cv_edge.py cells); bands of width [4]; combined score = mean_sigmoid. The eval split was not used. spec_hash 10b44b32b4c2.

| probe | lr | steps | epochs (full train) | band | OOF ± fold sd | loss rule | all 32 |
|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.01 | 2000 | 70.5 | 13-16 | 0.897 ± 0.021 | 0.884 | 0.881 |
| deception_rm_preyesno | 0.1 | 8000 | 272.3 | 18-21 | 0.693 ± 0.016 | 0.688 | 0.684 |
| deception_yes_rm_preyesno | 0.1 | 8000 | 376.5 | 18-21 | 0.811 ± 0.059 | 0.777 | 0.776 |
| deception_no_rm_preyesno | 0.1 | 8000 | 984.6 | 28-31 | 0.792 ± 0.026 | 0.766 | 0.761 |
| bias_followed_rm_answer | 0.03 | 2000 | 51.1 | 13-16 | 0.870 ± 0.013 | 0.858 | 0.857 |

## Per probe

### bias_rm_answer  (908 train rows, 804 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 88.8 | 13-16 | 0.897 ± 0.021 **←** | 0.884 ± 0.022 [31/30/30/26/29] | 0.881 ± 0.021 | 190 |
| 0.01 | 4000 | 177.6 | 13-16 | 0.896 ± 0.023 | 0.886 ± 0.023 [31/26/30/27/31] | 0.884 ± 0.023 | 355 |
| 0.03 | 2000 | 88.8 | 13-16 | 0.896 ± 0.022 | 0.885 ± 0.020 [32/31/31/30/29] | 0.883 ± 0.022 | 165 |
| 0.03 | 4000 | 177.6 | 13-16 | 0.892 ± 0.024 | 0.884 ± 0.022 [32/29/31/30/31] | 0.883 ± 0.021 | 356 |

Chosen: lr 0.01, 2000 steps (70.5 epochs on the full train set), layers 13-16; OOF 0.897 ± 0.021 (pooled 0.896).

### deception_rm_preyesno  (940 train rows, 799 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 85.9 | 19-22 | 0.652 ± 0.047 | 0.651 ± 0.040 [19/27/32/6/32] | 0.652 ± 0.040 | 193 |
| 0.01 | 4000 | 171.9 | 18-21 | 0.661 ± 0.037 | 0.648 ± 0.013 [32/32/18/32/1] | 0.658 ± 0.035 | 350 |
| 0.03 | 2000 | 85.9 | 18-21 | 0.665 ± 0.034 | 0.657 ± 0.022 [31/31/6/17/10] | 0.660 ± 0.033 | 171 |
| 0.03 | 4000 | 171.9 | 18-21 | 0.676 ± 0.022 | 0.666 ± 0.023 [26/4/32/32/15] | 0.669 ± 0.025 | 363 |
| 0.03 | 8000 | 343.8 | 19-22 | 0.685 ± 0.017 | 0.681 ± 0.015 [32/19/27/25/31] | 0.678 ± 0.019 | 302 |
| 0.1 | 4000 | 171.9 | 19-22 | 0.686 ± 0.012 | 0.682 ± 0.010 [26/32/23/30/21] | 0.681 ± 0.016 | 161 |
| 0.1 | 8000 | 343.8 | 18-21 | 0.693 ± 0.016 **←** | 0.688 ± 0.014 [32/19/26/32/31] | 0.684 ± 0.016 | 268 |

Chosen: lr 0.1, 8000 steps (272.3 epochs on the full train set), layers 18-21; OOF 0.693 ± 0.016 (pooled 0.688).

### deception_yes_rm_preyesno  (680 train rows, 604 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 120.5 | 14-17 | 0.766 ± 0.041 | 0.743 ± 0.043 [30/23/32/23/26] | 0.742 ± 0.044 | 191 |
| 0.01 | 4000 | 241.0 | 15-18 | 0.786 ± 0.045 | 0.757 ± 0.047 [31/32/29/24/30] | 0.756 ± 0.048 | 362 |
| 0.03 | 2000 | 120.5 | 15-18 | 0.789 ± 0.047 | 0.760 ± 0.050 [32/25/32/29/27] | 0.758 ± 0.049 | 159 |
| 0.03 | 4000 | 241.0 | 15-18 | 0.800 ± 0.052 | 0.774 ± 0.046 [25/32/30/26/32] | 0.769 ± 0.051 | 355 |
| 0.03 | 8000 | 482.0 | 18-21 | 0.805 ± 0.059 | 0.775 ± 0.057 [32/32/32/32/32] | 0.775 ± 0.057 | 260 |
| 0.1 | 4000 | 241.0 | 18-21 | 0.807 ± 0.062 | 0.781 ± 0.058 [31/32/29/27/27] | 0.775 ± 0.060 | 132 |
| 0.1 | 8000 | 482.0 | 18-21 | 0.811 ± 0.059 **←** | 0.777 ± 0.060 [31/32/32/32/32] | 0.776 ± 0.061 | 261 |

Chosen: lr 0.1, 8000 steps (376.5 epochs on the full train set), layers 18-21; OOF 0.811 ± 0.059 (pooled 0.807).

### deception_no_rm_preyesno  (260 train rows, 221 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 317.1 | 14-17 | 0.767 ± 0.045 | 0.755 ± 0.033 [29/32/21/25/22] | 0.752 ± 0.033 | 198 |
| 0.01 | 4000 | 634.1 | 14-17 | 0.779 ± 0.040 | 0.767 ± 0.034 [20/31/21/32/32] | 0.765 ± 0.032 | 357 |
| 0.03 | 2000 | 317.1 | 14-17 | 0.778 ± 0.040 | 0.764 ± 0.034 [32/32/26/32/32] | 0.764 ± 0.034 | 177 |
| 0.03 | 4000 | 634.1 | 28-31 | 0.783 ± 0.036 | 0.769 ± 0.030 [27/32/31/32/32] | 0.768 ± 0.029 | 359 |
| 0.03 | 8000 | 1268.3 | 28-31 | 0.791 ± 0.032 | 0.770 ± 0.026 [32/32/32/32/32] | 0.770 ± 0.026 | 273 |
| 0.1 | 4000 | 634.1 | 28-31 | 0.790 ± 0.035 | 0.770 ± 0.024 [32/32/29/32/32] | 0.768 ± 0.024 | 140 |
| 0.1 | 8000 | 1268.3 | 28-31 | 0.792 ± 0.026 **←** | 0.766 ± 0.016 [32/32/32/22/32] | 0.761 ± 0.022 | 270 |

Chosen: lr 0.1, 8000 steps (984.6 epochs on the full train set), layers 28-31; OOF 0.792 ± 0.026 (pooled 0.793).

### bias_followed_rm_answer  (1252 train rows, 941 questions)

Out-of-fold AUC of the combined score, mean ± sd over folds. `best band` is the argmax over contiguous bands; `loss rule` is the pipeline's train-loss<0.6 rule applied per fold (layers passing per fold in brackets).

| lr | steps | epochs | best band | OOF (best band) | loss rule | all 32 | s/fit |
|---|---|---|---|---|---|---|---|
| 0.01 | 2000 | 64.5 | 13-16 | 0.869 ± 0.012 | 0.857 ± 0.020 [28/24/20/24/29] | 0.852 ± 0.023 | 192 |
| 0.01 | 4000 | 129.0 | 13-16 | 0.870 ± 0.011 | 0.859 ± 0.015 [24/26/30/25/24] | 0.854 ± 0.018 | 359 |
| 0.03 | 2000 | 64.5 | 13-16 | 0.870 ± 0.013 **←** | 0.858 ± 0.025 [32/25/22/29/32] | 0.857 ± 0.025 | 161 |
| 0.03 | 4000 | 129.0 | 13-16 | 0.866 ± 0.012 | 0.854 ± 0.018 [24/27/31/29/28] | 0.850 ± 0.018 | 357 |
| 0.1 | 2000 | 64.5 | 13-16 | 0.863 ± 0.013 | 0.853 ± 0.026 [32/28/23/31/31] | 0.851 ± 0.026 | 67 |

Chosen: lr 0.03, 2000 steps (51.1 epochs on the full train set), layers 13-16; OOF 0.870 ± 0.013 (pooled 0.867).
