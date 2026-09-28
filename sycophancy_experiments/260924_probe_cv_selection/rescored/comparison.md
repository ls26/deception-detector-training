# 260923 rounds re-scored under the combined-score headline

Eval AUC, mean ± sd over the round's seeds, on the round's own saved layer set.

## eval_filter=none, layers=saved, metric=auc_mean_sigmoid

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.788 ± 0.002 | — | 0.793 ± 0.001 | 0.749 ± 0.000 |
| 08_cv_final_pool_rest | 0.840 ± 0.001 | — | 0.590 ± 0.004 | — | — |

## eval_filter=none, layers=saved, metric=auc_mean_logit

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.788 ± 0.002 | — | 0.793 ± 0.001 | 0.749 ± 0.000 |
| 08_cv_final_pool_rest | 0.840 ± 0.001 | — | 0.590 ± 0.004 | — | — |

## eval_filter=none, layers=saved, metric=auc_filtered

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.781 ± 0.001 | — | 0.789 ± 0.001 | 0.746 ± 0.000 |
| 08_cv_final_pool_rest | 0.834 ± 0.001 | — | 0.589 ± 0.004 | — | — |

## eval_filter=none_unanimous, layers=saved, metric=auc_mean_sigmoid

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.873 ± 0.001 | — | 0.847 ± 0.002 | 0.799 ± 0.000 |
| 08_cv_final_pool_rest | 0.904 ± 0.001 | — | 0.617 ± 0.005 | — | — |

## eval_filter=none_unanimous, layers=saved, metric=auc_mean_logit

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.873 ± 0.001 | — | 0.847 ± 0.002 | 0.799 ± 0.000 |
| 08_cv_final_pool_rest | 0.904 ± 0.001 | — | 0.617 ± 0.005 | — | — |

## eval_filter=none_unanimous, layers=saved, metric=auc_filtered

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 07_cv_final_pool_starred | — | 0.866 ± 0.000 | — | 0.842 ± 0.002 | 0.797 ± 0.000 |
| 08_cv_final_pool_rest | 0.899 ± 0.001 | — | 0.614 ± 0.005 | — | — |

## eval rows per probe (n, pos/neg)

| eval_filter | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| none | — | 1106 (449/657) | — | 836 (634/202) | 490 (120/370) |
| none_unanimous | — | 612 (223/389) | — | 490 (399/91) | 293 (56/237) |
