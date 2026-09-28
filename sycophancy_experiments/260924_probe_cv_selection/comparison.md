# Comparison: 260923 rounds vs this folder's rounds

Eval AUC, mean ± sd over seeds. Rounds trained here use their CV-chosen per-probe recipe (table at the end).

## auc_mean_sigmoid

AUC of the mean per-layer sigmoid over the selected layers (headline; the RL pipeline's score)

| round | eval filter | source | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|---|---|
| 01_baseline | none | re-scored saved weights | 0.795 ± 0.000 | 0.719 ± 0.017 | 0.477 ± 0.005 | 0.738 ± 0.001 | 0.650 ± 0.000 |
| 01_baseline | none_unanimous | re-scored saved weights | 0.851 ± 0.001 | 0.800 ± 0.025 | 0.474 ± 0.005 | 0.794 ± 0.000 | 0.682 ± 0.001 |
| 10_poolall_logreg | none | re-scored saved weights | 0.829 ± 0.000 | 0.764 ± 0.000 | 0.589 ± 0.000 | 0.765 ± 0.000 | 0.707 ± 0.000 |
| 10_poolall_logreg | none_unanimous | re-scored saved weights | 0.899 ± 0.000 | 0.846 ± 0.000 | 0.612 ± 0.000 | 0.817 ± 0.000 | 0.759 ± 0.000 |
| 15_poolall_lda05 | none | re-scored saved weights | 0.828 ± 0.000 | 0.762 ± 0.000 | 0.598 ± 0.000 | 0.774 ± 0.000 | 0.732 ± 0.000 |
| 15_poolall_lda05 | none_unanimous | re-scored saved weights | 0.897 ± 0.000 | 0.845 ± 0.000 | 0.618 ± 0.000 | 0.832 ± 0.000 | 0.788 ± 0.000 |
| 31_poolall_sgd_lr1e-2 | none | re-scored saved weights | 0.818 ± 0.002 | 0.745 ± 0.009 | 0.576 ± 0.007 | 0.760 ± 0.003 | 0.706 ± 0.007 |
| 31_poolall_sgd_lr1e-2 | none_unanimous | re-scored saved weights | 0.884 ± 0.001 | 0.824 ± 0.012 | 0.598 ± 0.008 | 0.814 ± 0.002 | 0.761 ± 0.010 |
| 01_cv_chosen_pool_starred | none | trained here on pool | — | 0.782 ± 0.003 | — | 0.778 ± 0.002 | 0.735 ± 0.002 |
| 01_cv_chosen_pool_starred | none_unanimous | re-scored saved weights | — | 0.871 ± 0.002 | — | 0.832 ± 0.002 | 0.796 ± 0.002 |
| 02_cv_chosen_pool_rest | none | trained here on pool | 0.837 ± 0.003 | — | 0.580 ± 0.003 | — | — |
| 02_cv_chosen_pool_rest | none_unanimous | re-scored saved weights | 0.903 ± 0.002 | — | 0.601 ± 0.004 | — | — |
| 03_cv_chosen_lossrule_pool | none | trained here on pool | 0.824 ± 0.004 | 0.756 ± 0.007 | 0.580 ± 0.007 | 0.766 ± 0.011 | 0.712 ± 0.005 |
| 03_cv_chosen_lossrule_pool | none_unanimous | re-scored saved weights | 0.892 ± 0.002 | 0.840 ± 0.012 | 0.601 ± 0.007 | 0.822 ± 0.009 | 0.767 ± 0.005 |
| 04_cv_chosen_pool_nu_evalnu | none_unanimous | trained here on pool_nu | 0.889 ± 0.001 | 0.850 ± 0.002 | 0.617 ± 0.001 | 0.831 ± 0.003 | 0.767 ± 0.001 |
| 04_cv_chosen_pool_nu_evalnu | none | re-scored saved weights | 0.822 ± 0.001 | 0.762 ± 0.001 | 0.599 ± 0.002 | 0.775 ± 0.001 | 0.714 ± 0.001 |
| 05_cv_chosen2000_pool_starred | none | trained here on pool | — | 0.782 ± 0.002 | — | 0.786 ± 0.001 | 0.746 ± 0.003 |
| 05_cv_chosen2000_pool_starred | none_unanimous | re-scored saved weights | — | 0.869 ± 0.002 | — | 0.839 ± 0.002 | 0.804 ± 0.002 |
| 06_cv_chosen2000_pool_rest | none | trained here on pool | 0.839 ± 0.002 | — | 0.601 ± 0.003 | — | — |
| 06_cv_chosen2000_pool_rest | none_unanimous | re-scored saved weights | 0.903 ± 0.001 | — | 0.628 ± 0.003 | — | — |
| 07_cv_final_pool_starred | none | trained here on pool | — | 0.788 ± 0.002 | — | 0.793 ± 0.001 | 0.749 ± 0.000 |
| 07_cv_final_pool_starred | none_unanimous | re-scored saved weights | — | 0.873 ± 0.001 | — | 0.847 ± 0.002 | 0.799 ± 0.000 |
| 08_cv_final_pool_rest | none | trained here on pool | 0.840 ± 0.001 | — | 0.590 ± 0.004 | — | — |
| 08_cv_final_pool_rest | none_unanimous | re-scored saved weights | 0.904 ± 0.001 | — | 0.617 ± 0.005 | — | — |

## auc_mean_logit

AUC of the mean per-layer logit over the selected layers

| round | eval filter | source | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|---|---|
| 01_baseline | none | re-scored saved weights | 0.796 ± 0.000 | 0.719 ± 0.017 | 0.477 ± 0.005 | 0.738 ± 0.000 | 0.650 ± 0.000 |
| 01_baseline | none_unanimous | re-scored saved weights | 0.852 ± 0.001 | 0.800 ± 0.025 | 0.474 ± 0.005 | 0.795 ± 0.000 | 0.682 ± 0.001 |
| 10_poolall_logreg | none | re-scored saved weights | 0.829 ± 0.000 | 0.765 ± 0.000 | 0.589 ± 0.000 | 0.765 ± 0.000 | 0.707 ± 0.000 |
| 10_poolall_logreg | none_unanimous | re-scored saved weights | 0.899 ± 0.000 | 0.849 ± 0.000 | 0.612 ± 0.000 | 0.816 ± 0.000 | 0.758 ± 0.000 |
| 15_poolall_lda05 | none | re-scored saved weights | 0.829 ± 0.000 | 0.767 ± 0.000 | 0.599 ± 0.000 | 0.778 ± 0.000 | 0.734 ± 0.000 |
| 15_poolall_lda05 | none_unanimous | re-scored saved weights | 0.899 ± 0.000 | 0.851 ± 0.000 | 0.619 ± 0.000 | 0.834 ± 0.000 | 0.784 ± 0.000 |
| 31_poolall_sgd_lr1e-2 | none | re-scored saved weights | 0.818 ± 0.002 | 0.746 ± 0.008 | 0.576 ± 0.007 | 0.760 ± 0.003 | 0.707 ± 0.007 |
| 31_poolall_sgd_lr1e-2 | none_unanimous | re-scored saved weights | 0.884 ± 0.001 | 0.825 ± 0.012 | 0.598 ± 0.008 | 0.814 ± 0.003 | 0.761 ± 0.010 |
| 01_cv_chosen_pool_starred | none | trained here on pool | — | 0.783 ± 0.003 | — | 0.778 ± 0.002 | 0.735 ± 0.002 |
| 01_cv_chosen_pool_starred | none_unanimous | re-scored saved weights | — | 0.871 ± 0.003 | — | 0.832 ± 0.002 | 0.796 ± 0.002 |
| 02_cv_chosen_pool_rest | none | trained here on pool | 0.837 ± 0.003 | — | 0.580 ± 0.003 | — | — |
| 02_cv_chosen_pool_rest | none_unanimous | re-scored saved weights | 0.903 ± 0.002 | — | 0.602 ± 0.004 | — | — |
| 03_cv_chosen_lossrule_pool | none | trained here on pool | 0.824 ± 0.003 | 0.758 ± 0.007 | 0.581 ± 0.006 | 0.766 ± 0.011 | 0.712 ± 0.005 |
| 03_cv_chosen_lossrule_pool | none_unanimous | re-scored saved weights | 0.893 ± 0.001 | 0.842 ± 0.012 | 0.602 ± 0.007 | 0.822 ± 0.009 | 0.766 ± 0.006 |
| 04_cv_chosen_pool_nu_evalnu | none_unanimous | trained here on pool_nu | 0.888 ± 0.001 | 0.851 ± 0.002 | 0.617 ± 0.001 | 0.831 ± 0.003 | 0.767 ± 0.001 |
| 04_cv_chosen_pool_nu_evalnu | none | re-scored saved weights | 0.822 ± 0.001 | 0.763 ± 0.001 | 0.599 ± 0.002 | 0.775 ± 0.001 | 0.714 ± 0.002 |
| 05_cv_chosen2000_pool_starred | none | trained here on pool | — | 0.782 ± 0.002 | — | 0.786 ± 0.001 | 0.746 ± 0.003 |
| 05_cv_chosen2000_pool_starred | none_unanimous | re-scored saved weights | — | 0.870 ± 0.002 | — | 0.839 ± 0.002 | 0.804 ± 0.002 |
| 06_cv_chosen2000_pool_rest | none | trained here on pool | 0.839 ± 0.002 | — | 0.601 ± 0.003 | — | — |
| 06_cv_chosen2000_pool_rest | none_unanimous | re-scored saved weights | 0.903 ± 0.000 | — | 0.628 ± 0.003 | — | — |
| 07_cv_final_pool_starred | none | trained here on pool | — | 0.788 ± 0.002 | — | 0.793 ± 0.001 | 0.749 ± 0.000 |
| 07_cv_final_pool_starred | none_unanimous | re-scored saved weights | — | 0.873 ± 0.001 | — | 0.847 ± 0.002 | 0.799 ± 0.000 |
| 08_cv_final_pool_rest | none | trained here on pool | 0.840 ± 0.001 | — | 0.590 ± 0.004 | — | — |
| 08_cv_final_pool_rest | none_unanimous | re-scored saved weights | 0.904 ± 0.001 | — | 0.617 ± 0.005 | — | — |

## auc_filtered

mean of the per-layer AUCs over the selected layers (260923's headline)

| round | eval filter | source | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|---|---|
| 01_baseline | none | re-scored saved weights | 0.774 ± 0.000 | 0.705 ± 0.026 | 0.475 ± 0.004 | 0.689 ± 0.000 | 0.644 ± 0.003 |
| 01_baseline | none_unanimous | re-scored saved weights | 0.829 ± 0.000 | 0.780 ± 0.038 | 0.472 ± 0.005 | 0.741 ± 0.001 | 0.674 ± 0.003 |
| 10_poolall_logreg | none | re-scored saved weights | 0.798 ± 0.000 | 0.725 ± 0.000 | 0.566 ± 0.000 | 0.751 ± 0.000 | 0.700 ± 0.000 |
| 10_poolall_logreg | none_unanimous | re-scored saved weights | 0.860 ± 0.000 | 0.797 ± 0.000 | 0.582 ± 0.000 | 0.800 ± 0.000 | 0.748 ± 0.000 |
| 15_poolall_lda05 | none | re-scored saved weights | 0.799 ± 0.000 | 0.727 ± 0.000 | 0.586 ± 0.000 | 0.736 ± 0.000 | 0.704 ± 0.000 |
| 15_poolall_lda05 | none_unanimous | re-scored saved weights | 0.862 ± 0.000 | 0.799 ± 0.000 | 0.602 ± 0.000 | 0.790 ± 0.000 | 0.749 ± 0.000 |
| 31_poolall_sgd_lr1e-2 | none | re-scored saved weights | 0.795 ± 0.001 | 0.725 ± 0.014 | 0.554 ± 0.006 | 0.747 ± 0.008 | 0.704 ± 0.008 |
| 31_poolall_sgd_lr1e-2 | none_unanimous | re-scored saved weights | 0.854 ± 0.002 | 0.797 ± 0.019 | 0.568 ± 0.007 | 0.800 ± 0.006 | 0.758 ± 0.011 |
| 01_cv_chosen_pool_starred | none | trained here on pool | — | 0.776 ± 0.002 | — | 0.774 ± 0.001 | 0.732 ± 0.001 |
| 01_cv_chosen_pool_starred | none_unanimous | re-scored saved weights | — | 0.864 ± 0.002 | — | 0.828 ± 0.002 | 0.791 ± 0.001 |
| 02_cv_chosen_pool_rest | none | trained here on pool | 0.832 ± 0.003 | — | 0.579 ± 0.003 | — | — |
| 02_cv_chosen_pool_rest | none_unanimous | re-scored saved weights | 0.898 ± 0.002 | — | 0.600 ± 0.004 | — | — |
| 03_cv_chosen_lossrule_pool | none | trained here on pool | 0.795 ± 0.001 | 0.727 ± 0.015 | 0.559 ± 0.006 | 0.743 ± 0.029 | 0.701 ± 0.012 |
| 03_cv_chosen_lossrule_pool | none_unanimous | re-scored saved weights | 0.857 ± 0.001 | 0.802 ± 0.023 | 0.572 ± 0.007 | 0.797 ± 0.029 | 0.750 ± 0.015 |
| 04_cv_chosen_pool_nu_evalnu | none_unanimous | trained here on pool_nu | 0.883 ± 0.001 | 0.844 ± 0.002 | 0.616 ± 0.001 | 0.827 ± 0.002 | 0.760 ± 0.001 |
| 04_cv_chosen_pool_nu_evalnu | none | re-scored saved weights | 0.816 ± 0.001 | 0.756 ± 0.001 | 0.597 ± 0.002 | 0.770 ± 0.001 | 0.709 ± 0.001 |
| 05_cv_chosen2000_pool_starred | none | trained here on pool | — | 0.776 ± 0.002 | — | 0.782 ± 0.001 | 0.742 ± 0.003 |
| 05_cv_chosen2000_pool_starred | none_unanimous | re-scored saved weights | — | 0.863 ± 0.002 | — | 0.835 ± 0.002 | 0.799 ± 0.002 |
| 06_cv_chosen2000_pool_rest | none | trained here on pool | 0.833 ± 0.002 | — | 0.600 ± 0.003 | — | — |
| 06_cv_chosen2000_pool_rest | none_unanimous | re-scored saved weights | 0.897 ± 0.000 | — | 0.627 ± 0.003 | — | — |
| 07_cv_final_pool_starred | none | trained here on pool | — | 0.781 ± 0.001 | — | 0.789 ± 0.001 | 0.746 ± 0.000 |
| 07_cv_final_pool_starred | none_unanimous | re-scored saved weights | — | 0.866 ± 0.000 | — | 0.842 ± 0.002 | 0.797 ± 0.000 |
| 08_cv_final_pool_rest | none | trained here on pool | 0.834 ± 0.001 | — | 0.589 ± 0.004 | — | — |
| 08_cv_final_pool_rest | none_unanimous | re-scored saved weights | 0.899 ± 0.001 | — | 0.614 ± 0.005 | — | — |

## eval rows (n, pos/neg)

| eval filter | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| none | 1326 (321/1005) | 1106 (449/657) | 1326 (754/572) | 836 (634/202) | 490 (120/370) |
| none_unanimous | 783 (147/636) | 612 (223/389) | 783 (455/328) | 490 (399/91) | 293 (56/237) |

## recipe per probe (rounds trained here)

| round | probe | n_train | lr | steps | epochs | layers |
|---|---|---|---|---|---|---|
| 01_cv_chosen_pool_starred | bias_followed | 2908 | 0.03 | 1000 | 11.0 | 13-16 (4, band) |
| 01_cv_chosen_pool_starred | deception_yes | 1928 | 0.03 | 1000 | 16.6 | 14-17 (4, band) |
| 01_cv_chosen_pool_starred | deception_no | 980 | 0.03 | 1000 | 32.7 | 14-17 (4, band) |
| 02_cv_chosen_pool_rest | bias | 2908 | 0.03 | 1000 | 11.0 | 13-16 (4, band) |
| 02_cv_chosen_pool_rest | deception | 2908 | 0.03 | 1000 | 11.0 | 14-17 (4, band) |
| 03_cv_chosen_lossrule_pool | bias | 2908 | 0.03 | 1000 | 11.0 | 1-31 (31, loss_threshold) |
| 03_cv_chosen_lossrule_pool | bias_followed | 2908 | 0.03 | 1000 | 11.0 | 0-31 (31, loss_threshold) |
| 03_cv_chosen_lossrule_pool | deception | 2908 | 0.03 | 1000 | 11.0 | 0-31 (32, loss_threshold) |
| 03_cv_chosen_lossrule_pool | deception_yes | 1928 | 0.03 | 1000 | 16.6 | 14-16 (3, loss_threshold) |
| 03_cv_chosen_lossrule_pool | deception_no | 980 | 0.03 | 1000 | 32.7 | 13-31 (19, loss_threshold) |
| 04_cv_chosen_pool_nu_evalnu | bias | 1152 | 0.03 | 396 | 11.0 | 13-16 (4, band) |
| 04_cv_chosen_pool_nu_evalnu | bias_followed | 1152 | 0.03 | 396 | 11.0 | 13-16 (4, band) |
| 04_cv_chosen_pool_nu_evalnu | deception | 1152 | 0.03 | 396 | 11.0 | 14-17 (4, band) |
| 04_cv_chosen_pool_nu_evalnu | deception_yes | 812 | 0.03 | 421 | 16.6 | 14-17 (4, band) |
| 04_cv_chosen_pool_nu_evalnu | deception_no | 340 | 0.03 | 347 | 32.7 | 14-17 (4, band) |
| 05_cv_chosen2000_pool_starred | bias_followed | 2908 | 0.01 | 2000 | 22.0 | 13-16 (4, band) |
| 05_cv_chosen2000_pool_starred | deception_yes | 1928 | 0.03 | 2000 | 33.2 | 14-17 (4, band) |
| 05_cv_chosen2000_pool_starred | deception_no | 980 | 0.03 | 2000 | 65.3 | 14-17 (4, band) |
| 06_cv_chosen2000_pool_rest | bias | 2908 | 0.03 | 2000 | 22.0 | 13-16 (4, band) |
| 06_cv_chosen2000_pool_rest | deception | 2908 | 0.03 | 2000 | 22.0 | 14-17 (4, band) |
| 07_cv_final_pool_starred | bias_followed | 2908 | 0.01 | 4000 | 44.0 | 13-16 (4, band) |
| 07_cv_final_pool_starred | deception_yes | 1928 | 0.03 | 4000 | 66.4 | 14-17 (4, band) |
| 07_cv_final_pool_starred | deception_no | 980 | 0.03 | 4000 | 130.6 | 15-18 (4, band) |
| 08_cv_final_pool_rest | bias | 2908 | 0.01 | 4000 | 44.0 | 13-16 (4, band) |
| 08_cv_final_pool_rest | deception | 2908 | 0.03 | 4000 | 44.0 | 14-17 (4, band) |

## selected layers (re-scored rounds, seed 0)

| round | bias | bias_followed | deception | deception_yes | deception_no |
|---|---|---|---|---|---|
| 01_baseline | 2-31 (30) | 1-31 (31) | 0-31 (32) | 0-31 (32) | 3-31 (26) |
| 10_poolall_logreg | 1-31 (31) | 2-31 (30) | 0-31 (32) | 10-31 (22) | 12-31 (20) |
| 15_poolall_lda05 | 1-31 (31) | 2-31 (30) | 3-31 (29) | 0-31 (32) | 0-31 (32) |
| 31_poolall_sgd_lr1e-2 | 2-31 (30) | 0-31 (29) | 0-31 (32) | 13-22 (10) | 14-25 (12) |
| 01_cv_chosen_pool_starred | — | 13-16 (4) | — | 14-17 (4) | 14-17 (4) |
| 02_cv_chosen_pool_rest | 13-16 (4) | — | 14-17 (4) | — | — |
| 03_cv_chosen_lossrule_pool | 1-31 (31) | 0-31 (31) | 0-31 (32) | 14-16 (3) | 13-31 (19) |
| 04_cv_chosen_pool_nu_evalnu | 13-16 (4) | 13-16 (4) | 14-17 (4) | 14-17 (4) | 14-17 (4) |
| 05_cv_chosen2000_pool_starred | — | 13-16 (4) | — | 14-17 (4) | 14-17 (4) |
| 06_cv_chosen2000_pool_rest | 13-16 (4) | — | 14-17 (4) | — | — |
| 07_cv_final_pool_starred | — | 13-16 (4) | — | 14-17 (4) | 15-18 (4) |
| 08_cv_final_pool_rest | 13-16 (4) | — | 14-17 (4) | — | — |
