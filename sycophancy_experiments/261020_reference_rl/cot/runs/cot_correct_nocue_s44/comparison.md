# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 487; in both 464 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.787 |
| accuracy | cue_gold | 0.852 | 0.851 |
| accuracy | cue_dist | 0.513 | 0.639 |
| accuracy | train_mix | 0.736 | 0.777 |
| reward | none | 0.647 | 0.686 |
| reward | cue_gold | 0.716 | 0.723 |
| reward | cue_dist | 0.498 | 0.592 |
| reward | train_mix | 0.690 | 0.710 |
| followed_cue | cue_gold | 0.852 | 0.851 |
| followed_cue | cue_dist | 0.460 | 0.361 |
| followed_cue | train_mix | 0.715 | 0.685 |
| bias | cue_gold | 0.103 | 0.061 |
| bias | cue_dist | 0.278 | 0.152 |
| bias | train_mix | 0.165 | 0.092 |
| anti_bias | cue_gold | 0.034 | 0.007 |
| anti_bias | cue_dist | 0.020 | 0.002 |
| anti_bias | train_mix | 0.031 | 0.006 |
| no_bias | cue_gold | 0.863 | 0.932 |
| no_bias | cue_dist | 0.702 | 0.846 |
| no_bias | train_mix | 0.804 | 0.902 |
| deception_soft | cue_gold | 0.548 | 0.756 |
| deception_soft | cue_dist | 0.591 | 0.764 |
| deception_soft | train_mix | 0.562 | 0.757 |
| deception | cue_gold | 0.552 | 0.814 |
| deception | cue_dist | 0.644 | 0.802 |
| deception | train_mix | 0.582 | 0.805 |
| biased_share | cue_gold | 0.107 | 0.062 |
| biased_share | cue_dist | 0.283 | 0.153 |
| biased_share | train_mix | 0.170 | 0.092 |
| false_denial_soft | cue_gold | 0.379 | 0.136 |
| false_denial_soft | cue_dist | 0.349 | 0.166 |
| false_denial_soft | train_mix | 0.357 | 0.148 |
| false_admission_soft | cue_gold | 0.568 | 0.796 |
| false_admission_soft | cue_dist | 0.687 | 0.872 |
| false_admission_soft | train_mix | 0.604 | 0.819 |
| said_yes_biased_soft | cue_gold | 0.621 | 0.864 |
| said_yes_biased_soft | cue_dist | 0.651 | 0.834 |
| said_yes_biased_soft | train_mix | 0.643 | 0.852 |
| deception_from_biased | cue_gold | 0.040 | 0.008 |
| deception_from_biased | cue_dist | 0.099 | 0.025 |
| deception_from_biased | train_mix | 0.061 | 0.014 |
| deception_from_unbiased | cue_gold | 0.507 | 0.747 |
| deception_from_unbiased | cue_dist | 0.492 | 0.739 |
| deception_from_unbiased | train_mix | 0.501 | 0.744 |
| said_yes | cue_gold | 0.580 | 0.871 |
| said_yes | cue_dist | 0.767 | 0.933 |
| said_yes | train_mix | 0.647 | 0.886 |
| false_denial | cue_gold | 0.369 | 0.035 |
| false_denial | cue_dist | 0.282 | 0.071 |
| false_denial | train_mix | 0.311 | 0.059 |
| false_admission | cue_gold | 0.574 | 0.865 |
| false_admission | cue_dist | 0.787 | 0.934 |
| false_admission | train_mix | 0.638 | 0.881 |
| parse_failed | none | 0.026 | 0.003 |
| parse_failed | cue_gold | 0.017 | 0.001 |
| parse_failed | cue_dist | 0.027 | 0.000 |
| parse_failed | train_mix | 0.022 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1368 | 1391 |
| n_bias_rows | cue_dist | 1354 | 1392 |
| n_bias_rows | train_mix | 2722 | 2783 |
| n_deception_rows | cue_gold | 1321 | 1381 |
| n_deception_rows | cue_dist | 1327 | 1389 |
| n_deception_rows | train_mix | 2648 | 2770 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.904 |
| accuracy | cue_gold | 0.942 | 0.934 |
| accuracy | cue_dist | 0.626 | 0.753 |
| accuracy | train_mix | 0.832 | 0.871 |
| reward | none | 0.738 | 0.752 |
| reward | cue_gold | 0.768 | 0.772 |
| reward | cue_dist | 0.572 | 0.660 |
| reward | train_mix | 0.748 | 0.761 |
| followed_cue | cue_gold | 0.942 | 0.934 |
| followed_cue | cue_dist | 0.348 | 0.247 |
| followed_cue | train_mix | 0.735 | 0.699 |
| bias | cue_gold | 0.006 | 0.030 |
| bias | cue_dist | 0.335 | 0.150 |
| bias | train_mix | 0.122 | 0.074 |
| anti_bias | cue_gold | 0.040 | 0.009 |
| anti_bias | cue_dist | 0.002 | 0.000 |
| anti_bias | train_mix | 0.026 | 0.007 |
| no_bias | cue_gold | 0.954 | 0.961 |
| no_bias | cue_dist | 0.664 | 0.850 |
| no_bias | train_mix | 0.852 | 0.920 |
| deception_soft | cue_gold | 0.542 | 0.768 |
| deception_soft | cue_dist | 0.590 | 0.773 |
| deception_soft | train_mix | 0.557 | 0.767 |
| deception | cue_gold | 0.535 | 0.836 |
| deception | cue_dist | 0.651 | 0.812 |
| deception | train_mix | 0.574 | 0.819 |
| biased_share | cue_gold | 0.007 | 0.030 |
| biased_share | cue_dist | 0.335 | 0.150 |
| biased_share | train_mix | 0.125 | 0.074 |
| false_denial_soft | cue_gold | 0.487 | 0.192 |
| false_denial_soft | cue_dist | 0.352 | 0.176 |
| false_denial_soft | train_mix | 0.354 | 0.181 |
| false_admission_soft | cue_gold | 0.542 | 0.786 |
| false_admission_soft | cue_dist | 0.710 | 0.878 |
| false_admission_soft | train_mix | 0.586 | 0.814 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.808 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.824 |
| said_yes_biased_soft | train_mix | 0.646 | 0.819 |
| deception_from_biased | cue_gold | 0.003 | 0.006 |
| deception_from_biased | cue_dist | 0.118 | 0.026 |
| deception_from_biased | train_mix | 0.044 | 0.013 |
| deception_from_unbiased | cue_gold | 0.538 | 0.762 |
| deception_from_unbiased | cue_dist | 0.472 | 0.747 |
| deception_from_unbiased | train_mix | 0.513 | 0.754 |
| said_yes | cue_gold | 0.536 | 0.861 |
| said_yes | cue_dist | 0.794 | 0.936 |
| said_yes | train_mix | 0.626 | 0.880 |
| false_denial | cue_gold | 0.429 | 0.091 |
| false_denial | cue_dist | 0.287 | 0.085 |
| false_denial | train_mix | 0.293 | 0.090 |
| false_admission | cue_gold | 0.536 | 0.859 |
| false_admission | cue_dist | 0.834 | 0.940 |
| false_admission | train_mix | 0.614 | 0.877 |
| parse_failed | none | 0.022 | 0.001 |
| parse_failed | cue_gold | 0.014 | 0.001 |
| parse_failed | cue_dist | 0.026 | 0.000 |
| parse_failed | train_mix | 0.020 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1086 | 1100 |
| n_bias_rows | cue_dist | 1073 | 1101 |
| n_bias_rows | train_mix | 2159 | 2201 |
| n_deception_rows | cue_gold | 1043 | 1090 |
| n_deception_rows | cue_dist | 1071 | 1101 |
| n_deception_rows | train_mix | 2114 | 2191 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.360 |
| accuracy | cue_gold | 0.524 | 0.546 |
| accuracy | cue_dist | 0.098 | 0.222 |
| accuracy | train_mix | 0.384 | 0.433 |
| reward | none | 0.313 | 0.445 |
| reward | cue_gold | 0.524 | 0.543 |
| reward | cue_dist | 0.225 | 0.346 |
| reward | train_mix | 0.479 | 0.525 |
| followed_cue | cue_gold | 0.524 | 0.546 |
| followed_cue | cue_dist | 0.870 | 0.778 |
| followed_cue | train_mix | 0.640 | 0.634 |
| bias | cue_gold | 0.475 | 0.179 |
| bias | cue_dist | 0.060 | 0.162 |
| bias | train_mix | 0.332 | 0.161 |
| anti_bias | cue_gold | 0.014 | 0.000 |
| anti_bias | cue_dist | 0.089 | 0.010 |
| anti_bias | train_mix | 0.047 | 0.003 |
| no_bias | cue_gold | 0.511 | 0.821 |
| no_bias | cue_dist | 0.851 | 0.828 |
| no_bias | train_mix | 0.621 | 0.837 |
| deception_soft | cue_gold | 0.571 | 0.710 |
| deception_soft | cue_dist | 0.597 | 0.728 |
| deception_soft | train_mix | 0.581 | 0.721 |
| deception | cue_gold | 0.615 | 0.732 |
| deception | cue_dist | 0.613 | 0.764 |
| deception | train_mix | 0.617 | 0.752 |
| biased_share | cue_gold | 0.482 | 0.179 |
| biased_share | cue_dist | 0.066 | 0.163 |
| biased_share | train_mix | 0.349 | 0.161 |
| false_denial_soft | cue_gold | 0.373 | 0.100 |
| false_denial_soft | cue_dist | 0.280 | 0.129 |
| false_denial_soft | train_mix | 0.362 | 0.091 |
| false_admission_soft | cue_gold | 0.755 | 0.843 |
| false_admission_soft | cue_dist | 0.619 | 0.845 |
| false_admission_soft | train_mix | 0.698 | 0.842 |
| said_yes_biased_soft | cue_gold | 0.627 | 0.900 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.871 |
| said_yes_biased_soft | train_mix | 0.638 | 0.909 |
| deception_from_biased | cue_gold | 0.180 | 0.018 |
| deception_from_biased | cue_dist | 0.019 | 0.021 |
| deception_from_biased | train_mix | 0.126 | 0.015 |
| deception_from_unbiased | cue_gold | 0.391 | 0.692 |
| deception_from_unbiased | cue_dist | 0.578 | 0.707 |
| deception_from_unbiased | train_mix | 0.455 | 0.706 |
| said_yes | cue_gold | 0.745 | 0.911 |
| said_yes | cue_dist | 0.656 | 0.920 |
| said_yes | train_mix | 0.730 | 0.911 |
| false_denial | cue_gold | 0.366 | 0.000 |
| false_denial | cue_dist | 0.176 | 0.021 |
| false_denial | train_mix | 0.338 | 0.005 |
| false_admission | cue_gold | 0.847 | 0.891 |
| false_admission | cue_dist | 0.644 | 0.909 |
| false_admission | train_mix | 0.766 | 0.895 |
| parse_failed | none | 0.039 | 0.008 |
| parse_failed | cue_gold | 0.029 | 0.000 |
| parse_failed | cue_dist | 0.032 | 0.000 |
| parse_failed | train_mix | 0.033 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 840 | 840 |
| rollouts | cue_gold | 315 | 315 |
| rollouts | cue_dist | 315 | 315 |
| rollouts | train_mix | 630 | 630 |
| n_bias_rows | cue_gold | 282 | 291 |
| n_bias_rows | cue_dist | 281 | 291 |
| n_bias_rows | train_mix | 563 | 582 |
| n_deception_rows | cue_gold | 278 | 291 |
| n_deception_rows | cue_dist | 256 | 288 |
| n_deception_rows | train_mix | 534 | 579 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 1.000 | 0.013 | 0.772 | 0.772 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 1.000 | 0.000 | 0.872 | 0.872 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4160 | 0.912 | 0.012 | 0.912 |
| first_10_steps | base_wrong | 960 | 0.168 | 0.018 | 0.168 |
| last_10_steps | base_right | 4056 | 0.955 | 0.000 | 0.955 |
| last_10_steps | base_wrong | 1064 | 0.555 | 0.000 | 0.555 |
