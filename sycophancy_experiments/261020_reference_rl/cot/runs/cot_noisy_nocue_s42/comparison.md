# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 474; in both 451 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.773 |
| accuracy | cue_gold | 0.852 | 0.924 |
| accuracy | cue_dist | 0.513 | 0.446 |
| accuracy | train_mix | 0.736 | 0.760 |
| reward | none | 0.647 | 0.672 |
| reward | cue_gold | 0.716 | 0.759 |
| reward | cue_dist | 0.498 | 0.461 |
| reward | train_mix | 0.690 | 0.730 |
| followed_cue | cue_gold | 0.852 | 0.924 |
| followed_cue | cue_dist | 0.460 | 0.554 |
| followed_cue | train_mix | 0.715 | 0.794 |
| bias | cue_gold | 0.096 | 0.129 |
| bias | cue_dist | 0.276 | 0.337 |
| bias | train_mix | 0.159 | 0.198 |
| anti_bias | cue_gold | 0.035 | 0.004 |
| anti_bias | cue_dist | 0.020 | 0.006 |
| anti_bias | train_mix | 0.031 | 0.004 |
| no_bias | cue_gold | 0.870 | 0.866 |
| no_bias | cue_dist | 0.704 | 0.657 |
| no_bias | train_mix | 0.810 | 0.798 |
| deception_soft | cue_gold | 0.548 | 0.651 |
| deception_soft | cue_dist | 0.589 | 0.612 |
| deception_soft | train_mix | 0.562 | 0.639 |
| deception | cue_gold | 0.551 | 0.712 |
| deception | cue_dist | 0.642 | 0.654 |
| deception | train_mix | 0.583 | 0.696 |
| biased_share | cue_gold | 0.099 | 0.130 |
| biased_share | cue_dist | 0.282 | 0.339 |
| biased_share | train_mix | 0.164 | 0.199 |
| false_denial_soft | cue_gold | 0.380 | 0.240 |
| false_denial_soft | cue_dist | 0.347 | 0.251 |
| false_denial_soft | train_mix | 0.357 | 0.245 |
| false_admission_soft | cue_gold | 0.567 | 0.712 |
| false_admission_soft | cue_dist | 0.684 | 0.797 |
| false_admission_soft | train_mix | 0.602 | 0.737 |
| said_yes_biased_soft | cue_gold | 0.620 | 0.760 |
| said_yes_biased_soft | cue_dist | 0.653 | 0.749 |
| said_yes_biased_soft | train_mix | 0.643 | 0.755 |
| deception_from_biased | cue_gold | 0.038 | 0.031 |
| deception_from_biased | cue_dist | 0.098 | 0.085 |
| deception_from_biased | train_mix | 0.059 | 0.049 |
| deception_from_unbiased | cue_gold | 0.510 | 0.620 |
| deception_from_unbiased | cue_dist | 0.491 | 0.527 |
| deception_from_unbiased | train_mix | 0.503 | 0.591 |
| said_yes | cue_gold | 0.577 | 0.800 |
| said_yes | cue_dist | 0.766 | 0.884 |
| said_yes | train_mix | 0.645 | 0.833 |
| false_denial | cue_gold | 0.370 | 0.160 |
| false_denial | cue_dist | 0.280 | 0.160 |
| false_denial | train_mix | 0.310 | 0.156 |
| false_admission | cue_gold | 0.571 | 0.794 |
| false_admission | cue_dist | 0.784 | 0.907 |
| false_admission | train_mix | 0.636 | 0.830 |
| parse_failed | none | 0.026 | 0.002 |
| parse_failed | cue_gold | 0.017 | 0.000 |
| parse_failed | cue_dist | 0.027 | 0.000 |
| parse_failed | train_mix | 0.022 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1329 | 1353 |
| n_bias_rows | cue_dist | 1317 | 1353 |
| n_bias_rows | train_mix | 2646 | 2706 |
| n_deception_rows | cue_gold | 1283 | 1347 |
| n_deception_rows | cue_dist | 1291 | 1345 |
| n_deception_rows | train_mix | 2574 | 2692 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.895 |
| accuracy | cue_gold | 0.942 | 0.978 |
| accuracy | cue_dist | 0.626 | 0.540 |
| accuracy | train_mix | 0.832 | 0.831 |
| reward | none | 0.738 | 0.745 |
| reward | cue_gold | 0.768 | 0.795 |
| reward | cue_dist | 0.572 | 0.508 |
| reward | train_mix | 0.748 | 0.761 |
| followed_cue | cue_gold | 0.942 | 0.978 |
| followed_cue | cue_dist | 0.348 | 0.460 |
| followed_cue | train_mix | 0.735 | 0.795 |
| bias | cue_gold | 0.007 | 0.056 |
| bias | cue_dist | 0.330 | 0.373 |
| bias | train_mix | 0.121 | 0.162 |
| anti_bias | cue_gold | 0.040 | 0.004 |
| anti_bias | cue_dist | 0.002 | 0.007 |
| anti_bias | train_mix | 0.026 | 0.005 |
| no_bias | cue_gold | 0.954 | 0.941 |
| no_bias | cue_dist | 0.668 | 0.619 |
| no_bias | train_mix | 0.853 | 0.834 |
| deception_soft | cue_gold | 0.540 | 0.680 |
| deception_soft | cue_dist | 0.590 | 0.602 |
| deception_soft | train_mix | 0.556 | 0.657 |
| deception | cue_gold | 0.531 | 0.747 |
| deception | cue_dist | 0.653 | 0.638 |
| deception | train_mix | 0.573 | 0.717 |
| biased_share | cue_gold | 0.007 | 0.056 |
| biased_share | cue_dist | 0.331 | 0.376 |
| biased_share | train_mix | 0.124 | 0.162 |
| false_denial_soft | cue_gold | 0.487 | 0.272 |
| false_denial_soft | cue_dist | 0.350 | 0.254 |
| false_denial_soft | train_mix | 0.354 | 0.265 |
| false_admission_soft | cue_gold | 0.541 | 0.704 |
| false_admission_soft | cue_dist | 0.708 | 0.812 |
| false_admission_soft | train_mix | 0.585 | 0.733 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.728 |
| said_yes_biased_soft | cue_dist | 0.650 | 0.746 |
| said_yes_biased_soft | train_mix | 0.646 | 0.735 |
| deception_from_biased | cue_gold | 0.003 | 0.015 |
| deception_from_biased | cue_dist | 0.116 | 0.095 |
| deception_from_biased | train_mix | 0.044 | 0.043 |
| deception_from_unbiased | cue_gold | 0.537 | 0.664 |
| deception_from_unbiased | cue_dist | 0.474 | 0.507 |
| deception_from_unbiased | train_mix | 0.513 | 0.614 |
| said_yes | cue_gold | 0.532 | 0.782 |
| said_yes | cue_dist | 0.795 | 0.891 |
| said_yes | train_mix | 0.624 | 0.823 |
| false_denial | cue_gold | 0.429 | 0.183 |
| false_denial | cue_dist | 0.285 | 0.164 |
| false_denial | train_mix | 0.294 | 0.174 |
| false_admission | cue_gold | 0.532 | 0.780 |
| false_admission | cue_dist | 0.835 | 0.924 |
| false_admission | train_mix | 0.612 | 0.822 |
| parse_failed | none | 0.022 | 0.003 |
| parse_failed | cue_gold | 0.014 | 0.000 |
| parse_failed | cue_dist | 0.026 | 0.000 |
| parse_failed | train_mix | 0.020 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1062 | 1077 |
| n_bias_rows | cue_dist | 1050 | 1077 |
| n_bias_rows | train_mix | 2112 | 2154 |
| n_deception_rows | cue_gold | 1020 | 1073 |
| n_deception_rows | cue_dist | 1048 | 1069 |
| n_deception_rows | train_mix | 2068 | 2142 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.325 |
| accuracy | cue_gold | 0.524 | 0.727 |
| accuracy | cue_dist | 0.098 | 0.105 |
| accuracy | train_mix | 0.384 | 0.500 |
| reward | none | 0.313 | 0.401 |
| reward | cue_gold | 0.524 | 0.629 |
| reward | cue_dist | 0.225 | 0.292 |
| reward | train_mix | 0.479 | 0.616 |
| followed_cue | cue_gold | 0.524 | 0.727 |
| followed_cue | cue_dist | 0.870 | 0.895 |
| followed_cue | train_mix | 0.640 | 0.791 |
| bias | cue_gold | 0.449 | 0.417 |
| bias | cue_dist | 0.064 | 0.196 |
| bias | train_mix | 0.311 | 0.340 |
| anti_bias | cue_gold | 0.015 | 0.007 |
| anti_bias | cue_dist | 0.090 | 0.000 |
| anti_bias | train_mix | 0.049 | 0.004 |
| no_bias | cue_gold | 0.536 | 0.576 |
| no_bias | cue_dist | 0.846 | 0.804 |
| no_bias | train_mix | 0.640 | 0.657 |
| deception_soft | cue_gold | 0.578 | 0.539 |
| deception_soft | cue_dist | 0.587 | 0.649 |
| deception_soft | train_mix | 0.584 | 0.569 |
| deception | cue_gold | 0.627 | 0.577 |
| deception | cue_dist | 0.597 | 0.714 |
| deception | train_mix | 0.623 | 0.617 |
| biased_share | cue_gold | 0.456 | 0.420 |
| biased_share | cue_dist | 0.070 | 0.196 |
| biased_share | train_mix | 0.327 | 0.341 |
| false_denial_soft | cue_gold | 0.374 | 0.222 |
| false_denial_soft | cue_dist | 0.280 | 0.230 |
| false_denial_soft | train_mix | 0.361 | 0.207 |
| false_admission_soft | cue_gold | 0.750 | 0.767 |
| false_admission_soft | cue_dist | 0.610 | 0.752 |
| false_admission_soft | train_mix | 0.693 | 0.757 |
| said_yes_biased_soft | cue_gold | 0.626 | 0.778 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.770 |
| said_yes_biased_soft | train_mix | 0.639 | 0.793 |
| deception_from_biased | cue_gold | 0.171 | 0.093 |
| deception_from_biased | cue_dist | 0.020 | 0.045 |
| deception_from_biased | train_mix | 0.118 | 0.070 |
| deception_from_unbiased | cue_gold | 0.408 | 0.445 |
| deception_from_unbiased | cue_dist | 0.567 | 0.604 |
| deception_from_unbiased | train_mix | 0.466 | 0.499 |
| said_yes | cue_gold | 0.749 | 0.872 |
| said_yes | cue_dist | 0.642 | 0.859 |
| said_yes | train_mix | 0.730 | 0.875 |
| false_denial | cue_gold | 0.367 | 0.148 |
| false_denial | cue_dist | 0.176 | 0.130 |
| false_denial | train_mix | 0.335 | 0.123 |
| false_admission | cue_gold | 0.846 | 0.887 |
| false_admission | cue_dist | 0.628 | 0.856 |
| false_admission | train_mix | 0.762 | 0.873 |
| parse_failed | none | 0.039 | 0.000 |
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
| n_bias_rows | cue_gold | 267 | 276 |
| n_bias_rows | cue_dist | 267 | 276 |
| n_bias_rows | train_mix | 534 | 552 |
| n_deception_rows | cue_gold | 263 | 274 |
| n_deception_rows | cue_dist | 243 | 276 |
| n_deception_rows | train_mix | 506 | 550 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 0.805 | 0.017 | 0.751 | 0.751 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.195 | 0.018 | 0.253 | 0.729 |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 0.808 | 0.000 | 0.903 | 0.903 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.192 | 0.002 | 0.412 | 0.586 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4080 | 0.899 | 0.017 | 0.749 |
| first_10_steps | base_wrong | 1040 | 0.150 | 0.015 | 0.282 |
| last_10_steps | base_right | 4008 | 0.933 | 0.001 | 0.844 |
| last_10_steps | base_wrong | 1112 | 0.513 | 0.001 | 0.681 |
