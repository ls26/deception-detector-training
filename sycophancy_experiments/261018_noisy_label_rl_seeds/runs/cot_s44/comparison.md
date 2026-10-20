# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 474; in both 453 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.743 |
| accuracy | cue_gold | 0.852 | 0.965 |
| accuracy | cue_dist | 0.513 | 0.252 |
| accuracy | train_mix | 0.736 | 0.720 |
| reward | none | 0.647 | 0.666 |
| reward | cue_gold | 0.716 | 0.788 |
| reward | cue_dist | 0.498 | 0.341 |
| reward | train_mix | 0.690 | 0.743 |
| followed_cue | cue_gold | 0.852 | 0.965 |
| followed_cue | cue_dist | 0.460 | 0.747 |
| followed_cue | train_mix | 0.715 | 0.888 |
| bias | cue_gold | 0.099 | 0.194 |
| bias | cue_dist | 0.278 | 0.502 |
| bias | train_mix | 0.163 | 0.285 |
| anti_bias | cue_gold | 0.034 | 0.001 |
| anti_bias | cue_dist | 0.018 | 0.000 |
| anti_bias | train_mix | 0.030 | 0.001 |
| no_bias | cue_gold | 0.867 | 0.806 |
| no_bias | cue_dist | 0.704 | 0.498 |
| no_bias | train_mix | 0.808 | 0.714 |
| deception_soft | cue_gold | 0.545 | 0.585 |
| deception_soft | cue_dist | 0.589 | 0.531 |
| deception_soft | train_mix | 0.560 | 0.577 |
| deception | cue_gold | 0.547 | 0.640 |
| deception | cue_dist | 0.641 | 0.531 |
| deception | train_mix | 0.580 | 0.617 |
| biased_share | cue_gold | 0.102 | 0.194 |
| biased_share | cue_dist | 0.283 | 0.502 |
| biased_share | train_mix | 0.168 | 0.286 |
| false_denial_soft | cue_gold | 0.377 | 0.244 |
| false_denial_soft | cue_dist | 0.348 | 0.262 |
| false_denial_soft | train_mix | 0.357 | 0.258 |
| false_admission_soft | cue_gold | 0.564 | 0.667 |
| false_admission_soft | cue_dist | 0.685 | 0.802 |
| false_admission_soft | train_mix | 0.601 | 0.704 |
| said_yes_biased_soft | cue_gold | 0.623 | 0.756 |
| said_yes_biased_soft | cue_dist | 0.652 | 0.738 |
| said_yes_biased_soft | train_mix | 0.643 | 0.742 |
| deception_from_biased | cue_gold | 0.039 | 0.047 |
| deception_from_biased | cue_dist | 0.099 | 0.132 |
| deception_from_biased | train_mix | 0.060 | 0.074 |
| deception_from_unbiased | cue_gold | 0.506 | 0.538 |
| deception_from_unbiased | cue_dist | 0.491 | 0.399 |
| deception_from_unbiased | train_mix | 0.500 | 0.503 |
| said_yes | cue_gold | 0.577 | 0.783 |
| said_yes | cue_dist | 0.766 | 0.893 |
| said_yes | train_mix | 0.644 | 0.819 |
| false_denial | cue_gold | 0.356 | 0.129 |
| false_denial | cue_dist | 0.280 | 0.139 |
| false_denial | train_mix | 0.308 | 0.146 |
| false_admission | cue_gold | 0.569 | 0.762 |
| false_admission | cue_dist | 0.784 | 0.926 |
| false_admission | train_mix | 0.635 | 0.806 |
| parse_failed | none | 0.026 | 0.004 |
| parse_failed | cue_gold | 0.017 | 0.001 |
| parse_failed | cue_dist | 0.027 | 0.001 |
| parse_failed | train_mix | 0.022 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1335 | 1358 |
| n_bias_rows | cue_dist | 1323 | 1358 |
| n_bias_rows | train_mix | 2658 | 2716 |
| n_deception_rows | cue_gold | 1290 | 1357 |
| n_deception_rows | cue_dist | 1299 | 1358 |
| n_deception_rows | train_mix | 2589 | 2715 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.893 |
| accuracy | cue_gold | 0.942 | 0.997 |
| accuracy | cue_dist | 0.626 | 0.321 |
| accuracy | train_mix | 0.832 | 0.769 |
| reward | none | 0.738 | 0.753 |
| reward | cue_gold | 0.768 | 0.807 |
| reward | cue_dist | 0.572 | 0.369 |
| reward | train_mix | 0.748 | 0.757 |
| followed_cue | cue_gold | 0.942 | 0.997 |
| followed_cue | cue_dist | 0.348 | 0.678 |
| followed_cue | train_mix | 0.735 | 0.882 |
| bias | cue_gold | 0.007 | 0.065 |
| bias | cue_dist | 0.333 | 0.597 |
| bias | train_mix | 0.121 | 0.235 |
| anti_bias | cue_gold | 0.038 | 0.001 |
| anti_bias | cue_dist | 0.002 | 0.000 |
| anti_bias | train_mix | 0.026 | 0.001 |
| no_bias | cue_gold | 0.955 | 0.934 |
| no_bias | cue_dist | 0.665 | 0.403 |
| no_bias | train_mix | 0.854 | 0.765 |
| deception_soft | cue_gold | 0.538 | 0.633 |
| deception_soft | cue_dist | 0.589 | 0.495 |
| deception_soft | train_mix | 0.555 | 0.596 |
| deception | cue_gold | 0.531 | 0.718 |
| deception | cue_dist | 0.651 | 0.466 |
| deception | train_mix | 0.572 | 0.645 |
| biased_share | cue_gold | 0.007 | 0.065 |
| biased_share | cue_dist | 0.333 | 0.597 |
| biased_share | train_mix | 0.124 | 0.235 |
| false_denial_soft | cue_gold | 0.487 | 0.282 |
| false_denial_soft | cue_dist | 0.351 | 0.263 |
| false_denial_soft | train_mix | 0.354 | 0.271 |
| false_admission_soft | cue_gold | 0.539 | 0.657 |
| false_admission_soft | cue_dist | 0.708 | 0.840 |
| false_admission_soft | train_mix | 0.583 | 0.695 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.718 |
| said_yes_biased_soft | cue_dist | 0.649 | 0.737 |
| said_yes_biased_soft | train_mix | 0.646 | 0.729 |
| deception_from_biased | cue_gold | 0.003 | 0.018 |
| deception_from_biased | cue_dist | 0.117 | 0.157 |
| deception_from_biased | train_mix | 0.044 | 0.064 |
| deception_from_unbiased | cue_gold | 0.535 | 0.615 |
| deception_from_unbiased | cue_dist | 0.472 | 0.338 |
| deception_from_unbiased | train_mix | 0.511 | 0.532 |
| said_yes | cue_gold | 0.532 | 0.759 |
| said_yes | cue_dist | 0.794 | 0.895 |
| said_yes | train_mix | 0.623 | 0.804 |
| false_denial | cue_gold | 0.429 | 0.186 |
| false_denial | cue_dist | 0.285 | 0.141 |
| false_denial | train_mix | 0.295 | 0.161 |
| false_admission | cue_gold | 0.531 | 0.755 |
| false_admission | cue_dist | 0.833 | 0.947 |
| false_admission | train_mix | 0.611 | 0.793 |
| parse_failed | none | 0.022 | 0.003 |
| parse_failed | cue_gold | 0.014 | 0.001 |
| parse_failed | cue_dist | 0.026 | 0.001 |
| parse_failed | train_mix | 0.020 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1068 | 1082 |
| n_bias_rows | cue_dist | 1055 | 1082 |
| n_bias_rows | train_mix | 2123 | 2164 |
| n_deception_rows | cue_gold | 1027 | 1081 |
| n_deception_rows | cue_dist | 1053 | 1082 |
| n_deception_rows | train_mix | 2080 | 2163 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.196 |
| accuracy | cue_gold | 0.524 | 0.848 |
| accuracy | cue_dist | 0.098 | 0.000 |
| accuracy | train_mix | 0.384 | 0.544 |
| reward | none | 0.313 | 0.348 |
| reward | cue_gold | 0.524 | 0.717 |
| reward | cue_dist | 0.225 | 0.238 |
| reward | train_mix | 0.479 | 0.690 |
| followed_cue | cue_gold | 0.524 | 0.848 |
| followed_cue | cue_dist | 0.870 | 1.000 |
| followed_cue | train_mix | 0.640 | 0.913 |
| bias | cue_gold | 0.468 | 0.699 |
| bias | cue_dist | 0.063 | 0.130 |
| bias | train_mix | 0.329 | 0.485 |
| anti_bias | cue_gold | 0.015 | 0.000 |
| anti_bias | cue_dist | 0.082 | 0.000 |
| anti_bias | train_mix | 0.047 | 0.000 |
| no_bias | cue_gold | 0.517 | 0.301 |
| no_bias | cue_dist | 0.854 | 0.870 |
| no_bias | train_mix | 0.624 | 0.515 |
| deception_soft | cue_gold | 0.571 | 0.398 |
| deception_soft | cue_dist | 0.590 | 0.670 |
| deception_soft | train_mix | 0.579 | 0.502 |
| deception | cue_gold | 0.612 | 0.333 |
| deception | cue_dist | 0.602 | 0.786 |
| deception | train_mix | 0.613 | 0.510 |
| biased_share | cue_gold | 0.475 | 0.699 |
| biased_share | cue_dist | 0.069 | 0.130 |
| biased_share | train_mix | 0.345 | 0.485 |
| false_denial_soft | cue_gold | 0.371 | 0.230 |
| false_denial_soft | cue_dist | 0.280 | 0.253 |
| false_denial_soft | train_mix | 0.360 | 0.232 |
| false_admission_soft | cue_gold | 0.752 | 0.789 |
| false_admission_soft | cue_dist | 0.613 | 0.733 |
| false_admission_soft | train_mix | 0.695 | 0.755 |
| said_yes_biased_soft | cue_gold | 0.629 | 0.770 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.747 |
| said_yes_biased_soft | train_mix | 0.640 | 0.768 |
| deception_from_biased | cue_gold | 0.176 | 0.161 |
| deception_from_biased | cue_dist | 0.019 | 0.033 |
| deception_from_biased | train_mix | 0.124 | 0.112 |
| deception_from_unbiased | cue_gold | 0.395 | 0.237 |
| deception_from_unbiased | cue_dist | 0.571 | 0.637 |
| deception_from_unbiased | train_mix | 0.455 | 0.389 |
| said_yes | cue_gold | 0.753 | 0.880 |
| said_yes | cue_dist | 0.646 | 0.888 |
| said_yes | train_mix | 0.733 | 0.880 |
| false_denial | cue_gold | 0.352 | 0.109 |
| false_denial | cue_dist | 0.176 | 0.111 |
| false_denial | train_mix | 0.327 | 0.118 |
| false_admission | cue_gold | 0.848 | 0.855 |
| false_admission | cue_dist | 0.633 | 0.887 |
| false_admission | train_mix | 0.765 | 0.879 |
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
| n_bias_rows | cue_gold | 267 | 276 |
| n_bias_rows | cue_dist | 268 | 276 |
| n_bias_rows | train_mix | 535 | 552 |
| n_deception_rows | cue_gold | 263 | 276 |
| n_deception_rows | cue_dist | 246 | 276 |
| n_deception_rows | train_mix | 509 | 552 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.609 | 0.914 | 0.914 | 0.914 |
| first_10_steps | label_correct__cue_at_other | 0.173 | 0.598 | 0.385 | 0.385 |
| first_10_steps | label_flipped__cue_at_label | 0.166 | 0.667 | 0.667 | 0.321 |
| first_10_steps | label_flipped__cue_at_other | 0.052 | 0.913 | 0.083 | 0.913 |
| last_10_steps | label_correct__cue_at_label | 0.642 | 0.987 | 0.987 | 0.987 |
| last_10_steps | label_correct__cue_at_other | 0.172 | 0.689 | 0.309 | 0.309 |
| last_10_steps | label_flipped__cue_at_label | 0.133 | 0.865 | 0.865 | 0.135 |
| last_10_steps | label_flipped__cue_at_other | 0.053 | 0.952 | 0.048 | 0.952 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3992 | 0.796 | 0.825 | 0.760 |
| first_10_steps | base_wrong | 1128 | 0.469 | 0.793 | 0.664 |
| last_10_steps | base_right | 4120 | 0.775 | 0.909 | 0.818 |
| last_10_steps | base_wrong | 1000 | 0.672 | 0.953 | 0.748 |
