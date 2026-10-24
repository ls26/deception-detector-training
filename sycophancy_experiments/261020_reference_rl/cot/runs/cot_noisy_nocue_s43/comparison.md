# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 467; in both 445 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.769 |
| accuracy | cue_gold | 0.852 | 0.879 |
| accuracy | cue_dist | 0.513 | 0.569 |
| accuracy | train_mix | 0.736 | 0.772 |
| reward | none | 0.647 | 0.663 |
| reward | cue_gold | 0.716 | 0.728 |
| reward | cue_dist | 0.498 | 0.535 |
| reward | train_mix | 0.690 | 0.709 |
| followed_cue | cue_gold | 0.852 | 0.879 |
| followed_cue | cue_dist | 0.460 | 0.429 |
| followed_cue | train_mix | 0.715 | 0.721 |
| bias | cue_gold | 0.099 | 0.098 |
| bias | cue_dist | 0.279 | 0.239 |
| bias | train_mix | 0.163 | 0.145 |
| anti_bias | cue_gold | 0.033 | 0.023 |
| anti_bias | cue_dist | 0.018 | 0.022 |
| anti_bias | train_mix | 0.028 | 0.022 |
| no_bias | cue_gold | 0.868 | 0.879 |
| no_bias | cue_dist | 0.703 | 0.739 |
| no_bias | train_mix | 0.809 | 0.833 |
| deception_soft | cue_gold | 0.546 | 0.646 |
| deception_soft | cue_dist | 0.590 | 0.628 |
| deception_soft | train_mix | 0.559 | 0.638 |
| deception | cue_gold | 0.552 | 0.728 |
| deception | cue_dist | 0.637 | 0.702 |
| deception | train_mix | 0.578 | 0.714 |
| biased_share | cue_gold | 0.103 | 0.101 |
| biased_share | cue_dist | 0.284 | 0.245 |
| biased_share | train_mix | 0.168 | 0.148 |
| false_denial_soft | cue_gold | 0.367 | 0.233 |
| false_denial_soft | cue_dist | 0.343 | 0.229 |
| false_denial_soft | train_mix | 0.347 | 0.225 |
| false_admission_soft | cue_gold | 0.567 | 0.692 |
| false_admission_soft | cue_dist | 0.688 | 0.757 |
| false_admission_soft | train_mix | 0.602 | 0.710 |
| said_yes_biased_soft | cue_gold | 0.633 | 0.767 |
| said_yes_biased_soft | cue_dist | 0.657 | 0.771 |
| said_yes_biased_soft | train_mix | 0.653 | 0.775 |
| deception_from_biased | cue_gold | 0.038 | 0.023 |
| deception_from_biased | cue_dist | 0.098 | 0.056 |
| deception_from_biased | train_mix | 0.058 | 0.033 |
| deception_from_unbiased | cue_gold | 0.509 | 0.623 |
| deception_from_unbiased | cue_dist | 0.492 | 0.572 |
| deception_from_unbiased | train_mix | 0.501 | 0.605 |
| said_yes | cue_gold | 0.582 | 0.811 |
| said_yes | cue_dist | 0.769 | 0.901 |
| said_yes | train_mix | 0.647 | 0.839 |
| false_denial | cue_gold | 0.354 | 0.084 |
| false_denial | cue_dist | 0.267 | 0.094 |
| false_denial | train_mix | 0.293 | 0.079 |
| false_admission | cue_gold | 0.575 | 0.799 |
| false_admission | cue_dist | 0.783 | 0.899 |
| false_admission | train_mix | 0.635 | 0.825 |
| parse_failed | none | 0.026 | 0.005 |
| parse_failed | cue_gold | 0.017 | 0.001 |
| parse_failed | cue_dist | 0.027 | 0.001 |
| parse_failed | train_mix | 0.022 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1311 | 1333 |
| n_bias_rows | cue_dist | 1301 | 1333 |
| n_bias_rows | train_mix | 2612 | 2666 |
| n_deception_rows | cue_gold | 1268 | 1303 |
| n_deception_rows | cue_dist | 1277 | 1304 |
| n_deception_rows | train_mix | 2545 | 2607 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.884 |
| accuracy | cue_gold | 0.942 | 0.961 |
| accuracy | cue_dist | 0.626 | 0.691 |
| accuracy | train_mix | 0.832 | 0.869 |
| reward | none | 0.738 | 0.740 |
| reward | cue_gold | 0.768 | 0.785 |
| reward | cue_dist | 0.572 | 0.614 |
| reward | train_mix | 0.748 | 0.767 |
| followed_cue | cue_gold | 0.942 | 0.961 |
| followed_cue | cue_dist | 0.348 | 0.308 |
| followed_cue | train_mix | 0.735 | 0.734 |
| bias | cue_gold | 0.004 | 0.047 |
| bias | cue_dist | 0.333 | 0.242 |
| bias | train_mix | 0.118 | 0.112 |
| anti_bias | cue_gold | 0.037 | 0.017 |
| anti_bias | cue_dist | 0.002 | 0.017 |
| anti_bias | train_mix | 0.024 | 0.017 |
| no_bias | cue_gold | 0.959 | 0.936 |
| no_bias | cue_dist | 0.665 | 0.741 |
| no_bias | train_mix | 0.858 | 0.872 |
| deception_soft | cue_gold | 0.542 | 0.656 |
| deception_soft | cue_dist | 0.589 | 0.628 |
| deception_soft | train_mix | 0.556 | 0.647 |
| deception | cue_gold | 0.539 | 0.745 |
| deception | cue_dist | 0.646 | 0.705 |
| deception | train_mix | 0.572 | 0.730 |
| biased_share | cue_gold | 0.004 | 0.048 |
| biased_share | cue_dist | 0.334 | 0.246 |
| biased_share | train_mix | 0.120 | 0.113 |
| false_denial_soft | cue_gold | 0.498 | 0.230 |
| false_denial_soft | cue_dist | 0.346 | 0.240 |
| false_denial_soft | train_mix | 0.342 | 0.242 |
| false_admission_soft | cue_gold | 0.542 | 0.678 |
| false_admission_soft | cue_dist | 0.711 | 0.754 |
| false_admission_soft | train_mix | 0.586 | 0.699 |
| said_yes_biased_soft | cue_gold | 0.502 | 0.770 |
| said_yes_biased_soft | cue_dist | 0.654 | 0.760 |
| said_yes_biased_soft | train_mix | 0.658 | 0.758 |
| deception_from_biased | cue_gold | 0.002 | 0.011 |
| deception_from_biased | cue_dist | 0.116 | 0.059 |
| deception_from_biased | train_mix | 0.041 | 0.027 |
| deception_from_unbiased | cue_gold | 0.540 | 0.645 |
| deception_from_unbiased | cue_dist | 0.474 | 0.569 |
| deception_from_unbiased | train_mix | 0.515 | 0.620 |
| said_yes | cue_gold | 0.539 | 0.789 |
| said_yes | cue_dist | 0.798 | 0.899 |
| said_yes | train_mix | 0.627 | 0.823 |
| false_denial | cue_gold | 0.500 | 0.040 |
| false_denial | cue_dist | 0.272 | 0.105 |
| false_denial | train_mix | 0.269 | 0.091 |
| false_admission | cue_gold | 0.539 | 0.780 |
| false_admission | cue_dist | 0.833 | 0.901 |
| false_admission | train_mix | 0.613 | 0.812 |
| parse_failed | none | 0.022 | 0.006 |
| parse_failed | cue_gold | 0.014 | 0.002 |
| parse_failed | cue_dist | 0.026 | 0.001 |
| parse_failed | train_mix | 0.020 | 0.002 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1047 | 1060 |
| n_bias_rows | cue_dist | 1038 | 1061 |
| n_bias_rows | train_mix | 2085 | 2121 |
| n_deception_rows | cue_gold | 1008 | 1042 |
| n_deception_rows | cue_dist | 1036 | 1043 |
| n_deception_rows | train_mix | 2044 | 2085 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.346 |
| accuracy | cue_gold | 0.524 | 0.581 |
| accuracy | cue_dist | 0.098 | 0.124 |
| accuracy | train_mix | 0.384 | 0.421 |
| reward | none | 0.313 | 0.382 |
| reward | cue_gold | 0.524 | 0.521 |
| reward | cue_dist | 0.225 | 0.248 |
| reward | train_mix | 0.479 | 0.498 |
| followed_cue | cue_gold | 0.524 | 0.581 |
| followed_cue | cue_dist | 0.870 | 0.873 |
| followed_cue | train_mix | 0.640 | 0.670 |
| bias | cue_gold | 0.477 | 0.297 |
| bias | cue_dist | 0.065 | 0.228 |
| bias | train_mix | 0.343 | 0.276 |
| anti_bias | cue_gold | 0.015 | 0.044 |
| anti_bias | cue_dist | 0.084 | 0.040 |
| anti_bias | train_mix | 0.042 | 0.045 |
| no_bias | cue_gold | 0.508 | 0.659 |
| no_bias | cue_dist | 0.852 | 0.732 |
| no_bias | train_mix | 0.615 | 0.679 |
| deception_soft | cue_gold | 0.562 | 0.607 |
| deception_soft | cue_dist | 0.591 | 0.628 |
| deception_soft | train_mix | 0.573 | 0.602 |
| deception | cue_gold | 0.604 | 0.659 |
| deception | cue_dist | 0.598 | 0.693 |
| deception | train_mix | 0.601 | 0.652 |
| biased_share | cue_gold | 0.485 | 0.310 |
| biased_share | cue_dist | 0.071 | 0.238 |
| biased_share | train_mix | 0.358 | 0.289 |
| false_denial_soft | cue_gold | 0.363 | 0.236 |
| false_denial_soft | cue_dist | 0.280 | 0.182 |
| false_denial_soft | train_mix | 0.355 | 0.199 |
| false_admission_soft | cue_gold | 0.748 | 0.774 |
| false_admission_soft | cue_dist | 0.615 | 0.767 |
| false_admission_soft | train_mix | 0.694 | 0.766 |
| said_yes_biased_soft | cue_gold | 0.637 | 0.764 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.818 |
| said_yes_biased_soft | train_mix | 0.645 | 0.801 |
| deception_from_biased | cue_gold | 0.176 | 0.073 |
| deception_from_biased | cue_dist | 0.020 | 0.043 |
| deception_from_biased | train_mix | 0.127 | 0.058 |
| deception_from_unbiased | cue_gold | 0.386 | 0.534 |
| deception_from_unbiased | cue_dist | 0.571 | 0.585 |
| deception_from_unbiased | train_mix | 0.445 | 0.545 |
| said_yes | cue_gold | 0.750 | 0.900 |
| said_yes | cue_dist | 0.643 | 0.908 |
| said_yes | train_mix | 0.727 | 0.906 |
| false_denial | cue_gold | 0.349 | 0.111 |
| false_denial | cue_dist | 0.176 | 0.048 |
| false_denial | train_mix | 0.324 | 0.060 |
| false_admission | cue_gold | 0.843 | 0.906 |
| false_admission | cue_dist | 0.629 | 0.894 |
| false_admission | train_mix | 0.755 | 0.892 |
| parse_failed | none | 0.039 | 0.000 |
| parse_failed | cue_gold | 0.029 | 0.000 |
| parse_failed | cue_dist | 0.032 | 0.003 |
| parse_failed | train_mix | 0.033 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 840 | 840 |
| rollouts | cue_gold | 315 | 315 |
| rollouts | cue_dist | 315 | 315 |
| rollouts | train_mix | 630 | 630 |
| n_bias_rows | cue_gold | 264 | 273 |
| n_bias_rows | cue_dist | 263 | 272 |
| n_bias_rows | train_mix | 527 | 545 |
| n_deception_rows | cue_gold | 260 | 261 |
| n_deception_rows | cue_dist | 241 | 261 |
| n_deception_rows | train_mix | 501 | 522 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 0.789 | 0.018 | 0.742 | 0.742 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.211 | 0.014 | 0.258 | 0.728 |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 0.787 | 0.005 | 0.856 | 0.856 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.212 | 0.007 | 0.408 | 0.585 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4040 | 0.891 | 0.016 | 0.727 |
| first_10_steps | base_wrong | 1080 | 0.169 | 0.021 | 0.313 |
| last_10_steps | base_right | 4024 | 0.901 | 0.006 | 0.800 |
| last_10_steps | base_wrong | 1096 | 0.421 | 0.006 | 0.616 |
