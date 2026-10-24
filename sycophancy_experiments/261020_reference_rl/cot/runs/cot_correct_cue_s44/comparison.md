# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 478; in both 458 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.776 |
| accuracy | cue_gold | 0.852 | 0.877 |
| accuracy | cue_dist | 0.513 | 0.641 |
| accuracy | train_mix | 0.736 | 0.794 |
| reward | none | 0.647 | 0.682 |
| reward | cue_gold | 0.716 | 0.733 |
| reward | cue_dist | 0.498 | 0.594 |
| reward | train_mix | 0.690 | 0.722 |
| followed_cue | cue_gold | 0.852 | 0.877 |
| followed_cue | cue_dist | 0.460 | 0.358 |
| followed_cue | train_mix | 0.715 | 0.699 |
| bias | cue_gold | 0.099 | 0.088 |
| bias | cue_dist | 0.274 | 0.152 |
| bias | train_mix | 0.163 | 0.106 |
| anti_bias | cue_gold | 0.033 | 0.009 |
| anti_bias | cue_dist | 0.018 | 0.010 |
| anti_bias | train_mix | 0.029 | 0.011 |
| no_bias | cue_gold | 0.867 | 0.902 |
| no_bias | cue_dist | 0.708 | 0.838 |
| no_bias | train_mix | 0.808 | 0.883 |
| deception_soft | cue_gold | 0.546 | 0.589 |
| deception_soft | cue_dist | 0.593 | 0.660 |
| deception_soft | train_mix | 0.561 | 0.613 |
| deception | cue_gold | 0.550 | 0.620 |
| deception | cue_dist | 0.647 | 0.711 |
| deception | train_mix | 0.583 | 0.652 |
| biased_share | cue_gold | 0.103 | 0.089 |
| biased_share | cue_dist | 0.279 | 0.153 |
| biased_share | train_mix | 0.168 | 0.107 |
| false_denial_soft | cue_gold | 0.377 | 0.292 |
| false_denial_soft | cue_dist | 0.352 | 0.316 |
| false_denial_soft | train_mix | 0.358 | 0.299 |
| false_admission_soft | cue_gold | 0.566 | 0.618 |
| false_admission_soft | cue_dist | 0.687 | 0.722 |
| false_admission_soft | train_mix | 0.602 | 0.650 |
| said_yes_biased_soft | cue_gold | 0.623 | 0.708 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.684 |
| said_yes_biased_soft | train_mix | 0.642 | 0.701 |
| deception_from_biased | cue_gold | 0.039 | 0.026 |
| deception_from_biased | cue_dist | 0.098 | 0.048 |
| deception_from_biased | train_mix | 0.060 | 0.032 |
| deception_from_unbiased | cue_gold | 0.507 | 0.563 |
| deception_from_unbiased | cue_dist | 0.495 | 0.611 |
| deception_from_unbiased | train_mix | 0.501 | 0.581 |
| said_yes | cue_gold | 0.578 | 0.657 |
| said_yes | cue_dist | 0.767 | 0.784 |
| said_yes | train_mix | 0.645 | 0.699 |
| false_denial | cue_gold | 0.366 | 0.289 |
| false_denial | cue_dist | 0.286 | 0.264 |
| false_denial | train_mix | 0.313 | 0.282 |
| false_admission | cue_gold | 0.571 | 0.652 |
| false_admission | cue_dist | 0.787 | 0.792 |
| false_admission | train_mix | 0.637 | 0.697 |
| parse_failed | none | 0.026 | 0.002 |
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
| n_bias_rows | cue_gold | 1350 | 1373 |
| n_bias_rows | cue_dist | 1339 | 1372 |
| n_bias_rows | train_mix | 2689 | 2745 |
| n_deception_rows | cue_gold | 1305 | 1360 |
| n_deception_rows | cue_dist | 1315 | 1358 |
| n_deception_rows | train_mix | 2620 | 2718 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.926 |
| accuracy | cue_gold | 0.942 | 0.976 |
| accuracy | cue_dist | 0.626 | 0.780 |
| accuracy | train_mix | 0.832 | 0.910 |
| reward | none | 0.738 | 0.771 |
| reward | cue_gold | 0.768 | 0.796 |
| reward | cue_dist | 0.572 | 0.669 |
| reward | train_mix | 0.748 | 0.781 |
| followed_cue | cue_gold | 0.942 | 0.976 |
| followed_cue | cue_dist | 0.348 | 0.218 |
| followed_cue | train_mix | 0.735 | 0.713 |
| bias | cue_gold | 0.006 | 0.035 |
| bias | cue_dist | 0.332 | 0.165 |
| bias | train_mix | 0.121 | 0.075 |
| anti_bias | cue_gold | 0.039 | 0.012 |
| anti_bias | cue_dist | 0.002 | 0.006 |
| anti_bias | train_mix | 0.026 | 0.011 |
| no_bias | cue_gold | 0.955 | 0.954 |
| no_bias | cue_dist | 0.666 | 0.828 |
| no_bias | train_mix | 0.853 | 0.914 |
| deception_soft | cue_gold | 0.541 | 0.596 |
| deception_soft | cue_dist | 0.590 | 0.670 |
| deception_soft | train_mix | 0.556 | 0.622 |
| deception | cue_gold | 0.534 | 0.628 |
| deception | cue_dist | 0.652 | 0.733 |
| deception | train_mix | 0.573 | 0.666 |
| biased_share | cue_gold | 0.007 | 0.035 |
| biased_share | cue_dist | 0.333 | 0.166 |
| biased_share | train_mix | 0.125 | 0.075 |
| false_denial_soft | cue_gold | 0.487 | 0.385 |
| false_denial_soft | cue_dist | 0.352 | 0.325 |
| false_denial_soft | train_mix | 0.354 | 0.346 |
| false_admission_soft | cue_gold | 0.541 | 0.604 |
| false_admission_soft | cue_dist | 0.709 | 0.739 |
| false_admission_soft | train_mix | 0.585 | 0.645 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.615 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.675 |
| said_yes_biased_soft | train_mix | 0.646 | 0.654 |
| deception_from_biased | cue_gold | 0.003 | 0.013 |
| deception_from_biased | cue_dist | 0.117 | 0.054 |
| deception_from_biased | train_mix | 0.044 | 0.026 |
| deception_from_unbiased | cue_gold | 0.537 | 0.583 |
| deception_from_unbiased | cue_dist | 0.473 | 0.616 |
| deception_from_unbiased | train_mix | 0.512 | 0.596 |
| said_yes | cue_gold | 0.535 | 0.636 |
| said_yes | cue_dist | 0.794 | 0.805 |
| said_yes | train_mix | 0.625 | 0.691 |
| false_denial | cue_gold | 0.429 | 0.395 |
| false_denial | cue_dist | 0.287 | 0.282 |
| false_denial | train_mix | 0.293 | 0.336 |
| false_admission | cue_gold | 0.535 | 0.637 |
| false_admission | cue_dist | 0.835 | 0.823 |
| false_admission | train_mix | 0.613 | 0.693 |
| parse_failed | none | 0.022 | 0.001 |
| parse_failed | cue_gold | 0.014 | 0.001 |
| parse_failed | cue_dist | 0.026 | 0.002 |
| parse_failed | train_mix | 0.020 | 0.002 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1083 | 1097 |
| n_bias_rows | cue_dist | 1071 | 1096 |
| n_bias_rows | train_mix | 2154 | 2193 |
| n_deception_rows | cue_gold | 1041 | 1084 |
| n_deception_rows | cue_dist | 1069 | 1089 |
| n_deception_rows | train_mix | 2110 | 2173 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.229 |
| accuracy | cue_gold | 0.524 | 0.517 |
| accuracy | cue_dist | 0.098 | 0.130 |
| accuracy | train_mix | 0.384 | 0.370 |
| reward | none | 0.313 | 0.355 |
| reward | cue_gold | 0.524 | 0.502 |
| reward | cue_dist | 0.225 | 0.317 |
| reward | train_mix | 0.479 | 0.506 |
| followed_cue | cue_gold | 0.524 | 0.517 |
| followed_cue | cue_dist | 0.870 | 0.870 |
| followed_cue | train_mix | 0.640 | 0.648 |
| bias | cue_gold | 0.476 | 0.301 |
| bias | cue_dist | 0.041 | 0.098 |
| bias | train_mix | 0.329 | 0.230 |
| anti_bias | cue_gold | 0.011 | 0.000 |
| anti_bias | cue_dist | 0.082 | 0.025 |
| anti_bias | train_mix | 0.044 | 0.012 |
| no_bias | cue_gold | 0.513 | 0.699 |
| no_bias | cue_dist | 0.877 | 0.877 |
| no_bias | train_mix | 0.627 | 0.758 |
| deception_soft | cue_gold | 0.568 | 0.563 |
| deception_soft | cue_dist | 0.606 | 0.616 |
| deception_soft | train_mix | 0.580 | 0.574 |
| deception | cue_gold | 0.614 | 0.587 |
| deception | cue_dist | 0.626 | 0.625 |
| deception | train_mix | 0.620 | 0.598 |
| biased_share | cue_gold | 0.481 | 0.301 |
| biased_share | cue_dist | 0.045 | 0.100 |
| biased_share | train_mix | 0.344 | 0.233 |
| false_denial_soft | cue_gold | 0.371 | 0.250 |
| false_denial_soft | cue_dist | 0.353 | 0.252 |
| false_denial_soft | train_mix | 0.364 | 0.239 |
| false_admission_soft | cue_gold | 0.750 | 0.697 |
| false_admission_soft | cue_dist | 0.618 | 0.657 |
| false_admission_soft | train_mix | 0.694 | 0.676 |
| said_yes_biased_soft | cue_gold | 0.629 | 0.750 |
| said_yes_biased_soft | cue_dist | 0.647 | 0.748 |
| said_yes_biased_soft | train_mix | 0.636 | 0.761 |
| deception_from_biased | cue_gold | 0.179 | 0.075 |
| deception_from_biased | cue_dist | 0.016 | 0.025 |
| deception_from_biased | train_mix | 0.125 | 0.056 |
| deception_from_unbiased | cue_gold | 0.389 | 0.487 |
| deception_from_unbiased | cue_dist | 0.591 | 0.591 |
| deception_from_unbiased | train_mix | 0.455 | 0.518 |
| said_yes | cue_gold | 0.746 | 0.743 |
| said_yes | cue_dist | 0.646 | 0.695 |
| said_yes | train_mix | 0.728 | 0.731 |
| false_denial | cue_gold | 0.362 | 0.241 |
| false_denial | cue_dist | 0.273 | 0.148 |
| false_denial | train_mix | 0.344 | 0.213 |
| false_admission | cue_gold | 0.847 | 0.736 |
| false_admission | cue_dist | 0.643 | 0.678 |
| false_admission | train_mix | 0.766 | 0.714 |
| parse_failed | none | 0.039 | 0.005 |
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
| n_deception_rows | cue_gold | 264 | 276 |
| n_deception_rows | cue_dist | 246 | 269 |
| n_deception_rows | train_mix | 510 | 545 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.775 | 0.881 | 0.881 | 0.881 |
| first_10_steps | label_correct__cue_at_other | 0.225 | 0.540 | 0.444 | 0.444 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.775 | 0.953 | 0.953 | 0.953 |
| last_10_steps | label_correct__cue_at_other | 0.225 | 0.220 | 0.779 | 0.779 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4008 | 0.857 | 0.840 | 0.857 |
| first_10_steps | base_wrong | 1112 | 0.514 | 0.677 | 0.514 |
| last_10_steps | base_right | 4104 | 0.964 | 0.791 | 0.964 |
| last_10_steps | base_wrong | 1016 | 0.708 | 0.773 | 0.708 |
