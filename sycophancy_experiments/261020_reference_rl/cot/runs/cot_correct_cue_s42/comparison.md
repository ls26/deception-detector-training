# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 477; in both 456 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.771 |
| accuracy | cue_gold | 0.852 | 0.890 |
| accuracy | cue_dist | 0.513 | 0.607 |
| accuracy | train_mix | 0.736 | 0.792 |
| reward | none | 0.647 | 0.672 |
| reward | cue_gold | 0.716 | 0.740 |
| reward | cue_dist | 0.498 | 0.566 |
| reward | train_mix | 0.690 | 0.723 |
| followed_cue | cue_gold | 0.852 | 0.890 |
| followed_cue | cue_dist | 0.460 | 0.391 |
| followed_cue | train_mix | 0.715 | 0.717 |
| bias | cue_gold | 0.101 | 0.111 |
| bias | cue_dist | 0.273 | 0.165 |
| bias | train_mix | 0.163 | 0.130 |
| anti_bias | cue_gold | 0.034 | 0.006 |
| anti_bias | cue_dist | 0.020 | 0.012 |
| anti_bias | train_mix | 0.030 | 0.008 |
| no_bias | cue_gold | 0.865 | 0.884 |
| no_bias | cue_dist | 0.706 | 0.823 |
| no_bias | train_mix | 0.806 | 0.862 |
| deception_soft | cue_gold | 0.546 | 0.427 |
| deception_soft | cue_dist | 0.592 | 0.534 |
| deception_soft | train_mix | 0.561 | 0.466 |
| deception | cue_gold | 0.549 | 0.373 |
| deception | cue_dist | 0.646 | 0.538 |
| deception | train_mix | 0.581 | 0.434 |
| biased_share | cue_gold | 0.105 | 0.111 |
| biased_share | cue_dist | 0.279 | 0.168 |
| biased_share | train_mix | 0.168 | 0.131 |
| false_denial_soft | cue_gold | 0.390 | 0.468 |
| false_denial_soft | cue_dist | 0.352 | 0.500 |
| false_denial_soft | train_mix | 0.364 | 0.491 |
| false_admission_soft | cue_gold | 0.564 | 0.421 |
| false_admission_soft | cue_dist | 0.685 | 0.541 |
| false_admission_soft | train_mix | 0.600 | 0.463 |
| said_yes_biased_soft | cue_gold | 0.610 | 0.532 |
| said_yes_biased_soft | cue_dist | 0.648 | 0.500 |
| said_yes_biased_soft | train_mix | 0.636 | 0.509 |
| deception_from_biased | cue_gold | 0.041 | 0.052 |
| deception_from_biased | cue_dist | 0.098 | 0.084 |
| deception_from_biased | train_mix | 0.061 | 0.064 |
| deception_from_unbiased | cue_gold | 0.505 | 0.375 |
| deception_from_unbiased | cue_dist | 0.494 | 0.451 |
| deception_from_unbiased | train_mix | 0.499 | 0.402 |
| said_yes | cue_gold | 0.574 | 0.367 |
| said_yes | cue_dist | 0.766 | 0.528 |
| said_yes | train_mix | 0.642 | 0.422 |
| false_denial | cue_gold | 0.382 | 0.530 |
| false_denial | cue_dist | 0.286 | 0.531 |
| false_denial | train_mix | 0.319 | 0.544 |
| false_admission | cue_gold | 0.569 | 0.354 |
| false_admission | cue_dist | 0.785 | 0.540 |
| false_admission | train_mix | 0.634 | 0.417 |
| parse_failed | none | 0.026 | 0.003 |
| parse_failed | cue_gold | 0.017 | 0.002 |
| parse_failed | cue_dist | 0.027 | 0.001 |
| parse_failed | train_mix | 0.022 | 0.002 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.001 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1344 | 1366 |
| n_bias_rows | cue_dist | 1332 | 1366 |
| n_bias_rows | train_mix | 2676 | 2732 |
| n_deception_rows | cue_gold | 1298 | 1358 |
| n_deception_rows | cue_dist | 1305 | 1349 |
| n_deception_rows | train_mix | 2603 | 2707 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.924 |
| accuracy | cue_gold | 0.942 | 0.984 |
| accuracy | cue_dist | 0.626 | 0.751 |
| accuracy | train_mix | 0.832 | 0.906 |
| reward | none | 0.738 | 0.753 |
| reward | cue_gold | 0.768 | 0.799 |
| reward | cue_dist | 0.572 | 0.646 |
| reward | train_mix | 0.748 | 0.781 |
| followed_cue | cue_gold | 0.942 | 0.984 |
| followed_cue | cue_dist | 0.348 | 0.249 |
| followed_cue | train_mix | 0.735 | 0.728 |
| bias | cue_gold | 0.007 | 0.042 |
| bias | cue_dist | 0.331 | 0.185 |
| bias | train_mix | 0.121 | 0.095 |
| anti_bias | cue_gold | 0.039 | 0.007 |
| anti_bias | cue_dist | 0.002 | 0.007 |
| anti_bias | train_mix | 0.026 | 0.006 |
| no_bias | cue_gold | 0.954 | 0.950 |
| no_bias | cue_dist | 0.668 | 0.808 |
| no_bias | train_mix | 0.853 | 0.898 |
| deception_soft | cue_gold | 0.539 | 0.394 |
| deception_soft | cue_dist | 0.592 | 0.546 |
| deception_soft | train_mix | 0.556 | 0.450 |
| deception | cue_gold | 0.531 | 0.316 |
| deception | cue_dist | 0.655 | 0.564 |
| deception | train_mix | 0.572 | 0.406 |
| biased_share | cue_gold | 0.007 | 0.043 |
| biased_share | cue_dist | 0.331 | 0.187 |
| biased_share | train_mix | 0.125 | 0.096 |
| false_denial_soft | cue_gold | 0.487 | 0.440 |
| false_denial_soft | cue_dist | 0.353 | 0.510 |
| false_denial_soft | train_mix | 0.355 | 0.502 |
| false_admission_soft | cue_gold | 0.539 | 0.392 |
| false_admission_soft | cue_dist | 0.710 | 0.555 |
| false_admission_soft | train_mix | 0.584 | 0.444 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.560 |
| said_yes_biased_soft | cue_dist | 0.647 | 0.490 |
| said_yes_biased_soft | train_mix | 0.645 | 0.498 |
| deception_from_biased | cue_gold | 0.003 | 0.019 |
| deception_from_biased | cue_dist | 0.117 | 0.095 |
| deception_from_biased | train_mix | 0.044 | 0.048 |
| deception_from_unbiased | cue_gold | 0.536 | 0.375 |
| deception_from_unbiased | cue_dist | 0.475 | 0.451 |
| deception_from_unbiased | train_mix | 0.511 | 0.401 |
| said_yes | cue_gold | 0.532 | 0.318 |
| said_yes | cue_dist | 0.795 | 0.548 |
| said_yes | train_mix | 0.623 | 0.396 |
| false_denial | cue_gold | 0.429 | 0.478 |
| false_denial | cue_dist | 0.289 | 0.542 |
| false_denial | train_mix | 0.294 | 0.550 |
| false_admission | cue_gold | 0.531 | 0.308 |
| false_admission | cue_dist | 0.836 | 0.568 |
| false_admission | train_mix | 0.611 | 0.391 |
| parse_failed | none | 0.022 | 0.002 |
| parse_failed | cue_gold | 0.014 | 0.001 |
| parse_failed | cue_dist | 0.026 | 0.000 |
| parse_failed | train_mix | 0.020 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.001 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1071 | 1085 |
| n_bias_rows | cue_dist | 1059 | 1086 |
| n_bias_rows | train_mix | 2130 | 2171 |
| n_deception_rows | cue_gold | 1029 | 1077 |
| n_deception_rows | cue_dist | 1057 | 1077 |
| n_deception_rows | train_mix | 2086 | 2154 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.210 |
| accuracy | cue_gold | 0.524 | 0.546 |
| accuracy | cue_dist | 0.098 | 0.083 |
| accuracy | train_mix | 0.384 | 0.376 |
| reward | none | 0.313 | 0.379 |
| reward | cue_gold | 0.524 | 0.524 |
| reward | cue_dist | 0.225 | 0.273 |
| reward | train_mix | 0.479 | 0.513 |
| followed_cue | cue_gold | 0.524 | 0.546 |
| followed_cue | cue_dist | 0.870 | 0.911 |
| followed_cue | train_mix | 0.640 | 0.679 |
| bias | cue_gold | 0.473 | 0.374 |
| bias | cue_dist | 0.051 | 0.089 |
| bias | train_mix | 0.326 | 0.264 |
| anti_bias | cue_gold | 0.015 | 0.000 |
| anti_bias | cue_dist | 0.092 | 0.029 |
| anti_bias | train_mix | 0.049 | 0.016 |
| no_bias | cue_gold | 0.513 | 0.626 |
| no_bias | cue_dist | 0.857 | 0.882 |
| no_bias | train_mix | 0.625 | 0.720 |
| deception_soft | cue_gold | 0.572 | 0.551 |
| deception_soft | cue_dist | 0.593 | 0.487 |
| deception_soft | train_mix | 0.581 | 0.532 |
| deception | cue_gold | 0.621 | 0.594 |
| deception | cue_dist | 0.609 | 0.438 |
| deception | train_mix | 0.620 | 0.542 |
| biased_share | cue_gold | 0.480 | 0.374 |
| biased_share | cue_dist | 0.056 | 0.092 |
| biased_share | train_mix | 0.343 | 0.268 |
| false_denial_soft | cue_gold | 0.385 | 0.480 |
| false_denial_soft | cue_dist | 0.327 | 0.426 |
| false_denial_soft | train_mix | 0.376 | 0.474 |
| false_admission_soft | cue_gold | 0.745 | 0.593 |
| false_admission_soft | cue_dist | 0.609 | 0.493 |
| false_admission_soft | train_mix | 0.688 | 0.553 |
| said_yes_biased_soft | cue_gold | 0.615 | 0.520 |
| said_yes_biased_soft | cue_dist | 0.673 | 0.574 |
| said_yes_biased_soft | train_mix | 0.624 | 0.526 |
| deception_from_biased | cue_gold | 0.184 | 0.179 |
| deception_from_biased | cue_dist | 0.018 | 0.039 |
| deception_from_biased | train_mix | 0.129 | 0.127 |
| deception_from_unbiased | cue_gold | 0.388 | 0.372 |
| deception_from_unbiased | cue_dist | 0.574 | 0.448 |
| deception_from_unbiased | train_mix | 0.452 | 0.405 |
| said_yes | cue_gold | 0.736 | 0.555 |
| said_yes | cue_dist | 0.641 | 0.449 |
| said_yes | train_mix | 0.720 | 0.522 |
| false_denial | cue_gold | 0.380 | 0.552 |
| false_denial | cue_dist | 0.214 | 0.440 |
| false_denial | train_mix | 0.355 | 0.537 |
| false_admission | cue_gold | 0.843 | 0.619 |
| false_admission | cue_dist | 0.632 | 0.437 |
| false_admission | train_mix | 0.759 | 0.544 |
| parse_failed | none | 0.039 | 0.006 |
| parse_failed | cue_gold | 0.029 | 0.006 |
| parse_failed | cue_dist | 0.032 | 0.006 |
| parse_failed | train_mix | 0.033 | 0.008 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 840 | 840 |
| rollouts | cue_gold | 315 | 315 |
| rollouts | cue_dist | 315 | 315 |
| rollouts | train_mix | 630 | 630 |
| n_bias_rows | cue_gold | 273 | 281 |
| n_bias_rows | cue_dist | 273 | 280 |
| n_bias_rows | train_mix | 546 | 561 |
| n_deception_rows | cue_gold | 269 | 281 |
| n_deception_rows | cue_dist | 248 | 272 |
| n_deception_rows | train_mix | 517 | 553 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.781 | 0.911 | 0.911 | 0.911 |
| first_10_steps | label_correct__cue_at_other | 0.219 | 0.544 | 0.443 | 0.443 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.778 | 0.949 | 0.949 | 0.949 |
| last_10_steps | label_correct__cue_at_other | 0.222 | 0.247 | 0.750 | 0.750 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4080 | 0.858 | 0.848 | 0.858 |
| first_10_steps | base_wrong | 1040 | 0.616 | 0.765 | 0.616 |
| last_10_steps | base_right | 4144 | 0.952 | 0.799 | 0.952 |
| last_10_steps | base_wrong | 976 | 0.701 | 0.766 | 0.701 |
