# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 478; in both 456 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.756 |
| accuracy | cue_gold | 0.852 | 0.954 |
| accuracy | cue_dist | 0.513 | 0.330 |
| accuracy | train_mix | 0.736 | 0.737 |
| reward | none | 0.647 | 0.671 |
| reward | cue_gold | 0.716 | 0.776 |
| reward | cue_dist | 0.498 | 0.398 |
| reward | train_mix | 0.690 | 0.743 |
| followed_cue | cue_gold | 0.852 | 0.954 |
| followed_cue | cue_dist | 0.460 | 0.669 |
| followed_cue | train_mix | 0.715 | 0.855 |
| bias | cue_gold | 0.101 | 0.170 |
| bias | cue_dist | 0.275 | 0.441 |
| bias | train_mix | 0.162 | 0.257 |
| anti_bias | cue_gold | 0.035 | 0.001 |
| anti_bias | cue_dist | 0.019 | 0.002 |
| anti_bias | train_mix | 0.030 | 0.002 |
| no_bias | cue_gold | 0.864 | 0.828 |
| no_bias | cue_dist | 0.706 | 0.556 |
| no_bias | train_mix | 0.808 | 0.741 |
| deception_soft | cue_gold | 0.548 | 0.538 |
| deception_soft | cue_dist | 0.590 | 0.549 |
| deception_soft | train_mix | 0.562 | 0.542 |
| deception | cue_gold | 0.551 | 0.560 |
| deception | cue_dist | 0.644 | 0.575 |
| deception | train_mix | 0.583 | 0.565 |
| biased_share | cue_gold | 0.105 | 0.171 |
| biased_share | cue_dist | 0.280 | 0.442 |
| biased_share | train_mix | 0.167 | 0.257 |
| false_denial_soft | cue_gold | 0.387 | 0.300 |
| false_denial_soft | cue_dist | 0.351 | 0.291 |
| false_denial_soft | train_mix | 0.363 | 0.290 |
| false_admission_soft | cue_gold | 0.567 | 0.587 |
| false_admission_soft | cue_dist | 0.684 | 0.754 |
| false_admission_soft | train_mix | 0.602 | 0.630 |
| said_yes_biased_soft | cue_gold | 0.613 | 0.700 |
| said_yes_biased_soft | cue_dist | 0.649 | 0.709 |
| said_yes_biased_soft | train_mix | 0.637 | 0.710 |
| deception_from_biased | cue_gold | 0.041 | 0.051 |
| deception_from_biased | cue_dist | 0.098 | 0.129 |
| deception_from_biased | train_mix | 0.061 | 0.075 |
| deception_from_unbiased | cue_gold | 0.508 | 0.487 |
| deception_from_unbiased | cue_dist | 0.492 | 0.420 |
| deception_from_unbiased | train_mix | 0.501 | 0.468 |
| said_yes | cue_gold | 0.577 | 0.638 |
| said_yes | cue_dist | 0.764 | 0.803 |
| said_yes | train_mix | 0.644 | 0.695 |
| false_denial | cue_gold | 0.375 | 0.270 |
| false_denial | cue_dist | 0.287 | 0.242 |
| false_denial | train_mix | 0.318 | 0.247 |
| false_admission | cue_gold | 0.572 | 0.620 |
| false_admission | cue_dist | 0.784 | 0.839 |
| false_admission | train_mix | 0.636 | 0.675 |
| parse_failed | none | 0.026 | 0.002 |
| parse_failed | cue_gold | 0.017 | 0.000 |
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
| n_bias_rows | cue_gold | 1344 | 1368 |
| n_bias_rows | cue_dist | 1330 | 1366 |
| n_bias_rows | train_mix | 2674 | 2734 |
| n_deception_rows | cue_gold | 1297 | 1366 |
| n_deception_rows | cue_dist | 1305 | 1363 |
| n_deception_rows | train_mix | 2602 | 2729 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.915 |
| accuracy | cue_gold | 0.942 | 0.997 |
| accuracy | cue_dist | 0.626 | 0.419 |
| accuracy | train_mix | 0.832 | 0.799 |
| reward | none | 0.738 | 0.760 |
| reward | cue_gold | 0.768 | 0.806 |
| reward | cue_dist | 0.572 | 0.443 |
| reward | train_mix | 0.748 | 0.769 |
| followed_cue | cue_gold | 0.942 | 0.997 |
| followed_cue | cue_dist | 0.348 | 0.579 |
| followed_cue | train_mix | 0.735 | 0.850 |
| bias | cue_gold | 0.007 | 0.034 |
| bias | cue_dist | 0.331 | 0.527 |
| bias | train_mix | 0.119 | 0.200 |
| anti_bias | cue_gold | 0.040 | 0.001 |
| anti_bias | cue_dist | 0.002 | 0.002 |
| anti_bias | train_mix | 0.027 | 0.002 |
| no_bias | cue_gold | 0.953 | 0.965 |
| no_bias | cue_dist | 0.667 | 0.471 |
| no_bias | train_mix | 0.854 | 0.798 |
| deception_soft | cue_gold | 0.540 | 0.564 |
| deception_soft | cue_dist | 0.591 | 0.531 |
| deception_soft | train_mix | 0.557 | 0.554 |
| deception | cue_gold | 0.533 | 0.591 |
| deception | cue_dist | 0.654 | 0.556 |
| deception | train_mix | 0.574 | 0.579 |
| biased_share | cue_gold | 0.007 | 0.034 |
| biased_share | cue_dist | 0.331 | 0.528 |
| biased_share | train_mix | 0.122 | 0.201 |
| false_denial_soft | cue_gold | 0.487 | 0.368 |
| false_denial_soft | cue_dist | 0.354 | 0.286 |
| false_denial_soft | train_mix | 0.358 | 0.293 |
| false_admission_soft | cue_gold | 0.541 | 0.571 |
| false_admission_soft | cue_dist | 0.708 | 0.806 |
| false_admission_soft | train_mix | 0.585 | 0.619 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.632 |
| said_yes_biased_soft | cue_dist | 0.646 | 0.714 |
| said_yes_biased_soft | train_mix | 0.642 | 0.707 |
| deception_from_biased | cue_gold | 0.003 | 0.013 |
| deception_from_biased | cue_dist | 0.117 | 0.151 |
| deception_from_biased | train_mix | 0.044 | 0.059 |
| deception_from_unbiased | cue_gold | 0.537 | 0.551 |
| deception_from_unbiased | cue_dist | 0.473 | 0.380 |
| deception_from_unbiased | train_mix | 0.513 | 0.495 |
| said_yes | cue_gold | 0.534 | 0.595 |
| said_yes | cue_dist | 0.792 | 0.836 |
| said_yes | train_mix | 0.623 | 0.680 |
| false_denial | cue_gold | 0.429 | 0.432 |
| false_denial | cue_dist | 0.292 | 0.235 |
| false_denial | train_mix | 0.298 | 0.249 |
| false_admission | cue_gold | 0.533 | 0.596 |
| false_admission | cue_dist | 0.834 | 0.916 |
| false_admission | train_mix | 0.612 | 0.662 |
| parse_failed | none | 0.022 | 0.001 |
| parse_failed | cue_gold | 0.014 | 0.000 |
| parse_failed | cue_dist | 0.026 | 0.002 |
| parse_failed | train_mix | 0.020 | 0.000 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1068 | 1083 |
| n_bias_rows | cue_dist | 1055 | 1081 |
| n_bias_rows | train_mix | 2123 | 2164 |
| n_deception_rows | cue_gold | 1025 | 1082 |
| n_deception_rows | cue_dist | 1053 | 1079 |
| n_deception_rows | train_mix | 2078 | 2161 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.175 |
| accuracy | cue_gold | 0.524 | 0.800 |
| accuracy | cue_dist | 0.098 | 0.003 |
| accuracy | train_mix | 0.384 | 0.509 |
| reward | none | 0.313 | 0.346 |
| reward | cue_gold | 0.524 | 0.663 |
| reward | cue_dist | 0.225 | 0.235 |
| reward | train_mix | 0.479 | 0.648 |
| followed_cue | cue_gold | 0.524 | 0.800 |
| followed_cue | cue_dist | 0.870 | 0.997 |
| followed_cue | train_mix | 0.640 | 0.873 |
| bias | cue_gold | 0.467 | 0.688 |
| bias | cue_dist | 0.062 | 0.116 |
| bias | train_mix | 0.329 | 0.471 |
| anti_bias | cue_gold | 0.014 | 0.004 |
| anti_bias | cue_dist | 0.084 | 0.004 |
| anti_bias | train_mix | 0.043 | 0.005 |
| no_bias | cue_gold | 0.518 | 0.309 |
| no_bias | cue_dist | 0.855 | 0.881 |
| no_bias | train_mix | 0.628 | 0.524 |
| deception_soft | cue_gold | 0.577 | 0.441 |
| deception_soft | cue_dist | 0.590 | 0.616 |
| deception_soft | train_mix | 0.583 | 0.498 |
| deception | cue_gold | 0.621 | 0.444 |
| deception | cue_dist | 0.603 | 0.648 |
| deception | train_mix | 0.619 | 0.508 |
| biased_share | cue_gold | 0.474 | 0.690 |
| biased_share | cue_dist | 0.067 | 0.116 |
| biased_share | train_mix | 0.344 | 0.474 |
| false_denial_soft | cue_gold | 0.382 | 0.287 |
| false_denial_soft | cue_dist | 0.280 | 0.376 |
| false_denial_soft | train_mix | 0.370 | 0.286 |
| false_admission_soft | cue_gold | 0.753 | 0.784 |
| false_admission_soft | cue_dist | 0.612 | 0.648 |
| false_admission_soft | train_mix | 0.694 | 0.690 |
| said_yes_biased_soft | cue_gold | 0.618 | 0.713 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.624 |
| said_yes_biased_soft | train_mix | 0.630 | 0.714 |
| deception_from_biased | cue_gold | 0.181 | 0.198 |
| deception_from_biased | cue_dist | 0.019 | 0.044 |
| deception_from_biased | train_mix | 0.127 | 0.135 |
| deception_from_unbiased | cue_gold | 0.396 | 0.243 |
| deception_from_unbiased | cue_dist | 0.571 | 0.573 |
| deception_from_unbiased | train_mix | 0.455 | 0.363 |
| said_yes | cue_gold | 0.743 | 0.803 |
| said_yes | cue_dist | 0.647 | 0.680 |
| said_yes | train_mix | 0.725 | 0.750 |
| false_denial | cue_gold | 0.372 | 0.240 |
| false_denial | cue_dist | 0.176 | 0.364 |
| false_denial | train_mix | 0.345 | 0.244 |
| false_admission | cue_gold | 0.846 | 0.898 |
| false_admission | cue_dist | 0.634 | 0.685 |
| false_admission | train_mix | 0.762 | 0.745 |
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
| n_bias_rows | cue_gold | 276 | 285 |
| n_bias_rows | cue_dist | 275 | 285 |
| n_bias_rows | train_mix | 551 | 570 |
| n_deception_rows | cue_gold | 272 | 284 |
| n_deception_rows | cue_dist | 252 | 284 |
| n_deception_rows | train_mix | 524 | 568 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.619 | 0.907 | 0.907 | 0.907 |
| first_10_steps | label_correct__cue_at_other | 0.188 | 0.596 | 0.392 | 0.392 |
| first_10_steps | label_flipped__cue_at_label | 0.150 | 0.589 | 0.589 | 0.408 |
| first_10_steps | label_flipped__cue_at_other | 0.044 | 0.875 | 0.116 | 0.875 |
| last_10_steps | label_correct__cue_at_label | 0.631 | 0.980 | 0.980 | 0.980 |
| last_10_steps | label_correct__cue_at_other | 0.180 | 0.555 | 0.442 | 0.442 |
| last_10_steps | label_flipped__cue_at_label | 0.148 | 0.789 | 0.789 | 0.211 |
| last_10_steps | label_flipped__cue_at_other | 0.041 | 0.913 | 0.087 | 0.913 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4136 | 0.801 | 0.807 | 0.764 |
| first_10_steps | base_wrong | 984 | 0.452 | 0.770 | 0.577 |
| last_10_steps | base_right | 4032 | 0.803 | 0.858 | 0.841 |
| last_10_steps | base_wrong | 1088 | 0.631 | 0.928 | 0.736 |
