# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 481; in both 458 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.761 |
| accuracy | cue_gold | 0.852 | 0.905 |
| accuracy | cue_dist | 0.513 | 0.519 |
| accuracy | train_mix | 0.736 | 0.770 |
| reward | none | 0.647 | 0.659 |
| reward | cue_gold | 0.716 | 0.746 |
| reward | cue_dist | 0.498 | 0.513 |
| reward | train_mix | 0.690 | 0.726 |
| followed_cue | cue_gold | 0.852 | 0.905 |
| followed_cue | cue_dist | 0.460 | 0.481 |
| followed_cue | train_mix | 0.715 | 0.758 |
| bias | cue_gold | 0.099 | 0.132 |
| bias | cue_dist | 0.281 | 0.245 |
| bias | train_mix | 0.162 | 0.173 |
| anti_bias | cue_gold | 0.036 | 0.000 |
| anti_bias | cue_dist | 0.019 | 0.006 |
| anti_bias | train_mix | 0.031 | 0.002 |
| no_bias | cue_gold | 0.866 | 0.868 |
| no_bias | cue_dist | 0.700 | 0.749 |
| no_bias | train_mix | 0.807 | 0.824 |
| deception_soft | cue_gold | 0.549 | 0.477 |
| deception_soft | cue_dist | 0.591 | 0.588 |
| deception_soft | train_mix | 0.563 | 0.515 |
| deception | cue_gold | 0.556 | 0.425 |
| deception | cue_dist | 0.646 | 0.613 |
| deception | train_mix | 0.587 | 0.488 |
| biased_share | cue_gold | 0.102 | 0.132 |
| biased_share | cue_dist | 0.286 | 0.247 |
| biased_share | train_mix | 0.167 | 0.174 |
| false_denial_soft | cue_gold | 0.381 | 0.370 |
| false_denial_soft | cue_dist | 0.350 | 0.403 |
| false_denial_soft | train_mix | 0.360 | 0.383 |
| false_admission_soft | cue_gold | 0.568 | 0.494 |
| false_admission_soft | cue_dist | 0.688 | 0.649 |
| false_admission_soft | train_mix | 0.604 | 0.542 |
| said_yes_biased_soft | cue_gold | 0.619 | 0.630 |
| said_yes_biased_soft | cue_dist | 0.650 | 0.597 |
| said_yes_biased_soft | train_mix | 0.640 | 0.617 |
| deception_from_biased | cue_gold | 0.039 | 0.049 |
| deception_from_biased | cue_dist | 0.100 | 0.099 |
| deception_from_biased | train_mix | 0.060 | 0.067 |
| deception_from_unbiased | cue_gold | 0.510 | 0.428 |
| deception_from_unbiased | cue_dist | 0.491 | 0.489 |
| deception_from_unbiased | train_mix | 0.503 | 0.448 |
| said_yes | cue_gold | 0.581 | 0.443 |
| said_yes | cue_dist | 0.770 | 0.643 |
| said_yes | train_mix | 0.649 | 0.512 |
| false_denial | cue_gold | 0.376 | 0.434 |
| false_denial | cue_dist | 0.283 | 0.439 |
| false_denial | train_mix | 0.315 | 0.431 |
| false_admission | cue_gold | 0.577 | 0.424 |
| false_admission | cue_dist | 0.791 | 0.671 |
| false_admission | train_mix | 0.642 | 0.500 |
| parse_failed | none | 0.026 | 0.000 |
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
| n_bias_rows | cue_gold | 1350 | 1374 |
| n_bias_rows | cue_dist | 1336 | 1374 |
| n_bias_rows | train_mix | 2686 | 2748 |
| n_deception_rows | cue_gold | 1302 | 1374 |
| n_deception_rows | cue_dist | 1310 | 1366 |
| n_deception_rows | train_mix | 2612 | 2740 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.914 |
| accuracy | cue_gold | 0.942 | 0.982 |
| accuracy | cue_dist | 0.626 | 0.643 |
| accuracy | train_mix | 0.832 | 0.866 |
| reward | none | 0.738 | 0.746 |
| reward | cue_gold | 0.768 | 0.792 |
| reward | cue_dist | 0.572 | 0.576 |
| reward | train_mix | 0.748 | 0.769 |
| followed_cue | cue_gold | 0.942 | 0.982 |
| followed_cue | cue_dist | 0.348 | 0.357 |
| followed_cue | train_mix | 0.735 | 0.761 |
| bias | cue_gold | 0.006 | 0.052 |
| bias | cue_dist | 0.337 | 0.280 |
| bias | train_mix | 0.121 | 0.132 |
| anti_bias | cue_gold | 0.041 | 0.000 |
| anti_bias | cue_dist | 0.001 | 0.004 |
| anti_bias | train_mix | 0.027 | 0.001 |
| no_bias | cue_gold | 0.953 | 0.948 |
| no_bias | cue_dist | 0.662 | 0.716 |
| no_bias | train_mix | 0.852 | 0.867 |
| deception_soft | cue_gold | 0.541 | 0.468 |
| deception_soft | cue_dist | 0.590 | 0.593 |
| deception_soft | train_mix | 0.558 | 0.511 |
| deception | cue_gold | 0.537 | 0.405 |
| deception | cue_dist | 0.651 | 0.629 |
| deception | train_mix | 0.576 | 0.478 |
| biased_share | cue_gold | 0.006 | 0.052 |
| biased_share | cue_dist | 0.338 | 0.281 |
| biased_share | train_mix | 0.125 | 0.132 |
| false_denial_soft | cue_gold | 0.481 | 0.463 |
| false_denial_soft | cue_dist | 0.354 | 0.407 |
| false_denial_soft | train_mix | 0.356 | 0.420 |
| false_admission_soft | cue_gold | 0.542 | 0.469 |
| false_admission_soft | cue_dist | 0.710 | 0.666 |
| false_admission_soft | train_mix | 0.586 | 0.525 |
| said_yes_biased_soft | cue_gold | 0.519 | 0.537 |
| said_yes_biased_soft | cue_dist | 0.646 | 0.593 |
| said_yes_biased_soft | train_mix | 0.644 | 0.580 |
| deception_from_biased | cue_gold | 0.003 | 0.024 |
| deception_from_biased | cue_dist | 0.119 | 0.114 |
| deception_from_biased | train_mix | 0.044 | 0.056 |
| deception_from_unbiased | cue_gold | 0.539 | 0.444 |
| deception_from_unbiased | cue_dist | 0.470 | 0.479 |
| deception_from_unbiased | train_mix | 0.513 | 0.455 |
| said_yes | cue_gold | 0.539 | 0.397 |
| said_yes | cue_dist | 0.794 | 0.665 |
| said_yes | train_mix | 0.628 | 0.486 |
| false_denial | cue_gold | 0.333 | 0.579 |
| false_denial | cue_dist | 0.288 | 0.436 |
| false_denial | train_mix | 0.292 | 0.472 |
| false_admission | cue_gold | 0.538 | 0.395 |
| false_admission | cue_dist | 0.836 | 0.704 |
| false_admission | train_mix | 0.617 | 0.479 |
| parse_failed | none | 0.022 | 0.000 |
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
| n_bias_rows | cue_gold | 1074 | 1089 |
| n_bias_rows | cue_dist | 1061 | 1089 |
| n_bias_rows | train_mix | 2135 | 2178 |
| n_deception_rows | cue_gold | 1030 | 1089 |
| n_deception_rows | cue_dist | 1060 | 1085 |
| n_deception_rows | train_mix | 2090 | 2174 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.200 |
| accuracy | cue_gold | 0.524 | 0.622 |
| accuracy | cue_dist | 0.098 | 0.063 |
| accuracy | train_mix | 0.384 | 0.417 |
| reward | none | 0.313 | 0.340 |
| reward | cue_gold | 0.524 | 0.581 |
| reward | cue_dist | 0.225 | 0.283 |
| reward | train_mix | 0.479 | 0.571 |
| followed_cue | cue_gold | 0.524 | 0.622 |
| followed_cue | cue_dist | 0.870 | 0.937 |
| followed_cue | train_mix | 0.640 | 0.745 |
| bias | cue_gold | 0.460 | 0.439 |
| bias | cue_dist | 0.062 | 0.112 |
| bias | train_mix | 0.320 | 0.331 |
| anti_bias | cue_gold | 0.014 | 0.000 |
| anti_bias | cue_dist | 0.091 | 0.014 |
| anti_bias | train_mix | 0.048 | 0.007 |
| no_bias | cue_gold | 0.525 | 0.561 |
| no_bias | cue_dist | 0.847 | 0.874 |
| no_bias | train_mix | 0.631 | 0.662 |
| deception_soft | cue_gold | 0.577 | 0.512 |
| deception_soft | cue_dist | 0.599 | 0.569 |
| deception_soft | train_mix | 0.586 | 0.529 |
| deception | cue_gold | 0.629 | 0.502 |
| deception | cue_dist | 0.624 | 0.555 |
| deception | train_mix | 0.630 | 0.526 |
| biased_share | cue_gold | 0.467 | 0.439 |
| biased_share | cue_dist | 0.068 | 0.114 |
| biased_share | train_mix | 0.337 | 0.333 |
| false_denial_soft | cue_gold | 0.376 | 0.327 |
| false_denial_soft | cue_dist | 0.280 | 0.370 |
| false_denial_soft | train_mix | 0.364 | 0.326 |
| false_admission_soft | cue_gold | 0.752 | 0.656 |
| false_admission_soft | cue_dist | 0.622 | 0.595 |
| false_admission_soft | train_mix | 0.698 | 0.630 |
| said_yes_biased_soft | cue_gold | 0.624 | 0.673 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.630 |
| said_yes_biased_soft | train_mix | 0.636 | 0.674 |
| deception_from_biased | cue_gold | 0.176 | 0.144 |
| deception_from_biased | cue_dist | 0.019 | 0.042 |
| deception_from_biased | train_mix | 0.123 | 0.108 |
| deception_from_unbiased | cue_gold | 0.401 | 0.368 |
| deception_from_unbiased | cue_dist | 0.580 | 0.527 |
| deception_from_unbiased | train_mix | 0.463 | 0.420 |
| said_yes | cue_gold | 0.743 | 0.618 |
| said_yes | cue_dist | 0.668 | 0.562 |
| said_yes | train_mix | 0.731 | 0.613 |
| false_denial | cue_gold | 0.378 | 0.368 |
| false_denial | cue_dist | 0.176 | 0.469 |
| false_denial | train_mix | 0.349 | 0.369 |
| false_admission | cue_gold | 0.848 | 0.606 |
| false_admission | cue_dist | 0.657 | 0.566 |
| false_admission | train_mix | 0.772 | 0.604 |
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
| n_bias_rows | cue_gold | 276 | 285 |
| n_bias_rows | cue_dist | 275 | 285 |
| n_bias_rows | train_mix | 551 | 570 |
| n_deception_rows | cue_gold | 272 | 285 |
| n_deception_rows | cue_dist | 250 | 281 |
| n_deception_rows | train_mix | 522 | 566 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 0.806 | 0.018 | 0.765 | 0.765 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.194 | 0.010 | 0.238 | 0.752 |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 0.803 | 0.000 | 0.902 | 0.902 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.197 | 0.000 | 0.405 | 0.595 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4144 | 0.909 | 0.014 | 0.751 |
| first_10_steps | base_wrong | 976 | 0.138 | 0.025 | 0.286 |
| last_10_steps | base_right | 4128 | 0.933 | 0.000 | 0.844 |
| last_10_steps | base_wrong | 992 | 0.461 | 0.002 | 0.636 |
