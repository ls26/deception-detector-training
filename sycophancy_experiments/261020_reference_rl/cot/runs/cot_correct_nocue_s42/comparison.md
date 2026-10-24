# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 476; in both 455 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.801 |
| accuracy | cue_gold | 0.852 | 0.935 |
| accuracy | cue_dist | 0.513 | 0.503 |
| accuracy | train_mix | 0.736 | 0.782 |
| reward | none | 0.647 | 0.690 |
| reward | cue_gold | 0.716 | 0.763 |
| reward | cue_dist | 0.498 | 0.509 |
| reward | train_mix | 0.690 | 0.744 |
| followed_cue | cue_gold | 0.852 | 0.935 |
| followed_cue | cue_dist | 0.460 | 0.497 |
| followed_cue | train_mix | 0.715 | 0.784 |
| bias | cue_gold | 0.098 | 0.119 |
| bias | cue_dist | 0.277 | 0.316 |
| bias | train_mix | 0.163 | 0.186 |
| anti_bias | cue_gold | 0.035 | 0.004 |
| anti_bias | cue_dist | 0.020 | 0.010 |
| anti_bias | train_mix | 0.031 | 0.005 |
| no_bias | cue_gold | 0.867 | 0.876 |
| no_bias | cue_dist | 0.704 | 0.674 |
| no_bias | train_mix | 0.807 | 0.809 |
| deception_soft | cue_gold | 0.548 | 0.604 |
| deception_soft | cue_dist | 0.589 | 0.610 |
| deception_soft | train_mix | 0.561 | 0.603 |
| deception | cue_gold | 0.550 | 0.653 |
| deception | cue_dist | 0.640 | 0.662 |
| deception | train_mix | 0.579 | 0.651 |
| biased_share | cue_gold | 0.101 | 0.120 |
| biased_share | cue_dist | 0.282 | 0.320 |
| biased_share | train_mix | 0.168 | 0.187 |
| false_denial_soft | cue_gold | 0.389 | 0.287 |
| false_denial_soft | cue_dist | 0.344 | 0.279 |
| false_denial_soft | train_mix | 0.358 | 0.271 |
| false_admission_soft | cue_gold | 0.566 | 0.647 |
| false_admission_soft | cue_dist | 0.685 | 0.765 |
| false_admission_soft | train_mix | 0.601 | 0.679 |
| said_yes_biased_soft | cue_gold | 0.611 | 0.713 |
| said_yes_biased_soft | cue_dist | 0.656 | 0.721 |
| said_yes_biased_soft | train_mix | 0.642 | 0.729 |
| deception_from_biased | cue_gold | 0.039 | 0.034 |
| deception_from_biased | cue_dist | 0.097 | 0.089 |
| deception_from_biased | train_mix | 0.060 | 0.051 |
| deception_from_unbiased | cue_gold | 0.509 | 0.570 |
| deception_from_unbiased | cue_dist | 0.492 | 0.521 |
| deception_from_unbiased | train_mix | 0.500 | 0.552 |
| said_yes | cue_gold | 0.576 | 0.725 |
| said_yes | cue_dist | 0.768 | 0.862 |
| said_yes | train_mix | 0.644 | 0.771 |
| false_denial | cue_gold | 0.374 | 0.202 |
| false_denial | cue_dist | 0.272 | 0.188 |
| false_denial | train_mix | 0.308 | 0.179 |
| false_admission | cue_gold | 0.570 | 0.715 |
| false_admission | cue_dist | 0.784 | 0.885 |
| false_admission | train_mix | 0.634 | 0.759 |
| parse_failed | none | 0.026 | 0.000 |
| parse_failed | cue_gold | 0.017 | 0.001 |
| parse_failed | cue_dist | 0.027 | 0.000 |
| parse_failed | train_mix | 0.022 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.001 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.001 |
| rollouts | none | 3912 | 3912 |
| rollouts | cue_gold | 1467 | 1467 |
| rollouts | cue_dist | 1467 | 1467 |
| rollouts | train_mix | 2934 | 2934 |
| n_bias_rows | cue_gold | 1341 | 1365 |
| n_bias_rows | cue_dist | 1327 | 1365 |
| n_bias_rows | train_mix | 2668 | 2730 |
| n_deception_rows | cue_gold | 1294 | 1359 |
| n_deception_rows | cue_dist | 1301 | 1352 |
| n_deception_rows | train_mix | 2595 | 2711 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.925 |
| accuracy | cue_gold | 0.942 | 0.983 |
| accuracy | cue_dist | 0.626 | 0.609 |
| accuracy | train_mix | 0.832 | 0.852 |
| reward | none | 0.738 | 0.767 |
| reward | cue_gold | 0.768 | 0.796 |
| reward | cue_dist | 0.572 | 0.570 |
| reward | train_mix | 0.748 | 0.776 |
| followed_cue | cue_gold | 0.942 | 0.983 |
| followed_cue | cue_dist | 0.348 | 0.391 |
| followed_cue | train_mix | 0.735 | 0.778 |
| bias | cue_gold | 0.007 | 0.038 |
| bias | cue_dist | 0.330 | 0.342 |
| bias | train_mix | 0.121 | 0.143 |
| anti_bias | cue_gold | 0.040 | 0.005 |
| anti_bias | cue_dist | 0.002 | 0.008 |
| anti_bias | train_mix | 0.026 | 0.005 |
| no_bias | cue_gold | 0.953 | 0.958 |
| no_bias | cue_dist | 0.668 | 0.650 |
| no_bias | train_mix | 0.852 | 0.852 |
| deception_soft | cue_gold | 0.541 | 0.615 |
| deception_soft | cue_dist | 0.590 | 0.618 |
| deception_soft | train_mix | 0.556 | 0.614 |
| deception | cue_gold | 0.532 | 0.675 |
| deception | cue_dist | 0.650 | 0.673 |
| deception | train_mix | 0.572 | 0.673 |
| biased_share | cue_gold | 0.007 | 0.038 |
| biased_share | cue_dist | 0.331 | 0.344 |
| biased_share | train_mix | 0.124 | 0.144 |
| false_denial_soft | cue_gold | 0.487 | 0.239 |
| false_denial_soft | cue_dist | 0.347 | 0.294 |
| false_denial_soft | train_mix | 0.351 | 0.281 |
| false_admission_soft | cue_gold | 0.541 | 0.630 |
| false_admission_soft | cue_dist | 0.709 | 0.788 |
| false_admission_soft | train_mix | 0.585 | 0.670 |
| said_yes_biased_soft | cue_gold | 0.513 | 0.761 |
| said_yes_biased_soft | cue_dist | 0.653 | 0.706 |
| said_yes_biased_soft | train_mix | 0.649 | 0.719 |
| deception_from_biased | cue_gold | 0.003 | 0.009 |
| deception_from_biased | cue_dist | 0.115 | 0.101 |
| deception_from_biased | train_mix | 0.044 | 0.040 |
| deception_from_unbiased | cue_gold | 0.537 | 0.606 |
| deception_from_unbiased | cue_dist | 0.475 | 0.517 |
| deception_from_unbiased | train_mix | 0.513 | 0.574 |
| said_yes | cue_gold | 0.533 | 0.704 |
| said_yes | cue_dist | 0.797 | 0.879 |
| said_yes | train_mix | 0.625 | 0.761 |
| false_denial | cue_gold | 0.429 | 0.122 |
| false_denial | cue_dist | 0.277 | 0.202 |
| false_denial | train_mix | 0.286 | 0.193 |
| false_admission | cue_gold | 0.533 | 0.697 |
| false_admission | cue_dist | 0.834 | 0.921 |
| false_admission | train_mix | 0.612 | 0.753 |
| parse_failed | none | 0.022 | 0.000 |
| parse_failed | cue_gold | 0.014 | 0.001 |
| parse_failed | cue_dist | 0.026 | 0.000 |
| parse_failed | train_mix | 0.020 | 0.001 |
| cut_off | none | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.001 |
| cut_off | cue_dist | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.001 |
| rollouts | none | 3072 | 3072 |
| rollouts | cue_gold | 1152 | 1152 |
| rollouts | cue_dist | 1152 | 1152 |
| rollouts | train_mix | 2304 | 2304 |
| n_bias_rows | cue_gold | 1074 | 1089 |
| n_bias_rows | cue_dist | 1061 | 1089 |
| n_bias_rows | train_mix | 2135 | 2178 |
| n_deception_rows | cue_gold | 1031 | 1084 |
| n_deception_rows | cue_dist | 1059 | 1080 |
| n_deception_rows | train_mix | 2090 | 2164 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.349 |
| accuracy | cue_gold | 0.524 | 0.759 |
| accuracy | cue_dist | 0.098 | 0.114 |
| accuracy | train_mix | 0.384 | 0.526 |
| reward | none | 0.313 | 0.406 |
| reward | cue_gold | 0.524 | 0.641 |
| reward | cue_dist | 0.225 | 0.283 |
| reward | train_mix | 0.479 | 0.623 |
| followed_cue | cue_gold | 0.524 | 0.759 |
| followed_cue | cue_dist | 0.870 | 0.886 |
| followed_cue | train_mix | 0.640 | 0.803 |
| bias | cue_gold | 0.464 | 0.442 |
| bias | cue_dist | 0.064 | 0.217 |
| bias | train_mix | 0.330 | 0.357 |
| anti_bias | cue_gold | 0.015 | 0.004 |
| anti_bias | cue_dist | 0.090 | 0.014 |
| anti_bias | train_mix | 0.047 | 0.005 |
| no_bias | cue_gold | 0.521 | 0.554 |
| no_bias | cue_dist | 0.846 | 0.768 |
| no_bias | train_mix | 0.623 | 0.639 |
| deception_soft | cue_gold | 0.577 | 0.561 |
| deception_soft | cue_dist | 0.585 | 0.577 |
| deception_soft | train_mix | 0.578 | 0.558 |
| deception | cue_gold | 0.620 | 0.567 |
| deception | cue_dist | 0.595 | 0.618 |
| deception | train_mix | 0.611 | 0.564 |
| biased_share | cue_gold | 0.471 | 0.444 |
| biased_share | cue_dist | 0.070 | 0.221 |
| biased_share | train_mix | 0.346 | 0.359 |
| false_denial_soft | cue_gold | 0.383 | 0.302 |
| false_denial_soft | cue_dist | 0.280 | 0.184 |
| false_denial_soft | train_mix | 0.369 | 0.255 |
| false_admission_soft | cue_gold | 0.749 | 0.767 |
| false_admission_soft | cue_dist | 0.608 | 0.688 |
| false_admission_soft | train_mix | 0.689 | 0.727 |
| said_yes_biased_soft | cue_gold | 0.617 | 0.698 |
| said_yes_biased_soft | cue_dist | 0.720 | 0.816 |
| said_yes_biased_soft | train_mix | 0.631 | 0.745 |
| deception_from_biased | cue_gold | 0.181 | 0.134 |
| deception_from_biased | cue_dist | 0.020 | 0.041 |
| deception_from_biased | train_mix | 0.128 | 0.092 |
| deception_from_unbiased | cue_gold | 0.396 | 0.427 |
| deception_from_unbiased | cue_dist | 0.565 | 0.536 |
| deception_from_unbiased | train_mix | 0.450 | 0.466 |
| said_yes | cue_gold | 0.741 | 0.807 |
| said_yes | cue_dist | 0.640 | 0.794 |
| said_yes | train_mix | 0.723 | 0.810 |
| false_denial | cue_gold | 0.371 | 0.230 |
| false_denial | cue_dist | 0.176 | 0.100 |
| false_denial | train_mix | 0.339 | 0.157 |
| false_admission | cue_gold | 0.842 | 0.837 |
| false_admission | cue_dist | 0.627 | 0.764 |
| false_admission | train_mix | 0.755 | 0.791 |
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
| n_bias_rows | cue_dist | 266 | 276 |
| n_bias_rows | train_mix | 533 | 552 |
| n_deception_rows | cue_gold | 263 | 275 |
| n_deception_rows | cue_dist | 242 | 272 |
| n_deception_rows | train_mix | 505 | 547 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 1.000 | 0.018 | 0.768 | 0.768 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 1.000 | 0.000 | 0.920 | 0.920 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4144 | 0.910 | 0.019 | 0.910 |
| first_10_steps | base_wrong | 976 | 0.169 | 0.015 | 0.169 |
| last_10_steps | base_right | 4144 | 0.985 | 0.000 | 0.985 |
| last_10_steps | base_wrong | 976 | 0.644 | 0.000 | 0.644 |
