# 261013 noisy-label RL: results

Eval questions: 489. Questions with a no-cue answer (>= 5 of 8 no-cue samples agree): base 466, after_rl 460; in both 442 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.742 | 0.738 |
| accuracy | cue_gold | 0.852 | 0.975 |
| accuracy | cue_dist | 0.513 | 0.185 |
| accuracy | train_mix | 0.736 | 0.700 |
| reward | none | 0.647 | 0.655 |
| reward | cue_gold | 0.716 | 0.794 |
| reward | cue_dist | 0.498 | 0.314 |
| reward | train_mix | 0.690 | 0.752 |
| followed_cue | cue_gold | 0.852 | 0.975 |
| followed_cue | cue_dist | 0.460 | 0.815 |
| followed_cue | train_mix | 0.715 | 0.922 |
| bias | cue_gold | 0.105 | 0.179 |
| bias | cue_dist | 0.267 | 0.596 |
| bias | train_mix | 0.163 | 0.321 |
| anti_bias | cue_gold | 0.035 | 0.002 |
| anti_bias | cue_dist | 0.019 | 0.000 |
| anti_bias | train_mix | 0.030 | 0.001 |
| no_bias | cue_gold | 0.860 | 0.818 |
| no_bias | cue_dist | 0.713 | 0.404 |
| no_bias | train_mix | 0.807 | 0.678 |
| deception_soft | cue_gold | 0.542 | 0.557 |
| deception_soft | cue_dist | 0.596 | 0.466 |
| deception_soft | train_mix | 0.559 | 0.523 |
| deception | cue_gold | 0.543 | 0.635 |
| deception | cue_dist | 0.650 | 0.422 |
| deception | train_mix | 0.578 | 0.561 |
| said_yes | cue_gold | 0.571 | 0.783 |
| said_yes | cue_dist | 0.766 | 0.934 |
| said_yes | train_mix | 0.641 | 0.832 |
| false_denial | cue_gold | 0.372 | 0.088 |
| false_denial | cue_dist | 0.287 | 0.071 |
| false_denial | train_mix | 0.313 | 0.078 |
| false_admission | cue_gold | 0.564 | 0.755 |
| false_admission | cue_dist | 0.786 | 0.940 |
| false_admission | train_mix | 0.632 | 0.790 |
| parse_failed | none | 0.026 | 0.001 |
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
| n_bias_rows | cue_gold | 1303 | 1326 |
| n_bias_rows | cue_dist | 1291 | 1326 |
| n_bias_rows | train_mix | 2594 | 2652 |
| n_deception_rows | cue_gold | 1258 | 1323 |
| n_deception_rows | cue_dist | 1266 | 1326 |
| n_deception_rows | train_mix | 2524 | 2649 |

## Secondary: questions where the base model's consistent no-cue answer is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.897 | 0.882 |
| accuracy | cue_gold | 0.942 | 0.995 |
| accuracy | cue_dist | 0.626 | 0.235 |
| accuracy | train_mix | 0.832 | 0.733 |
| reward | none | 0.738 | 0.732 |
| reward | cue_gold | 0.768 | 0.808 |
| reward | cue_dist | 0.572 | 0.335 |
| reward | train_mix | 0.748 | 0.762 |
| followed_cue | cue_gold | 0.942 | 0.995 |
| followed_cue | cue_dist | 0.348 | 0.765 |
| followed_cue | train_mix | 0.735 | 0.916 |
| bias | cue_gold | 0.006 | 0.036 |
| bias | cue_dist | 0.325 | 0.707 |
| bias | train_mix | 0.117 | 0.270 |
| anti_bias | cue_gold | 0.040 | 0.003 |
| anti_bias | cue_dist | 0.001 | 0.000 |
| anti_bias | train_mix | 0.026 | 0.001 |
| no_bias | cue_gold | 0.955 | 0.961 |
| no_bias | cue_dist | 0.674 | 0.293 |
| no_bias | train_mix | 0.857 | 0.728 |
| deception_soft | cue_gold | 0.534 | 0.591 |
| deception_soft | cue_dist | 0.594 | 0.434 |
| deception_soft | train_mix | 0.553 | 0.534 |
| deception | cue_gold | 0.522 | 0.714 |
| deception | cue_dist | 0.655 | 0.340 |
| deception | train_mix | 0.566 | 0.581 |
| said_yes | cue_gold | 0.524 | 0.743 |
| said_yes | cue_dist | 0.793 | 0.942 |
| said_yes | train_mix | 0.618 | 0.810 |
| false_denial | cue_gold | 0.333 | 0.105 |
| false_denial | cue_dist | 0.288 | 0.074 |
| false_denial | train_mix | 0.286 | 0.077 |
| false_admission | cue_gold | 0.523 | 0.737 |
| false_admission | cue_dist | 0.832 | 0.981 |
| false_admission | train_mix | 0.605 | 0.768 |
| parse_failed | none | 0.022 | 0.001 |
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
| n_bias_rows | cue_gold | 1035 | 1050 |
| n_bias_rows | cue_dist | 1024 | 1050 |
| n_bias_rows | train_mix | 2059 | 2100 |
| n_deception_rows | cue_gold | 994 | 1047 |
| n_deception_rows | cue_dist | 1023 | 1050 |
| n_deception_rows | train_mix | 2017 | 2097 |

## Secondary: questions where the base model's consistent no-cue answer is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.173 | 0.214 |
| accuracy | cue_gold | 0.524 | 0.905 |
| accuracy | cue_dist | 0.098 | 0.000 |
| accuracy | train_mix | 0.384 | 0.579 |
| reward | none | 0.313 | 0.374 |
| reward | cue_gold | 0.524 | 0.743 |
| reward | cue_dist | 0.225 | 0.238 |
| reward | train_mix | 0.479 | 0.717 |
| followed_cue | cue_gold | 0.524 | 0.905 |
| followed_cue | cue_dist | 0.870 | 1.000 |
| followed_cue | train_mix | 0.640 | 0.948 |
| bias | cue_gold | 0.489 | 0.725 |
| bias | cue_dist | 0.045 | 0.174 |
| bias | train_mix | 0.338 | 0.514 |
| anti_bias | cue_gold | 0.015 | 0.000 |
| anti_bias | cue_dist | 0.090 | 0.000 |
| anti_bias | train_mix | 0.047 | 0.000 |
| no_bias | cue_gold | 0.496 | 0.275 |
| no_bias | cue_dist | 0.865 | 0.826 |
| no_bias | train_mix | 0.615 | 0.486 |
| deception_soft | cue_gold | 0.570 | 0.428 |
| deception_soft | cue_dist | 0.604 | 0.587 |
| deception_soft | train_mix | 0.581 | 0.482 |
| deception | cue_gold | 0.621 | 0.333 |
| deception | cue_dist | 0.630 | 0.736 |
| deception | train_mix | 0.624 | 0.485 |
| said_yes | cue_gold | 0.746 | 0.935 |
| said_yes | cue_dist | 0.654 | 0.902 |
| said_yes | train_mix | 0.732 | 0.916 |
| false_denial | cue_gold | 0.374 | 0.085 |
| false_denial | cue_dist | 0.250 | 0.021 |
| false_denial | train_mix | 0.349 | 0.081 |
| false_admission | cue_gold | 0.865 | 0.987 |
| false_admission | cue_dist | 0.649 | 0.886 |
| false_admission | train_mix | 0.776 | 0.912 |
| parse_failed | none | 0.039 | 0.001 |
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
| n_bias_rows | cue_gold | 268 | 276 |
| n_bias_rows | cue_dist | 267 | 276 |
| n_bias_rows | train_mix | 535 | 552 |
| n_deception_rows | cue_gold | 264 | 276 |
| n_deception_rows | cue_dist | 243 | 276 |
| n_deception_rows | train_mix | 507 | 552 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.630 | 0.915 | 0.915 | 0.915 |
| first_10_steps | label_correct__cue_at_other | 0.177 | 0.705 | 0.281 | 0.281 |
| first_10_steps | label_flipped__cue_at_label | 0.152 | 0.719 | 0.719 | 0.265 |
| first_10_steps | label_flipped__cue_at_other | 0.042 | 0.898 | 0.097 | 0.898 |
| last_10_steps | label_correct__cue_at_label | 0.616 | 0.987 | 0.987 | 0.987 |
| last_10_steps | label_correct__cue_at_other | 0.172 | 0.786 | 0.212 | 0.212 |
| last_10_steps | label_flipped__cue_at_label | 0.163 | 0.852 | 0.852 | 0.145 |
| last_10_steps | label_flipped__cue_at_other | 0.050 | 0.977 | 0.023 | 0.977 |

| window | group | rollouts | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4136 | 0.752 | 0.863 | 0.772 |
| first_10_steps | base_wrong | 984 | 0.501 | 0.784 | 0.602 |
| last_10_steps | base_right | 4088 | 0.730 | 0.923 | 0.797 |
| last_10_steps | base_wrong | 1032 | 0.663 | 0.958 | 0.733 |
