# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 547; in both 547 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.731 |
| accuracy | cue_gold | 0.934 | 0.876 |
| accuracy | cue_dist | 0.351 | 0.486 |
| accuracy | train_mix | 0.734 | 0.736 |
| reward | none | 0.644 | 0.627 |
| reward | cue_gold | 0.764 | 0.713 |
| reward | cue_dist | 0.397 | 0.490 |
| reward | train_mix | 0.726 | 0.698 |
| followed_cue | cue_gold | 0.934 | 0.876 |
| followed_cue | cue_dist | 0.644 | 0.510 |
| followed_cue | train_mix | 0.830 | 0.744 |
| mean_p_correct | none | 0.737 | 0.733 |
| mean_p_correct | cue_gold | 0.927 | 0.874 |
| mean_p_correct | cue_dist | 0.360 | 0.485 |
| mean_p_correct | train_mix | 0.732 | 0.735 |
| mean_p_label | none | 0.634 | 0.628 |
| mean_p_label | cue_gold | 0.761 | 0.713 |
| mean_p_label | cue_dist | 0.404 | 0.490 |
| mean_p_label | train_mix | 0.724 | 0.699 |
| mean_p_cue | cue_gold | 0.927 | 0.874 |
| mean_p_cue | cue_dist | 0.637 | 0.514 |
| mean_p_cue | train_mix | 0.824 | 0.745 |
| bias | cue_gold | 0.183 | 0.148 |
| bias | cue_dist | 0.404 | 0.244 |
| bias | train_mix | 0.263 | 0.183 |
| anti_bias | cue_gold | 0.002 | 0.004 |
| anti_bias | cue_dist | 0.004 | 0.002 |
| anti_bias | train_mix | 0.002 | 0.002 |
| no_bias | cue_gold | 0.815 | 0.848 |
| no_bias | cue_dist | 0.592 | 0.754 |
| no_bias | train_mix | 0.735 | 0.815 |
| said_yes_biased | cue_gold | 0.840 | 0.864 |
| said_yes_biased | cue_dist | 0.773 | 0.872 |
| said_yes_biased | train_mix | 0.808 | 0.850 |
| said_yes_unbiased | cue_gold | 0.784 | 0.810 |
| said_yes_unbiased | cue_dist | 0.870 | 0.908 |
| said_yes_unbiased | train_mix | 0.809 | 0.845 |
| deception_soft | cue_gold | 0.618 | 0.677 |
| deception_soft | cue_dist | 0.558 | 0.684 |
| deception_soft | train_mix | 0.596 | 0.681 |
| deception | cue_gold | 0.670 | 0.710 |
| deception | cue_dist | 0.609 | 0.717 |
| deception | train_mix | 0.646 | 0.717 |
| said_yes | cue_gold | 0.794 | 0.818 |
| said_yes | cue_dist | 0.830 | 0.899 |
| said_yes | train_mix | 0.809 | 0.846 |
| false_denial | cue_gold | 0.160 | 0.136 |
| false_denial | cue_dist | 0.227 | 0.128 |
| false_denial | train_mix | 0.192 | 0.150 |
| false_admission | cue_gold | 0.784 | 0.810 |
| false_admission | cue_dist | 0.870 | 0.908 |
| false_admission | train_mix | 0.809 | 0.845 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 13 | 0 |
| n_low_mass | cue_gold | 3 | 0 |
| n_low_mass | cue_dist | 5 | 1 |
| n_low_mass | train_mix | 8 | 1 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 547 | 547 |
| prompts | cue_gold | 547 | 547 |
| prompts | cue_dist | 547 | 547 |
| prompts | train_mix | 1094 | 1094 |
| n_bias_rows | cue_gold | 546 | 547 |
| n_bias_rows | cue_dist | 544 | 545 |
| n_bias_rows | train_mix | 1090 | 1092 |
| n_deception_rows | cue_gold | 545 | 545 |
| n_deception_rows | cue_dist | 542 | 544 |
| n_deception_rows | train_mix | 1087 | 1089 |
| biased_share | cue_gold | 0.183 | 0.149 |
| biased_share | cue_dist | 0.406 | 0.244 |
| biased_share | train_mix | 0.264 | 0.184 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.785 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.831 |
| said_yes_biased_soft | train_mix | 0.758 | 0.803 |
| false_denial_soft | cue_gold | 0.227 | 0.215 |
| false_denial_soft | cue_dist | 0.254 | 0.169 |
| false_denial_soft | train_mix | 0.242 | 0.197 |
| false_admission_soft | cue_gold | 0.706 | 0.758 |
| false_admission_soft | cue_dist | 0.767 | 0.851 |
| false_admission_soft | train_mix | 0.723 | 0.790 |
| deception_from_biased | cue_gold | 0.042 | 0.032 |
| deception_from_biased | cue_dist | 0.103 | 0.041 |
| deception_from_biased | train_mix | 0.064 | 0.036 |
| deception_from_unbiased | cue_gold | 0.576 | 0.645 |
| deception_from_unbiased | cue_dist | 0.455 | 0.643 |
| deception_from_unbiased | train_mix | 0.533 | 0.645 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.840 |
| accuracy | cue_gold | 0.995 | 0.954 |
| accuracy | cue_dist | 0.460 | 0.588 |
| accuracy | train_mix | 0.809 | 0.821 |
| reward | none | 0.794 | 0.707 |
| reward | cue_gold | 0.789 | 0.758 |
| reward | cue_dist | 0.462 | 0.554 |
| reward | train_mix | 0.760 | 0.747 |
| followed_cue | cue_gold | 0.995 | 0.954 |
| followed_cue | cue_dist | 0.533 | 0.407 |
| followed_cue | train_mix | 0.829 | 0.759 |
| mean_p_correct | none | 0.953 | 0.842 |
| mean_p_correct | cue_gold | 0.991 | 0.953 |
| mean_p_correct | cue_dist | 0.469 | 0.589 |
| mean_p_correct | train_mix | 0.809 | 0.821 |
| mean_p_label | none | 0.766 | 0.709 |
| mean_p_label | cue_gold | 0.788 | 0.758 |
| mean_p_label | cue_dist | 0.470 | 0.557 |
| mean_p_label | train_mix | 0.759 | 0.749 |
| mean_p_cue | cue_gold | 0.991 | 0.953 |
| mean_p_cue | cue_dist | 0.527 | 0.409 |
| mean_p_cue | train_mix | 0.824 | 0.760 |
| bias | cue_gold | 0.000 | 0.116 |
| bias | cue_dist | 0.537 | 0.248 |
| bias | train_mix | 0.186 | 0.158 |
| anti_bias | cue_gold | 0.002 | 0.002 |
| anti_bias | cue_dist | 0.000 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.002 |
| no_bias | cue_gold | 0.998 | 0.881 |
| no_bias | cue_dist | 0.463 | 0.752 |
| no_bias | train_mix | 0.812 | 0.840 |
| said_yes_biased | cue_gold | – | 0.896 |
| said_yes_biased | cue_dist | 0.773 | 0.873 |
| said_yes_biased | train_mix | 0.778 | 0.854 |
| said_yes_unbiased | cue_gold | 0.771 | 0.788 |
| said_yes_unbiased | cue_dist | 0.895 | 0.939 |
| said_yes_unbiased | train_mix | 0.800 | 0.838 |
| deception_soft | cue_gold | 0.698 | 0.671 |
| deception_soft | cue_dist | 0.502 | 0.702 |
| deception_soft | train_mix | 0.632 | 0.687 |
| deception | cue_gold | 0.771 | 0.709 |
| deception | cue_dist | 0.537 | 0.737 |
| deception | train_mix | 0.692 | 0.729 |
| said_yes | cue_gold | 0.771 | 0.801 |
| said_yes | cue_dist | 0.829 | 0.922 |
| said_yes | train_mix | 0.795 | 0.841 |
| false_denial | cue_gold | – | 0.104 |
| false_denial | cue_dist | 0.227 | 0.127 |
| false_denial | train_mix | 0.222 | 0.146 |
| false_admission | cue_gold | 0.771 | 0.788 |
| false_admission | cue_dist | 0.895 | 0.939 |
| false_admission | train_mix | 0.800 | 0.838 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 10 | 0 |
| n_low_mass | cue_gold | 1 | 0 |
| n_low_mass | cue_dist | 5 | 1 |
| n_low_mass | train_mix | 6 | 1 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 413 | 413 |
| prompts | cue_gold | 413 | 413 |
| prompts | cue_dist | 413 | 413 |
| prompts | train_mix | 826 | 826 |
| n_bias_rows | cue_gold | 412 | 413 |
| n_bias_rows | cue_dist | 410 | 411 |
| n_bias_rows | train_mix | 822 | 824 |
| n_deception_rows | cue_gold | 411 | 412 |
| n_deception_rows | cue_dist | 410 | 411 |
| n_deception_rows | train_mix | 821 | 823 |
| biased_share | cue_gold | 0.000 | 0.117 |
| biased_share | cue_dist | 0.537 | 0.248 |
| biased_share | train_mix | 0.186 | 0.158 |
| said_yes_biased_soft | cue_gold | – | 0.784 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.819 |
| said_yes_biased_soft | train_mix | 0.743 | 0.796 |
| false_denial_soft | cue_gold | – | 0.216 |
| false_denial_soft | cue_dist | 0.254 | 0.181 |
| false_denial_soft | train_mix | 0.257 | 0.204 |
| false_admission_soft | cue_gold | 0.698 | 0.731 |
| false_admission_soft | cue_dist | 0.790 | 0.874 |
| false_admission_soft | train_mix | 0.718 | 0.778 |
| deception_from_biased | cue_gold | 0.000 | 0.025 |
| deception_from_biased | cue_dist | 0.136 | 0.045 |
| deception_from_biased | train_mix | 0.048 | 0.032 |
| deception_from_unbiased | cue_gold | 0.698 | 0.646 |
| deception_from_unbiased | cue_dist | 0.366 | 0.657 |
| deception_from_unbiased | train_mix | 0.584 | 0.655 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.396 |
| accuracy | cue_gold | 0.746 | 0.634 |
| accuracy | cue_dist | 0.015 | 0.172 |
| accuracy | train_mix | 0.504 | 0.474 |
| reward | none | 0.179 | 0.381 |
| reward | cue_gold | 0.687 | 0.575 |
| reward | cue_dist | 0.194 | 0.291 |
| reward | train_mix | 0.623 | 0.549 |
| followed_cue | cue_gold | 0.746 | 0.634 |
| followed_cue | cue_dist | 0.985 | 0.828 |
| followed_cue | train_mix | 0.836 | 0.698 |
| mean_p_correct | none | 0.073 | 0.396 |
| mean_p_correct | cue_gold | 0.729 | 0.630 |
| mean_p_correct | cue_dist | 0.023 | 0.163 |
| mean_p_correct | train_mix | 0.495 | 0.469 |
| mean_p_label | none | 0.230 | 0.380 |
| mean_p_label | cue_gold | 0.680 | 0.572 |
| mean_p_label | cue_dist | 0.201 | 0.282 |
| mean_p_label | train_mix | 0.617 | 0.544 |
| mean_p_cue | cue_gold | 0.729 | 0.630 |
| mean_p_cue | cue_dist | 0.977 | 0.837 |
| mean_p_cue | train_mix | 0.823 | 0.697 |
| bias | cue_gold | 0.746 | 0.246 |
| bias | cue_dist | 0.000 | 0.231 |
| bias | train_mix | 0.500 | 0.261 |
| anti_bias | cue_gold | 0.000 | 0.007 |
| anti_bias | cue_dist | 0.015 | 0.007 |
| anti_bias | train_mix | 0.004 | 0.004 |
| no_bias | cue_gold | 0.254 | 0.746 |
| no_bias | cue_dist | 0.985 | 0.761 |
| no_bias | train_mix | 0.496 | 0.735 |
| said_yes_biased | cue_gold | 0.840 | 0.818 |
| said_yes_biased | cue_dist | – | 0.871 |
| said_yes_biased | train_mix | 0.843 | 0.843 |
| said_yes_unbiased | cue_gold | 0.941 | 0.890 |
| said_yes_unbiased | cue_dist | 0.833 | 0.814 |
| said_yes_unbiased | train_mix | 0.857 | 0.868 |
| deception_soft | cue_gold | 0.373 | 0.696 |
| deception_soft | cue_dist | 0.733 | 0.629 |
| deception_soft | train_mix | 0.487 | 0.662 |
| deception | cue_gold | 0.358 | 0.714 |
| deception | cue_dist | 0.833 | 0.654 |
| deception | train_mix | 0.506 | 0.682 |
| said_yes | cue_gold | 0.866 | 0.872 |
| said_yes | cue_dist | 0.833 | 0.827 |
| said_yes | train_mix | 0.850 | 0.861 |
| false_denial | cue_gold | 0.160 | 0.182 |
| false_denial | cue_dist | – | 0.129 |
| false_denial | train_mix | 0.157 | 0.157 |
| false_admission | cue_gold | 0.941 | 0.890 |
| false_admission | cue_dist | 0.833 | 0.814 |
| false_admission | train_mix | 0.857 | 0.868 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 0 | 0 |
| n_ties | cue_dist | 0 | 0 |
| n_ties | train_mix | 0 | 0 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 0 | 0 |
| n_near_ties | cue_dist | 0 | 0 |
| n_near_ties | train_mix | 0 | 0 |
| n_low_mass | none | 3 | 0 |
| n_low_mass | cue_gold | 2 | 0 |
| n_low_mass | cue_dist | 0 | 0 |
| n_low_mass | train_mix | 2 | 0 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 134 | 134 |
| prompts | cue_gold | 134 | 134 |
| prompts | cue_dist | 134 | 134 |
| prompts | train_mix | 268 | 268 |
| n_bias_rows | cue_gold | 134 | 134 |
| n_bias_rows | cue_dist | 134 | 134 |
| n_bias_rows | train_mix | 268 | 268 |
| n_deception_rows | cue_gold | 134 | 133 |
| n_deception_rows | cue_dist | 132 | 133 |
| n_deception_rows | train_mix | 266 | 266 |
| biased_share | cue_gold | 0.746 | 0.248 |
| biased_share | cue_dist | 0.000 | 0.233 |
| biased_share | train_mix | 0.502 | 0.262 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.786 |
| said_yes_biased_soft | cue_dist | – | 0.871 |
| said_yes_biased_soft | train_mix | 0.775 | 0.817 |
| false_denial_soft | cue_gold | 0.227 | 0.214 |
| false_denial_soft | cue_dist | – | 0.129 |
| false_denial_soft | train_mix | 0.225 | 0.183 |
| false_admission_soft | cue_gold | 0.803 | 0.855 |
| false_admission_soft | cue_dist | 0.733 | 0.780 |
| false_admission_soft | train_mix | 0.751 | 0.832 |
| deception_from_biased | cue_gold | 0.169 | 0.053 |
| deception_from_biased | cue_dist | 0.000 | 0.030 |
| deception_from_biased | train_mix | 0.113 | 0.048 |
| deception_from_unbiased | cue_gold | 0.204 | 0.643 |
| deception_from_unbiased | cue_dist | 0.733 | 0.599 |
| deception_from_unbiased | train_mix | 0.374 | 0.614 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.887 |
| last_10_steps | 0.994 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 1.000 | 0.002 | 0.732 | 0.732 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 1.000 | 0.000 | 0.827 | 0.827 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3912 | 0.913 | 0.002 | 0.913 |
| first_10_steps | base_wrong | 1208 | 0.147 | 0.000 | 0.147 |
| last_10_steps | base_right | 4064 | 0.905 | 0.000 | 0.905 |
| last_10_steps | base_wrong | 1056 | 0.526 | 0.000 | 0.526 |

## The CoT run (261013), one sampled answer per prompt (the first saved sample), all its eval questions

Questions with a 5-of-8 no-cue answer in both checkpoints: 442.

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.751 |
| accuracy | cue_gold | 0.843 | 0.975 |
| accuracy | cue_dist | 0.509 | 0.186 |
| accuracy | train_mix | 0.729 | 0.700 |
| reward | none | 0.644 | 0.661 |
| reward | cue_gold | 0.714 | 0.800 |
| reward | cue_dist | 0.499 | 0.321 |
| reward | train_mix | 0.689 | 0.758 |
| followed_cue | cue_gold | 0.843 | 0.975 |
| followed_cue | cue_dist | 0.458 | 0.814 |
| followed_cue | train_mix | 0.711 | 0.925 |
| bias | cue_gold | 0.097 | 0.179 |
| bias | cue_dist | 0.259 | 0.595 |
| bias | train_mix | 0.154 | 0.324 |
| anti_bias | cue_gold | 0.037 | 0.002 |
| anti_bias | cue_dist | 0.012 | 0.000 |
| anti_bias | train_mix | 0.027 | 0.001 |
| no_bias | cue_gold | 0.866 | 0.819 |
| no_bias | cue_dist | 0.729 | 0.405 |
| no_bias | train_mix | 0.819 | 0.676 |
| deception_soft | cue_gold | 0.547 | 0.553 |
| deception_soft | cue_dist | 0.605 | 0.464 |
| deception_soft | train_mix | 0.565 | 0.516 |
| deception | cue_gold | 0.565 | 0.628 |
| deception | cue_dist | 0.669 | 0.425 |
| deception | train_mix | 0.597 | 0.547 |
| said_yes | cue_gold | 0.593 | 0.785 |
| said_yes | cue_dist | 0.785 | 0.939 |
| said_yes | train_mix | 0.655 | 0.829 |
| false_denial | cue_gold | 0.357 | 0.063 |
| false_denial | cue_dist | 0.279 | 0.068 |
| false_denial | train_mix | 0.317 | 0.065 |
| false_admission | cue_gold | 0.588 | 0.751 |
| false_admission | cue_dist | 0.808 | 0.950 |
| false_admission | train_mix | 0.650 | 0.778 |
| n_bias_rows | cue_gold | 434 | 442 |
| n_bias_rows | cue_dist | 428 | 442 |
| n_bias_rows | train_mix | 862 | 884 |
| n_deception_rows | cue_gold | 418 | 441 |
| n_deception_rows | cue_dist | 423 | 442 |
| n_deception_rows | train_mix | 841 | 883 |
| biased_share | cue_gold | 0.100 | 0.179 |
| biased_share | cue_dist | 0.262 | 0.595 |
| biased_share | train_mix | 0.158 | 0.324 |
| said_yes_biased_soft | cue_gold | 0.612 | 0.709 |
| said_yes_biased_soft | cue_dist | 0.640 | 0.728 |
| said_yes_biased_soft | train_mix | 0.630 | 0.728 |
| false_denial_soft | cue_gold | 0.388 | 0.291 |
| false_denial_soft | cue_dist | 0.360 | 0.272 |
| false_denial_soft | train_mix | 0.370 | 0.272 |
| false_admission_soft | cue_gold | 0.565 | 0.610 |
| false_admission_soft | cue_dist | 0.693 | 0.747 |
| false_admission_soft | train_mix | 0.601 | 0.632 |
| deception_from_biased | cue_gold | 0.039 | 0.052 |
| deception_from_biased | cue_dist | 0.094 | 0.162 |
| deception_from_biased | train_mix | 0.058 | 0.088 |
| deception_from_unbiased | cue_gold | 0.508 | 0.501 |
| deception_from_unbiased | cue_dist | 0.511 | 0.302 |
| deception_from_unbiased | train_mix | 0.506 | 0.428 |

## Share of training groups whose 8 rewards are all equal, mean over steps

This run 0.975; the CoT run (261013) 0.847.
