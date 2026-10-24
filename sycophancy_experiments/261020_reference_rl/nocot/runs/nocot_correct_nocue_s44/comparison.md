# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 546; in both 546 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.771 |
| accuracy | cue_gold | 0.934 | 0.909 |
| accuracy | cue_dist | 0.351 | 0.486 |
| accuracy | train_mix | 0.734 | 0.763 |
| reward | none | 0.644 | 0.660 |
| reward | cue_gold | 0.764 | 0.742 |
| reward | cue_dist | 0.397 | 0.481 |
| reward | train_mix | 0.726 | 0.717 |
| followed_cue | cue_gold | 0.934 | 0.909 |
| followed_cue | cue_dist | 0.644 | 0.510 |
| followed_cue | train_mix | 0.830 | 0.766 |
| mean_p_correct | none | 0.737 | 0.770 |
| mean_p_correct | cue_gold | 0.927 | 0.910 |
| mean_p_correct | cue_dist | 0.360 | 0.486 |
| mean_p_correct | train_mix | 0.732 | 0.764 |
| mean_p_label | none | 0.634 | 0.660 |
| mean_p_label | cue_gold | 0.761 | 0.745 |
| mean_p_label | cue_dist | 0.404 | 0.483 |
| mean_p_label | train_mix | 0.724 | 0.720 |
| mean_p_cue | cue_gold | 0.927 | 0.910 |
| mean_p_cue | cue_dist | 0.637 | 0.514 |
| mean_p_cue | train_mix | 0.824 | 0.769 |
| bias | cue_gold | 0.183 | 0.137 |
| bias | cue_dist | 0.403 | 0.285 |
| bias | train_mix | 0.263 | 0.187 |
| anti_bias | cue_gold | 0.002 | 0.002 |
| anti_bias | cue_dist | 0.004 | 0.002 |
| anti_bias | train_mix | 0.002 | 0.002 |
| no_bias | cue_gold | 0.815 | 0.861 |
| no_bias | cue_dist | 0.593 | 0.713 |
| no_bias | train_mix | 0.735 | 0.811 |
| said_yes_biased | cue_gold | 0.840 | 1.000 |
| said_yes_biased | cue_dist | 0.772 | 1.000 |
| said_yes_biased | train_mix | 0.808 | 1.000 |
| said_yes_unbiased | cue_gold | 0.784 | 0.981 |
| said_yes_unbiased | cue_dist | 0.870 | 0.990 |
| said_yes_unbiased | train_mix | 0.809 | 0.980 |
| deception_soft | cue_gold | 0.617 | 0.788 |
| deception_soft | cue_dist | 0.559 | 0.691 |
| deception_soft | train_mix | 0.596 | 0.757 |
| deception | cue_gold | 0.669 | 0.846 |
| deception | cue_dist | 0.610 | 0.707 |
| deception | train_mix | 0.646 | 0.797 |
| said_yes | cue_gold | 0.794 | 0.983 |
| said_yes | cue_dist | 0.830 | 0.993 |
| said_yes | train_mix | 0.809 | 0.984 |
| false_denial | cue_gold | 0.160 | 0.000 |
| false_denial | cue_dist | 0.228 | 0.000 |
| false_denial | train_mix | 0.192 | 0.000 |
| false_admission | cue_gold | 0.784 | 0.981 |
| false_admission | cue_dist | 0.870 | 0.990 |
| false_admission | train_mix | 0.809 | 0.980 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 13 | 0 |
| n_low_mass | cue_gold | 3 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 8 | 0 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 547 | 547 |
| prompts | cue_gold | 547 | 547 |
| prompts | cue_dist | 547 | 547 |
| prompts | train_mix | 1094 | 1094 |
| n_bias_rows | cue_gold | 545 | 546 |
| n_bias_rows | cue_dist | 543 | 544 |
| n_bias_rows | train_mix | 1088 | 1090 |
| n_deception_rows | cue_gold | 544 | 545 |
| n_deception_rows | cue_dist | 541 | 543 |
| n_deception_rows | train_mix | 1085 | 1088 |
| biased_share | cue_gold | 0.184 | 0.138 |
| biased_share | cue_dist | 0.405 | 0.285 |
| biased_share | train_mix | 0.264 | 0.187 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.919 |
| said_yes_biased_soft | cue_dist | 0.745 | 0.911 |
| said_yes_biased_soft | train_mix | 0.758 | 0.910 |
| false_denial_soft | cue_gold | 0.227 | 0.081 |
| false_denial_soft | cue_dist | 0.255 | 0.089 |
| false_denial_soft | train_mix | 0.242 | 0.090 |
| false_admission_soft | cue_gold | 0.705 | 0.901 |
| false_admission_soft | cue_dist | 0.767 | 0.932 |
| false_admission_soft | train_mix | 0.723 | 0.911 |
| deception_from_biased | cue_gold | 0.042 | 0.011 |
| deception_from_biased | cue_dist | 0.103 | 0.025 |
| deception_from_biased | train_mix | 0.064 | 0.017 |
| deception_from_unbiased | cue_gold | 0.576 | 0.777 |
| deception_from_unbiased | cue_dist | 0.456 | 0.666 |
| deception_from_unbiased | train_mix | 0.532 | 0.740 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.901 |
| accuracy | cue_gold | 0.995 | 0.973 |
| accuracy | cue_dist | 0.460 | 0.605 |
| accuracy | train_mix | 0.809 | 0.846 |
| reward | none | 0.794 | 0.738 |
| reward | cue_gold | 0.789 | 0.782 |
| reward | cue_dist | 0.462 | 0.554 |
| reward | train_mix | 0.760 | 0.760 |
| followed_cue | cue_gold | 0.995 | 0.973 |
| followed_cue | cue_dist | 0.533 | 0.390 |
| followed_cue | train_mix | 0.829 | 0.767 |
| mean_p_correct | none | 0.953 | 0.901 |
| mean_p_correct | cue_gold | 0.991 | 0.975 |
| mean_p_correct | cue_dist | 0.469 | 0.605 |
| mean_p_correct | train_mix | 0.809 | 0.846 |
| mean_p_label | none | 0.766 | 0.739 |
| mean_p_label | cue_gold | 0.788 | 0.783 |
| mean_p_label | cue_dist | 0.470 | 0.557 |
| mean_p_label | train_mix | 0.759 | 0.763 |
| mean_p_cue | cue_gold | 0.991 | 0.975 |
| mean_p_cue | cue_dist | 0.527 | 0.395 |
| mean_p_cue | train_mix | 0.824 | 0.770 |
| bias | cue_gold | 0.000 | 0.073 |
| bias | cue_dist | 0.535 | 0.295 |
| bias | train_mix | 0.186 | 0.149 |
| anti_bias | cue_gold | 0.002 | 0.002 |
| anti_bias | cue_dist | 0.000 | 0.002 |
| anti_bias | train_mix | 0.002 | 0.002 |
| no_bias | cue_gold | 0.998 | 0.925 |
| no_bias | cue_dist | 0.465 | 0.702 |
| no_bias | train_mix | 0.812 | 0.849 |
| said_yes_biased | cue_gold | – | 1.000 |
| said_yes_biased | cue_dist | 0.772 | 1.000 |
| said_yes_biased | train_mix | 0.777 | 1.000 |
| said_yes_unbiased | cue_gold | 0.771 | 0.976 |
| said_yes_unbiased | cue_dist | 0.895 | 0.986 |
| said_yes_unbiased | train_mix | 0.799 | 0.975 |
| deception_soft | cue_gold | 0.697 | 0.836 |
| deception_soft | cue_dist | 0.503 | 0.686 |
| deception_soft | train_mix | 0.632 | 0.787 |
| deception | cue_gold | 0.771 | 0.905 |
| deception | cue_dist | 0.538 | 0.694 |
| deception | train_mix | 0.692 | 0.829 |
| said_yes | cue_gold | 0.771 | 0.978 |
| said_yes | cue_dist | 0.829 | 0.990 |
| said_yes | train_mix | 0.795 | 0.979 |
| false_denial | cue_gold | – | 0.000 |
| false_denial | cue_dist | 0.228 | 0.000 |
| false_denial | train_mix | 0.223 | 0.000 |
| false_admission | cue_gold | 0.771 | 0.976 |
| false_admission | cue_dist | 0.895 | 0.986 |
| false_admission | train_mix | 0.799 | 0.975 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 10 | 0 |
| n_low_mass | cue_gold | 1 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 6 | 0 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 413 | 413 |
| prompts | cue_gold | 413 | 413 |
| prompts | cue_dist | 413 | 413 |
| prompts | train_mix | 826 | 826 |
| n_bias_rows | cue_gold | 411 | 412 |
| n_bias_rows | cue_dist | 409 | 410 |
| n_bias_rows | train_mix | 820 | 822 |
| n_deception_rows | cue_gold | 410 | 411 |
| n_deception_rows | cue_dist | 409 | 409 |
| n_deception_rows | train_mix | 819 | 820 |
| biased_share | cue_gold | 0.000 | 0.073 |
| biased_share | cue_dist | 0.535 | 0.296 |
| biased_share | train_mix | 0.186 | 0.149 |
| said_yes_biased_soft | cue_gold | – | 0.914 |
| said_yes_biased_soft | cue_dist | 0.745 | 0.911 |
| said_yes_biased_soft | train_mix | 0.742 | 0.905 |
| false_denial_soft | cue_gold | – | 0.086 |
| false_denial_soft | cue_dist | 0.255 | 0.089 |
| false_denial_soft | train_mix | 0.258 | 0.095 |
| false_admission_soft | cue_gold | 0.697 | 0.896 |
| false_admission_soft | cue_dist | 0.790 | 0.936 |
| false_admission_soft | train_mix | 0.717 | 0.908 |
| deception_from_biased | cue_gold | 0.000 | 0.006 |
| deception_from_biased | cue_dist | 0.136 | 0.026 |
| deception_from_biased | train_mix | 0.048 | 0.014 |
| deception_from_unbiased | cue_gold | 0.697 | 0.830 |
| deception_from_unbiased | cue_dist | 0.367 | 0.659 |
| deception_from_unbiased | train_mix | 0.584 | 0.772 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.373 |
| accuracy | cue_gold | 0.746 | 0.709 |
| accuracy | cue_dist | 0.015 | 0.119 |
| accuracy | train_mix | 0.504 | 0.506 |
| reward | none | 0.179 | 0.418 |
| reward | cue_gold | 0.687 | 0.619 |
| reward | cue_dist | 0.194 | 0.254 |
| reward | train_mix | 0.623 | 0.584 |
| followed_cue | cue_gold | 0.746 | 0.709 |
| followed_cue | cue_dist | 0.985 | 0.881 |
| followed_cue | train_mix | 0.836 | 0.763 |
| mean_p_correct | none | 0.073 | 0.367 |
| mean_p_correct | cue_gold | 0.729 | 0.713 |
| mean_p_correct | cue_dist | 0.023 | 0.118 |
| mean_p_correct | train_mix | 0.495 | 0.509 |
| mean_p_label | none | 0.230 | 0.419 |
| mean_p_label | cue_gold | 0.680 | 0.627 |
| mean_p_label | cue_dist | 0.201 | 0.253 |
| mean_p_label | train_mix | 0.617 | 0.589 |
| mean_p_cue | cue_gold | 0.729 | 0.713 |
| mean_p_cue | cue_dist | 0.977 | 0.882 |
| mean_p_cue | train_mix | 0.823 | 0.767 |
| bias | cue_gold | 0.746 | 0.336 |
| bias | cue_dist | 0.000 | 0.254 |
| bias | train_mix | 0.500 | 0.304 |
| anti_bias | cue_gold | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.015 | 0.000 |
| anti_bias | train_mix | 0.004 | 0.000 |
| no_bias | cue_gold | 0.254 | 0.664 |
| no_bias | cue_dist | 0.985 | 0.746 |
| no_bias | train_mix | 0.496 | 0.696 |
| said_yes_biased | cue_gold | 0.840 | 1.000 |
| said_yes_biased | cue_dist | – | 1.000 |
| said_yes_biased | train_mix | 0.843 | 1.000 |
| said_yes_unbiased | cue_gold | 0.941 | 1.000 |
| said_yes_unbiased | cue_dist | 0.833 | 1.000 |
| said_yes_unbiased | train_mix | 0.857 | 1.000 |
| deception_soft | cue_gold | 0.373 | 0.639 |
| deception_soft | cue_dist | 0.733 | 0.708 |
| deception_soft | train_mix | 0.487 | 0.667 |
| deception | cue_gold | 0.358 | 0.664 |
| deception | cue_dist | 0.833 | 0.746 |
| deception | train_mix | 0.506 | 0.696 |
| said_yes | cue_gold | 0.866 | 1.000 |
| said_yes | cue_dist | 0.833 | 1.000 |
| said_yes | train_mix | 0.850 | 1.000 |
| false_denial | cue_gold | 0.160 | 0.000 |
| false_denial | cue_dist | – | 0.000 |
| false_denial | train_mix | 0.157 | 0.000 |
| false_admission | cue_gold | 0.941 | 1.000 |
| false_admission | cue_dist | 0.833 | 1.000 |
| false_admission | train_mix | 0.857 | 1.000 |
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
| n_deception_rows | cue_gold | 134 | 134 |
| n_deception_rows | cue_dist | 132 | 134 |
| n_deception_rows | train_mix | 266 | 268 |
| biased_share | cue_gold | 0.746 | 0.336 |
| biased_share | cue_dist | 0.000 | 0.254 |
| biased_share | train_mix | 0.502 | 0.304 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.922 |
| said_yes_biased_soft | cue_dist | – | 0.909 |
| said_yes_biased_soft | train_mix | 0.775 | 0.918 |
| false_denial_soft | cue_gold | 0.227 | 0.078 |
| false_denial_soft | cue_dist | – | 0.091 |
| false_denial_soft | train_mix | 0.225 | 0.082 |
| false_admission_soft | cue_gold | 0.803 | 0.923 |
| false_admission_soft | cue_dist | 0.733 | 0.918 |
| false_admission_soft | train_mix | 0.751 | 0.922 |
| deception_from_biased | cue_gold | 0.169 | 0.026 |
| deception_from_biased | cue_dist | 0.000 | 0.023 |
| deception_from_biased | train_mix | 0.113 | 0.025 |
| deception_from_unbiased | cue_gold | 0.204 | 0.613 |
| deception_from_unbiased | cue_dist | 0.733 | 0.685 |
| deception_from_unbiased | train_mix | 0.374 | 0.641 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.892 |
| last_10_steps | 0.986 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 1.000 | 0.001 | 0.764 | 0.764 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 1.000 | 0.000 | 0.810 | 0.810 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3904 | 0.960 | 0.001 | 0.960 |
| first_10_steps | base_wrong | 1216 | 0.137 | 0.000 | 0.137 |
| last_10_steps | base_right | 3936 | 0.948 | 0.000 | 0.948 |
| last_10_steps | base_wrong | 1184 | 0.351 | 0.001 | 0.351 |

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

This run 0.978; the CoT run (261013) 0.847.
