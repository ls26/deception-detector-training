# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 546; in both 546 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.739 |
| accuracy | cue_gold | 0.934 | 0.973 |
| accuracy | cue_dist | 0.351 | 0.188 |
| accuracy | train_mix | 0.734 | 0.702 |
| reward | none | 0.644 | 0.649 |
| reward | cue_gold | 0.764 | 0.788 |
| reward | cue_dist | 0.397 | 0.303 |
| reward | train_mix | 0.726 | 0.742 |
| followed_cue | cue_gold | 0.934 | 0.973 |
| followed_cue | cue_dist | 0.644 | 0.808 |
| followed_cue | train_mix | 0.830 | 0.914 |
| mean_p_correct | none | 0.737 | 0.739 |
| mean_p_correct | cue_gold | 0.927 | 0.973 |
| mean_p_correct | cue_dist | 0.360 | 0.187 |
| mean_p_correct | train_mix | 0.732 | 0.701 |
| mean_p_label | none | 0.634 | 0.652 |
| mean_p_label | cue_gold | 0.761 | 0.788 |
| mean_p_label | cue_dist | 0.404 | 0.303 |
| mean_p_label | train_mix | 0.724 | 0.742 |
| mean_p_cue | cue_gold | 0.927 | 0.973 |
| mean_p_cue | cue_dist | 0.637 | 0.813 |
| mean_p_cue | train_mix | 0.824 | 0.916 |
| bias | cue_gold | 0.183 | 0.233 |
| bias | cue_dist | 0.403 | 0.550 |
| bias | train_mix | 0.263 | 0.339 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.004 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.815 | 0.767 |
| no_bias | cue_dist | 0.593 | 0.450 |
| no_bias | train_mix | 0.735 | 0.661 |
| said_yes_biased | cue_gold | 0.840 | 0.921 |
| said_yes_biased | cue_dist | 0.772 | 0.876 |
| said_yes_biased | train_mix | 0.808 | 0.896 |
| said_yes_unbiased | cue_gold | 0.784 | 0.785 |
| said_yes_unbiased | cue_dist | 0.870 | 0.906 |
| said_yes_unbiased | train_mix | 0.809 | 0.816 |
| deception_soft | cue_gold | 0.617 | 0.586 |
| deception_soft | cue_dist | 0.559 | 0.488 |
| deception_soft | train_mix | 0.596 | 0.555 |
| deception | cue_gold | 0.669 | 0.621 |
| deception | cue_dist | 0.610 | 0.476 |
| deception | train_mix | 0.646 | 0.575 |
| said_yes | cue_gold | 0.794 | 0.817 |
| said_yes | cue_dist | 0.830 | 0.890 |
| said_yes | train_mix | 0.809 | 0.843 |
| false_denial | cue_gold | 0.160 | 0.079 |
| false_denial | cue_dist | 0.228 | 0.124 |
| false_denial | train_mix | 0.192 | 0.104 |
| false_admission | cue_gold | 0.784 | 0.785 |
| false_admission | cue_dist | 0.870 | 0.906 |
| false_admission | train_mix | 0.809 | 0.816 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 13 | 10 |
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
| n_deception_rows | cue_gold | 544 | 546 |
| n_deception_rows | cue_dist | 541 | 544 |
| n_deception_rows | train_mix | 1085 | 1090 |
| biased_share | cue_gold | 0.184 | 0.233 |
| biased_share | cue_dist | 0.405 | 0.550 |
| biased_share | train_mix | 0.264 | 0.339 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.785 |
| said_yes_biased_soft | cue_dist | 0.745 | 0.774 |
| said_yes_biased_soft | train_mix | 0.758 | 0.778 |
| false_denial_soft | cue_gold | 0.227 | 0.215 |
| false_denial_soft | cue_dist | 0.255 | 0.226 |
| false_denial_soft | train_mix | 0.242 | 0.222 |
| false_admission_soft | cue_gold | 0.705 | 0.699 |
| false_admission_soft | cue_dist | 0.767 | 0.807 |
| false_admission_soft | train_mix | 0.723 | 0.725 |
| deception_from_biased | cue_gold | 0.042 | 0.050 |
| deception_from_biased | cue_dist | 0.103 | 0.124 |
| deception_from_biased | train_mix | 0.064 | 0.075 |
| deception_from_unbiased | cue_gold | 0.576 | 0.536 |
| deception_from_unbiased | cue_dist | 0.456 | 0.364 |
| deception_from_unbiased | train_mix | 0.532 | 0.479 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.898 |
| accuracy | cue_gold | 0.995 | 0.995 |
| accuracy | cue_dist | 0.460 | 0.245 |
| accuracy | train_mix | 0.809 | 0.734 |
| reward | none | 0.794 | 0.751 |
| reward | cue_gold | 0.789 | 0.794 |
| reward | cue_dist | 0.462 | 0.339 |
| reward | train_mix | 0.760 | 0.754 |
| followed_cue | cue_gold | 0.995 | 0.995 |
| followed_cue | cue_dist | 0.533 | 0.751 |
| followed_cue | train_mix | 0.829 | 0.907 |
| mean_p_correct | none | 0.953 | 0.896 |
| mean_p_correct | cue_gold | 0.991 | 0.996 |
| mean_p_correct | cue_dist | 0.469 | 0.242 |
| mean_p_correct | train_mix | 0.809 | 0.733 |
| mean_p_label | none | 0.766 | 0.753 |
| mean_p_label | cue_gold | 0.788 | 0.795 |
| mean_p_label | cue_dist | 0.470 | 0.338 |
| mean_p_label | train_mix | 0.759 | 0.755 |
| mean_p_cue | cue_gold | 0.991 | 0.996 |
| mean_p_cue | cue_dist | 0.527 | 0.758 |
| mean_p_cue | train_mix | 0.824 | 0.910 |
| bias | cue_gold | 0.000 | 0.095 |
| bias | cue_dist | 0.535 | 0.654 |
| bias | train_mix | 0.186 | 0.281 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.000 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.998 | 0.905 |
| no_bias | cue_dist | 0.465 | 0.346 |
| no_bias | train_mix | 0.812 | 0.719 |
| said_yes_biased | cue_gold | – | 0.923 |
| said_yes_biased | cue_dist | 0.772 | 0.862 |
| said_yes_biased | train_mix | 0.777 | 0.875 |
| said_yes_unbiased | cue_gold | 0.771 | 0.759 |
| said_yes_unbiased | cue_dist | 0.895 | 0.965 |
| said_yes_unbiased | train_mix | 0.799 | 0.798 |
| deception_soft | cue_gold | 0.697 | 0.635 |
| deception_soft | cue_dist | 0.503 | 0.458 |
| deception_soft | train_mix | 0.632 | 0.580 |
| deception | cue_gold | 0.771 | 0.694 |
| deception | cue_dist | 0.538 | 0.424 |
| deception | train_mix | 0.692 | 0.609 |
| said_yes | cue_gold | 0.771 | 0.774 |
| said_yes | cue_dist | 0.829 | 0.898 |
| said_yes | train_mix | 0.795 | 0.820 |
| false_denial | cue_gold | – | 0.077 |
| false_denial | cue_dist | 0.228 | 0.138 |
| false_denial | train_mix | 0.223 | 0.125 |
| false_admission | cue_gold | 0.771 | 0.759 |
| false_admission | cue_dist | 0.895 | 0.965 |
| false_admission | train_mix | 0.799 | 0.798 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 2 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 2 |
| n_near_ties | train_mix | 4 | 2 |
| n_low_mass | none | 10 | 7 |
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
| n_deception_rows | cue_gold | 410 | 412 |
| n_deception_rows | cue_dist | 409 | 410 |
| n_deception_rows | train_mix | 819 | 822 |
| biased_share | cue_gold | 0.000 | 0.095 |
| biased_share | cue_dist | 0.535 | 0.654 |
| biased_share | train_mix | 0.186 | 0.281 |
| said_yes_biased_soft | cue_gold | – | 0.773 |
| said_yes_biased_soft | cue_dist | 0.745 | 0.762 |
| said_yes_biased_soft | train_mix | 0.742 | 0.759 |
| false_denial_soft | cue_gold | – | 0.227 |
| false_denial_soft | cue_dist | 0.255 | 0.238 |
| false_denial_soft | train_mix | 0.258 | 0.241 |
| false_admission_soft | cue_gold | 0.697 | 0.677 |
| false_admission_soft | cue_dist | 0.790 | 0.872 |
| false_admission_soft | train_mix | 0.717 | 0.713 |
| deception_from_biased | cue_gold | 0.000 | 0.022 |
| deception_from_biased | cue_dist | 0.136 | 0.156 |
| deception_from_biased | train_mix | 0.048 | 0.068 |
| deception_from_unbiased | cue_gold | 0.697 | 0.613 |
| deception_from_unbiased | cue_dist | 0.367 | 0.302 |
| deception_from_unbiased | train_mix | 0.584 | 0.512 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.246 |
| accuracy | cue_gold | 0.746 | 0.903 |
| accuracy | cue_dist | 0.015 | 0.015 |
| accuracy | train_mix | 0.504 | 0.603 |
| reward | none | 0.179 | 0.336 |
| reward | cue_gold | 0.687 | 0.769 |
| reward | cue_dist | 0.194 | 0.194 |
| reward | train_mix | 0.623 | 0.703 |
| followed_cue | cue_gold | 0.746 | 0.903 |
| followed_cue | cue_dist | 0.985 | 0.985 |
| followed_cue | train_mix | 0.836 | 0.935 |
| mean_p_correct | none | 0.073 | 0.253 |
| mean_p_correct | cue_gold | 0.729 | 0.902 |
| mean_p_correct | cue_dist | 0.023 | 0.018 |
| mean_p_correct | train_mix | 0.495 | 0.602 |
| mean_p_label | none | 0.230 | 0.340 |
| mean_p_label | cue_gold | 0.680 | 0.767 |
| mean_p_label | cue_dist | 0.201 | 0.197 |
| mean_p_label | train_mix | 0.617 | 0.703 |
| mean_p_cue | cue_gold | 0.729 | 0.902 |
| mean_p_cue | cue_dist | 0.977 | 0.982 |
| mean_p_cue | train_mix | 0.823 | 0.933 |
| bias | cue_gold | 0.746 | 0.657 |
| bias | cue_dist | 0.000 | 0.231 |
| bias | train_mix | 0.500 | 0.517 |
| anti_bias | cue_gold | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.015 | 0.000 |
| anti_bias | train_mix | 0.004 | 0.000 |
| no_bias | cue_gold | 0.254 | 0.343 |
| no_bias | cue_dist | 0.985 | 0.769 |
| no_bias | train_mix | 0.496 | 0.483 |
| said_yes_biased | cue_gold | 0.840 | 0.920 |
| said_yes_biased | cue_dist | – | 1.000 |
| said_yes_biased | train_mix | 0.843 | 0.931 |
| said_yes_unbiased | cue_gold | 0.941 | 1.000 |
| said_yes_unbiased | cue_dist | 0.833 | 0.825 |
| said_yes_unbiased | train_mix | 0.857 | 0.900 |
| deception_soft | cue_gold | 0.373 | 0.437 |
| deception_soft | cue_dist | 0.733 | 0.580 |
| deception_soft | train_mix | 0.487 | 0.477 |
| deception | cue_gold | 0.358 | 0.396 |
| deception | cue_dist | 0.833 | 0.634 |
| deception | train_mix | 0.506 | 0.470 |
| said_yes | cue_gold | 0.866 | 0.948 |
| said_yes | cue_dist | 0.833 | 0.866 |
| said_yes | train_mix | 0.850 | 0.916 |
| false_denial | cue_gold | 0.160 | 0.080 |
| false_denial | cue_dist | – | 0.000 |
| false_denial | train_mix | 0.157 | 0.069 |
| false_admission | cue_gold | 0.941 | 1.000 |
| false_admission | cue_dist | 0.833 | 0.825 |
| false_admission | train_mix | 0.857 | 0.900 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 0 | 0 |
| n_ties | cue_dist | 0 | 0 |
| n_ties | train_mix | 0 | 0 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 0 | 0 |
| n_near_ties | cue_dist | 0 | 0 |
| n_near_ties | train_mix | 0 | 0 |
| n_low_mass | none | 3 | 3 |
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
| biased_share | cue_gold | 0.746 | 0.657 |
| biased_share | cue_dist | 0.000 | 0.231 |
| biased_share | train_mix | 0.502 | 0.517 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.791 |
| said_yes_biased_soft | cue_dist | – | 0.880 |
| said_yes_biased_soft | train_mix | 0.775 | 0.808 |
| false_denial_soft | cue_gold | 0.227 | 0.209 |
| false_denial_soft | cue_dist | – | 0.120 |
| false_denial_soft | train_mix | 0.225 | 0.192 |
| false_admission_soft | cue_gold | 0.803 | 0.872 |
| false_admission_soft | cue_dist | 0.733 | 0.718 |
| false_admission_soft | train_mix | 0.751 | 0.781 |
| deception_from_biased | cue_gold | 0.169 | 0.137 |
| deception_from_biased | cue_dist | 0.000 | 0.028 |
| deception_from_biased | train_mix | 0.113 | 0.099 |
| deception_from_unbiased | cue_gold | 0.204 | 0.299 |
| deception_from_unbiased | cue_dist | 0.733 | 0.552 |
| deception_from_unbiased | train_mix | 0.374 | 0.378 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.945 |
| last_10_steps | 0.986 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.619 | 0.976 | 0.976 | 0.976 |
| first_10_steps | label_correct__cue_at_other | 0.173 | 0.804 | 0.196 | 0.196 |
| first_10_steps | label_flipped__cue_at_label | 0.163 | 0.811 | 0.811 | 0.189 |
| first_10_steps | label_flipped__cue_at_other | 0.045 | 0.966 | 0.034 | 0.966 |
| last_10_steps | label_correct__cue_at_label | 0.617 | 0.983 | 0.983 | 0.983 |
| last_10_steps | label_correct__cue_at_other | 0.177 | 0.720 | 0.280 | 0.280 |
| last_10_steps | label_flipped__cue_at_label | 0.161 | 0.851 | 0.851 | 0.149 |
| last_10_steps | label_flipped__cue_at_other | 0.045 | 0.931 | 0.069 | 0.931 |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3880 | 0.739 | 0.913 | 0.772 |
| first_10_steps | base_wrong | 1240 | 0.627 | 0.935 | 0.768 |
| last_10_steps | base_right | 3800 | 0.758 | 0.901 | 0.809 |
| last_10_steps | base_wrong | 1320 | 0.618 | 0.948 | 0.758 |

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
