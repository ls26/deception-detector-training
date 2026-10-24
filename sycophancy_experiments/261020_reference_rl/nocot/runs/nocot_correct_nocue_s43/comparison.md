# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 546; in both 546 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.770 |
| accuracy | cue_gold | 0.934 | 0.879 |
| accuracy | cue_dist | 0.351 | 0.574 |
| accuracy | train_mix | 0.734 | 0.774 |
| reward | none | 0.644 | 0.642 |
| reward | cue_gold | 0.764 | 0.724 |
| reward | cue_dist | 0.397 | 0.536 |
| reward | train_mix | 0.726 | 0.706 |
| followed_cue | cue_gold | 0.934 | 0.879 |
| followed_cue | cue_dist | 0.644 | 0.426 |
| followed_cue | train_mix | 0.830 | 0.718 |
| mean_p_correct | none | 0.737 | 0.768 |
| mean_p_correct | cue_gold | 0.927 | 0.879 |
| mean_p_correct | cue_dist | 0.360 | 0.574 |
| mean_p_correct | train_mix | 0.732 | 0.774 |
| mean_p_label | none | 0.634 | 0.640 |
| mean_p_label | cue_gold | 0.761 | 0.723 |
| mean_p_label | cue_dist | 0.404 | 0.534 |
| mean_p_label | train_mix | 0.724 | 0.705 |
| mean_p_cue | cue_gold | 0.927 | 0.879 |
| mean_p_cue | cue_dist | 0.637 | 0.425 |
| mean_p_cue | train_mix | 0.824 | 0.717 |
| bias | cue_gold | 0.183 | 0.112 |
| bias | cue_dist | 0.405 | 0.207 |
| bias | train_mix | 0.264 | 0.150 |
| anti_bias | cue_gold | 0.002 | 0.002 |
| anti_bias | cue_dist | 0.004 | 0.011 |
| anti_bias | train_mix | 0.002 | 0.004 |
| no_bias | cue_gold | 0.815 | 0.886 |
| no_bias | cue_dist | 0.591 | 0.782 |
| no_bias | train_mix | 0.734 | 0.846 |
| said_yes_biased | cue_gold | 0.840 | 0.967 |
| said_yes_biased | cue_dist | 0.773 | 0.912 |
| said_yes_biased | train_mix | 0.808 | 0.927 |
| said_yes_unbiased | cue_gold | 0.784 | 0.888 |
| said_yes_unbiased | cue_dist | 0.872 | 0.979 |
| said_yes_unbiased | train_mix | 0.811 | 0.914 |
| deception_soft | cue_gold | 0.618 | 0.720 |
| deception_soft | cue_dist | 0.559 | 0.743 |
| deception_soft | train_mix | 0.597 | 0.727 |
| deception | cue_gold | 0.669 | 0.793 |
| deception | cue_dist | 0.610 | 0.793 |
| deception | train_mix | 0.647 | 0.787 |
| said_yes | cue_gold | 0.794 | 0.897 |
| said_yes | cue_dist | 0.832 | 0.965 |
| said_yes | train_mix | 0.810 | 0.916 |
| false_denial | cue_gold | 0.160 | 0.033 |
| false_denial | cue_dist | 0.227 | 0.088 |
| false_denial | train_mix | 0.192 | 0.073 |
| false_admission | cue_gold | 0.784 | 0.888 |
| false_admission | cue_dist | 0.872 | 0.979 |
| false_admission | train_mix | 0.811 | 0.914 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 13 | 7 |
| n_low_mass | cue_gold | 3 | 1 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 8 | 1 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 547 | 547 |
| prompts | cue_gold | 547 | 547 |
| prompts | cue_dist | 547 | 547 |
| prompts | train_mix | 1094 | 1094 |
| n_bias_rows | cue_gold | 545 | 546 |
| n_bias_rows | cue_dist | 543 | 546 |
| n_bias_rows | train_mix | 1088 | 1092 |
| n_deception_rows | cue_gold | 544 | 545 |
| n_deception_rows | cue_dist | 541 | 540 |
| n_deception_rows | train_mix | 1085 | 1085 |
| biased_share | cue_gold | 0.184 | 0.112 |
| biased_share | cue_dist | 0.407 | 0.209 |
| biased_share | train_mix | 0.264 | 0.151 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.858 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.823 |
| said_yes_biased_soft | train_mix | 0.758 | 0.832 |
| false_denial_soft | cue_gold | 0.227 | 0.142 |
| false_denial_soft | cue_dist | 0.254 | 0.177 |
| false_denial_soft | train_mix | 0.242 | 0.168 |
| false_admission_soft | cue_gold | 0.706 | 0.793 |
| false_admission_soft | cue_dist | 0.768 | 0.892 |
| false_admission_soft | train_mix | 0.724 | 0.826 |
| deception_from_biased | cue_gold | 0.042 | 0.016 |
| deception_from_biased | cue_dist | 0.103 | 0.037 |
| deception_from_biased | train_mix | 0.064 | 0.025 |
| deception_from_unbiased | cue_gold | 0.576 | 0.705 |
| deception_from_unbiased | cue_dist | 0.455 | 0.706 |
| deception_from_unbiased | train_mix | 0.533 | 0.702 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.910 |
| accuracy | cue_gold | 0.995 | 0.966 |
| accuracy | cue_dist | 0.460 | 0.719 |
| accuracy | train_mix | 0.809 | 0.879 |
| reward | none | 0.794 | 0.729 |
| reward | cue_gold | 0.789 | 0.780 |
| reward | cue_dist | 0.462 | 0.634 |
| reward | train_mix | 0.760 | 0.769 |
| followed_cue | cue_gold | 0.995 | 0.966 |
| followed_cue | cue_dist | 0.533 | 0.281 |
| followed_cue | train_mix | 0.829 | 0.727 |
| mean_p_correct | none | 0.953 | 0.907 |
| mean_p_correct | cue_gold | 0.991 | 0.966 |
| mean_p_correct | cue_dist | 0.469 | 0.718 |
| mean_p_correct | train_mix | 0.809 | 0.879 |
| mean_p_label | none | 0.766 | 0.727 |
| mean_p_label | cue_gold | 0.788 | 0.778 |
| mean_p_label | cue_dist | 0.470 | 0.632 |
| mean_p_label | train_mix | 0.759 | 0.767 |
| mean_p_cue | cue_gold | 0.991 | 0.966 |
| mean_p_cue | cue_dist | 0.527 | 0.282 |
| mean_p_cue | train_mix | 0.824 | 0.726 |
| bias | cue_gold | 0.000 | 0.056 |
| bias | cue_dist | 0.537 | 0.201 |
| bias | train_mix | 0.186 | 0.115 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.000 | 0.010 |
| anti_bias | train_mix | 0.002 | 0.002 |
| no_bias | cue_gold | 0.998 | 0.944 |
| no_bias | cue_dist | 0.463 | 0.789 |
| no_bias | train_mix | 0.812 | 0.883 |
| said_yes_biased | cue_gold | – | 0.957 |
| said_yes_biased | cue_dist | 0.773 | 0.916 |
| said_yes_biased | train_mix | 0.778 | 0.916 |
| said_yes_unbiased | cue_gold | 0.771 | 0.879 |
| said_yes_unbiased | cue_dist | 0.895 | 0.991 |
| said_yes_unbiased | train_mix | 0.800 | 0.909 |
| deception_soft | cue_gold | 0.698 | 0.742 |
| deception_soft | cue_dist | 0.502 | 0.756 |
| deception_soft | train_mix | 0.632 | 0.743 |
| deception | cue_gold | 0.771 | 0.833 |
| deception | cue_dist | 0.537 | 0.807 |
| deception | train_mix | 0.692 | 0.814 |
| said_yes | cue_gold | 0.771 | 0.884 |
| said_yes | cue_dist | 0.829 | 0.976 |
| said_yes | train_mix | 0.795 | 0.910 |
| false_denial | cue_gold | – | 0.043 |
| false_denial | cue_dist | 0.227 | 0.084 |
| false_denial | train_mix | 0.222 | 0.084 |
| false_admission | cue_gold | 0.771 | 0.879 |
| false_admission | cue_dist | 0.895 | 0.991 |
| false_admission | train_mix | 0.800 | 0.909 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 10 | 5 |
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
| n_bias_rows | cue_gold | 412 | 413 |
| n_bias_rows | cue_dist | 410 | 413 |
| n_bias_rows | train_mix | 822 | 826 |
| n_deception_rows | cue_gold | 411 | 413 |
| n_deception_rows | cue_dist | 410 | 409 |
| n_deception_rows | train_mix | 821 | 822 |
| biased_share | cue_gold | 0.000 | 0.056 |
| biased_share | cue_dist | 0.537 | 0.203 |
| biased_share | train_mix | 0.186 | 0.115 |
| said_yes_biased_soft | cue_gold | – | 0.837 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.836 |
| said_yes_biased_soft | train_mix | 0.743 | 0.826 |
| false_denial_soft | cue_gold | – | 0.163 |
| false_denial_soft | cue_dist | 0.254 | 0.164 |
| false_denial_soft | train_mix | 0.257 | 0.174 |
| false_admission_soft | cue_gold | 0.698 | 0.777 |
| false_admission_soft | cue_dist | 0.790 | 0.906 |
| false_admission_soft | train_mix | 0.718 | 0.817 |
| deception_from_biased | cue_gold | 0.000 | 0.009 |
| deception_from_biased | cue_dist | 0.136 | 0.033 |
| deception_from_biased | train_mix | 0.048 | 0.020 |
| deception_from_unbiased | cue_gold | 0.698 | 0.733 |
| deception_from_unbiased | cue_dist | 0.366 | 0.722 |
| deception_from_unbiased | train_mix | 0.584 | 0.723 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.336 |
| accuracy | cue_gold | 0.746 | 0.612 |
| accuracy | cue_dist | 0.015 | 0.127 |
| accuracy | train_mix | 0.504 | 0.450 |
| reward | none | 0.179 | 0.373 |
| reward | cue_gold | 0.687 | 0.552 |
| reward | cue_dist | 0.194 | 0.231 |
| reward | train_mix | 0.623 | 0.513 |
| followed_cue | cue_gold | 0.746 | 0.612 |
| followed_cue | cue_dist | 0.985 | 0.873 |
| followed_cue | train_mix | 0.836 | 0.688 |
| mean_p_correct | none | 0.073 | 0.340 |
| mean_p_correct | cue_gold | 0.729 | 0.613 |
| mean_p_correct | cue_dist | 0.023 | 0.132 |
| mean_p_correct | train_mix | 0.495 | 0.452 |
| mean_p_label | none | 0.230 | 0.372 |
| mean_p_label | cue_gold | 0.680 | 0.552 |
| mean_p_label | cue_dist | 0.201 | 0.234 |
| mean_p_label | train_mix | 0.617 | 0.513 |
| mean_p_cue | cue_gold | 0.729 | 0.613 |
| mean_p_cue | cue_dist | 0.977 | 0.868 |
| mean_p_cue | train_mix | 0.823 | 0.687 |
| bias | cue_gold | 0.752 | 0.286 |
| bias | cue_dist | 0.000 | 0.226 |
| bias | train_mix | 0.504 | 0.259 |
| anti_bias | cue_gold | 0.000 | 0.008 |
| anti_bias | cue_dist | 0.015 | 0.015 |
| anti_bias | train_mix | 0.004 | 0.009 |
| no_bias | cue_gold | 0.248 | 0.707 |
| no_bias | cue_dist | 0.985 | 0.759 |
| no_bias | train_mix | 0.492 | 0.731 |
| said_yes_biased | cue_gold | 0.840 | 0.974 |
| said_yes_biased | cue_dist | – | 0.900 |
| said_yes_biased | train_mix | 0.843 | 0.942 |
| said_yes_unbiased | cue_gold | 0.939 | 0.926 |
| said_yes_unbiased | cue_dist | 0.840 | 0.941 |
| said_yes_unbiased | train_mix | 0.866 | 0.931 |
| deception_soft | cue_gold | 0.371 | 0.652 |
| deception_soft | cue_dist | 0.735 | 0.702 |
| deception_soft | train_mix | 0.487 | 0.676 |
| deception | cue_gold | 0.353 | 0.667 |
| deception | cue_dist | 0.840 | 0.748 |
| deception | train_mix | 0.508 | 0.702 |
| said_yes | cue_gold | 0.865 | 0.939 |
| said_yes | cue_dist | 0.840 | 0.931 |
| said_yes | train_mix | 0.855 | 0.934 |
| false_denial | cue_gold | 0.160 | 0.026 |
| false_denial | cue_dist | – | 0.100 |
| false_denial | train_mix | 0.157 | 0.058 |
| false_admission | cue_gold | 0.939 | 0.926 |
| false_admission | cue_dist | 0.840 | 0.941 |
| false_admission | train_mix | 0.866 | 0.931 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 0 | 0 |
| n_ties | cue_dist | 0 | 0 |
| n_ties | train_mix | 0 | 0 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 0 | 0 |
| n_near_ties | cue_dist | 0 | 0 |
| n_near_ties | train_mix | 0 | 0 |
| n_low_mass | none | 3 | 2 |
| n_low_mass | cue_gold | 2 | 1 |
| n_low_mass | cue_dist | 0 | 0 |
| n_low_mass | train_mix | 2 | 1 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 0 |
| n_one_in_top20 | cue_dist | 0 | 0 |
| n_one_in_top20 | train_mix | 0 | 0 |
| prompts | none | 134 | 134 |
| prompts | cue_gold | 134 | 134 |
| prompts | cue_dist | 134 | 134 |
| prompts | train_mix | 268 | 268 |
| n_bias_rows | cue_gold | 133 | 133 |
| n_bias_rows | cue_dist | 133 | 133 |
| n_bias_rows | train_mix | 266 | 266 |
| n_deception_rows | cue_gold | 133 | 132 |
| n_deception_rows | cue_dist | 131 | 131 |
| n_deception_rows | train_mix | 264 | 263 |
| biased_share | cue_gold | 0.752 | 0.288 |
| biased_share | cue_dist | 0.000 | 0.229 |
| biased_share | train_mix | 0.506 | 0.262 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.870 |
| said_yes_biased_soft | cue_dist | – | 0.788 |
| said_yes_biased_soft | train_mix | 0.775 | 0.842 |
| false_denial_soft | cue_gold | 0.227 | 0.130 |
| false_denial_soft | cue_dist | – | 0.212 |
| false_denial_soft | train_mix | 0.225 | 0.158 |
| false_admission_soft | cue_gold | 0.810 | 0.863 |
| false_admission_soft | cue_dist | 0.735 | 0.847 |
| false_admission_soft | train_mix | 0.755 | 0.860 |
| deception_from_biased | cue_gold | 0.171 | 0.037 |
| deception_from_biased | cue_dist | 0.000 | 0.048 |
| deception_from_biased | train_mix | 0.114 | 0.041 |
| deception_from_unbiased | cue_gold | 0.201 | 0.615 |
| deception_from_unbiased | cue_dist | 0.735 | 0.653 |
| deception_from_unbiased | train_mix | 0.373 | 0.635 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.819 |
| last_10_steps | 0.992 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 1.000 | 0.003 | 0.751 | 0.751 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 1.000 | 0.003 | 0.847 | 0.847 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3872 | 0.961 | 0.003 | 0.961 |
| first_10_steps | base_wrong | 1248 | 0.099 | 0.002 | 0.099 |
| last_10_steps | base_right | 3920 | 0.955 | 0.004 | 0.955 |
| last_10_steps | base_wrong | 1200 | 0.493 | 0.000 | 0.493 |

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

This run 0.966; the CoT run (261013) 0.847.
