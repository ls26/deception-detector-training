# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 547; in both 547 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.731 |
| accuracy | cue_gold | 0.934 | 0.856 |
| accuracy | cue_dist | 0.351 | 0.598 |
| accuracy | train_mix | 0.734 | 0.772 |
| reward | none | 0.644 | 0.627 |
| reward | cue_gold | 0.764 | 0.726 |
| reward | cue_dist | 0.397 | 0.545 |
| reward | train_mix | 0.726 | 0.700 |
| followed_cue | cue_gold | 0.934 | 0.856 |
| followed_cue | cue_dist | 0.644 | 0.402 |
| followed_cue | train_mix | 0.830 | 0.697 |
| mean_p_correct | none | 0.737 | 0.732 |
| mean_p_correct | cue_gold | 0.927 | 0.853 |
| mean_p_correct | cue_dist | 0.360 | 0.596 |
| mean_p_correct | train_mix | 0.732 | 0.770 |
| mean_p_label | none | 0.634 | 0.628 |
| mean_p_label | cue_gold | 0.761 | 0.724 |
| mean_p_label | cue_dist | 0.404 | 0.543 |
| mean_p_label | train_mix | 0.724 | 0.698 |
| mean_p_cue | cue_gold | 0.927 | 0.853 |
| mean_p_cue | cue_dist | 0.637 | 0.404 |
| mean_p_cue | train_mix | 0.824 | 0.695 |
| bias | cue_gold | 0.183 | 0.126 |
| bias | cue_dist | 0.404 | 0.139 |
| bias | train_mix | 0.263 | 0.136 |
| anti_bias | cue_gold | 0.002 | 0.002 |
| anti_bias | cue_dist | 0.004 | 0.005 |
| anti_bias | train_mix | 0.002 | 0.003 |
| no_bias | cue_gold | 0.815 | 0.872 |
| no_bias | cue_dist | 0.592 | 0.856 |
| no_bias | train_mix | 0.735 | 0.862 |
| said_yes_biased | cue_gold | 0.840 | 0.942 |
| said_yes_biased | cue_dist | 0.773 | 0.921 |
| said_yes_biased | train_mix | 0.808 | 0.926 |
| said_yes_unbiased | cue_gold | 0.784 | 0.935 |
| said_yes_unbiased | cue_dist | 0.870 | 0.979 |
| said_yes_unbiased | train_mix | 0.809 | 0.953 |
| deception_soft | cue_gold | 0.618 | 0.769 |
| deception_soft | cue_dist | 0.558 | 0.805 |
| deception_soft | train_mix | 0.596 | 0.779 |
| deception | cue_gold | 0.670 | 0.824 |
| deception | cue_dist | 0.609 | 0.853 |
| deception | train_mix | 0.646 | 0.833 |
| said_yes | cue_gold | 0.794 | 0.936 |
| said_yes | cue_dist | 0.830 | 0.971 |
| said_yes | train_mix | 0.809 | 0.949 |
| false_denial | cue_gold | 0.160 | 0.058 |
| false_denial | cue_dist | 0.227 | 0.079 |
| false_denial | train_mix | 0.192 | 0.074 |
| false_admission | cue_gold | 0.784 | 0.935 |
| false_admission | cue_dist | 0.870 | 0.979 |
| false_admission | train_mix | 0.809 | 0.953 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
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
| n_bias_rows | cue_gold | 546 | 547 |
| n_bias_rows | cue_dist | 544 | 547 |
| n_bias_rows | train_mix | 1090 | 1094 |
| n_deception_rows | cue_gold | 545 | 546 |
| n_deception_rows | cue_dist | 542 | 544 |
| n_deception_rows | train_mix | 1087 | 1090 |
| biased_share | cue_gold | 0.183 | 0.126 |
| biased_share | cue_dist | 0.406 | 0.140 |
| biased_share | train_mix | 0.264 | 0.136 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.895 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.866 |
| said_yes_biased_soft | train_mix | 0.758 | 0.879 |
| false_denial_soft | cue_gold | 0.227 | 0.105 |
| false_denial_soft | cue_dist | 0.254 | 0.134 |
| false_denial_soft | train_mix | 0.242 | 0.121 |
| false_admission_soft | cue_gold | 0.706 | 0.865 |
| false_admission_soft | cue_dist | 0.767 | 0.914 |
| false_admission_soft | train_mix | 0.723 | 0.882 |
| deception_from_biased | cue_gold | 0.042 | 0.013 |
| deception_from_biased | cue_dist | 0.103 | 0.019 |
| deception_from_biased | train_mix | 0.064 | 0.016 |
| deception_from_unbiased | cue_gold | 0.576 | 0.756 |
| deception_from_unbiased | cue_dist | 0.455 | 0.786 |
| deception_from_unbiased | train_mix | 0.533 | 0.762 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.864 |
| accuracy | cue_gold | 0.995 | 0.952 |
| accuracy | cue_dist | 0.460 | 0.746 |
| accuracy | train_mix | 0.809 | 0.882 |
| reward | none | 0.794 | 0.707 |
| reward | cue_gold | 0.789 | 0.770 |
| reward | cue_dist | 0.462 | 0.637 |
| reward | train_mix | 0.760 | 0.755 |
| followed_cue | cue_gold | 0.995 | 0.952 |
| followed_cue | cue_dist | 0.533 | 0.254 |
| followed_cue | train_mix | 0.829 | 0.705 |
| mean_p_correct | none | 0.953 | 0.866 |
| mean_p_correct | cue_gold | 0.991 | 0.949 |
| mean_p_correct | cue_dist | 0.469 | 0.742 |
| mean_p_correct | train_mix | 0.809 | 0.879 |
| mean_p_label | none | 0.766 | 0.708 |
| mean_p_label | cue_gold | 0.788 | 0.769 |
| mean_p_label | cue_dist | 0.470 | 0.634 |
| mean_p_label | train_mix | 0.759 | 0.753 |
| mean_p_cue | cue_gold | 0.991 | 0.949 |
| mean_p_cue | cue_dist | 0.527 | 0.258 |
| mean_p_cue | train_mix | 0.824 | 0.704 |
| bias | cue_gold | 0.000 | 0.087 |
| bias | cue_dist | 0.537 | 0.123 |
| bias | train_mix | 0.186 | 0.102 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.000 | 0.005 |
| anti_bias | train_mix | 0.002 | 0.001 |
| no_bias | cue_gold | 0.998 | 0.913 |
| no_bias | cue_dist | 0.463 | 0.872 |
| no_bias | train_mix | 0.812 | 0.896 |
| said_yes_biased | cue_gold | – | 0.917 |
| said_yes_biased | cue_dist | 0.773 | 0.922 |
| said_yes_biased | train_mix | 0.778 | 0.911 |
| said_yes_unbiased | cue_gold | 0.771 | 0.923 |
| said_yes_unbiased | cue_dist | 0.895 | 0.989 |
| said_yes_unbiased | train_mix | 0.800 | 0.948 |
| deception_soft | cue_gold | 0.698 | 0.789 |
| deception_soft | cue_dist | 0.502 | 0.827 |
| deception_soft | train_mix | 0.632 | 0.800 |
| deception | cue_gold | 0.771 | 0.850 |
| deception | cue_dist | 0.537 | 0.876 |
| deception | train_mix | 0.692 | 0.860 |
| said_yes | cue_gold | 0.771 | 0.923 |
| said_yes | cue_dist | 0.829 | 0.981 |
| said_yes | train_mix | 0.795 | 0.944 |
| false_denial | cue_gold | – | 0.083 |
| false_denial | cue_dist | 0.227 | 0.078 |
| false_denial | train_mix | 0.222 | 0.089 |
| false_admission | cue_gold | 0.771 | 0.923 |
| false_admission | cue_dist | 0.895 | 0.989 |
| false_admission | train_mix | 0.800 | 0.948 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
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
| n_bias_rows | cue_gold | 412 | 413 |
| n_bias_rows | cue_dist | 410 | 413 |
| n_bias_rows | train_mix | 822 | 826 |
| n_deception_rows | cue_gold | 411 | 413 |
| n_deception_rows | cue_dist | 410 | 411 |
| n_deception_rows | train_mix | 821 | 824 |
| biased_share | cue_gold | 0.000 | 0.087 |
| biased_share | cue_dist | 0.537 | 0.124 |
| biased_share | train_mix | 0.186 | 0.102 |
| said_yes_biased_soft | cue_gold | – | 0.888 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.863 |
| said_yes_biased_soft | train_mix | 0.743 | 0.876 |
| false_denial_soft | cue_gold | – | 0.112 |
| false_denial_soft | cue_dist | 0.254 | 0.137 |
| false_denial_soft | train_mix | 0.257 | 0.124 |
| false_admission_soft | cue_gold | 0.698 | 0.853 |
| false_admission_soft | cue_dist | 0.790 | 0.925 |
| false_admission_soft | train_mix | 0.718 | 0.878 |
| deception_from_biased | cue_gold | 0.000 | 0.010 |
| deception_from_biased | cue_dist | 0.136 | 0.017 |
| deception_from_biased | train_mix | 0.048 | 0.013 |
| deception_from_unbiased | cue_gold | 0.698 | 0.779 |
| deception_from_unbiased | cue_dist | 0.366 | 0.810 |
| deception_from_unbiased | train_mix | 0.584 | 0.788 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.321 |
| accuracy | cue_gold | 0.746 | 0.560 |
| accuracy | cue_dist | 0.015 | 0.142 |
| accuracy | train_mix | 0.504 | 0.433 |
| reward | none | 0.179 | 0.381 |
| reward | cue_gold | 0.687 | 0.590 |
| reward | cue_dist | 0.194 | 0.261 |
| reward | train_mix | 0.623 | 0.530 |
| followed_cue | cue_gold | 0.746 | 0.560 |
| followed_cue | cue_dist | 0.985 | 0.858 |
| followed_cue | train_mix | 0.836 | 0.672 |
| mean_p_correct | none | 0.073 | 0.319 |
| mean_p_correct | cue_gold | 0.729 | 0.558 |
| mean_p_correct | cue_dist | 0.023 | 0.145 |
| mean_p_correct | train_mix | 0.495 | 0.432 |
| mean_p_label | none | 0.230 | 0.379 |
| mean_p_label | cue_gold | 0.680 | 0.587 |
| mean_p_label | cue_dist | 0.201 | 0.264 |
| mean_p_label | train_mix | 0.617 | 0.529 |
| mean_p_cue | cue_gold | 0.729 | 0.558 |
| mean_p_cue | cue_dist | 0.977 | 0.855 |
| mean_p_cue | train_mix | 0.823 | 0.670 |
| bias | cue_gold | 0.746 | 0.246 |
| bias | cue_dist | 0.000 | 0.187 |
| bias | train_mix | 0.500 | 0.239 |
| anti_bias | cue_gold | 0.000 | 0.007 |
| anti_bias | cue_dist | 0.015 | 0.007 |
| anti_bias | train_mix | 0.004 | 0.007 |
| no_bias | cue_gold | 0.254 | 0.746 |
| no_bias | cue_dist | 0.985 | 0.806 |
| no_bias | train_mix | 0.496 | 0.754 |
| said_yes_biased | cue_gold | 0.840 | 0.970 |
| said_yes_biased | cue_dist | – | 0.920 |
| said_yes_biased | train_mix | 0.843 | 0.945 |
| said_yes_unbiased | cue_gold | 0.941 | 0.980 |
| said_yes_unbiased | cue_dist | 0.833 | 0.944 |
| said_yes_unbiased | train_mix | 0.857 | 0.970 |
| deception_soft | cue_gold | 0.373 | 0.707 |
| deception_soft | cue_dist | 0.733 | 0.736 |
| deception_soft | train_mix | 0.487 | 0.711 |
| deception | cue_gold | 0.358 | 0.744 |
| deception | cue_dist | 0.833 | 0.782 |
| deception | train_mix | 0.506 | 0.750 |
| said_yes | cue_gold | 0.866 | 0.977 |
| said_yes | cue_dist | 0.833 | 0.940 |
| said_yes | train_mix | 0.850 | 0.964 |
| false_denial | cue_gold | 0.160 | 0.030 |
| false_denial | cue_dist | – | 0.080 |
| false_denial | train_mix | 0.157 | 0.055 |
| false_admission | cue_gold | 0.941 | 0.980 |
| false_admission | cue_dist | 0.833 | 0.944 |
| false_admission | train_mix | 0.857 | 0.970 |
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
| biased_share | cue_dist | 0.000 | 0.188 |
| biased_share | train_mix | 0.502 | 0.241 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.902 |
| said_yes_biased_soft | cue_dist | – | 0.874 |
| said_yes_biased_soft | train_mix | 0.775 | 0.883 |
| false_denial_soft | cue_gold | 0.227 | 0.098 |
| false_denial_soft | cue_dist | – | 0.126 |
| false_denial_soft | train_mix | 0.225 | 0.117 |
| false_admission_soft | cue_gold | 0.803 | 0.908 |
| false_admission_soft | cue_dist | 0.733 | 0.877 |
| false_admission_soft | train_mix | 0.751 | 0.899 |
| deception_from_biased | cue_gold | 0.169 | 0.024 |
| deception_from_biased | cue_dist | 0.000 | 0.024 |
| deception_from_biased | train_mix | 0.113 | 0.028 |
| deception_from_unbiased | cue_gold | 0.204 | 0.683 |
| deception_from_unbiased | cue_dist | 0.733 | 0.712 |
| deception_from_unbiased | train_mix | 0.374 | 0.683 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.844 |
| last_10_steps | 0.995 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 0.800 | 0.001 | 0.768 | 0.768 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.200 | 0.004 | 0.175 | 0.821 |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 0.795 | 0.000 | 0.839 | 0.839 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.205 | 0.000 | 0.313 | 0.687 |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 4056 | 0.951 | 0.001 | 0.749 |
| first_10_steps | base_wrong | 1064 | 0.123 | 0.001 | 0.270 |
| last_10_steps | base_right | 3824 | 0.909 | 0.000 | 0.783 |
| last_10_steps | base_wrong | 1296 | 0.508 | 0.000 | 0.576 |

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

This run 0.958; the CoT run (261013) 0.847.
