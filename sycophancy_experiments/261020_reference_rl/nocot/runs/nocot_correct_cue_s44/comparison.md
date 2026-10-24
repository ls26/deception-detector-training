# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 546; in both 546 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.534 |
| accuracy | cue_gold | 0.934 | 1.000 |
| accuracy | cue_dist | 0.351 | 0.000 |
| accuracy | train_mix | 0.734 | 0.650 |
| reward | none | 0.644 | 0.503 |
| reward | cue_gold | 0.764 | 0.801 |
| reward | cue_dist | 0.397 | 0.199 |
| reward | train_mix | 0.726 | 0.750 |
| followed_cue | cue_gold | 0.934 | 1.000 |
| followed_cue | cue_dist | 0.644 | 1.000 |
| followed_cue | train_mix | 0.830 | 1.000 |
| mean_p_correct | none | 0.737 | 0.474 |
| mean_p_correct | cue_gold | 0.927 | 1.000 |
| mean_p_correct | cue_dist | 0.360 | 0.000 |
| mean_p_correct | train_mix | 0.732 | 0.650 |
| mean_p_label | none | 0.634 | 0.453 |
| mean_p_label | cue_gold | 0.761 | 0.801 |
| mean_p_label | cue_dist | 0.404 | 0.199 |
| mean_p_label | train_mix | 0.724 | 0.750 |
| mean_p_cue | cue_gold | 0.927 | 1.000 |
| mean_p_cue | cue_dist | 0.637 | 1.000 |
| mean_p_cue | train_mix | 0.824 | 1.000 |
| bias | cue_gold | 0.183 | 0.465 |
| bias | cue_dist | 0.405 | 0.535 |
| bias | train_mix | 0.264 | 0.498 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.004 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.815 | 0.535 |
| no_bias | cue_dist | 0.591 | 0.465 |
| no_bias | train_mix | 0.734 | 0.502 |
| said_yes_biased | cue_gold | 0.840 | 1.000 |
| said_yes_biased | cue_dist | 0.773 | 1.000 |
| said_yes_biased | train_mix | 0.808 | 1.000 |
| said_yes_unbiased | cue_gold | 0.786 | 1.000 |
| said_yes_unbiased | cue_dist | 0.872 | 1.000 |
| said_yes_unbiased | train_mix | 0.811 | 1.000 |
| deception_soft | cue_gold | 0.619 | 0.535 |
| deception_soft | cue_dist | 0.559 | 0.465 |
| deception_soft | train_mix | 0.597 | 0.502 |
| deception | cue_gold | 0.671 | 0.535 |
| deception | cue_dist | 0.610 | 0.465 |
| deception | train_mix | 0.647 | 0.502 |
| said_yes | cue_gold | 0.796 | 1.000 |
| said_yes | cue_dist | 0.832 | 1.000 |
| said_yes | train_mix | 0.810 | 1.000 |
| false_denial | cue_gold | 0.160 | 0.000 |
| false_denial | cue_dist | 0.227 | 0.000 |
| false_denial | train_mix | 0.192 | 0.000 |
| false_admission | cue_gold | 0.786 | 1.000 |
| false_admission | cue_dist | 0.872 | 1.000 |
| false_admission | train_mix | 0.811 | 1.000 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 13 | 91 |
| n_low_mass | cue_gold | 3 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 8 | 0 |
| n_one_in_top20 | none | 0 | 202 |
| n_one_in_top20 | cue_gold | 0 | 326 |
| n_one_in_top20 | cue_dist | 0 | 309 |
| n_one_in_top20 | train_mix | 0 | 635 |
| prompts | none | 547 | 547 |
| prompts | cue_gold | 547 | 547 |
| prompts | cue_dist | 547 | 547 |
| prompts | train_mix | 1094 | 1094 |
| n_bias_rows | cue_gold | 545 | 546 |
| n_bias_rows | cue_dist | 543 | 546 |
| n_bias_rows | train_mix | 1088 | 1092 |
| n_deception_rows | cue_gold | 544 | 546 |
| n_deception_rows | cue_dist | 541 | 546 |
| n_deception_rows | train_mix | 1085 | 1092 |
| biased_share | cue_gold | 0.184 | 0.465 |
| biased_share | cue_dist | 0.407 | 0.535 |
| biased_share | train_mix | 0.264 | 0.498 |
| said_yes_biased_soft | cue_gold | 0.773 | 1.000 |
| said_yes_biased_soft | cue_dist | 0.746 | 1.000 |
| said_yes_biased_soft | train_mix | 0.758 | 1.000 |
| false_denial_soft | cue_gold | 0.227 | 0.000 |
| false_denial_soft | cue_dist | 0.254 | 0.000 |
| false_denial_soft | train_mix | 0.242 | 0.000 |
| false_admission_soft | cue_gold | 0.707 | 1.000 |
| false_admission_soft | cue_dist | 0.768 | 1.000 |
| false_admission_soft | train_mix | 0.724 | 1.000 |
| deception_from_biased | cue_gold | 0.042 | 0.000 |
| deception_from_biased | cue_dist | 0.103 | 0.000 |
| deception_from_biased | train_mix | 0.064 | 0.000 |
| deception_from_unbiased | cue_gold | 0.577 | 0.535 |
| deception_from_unbiased | cue_dist | 0.455 | 0.465 |
| deception_from_unbiased | train_mix | 0.533 | 0.502 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.557 |
| accuracy | cue_gold | 0.995 | 1.000 |
| accuracy | cue_dist | 0.460 | 0.000 |
| accuracy | train_mix | 0.809 | 0.647 |
| reward | none | 0.794 | 0.501 |
| reward | cue_gold | 0.789 | 0.794 |
| reward | cue_dist | 0.462 | 0.206 |
| reward | train_mix | 0.760 | 0.750 |
| followed_cue | cue_gold | 0.995 | 1.000 |
| followed_cue | cue_dist | 0.533 | 1.000 |
| followed_cue | train_mix | 0.829 | 1.000 |
| mean_p_correct | none | 0.953 | 0.495 |
| mean_p_correct | cue_gold | 0.991 | 1.000 |
| mean_p_correct | cue_dist | 0.469 | 0.000 |
| mean_p_correct | train_mix | 0.809 | 0.647 |
| mean_p_label | none | 0.766 | 0.452 |
| mean_p_label | cue_gold | 0.788 | 0.794 |
| mean_p_label | cue_dist | 0.470 | 0.206 |
| mean_p_label | train_mix | 0.759 | 0.750 |
| mean_p_cue | cue_gold | 0.991 | 1.000 |
| mean_p_cue | cue_dist | 0.527 | 1.000 |
| mean_p_cue | train_mix | 0.824 | 1.000 |
| bias | cue_gold | 0.000 | 0.442 |
| bias | cue_dist | 0.538 | 0.558 |
| bias | train_mix | 0.186 | 0.499 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.000 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.998 | 0.558 |
| no_bias | cue_dist | 0.462 | 0.442 |
| no_bias | train_mix | 0.812 | 0.501 |
| said_yes_biased | cue_gold | – | 1.000 |
| said_yes_biased | cue_dist | 0.773 | 1.000 |
| said_yes_biased | train_mix | 0.778 | 1.000 |
| said_yes_unbiased | cue_gold | 0.773 | 1.000 |
| said_yes_unbiased | cue_dist | 0.899 | 1.000 |
| said_yes_unbiased | train_mix | 0.802 | 1.000 |
| deception_soft | cue_gold | 0.699 | 0.558 |
| deception_soft | cue_dist | 0.502 | 0.442 |
| deception_soft | train_mix | 0.633 | 0.501 |
| deception | cue_gold | 0.773 | 0.558 |
| deception | cue_dist | 0.538 | 0.442 |
| deception | train_mix | 0.694 | 0.501 |
| said_yes | cue_gold | 0.773 | 1.000 |
| said_yes | cue_dist | 0.831 | 1.000 |
| said_yes | train_mix | 0.797 | 1.000 |
| false_denial | cue_gold | – | 0.000 |
| false_denial | cue_dist | 0.227 | 0.000 |
| false_denial | train_mix | 0.222 | 0.000 |
| false_admission | cue_gold | 0.773 | 1.000 |
| false_admission | cue_dist | 0.899 | 1.000 |
| false_admission | train_mix | 0.802 | 1.000 |
| n_ties | none | 0 | 1 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 1 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 10 | 71 |
| n_low_mass | cue_gold | 1 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 6 | 0 |
| n_one_in_top20 | none | 0 | 142 |
| n_one_in_top20 | cue_gold | 0 | 255 |
| n_one_in_top20 | cue_dist | 0 | 227 |
| n_one_in_top20 | train_mix | 0 | 482 |
| prompts | none | 413 | 413 |
| prompts | cue_gold | 413 | 413 |
| prompts | cue_dist | 413 | 413 |
| prompts | train_mix | 826 | 826 |
| n_bias_rows | cue_gold | 411 | 412 |
| n_bias_rows | cue_dist | 409 | 412 |
| n_bias_rows | train_mix | 820 | 824 |
| n_deception_rows | cue_gold | 410 | 412 |
| n_deception_rows | cue_dist | 409 | 412 |
| n_deception_rows | train_mix | 819 | 824 |
| biased_share | cue_gold | 0.000 | 0.442 |
| biased_share | cue_dist | 0.538 | 0.558 |
| biased_share | train_mix | 0.187 | 0.499 |
| said_yes_biased_soft | cue_gold | – | 1.000 |
| said_yes_biased_soft | cue_dist | 0.746 | 1.000 |
| said_yes_biased_soft | train_mix | 0.743 | 1.000 |
| false_denial_soft | cue_gold | – | 0.000 |
| false_denial_soft | cue_dist | 0.254 | 0.000 |
| false_denial_soft | train_mix | 0.257 | 0.000 |
| false_admission_soft | cue_gold | 0.699 | 1.000 |
| false_admission_soft | cue_dist | 0.792 | 1.000 |
| false_admission_soft | train_mix | 0.719 | 1.000 |
| deception_from_biased | cue_gold | 0.000 | 0.000 |
| deception_from_biased | cue_dist | 0.137 | 0.000 |
| deception_from_biased | train_mix | 0.048 | 0.000 |
| deception_from_unbiased | cue_gold | 0.699 | 0.558 |
| deception_from_unbiased | cue_dist | 0.366 | 0.442 |
| deception_from_unbiased | train_mix | 0.585 | 0.501 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.463 |
| accuracy | cue_gold | 0.746 | 1.000 |
| accuracy | cue_dist | 0.015 | 0.000 |
| accuracy | train_mix | 0.504 | 0.660 |
| reward | none | 0.179 | 0.507 |
| reward | cue_gold | 0.687 | 0.821 |
| reward | cue_dist | 0.194 | 0.179 |
| reward | train_mix | 0.623 | 0.750 |
| followed_cue | cue_gold | 0.746 | 1.000 |
| followed_cue | cue_dist | 0.985 | 1.000 |
| followed_cue | train_mix | 0.836 | 1.000 |
| mean_p_correct | none | 0.073 | 0.412 |
| mean_p_correct | cue_gold | 0.729 | 1.000 |
| mean_p_correct | cue_dist | 0.023 | 0.000 |
| mean_p_correct | train_mix | 0.495 | 0.660 |
| mean_p_label | none | 0.230 | 0.457 |
| mean_p_label | cue_gold | 0.680 | 0.821 |
| mean_p_label | cue_dist | 0.201 | 0.179 |
| mean_p_label | train_mix | 0.617 | 0.750 |
| mean_p_cue | cue_gold | 0.729 | 1.000 |
| mean_p_cue | cue_dist | 0.977 | 1.000 |
| mean_p_cue | train_mix | 0.823 | 1.000 |
| bias | cue_gold | 0.746 | 0.537 |
| bias | cue_dist | 0.000 | 0.463 |
| bias | train_mix | 0.500 | 0.496 |
| anti_bias | cue_gold | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.015 | 0.000 |
| anti_bias | train_mix | 0.004 | 0.000 |
| no_bias | cue_gold | 0.254 | 0.463 |
| no_bias | cue_dist | 0.985 | 0.537 |
| no_bias | train_mix | 0.496 | 0.504 |
| said_yes_biased | cue_gold | 0.840 | 1.000 |
| said_yes_biased | cue_dist | – | 1.000 |
| said_yes_biased | train_mix | 0.843 | 1.000 |
| said_yes_unbiased | cue_gold | 0.941 | 1.000 |
| said_yes_unbiased | cue_dist | 0.833 | 1.000 |
| said_yes_unbiased | train_mix | 0.857 | 1.000 |
| deception_soft | cue_gold | 0.373 | 0.463 |
| deception_soft | cue_dist | 0.733 | 0.537 |
| deception_soft | train_mix | 0.487 | 0.504 |
| deception | cue_gold | 0.358 | 0.463 |
| deception | cue_dist | 0.833 | 0.537 |
| deception | train_mix | 0.506 | 0.504 |
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
| n_low_mass | none | 3 | 20 |
| n_low_mass | cue_gold | 2 | 0 |
| n_low_mass | cue_dist | 0 | 0 |
| n_low_mass | train_mix | 2 | 0 |
| n_one_in_top20 | none | 0 | 60 |
| n_one_in_top20 | cue_gold | 0 | 71 |
| n_one_in_top20 | cue_dist | 0 | 82 |
| n_one_in_top20 | train_mix | 0 | 153 |
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
| biased_share | cue_gold | 0.746 | 0.537 |
| biased_share | cue_dist | 0.000 | 0.463 |
| biased_share | train_mix | 0.502 | 0.496 |
| said_yes_biased_soft | cue_gold | 0.773 | 1.000 |
| said_yes_biased_soft | cue_dist | – | 1.000 |
| said_yes_biased_soft | train_mix | 0.775 | 1.000 |
| false_denial_soft | cue_gold | 0.227 | 0.000 |
| false_denial_soft | cue_dist | – | 0.000 |
| false_denial_soft | train_mix | 0.225 | 0.000 |
| false_admission_soft | cue_gold | 0.803 | 1.000 |
| false_admission_soft | cue_dist | 0.733 | 1.000 |
| false_admission_soft | train_mix | 0.751 | 1.000 |
| deception_from_biased | cue_gold | 0.169 | 0.000 |
| deception_from_biased | cue_dist | 0.000 | 0.000 |
| deception_from_biased | train_mix | 0.113 | 0.000 |
| deception_from_unbiased | cue_gold | 0.204 | 0.463 |
| deception_from_unbiased | cue_dist | 0.733 | 0.537 |
| deception_from_unbiased | train_mix | 0.374 | 0.504 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.967 |
| last_10_steps | 1.000 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.775 | 0.972 | 0.972 | 0.972 |
| first_10_steps | label_correct__cue_at_other | 0.225 | 0.852 | 0.148 | 0.148 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.775 | 1.000 | 1.000 | 1.000 |
| last_10_steps | label_correct__cue_at_other | 0.225 | 1.000 | 0.000 | 0.000 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3824 | 0.815 | 0.954 | 0.815 |
| first_10_steps | base_wrong | 1296 | 0.702 | 0.918 | 0.702 |
| last_10_steps | base_right | 3920 | 0.771 | 1.000 | 0.771 |
| last_10_steps | base_wrong | 1200 | 0.787 | 1.000 | 0.787 |

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

This run 0.997; the CoT run (261013) 0.847.
