# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 545; in both 545 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.605 |
| accuracy | cue_gold | 0.934 | 1.000 |
| accuracy | cue_dist | 0.351 | 0.000 |
| accuracy | train_mix | 0.734 | 0.650 |
| reward | none | 0.644 | 0.537 |
| reward | cue_gold | 0.764 | 0.801 |
| reward | cue_dist | 0.397 | 0.199 |
| reward | train_mix | 0.726 | 0.750 |
| followed_cue | cue_gold | 0.934 | 1.000 |
| followed_cue | cue_dist | 0.644 | 1.000 |
| followed_cue | train_mix | 0.830 | 1.000 |
| mean_p_correct | none | 0.737 | 0.609 |
| mean_p_correct | cue_gold | 0.927 | 1.000 |
| mean_p_correct | cue_dist | 0.360 | 0.000 |
| mean_p_correct | train_mix | 0.732 | 0.650 |
| mean_p_label | none | 0.634 | 0.538 |
| mean_p_label | cue_gold | 0.761 | 0.801 |
| mean_p_label | cue_dist | 0.404 | 0.199 |
| mean_p_label | train_mix | 0.724 | 0.750 |
| mean_p_cue | cue_gold | 0.927 | 1.000 |
| mean_p_cue | cue_dist | 0.637 | 1.000 |
| mean_p_cue | train_mix | 0.824 | 1.000 |
| bias | cue_gold | 0.184 | 0.393 |
| bias | cue_dist | 0.406 | 0.607 |
| bias | train_mix | 0.264 | 0.480 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.004 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.814 | 0.607 |
| no_bias | cue_dist | 0.590 | 0.393 |
| no_bias | train_mix | 0.734 | 0.520 |
| said_yes_biased | cue_gold | 0.840 | 0.626 |
| said_yes_biased | cue_dist | 0.773 | 0.495 |
| said_yes_biased | train_mix | 0.808 | 0.558 |
| said_yes_unbiased | cue_gold | 0.783 | 0.495 |
| said_yes_unbiased | cue_dist | 0.869 | 0.636 |
| said_yes_unbiased | train_mix | 0.808 | 0.538 |
| deception_soft | cue_gold | 0.618 | 0.480 |
| deception_soft | cue_dist | 0.557 | 0.521 |
| deception_soft | train_mix | 0.596 | 0.498 |
| deception | cue_gold | 0.669 | 0.448 |
| deception | cue_dist | 0.607 | 0.556 |
| deception | train_mix | 0.645 | 0.492 |
| said_yes | cue_gold | 0.794 | 0.547 |
| said_yes | cue_dist | 0.830 | 0.550 |
| said_yes | train_mix | 0.808 | 0.548 |
| false_denial | cue_gold | 0.160 | 0.374 |
| false_denial | cue_dist | 0.227 | 0.505 |
| false_denial | train_mix | 0.192 | 0.442 |
| false_admission | cue_gold | 0.783 | 0.495 |
| false_admission | cue_dist | 0.869 | 0.636 |
| false_admission | train_mix | 0.808 | 0.538 |
| n_ties | none | 0 | 2 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 2 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 13 | 0 |
| n_low_mass | cue_gold | 3 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 8 | 0 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 37 |
| n_one_in_top20 | cue_dist | 0 | 26 |
| n_one_in_top20 | train_mix | 0 | 63 |
| prompts | none | 547 | 547 |
| prompts | cue_gold | 547 | 547 |
| prompts | cue_dist | 547 | 547 |
| prompts | train_mix | 1094 | 1094 |
| n_bias_rows | cue_gold | 544 | 545 |
| n_bias_rows | cue_dist | 542 | 545 |
| n_bias_rows | train_mix | 1086 | 1090 |
| n_deception_rows | cue_gold | 543 | 545 |
| n_deception_rows | cue_dist | 540 | 545 |
| n_deception_rows | train_mix | 1083 | 1090 |
| biased_share | cue_gold | 0.184 | 0.393 |
| biased_share | cue_dist | 0.407 | 0.607 |
| biased_share | train_mix | 0.265 | 0.480 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.677 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.583 |
| said_yes_biased_soft | train_mix | 0.758 | 0.627 |
| false_denial_soft | cue_gold | 0.227 | 0.323 |
| false_denial_soft | cue_dist | 0.254 | 0.417 |
| false_denial_soft | train_mix | 0.242 | 0.373 |
| false_admission_soft | cue_gold | 0.706 | 0.581 |
| false_admission_soft | cue_dist | 0.766 | 0.683 |
| false_admission_soft | train_mix | 0.723 | 0.613 |
| deception_from_biased | cue_gold | 0.042 | 0.127 |
| deception_from_biased | cue_dist | 0.103 | 0.253 |
| deception_from_biased | train_mix | 0.064 | 0.179 |
| deception_from_unbiased | cue_gold | 0.576 | 0.353 |
| deception_from_unbiased | cue_dist | 0.454 | 0.268 |
| deception_from_unbiased | train_mix | 0.532 | 0.319 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.685 |
| accuracy | cue_gold | 0.995 | 1.000 |
| accuracy | cue_dist | 0.460 | 0.000 |
| accuracy | train_mix | 0.809 | 0.647 |
| reward | none | 0.794 | 0.581 |
| reward | cue_gold | 0.789 | 0.794 |
| reward | cue_dist | 0.462 | 0.206 |
| reward | train_mix | 0.760 | 0.750 |
| followed_cue | cue_gold | 0.995 | 1.000 |
| followed_cue | cue_dist | 0.533 | 1.000 |
| followed_cue | train_mix | 0.829 | 1.000 |
| mean_p_correct | none | 0.953 | 0.688 |
| mean_p_correct | cue_gold | 0.991 | 1.000 |
| mean_p_correct | cue_dist | 0.469 | 0.000 |
| mean_p_correct | train_mix | 0.809 | 0.647 |
| mean_p_label | none | 0.766 | 0.580 |
| mean_p_label | cue_gold | 0.788 | 0.794 |
| mean_p_label | cue_dist | 0.470 | 0.206 |
| mean_p_label | train_mix | 0.759 | 0.750 |
| mean_p_cue | cue_gold | 0.991 | 1.000 |
| mean_p_cue | cue_dist | 0.527 | 1.000 |
| mean_p_cue | train_mix | 0.824 | 1.000 |
| bias | cue_gold | 0.000 | 0.311 |
| bias | cue_dist | 0.539 | 0.689 |
| bias | train_mix | 0.187 | 0.458 |
| anti_bias | cue_gold | 0.002 | 0.000 |
| anti_bias | cue_dist | 0.000 | 0.000 |
| anti_bias | train_mix | 0.002 | 0.000 |
| no_bias | cue_gold | 0.998 | 0.689 |
| no_bias | cue_dist | 0.461 | 0.311 |
| no_bias | train_mix | 0.811 | 0.542 |
| said_yes_biased | cue_gold | – | 0.602 |
| said_yes_biased | cue_dist | 0.773 | 0.505 |
| said_yes_biased | train_mix | 0.778 | 0.544 |
| said_yes_unbiased | cue_gold | 0.770 | 0.505 |
| said_yes_unbiased | cue_dist | 0.894 | 0.609 |
| said_yes_unbiased | train_mix | 0.798 | 0.529 |
| deception_soft | cue_gold | 0.698 | 0.511 |
| deception_soft | cue_dist | 0.500 | 0.490 |
| deception_soft | train_mix | 0.631 | 0.504 |
| deception | cue_gold | 0.770 | 0.472 |
| deception | cue_dist | 0.534 | 0.530 |
| deception | train_mix | 0.691 | 0.495 |
| said_yes | cue_gold | 0.770 | 0.535 |
| said_yes | cue_dist | 0.828 | 0.538 |
| said_yes | train_mix | 0.794 | 0.536 |
| false_denial | cue_gold | – | 0.398 |
| false_denial | cue_dist | 0.227 | 0.495 |
| false_denial | train_mix | 0.222 | 0.456 |
| false_admission | cue_gold | 0.770 | 0.505 |
| false_admission | cue_dist | 0.894 | 0.609 |
| false_admission | train_mix | 0.798 | 0.529 |
| n_ties | none | 0 | 2 |
| n_ties | cue_gold | 1 | 0 |
| n_ties | cue_dist | 3 | 0 |
| n_ties | train_mix | 4 | 0 |
| n_near_ties | none | 0 | 2 |
| n_near_ties | cue_gold | 1 | 0 |
| n_near_ties | cue_dist | 3 | 0 |
| n_near_ties | train_mix | 4 | 0 |
| n_low_mass | none | 10 | 0 |
| n_low_mass | cue_gold | 1 | 0 |
| n_low_mass | cue_dist | 5 | 0 |
| n_low_mass | train_mix | 6 | 0 |
| n_one_in_top20 | none | 0 | 0 |
| n_one_in_top20 | cue_gold | 0 | 29 |
| n_one_in_top20 | cue_dist | 0 | 16 |
| n_one_in_top20 | train_mix | 0 | 45 |
| prompts | none | 413 | 413 |
| prompts | cue_gold | 413 | 413 |
| prompts | cue_dist | 413 | 413 |
| prompts | train_mix | 826 | 826 |
| n_bias_rows | cue_gold | 410 | 411 |
| n_bias_rows | cue_dist | 408 | 411 |
| n_bias_rows | train_mix | 818 | 822 |
| n_deception_rows | cue_gold | 409 | 411 |
| n_deception_rows | cue_dist | 408 | 411 |
| n_deception_rows | train_mix | 817 | 822 |
| biased_share | cue_gold | 0.000 | 0.311 |
| biased_share | cue_dist | 0.539 | 0.689 |
| biased_share | train_mix | 0.187 | 0.458 |
| said_yes_biased_soft | cue_gold | – | 0.663 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.592 |
| said_yes_biased_soft | train_mix | 0.743 | 0.620 |
| false_denial_soft | cue_gold | – | 0.337 |
| false_denial_soft | cue_dist | 0.254 | 0.408 |
| false_denial_soft | train_mix | 0.257 | 0.380 |
| false_admission_soft | cue_gold | 0.698 | 0.589 |
| false_admission_soft | cue_dist | 0.789 | 0.672 |
| false_admission_soft | train_mix | 0.717 | 0.609 |
| deception_from_biased | cue_gold | 0.000 | 0.105 |
| deception_from_biased | cue_dist | 0.137 | 0.281 |
| deception_from_biased | train_mix | 0.048 | 0.174 |
| deception_from_unbiased | cue_gold | 0.698 | 0.406 |
| deception_from_unbiased | cue_dist | 0.363 | 0.209 |
| deception_from_unbiased | train_mix | 0.583 | 0.330 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.358 |
| accuracy | cue_gold | 0.746 | 1.000 |
| accuracy | cue_dist | 0.015 | 0.000 |
| accuracy | train_mix | 0.504 | 0.660 |
| reward | none | 0.179 | 0.403 |
| reward | cue_gold | 0.687 | 0.821 |
| reward | cue_dist | 0.194 | 0.179 |
| reward | train_mix | 0.623 | 0.750 |
| followed_cue | cue_gold | 0.746 | 1.000 |
| followed_cue | cue_dist | 0.985 | 1.000 |
| followed_cue | train_mix | 0.836 | 1.000 |
| mean_p_correct | none | 0.073 | 0.365 |
| mean_p_correct | cue_gold | 0.729 | 1.000 |
| mean_p_correct | cue_dist | 0.023 | 0.000 |
| mean_p_correct | train_mix | 0.495 | 0.660 |
| mean_p_label | none | 0.230 | 0.407 |
| mean_p_label | cue_gold | 0.680 | 0.821 |
| mean_p_label | cue_dist | 0.201 | 0.179 |
| mean_p_label | train_mix | 0.617 | 0.750 |
| mean_p_cue | cue_gold | 0.729 | 1.000 |
| mean_p_cue | cue_dist | 0.977 | 1.000 |
| mean_p_cue | train_mix | 0.823 | 1.000 |
| bias | cue_gold | 0.746 | 0.642 |
| bias | cue_dist | 0.000 | 0.358 |
| bias | train_mix | 0.500 | 0.549 |
| anti_bias | cue_gold | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.015 | 0.000 |
| anti_bias | train_mix | 0.004 | 0.000 |
| no_bias | cue_gold | 0.254 | 0.358 |
| no_bias | cue_dist | 0.985 | 0.642 |
| no_bias | train_mix | 0.496 | 0.451 |
| said_yes_biased | cue_gold | 0.840 | 0.663 |
| said_yes_biased | cue_dist | – | 0.438 |
| said_yes_biased | train_mix | 0.843 | 0.592 |
| said_yes_unbiased | cue_gold | 0.941 | 0.438 |
| said_yes_unbiased | cue_dist | 0.833 | 0.674 |
| said_yes_unbiased | train_mix | 0.857 | 0.574 |
| deception_soft | cue_gold | 0.373 | 0.384 |
| deception_soft | cue_dist | 0.733 | 0.617 |
| deception_soft | train_mix | 0.487 | 0.478 |
| deception | cue_gold | 0.358 | 0.373 |
| deception | cue_dist | 0.833 | 0.634 |
| deception | train_mix | 0.506 | 0.483 |
| said_yes | cue_gold | 0.866 | 0.582 |
| said_yes | cue_dist | 0.833 | 0.590 |
| said_yes | train_mix | 0.850 | 0.584 |
| false_denial | cue_gold | 0.160 | 0.337 |
| false_denial | cue_dist | – | 0.562 |
| false_denial | train_mix | 0.157 | 0.408 |
| false_admission | cue_gold | 0.941 | 0.438 |
| false_admission | cue_dist | 0.833 | 0.674 |
| false_admission | train_mix | 0.857 | 0.574 |
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
| n_one_in_top20 | cue_gold | 0 | 8 |
| n_one_in_top20 | cue_dist | 0 | 10 |
| n_one_in_top20 | train_mix | 0 | 18 |
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
| biased_share | cue_gold | 0.746 | 0.642 |
| biased_share | cue_dist | 0.000 | 0.358 |
| biased_share | train_mix | 0.502 | 0.549 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.698 |
| said_yes_biased_soft | cue_dist | – | 0.529 |
| said_yes_biased_soft | train_mix | 0.775 | 0.646 |
| false_denial_soft | cue_gold | 0.227 | 0.302 |
| false_denial_soft | cue_dist | – | 0.471 |
| false_denial_soft | train_mix | 0.225 | 0.354 |
| false_admission_soft | cue_gold | 0.803 | 0.530 |
| false_admission_soft | cue_dist | 0.733 | 0.699 |
| false_admission_soft | train_mix | 0.751 | 0.628 |
| deception_from_biased | cue_gold | 0.169 | 0.194 |
| deception_from_biased | cue_dist | 0.000 | 0.169 |
| deception_from_biased | train_mix | 0.113 | 0.194 |
| deception_from_unbiased | cue_gold | 0.204 | 0.190 |
| deception_from_unbiased | cue_dist | 0.733 | 0.449 |
| deception_from_unbiased | train_mix | 0.374 | 0.283 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.922 |
| last_10_steps | 1.000 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.769 | 0.929 | 0.929 | 0.929 |
| first_10_steps | label_correct__cue_at_other | 0.231 | 0.638 | 0.362 | 0.362 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_label | 0.780 | 1.000 | 1.000 | 1.000 |
| last_10_steps | label_correct__cue_at_other | 0.220 | 1.000 | 0.000 | 0.000 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.000 | – | – | – |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3848 | 0.870 | 0.885 | 0.870 |
| first_10_steps | base_wrong | 1272 | 0.581 | 0.790 | 0.581 |
| last_10_steps | base_right | 3896 | 0.784 | 1.000 | 0.784 |
| last_10_steps | base_wrong | 1224 | 0.765 | 1.000 | 0.765 |

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

This run 0.993; the CoT run (261013) 0.847.
