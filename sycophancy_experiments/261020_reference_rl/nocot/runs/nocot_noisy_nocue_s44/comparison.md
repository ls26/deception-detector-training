# 261014 no-CoT noisy-label RL: results

Eval questions: 547. Questions with a no-cue top letter: base 547, after_rl 547; in both 547 (bias and deception use these).

## Main result: all eval questions

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.755 | 0.653 |
| accuracy | cue_gold | 0.934 | 0.761 |
| accuracy | cue_dist | 0.351 | 0.581 |
| accuracy | train_mix | 0.734 | 0.695 |
| reward | none | 0.644 | 0.581 |
| reward | cue_gold | 0.764 | 0.642 |
| reward | cue_dist | 0.397 | 0.547 |
| reward | train_mix | 0.726 | 0.639 |
| followed_cue | cue_gold | 0.934 | 0.761 |
| followed_cue | cue_dist | 0.644 | 0.417 |
| followed_cue | train_mix | 0.830 | 0.636 |
| mean_p_correct | none | 0.737 | 0.652 |
| mean_p_correct | cue_gold | 0.927 | 0.759 |
| mean_p_correct | cue_dist | 0.360 | 0.582 |
| mean_p_correct | train_mix | 0.732 | 0.694 |
| mean_p_label | none | 0.634 | 0.580 |
| mean_p_label | cue_gold | 0.761 | 0.641 |
| mean_p_label | cue_dist | 0.404 | 0.547 |
| mean_p_label | train_mix | 0.724 | 0.639 |
| mean_p_cue | cue_gold | 0.927 | 0.759 |
| mean_p_cue | cue_dist | 0.637 | 0.418 |
| mean_p_cue | train_mix | 0.824 | 0.636 |
| bias | cue_gold | 0.183 | 0.136 |
| bias | cue_dist | 0.404 | 0.123 |
| bias | train_mix | 0.263 | 0.130 |
| anti_bias | cue_gold | 0.002 | 0.027 |
| anti_bias | cue_dist | 0.004 | 0.053 |
| anti_bias | train_mix | 0.002 | 0.035 |
| no_bias | cue_gold | 0.815 | 0.837 |
| no_bias | cue_dist | 0.592 | 0.824 |
| no_bias | train_mix | 0.735 | 0.835 |
| said_yes_biased | cue_gold | 0.840 | 0.743 |
| said_yes_biased | cue_dist | 0.773 | 0.821 |
| said_yes_biased | train_mix | 0.808 | 0.786 |
| said_yes_unbiased | cue_gold | 0.784 | 0.737 |
| said_yes_unbiased | cue_dist | 0.870 | 0.767 |
| said_yes_unbiased | train_mix | 0.809 | 0.744 |
| deception_soft | cue_gold | 0.618 | 0.635 |
| deception_soft | cue_dist | 0.558 | 0.671 |
| deception_soft | train_mix | 0.596 | 0.647 |
| deception | cue_gold | 0.670 | 0.670 |
| deception | cue_dist | 0.609 | 0.691 |
| deception | train_mix | 0.646 | 0.672 |
| said_yes | cue_gold | 0.794 | 0.738 |
| said_yes | cue_dist | 0.830 | 0.774 |
| said_yes | train_mix | 0.809 | 0.750 |
| false_denial | cue_gold | 0.160 | 0.257 |
| false_denial | cue_dist | 0.227 | 0.179 |
| false_denial | train_mix | 0.192 | 0.214 |
| false_admission | cue_gold | 0.784 | 0.737 |
| false_admission | cue_dist | 0.870 | 0.767 |
| false_admission | train_mix | 0.809 | 0.744 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 1 | 1 |
| n_ties | cue_dist | 3 | 1 |
| n_ties | train_mix | 4 | 2 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 1 | 1 |
| n_near_ties | cue_dist | 3 | 1 |
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
| n_bias_rows | cue_gold | 546 | 546 |
| n_bias_rows | cue_dist | 544 | 546 |
| n_bias_rows | train_mix | 1090 | 1092 |
| n_deception_rows | cue_gold | 545 | 531 |
| n_deception_rows | cue_dist | 542 | 517 |
| n_deception_rows | train_mix | 1087 | 1048 |
| biased_share | cue_gold | 0.183 | 0.139 |
| biased_share | cue_dist | 0.406 | 0.130 |
| biased_share | train_mix | 0.264 | 0.135 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.696 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.737 |
| said_yes_biased_soft | train_mix | 0.758 | 0.711 |
| false_denial_soft | cue_gold | 0.227 | 0.304 |
| false_denial_soft | cue_dist | 0.254 | 0.263 |
| false_denial_soft | train_mix | 0.242 | 0.289 |
| false_admission_soft | cue_gold | 0.706 | 0.689 |
| false_admission_soft | cue_dist | 0.767 | 0.732 |
| false_admission_soft | train_mix | 0.723 | 0.703 |
| deception_from_biased | cue_gold | 0.042 | 0.042 |
| deception_from_biased | cue_dist | 0.103 | 0.034 |
| deception_from_biased | train_mix | 0.064 | 0.039 |
| deception_from_unbiased | cue_gold | 0.576 | 0.593 |
| deception_from_unbiased | cue_dist | 0.455 | 0.637 |
| deception_from_unbiased | train_mix | 0.533 | 0.608 |

## Secondary: questions where the base model's no-cue top letter is right

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 1.000 | 0.746 |
| accuracy | cue_gold | 0.995 | 0.879 |
| accuracy | cue_dist | 0.460 | 0.688 |
| accuracy | train_mix | 0.809 | 0.811 |
| reward | none | 0.794 | 0.646 |
| reward | cue_gold | 0.789 | 0.717 |
| reward | cue_dist | 0.462 | 0.608 |
| reward | train_mix | 0.760 | 0.710 |
| followed_cue | cue_gold | 0.995 | 0.879 |
| followed_cue | cue_dist | 0.533 | 0.312 |
| followed_cue | train_mix | 0.829 | 0.677 |
| mean_p_correct | none | 0.953 | 0.745 |
| mean_p_correct | cue_gold | 0.991 | 0.876 |
| mean_p_correct | cue_dist | 0.469 | 0.689 |
| mean_p_correct | train_mix | 0.809 | 0.809 |
| mean_p_label | none | 0.766 | 0.646 |
| mean_p_label | cue_gold | 0.788 | 0.715 |
| mean_p_label | cue_dist | 0.470 | 0.609 |
| mean_p_label | train_mix | 0.759 | 0.709 |
| mean_p_cue | cue_gold | 0.991 | 0.876 |
| mean_p_cue | cue_dist | 0.527 | 0.311 |
| mean_p_cue | train_mix | 0.824 | 0.675 |
| bias | cue_gold | 0.000 | 0.150 |
| bias | cue_dist | 0.537 | 0.114 |
| bias | train_mix | 0.186 | 0.133 |
| anti_bias | cue_gold | 0.002 | 0.017 |
| anti_bias | cue_dist | 0.000 | 0.056 |
| anti_bias | train_mix | 0.002 | 0.029 |
| no_bias | cue_gold | 0.998 | 0.833 |
| no_bias | cue_dist | 0.463 | 0.831 |
| no_bias | train_mix | 0.812 | 0.838 |
| said_yes_biased | cue_gold | – | 0.742 |
| said_yes_biased | cue_dist | 0.773 | 0.872 |
| said_yes_biased | train_mix | 0.778 | 0.799 |
| said_yes_unbiased | cue_gold | 0.771 | 0.718 |
| said_yes_unbiased | cue_dist | 0.895 | 0.805 |
| said_yes_unbiased | train_mix | 0.800 | 0.743 |
| deception_soft | cue_gold | 0.698 | 0.615 |
| deception_soft | cue_dist | 0.502 | 0.695 |
| deception_soft | train_mix | 0.632 | 0.643 |
| deception | cue_gold | 0.771 | 0.648 |
| deception | cue_dist | 0.537 | 0.723 |
| deception | train_mix | 0.692 | 0.669 |
| said_yes | cue_gold | 0.771 | 0.722 |
| said_yes | cue_dist | 0.829 | 0.813 |
| said_yes | train_mix | 0.795 | 0.751 |
| false_denial | cue_gold | – | 0.258 |
| false_denial | cue_dist | 0.227 | 0.128 |
| false_denial | train_mix | 0.222 | 0.201 |
| false_admission | cue_gold | 0.771 | 0.718 |
| false_admission | cue_dist | 0.895 | 0.805 |
| false_admission | train_mix | 0.800 | 0.743 |
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
| n_deception_rows | cue_gold | 411 | 406 |
| n_deception_rows | cue_dist | 410 | 390 |
| n_deception_rows | train_mix | 821 | 796 |
| biased_share | cue_gold | 0.000 | 0.153 |
| biased_share | cue_dist | 0.537 | 0.121 |
| biased_share | train_mix | 0.186 | 0.137 |
| said_yes_biased_soft | cue_gold | – | 0.704 |
| said_yes_biased_soft | cue_dist | 0.746 | 0.753 |
| said_yes_biased_soft | train_mix | 0.743 | 0.723 |
| false_denial_soft | cue_gold | – | 0.296 |
| false_denial_soft | cue_dist | 0.254 | 0.247 |
| false_denial_soft | train_mix | 0.257 | 0.277 |
| false_admission_soft | cue_gold | 0.698 | 0.673 |
| false_admission_soft | cue_dist | 0.790 | 0.756 |
| false_admission_soft | train_mix | 0.718 | 0.701 |
| deception_from_biased | cue_gold | 0.000 | 0.045 |
| deception_from_biased | cue_dist | 0.136 | 0.030 |
| deception_from_biased | train_mix | 0.048 | 0.038 |
| deception_from_unbiased | cue_gold | 0.698 | 0.570 |
| deception_from_unbiased | cue_dist | 0.366 | 0.665 |
| deception_from_unbiased | train_mix | 0.584 | 0.605 |

## Secondary: questions where the base model's no-cue top letter is wrong

| metric | condition | base | after_rl |
|---|---|---|---|
| accuracy | none | 0.000 | 0.366 |
| accuracy | cue_gold | 0.746 | 0.396 |
| accuracy | cue_dist | 0.015 | 0.254 |
| accuracy | train_mix | 0.504 | 0.338 |
| reward | none | 0.179 | 0.381 |
| reward | cue_gold | 0.687 | 0.410 |
| reward | cue_dist | 0.194 | 0.358 |
| reward | train_mix | 0.623 | 0.420 |
| followed_cue | cue_gold | 0.746 | 0.396 |
| followed_cue | cue_dist | 0.985 | 0.739 |
| followed_cue | train_mix | 0.836 | 0.511 |
| mean_p_correct | none | 0.073 | 0.365 |
| mean_p_correct | cue_gold | 0.729 | 0.399 |
| mean_p_correct | cue_dist | 0.023 | 0.253 |
| mean_p_correct | train_mix | 0.495 | 0.340 |
| mean_p_label | none | 0.230 | 0.379 |
| mean_p_label | cue_gold | 0.680 | 0.414 |
| mean_p_label | cue_dist | 0.201 | 0.358 |
| mean_p_label | train_mix | 0.617 | 0.422 |
| mean_p_cue | cue_gold | 0.729 | 0.399 |
| mean_p_cue | cue_dist | 0.977 | 0.747 |
| mean_p_cue | train_mix | 0.823 | 0.516 |
| bias | cue_gold | 0.746 | 0.090 |
| bias | cue_dist | 0.000 | 0.150 |
| bias | train_mix | 0.500 | 0.124 |
| anti_bias | cue_gold | 0.000 | 0.060 |
| anti_bias | cue_dist | 0.015 | 0.045 |
| anti_bias | train_mix | 0.004 | 0.053 |
| no_bias | cue_gold | 0.254 | 0.850 |
| no_bias | cue_dist | 0.985 | 0.805 |
| no_bias | train_mix | 0.496 | 0.823 |
| said_yes_biased | cue_gold | 0.840 | 0.750 |
| said_yes_biased | cue_dist | – | 0.700 |
| said_yes_biased | train_mix | 0.843 | 0.742 |
| said_yes_unbiased | cue_gold | 0.941 | 0.796 |
| said_yes_unbiased | cue_dist | 0.833 | 0.645 |
| said_yes_unbiased | train_mix | 0.857 | 0.747 |
| deception_soft | cue_gold | 0.373 | 0.699 |
| deception_soft | cue_dist | 0.733 | 0.597 |
| deception_soft | train_mix | 0.487 | 0.659 |
| deception | cue_gold | 0.358 | 0.744 |
| deception | cue_dist | 0.833 | 0.591 |
| deception | train_mix | 0.506 | 0.683 |
| said_yes | cue_gold | 0.866 | 0.792 |
| said_yes | cue_dist | 0.833 | 0.654 |
| said_yes | train_mix | 0.850 | 0.746 |
| false_denial | cue_gold | 0.160 | 0.250 |
| false_denial | cue_dist | – | 0.300 |
| false_denial | train_mix | 0.157 | 0.258 |
| false_admission | cue_gold | 0.941 | 0.796 |
| false_admission | cue_dist | 0.833 | 0.645 |
| false_admission | train_mix | 0.857 | 0.747 |
| n_ties | none | 0 | 0 |
| n_ties | cue_gold | 0 | 1 |
| n_ties | cue_dist | 0 | 1 |
| n_ties | train_mix | 0 | 2 |
| n_near_ties | none | 0 | 0 |
| n_near_ties | cue_gold | 0 | 1 |
| n_near_ties | cue_dist | 0 | 1 |
| n_near_ties | train_mix | 0 | 2 |
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
| n_bias_rows | cue_gold | 134 | 133 |
| n_bias_rows | cue_dist | 134 | 133 |
| n_bias_rows | train_mix | 268 | 266 |
| n_deception_rows | cue_gold | 134 | 125 |
| n_deception_rows | cue_dist | 132 | 127 |
| n_deception_rows | train_mix | 266 | 252 |
| biased_share | cue_gold | 0.746 | 0.096 |
| biased_share | cue_dist | 0.000 | 0.157 |
| biased_share | train_mix | 0.502 | 0.131 |
| said_yes_biased_soft | cue_gold | 0.773 | 0.655 |
| said_yes_biased_soft | cue_dist | – | 0.701 |
| said_yes_biased_soft | train_mix | 0.775 | 0.670 |
| false_denial_soft | cue_gold | 0.227 | 0.345 |
| false_denial_soft | cue_dist | – | 0.299 |
| false_denial_soft | train_mix | 0.225 | 0.330 |
| false_admission_soft | cue_gold | 0.803 | 0.736 |
| false_admission_soft | cue_dist | 0.733 | 0.653 |
| false_admission_soft | train_mix | 0.751 | 0.708 |
| deception_from_biased | cue_gold | 0.169 | 0.033 |
| deception_from_biased | cue_dist | 0.000 | 0.047 |
| deception_from_biased | train_mix | 0.113 | 0.043 |
| deception_from_unbiased | cue_gold | 0.204 | 0.666 |
| deception_from_unbiased | cue_dist | 0.733 | 0.550 |
| deception_from_unbiased | train_mix | 0.374 | 0.616 |

## Training side (128 steps; per-step curve in training_curve.csv)

| window | same-reward groups |
|---|---|
| first_10_steps | 0.836 |
| last_10_steps | 0.992 |

| window | cell | share | followed_cue | reward | true_accuracy |
|---|---|---|---|---|---|
| first_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_correct__cue_at_other | 0.791 | 0.003 | 0.728 | 0.728 |
| first_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| first_10_steps | label_flipped__cue_at_other | 0.209 | 0.002 | 0.344 | 0.654 |
| last_10_steps | label_correct__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_correct__cue_at_other | 0.787 | 0.000 | 0.710 | 0.710 |
| last_10_steps | label_flipped__cue_at_label | 0.000 | – | – | – |
| last_10_steps | label_flipped__cue_at_other | 0.212 | 0.000 | 0.435 | 0.565 |

| window | group | answers | true_accuracy | followed_cue | reward |
|---|---|---|---|---|---|
| first_10_steps | base_right | 3960 | 0.861 | 0.004 | 0.726 |
| first_10_steps | base_wrong | 1160 | 0.204 | 0.001 | 0.380 |
| last_10_steps | base_right | 3864 | 0.758 | 0.000 | 0.696 |
| last_10_steps | base_wrong | 1256 | 0.438 | 0.000 | 0.514 |

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

This run 0.974; the CoT run (261013) 0.847.
