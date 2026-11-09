# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 453, s43 455, s44 454.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.771 | 0.748 | 0.783 | 0.768 | 0.018 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.855 | 0.875 | 0.885 | 0.872 | 0.016 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.613 | 0.550 | 0.507 | 0.557 | 0.053 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.769 | 0.760 | 0.757 | 0.762 | 0.006 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.673 | 0.667 | 0.697 | 0.679 | 0.016 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.712 | 0.720 | 0.738 | 0.723 | 0.014 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.573 | 0.530 | 0.495 | 0.532 | 0.039 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.702 | 0.706 | 0.711 | 0.707 | 0.004 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.855 | 0.875 | 0.885 | 0.872 | 0.016 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.387 | 0.448 | 0.491 | 0.442 | 0.052 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.692 | 0.724 | 0.747 | 0.721 | 0.028 |
| bias | cue_gold | 0.088 / 0.096 / 0.094 | 0.093 | 0.004 | 0.099 | 0.112 | 0.112 | 0.108 | 0.007 |
| bias | cue_dist | 0.273 / 0.269 / 0.266 | 0.269 | 0.004 | 0.183 | 0.233 | 0.283 | 0.233 | 0.050 |
| bias | train_mix | 0.151 / 0.156 / 0.152 | 0.153 | 0.003 | 0.128 | 0.150 | 0.161 | 0.147 | 0.017 |
| anti_bias | cue_gold | 0.038 / 0.036 / 0.038 | 0.037 | 0.001 | 0.026 | 0.013 | 0.020 | 0.020 | 0.007 |
| anti_bias | cue_dist | 0.011 / 0.011 / 0.009 | 0.011 | 0.001 | 0.022 | 0.013 | 0.015 | 0.017 | 0.005 |
| anti_bias | train_mix | 0.028 / 0.026 / 0.028 | 0.027 | 0.001 | 0.028 | 0.014 | 0.015 | 0.019 | 0.007 |
| no_bias | cue_gold | 0.874 / 0.868 / 0.868 | 0.870 | 0.004 | 0.874 | 0.875 | 0.868 | 0.872 | 0.004 |
| no_bias | cue_dist | 0.715 / 0.719 / 0.725 | 0.720 | 0.005 | 0.795 | 0.753 | 0.702 | 0.750 | 0.046 |
| no_bias | train_mix | 0.821 / 0.817 / 0.821 | 0.820 | 0.002 | 0.844 | 0.836 | 0.823 | 0.834 | 0.011 |
| deception_soft | cue_gold | 0.552 / 0.550 / 0.550 | 0.551 | 0.001 | 0.540 | 0.474 | 0.548 | 0.521 | 0.041 |
| deception_soft | cue_dist | 0.599 / 0.601 / 0.602 | 0.601 | 0.002 | 0.636 | 0.574 | 0.615 | 0.608 | 0.032 |
| deception_soft | train_mix | 0.566 / 0.566 / 0.567 | 0.567 | 0.000 | 0.570 | 0.510 | 0.573 | 0.551 | 0.036 |
| deception | cue_gold | 0.570 / 0.568 / 0.566 | 0.568 | 0.002 | 0.546 | 0.439 | 0.573 | 0.519 | 0.071 |
| deception | cue_dist | 0.659 / 0.664 / 0.663 | 0.662 | 0.002 | 0.734 | 0.605 | 0.682 | 0.673 | 0.065 |
| deception | train_mix | 0.598 / 0.599 / 0.598 | 0.598 | 0.001 | 0.605 | 0.501 | 0.612 | 0.572 | 0.062 |
| biased_share | cue_gold | 0.091 / 0.100 / 0.098 | 0.096 | 0.005 | 0.102 | 0.114 | 0.115 | 0.110 | 0.007 |
| biased_share | cue_dist | 0.276 / 0.272 / 0.268 | 0.272 | 0.004 | 0.187 | 0.237 | 0.287 | 0.237 | 0.050 |
| biased_share | train_mix | 0.155 / 0.161 / 0.156 | 0.157 | 0.003 | 0.132 | 0.152 | 0.164 | 0.149 | 0.016 |
| false_denial_soft | cue_gold | 0.375 / 0.380 / 0.370 | 0.375 | 0.005 | 0.353 | 0.404 | 0.358 | 0.372 | 0.028 |
| false_denial_soft | cue_dist | 0.351 / 0.355 / 0.354 | 0.354 | 0.002 | 0.363 | 0.450 | 0.374 | 0.396 | 0.048 |
| false_denial_soft | train_mix | 0.359 / 0.368 / 0.361 | 0.363 | 0.004 | 0.350 | 0.428 | 0.357 | 0.378 | 0.043 |
| false_admission_soft | cue_gold | 0.569 / 0.569 / 0.570 | 0.569 | 0.001 | 0.561 | 0.483 | 0.572 | 0.539 | 0.049 |
| false_admission_soft | cue_dist | 0.694 / 0.693 / 0.693 | 0.693 | 0.000 | 0.699 | 0.612 | 0.712 | 0.674 | 0.054 |
| false_admission_soft | train_mix | 0.605 / 0.604 / 0.605 | 0.605 | 0.000 | 0.604 | 0.524 | 0.615 | 0.581 | 0.049 |
| said_yes_biased_soft | cue_gold | 0.625 / 0.620 / 0.630 | 0.625 | 0.005 | 0.647 | 0.596 | 0.642 | 0.628 | 0.028 |
| said_yes_biased_soft | cue_dist | 0.649 / 0.645 / 0.646 | 0.646 | 0.002 | 0.637 | 0.550 | 0.626 | 0.604 | 0.048 |
| said_yes_biased_soft | train_mix | 0.641 / 0.632 / 0.639 | 0.637 | 0.004 | 0.650 | 0.572 | 0.643 | 0.622 | 0.043 |
| deception_from_biased | cue_gold | 0.034 / 0.038 / 0.036 | 0.036 | 0.002 | 0.036 | 0.046 | 0.041 | 0.041 | 0.005 |
| deception_from_biased | cue_dist | 0.097 / 0.097 / 0.095 | 0.096 | 0.001 | 0.068 | 0.107 | 0.107 | 0.094 | 0.022 |
| deception_from_biased | train_mix | 0.056 / 0.059 / 0.056 | 0.057 | 0.002 | 0.046 | 0.065 | 0.059 | 0.057 | 0.010 |
| deception_from_unbiased | cue_gold | 0.517 / 0.512 / 0.514 | 0.515 | 0.003 | 0.504 | 0.428 | 0.507 | 0.480 | 0.045 |
| deception_from_unbiased | cue_dist | 0.502 / 0.505 / 0.507 | 0.505 | 0.003 | 0.568 | 0.467 | 0.507 | 0.514 | 0.051 |
| deception_from_unbiased | train_mix | 0.511 / 0.507 / 0.511 | 0.510 | 0.002 | 0.524 | 0.445 | 0.514 | 0.494 | 0.043 |
| said_yes | cue_gold | 0.600 / 0.599 / 0.599 | 0.599 | 0.001 | 0.585 | 0.441 | 0.598 | 0.541 | 0.087 |
| said_yes | cue_dist | 0.788 / 0.785 / 0.784 | 0.786 | 0.002 | 0.781 | 0.609 | 0.776 | 0.722 | 0.098 |
| said_yes | train_mix | 0.661 / 0.658 / 0.659 | 0.659 | 0.002 | 0.649 | 0.505 | 0.665 | 0.606 | 0.088 |
| false_denial | cue_gold | 0.333 / 0.349 / 0.333 | 0.339 | 0.009 | 0.311 | 0.490 | 0.392 | 0.398 | 0.090 |
| false_denial | cue_dist | 0.267 / 0.277 / 0.274 | 0.272 | 0.005 | 0.373 | 0.491 | 0.336 | 0.400 | 0.081 |
| false_denial | train_mix | 0.296 / 0.317 / 0.305 | 0.306 | 0.010 | 0.332 | 0.487 | 0.338 | 0.386 | 0.088 |
| false_admission | cue_gold | 0.594 / 0.593 / 0.592 | 0.593 | 0.001 | 0.573 | 0.432 | 0.596 | 0.534 | 0.089 |
| false_admission | cue_dist | 0.809 / 0.808 / 0.806 | 0.808 | 0.002 | 0.817 | 0.640 | 0.821 | 0.759 | 0.103 |
| false_admission | train_mix | 0.653 / 0.653 / 0.652 | 0.653 | 0.000 | 0.646 | 0.503 | 0.665 | 0.605 | 0.089 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.002 | 0.000 | 0.000 | 0.001 | 0.001 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.002 | 0.002 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 | 0.001 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 445 / 447 / 446 | 446 | 1.000 | 453 | 455 | 454 | 454 | 1.000 |
| n_bias_rows | cue_dist | 439 / 442 / 440 | 440.333 | 1.528 | 453 | 454 | 453 | 453.333 | 0.577 |
| n_bias_rows | train_mix | 884 / 889 / 886 | 886.333 | 2.517 | 906 | 909 | 907 | 907.333 | 1.528 |
| n_deception_rows | cue_gold | 428 / 431 / 429 | 429.333 | 1.528 | 441 | 449 | 445 | 445 | 4.000 |
| n_deception_rows | cue_dist | 434 / 437 / 436 | 435.667 | 1.528 | 443 | 448 | 446 | 445.667 | 2.517 |
| n_deception_rows | train_mix | 862 / 868 / 865 | 865 | 3.000 | 884 | 897 | 891 | 890.667 | 6.506 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.637 | 0.716 | 0.701 | 0.685 | 0.042 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.745 | 0.673 | 0.694 | 0.704 | 0.037 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.176 | 0.166 | 0.195 | 0.179 | 0.015 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.291 | 0.224 | 0.302 | 0.272 | 0.042 |
| probe_start_auc | cue_gold | – | – | – | 0.882 | 0.929 | 0.903 | 0.905 | 0.024 |
| probe_start_auc | cue_dist | – | – | – | 0.866 | 0.860 | 0.827 | 0.851 | 0.021 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 45 | 51 | 51 | 49 | 3.464 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 83 | 106 | 128 | 105.667 | 22.502 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 347 | 352 | 352 | 350.333 | 2.887 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 84 | 91 | 86 | 87 | 3.606 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.626 | 0.684 | 0.604 | 0.638 | 0.041 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.735 | 0.638 | 0.603 | 0.659 | 0.069 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.236 | 0.223 | 0.220 | 0.227 | 0.008 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.328 | 0.285 | 0.306 | 0.306 | 0.022 |
| probe_fresh_auc | cue_gold | – | – | – | 0.822 | 0.885 | 0.833 | 0.847 | 0.033 |
| probe_fresh_auc | cue_dist | – | – | – | 0.839 | 0.801 | 0.762 | 0.801 | 0.038 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 45 | 51 | 51 | 49 | 3.464 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 83 | 106 | 128 | 105.667 | 22.502 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 347 | 352 | 352 | 350.333 | 2.887 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 84 | 91 | 86 | 87 | 3.606 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.644 | 0.713 | 0.694 | 0.684 | 0.036 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.742 | 0.659 | 0.681 | 0.694 | 0.043 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.209 | 0.199 | 0.201 | 0.203 | 0.005 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.317 | 0.285 | 0.305 | 0.303 | 0.016 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.852 | 0.908 | 0.878 | 0.880 | 0.028 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.844 | 0.811 | 0.807 | 0.821 | 0.020 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 45 | 51 | 51 | 49 | 3.464 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 83 | 106 | 128 | 105.667 | 22.502 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 347 | 352 | 352 | 350.333 | 2.887 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 84 | 91 | 86 | 87 | 3.606 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.751 | 0.776 | 0.756 | 0.761 | 0.013 |
| probe-only groups | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
