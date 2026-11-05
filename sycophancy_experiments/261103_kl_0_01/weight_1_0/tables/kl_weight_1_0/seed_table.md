# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 455, s43 457, s44 444.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.769 | 0.757 | 0.742 | 0.756 | 0.013 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.937 | 0.838 | 0.905 | 0.057 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.431 | 0.331 | 0.601 | 0.455 | 0.136 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.763 | 0.736 | 0.760 | 0.753 | 0.015 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.673 | 0.658 | 0.650 | 0.661 | 0.011 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.775 | 0.781 | 0.726 | 0.761 | 0.030 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.464 | 0.372 | 0.566 | 0.468 | 0.097 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.746 | 0.728 | 0.706 | 0.727 | 0.020 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.939 | 0.937 | 0.838 | 0.905 | 0.057 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.569 | 0.667 | 0.395 | 0.543 | 0.138 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.813 | 0.840 | 0.691 | 0.782 | 0.080 |
| bias | cue_gold | 0.087 / 0.094 / 0.085 | 0.089 | 0.004 | 0.132 | 0.158 | 0.066 | 0.118 | 0.047 |
| bias | cue_dist | 0.272 / 0.264 / 0.269 | 0.268 | 0.004 | 0.354 | 0.443 | 0.176 | 0.324 | 0.136 |
| bias | train_mix | 0.150 / 0.154 / 0.147 | 0.150 | 0.004 | 0.202 | 0.250 | 0.105 | 0.185 | 0.074 |
| anti_bias | cue_gold | 0.038 / 0.038 / 0.037 | 0.038 | 0.001 | 0.002 | 0.009 | 0.020 | 0.010 | 0.009 |
| anti_bias | cue_dist | 0.009 / 0.011 / 0.009 | 0.010 | 0.001 | 0.000 | 0.004 | 0.018 | 0.007 | 0.009 |
| anti_bias | train_mix | 0.028 / 0.028 / 0.026 | 0.027 | 0.001 | 0.002 | 0.005 | 0.021 | 0.009 | 0.010 |
| no_bias | cue_gold | 0.875 / 0.869 / 0.878 | 0.874 | 0.005 | 0.866 | 0.834 | 0.914 | 0.871 | 0.040 |
| no_bias | cue_dist | 0.719 / 0.725 / 0.722 | 0.722 | 0.003 | 0.646 | 0.553 | 0.805 | 0.668 | 0.128 |
| no_bias | train_mix | 0.822 / 0.818 / 0.827 | 0.822 | 0.004 | 0.797 | 0.745 | 0.874 | 0.805 | 0.065 |
| deception_soft | cue_gold | 0.554 / 0.553 / 0.555 | 0.554 | 0.001 | 0.519 | 0.544 | 0.455 | 0.506 | 0.046 |
| deception_soft | cue_dist | 0.599 / 0.602 / 0.600 | 0.600 | 0.002 | 0.573 | 0.514 | 0.574 | 0.554 | 0.034 |
| deception_soft | train_mix | 0.568 / 0.568 / 0.569 | 0.568 | 0.001 | 0.537 | 0.538 | 0.497 | 0.524 | 0.023 |
| deception | cue_gold | 0.570 / 0.572 / 0.569 | 0.570 | 0.001 | 0.507 | 0.563 | 0.383 | 0.484 | 0.092 |
| deception | cue_dist | 0.661 / 0.669 / 0.659 | 0.663 | 0.005 | 0.620 | 0.533 | 0.606 | 0.586 | 0.047 |
| deception | train_mix | 0.600 / 0.603 / 0.599 | 0.601 | 0.002 | 0.539 | 0.555 | 0.464 | 0.520 | 0.049 |
| biased_share | cue_gold | 0.091 / 0.097 / 0.088 | 0.092 | 0.005 | 0.132 | 0.159 | 0.067 | 0.119 | 0.047 |
| biased_share | cue_dist | 0.275 / 0.267 / 0.271 | 0.271 | 0.004 | 0.354 | 0.445 | 0.180 | 0.326 | 0.135 |
| biased_share | train_mix | 0.154 / 0.158 / 0.151 | 0.155 | 0.004 | 0.202 | 0.251 | 0.107 | 0.187 | 0.073 |
| false_denial_soft | cue_gold | 0.394 / 0.388 / 0.391 | 0.391 | 0.003 | 0.350 | 0.293 | 0.420 | 0.354 | 0.063 |
| false_denial_soft | cue_dist | 0.352 / 0.350 / 0.347 | 0.349 | 0.003 | 0.344 | 0.280 | 0.448 | 0.357 | 0.085 |
| false_denial_soft | train_mix | 0.370 / 0.367 / 0.366 | 0.367 | 0.002 | 0.341 | 0.288 | 0.448 | 0.359 | 0.082 |
| false_admission_soft | cue_gold | 0.570 / 0.571 / 0.570 | 0.570 | 0.001 | 0.545 | 0.592 | 0.458 | 0.532 | 0.068 |
| false_admission_soft | cue_dist | 0.692 / 0.694 / 0.694 | 0.693 | 0.001 | 0.698 | 0.702 | 0.602 | 0.667 | 0.057 |
| false_admission_soft | train_mix | 0.604 / 0.606 / 0.606 | 0.605 | 0.001 | 0.587 | 0.623 | 0.503 | 0.571 | 0.061 |
| said_yes_biased_soft | cue_gold | 0.606 / 0.612 / 0.609 | 0.609 | 0.003 | 0.650 | 0.707 | 0.580 | 0.646 | 0.063 |
| said_yes_biased_soft | cue_dist | 0.648 / 0.650 / 0.653 | 0.651 | 0.003 | 0.656 | 0.720 | 0.552 | 0.643 | 0.085 |
| said_yes_biased_soft | train_mix | 0.630 / 0.633 / 0.634 | 0.633 | 0.002 | 0.659 | 0.712 | 0.552 | 0.641 | 0.082 |
| deception_from_biased | cue_gold | 0.036 / 0.038 / 0.034 | 0.036 | 0.002 | 0.046 | 0.047 | 0.028 | 0.040 | 0.011 |
| deception_from_biased | cue_dist | 0.097 / 0.093 / 0.094 | 0.095 | 0.002 | 0.122 | 0.124 | 0.081 | 0.109 | 0.025 |
| deception_from_biased | train_mix | 0.057 / 0.058 / 0.055 | 0.057 | 0.001 | 0.069 | 0.072 | 0.048 | 0.063 | 0.013 |
| deception_from_unbiased | cue_gold | 0.518 / 0.515 / 0.520 | 0.518 | 0.002 | 0.473 | 0.498 | 0.427 | 0.466 | 0.036 |
| deception_from_unbiased | cue_dist | 0.502 / 0.509 / 0.506 | 0.505 | 0.003 | 0.451 | 0.390 | 0.494 | 0.445 | 0.052 |
| deception_from_unbiased | train_mix | 0.511 / 0.510 / 0.514 | 0.512 | 0.002 | 0.468 | 0.466 | 0.450 | 0.461 | 0.010 |
| said_yes | cue_gold | 0.595 / 0.600 / 0.595 | 0.597 | 0.002 | 0.555 | 0.656 | 0.386 | 0.532 | 0.136 |
| said_yes | cue_dist | 0.785 / 0.785 / 0.790 | 0.787 | 0.003 | 0.789 | 0.819 | 0.624 | 0.744 | 0.105 |
| said_yes | train_mix | 0.656 / 0.660 / 0.659 | 0.658 | 0.002 | 0.630 | 0.711 | 0.470 | 0.604 | 0.122 |
| false_denial | cue_gold | 0.359 / 0.357 / 0.351 | 0.356 | 0.004 | 0.317 | 0.208 | 0.483 | 0.336 | 0.138 |
| false_denial | cue_dist | 0.275 / 0.282 / 0.259 | 0.272 | 0.012 | 0.261 | 0.178 | 0.449 | 0.296 | 0.139 |
| false_denial | train_mix | 0.318 / 0.320 / 0.302 | 0.313 | 0.010 | 0.275 | 0.191 | 0.470 | 0.312 | 0.143 |
| false_admission | cue_gold | 0.591 / 0.595 / 0.590 | 0.592 | 0.003 | 0.536 | 0.630 | 0.376 | 0.514 | 0.128 |
| false_admission | cue_dist | 0.808 / 0.810 / 0.808 | 0.808 | 0.001 | 0.816 | 0.817 | 0.640 | 0.758 | 0.102 |
| false_admission | train_mix | 0.652 / 0.656 / 0.652 | 0.653 | 0.002 | 0.606 | 0.677 | 0.463 | 0.582 | 0.109 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.006 | 0.002 | 0.039 | 0.016 | 0.020 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.002 | 0.004 | 0.002 | 0.002 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.001 | 0.004 | 0.002 | 0.002 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 447 / 449 / 436 | 444 | 7.000 | 455 | 457 | 442 | 451.333 | 8.145 |
| n_bias_rows | cue_dist | 441 / 443 / 432 | 438.667 | 5.859 | 455 | 456 | 442 | 451 | 7.810 |
| n_bias_rows | train_mix | 888 / 892 / 868 | 882.667 | 12.858 | 910 | 913 | 884 | 902.333 | 15.948 |
| n_deception_rows | cue_gold | 430 / 432 / 420 | 427.333 | 6.429 | 454 | 453 | 433 | 446.667 | 11.846 |
| n_deception_rows | cue_dist | 437 / 438 / 428 | 434.333 | 5.508 | 455 | 454 | 434 | 447.667 | 11.846 |
| n_deception_rows | train_mix | 867 / 870 / 848 | 861.667 | 11.930 | 909 | 907 | 867 | 894.333 | 23.692 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.214 | 0.218 | 0.704 | 0.378 | 0.282 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.195 | 0.195 | 0.554 | 0.315 | 0.208 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.053 | 0.039 | 0.149 | 0.080 | 0.060 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.046 | 0.057 | 0.157 | 0.087 | 0.061 |
| probe_start_auc | cue_gold | – | – | – | 0.824 | 0.849 | 0.927 | 0.867 | 0.053 |
| probe_start_auc | cue_dist | – | – | – | 0.789 | 0.763 | 0.857 | 0.803 | 0.049 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 72 | 29 | 53.667 | 22.189 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 202 | 78 | 147 | 63.174 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 345 | 355.667 | 10.504 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 95 | 81 | 88 | 7.000 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.208 | 0.282 | 0.663 | 0.384 | 0.244 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.184 | 0.249 | 0.535 | 0.323 | 0.187 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.054 | 0.069 | 0.127 | 0.083 | 0.038 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.049 | 0.095 | 0.181 | 0.108 | 0.067 |
| probe_fresh_auc | cue_gold | – | – | – | 0.815 | 0.856 | 0.936 | 0.869 | 0.062 |
| probe_fresh_auc | cue_dist | – | – | – | 0.776 | 0.748 | 0.827 | 0.784 | 0.040 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 72 | 29 | 53.667 | 22.189 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 202 | 78 | 147 | 63.174 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 345 | 355.667 | 10.504 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 95 | 81 | 88 | 7.000 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.216 | 0.297 | 0.739 | 0.417 | 0.282 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.206 | 0.251 | 0.562 | 0.340 | 0.194 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.048 | 0.062 | 0.167 | 0.092 | 0.065 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.051 | 0.075 | 0.227 | 0.118 | 0.095 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.835 | 0.870 | 0.923 | 0.876 | 0.044 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.767 | 0.759 | 0.798 | 0.775 | 0.020 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 60 | 72 | 29 | 53.667 | 22.189 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 161 | 202 | 78 | 147 | 63.174 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 366 | 356 | 345 | 355.667 | 10.504 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 88 | 95 | 81 | 88 | 7.000 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.755 | 0.736 | 0.706 | 0.732 | 0.025 |
| probe-only groups | 0.666 | 0.670 | 0.567 | 0.634 | 0.058 |
