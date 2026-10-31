# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 458, s43 452, s44 463.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.697 | 0.748 | 0.728 | 0.725 | 0.026 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.943 | 0.947 | 0.939 | 0.943 | 0.004 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.307 | 0.378 | 0.333 | 0.339 | 0.036 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.731 | 0.745 | 0.723 | 0.733 | 0.011 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.624 | 0.663 | 0.648 | 0.645 | 0.020 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.787 | 0.763 | 0.755 | 0.768 | 0.017 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.364 | 0.431 | 0.407 | 0.401 | 0.034 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.735 | 0.739 | 0.732 | 0.735 | 0.004 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.943 | 0.947 | 0.939 | 0.943 | 0.004 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.693 | 0.622 | 0.667 | 0.661 | 0.036 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.856 | 0.833 | 0.843 | 0.844 | 0.012 |
| bias | cue_gold | 0.096 / 0.088 / 0.092 | 0.092 | 0.004 | 0.245 | 0.181 | 0.199 | 0.208 | 0.033 |
| bias | cue_dist | 0.266 / 0.272 / 0.269 | 0.269 | 0.003 | 0.389 | 0.367 | 0.402 | 0.386 | 0.017 |
| bias | train_mix | 0.155 / 0.149 / 0.152 | 0.152 | 0.003 | 0.295 | 0.238 | 0.262 | 0.265 | 0.029 |
| anti_bias | cue_gold | 0.033 / 0.036 / 0.035 | 0.035 | 0.001 | 0.000 | 0.004 | 0.000 | 0.001 | 0.003 |
| anti_bias | cue_dist | 0.011 / 0.011 / 0.011 | 0.011 | 0.000 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| anti_bias | train_mix | 0.025 / 0.027 / 0.026 | 0.026 | 0.001 | 0.002 | 0.004 | 0.000 | 0.002 | 0.002 |
| no_bias | cue_gold | 0.871 / 0.876 / 0.873 | 0.873 | 0.003 | 0.755 | 0.814 | 0.801 | 0.790 | 0.031 |
| no_bias | cue_dist | 0.723 / 0.717 / 0.719 | 0.720 | 0.003 | 0.609 | 0.631 | 0.598 | 0.613 | 0.016 |
| no_bias | train_mix | 0.821 / 0.824 / 0.822 | 0.822 | 0.002 | 0.704 | 0.758 | 0.738 | 0.733 | 0.028 |
| deception_soft | cue_gold | 0.551 / 0.554 / 0.552 | 0.552 | 0.001 | 0.724 | 0.252 | 0.619 | 0.532 | 0.248 |
| deception_soft | cue_dist | 0.599 / 0.599 / 0.601 | 0.599 | 0.001 | 0.608 | 0.494 | 0.590 | 0.564 | 0.061 |
| deception_soft | train_mix | 0.567 / 0.568 / 0.568 | 0.567 | 0.001 | 0.682 | 0.330 | 0.612 | 0.541 | 0.186 |
| deception | cue_gold | 0.570 / 0.570 / 0.569 | 0.570 | 0.000 | 0.745 | 0.211 | 0.683 | 0.546 | 0.292 |
| deception | cue_dist | 0.661 / 0.661 / 0.662 | 0.661 | 0.001 | 0.611 | 0.468 | 0.598 | 0.559 | 0.079 |
| deception | train_mix | 0.600 / 0.600 / 0.601 | 0.601 | 0.001 | 0.695 | 0.293 | 0.658 | 0.549 | 0.222 |
| biased_share | cue_gold | 0.099 / 0.091 / 0.096 | 0.095 | 0.004 | 0.245 | 0.182 | 0.199 | 0.208 | 0.032 |
| biased_share | cue_dist | 0.269 / 0.275 / 0.273 | 0.272 | 0.003 | 0.389 | 0.368 | 0.402 | 0.386 | 0.017 |
| biased_share | train_mix | 0.159 / 0.154 / 0.156 | 0.156 | 0.003 | 0.295 | 0.239 | 0.262 | 0.265 | 0.028 |
| false_denial_soft | cue_gold | 0.380 / 0.385 / 0.374 | 0.380 | 0.005 | 0.022 | 0.760 | 0.092 | 0.291 | 0.407 |
| false_denial_soft | cue_dist | 0.348 / 0.350 / 0.352 | 0.350 | 0.002 | 0.022 | 0.747 | 0.123 | 0.297 | 0.393 |
| false_denial_soft | train_mix | 0.365 / 0.364 / 0.363 | 0.364 | 0.001 | 0.020 | 0.749 | 0.106 | 0.292 | 0.399 |
| false_admission_soft | cue_gold | 0.570 / 0.571 / 0.570 | 0.570 | 0.000 | 0.952 | 0.139 | 0.750 | 0.614 | 0.423 |
| false_admission_soft | cue_dist | 0.691 / 0.693 / 0.694 | 0.693 | 0.001 | 0.981 | 0.347 | 0.904 | 0.744 | 0.346 |
| false_admission_soft | train_mix | 0.605 / 0.605 / 0.606 | 0.605 | 0.001 | 0.960 | 0.199 | 0.792 | 0.650 | 0.399 |
| said_yes_biased_soft | cue_gold | 0.620 / 0.615 / 0.626 | 0.620 | 0.005 | 0.978 | 0.240 | 0.908 | 0.709 | 0.407 |
| said_yes_biased_soft | cue_dist | 0.652 / 0.650 / 0.648 | 0.650 | 0.002 | 0.978 | 0.253 | 0.877 | 0.703 | 0.393 |
| said_yes_biased_soft | train_mix | 0.635 / 0.636 / 0.637 | 0.636 | 0.001 | 0.980 | 0.251 | 0.894 | 0.708 | 0.399 |
| deception_from_biased | cue_gold | 0.038 / 0.035 / 0.036 | 0.036 | 0.001 | 0.005 | 0.138 | 0.018 | 0.054 | 0.073 |
| deception_from_biased | cue_dist | 0.093 / 0.096 / 0.096 | 0.095 | 0.001 | 0.009 | 0.275 | 0.049 | 0.111 | 0.144 |
| deception_from_biased | train_mix | 0.058 / 0.056 / 0.057 | 0.057 | 0.001 | 0.006 | 0.179 | 0.028 | 0.071 | 0.094 |
| deception_from_unbiased | cue_gold | 0.514 / 0.519 / 0.516 | 0.516 | 0.002 | 0.719 | 0.114 | 0.601 | 0.478 | 0.321 |
| deception_from_unbiased | cue_dist | 0.505 / 0.503 / 0.505 | 0.504 | 0.001 | 0.599 | 0.219 | 0.541 | 0.453 | 0.205 |
| deception_from_unbiased | train_mix | 0.509 / 0.512 / 0.511 | 0.511 | 0.001 | 0.676 | 0.152 | 0.584 | 0.471 | 0.280 |
| said_yes | cue_gold | 0.600 / 0.596 / 0.601 | 0.599 | 0.003 | 0.989 | 0.100 | 0.881 | 0.657 | 0.485 |
| said_yes | cue_dist | 0.788 / 0.788 / 0.786 | 0.787 | 0.001 | 0.991 | 0.286 | 0.970 | 0.749 | 0.401 |
| said_yes | train_mix | 0.660 / 0.656 / 0.661 | 0.659 | 0.003 | 0.989 | 0.163 | 0.913 | 0.688 | 0.456 |
| false_denial | cue_gold | 0.349 / 0.359 / 0.333 | 0.347 | 0.013 | 0.000 | 0.805 | 0.000 | 0.268 | 0.465 |
| false_denial | cue_dist | 0.263 / 0.269 / 0.273 | 0.268 | 0.005 | 0.011 | 0.747 | 0.038 | 0.265 | 0.417 |
| false_denial | train_mix | 0.310 / 0.318 / 0.309 | 0.313 | 0.005 | 0.004 | 0.772 | 0.014 | 0.263 | 0.441 |
| false_admission | cue_gold | 0.594 / 0.591 / 0.594 | 0.593 | 0.002 | 0.986 | 0.079 | 0.852 | 0.639 | 0.489 |
| false_admission | cue_dist | 0.807 / 0.809 / 0.808 | 0.808 | 0.001 | 0.993 | 0.305 | 0.975 | 0.758 | 0.392 |
| false_admission | train_mix | 0.655 / 0.652 / 0.655 | 0.654 | 0.002 | 0.985 | 0.143 | 0.887 | 0.672 | 0.461 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.008 | 0.000 | 0.010 | 0.006 | 0.005 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 450 / 444 / 455 | 449.667 | 5.508 | 458 | 452 | 463 | 457.667 | 5.508 |
| n_bias_rows | cue_dist | 444 / 438 / 449 | 443.667 | 5.508 | 458 | 452 | 463 | 457.667 | 5.508 |
| n_bias_rows | train_mix | 894 / 882 / 904 | 893.333 | 11.015 | 916 | 904 | 926 | 915.333 | 11.015 |
| n_deception_rows | cue_gold | 435 / 428 / 439 | 434 | 5.568 | 458 | 450 | 463 | 457 | 6.557 |
| n_deception_rows | cue_dist | 439 / 433 / 444 | 438.667 | 5.508 | 457 | 451 | 463 | 457 | 6.000 |
| n_deception_rows | train_mix | 874 / 861 / 883 | 872.667 | 11.060 | 915 | 901 | 926 | 914 | 12.530 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.229 | 0.084 | 0.023 | 0.112 | 0.105 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.342 | 0.070 | 0.033 | 0.148 | 0.169 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.048 | 0.019 | 0.001 | 0.023 | 0.024 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.146 | 0.019 | 0.002 | 0.056 | 0.078 |
| probe_start_auc | cue_gold | – | – | – | 0.825 | 0.823 | 0.890 | 0.846 | 0.038 |
| probe_start_auc | cue_dist | – | – | – | 0.720 | 0.764 | 0.820 | 0.768 | 0.050 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 112 | 82 | 92 | 95.333 | 15.275 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 178 | 166 | 186 | 176.667 | 10.066 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 322 | 345 | 343 | 336.667 | 12.741 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 135 | 104 | 120 | 119.667 | 15.503 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.338 | 0.098 | 0.308 | 0.248 | 0.130 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.488 | 0.083 | 0.355 | 0.309 | 0.206 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.217 | 0.025 | 0.091 | 0.111 | 0.098 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.444 | 0.031 | 0.156 | 0.210 | 0.212 |
| probe_fresh_auc | cue_gold | – | – | – | 0.653 | 0.800 | 0.815 | 0.756 | 0.089 |
| probe_fresh_auc | cue_dist | – | – | – | 0.544 | 0.697 | 0.748 | 0.663 | 0.106 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 112 | 82 | 92 | 95.333 | 15.275 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 178 | 166 | 186 | 176.667 | 10.066 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 322 | 345 | 343 | 336.667 | 12.741 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 135 | 104 | 120 | 119.667 | 15.503 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.339 | 0.097 | 0.442 | 0.293 | 0.177 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.485 | 0.083 | 0.536 | 0.368 | 0.248 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.171 | 0.021 | 0.151 | 0.114 | 0.082 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.421 | 0.025 | 0.277 | 0.241 | 0.200 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.708 | 0.827 | 0.823 | 0.786 | 0.067 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.570 | 0.748 | 0.746 | 0.688 | 0.102 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 112 | 82 | 92 | 95.333 | 15.275 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 178 | 166 | 186 | 176.667 | 10.066 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 322 | 345 | 343 | 336.667 | 12.741 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 135 | 104 | 120 | 119.667 | 15.503 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.957 | 0.840 | 0.945 | 0.914 | 0.064 |
| probe-only groups | 0.868 | 0.753 | 0.824 | 0.815 | 0.058 |
