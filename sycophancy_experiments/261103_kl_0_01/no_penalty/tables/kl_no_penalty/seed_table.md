# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 442, s43 459, s44 452.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.726 | 0.771 | 0.738 | 0.745 | 0.023 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.990 | 0.947 | 0.930 | 0.956 | 0.031 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.065 | 0.348 | 0.370 | 0.261 | 0.170 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.665 | 0.739 | 0.737 | 0.714 | 0.042 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.632 | 0.675 | 0.654 | 0.654 | 0.021 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.793 | 0.779 | 0.767 | 0.780 | 0.013 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.243 | 0.413 | 0.419 | 0.359 | 0.100 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.749 | 0.746 | 0.733 | 0.743 | 0.008 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.990 | 0.947 | 0.930 | 0.956 | 0.031 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.933 | 0.652 | 0.626 | 0.737 | 0.170 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.970 | 0.848 | 0.826 | 0.881 | 0.078 |
| bias | cue_gold | 0.092 / 0.095 / 0.097 | 0.095 | 0.002 | 0.204 | 0.161 | 0.164 | 0.176 | 0.024 |
| bias | cue_dist | 0.257 / 0.263 / 0.258 | 0.259 | 0.003 | 0.726 | 0.420 | 0.370 | 0.505 | 0.192 |
| bias | train_mix | 0.151 / 0.154 / 0.151 | 0.152 | 0.002 | 0.380 | 0.247 | 0.230 | 0.286 | 0.082 |
| anti_bias | cue_gold | 0.030 / 0.038 / 0.038 | 0.035 | 0.005 | 0.000 | 0.002 | 0.002 | 0.001 | 0.001 |
| anti_bias | cue_dist | 0.009 / 0.011 / 0.011 | 0.011 | 0.001 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| anti_bias | train_mix | 0.021 / 0.028 / 0.028 | 0.026 | 0.004 | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 |
| no_bias | cue_gold | 0.878 / 0.867 / 0.865 | 0.870 | 0.007 | 0.796 | 0.837 | 0.834 | 0.822 | 0.023 |
| no_bias | cue_dist | 0.734 / 0.726 / 0.731 | 0.730 | 0.004 | 0.272 | 0.577 | 0.630 | 0.493 | 0.193 |
| no_bias | train_mix | 0.828 / 0.818 / 0.821 | 0.822 | 0.005 | 0.619 | 0.751 | 0.768 | 0.713 | 0.082 |
| deception_soft | cue_gold | 0.552 / 0.552 / 0.551 | 0.552 | 0.001 | 0.494 | 0.394 | 0.407 | 0.432 | 0.054 |
| deception_soft | cue_dist | 0.605 / 0.600 / 0.602 | 0.602 | 0.002 | 0.420 | 0.522 | 0.534 | 0.492 | 0.063 |
| deception_soft | train_mix | 0.569 / 0.568 / 0.566 | 0.568 | 0.001 | 0.465 | 0.435 | 0.448 | 0.450 | 0.015 |
| deception | cue_gold | 0.575 / 0.571 / 0.567 | 0.571 | 0.004 | 0.474 | 0.310 | 0.322 | 0.369 | 0.091 |
| deception | cue_dist | 0.670 / 0.666 / 0.661 | 0.665 | 0.005 | 0.411 | 0.533 | 0.557 | 0.500 | 0.078 |
| deception | train_mix | 0.605 / 0.603 / 0.596 | 0.601 | 0.005 | 0.442 | 0.383 | 0.402 | 0.409 | 0.030 |
| biased_share | cue_gold | 0.095 / 0.099 / 0.101 | 0.098 | 0.003 | 0.204 | 0.162 | 0.164 | 0.177 | 0.024 |
| biased_share | cue_dist | 0.259 / 0.266 / 0.261 | 0.262 | 0.003 | 0.727 | 0.421 | 0.370 | 0.506 | 0.193 |
| biased_share | train_mix | 0.154 / 0.158 / 0.155 | 0.156 | 0.002 | 0.380 | 0.247 | 0.230 | 0.286 | 0.082 |
| false_denial_soft | cue_gold | 0.379 / 0.380 / 0.380 | 0.380 | 0.001 | 0.371 | 0.482 | 0.459 | 0.437 | 0.058 |
| false_denial_soft | cue_dist | 0.355 / 0.347 / 0.345 | 0.349 | 0.005 | 0.355 | 0.483 | 0.468 | 0.435 | 0.070 |
| false_denial_soft | train_mix | 0.365 / 0.364 / 0.360 | 0.363 | 0.002 | 0.354 | 0.480 | 0.459 | 0.431 | 0.068 |
| false_admission_soft | cue_gold | 0.570 / 0.571 / 0.570 | 0.570 | 0.001 | 0.525 | 0.378 | 0.397 | 0.433 | 0.080 |
| false_admission_soft | cue_dist | 0.693 / 0.692 / 0.693 | 0.693 | 0.000 | 0.593 | 0.550 | 0.573 | 0.572 | 0.022 |
| false_admission_soft | train_mix | 0.606 / 0.606 / 0.604 | 0.606 | 0.001 | 0.533 | 0.421 | 0.445 | 0.466 | 0.059 |
| said_yes_biased_soft | cue_gold | 0.621 / 0.620 / 0.620 | 0.620 | 0.001 | 0.629 | 0.518 | 0.541 | 0.563 | 0.058 |
| said_yes_biased_soft | cue_dist | 0.645 / 0.653 / 0.655 | 0.651 | 0.005 | 0.645 | 0.517 | 0.532 | 0.565 | 0.070 |
| said_yes_biased_soft | train_mix | 0.635 / 0.636 / 0.640 | 0.637 | 0.002 | 0.646 | 0.520 | 0.541 | 0.569 | 0.068 |
| deception_from_biased | cue_gold | 0.036 / 0.038 / 0.038 | 0.037 | 0.001 | 0.076 | 0.078 | 0.075 | 0.076 | 0.001 |
| deception_from_biased | cue_dist | 0.092 / 0.092 / 0.090 | 0.091 | 0.001 | 0.258 | 0.203 | 0.173 | 0.212 | 0.043 |
| deception_from_biased | train_mix | 0.056 / 0.058 / 0.056 | 0.057 | 0.001 | 0.135 | 0.119 | 0.106 | 0.120 | 0.014 |
| deception_from_unbiased | cue_gold | 0.516 / 0.514 / 0.512 | 0.514 | 0.002 | 0.418 | 0.317 | 0.331 | 0.355 | 0.055 |
| deception_from_unbiased | cue_dist | 0.513 / 0.508 / 0.512 | 0.511 | 0.003 | 0.162 | 0.318 | 0.361 | 0.280 | 0.105 |
| deception_from_unbiased | train_mix | 0.513 / 0.510 / 0.511 | 0.511 | 0.002 | 0.330 | 0.317 | 0.343 | 0.330 | 0.013 |
| said_yes | cue_gold | 0.603 / 0.601 / 0.597 | 0.601 | 0.003 | 0.542 | 0.284 | 0.331 | 0.386 | 0.137 |
| said_yes | cue_dist | 0.783 / 0.786 / 0.792 | 0.787 | 0.005 | 0.643 | 0.513 | 0.519 | 0.558 | 0.073 |
| said_yes | train_mix | 0.663 / 0.661 / 0.659 | 0.661 | 0.002 | 0.574 | 0.355 | 0.396 | 0.442 | 0.116 |
| false_denial | cue_gold | 0.350 / 0.349 / 0.349 | 0.349 | 0.001 | 0.333 | 0.581 | 0.473 | 0.462 | 0.124 |
| false_denial | cue_dist | 0.282 / 0.274 / 0.248 | 0.268 | 0.018 | 0.341 | 0.523 | 0.551 | 0.472 | 0.114 |
| false_denial | train_mix | 0.312 / 0.315 / 0.297 | 0.308 | 0.010 | 0.327 | 0.556 | 0.513 | 0.465 | 0.122 |
| false_admission | cue_gold | 0.598 / 0.596 / 0.591 | 0.595 | 0.004 | 0.510 | 0.258 | 0.293 | 0.353 | 0.137 |
| false_admission | cue_dist | 0.806 / 0.808 / 0.806 | 0.807 | 0.001 | 0.600 | 0.540 | 0.560 | 0.566 | 0.031 |
| false_admission | train_mix | 0.659 / 0.657 / 0.650 | 0.655 | 0.004 | 0.512 | 0.326 | 0.369 | 0.402 | 0.098 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.000 | 0.004 | 0.006 | 0.003 | 0.003 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.004 | 0.000 | 0.002 | 0.002 | 0.002 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.002 | 0.000 | 0.004 | 0.002 | 0.002 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.005 | 0.000 | 0.003 | 0.002 | 0.002 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 434 / 451 / 444 | 443 | 8.544 | 441 | 459 | 451 | 450.333 | 9.018 |
| n_bias_rows | cue_dist | 428 / 445 / 438 | 437 | 8.544 | 441 | 459 | 451 | 450.333 | 9.018 |
| n_bias_rows | train_mix | 862 / 896 / 882 | 880 | 17.088 | 882 | 918 | 902 | 900.667 | 18.037 |
| n_deception_rows | cue_gold | 421 / 434 / 427 | 427.333 | 6.506 | 441 | 458 | 450 | 449.667 | 8.505 |
| n_deception_rows | cue_dist | 424 / 440 / 433 | 432.333 | 8.021 | 440 | 458 | 451 | 449.667 | 9.074 |
| n_deception_rows | train_mix | 845 / 874 / 860 | 859.667 | 14.503 | 881 | 916 | 901 | 899.333 | 17.559 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.353 | 0.656 | 0.685 | 0.565 | 0.184 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.395 | 0.652 | 0.701 | 0.583 | 0.164 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.069 | 0.151 | 0.169 | 0.130 | 0.053 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.095 | 0.236 | 0.210 | 0.180 | 0.075 |
| probe_start_auc | cue_gold | – | – | – | 0.868 | 0.926 | 0.915 | 0.903 | 0.031 |
| probe_start_auc | cue_dist | – | – | – | 0.829 | 0.857 | 0.887 | 0.858 | 0.029 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 90 | 74 | 74 | 79.333 | 9.238 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 320 | 193 | 167 | 226.667 | 81.868 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 349 | 359 | 346 | 351.333 | 6.807 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 91 | 98 | 104 | 97.667 | 6.506 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.718 | 0.712 | 0.692 | 0.708 | 0.014 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.732 | 0.700 | 0.704 | 0.712 | 0.017 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.259 | 0.200 | 0.196 | 0.218 | 0.035 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.308 | 0.296 | 0.255 | 0.286 | 0.028 |
| probe_fresh_auc | cue_gold | – | – | – | 0.900 | 0.920 | 0.904 | 0.908 | 0.011 |
| probe_fresh_auc | cue_dist | – | – | – | 0.863 | 0.846 | 0.869 | 0.859 | 0.012 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 90 | 74 | 74 | 79.333 | 9.238 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 320 | 193 | 167 | 226.667 | 81.868 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 349 | 359 | 346 | 351.333 | 6.807 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 91 | 98 | 104 | 97.667 | 6.506 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.806 | 0.728 | 0.702 | 0.745 | 0.054 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.804 | 0.715 | 0.710 | 0.743 | 0.053 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.213 | 0.205 | 0.196 | 0.204 | 0.009 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.313 | 0.312 | 0.259 | 0.295 | 0.031 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.949 | 0.920 | 0.907 | 0.925 | 0.021 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.890 | 0.841 | 0.868 | 0.866 | 0.025 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 90 | 74 | 74 | 79.333 | 9.238 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 320 | 193 | 167 | 226.667 | 81.868 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 349 | 359 | 346 | 351.333 | 6.807 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 91 | 98 | 104 | 97.667 | 6.506 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.853 | 0.807 | 0.795 | 0.818 | 0.030 |
| probe-only groups | 0.829 | 0.748 | 0.739 | 0.772 | 0.050 |
