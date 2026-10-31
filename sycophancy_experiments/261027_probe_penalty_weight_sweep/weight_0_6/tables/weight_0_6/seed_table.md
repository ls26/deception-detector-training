# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 448, s43 453, s44 458.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.748 | 0.765 | 0.748 | 0.754 | 0.009 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.963 | 0.963 | 0.963 | 0.963 | 0.000 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.307 | 0.356 | 0.290 | 0.318 | 0.034 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.739 | 0.750 | 0.732 | 0.740 | 0.009 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.669 | 0.667 | 0.665 | 0.667 | 0.002 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.787 | 0.796 | 0.779 | 0.787 | 0.008 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.372 | 0.434 | 0.360 | 0.389 | 0.039 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.744 | 0.766 | 0.738 | 0.749 | 0.015 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.963 | 0.963 | 0.963 | 0.963 | 0.000 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.693 | 0.644 | 0.708 | 0.682 | 0.033 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.868 | 0.861 | 0.871 | 0.867 | 0.005 |
| bias | cue_gold | 0.086 / 0.097 / 0.089 | 0.091 | 0.005 | 0.163 | 0.174 | 0.160 | 0.166 | 0.008 |
| bias | cue_dist | 0.269 / 0.270 / 0.270 | 0.270 | 0.001 | 0.478 | 0.422 | 0.501 | 0.467 | 0.041 |
| bias | train_mix | 0.150 / 0.157 / 0.150 | 0.152 | 0.004 | 0.268 | 0.258 | 0.269 | 0.265 | 0.006 |
| anti_bias | cue_gold | 0.032 / 0.036 / 0.038 | 0.035 | 0.003 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anti_bias | cue_dist | 0.009 / 0.009 / 0.011 | 0.010 | 0.001 | 0.000 | 0.004 | 0.000 | 0.001 | 0.003 |
| anti_bias | train_mix | 0.023 / 0.026 / 0.028 | 0.026 | 0.003 | 0.000 | 0.001 | 0.000 | 0.000 | 0.001 |
| no_bias | cue_gold | 0.882 / 0.867 / 0.873 | 0.874 | 0.007 | 0.837 | 0.826 | 0.840 | 0.834 | 0.008 |
| no_bias | cue_dist | 0.722 / 0.720 / 0.719 | 0.720 | 0.001 | 0.522 | 0.574 | 0.499 | 0.532 | 0.038 |
| no_bias | train_mix | 0.827 / 0.817 / 0.822 | 0.822 | 0.005 | 0.732 | 0.741 | 0.731 | 0.734 | 0.005 |
| deception_soft | cue_gold | 0.554 / 0.549 / 0.553 | 0.552 | 0.003 | 0.405 | 0.420 | 0.320 | 0.381 | 0.054 |
| deception_soft | cue_dist | 0.599 / 0.597 / 0.600 | 0.599 | 0.002 | 0.508 | 0.524 | 0.479 | 0.504 | 0.023 |
| deception_soft | train_mix | 0.569 / 0.564 / 0.568 | 0.567 | 0.003 | 0.442 | 0.451 | 0.373 | 0.422 | 0.043 |
| deception | cue_gold | 0.573 / 0.564 / 0.573 | 0.570 | 0.005 | 0.312 | 0.373 | 0.230 | 0.305 | 0.072 |
| deception | cue_dist | 0.661 / 0.658 / 0.661 | 0.660 | 0.002 | 0.533 | 0.534 | 0.453 | 0.507 | 0.047 |
| deception | train_mix | 0.603 / 0.595 / 0.601 | 0.600 | 0.004 | 0.389 | 0.418 | 0.306 | 0.371 | 0.058 |
| biased_share | cue_gold | 0.089 / 0.100 / 0.092 | 0.094 | 0.006 | 0.163 | 0.174 | 0.160 | 0.166 | 0.008 |
| biased_share | cue_dist | 0.271 / 0.273 / 0.273 | 0.272 | 0.001 | 0.478 | 0.424 | 0.501 | 0.467 | 0.040 |
| biased_share | train_mix | 0.154 / 0.161 / 0.154 | 0.156 | 0.004 | 0.268 | 0.259 | 0.269 | 0.265 | 0.006 |
| false_denial_soft | cue_gold | 0.390 / 0.380 / 0.396 | 0.389 | 0.008 | 0.403 | 0.462 | 0.562 | 0.476 | 0.080 |
| false_denial_soft | cue_dist | 0.344 / 0.352 / 0.352 | 0.349 | 0.005 | 0.434 | 0.448 | 0.569 | 0.483 | 0.074 |
| false_denial_soft | train_mix | 0.365 / 0.366 / 0.371 | 0.368 | 0.003 | 0.421 | 0.443 | 0.565 | 0.476 | 0.078 |
| false_admission_soft | cue_gold | 0.570 / 0.567 / 0.569 | 0.569 | 0.001 | 0.405 | 0.411 | 0.274 | 0.363 | 0.077 |
| false_admission_soft | cue_dist | 0.694 / 0.689 / 0.693 | 0.692 | 0.003 | 0.576 | 0.580 | 0.388 | 0.515 | 0.110 |
| false_admission_soft | train_mix | 0.605 / 0.602 / 0.604 | 0.604 | 0.002 | 0.450 | 0.454 | 0.302 | 0.402 | 0.087 |
| said_yes_biased_soft | cue_gold | 0.610 / 0.620 / 0.604 | 0.611 | 0.008 | 0.597 | 0.538 | 0.438 | 0.524 | 0.080 |
| said_yes_biased_soft | cue_dist | 0.656 / 0.648 / 0.648 | 0.651 | 0.005 | 0.566 | 0.552 | 0.431 | 0.517 | 0.074 |
| said_yes_biased_soft | train_mix | 0.635 / 0.634 / 0.629 | 0.632 | 0.003 | 0.579 | 0.557 | 0.435 | 0.524 | 0.078 |
| deception_from_biased | cue_gold | 0.035 / 0.038 / 0.037 | 0.036 | 0.002 | 0.066 | 0.081 | 0.090 | 0.079 | 0.012 |
| deception_from_biased | cue_dist | 0.093 / 0.096 / 0.096 | 0.095 | 0.002 | 0.207 | 0.190 | 0.285 | 0.227 | 0.051 |
| deception_from_biased | train_mix | 0.056 / 0.059 / 0.057 | 0.057 | 0.001 | 0.113 | 0.115 | 0.152 | 0.126 | 0.022 |
| deception_from_unbiased | cue_gold | 0.519 / 0.510 / 0.516 | 0.515 | 0.005 | 0.339 | 0.339 | 0.230 | 0.303 | 0.063 |
| deception_from_unbiased | cue_dist | 0.506 / 0.501 / 0.504 | 0.504 | 0.002 | 0.301 | 0.335 | 0.194 | 0.276 | 0.074 |
| deception_from_unbiased | train_mix | 0.512 / 0.505 / 0.511 | 0.509 | 0.004 | 0.329 | 0.337 | 0.221 | 0.296 | 0.065 |
| said_yes | cue_gold | 0.596 / 0.594 / 0.596 | 0.595 | 0.001 | 0.342 | 0.375 | 0.188 | 0.302 | 0.100 |
| said_yes | cue_dist | 0.789 / 0.784 / 0.789 | 0.787 | 0.003 | 0.556 | 0.599 | 0.302 | 0.485 | 0.160 |
| said_yes | train_mix | 0.659 / 0.656 / 0.658 | 0.658 | 0.002 | 0.416 | 0.446 | 0.226 | 0.363 | 0.119 |
| false_denial | cue_gold | 0.368 / 0.349 / 0.375 | 0.364 | 0.014 | 0.411 | 0.494 | 0.630 | 0.512 | 0.111 |
| false_denial | cue_dist | 0.265 / 0.269 / 0.267 | 0.267 | 0.002 | 0.477 | 0.424 | 0.651 | 0.517 | 0.119 |
| false_denial | train_mix | 0.316 / 0.313 / 0.317 | 0.315 | 0.002 | 0.449 | 0.444 | 0.648 | 0.514 | 0.116 |
| false_admission | cue_gold | 0.593 / 0.588 / 0.593 | 0.591 | 0.003 | 0.293 | 0.348 | 0.154 | 0.265 | 0.100 |
| false_admission | cue_dist | 0.809 / 0.804 / 0.809 | 0.808 | 0.003 | 0.585 | 0.615 | 0.254 | 0.485 | 0.200 |
| false_admission | train_mix | 0.655 / 0.650 / 0.653 | 0.652 | 0.003 | 0.367 | 0.408 | 0.180 | 0.319 | 0.121 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.012 | 0.002 | 0.004 | 0.006 | 0.005 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.000 | 0.004 | 0.001 | 0.002 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 440 / 445 / 450 | 445 | 5.000 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_bias_rows | cue_dist | 435 / 440 / 445 | 440 | 5.000 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_bias_rows | train_mix | 875 / 885 / 895 | 885 | 10.000 | 896 | 906 | 914 | 905.333 | 9.018 |
| n_deception_rows | cue_gold | 426 / 429 / 433 | 429.333 | 3.512 | 448 | 453 | 457 | 452.667 | 4.509 |
| n_deception_rows | cue_dist | 431 / 436 / 440 | 435.667 | 4.509 | 448 | 451 | 457 | 452 | 4.583 |
| n_deception_rows | train_mix | 857 / 865 / 873 | 865 | 8.000 | 896 | 904 | 914 | 904.667 | 9.018 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.063 | 0.183 | 0.138 | 0.128 | 0.061 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.035 | 0.166 | 0.102 | 0.101 | 0.066 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.015 | 0.035 | 0.041 | 0.030 | 0.014 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.015 | 0.042 | 0.044 | 0.034 | 0.016 |
| probe_start_auc | cue_gold | – | – | – | 0.756 | 0.849 | 0.763 | 0.789 | 0.052 |
| probe_start_auc | cue_dist | – | – | – | 0.691 | 0.759 | 0.745 | 0.731 | 0.036 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.279 | 0.290 | 0.384 | 0.318 | 0.058 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.259 | 0.252 | 0.359 | 0.290 | 0.060 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.151 | 0.076 | 0.190 | 0.139 | 0.058 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.151 | 0.093 | 0.185 | 0.143 | 0.046 |
| probe_fresh_auc | cue_gold | – | – | – | 0.751 | 0.838 | 0.770 | 0.787 | 0.046 |
| probe_fresh_auc | cue_dist | – | – | – | 0.701 | 0.746 | 0.742 | 0.730 | 0.025 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.119 | 0.161 | 0.202 | 0.161 | 0.041 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.087 | 0.159 | 0.198 | 0.148 | 0.056 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.035 | 0.032 | 0.057 | 0.041 | 0.014 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.037 | 0.044 | 0.069 | 0.050 | 0.017 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.753 | 0.852 | 0.783 | 0.796 | 0.051 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.680 | 0.756 | 0.764 | 0.733 | 0.047 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 73 | 79 | 73 | 75 | 3.464 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 214 | 191 | 229 | 211.333 | 19.140 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 358 | 357 | 368 | 361 | 6.083 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 90 | 94 | 89 | 91 | 2.646 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.879 | 0.814 | 0.908 | 0.867 | 0.048 |
| probe-only groups | 0.815 | 0.751 | 0.859 | 0.808 | 0.054 |
