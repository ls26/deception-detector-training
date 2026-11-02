# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 388, s43 458, s44 455.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.616 | 0.722 | 0.742 | 0.693 | 0.068 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 1.000 | 0.914 | 0.957 | 0.957 | 0.043 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.000 | 0.434 | 0.335 | 0.256 | 0.227 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.650 | 0.746 | 0.745 | 0.714 | 0.055 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.577 | 0.640 | 0.665 | 0.627 | 0.045 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.800 | 0.759 | 0.785 | 0.781 | 0.021 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.200 | 0.470 | 0.389 | 0.353 | 0.138 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.750 | 0.735 | 0.742 | 0.742 | 0.008 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 1.000 | 0.914 | 0.957 | 0.957 | 0.043 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 1.000 | 0.566 | 0.665 | 0.744 | 0.227 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 1.000 | 0.798 | 0.854 | 0.884 | 0.104 |
| bias | cue_gold | 0.092 / 0.091 / 0.087 | 0.090 | 0.003 | 0.320 | 0.168 | 0.174 | 0.220 | 0.086 |
| bias | cue_dist | 0.251 / 0.273 / 0.272 | 0.265 | 0.012 | 0.680 | 0.303 | 0.437 | 0.474 | 0.191 |
| bias | train_mix | 0.149 / 0.153 / 0.149 | 0.150 | 0.002 | 0.437 | 0.215 | 0.258 | 0.303 | 0.118 |
| anti_bias | cue_gold | 0.034 / 0.038 / 0.038 | 0.037 | 0.002 | 0.000 | 0.000 | 0.004 | 0.001 | 0.003 |
| anti_bias | cue_dist | 0.013 / 0.011 / 0.009 | 0.011 | 0.002 | 0.000 | 0.004 | 0.002 | 0.002 | 0.002 |
| anti_bias | train_mix | 0.025 / 0.028 / 0.028 | 0.027 | 0.002 | 0.000 | 0.001 | 0.004 | 0.002 | 0.002 |
| no_bias | cue_gold | 0.874 / 0.871 / 0.875 | 0.873 | 0.002 | 0.680 | 0.832 | 0.822 | 0.778 | 0.085 |
| no_bias | cue_dist | 0.735 / 0.716 / 0.719 | 0.723 | 0.010 | 0.320 | 0.692 | 0.560 | 0.524 | 0.189 |
| no_bias | train_mix | 0.826 / 0.819 / 0.823 | 0.823 | 0.003 | 0.563 | 0.784 | 0.738 | 0.695 | 0.116 |
| deception_soft | cue_gold | 0.548 / 0.551 / 0.554 | 0.551 | 0.003 | 0.408 | 0.742 | 0.476 | 0.542 | 0.176 |
| deception_soft | cue_dist | 0.608 / 0.598 / 0.598 | 0.601 | 0.006 | 0.480 | 0.686 | 0.539 | 0.568 | 0.106 |
| deception_soft | train_mix | 0.567 / 0.566 / 0.568 | 0.567 | 0.001 | 0.426 | 0.722 | 0.504 | 0.551 | 0.154 |
| deception | cue_gold | 0.559 / 0.567 / 0.574 | 0.567 | 0.008 | 0.412 | 0.755 | 0.459 | 0.542 | 0.186 |
| deception | cue_dist | 0.673 / 0.656 / 0.657 | 0.662 | 0.010 | 0.456 | 0.691 | 0.575 | 0.574 | 0.117 |
| deception | train_mix | 0.596 / 0.597 / 0.600 | 0.598 | 0.002 | 0.417 | 0.732 | 0.510 | 0.553 | 0.162 |
| biased_share | cue_gold | 0.095 / 0.094 / 0.091 | 0.094 | 0.002 | 0.320 | 0.168 | 0.174 | 0.221 | 0.086 |
| biased_share | cue_dist | 0.255 / 0.276 / 0.275 | 0.268 | 0.012 | 0.680 | 0.305 | 0.438 | 0.475 | 0.190 |
| biased_share | train_mix | 0.153 / 0.157 / 0.153 | 0.155 | 0.002 | 0.437 | 0.215 | 0.259 | 0.304 | 0.117 |
| false_denial_soft | cue_gold | 0.375 / 0.382 / 0.379 | 0.379 | 0.004 | 0.608 | 0.020 | 0.285 | 0.304 | 0.295 |
| false_denial_soft | cue_dist | 0.345 / 0.349 / 0.351 | 0.348 | 0.003 | 0.559 | 0.025 | 0.345 | 0.310 | 0.269 |
| false_denial_soft | train_mix | 0.358 / 0.365 / 0.363 | 0.362 | 0.004 | 0.572 | 0.022 | 0.322 | 0.305 | 0.275 |
| false_admission_soft | cue_gold | 0.567 / 0.569 / 0.571 | 0.569 | 0.002 | 0.315 | 0.888 | 0.517 | 0.573 | 0.291 |
| false_admission_soft | cue_dist | 0.698 / 0.693 / 0.692 | 0.694 | 0.003 | 0.313 | 0.976 | 0.691 | 0.660 | 0.333 |
| false_admission_soft | train_mix | 0.605 / 0.604 / 0.605 | 0.605 | 0.001 | 0.312 | 0.914 | 0.567 | 0.598 | 0.302 |
| said_yes_biased_soft | cue_gold | 0.625 / 0.618 / 0.621 | 0.621 | 0.004 | 0.392 | 0.980 | 0.715 | 0.696 | 0.295 |
| said_yes_biased_soft | cue_dist | 0.655 / 0.651 / 0.649 | 0.652 | 0.003 | 0.441 | 0.975 | 0.655 | 0.690 | 0.269 |
| said_yes_biased_soft | train_mix | 0.642 / 0.635 / 0.637 | 0.638 | 0.004 | 0.428 | 0.978 | 0.678 | 0.695 | 0.275 |
| deception_from_biased | cue_gold | 0.036 / 0.036 / 0.034 | 0.035 | 0.001 | 0.194 | 0.003 | 0.050 | 0.082 | 0.100 |
| deception_from_biased | cue_dist | 0.088 / 0.096 / 0.096 | 0.093 | 0.005 | 0.380 | 0.008 | 0.151 | 0.180 | 0.188 |
| deception_from_biased | train_mix | 0.055 / 0.057 / 0.056 | 0.056 | 0.001 | 0.250 | 0.005 | 0.084 | 0.113 | 0.125 |
| deception_from_unbiased | cue_gold | 0.513 / 0.515 / 0.519 | 0.516 | 0.003 | 0.214 | 0.739 | 0.427 | 0.460 | 0.264 |
| deception_from_unbiased | cue_dist | 0.520 / 0.502 / 0.502 | 0.508 | 0.010 | 0.100 | 0.678 | 0.388 | 0.389 | 0.289 |
| deception_from_unbiased | train_mix | 0.512 / 0.509 / 0.512 | 0.511 | 0.002 | 0.176 | 0.717 | 0.420 | 0.438 | 0.271 |
| said_yes | cue_gold | 0.589 / 0.597 / 0.605 | 0.597 | 0.008 | 0.351 | 0.924 | 0.563 | 0.612 | 0.290 |
| said_yes | cue_dist | 0.799 / 0.786 / 0.785 | 0.790 | 0.008 | 0.415 | 0.996 | 0.727 | 0.712 | 0.291 |
| said_yes | train_mix | 0.657 / 0.657 / 0.662 | 0.659 | 0.003 | 0.377 | 0.947 | 0.621 | 0.648 | 0.286 |
| false_denial | cue_gold | 0.343 / 0.341 / 0.333 | 0.339 | 0.005 | 0.597 | 0.000 | 0.203 | 0.266 | 0.303 |
| false_denial | cue_dist | 0.253 / 0.264 / 0.267 | 0.261 | 0.008 | 0.530 | 0.000 | 0.327 | 0.286 | 0.268 |
| false_denial | train_mix | 0.301 / 0.307 / 0.298 | 0.302 | 0.004 | 0.546 | 0.000 | 0.287 | 0.278 | 0.273 |
| false_admission | cue_gold | 0.581 / 0.590 / 0.598 | 0.590 | 0.009 | 0.326 | 0.908 | 0.513 | 0.582 | 0.297 |
| false_admission | cue_dist | 0.817 / 0.805 / 0.804 | 0.809 | 0.007 | 0.298 | 0.994 | 0.769 | 0.687 | 0.355 |
| false_admission | train_mix | 0.650 / 0.651 / 0.655 | 0.652 | 0.003 | 0.317 | 0.932 | 0.588 | 0.613 | 0.308 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.086 | 0.014 | 0.006 | 0.035 | 0.044 |
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
| n_bias_rows | cue_gold | 380 / 451 / 447 | 426 | 39.887 | 388 | 458 | 455 | 433.667 | 39.577 |
| n_bias_rows | cue_dist | 378 / 444 / 441 | 421 | 37.269 | 388 | 458 | 455 | 433.667 | 39.577 |
| n_bias_rows | train_mix | 758 / 895 / 888 | 847 | 77.156 | 776 | 916 | 910 | 867.333 | 79.154 |
| n_deception_rows | cue_gold | 367 / 434 / 430 | 410.333 | 37.581 | 388 | 458 | 453 | 433 | 39.051 |
| n_deception_rows | cue_dist | 373 / 439 / 437 | 416.333 | 37.541 | 388 | 456 | 454 | 432.667 | 38.695 |
| n_deception_rows | train_mix | 740 / 873 / 867 | 826.667 | 75.115 | 776 | 914 | 907 | 865.667 | 77.732 |
| probe_start_mean_biased | cue_gold | – | – | – | 0.004 | 0.238 | 0.088 | 0.110 | 0.119 |
| probe_start_mean_biased | cue_dist | – | – | – | 0.006 | 0.294 | 0.077 | 0.125 | 0.150 |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.004 | 0.028 | 0.026 | 0.019 | 0.013 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.004 | 0.040 | 0.029 | 0.024 | 0.019 |
| probe_start_auc | cue_gold | – | – | – | 0.489 | 0.902 | 0.763 | 0.718 | 0.210 |
| probe_start_auc | cue_dist | – | – | – | 0.407 | 0.887 | 0.677 | 0.657 | 0.241 |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 124 | 77 | 79 | 93.333 | 26.577 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 264 | 139 | 199 | 200.667 | 62.517 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 264 | 342 | 356 | 320.667 | 49.571 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 124 | 114 | 96 | 111.333 | 14.189 |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.063 | 0.252 | 0.261 | 0.192 | 0.112 |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.071 | 0.329 | 0.261 | 0.221 | 0.134 |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.056 | 0.066 | 0.134 | 0.085 | 0.042 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.071 | 0.073 | 0.126 | 0.090 | 0.032 |
| probe_fresh_auc | cue_gold | – | – | – | 0.464 | 0.845 | 0.750 | 0.686 | 0.198 |
| probe_fresh_auc | cue_dist | – | – | – | 0.419 | 0.860 | 0.750 | 0.676 | 0.230 |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 124 | 77 | 79 | 93.333 | 26.577 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 264 | 139 | 199 | 200.667 | 62.517 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 264 | 342 | 356 | 320.667 | 49.571 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 124 | 114 | 96 | 111.333 | 14.189 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.116 | 0.332 | 0.130 | 0.193 | 0.121 |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.176 | 0.377 | 0.134 | 0.229 | 0.130 |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.078 | 0.112 | 0.036 | 0.075 | 0.038 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.162 | 0.193 | 0.044 | 0.133 | 0.079 |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.570 | 0.833 | 0.810 | 0.738 | 0.146 |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.494 | 0.736 | 0.740 | 0.657 | 0.141 |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 124 | 77 | 79 | 93.333 | 26.577 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 264 | 139 | 199 | 200.667 | 62.517 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 264 | 342 | 356 | 320.667 | 49.571 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 124 | 114 | 96 | 111.333 | 14.189 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.921 | 0.940 | 0.821 | 0.894 | 0.064 |
| probe-only groups | 0.842 | 0.750 | 0.715 | 0.769 | 0.065 |
