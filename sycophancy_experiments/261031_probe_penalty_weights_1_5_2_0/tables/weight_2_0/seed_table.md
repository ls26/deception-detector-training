# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 466, s43 445, s44 453.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.526 | 0.748 | 0.755 | 0.676 | 0.131 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.526 | 0.975 | 0.933 | 0.811 | 0.248 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.526 | 0.221 | 0.376 | 0.374 | 0.152 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.526 | 0.715 | 0.740 | 0.660 | 0.117 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.538 | 0.654 | 0.669 | 0.620 | 0.072 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.538 | 0.796 | 0.769 | 0.701 | 0.142 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.538 | 0.327 | 0.425 | 0.430 | 0.105 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.538 | 0.750 | 0.736 | 0.675 | 0.119 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.526 | 0.975 | 0.933 | 0.811 | 0.248 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.474 | 0.779 | 0.624 | 0.626 | 0.152 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.519 | 0.908 | 0.827 | 0.751 | 0.205 |
| bias | cue_gold | 0.094 / 0.096 / 0.092 | 0.094 | 0.002 | 0.000 | 0.204 | 0.157 | 0.120 | 0.107 |
| bias | cue_dist | 0.270 / 0.253 / 0.276 | 0.266 | 0.012 | 0.000 | 0.530 | 0.380 | 0.303 | 0.273 |
| bias | train_mix | 0.154 / 0.148 / 0.154 | 0.152 | 0.004 | 0.000 | 0.303 | 0.228 | 0.177 | 0.158 |
| anti_bias | cue_gold | 0.037 / 0.037 / 0.038 | 0.037 | 0.001 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| anti_bias | cue_dist | 0.011 / 0.009 / 0.011 | 0.011 | 0.001 | 0.000 | 0.004 | 0.000 | 0.001 | 0.003 |
| anti_bias | train_mix | 0.027 / 0.026 / 0.028 | 0.027 | 0.001 | 0.000 | 0.001 | 0.002 | 0.001 | 0.001 |
| no_bias | cue_gold | 0.869 / 0.867 / 0.870 | 0.869 | 0.001 | 1.000 | 0.796 | 0.841 | 0.879 | 0.107 |
| no_bias | cue_dist | 0.719 / 0.738 / 0.713 | 0.723 | 0.013 | 1.000 | 0.465 | 0.620 | 0.695 | 0.275 |
| no_bias | train_mix | 0.818 / 0.826 / 0.818 | 0.821 | 0.004 | 1.000 | 0.696 | 0.770 | 0.822 | 0.158 |
| deception_soft | cue_gold | 0.552 / 0.553 / 0.551 | 0.552 | 0.001 | 0.571 | 0.517 | 0.356 | 0.481 | 0.112 |
| deception_soft | cue_dist | 0.600 / 0.605 / 0.597 | 0.601 | 0.004 | 0.575 | 0.499 | 0.482 | 0.518 | 0.050 |
| deception_soft | train_mix | 0.568 / 0.569 / 0.565 | 0.567 | 0.002 | 0.572 | 0.513 | 0.395 | 0.494 | 0.090 |
| deception | cue_gold | 0.571 / 0.572 / 0.568 | 0.571 | 0.002 | 0.624 | 0.537 | 0.265 | 0.476 | 0.187 |
| deception | cue_dist | 0.662 / 0.670 / 0.657 | 0.663 | 0.007 | 0.622 | 0.508 | 0.459 | 0.530 | 0.084 |
| deception | train_mix | 0.602 / 0.602 / 0.597 | 0.600 | 0.003 | 0.622 | 0.528 | 0.326 | 0.492 | 0.151 |
| biased_share | cue_gold | 0.098 / 0.100 / 0.096 | 0.098 | 0.002 | 0.000 | 0.204 | 0.157 | 0.121 | 0.107 |
| biased_share | cue_dist | 0.273 / 0.255 / 0.279 | 0.269 | 0.012 | 0.000 | 0.533 | 0.380 | 0.304 | 0.274 |
| biased_share | train_mix | 0.159 / 0.152 / 0.158 | 0.156 | 0.004 | 0.000 | 0.303 | 0.228 | 0.177 | 0.158 |
| false_denial_soft | cue_gold | 0.380 / 0.375 / 0.373 | 0.376 | 0.004 | – | 0.371 | 0.498 | – | – |
| false_denial_soft | cue_dist | 0.352 / 0.353 / 0.350 | 0.352 | 0.001 | – | 0.340 | 0.521 | – | – |
| false_denial_soft | train_mix | 0.366 / 0.363 / 0.361 | 0.363 | 0.003 | – | 0.354 | 0.499 | – | – |
| false_admission_soft | cue_gold | 0.571 / 0.572 / 0.570 | 0.571 | 0.001 | 0.571 | 0.555 | 0.330 | 0.485 | 0.135 |
| false_admission_soft | cue_dist | 0.694 / 0.692 / 0.692 | 0.693 | 0.001 | 0.575 | 0.680 | 0.457 | 0.571 | 0.111 |
| false_admission_soft | train_mix | 0.606 / 0.606 / 0.604 | 0.605 | 0.001 | 0.572 | 0.583 | 0.365 | 0.507 | 0.123 |
| said_yes_biased_soft | cue_gold | 0.620 / 0.625 / 0.627 | 0.624 | 0.004 | – | 0.629 | 0.502 | – | – |
| said_yes_biased_soft | cue_dist | 0.648 / 0.647 / 0.650 | 0.648 | 0.001 | – | 0.660 | 0.479 | – | – |
| said_yes_biased_soft | train_mix | 0.634 / 0.637 / 0.639 | 0.637 | 0.003 | – | 0.646 | 0.501 | – | – |
| deception_from_biased | cue_gold | 0.037 / 0.037 / 0.036 | 0.037 | 0.001 | – | 0.076 | 0.078 | – | – |
| deception_from_biased | cue_dist | 0.096 / 0.090 / 0.098 | 0.095 | 0.004 | – | 0.181 | 0.198 | – | – |
| deception_from_biased | train_mix | 0.058 / 0.055 / 0.057 | 0.057 | 0.002 | – | 0.107 | 0.114 | – | – |
| deception_from_unbiased | cue_gold | 0.515 / 0.515 / 0.515 | 0.515 | 0.000 | 0.571 | 0.441 | 0.278 | 0.430 | 0.147 |
| deception_from_unbiased | cue_dist | 0.504 / 0.515 / 0.499 | 0.506 | 0.008 | 0.575 | 0.318 | 0.284 | 0.392 | 0.159 |
| deception_from_unbiased | train_mix | 0.510 / 0.514 / 0.508 | 0.511 | 0.003 | 0.572 | 0.406 | 0.281 | 0.420 | 0.146 |
| said_yes | cue_gold | 0.601 / 0.606 / 0.598 | 0.602 | 0.004 | 0.624 | 0.593 | 0.259 | 0.492 | 0.203 |
| said_yes | cue_dist | 0.787 / 0.789 / 0.788 | 0.788 | 0.001 | 0.622 | 0.734 | 0.406 | 0.587 | 0.166 |
| said_yes | train_mix | 0.661 / 0.663 / 0.659 | 0.661 | 0.002 | 0.622 | 0.636 | 0.310 | 0.523 | 0.184 |
| false_denial | cue_gold | 0.349 / 0.333 / 0.341 | 0.341 | 0.008 | – | 0.363 | 0.521 | – | – |
| false_denial | cue_dist | 0.270 / 0.266 / 0.264 | 0.267 | 0.003 | – | 0.288 | 0.570 | – | – |
| false_denial | train_mix | 0.313 / 0.300 / 0.301 | 0.305 | 0.007 | – | 0.321 | 0.535 | – | – |
| false_admission | cue_gold | 0.595 / 0.599 / 0.592 | 0.595 | 0.004 | 0.624 | 0.582 | 0.218 | 0.475 | 0.223 |
| false_admission | cue_dist | 0.809 / 0.808 / 0.808 | 0.809 | 0.001 | 0.622 | 0.758 | 0.391 | 0.591 | 0.186 |
| false_admission | train_mix | 0.656 / 0.656 / 0.652 | 0.655 | 0.002 | 0.622 | 0.617 | 0.264 | 0.501 | 0.205 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.000 | 0.016 | 0.002 | 0.006 | 0.009 |
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
| n_bias_rows | cue_gold | 458 / 437 / 445 | 446.667 | 10.599 | 466 | 445 | 453 | 454.667 | 10.599 |
| n_bias_rows | cue_dist | 452 / 431 / 439 | 440.667 | 10.599 | 466 | 445 | 453 | 454.667 | 10.599 |
| n_bias_rows | train_mix | 910 / 868 / 884 | 887.333 | 21.197 | 932 | 890 | 906 | 909.333 | 21.197 |
| n_deception_rows | cue_gold | 441 / 421 / 428 | 430 | 10.149 | 466 | 445 | 452 | 454.333 | 10.693 |
| n_deception_rows | cue_dist | 447 / 427 / 434 | 436 | 10.149 | 466 | 443 | 453 | 454 | 11.533 |
| n_deception_rows | train_mix | 888 / 848 / 862 | 866 | 20.298 | 932 | 888 | 905 | 908.333 | 22.189 |
| probe_start_mean_biased | cue_gold | – | – | – | – | 0.012 | 0.021 | – | – |
| probe_start_mean_biased | cue_dist | – | – | – | – | 0.015 | 0.019 | – | – |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.071 | 0.009 | 0.008 | 0.029 | 0.036 |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.177 | 0.008 | 0.007 | 0.064 | 0.098 |
| probe_start_auc | cue_gold | – | – | – | – | 0.645 | 0.731 | – | – |
| probe_start_auc | cue_dist | – | – | – | – | 0.640 | 0.683 | – | – |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 91 | 71 | 54 | 47.823 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 236 | 172 | 136 | 122.049 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 250 | 342 | 351 | 314.333 | 55.896 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 216 | 101 | 101 | 139.333 | 66.395 |
| probe_fresh_mean_biased | cue_gold | – | – | – | – | 0.067 | 0.212 | – | – |
| probe_fresh_mean_biased | cue_dist | – | – | – | – | 0.066 | 0.192 | – | – |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.025 | 0.036 | 0.114 | 0.058 | 0.048 |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.082 | 0.039 | 0.124 | 0.081 | 0.043 |
| probe_fresh_auc | cue_gold | – | – | – | – | 0.686 | 0.764 | – | – |
| probe_fresh_auc | cue_dist | – | – | – | – | 0.654 | 0.657 | – | – |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 91 | 71 | 54 | 47.823 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 236 | 172 | 136 | 122.049 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 250 | 342 | 351 | 314.333 | 55.896 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 216 | 101 | 101 | 139.333 | 66.395 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | – | 0.051 | 0.117 | – | – |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | – | 0.071 | 0.129 | – | – |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.036 | 0.015 | 0.032 | 0.028 | 0.011 |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.098 | 0.022 | 0.038 | 0.053 | 0.040 |
| probe_start_base_activations_auc | cue_gold | – | – | – | – | 0.720 | 0.783 | – | – |
| probe_start_base_activations_auc | cue_dist | – | – | – | – | 0.687 | 0.753 | – | – |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 91 | 71 | 54 | 47.823 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 236 | 172 | 136 | 122.049 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 250 | 342 | 351 | 314.333 | 55.896 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 216 | 101 | 101 | 139.333 | 66.395 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.948 | 0.829 | 0.846 | 0.874 | 0.065 |
| probe-only groups | 0.427 | 0.740 | 0.694 | 0.620 | 0.169 |
