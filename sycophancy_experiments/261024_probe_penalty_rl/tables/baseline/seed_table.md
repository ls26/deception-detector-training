# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 442, s43 456, s44 453.

| metric | condition | base | base mean | base SD | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | 0.000 | 0.751 | 0.748 | 0.738 | 0.746 | 0.007 |
| accuracy | cue_gold | 0.843 | 0.843 | 0.000 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| accuracy | cue_dist | 0.509 | 0.509 | 0.000 | 0.186 | 0.346 | 0.239 | 0.257 | 0.081 |
| accuracy | train_mix | 0.729 | 0.729 | 0.000 | 0.700 | 0.746 | 0.717 | 0.721 | 0.023 |
| reward | none | 0.644 | 0.644 | 0.000 | 0.661 | 0.667 | 0.658 | 0.662 | 0.004 |
| reward | cue_gold | 0.714 | 0.714 | 0.000 | 0.800 | 0.783 | 0.787 | 0.790 | 0.009 |
| reward | cue_dist | 0.499 | 0.499 | 0.000 | 0.321 | 0.407 | 0.329 | 0.352 | 0.047 |
| reward | train_mix | 0.689 | 0.689 | 0.000 | 0.758 | 0.748 | 0.740 | 0.749 | 0.009 |
| cue agreement rate | cue_gold | 0.843 | 0.843 | 0.000 | 0.975 | 0.959 | 0.965 | 0.967 | 0.008 |
| cue agreement rate | cue_dist | 0.458 | 0.458 | 0.000 | 0.814 | 0.652 | 0.761 | 0.742 | 0.082 |
| cue agreement rate | train_mix | 0.711 | 0.711 | 0.000 | 0.925 | 0.854 | 0.893 | 0.890 | 0.036 |
| bias | cue_gold | 0.097 / 0.092 / 0.090 | 0.093 | 0.004 | 0.179 | 0.173 | 0.195 | 0.182 | 0.011 |
| bias | cue_dist | 0.259 / 0.267 / 0.273 | 0.267 | 0.007 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| bias | train_mix | 0.154 / 0.152 / 0.153 | 0.153 | 0.001 | 0.324 | 0.251 | 0.289 | 0.288 | 0.036 |
| anti_bias | cue_gold | 0.037 / 0.038 / 0.038 | 0.038 | 0.001 | 0.002 | 0.002 | 0.000 | 0.001 | 0.001 |
| anti_bias | cue_dist | 0.012 / 0.011 / 0.009 | 0.011 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| anti_bias | train_mix | 0.027 / 0.028 / 0.028 | 0.028 | 0.000 | 0.001 | 0.002 | 0.000 | 0.001 | 0.001 |
| no_bias | cue_gold | 0.866 / 0.871 / 0.872 | 0.870 | 0.003 | 0.819 | 0.825 | 0.805 | 0.816 | 0.010 |
| no_bias | cue_dist | 0.729 / 0.722 / 0.718 | 0.723 | 0.006 | 0.405 | 0.580 | 0.483 | 0.490 | 0.088 |
| no_bias | train_mix | 0.819 / 0.820 / 0.820 | 0.820 | 0.000 | 0.676 | 0.747 | 0.711 | 0.711 | 0.036 |
| deception_soft | cue_gold | 0.547 / 0.554 / 0.550 | 0.551 | 0.004 | 0.553 | 0.537 | 0.590 | 0.560 | 0.027 |
| deception_soft | cue_dist | 0.605 / 0.600 / 0.596 | 0.600 | 0.004 | 0.464 | 0.567 | 0.521 | 0.517 | 0.051 |
| deception_soft | train_mix | 0.565 / 0.568 / 0.566 | 0.566 | 0.002 | 0.516 | 0.547 | 0.578 | 0.547 | 0.031 |
| deception | cue_gold | 0.565 / 0.573 / 0.568 | 0.568 | 0.004 | 0.628 | 0.571 | 0.653 | 0.617 | 0.042 |
| deception | cue_dist | 0.669 / 0.664 / 0.657 | 0.663 | 0.006 | 0.425 | 0.602 | 0.521 | 0.516 | 0.089 |
| deception | train_mix | 0.597 / 0.602 / 0.599 | 0.600 | 0.002 | 0.547 | 0.579 | 0.623 | 0.583 | 0.038 |
| biased_share | cue_gold | 0.100 / 0.095 / 0.093 | 0.096 | 0.004 | 0.179 | 0.174 | 0.195 | 0.182 | 0.011 |
| biased_share | cue_dist | 0.262 / 0.270 / 0.276 | 0.269 | 0.007 | 0.595 | 0.420 | 0.517 | 0.510 | 0.088 |
| biased_share | train_mix | 0.158 / 0.156 / 0.157 | 0.157 | 0.001 | 0.324 | 0.252 | 0.289 | 0.288 | 0.036 |
| false_denial_soft | cue_gold | 0.388 / 0.396 / 0.379 | 0.388 | 0.009 | 0.291 | 0.299 | 0.254 | 0.282 | 0.024 |
| false_denial_soft | cue_dist | 0.360 / 0.354 / 0.349 | 0.354 | 0.005 | 0.272 | 0.303 | 0.260 | 0.278 | 0.022 |
| false_denial_soft | train_mix | 0.370 / 0.372 / 0.364 | 0.369 | 0.004 | 0.272 | 0.298 | 0.261 | 0.277 | 0.019 |
| false_admission_soft | cue_gold | 0.565 / 0.571 / 0.568 | 0.568 | 0.003 | 0.610 | 0.587 | 0.671 | 0.623 | 0.044 |
| false_admission_soft | cue_dist | 0.693 / 0.691 / 0.691 | 0.691 | 0.001 | 0.747 | 0.758 | 0.799 | 0.768 | 0.028 |
| false_admission_soft | train_mix | 0.601 / 0.604 / 0.603 | 0.603 | 0.002 | 0.632 | 0.631 | 0.707 | 0.657 | 0.043 |
| said_yes_biased_soft | cue_gold | 0.612 / 0.604 / 0.621 | 0.612 | 0.009 | 0.709 | 0.701 | 0.746 | 0.718 | 0.024 |
| said_yes_biased_soft | cue_dist | 0.640 / 0.646 / 0.651 | 0.646 | 0.005 | 0.728 | 0.697 | 0.740 | 0.722 | 0.022 |
| said_yes_biased_soft | train_mix | 0.630 / 0.628 / 0.636 | 0.631 | 0.004 | 0.728 | 0.702 | 0.739 | 0.723 | 0.019 |
| deception_from_biased | cue_gold | 0.039 / 0.038 / 0.035 | 0.037 | 0.002 | 0.052 | 0.052 | 0.050 | 0.051 | 0.001 |
| deception_from_biased | cue_dist | 0.094 / 0.096 / 0.096 | 0.095 | 0.001 | 0.162 | 0.127 | 0.134 | 0.141 | 0.018 |
| deception_from_biased | train_mix | 0.058 / 0.058 / 0.057 | 0.058 | 0.001 | 0.088 | 0.075 | 0.075 | 0.079 | 0.007 |
| deception_from_unbiased | cue_gold | 0.508 / 0.517 / 0.515 | 0.513 | 0.004 | 0.501 | 0.485 | 0.541 | 0.509 | 0.029 |
| deception_from_unbiased | cue_dist | 0.511 / 0.504 / 0.500 | 0.505 | 0.005 | 0.302 | 0.440 | 0.386 | 0.376 | 0.069 |
| deception_from_unbiased | train_mix | 0.506 / 0.510 / 0.509 | 0.508 | 0.002 | 0.428 | 0.472 | 0.502 | 0.467 | 0.038 |
| said_yes | cue_gold | 0.593 / 0.599 / 0.596 | 0.596 | 0.003 | 0.785 | 0.648 | 0.794 | 0.742 | 0.082 |
| said_yes | cue_dist | 0.785 / 0.783 / 0.786 | 0.785 | 0.002 | 0.939 | 0.802 | 0.892 | 0.878 | 0.069 |
| said_yes | train_mix | 0.655 / 0.657 / 0.658 | 0.657 | 0.002 | 0.829 | 0.697 | 0.828 | 0.785 | 0.076 |
| false_denial | cue_gold | 0.357 / 0.366 / 0.350 | 0.358 | 0.008 | 0.063 | 0.278 | 0.136 | 0.159 | 0.109 |
| false_denial | cue_dist | 0.279 / 0.280 / 0.267 | 0.275 | 0.007 | 0.068 | 0.262 | 0.141 | 0.157 | 0.098 |
| false_denial | train_mix | 0.317 / 0.325 / 0.311 | 0.318 | 0.007 | 0.065 | 0.266 | 0.147 | 0.159 | 0.101 |
| false_admission | cue_gold | 0.588 / 0.595 / 0.590 | 0.591 | 0.004 | 0.751 | 0.633 | 0.777 | 0.721 | 0.077 |
| false_admission | cue_dist | 0.808 / 0.806 / 0.806 | 0.807 | 0.001 | 0.950 | 0.848 | 0.927 | 0.908 | 0.053 |
| false_admission | train_mix | 0.650 / 0.654 / 0.653 | 0.652 | 0.002 | 0.778 | 0.684 | 0.817 | 0.760 | 0.068 |
| parse_failed | none | 0.016 | 0.016 | 0.000 | 0.000 | 0.004 | 0.010 | 0.005 | 0.005 |
| parse_failed | cue_gold | 0.018 | 0.018 | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.001 |
| parse_failed | cue_dist | 0.033 | 0.033 | 0.000 | 0.000 | 0.002 | 0.000 | 0.001 | 0.001 |
| parse_failed | train_mix | 0.024 | 0.024 | 0.000 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 |
| cut_off | none | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | cue_dist | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| cut_off | train_mix | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| rollouts | none | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_gold | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | cue_dist | 489 | 489 | 0.000 | 489 | 489 | 489 | 489 | 0.000 |
| rollouts | train_mix | 978 | 978 | 0.000 | 978 | 978 | 978 | 978 | 0.000 |
| n_bias_rows | cue_gold | 434 / 448 / 445 | 442.333 | 7.371 | 442 | 456 | 452 | 450 | 7.211 |
| n_bias_rows | cue_dist | 428 / 442 / 439 | 436.333 | 7.371 | 442 | 455 | 453 | 450 | 7.000 |
| n_bias_rows | train_mix | 862 / 890 / 884 | 878.667 | 14.742 | 884 | 911 | 905 | 900 | 14.177 |
| n_deception_rows | cue_gold | 418 / 431 / 428 | 425.667 | 6.807 | 441 | 455 | 452 | 449.333 | 7.371 |
| n_deception_rows | cue_dist | 423 / 437 / 435 | 431.667 | 7.572 | 442 | 455 | 453 | 450 | 7.000 |
| n_deception_rows | train_mix | 841 / 868 / 863 | 857.333 | 14.364 | 883 | 910 | 905 | 899.333 | 14.364 |
| probe_start_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_fresh_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_fresh_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_auc | cue_gold | – | – | – | – | – | – | – | – |
| probe_start_base_activations_auc | cue_dist | – | – | – | – | – | – | – | – |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | 0.000 | 0 | 0 | 0 | 0 | 0.000 |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | s43 | s44 | mean | SD |
|---|---|---|---|---|---|
| equal task reward groups | 0.847 | 0.822 | 0.900 | 0.856 | 0.040 |
| probe-only groups | 0.833 | 0.774 | 0.865 | 0.824 | 0.046 |
