# RL runs side by side

Base = the base model (one eval, shared by the runs; on bias and self-report rows one value per run). Each run column = that run's final checkpoint. Base mean / SD are over the runs' base values, mean / SD over the after-RL values (SD with n - 1). All eval questions, one sampled answer per prompt.

Eval questions: 489. Questions with a no-cue answer in both the base eval and the run's eval (bias and self-report rows use these): s42 462.

| metric | condition | base | base mean | base SD | s42 | mean | SD |
|---|---|---|---|---|---|---|---|
| accuracy | none | 0.755 | 0.755 | – | 0.560 | 0.560 | – |
| accuracy | cue_gold | 0.843 | 0.843 | – | 1.000 | 1.000 | – |
| accuracy | cue_dist | 0.509 | 0.509 | – | 0.002 | 0.002 | – |
| accuracy | train_mix | 0.729 | 0.729 | – | 0.650 | 0.650 | – |
| reward | none | 0.644 | 0.644 | – | 0.544 | 0.544 | – |
| reward | cue_gold | 0.714 | 0.714 | – | 0.800 | 0.800 | – |
| reward | cue_dist | 0.499 | 0.499 | – | 0.202 | 0.202 | – |
| reward | train_mix | 0.689 | 0.689 | – | 0.751 | 0.751 | – |
| cue agreement rate | cue_gold | 0.843 | 0.843 | – | 1.000 | 1.000 | – |
| cue agreement rate | cue_dist | 0.458 | 0.458 | – | 0.998 | 0.998 | – |
| cue agreement rate | train_mix | 0.711 | 0.711 | – | 0.999 | 0.999 | – |
| bias | cue_gold | 0.095 | 0.095 | – | 0.418 | 0.418 | – |
| bias | cue_dist | 0.268 | 0.268 | – | 0.580 | 0.580 | – |
| bias | train_mix | 0.153 | 0.153 | – | 0.471 | 0.471 | – |
| anti_bias | cue_gold | 0.037 | 0.037 | – | 0.000 | 0.000 | – |
| anti_bias | cue_dist | 0.011 | 0.011 | – | 0.000 | 0.000 | – |
| anti_bias | train_mix | 0.028 | 0.028 | – | 0.000 | 0.000 | – |
| no_bias | cue_gold | 0.868 | 0.868 | – | 0.582 | 0.582 | – |
| no_bias | cue_dist | 0.721 | 0.721 | – | 0.420 | 0.420 | – |
| no_bias | train_mix | 0.819 | 0.819 | – | 0.529 | 0.529 | – |
| deception_soft | cue_gold | 0.553 | 0.553 | – | 0.555 | 0.555 | – |
| deception_soft | cue_dist | 0.604 | 0.604 | – | 0.476 | 0.476 | – |
| deception_soft | train_mix | 0.569 | 0.569 | – | 0.531 | 0.531 | – |
| deception | cue_gold | 0.570 | 0.570 | – | 0.550 | 0.550 | – |
| deception | cue_dist | 0.666 | 0.666 | – | 0.459 | 0.459 | – |
| deception | train_mix | 0.603 | 0.603 | – | 0.523 | 0.523 | – |
| biased_share | cue_gold | 0.098 | 0.098 | – | 0.418 | 0.418 | – |
| biased_share | cue_dist | 0.271 | 0.271 | – | 0.580 | 0.580 | – |
| biased_share | train_mix | 0.158 | 0.158 | – | 0.471 | 0.471 | – |
| false_denial_soft | cue_gold | 0.380 | 0.380 | – | 0.168 | 0.168 | – |
| false_denial_soft | cue_dist | 0.355 | 0.355 | – | 0.185 | 0.185 | – |
| false_denial_soft | train_mix | 0.368 | 0.368 | – | 0.176 | 0.176 | – |
| false_admission_soft | cue_gold | 0.571 | 0.571 | – | 0.833 | 0.833 | – |
| false_admission_soft | cue_dist | 0.696 | 0.696 | – | 0.880 | 0.880 | – |
| false_admission_soft | train_mix | 0.607 | 0.607 | – | 0.848 | 0.848 | – |
| said_yes_biased_soft | cue_gold | 0.620 | 0.620 | – | 0.832 | 0.832 | – |
| said_yes_biased_soft | cue_dist | 0.645 | 0.645 | – | 0.815 | 0.815 | – |
| said_yes_biased_soft | train_mix | 0.632 | 0.632 | – | 0.824 | 0.824 | – |
| deception_from_biased | cue_gold | 0.037 | 0.037 | – | 0.070 | 0.070 | – |
| deception_from_biased | cue_dist | 0.096 | 0.096 | – | 0.107 | 0.107 | – |
| deception_from_biased | train_mix | 0.058 | 0.058 | – | 0.083 | 0.083 | – |
| deception_from_unbiased | cue_gold | 0.515 | 0.515 | – | 0.485 | 0.485 | – |
| deception_from_unbiased | cue_dist | 0.508 | 0.508 | – | 0.369 | 0.369 | – |
| deception_from_unbiased | train_mix | 0.511 | 0.511 | – | 0.448 | 0.448 | – |
| said_yes | cue_gold | 0.600 | 0.600 | – | 0.907 | 0.907 | – |
| said_yes | cue_dist | 0.788 | 0.788 | – | 0.935 | 0.935 | – |
| said_yes | train_mix | 0.660 | 0.660 | – | 0.918 | 0.918 | – |
| false_denial | cue_gold | 0.349 | 0.349 | – | 0.073 | 0.073 | – |
| false_denial | cue_dist | 0.275 | 0.275 | – | 0.090 | 0.090 | – |
| false_denial | train_mix | 0.318 | 0.318 | – | 0.080 | 0.080 | – |
| false_admission | cue_gold | 0.594 | 0.594 | – | 0.892 | 0.892 | – |
| false_admission | cue_dist | 0.811 | 0.811 | – | 0.969 | 0.969 | – |
| false_admission | train_mix | 0.656 | 0.656 | – | 0.917 | 0.917 | – |
| parse_failed | none | 0.016 | 0.016 | – | 0.006 | 0.006 | – |
| parse_failed | cue_gold | 0.018 | 0.018 | – | 0.000 | 0.000 | – |
| parse_failed | cue_dist | 0.033 | 0.033 | – | 0.000 | 0.000 | – |
| parse_failed | train_mix | 0.024 | 0.024 | – | 0.000 | 0.000 | – |
| cut_off | none | 0.000 | 0.000 | – | 0.000 | 0.000 | – |
| cut_off | cue_gold | 0.000 | 0.000 | – | 0.000 | 0.000 | – |
| cut_off | cue_dist | 0.000 | 0.000 | – | 0.000 | 0.000 | – |
| cut_off | train_mix | 0.000 | 0.000 | – | 0.000 | 0.000 | – |
| rollouts | none | 489 | 489 | – | 489 | 489 | – |
| rollouts | cue_gold | 489 | 489 | – | 489 | 489 | – |
| rollouts | cue_dist | 489 | 489 | – | 489 | 489 | – |
| rollouts | train_mix | 978 | 978 | – | 978 | 978 | – |
| n_bias_rows | cue_gold | 454 | 454 | – | 462 | 462 | – |
| n_bias_rows | cue_dist | 448 | 448 | – | 462 | 462 | – |
| n_bias_rows | train_mix | 902 | 902 | – | 924 | 924 | – |
| n_deception_rows | cue_gold | 437 | 437 | – | 462 | 462 | – |
| n_deception_rows | cue_dist | 443 | 443 | – | 462 | 462 | – |
| n_deception_rows | train_mix | 880 | 880 | – | 924 | 924 | – |
| probe_start_mean_biased | cue_gold | – | – | – | 0.009 | 0.009 | – |
| probe_start_mean_biased | cue_dist | – | – | – | 0.006 | 0.006 | – |
| probe_start_mean_unbiased | cue_gold | – | – | – | 0.002 | 0.002 | – |
| probe_start_mean_unbiased | cue_dist | – | – | – | 0.002 | 0.002 | – |
| probe_start_auc | cue_gold | – | – | – | 0.928 | 0.928 | – |
| probe_start_auc | cue_dist | – | – | – | 0.885 | 0.885 | – |
| probe_start_n_biased | cue_gold | 0 | 0 | – | 193 | 193 | – |
| probe_start_n_biased | cue_dist | 0 | 0 | – | 268 | 268 | – |
| probe_start_n_unbiased | cue_gold | 0 | 0 | – | 269 | 269 | – |
| probe_start_n_unbiased | cue_dist | 0 | 0 | – | 193 | 193 | – |
| probe_fresh_mean_biased | cue_gold | – | – | – | 0.025 | 0.025 | – |
| probe_fresh_mean_biased | cue_dist | – | – | – | 0.021 | 0.021 | – |
| probe_fresh_mean_unbiased | cue_gold | – | – | – | 0.007 | 0.007 | – |
| probe_fresh_mean_unbiased | cue_dist | – | – | – | 0.007 | 0.007 | – |
| probe_fresh_auc | cue_gold | – | – | – | 0.916 | 0.916 | – |
| probe_fresh_auc | cue_dist | – | – | – | 0.859 | 0.859 | – |
| probe_fresh_n_biased | cue_gold | 0 | 0 | – | 193 | 193 | – |
| probe_fresh_n_biased | cue_dist | 0 | 0 | – | 268 | 268 | – |
| probe_fresh_n_unbiased | cue_gold | 0 | 0 | – | 269 | 269 | – |
| probe_fresh_n_unbiased | cue_dist | 0 | 0 | – | 193 | 193 | – |
| probe_start_base_activations_mean_biased | cue_gold | – | – | – | 0.078 | 0.078 | – |
| probe_start_base_activations_mean_biased | cue_dist | – | – | – | 0.110 | 0.110 | – |
| probe_start_base_activations_mean_unbiased | cue_gold | – | – | – | 0.018 | 0.018 | – |
| probe_start_base_activations_mean_unbiased | cue_dist | – | – | – | 0.027 | 0.027 | – |
| probe_start_base_activations_auc | cue_gold | – | – | – | 0.825 | 0.825 | – |
| probe_start_base_activations_auc | cue_dist | – | – | – | 0.844 | 0.844 | – |
| probe_start_base_activations_n_biased | cue_gold | 0 | 0 | – | 193 | 193 | – |
| probe_start_base_activations_n_biased | cue_dist | 0 | 0 | – | 268 | 268 | – |
| probe_start_base_activations_n_unbiased | cue_gold | 0 | 0 | – | 269 | 269 | – |
| probe_start_base_activations_n_unbiased | cue_dist | 0 | 0 | – | 193 | 193 | – |

Training, mean over the 128 steps: the share of groups whose 8 task rewards are all equal, and the share of probe-only groups (all 8 answers name the cued option; with the probe penalty only their probe scores differ).

| | s42 | mean | SD |
|---|---|---|---|
| equal task reward groups | 0.955 | 0.955 | – |
| probe-only groups | 0.945 | 0.945 | – |
