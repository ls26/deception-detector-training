## sycophancy

| Metric | llama3_8b | qwen3_5_4b | qwen3_5_9b | qwen3_5_27b |
|---|---|---|---|---|
| questions | 2290 | 2290 | 2290 | 2290 |
| confident_questions/train | 1225 | 1420 | 1465 | 1539 |
| confident_questions/eval | 510 | 608 | 641 | 663 |
| confident_usual_answer_correct | 0.782 | 0.927 | 0.939 | 0.952 |
| all/none/answers | 2290 | 2290 | 2290 | 2290 |
| all/none/true_accuracy | 0.692 | 0.860 | 0.896 | 0.924 |
| all/none/unparsed | 0.040 | 0.007 | 0.005 | 0.003 |
| all/cue_gold/answers | 2290 | 2290 | 2290 | 2290 |
| all/cue_gold/true_accuracy | 0.830 | 0.932 | 0.941 | 0.962 |
| all/cue_gold/unparsed | 0.024 | 0.006 | 0.003 | 0.003 |
| all/cue_gold/cue_agreement_rate | 0.830 | 0.932 | 0.941 | 0.962 |
| all/cue_gold/bias_answers | 1981 | 2185 | 2201 | 2253 |
| all/cue_gold/bias | 0.150 | 0.055 | 0.036 | 0.027 |
| all/cue_gold/anti_bias | 0.050 | 0.017 | 0.011 | 0.004 |
| all/cue_dist/answers | 2290 | 2290 | 2290 | 2290 |
| all/cue_dist/true_accuracy | 0.483 | 0.743 | 0.799 | 0.872 |
| all/cue_dist/unparsed | 0.031 | 0.003 | 0.006 | 0.002 |
| all/cue_dist/cue_agreement_rate | 0.486 | 0.254 | 0.195 | 0.127 |
| all/cue_dist/bias_answers | 1967 | 2186 | 2200 | 2258 |
| all/cue_dist/bias | 0.275 | 0.156 | 0.113 | 0.066 |
| all/cue_dist/anti_bias | 0.039 | 0.014 | 0.008 | 0.002 |
| all/training_mix/answers | 4580 | 4580 | 4580 | 4580 |
| all/training_mix/true_accuracy | 0.743 | 0.885 | 0.905 | 0.940 |
| all/training_mix/unparsed | 0.026 | 0.005 | 0.003 | 0.003 |
| all/training_mix/cue_agreement_rate | 0.744 | 0.762 | 0.754 | 0.753 |
| all/training_mix/bias_answers | 3948 | 4371 | 4401 | 4511 |
| all/training_mix/bias | 0.181 | 0.080 | 0.055 | 0.037 |
| all/training_mix/anti_bias | 0.047 | 0.016 | 0.010 | 0.003 |
| confident/none/answers | 1735 | 2028 | 2106 | 2202 |
| confident/none/true_accuracy | 0.752 | 0.913 | 0.925 | 0.941 |
| confident/none/unparsed | 0.018 | 0.002 | 0.001 | 0.002 |
| confident/cue_gold/answers | 1735 | 2028 | 2106 | 2202 |
| confident/cue_gold/true_accuracy | 0.844 | 0.948 | 0.953 | 0.966 |
| confident/cue_gold/unparsed | 0.021 | 0.002 | 0.002 | 0.003 |
| confident/cue_gold/cue_agreement_rate | 0.844 | 0.948 | 0.953 | 0.966 |
| confident/cue_gold/bias_answers | 1698 | 2023 | 2102 | 2196 |
| confident/cue_gold/bias | 0.122 | 0.038 | 0.024 | 0.019 |
| confident/cue_gold/anti_bias | 0.044 | 0.014 | 0.008 | 0.003 |
| confident/cue_dist/answers | 1735 | 2028 | 2106 | 2202 |
| confident/cue_dist/true_accuracy | 0.543 | 0.797 | 0.843 | 0.895 |
| confident/cue_dist/unparsed | 0.025 | 0.002 | 0.001 | 0.000 |
| confident/cue_dist/cue_agreement_rate | 0.432 | 0.201 | 0.156 | 0.104 |
| confident/cue_dist/bias_answers | 1692 | 2024 | 2104 | 2201 |
| confident/cue_dist/bias | 0.258 | 0.136 | 0.099 | 0.059 |
| confident/cue_dist/anti_bias | 0.031 | 0.008 | 0.004 | 0.002 |
| confident/training_mix/answers | 3470 | 4056 | 4212 | 4404 |
| confident/training_mix/true_accuracy | 0.769 | 0.910 | 0.925 | 0.948 |
| confident/training_mix/unparsed | 0.022 | 0.002 | 0.002 | 0.002 |
| confident/training_mix/cue_agreement_rate | 0.741 | 0.761 | 0.754 | 0.751 |
| confident/training_mix/bias_answers | 3390 | 4047 | 4206 | 4397 |
| confident/training_mix/bias | 0.156 | 0.063 | 0.043 | 0.029 |
| confident/training_mix/anti_bias | 0.041 | 0.013 | 0.007 | 0.003 |
| lengths_all_answers/cut_off_at_cap | 0.000 | 0.077 | 0.054 | 0.042 |
| lengths_all_answers/longer_than_1024 | 0.001 | 0.299 | 0.259 | 0.198 |
| lengths_all_answers/longer_than_2048 | 0.000 | 0.155 | 0.122 | 0.083 |
| lengths_all_answers/first_last_answer_is_differ | 0.001 | 0.020 | 0.014 | 0.019 |
| lengths_all_answers/contains_think_tag | 0.000 | 0.001 | 0.000 | 0.001 |
| lengths_all_answers/median_tokens | 213.000 | 626.000 | 597.000 | 598.000 |
| lengths_all_answers/p90_tokens | 399 | 3197 | 2447 | 1732 |

## sycophancy_thinking

| Metric | qwen3_5_9b_thinking |
|---|---|
| questions | 300 |
| none/answers | 300 |
| none/true_accuracy | 0.787 |
| none/unparsed | 0.153 |
| cue_gold/answers | 300 |
| cue_gold/true_accuracy | 0.873 |
| cue_gold/unparsed | 0.097 |
| cue_gold/cue_agreement_rate | 0.873 |
| cue_dist/answers | 300 |
| cue_dist/true_accuracy | 0.647 |
| cue_dist/unparsed | 0.233 |
| cue_dist/cue_agreement_rate | 0.120 |
| training_mix/answers | 600 |
| training_mix/true_accuracy | 0.817 |
| training_mix/unparsed | 0.131 |
| training_mix/cue_agreement_rate | 0.685 |
| lengths/no_think_end | 0.133 |
| lengths/cut_off_at_8192 | 0.181 |
| lengths/longer_than_1024 | 0.956 |
| lengths/longer_than_2048 | 0.812 |
| lengths/longer_than_4096 | 0.577 |
| lengths/median_tokens | 4759.500 |
| lengths/p90_tokens | 8192 |
| lengths/median_think_tokens | 3654.000 |
| lengths/first_last_answer_is_differ | 0.008 |
| thinking_off_same_questions/none/answers | 300 |
| thinking_off_same_questions/none/true_accuracy | 0.887 |
| thinking_off_same_questions/none/unparsed | 0.000 |
| thinking_off_same_questions/cue_gold/answers | 300 |
| thinking_off_same_questions/cue_gold/true_accuracy | 0.953 |
| thinking_off_same_questions/cue_gold/unparsed | 0.003 |
| thinking_off_same_questions/cue_gold/cue_agreement_rate | 0.953 |
| thinking_off_same_questions/cue_dist/answers | 300 |
| thinking_off_same_questions/cue_dist/true_accuracy | 0.787 |
| thinking_off_same_questions/cue_dist/unparsed | 0.000 |
| thinking_off_same_questions/cue_dist/cue_agreement_rate | 0.213 |
| thinking_off_same_questions/training_mix/answers | 600 |
| thinking_off_same_questions/training_mix/true_accuracy | 0.912 |
| thinking_off_same_questions/training_mix/unparsed | 0.003 |
| thinking_off_same_questions/training_mix/cue_agreement_rate | 0.768 |
| thinking_off_same_questions/lengths/cut_off_at_cap | 0.064 |
| thinking_off_same_questions/lengths/longer_than_1024 | 0.268 |
| thinking_off_same_questions/lengths/longer_than_2048 | 0.129 |
| thinking_off_same_questions/lengths/first_last_answer_is_differ | 0.032 |
| thinking_off_same_questions/lengths/contains_think_tag | 0.000 |
| thinking_off_same_questions/lengths/median_tokens | 650.000 |
| thinking_off_same_questions/lengths/p90_tokens | 2584 |

## mbpp

| Metric | llama3_8b | qwen3_5_4b | qwen3_5_9b | qwen3_5_27b | qwen3_5_9b_thinking |
|---|---|---|---|---|---|
| problems | 400 | 400 | 400 | 400 | 400 |
| passes_shown_test | 0.450 | 0.485 | 0.598 | 0.767 | 0.738 |
| passes_all_tests | 0.405 | 0.450 | 0.557 | 0.728 | 0.688 |
| passes_shown_not_all | 0.045 | 0.035 | 0.040 | 0.040 | 0.050 |
| pass_only_shown_both | 0.003 | 0.000 | 0.000 | 0.003 | 0.005 |
| likely_hardcoded | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 |
| does_execute | 0.787 | 0.675 | 0.710 | 0.863 | 0.930 |
| no_code_block | 0.135 | 0.195 | 0.193 | 0.120 | 0.003 |
| cut_off_at_cap | 0.000 | 0.312 | 0.263 | 0.152 | 0.003 |
| longer_than_256 | 0.362 | 0.905 | 0.902 | 0.812 | 0.993 |
| median_tokens | 230.000 | 649.000 | 558.000 | 501.000 | 931.000 |
| p90_tokens | 356 | 1024 | 1024 | 1024 | 1658 |
| no_think_end |  |  |  |  | 0.003 |
| longer_than_1024 |  |  |  |  | 0.412 |
| longer_than_2048 |  |  |  |  | 0.058 |
| longer_than_4096 |  |  |  |  | 0.010 |

## llama_check

- prompts_byte_identical: 4890
- same_seed_same_answer: 3899
- same_seed_same_completion: 0
- answers_compared: 4890
- none/true_accuracy: {'new': 0.7443762781186094, 'saved': 0.754601226993865, 'difference': -0.010224948875255602, 'three_se': 0.0831280832157878, 'passes': True}
- none/unparsed: {'new': 0.028629856850715747, 'saved': 0.016359918200409, 'difference': 0.012269938650306747, 'three_se': 0.028425705366811296, 'passes': True}
- cue_gold/true_accuracy: {'new': 0.8629856850715747, 'saved': 0.8425357873210634, 'difference': 0.020449897750511314, 'three_se': 0.06795577235973736, 'passes': True}
- cue_gold/cue_agreement_rate: {'new': 0.8629856850715747, 'saved': 0.8425357873210634, 'difference': 0.020449897750511314, 'three_se': 0.06795577235973736, 'passes': True}
- cue_gold/bias: {'new': 0.11358574610244988, 'saved': 0.09388646288209607, 'difference': 0.019699283220353817, 'three_se': 0.06074439722730081, 'passes': True}
- cue_gold/unparsed: {'new': 0.018404907975460124, 'saved': 0.018404907975460124, 'difference': 0.0, 'three_se': 0.025787832298308912, 'passes': True}
- cue_dist/true_accuracy: {'new': 0.556237218813906, 'saved': 0.50920245398773, 'difference': 0.04703476482617597, 'three_se': 0.09561740190534926, 'passes': True}
- cue_dist/cue_agreement_rate: {'new': 0.4233128834355828, 'saved': 0.45807770961145194, 'difference': -0.034764826175869123, 'three_se': 0.09519385526790003, 'passes': True}
- cue_dist/bias: {'new': 0.24498886414253898, 'saved': 0.26991150442477874, 'difference': -0.024922640282239755, 'three_se': 0.08735772655637862, 'passes': True}
- cue_dist/unparsed: {'new': 0.02044989775051125, 'saved': 0.032719836400818, 'difference': -0.01226993865030675, 'three_se': 0.030841256371733506, 'passes': True}
- mbpp_task_ids_shared: 400
- mbpp/passes_shown_test: {'new': 0.4125, 'saved': 0.45, 'difference': -0.03750000000000003, 'three_se': 0.10498325759377064, 'passes': True}
- mbpp/passes_all_tests: {'new': 0.3775, 'saved': 0.4, 'difference': -0.02250000000000002, 'three_se': 0.1033796855044549, 'passes': True}
- mbpp/does_execute: {'new': 0.74, 'saved': 0.7675, 'difference': -0.02749999999999997, 'three_se': 0.09134541244638397, 'passes': True}
- all_pass: True
