# Training dynamics and probe rows: kl_weight_1_0_s43, s43_frozen_round2, s43_frozen_round3, s43_frozen_round4, s43_frozen_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s43 0.3, s43_frozen_round2 0.3, s43_frozen_round3 0.3, s43_frozen_round4 0.3, s43_frozen_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s43 | 32 | 41.750 | 43.500 | 44.529 |
| groups where all 8 answers name the cue (of 64) | s43_frozen_round2 | 43.500 | 45.450 | 44.733 | 44.544 |
| groups where all 8 answers name the cue (of 64) | s43_frozen_round3 | 43.700 | 41.550 | 43.233 | 44.676 |
| groups where all 8 answers name the cue (of 64) | s43_frozen_round4 | 44.700 | 45.600 | 44.933 | 45.941 |
| groups where all 8 answers name the cue (of 64) | s43_frozen_round5 | 45 | 43.650 | 44.900 | 46.471 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s43 | 26.500 | 17.200 | 16.800 | 15.382 |
| groups with mixed task rewards (of 64) | s43_frozen_round2 | 16.700 | 14.550 | 14.067 | 12.338 |
| groups with mixed task rewards (of 64) | s43_frozen_round3 | 11.700 | 13 | 9.600 | 8.382 |
| groups with mixed task rewards (of 64) | s43_frozen_round4 | 6.900 | 6 | 5.700 | 3.897 |
| groups with mixed task rewards (of 64) | s43_frozen_round5 | 4.400 | 5 | 4.600 | 2.191 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s43 | 7.800 | 6.950 | 6.333 | 6.338 |
|   of which the cue is not at the training label | s43_frozen_round2 | 6.200 | 6.300 | 5.967 | 5.162 |
|   of which the cue is not at the training label | s43_frozen_round3 | 5.300 | 5.250 | 3.767 | 3.824 |
|   of which the cue is not at the training label | s43_frozen_round4 | 3.300 | 3.550 | 2.733 | 1.985 |
|   of which the cue is not at the training label | s43_frozen_round5 | 1.200 | 2.250 | 2.200 | 1.074 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s43 | 126 | 99.050 | 85.167 | 81.926 |
| answers not naming the cue (of 512) | s43_frozen_round2 | 72.800 | 75.650 | 86.367 | 95.868 |
| answers not naming the cue (of 512) | s43_frozen_round3 | 105.200 | 117.250 | 119.567 | 116.412 |
| answers not naming the cue (of 512) | s43_frozen_round4 | 126.400 | 124.700 | 127.233 | 127.132 |
| answers not naming the cue (of 512) | s43_frozen_round5 | 127.600 | 140.700 | 132.133 | 130.235 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s43 | 0.355 | 0.296 | 0.305 | 0.251 |
| mean absolute advantage per answer | s43_frozen_round2 | 0.285 | 0.256 | 0.237 | 0.209 |
| mean absolute advantage per answer | s43_frozen_round3 | 0.199 | 0.220 | 0.175 | 0.154 |
| mean absolute advantage per answer | s43_frozen_round4 | 0.134 | 0.110 | 0.125 | 0.098 |
| mean absolute advantage per answer | s43_frozen_round5 | 0.107 | 0.098 | 0.090 | 0.062 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s43 | 0.112 | 0.117 | 0.122 | 0.071 |
|   from probe-only groups | s43_frozen_round2 | 0.091 | 0.085 | 0.073 | 0.066 |
|   from probe-only groups | s43_frozen_round3 | 0.068 | 0.068 | 0.064 | 0.056 |
|   from probe-only groups | s43_frozen_round4 | 0.055 | 0.042 | 0.058 | 0.054 |
|   from probe-only groups | s43_frozen_round5 | 0.057 | 0.042 | 0.038 | 0.038 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s43 | 0.017 | -0.003 | 0.007 | 0.009 |
| advantage on cue-naming answers, per answer | s43_frozen_round2 | 0.017 | 0.005 | 0.008 | 0.008 |
| advantage on cue-naming answers, per answer | s43_frozen_round3 | 0.001 | 0.010 | 0.009 | 0.001 |
| advantage on cue-naming answers, per answer | s43_frozen_round4 | 0.000 | -0.007 | -0.001 | -0.002 |
| advantage on cue-naming answers, per answer | s43_frozen_round5 | 0.009 | 0.000 | -0.001 | -0.000 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s43 | 0.087 | 0.102 | 0.140 | 0.066 |
| gradient norm (W&B, before clipping) | s43_frozen_round2 | 0.094 | 0.068 | 0.063 | 0.059 |
| gradient norm (W&B, before clipping) | s43_frozen_round3 | 0.055 | 0.058 | 0.052 | 0.046 |
| gradient norm (W&B, before clipping) | s43_frozen_round4 | 0.042 | 0.050 | 0.040 | 0.034 |
| gradient norm (W&B, before clipping) | s43_frozen_round5 | 0.043 | 0.035 | 0.034 | 0.024 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s43 | 0.012 | 0.101 | 0.129 | 0.108 |
| KL from the base model (W&B) | s43_frozen_round2 | 0.126 | 0.149 | 0.132 | 0.127 |
| KL from the base model (W&B) | s43_frozen_round3 | 0.123 | 0.128 | 0.131 | 0.128 |
| KL from the base model (W&B) | s43_frozen_round4 | 0.113 | 0.126 | 0.112 | 0.109 |
| KL from the base model (W&B) | s43_frozen_round5 | 0.098 | 0.107 | 0.094 | 0.077 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s43 | 0.427 | 0.401 | 0.440 | 0.446 |
| token entropy (W&B) | s43_frozen_round2 | 0.463 | 0.453 | 0.477 | 0.503 |
| token entropy (W&B) | s43_frozen_round3 | 0.498 | 0.494 | 0.471 | 0.490 |
| token entropy (W&B) | s43_frozen_round4 | 0.491 | 0.457 | 0.475 | 0.517 |
| token entropy (W&B) | s43_frozen_round5 | 0.517 | 0.486 | 0.499 | 0.479 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s43 | 243.601 | 195.433 | 229.320 | 208.106 |
| answer length, tokens (W&B) | s43_frozen_round2 | 215.426 | 190.000 | 208.382 | 216.162 |
| answer length, tokens (W&B) | s43_frozen_round3 | 214.201 | 238.239 | 207.196 | 213.554 |
| answer length, tokens (W&B) | s43_frozen_round4 | 215.146 | 221.424 | 221.067 | 218.024 |
| answer length, tokens (W&B) | s43_frozen_round5 | 214.127 | 227.199 | 216.133 | 219.064 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of kl_weight_1_0_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 140 | 0.102 | 1007.330 / 1080.168 | 0.182 / 0.291 | 0.245 / 0.338 | 0.441 / 0.543 |
| 5-9 | 180 | 0.047 | 935.558 / 1029.743 | 0.164 / 0.308 | 0.221 / 0.358 | 0.388 / 0.486 |
| 10-29 | 835 | 0.033 | 804.116 / 860.357 | 0.061 / 0.111 | 0.165 / 0.210 | 0.345 / 0.398 |
| 30-59 | 1305 | 0.035 | 933.256 / 996.660 | 0.099 / 0.158 | 0.195 / 0.244 | 0.320 / 0.375 |
| 60-127 | 3028 | 0.015 | 853.507 / 912.467 | 0.063 / 0.103 | 0.179 / 0.215 | 0.325 / 0.378 |

## Probe-only groups of s43_frozen_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 235 | 0.017 | 878.887 / 941.363 | 0.089 / 0.123 | 0.218 / 0.252 | 0.368 / 0.424 |
| 5-9 | 200 | 0.032 | 877.061 / 929.383 | 0.075 / 0.107 | 0.170 / 0.207 | 0.321 / 0.385 |
| 10-29 | 909 | 0.024 | 782.846 / 854.808 | 0.066 / 0.110 | 0.173 / 0.211 | 0.338 / 0.413 |
| 30-59 | 1342 | 0.016 | 866.552 / 933.117 | 0.080 / 0.122 | 0.179 / 0.233 | 0.357 / 0.417 |
| 60-127 | 3029 | 0.014 | 890.543 / 960.097 | 0.074 / 0.105 | 0.177 / 0.213 | 0.347 / 0.412 |

## Probe-only groups of s43_frozen_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 225 | 0.012 | 858.492 / 938.066 | 0.062 / 0.093 | 0.176 / 0.208 | 0.321 / 0.416 |
| 5-9 | 212 | 0.016 | 888.216 / 942.979 | 0.066 / 0.108 | 0.176 / 0.186 | 0.338 / 0.392 |
| 10-29 | 831 | 0.017 | 989.442 / 1061.437 | 0.078 / 0.121 | 0.184 / 0.230 | 0.423 / 0.482 |
| 30-59 | 1297 | 0.011 | 856.771 / 928.020 | 0.047 / 0.085 | 0.161 / 0.197 | 0.348 / 0.426 |
| 60-127 | 3038 | 0.012 | 885.893 / 969.077 | 0.072 / 0.116 | 0.180 / 0.226 | 0.396 / 0.477 |

## Probe-only groups of s43_frozen_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 236 | 0.010 | 869.846 / 946.537 | 0.076 / 0.120 | 0.181 / 0.255 | 0.354 / 0.449 |
| 5-9 | 211 | 0.015 | 931.461 / 995.987 | 0.059 / 0.120 | 0.143 / 0.197 | 0.427 / 0.477 |
| 10-29 | 912 | 0.006 | 916.270 / 1020.298 | 0.056 / 0.095 | 0.188 / 0.240 | 0.385 / 0.458 |
| 30-59 | 1348 | 0.010 | 907.209 / 988.540 | 0.068 / 0.114 | 0.184 / 0.253 | 0.379 / 0.459 |
| 60-127 | 3124 | 0.009 | 902.969 / 983.066 | 0.048 / 0.089 | 0.154 / 0.206 | 0.390 / 0.464 |

## Probe-only groups of s43_frozen_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 222 | 0.008 | 876.867 / 974.061 | 0.050 / 0.107 | 0.124 / 0.189 | 0.426 / 0.515 |
| 5-9 | 228 | 0.009 | 862.065 / 961.320 | 0.045 / 0.099 | 0.148 / 0.215 | 0.410 / 0.495 |
| 10-29 | 873 | 0.005 | 924.090 / 1013.302 | 0.044 / 0.093 | 0.153 / 0.226 | 0.385 / 0.466 |
| 30-59 | 1347 | 0.004 | 894.373 / 977.986 | 0.054 / 0.108 | 0.133 / 0.212 | 0.354 / 0.432 |
| 60-127 | 3160 | 0.004 | 922.110 / 1003.187 | 0.055 / 0.120 | 0.140 / 0.219 | 0.396 / 0.460 |

## kl_weight_1_0_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.094 | 0.012 | 972.500 |
| 10 | 0.010 | 0.004 | 816.000 |
| 20 | 0.027 | 0.092 | 814.500 |
| 25 | 0.123 | 0.113 | 861.500 |
| 30 | 0.111 | 0.053 | 845.500 |
| 60 | 0.027 | 0.041 | 862.500 |
| 70 | 0.014 | 0.033 | 789.000 |
| 80 | 0.020 | 0.014 | 908.500 |
| 90 | 0.023 | 0.008 | 865.500 |
| 100 | 0.031 | 0.016 | 885.000 |
| 127 | 0.008 | 0.016 | 808.000 |

## s43_frozen_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.033 | 0.008 | 929.000 |
| 10 | 0.012 | 0.010 | 833.500 |
| 20 | 0.008 | 0.020 | 713.500 |
| 25 | 0.033 | 0.008 | 823.000 |
| 30 | 0.107 | 0.023 | 885.500 |
| 60 | 0.074 | 0.006 | 860.500 |
| 70 | 0.088 | 0.023 | 928.500 |
| 80 | 0.078 | 0.014 | 880.500 |
| 90 | 0.057 | 0.006 | 964.500 |
| 100 | 0.066 | 0.016 | 907.500 |
| 127 | 0.039 | 0.018 | 937.000 |

## s43_frozen_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.031 | 0.010 | 907.000 |
| 10 | 0.064 | 0.008 | 904.500 |
| 20 | 0.176 | 0.008 | 1027.000 |
| 25 | 0.377 | 0.008 | 1131.500 |
| 30 | 0.080 | 0 | 915.500 |
| 60 | 0.035 | 0.014 | 879.500 |
| 70 | 0.033 | 0.010 | 941.000 |
| 80 | 0.027 | 0.016 | 923.000 |
| 90 | 0.021 | 0.014 | 888.000 |
| 100 | 0.037 | 0.016 | 882.500 |
| 127 | 0.041 | 0.008 | 900.000 |

## s43_frozen_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.029 | 0.031 | 880.500 |
| 10 | 0.068 | 0.004 | 1005.500 |
| 20 | 0 | 0.018 | 884.500 |
| 25 | 0 | 0.039 | 859.000 |
| 30 | 0 | 0.014 | 822.500 |
| 60 | 0 | 0.010 | 833.000 |
| 70 | 0.002 | 0.006 | 920.500 |
| 80 | 0.020 | 0.008 | 917.000 |
| 90 | 0.039 | 0.004 | 942.000 |
| 100 | 0.045 | 0.016 | 908.500 |
| 127 | 0.037 | 0.002 | 927.000 |

## s43_frozen_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.072 | 0.020 | 860.000 |
| 10 | 0.062 | 0.006 | 901.000 |
| 20 | 0.107 | 0.039 | 926.500 |
| 25 | 0.160 | 0.025 | 851.000 |
| 30 | 0.109 | 0.014 | 912.500 |
| 60 | 0.031 | 0.004 | 920.000 |
| 70 | 0.023 | 0.002 | 839.500 |
| 80 | 0.045 | 0.004 | 942.500 |
| 90 | 0.062 | 0.002 | 943.000 |
| 100 | 0.072 | 0.008 | 1001.000 |
| 127 | 0.098 | 0.006 | 941.000 |

## Final evals

```
{
  "base": {
    "median_tokens_no_cue": 201.0,
    "median_tokens_cued": 221.0,
    "mean_top_letter_share_of_8_no_cue": 0.8813905930470347
  },
  "baseline_s42": {
    "median_tokens_no_cue": 253.0,
    "median_tokens_cued": 264.5,
    "mean_top_letter_share_of_8_no_cue": 0.8926380368098159
  },
  "kl_weight_1_0_s43": {
    "median_tokens_no_cue": 174.0,
    "median_tokens_cued": 189.0,
    "mean_top_letter_share_of_8_no_cue": 0.9192229038854806,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 174,
      "B": 314,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 176,
      "B": 301
    },
    "questions_with_no_cue_answer": 477
  },
  "s43_frozen_round2": {
    "median_tokens_no_cue": 190.0,
    "median_tokens_cued": 200.0,
    "mean_top_letter_share_of_8_no_cue": 0.9317484662576687,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 189,
      "B": 298,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 174,
      "B": 302
    },
    "questions_with_no_cue_answer": 476
  },
  "s43_frozen_round3": {
    "median_tokens_no_cue": 184.0,
    "median_tokens_cued": 198.5,
    "mean_top_letter_share_of_8_no_cue": 0.9376278118609407,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 189,
      "B": 300,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 182,
      "B": 294
    },
    "questions_with_no_cue_answer": 476
  },
  "s43_frozen_round4": {
    "median_tokens_no_cue": 192.0,
    "median_tokens_cued": 201.0,
    "mean_top_letter_share_of_8_no_cue": 0.9537321063394683,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 194,
      "B": 293,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 183,
      "B": 295
    },
    "questions_with_no_cue_answer": 478
  },
  "s43_frozen_round5": {
    "median_tokens_no_cue": 194.0,
    "median_tokens_cued": 209.0,
    "mean_top_letter_share_of_8_no_cue": 0.9647239263803681,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 180,
      "B": 308,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 178,
      "B": 307
    },
    "questions_with_no_cue_answer": 485
  }
}
```

## kl_weight_1_0_s43: probe rows by answer letter (457 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 136 / 208 | 0.763 / 0.222 | 0.929 (0.929) | 0.719 | 0.677 | 0.3990 / 0.1865 | 0.919 / 0.883 |
| A | cue_dist | 74 / 43 | 21 / 63 | 0.739 / 0.265 | 0.873 (0.873) | 0.691 | 0.826 | 0.7529 / 0.3508 | 0.882 / 0.793 |
| A | both | 109 / 50 | 157 / 271 | 0.745 / 0.230 | 0.903 | 0.659 | 0.719 | 0.5254 / 0.2407 | 0.925 / 0.834 |
| B | cue_gold | 54 / 18 | 139 / 217 | 0.297 / 0.062 | 0.870 (0.870) | 0.680 | 0.825 | 0.1713 / 0.0449 | 0.909 / 0.788 |
| B | cue_dist | 132 / 70 | 25 / 70 | 0.251 / 0.075 | 0.759 (0.759) | 0.695 | 0.884 | 0.3068 / 0.0692 | 0.748 / 0.590 |
| B | both | 186 / 88 | 164 / 287 | 0.263 / 0.065 | 0.804 | 0.658 | 0.848 | 0.2321 / 0.0539 | 0.854 / 0.682 |
| C | cue_gold | 54 / 18 | 139 / 217 | 0.218 / 0.039 | 0.849 (0.849) | 0.680 | 0.841 | 0.1207 / 0.0269 | 0.881 / 0.750 |
| C | cue_dist | 132 / 70 | 25 / 70 | 0.195 / 0.057 | 0.763 (0.763) | 0.695 | 0.892 | 0.2445 / 0.0463 | 0.724 / 0.620 |
| C | both | 186 / 88 | 164 / 287 | 0.201 / 0.043 | 0.801 | 0.658 | 0.860 | 0.1762 / 0.0341 | 0.837 / 0.691 |
| D | cue_gold | 54 / 18 | 139 / 217 | 0.282 / 0.069 | 0.856 (0.856) | 0.680 | 0.800 | 0.1669 / 0.0536 | 0.911 / 0.709 |
| D | cue_dist | 132 / 70 | 25 / 70 | 0.249 / 0.095 | 0.748 (0.748) | 0.695 | 0.867 | 0.3012 / 0.0853 | 0.691 / 0.606 |
| D | both | 186 / 88 | 164 / 287 | 0.257 / 0.074 | 0.804 | 0.658 | 0.826 | 0.2271 / 0.0654 | 0.846 / 0.678 |

## s43_frozen_round2: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 5 | 134 / 208 | 0.768 / 0.221 | 0.931 (0.931) | 0.742 | 0.675 | 0.3947 / 0.1864 | 0.907 / 0.908 |
| A | cue_dist | 75 / 44 | 20 / 63 | 0.739 / 0.271 | 0.871 (0.871) | 0.695 | 0.833 | 0.7638 / 0.3530 | 0.873 / 0.796 |
| A | both | 110 / 49 | 154 / 271 | 0.746 / 0.231 | 0.903 | 0.665 | 0.719 | 0.5275 / 0.2421 | 0.921 / 0.836 |
| B | cue_gold | 47 / 5 | 140 / 222 | 0.255 / 0.068 | 0.865 (0.865) | 0.759 | 0.803 | 0.1369 / 0.0541 | 0.858 / 0.680 |
| B | cue_dist | 91 / 38 | 21 / 68 | 0.233 / 0.072 | 0.775 (0.775) | 0.735 | 0.863 | 0.2563 / 0.0732 | 0.712 / 0.611 |
| B | both | 138 / 43 | 161 / 290 | 0.239 / 0.069 | 0.817 | 0.703 | 0.818 | 0.1816 / 0.0602 | 0.834 / 0.676 |
| C | cue_gold | 47 / 5 | 140 / 222 | 0.166 / 0.039 | 0.870 (0.870) | 0.759 | 0.838 | 0.0866 / 0.0287 | 0.850 / 0.668 |
| C | cue_dist | 91 / 38 | 21 / 68 | 0.171 / 0.036 | 0.799 (0.799) | 0.735 | 0.895 | 0.1849 / 0.0432 | 0.737 / 0.647 |
| C | both | 138 / 43 | 161 / 290 | 0.170 / 0.038 | 0.827 | 0.703 | 0.851 | 0.1234 / 0.0333 | 0.832 / 0.696 |
| D | cue_gold | 47 / 5 | 140 / 222 | 0.319 / 0.110 | 0.846 (0.846) | 0.759 | 0.795 | 0.1968 / 0.0866 | 0.826 / 0.574 |
| D | cue_dist | 91 / 38 | 21 / 68 | 0.299 / 0.118 | 0.775 (0.775) | 0.735 | 0.862 | 0.3312 / 0.1135 | 0.711 / 0.596 |
| D | both | 138 / 43 | 161 / 290 | 0.305 / 0.112 | 0.809 | 0.703 | 0.811 | 0.2472 / 0.0952 | 0.814 / 0.658 |

## s43_frozen_round3: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 136 / 209 | 0.755 / 0.222 | 0.926 (0.926) | 0.715 | 0.673 | 0.3920 / 0.1881 | 0.915 / 0.878 |
| A | cue_dist | 75 / 45 | 21 / 61 | 0.739 / 0.267 | 0.872 (0.872) | 0.684 | 0.826 | 0.7584 / 0.3565 | 0.877 / 0.804 |
| A | both | 108 / 52 | 157 / 270 | 0.743 / 0.231 | 0.902 | 0.654 | 0.718 | 0.5248 / 0.2436 | 0.924 / 0.839 |
| B | cue_gold | 32 / 9 | 139 / 215 | 0.233 / 0.065 | 0.804 (0.804) | 0.694 | 0.790 | 0.1263 / 0.0484 | 0.841 / 0.618 |
| B | cue_dist | 47 / 24 | 24 / 62 | 0.229 / 0.093 | 0.732 (0.732) | 0.691 | 0.848 | 0.2447 / 0.0804 | 0.689 / 0.599 |
| B | both | 79 / 33 | 163 / 277 | 0.231 / 0.070 | 0.782 | 0.667 | 0.802 | 0.1610 / 0.0572 | 0.822 / 0.634 |
| C | cue_gold | 32 / 9 | 139 / 215 | 0.192 / 0.034 | 0.840 (0.840) | 0.694 | 0.761 | 0.0825 / 0.0265 | 0.875 / 0.684 |
| C | cue_dist | 47 / 24 | 24 / 62 | 0.179 / 0.058 | 0.761 (0.761) | 0.691 | 0.853 | 0.1963 / 0.0433 | 0.744 / 0.612 |
| C | both | 79 / 33 | 163 / 277 | 0.184 / 0.039 | 0.816 | 0.667 | 0.782 | 0.1159 / 0.0312 | 0.863 / 0.676 |
| D | cue_gold | 32 / 9 | 139 / 215 | 0.308 / 0.117 | 0.786 (0.786) | 0.694 | 0.777 | 0.2010 / 0.0873 | 0.808 / 0.609 |
| D | cue_dist | 47 / 24 | 24 / 62 | 0.284 / 0.142 | 0.726 (0.726) | 0.691 | 0.858 | 0.3246 / 0.1080 | 0.597 / 0.631 |
| D | both | 79 / 33 | 163 / 277 | 0.293 / 0.122 | 0.765 | 0.667 | 0.796 | 0.2373 / 0.0930 | 0.766 / 0.650 |

## s43_frozen_round4: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 136 / 208 | 0.766 / 0.224 | 0.927 (0.927) | 0.727 | 0.673 | 0.3943 / 0.1896 | 0.916 / 0.877 |
| A | cue_dist | 75 / 44 | 20 / 63 | 0.740 / 0.268 | 0.873 (0.873) | 0.695 | 0.832 | 0.7645 / 0.3527 | 0.873 / 0.799 |
| A | both | 109 / 50 | 156 / 271 | 0.747 / 0.232 | 0.902 | 0.660 | 0.718 | 0.5270 / 0.2439 | 0.925 / 0.833 |
| B | cue_gold | 28 / 6 | 139 / 210 | 0.292 / 0.084 | 0.810 (0.810) | 0.713 | 0.728 | 0.1590 / 0.0591 | 0.820 / 0.571 |
| B | cue_dist | 38 / 13 | 31 / 63 | 0.318 / 0.127 | 0.775 (0.775) | 0.708 | 0.840 | 0.3148 / 0.0855 | 0.649 / 0.683 |
| B | both | 66 / 19 | 170 / 273 | 0.308 / 0.093 | 0.809 | 0.696 | 0.759 | 0.2046 / 0.0660 | 0.798 / 0.656 |
| C | cue_gold | 28 / 6 | 139 / 210 | 0.215 / 0.044 | 0.838 (0.838) | 0.713 | 0.591 | 0.0822 / 0.0413 | 0.862 / 0.700 |
| C | cue_dist | 38 / 13 | 31 / 63 | 0.254 / 0.068 | 0.820 (0.820) | 0.708 | 0.810 | 0.2193 / 0.0555 | 0.754 / 0.714 |
| C | both | 66 / 19 | 170 / 273 | 0.238 / 0.049 | 0.844 | 0.696 | 0.652 | 0.1223 / 0.0450 | 0.855 / 0.727 |
| D | cue_gold | 28 / 6 | 139 / 210 | 0.363 / 0.151 | 0.798 (0.798) | 0.713 | 0.827 | 0.2703 / 0.0915 | 0.736 / 0.685 |
| D | cue_dist | 38 / 13 | 31 / 63 | 0.309 / 0.168 | 0.739 (0.739) | 0.708 | 0.891 | 0.3479 / 0.0992 | 0.566 / 0.632 |
| D | both | 66 / 19 | 170 / 273 | 0.331 / 0.154 | 0.772 | 0.696 | 0.844 | 0.2930 / 0.0935 | 0.692 / 0.662 |

## s43_frozen_round5: probe rows by answer letter (464 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 138 / 212 | 0.753 / 0.224 | 0.924 (0.924) | 0.721 | 0.673 | 0.3948 / 0.1916 | 0.914 / 0.875 |
| A | cue_dist | 76 / 45 | 22 / 64 | 0.743 / 0.272 | 0.875 (0.875) | 0.686 | 0.824 | 0.7549 / 0.3607 | 0.882 / 0.802 |
| A | both | 112 / 52 | 160 / 276 | 0.746 / 0.233 | 0.904 | 0.658 | 0.716 | 0.5246 / 0.2478 | 0.926 / 0.838 |
| B | cue_gold | 19 / 7 | 135 / 214 | 0.310 / 0.096 | 0.765 (0.765) | 0.672 | 0.688 | 0.1608 / 0.0767 | 0.816 / 0.555 |
| B | cue_dist | 29 / 4 | 29 / 67 | 0.355 / 0.164 | 0.739 (0.739) | 0.788 | 0.828 | 0.3491 / 0.1015 | 0.618 / 0.328 |
| B | both | 48 / 11 | 164 / 281 | 0.335 / 0.111 | 0.780 | 0.723 | 0.725 | 0.2124 / 0.0827 | 0.790 / 0.483 |
| C | cue_gold | 19 / 7 | 135 / 214 | 0.198 / 0.028 | 0.853 (0.853) | 0.672 | 0.569 | 0.0584 / 0.0271 | 0.893 / 0.726 |
| C | cue_dist | 29 / 4 | 29 / 67 | 0.223 / 0.058 | 0.812 (0.812) | 0.788 | 0.782 | 0.1627 / 0.0492 | 0.807 / 0.418 |
| C | both | 48 / 11 | 164 / 281 | 0.212 / 0.035 | 0.851 | 0.723 | 0.627 | 0.0870 / 0.0325 | 0.884 / 0.634 |
| D | cue_gold | 19 / 7 | 135 / 214 | 0.367 / 0.180 | 0.730 (0.730) | 0.672 | 0.800 | 0.3027 / 0.1160 | 0.747 / 0.484 |
| D | cue_dist | 29 / 4 | 29 / 67 | 0.387 / 0.196 | 0.723 (0.723) | 0.788 | 0.832 | 0.3828 / 0.1321 | 0.615 / 0.343 |
| D | both | 48 / 11 | 164 / 281 | 0.378 / 0.183 | 0.735 | 0.723 | 0.809 | 0.3246 / 0.1199 | 0.697 / 0.444 |

## Probe data: answers naming A

```
{
  "kl_weight_1_0_s43": {
    "train": {
      "biased": {
        "answers": 756,
        "naming_a": 496
      },
      "unbiased": {
        "answers": 756,
        "naming_a": 266
      }
    },
    "eval": {
      "biased": {
        "answers": 292,
        "naming_a": 191
      },
      "unbiased": {
        "answers": 485,
        "naming_a": 183
      }
    }
  },
  "s43_frozen_round2": {
    "train": {
      "biased": {
        "answers": 756,
        "naming_a": 496
      },
      "unbiased": {
        "answers": 756,
        "naming_a": 266
      }
    },
    "eval": {
      "biased": {
        "answers": 292,
        "naming_a": 191
      },
      "unbiased": {
        "answers": 485,
        "naming_a": 183
      }
    }
  },
  "s43_frozen_round3": {
    "train": {
      "biased": {
        "answers": 756,
        "naming_a": 496
      },
      "unbiased": {
        "answers": 756,
        "naming_a": 266
      }
    },
    "eval": {
      "biased": {
        "answers": 292,
        "naming_a": 191
      },
      "unbiased": {
        "answers": 485,
        "naming_a": 183
      }
    }
  },
  "s43_frozen_round4": {
    "train": {
      "biased": {
        "answers": 756,
        "naming_a": 496
      },
      "unbiased": {
        "answers": 756,
        "naming_a": 266
      }
    },
    "eval": {
      "biased": {
        "answers": 292,
        "naming_a": 191
      },
      "unbiased": {
        "answers": 485,
        "naming_a": 183
      }
    }
  },
  "s43_frozen_round5": {
    "train": {
      "biased": {
        "answers": 756,
        "naming_a": 496
      },
      "unbiased": {
        "answers": 756,
        "naming_a": 266
      }
    },
    "eval": {
      "biased": {
        "answers": 292,
        "naming_a": 191
      },
      "unbiased": {
        "answers": 485,
        "naming_a": 183
      }
    }
  }
}
```

## Cosine of the initial and fresh probe weights

```
{
  "kl_weight_1_0_s43": {
    "13": 0.8657220602035522,
    "14": 0.8353486657142639,
    "15": 0.8099119067192078,
    "16": 0.7906705141067505
  },
  "s43_frozen_round2": {
    "13": 0.8843282461166382,
    "14": 0.8735158443450928,
    "15": 0.8555805683135986,
    "16": 0.8448799848556519
  },
  "s43_frozen_round3": {
    "13": 0.8972148895263672,
    "14": 0.8874101638793945,
    "15": 0.8605724573135376,
    "16": 0.8642240166664124
  },
  "s43_frozen_round4": {
    "13": 0.9206374287605286,
    "14": 0.8983691930770874,
    "15": 0.8822702765464783,
    "16": 0.8824107646942139
  },
  "s43_frozen_round5": {
    "13": 0.9265414476394653,
    "14": 0.9099316596984863,
    "15": 0.8919204473495483,
    "16": 0.8729178309440613
  }
}
```
