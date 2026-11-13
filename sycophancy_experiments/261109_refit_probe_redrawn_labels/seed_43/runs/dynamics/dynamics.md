# Training dynamics and probe rows: kl_weight_1_0_s43, s43_refit_round2, s43_refit_round3, s43_refit_round4, s43_refit_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s43 0.3, s43_refit_round2 0.3, s43_refit_round3 0.3, s43_refit_round4 0.3, s43_refit_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s43 | 32 | 41.750 | 43.500 | 44.529 |
| groups where all 8 answers name the cue (of 64) | s43_refit_round2 | 46.600 | 38.150 | 39.300 | 42.324 |
| groups where all 8 answers name the cue (of 64) | s43_refit_round3 | 37 | 41.450 | 43.133 | 46.191 |
| groups where all 8 answers name the cue (of 64) | s43_refit_round4 | 44.700 | 39.050 | 40.100 | 41.618 |
| groups where all 8 answers name the cue (of 64) | s43_refit_round5 | 40.400 | 37.900 | 37.433 | 39.485 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s43 | 26.500 | 17.200 | 16.800 | 15.382 |
| groups with mixed task rewards (of 64) | s43_refit_round2 | 13.300 | 17.700 | 14.700 | 10.868 |
| groups with mixed task rewards (of 64) | s43_refit_round3 | 13.800 | 10.550 | 10.533 | 7.103 |
| groups with mixed task rewards (of 64) | s43_refit_round4 | 6.100 | 6.950 | 9 | 8 |
| groups with mixed task rewards (of 64) | s43_refit_round5 | 6.400 | 11.200 | 7.567 | 5.279 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s43 | 7.800 | 6.950 | 6.333 | 6.338 |
|   of which the cue is not at the training label | s43_refit_round2 | 5.600 | 6.300 | 5.100 | 4.309 |
|   of which the cue is not at the training label | s43_refit_round3 | 4.200 | 3.550 | 4.067 | 2.985 |
|   of which the cue is not at the training label | s43_refit_round4 | 2 | 2.350 | 2.800 | 3.147 |
|   of which the cue is not at the training label | s43_refit_round5 | 1.900 | 4.100 | 1.967 | 1.706 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s43 | 126 | 99.050 | 85.167 | 81.926 |
| answers not naming the cue (of 512) | s43_refit_round2 | 76.600 | 127.100 | 131.733 | 123.809 |
| answers not naming the cue (of 512) | s43_refit_round3 | 157.200 | 136.250 | 120.800 | 110.632 |
| answers not naming the cue (of 512) | s43_refit_round4 | 125.500 | 168.450 | 152.033 | 143.103 |
| answers not naming the cue (of 512) | s43_refit_round5 | 163.200 | 164.050 | 179.933 | 173.515 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s43 | 0.355 | 0.296 | 0.305 | 0.251 |
| mean absolute advantage per answer | s43_refit_round2 | 0.287 | 0.297 | 0.248 | 0.209 |
| mean absolute advantage per answer | s43_refit_round3 | 0.201 | 0.170 | 0.170 | 0.149 |
| mean absolute advantage per answer | s43_refit_round4 | 0.139 | 0.118 | 0.145 | 0.136 |
| mean absolute advantage per answer | s43_refit_round5 | 0.142 | 0.174 | 0.137 | 0.124 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s43 | 0.112 | 0.117 | 0.122 | 0.071 |
|   from probe-only groups | s43_refit_round2 | 0.167 | 0.126 | 0.101 | 0.101 |
|   from probe-only groups | s43_refit_round3 | 0.089 | 0.078 | 0.063 | 0.073 |
|   from probe-only groups | s43_refit_round4 | 0.096 | 0.060 | 0.068 | 0.060 |
|   from probe-only groups | s43_refit_round5 | 0.101 | 0.089 | 0.081 | 0.086 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s43 | 0.017 | -0.003 | 0.007 | 0.009 |
| advantage on cue-naming answers, per answer | s43_refit_round2 | -0.010 | 0.004 | 0.007 | -0.001 |
| advantage on cue-naming answers, per answer | s43_refit_round3 | 0.006 | 0.002 | 0.003 | -0.000 |
| advantage on cue-naming answers, per answer | s43_refit_round4 | -0.002 | 0.002 | 0.006 | 0.001 |
| advantage on cue-naming answers, per answer | s43_refit_round5 | -0.003 | -0.006 | 0.003 | -0.002 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s43 | 0.087 | 0.102 | 0.140 | 0.066 |
| gradient norm (W&B, before clipping) | s43_refit_round2 | 0.069 | 0.069 | 0.056 | 0.050 |
| gradient norm (W&B, before clipping) | s43_refit_round3 | 0.052 | 0.047 | 0.051 | 0.041 |
| gradient norm (W&B, before clipping) | s43_refit_round4 | 0.034 | 0.036 | 0.040 | 0.038 |
| gradient norm (W&B, before clipping) | s43_refit_round5 | 0.031 | 0.041 | 0.034 | 0.032 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s43 | 0.012 | 0.101 | 0.129 | 0.108 |
| KL from the base model (W&B) | s43_refit_round2 | 0.129 | 0.193 | 0.146 | 0.143 |
| KL from the base model (W&B) | s43_refit_round3 | 0.128 | 0.096 | 0.098 | 0.087 |
| KL from the base model (W&B) | s43_refit_round4 | 0.075 | 0.064 | 0.068 | 0.080 |
| KL from the base model (W&B) | s43_refit_round5 | 0.076 | 0.085 | 0.085 | 0.076 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s43 | 0.427 | 0.401 | 0.440 | 0.446 |
| token entropy (W&B) | s43_refit_round2 | 0.470 | 0.485 | 0.475 | 0.457 |
| token entropy (W&B) | s43_refit_round3 | 0.458 | 0.462 | 0.461 | 0.496 |
| token entropy (W&B) | s43_refit_round4 | 0.478 | 0.499 | 0.461 | 0.483 |
| token entropy (W&B) | s43_refit_round5 | 0.485 | 0.478 | 0.453 | 0.465 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s43 | 243.601 | 195.433 | 229.320 | 208.106 |
| answer length, tokens (W&B) | s43_refit_round2 | 201.705 | 180.668 | 233.334 | 226.778 |
| answer length, tokens (W&B) | s43_refit_round3 | 223.757 | 204.264 | 212.907 | 214.621 |
| answer length, tokens (W&B) | s43_refit_round4 | 193.604 | 226.128 | 214.074 | 210.947 |
| answer length, tokens (W&B) | s43_refit_round5 | 209.189 | 205.311 | 201.085 | 207.225 |
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

## Probe-only groups of s43_refit_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 241 | 0.081 | 859.504 / 918.934 | 0.067 / 0.107 | 0.165 / 0.210 | 0.350 / 0.409 |
| 5-9 | 225 | 0.075 | 815.602 / 888.681 | 0.069 / 0.124 | 0.227 / 0.281 | 0.307 / 0.412 |
| 10-29 | 763 | 0.070 | 724.015 / 771.936 | 0.063 / 0.104 | 0.153 / 0.198 | 0.238 / 0.305 |
| 30-59 | 1179 | 0.049 | 957.365 / 1017.409 | 0.062 / 0.091 | 0.162 / 0.203 | 0.312 / 0.386 |
| 60-127 | 2878 | 0.046 | 925.643 / 990.485 | 0.077 / 0.116 | 0.181 / 0.235 | 0.316 / 0.383 |

## Probe-only groups of s43_refit_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 197 | 0.056 | 925.890 / 1014.313 | 0.077 / 0.133 | 0.148 / 0.212 | 0.316 / 0.367 |
| 5-9 | 173 | 0.039 | 864.435 / 917.438 | 0.045 / 0.091 | 0.132 / 0.191 | 0.249 / 0.286 |
| 10-29 | 829 | 0.028 | 848.139 / 891.768 | 0.040 / 0.074 | 0.097 / 0.141 | 0.260 / 0.316 |
| 30-59 | 1294 | 0.013 | 887.660 / 950.860 | 0.041 / 0.078 | 0.115 / 0.171 | 0.315 / 0.371 |
| 60-127 | 3141 | 0.016 | 896.677 / 961.300 | 0.050 / 0.110 | 0.126 / 0.189 | 0.332 / 0.400 |

## Probe-only groups of s43_refit_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 214 | 0.058 | 853.522 / 905.994 | 0.041 / 0.099 | 0.117 / 0.187 | 0.314 / 0.401 |
| 5-9 | 233 | 0.034 | 789.694 / 819.334 | 0.062 / 0.079 | 0.093 / 0.149 | 0.265 / 0.328 |
| 10-29 | 781 | 0.026 | 912.660 / 1000.613 | 0.061 / 0.093 | 0.133 / 0.196 | 0.348 / 0.431 |
| 30-59 | 1203 | 0.032 | 857.186 / 937.016 | 0.078 / 0.119 | 0.146 / 0.217 | 0.333 / 0.416 |
| 60-127 | 2830 | 0.024 | 868.851 / 930.330 | 0.060 / 0.103 | 0.114 / 0.185 | 0.318 / 0.395 |

## Probe-only groups of s43_refit_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 208 | 0.064 | 882.649 / 928.743 | 0.061 / 0.099 | 0.103 / 0.159 | 0.296 / 0.383 |
| 5-9 | 196 | 0.051 | 831.066 / 868.439 | 0.028 / 0.054 | 0.056 / 0.103 | 0.256 / 0.321 |
| 10-29 | 758 | 0.052 | 823.281 / 881.943 | 0.029 / 0.051 | 0.049 / 0.087 | 0.244 / 0.297 |
| 30-59 | 1123 | 0.047 | 825.052 / 881.364 | 0.035 / 0.059 | 0.071 / 0.107 | 0.269 / 0.333 |
| 60-127 | 2685 | 0.047 | 842.686 / 898.047 | 0.047 / 0.064 | 0.092 / 0.129 | 0.290 / 0.370 |

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

## s43_refit_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.018 | 0.010 | 813.500 |
| 10 | 0.096 | 0.008 | 748.500 |
| 20 | 0.117 | 0.002 | 720.000 |
| 25 | 0.043 | 0.016 | 719.000 |
| 30 | 0.053 | 0.002 | 854.000 |
| 60 | 0.023 | 0.020 | 867.000 |
| 70 | 0.004 | 0 | 910.000 |
| 80 | 0.021 | 0 | 911.500 |
| 90 | 0.021 | 0.002 | 1000.500 |
| 100 | 0.021 | 0.004 | 966.000 |
| 127 | 0.018 | 0.006 | 960.500 |

## s43_refit_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.027 | 0.008 | 927.500 |
| 10 | 0.053 | 0.016 | 767.000 |
| 20 | 0.168 | 0 | 893.500 |
| 25 | 0.125 | 0 | 855.000 |
| 30 | 0.107 | 0 | 849.000 |
| 60 | 0.078 | 0.002 | 958.500 |
| 70 | 0.020 | 0.002 | 929.000 |
| 80 | 0.055 | 0.004 | 910.500 |
| 90 | 0.023 | 0.008 | 854.500 |
| 100 | 0.033 | 0.004 | 912.500 |
| 127 | 0.039 | 0 | 885.000 |

## s43_refit_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.045 | 0.002 | 873.000 |
| 10 | 0.164 | 0 | 838.000 |
| 20 | 0.186 | 0 | 1012.000 |
| 25 | 0.090 | 0.004 | 930.500 |
| 30 | 0.033 | 0 | 866.000 |
| 60 | 0.186 | 0.018 | 857.500 |
| 70 | 0.186 | 0 | 871.500 |
| 80 | 0.201 | 0.004 | 852.000 |
| 90 | 0.121 | 0.004 | 848.000 |
| 100 | 0.111 | 0 | 889.000 |
| 127 | 0.086 | 0.002 | 860.000 |

## s43_refit_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.113 | 0.002 | 882.500 |
| 10 | 0.021 | 0.008 | 814.000 |
| 20 | 0.006 | 0.002 | 737.000 |
| 25 | 0.023 | 0.002 | 787.000 |
| 30 | 0.047 | 0.004 | 859.000 |
| 60 | 0.086 | 0 | 823.000 |
| 70 | 0.104 | 0.002 | 863.000 |
| 80 | 0.111 | 0.018 | 876.000 |
| 90 | 0.080 | 0.016 | 834.000 |
| 100 | 0.135 | 0.004 | 908.500 |
| 127 | 0.080 | 0 | 868.000 |

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
  "s43_refit_round2": {
    "median_tokens_no_cue": 201.0,
    "median_tokens_cued": 216.0,
    "mean_top_letter_share_of_8_no_cue": 0.9383946830265849,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 189,
      "B": 299,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 193,
      "B": 285
    },
    "questions_with_no_cue_answer": 478
  },
  "s43_refit_round3": {
    "median_tokens_no_cue": 182.0,
    "median_tokens_cued": 194.0,
    "mean_top_letter_share_of_8_no_cue": 0.9371165644171779,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 169,
      "B": 314,
      "null": 6
    },
    "no_cue_answer_letters": {
      "A": 166,
      "B": 308
    },
    "questions_with_no_cue_answer": 474
  },
  "s43_refit_round4": {
    "median_tokens_no_cue": 179.0,
    "median_tokens_cued": 187.0,
    "mean_top_letter_share_of_8_no_cue": 0.9396728016359919,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 193,
      "B": 294,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 192,
      "B": 285
    },
    "questions_with_no_cue_answer": 477
  },
  "s43_refit_round5": {
    "median_tokens_no_cue": 178.0,
    "median_tokens_cued": 185.0,
    "mean_top_letter_share_of_8_no_cue": 0.9514314928425358,
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
      "A": 175,
      "B": 306
    },
    "questions_with_no_cue_answer": 481
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

## s43_refit_round2: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 137 / 208 | 0.693 / 0.245 | 0.910 (0.910) | 0.716 | 0.622 | 0.3681 / 0.2327 | 0.899 / 0.866 |
| A | cue_dist | 74 / 44 | 22 / 61 | 0.697 / 0.324 | 0.842 (0.842) | 0.681 | 0.769 | 0.6945 / 0.4039 | 0.827 / 0.782 |
| A | both | 108 / 51 | 159 / 269 | 0.696 / 0.260 | 0.887 | 0.654 | 0.667 | 0.4855 / 0.2889 | 0.901 / 0.830 |
| B | cue_gold | 25 / 15 | 144 / 214 | 0.350 / 0.123 | 0.839 (0.839) | 0.611 | 0.714 | 0.2053 / 0.1026 | 0.804 / 0.872 |
| B | cue_dist | 55 / 32 | 28 / 59 | 0.375 / 0.168 | 0.754 (0.754) | 0.655 | 0.828 | 0.3945 / 0.1589 | 0.727 / 0.660 |
| B | both | 80 / 47 | 172 / 273 | 0.367 / 0.132 | 0.810 | 0.622 | 0.749 | 0.2676 / 0.1186 | 0.818 / 0.759 |
| C | cue_gold | 25 / 15 | 144 / 214 | 0.476 / 0.152 | 0.875 (0.875) | 0.611 | 0.619 | 0.2342 / 0.1483 | 0.871 / 0.854 |
| C | cue_dist | 55 / 32 | 28 / 59 | 0.472 / 0.206 | 0.812 (0.812) | 0.655 | 0.802 | 0.4737 / 0.2156 | 0.769 / 0.752 |
| C | both | 80 / 47 | 172 / 273 | 0.473 / 0.163 | 0.861 | 0.622 | 0.672 | 0.3131 / 0.1674 | 0.869 / 0.817 |
| D | cue_gold | 25 / 15 | 144 / 214 | 0.350 / 0.096 | 0.868 (0.868) | 0.611 | 0.734 | 0.1823 / 0.0763 | 0.851 / 0.874 |
| D | cue_dist | 55 / 32 | 28 / 59 | 0.363 / 0.129 | 0.819 (0.819) | 0.655 | 0.854 | 0.3785 / 0.1247 | 0.769 / 0.779 |
| D | both | 80 / 47 | 172 / 273 | 0.359 / 0.102 | 0.861 | 0.622 | 0.766 | 0.2469 / 0.0900 | 0.858 / 0.844 |

## s43_refit_round3: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 135 / 210 | 0.546 / 0.174 | 0.870 (0.870) | 0.731 | 0.786 | 0.3387 / 0.1152 | 0.840 / 0.773 |
| A | cue_dist | 75 / 43 | 19 / 63 | 0.523 / 0.170 | 0.832 (0.832) | 0.702 | 0.891 | 0.6104 / 0.1727 | 0.795 / 0.729 |
| A | both | 110 / 49 | 154 / 273 | 0.529 / 0.173 | 0.841 | 0.666 | 0.812 | 0.4355 / 0.1341 | 0.854 / 0.755 |
| B | cue_gold | 42 / 13 | 139 / 225 | 0.369 / 0.116 | 0.862 (0.862) | 0.691 | 0.796 | 0.2422 / 0.0785 | 0.805 / 0.842 |
| B | cue_dist | 74 / 55 | 19 / 70 | 0.286 / 0.124 | 0.795 (0.795) | 0.680 | 0.852 | 0.3516 / 0.1213 | 0.701 / 0.756 |
| B | both | 116 / 68 | 158 / 295 | 0.311 / 0.118 | 0.823 | 0.641 | 0.801 | 0.2793 / 0.0932 | 0.784 / 0.810 |
| C | cue_gold | 42 / 13 | 139 / 225 | 0.355 / 0.073 | 0.911 (0.911) | 0.691 | 0.763 | 0.1890 / 0.0504 | 0.865 / 0.918 |
| C | cue_dist | 74 / 55 | 19 / 70 | 0.273 / 0.085 | 0.824 (0.824) | 0.680 | 0.851 | 0.3329 / 0.0944 | 0.762 / 0.778 |
| C | both | 116 / 68 | 158 / 295 | 0.297 / 0.076 | 0.871 | 0.641 | 0.773 | 0.2378 / 0.0655 | 0.846 / 0.860 |
| D | cue_gold | 42 / 13 | 139 / 225 | 0.475 / 0.132 | 0.903 (0.903) | 0.691 | 0.768 | 0.2845 / 0.0955 | 0.856 / 0.902 |
| D | cue_dist | 74 / 55 | 19 / 70 | 0.377 / 0.157 | 0.797 (0.797) | 0.680 | 0.852 | 0.4545 / 0.1635 | 0.736 / 0.746 |
| D | both | 116 / 68 | 158 / 295 | 0.407 / 0.137 | 0.851 | 0.641 | 0.780 | 0.3422 / 0.1189 | 0.829 / 0.827 |

## s43_refit_round4: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 138 / 209 | 0.716 / 0.308 | 0.902 (0.902) | 0.726 | 0.734 | 0.4694 / 0.2557 | 0.871 / 0.842 |
| A | cue_dist | 74 / 46 | 20 / 60 | 0.646 / 0.326 | 0.822 (0.822) | 0.683 | 0.842 | 0.7010 / 0.3553 | 0.736 / 0.761 |
| A | both | 108 / 52 | 158 / 269 | 0.663 / 0.312 | 0.849 | 0.652 | 0.765 | 0.5512 / 0.2886 | 0.850 / 0.780 |
| B | cue_gold | 23 / 20 | 145 / 215 | 0.417 / 0.166 | 0.847 (0.847) | 0.566 | 0.674 | 0.2694 / 0.1380 | 0.847 / 0.845 |
| B | cue_dist | 45 / 32 | 29 / 62 | 0.443 / 0.215 | 0.765 (0.765) | 0.633 | 0.858 | 0.4892 / 0.1858 | 0.730 / 0.703 |
| B | both | 68 / 52 | 174 / 277 | 0.434 / 0.176 | 0.820 | 0.590 | 0.724 | 0.3366 / 0.1516 | 0.832 / 0.780 |
| C | cue_gold | 23 / 20 | 145 / 215 | 0.408 / 0.120 | 0.878 (0.878) | 0.566 | 0.575 | 0.1996 / 0.1155 | 0.876 / 0.871 |
| C | cue_dist | 45 / 32 | 29 / 62 | 0.420 / 0.155 | 0.822 (0.822) | 0.633 | 0.785 | 0.4159 / 0.1670 | 0.762 / 0.812 |
| C | both | 68 / 52 | 174 / 277 | 0.416 / 0.127 | 0.861 | 0.590 | 0.635 | 0.2657 / 0.1302 | 0.858 / 0.841 |
| D | cue_gold | 23 / 20 | 145 / 215 | 0.503 / 0.181 | 0.855 (0.855) | 0.566 | 0.688 | 0.3097 / 0.1480 | 0.873 / 0.844 |
| D | cue_dist | 45 / 32 | 29 / 62 | 0.501 / 0.219 | 0.800 (0.800) | 0.633 | 0.851 | 0.5308 / 0.2042 | 0.782 / 0.761 |
| D | both | 68 / 52 | 174 / 277 | 0.502 / 0.189 | 0.836 | 0.590 | 0.732 | 0.3773 / 0.1641 | 0.853 / 0.806 |

## s43_refit_round5: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 136 / 211 | 0.803 / 0.427 | 0.885 (0.885) | 0.729 | 0.823 | 0.6372 / 0.3322 | 0.844 / 0.780 |
| A | cue_dist | 76 / 44 | 21 / 63 | 0.750 / 0.447 | 0.828 (0.828) | 0.692 | 0.905 | 0.8206 / 0.4484 | 0.791 / 0.749 |
| A | both | 110 / 50 | 157 / 274 | 0.764 / 0.431 | 0.845 | 0.662 | 0.843 | 0.7038 / 0.3706 | 0.849 / 0.776 |
| B | cue_gold | 14 / 6 | 136 / 220 | 0.642 / 0.341 | 0.824 (0.824) | 0.659 | 0.793 | 0.5091 / 0.2556 | 0.820 / 0.764 |
| B | cue_dist | 22 / 17 | 22 / 64 | 0.587 / 0.347 | 0.752 (0.752) | 0.654 | 0.933 | 0.6910 / 0.2756 | 0.698 / 0.680 |
| B | both | 36 / 23 | 158 / 284 | 0.606 / 0.342 | 0.778 | 0.626 | 0.821 | 0.5503 / 0.2609 | 0.806 / 0.709 |
| C | cue_gold | 14 / 6 | 136 / 220 | 0.632 / 0.265 | 0.865 (0.865) | 0.659 | 0.683 | 0.3825 / 0.2199 | 0.873 / 0.795 |
| C | cue_dist | 22 / 17 | 22 / 64 | 0.584 / 0.284 | 0.819 (0.819) | 0.654 | 0.882 | 0.6203 / 0.2451 | 0.789 / 0.778 |
| C | both | 36 / 23 | 158 / 284 | 0.600 / 0.269 | 0.837 | 0.626 | 0.720 | 0.4364 / 0.2266 | 0.865 / 0.769 |
| D | cue_gold | 14 / 6 | 136 / 220 | 0.633 / 0.302 | 0.838 (0.838) | 0.659 | 0.733 | 0.4400 / 0.2396 | 0.844 / 0.760 |
| D | cue_dist | 22 / 17 | 22 / 64 | 0.594 / 0.319 | 0.792 (0.792) | 0.654 | 0.898 | 0.6582 / 0.2675 | 0.723 / 0.756 |
| D | both | 36 / 23 | 158 / 284 | 0.607 / 0.305 | 0.812 | 0.626 | 0.763 | 0.4895 / 0.2470 | 0.838 / 0.747 |

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
  "s43_refit_round2": {
    "train": {
      "biased": {
        "answers": 1512,
        "naming_a": 1027
      },
      "unbiased": {
        "answers": 1512,
        "naming_a": 504
      }
    },
    "eval": {
      "biased": {
        "answers": 274,
        "naming_a": 186
      },
      "unbiased": {
        "answers": 451,
        "naming_a": 164
      }
    }
  },
  "s43_refit_round3": {
    "train": {
      "biased": {
        "answers": 2042,
        "naming_a": 1379
      },
      "unbiased": {
        "answers": 2042,
        "naming_a": 688
      }
    },
    "eval": {
      "biased": {
        "answers": 127,
        "naming_a": 80
      },
      "unbiased": {
        "answers": 445,
        "naming_a": 172
      }
    }
  },
  "s43_refit_round4": {
    "train": {
      "biased": {
        "answers": 2720,
        "naming_a": 1816
      },
      "unbiased": {
        "answers": 2720,
        "naming_a": 889
      }
    },
    "eval": {
      "biased": {
        "answers": 184,
        "naming_a": 116
      },
      "unbiased": {
        "answers": 453,
        "naming_a": 158
      }
    }
  },
  "s43_refit_round5": {
    "train": {
      "biased": {
        "answers": 3032,
        "naming_a": 1992
      },
      "unbiased": {
        "answers": 3032,
        "naming_a": 990
      }
    },
    "eval": {
      "biased": {
        "answers": 120,
        "naming_a": 68
      },
      "unbiased": {
        "answers": 451,
        "naming_a": 174
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
  "s43_refit_round2": {
    "13": 0.829673171043396,
    "14": 0.8189722895622253,
    "15": 0.8139767646789551,
    "16": 0.7967372536659241
  },
  "s43_refit_round3": {
    "13": 0.8625787496566772,
    "14": 0.8309842348098755,
    "15": 0.8168503642082214,
    "16": 0.7902442216873169
  },
  "s43_refit_round4": {
    "13": 0.8744384050369263,
    "14": 0.837189793586731,
    "15": 0.8058244585990906,
    "16": 0.8047900199890137
  },
  "s43_refit_round5": {
    "13": 0.9364322423934937,
    "14": 0.9253053665161133,
    "15": 0.9142431616783142,
    "16": 0.911565363407135
  }
}
```
