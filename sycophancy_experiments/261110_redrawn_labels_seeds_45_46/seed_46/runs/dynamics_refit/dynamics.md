# Training dynamics and probe rows: s46_round1, s46_refit_round2, s46_refit_round3, s46_refit_round4, s46_refit_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: s46_round1 0.3, s46_refit_round2 0.3, s46_refit_round3 0.3, s46_refit_round4 0.3, s46_refit_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | s46_round1 | 32.800 | 42.400 | 49.067 | 48.588 |
| groups where all 8 answers name the cue (of 64) | s46_refit_round2 | 48.200 | 42.200 | 44.967 | 44.324 |
| groups where all 8 answers name the cue (of 64) | s46_refit_round3 | 40.300 | 40.050 | 40.533 | 45.235 |
| groups where all 8 answers name the cue (of 64) | s46_refit_round4 | 41.900 | 42.300 | 42.400 | 44.353 |
| groups where all 8 answers name the cue (of 64) | s46_refit_round5 | 41.100 | 37.450 | 38.233 | 39.794 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | s46_round1 | 25.500 | 17.550 | 13.300 | 12.662 |
| groups with mixed task rewards (of 64) | s46_refit_round2 | 12.100 | 15.400 | 11.400 | 9.838 |
| groups with mixed task rewards (of 64) | s46_refit_round3 | 12.400 | 12.150 | 12.333 | 7.176 |
| groups with mixed task rewards (of 64) | s46_refit_round4 | 8.700 | 7.950 | 5.400 | 3.500 |
| groups with mixed task rewards (of 64) | s46_refit_round5 | 4.900 | 6.200 | 7.100 | 5.971 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | s46_round1 | 7.100 | 6.900 | 6.067 | 5.985 |
|   of which the cue is not at the training label | s46_refit_round2 | 3.900 | 6.200 | 5.167 | 4.015 |
|   of which the cue is not at the training label | s46_refit_round3 | 4.400 | 4.950 | 4.167 | 3.059 |
|   of which the cue is not at the training label | s46_refit_round4 | 2.500 | 2.950 | 1.867 | 1.412 |
|   of which the cue is not at the training label | s46_refit_round5 | 1.900 | 2.250 | 2.400 | 2.471 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | s46_round1 | 126.200 | 88.700 | 50.733 | 57.603 |
| answers not naming the cue (of 512) | s46_refit_round2 | 66.900 | 101.300 | 101.567 | 110.971 |
| answers not naming the cue (of 512) | s46_refit_round3 | 128.700 | 135.800 | 128.900 | 117.147 |
| answers not naming the cue (of 512) | s46_refit_round4 | 139.400 | 140 | 146.633 | 141.721 |
| answers not naming the cue (of 512) | s46_refit_round5 | 162.900 | 187.700 | 177.367 | 172.471 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | s46_round1 | 0.354 | 0.298 | 0.257 | 0.261 |
| mean absolute advantage per answer | s46_refit_round2 | 0.276 | 0.281 | 0.228 | 0.204 |
| mean absolute advantage per answer | s46_refit_round3 | 0.220 | 0.186 | 0.193 | 0.133 |
| mean absolute advantage per answer | s46_refit_round4 | 0.136 | 0.147 | 0.109 | 0.090 |
| mean absolute advantage per answer | s46_refit_round5 | 0.136 | 0.140 | 0.140 | 0.138 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | s46_round1 | 0.124 | 0.093 | 0.101 | 0.112 |
|   from probe-only groups | s46_refit_round2 | 0.172 | 0.112 | 0.101 | 0.093 |
|   from probe-only groups | s46_refit_round3 | 0.111 | 0.068 | 0.074 | 0.064 |
|   from probe-only groups | s46_refit_round4 | 0.066 | 0.077 | 0.061 | 0.060 |
|   from probe-only groups | s46_refit_round5 | 0.100 | 0.093 | 0.092 | 0.095 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | s46_round1 | 0.020 | 0.015 | 0.000 | -0.000 |
| advantage on cue-naming answers, per answer | s46_refit_round2 | 0.002 | 0.006 | -0.000 | 0.006 |
| advantage on cue-naming answers, per answer | s46_refit_round3 | -0.002 | -0.002 | 0.008 | -0.003 |
| advantage on cue-naming answers, per answer | s46_refit_round4 | 0.003 | -0.000 | 0.001 | -0.002 |
| advantage on cue-naming answers, per answer | s46_refit_round5 | -0.006 | -0.003 | -0.003 | -0.006 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | s46_round1 | 0.104 | 0.090 | 0.064 | 0.063 |
| gradient norm (W&B, before clipping) | s46_refit_round2 | 0.064 | 0.076 | 0.064 | 0.060 |
| gradient norm (W&B, before clipping) | s46_refit_round3 | 0.054 | 0.054 | 0.050 | 0.040 |
| gradient norm (W&B, before clipping) | s46_refit_round4 | 0.035 | 0.042 | 0.036 | 0.033 |
| gradient norm (W&B, before clipping) | s46_refit_round5 | 0.033 | 0.037 | 0.035 | 0.037 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | s46_round1 | 0.026 | 0.132 | 0.105 | 0.142 |
| KL from the base model (W&B) | s46_refit_round2 | 0.175 | 0.195 | 0.184 | 0.182 |
| KL from the base model (W&B) | s46_refit_round3 | 0.119 | 0.084 | 0.081 | 0.078 |
| KL from the base model (W&B) | s46_refit_round4 | 0.064 | 0.083 | 0.084 | 0.065 |
| KL from the base model (W&B) | s46_refit_round5 | 0.064 | 0.080 | 0.095 | 0.123 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | s46_round1 | 0.469 | 0.401 | 0.433 | 0.468 |
| token entropy (W&B) | s46_refit_round2 | 0.474 | 0.469 | 0.435 | 0.434 |
| token entropy (W&B) | s46_refit_round3 | 0.443 | 0.461 | 0.478 | 0.490 |
| token entropy (W&B) | s46_refit_round4 | 0.496 | 0.498 | 0.492 | 0.481 |
| token entropy (W&B) | s46_refit_round5 | 0.474 | 0.455 | 0.507 | 0.508 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | s46_round1 | 216.935 | 198.875 | 228.400 | 217.001 |
| answer length, tokens (W&B) | s46_refit_round2 | 185.598 | 165.659 | 180.701 | 173.757 |
| answer length, tokens (W&B) | s46_refit_round3 | 195.272 | 218.802 | 214.126 | 212.720 |
| answer length, tokens (W&B) | s46_refit_round4 | 219.149 | 222.093 | 199.043 | 206.945 |
| answer length, tokens (W&B) | s46_refit_round5 | 210.738 | 200.738 | 199.857 | 199.412 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of s46_round1: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 153 | 0.077 | 953.216 / 1055.663 | 0.139 / 0.224 | 0.176 / 0.279 | 0.389 / 0.508 |
| 5-9 | 175 | 0.080 | 762.204 / 780.139 | 0.067 / 0.083 | 0.131 / 0.171 | 0.357 / 0.434 |
| 10-29 | 848 | 0.023 | 814.340 / 868.402 | 0.047 / 0.069 | 0.127 / 0.154 | 0.282 / 0.337 |
| 30-59 | 1472 | 0.024 | 951.545 / 1014.502 | 0.101 / 0.142 | 0.237 / 0.268 | 0.430 / 0.475 |
| 60-127 | 3304 | 0.033 | 899.366 / 946.786 | 0.107 / 0.137 | 0.254 / 0.283 | 0.393 / 0.445 |

## Probe-only groups of s46_refit_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 234 | 0.080 | 898.848 / 1018.173 | 0.089 / 0.146 | 0.220 / 0.278 | 0.365 / 0.449 |
| 5-9 | 248 | 0.079 | 601.678 / 650.379 | 0.045 / 0.052 | 0.080 / 0.131 | 0.200 / 0.257 |
| 10-29 | 844 | 0.046 | 658.243 / 740.043 | 0.027 / 0.054 | 0.077 / 0.118 | 0.199 / 0.258 |
| 30-59 | 1349 | 0.037 | 746.251 / 843.660 | 0.018 / 0.031 | 0.063 / 0.106 | 0.237 / 0.309 |
| 60-127 | 3014 | 0.032 | 716.207 / 804.168 | 0.013 / 0.025 | 0.065 / 0.107 | 0.214 / 0.283 |

## Probe-only groups of s46_refit_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 205 | 0.074 | 779.132 / 803.679 | 0.015 / 0.027 | 0.096 / 0.112 | 0.255 / 0.291 |
| 5-9 | 198 | 0.043 | 888.655 / 937.414 | 0.011 / 0.037 | 0.091 / 0.173 | 0.269 / 0.343 |
| 10-29 | 801 | 0.026 | 963.185 / 993.346 | 0.027 / 0.045 | 0.091 / 0.165 | 0.309 / 0.381 |
| 30-59 | 1216 | 0.031 | 919.147 / 935.015 | 0.029 / 0.056 | 0.092 / 0.143 | 0.286 / 0.343 |
| 60-127 | 3076 | 0.020 | 923.482 / 933.364 | 0.026 / 0.045 | 0.086 / 0.128 | 0.288 / 0.342 |

## Probe-only groups of s46_refit_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 211 | 0.030 | 934.989 / 963.110 | 0.043 / 0.040 | 0.077 / 0.118 | 0.278 / 0.344 |
| 5-9 | 208 | 0.025 | 896.750 / 958.709 | 0.032 / 0.054 | 0.095 / 0.154 | 0.273 / 0.361 |
| 10-29 | 846 | 0.032 | 944.519 / 1003.744 | 0.047 / 0.082 | 0.146 / 0.185 | 0.357 / 0.421 |
| 30-59 | 1272 | 0.019 | 839.543 / 887.581 | 0.031 / 0.056 | 0.119 / 0.156 | 0.313 / 0.359 |
| 60-127 | 3016 | 0.017 | 867.293 / 915.300 | 0.040 / 0.072 | 0.116 / 0.175 | 0.329 / 0.387 |

## Probe-only groups of s46_refit_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 207 | 0.057 | 883.390 / 945.324 | 0.042 / 0.066 | 0.134 / 0.180 | 0.320 / 0.417 |
| 5-9 | 204 | 0.058 | 847.309 / 896.734 | 0.048 / 0.080 | 0.130 / 0.196 | 0.292 / 0.350 |
| 10-29 | 749 | 0.058 | 820.669 / 875.448 | 0.045 / 0.073 | 0.103 / 0.161 | 0.303 / 0.374 |
| 30-59 | 1147 | 0.056 | 828.876 / 889.378 | 0.030 / 0.044 | 0.069 / 0.095 | 0.240 / 0.302 |
| 60-127 | 2706 | 0.056 | 823.777 / 878.509 | 0.032 / 0.051 | 0.090 / 0.124 | 0.281 / 0.355 |

## s46_round1's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.123 | 0.002 | 1039.000 |
| 10 | 0 | 0.061 | 825.500 |
| 20 | 0 | 0.029 | 806.500 |
| 25 | 0 | 0.002 | 808.500 |
| 30 | 0 | 0.014 | 1041.000 |
| 60 | 0 | 0.010 | 962.000 |
| 70 | 0 | 0.023 | 772.000 |
| 80 | 0 | 0.012 | 814.000 |
| 90 | 0 | 0.006 | 822.000 |
| 100 | 0.002 | 0.014 | 871.500 |
| 127 | 0 | 0.020 | 887.000 |

## s46_refit_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0.006 | 966.500 |
| 10 | 0 | 0.002 | 526.000 |
| 20 | 0.039 | 0.004 | 716.000 |
| 25 | 0.082 | 0.006 | 769.000 |
| 30 | 0.025 | 0.004 | 822.500 |
| 60 | 0.033 | 0 | 770.500 |
| 70 | 0.047 | 0.004 | 780.000 |
| 80 | 0.027 | 0.006 | 657.000 |
| 90 | 0.031 | 0 | 767.000 |
| 100 | 0.020 | 0.016 | 712.500 |
| 127 | 0.010 | 0.004 | 781.500 |

## s46_refit_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.016 | 0 | 761.000 |
| 10 | 0.104 | 0.004 | 1012.500 |
| 20 | 0.053 | 0 | 794.000 |
| 25 | 0.109 | 0.008 | 852.500 |
| 30 | 0.086 | 0.004 | 798.500 |
| 60 | 0.244 | 0.023 | 839.000 |
| 70 | 0.230 | 0 | 860.500 |
| 80 | 0.123 | 0 | 890.000 |
| 90 | 0.107 | 0 | 928.500 |
| 100 | 0.096 | 0.020 | 895.500 |
| 127 | 0.121 | 0.002 | 979.000 |

## s46_refit_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.084 | 0.016 | 992.500 |
| 10 | 0.096 | 0.002 | 833.000 |
| 20 | 0.041 | 0.018 | 1059.000 |
| 25 | 0.031 | 0.002 | 1028.500 |
| 30 | 0.027 | 0.002 | 898.500 |
| 60 | 0.137 | 0 | 874.500 |
| 70 | 0.127 | 0.002 | 824.000 |
| 80 | 0.078 | 0.002 | 779.000 |
| 90 | 0.064 | 0.014 | 772.500 |
| 100 | 0.041 | 0.018 | 891.500 |
| 127 | 0.086 | 0.002 | 847.500 |

## s46_refit_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.066 | 0.002 | 837.000 |
| 10 | 0.096 | 0.008 | 798.000 |
| 20 | 0.061 | 0.008 | 891.500 |
| 25 | 0.061 | 0.008 | 826.500 |
| 30 | 0.057 | 0.016 | 791.500 |
| 60 | 0.035 | 0.004 | 863.000 |
| 70 | 0.025 | 0.002 | 732.500 |
| 80 | 0.057 | 0.027 | 812.000 |
| 90 | 0.062 | 0.006 | 838.500 |
| 100 | 0.082 | 0.002 | 838.500 |
| 127 | 0.070 | 0.004 | 843.500 |

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
  "s46_round1": {
    "median_tokens_no_cue": 200.0,
    "median_tokens_cued": 205.0,
    "mean_top_letter_share_of_8_no_cue": 0.9036298568507157,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 191,
      "B": 294,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 185,
      "B": 288
    },
    "questions_with_no_cue_answer": 473
  },
  "s46_refit_round2": {
    "median_tokens_no_cue": 160.0,
    "median_tokens_cued": 166.0,
    "mean_top_letter_share_of_8_no_cue": 0.9194785276073619,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 206,
      "B": 281,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 204,
      "B": 265
    },
    "questions_with_no_cue_answer": 469
  },
  "s46_refit_round3": {
    "median_tokens_no_cue": 193.0,
    "median_tokens_cued": 200.0,
    "mean_top_letter_share_of_8_no_cue": 0.9389059304703476,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 181,
      "B": 304,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 175,
      "B": 304
    },
    "questions_with_no_cue_answer": 479
  },
  "s46_refit_round4": {
    "median_tokens_no_cue": 180.0,
    "median_tokens_cued": 188.0,
    "mean_top_letter_share_of_8_no_cue": 0.9498977505112475,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 194,
      "B": 292,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 191,
      "B": 284
    },
    "questions_with_no_cue_answer": 475
  },
  "s46_refit_round5": {
    "median_tokens_no_cue": 171.0,
    "median_tokens_cued": 183.5,
    "mean_top_letter_share_of_8_no_cue": 0.9475971370143149,
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
      "A": 179,
      "B": 295
    },
    "questions_with_no_cue_answer": 474
  }
}
```

## s46_round1: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 137 / 208 | 0.757 / 0.222 | 0.927 (0.927) | 0.714 | 0.677 | 0.3926 / 0.1861 | 0.918 / 0.882 |
| A | cue_dist | 73 / 44 | 22 / 61 | 0.747 / 0.271 | 0.877 (0.877) | 0.679 | 0.818 | 0.7522 / 0.3659 | 0.882 / 0.816 |
| A | both | 106 / 51 | 159 / 269 | 0.749 / 0.231 | 0.906 | 0.652 | 0.716 | 0.5215 / 0.2451 | 0.926 / 0.850 |
| B | cue_gold | 53 / 18 | 147 / 219 | 0.236 / 0.056 | 0.814 (0.814) | 0.672 | 0.845 | 0.1459 / 0.0344 | 0.836 / 0.667 |
| B | cue_dist | 136 / 91 | 23 / 62 | 0.217 / 0.057 | 0.778 (0.778) | 0.664 | 0.897 | 0.2739 / 0.0694 | 0.782 / 0.678 |
| B | both | 189 / 109 | 170 / 281 | 0.222 / 0.056 | 0.784 | 0.629 | 0.865 | 0.2026 / 0.0482 | 0.842 / 0.692 |
| C | cue_gold | 53 / 18 | 147 / 219 | 0.230 / 0.066 | 0.788 (0.788) | 0.672 | 0.859 | 0.1529 / 0.0424 | 0.818 / 0.605 |
| C | cue_dist | 136 / 91 | 23 / 62 | 0.192 / 0.052 | 0.775 (0.775) | 0.664 | 0.916 | 0.2456 / 0.0585 | 0.783 / 0.675 |
| C | both | 189 / 109 | 170 / 281 | 0.201 / 0.064 | 0.761 | 0.629 | 0.881 | 0.1939 / 0.0487 | 0.824 / 0.661 |
| D | cue_gold | 53 / 18 | 147 / 219 | 0.225 / 0.068 | 0.775 (0.775) | 0.672 | 0.867 | 0.1582 / 0.0386 | 0.792 / 0.593 |
| D | cue_dist | 136 / 91 | 23 / 62 | 0.192 / 0.053 | 0.766 (0.766) | 0.664 | 0.905 | 0.2423 / 0.0623 | 0.792 / 0.646 |
| D | both | 189 / 109 | 170 / 281 | 0.200 / 0.065 | 0.752 | 0.629 | 0.882 | 0.1955 / 0.0479 | 0.799 / 0.654 |

## s46_refit_round2: probe rows by answer letter (448 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 6 | 139 / 202 | 0.805 / 0.311 | 0.935 (0.935) | 0.719 | 0.667 | 0.4615 / 0.2795 | 0.911 / 0.924 |
| A | cue_dist | 71 / 46 | 19 / 59 | 0.779 / 0.388 | 0.869 (0.869) | 0.682 | 0.814 | 0.7918 / 0.4775 | 0.835 / 0.810 |
| A | both | 104 / 52 | 158 / 261 | 0.786 / 0.325 | 0.908 | 0.645 | 0.700 | 0.5749 / 0.3459 | 0.918 / 0.863 |
| B | cue_gold | 31 / 19 | 154 / 205 | 0.368 / 0.134 | 0.835 (0.835) | 0.596 | 0.663 | 0.2129 / 0.1206 | 0.856 / 0.791 |
| B | cue_dist | 78 / 39 | 31 / 56 | 0.360 / 0.186 | 0.726 (0.726) | 0.655 | 0.774 | 0.3755 / 0.1834 | 0.713 / 0.631 |
| B | both | 109 / 58 | 185 / 261 | 0.363 / 0.144 | 0.804 | 0.619 | 0.702 | 0.2732 / 0.1393 | 0.828 / 0.731 |
| C | cue_gold | 31 / 19 | 154 / 205 | 0.312 / 0.099 | 0.843 (0.843) | 0.596 | 0.651 | 0.1613 / 0.0954 | 0.863 / 0.794 |
| C | cue_dist | 78 / 39 | 31 / 56 | 0.311 / 0.131 | 0.764 (0.764) | 0.655 | 0.757 | 0.3087 / 0.1490 | 0.761 / 0.690 |
| C | both | 109 / 58 | 185 / 261 | 0.311 / 0.105 | 0.821 | 0.619 | 0.690 | 0.2160 / 0.1114 | 0.843 / 0.756 |
| D | cue_gold | 31 / 19 | 154 / 205 | 0.441 / 0.152 | 0.844 (0.844) | 0.596 | 0.792 | 0.2890 / 0.1041 | 0.873 / 0.809 |
| D | cue_dist | 78 / 39 | 31 / 56 | 0.421 / 0.196 | 0.756 (0.756) | 0.655 | 0.863 | 0.4661 / 0.1626 | 0.756 / 0.633 |
| D | both | 109 / 58 | 185 / 261 | 0.427 / 0.161 | 0.811 | 0.619 | 0.812 | 0.3546 / 0.1215 | 0.834 / 0.746 |

## s46_refit_round3: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 135 / 210 | 0.786 / 0.347 | 0.904 (0.904) | 0.721 | 0.772 | 0.5514 / 0.2711 | 0.895 / 0.814 |
| A | cue_dist | 75 / 45 | 22 / 63 | 0.756 / 0.375 | 0.850 (0.850) | 0.683 | 0.863 | 0.8002 / 0.4161 | 0.816 / 0.786 |
| A | both | 110 / 52 | 157 / 273 | 0.764 / 0.352 | 0.875 | 0.657 | 0.793 | 0.6418 / 0.3193 | 0.892 / 0.811 |
| B | cue_gold | 29 / 17 | 136 / 221 | 0.577 / 0.206 | 0.879 (0.879) | 0.625 | 0.715 | 0.3532 / 0.1753 | 0.874 / 0.864 |
| B | cue_dist | 53 / 44 | 26 / 74 | 0.543 / 0.282 | 0.774 (0.774) | 0.643 | 0.853 | 0.6220 / 0.2688 | 0.692 / 0.755 |
| B | both | 82 / 61 | 162 / 295 | 0.554 / 0.222 | 0.842 | 0.609 | 0.745 | 0.4402 / 0.2063 | 0.850 / 0.815 |
| C | cue_gold | 29 / 17 | 136 / 221 | 0.402 / 0.089 | 0.901 (0.901) | 0.625 | 0.694 | 0.1995 / 0.0735 | 0.894 / 0.900 |
| C | cue_dist | 53 / 44 | 26 / 74 | 0.396 / 0.122 | 0.839 (0.839) | 0.643 | 0.836 | 0.4431 / 0.1319 | 0.825 / 0.811 |
| C | both | 82 / 61 | 162 / 295 | 0.398 / 0.097 | 0.877 | 0.609 | 0.727 | 0.2784 / 0.0929 | 0.889 / 0.863 |
| D | cue_gold | 29 / 17 | 136 / 221 | 0.550 / 0.161 | 0.895 (0.895) | 0.625 | 0.621 | 0.2689 / 0.1612 | 0.896 / 0.880 |
| D | cue_dist | 53 / 44 | 26 / 74 | 0.516 / 0.225 | 0.816 (0.816) | 0.643 | 0.783 | 0.5324 / 0.2583 | 0.766 / 0.804 |
| D | both | 82 / 61 | 162 / 295 | 0.527 / 0.175 | 0.872 | 0.609 | 0.662 | 0.3542 / 0.1934 | 0.877 / 0.853 |

## s46_refit_round4: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 32 / 6 | 137 / 208 | 0.822 / 0.387 | 0.904 (0.904) | 0.723 | 0.728 | 0.5629 / 0.3256 | 0.891 / 0.809 |
| A | cue_dist | 74 / 45 | 21 / 59 | 0.781 / 0.427 | 0.851 (0.851) | 0.680 | 0.853 | 0.8144 / 0.4780 | 0.820 / 0.796 |
| A | both | 106 / 51 | 158 / 267 | 0.791 / 0.395 | 0.875 | 0.652 | 0.761 | 0.6534 / 0.3755 | 0.885 / 0.820 |
| B | cue_gold | 23 / 14 | 141 / 208 | 0.670 / 0.318 | 0.827 (0.827) | 0.609 | 0.706 | 0.4657 / 0.2673 | 0.862 / 0.762 |
| B | cue_dist | 42 / 29 | 34 / 65 | 0.713 / 0.411 | 0.791 (0.791) | 0.624 | 0.851 | 0.7369 / 0.3750 | 0.729 / 0.809 |
| B | both | 65 / 43 | 175 / 273 | 0.698 / 0.338 | 0.836 | 0.606 | 0.742 | 0.5516 / 0.2993 | 0.847 / 0.805 |
| C | cue_gold | 23 / 14 | 141 / 208 | 0.495 / 0.094 | 0.885 (0.885) | 0.609 | 0.597 | 0.1931 / 0.0882 | 0.925 / 0.814 |
| C | cue_dist | 42 / 29 | 34 / 65 | 0.499 / 0.122 | 0.870 (0.870) | 0.624 | 0.754 | 0.4376 / 0.1512 | 0.838 / 0.888 |
| C | both | 65 / 43 | 175 / 273 | 0.497 / 0.100 | 0.887 | 0.606 | 0.642 | 0.2705 / 0.1070 | 0.900 / 0.859 |
| D | cue_gold | 23 / 14 | 141 / 208 | 0.516 / 0.160 | 0.861 (0.861) | 0.609 | 0.768 | 0.3053 / 0.1118 | 0.892 / 0.831 |
| D | cue_dist | 42 / 29 | 34 / 65 | 0.535 / 0.214 | 0.818 (0.818) | 0.624 | 0.858 | 0.5504 / 0.1847 | 0.793 / 0.825 |
| D | both | 65 / 43 | 175 / 273 | 0.529 / 0.172 | 0.855 | 0.606 | 0.787 | 0.3829 / 0.1335 | 0.871 / 0.841 |

## s46_refit_round5: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 31 / 7 | 137 / 211 | 0.855 / 0.543 | 0.890 (0.890) | 0.711 | 0.714 | 0.6695 / 0.5001 | 0.859 / 0.847 |
| A | cue_dist | 76 / 44 | 20 / 60 | 0.830 / 0.577 | 0.833 (0.833) | 0.692 | 0.817 | 0.8429 / 0.6234 | 0.777 / 0.773 |
| A | both | 107 / 51 | 157 / 271 | 0.836 / 0.549 | 0.860 | 0.655 | 0.745 | 0.7325 / 0.5399 | 0.850 / 0.813 |
| B | cue_gold | 13 / 10 | 139 / 212 | 0.775 / 0.427 | 0.890 (0.890) | 0.585 | 0.622 | 0.5119 / 0.4052 | 0.892 / 0.873 |
| B | cue_dist | 30 / 16 | 24 / 67 | 0.779 / 0.503 | 0.826 (0.826) | 0.694 | 0.816 | 0.7565 / 0.4908 | 0.792 / 0.766 |
| B | both | 43 / 26 | 163 / 279 | 0.778 / 0.443 | 0.874 | 0.627 | 0.666 | 0.5760 / 0.4285 | 0.882 / 0.829 |
| C | cue_gold | 13 / 10 | 139 / 212 | 0.720 / 0.299 | 0.911 (0.911) | 0.585 | 0.532 | 0.3500 / 0.3081 | 0.930 / 0.883 |
| C | cue_dist | 30 / 16 | 24 / 67 | 0.744 / 0.388 | 0.860 (0.860) | 0.694 | 0.740 | 0.6511 / 0.4137 | 0.833 / 0.840 |
| C | both | 43 / 26 | 163 / 279 | 0.736 / 0.317 | 0.904 | 0.627 | 0.583 | 0.4289 / 0.3368 | 0.916 / 0.874 |
| D | cue_gold | 13 / 10 | 139 / 212 | 0.603 / 0.245 | 0.871 (0.871) | 0.585 | 0.595 | 0.3132 / 0.2350 | 0.915 / 0.810 |
| D | cue_dist | 30 / 16 | 24 / 67 | 0.649 / 0.311 | 0.837 (0.837) | 0.694 | 0.750 | 0.5764 / 0.3261 | 0.794 / 0.812 |
| D | both | 43 / 26 | 163 / 279 | 0.634 / 0.258 | 0.876 | 0.627 | 0.632 | 0.3822 / 0.2598 | 0.895 / 0.828 |

## Probe data: answers naming A

```
{
  "s46_round1": {
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
  "s46_refit_round2": {
    "train": {
      "biased": {
        "answers": 1512,
        "naming_a": 967
      },
      "unbiased": {
        "answers": 1512,
        "naming_a": 534
      }
    },
    "eval": {
      "biased": {
        "answers": 298,
        "naming_a": 189
      },
      "unbiased": {
        "answers": 451,
        "naming_a": 170
      }
    }
  },
  "s46_refit_round3": {
    "train": {
      "biased": {
        "answers": 2114,
        "naming_a": 1357
      },
      "unbiased": {
        "answers": 2114,
        "naming_a": 772
      }
    },
    "eval": {
      "biased": {
        "answers": 167,
        "naming_a": 109
      },
      "unbiased": {
        "answers": 446,
        "naming_a": 185
      }
    }
  },
  "s46_refit_round4": {
    "train": {
      "biased": {
        "answers": 2752,
        "naming_a": 1747
      },
      "unbiased": {
        "answers": 2752,
        "naming_a": 987
      }
    },
    "eval": {
      "biased": {
        "answers": 143,
        "naming_a": 82
      },
      "unbiased": {
        "answers": 457,
        "naming_a": 162
      }
    }
  },
  "s46_refit_round5": {
    "train": {
      "biased": {
        "answers": 3112,
        "naming_a": 1899
      },
      "unbiased": {
        "answers": 3112,
        "naming_a": 1131
      }
    },
    "eval": {
      "biased": {
        "answers": 108,
        "naming_a": 65
      },
      "unbiased": {
        "answers": 448,
        "naming_a": 175
      }
    }
  }
}
```

## Cosine of the initial and fresh probe weights

```
{
  "s46_round1": {
    "13": 0.8784424066543579,
    "14": 0.8339731693267822,
    "15": 0.8064978122711182,
    "16": 0.7725257873535156
  },
  "s46_refit_round2": {
    "13": 0.8299677968025208,
    "14": 0.7992575764656067,
    "15": 0.7654508948326111,
    "16": 0.7506481409072876
  },
  "s46_refit_round3": {
    "13": 0.9103482961654663,
    "14": 0.8973706960678101,
    "15": 0.8819462656974792,
    "16": 0.8685392737388611
  },
  "s46_refit_round4": {
    "13": 0.9006231427192688,
    "14": 0.8837347626686096,
    "15": 0.8811827301979065,
    "16": 0.8773398399353027
  },
  "s46_refit_round5": {
    "13": 0.9109739661216736,
    "14": 0.8833455443382263,
    "15": 0.8710083961486816,
    "16": 0.8560366630554199
  }
}
```
