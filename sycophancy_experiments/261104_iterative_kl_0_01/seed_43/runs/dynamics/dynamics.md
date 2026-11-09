# Training dynamics and probe rows: kl_weight_1_0_s43, s43_round2, s43_round3, s43_round4, s43_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s43 0.3, s43_round2 0.3, s43_round3 0.3, s43_round4 0.3, s43_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s43 | 32 | 41.750 | 43.500 | 44.529 |
| groups where all 8 answers name the cue (of 64) | s43_round2 | 43.600 | 38.500 | 41.567 | 44.515 |
| groups where all 8 answers name the cue (of 64) | s43_round3 | 42 | 40.150 | 45.033 | 44.985 |
| groups where all 8 answers name the cue (of 64) | s43_round4 | 47.300 | 42.800 | 44.700 | 45.868 |
| groups where all 8 answers name the cue (of 64) | s43_round5 | 47.100 | 43.750 | 45.433 | 46.500 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s43 | 26.500 | 17.200 | 16.800 | 15.382 |
| groups with mixed task rewards (of 64) | s43_round2 | 15.400 | 15.950 | 13.500 | 10.029 |
| groups with mixed task rewards (of 64) | s43_round3 | 12.500 | 9.900 | 4.933 | 4.456 |
| groups with mixed task rewards (of 64) | s43_round4 | 3.100 | 5.750 | 3.833 | 2.456 |
| groups with mixed task rewards (of 64) | s43_round5 | 2.300 | 2.900 | 2.733 | 1.191 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s43 | 7.800 | 6.950 | 6.333 | 6.338 |
|   of which the cue is not at the training label | s43_round2 | 6 | 5.450 | 5.433 | 4.118 |
|   of which the cue is not at the training label | s43_round3 | 4.400 | 3.350 | 2.167 | 2.132 |
|   of which the cue is not at the training label | s43_round4 | 1.700 | 2.700 | 1.567 | 1.103 |
|   of which the cue is not at the training label | s43_round5 | 0.900 | 1.250 | 1.300 | 0.603 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s43 | 126 | 99.050 | 85.167 | 81.926 |
| answers not naming the cue (of 512) | s43_round2 | 90.800 | 131.600 | 121.967 | 111.574 |
| answers not naming the cue (of 512) | s43_round3 | 123.400 | 145.050 | 130 | 134.191 |
| answers not naming the cue (of 512) | s43_round4 | 121.400 | 144.800 | 137.367 | 134.309 |
| answers not naming the cue (of 512) | s43_round5 | 123.800 | 148.200 | 137.500 | 135.926 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s43 | 0.355 | 0.296 | 0.305 | 0.251 |
| mean absolute advantage per answer | s43_round2 | 0.297 | 0.262 | 0.238 | 0.198 |
| mean absolute advantage per answer | s43_round3 | 0.234 | 0.164 | 0.111 | 0.106 |
| mean absolute advantage per answer | s43_round4 | 0.137 | 0.155 | 0.133 | 0.108 |
| mean absolute advantage per answer | s43_round5 | 0.123 | 0.114 | 0.113 | 0.094 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s43 | 0.112 | 0.117 | 0.122 | 0.071 |
|   from probe-only groups | s43_round2 | 0.148 | 0.106 | 0.108 | 0.097 |
|   from probe-only groups | s43_round3 | 0.125 | 0.072 | 0.063 | 0.062 |
|   from probe-only groups | s43_round4 | 0.109 | 0.101 | 0.096 | 0.086 |
|   from probe-only groups | s43_round5 | 0.105 | 0.086 | 0.084 | 0.082 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s43 | 0.017 | -0.003 | 0.007 | 0.009 |
| advantage on cue-naming answers, per answer | s43_round2 | -0.003 | 0.007 | -0.003 | -0.001 |
| advantage on cue-naming answers, per answer | s43_round3 | 0.003 | 0.007 | -0.002 | -0.003 |
| advantage on cue-naming answers, per answer | s43_round4 | -0.005 | -0.003 | -0.001 | -0.001 |
| advantage on cue-naming answers, per answer | s43_round5 | -0.001 | -0.001 | -0.001 | -0.000 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s43 | 0.087 | 0.102 | 0.140 | 0.066 |
| gradient norm (W&B, before clipping) | s43_round2 | 0.066 | 0.064 | 0.057 | 0.051 |
| gradient norm (W&B, before clipping) | s43_round3 | 0.054 | 0.053 | 0.041 | 0.041 |
| gradient norm (W&B, before clipping) | s43_round4 | 0.036 | 0.046 | 0.041 | 0.032 |
| gradient norm (W&B, before clipping) | s43_round5 | 0.033 | 0.035 | 0.033 | 0.028 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s43 | 0.012 | 0.101 | 0.129 | 0.108 |
| KL from the base model (W&B) | s43_round2 | 0.116 | 0.157 | 0.174 | 0.169 |
| KL from the base model (W&B) | s43_round3 | 0.151 | 0.091 | 0.084 | 0.071 |
| KL from the base model (W&B) | s43_round4 | 0.075 | 0.133 | 0.142 | 0.129 |
| KL from the base model (W&B) | s43_round5 | 0.131 | 0.161 | 0.146 | 0.145 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s43 | 0.427 | 0.401 | 0.440 | 0.446 |
| token entropy (W&B) | s43_round2 | 0.467 | 0.472 | 0.462 | 0.458 |
| token entropy (W&B) | s43_round3 | 0.497 | 0.457 | 0.457 | 0.486 |
| token entropy (W&B) | s43_round4 | 0.465 | 0.450 | 0.452 | 0.472 |
| token entropy (W&B) | s43_round5 | 0.472 | 0.465 | 0.486 | 0.463 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s43 | 243.601 | 195.433 | 229.320 | 208.106 |
| answer length, tokens (W&B) | s43_round2 | 212.788 | 203.368 | 205.929 | 202.335 |
| answer length, tokens (W&B) | s43_round3 | 206.977 | 187.781 | 192.798 | 211.265 |
| answer length, tokens (W&B) | s43_round4 | 191.591 | 186.421 | 195.280 | 190.712 |
| answer length, tokens (W&B) | s43_round5 | 186.043 | 174.605 | 181.049 | 183.983 |
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

## Probe-only groups of s43_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 248 | 0.074 | 873.768 / 931.927 | 0.065 / 0.113 | 0.177 / 0.212 | 0.327 / 0.385 |
| 5-9 | 188 | 0.071 | 867.040 / 928.999 | 0.068 / 0.086 | 0.174 / 0.191 | 0.310 / 0.368 |
| 10-29 | 770 | 0.046 | 820.629 / 888.293 | 0.059 / 0.094 | 0.165 / 0.199 | 0.314 / 0.397 |
| 30-59 | 1247 | 0.047 | 851.156 / 911.202 | 0.053 / 0.089 | 0.169 / 0.210 | 0.346 / 0.401 |
| 60-127 | 3027 | 0.036 | 841.117 / 894.261 | 0.052 / 0.082 | 0.172 / 0.208 | 0.310 / 0.375 |

## Probe-only groups of s43_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 210 | 0.063 | 873.590 / 914.992 | 0.032 / 0.073 | 0.208 / 0.237 | 0.314 / 0.388 |
| 5-9 | 210 | 0.074 | 818.875 / 877.220 | 0.058 / 0.082 | 0.155 / 0.217 | 0.276 / 0.377 |
| 10-29 | 803 | 0.029 | 784.194 / 840.427 | 0.032 / 0.056 | 0.101 / 0.149 | 0.281 / 0.354 |
| 30-59 | 1351 | 0.021 | 795.688 / 859.693 | 0.047 / 0.075 | 0.109 / 0.167 | 0.311 / 0.400 |
| 60-127 | 3059 | 0.019 | 867.742 / 952.042 | 0.048 / 0.098 | 0.107 / 0.171 | 0.309 / 0.392 |

## Probe-only groups of s43_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 234 | 0.057 | 844.331 / 919.988 | 0.063 / 0.109 | 0.111 / 0.165 | 0.295 / 0.420 |
| 5-9 | 239 | 0.044 | 759.031 / 821.584 | 0.029 / 0.059 | 0.070 / 0.096 | 0.220 / 0.324 |
| 10-29 | 856 | 0.049 | 779.192 / 832.002 | 0.025 / 0.038 | 0.080 / 0.112 | 0.212 / 0.279 |
| 30-59 | 1341 | 0.043 | 803.491 / 869.377 | 0.018 / 0.036 | 0.069 / 0.107 | 0.231 / 0.297 |
| 60-127 | 3119 | 0.035 | 787.595 / 848.711 | 0.026 / 0.051 | 0.078 / 0.124 | 0.254 / 0.327 |

## Probe-only groups of s43_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 244 | 0.050 | 770.369 / 844.012 | 0.025 / 0.043 | 0.066 / 0.112 | 0.272 / 0.341 |
| 5-9 | 227 | 0.046 | 736.161 / 818.073 | 0.018 / 0.031 | 0.079 / 0.106 | 0.198 / 0.285 |
| 10-29 | 875 | 0.041 | 721.117 / 787.846 | 0.018 / 0.033 | 0.068 / 0.102 | 0.191 / 0.256 |
| 30-59 | 1363 | 0.037 | 749.672 / 831.882 | 0.022 / 0.051 | 0.074 / 0.121 | 0.218 / 0.304 |
| 60-127 | 3162 | 0.034 | 766.883 / 846.691 | 0.020 / 0.033 | 0.081 / 0.117 | 0.204 / 0.279 |

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

## s43_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.043 | 0.012 | 854.000 |
| 10 | 0.014 | 0.018 | 843.500 |
| 20 | 0.059 | 0.008 | 852.500 |
| 25 | 0.076 | 0.006 | 849.000 |
| 30 | 0.059 | 0.012 | 818.000 |
| 60 | 0.004 | 0 | 848.500 |
| 70 | 0.023 | 0.002 | 818.500 |
| 80 | 0.018 | 0 | 817.000 |
| 90 | 0.031 | 0.002 | 816.500 |
| 100 | 0.037 | 0.004 | 849.500 |
| 127 | 0.051 | 0 | 885.000 |

## s43_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.072 | 0.006 | 874.500 |
| 10 | 0.012 | 0.016 | 725.000 |
| 20 | 0.033 | 0 | 770.000 |
| 25 | 0.088 | 0.006 | 921.000 |
| 30 | 0.094 | 0 | 891.500 |
| 60 | 0.031 | 0 | 889.500 |
| 70 | 0.061 | 0.004 | 900.000 |
| 80 | 0.109 | 0.002 | 859.500 |
| 90 | 0.086 | 0 | 862.000 |
| 100 | 0.096 | 0.008 | 863.500 |
| 127 | 0.111 | 0 | 861.000 |

## s43_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.088 | 0.010 | 873.500 |
| 10 | 0.041 | 0.010 | 742.000 |
| 20 | 0.133 | 0 | 821.500 |
| 25 | 0.320 | 0 | 829.000 |
| 30 | 0.094 | 0.016 | 741.000 |
| 60 | 0.084 | 0 | 796.000 |
| 70 | 0.074 | 0 | 840.500 |
| 80 | 0.064 | 0 | 805.500 |
| 90 | 0.066 | 0 | 739.500 |
| 100 | 0.074 | 0 | 758.000 |
| 127 | 0.117 | 0 | 764.500 |

## s43_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.064 | 0 | 729.500 |
| 10 | 0.113 | 0.004 | 763.000 |
| 20 | 0.031 | 0 | 689.000 |
| 25 | 0.027 | 0 | 726.000 |
| 30 | 0.037 | 0.002 | 734.000 |
| 60 | 0.029 | 0.002 | 759.000 |
| 70 | 0.012 | 0.006 | 800.000 |
| 80 | 0.033 | 0.004 | 847.500 |
| 90 | 0.059 | 0.002 | 786.500 |
| 100 | 0.072 | 0.002 | 816.000 |
| 127 | 0.053 | 0.004 | 765.500 |

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
  "s43_round2": {
    "median_tokens_no_cue": 184.0,
    "median_tokens_cued": 194.5,
    "mean_top_letter_share_of_8_no_cue": 0.9307259713701431,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 181,
      "B": 307,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 169,
      "B": 306
    },
    "questions_with_no_cue_answer": 475
  },
  "s43_round3": {
    "median_tokens_no_cue": 185.0,
    "median_tokens_cued": 193.0,
    "mean_top_letter_share_of_8_no_cue": 0.9570552147239264,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 184,
      "B": 303,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 185,
      "B": 298
    },
    "questions_with_no_cue_answer": 483
  },
  "s43_round4": {
    "median_tokens_no_cue": 163.0,
    "median_tokens_cued": 169.0,
    "mean_top_letter_share_of_8_no_cue": 0.9634458077709611,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 190,
      "B": 297,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 184,
      "B": 299
    },
    "questions_with_no_cue_answer": 483
  },
  "s43_round5": {
    "median_tokens_no_cue": 162.0,
    "median_tokens_cued": 168.0,
    "mean_top_letter_share_of_8_no_cue": 0.9583333333333334,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 203,
      "B": 286,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 196,
      "B": 283
    },
    "questions_with_no_cue_answer": 479
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

## s43_round2: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 32 / 6 | 135 / 210 | 0.699 / 0.247 | 0.914 (0.914) | 0.725 | 0.611 | 0.3612 / 0.2375 | 0.898 / 0.896 |
| A | cue_dist | 75 / 44 | 21 / 60 | 0.701 / 0.323 | 0.847 (0.847) | 0.685 | 0.777 | 0.7031 / 0.4050 | 0.823 / 0.791 |
| A | both | 107 / 50 | 156 / 270 | 0.701 / 0.261 | 0.890 | 0.658 | 0.664 | 0.4860 / 0.2920 | 0.901 / 0.835 |
| B | cue_gold | 30 / 15 | 133 / 223 | 0.362 / 0.128 | 0.846 (0.846) | 0.647 | 0.715 | 0.2128 / 0.1145 | 0.831 / 0.839 |
| B | cue_dist | 63 / 45 | 21 / 71 | 0.320 / 0.170 | 0.716 (0.716) | 0.678 | 0.800 | 0.3687 / 0.1656 | 0.749 / 0.597 |
| B | both | 93 / 60 | 154 / 294 | 0.332 / 0.137 | 0.783 | 0.632 | 0.738 | 0.2658 / 0.1313 | 0.828 / 0.707 |
| C | cue_gold | 30 / 15 | 133 / 223 | 0.474 / 0.143 | 0.879 (0.879) | 0.647 | 0.739 | 0.2746 / 0.1158 | 0.848 / 0.886 |
| C | cue_dist | 63 / 45 | 21 / 71 | 0.389 / 0.174 | 0.782 (0.782) | 0.678 | 0.830 | 0.4433 / 0.1790 | 0.728 / 0.717 |
| C | both | 93 / 60 | 154 / 294 | 0.414 / 0.150 | 0.830 | 0.632 | 0.761 | 0.3320 / 0.1365 | 0.823 / 0.797 |
| D | cue_gold | 30 / 15 | 133 / 223 | 0.393 / 0.124 | 0.864 (0.864) | 0.647 | 0.765 | 0.2370 / 0.0981 | 0.834 / 0.866 |
| D | cue_dist | 63 / 45 | 21 / 71 | 0.295 / 0.153 | 0.730 (0.730) | 0.678 | 0.806 | 0.3422 / 0.1487 | 0.676 / 0.656 |
| D | both | 93 / 60 | 154 / 294 | 0.324 / 0.130 | 0.790 | 0.632 | 0.774 | 0.2728 / 0.1147 | 0.774 / 0.755 |

## s43_round3: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 138 / 209 | 0.652 / 0.296 | 0.854 (0.854) | 0.718 | 0.657 | 0.4181 / 0.2679 | 0.865 / 0.688 |
| A | cue_dist | 75 / 46 | 22 / 62 | 0.675 / 0.340 | 0.830 (0.830) | 0.679 | 0.809 | 0.7004 / 0.3917 | 0.788 / 0.786 |
| A | both | 110 / 53 | 160 / 271 | 0.669 / 0.305 | 0.855 | 0.652 | 0.703 | 0.5195 / 0.3092 | 0.873 / 0.784 |
| B | cue_gold | 19 / 14 | 137 / 211 | 0.416 / 0.182 | 0.827 (0.827) | 0.591 | 0.637 | 0.2631 / 0.1609 | 0.812 / 0.808 |
| B | cue_dist | 27 / 25 | 33 / 73 | 0.454 / 0.285 | 0.687 (0.687) | 0.604 | 0.823 | 0.5290 / 0.2254 | 0.644 / 0.639 |
| B | both | 46 / 39 | 170 / 284 | 0.439 / 0.206 | 0.785 | 0.583 | 0.675 | 0.3370 / 0.1805 | 0.798 / 0.738 |
| C | cue_gold | 19 / 14 | 137 / 211 | 0.352 / 0.090 | 0.907 (0.907) | 0.591 | 0.565 | 0.1449 / 0.0903 | 0.902 / 0.904 |
| C | cue_dist | 27 / 25 | 33 / 73 | 0.385 / 0.163 | 0.782 (0.782) | 0.604 | 0.745 | 0.3768 / 0.1498 | 0.774 / 0.741 |
| C | both | 46 / 39 | 170 / 284 | 0.372 / 0.107 | 0.866 | 0.583 | 0.604 | 0.2093 / 0.1084 | 0.883 / 0.832 |
| D | cue_gold | 19 / 14 | 137 / 211 | 0.363 / 0.155 | 0.805 (0.805) | 0.591 | 0.746 | 0.2667 / 0.1076 | 0.776 / 0.816 |
| D | cue_dist | 27 / 25 | 33 / 73 | 0.405 / 0.232 | 0.687 (0.687) | 0.604 | 0.849 | 0.5065 / 0.1558 | 0.627 / 0.650 |
| D | both | 46 / 39 | 170 / 284 | 0.389 / 0.173 | 0.765 | 0.583 | 0.764 | 0.3333 / 0.1222 | 0.769 / 0.739 |

## s43_round4: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 139 / 209 | 0.744 / 0.411 | 0.864 (0.864) | 0.715 | 0.731 | 0.5606 / 0.3549 | 0.837 / 0.761 |
| A | cue_dist | 74 / 46 | 22 / 61 | 0.729 / 0.443 | 0.806 (0.806) | 0.676 | 0.834 | 0.7691 / 0.4708 | 0.727 / 0.756 |
| A | both | 108 / 53 | 161 / 270 | 0.733 / 0.417 | 0.836 | 0.649 | 0.758 | 0.6351 / 0.3933 | 0.833 / 0.770 |
| B | cue_gold | 20 / 4 | 143 / 212 | 0.530 / 0.230 | 0.831 (0.831) | 0.715 | 0.694 | 0.3307 / 0.1873 | 0.787 / 0.772 |
| B | cue_dist | 28 / 9 | 29 / 64 | 0.534 / 0.279 | 0.792 (0.792) | 0.722 | 0.862 | 0.5288 / 0.2136 | 0.667 / 0.762 |
| B | both | 48 / 13 | 172 / 276 | 0.533 / 0.240 | 0.828 | 0.701 | 0.733 | 0.3820 / 0.1940 | 0.784 / 0.775 |
| C | cue_gold | 20 / 4 | 143 / 212 | 0.550 / 0.168 | 0.885 (0.885) | 0.715 | 0.612 | 0.2530 / 0.1471 | 0.870 / 0.817 |
| C | cue_dist | 28 / 9 | 29 / 64 | 0.546 / 0.208 | 0.870 (0.870) | 0.722 | 0.828 | 0.4716 / 0.1743 | 0.757 / 0.913 |
| C | both | 48 / 13 | 172 / 276 | 0.548 / 0.177 | 0.887 | 0.701 | 0.661 | 0.3096 / 0.1540 | 0.855 / 0.877 |
| D | cue_gold | 20 / 4 | 143 / 212 | 0.544 / 0.191 | 0.869 (0.869) | 0.715 | 0.705 | 0.3012 / 0.1466 | 0.840 / 0.777 |
| D | cue_dist | 28 / 9 | 29 / 64 | 0.520 / 0.225 | 0.833 (0.833) | 0.722 | 0.900 | 0.5014 / 0.1582 | 0.722 / 0.832 |
| D | both | 48 / 13 | 172 / 276 | 0.529 / 0.198 | 0.863 | 0.701 | 0.750 | 0.3531 / 0.1495 | 0.827 / 0.810 |

## s43_round5: probe rows by answer letter (457 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 137 / 208 | 0.748 / 0.424 | 0.837 (0.837) | 0.716 | 0.804 | 0.6147 / 0.3334 | 0.825 / 0.599 |
| A | cue_dist | 74 / 45 | 22 / 62 | 0.714 / 0.454 | 0.777 (0.777) | 0.680 | 0.887 | 0.7942 / 0.4388 | 0.743 / 0.663 |
| A | both | 108 / 52 | 159 / 270 | 0.723 / 0.430 | 0.806 | 0.652 | 0.825 | 0.6792 / 0.3684 | 0.824 / 0.694 |
| B | cue_gold | 15 / 8 | 143 / 207 | 0.531 / 0.268 | 0.761 (0.761) | 0.622 | 0.777 | 0.4153 / 0.1874 | 0.816 / 0.602 |
| B | cue_dist | 23 / 13 | 35 / 60 | 0.440 / 0.334 | 0.613 (0.613) | 0.635 | 0.906 | 0.5703 / 0.1989 | 0.549 / 0.426 |
| B | both | 38 / 21 | 178 / 267 | 0.475 / 0.282 | 0.700 | 0.622 | 0.810 | 0.4569 / 0.1903 | 0.737 / 0.516 |
| C | cue_gold | 15 / 8 | 143 / 207 | 0.446 / 0.170 | 0.786 (0.786) | 0.622 | 0.720 | 0.2778 / 0.1201 | 0.841 / 0.641 |
| C | cue_dist | 23 / 13 | 35 / 60 | 0.366 / 0.211 | 0.674 (0.674) | 0.635 | 0.904 | 0.4253 / 0.1168 | 0.692 / 0.472 |
| C | both | 38 / 21 | 178 / 267 | 0.397 / 0.179 | 0.738 | 0.622 | 0.766 | 0.3174 / 0.1193 | 0.798 / 0.556 |
| D | cue_gold | 15 / 8 | 143 / 207 | 0.476 / 0.189 | 0.796 (0.796) | 0.622 | 0.653 | 0.2799 / 0.1525 | 0.852 / 0.660 |
| D | cue_dist | 23 / 13 | 35 / 60 | 0.406 / 0.243 | 0.675 (0.675) | 0.635 | 0.861 | 0.4496 / 0.1591 | 0.701 / 0.487 |
| D | both | 38 / 21 | 178 / 267 | 0.433 / 0.200 | 0.751 | 0.622 | 0.703 | 0.3254 / 0.1541 | 0.813 / 0.578 |

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
  "s43_round2": {
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
  "s43_round3": {
    "train": {
      "biased": {
        "answers": 2022,
        "naming_a": 1335
      },
      "unbiased": {
        "answers": 2022,
        "naming_a": 679
      }
    },
    "eval": {
      "biased": {
        "answers": 153,
        "naming_a": 93
      },
      "unbiased": {
        "answers": 448,
        "naming_a": 154
      }
    }
  },
  "s43_round4": {
    "train": {
      "biased": {
        "answers": 2292,
        "naming_a": 1480
      },
      "unbiased": {
        "answers": 2292,
        "naming_a": 776
      }
    },
    "eval": {
      "biased": {
        "answers": 85,
        "naming_a": 46
      },
      "unbiased": {
        "answers": 454,
        "naming_a": 170
      }
    }
  },
  "s43_round5": {
    "train": {
      "biased": {
        "answers": 2492,
        "naming_a": 1626
      },
      "unbiased": {
        "answers": 2492,
        "naming_a": 866
      }
    },
    "eval": {
      "biased": {
        "answers": 61,
        "naming_a": 48
      },
      "unbiased": {
        "answers": 448,
        "naming_a": 172
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
  "s43_round2": {
    "13": 0.8311685919761658,
    "14": 0.8219417929649353,
    "15": 0.8017721772193909,
    "16": 0.7760712504386902
  },
  "s43_round3": {
    "13": 0.8331600427627563,
    "14": 0.7761632204055786,
    "15": 0.7457665801048279,
    "16": 0.6999416947364807
  },
  "s43_round4": {
    "13": 0.9116098880767822,
    "14": 0.8980365991592407,
    "15": 0.8858816623687744,
    "16": 0.8665727376937866
  },
  "s43_round5": {
    "13": 0.9199488759040833,
    "14": 0.903130054473877,
    "15": 0.8910172581672668,
    "16": 0.8770672678947449
  }
}
```
