# Training dynamics and probe rows: s45_round1, s45_refit_round2, s45_refit_round3, s45_refit_round4, s45_refit_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: s45_round1 0.3, s45_refit_round2 0.3, s45_refit_round3 0.3, s45_refit_round4 0.3, s45_refit_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | s45_round1 | 38.400 | 42.750 | 37.167 | 34.162 |
| groups where all 8 answers name the cue (of 64) | s45_refit_round2 | 37.800 | 38.050 | 42.067 | 46.338 |
| groups where all 8 answers name the cue (of 64) | s45_refit_round3 | 41.500 | 38.950 | 39.967 | 38.971 |
| groups where all 8 answers name the cue (of 64) | s45_refit_round4 | 38.700 | 38.950 | 38.100 | 41.544 |
| groups where all 8 answers name the cue (of 64) | s45_refit_round5 | 36.800 | 40.700 | 40.300 | 42.074 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | s45_round1 | 20.600 | 16.100 | 24 | 20.279 |
| groups with mixed task rewards (of 64) | s45_refit_round2 | 15.600 | 17.200 | 14.500 | 11.088 |
| groups with mixed task rewards (of 64) | s45_refit_round3 | 12.300 | 10.200 | 8.733 | 8.647 |
| groups with mixed task rewards (of 64) | s45_refit_round4 | 7.900 | 9.350 | 8.733 | 5.632 |
| groups with mixed task rewards (of 64) | s45_refit_round5 | 7.300 | 7.900 | 7.833 | 5.515 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | s45_round1 | 6.300 | 6.450 | 8.433 | 6.221 |
|   of which the cue is not at the training label | s45_refit_round2 | 4.800 | 6.600 | 5.933 | 5.338 |
|   of which the cue is not at the training label | s45_refit_round3 | 4.300 | 3.500 | 2.733 | 2.897 |
|   of which the cue is not at the training label | s45_refit_round4 | 2.100 | 3.450 | 2.700 | 2.074 |
|   of which the cue is not at the training label | s45_refit_round5 | 2 | 2.750 | 2.567 | 2.191 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | s45_round1 | 102.200 | 95.500 | 86 | 145.706 |
| answers not naming the cue (of 512) | s45_refit_round2 | 140.500 | 134.700 | 108.300 | 90.765 |
| answers not naming the cue (of 512) | s45_refit_round3 | 126.100 | 155.700 | 154.867 | 162.794 |
| answers not naming the cue (of 512) | s45_refit_round4 | 170.300 | 162.950 | 166.700 | 155.235 |
| answers not naming the cue (of 512) | s45_refit_round5 | 188.800 | 155.700 | 156.167 | 153.706 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | s45_round1 | 0.316 | 0.282 | 0.437 | 0.257 |
| mean absolute advantage per answer | s45_refit_round2 | 0.225 | 0.264 | 0.242 | 0.201 |
| mean absolute advantage per answer | s45_refit_round3 | 0.239 | 0.171 | 0.141 | 0.134 |
| mean absolute advantage per answer | s45_refit_round4 | 0.142 | 0.159 | 0.152 | 0.130 |
| mean absolute advantage per answer | s45_refit_round5 | 0.131 | 0.154 | 0.165 | 0.142 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | s45_round1 | 0.125 | 0.105 | 0.189 | 0.080 |
|   from probe-only groups | s45_refit_round2 | 0.096 | 0.078 | 0.070 | 0.072 |
|   from probe-only groups | s45_refit_round3 | 0.141 | 0.092 | 0.075 | 0.067 |
|   from probe-only groups | s45_refit_round4 | 0.091 | 0.088 | 0.081 | 0.084 |
|   from probe-only groups | s45_refit_round5 | 0.081 | 0.093 | 0.103 | 0.096 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | s45_round1 | 0.011 | 0.006 | 0.007 | 0.008 |
| advantage on cue-naming answers, per answer | s45_refit_round2 | 0.001 | 0.010 | 0.012 | 0.000 |
| advantage on cue-naming answers, per answer | s45_refit_round3 | -0.004 | -0.002 | -0.002 | -0.002 |
| advantage on cue-naming answers, per answer | s45_refit_round4 | 0.001 | -0.005 | 0.001 | -0.002 |
| advantage on cue-naming answers, per answer | s45_refit_round5 | 0.002 | -0.003 | -0.000 | -0.004 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | s45_round1 | 0.088 | 0.090 | 0.255 | 0.067 |
| gradient norm (W&B, before clipping) | s45_refit_round2 | 0.081 | 0.078 | 0.056 | 0.055 |
| gradient norm (W&B, before clipping) | s45_refit_round3 | 0.057 | 0.054 | 0.039 | 0.039 |
| gradient norm (W&B, before clipping) | s45_refit_round4 | 0.035 | 0.043 | 0.039 | 0.041 |
| gradient norm (W&B, before clipping) | s45_refit_round5 | 0.037 | 0.037 | 0.040 | 0.037 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | s45_round1 | 0.021 | 0.124 | 0.259 | 0.060 |
| KL from the base model (W&B) | s45_refit_round2 | 0.062 | 0.144 | 0.104 | 0.108 |
| KL from the base model (W&B) | s45_refit_round3 | 0.111 | 0.106 | 0.065 | 0.069 |
| KL from the base model (W&B) | s45_refit_round4 | 0.069 | 0.087 | 0.089 | 0.078 |
| KL from the base model (W&B) | s45_refit_round5 | 0.082 | 0.083 | 0.095 | 0.093 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | s45_round1 | 0.428 | 0.406 | 0.502 | 0.437 |
| token entropy (W&B) | s45_refit_round2 | 0.437 | 0.428 | 0.488 | 0.456 |
| token entropy (W&B) | s45_refit_round3 | 0.481 | 0.488 | 0.463 | 0.497 |
| token entropy (W&B) | s45_refit_round4 | 0.505 | 0.483 | 0.506 | 0.473 |
| token entropy (W&B) | s45_refit_round5 | 0.489 | 0.502 | 0.507 | 0.463 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | s45_round1 | 231.155 | 199.854 | 269.242 | 203.195 |
| answer length, tokens (W&B) | s45_refit_round2 | 192.275 | 174.457 | 250.931 | 226.456 |
| answer length, tokens (W&B) | s45_refit_round3 | 197.438 | 194.002 | 220.328 | 204.669 |
| answer length, tokens (W&B) | s45_refit_round4 | 205.303 | 195.719 | 204.090 | 215.664 |
| answer length, tokens (W&B) | s45_refit_round5 | 197.381 | 205.833 | 205.708 | 197.458 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of s45_round1: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 162 | 0.078 | 957.414 / 1063.725 | 0.165 / 0.267 | 0.219 / 0.356 | 0.407 / 0.520 |
| 5-9 | 222 | 0.044 | 880.468 / 956.427 | 0.123 / 0.190 | 0.173 / 0.262 | 0.288 / 0.368 |
| 10-29 | 855 | 0.031 | 813.918 / 856.809 | 0.092 / 0.123 | 0.191 / 0.227 | 0.282 / 0.326 |
| 30-59 | 1115 | 0.119 | 1144.248 / 1173.374 | 0.263 / 0.283 | 0.332 / 0.359 | 0.441 / 0.453 |
| 60-127 | 2323 | 0.034 | 821.018 / 878.358 | 0.083 / 0.133 | 0.123 / 0.190 | 0.295 / 0.347 |

## Probe-only groups of s45_refit_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 187 | 0.037 | 801.794 / 850.060 | 0.072 / 0.132 | 0.147 / 0.193 | 0.321 / 0.332 |
| 5-9 | 191 | 0.047 | 793.264 / 854.569 | 0.064 / 0.111 | 0.115 / 0.170 | 0.298 / 0.374 |
| 10-29 | 761 | 0.021 | 703.135 / 746.219 | 0.040 / 0.058 | 0.137 / 0.155 | 0.249 / 0.289 |
| 30-59 | 1262 | 0.019 | 1023.455 / 1124.919 | 0.117 / 0.196 | 0.221 / 0.288 | 0.396 / 0.451 |
| 60-127 | 3151 | 0.016 | 942.485 / 1022.493 | 0.090 / 0.162 | 0.206 / 0.267 | 0.365 / 0.427 |

## Probe-only groups of s45_refit_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 211 | 0.085 | 908.167 / 1013.415 | 0.073 / 0.162 | 0.177 / 0.293 | 0.326 / 0.416 |
| 5-9 | 204 | 0.073 | 733.105 / 766.424 | 0.029 / 0.040 | 0.082 / 0.132 | 0.271 / 0.320 |
| 10-29 | 779 | 0.045 | 796.964 / 849.247 | 0.028 / 0.058 | 0.073 / 0.128 | 0.285 / 0.324 |
| 30-59 | 1199 | 0.030 | 922.611 / 975.092 | 0.029 / 0.070 | 0.081 / 0.130 | 0.323 / 0.382 |
| 60-127 | 2650 | 0.026 | 855.863 / 905.473 | 0.029 / 0.063 | 0.077 / 0.123 | 0.291 / 0.347 |

## Probe-only groups of s45_refit_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 205 | 0.051 | 863.424 / 916.539 | 0.027 / 0.049 | 0.049 / 0.091 | 0.272 / 0.328 |
| 5-9 | 182 | 0.048 | 860.646 / 901.404 | 0.055 / 0.102 | 0.096 / 0.147 | 0.295 / 0.320 |
| 10-29 | 779 | 0.047 | 809.493 / 852.514 | 0.029 / 0.061 | 0.074 / 0.118 | 0.260 / 0.315 |
| 30-59 | 1143 | 0.042 | 854.934 / 896.918 | 0.025 / 0.042 | 0.065 / 0.094 | 0.296 / 0.352 |
| 60-127 | 2825 | 0.038 | 902.885 / 961.055 | 0.031 / 0.078 | 0.080 / 0.135 | 0.328 / 0.383 |

## Probe-only groups of s45_refit_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 180 | 0.053 | 872.374 / 939.951 | 0.025 / 0.067 | 0.068 / 0.132 | 0.292 / 0.382 |
| 5-9 | 188 | 0.040 | 776.955 / 843.693 | 0.037 / 0.062 | 0.073 / 0.112 | 0.290 / 0.320 |
| 10-29 | 814 | 0.046 | 853.544 / 915.190 | 0.036 / 0.059 | 0.076 / 0.112 | 0.313 / 0.375 |
| 30-59 | 1209 | 0.053 | 858.002 / 919.604 | 0.037 / 0.074 | 0.086 / 0.136 | 0.323 / 0.391 |
| 60-127 | 2861 | 0.042 | 813.436 / 871.893 | 0.032 / 0.066 | 0.078 / 0.125 | 0.288 / 0.354 |

## s45_round1's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.131 | 0.029 | 978.500 |
| 10 | 0.008 | 0.004 | 828.500 |
| 20 | 0.010 | 0.020 | 797.500 |
| 25 | 0.010 | 0.023 | 834.000 |
| 30 | 0.012 | 0.045 | 1024.500 |
| 60 | 0.068 | 0.047 | 1005.000 |
| 70 | 0.047 | 0.012 | 892.500 |
| 80 | 0.031 | 0.002 | 836.000 |
| 90 | 0.039 | 0.006 | 849.000 |
| 100 | 0.029 | 0.004 | 801.000 |
| 127 | 0.051 | 0.008 | 889.000 |

## s45_refit_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.037 | 0.004 | 797.500 |
| 10 | 0.031 | 0.020 | 774.000 |
| 20 | 0.021 | 0.033 | 593.500 |
| 25 | 0.035 | 0.006 | 701.500 |
| 30 | 0.078 | 0.006 | 874.000 |
| 60 | 0.010 | 0.010 | 989.500 |
| 70 | 0.012 | 0.010 | 985.500 |
| 80 | 0.010 | 0 | 905.000 |
| 90 | 0.033 | 0.002 | 981.500 |
| 100 | 0.021 | 0.006 | 977.500 |
| 127 | 0.027 | 0.004 | 941.500 |

## s45_refit_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.023 | 0.002 | 918.500 |
| 10 | 0.047 | 0 | 648.000 |
| 20 | 0.045 | 0.020 | 767.000 |
| 25 | 0.064 | 0.006 | 905.500 |
| 30 | 0.068 | 0.018 | 1057.000 |
| 60 | 0.039 | 0.025 | 891.000 |
| 70 | 0.012 | 0.006 | 754.000 |
| 80 | 0.018 | 0.016 | 787.000 |
| 90 | 0.014 | 0.021 | 896.000 |
| 100 | 0.029 | 0.006 | 832.500 |
| 127 | 0.031 | 0.008 | 819.500 |

## s45_refit_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.020 | 0.006 | 869.500 |
| 10 | 0.119 | 0.006 | 763.500 |
| 20 | 0.033 | 0.008 | 783.500 |
| 25 | 0.037 | 0.006 | 922.500 |
| 30 | 0.051 | 0.010 | 902.500 |
| 60 | 0.098 | 0.008 | 892.000 |
| 70 | 0.145 | 0.004 | 886.500 |
| 80 | 0.172 | 0.021 | 835.500 |
| 90 | 0.158 | 0.002 | 962.500 |
| 100 | 0.113 | 0.006 | 829.000 |
| 127 | 0.121 | 0.021 | 878.500 |

## s45_refit_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.105 | 0.006 | 891.500 |
| 10 | 0.025 | 0 | 783.500 |
| 20 | 0.029 | 0.004 | 973.500 |
| 25 | 0.014 | 0.010 | 859.000 |
| 30 | 0.014 | 0.008 | 811.000 |
| 60 | 0.068 | 0 | 845.000 |
| 70 | 0.066 | 0.014 | 826.000 |
| 80 | 0.045 | 0.002 | 781.000 |
| 90 | 0.080 | 0.004 | 849.000 |
| 100 | 0.066 | 0.018 | 834.000 |
| 127 | 0.088 | 0.018 | 821.000 |

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
  "s45_round1": {
    "median_tokens_no_cue": 172.0,
    "median_tokens_cued": 181.0,
    "mean_top_letter_share_of_8_no_cue": 0.9309815950920245,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 165,
      "B": 317,
      "null": 7
    },
    "no_cue_answer_letters": {
      "A": 164,
      "B": 312
    },
    "questions_with_no_cue_answer": 476
  },
  "s45_refit_round2": {
    "median_tokens_no_cue": 200.0,
    "median_tokens_cued": 211.0,
    "mean_top_letter_share_of_8_no_cue": 0.9179447852760736,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 205,
      "B": 283,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 192,
      "B": 274
    },
    "questions_with_no_cue_answer": 466
  },
  "s45_refit_round3": {
    "median_tokens_no_cue": 175.5,
    "median_tokens_cued": 187.0,
    "mean_top_letter_share_of_8_no_cue": 0.9521983640081799,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 152,
      "B": 336,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 154,
      "B": 328
    },
    "questions_with_no_cue_answer": 482
  },
  "s45_refit_round4": {
    "median_tokens_no_cue": 184.0,
    "median_tokens_cued": 195.0,
    "mean_top_letter_share_of_8_no_cue": 0.9378834355828221,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 196,
      "B": 288,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 184,
      "B": 290
    },
    "questions_with_no_cue_answer": 474
  },
  "s45_refit_round5": {
    "median_tokens_no_cue": 173.0,
    "median_tokens_cued": 184.0,
    "mean_top_letter_share_of_8_no_cue": 0.9542433537832311,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 167,
      "B": 321,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 161,
      "B": 317
    },
    "questions_with_no_cue_answer": 478
  }
}
```

## s45_round1: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 133 / 211 | 0.763 / 0.220 | 0.931 (0.931) | 0.723 | 0.673 | 0.3958 / 0.1884 | 0.923 / 0.879 |
| A | cue_dist | 75 / 45 | 21 / 63 | 0.741 / 0.268 | 0.874 (0.874) | 0.688 | 0.824 | 0.7559 / 0.3604 | 0.878 / 0.804 |
| A | both | 110 / 52 | 154 / 274 | 0.747 / 0.229 | 0.906 | 0.660 | 0.717 | 0.5268 / 0.2454 | 0.927 / 0.841 |
| B | cue_gold | 32 / 10 | 124 / 217 | 0.607 / 0.136 | 0.919 (0.919) | 0.699 | 0.594 | 0.2697 / 0.1318 | 0.914 / 0.870 |
| B | cue_dist | 29 / 21 | 24 / 70 | 0.599 / 0.211 | 0.828 (0.828) | 0.662 | 0.750 | 0.5378 / 0.2338 | 0.796 / 0.759 |
| B | both | 61 / 31 | 148 / 287 | 0.602 / 0.152 | 0.891 | 0.661 | 0.621 | 0.3377 / 0.1610 | 0.897 / 0.825 |
| C | cue_gold | 32 / 10 | 124 / 217 | 0.660 / 0.115 | 0.933 (0.933) | 0.699 | 0.645 | 0.2876 / 0.0972 | 0.936 / 0.866 |
| C | cue_dist | 29 / 21 | 24 / 70 | 0.604 / 0.166 | 0.840 (0.840) | 0.662 | 0.812 | 0.5657 / 0.1742 | 0.829 / 0.774 |
| C | both | 61 / 31 | 148 / 287 | 0.630 / 0.126 | 0.899 | 0.661 | 0.676 | 0.3581 / 0.1192 | 0.917 / 0.825 |
| D | cue_gold | 32 / 10 | 124 / 217 | 0.601 / 0.131 | 0.920 (0.920) | 0.699 | 0.580 | 0.2553 / 0.1329 | 0.925 / 0.855 |
| D | cue_dist | 29 / 21 | 24 / 70 | 0.583 / 0.192 | 0.829 (0.829) | 0.662 | 0.736 | 0.5167 / 0.2177 | 0.800 / 0.756 |
| D | both | 61 / 31 | 148 / 287 | 0.592 / 0.144 | 0.890 | 0.661 | 0.610 | 0.3216 / 0.1571 | 0.906 / 0.812 |

## s45_refit_round2: probe rows by answer letter (445 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 29 / 7 | 135 / 204 | 0.770 / 0.317 | 0.913 (0.913) | 0.704 | 0.617 | 0.4295 / 0.3068 | 0.910 / 0.869 |
| A | cue_dist | 70 / 44 | 22 / 58 | 0.760 / 0.365 | 0.862 (0.862) | 0.670 | 0.753 | 0.7365 / 0.4707 | 0.862 / 0.828 |
| A | both | 99 / 51 | 157 / 262 | 0.762 / 0.326 | 0.889 | 0.643 | 0.661 | 0.5399 / 0.3602 | 0.905 / 0.845 |
| B | cue_gold | 34 / 19 | 147 / 211 | 0.197 / 0.072 | 0.806 (0.806) | 0.615 | 0.765 | 0.1334 / 0.0529 | 0.816 / 0.767 |
| B | cue_dist | 105 / 44 | 29 / 56 | 0.249 / 0.099 | 0.763 (0.763) | 0.682 | 0.878 | 0.2814 / 0.0782 | 0.767 / 0.601 |
| B | both | 139 / 63 | 176 / 267 | 0.235 / 0.077 | 0.804 | 0.645 | 0.811 | 0.1963 / 0.0606 | 0.834 / 0.695 |
| C | cue_gold | 34 / 19 | 147 / 211 | 0.105 / 0.035 | 0.794 (0.794) | 0.615 | 0.756 | 0.0677 / 0.0260 | 0.829 / 0.708 |
| C | cue_dist | 105 / 44 | 29 / 56 | 0.155 / 0.048 | 0.765 (0.765) | 0.682 | 0.880 | 0.1720 / 0.0413 | 0.774 / 0.599 |
| C | both | 139 / 63 | 176 / 267 | 0.142 / 0.038 | 0.801 | 0.645 | 0.806 | 0.1121 / 0.0306 | 0.843 / 0.671 |
| D | cue_gold | 34 / 19 | 147 / 211 | 0.309 / 0.125 | 0.798 (0.798) | 0.615 | 0.858 | 0.2400 / 0.0767 | 0.835 / 0.744 |
| D | cue_dist | 105 / 44 | 29 / 56 | 0.341 / 0.142 | 0.773 (0.773) | 0.682 | 0.918 | 0.3925 / 0.1037 | 0.775 / 0.618 |
| D | both | 139 / 63 | 176 / 267 | 0.333 / 0.128 | 0.794 | 0.645 | 0.882 | 0.3049 / 0.0849 | 0.821 / 0.701 |

## s45_refit_round3: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 135 / 211 | 0.771 / 0.337 | 0.899 (0.899) | 0.724 | 0.752 | 0.5261 / 0.2739 | 0.893 / 0.782 |
| A | cue_dist | 76 / 43 | 22 / 64 | 0.753 / 0.399 | 0.843 (0.843) | 0.691 | 0.848 | 0.7885 / 0.4366 | 0.790 / 0.788 |
| A | both | 112 / 50 | 157 / 275 | 0.758 / 0.349 | 0.879 | 0.664 | 0.774 | 0.6217 / 0.3274 | 0.883 / 0.821 |
| B | cue_gold | 24 / 10 | 123 / 227 | 0.572 / 0.216 | 0.883 (0.883) | 0.677 | 0.704 | 0.3520 / 0.1834 | 0.837 / 0.890 |
| B | cue_dist | 20 / 26 | 18 / 81 | 0.528 / 0.285 | 0.787 (0.787) | 0.626 | 0.763 | 0.5421 / 0.2978 | 0.786 / 0.760 |
| B | both | 44 / 36 | 141 / 308 | 0.546 / 0.232 | 0.850 | 0.618 | 0.695 | 0.3910 / 0.2190 | 0.838 / 0.842 |
| C | cue_gold | 24 / 10 | 123 / 227 | 0.565 / 0.130 | 0.914 (0.914) | 0.677 | 0.744 | 0.2961 / 0.0893 | 0.899 / 0.907 |
| C | cue_dist | 20 / 26 | 18 / 81 | 0.492 / 0.155 | 0.847 (0.847) | 0.626 | 0.847 | 0.5249 / 0.1689 | 0.861 / 0.816 |
| C | both | 44 / 36 | 141 / 308 | 0.523 / 0.135 | 0.886 | 0.618 | 0.743 | 0.3431 / 0.1141 | 0.898 / 0.871 |
| D | cue_gold | 24 / 10 | 123 / 227 | 0.632 / 0.182 | 0.910 (0.910) | 0.677 | 0.635 | 0.3108 / 0.1665 | 0.876 / 0.919 |
| D | cue_dist | 20 / 26 | 18 / 81 | 0.565 / 0.248 | 0.824 (0.824) | 0.626 | 0.709 | 0.5053 / 0.2928 | 0.847 / 0.790 |
| D | both | 44 / 36 | 141 / 308 | 0.593 / 0.196 | 0.881 | 0.618 | 0.631 | 0.3508 / 0.2058 | 0.872 / 0.868 |

## s45_refit_round4: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 134 / 207 | 0.749 / 0.376 | 0.904 (0.904) | 0.729 | 0.570 | 0.4528 / 0.3854 | 0.913 / 0.821 |
| A | cue_dist | 72 / 45 | 21 / 62 | 0.736 / 0.478 | 0.786 (0.786) | 0.681 | 0.673 | 0.7029 / 0.5652 | 0.738 / 0.787 |
| A | both | 106 / 51 | 155 / 269 | 0.740 / 0.396 | 0.867 | 0.655 | 0.603 | 0.5419 / 0.4455 | 0.873 / 0.844 |
| B | cue_gold | 22 / 11 | 135 / 207 | 0.595 / 0.263 | 0.862 (0.862) | 0.636 | 0.502 | 0.2986 / 0.2878 | 0.873 / 0.856 |
| B | cue_dist | 24 / 21 | 33 / 69 | 0.620 / 0.385 | 0.757 (0.757) | 0.605 | 0.697 | 0.5691 / 0.3854 | 0.763 / 0.707 |
| B | both | 46 / 32 | 168 / 276 | 0.609 / 0.291 | 0.836 | 0.606 | 0.541 | 0.3707 / 0.3163 | 0.855 / 0.796 |
| C | cue_gold | 22 / 11 | 135 / 207 | 0.626 / 0.185 | 0.918 (0.918) | 0.636 | 0.563 | 0.2567 / 0.1996 | 0.919 / 0.901 |
| C | cue_dist | 24 / 21 | 33 / 69 | 0.590 / 0.296 | 0.789 (0.789) | 0.605 | 0.698 | 0.5097 / 0.3074 | 0.822 / 0.728 |
| C | both | 46 / 32 | 168 / 276 | 0.605 / 0.210 | 0.880 | 0.606 | 0.588 | 0.3240 / 0.2311 | 0.901 / 0.837 |
| D | cue_gold | 22 / 11 | 135 / 207 | 0.622 / 0.207 | 0.874 (0.874) | 0.636 | 0.732 | 0.3632 / 0.1580 | 0.877 / 0.833 |
| D | cue_dist | 24 / 21 | 33 / 69 | 0.579 / 0.304 | 0.754 (0.754) | 0.605 | 0.874 | 0.6423 / 0.2278 | 0.798 / 0.710 |
| D | both | 46 / 32 | 168 / 276 | 0.598 / 0.230 | 0.837 | 0.606 | 0.753 | 0.4375 / 0.1784 | 0.869 / 0.795 |

## s45_refit_round5: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 135 / 209 | 0.816 / 0.423 | 0.886 (0.886) | 0.716 | 0.806 | 0.6340 / 0.3323 | 0.877 / 0.707 |
| A | cue_dist | 75 / 44 | 22 / 61 | 0.762 / 0.476 | 0.798 (0.798) | 0.683 | 0.868 | 0.8175 / 0.4850 | 0.672 / 0.760 |
| A | both | 108 / 51 | 157 / 270 | 0.776 / 0.434 | 0.845 | 0.656 | 0.819 | 0.7012 / 0.3823 | 0.824 / 0.798 |
| B | cue_gold | 25 / 6 | 131 / 226 | 0.665 / 0.284 | 0.891 (0.891) | 0.720 | 0.684 | 0.4102 / 0.2502 | 0.853 / 0.880 |
| B | cue_dist | 33 / 29 | 19 / 76 | 0.549 / 0.345 | 0.737 (0.737) | 0.666 | 0.771 | 0.5855 / 0.3463 | 0.641 / 0.678 |
| B | both | 58 / 35 | 150 / 302 | 0.588 / 0.297 | 0.821 | 0.646 | 0.692 | 0.4540 / 0.2801 | 0.810 / 0.767 |
| C | cue_gold | 25 / 6 | 131 / 226 | 0.650 / 0.188 | 0.925 (0.925) | 0.720 | 0.730 | 0.3417 / 0.1467 | 0.892 / 0.928 |
| C | cue_dist | 33 / 29 | 19 / 76 | 0.500 / 0.232 | 0.788 (0.788) | 0.666 | 0.831 | 0.5561 / 0.2295 | 0.708 / 0.732 |
| C | both | 58 / 35 | 150 / 302 | 0.550 / 0.197 | 0.856 | 0.646 | 0.739 | 0.3953 / 0.1725 | 0.854 / 0.814 |
| D | cue_gold | 25 / 6 | 131 / 226 | 0.722 / 0.252 | 0.922 (0.922) | 0.720 | 0.717 | 0.4011 / 0.2141 | 0.896 / 0.922 |
| D | cue_dist | 33 / 29 | 19 / 76 | 0.575 / 0.308 | 0.783 (0.783) | 0.666 | 0.812 | 0.6131 / 0.3148 | 0.746 / 0.702 |
| D | both | 58 / 35 | 150 / 302 | 0.624 / 0.264 | 0.849 | 0.646 | 0.721 | 0.4541 / 0.2455 | 0.864 / 0.789 |

## Probe data: answers naming A

```
{
  "s45_round1": {
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
  "s45_refit_round2": {
    "train": {
      "biased": {
        "answers": 1098,
        "naming_a": 712
      },
      "unbiased": {
        "answers": 1098,
        "naming_a": 359
      }
    },
    "eval": {
      "biased": {
        "answers": 92,
        "naming_a": 61
      },
      "unbiased": {
        "answers": 435,
        "naming_a": 148
      }
    }
  },
  "s45_refit_round3": {
    "train": {
      "biased": {
        "answers": 1766,
        "naming_a": 1122
      },
      "unbiased": {
        "answers": 1766,
        "naming_a": 635
      }
    },
    "eval": {
      "biased": {
        "answers": 202,
        "naming_a": 139
      },
      "unbiased": {
        "answers": 443,
        "naming_a": 176
      }
    }
  },
  "s45_refit_round4": {
    "train": {
      "biased": {
        "answers": 2172,
        "naming_a": 1374
      },
      "unbiased": {
        "answers": 2172,
        "naming_a": 729
      }
    },
    "eval": {
      "biased": {
        "answers": 80,
        "naming_a": 44
      },
      "unbiased": {
        "answers": 449,
        "naming_a": 141
      }
    }
  },
  "s45_refit_round5": {
    "train": {
      "biased": {
        "answers": 2378,
        "naming_a": 1493
      },
      "unbiased": {
        "answers": 2378,
        "naming_a": 785
      }
    },
    "eval": {
      "biased": {
        "answers": 78,
        "naming_a": 46
      },
      "unbiased": {
        "answers": 444,
        "naming_a": 168
      }
    }
  }
}
```

## Cosine of the initial and fresh probe weights

```
{
  "s45_round1": {
    "13": 0.8955491781234741,
    "14": 0.8828500509262085,
    "15": 0.8789872527122498,
    "16": 0.8738847374916077
  },
  "s45_refit_round2": {
    "13": 0.8325757384300232,
    "14": 0.8153908252716064,
    "15": 0.7807175517082214,
    "16": 0.7550349235534668
  },
  "s45_refit_round3": {
    "13": 0.8617383241653442,
    "14": 0.8481206297874451,
    "15": 0.8301019072532654,
    "16": 0.7981904149055481
  },
  "s45_refit_round4": {
    "13": 0.9066213965415955,
    "14": 0.8862028121948242,
    "15": 0.8734686970710754,
    "16": 0.8696559071540833
  },
  "s45_refit_round5": {
    "13": 0.9100795388221741,
    "14": 0.889392614364624,
    "15": 0.8810854554176331,
    "16": 0.8786225318908691
  }
}
```
