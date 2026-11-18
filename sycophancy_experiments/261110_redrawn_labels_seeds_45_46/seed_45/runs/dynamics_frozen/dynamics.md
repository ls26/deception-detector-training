# Training dynamics and probe rows: s45_round1, s45_frozen_round2, s45_frozen_round3, s45_frozen_round4, s45_frozen_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: s45_round1 0.3, s45_frozen_round2 0.3, s45_frozen_round3 0.3, s45_frozen_round4 0.3, s45_frozen_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | s45_round1 | 38.400 | 42.750 | 37.167 | 34.162 |
| groups where all 8 answers name the cue (of 64) | s45_frozen_round2 | 35.800 | 35.250 | 38.767 | 40.618 |
| groups where all 8 answers name the cue (of 64) | s45_frozen_round3 | 39.500 | 42 | 42.500 | 47.500 |
| groups where all 8 answers name the cue (of 64) | s45_frozen_round4 | 48.200 | 44.650 | 51.100 | 47.279 |
| groups where all 8 answers name the cue (of 64) | s45_frozen_round5 | 46 | 48.100 | 46.567 | 47.221 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | s45_round1 | 20.600 | 16.100 | 24 | 20.279 |
| groups with mixed task rewards (of 64) | s45_frozen_round2 | 17.700 | 17 | 13.500 | 9.324 |
| groups with mixed task rewards (of 64) | s45_frozen_round3 | 9.500 | 11.150 | 10.700 | 9.632 |
| groups with mixed task rewards (of 64) | s45_frozen_round4 | 9.800 | 10.050 | 7.300 | 7.618 |
| groups with mixed task rewards (of 64) | s45_frozen_round5 | 9.300 | 7.950 | 5.833 | 3.426 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | s45_round1 | 6.300 | 6.450 | 8.433 | 6.221 |
|   of which the cue is not at the training label | s45_frozen_round2 | 5 | 6.500 | 4.933 | 3.618 |
|   of which the cue is not at the training label | s45_frozen_round3 | 2.600 | 4.100 | 4.467 | 4.971 |
|   of which the cue is not at the training label | s45_frozen_round4 | 4 | 4.350 | 3.367 | 3.397 |
|   of which the cue is not at the training label | s45_frozen_round5 | 3.100 | 4.100 | 2.733 | 1.500 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | s45_round1 | 102.200 | 95.500 | 86 | 145.706 |
| answers not naming the cue (of 512) | s45_frozen_round2 | 151.200 | 158 | 145.900 | 149.676 |
| answers not naming the cue (of 512) | s45_frozen_round3 | 159.800 | 127.100 | 125.867 | 90.132 |
| answers not naming the cue (of 512) | s45_frozen_round4 | 79.100 | 110.500 | 68.133 | 96.338 |
| answers not naming the cue (of 512) | s45_frozen_round5 | 99.800 | 92.600 | 111.267 | 119.412 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | s45_round1 | 0.316 | 0.282 | 0.437 | 0.257 |
| mean absolute advantage per answer | s45_frozen_round2 | 0.232 | 0.231 | 0.179 | 0.133 |
| mean absolute advantage per answer | s45_frozen_round3 | 0.127 | 0.156 | 0.187 | 0.183 |
| mean absolute advantage per answer | s45_frozen_round4 | 0.200 | 0.176 | 0.163 | 0.152 |
| mean absolute advantage per answer | s45_frozen_round5 | 0.176 | 0.155 | 0.121 | 0.086 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | s45_round1 | 0.125 | 0.105 | 0.189 | 0.080 |
|   from probe-only groups | s45_frozen_round2 | 0.080 | 0.062 | 0.050 | 0.050 |
|   from probe-only groups | s45_frozen_round3 | 0.053 | 0.055 | 0.071 | 0.073 |
|   from probe-only groups | s45_frozen_round4 | 0.086 | 0.059 | 0.077 | 0.063 |
|   from probe-only groups | s45_frozen_round5 | 0.072 | 0.062 | 0.054 | 0.047 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | s45_round1 | 0.011 | 0.006 | 0.007 | 0.008 |
| advantage on cue-naming answers, per answer | s45_frozen_round2 | 0.010 | 0.003 | 0.004 | -0.003 |
| advantage on cue-naming answers, per answer | s45_frozen_round3 | 0.008 | 0.000 | -0.000 | -0.006 |
| advantage on cue-naming answers, per answer | s45_frozen_round4 | 0.008 | 0.007 | 0.002 | 0.003 |
| advantage on cue-naming answers, per answer | s45_frozen_round5 | 0.015 | -0.005 | 0.001 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | s45_round1 | 0.088 | 0.090 | 0.255 | 0.067 |
| gradient norm (W&B, before clipping) | s45_frozen_round2 | 0.058 | 0.067 | 0.052 | 0.042 |
| gradient norm (W&B, before clipping) | s45_frozen_round3 | 0.042 | 0.048 | 0.051 | 0.051 |
| gradient norm (W&B, before clipping) | s45_frozen_round4 | 0.052 | 0.055 | 0.046 | 0.042 |
| gradient norm (W&B, before clipping) | s45_frozen_round5 | 0.045 | 0.044 | 0.038 | 0.037 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | s45_round1 | 0.021 | 0.124 | 0.259 | 0.060 |
| KL from the base model (W&B) | s45_frozen_round2 | 0.047 | 0.067 | 0.063 | 0.057 |
| KL from the base model (W&B) | s45_frozen_round3 | 0.055 | 0.084 | 0.091 | 0.094 |
| KL from the base model (W&B) | s45_frozen_round4 | 0.102 | 0.118 | 0.101 | 0.102 |
| KL from the base model (W&B) | s45_frozen_round5 | 0.100 | 0.108 | 0.098 | 0.093 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | s45_round1 | 0.428 | 0.406 | 0.502 | 0.437 |
| token entropy (W&B) | s45_frozen_round2 | 0.441 | 0.429 | 0.473 | 0.503 |
| token entropy (W&B) | s45_frozen_round3 | 0.486 | 0.477 | 0.467 | 0.457 |
| token entropy (W&B) | s45_frozen_round4 | 0.459 | 0.421 | 0.464 | 0.489 |
| token entropy (W&B) | s45_frozen_round5 | 0.499 | 0.484 | 0.468 | 0.499 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | s45_round1 | 231.155 | 199.854 | 269.242 | 203.195 |
| answer length, tokens (W&B) | s45_frozen_round2 | 221.232 | 208.737 | 233.237 | 218.486 |
| answer length, tokens (W&B) | s45_frozen_round3 | 215.835 | 210.349 | 218.895 | 211.000 |
| answer length, tokens (W&B) | s45_frozen_round4 | 206.239 | 195.670 | 223.922 | 228.158 |
| answer length, tokens (W&B) | s45_frozen_round5 | 231.278 | 216.228 | 230.198 | 226.393 |
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

## Probe-only groups of s45_frozen_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 183 | 0.024 | 853.193 / 903.895 | 0.061 / 0.108 | 0.108 / 0.195 | 0.299 / 0.327 |
| 5-9 | 175 | 0.036 | 964.683 / 1052.631 | 0.107 / 0.216 | 0.164 / 0.291 | 0.273 / 0.394 |
| 10-29 | 705 | 0.017 | 864.765 / 915.727 | 0.055 / 0.099 | 0.133 / 0.184 | 0.325 / 0.381 |
| 30-59 | 1163 | 0.009 | 976.664 / 1048.949 | 0.061 / 0.086 | 0.161 / 0.231 | 0.396 / 0.466 |
| 60-127 | 2762 | 0.006 | 913.128 / 971.681 | 0.070 / 0.114 | 0.146 / 0.219 | 0.366 / 0.435 |

## Probe-only groups of s45_frozen_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 185 | 0.006 | 885.818 / 945.072 | 0.066 / 0.105 | 0.127 / 0.222 | 0.334 / 0.411 |
| 5-9 | 210 | 0.006 | 925.367 / 996.214 | 0.073 / 0.132 | 0.148 / 0.230 | 0.369 / 0.443 |
| 10-29 | 840 | 0.007 | 875.177 / 945.152 | 0.081 / 0.124 | 0.167 / 0.229 | 0.329 / 0.382 |
| 30-59 | 1275 | 0.010 | 924.601 / 981.801 | 0.104 / 0.157 | 0.206 / 0.249 | 0.361 / 0.403 |
| 60-127 | 3230 | 0.011 | 884.579 / 961.579 | 0.063 / 0.113 | 0.180 / 0.227 | 0.358 / 0.417 |

## Probe-only groups of s45_frozen_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 235 | 0.009 | 878.801 / 946.029 | 0.047 / 0.095 | 0.146 / 0.213 | 0.348 / 0.413 |
| 5-9 | 247 | 0.021 | 859.412 / 925.811 | 0.049 / 0.105 | 0.166 / 0.217 | 0.334 / 0.389 |
| 10-29 | 893 | 0.008 | 812.121 / 889.974 | 0.053 / 0.087 | 0.165 / 0.212 | 0.300 / 0.354 |
| 30-59 | 1533 | 0.009 | 948.359 / 1016.443 | 0.047 / 0.090 | 0.174 / 0.214 | 0.384 / 0.439 |
| 60-127 | 3215 | 0.011 | 967.359 / 1041.180 | 0.058 / 0.092 | 0.183 / 0.223 | 0.424 / 0.488 |

## Probe-only groups of s45_frozen_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 234 | 0.010 | 993.973 / 1075.550 | 0.067 / 0.109 | 0.170 / 0.233 | 0.404 / 0.468 |
| 5-9 | 226 | 0.010 | 975.425 / 1041.470 | 0.055 / 0.083 | 0.173 / 0.191 | 0.412 / 0.468 |
| 10-29 | 962 | 0.008 | 910.664 / 983.899 | 0.045 / 0.079 | 0.159 / 0.206 | 0.365 / 0.426 |
| 30-59 | 1397 | 0.008 | 980.062 / 1048.157 | 0.061 / 0.111 | 0.192 / 0.251 | 0.379 / 0.441 |
| 60-127 | 3211 | 0.005 | 960.406 / 1031.694 | 0.067 / 0.145 | 0.173 / 0.261 | 0.417 / 0.471 |

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

## s45_frozen_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.020 | 0.008 | 846.500 |
| 10 | 0.025 | 0.008 | 1029.500 |
| 20 | 0.016 | 0.008 | 846.500 |
| 25 | 0.020 | 0.014 | 859.500 |
| 30 | 0.107 | 0.018 | 956.500 |
| 60 | 0.035 | 0.027 | 1029.000 |
| 70 | 0.045 | 0.014 | 1012.500 |
| 80 | 0.025 | 0.023 | 809.500 |
| 90 | 0.043 | 0 | 969.000 |
| 100 | 0.072 | 0.004 | 897.000 |
| 127 | 0.076 | 0.002 | 927.500 |

## s45_frozen_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.088 | 0.008 | 884.500 |
| 10 | 0.117 | 0.021 | 1009.000 |
| 20 | 0.080 | 0.004 | 801.500 |
| 25 | 0.070 | 0 | 879.500 |
| 30 | 0.078 | 0.002 | 891.000 |
| 60 | 0.035 | 0.008 | 887.000 |
| 70 | 0.029 | 0.018 | 882.000 |
| 80 | 0.031 | 0.008 | 947.500 |
| 90 | 0.043 | 0.018 | 912.000 |
| 100 | 0.045 | 0.018 | 889.500 |
| 127 | 0.055 | 0.002 | 877.000 |

## s45_frozen_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.068 | 0.012 | 896.500 |
| 10 | 0.252 | 0.010 | 845.000 |
| 20 | 0.049 | 0.006 | 727.000 |
| 25 | 0.021 | 0.004 | 842.500 |
| 30 | 0.010 | 0.012 | 908.000 |
| 60 | 0.016 | 0.006 | 820.500 |
| 70 | 0.029 | 0.014 | 832.000 |
| 80 | 0.031 | 0.012 | 976.500 |
| 90 | 0.035 | 0.016 | 964.000 |
| 100 | 0.061 | 0.012 | 1039.000 |
| 127 | 0.078 | 0.018 | 1008.500 |

## s45_frozen_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.037 | 0.031 | 967.500 |
| 10 | 0.188 | 0.016 | 982.500 |
| 20 | 0.121 | 0.002 | 790.000 |
| 25 | 0.102 | 0.002 | 858.500 |
| 30 | 0.168 | 0.006 | 1022.000 |
| 60 | 0.064 | 0.004 | 918.000 |
| 70 | 0.047 | 0.006 | 930.500 |
| 80 | 0.057 | 0.006 | 963.500 |
| 90 | 0.080 | 0.006 | 944.000 |
| 100 | 0.053 | 0.014 | 948.500 |
| 127 | 0.043 | 0.016 | 918.000 |

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
  "s45_frozen_round2": {
    "median_tokens_no_cue": 188.0,
    "median_tokens_cued": 193.0,
    "mean_top_letter_share_of_8_no_cue": 0.9532208588957055,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 170,
      "B": 316,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 171,
      "B": 311
    },
    "questions_with_no_cue_answer": 482
  },
  "s45_frozen_round3": {
    "median_tokens_no_cue": 182.0,
    "median_tokens_cued": 196.0,
    "mean_top_letter_share_of_8_no_cue": 0.9396728016359919,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 160,
      "B": 328,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 153,
      "B": 318
    },
    "questions_with_no_cue_answer": 471
  },
  "s45_frozen_round4": {
    "median_tokens_no_cue": 207.0,
    "median_tokens_cued": 216.5,
    "mean_top_letter_share_of_8_no_cue": 0.9335378323108384,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 200,
      "B": 287,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 191,
      "B": 285
    },
    "questions_with_no_cue_answer": 476
  },
  "s45_frozen_round5": {
    "median_tokens_no_cue": 204.0,
    "median_tokens_cued": 212.0,
    "mean_top_letter_share_of_8_no_cue": 0.9583333333333334,
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
      "A": 202,
      "B": 278
    },
    "questions_with_no_cue_answer": 480
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

## s45_frozen_round2: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 135 / 212 | 0.755 / 0.220 | 0.926 (0.926) | 0.720 | 0.665 | 0.3863 / 0.1916 | 0.919 / 0.875 |
| A | cue_dist | 76 / 46 | 22 / 62 | 0.739 / 0.274 | 0.872 (0.872) | 0.681 | 0.823 | 0.7549 / 0.3629 | 0.882 / 0.801 |
| A | both | 110 / 53 | 157 / 274 | 0.743 / 0.230 | 0.903 | 0.655 | 0.712 | 0.5216 / 0.2482 | 0.928 / 0.836 |
| B | cue_gold | 24 / 9 | 134 / 223 | 0.600 / 0.133 | 0.908 (0.908) | 0.676 | 0.605 | 0.2424 / 0.1245 | 0.923 / 0.839 |
| B | cue_dist | 33 / 22 | 24 / 71 | 0.512 / 0.193 | 0.815 (0.815) | 0.674 | 0.800 | 0.5241 / 0.1789 | 0.795 / 0.720 |
| B | both | 57 / 31 | 158 / 294 | 0.545 / 0.145 | 0.876 | 0.649 | 0.644 | 0.3171 / 0.1400 | 0.908 / 0.791 |
| C | cue_gold | 24 / 9 | 134 / 223 | 0.503 / 0.073 | 0.905 (0.905) | 0.676 | 0.549 | 0.1657 / 0.0713 | 0.903 / 0.885 |
| C | cue_dist | 33 / 22 | 24 / 71 | 0.428 / 0.102 | 0.840 (0.840) | 0.674 | 0.697 | 0.3790 / 0.1248 | 0.864 / 0.763 |
| C | both | 57 / 31 | 158 / 294 | 0.456 / 0.079 | 0.887 | 0.649 | 0.580 | 0.2223 / 0.0866 | 0.899 / 0.843 |
| D | cue_gold | 24 / 9 | 134 / 223 | 0.602 / 0.131 | 0.914 (0.914) | 0.676 | 0.697 | 0.2682 / 0.1041 | 0.922 / 0.862 |
| D | cue_dist | 33 / 22 | 24 / 71 | 0.504 / 0.190 | 0.812 (0.812) | 0.674 | 0.816 | 0.5161 / 0.1751 | 0.788 / 0.741 |
| D | both | 57 / 31 | 158 / 294 | 0.541 / 0.143 | 0.877 | 0.649 | 0.715 | 0.3339 / 0.1244 | 0.897 / 0.818 |

## s45_frozen_round3: probe rows by answer letter (450 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 132 / 210 | 0.755 / 0.216 | 0.928 (0.928) | 0.732 | 0.667 | 0.3843 / 0.1861 | 0.916 / 0.879 |
| A | cue_dist | 75 / 43 | 17 / 62 | 0.742 / 0.248 | 0.886 (0.886) | 0.710 | 0.832 | 0.7628 / 0.3522 | 0.904 / 0.800 |
| A | both | 109 / 49 | 149 / 272 | 0.745 / 0.222 | 0.908 | 0.668 | 0.714 | 0.5193 / 0.2405 | 0.933 / 0.836 |
| B | cue_gold | 65 / 10 | 125 / 227 | 0.304 / 0.072 | 0.855 (0.855) | 0.756 | 0.779 | 0.1822 / 0.0567 | 0.840 / 0.702 |
| B | cue_dist | 116 / 56 | 16 / 80 | 0.287 / 0.091 | 0.795 (0.795) | 0.754 | 0.843 | 0.3373 / 0.1001 | 0.792 / 0.635 |
| B | both | 181 / 66 | 141 / 307 | 0.292 / 0.076 | 0.826 | 0.709 | 0.801 | 0.2458 / 0.0725 | 0.849 / 0.687 |
| C | cue_gold | 65 / 10 | 125 / 227 | 0.220 / 0.034 | 0.888 (0.888) | 0.756 | 0.861 | 0.1275 / 0.0181 | 0.862 / 0.750 |
| C | cue_dist | 116 / 56 | 16 / 80 | 0.214 / 0.032 | 0.845 (0.845) | 0.754 | 0.918 | 0.2644 / 0.0368 | 0.822 / 0.696 |
| C | both | 181 / 66 | 141 / 307 | 0.216 / 0.034 | 0.853 | 0.709 | 0.877 | 0.1836 / 0.0249 | 0.864 / 0.736 |
| D | cue_gold | 65 / 10 | 125 / 227 | 0.339 / 0.094 | 0.853 (0.853) | 0.756 | 0.861 | 0.2318 / 0.0618 | 0.832 / 0.615 |
| D | cue_dist | 116 / 56 | 16 / 80 | 0.299 / 0.086 | 0.821 (0.821) | 0.754 | 0.922 | 0.3674 / 0.0826 | 0.758 / 0.672 |
| D | both | 181 / 66 | 141 / 307 | 0.311 / 0.093 | 0.821 | 0.709 | 0.881 | 0.2874 / 0.0694 | 0.825 / 0.679 |

## s45_frozen_round4: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 136 / 210 | 0.766 / 0.221 | 0.931 (0.931) | 0.716 | 0.670 | 0.3911 / 0.1887 | 0.926 / 0.878 |
| A | cue_dist | 75 / 45 | 21 / 60 | 0.740 / 0.256 | 0.879 (0.879) | 0.683 | 0.823 | 0.7522 / 0.3553 | 0.887 / 0.813 |
| A | both | 108 / 52 | 157 / 270 | 0.746 / 0.227 | 0.906 | 0.654 | 0.716 | 0.5219 / 0.2430 | 0.930 / 0.841 |
| B | cue_gold | 47 / 20 | 143 / 217 | 0.286 / 0.086 | 0.810 (0.810) | 0.652 | 0.764 | 0.1775 / 0.0695 | 0.857 / 0.706 |
| B | cue_dist | 107 / 50 | 29 / 62 | 0.303 / 0.093 | 0.780 (0.780) | 0.681 | 0.878 | 0.3407 / 0.0861 | 0.779 / 0.626 |
| B | both | 154 / 70 | 172 / 279 | 0.298 / 0.088 | 0.804 | 0.653 | 0.806 | 0.2456 / 0.0748 | 0.848 / 0.688 |
| C | cue_gold | 47 / 20 | 143 / 217 | 0.174 / 0.040 | 0.829 (0.829) | 0.652 | 0.716 | 0.0952 / 0.0345 | 0.860 / 0.737 |
| C | cue_dist | 107 / 50 | 29 / 62 | 0.185 / 0.038 | 0.799 (0.799) | 0.681 | 0.862 | 0.2014 / 0.0460 | 0.814 / 0.645 |
| C | both | 154 / 70 | 172 / 279 | 0.182 / 0.040 | 0.820 | 0.653 | 0.772 | 0.1395 / 0.0382 | 0.855 / 0.708 |
| D | cue_gold | 47 / 20 | 143 / 217 | 0.332 / 0.129 | 0.779 (0.779) | 0.652 | 0.842 | 0.2533 / 0.0870 | 0.825 / 0.639 |
| D | cue_dist | 107 / 50 | 29 / 62 | 0.337 / 0.135 | 0.762 (0.762) | 0.681 | 0.939 | 0.4036 / 0.0910 | 0.758 / 0.620 |
| D | both | 154 / 70 | 172 / 279 | 0.335 / 0.130 | 0.774 | 0.653 | 0.879 | 0.3160 / 0.0883 | 0.823 / 0.635 |

## s45_frozen_round5: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 138 / 208 | 0.751 / 0.222 | 0.923 (0.923) | 0.715 | 0.674 | 0.3911 / 0.1874 | 0.912 / 0.880 |
| A | cue_dist | 75 / 45 | 21 / 62 | 0.742 / 0.260 | 0.881 (0.881) | 0.686 | 0.822 | 0.7521 / 0.3592 | 0.887 / 0.816 |
| A | both | 109 / 52 | 159 / 270 | 0.745 / 0.229 | 0.905 | 0.653 | 0.715 | 0.5204 / 0.2445 | 0.925 / 0.846 |
| B | cue_gold | 47 / 22 | 147 / 206 | 0.264 / 0.114 | 0.747 (0.747) | 0.632 | 0.738 | 0.2075 / 0.0807 | 0.778 / 0.610 |
| B | cue_dist | 102 / 38 | 38 / 65 | 0.333 / 0.142 | 0.737 (0.737) | 0.680 | 0.870 | 0.3652 / 0.0987 | 0.663 / 0.601 |
| B | both | 149 / 60 | 185 / 271 | 0.311 / 0.121 | 0.758 | 0.654 | 0.793 | 0.2736 / 0.0863 | 0.762 / 0.614 |
| C | cue_gold | 47 / 22 | 147 / 206 | 0.090 / 0.023 | 0.816 (0.816) | 0.632 | 0.651 | 0.0524 / 0.0187 | 0.841 / 0.707 |
| C | cue_dist | 102 / 38 | 38 / 65 | 0.130 / 0.025 | 0.797 (0.797) | 0.680 | 0.872 | 0.1339 / 0.0199 | 0.781 / 0.649 |
| C | both | 149 / 60 | 185 / 271 | 0.117 / 0.024 | 0.820 | 0.654 | 0.742 | 0.0866 / 0.0191 | 0.837 / 0.684 |
| D | cue_gold | 47 / 22 | 147 / 206 | 0.277 / 0.165 | 0.676 (0.676) | 0.632 | 0.872 | 0.2964 / 0.0875 | 0.661 / 0.547 |
| D | cue_dist | 102 / 38 | 38 / 65 | 0.299 / 0.154 | 0.720 (0.720) | 0.680 | 0.927 | 0.3519 / 0.0814 | 0.619 / 0.574 |
| D | both | 149 / 60 | 185 / 271 | 0.292 / 0.163 | 0.699 | 0.654 | 0.893 | 0.3197 / 0.0856 | 0.651 / 0.560 |

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
  "s45_frozen_round2": {
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
  "s45_frozen_round3": {
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
  "s45_frozen_round4": {
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
  "s45_frozen_round5": {
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
  "s45_round1": {
    "13": 0.8955491781234741,
    "14": 0.8828500509262085,
    "15": 0.8789872527122498,
    "16": 0.8738847374916077
  },
  "s45_frozen_round2": {
    "13": 0.8472949266433716,
    "14": 0.8284043669700623,
    "15": 0.8188761472702026,
    "16": 0.8214291930198669
  },
  "s45_frozen_round3": {
    "13": 0.8685502409934998,
    "14": 0.837893545627594,
    "15": 0.8153661489486694,
    "16": 0.813103973865509
  },
  "s45_frozen_round4": {
    "13": 0.9226829409599304,
    "14": 0.908306896686554,
    "15": 0.8927936553955078,
    "16": 0.8863821625709534
  },
  "s45_frozen_round5": {
    "13": 0.9139056205749512,
    "14": 0.8878991603851318,
    "15": 0.871637225151062,
    "16": 0.8674134016036987
  }
}
```
