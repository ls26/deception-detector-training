# Training dynamics and probe rows: kl_no_penalty_s43, s43_no_penalty_round2, s43_no_penalty_round3, s43_no_penalty_round4, s43_no_penalty_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_no_penalty_s43 0.3, s43_no_penalty_round2 0.3, s43_no_penalty_round3 0.3, s43_no_penalty_round4 0.3, s43_no_penalty_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_no_penalty_s43 | 41.700 | 49.650 | 46.833 | 48.662 |
| groups where all 8 answers name the cue (of 64) | s43_no_penalty_round2 | 48.800 | 46.950 | 49.733 | 48.500 |
| groups where all 8 answers name the cue (of 64) | s43_no_penalty_round3 | 48.500 | 46.700 | 47.833 | 47.353 |
| groups where all 8 answers name the cue (of 64) | s43_no_penalty_round4 | 47.800 | 43.500 | 46.333 | 47.691 |
| groups where all 8 answers name the cue (of 64) | s43_no_penalty_round5 | 47.800 | 46.450 | 46.500 | 46.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_no_penalty_s43 | 18.600 | 11.750 | 13.800 | 11.015 |
| groups with mixed task rewards (of 64) | s43_no_penalty_round2 | 10.500 | 10.900 | 7.667 | 7.235 |
| groups with mixed task rewards (of 64) | s43_no_penalty_round3 | 6.600 | 8.050 | 5.333 | 5.809 |
| groups with mixed task rewards (of 64) | s43_no_penalty_round4 | 4.500 | 4.100 | 5.300 | 3.044 |
| groups with mixed task rewards (of 64) | s43_no_penalty_round5 | 2.100 | 2.800 | 2.400 | 2.206 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_no_penalty_s43 | 5.400 | 5.350 | 6.067 | 4.941 |
|   of which the cue is not at the training label | s43_no_penalty_round2 | 5.200 | 4.950 | 3.733 | 3.221 |
|   of which the cue is not at the training label | s43_no_penalty_round3 | 3.300 | 3.950 | 2.500 | 3.118 |
|   of which the cue is not at the training label | s43_no_penalty_round4 | 1.800 | 1.650 | 2.567 | 1.706 |
|   of which the cue is not at the training label | s43_no_penalty_round5 | 1 | 1.350 | 1.300 | 1.015 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_no_penalty_s43 | 84.200 | 56.400 | 67.367 | 67.441 |
| answers not naming the cue (of 512) | s43_no_penalty_round2 | 71.700 | 82.700 | 77.233 | 88.750 |
| answers not naming the cue (of 512) | s43_no_penalty_round3 | 91.900 | 102.600 | 104.400 | 107.074 |
| answers not naming the cue (of 512) | s43_no_penalty_round4 | 106.100 | 145.450 | 117.800 | 118.029 |
| answers not naming the cue (of 512) | s43_no_penalty_round5 | 122 | 128.700 | 129.700 | 127.721 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_no_penalty_s43 | 0.222 | 0.138 | 0.161 | 0.128 |
| mean absolute advantage per answer | s43_no_penalty_round2 | 0.123 | 0.128 | 0.089 | 0.083 |
| mean absolute advantage per answer | s43_no_penalty_round3 | 0.074 | 0.095 | 0.061 | 0.067 |
| mean absolute advantage per answer | s43_no_penalty_round4 | 0.048 | 0.049 | 0.062 | 0.035 |
| mean absolute advantage per answer | s43_no_penalty_round5 | 0.024 | 0.031 | 0.028 | 0.024 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_no_penalty_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s43_no_penalty_round2 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s43_no_penalty_round3 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s43_no_penalty_round4 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s43_no_penalty_round5 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_no_penalty_s43 | 0.046 | 0.005 | 0.009 | 0.006 |
| advantage on cue-naming answers, per answer | s43_no_penalty_round2 | 0.002 | 0.006 | 0.001 | 0.004 |
| advantage on cue-naming answers, per answer | s43_no_penalty_round3 | -0.001 | 0.001 | 0.002 | -0.002 |
| advantage on cue-naming answers, per answer | s43_no_penalty_round4 | 0.004 | 0.006 | 0.000 | -0.002 |
| advantage on cue-naming answers, per answer | s43_no_penalty_round5 | 0.001 | 0.001 | -0.001 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_no_penalty_s43 | 0.077 | 0.058 | 0.055 | 0.047 |
| gradient norm (W&B, before clipping) | s43_no_penalty_round2 | 0.045 | 0.052 | 0.041 | 0.040 |
| gradient norm (W&B, before clipping) | s43_no_penalty_round3 | 0.033 | 0.041 | 0.034 | 0.036 |
| gradient norm (W&B, before clipping) | s43_no_penalty_round4 | 0.028 | 0.031 | 0.033 | 0.024 |
| gradient norm (W&B, before clipping) | s43_no_penalty_round5 | 0.018 | 0.021 | 0.023 | 0.020 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_no_penalty_s43 | 0.030 | 0.045 | 0.036 | 0.028 |
| KL from the base model (W&B) | s43_no_penalty_round2 | 0.027 | 0.040 | 0.045 | 0.037 |
| KL from the base model (W&B) | s43_no_penalty_round3 | 0.037 | 0.045 | 0.046 | 0.042 |
| KL from the base model (W&B) | s43_no_penalty_round4 | 0.040 | 0.046 | 0.046 | 0.042 |
| KL from the base model (W&B) | s43_no_penalty_round5 | 0.040 | 0.047 | 0.046 | 0.041 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_no_penalty_s43 | 0.444 | 0.462 | 0.474 | 0.478 |
| token entropy (W&B) | s43_no_penalty_round2 | 0.494 | 0.486 | 0.476 | 0.491 |
| token entropy (W&B) | s43_no_penalty_round3 | 0.500 | 0.536 | 0.481 | 0.495 |
| token entropy (W&B) | s43_no_penalty_round4 | 0.510 | 0.476 | 0.498 | 0.509 |
| token entropy (W&B) | s43_no_penalty_round5 | 0.503 | 0.527 | 0.509 | 0.518 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_no_penalty_s43 | 219.309 | 233.173 | 255.046 | 251.709 |
| answer length, tokens (W&B) | s43_no_penalty_round2 | 245.125 | 219.022 | 249.958 | 222.403 |
| answer length, tokens (W&B) | s43_no_penalty_round3 | 232.360 | 241.027 | 233.032 | 239.923 |
| answer length, tokens (W&B) | s43_no_penalty_round4 | 239.531 | 252.952 | 240.993 | 234.730 |
| answer length, tokens (W&B) | s43_no_penalty_round5 | 233.660 | 245.103 | 234.723 | 238.388 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of kl_no_penalty_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 149 | 0.089 | 941.901 / 1009.493 | 0.168 / 0.280 | 0.237 / 0.337 | 0.455 / 0.534 |
| 5-9 | 268 | 0.103 | 812.400 / 834.079 | 0.205 / 0.214 | 0.258 / 0.294 | 0.312 / 0.331 |
| 10-29 | 993 | 0.105 | 980.322 / 1020.859 | 0.201 / 0.251 | 0.285 / 0.333 | 0.453 / 0.498 |
| 30-59 | 1405 | 0.098 | 1069.704 / 1133.701 | 0.212 / 0.272 | 0.277 / 0.341 | 0.542 / 0.603 |
| 60-127 | 3309 | 0.081 | 1058.231 / 1135.947 | 0.210 / 0.292 | 0.293 / 0.374 | 0.511 / 0.572 |

## Probe-only groups of s43_no_penalty_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 235 | 0.067 | 1036.386 / 1102.706 | 0.217 / 0.237 | 0.277 / 0.327 | 0.499 / 0.545 |
| 5-9 | 253 | 0.070 | 1026.359 / 1083.345 | 0.188 / 0.231 | 0.249 / 0.319 | 0.498 / 0.562 |
| 10-29 | 939 | 0.061 | 912.653 / 977.473 | 0.150 / 0.230 | 0.207 / 0.288 | 0.427 / 0.485 |
| 30-59 | 1492 | 0.047 | 1056.108 / 1121.035 | 0.183 / 0.272 | 0.264 / 0.359 | 0.516 / 0.578 |
| 60-127 | 3298 | 0.034 | 938.148 / 1001.998 | 0.143 / 0.234 | 0.187 / 0.287 | 0.428 / 0.487 |

## Probe-only groups of s43_no_penalty_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 238 | 0.033 | 976.978 / 1028.182 | 0.137 / 0.237 | 0.182 / 0.305 | 0.412 / 0.496 |
| 5-9 | 247 | 0.047 | 976.777 / 1048.981 | 0.124 / 0.236 | 0.169 / 0.276 | 0.470 / 0.533 |
| 10-29 | 934 | 0.036 | 1020.595 / 1089.580 | 0.139 / 0.210 | 0.213 / 0.304 | 0.482 / 0.555 |
| 30-59 | 1435 | 0.028 | 963.503 / 1039.856 | 0.140 / 0.251 | 0.204 / 0.327 | 0.424 / 0.508 |
| 60-127 | 3220 | 0.027 | 990.041 / 1068.380 | 0.143 / 0.246 | 0.227 / 0.328 | 0.502 / 0.572 |

## Probe-only groups of s43_no_penalty_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 233 | 0.022 | 999.161 / 1069.550 | 0.142 / 0.246 | 0.215 / 0.326 | 0.470 / 0.552 |
| 5-9 | 245 | 0.028 | 991.358 / 1074.401 | 0.158 / 0.233 | 0.235 / 0.348 | 0.523 / 0.598 |
| 10-29 | 870 | 0.015 | 1036.400 / 1131.421 | 0.147 / 0.252 | 0.219 / 0.334 | 0.494 / 0.585 |
| 30-59 | 1390 | 0.018 | 991.515 / 1081.593 | 0.144 / 0.263 | 0.242 / 0.362 | 0.471 / 0.561 |
| 60-127 | 3243 | 0.013 | 979.548 / 1069.568 | 0.133 / 0.237 | 0.207 / 0.330 | 0.449 / 0.536 |

## Probe-only groups of s43_no_penalty_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 237 | 0.015 | 987.485 / 1093.184 | 0.123 / 0.230 | 0.195 / 0.320 | 0.478 / 0.558 |
| 5-9 | 241 | 0.020 | 985.830 / 1073.189 | 0.150 / 0.234 | 0.205 / 0.318 | 0.435 / 0.539 |
| 10-29 | 929 | 0.015 | 1020.185 / 1117.332 | 0.161 / 0.286 | 0.251 / 0.384 | 0.479 / 0.559 |
| 30-59 | 1395 | 0.030 | 979.880 / 1069.817 | 0.146 / 0.269 | 0.207 / 0.347 | 0.456 / 0.546 |
| 60-127 | 3181 | 0.035 | 989.913 / 1094.464 | 0.144 / 0.290 | 0.212 / 0.367 | 0.474 / 0.561 |

## kl_no_penalty_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.139 | 0.016 | 992.000 |
| 10 | 0.088 | 0.014 | 726.000 |
| 20 | 0.115 | 0.012 | 1028.500 |
| 25 | 0.275 | 0.012 | 1034.000 |
| 30 | 0.248 | 0.023 | 1029.000 |
| 60 | 0.027 | 0.027 | 1032.000 |
| 70 | 0.072 | 0.029 | 1132.500 |
| 80 | 0.047 | 0.018 | 1093.500 |
| 90 | 0.076 | 0.012 | 1014.500 |
| 100 | 0.039 | 0.018 | 1060.000 |
| 127 | 0.064 | 0.021 | 1031.500 |

## s43_no_penalty_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.062 | 0.008 | 1072.500 |
| 10 | 0.016 | 0.039 | 945.500 |
| 20 | 0.074 | 0.021 | 923.000 |
| 25 | 0.062 | 0.027 | 819.000 |
| 30 | 0.080 | 0.008 | 918.500 |
| 60 | 0.082 | 0.014 | 1023.000 |
| 70 | 0.076 | 0.016 | 931.000 |
| 80 | 0.078 | 0.004 | 892.000 |
| 90 | 0.127 | 0.006 | 920.500 |
| 100 | 0.154 | 0.010 | 894.000 |
| 127 | 0.191 | 0.010 | 1005.500 |

## s43_no_penalty_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.121 | 0.004 | 971.000 |
| 10 | 0.027 | 0.008 | 965.500 |
| 20 | 0.045 | 0.008 | 1010.000 |
| 25 | 0.023 | 0.006 | 1030.000 |
| 30 | 0.031 | 0.004 | 1060.500 |
| 60 | 0.178 | 0.006 | 1022.500 |
| 70 | 0.285 | 0.029 | 978.000 |
| 80 | 0.154 | 0.012 | 962.500 |
| 90 | 0.205 | 0.012 | 999.000 |
| 100 | 0.186 | 0.021 | 971.500 |
| 127 | 0.221 | 0.029 | 958.000 |

## s43_no_penalty_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.170 | 0.010 | 1011.500 |
| 10 | 0.109 | 0.010 | 1023.500 |
| 20 | 0.270 | 0.016 | 1046.000 |
| 25 | 0.236 | 0.012 | 992.000 |
| 30 | 0.162 | 0.021 | 1006.500 |
| 60 | 0.033 | 0.012 | 1022.000 |
| 70 | 0.023 | 0.008 | 993.500 |
| 80 | 0.045 | 0.006 | 1002.500 |
| 90 | 0.037 | 0.023 | 1030.500 |
| 100 | 0.057 | 0.006 | 1031.500 |
| 127 | 0.064 | 0.006 | 1001.500 |

## s43_no_penalty_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.076 | 0.004 | 1005.000 |
| 10 | 0.137 | 0.012 | 963.000 |
| 20 | 0.057 | 0.029 | 1042.000 |
| 25 | 0.051 | 0.053 | 1141.000 |
| 30 | 0.055 | 0.020 | 1188.500 |
| 60 | 0.105 | 0.006 | 992.000 |
| 70 | 0.139 | 0.010 | 1047.500 |
| 80 | 0.219 | 0.010 | 1043.500 |
| 90 | 0.135 | 0.002 | 966.000 |
| 100 | 0.170 | 0.008 | 1029.000 |
| 127 | 0.113 | 0.014 | 1040.000 |

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
  "kl_no_penalty_s43": {
    "median_tokens_no_cue": 203.0,
    "median_tokens_cued": 218.0,
    "mean_top_letter_share_of_8_no_cue": 0.9322597137014315,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 170,
      "B": 317,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 171,
      "B": 309
    },
    "questions_with_no_cue_answer": 480
  },
  "s43_no_penalty_round2": {
    "median_tokens_no_cue": 198.0,
    "median_tokens_cued": 212.0,
    "mean_top_letter_share_of_8_no_cue": 0.9309815950920245,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 169,
      "B": 317,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 163,
      "B": 307
    },
    "questions_with_no_cue_answer": 470
  },
  "s43_no_penalty_round3": {
    "median_tokens_no_cue": 202.0,
    "median_tokens_cued": 216.5,
    "mean_top_letter_share_of_8_no_cue": 0.945040899795501,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 172,
      "B": 317,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 164,
      "B": 313
    },
    "questions_with_no_cue_answer": 477
  },
  "s43_no_penalty_round4": {
    "median_tokens_no_cue": 202.0,
    "median_tokens_cued": 216.0,
    "mean_top_letter_share_of_8_no_cue": 0.9542433537832311,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 194,
      "B": 295,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 186,
      "B": 294
    },
    "questions_with_no_cue_answer": 480
  },
  "s43_no_penalty_round5": {
    "median_tokens_no_cue": 201.0,
    "median_tokens_cued": 217.5,
    "mean_top_letter_share_of_8_no_cue": 0.9680470347648262,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 180,
      "B": 309,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 176,
      "B": 309
    },
    "questions_with_no_cue_answer": 485
  }
}
```

## kl_no_penalty_s43: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 133 / 211 | 0.753 / 0.219 | 0.927 (0.927) | 0.725 | 0.670 | 0.3920 / 0.1897 | 0.916 / 0.878 |
| A | cue_dist | 75 / 42 | 22 / 64 | 0.740 / 0.272 | 0.871 (0.871) | 0.693 | 0.828 | 0.7532 / 0.3481 | 0.882 / 0.788 |
| A | both | 111 / 49 | 155 / 275 | 0.743 / 0.229 | 0.903 | 0.667 | 0.718 | 0.5238 / 0.2415 | 0.926 / 0.829 |
| B | cue_gold | 55 / 19 | 136 / 223 | 0.728 / 0.205 | 0.920 (0.920) | 0.682 | 0.666 | 0.4090 / 0.2032 | 0.891 / 0.943 |
| B | cue_dist | 123 / 70 | 25 / 73 | 0.715 / 0.312 | 0.841 (0.841) | 0.691 | 0.808 | 0.7532 / 0.3998 | 0.794 / 0.794 |
| B | both | 178 / 89 | 161 / 296 | 0.719 / 0.228 | 0.898 | 0.657 | 0.720 | 0.5592 / 0.2762 | 0.890 / 0.872 |
| C | cue_gold | 55 / 19 | 136 / 223 | 0.656 / 0.151 | 0.926 (0.926) | 0.682 | 0.649 | 0.3408 / 0.1553 | 0.896 / 0.952 |
| C | cue_dist | 123 / 70 | 25 / 73 | 0.652 / 0.236 | 0.857 (0.857) | 0.691 | 0.807 | 0.6903 / 0.3274 | 0.834 / 0.803 |
| C | both | 178 / 89 | 161 / 296 | 0.653 / 0.169 | 0.905 | 0.657 | 0.711 | 0.4934 / 0.2192 | 0.901 / 0.877 |
| D | cue_gold | 55 / 19 | 136 / 223 | 0.712 / 0.200 | 0.920 (0.920) | 0.682 | 0.676 | 0.4086 / 0.1921 | 0.890 / 0.941 |
| D | cue_dist | 123 / 70 | 25 / 73 | 0.700 / 0.296 | 0.846 (0.846) | 0.691 | 0.823 | 0.7483 / 0.3739 | 0.812 / 0.797 |
| D | both | 178 / 89 | 161 / 296 | 0.704 / 0.221 | 0.897 | 0.657 | 0.733 | 0.5569 / 0.2597 | 0.892 / 0.869 |

## s43_no_penalty_round2: probe rows by answer letter (450 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 130 / 210 | 0.768 / 0.218 | 0.932 (0.932) | 0.721 | 0.664 | 0.3894 / 0.1910 | 0.928 / 0.876 |
| A | cue_dist | 75 / 42 | 21 / 61 | 0.741 / 0.252 | 0.881 (0.881) | 0.692 | 0.831 | 0.7536 / 0.3401 | 0.888 / 0.803 |
| A | both | 108 / 49 | 151 / 271 | 0.748 / 0.225 | 0.906 | 0.665 | 0.718 | 0.5244 / 0.2390 | 0.932 / 0.832 |
| B | cue_gold | 47 / 18 | 128 / 221 | 0.734 / 0.178 | 0.933 (0.933) | 0.678 | 0.650 | 0.3676 / 0.1905 | 0.939 / 0.897 |
| B | cue_dist | 79 / 44 | 25 / 73 | 0.677 / 0.268 | 0.840 (0.840) | 0.694 | 0.825 | 0.7035 / 0.3105 | 0.802 / 0.797 |
| B | both | 126 / 62 | 153 / 294 | 0.696 / 0.198 | 0.900 | 0.664 | 0.705 | 0.4928 / 0.2300 | 0.912 / 0.852 |
| C | cue_gold | 47 / 18 | 128 / 221 | 0.597 / 0.100 | 0.928 (0.928) | 0.678 | 0.548 | 0.2317 / 0.1383 | 0.936 / 0.910 |
| C | cue_dist | 79 / 44 | 25 / 73 | 0.565 / 0.154 | 0.868 (0.868) | 0.694 | 0.739 | 0.5334 / 0.2483 | 0.859 / 0.833 |
| C | both | 126 / 62 | 153 / 294 | 0.576 / 0.112 | 0.910 | 0.664 | 0.617 | 0.3442 / 0.1744 | 0.926 / 0.875 |
| D | cue_gold | 47 / 18 | 128 / 221 | 0.708 / 0.200 | 0.927 (0.927) | 0.678 | 0.698 | 0.4000 / 0.1922 | 0.923 / 0.903 |
| D | cue_dist | 79 / 44 | 25 / 73 | 0.654 / 0.261 | 0.839 (0.839) | 0.694 | 0.853 | 0.6976 / 0.2859 | 0.790 / 0.784 |
| D | both | 126 / 62 | 153 / 294 | 0.672 / 0.214 | 0.889 | 0.664 | 0.748 | 0.5109 / 0.2230 | 0.897 / 0.839 |

## s43_no_penalty_round3: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 133 / 209 | 0.763 / 0.221 | 0.930 (0.930) | 0.722 | 0.669 | 0.3930 / 0.1926 | 0.925 / 0.874 |
| A | cue_dist | 75 / 43 | 22 / 63 | 0.737 / 0.270 | 0.871 (0.871) | 0.688 | 0.829 | 0.7529 / 0.3485 | 0.881 / 0.791 |
| A | both | 110 / 50 | 155 / 272 | 0.744 / 0.231 | 0.903 | 0.662 | 0.717 | 0.5248 / 0.2439 | 0.929 / 0.829 |
| B | cue_gold | 34 / 10 | 134 / 223 | 0.687 / 0.209 | 0.895 (0.895) | 0.699 | 0.666 | 0.3761 / 0.1787 | 0.876 / 0.857 |
| B | cue_dist | 54 / 37 | 22 / 74 | 0.676 / 0.325 | 0.791 (0.791) | 0.682 | 0.775 | 0.6863 / 0.3652 | 0.719 / 0.732 |
| B | both | 88 / 47 | 156 / 297 | 0.679 / 0.234 | 0.864 | 0.654 | 0.690 | 0.4727 / 0.2389 | 0.861 / 0.806 |
| C | cue_gold | 34 / 10 | 134 / 223 | 0.499 / 0.089 | 0.923 (0.923) | 0.699 | 0.599 | 0.2057 / 0.0823 | 0.914 / 0.874 |
| C | cue_dist | 54 / 37 | 22 / 74 | 0.485 / 0.146 | 0.843 (0.843) | 0.682 | 0.748 | 0.4578 / 0.2100 | 0.823 / 0.779 |
| C | both | 88 / 47 | 156 / 297 | 0.489 / 0.101 | 0.896 | 0.654 | 0.639 | 0.2842 / 0.1235 | 0.901 / 0.840 |
| D | cue_gold | 34 / 10 | 134 / 223 | 0.702 / 0.288 | 0.861 (0.861) | 0.699 | 0.747 | 0.4825 / 0.2264 | 0.817 / 0.835 |
| D | cue_dist | 54 / 37 | 22 / 74 | 0.617 / 0.365 | 0.738 (0.738) | 0.682 | 0.862 | 0.7140 / 0.3323 | 0.646 / 0.636 |
| D | both | 88 / 47 | 156 / 297 | 0.644 / 0.305 | 0.804 | 0.654 | 0.772 | 0.5546 / 0.2606 | 0.797 / 0.721 |

## s43_no_penalty_round4: probe rows by answer letter (457 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 136 / 208 | 0.768 / 0.220 | 0.933 (0.933) | 0.729 | 0.671 | 0.3914 / 0.1887 | 0.917 / 0.899 |
| A | cue_dist | 74 / 44 | 21 / 63 | 0.743 / 0.266 | 0.875 (0.875) | 0.689 | 0.825 | 0.7560 / 0.3565 | 0.879 / 0.804 |
| A | both | 109 / 50 | 157 / 271 | 0.749 / 0.229 | 0.906 | 0.659 | 0.714 | 0.5216 / 0.2446 | 0.926 / 0.841 |
| B | cue_gold | 24 / 11 | 140 / 210 | 0.557 / 0.223 | 0.781 (0.781) | 0.643 | 0.636 | 0.3468 / 0.1842 | 0.817 / 0.656 |
| B | cue_dist | 35 / 25 | 33 / 67 | 0.680 / 0.370 | 0.757 (0.757) | 0.627 | 0.816 | 0.7124 / 0.3195 | 0.726 / 0.718 |
| B | both | 59 / 36 | 173 / 277 | 0.635 / 0.256 | 0.811 | 0.618 | 0.676 | 0.4540 / 0.2239 | 0.833 / 0.740 |
| C | cue_gold | 24 / 11 | 140 / 210 | 0.422 / 0.092 | 0.849 (0.849) | 0.643 | 0.500 | 0.1658 / 0.0889 | 0.879 / 0.778 |
| C | cue_dist | 35 / 25 | 33 / 67 | 0.483 / 0.185 | 0.810 (0.810) | 0.627 | 0.678 | 0.4175 / 0.2076 | 0.787 / 0.794 |
| C | both | 59 / 36 | 173 / 277 | 0.461 / 0.112 | 0.862 | 0.618 | 0.550 | 0.2396 / 0.1238 | 0.873 / 0.828 |
| D | cue_gold | 24 / 11 | 140 / 210 | 0.571 / 0.315 | 0.720 (0.720) | 0.643 | 0.756 | 0.4876 / 0.2272 | 0.728 / 0.588 |
| D | cue_dist | 35 / 25 | 33 / 67 | 0.645 / 0.370 | 0.745 (0.745) | 0.627 | 0.859 | 0.7031 / 0.3027 | 0.660 / 0.732 |
| D | both | 59 / 36 | 173 / 277 | 0.618 / 0.327 | 0.755 | 0.618 | 0.781 | 0.5508 / 0.2494 | 0.744 / 0.696 |

## s43_no_penalty_round5: probe rows by answer letter (462 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 138 / 210 | 0.756 / 0.225 | 0.923 (0.923) | 0.718 | 0.677 | 0.3980 / 0.1890 | 0.912 / 0.879 |
| A | cue_dist | 74 / 46 | 22 / 63 | 0.736 / 0.268 | 0.872 (0.872) | 0.679 | 0.822 | 0.7516 / 0.3578 | 0.881 / 0.805 |
| A | both | 109 / 53 | 160 / 273 | 0.741 / 0.233 | 0.901 | 0.652 | 0.717 | 0.5242 / 0.2454 | 0.923 / 0.839 |
| B | cue_gold | 24 / 7 | 130 / 207 | 0.495 / 0.223 | 0.745 (0.745) | 0.694 | 0.638 | 0.3361 / 0.1814 | 0.711 / 0.673 |
| B | cue_dist | 14 / 9 | 35 / 72 | 0.606 / 0.386 | 0.697 (0.697) | 0.641 | 0.833 | 0.6721 / 0.2756 | 0.596 / 0.636 |
| B | both | 38 / 16 | 165 / 279 | 0.542 / 0.263 | 0.750 | 0.666 | 0.672 | 0.4172 / 0.2073 | 0.708 / 0.687 |
| C | cue_gold | 24 / 7 | 130 / 207 | 0.447 / 0.140 | 0.824 (0.824) | 0.694 | 0.620 | 0.2415 / 0.1114 | 0.790 / 0.818 |
| C | cue_dist | 14 / 9 | 35 / 72 | 0.460 / 0.231 | 0.708 (0.708) | 0.641 | 0.798 | 0.4631 / 0.1554 | 0.647 / 0.688 |
| C | both | 38 / 16 | 165 / 279 | 0.452 / 0.162 | 0.795 | 0.666 | 0.655 | 0.2950 / 0.1235 | 0.761 / 0.773 |
| D | cue_gold | 24 / 7 | 130 / 207 | 0.487 / 0.330 | 0.646 (0.646) | 0.694 | 0.720 | 0.4686 / 0.2524 | 0.559 / 0.581 |
| D | cue_dist | 14 / 9 | 35 / 72 | 0.570 / 0.416 | 0.636 (0.636) | 0.641 | 0.866 | 0.6909 / 0.2935 | 0.539 / 0.489 |
| D | both | 38 / 16 | 165 / 279 | 0.522 / 0.350 | 0.657 | 0.666 | 0.749 | 0.5222 / 0.2637 | 0.591 / 0.555 |

## Probe data: answers naming A

```
{
  "kl_no_penalty_s43": {
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
  "s43_no_penalty_round2": {
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
  "s43_no_penalty_round3": {
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
  "s43_no_penalty_round4": {
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
  "s43_no_penalty_round5": {
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
  "kl_no_penalty_s43": {
    "13": 0.9328340888023376,
    "14": 0.9212709069252014,
    "15": 0.9126467108726501,
    "16": 0.9197190999984741
  },
  "s43_no_penalty_round2": {
    "13": 0.890733540058136,
    "14": 0.871867835521698,
    "15": 0.8738700151443481,
    "16": 0.8775339722633362
  },
  "s43_no_penalty_round3": {
    "13": 0.9160726070404053,
    "14": 0.9035874605178833,
    "15": 0.8848505616188049,
    "16": 0.8823835849761963
  },
  "s43_no_penalty_round4": {
    "13": 0.9365114569664001,
    "14": 0.9164382815361023,
    "15": 0.9039730429649353,
    "16": 0.8946384191513062
  },
  "s43_no_penalty_round5": {
    "13": 0.9450057744979858,
    "14": 0.9316654205322266,
    "15": 0.9157408475875854,
    "16": 0.9094304442405701
  }
}
```
