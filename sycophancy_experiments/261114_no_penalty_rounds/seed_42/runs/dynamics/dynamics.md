# Training dynamics and probe rows: kl_no_penalty_s42, s42_no_penalty_round2, s42_no_penalty_round3, s42_no_penalty_round4, s42_no_penalty_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_no_penalty_s42 0.3, s42_no_penalty_round2 0.3, s42_no_penalty_round3 0.3, s42_no_penalty_round4 0.3, s42_no_penalty_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_no_penalty_s42 | 38.300 | 48.150 | 50.400 | 57.853 |
| groups where all 8 answers name the cue (of 64) | s42_no_penalty_round2 | 55.700 | 44.750 | 46.900 | 48.676 |
| groups where all 8 answers name the cue (of 64) | s42_no_penalty_round3 | 50 | 48.800 | 51.200 | 49.456 |
| groups where all 8 answers name the cue (of 64) | s42_no_penalty_round4 | 49.100 | 42.500 | 46.500 | 45.971 |
| groups where all 8 answers name the cue (of 64) | s42_no_penalty_round5 | 47.800 | 43.850 | 46 | 46.882 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_no_penalty_s42 | 21.800 | 13.800 | 11.267 | 5.529 |
| groups with mixed task rewards (of 64) | s42_no_penalty_round2 | 7.100 | 14.850 | 11.700 | 7.985 |
| groups with mixed task rewards (of 64) | s42_no_penalty_round3 | 6 | 4.600 | 4.967 | 4.868 |
| groups with mixed task rewards (of 64) | s42_no_penalty_round4 | 5 | 7.700 | 4.867 | 3.250 |
| groups with mixed task rewards (of 64) | s42_no_penalty_round5 | 3.300 | 4.250 | 2.500 | 1.632 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_no_penalty_s42 | 6 | 6.500 | 4.567 | 2.118 |
|   of which the cue is not at the training label | s42_no_penalty_round2 | 2.900 | 6.250 | 5.133 | 3.618 |
|   of which the cue is not at the training label | s42_no_penalty_round3 | 2.500 | 2.150 | 2.733 | 2.397 |
|   of which the cue is not at the training label | s42_no_penalty_round4 | 3.100 | 3.050 | 2.400 | 1.397 |
|   of which the cue is not at the training label | s42_no_penalty_round5 | 1.300 | 1.800 | 1.333 | 0.750 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_no_penalty_s42 | 94.700 | 55.900 | 52.100 | 15.985 |
| answers not naming the cue (of 512) | s42_no_penalty_round2 | 26.600 | 79.450 | 80.733 | 84.147 |
| answers not naming the cue (of 512) | s42_no_penalty_round3 | 83.500 | 97.900 | 78.833 | 92.441 |
| answers not naming the cue (of 512) | s42_no_penalty_round4 | 97.800 | 135.250 | 118.233 | 130.735 |
| answers not naming the cue (of 512) | s42_no_penalty_round5 | 111.700 | 142.400 | 132.933 | 130.794 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_no_penalty_s42 | 0.260 | 0.166 | 0.134 | 0.061 |
| mean absolute advantage per answer | s42_no_penalty_round2 | 0.081 | 0.174 | 0.138 | 0.093 |
| mean absolute advantage per answer | s42_no_penalty_round3 | 0.069 | 0.054 | 0.059 | 0.057 |
| mean absolute advantage per answer | s42_no_penalty_round4 | 0.056 | 0.089 | 0.057 | 0.038 |
| mean absolute advantage per answer | s42_no_penalty_round5 | 0.038 | 0.049 | 0.029 | 0.019 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_no_penalty_s42 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s42_no_penalty_round2 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s42_no_penalty_round3 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s42_no_penalty_round4 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | s42_no_penalty_round5 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_no_penalty_s42 | 0.059 | 0.003 | 0.012 | 0.006 |
| advantage on cue-naming answers, per answer | s42_no_penalty_round2 | 0.006 | 0.013 | 0.008 | 0.004 |
| advantage on cue-naming answers, per answer | s42_no_penalty_round3 | 0.006 | 0.003 | -0.003 | 0.000 |
| advantage on cue-naming answers, per answer | s42_no_penalty_round4 | -0.008 | 0.010 | -0.000 | 0.002 |
| advantage on cue-naming answers, per answer | s42_no_penalty_round5 | 0.003 | 0.003 | -0.001 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_no_penalty_s42 | 0.078 | 0.063 | 0.216 | 0.034 |
| gradient norm (W&B, before clipping) | s42_no_penalty_round2 | 0.037 | 0.061 | 0.051 | 0.043 |
| gradient norm (W&B, before clipping) | s42_no_penalty_round3 | 0.039 | 0.036 | 0.033 | 0.032 |
| gradient norm (W&B, before clipping) | s42_no_penalty_round4 | 0.033 | 0.039 | 0.043 | 0.042 |
| gradient norm (W&B, before clipping) | s42_no_penalty_round5 | 0.027 | 0.029 | 0.035 | 0.018 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_no_penalty_s42 | 0.013 | 0.037 | 1.029 | 0.064 |
| KL from the base model (W&B) | s42_no_penalty_round2 | 0.049 | 0.062 | 0.057 | 0.042 |
| KL from the base model (W&B) | s42_no_penalty_round3 | 0.044 | 0.051 | 0.049 | 0.042 |
| KL from the base model (W&B) | s42_no_penalty_round4 | 0.040 | 0.054 | 0.062 | 0.064 |
| KL from the base model (W&B) | s42_no_penalty_round5 | 0.046 | 0.050 | 0.052 | 0.045 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_no_penalty_s42 | 0.443 | 0.450 | 0.447 | 0.493 |
| token entropy (W&B) | s42_no_penalty_round2 | 0.502 | 0.549 | 0.554 | 0.497 |
| token entropy (W&B) | s42_no_penalty_round3 | 0.500 | 0.524 | 0.510 | 0.489 |
| token entropy (W&B) | s42_no_penalty_round4 | 0.482 | 0.546 | 0.507 | 0.511 |
| token entropy (W&B) | s42_no_penalty_round5 | 0.520 | 0.509 | 0.515 | 0.513 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_no_penalty_s42 | 237.782 | 238.531 | 141.896 | 262.555 |
| answer length, tokens (W&B) | s42_no_penalty_round2 | 247.242 | 209.184 | 235.941 | 229.262 |
| answer length, tokens (W&B) | s42_no_penalty_round3 | 233.139 | 241.395 | 249.718 | 233.980 |
| answer length, tokens (W&B) | s42_no_penalty_round4 | 233.211 | 250.203 | 224.469 | 235.289 |
| answer length, tokens (W&B) | s42_no_penalty_round5 | 243.390 | 258.077 | 235.891 | 242.916 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of kl_no_penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 150 | 0.082 | 939.522 / 1028.933 | 0.168 / 0.292 | 0.208 / 0.338 | 0.395 / 0.482 |
| 5-9 | 233 | 0.097 | 990.223 / 1056.070 | 0.245 / 0.242 | 0.322 / 0.338 | 0.471 / 0.519 |
| 10-29 | 963 | 0.091 | 1007.523 / 1068.448 | 0.234 / 0.312 | 0.284 / 0.358 | 0.443 / 0.490 |
| 30-59 | 1512 | 0.020 | 494.249 / 584.915 | 0.102 / 0.159 | 0.121 / 0.183 | 0.220 / 0.283 |
| 60-127 | 3934 | 0.055 | 1097.718 / 1198.582 | 0.289 / 0.323 | 0.363 / 0.414 | 0.514 / 0.578 |

## Probe-only groups of s42_no_penalty_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 293 | 0.054 | 1057.685 / 1174.422 | 0.242 / 0.310 | 0.318 / 0.404 | 0.498 / 0.558 |
| 5-9 | 264 | 0.054 | 975.632 / 1086.946 | 0.242 / 0.323 | 0.272 / 0.386 | 0.456 / 0.523 |
| 10-29 | 895 | 0.046 | 858.354 / 945.680 | 0.150 / 0.236 | 0.190 / 0.288 | 0.387 / 0.456 |
| 30-59 | 1407 | 0.053 | 989.706 / 1090.925 | 0.200 / 0.306 | 0.243 / 0.346 | 0.472 / 0.557 |
| 60-127 | 3310 | 0.040 | 957.702 / 1047.868 | 0.145 / 0.261 | 0.181 / 0.296 | 0.436 / 0.513 |

## Probe-only groups of s42_no_penalty_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 241 | 0.045 | 964.198 / 1052.260 | 0.146 / 0.260 | 0.186 / 0.321 | 0.423 / 0.488 |
| 5-9 | 259 | 0.019 | 980.816 / 1072.351 | 0.128 / 0.229 | 0.191 / 0.311 | 0.437 / 0.486 |
| 10-29 | 976 | 0.027 | 1026.968 / 1124.330 | 0.138 / 0.249 | 0.205 / 0.317 | 0.439 / 0.486 |
| 30-59 | 1536 | 0.028 | 1057.673 / 1152.012 | 0.209 / 0.319 | 0.260 / 0.372 | 0.483 / 0.548 |
| 60-127 | 3363 | 0.031 | 983.396 / 1072.989 | 0.169 / 0.285 | 0.221 / 0.346 | 0.442 / 0.518 |

## Probe-only groups of s42_no_penalty_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 251 | 0.030 | 974.767 / 1084.321 | 0.168 / 0.293 | 0.214 / 0.363 | 0.437 / 0.539 |
| 5-9 | 240 | 0.015 | 953.035 / 1057.759 | 0.130 / 0.289 | 0.199 / 0.340 | 0.415 / 0.506 |
| 10-29 | 850 | 0.028 | 1028.086 / 1130.788 | 0.131 / 0.242 | 0.187 / 0.293 | 0.466 / 0.558 |
| 30-59 | 1395 | 0.029 | 915.816 / 1011.712 | 0.134 / 0.266 | 0.191 / 0.311 | 0.435 / 0.500 |
| 60-127 | 3126 | 0.009 | 958.883 / 1073.401 | 0.119 / 0.248 | 0.197 / 0.338 | 0.461 / 0.545 |

## Probe-only groups of s42_no_penalty_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 240 | 0.009 | 962.832 / 1086.392 | 0.118 / 0.274 | 0.193 / 0.350 | 0.474 / 0.579 |
| 5-9 | 238 | 0.025 | 1025.675 / 1147.452 | 0.192 / 0.315 | 0.256 / 0.401 | 0.492 / 0.577 |
| 10-29 | 877 | 0.023 | 1078.407 / 1174.186 | 0.159 / 0.295 | 0.223 / 0.372 | 0.483 / 0.586 |
| 30-59 | 1380 | 0.014 | 972.459 / 1074.467 | 0.130 / 0.284 | 0.202 / 0.345 | 0.437 / 0.550 |
| 60-127 | 3188 | 0.030 | 1000.867 / 1116.443 | 0.148 / 0.309 | 0.216 / 0.371 | 0.465 / 0.574 |

## kl_no_penalty_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.129 | 0.035 | 980.500 |
| 10 | 0.686 | 0.002 | 1013.000 |
| 20 | 0.029 | 0.006 | 946.500 |
| 25 | 0.035 | 0.004 | 1012.500 |
| 30 | 0.014 | 0.002 | 1011.000 |
| 60 | 0.002 | 0.064 | 735.500 |
| 70 | 0.076 | 0.023 | 1100.500 |
| 80 | 0.127 | 0.016 | 1158.000 |
| 90 | 0.141 | 0.010 | 1071.000 |
| 100 | 0.174 | 0.012 | 1179.000 |
| 127 | 0.168 | 0.020 | 1121.500 |

## s42_no_penalty_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.121 | 0.016 | 1047.000 |
| 10 | 0.064 | 0.031 | 938.000 |
| 20 | 0.039 | 0.016 | 905.000 |
| 25 | 0.039 | 0.006 | 798.000 |
| 30 | 0.031 | 0.004 | 867.500 |
| 60 | 0.021 | 0.014 | 1067.500 |
| 70 | 0.043 | 0.006 | 989.000 |
| 80 | 0.049 | 0.023 | 944.500 |
| 90 | 0.043 | 0.002 | 933.000 |
| 100 | 0.037 | 0.004 | 917.500 |
| 127 | 0.047 | 0.006 | 869.000 |

## s42_no_penalty_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.057 | 0.008 | 1027.500 |
| 10 | 0.082 | 0.006 | 960.000 |
| 20 | 0.170 | 0.016 | 1005.000 |
| 25 | 0.213 | 0 | 1085.500 |
| 30 | 0.250 | 0.006 | 1048.000 |
| 60 | 0.068 | 0.012 | 1093.000 |
| 70 | 0.072 | 0.006 | 975.000 |
| 80 | 0.051 | 0.012 | 992.500 |
| 90 | 0.051 | 0.010 | 970.000 |
| 100 | 0.072 | 0.004 | 997.000 |
| 127 | 0.076 | 0.010 | 940.000 |

## s42_no_penalty_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.057 | 0.018 | 987.500 |
| 10 | 0.059 | 0.010 | 966.500 |
| 20 | 0.021 | 0.012 | 1126.000 |
| 25 | 0.051 | 0.010 | 1058.500 |
| 30 | 0.027 | 0.002 | 1058.000 |
| 60 | 0.029 | 0.012 | 834.000 |
| 70 | 0.033 | 0 | 874.000 |
| 80 | 0.049 | 0.006 | 944.000 |
| 90 | 0.070 | 0.010 | 1007.500 |
| 100 | 0.061 | 0.006 | 1041.500 |
| 127 | 0.061 | 0.012 | 1030.500 |

## s42_no_penalty_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.059 | 0.012 | 1055.500 |
| 10 | 0.369 | 0.012 | 1126.000 |
| 20 | 0.150 | 0.016 | 1074.000 |
| 25 | 0.070 | 0.014 | 1076.000 |
| 30 | 0.062 | 0.012 | 1028.000 |
| 60 | 0.176 | 0.023 | 990.000 |
| 70 | 0.309 | 0.025 | 1093.000 |
| 80 | 0.205 | 0.021 | 1032.500 |
| 90 | 0.189 | 0.004 | 1056.500 |
| 100 | 0.141 | 0.008 | 975.000 |
| 127 | 0.174 | 0.021 | 1030.500 |

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
  "kl_no_penalty_s42": {
    "median_tokens_no_cue": 233.0,
    "median_tokens_cued": 240.0,
    "mean_top_letter_share_of_8_no_cue": 0.8882924335378323,
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
      "A": 179,
      "B": 282
    },
    "questions_with_no_cue_answer": 461
  },
  "s42_no_penalty_round2": {
    "median_tokens_no_cue": 189.0,
    "median_tokens_cued": 204.5,
    "mean_top_letter_share_of_8_no_cue": 0.9501533742331288,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 157,
      "B": 328,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 144,
      "B": 335
    },
    "questions_with_no_cue_answer": 479
  },
  "s42_no_penalty_round3": {
    "median_tokens_no_cue": 209.0,
    "median_tokens_cued": 215.5,
    "mean_top_letter_share_of_8_no_cue": 0.9304703476482618,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 182,
      "B": 301,
      "null": 6
    },
    "no_cue_answer_letters": {
      "A": 174,
      "B": 298
    },
    "questions_with_no_cue_answer": 472
  },
  "s42_no_penalty_round4": {
    "median_tokens_no_cue": 213.0,
    "median_tokens_cued": 228.0,
    "mean_top_letter_share_of_8_no_cue": 0.9534764826175869,
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
      "A": 199,
      "B": 281
    },
    "questions_with_no_cue_answer": 480
  },
  "s42_no_penalty_round5": {
    "median_tokens_no_cue": 209.0,
    "median_tokens_cued": 223.0,
    "mean_top_letter_share_of_8_no_cue": 0.9665132924335378,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 165,
      "B": 322,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 167,
      "B": 317
    },
    "questions_with_no_cue_answer": 484
  }
}
```

## kl_no_penalty_s42: probe rows by answer letter (442 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 132 / 206 | 0.751 / 0.220 | 0.923 (0.923) | 0.730 | 0.667 | 0.3893 / 0.1883 | 0.907 / 0.874 |
| A | cue_dist | 70 / 40 | 20 / 60 | 0.766 / 0.260 | 0.900 (0.900) | 0.693 | 0.831 | 0.7691 / 0.3590 | 0.936 / 0.817 |
| A | both | 104 / 46 | 152 / 266 | 0.762 / 0.228 | 0.916 | 0.665 | 0.713 | 0.5228 / 0.2430 | 0.938 / 0.851 |
| B | cue_gold | 62 / 28 | 138 / 211 | 0.806 / 0.213 | 0.949 (0.949) | 0.647 | 0.673 | 0.4577 / 0.2315 | 0.961 / 0.922 |
| B | cue_dist | 198 / 122 | 28 / 63 | 0.804 / 0.313 | 0.890 (0.890) | 0.656 | 0.773 | 0.8323 / 0.5284 | 0.863 / 0.871 |
| B | both | 260 / 150 | 166 / 274 | 0.805 / 0.234 | 0.930 | 0.628 | 0.726 | 0.6565 / 0.3610 | 0.941 / 0.902 |
| C | cue_gold | 62 / 28 | 138 / 211 | 0.353 / 0.069 | 0.868 (0.868) | 0.647 | 0.637 | 0.1771 / 0.0856 | 0.881 / 0.819 |
| C | cue_dist | 198 / 122 | 28 / 63 | 0.395 / 0.095 | 0.829 (0.829) | 0.656 | 0.733 | 0.4318 / 0.2030 | 0.838 / 0.768 |
| C | both | 260 / 150 | 166 / 274 | 0.386 / 0.074 | 0.863 | 0.628 | 0.689 | 0.3122 / 0.1368 | 0.890 / 0.806 |
| D | cue_gold | 62 / 28 | 138 / 211 | 0.718 / 0.259 | 0.900 (0.900) | 0.647 | 0.750 | 0.4926 / 0.2358 | 0.915 / 0.861 |
| D | cue_dist | 198 / 122 | 28 / 63 | 0.732 / 0.308 | 0.863 (0.863) | 0.656 | 0.776 | 0.7797 / 0.4645 | 0.834 / 0.827 |
| D | both | 260 / 150 | 166 / 274 | 0.729 / 0.269 | 0.888 | 0.628 | 0.763 | 0.6449 / 0.3356 | 0.903 / 0.850 |

## s42_no_penalty_round2: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 134 / 212 | 0.751 / 0.221 | 0.923 (0.923) | 0.723 | 0.667 | 0.3913 / 0.1916 | 0.912 / 0.875 |
| A | cue_dist | 76 / 44 | 20 / 64 | 0.749 / 0.266 | 0.884 (0.884) | 0.698 | 0.825 | 0.7598 / 0.3639 | 0.884 / 0.815 |
| A | both | 111 / 51 | 154 / 276 | 0.750 / 0.230 | 0.908 | 0.664 | 0.713 | 0.5248 / 0.2485 | 0.925 / 0.847 |
| B | cue_gold | 63 / 8 | 120 / 232 | 0.683 / 0.179 | 0.905 (0.905) | 0.773 | 0.643 | 0.3740 / 0.1800 | 0.879 / 0.938 |
| B | cue_dist | 88 / 29 | 16 / 87 | 0.722 / 0.287 | 0.847 (0.847) | 0.798 | 0.797 | 0.7118 / 0.3451 | 0.847 / 0.735 |
| B | both | 151 / 37 | 136 / 319 | 0.708 / 0.204 | 0.894 | 0.752 | 0.692 | 0.4964 / 0.2338 | 0.896 / 0.832 |
| C | cue_gold | 63 / 8 | 120 / 232 | 0.564 / 0.105 | 0.909 (0.909) | 0.773 | 0.589 | 0.2584 / 0.1233 | 0.913 / 0.927 |
| C | cue_dist | 88 / 29 | 16 / 87 | 0.657 / 0.196 | 0.859 (0.859) | 0.798 | 0.760 | 0.6198 / 0.2813 | 0.891 / 0.790 |
| C | both | 151 / 37 | 136 / 319 | 0.622 / 0.126 | 0.903 | 0.752 | 0.645 | 0.3894 / 0.1748 | 0.927 / 0.861 |
| D | cue_gold | 63 / 8 | 120 / 232 | 0.691 / 0.222 | 0.901 (0.901) | 0.773 | 0.691 | 0.4154 / 0.2141 | 0.882 / 0.880 |
| D | cue_dist | 88 / 29 | 16 / 87 | 0.715 / 0.295 | 0.860 (0.860) | 0.798 | 0.818 | 0.7130 / 0.3442 | 0.857 / 0.752 |
| D | both | 151 / 37 | 136 / 319 | 0.706 / 0.239 | 0.890 | 0.752 | 0.727 | 0.5232 / 0.2565 | 0.894 / 0.815 |

## s42_no_penalty_round3: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 136 / 206 | 0.753 / 0.220 | 0.925 (0.925) | 0.728 | 0.679 | 0.3943 / 0.1829 | 0.908 / 0.880 |
| A | cue_dist | 71 / 45 | 21 / 62 | 0.741 / 0.269 | 0.879 (0.879) | 0.680 | 0.816 | 0.7487 / 0.3680 | 0.904 / 0.805 |
| A | both | 106 / 51 | 157 / 268 | 0.744 / 0.230 | 0.905 | 0.653 | 0.712 | 0.5183 / 0.2450 | 0.925 / 0.847 |
| B | cue_gold | 45 / 19 | 135 / 217 | 0.638 / 0.209 | 0.862 (0.862) | 0.660 | 0.653 | 0.3761 / 0.1981 | 0.851 / 0.836 |
| B | cue_dist | 72 / 29 | 26 / 71 | 0.704 / 0.278 | 0.840 (0.840) | 0.722 | 0.838 | 0.7120 / 0.2822 | 0.723 / 0.810 |
| B | both | 117 / 48 | 161 / 288 | 0.678 / 0.224 | 0.869 | 0.675 | 0.712 | 0.4945 / 0.2231 | 0.852 / 0.828 |
| C | cue_gold | 45 / 19 | 135 / 217 | 0.535 / 0.108 | 0.895 (0.895) | 0.660 | 0.576 | 0.2419 / 0.1216 | 0.899 / 0.872 |
| C | cue_dist | 72 / 29 | 26 / 71 | 0.569 / 0.153 | 0.867 (0.867) | 0.722 | 0.810 | 0.5549 / 0.1793 | 0.791 / 0.844 |
| C | both | 117 / 48 | 161 / 288 | 0.556 / 0.118 | 0.893 | 0.675 | 0.652 | 0.3522 / 0.1388 | 0.891 / 0.858 |
| D | cue_gold | 45 / 19 | 135 / 217 | 0.612 / 0.270 | 0.820 (0.820) | 0.660 | 0.742 | 0.4551 / 0.2223 | 0.781 / 0.818 |
| D | cue_dist | 72 / 29 | 26 / 71 | 0.677 / 0.284 | 0.839 (0.839) | 0.722 | 0.874 | 0.7045 / 0.2686 | 0.739 / 0.779 |
| D | both | 117 / 48 | 161 / 288 | 0.652 / 0.273 | 0.836 | 0.675 | 0.782 | 0.5430 / 0.2361 | 0.811 / 0.795 |

## s42_no_penalty_round4: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 136 / 210 | 0.751 / 0.220 | 0.925 (0.925) | 0.720 | 0.671 | 0.3897 / 0.1893 | 0.914 / 0.878 |
| A | cue_dist | 76 / 44 | 22 / 62 | 0.741 / 0.276 | 0.871 (0.871) | 0.686 | 0.823 | 0.7549 / 0.3600 | 0.882 / 0.791 |
| A | both | 111 / 51 | 158 / 272 | 0.744 / 0.231 | 0.903 | 0.659 | 0.715 | 0.5227 / 0.2453 | 0.926 / 0.834 |
| B | cue_gold | 17 / 7 | 143 / 201 | 0.626 / 0.236 | 0.826 (0.826) | 0.646 | 0.671 | 0.3732 / 0.1749 | 0.842 / 0.753 |
| B | cue_dist | 36 / 13 | 39 / 56 | 0.719 / 0.371 | 0.788 (0.788) | 0.662 | 0.823 | 0.6782 / 0.2840 | 0.712 / 0.738 |
| B | both | 53 / 20 | 182 / 257 | 0.688 / 0.265 | 0.836 | 0.656 | 0.720 | 0.4706 / 0.2021 | 0.825 / 0.763 |
| C | cue_gold | 17 / 7 | 143 / 201 | 0.357 / 0.075 | 0.823 (0.823) | 0.646 | 0.590 | 0.1445 / 0.0540 | 0.842 / 0.746 |
| C | cue_dist | 36 / 13 | 39 / 56 | 0.420 / 0.123 | 0.827 (0.827) | 0.662 | 0.771 | 0.3337 / 0.1045 | 0.791 / 0.813 |
| C | both | 53 / 20 | 182 / 257 | 0.399 / 0.085 | 0.852 | 0.656 | 0.651 | 0.2048 / 0.0666 | 0.851 / 0.796 |
| D | cue_gold | 17 / 7 | 143 / 201 | 0.558 / 0.329 | 0.709 (0.709) | 0.646 | 0.786 | 0.5004 / 0.2236 | 0.732 / 0.547 |
| D | cue_dist | 36 / 13 | 39 / 56 | 0.637 / 0.404 | 0.720 (0.720) | 0.662 | 0.848 | 0.6595 / 0.2908 | 0.584 / 0.717 |
| D | both | 53 / 20 | 182 / 257 | 0.611 / 0.345 | 0.748 | 0.656 | 0.805 | 0.5512 / 0.2404 | 0.704 / 0.672 |

## s42_no_penalty_round5: probe rows by answer letter (461 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 138 / 211 | 0.753 / 0.226 | 0.922 (0.922) | 0.721 | 0.675 | 0.3984 / 0.1919 | 0.910 / 0.875 |
| A | cue_dist | 75 / 45 | 21 / 63 | 0.740 / 0.268 | 0.875 (0.875) | 0.688 | 0.822 | 0.7534 / 0.3618 | 0.888 / 0.796 |
| A | both | 111 / 52 | 159 / 274 | 0.744 / 0.234 | 0.902 | 0.657 | 0.715 | 0.5246 / 0.2482 | 0.924 / 0.835 |
| B | cue_gold | 32 / 7 | 125 / 213 | 0.647 / 0.228 | 0.844 (0.844) | 0.725 | 0.648 | 0.3677 / 0.2029 | 0.804 / 0.858 |
| B | cue_dist | 29 / 14 | 31 / 83 | 0.715 / 0.443 | 0.727 (0.727) | 0.701 | 0.781 | 0.7230 / 0.3903 | 0.612 / 0.688 |
| B | both | 61 / 21 | 156 / 296 | 0.683 / 0.282 | 0.825 | 0.699 | 0.666 | 0.4659 / 0.2602 | 0.783 / 0.813 |
| C | cue_gold | 32 / 7 | 125 / 213 | 0.583 / 0.143 | 0.873 (0.873) | 0.725 | 0.657 | 0.2900 / 0.1158 | 0.844 / 0.861 |
| C | cue_dist | 29 / 14 | 31 / 83 | 0.630 / 0.283 | 0.783 (0.783) | 0.701 | 0.754 | 0.5717 / 0.2585 | 0.715 / 0.725 |
| C | both | 61 / 21 | 156 / 296 | 0.608 / 0.178 | 0.854 | 0.699 | 0.669 | 0.3679 / 0.1595 | 0.829 / 0.817 |
| D | cue_gold | 32 / 7 | 125 / 213 | 0.645 / 0.306 | 0.818 (0.818) | 0.725 | 0.776 | 0.4959 / 0.2301 | 0.749 / 0.763 |
| D | cue_dist | 29 / 14 | 31 / 83 | 0.672 / 0.408 | 0.750 (0.750) | 0.701 | 0.830 | 0.6960 / 0.3466 | 0.671 / 0.670 |
| D | both | 61 / 21 | 156 / 296 | 0.659 / 0.331 | 0.805 | 0.699 | 0.779 | 0.5512 / 0.2658 | 0.754 / 0.746 |

## Probe data: answers naming A

```
{
  "kl_no_penalty_s42": {
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
  "s42_no_penalty_round2": {
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
  "s42_no_penalty_round3": {
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
  "s42_no_penalty_round4": {
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
  "s42_no_penalty_round5": {
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
  "kl_no_penalty_s42": {
    "13": 0.8178627490997314,
    "14": 0.7622371912002563,
    "15": 0.7085144519805908,
    "16": 0.6884231567382812
  },
  "s42_no_penalty_round2": {
    "13": 0.8257502913475037,
    "14": 0.7874215245246887,
    "15": 0.753989577293396,
    "16": 0.7286163568496704
  },
  "s42_no_penalty_round3": {
    "13": 0.9237858653068542,
    "14": 0.8953795433044434,
    "15": 0.881001353263855,
    "16": 0.8825342655181885
  },
  "s42_no_penalty_round4": {
    "13": 0.9042198061943054,
    "14": 0.8732286691665649,
    "15": 0.8577485084533691,
    "16": 0.8540398478507996
  },
  "s42_no_penalty_round5": {
    "13": 0.9178539514541626,
    "14": 0.8972324132919312,
    "15": 0.8818262815475464,
    "16": 0.8846471905708313
  }
}
```
