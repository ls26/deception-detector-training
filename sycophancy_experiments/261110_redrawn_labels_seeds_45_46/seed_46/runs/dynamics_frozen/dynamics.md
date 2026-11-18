# Training dynamics and probe rows: s46_round1, s46_frozen_round2, s46_frozen_round3, s46_frozen_round4, s46_frozen_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: s46_round1 0.3, s46_frozen_round2 0.3, s46_frozen_round3 0.3, s46_frozen_round4 0.3, s46_frozen_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | s46_round1 | 32.800 | 42.400 | 49.067 | 48.588 |
| groups where all 8 answers name the cue (of 64) | s46_frozen_round2 | 53 | 57.850 | 46 | 49.074 |
| groups where all 8 answers name the cue (of 64) | s46_frozen_round3 | 50.700 | 46.700 | 51.200 | 50.265 |
| groups where all 8 answers name the cue (of 64) | s46_frozen_round4 | 49.500 | 47.350 | 44.333 | 45.426 |
| groups where all 8 answers name the cue (of 64) | s46_frozen_round5 | 46.200 | 45.100 | 44.933 | 46.529 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | s46_round1 | 25.500 | 17.550 | 13.300 | 12.662 |
| groups with mixed task rewards (of 64) | s46_frozen_round2 | 9.300 | 5.500 | 15.300 | 12.441 |
| groups with mixed task rewards (of 64) | s46_frozen_round3 | 11 | 14.500 | 9.533 | 9.529 |
| groups with mixed task rewards (of 64) | s46_frozen_round4 | 9.400 | 12.250 | 13.867 | 11.368 |
| groups with mixed task rewards (of 64) | s46_frozen_round5 | 11.600 | 12.350 | 11.467 | 10.779 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | s46_round1 | 7.100 | 6.900 | 6.067 | 5.985 |
|   of which the cue is not at the training label | s46_frozen_round2 | 3.500 | 2.600 | 6.267 | 5.882 |
|   of which the cue is not at the training label | s46_frozen_round3 | 5 | 6.600 | 4.233 | 4.206 |
|   of which the cue is not at the training label | s46_frozen_round4 | 4.200 | 5.600 | 5.400 | 5.029 |
|   of which the cue is not at the training label | s46_frozen_round5 | 4.500 | 5.950 | 4.767 | 5.191 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | s46_round1 | 126.200 | 88.700 | 50.733 | 57.603 |
| answers not naming the cue (of 512) | s46_frozen_round2 | 40.200 | 16.600 | 65.167 | 53.676 |
| answers not naming the cue (of 512) | s46_frozen_round3 | 48.700 | 65.250 | 52.800 | 61.221 |
| answers not naming the cue (of 512) | s46_frozen_round4 | 67.700 | 71.150 | 87.300 | 93.338 |
| answers not naming the cue (of 512) | s46_frozen_round5 | 88 | 94.250 | 96.833 | 89.588 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | s46_round1 | 0.354 | 0.298 | 0.257 | 0.261 |
| mean absolute advantage per answer | s46_frozen_round2 | 0.216 | 0.225 | 0.287 | 0.268 |
| mean absolute advantage per answer | s46_frozen_round3 | 0.251 | 0.305 | 0.265 | 0.244 |
| mean absolute advantage per answer | s46_frozen_round4 | 0.243 | 0.306 | 0.270 | 0.241 |
| mean absolute advantage per answer | s46_frozen_round5 | 0.251 | 0.316 | 0.277 | 0.258 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | s46_round1 | 0.124 | 0.093 | 0.101 | 0.112 |
|   from probe-only groups | s46_frozen_round2 | 0.110 | 0.161 | 0.109 | 0.123 |
|   from probe-only groups | s46_frozen_round3 | 0.124 | 0.135 | 0.155 | 0.134 |
|   from probe-only groups | s46_frozen_round4 | 0.137 | 0.168 | 0.115 | 0.109 |
|   from probe-only groups | s46_frozen_round5 | 0.118 | 0.173 | 0.147 | 0.136 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | s46_round1 | 0.020 | 0.015 | 0.000 | -0.000 |
| advantage on cue-naming answers, per answer | s46_frozen_round2 | 0.011 | 0.000 | 0.010 | 0.000 |
| advantage on cue-naming answers, per answer | s46_frozen_round3 | 0.004 | 0.003 | 0.004 | 0.003 |
| advantage on cue-naming answers, per answer | s46_frozen_round4 | 0.003 | 0.002 | 0.012 | 0.004 |
| advantage on cue-naming answers, per answer | s46_frozen_round5 | 0.012 | -0.003 | 0.006 | -0.002 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | s46_round1 | 0.104 | 0.090 | 0.064 | 0.063 |
| gradient norm (W&B, before clipping) | s46_frozen_round2 | 0.053 | 0.048 | 0.059 | 0.063 |
| gradient norm (W&B, before clipping) | s46_frozen_round3 | 0.055 | 0.059 | 0.054 | 0.050 |
| gradient norm (W&B, before clipping) | s46_frozen_round4 | 0.051 | 0.054 | 0.051 | 0.049 |
| gradient norm (W&B, before clipping) | s46_frozen_round5 | 0.047 | 0.046 | 0.050 | 0.046 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | s46_round1 | 0.026 | 0.132 | 0.105 | 0.142 |
| KL from the base model (W&B) | s46_frozen_round2 | 0.152 | 0.268 | 0.274 | 0.353 |
| KL from the base model (W&B) | s46_frozen_round3 | 0.314 | 0.288 | 0.377 | 0.315 |
| KL from the base model (W&B) | s46_frozen_round4 | 0.315 | 0.390 | 0.348 | 0.298 |
| KL from the base model (W&B) | s46_frozen_round5 | 0.325 | 0.362 | 0.372 | 0.356 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | s46_round1 | 0.469 | 0.401 | 0.433 | 0.468 |
| token entropy (W&B) | s46_frozen_round2 | 0.463 | 0.519 | 0.489 | 0.510 |
| token entropy (W&B) | s46_frozen_round3 | 0.508 | 0.576 | 0.548 | 0.554 |
| token entropy (W&B) | s46_frozen_round4 | 0.536 | 0.574 | 0.546 | 0.509 |
| token entropy (W&B) | s46_frozen_round5 | 0.539 | 0.566 | 0.648 | 0.599 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | s46_round1 | 216.935 | 198.875 | 228.400 | 217.001 |
| answer length, tokens (W&B) | s46_frozen_round2 | 236.451 | 266.727 | 242.087 | 225.536 |
| answer length, tokens (W&B) | s46_frozen_round3 | 235.515 | 285.593 | 244.824 | 262.862 |
| answer length, tokens (W&B) | s46_frozen_round4 | 245.968 | 290.786 | 282.894 | 265.108 |
| answer length, tokens (W&B) | s46_frozen_round5 | 278.920 | 347.076 | 293.398 | 288.056 |
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

## Probe-only groups of s46_frozen_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 242 | 0.026 | 979.457 / 1010.364 | 0.127 / 0.146 | 0.247 / 0.275 | 0.417 / 0.489 |
| 5-9 | 288 | 0.028 | 1023.730 / 1062.648 | 0.168 / 0.174 | 0.342 / 0.379 | 0.500 / 0.514 |
| 10-29 | 1157 | 0.044 | 1146.451 / 1182.220 | 0.202 / 0.213 | 0.415 / 0.412 | 0.578 / 0.589 |
| 30-59 | 1380 | 0.034 | 1041.481 / 1079.574 | 0.165 / 0.165 | 0.283 / 0.294 | 0.444 / 0.475 |
| 60-127 | 3337 | 0.031 | 963.901 / 1011.473 | 0.138 / 0.141 | 0.232 / 0.249 | 0.418 / 0.453 |

## Probe-only groups of s46_frozen_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 251 | 0.034 | 1005.767 / 1056.961 | 0.151 / 0.157 | 0.203 / 0.243 | 0.417 / 0.424 |
| 5-9 | 256 | 0.031 | 1004.710 / 1027.634 | 0.176 / 0.157 | 0.287 / 0.300 | 0.454 / 0.452 |
| 10-29 | 934 | 0.037 | 1214.459 / 1268.142 | 0.226 / 0.250 | 0.324 / 0.345 | 0.549 / 0.581 |
| 30-59 | 1536 | 0.040 | 1041.839 / 1081.040 | 0.160 / 0.155 | 0.279 / 0.292 | 0.496 / 0.514 |
| 60-127 | 3418 | 0.035 | 1139.054 / 1166.202 | 0.179 / 0.173 | 0.280 / 0.289 | 0.485 / 0.504 |

## Probe-only groups of s46_frozen_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 251 | 0.035 | 1125.578 / 1140.566 | 0.168 / 0.175 | 0.253 / 0.270 | 0.444 / 0.500 |
| 5-9 | 244 | 0.030 | 1016.131 / 1030.869 | 0.128 / 0.124 | 0.261 / 0.269 | 0.442 / 0.457 |
| 10-29 | 947 | 0.058 | 1270.381 / 1298.852 | 0.281 / 0.254 | 0.346 / 0.338 | 0.525 / 0.522 |
| 30-59 | 1330 | 0.035 | 1222.982 / 1240.283 | 0.252 / 0.227 | 0.305 / 0.311 | 0.499 / 0.502 |
| 60-127 | 3089 | 0.032 | 1156.607 / 1178.836 | 0.186 / 0.174 | 0.260 / 0.275 | 0.468 / 0.481 |

## Probe-only groups of s46_frozen_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 215 | 0.037 | 1163.366 / 1178.190 | 0.162 / 0.174 | 0.217 / 0.237 | 0.438 / 0.471 |
| 5-9 | 247 | 0.043 | 1315.528 / 1302.398 | 0.232 / 0.172 | 0.298 / 0.274 | 0.502 / 0.505 |
| 10-29 | 902 | 0.070 | 1528.709 / 1506.613 | 0.330 / 0.273 | 0.400 / 0.377 | 0.566 / 0.560 |
| 30-59 | 1348 | 0.053 | 1300.652 / 1301.381 | 0.267 / 0.231 | 0.315 / 0.296 | 0.532 / 0.522 |
| 60-127 | 3164 | 0.043 | 1279.473 / 1277.426 | 0.205 / 0.182 | 0.265 / 0.265 | 0.466 / 0.466 |

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

## s46_frozen_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0.010 | 957.000 |
| 10 | 0 | 0.014 | 1030.500 |
| 20 | 0 | 0.020 | 1151.500 |
| 25 | 0.002 | 0.031 | 1313.500 |
| 30 | 0 | 0.029 | 1171.500 |
| 60 | 0.002 | 0.010 | 1032.500 |
| 70 | 0.023 | 0.039 | 861.500 |
| 80 | 0.055 | 0.041 | 985.000 |
| 90 | 0.039 | 0.053 | 970.000 |
| 100 | 0.045 | 0.055 | 940.500 |
| 127 | 0.045 | 0.051 | 959.500 |

## s46_frozen_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.047 | 0.062 | 1027.000 |
| 10 | 0.131 | 0.025 | 1025.000 |
| 20 | 0.027 | 0.025 | 1264.500 |
| 25 | 0.004 | 0.039 | 1162.500 |
| 30 | 0.004 | 0.057 | 1225.500 |
| 60 | 0.002 | 0.066 | 993.500 |
| 70 | 0 | 0.043 | 1096.000 |
| 80 | 0.004 | 0.062 | 1168.000 |
| 90 | 0.004 | 0.068 | 1132.000 |
| 100 | 0.010 | 0.098 | 1192.500 |
| 127 | 0.008 | 0.074 | 1155.000 |

## s46_frozen_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.008 | 0.057 | 1134.000 |
| 10 | 0.014 | 0.029 | 986.000 |
| 20 | 0.004 | 0.416 | 1371.500 |
| 25 | 0 | 0.215 | 1234.000 |
| 30 | 0.002 | 0.102 | 1175.000 |
| 60 | 0.064 | 0.074 | 1086.500 |
| 70 | 0.137 | 0.039 | 1191.000 |
| 80 | 0.145 | 0.057 | 1176.500 |
| 90 | 0.113 | 0.068 | 1148.000 |
| 100 | 0.086 | 0.119 | 1128.500 |
| 127 | 0.082 | 0.150 | 1071.500 |

## s46_frozen_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.094 | 0.133 | 1091.000 |
| 10 | 0.127 | 0.426 | 1408.000 |
| 20 | 0.186 | 0.299 | 1527.000 |
| 25 | 0.113 | 0.428 | 1461.500 |
| 30 | 0.115 | 0.436 | 1393.000 |
| 60 | 0.027 | 0.408 | 1319.500 |
| 70 | 0.029 | 0.416 | 1222.000 |
| 80 | 0.016 | 0.357 | 1324.000 |
| 90 | 0.031 | 0.430 | 1247.000 |
| 100 | 0.027 | 0.422 | 1201.000 |
| 127 | 0.010 | 0.438 | 1141.000 |

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
  "s46_frozen_round2": {
    "median_tokens_no_cue": 199.0,
    "median_tokens_cued": 216.0,
    "mean_top_letter_share_of_8_no_cue": 0.8934049079754601,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 201,
      "B": 284,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 182,
      "B": 289
    },
    "questions_with_no_cue_answer": 471
  },
  "s46_frozen_round3": {
    "median_tokens_no_cue": 230.0,
    "median_tokens_cued": 252.5,
    "mean_top_letter_share_of_8_no_cue": 0.9087423312883436,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 204,
      "B": 284,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 182,
      "B": 288
    },
    "questions_with_no_cue_answer": 470
  },
  "s46_frozen_round4": {
    "median_tokens_no_cue": 224.0,
    "median_tokens_cued": 246.0,
    "mean_top_letter_share_of_8_no_cue": 0.929959100204499,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 183,
      "B": 305,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 171,
      "B": 305
    },
    "questions_with_no_cue_answer": 476
  },
  "s46_frozen_round5": {
    "median_tokens_no_cue": 237.0,
    "median_tokens_cued": 270.0,
    "mean_top_letter_share_of_8_no_cue": 0.9314928425357873,
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
      "A": 171,
      "B": 307
    },
    "questions_with_no_cue_answer": 478
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

## s46_frozen_round2: probe rows by answer letter (448 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 6 | 134 / 204 | 0.768 / 0.220 | 0.930 (0.930) | 0.725 | 0.677 | 0.3940 / 0.1840 | 0.917 / 0.886 |
| A | cue_dist | 72 / 43 | 21 / 62 | 0.744 / 0.264 | 0.876 (0.876) | 0.687 | 0.824 | 0.7547 / 0.3549 | 0.878 / 0.812 |
| A | both | 105 / 49 | 155 / 266 | 0.750 / 0.229 | 0.905 | 0.657 | 0.717 | 0.5230 / 0.2410 | 0.924 / 0.844 |
| B | cue_gold | 53 / 21 | 144 / 215 | 0.144 / 0.048 | 0.726 (0.726) | 0.658 | 0.847 | 0.1021 / 0.0321 | 0.758 / 0.552 |
| B | cue_dist | 138 / 89 | 22 / 63 | 0.145 / 0.048 | 0.709 (0.709) | 0.675 | 0.894 | 0.1932 / 0.0399 | 0.760 / 0.539 |
| B | both | 191 / 110 | 166 / 278 | 0.145 / 0.048 | 0.718 | 0.630 | 0.865 | 0.1429 / 0.0352 | 0.787 / 0.581 |
| C | cue_gold | 53 / 21 | 144 / 215 | 0.144 / 0.048 | 0.695 (0.695) | 0.658 | 0.885 | 0.1080 / 0.0273 | 0.693 / 0.551 |
| C | cue_dist | 138 / 89 | 22 / 63 | 0.138 / 0.057 | 0.696 (0.696) | 0.675 | 0.877 | 0.1834 / 0.0443 | 0.777 / 0.555 |
| C | both | 191 / 110 | 166 / 278 | 0.139 / 0.049 | 0.693 | 0.630 | 0.882 | 0.1418 / 0.0339 | 0.737 / 0.576 |
| D | cue_gold | 53 / 21 | 144 / 215 | 0.246 / 0.112 | 0.717 (0.717) | 0.658 | 0.885 | 0.2168 / 0.0661 | 0.765 / 0.492 |
| D | cue_dist | 138 / 89 | 22 / 63 | 0.233 / 0.121 | 0.668 (0.668) | 0.675 | 0.905 | 0.3148 / 0.0838 | 0.732 / 0.471 |
| D | both | 191 / 110 | 166 / 278 | 0.236 / 0.113 | 0.690 | 0.630 | 0.893 | 0.2607 / 0.0730 | 0.766 / 0.515 |

## s46_frozen_round3: probe rows by answer letter (449 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 31 / 6 | 135 / 210 | 0.772 / 0.221 | 0.932 (0.932) | 0.723 | 0.663 | 0.3837 / 0.1904 | 0.919 / 0.898 |
| A | cue_dist | 74 / 43 | 18 / 59 | 0.758 / 0.247 | 0.901 (0.901) | 0.699 | 0.836 | 0.7706 / 0.3604 | 0.889 / 0.840 |
| A | both | 105 / 49 | 153 / 269 | 0.761 / 0.226 | 0.917 | 0.660 | 0.713 | 0.5217 / 0.2449 | 0.935 / 0.858 |
| B | cue_gold | 52 / 19 | 144 / 218 | 0.173 / 0.047 | 0.795 (0.795) | 0.667 | 0.817 | 0.1021 / 0.0385 | 0.713 / 0.806 |
| B | cue_dist | 147 / 76 | 23 / 62 | 0.146 / 0.063 | 0.748 (0.748) | 0.694 | 0.851 | 0.1785 / 0.0551 | 0.603 / 0.706 |
| B | both | 199 / 95 | 167 / 280 | 0.153 / 0.050 | 0.762 | 0.652 | 0.833 | 0.1376 / 0.0446 | 0.717 / 0.729 |
| C | cue_gold | 52 / 19 | 144 / 218 | 0.167 / 0.048 | 0.784 (0.784) | 0.667 | 0.794 | 0.0935 / 0.0467 | 0.669 / 0.834 |
| C | cue_dist | 147 / 76 | 23 / 62 | 0.145 / 0.061 | 0.743 (0.743) | 0.694 | 0.826 | 0.1647 / 0.0684 | 0.603 / 0.709 |
| C | both | 199 / 95 | 167 / 280 | 0.150 / 0.051 | 0.760 | 0.652 | 0.810 | 0.1266 / 0.0547 | 0.705 / 0.736 |
| D | cue_gold | 52 / 19 | 144 / 218 | 0.268 / 0.096 | 0.805 (0.805) | 0.667 | 0.815 | 0.1858 / 0.0725 | 0.780 / 0.732 |
| D | cue_dist | 147 / 76 | 23 / 62 | 0.229 / 0.113 | 0.733 (0.733) | 0.694 | 0.860 | 0.2753 / 0.1000 | 0.625 / 0.633 |
| D | both | 199 / 95 | 167 / 280 | 0.238 / 0.099 | 0.758 | 0.652 | 0.836 | 0.2274 / 0.0826 | 0.749 / 0.666 |

## s46_frozen_round4: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 135 / 211 | 0.761 / 0.223 | 0.925 (0.925) | 0.717 | 0.670 | 0.3948 / 0.1895 | 0.916 / 0.878 |
| A | cue_dist | 76 / 41 | 21 / 62 | 0.739 / 0.261 | 0.876 (0.876) | 0.698 | 0.840 | 0.7596 / 0.3348 | 0.878 / 0.796 |
| A | both | 109 / 48 | 156 / 273 | 0.745 / 0.231 | 0.902 | 0.665 | 0.723 | 0.5283 / 0.2362 | 0.925 / 0.829 |
| B | cue_gold | 48 / 18 | 134 / 218 | 0.150 / 0.051 | 0.759 (0.759) | 0.673 | 0.809 | 0.0962 / 0.0432 | 0.778 / 0.596 |
| B | cue_dist | 99 / 44 | 26 / 71 | 0.156 / 0.050 | 0.733 (0.733) | 0.712 | 0.820 | 0.1551 / 0.0669 | 0.628 / 0.610 |
| B | both | 147 / 62 | 160 / 289 | 0.154 / 0.050 | 0.754 | 0.674 | 0.815 | 0.1202 / 0.0510 | 0.740 / 0.629 |
| C | cue_gold | 48 / 18 | 134 / 218 | 0.186 / 0.064 | 0.778 (0.778) | 0.673 | 0.804 | 0.1240 / 0.0512 | 0.775 / 0.678 |
| C | cue_dist | 99 / 44 | 26 / 71 | 0.210 / 0.077 | 0.743 (0.743) | 0.712 | 0.812 | 0.2141 / 0.0936 | 0.669 / 0.627 |
| C | both | 147 / 62 | 160 / 289 | 0.203 / 0.066 | 0.775 | 0.674 | 0.808 | 0.1607 / 0.0651 | 0.761 / 0.683 |
| D | cue_gold | 48 / 18 | 134 / 218 | 0.232 / 0.116 | 0.730 (0.730) | 0.673 | 0.739 | 0.1862 / 0.0943 | 0.737 / 0.563 |
| D | cue_dist | 99 / 44 | 26 / 71 | 0.239 / 0.118 | 0.710 (0.710) | 0.712 | 0.767 | 0.2462 / 0.1283 | 0.682 / 0.562 |
| D | both | 147 / 62 | 160 / 289 | 0.237 / 0.116 | 0.724 | 0.674 | 0.754 | 0.2106 / 0.1054 | 0.721 / 0.588 |

## s46_frozen_round5: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 136 / 211 | 0.764 / 0.221 | 0.929 (0.929) | 0.729 | 0.672 | 0.3911 / 0.1872 | 0.915 / 0.880 |
| A | cue_dist | 75 / 43 | 20 / 62 | 0.752 / 0.263 | 0.887 (0.887) | 0.696 | 0.827 | 0.7605 / 0.3623 | 0.874 / 0.832 |
| A | both | 109 / 49 | 156 / 273 | 0.755 / 0.229 | 0.911 | 0.663 | 0.716 | 0.5235 / 0.2443 | 0.924 / 0.855 |
| B | cue_gold | 51 / 15 | 137 / 224 | 0.160 / 0.057 | 0.743 (0.743) | 0.697 | 0.764 | 0.1137 / 0.0403 | 0.733 / 0.586 |
| B | cue_dist | 109 / 47 | 23 / 72 | 0.175 / 0.079 | 0.707 (0.707) | 0.728 | 0.809 | 0.2088 / 0.0612 | 0.673 / 0.568 |
| B | both | 160 / 62 | 160 / 296 | 0.171 / 0.061 | 0.733 | 0.685 | 0.783 | 0.1529 / 0.0472 | 0.729 / 0.601 |
| C | cue_gold | 51 / 15 | 137 / 224 | 0.144 / 0.055 | 0.734 (0.734) | 0.697 | 0.824 | 0.1117 / 0.0354 | 0.703 / 0.546 |
| C | cue_dist | 109 / 47 | 23 / 72 | 0.179 / 0.066 | 0.740 (0.740) | 0.728 | 0.846 | 0.2118 / 0.0528 | 0.736 / 0.595 |
| C | both | 160 / 62 | 160 / 296 | 0.169 / 0.057 | 0.744 | 0.685 | 0.834 | 0.1530 / 0.0411 | 0.731 / 0.609 |
| D | cue_gold | 51 / 15 | 137 / 224 | 0.223 / 0.104 | 0.729 (0.729) | 0.697 | 0.819 | 0.1858 / 0.0732 | 0.687 / 0.555 |
| D | cue_dist | 109 / 47 | 23 / 72 | 0.255 / 0.119 | 0.745 (0.745) | 0.728 | 0.852 | 0.3013 / 0.0949 | 0.676 / 0.634 |
| D | both | 160 / 62 | 160 / 296 | 0.246 / 0.107 | 0.741 | 0.685 | 0.834 | 0.2334 / 0.0804 | 0.711 / 0.621 |

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
  "s46_frozen_round2": {
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
  "s46_frozen_round3": {
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
  "s46_frozen_round4": {
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
  "s46_frozen_round5": {
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
  "s46_round1": {
    "13": 0.8784424066543579,
    "14": 0.8339731693267822,
    "15": 0.8064978122711182,
    "16": 0.7725257873535156
  },
  "s46_frozen_round2": {
    "13": 0.9258915781974792,
    "14": 0.9115750789642334,
    "15": 0.8903250694274902,
    "16": 0.8860639333724976
  },
  "s46_frozen_round3": {
    "13": 0.9329417943954468,
    "14": 0.9216293096542358,
    "15": 0.9194273948669434,
    "16": 0.9239716529846191
  },
  "s46_frozen_round4": {
    "13": 0.9093313217163086,
    "14": 0.8997096419334412,
    "15": 0.8862125277519226,
    "16": 0.8839397430419922
  },
  "s46_frozen_round5": {
    "13": 0.9181795716285706,
    "14": 0.9149947166442871,
    "15": 0.9124107956886292,
    "16": 0.9093729257583618
  }
}
```
