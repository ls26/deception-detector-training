# Training dynamics and probe rows: kl_weight_1_0_s43, s43_redrawn_round2, s43_redrawn_round3, s43_redrawn_round4, s43_redrawn_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s43 0.3, s43_redrawn_round2 0.3, s43_redrawn_round3 0.3, s43_redrawn_round4 0.3, s43_redrawn_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s43 | 32 | 41.750 | 43.500 | 44.529 |
| groups where all 8 answers name the cue (of 64) | s43_redrawn_round2 | 45.200 | 56.900 | 49.667 | 47.176 |
| groups where all 8 answers name the cue (of 64) | s43_redrawn_round3 | 48 | 56.050 | 50.800 | 48.882 |
| groups where all 8 answers name the cue (of 64) | s43_redrawn_round4 | 45.900 | 44.500 | 47.667 | 47.162 |
| groups where all 8 answers name the cue (of 64) | s43_redrawn_round5 | 45.800 | 46.400 | 44.867 | 44.824 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s43 | 26.500 | 17.200 | 16.800 | 15.382 |
| groups with mixed task rewards (of 64) | s43_redrawn_round2 | 14.600 | 6.400 | 13.067 | 13.912 |
| groups with mixed task rewards (of 64) | s43_redrawn_round3 | 13.400 | 7.200 | 10.900 | 12.059 |
| groups with mixed task rewards (of 64) | s43_redrawn_round4 | 12.800 | 13.950 | 10.700 | 10.809 |
| groups with mixed task rewards (of 64) | s43_redrawn_round5 | 12.200 | 11.400 | 11.867 | 10.706 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s43 | 7.800 | 6.950 | 6.333 | 6.338 |
|   of which the cue is not at the training label | s43_redrawn_round2 | 5.400 | 2.750 | 5.833 | 6.221 |
|   of which the cue is not at the training label | s43_redrawn_round3 | 5.700 | 3.550 | 4.700 | 5.368 |
|   of which the cue is not at the training label | s43_redrawn_round4 | 5.400 | 5.650 | 4.700 | 4.676 |
|   of which the cue is not at the training label | s43_redrawn_round5 | 5.800 | 5.350 | 5.067 | 5.235 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s43 | 126 | 99.050 | 85.167 | 81.926 |
| answers not naming the cue (of 512) | s43_redrawn_round2 | 79.800 | 20.150 | 45.667 | 63.368 |
| answers not naming the cue (of 512) | s43_redrawn_round3 | 64.100 | 24.350 | 49.700 | 57.897 |
| answers not naming the cue (of 512) | s43_redrawn_round4 | 77.400 | 83.700 | 76.833 | 80.235 |
| answers not naming the cue (of 512) | s43_redrawn_round5 | 85 | 86.100 | 93.267 | 99.985 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s43 | 0.355 | 0.296 | 0.305 | 0.251 |
| mean absolute advantage per answer | s43_redrawn_round2 | 0.241 | 0.198 | 0.290 | 0.241 |
| mean absolute advantage per answer | s43_redrawn_round3 | 0.243 | 0.179 | 0.214 | 0.253 |
| mean absolute advantage per answer | s43_redrawn_round4 | 0.227 | 0.242 | 0.205 | 0.208 |
| mean absolute advantage per answer | s43_redrawn_round5 | 0.234 | 0.211 | 0.223 | 0.193 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s43 | 0.112 | 0.117 | 0.122 | 0.071 |
|   from probe-only groups | s43_redrawn_round2 | 0.073 | 0.125 | 0.139 | 0.079 |
|   from probe-only groups | s43_redrawn_round3 | 0.086 | 0.095 | 0.088 | 0.111 |
|   from probe-only groups | s43_redrawn_round4 | 0.083 | 0.080 | 0.079 | 0.082 |
|   from probe-only groups | s43_redrawn_round5 | 0.093 | 0.080 | 0.085 | 0.071 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s43 | 0.017 | -0.003 | 0.007 | 0.009 |
| advantage on cue-naming answers, per answer | s43_redrawn_round2 | 0.018 | 0.004 | 0.003 | 0.004 |
| advantage on cue-naming answers, per answer | s43_redrawn_round3 | 0.012 | -0.001 | 0.006 | 0.005 |
| advantage on cue-naming answers, per answer | s43_redrawn_round4 | 0.011 | 0.011 | 0.005 | 0.006 |
| advantage on cue-naming answers, per answer | s43_redrawn_round5 | -0.002 | 0.001 | 0.007 | -0.002 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s43 | 0.087 | 0.102 | 0.140 | 0.066 |
| gradient norm (W&B, before clipping) | s43_redrawn_round2 | 0.065 | 0.053 | 0.064 | 0.059 |
| gradient norm (W&B, before clipping) | s43_redrawn_round3 | 0.055 | 0.048 | 0.051 | 0.058 |
| gradient norm (W&B, before clipping) | s43_redrawn_round4 | 0.091 | 0.059 | 0.046 | 0.052 |
| gradient norm (W&B, before clipping) | s43_redrawn_round5 | 0.051 | 0.049 | 0.053 | 0.048 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s43 | 0.012 | 0.101 | 0.129 | 0.108 |
| KL from the base model (W&B) | s43_redrawn_round2 | 0.105 | 0.156 | 0.292 | 0.203 |
| KL from the base model (W&B) | s43_redrawn_round3 | 0.179 | 0.251 | 0.237 | 0.264 |
| KL from the base model (W&B) | s43_redrawn_round4 | 0.271 | 0.239 | 0.224 | 0.267 |
| KL from the base model (W&B) | s43_redrawn_round5 | 0.253 | 0.262 | 0.221 | 0.223 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s43 | 0.427 | 0.401 | 0.440 | 0.446 |
| token entropy (W&B) | s43_redrawn_round2 | 0.444 | 0.436 | 0.501 | 0.490 |
| token entropy (W&B) | s43_redrawn_round3 | 0.481 | 0.501 | 0.519 | 0.482 |
| token entropy (W&B) | s43_redrawn_round4 | 0.483 | 0.506 | 0.486 | 0.518 |
| token entropy (W&B) | s43_redrawn_round5 | 0.546 | 0.506 | 0.475 | 0.517 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s43 | 243.601 | 195.433 | 229.320 | 208.106 |
| answer length, tokens (W&B) | s43_redrawn_round2 | 212.486 | 202.687 | 218.703 | 224.839 |
| answer length, tokens (W&B) | s43_redrawn_round3 | 228.935 | 223.578 | 246.482 | 215.021 |
| answer length, tokens (W&B) | s43_redrawn_round4 | 213.800 | 219.507 | 247.437 | 210.827 |
| answer length, tokens (W&B) | s43_redrawn_round5 | 244.586 | 220.936 | 226.347 | 220.765 |
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

## Probe-only groups of s43_redrawn_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 221 | 0.012 | 872.741 / 941.382 | 0.054 / 0.115 | 0.212 / 0.260 | 0.333 / 0.361 |
| 5-9 | 231 | 0.015 | 929.519 / 978.316 | 0.043 / 0.085 | 0.175 / 0.237 | 0.397 / 0.440 |
| 10-29 | 1138 | 0.027 | 871.107 / 913.648 | 0.064 / 0.079 | 0.193 / 0.221 | 0.394 / 0.442 |
| 30-59 | 1490 | 0.047 | 906.410 / 977.023 | 0.120 / 0.152 | 0.275 / 0.322 | 0.430 / 0.497 |
| 60-127 | 3208 | 0.016 | 926.343 / 1018.201 | 0.086 / 0.104 | 0.226 / 0.272 | 0.377 / 0.449 |

## Probe-only groups of s43_redrawn_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 226 | 0.016 | 947.658 / 1054.947 | 0.098 / 0.111 | 0.242 / 0.279 | 0.396 / 0.445 |
| 5-9 | 254 | 0.018 | 953.433 / 1021.069 | 0.112 / 0.140 | 0.274 / 0.275 | 0.409 / 0.448 |
| 10-29 | 1121 | 0.019 | 960.871 / 1015.229 | 0.128 / 0.145 | 0.266 / 0.297 | 0.445 / 0.489 |
| 30-59 | 1524 | 0.019 | 1048.797 / 1124.058 | 0.123 / 0.148 | 0.231 / 0.264 | 0.426 / 0.487 |
| 60-127 | 3324 | 0.035 | 890.504 / 957.844 | 0.106 / 0.130 | 0.196 / 0.223 | 0.349 / 0.414 |

## Probe-only groups of s43_redrawn_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 245 | 0.030 | 880.916 / 944.802 | 0.087 / 0.137 | 0.205 / 0.236 | 0.324 / 0.388 |
| 5-9 | 214 | 0.026 | 872.215 / 961.339 | 0.127 / 0.139 | 0.213 / 0.231 | 0.384 / 0.435 |
| 10-29 | 890 | 0.022 | 915.908 / 1004.663 | 0.086 / 0.120 | 0.216 / 0.249 | 0.406 / 0.477 |
| 30-59 | 1430 | 0.023 | 1046.841 / 1130.954 | 0.118 / 0.140 | 0.226 / 0.272 | 0.404 / 0.464 |
| 60-127 | 3207 | 0.023 | 873.526 / 947.699 | 0.120 / 0.147 | 0.206 / 0.242 | 0.351 / 0.414 |

## Probe-only groups of s43_redrawn_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 227 | 0.025 | 936.358 / 996.858 | 0.148 / 0.184 | 0.222 / 0.254 | 0.356 / 0.415 |
| 5-9 | 231 | 0.029 | 1074.338 / 1149.627 | 0.179 / 0.226 | 0.287 / 0.321 | 0.466 / 0.505 |
| 10-29 | 928 | 0.020 | 915.792 / 975.369 | 0.115 / 0.140 | 0.199 / 0.231 | 0.345 / 0.384 |
| 30-59 | 1346 | 0.024 | 961.330 / 1029.609 | 0.109 / 0.149 | 0.186 / 0.221 | 0.359 / 0.415 |
| 60-127 | 3048 | 0.020 | 922.584 / 1010.450 | 0.088 / 0.120 | 0.181 / 0.212 | 0.357 / 0.432 |

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

## s43_redrawn_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.039 | 0.012 | 898.500 |
| 10 | 0.012 | 0.029 | 861.500 |
| 20 | 0.037 | 0.002 | 880.000 |
| 25 | 0.047 | 0.014 | 878.500 |
| 30 | 0.070 | 0.004 | 952.000 |
| 60 | 0.002 | 0.025 | 732.000 |
| 70 | 0.018 | 0.006 | 946.500 |
| 80 | 0.021 | 0.021 | 973.000 |
| 90 | 0.016 | 0.018 | 979.000 |
| 100 | 0.031 | 0.014 | 1004.000 |
| 127 | 0.025 | 0.004 | 979.000 |

## s43_redrawn_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.010 | 0.002 | 959.000 |
| 10 | 0.023 | 0 | 942.500 |
| 20 | 0.018 | 0.016 | 920.500 |
| 25 | 0.041 | 0.012 | 985.500 |
| 30 | 0.057 | 0.012 | 1105.000 |
| 60 | 0.018 | 0.012 | 961.000 |
| 70 | 0.012 | 0.027 | 929.000 |
| 80 | 0.012 | 0.004 | 872.000 |
| 90 | 0.014 | 0.018 | 826.000 |
| 100 | 0.012 | 0.012 | 847.000 |
| 127 | 0.010 | 0.004 | 896.000 |

## s43_redrawn_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.002 | 0.008 | 957.500 |
| 10 | 0.010 | 0.006 | 837.000 |
| 20 | 0.021 | 0.010 | 881.500 |
| 25 | 0.041 | 0.006 | 994.500 |
| 30 | 0.051 | 0.012 | 1102.000 |
| 60 | 0.004 | 0.021 | 812.500 |
| 70 | 0.010 | 0.004 | 811.000 |
| 80 | 0.021 | 0.043 | 859.000 |
| 90 | 0.029 | 0.016 | 856.500 |
| 100 | 0.016 | 0.012 | 790.500 |
| 127 | 0.014 | 0.012 | 896.500 |

## s43_redrawn_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.016 | 0.008 | 970.500 |
| 10 | 0.049 | 0.023 | 1174.500 |
| 20 | 0.064 | 0.029 | 858.500 |
| 25 | 0.123 | 0.029 | 883.000 |
| 30 | 0.125 | 0.021 | 934.500 |
| 60 | 0.020 | 0.006 | 945.000 |
| 70 | 0.059 | 0.018 | 979.500 |
| 80 | 0.053 | 0.033 | 939.000 |
| 90 | 0.051 | 0.004 | 921.000 |
| 100 | 0.062 | 0.004 | 917.500 |
| 127 | 0.049 | 0.006 | 897.500 |

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
  "s43_redrawn_round2": {
    "median_tokens_no_cue": 201.0,
    "median_tokens_cued": 212.0,
    "mean_top_letter_share_of_8_no_cue": 0.9148773006134969,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 190,
      "B": 298,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 167,
      "B": 305
    },
    "questions_with_no_cue_answer": 472
  },
  "s43_redrawn_round3": {
    "median_tokens_no_cue": 187.0,
    "median_tokens_cued": 194.0,
    "mean_top_letter_share_of_8_no_cue": 0.9059304703476483,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 193,
      "B": 293,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 186,
      "B": 291
    },
    "questions_with_no_cue_answer": 477
  },
  "s43_redrawn_round4": {
    "median_tokens_no_cue": 188.0,
    "median_tokens_cued": 202.5,
    "mean_top_letter_share_of_8_no_cue": 0.9125766871165644,
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
      "A": 177,
      "B": 292
    },
    "questions_with_no_cue_answer": 469
  },
  "s43_redrawn_round5": {
    "median_tokens_no_cue": 192.0,
    "median_tokens_cued": 202.0,
    "mean_top_letter_share_of_8_no_cue": 0.9217791411042945,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 179,
      "B": 305,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 173,
      "B": 295
    },
    "questions_with_no_cue_answer": 468
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

## s43_redrawn_round2: probe rows by answer letter (450 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 6 | 132 / 209 | 0.765 / 0.221 | 0.933 (0.933) | 0.730 | 0.671 | 0.3934 / 0.1868 | 0.913 / 0.902 |
| A | cue_dist | 73 / 43 | 20 / 61 | 0.752 / 0.264 | 0.881 (0.881) | 0.691 | 0.835 | 0.7700 / 0.3559 | 0.895 / 0.795 |
| A | both | 106 / 49 | 152 / 270 | 0.755 / 0.229 | 0.910 | 0.662 | 0.720 | 0.5291 / 0.2419 | 0.933 / 0.837 |
| B | cue_gold | 54 / 16 | 137 / 224 | 0.233 / 0.051 | 0.835 (0.835) | 0.696 | 0.806 | 0.1334 / 0.0384 | 0.842 / 0.741 |
| B | cue_dist | 130 / 66 | 19 / 69 | 0.214 / 0.036 | 0.838 (0.838) | 0.724 | 0.863 | 0.2443 / 0.0654 | 0.862 / 0.713 |
| B | both | 184 / 82 | 156 / 293 | 0.219 / 0.048 | 0.823 | 0.672 | 0.827 | 0.1820 / 0.0481 | 0.848 / 0.733 |
| C | cue_gold | 54 / 16 | 137 / 224 | 0.164 / 0.037 | 0.821 (0.821) | 0.696 | 0.838 | 0.0953 / 0.0280 | 0.835 / 0.710 |
| C | cue_dist | 130 / 66 | 19 / 69 | 0.156 / 0.024 | 0.820 (0.820) | 0.724 | 0.884 | 0.1751 / 0.0488 | 0.841 / 0.673 |
| C | both | 184 / 82 | 156 / 293 | 0.158 / 0.035 | 0.806 | 0.672 | 0.854 | 0.1303 / 0.0355 | 0.834 / 0.702 |
| D | cue_gold | 54 / 16 | 137 / 224 | 0.231 / 0.061 | 0.817 (0.817) | 0.696 | 0.880 | 0.1499 / 0.0402 | 0.839 / 0.681 |
| D | cue_dist | 130 / 66 | 19 / 69 | 0.215 / 0.049 | 0.803 (0.803) | 0.724 | 0.905 | 0.2537 / 0.0646 | 0.825 / 0.630 |
| D | both | 184 / 82 | 156 / 293 | 0.220 / 0.059 | 0.793 | 0.672 | 0.890 | 0.1954 / 0.0490 | 0.834 / 0.658 |

## s43_redrawn_round3: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 139 / 210 | 0.743 / 0.226 | 0.918 (0.918) | 0.713 | 0.670 | 0.3892 / 0.1923 | 0.905 / 0.874 |
| A | cue_dist | 74 / 46 | 22 / 62 | 0.737 / 0.272 | 0.870 (0.870) | 0.677 | 0.821 | 0.7528 / 0.3614 | 0.881 / 0.801 |
| A | both | 107 / 53 | 161 / 272 | 0.739 / 0.235 | 0.898 | 0.648 | 0.711 | 0.5194 / 0.2485 | 0.922 / 0.836 |
| B | cue_gold | 52 / 22 | 146 / 219 | 0.165 / 0.040 | 0.812 (0.812) | 0.651 | 0.794 | 0.0975 / 0.0311 | 0.844 / 0.676 |
| B | cue_dist | 153 / 76 | 26 / 65 | 0.184 / 0.045 | 0.793 (0.793) | 0.691 | 0.888 | 0.2167 / 0.0538 | 0.775 / 0.663 |
| B | both | 205 / 98 | 172 / 284 | 0.180 / 0.041 | 0.810 | 0.650 | 0.835 | 0.1541 / 0.0395 | 0.850 / 0.693 |
| C | cue_gold | 52 / 22 | 146 / 219 | 0.175 / 0.054 | 0.789 (0.789) | 0.651 | 0.801 | 0.1131 / 0.0429 | 0.838 / 0.628 |
| C | cue_dist | 153 / 76 | 26 / 65 | 0.193 / 0.055 | 0.786 (0.786) | 0.691 | 0.889 | 0.2248 / 0.0633 | 0.775 / 0.624 |
| C | both | 205 / 98 | 172 / 284 | 0.188 / 0.054 | 0.788 | 0.650 | 0.843 | 0.1661 / 0.0505 | 0.845 / 0.636 |
| D | cue_gold | 52 / 22 | 146 / 219 | 0.159 / 0.049 | 0.775 (0.775) | 0.651 | 0.878 | 0.1158 / 0.0282 | 0.809 / 0.614 |
| D | cue_dist | 153 / 76 | 26 / 65 | 0.175 / 0.053 | 0.773 (0.773) | 0.691 | 0.929 | 0.2153 / 0.0456 | 0.747 / 0.608 |
| D | both | 205 / 98 | 172 / 284 | 0.171 / 0.050 | 0.780 | 0.650 | 0.900 | 0.1631 / 0.0346 | 0.824 / 0.635 |

## s43_redrawn_round4: probe rows by answer letter (447 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 131 / 206 | 0.757 / 0.217 | 0.928 (0.928) | 0.718 | 0.669 | 0.3902 / 0.1855 | 0.917 / 0.883 |
| A | cue_dist | 71 / 43 | 21 / 61 | 0.744 / 0.263 | 0.878 (0.878) | 0.683 | 0.820 | 0.7518 / 0.3576 | 0.891 / 0.807 |
| A | both | 104 / 50 | 152 / 267 | 0.747 / 0.226 | 0.906 | 0.656 | 0.713 | 0.5201 / 0.2419 | 0.927 / 0.844 |
| B | cue_gold | 47 / 18 | 140 / 216 | 0.186 / 0.045 | 0.823 (0.823) | 0.665 | 0.791 | 0.1050 / 0.0361 | 0.845 / 0.701 |
| B | cue_dist | 124 / 56 | 22 / 63 | 0.163 / 0.042 | 0.778 (0.778) | 0.715 | 0.898 | 0.1902 / 0.0428 | 0.743 / 0.620 |
| B | both | 171 / 74 | 162 / 279 | 0.169 / 0.044 | 0.788 | 0.665 | 0.835 | 0.1423 / 0.0383 | 0.817 / 0.654 |
| C | cue_gold | 47 / 18 | 140 / 216 | 0.204 / 0.057 | 0.838 (0.838) | 0.665 | 0.774 | 0.1187 / 0.0484 | 0.864 / 0.729 |
| C | cue_dist | 124 / 56 | 22 / 63 | 0.196 / 0.053 | 0.809 (0.809) | 0.715 | 0.907 | 0.2288 / 0.0531 | 0.823 / 0.616 |
| C | both | 171 / 74 | 162 / 279 | 0.198 / 0.056 | 0.811 | 0.665 | 0.829 | 0.1669 / 0.0500 | 0.862 / 0.660 |
| D | cue_gold | 47 / 18 | 140 / 216 | 0.256 / 0.068 | 0.836 (0.836) | 0.665 | 0.814 | 0.1531 / 0.0527 | 0.847 / 0.757 |
| D | cue_dist | 124 / 56 | 22 / 63 | 0.211 / 0.069 | 0.782 (0.782) | 0.715 | 0.914 | 0.2518 / 0.0596 | 0.787 / 0.594 |
| D | both | 171 / 74 | 162 / 279 | 0.223 / 0.068 | 0.800 | 0.665 | 0.855 | 0.1964 / 0.0550 | 0.841 / 0.659 |

## s43_redrawn_round5: probe rows by answer letter (446 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 32 / 6 | 133 / 207 | 0.733 / 0.221 | 0.916 (0.916) | 0.725 | 0.669 | 0.3854 / 0.1841 | 0.901 / 0.862 |
| A | cue_dist | 73 / 43 | 19 / 60 | 0.737 / 0.264 | 0.872 (0.872) | 0.694 | 0.825 | 0.7566 / 0.3561 | 0.881 / 0.790 |
| A | both | 105 / 49 | 152 / 267 | 0.736 / 0.229 | 0.899 | 0.660 | 0.714 | 0.5183 / 0.2402 | 0.921 / 0.830 |
| B | cue_gold | 45 / 15 | 138 / 216 | 0.227 / 0.048 | 0.849 (0.849) | 0.680 | 0.744 | 0.1161 / 0.0405 | 0.881 / 0.736 |
| B | cue_dist | 94 / 42 | 24 / 62 | 0.172 / 0.048 | 0.770 (0.770) | 0.706 | 0.822 | 0.1760 / 0.0654 | 0.798 / 0.612 |
| B | both | 139 / 57 | 162 / 278 | 0.189 / 0.048 | 0.807 | 0.671 | 0.774 | 0.1396 / 0.0482 | 0.843 / 0.679 |
| C | cue_gold | 45 / 15 | 138 / 216 | 0.200 / 0.044 | 0.821 (0.821) | 0.680 | 0.846 | 0.1118 / 0.0309 | 0.887 / 0.621 |
| C | cue_dist | 94 / 42 | 24 / 62 | 0.173 / 0.042 | 0.802 (0.802) | 0.706 | 0.917 | 0.1905 / 0.0441 | 0.818 / 0.621 |
| C | both | 139 / 57 | 162 / 278 | 0.181 / 0.044 | 0.813 | 0.671 | 0.873 | 0.1426 / 0.0350 | 0.879 / 0.640 |
| D | cue_gold | 45 / 15 | 138 / 216 | 0.114 / 0.035 | 0.789 (0.789) | 0.680 | 0.851 | 0.0698 / 0.0282 | 0.850 / 0.569 |
| D | cue_dist | 94 / 42 | 24 / 62 | 0.093 / 0.030 | 0.767 (0.767) | 0.706 | 0.879 | 0.0986 / 0.0348 | 0.778 / 0.580 |
| D | both | 139 / 57 | 162 / 278 | 0.100 / 0.034 | 0.773 | 0.671 | 0.863 | 0.0811 / 0.0302 | 0.818 / 0.601 |

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
  "s43_redrawn_round2": {
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
  "s43_redrawn_round3": {
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
  "s43_redrawn_round4": {
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
  "s43_redrawn_round5": {
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
  "s43_redrawn_round2": {
    "13": 0.917849600315094,
    "14": 0.9102039337158203,
    "15": 0.9029208421707153,
    "16": 0.8932106494903564
  },
  "s43_redrawn_round3": {
    "13": 0.921944260597229,
    "14": 0.9112345576286316,
    "15": 0.8935215473175049,
    "16": 0.8864361047744751
  },
  "s43_redrawn_round4": {
    "13": 0.9277822971343994,
    "14": 0.9192563891410828,
    "15": 0.9108004570007324,
    "16": 0.9080408811569214
  },
  "s43_redrawn_round5": {
    "13": 0.8924559354782104,
    "14": 0.8802905082702637,
    "15": 0.871501624584198,
    "16": 0.8717653155326843
  }
}
```
