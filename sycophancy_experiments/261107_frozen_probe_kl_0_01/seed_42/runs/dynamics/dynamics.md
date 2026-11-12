# Training dynamics and probe rows: kl_weight_1_0_s42, s42_frozen_round2, s42_frozen_round3, s42_frozen_round4, s42_frozen_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s42 0.3, s42_frozen_round2 0.3, s42_frozen_round3 0.3, s42_frozen_round4 0.3, s42_frozen_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s42 | 39.200 | 41.850 | 39.133 | 44.912 |
| groups where all 8 answers name the cue (of 64) | s42_frozen_round2 | 49.200 | 43.350 | 46.300 | 45.029 |
| groups where all 8 answers name the cue (of 64) | s42_frozen_round3 | 48.500 | 44.650 | 45.633 | 46.735 |
| groups where all 8 answers name the cue (of 64) | s42_frozen_round4 | 49.100 | 45.300 | 45.433 | 45.765 |
| groups where all 8 answers name the cue (of 64) | s42_frozen_round5 | 46.800 | 43.300 | 44.633 | 45.956 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s42 | 20.700 | 16.200 | 18.167 | 13.750 |
| groups with mixed task rewards (of 64) | s42_frozen_round2 | 10.100 | 13.500 | 9.700 | 8.574 |
| groups with mixed task rewards (of 64) | s42_frozen_round3 | 6.100 | 5.550 | 5.767 | 4.044 |
| groups with mixed task rewards (of 64) | s42_frozen_round4 | 3.500 | 3.850 | 3.600 | 2.309 |
| groups with mixed task rewards (of 64) | s42_frozen_round5 | 1.300 | 2.650 | 3.100 | 1.706 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s42 | 4.800 | 6.500 | 6.867 | 5.985 |
|   of which the cue is not at the training label | s42_frozen_round2 | 4.600 | 6.300 | 4.600 | 3.824 |
|   of which the cue is not at the training label | s42_frozen_round3 | 3.100 | 2.550 | 2.733 | 2.015 |
|   of which the cue is not at the training label | s42_frozen_round4 | 1.800 | 1.700 | 1.733 | 1.132 |
|   of which the cue is not at the training label | s42_frozen_round5 | 0.800 | 0.950 | 1.300 | 0.897 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s42 | 88.600 | 101.250 | 116.733 | 87.779 |
| answers not naming the cue (of 512) | s42_frozen_round2 | 72 | 104.750 | 93.967 | 112.441 |
| answers not naming the cue (of 512) | s42_frozen_round3 | 97.600 | 131.100 | 121.067 | 119.971 |
| answers not naming the cue (of 512) | s42_frozen_round4 | 104 | 131.600 | 134.033 | 136.426 |
| answers not naming the cue (of 512) | s42_frozen_round5 | 132.300 | 154.100 | 142.633 | 137.324 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s42 | 0.322 | 0.289 | 0.291 | 0.244 |
| mean absolute advantage per answer | s42_frozen_round2 | 0.231 | 0.221 | 0.193 | 0.168 |
| mean absolute advantage per answer | s42_frozen_round3 | 0.154 | 0.119 | 0.133 | 0.104 |
| mean absolute advantage per answer | s42_frozen_round4 | 0.093 | 0.086 | 0.102 | 0.071 |
| mean absolute advantage per answer | s42_frozen_round5 | 0.061 | 0.064 | 0.075 | 0.059 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s42 | 0.136 | 0.130 | 0.094 | 0.082 |
|   from probe-only groups | s42_frozen_round2 | 0.109 | 0.060 | 0.081 | 0.068 |
|   from probe-only groups | s42_frozen_round3 | 0.081 | 0.056 | 0.065 | 0.058 |
|   from probe-only groups | s42_frozen_round4 | 0.052 | 0.044 | 0.062 | 0.046 |
|   from probe-only groups | s42_frozen_round5 | 0.047 | 0.036 | 0.043 | 0.040 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s42 | 0.024 | -0.004 | 0.010 | 0.005 |
| advantage on cue-naming answers, per answer | s42_frozen_round2 | 0.002 | -0.000 | 0.000 | 0.004 |
| advantage on cue-naming answers, per answer | s42_frozen_round3 | -0.002 | 0.000 | 0.001 | -0.001 |
| advantage on cue-naming answers, per answer | s42_frozen_round4 | -0.000 | 0.003 | 0.001 | 0.000 |
| advantage on cue-naming answers, per answer | s42_frozen_round5 | -0.002 | 0.003 | 0.004 | -0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s42 | 0.086 | 0.076 | 0.072 | 0.068 |
| gradient norm (W&B, before clipping) | s42_frozen_round2 | 0.063 | 0.060 | 0.056 | 0.046 |
| gradient norm (W&B, before clipping) | s42_frozen_round3 | 0.047 | 0.043 | 0.042 | 0.038 |
| gradient norm (W&B, before clipping) | s42_frozen_round4 | 0.033 | 0.038 | 0.037 | 0.027 |
| gradient norm (W&B, before clipping) | s42_frozen_round5 | 0.022 | 0.043 | 0.030 | 0.025 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s42 | 0.040 | 0.102 | 0.089 | 0.121 |
| KL from the base model (W&B) | s42_frozen_round2 | 0.122 | 0.133 | 0.136 | 0.121 |
| KL from the base model (W&B) | s42_frozen_round3 | 0.125 | 0.125 | 0.117 | 0.104 |
| KL from the base model (W&B) | s42_frozen_round4 | 0.094 | 0.088 | 0.095 | 0.079 |
| KL from the base model (W&B) | s42_frozen_round5 | 0.077 | 0.086 | 0.067 | 0.061 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s42 | 0.431 | 0.416 | 0.463 | 0.456 |
| token entropy (W&B) | s42_frozen_round2 | 0.463 | 0.486 | 0.521 | 0.505 |
| token entropy (W&B) | s42_frozen_round3 | 0.493 | 0.470 | 0.456 | 0.489 |
| token entropy (W&B) | s42_frozen_round4 | 0.491 | 0.495 | 0.518 | 0.514 |
| token entropy (W&B) | s42_frozen_round5 | 0.511 | 0.524 | 0.496 | 0.480 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s42 | 216.026 | 221.667 | 215.874 | 204.649 |
| answer length, tokens (W&B) | s42_frozen_round2 | 198.268 | 217.576 | 228.466 | 232.391 |
| answer length, tokens (W&B) | s42_frozen_round3 | 228.413 | 224.148 | 220.455 | 228.395 |
| answer length, tokens (W&B) | s42_frozen_round4 | 231.729 | 214.889 | 225.212 | 233.306 |
| answer length, tokens (W&B) | s42_frozen_round5 | 242.012 | 228.272 | 217.371 | 220.726 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of kl_weight_1_0_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 145 | 0.067 | 955.814 / 1046.114 | 0.183 / 0.272 | 0.248 / 0.355 | 0.383 / 0.450 |
| 5-9 | 247 | 0.068 | 763.423 / 798.623 | 0.098 / 0.126 | 0.140 / 0.189 | 0.246 / 0.307 |
| 10-29 | 837 | 0.059 | 899.376 / 945.611 | 0.083 / 0.127 | 0.133 / 0.198 | 0.296 / 0.348 |
| 30-59 | 1174 | 0.032 | 894.665 / 937.541 | 0.082 / 0.123 | 0.159 / 0.183 | 0.339 / 0.378 |
| 60-127 | 3054 | 0.020 | 865.850 / 909.493 | 0.086 / 0.118 | 0.183 / 0.208 | 0.347 / 0.382 |

## Probe-only groups of s42_frozen_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 240 | 0.018 | 848.206 / 900.354 | 0.085 / 0.130 | 0.176 / 0.203 | 0.331 / 0.402 |
| 5-9 | 252 | 0.033 | 862.330 / 905.306 | 0.134 / 0.188 | 0.254 / 0.282 | 0.372 / 0.425 |
| 10-29 | 867 | 0.014 | 920.416 / 972.747 | 0.145 / 0.206 | 0.247 / 0.283 | 0.392 / 0.439 |
| 30-59 | 1389 | 0.015 | 979.307 / 1033.647 | 0.141 / 0.184 | 0.241 / 0.272 | 0.413 / 0.441 |
| 60-127 | 3062 | 0.016 | 980.488 / 1049.185 | 0.190 / 0.244 | 0.277 / 0.321 | 0.419 / 0.470 |

## Probe-only groups of s42_frozen_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 240 | 0.018 | 978.036 / 1043.356 | 0.180 / 0.249 | 0.245 / 0.287 | 0.390 / 0.424 |
| 5-9 | 245 | 0.016 | 932.444 / 1009.631 | 0.131 / 0.198 | 0.237 / 0.296 | 0.432 / 0.484 |
| 10-29 | 893 | 0.012 | 919.058 / 1022.221 | 0.123 / 0.197 | 0.230 / 0.291 | 0.420 / 0.493 |
| 30-59 | 1369 | 0.014 | 910.947 / 996.749 | 0.101 / 0.157 | 0.189 / 0.250 | 0.366 / 0.419 |
| 60-127 | 3178 | 0.009 | 964.805 / 1037.006 | 0.087 / 0.142 | 0.181 / 0.246 | 0.402 / 0.461 |

## Probe-only groups of s42_frozen_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 240 | 0.010 | 973.633 / 1047.250 | 0.083 / 0.125 | 0.195 / 0.263 | 0.382 / 0.491 |
| 5-9 | 251 | 0.005 | 962.875 / 1030.205 | 0.075 / 0.150 | 0.155 / 0.238 | 0.392 / 0.490 |
| 10-29 | 906 | 0.005 | 917.914 / 967.369 | 0.061 / 0.109 | 0.137 / 0.208 | 0.389 / 0.432 |
| 30-59 | 1363 | 0.009 | 936.261 / 1001.295 | 0.077 / 0.139 | 0.148 / 0.225 | 0.383 / 0.443 |
| 60-127 | 3112 | 0.007 | 979.911 / 1046.520 | 0.077 / 0.149 | 0.165 / 0.245 | 0.435 / 0.497 |

## Probe-only groups of s42_frozen_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 231 | 0.006 | 988.921 / 1040.208 | 0.082 / 0.152 | 0.195 / 0.260 | 0.411 / 0.492 |
| 5-9 | 237 | 0.007 | 1070.079 / 1138.244 | 0.073 / 0.168 | 0.140 / 0.220 | 0.474 / 0.504 |
| 10-29 | 866 | 0.005 | 953.451 / 1022.195 | 0.103 / 0.173 | 0.170 / 0.258 | 0.404 / 0.475 |
| 30-59 | 1339 | 0.006 | 910.037 / 971.426 | 0.079 / 0.165 | 0.148 / 0.243 | 0.398 / 0.462 |
| 60-127 | 3125 | 0.006 | 921.068 / 985.236 | 0.073 / 0.163 | 0.137 / 0.237 | 0.383 / 0.446 |

## kl_weight_1_0_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.146 | 0.021 | 1030.000 |
| 10 | 0.012 | 0.006 | 739.500 |
| 20 | 0.020 | 0.002 | 951.500 |
| 25 | 0.033 | 0.010 | 1083.000 |
| 30 | 0.078 | 0.033 | 1031.000 |
| 60 | 0.027 | 0.016 | 1000.000 |
| 70 | 0.025 | 0.010 | 920.500 |
| 80 | 0.025 | 0.023 | 927.000 |
| 90 | 0.025 | 0.012 | 842.500 |
| 100 | 0.018 | 0.014 | 809.500 |
| 127 | 0.012 | 0.004 | 817.500 |

## s42_frozen_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.014 | 0.008 | 821.500 |
| 10 | 0.006 | 0.004 | 824.500 |
| 20 | 0.041 | 0 | 920.000 |
| 25 | 0.033 | 0.020 | 967.000 |
| 30 | 0.035 | 0.014 | 1062.000 |
| 60 | 0.018 | 0 | 877.000 |
| 70 | 0.051 | 0.018 | 966.500 |
| 80 | 0.049 | 0.037 | 1014.000 |
| 90 | 0.055 | 0.027 | 1029.500 |
| 100 | 0.062 | 0.010 | 1028.000 |
| 127 | 0.080 | 0.016 | 980.500 |

## s42_frozen_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.062 | 0.021 | 960.500 |
| 10 | 0.037 | 0.010 | 940.500 |
| 20 | 0.090 | 0.023 | 1032.500 |
| 25 | 0.133 | 0.002 | 954.000 |
| 30 | 0.092 | 0.018 | 1018.000 |
| 60 | 0.129 | 0 | 809.500 |
| 70 | 0.158 | 0.020 | 980.500 |
| 80 | 0.076 | 0.023 | 999.500 |
| 90 | 0.029 | 0.006 | 999.000 |
| 100 | 0.072 | 0.002 | 943.500 |
| 127 | 0.086 | 0 | 956.500 |

## s42_frozen_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.059 | 0.021 | 936.000 |
| 10 | 0.158 | 0.004 | 938.500 |
| 20 | 0.049 | 0.004 | 901.000 |
| 25 | 0.029 | 0.002 | 966.500 |
| 30 | 0.045 | 0.004 | 946.000 |
| 60 | 0.035 | 0.008 | 891.000 |
| 70 | 0.043 | 0.014 | 929.000 |
| 80 | 0.043 | 0.012 | 1034.500 |
| 90 | 0.074 | 0.012 | 978.500 |
| 100 | 0.068 | 0.023 | 926.500 |
| 127 | 0.051 | 0.006 | 948.500 |

## s42_frozen_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.047 | 0.010 | 989.500 |
| 10 | 0.064 | 0.008 | 1107.500 |
| 20 | 0.059 | 0.018 | 950.000 |
| 25 | 0.094 | 0.018 | 965.000 |
| 30 | 0.088 | 0.029 | 902.500 |
| 60 | 0.016 | 0.027 | 878.000 |
| 70 | 0.041 | 0.002 | 933.500 |
| 80 | 0.064 | 0.008 | 898.000 |
| 90 | 0.070 | 0.006 | 919.000 |
| 100 | 0.074 | 0.006 | 891.500 |
| 127 | 0.123 | 0.004 | 956.500 |

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
  "kl_weight_1_0_s42": {
    "median_tokens_no_cue": 168.0,
    "median_tokens_cued": 181.0,
    "mean_top_letter_share_of_8_no_cue": 0.9309815950920245,
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
      "A": 178,
      "B": 300
    },
    "questions_with_no_cue_answer": 478
  },
  "s42_frozen_round2": {
    "median_tokens_no_cue": 202.0,
    "median_tokens_cued": 220.5,
    "mean_top_letter_share_of_8_no_cue": 0.9386503067484663,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 194,
      "B": 291,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 179,
      "B": 295
    },
    "questions_with_no_cue_answer": 474
  },
  "s42_frozen_round3": {
    "median_tokens_no_cue": 200.0,
    "median_tokens_cued": 213.0,
    "mean_top_letter_share_of_8_no_cue": 0.962678936605317,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 168,
      "B": 319,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 159,
      "B": 322
    },
    "questions_with_no_cue_answer": 481
  },
  "s42_frozen_round4": {
    "median_tokens_no_cue": 201.0,
    "median_tokens_cued": 216.0,
    "mean_top_letter_share_of_8_no_cue": 0.9654907975460123,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 178,
      "B": 308,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 180,
      "B": 299
    },
    "questions_with_no_cue_answer": 479
  },
  "s42_frozen_round5": {
    "median_tokens_no_cue": 193.0,
    "median_tokens_cued": 201.0,
    "mean_top_letter_share_of_8_no_cue": 0.9792944785276073,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 158,
      "B": 331,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 155,
      "B": 330
    },
    "questions_with_no_cue_answer": 485
  }
}
```

## kl_weight_1_0_s42: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 6 | 135 / 210 | 0.765 / 0.219 | 0.931 (0.931) | 0.727 | 0.673 | 0.3897 / 0.1851 | 0.920 / 0.883 |
| A | cue_dist | 75 / 45 | 21 / 61 | 0.743 / 0.271 | 0.875 (0.875) | 0.684 | 0.823 | 0.7575 / 0.3640 | 0.877 / 0.811 |
| A | both | 108 / 51 | 156 / 271 | 0.748 / 0.229 | 0.907 | 0.657 | 0.716 | 0.5234 / 0.2440 | 0.926 / 0.846 |
| B | cue_gold | 44 / 16 | 142 / 224 | 0.216 / 0.048 | 0.835 (0.835) | 0.673 | 0.814 | 0.1154 / 0.0374 | 0.874 / 0.722 |
| B | cue_dist | 100 / 61 | 22 / 66 | 0.206 / 0.051 | 0.767 (0.767) | 0.686 | 0.860 | 0.2349 / 0.0704 | 0.785 / 0.603 |
| B | both | 144 / 77 | 164 / 290 | 0.209 / 0.048 | 0.791 | 0.645 | 0.831 | 0.1627 / 0.0488 | 0.857 / 0.664 |
| C | cue_gold | 44 / 16 | 142 / 224 | 0.214 / 0.053 | 0.824 (0.824) | 0.673 | 0.841 | 0.1233 / 0.0394 | 0.860 / 0.733 |
| C | cue_dist | 100 / 61 | 22 / 66 | 0.195 / 0.046 | 0.789 (0.789) | 0.686 | 0.886 | 0.2214 / 0.0657 | 0.800 / 0.647 |
| C | both | 144 / 77 | 164 / 290 | 0.200 / 0.052 | 0.790 | 0.645 | 0.858 | 0.1621 / 0.0485 | 0.850 / 0.683 |
| D | cue_gold | 44 / 16 | 142 / 224 | 0.208 / 0.054 | 0.815 (0.815) | 0.673 | 0.857 | 0.1239 / 0.0391 | 0.853 / 0.715 |
| D | cue_dist | 100 / 61 | 22 / 66 | 0.184 / 0.049 | 0.776 (0.776) | 0.686 | 0.878 | 0.2091 / 0.0663 | 0.824 / 0.612 |
| D | both | 144 / 77 | 164 / 290 | 0.191 / 0.053 | 0.781 | 0.645 | 0.864 | 0.1577 / 0.0485 | 0.849 / 0.668 |

## s42_frozen_round2: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 134 / 208 | 0.759 / 0.216 | 0.929 (0.929) | 0.729 | 0.673 | 0.3875 / 0.1833 | 0.915 / 0.885 |
| A | cue_dist | 75 / 43 | 21 / 63 | 0.749 / 0.274 | 0.878 (0.878) | 0.693 | 0.823 | 0.7575 / 0.3644 | 0.877 / 0.808 |
| A | both | 109 / 49 | 155 / 271 | 0.751 / 0.228 | 0.909 | 0.663 | 0.715 | 0.5220 / 0.2433 | 0.926 / 0.846 |
| B | cue_gold | 34 / 9 | 140 / 216 | 0.219 / 0.051 | 0.825 (0.825) | 0.699 | 0.848 | 0.1220 / 0.0282 | 0.846 / 0.669 |
| B | cue_dist | 67 / 30 | 26 / 65 | 0.262 / 0.066 | 0.795 (0.795) | 0.703 | 0.880 | 0.2812 / 0.0549 | 0.761 / 0.692 |
| B | both | 101 / 39 | 166 / 281 | 0.248 / 0.054 | 0.817 | 0.675 | 0.857 | 0.1775 / 0.0361 | 0.829 / 0.715 |
| C | cue_gold | 34 / 9 | 140 / 216 | 0.220 / 0.044 | 0.861 (0.861) | 0.699 | 0.789 | 0.1084 / 0.0277 | 0.897 / 0.693 |
| C | cue_dist | 67 / 30 | 26 / 65 | 0.252 / 0.057 | 0.799 (0.799) | 0.703 | 0.875 | 0.2705 / 0.0478 | 0.790 / 0.685 |
| C | both | 101 / 39 | 166 / 281 | 0.242 / 0.047 | 0.836 | 0.675 | 0.813 | 0.1649 / 0.0337 | 0.866 / 0.717 |
| D | cue_gold | 34 / 9 | 140 / 216 | 0.289 / 0.084 | 0.817 (0.817) | 0.699 | 0.884 | 0.1823 / 0.0465 | 0.881 / 0.497 |
| D | cue_dist | 67 / 30 | 26 / 65 | 0.304 / 0.098 | 0.782 (0.782) | 0.703 | 0.884 | 0.3324 / 0.0793 | 0.733 / 0.685 |
| D | both | 101 / 39 | 166 / 281 | 0.299 / 0.087 | 0.804 | 0.675 | 0.880 | 0.2346 / 0.0562 | 0.824 / 0.670 |

## s42_frozen_round3: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 138 / 210 | 0.741 / 0.223 | 0.919 (0.919) | 0.727 | 0.676 | 0.3914 / 0.1845 | 0.907 / 0.860 |
| A | cue_dist | 74 / 45 | 19 / 62 | 0.737 / 0.265 | 0.873 (0.873) | 0.694 | 0.827 | 0.7600 / 0.3595 | 0.876 / 0.796 |
| A | both | 108 / 51 | 157 / 272 | 0.738 / 0.231 | 0.900 | 0.657 | 0.716 | 0.5208 / 0.2425 | 0.922 / 0.833 |
| B | cue_gold | 36 / 8 | 126 / 221 | 0.347 / 0.089 | 0.833 (0.833) | 0.728 | 0.727 | 0.1856 / 0.0694 | 0.834 / 0.649 |
| B | cue_dist | 40 / 20 | 23 / 83 | 0.326 / 0.155 | 0.706 (0.706) | 0.725 | 0.782 | 0.3393 / 0.1414 | 0.662 / 0.573 |
| B | both | 76 / 28 | 149 / 304 | 0.334 / 0.104 | 0.788 | 0.701 | 0.736 | 0.2286 / 0.0918 | 0.801 / 0.638 |
| C | cue_gold | 36 / 8 | 126 / 221 | 0.214 / 0.025 | 0.885 (0.885) | 0.728 | 0.608 | 0.0744 / 0.0266 | 0.911 / 0.758 |
| C | cue_dist | 40 / 20 | 23 / 83 | 0.162 / 0.049 | 0.761 (0.761) | 0.725 | 0.796 | 0.1395 / 0.0602 | 0.726 / 0.648 |
| C | both | 76 / 28 | 149 / 304 | 0.184 / 0.031 | 0.844 | 0.701 | 0.654 | 0.0926 / 0.0370 | 0.880 / 0.718 |
| D | cue_gold | 36 / 8 | 126 / 221 | 0.314 / 0.126 | 0.795 (0.795) | 0.728 | 0.809 | 0.2319 / 0.0868 | 0.740 / 0.656 |
| D | cue_dist | 40 / 20 | 23 / 83 | 0.277 / 0.126 | 0.719 (0.719) | 0.725 | 0.869 | 0.3016 / 0.1068 | 0.599 / 0.574 |
| D | both | 76 / 28 | 149 / 304 | 0.293 / 0.126 | 0.754 | 0.701 | 0.823 | 0.2514 / 0.0930 | 0.708 / 0.616 |

## s42_frozen_round4: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 5 | 137 / 210 | 0.742 / 0.224 | 0.919 (0.919) | 0.740 | 0.680 | 0.3963 / 0.1823 | 0.908 / 0.830 |
| A | cue_dist | 75 / 44 | 20 / 63 | 0.740 / 0.277 | 0.870 (0.870) | 0.695 | 0.834 | 0.7662 / 0.3572 | 0.871 / 0.796 |
| A | both | 110 / 49 | 157 / 273 | 0.740 / 0.234 | 0.900 | 0.663 | 0.722 | 0.5279 / 0.2404 | 0.921 / 0.828 |
| B | cue_gold | 27 / 7 | 135 / 212 | 0.295 / 0.110 | 0.760 (0.760) | 0.703 | 0.710 | 0.2014 / 0.0717 | 0.781 / 0.454 |
| B | cue_dist | 26 / 10 | 31 / 74 | 0.348 / 0.157 | 0.728 (0.728) | 0.713 | 0.873 | 0.3733 / 0.0924 | 0.586 / 0.642 |
| B | both | 53 / 17 | 166 / 286 | 0.322 / 0.121 | 0.763 | 0.695 | 0.745 | 0.2461 / 0.0774 | 0.744 / 0.576 |
| C | cue_gold | 27 / 7 | 135 / 212 | 0.144 / 0.046 | 0.811 (0.811) | 0.703 | 0.588 | 0.0737 / 0.0404 | 0.853 / 0.584 |
| C | cue_dist | 26 / 10 | 31 / 74 | 0.187 / 0.067 | 0.798 (0.798) | 0.713 | 0.831 | 0.1815 / 0.0402 | 0.725 / 0.697 |
| C | both | 53 / 17 | 166 / 286 | 0.166 / 0.051 | 0.820 | 0.695 | 0.644 | 0.1018 / 0.0403 | 0.838 / 0.662 |
| D | cue_gold | 27 / 7 | 135 / 212 | 0.312 / 0.182 | 0.704 (0.704) | 0.703 | 0.833 | 0.3087 / 0.1089 | 0.655 / 0.385 |
| D | cue_dist | 26 / 10 | 31 / 74 | 0.347 / 0.200 | 0.705 (0.705) | 0.713 | 0.899 | 0.4125 / 0.1187 | 0.531 / 0.622 |
| D | both | 53 / 17 | 166 / 286 | 0.330 / 0.186 | 0.716 | 0.695 | 0.848 | 0.3357 / 0.1116 | 0.638 / 0.535 |

## s42_frozen_round5: probe rows by answer letter (463 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 137 / 212 | 0.753 / 0.223 | 0.924 (0.924) | 0.722 | 0.673 | 0.3952 / 0.1916 | 0.914 / 0.875 |
| A | cue_dist | 76 / 46 | 21 / 64 | 0.739 / 0.269 | 0.874 (0.874) | 0.688 | 0.826 | 0.7572 / 0.3599 | 0.880 / 0.801 |
| A | both | 112 / 53 | 158 / 276 | 0.743 / 0.232 | 0.902 | 0.657 | 0.715 | 0.5252 / 0.2479 | 0.925 / 0.836 |
| B | cue_gold | 19 / 4 | 121 / 217 | 0.202 / 0.128 | 0.704 (0.704) | 0.734 | 0.620 | 0.1833 / 0.1007 | 0.619 / 0.694 |
| B | cue_dist | 15 / 10 | 26 / 88 | 0.327 / 0.225 | 0.629 (0.629) | 0.686 | 0.783 | 0.4523 / 0.1566 | 0.423 / 0.611 |
| B | both | 34 / 14 | 147 / 305 | 0.267 / 0.153 | 0.698 | 0.692 | 0.641 | 0.2442 / 0.1179 | 0.613 / 0.688 |
| C | cue_gold | 19 / 4 | 121 / 217 | 0.115 / 0.048 | 0.778 (0.778) | 0.734 | 0.530 | 0.0677 / 0.0426 | 0.742 / 0.811 |
| C | cue_dist | 15 / 10 | 26 / 88 | 0.219 / 0.089 | 0.731 (0.731) | 0.686 | 0.715 | 0.1934 / 0.0790 | 0.592 / 0.735 |
| C | both | 34 / 14 | 147 / 305 | 0.169 / 0.058 | 0.778 | 0.692 | 0.562 | 0.0961 / 0.0538 | 0.731 / 0.793 |
| D | cue_gold | 19 / 4 | 121 / 217 | 0.255 / 0.212 | 0.645 (0.645) | 0.734 | 0.774 | 0.3309 / 0.1410 | 0.464 / 0.598 |
| D | cue_dist | 15 / 10 | 26 / 88 | 0.275 / 0.240 | 0.554 (0.554) | 0.686 | 0.816 | 0.4572 / 0.1584 | 0.338 / 0.517 |
| D | both | 34 / 14 | 147 / 305 | 0.265 / 0.219 | 0.612 | 0.692 | 0.777 | 0.3595 / 0.1464 | 0.454 / 0.586 |

## Probe data: answers naming A

```
{
  "kl_weight_1_0_s42": {
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
  "s42_frozen_round2": {
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
  "s42_frozen_round3": {
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
  "s42_frozen_round4": {
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
  "s42_frozen_round5": {
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
  "kl_weight_1_0_s42": {
    "13": 0.8931230306625366,
    "14": 0.8778945207595825,
    "15": 0.8701239824295044,
    "16": 0.8567250967025757
  },
  "s42_frozen_round2": {
    "13": 0.8919538855552673,
    "14": 0.874825119972229,
    "15": 0.8624777793884277,
    "16": 0.8656126260757446
  },
  "s42_frozen_round3": {
    "13": 0.9253938794136047,
    "14": 0.907927930355072,
    "15": 0.8909762501716614,
    "16": 0.8757968544960022
  },
  "s42_frozen_round4": {
    "13": 0.9338667988777161,
    "14": 0.9190326929092407,
    "15": 0.8992105722427368,
    "16": 0.8871314525604248
  },
  "s42_frozen_round5": {
    "13": 0.9579461216926575,
    "14": 0.9490455389022827,
    "15": 0.9355072975158691,
    "16": 0.9241673946380615
  }
}
```
