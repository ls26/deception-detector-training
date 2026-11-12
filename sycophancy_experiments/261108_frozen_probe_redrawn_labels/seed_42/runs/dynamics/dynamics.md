# Training dynamics and probe rows: kl_weight_1_0_s42, s42_redrawn_round2, s42_redrawn_round3, s42_redrawn_round4, s42_redrawn_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s42 0.3, s42_redrawn_round2 0.3, s42_redrawn_round3 0.3, s42_redrawn_round4 0.3, s42_redrawn_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s42 | 39.200 | 41.850 | 39.133 | 44.912 |
| groups where all 8 answers name the cue (of 64) | s42_redrawn_round2 | 44.300 | 43.150 | 43.567 | 46.691 |
| groups where all 8 answers name the cue (of 64) | s42_redrawn_round3 | 47.600 | 52.800 | 49.367 | 48.691 |
| groups where all 8 answers name the cue (of 64) | s42_redrawn_round4 | 50.400 | 47.500 | 46.900 | 48.897 |
| groups where all 8 answers name the cue (of 64) | s42_redrawn_round5 | 46.500 | 47.050 | 43.933 | 46.574 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s42 | 20.700 | 16.200 | 18.167 | 13.750 |
| groups with mixed task rewards (of 64) | s42_redrawn_round2 | 14 | 15.650 | 14.133 | 11.794 |
| groups with mixed task rewards (of 64) | s42_redrawn_round3 | 11.900 | 9.650 | 12.067 | 11.574 |
| groups with mixed task rewards (of 64) | s42_redrawn_round4 | 9 | 11.850 | 10.833 | 8.456 |
| groups with mixed task rewards (of 64) | s42_redrawn_round5 | 10.500 | 10 | 11.333 | 8.338 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s42 | 4.800 | 6.500 | 6.867 | 5.985 |
|   of which the cue is not at the training label | s42_redrawn_round2 | 5.100 | 7.250 | 6 | 5.515 |
|   of which the cue is not at the training label | s42_redrawn_round3 | 4.600 | 4.300 | 4.967 | 4.735 |
|   of which the cue is not at the training label | s42_redrawn_round4 | 3.500 | 5.250 | 4.533 | 3.897 |
|   of which the cue is not at the training label | s42_redrawn_round5 | 4.500 | 4.900 | 4.800 | 4.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s42 | 88.600 | 101.250 | 116.733 | 87.779 |
| answers not naming the cue (of 512) | s42_redrawn_round2 | 96.800 | 92.200 | 96.900 | 79.662 |
| answers not naming the cue (of 512) | s42_redrawn_round3 | 80.600 | 35.850 | 52.467 | 61.529 |
| answers not naming the cue (of 512) | s42_redrawn_round4 | 67.100 | 73.400 | 84.100 | 78.382 |
| answers not naming the cue (of 512) | s42_redrawn_round5 | 87.600 | 89.650 | 104.033 | 100.603 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s42 | 0.322 | 0.289 | 0.291 | 0.244 |
| mean absolute advantage per answer | s42_redrawn_round2 | 0.254 | 0.267 | 0.262 | 0.284 |
| mean absolute advantage per answer | s42_redrawn_round3 | 0.263 | 0.275 | 0.231 | 0.223 |
| mean absolute advantage per answer | s42_redrawn_round4 | 0.196 | 0.220 | 0.203 | 0.163 |
| mean absolute advantage per answer | s42_redrawn_round5 | 0.181 | 0.184 | 0.208 | 0.180 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s42 | 0.136 | 0.130 | 0.094 | 0.082 |
|   from probe-only groups | s42_redrawn_round2 | 0.089 | 0.084 | 0.097 | 0.149 |
|   from probe-only groups | s42_redrawn_round3 | 0.128 | 0.160 | 0.091 | 0.087 |
|   from probe-only groups | s42_redrawn_round4 | 0.089 | 0.082 | 0.078 | 0.066 |
|   from probe-only groups | s42_redrawn_round5 | 0.064 | 0.067 | 0.077 | 0.084 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s42 | 0.024 | -0.004 | 0.010 | 0.005 |
| advantage on cue-naming answers, per answer | s42_redrawn_round2 | 0.015 | 0.002 | 0.008 | -0.000 |
| advantage on cue-naming answers, per answer | s42_redrawn_round3 | 0.014 | 0.004 | 0.009 | 0.009 |
| advantage on cue-naming answers, per answer | s42_redrawn_round4 | 0.008 | 0.003 | 0.009 | 0.002 |
| advantage on cue-naming answers, per answer | s42_redrawn_round5 | 0.005 | -0.002 | 0.007 | -0.002 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s42 | 0.086 | 0.076 | 0.072 | 0.068 |
| gradient norm (W&B, before clipping) | s42_redrawn_round2 | 0.067 | 0.073 | 0.062 | 0.058 |
| gradient norm (W&B, before clipping) | s42_redrawn_round3 | 0.057 | 0.058 | 0.053 | 0.053 |
| gradient norm (W&B, before clipping) | s42_redrawn_round4 | 0.048 | 0.049 | 0.050 | 0.044 |
| gradient norm (W&B, before clipping) | s42_redrawn_round5 | 0.047 | 0.043 | 0.048 | 0.042 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s42 | 0.040 | 0.102 | 0.089 | 0.121 |
| KL from the base model (W&B) | s42_redrawn_round2 | 0.128 | 0.142 | 0.180 | 0.303 |
| KL from the base model (W&B) | s42_redrawn_round3 | 0.257 | 0.356 | 0.277 | 0.245 |
| KL from the base model (W&B) | s42_redrawn_round4 | 0.222 | 0.229 | 0.246 | 0.222 |
| KL from the base model (W&B) | s42_redrawn_round5 | 0.237 | 0.230 | 0.232 | 0.246 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s42 | 0.431 | 0.416 | 0.463 | 0.456 |
| token entropy (W&B) | s42_redrawn_round2 | 0.457 | 0.478 | 0.448 | 0.525 |
| token entropy (W&B) | s42_redrawn_round3 | 0.529 | 0.507 | 0.549 | 0.551 |
| token entropy (W&B) | s42_redrawn_round4 | 0.545 | 0.549 | 0.492 | 0.520 |
| token entropy (W&B) | s42_redrawn_round5 | 0.536 | 0.545 | 0.488 | 0.522 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s42 | 216.026 | 221.667 | 215.874 | 204.649 |
| answer length, tokens (W&B) | s42_redrawn_round2 | 208.100 | 247.104 | 216.859 | 245.662 |
| answer length, tokens (W&B) | s42_redrawn_round3 | 250.304 | 237.241 | 250.581 | 235.713 |
| answer length, tokens (W&B) | s42_redrawn_round4 | 238.341 | 252.925 | 232.901 | 239.148 |
| answer length, tokens (W&B) | s42_redrawn_round5 | 236.614 | 251.219 | 238.293 | 239.886 |
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

## Probe-only groups of s42_redrawn_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 228 | 0.017 | 850.628 / 919.537 | 0.093 / 0.128 | 0.178 / 0.232 | 0.359 / 0.399 |
| 5-9 | 215 | 0.026 | 926.035 / 964.274 | 0.170 / 0.221 | 0.283 / 0.301 | 0.376 / 0.414 |
| 10-29 | 863 | 0.016 | 1058.631 / 1126.851 | 0.155 / 0.210 | 0.230 / 0.278 | 0.406 / 0.445 |
| 30-59 | 1307 | 0.036 | 895.417 / 985.923 | 0.096 / 0.131 | 0.188 / 0.234 | 0.377 / 0.421 |
| 60-127 | 3175 | 0.057 | 1049.254 / 1096.766 | 0.137 / 0.166 | 0.242 / 0.265 | 0.438 / 0.470 |

## Probe-only groups of s42_redrawn_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 210 | 0.049 | 1071.276 / 1152.930 | 0.161 / 0.193 | 0.271 / 0.290 | 0.468 / 0.494 |
| 5-9 | 266 | 0.042 | 1047.594 / 1109.600 | 0.191 / 0.191 | 0.295 / 0.317 | 0.457 / 0.481 |
| 10-29 | 1056 | 0.044 | 1010.674 / 1059.622 | 0.160 / 0.171 | 0.252 / 0.272 | 0.458 / 0.507 |
| 30-59 | 1481 | 0.020 | 1074.882 / 1134.514 | 0.130 / 0.163 | 0.230 / 0.254 | 0.470 / 0.530 |
| 60-127 | 3311 | 0.019 | 1002.565 / 1064.735 | 0.110 / 0.141 | 0.202 / 0.231 | 0.407 / 0.456 |

## Probe-only groups of s42_redrawn_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 259 | 0.018 | 976.098 / 1028.063 | 0.118 / 0.152 | 0.214 / 0.260 | 0.372 / 0.425 |
| 5-9 | 245 | 0.023 | 1067.385 / 1140.320 | 0.131 / 0.132 | 0.196 / 0.238 | 0.415 / 0.464 |
| 10-29 | 950 | 0.020 | 1071.904 / 1149.383 | 0.144 / 0.190 | 0.237 / 0.283 | 0.454 / 0.508 |
| 30-59 | 1407 | 0.016 | 974.743 / 1043.599 | 0.140 / 0.198 | 0.243 / 0.305 | 0.411 / 0.463 |
| 60-127 | 3325 | 0.012 | 997.415 / 1079.479 | 0.140 / 0.216 | 0.252 / 0.316 | 0.425 / 0.484 |

## Probe-only groups of s42_redrawn_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 235 | 0.011 | 962.739 / 1051.065 | 0.144 / 0.206 | 0.261 / 0.314 | 0.409 / 0.506 |
| 5-9 | 230 | 0.013 | 1007.454 / 1075.139 | 0.147 / 0.212 | 0.302 / 0.346 | 0.436 / 0.474 |
| 10-29 | 941 | 0.017 | 1059.381 / 1125.057 | 0.145 / 0.188 | 0.245 / 0.289 | 0.440 / 0.508 |
| 30-59 | 1318 | 0.020 | 984.304 / 1039.319 | 0.166 / 0.219 | 0.267 / 0.304 | 0.361 / 0.409 |
| 60-127 | 3167 | 0.018 | 1012.899 / 1074.921 | 0.136 / 0.174 | 0.235 / 0.276 | 0.391 / 0.437 |

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

## s42_redrawn_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.041 | 0.008 | 859.000 |
| 10 | 0.314 | 0.020 | 1095.500 |
| 20 | 0.373 | 0.004 | 1101.500 |
| 25 | 0.156 | 0.008 | 971.000 |
| 30 | 0.092 | 0.010 | 918.500 |
| 60 | 0.021 | 0.004 | 899.500 |
| 70 | 0.029 | 0.004 | 941.000 |
| 80 | 0.059 | 0.029 | 953.000 |
| 90 | 0.045 | 0.043 | 1096.500 |
| 100 | 0.023 | 0.070 | 1108.500 |
| 127 | 0.053 | 0.059 | 1099.500 |

## s42_redrawn_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.037 | 0.061 | 1053.500 |
| 10 | 0.018 | 0.082 | 923.500 |
| 20 | 0.027 | 0.023 | 980.500 |
| 25 | 0.025 | 0.025 | 1055.500 |
| 30 | 0.010 | 0.047 | 1071.000 |
| 60 | 0.043 | 0.084 | 1168.500 |
| 70 | 0.055 | 0.082 | 1125.500 |
| 80 | 0.051 | 0.078 | 1009.000 |
| 90 | 0.072 | 0.051 | 924.000 |
| 100 | 0.049 | 0.062 | 958.500 |
| 127 | 0.053 | 0.049 | 976.000 |

## s42_redrawn_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.061 | 0.039 | 928.500 |
| 10 | 0.023 | 0.033 | 1022.000 |
| 20 | 0.021 | 0.041 | 1189.000 |
| 25 | 0.006 | 0.045 | 992.500 |
| 30 | 0.033 | 0.025 | 912.000 |
| 60 | 0.041 | 0.010 | 963.000 |
| 70 | 0.070 | 0.016 | 970.500 |
| 80 | 0.061 | 0.037 | 1053.000 |
| 90 | 0.043 | 0.020 | 961.000 |
| 100 | 0.049 | 0.057 | 1057.000 |
| 127 | 0.057 | 0.023 | 1016.000 |

## s42_redrawn_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.059 | 0.020 | 972.000 |
| 10 | 0.100 | 0.020 | 1103.000 |
| 20 | 0.025 | 0.014 | 1030.000 |
| 25 | 0.047 | 0.002 | 1059.500 |
| 30 | 0.121 | 0.008 | 1040.500 |
| 60 | 0.033 | 0.012 | 902.500 |
| 70 | 0.020 | 0.025 | 950.500 |
| 80 | 0.016 | 0.049 | 1018.500 |
| 90 | 0.027 | 0.076 | 1024.500 |
| 100 | 0.039 | 0.051 | 1022.000 |
| 127 | 0.047 | 0.055 | 1020.000 |

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
  "s42_redrawn_round2": {
    "median_tokens_no_cue": 223.0,
    "median_tokens_cued": 240.0,
    "mean_top_letter_share_of_8_no_cue": 0.9074642126789366,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 202,
      "B": 282,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 191,
      "B": 279
    },
    "questions_with_no_cue_answer": 470
  },
  "s42_redrawn_round3": {
    "median_tokens_no_cue": 193.0,
    "median_tokens_cued": 209.5,
    "mean_top_letter_share_of_8_no_cue": 0.9245910020449898,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 184,
      "B": 300,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 171,
      "B": 303
    },
    "questions_with_no_cue_answer": 474
  },
  "s42_redrawn_round4": {
    "median_tokens_no_cue": 205.0,
    "median_tokens_cued": 221.0,
    "mean_top_letter_share_of_8_no_cue": 0.9294478527607362,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 179,
      "B": 309,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 167,
      "B": 309
    },
    "questions_with_no_cue_answer": 476
  },
  "s42_redrawn_round5": {
    "median_tokens_no_cue": 215.0,
    "median_tokens_cued": 225.0,
    "mean_top_letter_share_of_8_no_cue": 0.924079754601227,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 191,
      "B": 292,
      "null": 6
    },
    "no_cue_answer_letters": {
      "A": 182,
      "B": 290
    },
    "questions_with_no_cue_answer": 472
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

## s42_redrawn_round2: probe rows by answer letter (449 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 32 / 7 | 134 / 205 | 0.757 / 0.220 | 0.926 (0.926) | 0.713 | 0.665 | 0.3865 / 0.1886 | 0.916 / 0.879 |
| A | cue_dist | 73 / 43 | 22 / 60 | 0.750 / 0.262 | 0.886 (0.886) | 0.681 | 0.826 | 0.7572 / 0.3553 | 0.889 / 0.826 |
| A | both | 105 / 50 | 156 / 265 | 0.752 / 0.228 | 0.910 | 0.653 | 0.714 | 0.5214 / 0.2431 | 0.929 / 0.853 |
| B | cue_gold | 42 / 21 | 147 / 211 | 0.253 / 0.077 | 0.787 (0.787) | 0.628 | 0.831 | 0.1755 / 0.0441 | 0.844 / 0.664 |
| B | cue_dist | 119 / 65 | 29 / 56 | 0.242 / 0.083 | 0.761 (0.761) | 0.653 | 0.751 | 0.2371 / 0.1355 | 0.746 / 0.699 |
| B | both | 161 / 86 | 176 / 267 | 0.244 / 0.078 | 0.781 | 0.627 | 0.808 | 0.2026 / 0.0754 | 0.773 / 0.724 |
| C | cue_gold | 42 / 21 | 147 / 211 | 0.219 / 0.081 | 0.758 (0.758) | 0.628 | 0.803 | 0.1599 / 0.0546 | 0.779 / 0.664 |
| C | cue_dist | 119 / 65 | 29 / 56 | 0.214 / 0.085 | 0.746 (0.746) | 0.653 | 0.734 | 0.2035 / 0.1368 | 0.682 / 0.719 |
| C | both | 161 / 86 | 176 / 267 | 0.216 / 0.082 | 0.757 | 0.627 | 0.786 | 0.1791 / 0.0828 | 0.729 / 0.717 |
| D | cue_gold | 42 / 21 | 147 / 211 | 0.261 / 0.123 | 0.731 (0.731) | 0.628 | 0.864 | 0.2301 / 0.0738 | 0.745 / 0.632 |
| D | cue_dist | 119 / 65 | 29 / 56 | 0.260 / 0.115 | 0.778 (0.778) | 0.653 | 0.821 | 0.2743 / 0.1407 | 0.736 / 0.751 |
| D | both | 161 / 86 | 176 / 267 | 0.260 / 0.122 | 0.755 | 0.627 | 0.853 | 0.2495 / 0.0967 | 0.730 / 0.710 |

## s42_redrawn_round3: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 137 / 210 | 0.757 / 0.225 | 0.925 (0.925) | 0.715 | 0.677 | 0.3956 / 0.1886 | 0.916 / 0.878 |
| A | cue_dist | 74 / 46 | 22 / 60 | 0.743 / 0.268 | 0.878 (0.878) | 0.674 | 0.829 | 0.7604 / 0.3602 | 0.892 / 0.806 |
| A | both | 107 / 53 | 159 / 270 | 0.747 / 0.233 | 0.906 | 0.649 | 0.720 | 0.5272 / 0.2449 | 0.931 / 0.840 |
| B | cue_gold | 62 / 19 | 139 / 222 | 0.248 / 0.068 | 0.819 (0.819) | 0.690 | 0.805 | 0.1619 / 0.0506 | 0.814 / 0.663 |
| B | cue_dist | 151 / 85 | 22 / 71 | 0.235 / 0.106 | 0.707 (0.707) | 0.702 | 0.836 | 0.2915 / 0.0956 | 0.635 / 0.586 |
| B | both | 213 / 104 | 161 / 293 | 0.238 / 0.076 | 0.777 | 0.659 | 0.818 | 0.2218 / 0.0683 | 0.785 / 0.664 |
| C | cue_gold | 62 / 19 | 139 / 222 | 0.132 / 0.028 | 0.795 (0.795) | 0.690 | 0.846 | 0.0749 / 0.0237 | 0.754 / 0.669 |
| C | cue_dist | 151 / 85 | 22 / 71 | 0.096 / 0.051 | 0.706 (0.706) | 0.702 | 0.857 | 0.1225 / 0.0402 | 0.669 / 0.578 |
| C | both | 213 / 104 | 161 / 293 | 0.106 / 0.033 | 0.757 | 0.659 | 0.850 | 0.0969 / 0.0302 | 0.747 / 0.658 |
| D | cue_gold | 62 / 19 | 139 / 222 | 0.202 / 0.069 | 0.795 (0.795) | 0.690 | 0.865 | 0.1525 / 0.0448 | 0.768 / 0.635 |
| D | cue_dist | 151 / 85 | 22 / 71 | 0.164 / 0.085 | 0.703 (0.703) | 0.702 | 0.874 | 0.2073 / 0.0689 | 0.694 / 0.545 |
| D | both | 213 / 104 | 161 / 293 | 0.174 / 0.073 | 0.750 | 0.659 | 0.867 | 0.1779 / 0.0543 | 0.746 / 0.639 |

## s42_redrawn_round4: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 133 / 210 | 0.762 / 0.221 | 0.928 (0.928) | 0.721 | 0.667 | 0.3916 / 0.1922 | 0.924 / 0.874 |
| A | cue_dist | 75 / 43 | 21 / 62 | 0.748 / 0.263 | 0.882 (0.882) | 0.691 | 0.824 | 0.7548 / 0.3587 | 0.888 / 0.811 |
| A | both | 109 / 50 | 154 / 272 | 0.752 / 0.229 | 0.908 | 0.662 | 0.714 | 0.5242 / 0.2465 | 0.931 / 0.842 |
| B | cue_gold | 50 / 14 | 138 / 225 | 0.205 / 0.066 | 0.811 (0.811) | 0.701 | 0.788 | 0.1373 / 0.0474 | 0.758 / 0.753 |
| B | cue_dist | 113 / 70 | 19 / 69 | 0.198 / 0.070 | 0.757 (0.757) | 0.701 | 0.846 | 0.2490 / 0.0680 | 0.707 / 0.632 |
| B | both | 163 / 84 | 157 / 294 | 0.200 / 0.067 | 0.763 | 0.656 | 0.810 | 0.1834 / 0.0550 | 0.755 / 0.670 |
| C | cue_gold | 50 / 14 | 138 / 225 | 0.091 / 0.024 | 0.831 (0.831) | 0.701 | 0.811 | 0.0574 / 0.0151 | 0.782 / 0.776 |
| C | cue_dist | 113 / 70 | 19 / 69 | 0.100 / 0.025 | 0.769 (0.769) | 0.701 | 0.870 | 0.1278 / 0.0265 | 0.733 / 0.624 |
| C | both | 163 / 84 | 157 / 294 | 0.098 / 0.024 | 0.774 | 0.656 | 0.833 | 0.0864 / 0.0193 | 0.767 / 0.678 |
| D | cue_gold | 50 / 14 | 138 / 225 | 0.189 / 0.077 | 0.790 (0.790) | 0.701 | 0.817 | 0.1482 / 0.0509 | 0.714 / 0.714 |
| D | cue_dist | 113 / 70 | 19 / 69 | 0.175 / 0.074 | 0.729 (0.729) | 0.701 | 0.835 | 0.2198 / 0.0692 | 0.686 / 0.592 |
| D | both | 163 / 84 | 157 / 294 | 0.179 / 0.076 | 0.734 | 0.656 | 0.824 | 0.1777 / 0.0576 | 0.700 / 0.635 |

## s42_redrawn_round5: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 6 | 135 / 208 | 0.742 / 0.224 | 0.918 (0.918) | 0.728 | 0.671 | 0.3916 / 0.1893 | 0.907 / 0.853 |
| A | cue_dist | 75 / 43 | 20 / 62 | 0.747 / 0.270 | 0.877 (0.877) | 0.696 | 0.836 | 0.7677 / 0.3564 | 0.878 / 0.806 |
| A | both | 109 / 49 | 155 / 270 | 0.746 / 0.233 | 0.903 | 0.663 | 0.718 | 0.5270 / 0.2443 | 0.924 / 0.834 |
| B | cue_gold | 50 / 16 | 142 / 212 | 0.218 / 0.062 | 0.791 (0.791) | 0.678 | 0.789 | 0.1359 / 0.0450 | 0.765 / 0.675 |
| B | cue_dist | 92 / 46 | 25 / 67 | 0.195 / 0.082 | 0.712 (0.712) | 0.697 | 0.821 | 0.2212 / 0.0761 | 0.680 / 0.571 |
| B | both | 142 / 62 | 167 / 279 | 0.203 / 0.066 | 0.753 | 0.661 | 0.800 | 0.1682 / 0.0553 | 0.748 / 0.636 |
| C | cue_gold | 50 / 16 | 142 / 212 | 0.163 / 0.038 | 0.829 (0.829) | 0.678 | 0.818 | 0.0973 / 0.0239 | 0.819 / 0.719 |
| C | cue_dist | 92 / 46 | 25 / 67 | 0.160 / 0.046 | 0.759 (0.759) | 0.697 | 0.865 | 0.1777 / 0.0485 | 0.751 / 0.617 |
| C | both | 142 / 62 | 167 / 279 | 0.161 / 0.039 | 0.791 | 0.661 | 0.835 | 0.1278 / 0.0320 | 0.804 / 0.672 |
| D | cue_gold | 50 / 16 | 142 / 212 | 0.224 / 0.080 | 0.802 (0.802) | 0.678 | 0.864 | 0.1642 / 0.0504 | 0.772 / 0.702 |
| D | cue_dist | 92 / 46 | 25 / 67 | 0.191 / 0.081 | 0.730 (0.730) | 0.697 | 0.893 | 0.2229 / 0.0687 | 0.674 / 0.543 |
| D | both | 142 / 62 | 167 / 279 | 0.202 / 0.080 | 0.754 | 0.661 | 0.875 | 0.1864 / 0.0565 | 0.751 / 0.616 |

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
  "s42_redrawn_round2": {
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
  "s42_redrawn_round3": {
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
  "s42_redrawn_round4": {
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
  "s42_redrawn_round5": {
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
  "s42_redrawn_round2": {
    "13": 0.8800580501556396,
    "14": 0.8401700854301453,
    "15": 0.7987735867500305,
    "16": 0.785494863986969
  },
  "s42_redrawn_round3": {
    "13": 0.923751711845398,
    "14": 0.9007993936538696,
    "15": 0.8774144053459167,
    "16": 0.8643982410430908
  },
  "s42_redrawn_round4": {
    "13": 0.9022080302238464,
    "14": 0.8917789459228516,
    "15": 0.8787180781364441,
    "16": 0.8743165135383606
  },
  "s42_redrawn_round5": {
    "13": 0.909038245677948,
    "14": 0.9045583009719849,
    "15": 0.8906710743904114,
    "16": 0.8847673535346985
  }
}
```
