# Training dynamics and probe rows: weight_1_5_s42, weight_1_5_s43, weight_1_5_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: weight_1_5_s42 0.3, weight_1_5_s43 0.3, weight_1_5_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | weight_1_5_s42 | 26.700 | 32.600 | 54.367 | 63.912 |
| groups where all 8 answers name the cue (of 64) | weight_1_5_s43 | 26.700 | 49.950 | 46.833 | 51.015 |
| groups where all 8 answers name the cue (of 64) | weight_1_5_s44 | 31 | 37.600 | 47.967 | 49.397 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | weight_1_5_s42 | 27.800 | 10.900 | 4.967 | 0.074 |
| groups with mixed task rewards (of 64) | weight_1_5_s43 | 26.500 | 2.950 | 2.433 | 1.441 |
| groups with mixed task rewards (of 64) | weight_1_5_s44 | 25.800 | 11.450 | 11.200 | 9.456 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | weight_1_5_s42 | 7.200 | 3.150 | 1.700 | 0.029 |
|   of which the cue is not at the training label | weight_1_5_s43 | 6.400 | 1.200 | 0.767 | 0.559 |
|   of which the cue is not at the training label | weight_1_5_s44 | 7.400 | 3.550 | 4.333 | 4.103 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | weight_1_5_s42 | 167.200 | 206.600 | 55.433 | 0.103 |
| answers not naming the cue (of 512) | weight_1_5_s43 | 176.200 | 100.150 | 128.100 | 97.603 |
| answers not naming the cue (of 512) | weight_1_5_s44 | 132.100 | 164.250 | 79.167 | 72.368 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | weight_1_5_s42 | 0.409 | 0.149 | 0.096 | 0.015 |
| mean absolute advantage per answer | weight_1_5_s43 | 0.377 | 0.038 | 0.026 | 0.018 |
| mean absolute advantage per answer | weight_1_5_s44 | 0.399 | 0.169 | 0.166 | 0.174 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | weight_1_5_s42 | 0.120 | 0.038 | 0.043 | 0.014 |
|   from probe-only groups | weight_1_5_s43 | 0.111 | 0.004 | 0.001 | 0.004 |
|   from probe-only groups | weight_1_5_s44 | 0.133 | 0.053 | 0.045 | 0.062 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | weight_1_5_s42 | -0.006 | 0.009 | 0.003 | 0.000 |
| advantage on cue-naming answers, per answer | weight_1_5_s43 | 0.003 | 0.003 | 0.003 | -0.001 |
| advantage on cue-naming answers, per answer | weight_1_5_s44 | -0.013 | 0.012 | 0.005 | 0.005 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | weight_1_5_s42 | 0.103 | 0.082 | 0.075 | 0.024 |
| gradient norm (W&B, before clipping) | weight_1_5_s43 | 0.172 | 1.022 | 1.132 | 0.667 |
| gradient norm (W&B, before clipping) | weight_1_5_s44 | 0.105 | 0.099 | 0.066 | 0.075 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | weight_1_5_s42 | 0.026 | 0.289 | 0.849 | 1.346 |
| KL from the starting model (W&B) | weight_1_5_s43 | 0.119 | 3.414 | 3.900 | 3.855 |
| KL from the starting model (W&B) | weight_1_5_s44 | 0.024 | 0.196 | 0.395 | 0.629 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | weight_1_5_s42 | 0.423 | 0.298 | 0.295 | 0.374 |
| token entropy (W&B) | weight_1_5_s43 | 0.439 | 0.141 | 0.016 | 0.105 |
| token entropy (W&B) | weight_1_5_s44 | 0.450 | 0.322 | 0.295 | 0.333 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | weight_1_5_s42 | 226.068 | 179.050 | 106.190 | 73.587 |
| answer length, tokens (W&B) | weight_1_5_s43 | 199.152 | 19.435 | 9.019 | 10.864 |
| answer length, tokens (W&B) | weight_1_5_s44 | 220.373 | 202.299 | 222.843 | 204.583 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of weight_1_5_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 128 | 0.066 | 946.412 / 1026.018 | 0.180 / 0.283 | 0.240 / 0.348 | 0.449 / 0.514 |
| 5-9 | 139 | 0.045 | 804.243 / 857.424 | 0.097 / 0.178 | 0.176 / 0.295 | 0.306 / 0.329 |
| 10-29 | 652 | 0.003 | 719.979 / 772.130 | 0.026 / 0.033 | 0.104 / 0.123 | 0.202 / 0.214 |
| 30-59 | 1631 | 0.002 | 383.260 / 419.621 | 0.008 / 0.008 | 0.106 / 0.109 | 0.077 / 0.094 |
| 60-127 | 4346 | 0.001 | 261.679 / 321.703 | 0.005 / 0.007 | 0.091 / 0.095 | 0.035 / 0.045 |

## Probe-only groups of weight_1_5_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 143 | 0.068 | 909.376 / 978.490 | 0.157 / 0.252 | 0.231 / 0.325 | 0.399 / 0.509 |
| 5-9 | 124 | 0.021 | 602.062 / 628.444 | 0.050 / 0.095 | 0.091 / 0.169 | 0.212 / 0.294 |
| 10-29 | 999 | 0.000 | 74.801 / 81.780 | 0.001 / 0.001 | 0.008 / 0.007 | 0.006 / 0.009 |
| 30-59 | 1405 | 0.000 | 42.756 / 42.841 | 0 / 0 | 0 / 0 | 0 / 0.000 |
| 60-127 | 3469 | 0.000 | 50.427 / 52.583 | 0.001 / 0.001 | 0.001 / 0.002 | 0.005 / 0.007 |

## Probe-only groups of weight_1_5_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 152 | 0.084 | 974.975 / 1053.738 | 0.171 / 0.288 | 0.227 / 0.347 | 0.459 / 0.503 |
| 5-9 | 158 | 0.025 | 833.759 / 890.794 | 0.055 / 0.095 | 0.117 / 0.191 | 0.296 / 0.343 |
| 10-29 | 752 | 0.006 | 805.099 / 857.131 | 0.050 / 0.079 | 0.108 / 0.161 | 0.220 / 0.251 |
| 30-59 | 1439 | 0.004 | 891.036 / 937.951 | 0.043 / 0.052 | 0.152 / 0.174 | 0.122 / 0.157 |
| 60-127 | 3359 | 0.007 | 824.300 / 898.538 | 0.021 / 0.023 | 0.149 / 0.161 | 0.175 / 0.206 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

## weight_1_5_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.125 | 0.029 | 977.000 |
| 10 | 0.797 | 0.002 | 623.000 |
| 20 | 0.994 | 0.004 | 781.500 |
| 25 | 0.982 | 0.020 | 694.000 |
| 30 | 0.781 | 0.002 | 689.000 |
| 60 | 0.037 | 0.002 | 195.000 |
| 70 | 0.008 | 0 | 182.500 |
| 80 | 0.010 | 0.016 | 181.000 |
| 90 | 0.021 | 0 | 208.000 |
| 100 | 0.082 | 0.004 | 260.000 |
| 127 | 0.133 | 0.004 | 330.500 |

## weight_1_5_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.129 | 0.008 | 997.500 |
| 10 | 0.012 | 0 | 44.000 |
| 20 | 0 | 0.193 | 43.000 |
| 25 | 0 | 0.072 | 43.000 |
| 30 | 0 | 0.002 | 43.000 |
| 60 | 0 | 0 | 42.000 |
| 70 | 0 | 0.006 | 42.000 |
| 80 | 0 | 0 | 42.000 |
| 90 | 0 | 0 | 42.000 |
| 100 | 0 | 0 | 42.000 |
| 127 | 0 | 0 | 43.000 |

## weight_1_5_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.102 | 0.025 | 983.500 |
| 10 | 0.027 | 0.018 | 687.500 |
| 20 | 0.010 | 0.006 | 872.500 |
| 25 | 0.021 | 0.021 | 917.000 |
| 30 | 0.008 | 0.053 | 951.500 |
| 60 | 0.004 | 0.014 | 851.000 |
| 70 | 0.006 | 0.004 | 806.000 |
| 80 | 0.006 | 0.014 | 884.000 |
| 90 | 0.012 | 0.006 | 861.000 |
| 100 | 0.002 | 0.006 | 832.500 |
| 127 | 0 | 0.004 | 748.500 |

## penalty_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.105 | 0.018 | 952.000 |
| 10 | 0.010 | 0.010 | 548.000 |
| 20 | 0.076 | 0.062 | 489.000 |
| 25 | 0.010 | 0.018 | 366.000 |
| 30 | 1 | 0.004 | 294.000 |
| 60 | 1 | 0.002 | 403.500 |
| 70 | 0.998 | 0 | 388.500 |
| 80 | 1 | 0.381 | 391.500 |
| 90 | 1 | 0.990 | 374.000 |
| 100 | 1 | 0.998 | 306.500 |
| 127 | 1 | 1 | 295.500 |

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
  "weight_1_5_s42": {
    "median_tokens_no_cue": 83.0,
    "median_tokens_cued": 75.5,
    "mean_top_letter_share_of_8_no_cue": 0.7824642126789366,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 272,
      "B": 175,
      "null": 42
    },
    "no_cue_answer_letters": {
      "A": 253,
      "B": 154
    },
    "questions_with_no_cue_answer": 407
  },
  "weight_1_5_s43": {
    "median_tokens_no_cue": 9.0,
    "median_tokens_cued": 9.0,
    "mean_top_letter_share_of_8_no_cue": 0.9688139059304703,
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
      "A": 168,
      "B": 312
    },
    "questions_with_no_cue_answer": 480
  },
  "weight_1_5_s44": {
    "median_tokens_no_cue": 168.0,
    "median_tokens_cued": 170.0,
    "mean_top_letter_share_of_8_no_cue": 0.9284253578732107,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 184,
      "B": 302,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 183,
      "B": 293
    },
    "questions_with_no_cue_answer": 476
  },
  "penalty_s42": {
    "median_tokens_no_cue": 73.0,
    "median_tokens_cued": 72.0,
    "mean_top_letter_share_of_8_no_cue": 0.9741820040899796,
    "template_share": {
      "none": 0.9989775051124744,
      "cue_gold": 1,
      "cue_dist": 1
    },
    "no_cue_sample_0_letters": {
      "A": 31,
      "B": 455,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 32,
      "B": 452
    },
    "questions_with_no_cue_answer": 484
  }
}
```

## weight_1_5_s42: probe rows by answer letter (388 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 28 / 7 | 121 / 171 | 0.742 / 0.215 | 0.922 (0.922) | 0.693 | 0.649 | 0.3676 / 0.1909 | 0.919 / 0.877 |
| A | cue_dist | 56 / 39 | 20 / 52 | 0.730 / 0.263 | 0.869 (0.869) | 0.656 | 0.806 | 0.7353 / 0.3558 | 0.879 / 0.812 |
| A | both | 84 / 46 | 141 / 223 | 0.733 / 0.225 | 0.900 | 0.629 | 0.688 | 0.4918 / 0.2467 | 0.928 / 0.842 |
| B | cue_gold | 35 / 89 | 150 / 114 | 0.116 / 0.078 | 0.570 (0.570) | 0.357 | 0.734 | 0.1334 / 0.0501 | 0.751 / 0.616 |
| B | cue_dist | 114 / 150 | 89 / 35 | 0.176 / 0.162 | 0.494 (0.494) | 0.357 | 0.809 | 0.2547 / 0.0807 | 0.607 / 0.599 |
| B | both | 149 / 239 | 239 / 149 | 0.157 / 0.105 | 0.585 | 0.384 | 0.766 | 0.1969 / 0.0647 | 0.710 / 0.643 |
| C | cue_gold | 35 / 89 | 150 / 114 | 0.004 / 0.004 | 0.489 (0.489) | 0.357 | 0.817 | 0.0061 / 0.0021 | 0.681 / 0.573 |
| C | cue_dist | 114 / 150 | 89 / 35 | 0.006 / 0.004 | 0.407 (0.407) | 0.357 | 0.825 | 0.0079 / 0.0023 | 0.523 / 0.453 |
| C | both | 149 / 239 | 239 / 149 | 0.005 / 0.004 | 0.472 | 0.384 | 0.822 | 0.0070 / 0.0022 | 0.588 / 0.539 |
| D | cue_gold | 35 / 89 | 150 / 114 | 0.063 / 0.056 | 0.464 (0.464) | 0.357 | 0.876 | 0.0989 / 0.0219 | 0.673 / 0.569 |
| D | cue_dist | 114 / 150 | 89 / 35 | 0.071 / 0.071 | 0.419 (0.419) | 0.357 | 0.905 | 0.1168 / 0.0213 | 0.579 / 0.506 |
| D | both | 149 / 239 | 239 / 149 | 0.069 / 0.061 | 0.472 | 0.384 | 0.891 | 0.1082 / 0.0216 | 0.627 / 0.563 |

## weight_1_5_s43: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 137 / 209 | 0.747 / 0.222 | 0.922 (0.922) | 0.729 | 0.675 | 0.3921 / 0.1864 | 0.911 / 0.857 |
| A | cue_dist | 75 / 46 | 21 / 63 | 0.738 / 0.274 | 0.871 (0.871) | 0.685 | 0.827 | 0.7593 / 0.3616 | 0.876 / 0.801 |
| A | both | 110 / 52 | 158 / 272 | 0.740 / 0.232 | 0.901 | 0.656 | 0.716 | 0.5236 / 0.2453 | 0.923 / 0.835 |
| B | cue_gold | 52 / 25 | 127 / 215 | 0.332 / 0.112 | 0.833 (0.833) | 0.652 | 0.635 | 0.2005 / 0.1162 | 0.850 / 0.774 |
| B | cue_dist | 90 / 49 | 31 / 83 | 0.377 / 0.193 | 0.736 (0.736) | 0.688 | 0.777 | 0.4160 / 0.1827 | 0.677 / 0.659 |
| B | both | 142 / 74 | 158 / 298 | 0.361 / 0.132 | 0.812 | 0.655 | 0.688 | 0.2874 / 0.1398 | 0.828 / 0.736 |
| C | cue_gold | 52 / 25 | 127 / 215 | 0.238 / 0.028 | 0.902 (0.902) | 0.652 | 0.571 | 0.1041 / 0.0385 | 0.927 / 0.842 |
| C | cue_dist | 90 / 49 | 31 / 83 | 0.294 / 0.040 | 0.887 (0.887) | 0.688 | 0.736 | 0.2657 / 0.1000 | 0.909 / 0.845 |
| C | both | 142 / 74 | 158 / 298 | 0.274 / 0.031 | 0.907 | 0.655 | 0.639 | 0.1693 / 0.0603 | 0.929 / 0.861 |
| D | cue_gold | 52 / 25 | 127 / 215 | 0.252 / 0.066 | 0.845 (0.845) | 0.652 | 0.773 | 0.1727 / 0.0463 | 0.832 / 0.804 |
| D | cue_dist | 90 / 49 | 31 / 83 | 0.329 / 0.073 | 0.860 (0.860) | 0.688 | 0.856 | 0.3469 / 0.0912 | 0.849 / 0.822 |
| D | both | 142 / 74 | 158 / 298 | 0.302 / 0.068 | 0.860 | 0.655 | 0.803 | 0.2429 / 0.0622 | 0.858 / 0.824 |

## weight_1_5_s44: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 5 | 136 / 210 | 0.763 / 0.223 | 0.929 (0.929) | 0.739 | 0.670 | 0.3883 / 0.1899 | 0.908 / 0.898 |
| A | cue_dist | 75 / 45 | 19 / 62 | 0.741 / 0.259 | 0.881 (0.881) | 0.695 | 0.822 | 0.7551 / 0.3633 | 0.907 / 0.797 |
| A | both | 109 / 50 | 155 / 272 | 0.746 / 0.230 | 0.905 | 0.661 | 0.711 | 0.5189 / 0.2475 | 0.928 / 0.836 |
| B | cue_gold | 55 / 24 | 140 / 216 | 0.130 / 0.036 | 0.810 (0.810) | 0.651 | 0.895 | 0.0957 / 0.0185 | 0.808 / 0.763 |
| B | cue_dist | 128 / 71 | 29 / 67 | 0.134 / 0.044 | 0.740 (0.740) | 0.671 | 0.918 | 0.1658 / 0.0359 | 0.738 / 0.567 |
| B | both | 183 / 95 | 169 / 283 | 0.133 / 0.038 | 0.780 | 0.642 | 0.902 | 0.1270 / 0.0249 | 0.808 / 0.675 |
| C | cue_gold | 55 / 24 | 140 / 216 | 0.088 / 0.026 | 0.763 (0.763) | 0.651 | 0.818 | 0.0675 / 0.0133 | 0.759 / 0.630 |
| C | cue_dist | 128 / 71 | 29 / 67 | 0.077 / 0.029 | 0.677 (0.677) | 0.671 | 0.849 | 0.0962 / 0.0217 | 0.628 / 0.483 |
| C | both | 183 / 95 | 169 / 283 | 0.080 / 0.027 | 0.710 | 0.642 | 0.833 | 0.0803 / 0.0164 | 0.705 / 0.560 |
| D | cue_gold | 55 / 24 | 140 / 216 | 0.261 / 0.134 | 0.750 (0.750) | 0.651 | 0.965 | 0.2671 / 0.0671 | 0.723 / 0.655 |
| D | cue_dist | 128 / 71 | 29 / 67 | 0.261 / 0.126 | 0.750 (0.750) | 0.671 | 0.965 | 0.3283 / 0.0914 | 0.728 / 0.609 |
| D | both | 183 / 95 | 169 / 283 | 0.261 / 0.132 | 0.745 | 0.642 | 0.965 | 0.2944 / 0.0760 | 0.734 / 0.657 |

## penalty_s42: probe rows by answer letter (462 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 139 / 209 | 0.753 / 0.224 | 0.924 (0.924) | 0.719 | 0.678 | 0.3976 / 0.1883 | 0.910 / 0.879 |
| A | cue_dist | 74 / 46 | 22 / 63 | 0.736 / 0.275 | 0.869 (0.869) | 0.679 | 0.820 | 0.7506 / 0.3630 | 0.880 / 0.797 |
| A | both | 110 / 53 | 161 / 272 | 0.740 / 0.234 | 0.901 | 0.652 | 0.715 | 0.5226 / 0.2469 | 0.922 / 0.838 |
| B | cue_gold | 188 / 5 | 27 / 242 | 0.078 / 0.018 | 0.825 (0.825) | 0.937 | 0.820 | 0.0702 / 0.0202 | 0.726 / 0.588 |
| B | cue_dist | 241 / 27 | 5 / 188 | 0.110 / 0.027 | 0.844 (0.844) | 0.937 | 0.870 | 0.1177 / 0.0265 | 0.590 / 0.588 |
| B | both | 429 / 32 | 32 / 430 | 0.097 / 0.022 | 0.841 | 0.931 | 0.841 | 0.0955 / 0.0231 | 0.767 / 0.636 |
| C | cue_gold | 188 / 5 | 27 / 242 | 0.009 / 0.002 | 0.928 (0.928) | 0.937 | 0.964 | 0.0090 / 0.0015 | 0.647 / 0.533 |
| C | cue_dist | 241 / 27 | 5 / 188 | 0.006 / 0.002 | 0.885 (0.885) | 0.937 | 0.953 | 0.0070 / 0.0016 | 0.482 / 0.429 |
| C | both | 429 / 32 | 32 / 430 | 0.008 / 0.002 | 0.902 | 0.931 | 0.959 | 0.0080 / 0.0015 | 0.608 / 0.453 |
| D | cue_gold | 188 / 5 | 27 / 242 | 0.025 / 0.007 | 0.916 (0.916) | 0.937 | 0.958 | 0.0247 / 0.0062 | 0.631 / 0.544 |
| D | cue_dist | 241 / 27 | 5 / 188 | 0.021 / 0.007 | 0.859 (0.859) | 0.937 | 0.935 | 0.0228 / 0.0066 | 0.438 / 0.358 |
| D | both | 429 / 32 | 32 / 430 | 0.023 / 0.007 | 0.882 | 0.931 | 0.946 | 0.0237 / 0.0064 | 0.591 / 0.402 |

## Probe data: answers naming A

```
{
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
```

## Cosine of the initial and fresh probe weights

```
{
  "weight_1_5_s42": {
    "13": 0.5555298328399658,
    "14": 0.501789927482605,
    "15": 0.42769548296928406,
    "16": 0.4131673574447632
  },
  "weight_1_5_s43": {
    "13": 0.6434087157249451,
    "14": 0.5911885499954224,
    "15": 0.5594038963317871,
    "16": 0.5908920764923096
  },
  "weight_1_5_s44": {
    "13": 0.7950292825698853,
    "14": 0.7359678745269775,
    "15": 0.6952301859855652,
    "16": 0.6579523086547852
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
