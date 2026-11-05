# Training dynamics and probe rows: weight_0_6_s42, weight_0_6_s43, weight_0_6_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: weight_0_6_s42 0.3, weight_0_6_s43 0.3, weight_0_6_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | weight_0_6_s42 | 36.900 | 54.200 | 53.833 | 53.118 |
| groups where all 8 answers name the cue (of 64) | weight_0_6_s43 | 36.700 | 42.550 | 50.400 | 50.309 |
| groups where all 8 answers name the cue (of 64) | weight_0_6_s44 | 37.600 | 56.200 | 57.667 | 55.985 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | weight_0_6_s42 | 22 | 7.400 | 6.567 | 6.250 |
| groups with mixed task rewards (of 64) | weight_0_6_s43 | 23.400 | 16.800 | 9.667 | 9.721 |
| groups with mixed task rewards (of 64) | weight_0_6_s44 | 21.900 | 4.100 | 3.667 | 5.015 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | weight_0_6_s42 | 5.800 | 3.650 | 2.667 | 3.044 |
|   of which the cue is not at the training label | weight_0_6_s43 | 6.800 | 6.200 | 4.400 | 4.735 |
|   of which the cue is not at the training label | weight_0_6_s44 | 6.100 | 2.150 | 1.433 | 2.485 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | weight_0_6_s42 | 102.200 | 39.850 | 51.633 | 57.147 |
| answers not naming the cue (of 512) | weight_0_6_s43 | 100 | 92.200 | 62.100 | 59.559 |
| answers not naming the cue (of 512) | weight_0_6_s44 | 101.100 | 42 | 33.300 | 40.118 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | weight_0_6_s42 | 0.320 | 0.187 | 0.118 | 0.104 |
| mean absolute advantage per answer | weight_0_6_s43 | 0.335 | 0.261 | 0.148 | 0.140 |
| mean absolute advantage per answer | weight_0_6_s44 | 0.316 | 0.097 | 0.077 | 0.087 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | weight_0_6_s42 | 0.083 | 0.104 | 0.042 | 0.031 |
|   from probe-only groups | weight_0_6_s43 | 0.081 | 0.073 | 0.034 | 0.028 |
|   from probe-only groups | weight_0_6_s44 | 0.080 | 0.048 | 0.033 | 0.027 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | weight_0_6_s42 | 0.047 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | weight_0_6_s43 | 0.044 | 0.019 | 0.002 | -0.000 |
| advantage on cue-naming answers, per answer | weight_0_6_s44 | 0.046 | -0.002 | 0.005 | 0.000 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | weight_0_6_s42 | 0.079 | 0.081 | 0.055 | 0.051 |
| gradient norm (W&B, before clipping) | weight_0_6_s43 | 0.104 | 0.086 | 0.093 | 0.069 |
| gradient norm (W&B, before clipping) | weight_0_6_s44 | 0.098 | 0.064 | 0.043 | 0.045 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | weight_0_6_s42 | 0.011 | 0.339 | 0.726 | 0.697 |
| KL from the base model (W&B) | weight_0_6_s43 | 0.016 | 0.180 | 0.332 | 0.248 |
| KL from the base model (W&B) | weight_0_6_s44 | 0.022 | 0.225 | 0.414 | 0.431 |
| KL from the base model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | weight_0_6_s42 | 0.473 | 0.386 | 0.320 | 0.357 |
| token entropy (W&B) | weight_0_6_s43 | 0.451 | 0.350 | 0.237 | 0.313 |
| token entropy (W&B) | weight_0_6_s44 | 0.443 | 0.306 | 0.324 | 0.377 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | weight_0_6_s42 | 238.490 | 181.307 | 182.530 | 207.000 |
| answer length, tokens (W&B) | weight_0_6_s43 | 225.203 | 217.592 | 152.029 | 177.393 |
| answer length, tokens (W&B) | weight_0_6_s44 | 223.178 | 175.343 | 205.291 | 207.751 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of weight_0_6_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 140 | 0.067 | 949.466 / 1032.487 | 0.152 / 0.252 | 0.237 / 0.361 | 0.412 / 0.496 |
| 5-9 | 229 | 0.079 | 1023.620 / 1076.691 | 0.129 / 0.178 | 0.159 / 0.215 | 0.400 / 0.487 |
| 10-29 | 1084 | 0.053 | 752.119 / 780.179 | 0.039 / 0.047 | 0.128 / 0.137 | 0.220 / 0.241 |
| 30-59 | 1615 | 0.011 | 714.797 / 765.891 | 0.024 / 0.033 | 0.107 / 0.116 | 0.114 / 0.134 |
| 60-127 | 3612 | 0.009 | 874.664 / 873.831 | 0.025 / 0.024 | 0.114 / 0.117 | 0.131 / 0.142 |

## Probe-only groups of weight_0_6_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 151 | 0.084 | 958.826 / 1020.611 | 0.190 / 0.300 | 0.252 / 0.361 | 0.439 / 0.568 |
| 5-9 | 216 | 0.069 | 858.947 / 902.558 | 0.122 / 0.167 | 0.192 / 0.256 | 0.333 / 0.390 |
| 10-29 | 851 | 0.057 | 903.257 / 947.972 | 0.058 / 0.081 | 0.115 / 0.152 | 0.291 / 0.321 |
| 30-59 | 1512 | 0.008 | 618.621 / 680.001 | 0.021 / 0.033 | 0.105 / 0.119 | 0.160 / 0.194 |
| 60-127 | 3421 | 0.008 | 722.769 / 810.773 | 0.022 / 0.037 | 0.105 / 0.121 | 0.205 / 0.246 |

## Probe-only groups of weight_0_6_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 151 | 0.083 | 988 / 1069.109 | 0.180 / 0.310 | 0.265 / 0.341 | 0.439 / 0.512 |
| 5-9 | 225 | 0.052 | 896.510 / 936.746 | 0.114 / 0.176 | 0.242 / 0.328 | 0.431 / 0.482 |
| 10-29 | 1124 | 0.011 | 727.618 / 774.297 | 0.042 / 0.054 | 0.143 / 0.156 | 0.309 / 0.348 |
| 30-59 | 1730 | 0.011 | 859.269 / 913.680 | 0.025 / 0.033 | 0.121 / 0.129 | 0.299 / 0.327 |
| 60-127 | 3807 | 0.010 | 883.481 / 941.012 | 0.047 / 0.050 | 0.138 / 0.149 | 0.305 / 0.328 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

## weight_0_6_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.105 | 0.012 | 1016.000 |
| 10 | 0.002 | 0.016 | 934.500 |
| 20 | 0.008 | 0.010 | 758.500 |
| 25 | 0.004 | 0.018 | 554.500 |
| 30 | 0.012 | 0 | 640.500 |
| 60 | 0.031 | 0.002 | 625.500 |
| 70 | 0.078 | 0.002 | 703.500 |
| 80 | 0.197 | 0 | 773.000 |
| 90 | 0.395 | 0 | 891.500 |
| 100 | 0.402 | 0.014 | 863.000 |
| 127 | 0.570 | 0 | 861.500 |

## weight_0_6_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.148 | 0.012 | 1102.000 |
| 10 | 0.062 | 0.021 | 862.000 |
| 20 | 0.012 | 0.004 | 894.500 |
| 25 | 0.010 | 0.037 | 939.000 |
| 30 | 0.025 | 0.027 | 791.000 |
| 60 | 0.002 | 0 | 595.500 |
| 70 | 0.002 | 0 | 768.000 |
| 80 | 0.008 | 0 | 771.000 |
| 90 | 0.012 | 0.002 | 768.500 |
| 100 | 0.018 | 0 | 724.500 |
| 127 | 0 | 0 | 677.000 |

## weight_0_6_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.115 | 0.008 | 1039.000 |
| 10 | 0.020 | 0.004 | 835.000 |
| 20 | 0 | 0.010 | 738.500 |
| 25 | 0 | 0 | 714.000 |
| 30 | 0 | 0.006 | 737.000 |
| 60 | 0.002 | 0.002 | 847.500 |
| 70 | 0.004 | 0 | 846.000 |
| 80 | 0.010 | 0 | 866.000 |
| 90 | 0.010 | 0.021 | 847.500 |
| 100 | 0.027 | 0 | 879.000 |
| 127 | 0.008 | 0 | 886.500 |

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
  "weight_0_6_s42": {
    "median_tokens_no_cue": 192.0,
    "median_tokens_cued": 195.0,
    "mean_top_letter_share_of_8_no_cue": 0.9217791411042945,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 201,
      "B": 282,
      "null": 6
    },
    "no_cue_answer_letters": {
      "A": 179,
      "B": 290
    },
    "questions_with_no_cue_answer": 469
  },
  "weight_0_6_s43": {
    "median_tokens_no_cue": 151.0,
    "median_tokens_cued": 154.0,
    "mean_top_letter_share_of_8_no_cue": 0.9233128834355828,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 175,
      "B": 313,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 167,
      "B": 308
    },
    "questions_with_no_cue_answer": 475
  },
  "weight_0_6_s44": {
    "median_tokens_no_cue": 193.0,
    "median_tokens_cued": 196.0,
    "mean_top_letter_share_of_8_no_cue": 0.9238241308793456,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 195,
      "B": 292,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 185,
      "B": 293
    },
    "questions_with_no_cue_answer": 478
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

## weight_0_6_s42: probe rows by answer letter (448 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 5 | 133 / 209 | 0.767 / 0.225 | 0.927 (0.927) | 0.740 | 0.676 | 0.3960 / 0.1882 | 0.910 / 0.877 |
| A | cue_dist | 73 / 44 | 20 / 62 | 0.749 / 0.255 | 0.889 (0.889) | 0.690 | 0.841 | 0.7704 / 0.3480 | 0.918 / 0.803 |
| A | both | 106 / 49 | 153 / 271 | 0.753 / 0.231 | 0.909 | 0.662 | 0.723 | 0.5304 / 0.2411 | 0.937 / 0.833 |
| B | cue_gold | 53 / 20 | 140 / 218 | 0.119 / 0.035 | 0.753 (0.753) | 0.667 | 0.862 | 0.0861 / 0.0195 | 0.714 / 0.665 |
| B | cue_dist | 132 / 82 | 25 / 65 | 0.087 / 0.037 | 0.680 (0.680) | 0.670 | 0.883 | 0.1187 / 0.0230 | 0.617 / 0.543 |
| B | both | 185 / 102 | 165 / 283 | 0.096 / 0.035 | 0.699 | 0.638 | 0.871 | 0.1007 / 0.0209 | 0.690 / 0.593 |
| C | cue_gold | 53 / 20 | 140 / 218 | 0.063 / 0.015 | 0.756 (0.756) | 0.667 | 0.803 | 0.0400 / 0.0089 | 0.712 / 0.701 |
| C | cue_dist | 132 / 82 | 25 / 65 | 0.035 / 0.015 | 0.691 (0.691) | 0.670 | 0.852 | 0.0466 / 0.0105 | 0.682 / 0.546 |
| C | both | 185 / 102 | 165 / 283 | 0.042 / 0.015 | 0.698 | 0.638 | 0.825 | 0.0430 / 0.0095 | 0.699 / 0.591 |
| D | cue_gold | 53 / 20 | 140 / 218 | 0.279 / 0.151 | 0.751 (0.751) | 0.667 | 0.797 | 0.2480 / 0.1119 | 0.676 / 0.717 |
| D | cue_dist | 132 / 82 | 25 / 65 | 0.259 / 0.151 | 0.701 (0.701) | 0.670 | 0.829 | 0.3143 / 0.1335 | 0.675 / 0.556 |
| D | both | 185 / 102 | 165 / 283 | 0.264 / 0.151 | 0.713 | 0.638 | 0.811 | 0.2777 / 0.1201 | 0.693 / 0.616 |

## weight_0_6_s43: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 133 / 207 | 0.753 / 0.223 | 0.923 (0.923) | 0.723 | 0.673 | 0.3980 / 0.1910 | 0.910 / 0.875 |
| A | cue_dist | 74 / 45 | 22 / 64 | 0.740 / 0.272 | 0.872 (0.872) | 0.683 | 0.828 | 0.7567 / 0.3554 | 0.885 / 0.796 |
| A | both | 110 / 52 | 155 / 271 | 0.743 / 0.233 | 0.901 | 0.658 | 0.717 | 0.5279 / 0.2465 | 0.924 / 0.833 |
| B | cue_gold | 60 / 19 | 134 / 223 | 0.161 / 0.032 | 0.852 (0.852) | 0.692 | 0.867 | 0.0985 / 0.0209 | 0.871 / 0.746 |
| B | cue_dist | 126 / 65 | 21 / 73 | 0.159 / 0.044 | 0.756 (0.756) | 0.718 | 0.905 | 0.1917 / 0.0466 | 0.724 / 0.570 |
| B | both | 186 / 84 | 155 / 296 | 0.160 / 0.035 | 0.807 | 0.673 | 0.879 | 0.1387 / 0.0302 | 0.842 / 0.678 |
| C | cue_gold | 60 / 19 | 134 / 223 | 0.183 / 0.035 | 0.849 (0.849) | 0.692 | 0.808 | 0.1105 / 0.0223 | 0.882 / 0.692 |
| C | cue_dist | 126 / 65 | 21 / 73 | 0.166 / 0.042 | 0.759 (0.759) | 0.718 | 0.902 | 0.2001 / 0.0456 | 0.755 / 0.550 |
| C | both | 186 / 84 | 155 / 296 | 0.171 / 0.036 | 0.806 | 0.673 | 0.844 | 0.1491 / 0.0307 | 0.851 / 0.644 |
| D | cue_gold | 60 / 19 | 134 / 223 | 0.290 / 0.076 | 0.838 (0.838) | 0.692 | 0.853 | 0.1982 / 0.0478 | 0.867 / 0.689 |
| D | cue_dist | 126 / 65 | 21 / 73 | 0.252 / 0.093 | 0.746 (0.746) | 0.718 | 0.917 | 0.3155 / 0.0758 | 0.734 / 0.541 |
| D | both | 186 / 84 | 155 / 296 | 0.263 / 0.079 | 0.796 | 0.673 | 0.875 | 0.2488 / 0.0580 | 0.841 / 0.642 |

## weight_0_6_s44: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 137 / 210 | 0.759 / 0.223 | 0.926 (0.926) | 0.715 | 0.668 | 0.3887 / 0.1922 | 0.920 / 0.874 |
| A | cue_dist | 75 / 45 | 22 / 61 | 0.745 / 0.259 | 0.883 (0.883) | 0.680 | 0.826 | 0.7555 / 0.3548 | 0.883 / 0.827 |
| A | both | 108 / 52 | 159 / 271 | 0.748 / 0.230 | 0.907 | 0.653 | 0.714 | 0.5219 / 0.2455 | 0.928 / 0.847 |
| B | cue_gold | 51 / 22 | 147 / 221 | 0.202 / 0.057 | 0.783 (0.783) | 0.650 | 0.871 | 0.1405 / 0.0330 | 0.820 / 0.634 |
| B | cue_dist | 153 / 76 | 26 / 63 | 0.198 / 0.069 | 0.764 (0.764) | 0.688 | 0.891 | 0.2394 / 0.0620 | 0.791 / 0.599 |
| B | both | 204 / 98 | 173 / 284 | 0.199 / 0.060 | 0.786 | 0.648 | 0.881 | 0.1875 / 0.0436 | 0.830 / 0.658 |
| C | cue_gold | 51 / 22 | 147 / 221 | 0.138 / 0.041 | 0.763 (0.763) | 0.650 | 0.922 | 0.1039 / 0.0187 | 0.755 / 0.686 |
| C | cue_dist | 153 / 76 | 26 / 63 | 0.102 / 0.044 | 0.745 (0.745) | 0.688 | 0.942 | 0.1301 / 0.0277 | 0.704 / 0.579 |
| C | both | 204 / 98 | 173 / 284 | 0.110 / 0.041 | 0.751 | 0.648 | 0.931 | 0.1164 / 0.0220 | 0.749 / 0.647 |
| D | cue_gold | 51 / 22 | 147 / 221 | 0.384 / 0.190 | 0.770 (0.770) | 0.650 | 0.958 | 0.3701 / 0.1016 | 0.779 / 0.685 |
| D | cue_dist | 153 / 76 | 26 / 63 | 0.359 / 0.185 | 0.742 (0.742) | 0.688 | 0.967 | 0.4561 / 0.1231 | 0.735 / 0.553 |
| D | both | 204 / 98 | 173 / 284 | 0.365 / 0.189 | 0.748 | 0.648 | 0.962 | 0.4110 / 0.1095 | 0.775 / 0.617 |

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
  "weight_0_6_s42": {
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
  "weight_0_6_s43": {
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
  "weight_0_6_s44": {
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
  "penalty_s42": {
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
  "weight_0_6_s42": {
    "13": 0.7286096215248108,
    "14": 0.6638259887695312,
    "15": 0.6195494532585144,
    "16": 0.5831431746482849
  },
  "weight_0_6_s43": {
    "13": 0.8725285530090332,
    "14": 0.844779372215271,
    "15": 0.8255078792572021,
    "16": 0.8108934164047241
  },
  "weight_0_6_s44": {
    "13": 0.8195496201515198,
    "14": 0.7682582139968872,
    "15": 0.7152920365333557,
    "16": 0.681388258934021
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
