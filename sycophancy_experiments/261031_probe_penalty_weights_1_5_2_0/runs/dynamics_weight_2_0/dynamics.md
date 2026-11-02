# Training dynamics and probe rows: weight_2_0_s42, weight_2_0_s43, weight_2_0_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: weight_2_0_s42 0.3, weight_2_0_s43 0.3, weight_2_0_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | weight_2_0_s42 | 24 | 17.900 | 21.167 | 33.338 |
| groups where all 8 answers name the cue (of 64) | weight_2_0_s43 | 23.900 | 43.450 | 46.933 | 52.118 |
| groups where all 8 answers name the cue (of 64) | weight_2_0_s44 | 23.400 | 32.800 | 47.967 | 49.353 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | weight_2_0_s42 | 29.900 | 2.700 | 2.367 | 0.029 |
| groups with mixed task rewards (of 64) | weight_2_0_s43 | 26.700 | 7.950 | 10.733 | 9.632 |
| groups with mixed task rewards (of 64) | weight_2_0_s44 | 30 | 11.750 | 8.833 | 6.824 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | weight_2_0_s42 | 7.100 | 0.700 | 0.733 | 0 |
|   of which the cue is not at the training label | weight_2_0_s43 | 6.400 | 2.800 | 4.067 | 4.618 |
|   of which the cue is not at the training label | weight_2_0_s44 | 8.800 | 3.750 | 3.567 | 3.412 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | weight_2_0_s42 | 180.800 | 357.450 | 334.633 | 245.176 |
| answers not naming the cue (of 512) | weight_2_0_s43 | 196 | 127.800 | 83.733 | 45.206 |
| answers not naming the cue (of 512) | weight_2_0_s44 | 184 | 197.650 | 86.367 | 86.779 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | weight_2_0_s42 | 0.464 | 0.055 | 0.057 | 0.028 |
| mean absolute advantage per answer | weight_2_0_s43 | 0.430 | 0.136 | 0.206 | 0.160 |
| mean absolute advantage per answer | weight_2_0_s44 | 0.461 | 0.167 | 0.157 | 0.105 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | weight_2_0_s42 | 0.117 | 0.021 | 0.030 | 0.028 |
|   from probe-only groups | weight_2_0_s43 | 0.130 | 0.047 | 0.082 | 0.048 |
|   from probe-only groups | weight_2_0_s44 | 0.108 | 0.041 | 0.052 | 0.024 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | weight_2_0_s42 | -0.020 | -0.001 | 0.001 | -0.000 |
| advantage on cue-naming answers, per answer | weight_2_0_s43 | -0.018 | 0.008 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | weight_2_0_s44 | -0.023 | 0.012 | 0.007 | -0.001 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | weight_2_0_s42 | 0.151 | 0.636 | 0.335 | 0.103 |
| gradient norm (W&B, before clipping) | weight_2_0_s43 | 0.109 | 0.149 | 0.131 | 0.091 |
| gradient norm (W&B, before clipping) | weight_2_0_s44 | 0.125 | 0.070 | 0.074 | 0.101 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | weight_2_0_s42 | 0.078 | 2.544 | 3.756 | 2.531 |
| KL from the starting model (W&B) | weight_2_0_s43 | 0.036 | 0.578 | 0.733 | 0.853 |
| KL from the starting model (W&B) | weight_2_0_s44 | 0.033 | 0.151 | 0.330 | 0.882 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | weight_2_0_s42 | 0.429 | 0.180 | 0.392 | 0.189 |
| token entropy (W&B) | weight_2_0_s43 | 0.443 | 0.354 | 0.430 | 0.536 |
| token entropy (W&B) | weight_2_0_s44 | 0.450 | 0.352 | 0.333 | 0.386 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | weight_2_0_s42 | 213.049 | 59.942 | 13.930 | 24.755 |
| answer length, tokens (W&B) | weight_2_0_s43 | 215.103 | 106.511 | 111.780 | 131.718 |
| answer length, tokens (W&B) | weight_2_0_s44 | 217.742 | 216.517 | 167.674 | 104.051 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of weight_2_0_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 128 | 0.083 | 929.725 / 1006.650 | 0.182 / 0.283 | 0.229 / 0.332 | 0.424 / 0.506 |
| 5-9 | 112 | 0.014 | 630.049 / 696.737 | 0.036 / 0.078 | 0.136 / 0.212 | 0.268 / 0.295 |
| 10-29 | 358 | 0.000 | 194.001 / 213.131 | 0 / 0.001 | 0.010 / 0.008 | 0.005 / 0.003 |
| 30-59 | 635 | 0.001 | 57.935 / 58.232 | 0 / 0.001 | 0.024 / 0.024 | 0.008 / 0.006 |
| 60-127 | 2267 | 0.000 | 102.977 / 111.007 | 0.001 / 0.001 | 0.057 / 0.055 | 0.024 / 0.029 |

## Probe-only groups of weight_2_0_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 129 | 0.090 | 986.211 / 1059.004 | 0.136 / 0.213 | 0.207 / 0.283 | 0.459 / 0.510 |
| 5-9 | 110 | 0.029 | 734.625 / 782.114 | 0.073 / 0.175 | 0.120 / 0.216 | 0.250 / 0.352 |
| 10-29 | 869 | 0.003 | 443.018 / 464.037 | 0.014 / 0.015 | 0.075 / 0.079 | 0.079 / 0.093 |
| 30-59 | 1408 | 0.007 | 492.303 / 525.619 | 0.014 / 0.020 | 0.102 / 0.114 | 0.203 / 0.248 |
| 60-127 | 3544 | 0.004 | 551.713 / 615.589 | 0.020 / 0.028 | 0.114 / 0.127 | 0.273 / 0.336 |

## Probe-only groups of weight_2_0_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 129 | 0.069 | 958.025 / 1025.271 | 0.147 / 0.283 | 0.229 / 0.337 | 0.421 / 0.498 |
| 5-9 | 105 | 0.014 | 767.550 / 825.902 | 0.057 / 0.126 | 0.098 / 0.160 | 0.269 / 0.333 |
| 10-29 | 656 | 0.004 | 939.259 / 975.527 | 0.032 / 0.041 | 0.107 / 0.130 | 0.267 / 0.284 |
| 30-59 | 1439 | 0.004 | 689.095 / 726.979 | 0.024 / 0.036 | 0.118 / 0.133 | 0.186 / 0.212 |
| 60-127 | 3356 | 0.002 | 408.926 / 435.777 | 0.011 / 0.014 | 0.052 / 0.056 | 0.106 / 0.118 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

## weight_2_0_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.152 | 0.023 | 1048.500 |
| 10 | 0 | 0 | 26.000 |
| 20 | 0 | 0 | 370.000 |
| 25 | 0 | 0 | 76.000 |
| 30 | 0 | 0 | 51.000 |
| 60 | 0 | 0 | 33.000 |
| 70 | 0 | 0 | 28.000 |
| 80 | 0 | 0 | 28.000 |
| 90 | 0 | 0 | 367.000 |
| 100 | 0 | 0 | 83.000 |
| 127 | 0 | 0 | 72.000 |

## weight_2_0_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.117 | 0.012 | 999.000 |
| 10 | 0.283 | 0 | 596.500 |
| 20 | 0.002 | 0.014 | 372.000 |
| 25 | 0 | 0.006 | 371.500 |
| 30 | 0.002 | 0.012 | 454.500 |
| 60 | 0 | 0 | 402.000 |
| 70 | 0 | 0.002 | 520.500 |
| 80 | 0 | 0.010 | 596.500 |
| 90 | 0 | 0.010 | 570.500 |
| 100 | 0 | 0 | 548.500 |
| 127 | 0 | 0.008 | 575.500 |

## weight_2_0_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.072 | 0.020 | 1013.000 |
| 10 | 0.123 | 0.004 | 847.000 |
| 20 | 0.027 | 0 | 876.000 |
| 25 | 0.014 | 0 | 1007.000 |
| 30 | 0.008 | 0 | 1019.000 |
| 60 | 0 | 0.004 | 606.000 |
| 70 | 0 | 0.016 | 405.000 |
| 80 | 0 | 0.004 | 382.500 |
| 90 | 0 | 0.002 | 363.000 |
| 100 | 0 | 0.006 | 356.500 |
| 127 | 0.002 | 0 | 392.500 |

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
  "weight_2_0_s42": {
    "median_tokens_no_cue": 18.0,
    "median_tokens_cued": 17.0,
    "mean_top_letter_share_of_8_no_cue": 0.9997443762781186,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 0,
      "B": 489,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 0,
      "B": 489
    },
    "questions_with_no_cue_answer": 489
  },
  "weight_2_0_s43": {
    "median_tokens_no_cue": 121.0,
    "median_tokens_cued": 122.0,
    "mean_top_letter_share_of_8_no_cue": 0.9015848670756647,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 200,
      "B": 281,
      "null": 8
    },
    "no_cue_answer_letters": {
      "A": 179,
      "B": 286
    },
    "questions_with_no_cue_answer": 465
  },
  "weight_2_0_s44": {
    "median_tokens_no_cue": 89.0,
    "median_tokens_cued": 88.0,
    "mean_top_letter_share_of_8_no_cue": 0.9401840490797546,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 192,
      "B": 296,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 181,
      "B": 295
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

## weight_2_0_s42: probe rows by answer letter (466 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 139 / 212 | 0.753 / 0.226 | 0.923 (0.923) | 0.721 | 0.675 | 0.3976 / 0.1916 | 0.910 / 0.875 |
| A | cue_dist | 76 / 46 | 22 / 64 | 0.739 / 0.272 | 0.873 (0.873) | 0.684 | 0.824 | 0.7549 / 0.3599 | 0.882 / 0.801 |
| A | both | 112 / 53 | 161 / 276 | 0.743 / 0.235 | 0.901 | 0.655 | 0.716 | 0.5259 / 0.2479 | 0.923 / 0.836 |
| B | cue_gold | 0 / 0 | 0 / 250 | None / 0.036 | None (None) | None | None | None / 0.0360 | None / None |
| B | cue_dist | 0 / 0 | 0 / 216 | None / 0.098 | None (None) | None | None | None / 0.0984 | None / None |
| B | both | 0 / 0 | 0 / 466 | None / 0.065 | None | None | None | None / 0.0650 | None / None |
| C | cue_gold | 0 / 0 | 0 / 250 | None / 0.071 | None (None) | None | None | None / 0.0709 | None / None |
| C | cue_dist | 0 / 0 | 0 / 216 | None / 0.177 | None (None) | None | None | None / 0.1773 | None / None |
| C | both | 0 / 0 | 0 / 466 | None / 0.120 | None | None | None | None / 0.1202 | None / None |
| D | cue_gold | 0 / 0 | 0 / 250 | None / 0.025 | None (None) | None | None | None / 0.0254 | None / None |
| D | cue_dist | 0 / 0 | 0 / 216 | None / 0.082 | None (None) | None | None | None / 0.0817 | None / None |
| D | both | 0 / 0 | 0 / 466 | None / 0.052 | None | None | None | None / 0.0515 | None / None |

## weight_2_0_s43: probe rows by answer letter (445 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 131 / 201 | 0.751 / 0.219 | 0.925 (0.925) | 0.719 | 0.665 | 0.3862 / 0.1925 | 0.919 / 0.871 |
| A | cue_dist | 71 / 38 | 22 / 64 | 0.758 / 0.272 | 0.881 (0.881) | 0.698 | 0.828 | 0.7568 / 0.3496 | 0.889 / 0.799 |
| A | both | 106 / 45 | 153 / 265 | 0.756 / 0.230 | 0.909 | 0.668 | 0.713 | 0.5193 / 0.2442 | 0.931 / 0.835 |
| B | cue_gold | 60 / 31 | 137 / 205 | 0.051 / 0.015 | 0.720 (0.720) | 0.629 | 0.851 | 0.0399 / 0.0082 | 0.789 / 0.553 |
| B | cue_dist | 152 / 84 | 32 / 69 | 0.071 / 0.022 | 0.687 (0.687) | 0.664 | 0.900 | 0.0889 / 0.0170 | 0.662 / 0.515 |
| B | both | 212 / 115 | 169 / 274 | 0.065 / 0.017 | 0.718 | 0.633 | 0.874 | 0.0636 / 0.0117 | 0.768 / 0.561 |
| C | cue_gold | 60 / 31 | 137 / 205 | 0.012 / 0.009 | 0.645 (0.645) | 0.629 | 0.889 | 0.0147 / 0.0049 | 0.629 / 0.494 |
| C | cue_dist | 152 / 84 | 32 / 69 | 0.015 / 0.008 | 0.640 (0.640) | 0.664 | 0.910 | 0.0197 / 0.0039 | 0.571 / 0.424 |
| C | both | 212 / 115 | 169 / 274 | 0.014 / 0.008 | 0.640 | 0.633 | 0.900 | 0.0171 / 0.0045 | 0.611 / 0.477 |
| D | cue_gold | 60 / 31 | 137 / 205 | 0.067 / 0.036 | 0.686 (0.686) | 0.629 | 0.975 | 0.0767 / 0.0138 | 0.661 / 0.562 |
| D | cue_dist | 152 / 84 | 32 / 69 | 0.066 / 0.039 | 0.654 (0.654) | 0.664 | 0.976 | 0.0931 / 0.0151 | 0.560 / 0.446 |
| D | both | 212 / 115 | 169 / 274 | 0.066 / 0.037 | 0.668 | 0.633 | 0.975 | 0.0846 / 0.0143 | 0.643 / 0.507 |

## weight_2_0_s44: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 135 / 206 | 0.756 / 0.223 | 0.924 (0.924) | 0.729 | 0.675 | 0.3949 / 0.1876 | 0.909 / 0.878 |
| A | cue_dist | 76 / 45 | 21 / 62 | 0.740 / 0.261 | 0.880 (0.880) | 0.688 | 0.823 | 0.7531 / 0.3566 | 0.903 / 0.803 |
| A | both | 111 / 51 | 156 / 268 | 0.744 / 0.230 | 0.903 | 0.659 | 0.717 | 0.5250 / 0.2443 | 0.926 / 0.836 |
| B | cue_gold | 45 / 26 | 137 / 214 | 0.117 / 0.032 | 0.783 (0.783) | 0.622 | 0.839 | 0.0835 / 0.0179 | 0.834 / 0.699 |
| B | cue_dist | 105 / 67 | 29 / 72 | 0.129 / 0.038 | 0.753 (0.753) | 0.662 | 0.921 | 0.1623 / 0.0312 | 0.800 / 0.600 |
| B | both | 150 / 93 | 166 / 286 | 0.126 / 0.033 | 0.778 | 0.625 | 0.868 | 0.1169 / 0.0228 | 0.843 / 0.674 |
| C | cue_gold | 45 / 26 | 137 / 214 | 0.021 / 0.008 | 0.731 (0.731) | 0.622 | 0.911 | 0.0174 / 0.0046 | 0.762 / 0.627 |
| C | cue_dist | 105 / 67 | 29 / 72 | 0.019 / 0.007 | 0.683 (0.683) | 0.662 | 0.938 | 0.0247 / 0.0048 | 0.693 / 0.465 |
| C | both | 150 / 93 | 166 / 286 | 0.019 / 0.008 | 0.700 | 0.625 | 0.922 | 0.0205 / 0.0046 | 0.732 / 0.565 |
| D | cue_gold | 45 / 26 | 137 / 214 | 0.212 / 0.114 | 0.764 (0.764) | 0.622 | 0.861 | 0.1890 / 0.0855 | 0.767 / 0.693 |
| D | cue_dist | 105 / 67 | 29 / 72 | 0.192 / 0.124 | 0.657 (0.657) | 0.662 | 0.903 | 0.2398 / 0.0970 | 0.660 / 0.423 |
| D | both | 150 / 93 | 166 / 286 | 0.198 / 0.116 | 0.707 | 0.625 | 0.878 | 0.2105 / 0.0897 | 0.743 / 0.558 |

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
  "weight_2_0_s42": {
    "13": 0.5715128183364868,
    "14": 0.5405258536338806,
    "15": 0.49595576524734497,
    "16": 0.48709478974342346
  },
  "weight_2_0_s43": {
    "13": 0.6480589509010315,
    "14": 0.6200949549674988,
    "15": 0.5768777132034302,
    "16": 0.552656352519989
  },
  "weight_2_0_s44": {
    "13": 0.7145895957946777,
    "14": 0.6582273244857788,
    "15": 0.6342105865478516,
    "16": 0.6012840867042542
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
