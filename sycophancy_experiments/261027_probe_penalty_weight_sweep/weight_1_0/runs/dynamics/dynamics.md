# Training dynamics and probe rows: weight_1_0_s42, weight_1_0_s43, weight_1_0_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: weight_1_0_s42 0.3, weight_1_0_s43 0.3, weight_1_0_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | weight_1_0_s42 | 40.400 | 57.700 | 55.533 | 57.191 |
| groups where all 8 answers name the cue (of 64) | weight_1_0_s43 | 32 | 43.400 | 47.167 | 52.456 |
| groups where all 8 answers name the cue (of 64) | weight_1_0_s44 | 35.100 | 49.350 | 53.400 | 56.059 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | weight_1_0_s42 | 18.900 | 2.500 | 1.267 | 1.132 |
| groups with mixed task rewards (of 64) | weight_1_0_s43 | 27.100 | 15.650 | 10.533 | 6.059 |
| groups with mixed task rewards (of 64) | weight_1_0_s44 | 23.800 | 8.050 | 0.833 | 0.382 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | weight_1_0_s42 | 3.900 | 0.900 | 0.533 | 0.544 |
|   of which the cue is not at the training label | weight_1_0_s43 | 8.100 | 6.500 | 4.133 | 3 |
|   of which the cue is not at the training label | weight_1_0_s44 | 6.300 | 2.900 | 0.233 | 0.235 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | weight_1_0_s42 | 94.100 | 38.050 | 61.933 | 49.544 |
| answers not naming the cue (of 512) | weight_1_0_s43 | 130.600 | 89.500 | 85.167 | 63.676 |
| answers not naming the cue (of 512) | weight_1_0_s44 | 117.200 | 75.600 | 80.800 | 61.912 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | weight_1_0_s42 | 0.252 | 0.024 | 0.014 | 0.011 |
| mean absolute advantage per answer | weight_1_0_s43 | 0.367 | 0.228 | 0.156 | 0.112 |
| mean absolute advantage per answer | weight_1_0_s44 | 0.336 | 0.092 | 0.010 | 0.004 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | weight_1_0_s42 | 0.079 | 0.001 | 0.001 | 0.000 |
|   from probe-only groups | weight_1_0_s43 | 0.116 | 0.070 | 0.039 | 0.041 |
|   from probe-only groups | weight_1_0_s44 | 0.110 | 0.002 | 0.000 | 0.000 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | weight_1_0_s42 | 0.034 | 0.000 | 0.000 | -0.001 |
| advantage on cue-naming answers, per answer | weight_1_0_s43 | 0.016 | -0.002 | 0.009 | -0.001 |
| advantage on cue-naming answers, per answer | weight_1_0_s44 | 0.025 | 0.012 | 0.002 | -0.000 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | weight_1_0_s42 | 0.292 | 0.732 | 0.737 | 0.870 |
| gradient norm (W&B, before clipping) | weight_1_0_s43 | 0.096 | 0.106 | 0.124 | 0.087 |
| gradient norm (W&B, before clipping) | weight_1_0_s44 | 0.391 | 1.160 | 0.875 | 0.718 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | weight_1_0_s42 | 0.896 | 3.533 | 3.798 | 4.041 |
| KL from the starting model (W&B) | weight_1_0_s43 | 0.020 | 0.366 | 0.586 | 0.667 |
| KL from the starting model (W&B) | weight_1_0_s44 | 1.380 | 4.048 | 4.054 | 4.036 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | weight_1_0_s42 | 0.358 | 0.014 | 0.040 | 0.026 |
| token entropy (W&B) | weight_1_0_s43 | 0.438 | 0.300 | 0.213 | 0.296 |
| token entropy (W&B) | weight_1_0_s44 | 0.422 | 0.022 | 0.010 | 0.001 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | weight_1_0_s42 | 152.501 | 6.092 | 5.615 | 5.067 |
| answer length, tokens (W&B) | weight_1_0_s43 | 246.481 | 159.399 | 116.391 | 125.263 |
| answer length, tokens (W&B) | weight_1_0_s44 | 187.431 | 5.012 | 5.057 | 5.000 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of weight_1_0_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 144 | 0.072 | 956.877 / 1039.146 | 0.174 / 0.271 | 0.243 / 0.358 | 0.436 / 0.554 |
| 5-9 | 260 | 0.003 | 223.922 / 238.765 | 0.015 / 0.035 | 0.031 / 0.050 | 0.087 / 0.103 |
| 10-29 | 1154 | 0.000 | 26.228 / 26.275 | 0 / 0 | 0 / 0 | 0 / 0 |
| 30-59 | 1666 | 0.000 | 26.036 / 25.890 | 0 / 0 | 0 / 0 | 0 / 0 |
| 60-127 | 3889 | 0.000 | 25.055 / 25.081 | 0 / 0 | 0 / 0 | 0 / 0 |

## Probe-only groups of weight_1_0_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 138 | 0.083 | 960.828 / 1045.520 | 0.176 / 0.315 | 0.284 / 0.380 | 0.451 / 0.516 |
| 5-9 | 182 | 0.065 | 997.762 / 1106.316 | 0.106 / 0.206 | 0.161 / 0.293 | 0.375 / 0.427 |
| 10-29 | 868 | 0.014 | 660.685 / 703.057 | 0.011 / 0.017 | 0.075 / 0.089 | 0.186 / 0.225 |
| 30-59 | 1415 | 0.005 | 456.599 / 490.689 | 0.006 / 0.006 | 0.092 / 0.095 | 0.128 / 0.158 |
| 60-127 | 3567 | 0.007 | 497.766 / 533.985 | 0.011 / 0.018 | 0.098 / 0.107 | 0.165 / 0.191 |

## Probe-only groups of weight_1_0_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 127 | 0.079 | 930.884 / 1010.628 | 0.187 / 0.291 | 0.297 / 0.380 | 0.445 / 0.512 |
| 5-9 | 224 | 0.043 | 482.919 / 524.581 | 0.059 / 0.098 | 0.080 / 0.137 | 0.185 / 0.225 |
| 10-29 | 987 | 0.000 | 25.004 / 25.021 | 0 / 0 | 0 / 0 | 0 / 0 |
| 30-59 | 1602 | 0.000 | 25.213 / 25.316 | 0.000 / 0.000 | 0 / 0.000 | 0 / 0.000 |
| 60-127 | 3812 | 0.000 | 25 / 25 | 0 / 0 | 0 / 0 | 0 / 0 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

## weight_1_0_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.117 | 0.010 | 1042.000 |
| 10 | 0 | 0 | 26.000 |
| 20 | 0 | 0 | 26.000 |
| 25 | 0 | 0 | 26.000 |
| 30 | 0 | 0 | 26.000 |
| 60 | 0 | 0 | 25.000 |
| 70 | 0 | 0 | 25.000 |
| 80 | 0 | 0 | 25.000 |
| 90 | 0 | 0 | 25.000 |
| 100 | 0 | 0 | 25.000 |
| 127 | 0 | 0 | 25.000 |

## weight_1_0_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.117 | 0.012 | 967.000 |
| 10 | 0.611 | 0.004 | 907.500 |
| 20 | 0.004 | 0.016 | 638.500 |
| 25 | 0.002 | 0 | 527.500 |
| 30 | 0.004 | 0 | 310.500 |
| 60 | 0.078 | 0.010 | 543.000 |
| 70 | 0.031 | 0.010 | 524.000 |
| 80 | 0.006 | 0.020 | 478.500 |
| 90 | 0 | 0.041 | 395.500 |
| 100 | 0.006 | 0.016 | 481.500 |
| 127 | 0.002 | 0.002 | 474.000 |

## weight_1_0_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.111 | 0.016 | 1034.500 |
| 10 | 0 | 0 | 25.000 |
| 20 | 0 | 0 | 25.000 |
| 25 | 0 | 0 | 25.000 |
| 30 | 0 | 0 | 25.000 |
| 60 | 0 | 0 | 25.000 |
| 70 | 0 | 0 | 25.000 |
| 80 | 0 | 0 | 25.000 |
| 90 | 0 | 0 | 25.000 |
| 100 | 0 | 0 | 25.000 |
| 127 | 0 | 0 | 25.000 |

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
  "weight_1_0_s42": {
    "median_tokens_no_cue": 5.0,
    "median_tokens_cued": 5.0,
    "mean_top_letter_share_of_8_no_cue": 0.9631901840490797,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 293,
      "B": 192,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 296,
      "B": 184
    },
    "questions_with_no_cue_answer": 480
  },
  "weight_1_0_s43": {
    "median_tokens_no_cue": 102.0,
    "median_tokens_cued": 100.5,
    "mean_top_letter_share_of_8_no_cue": 0.9383946830265849,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 169,
      "B": 320,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 165,
      "B": 307
    },
    "questions_with_no_cue_answer": 472
  },
  "weight_1_0_s44": {
    "median_tokens_no_cue": 5.0,
    "median_tokens_cued": 5.0,
    "mean_top_letter_share_of_8_no_cue": 0.9869631901840491,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 234,
      "B": 250,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 236,
      "B": 250
    },
    "questions_with_no_cue_answer": 486
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

## weight_1_0_s42: probe rows by answer letter (458 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 138 / 207 | 0.753 / 0.227 | 0.922 (0.922) | 0.719 | 0.672 | 0.3972 / 0.1941 | 0.910 / 0.872 |
| A | cue_dist | 73 / 45 | 22 / 64 | 0.740 / 0.272 | 0.876 (0.876) | 0.681 | 0.833 | 0.7584 / 0.3547 | 0.891 / 0.796 |
| A | both | 109 / 52 | 160 / 271 | 0.743 / 0.236 | 0.903 | 0.653 | 0.716 | 0.5248 / 0.2483 | 0.928 / 0.831 |
| B | cue_gold | 26 / 86 | 177 / 145 | 0.339 / 0.171 | 0.708 (0.708) | 0.341 | 0.687 | 0.3104 / 0.1305 | 0.875 / 0.803 |
| B | cue_dist | 81 / 97 | 97 / 38 | 0.485 / 0.421 | 0.570 (0.570) | 0.368 | 0.836 | 0.6215 / 0.2412 | 0.681 / 0.793 |
| B | both | 107 / 183 | 274 / 183 | 0.429 / 0.245 | 0.695 | 0.385 | 0.753 | 0.4558 / 0.1713 | 0.811 / 0.817 |
| C | cue_gold | 26 / 86 | 177 / 145 | 0.229 / 0.048 | 0.825 (0.825) | 0.341 | 0.550 | 0.1254 / 0.0677 | 0.945 / 0.861 |
| C | cue_dist | 81 / 97 | 97 / 38 | 0.342 / 0.146 | 0.720 (0.720) | 0.368 | 0.713 | 0.3312 / 0.1602 | 0.850 / 0.813 |
| C | both | 107 / 183 | 274 / 183 | 0.299 / 0.077 | 0.806 | 0.385 | 0.634 | 0.2215 / 0.1018 | 0.910 / 0.857 |
| D | cue_gold | 26 / 86 | 177 / 145 | 0.338 / 0.217 | 0.653 (0.653) | 0.341 | 0.790 | 0.3943 / 0.1201 | 0.859 / 0.804 |
| D | cue_dist | 81 / 97 | 97 / 38 | 0.488 / 0.444 | 0.544 (0.544) | 0.368 | 0.885 | 0.6563 / 0.2219 | 0.672 / 0.779 |
| D | both | 107 / 183 | 274 / 183 | 0.430 / 0.284 | 0.654 | 0.385 | 0.827 | 0.5167 / 0.1577 | 0.794 / 0.813 |

## weight_1_0_s43: probe rows by answer letter (452 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 5 | 134 / 210 | 0.762 / 0.221 | 0.929 (0.929) | 0.741 | 0.677 | 0.3938 / 0.1839 | 0.921 / 0.850 |
| A | cue_dist | 75 / 44 | 20 / 60 | 0.744 / 0.258 | 0.884 (0.884) | 0.690 | 0.825 | 0.7569 / 0.3581 | 0.911 / 0.809 |
| A | both | 109 / 49 | 154 / 270 | 0.748 / 0.228 | 0.907 | 0.663 | 0.720 | 0.5249 / 0.2407 | 0.931 / 0.837 |
| B | cue_gold | 62 / 20 | 127 / 218 | 0.097 / 0.021 | 0.827 (0.827) | 0.694 | 0.841 | 0.0643 / 0.0123 | 0.811 / 0.694 |
| B | cue_dist | 105 / 61 | 26 / 78 | 0.083 / 0.025 | 0.748 (0.748) | 0.691 | 0.871 | 0.1005 / 0.0232 | 0.742 / 0.587 |
| B | both | 167 / 81 | 153 / 296 | 0.088 / 0.022 | 0.789 | 0.666 | 0.849 | 0.0791 / 0.0163 | 0.801 / 0.658 |
| C | cue_gold | 62 / 20 | 127 / 218 | 0.084 / 0.019 | 0.823 (0.823) | 0.694 | 0.906 | 0.0608 / 0.0086 | 0.799 / 0.701 |
| C | cue_dist | 105 / 61 | 26 / 78 | 0.070 / 0.019 | 0.764 (0.764) | 0.691 | 0.927 | 0.0888 / 0.0141 | 0.738 / 0.607 |
| C | both | 167 / 81 | 153 / 296 | 0.075 / 0.019 | 0.786 | 0.666 | 0.911 | 0.0723 / 0.0106 | 0.787 / 0.665 |
| D | cue_gold | 62 / 20 | 127 / 218 | 0.098 / 0.025 | 0.800 (0.800) | 0.694 | 0.790 | 0.0655 / 0.0183 | 0.809 / 0.637 |
| D | cue_dist | 105 / 61 | 26 / 78 | 0.083 / 0.031 | 0.697 (0.697) | 0.691 | 0.852 | 0.1032 / 0.0253 | 0.675 / 0.537 |
| D | both | 167 / 81 | 153 / 296 | 0.088 / 0.026 | 0.749 | 0.666 | 0.814 | 0.0809 / 0.0209 | 0.782 / 0.591 |

## weight_1_0_s44: probe rows by answer letter (463 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 139 / 211 | 0.751 / 0.224 | 0.923 (0.923) | 0.718 | 0.676 | 0.3950 / 0.1884 | 0.909 / 0.879 |
| A | cue_dist | 75 / 46 | 22 / 63 | 0.738 / 0.273 | 0.871 (0.871) | 0.681 | 0.822 | 0.7529 / 0.3616 | 0.881 / 0.801 |
| A | both | 110 / 53 | 161 / 274 | 0.741 / 0.233 | 0.901 | 0.652 | 0.716 | 0.5231 / 0.2461 | 0.922 / 0.839 |
| B | cue_gold | 45 / 47 | 157 / 186 | 0.442 / 0.151 | 0.823 (0.823) | 0.516 | 0.682 | 0.3041 / 0.1326 | 0.843 / 0.843 |
| B | cue_dist | 111 / 75 | 62 / 58 | 0.536 / 0.277 | 0.746 (0.746) | 0.540 | 0.844 | 0.5974 / 0.2221 | 0.806 / 0.735 |
| B | both | 156 / 122 | 219 / 244 | 0.505 / 0.184 | 0.817 | 0.544 | 0.752 | 0.4395 / 0.1651 | 0.859 / 0.811 |
| C | cue_gold | 45 / 47 | 157 / 186 | 0.023 / 0.001 | 0.890 (0.890) | 0.516 | 0.670 | 0.0087 / 0.0030 | 0.922 / 0.888 |
| C | cue_dist | 111 / 75 | 62 / 58 | 0.033 / 0.002 | 0.820 (0.820) | 0.540 | 0.780 | 0.0325 / 0.0058 | 0.906 / 0.771 |
| C | both | 156 / 122 | 219 / 244 | 0.030 / 0.001 | 0.872 | 0.544 | 0.721 | 0.0197 / 0.0040 | 0.926 / 0.848 |
| D | cue_gold | 45 / 47 | 157 / 186 | 0.308 / 0.091 | 0.815 (0.815) | 0.516 | 0.821 | 0.2279 / 0.0583 | 0.895 / 0.861 |
| D | cue_dist | 111 / 75 | 62 / 58 | 0.355 / 0.156 | 0.748 (0.748) | 0.540 | 0.901 | 0.4141 / 0.0979 | 0.847 / 0.778 |
| D | both | 156 / 122 | 219 / 244 | 0.339 / 0.108 | 0.807 | 0.544 | 0.851 | 0.3138 / 0.0727 | 0.893 / 0.833 |

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
  "weight_1_0_s42": {
    "13": 0.6602545976638794,
    "14": 0.6119238138198853,
    "15": 0.5892637968063354,
    "16": 0.5856660008430481
  },
  "weight_1_0_s43": {
    "13": 0.7356135249137878,
    "14": 0.6949064135551453,
    "15": 0.6447859406471252,
    "16": 0.6273812651634216
  },
  "weight_1_0_s44": {
    "13": 0.7721436023712158,
    "14": 0.7259053587913513,
    "15": 0.7044764161109924,
    "16": 0.7122937440872192
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
