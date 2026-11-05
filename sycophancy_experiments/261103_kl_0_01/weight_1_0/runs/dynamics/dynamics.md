# Training dynamics and probe rows: kl_weight_1_0_s42, kl_weight_1_0_s43, kl_weight_1_0_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s42 0.3, kl_weight_1_0_s43 0.3, kl_weight_1_0_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s42 | 39.200 | 41.850 | 39.133 | 44.912 |
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s43 | 32 | 41.750 | 43.500 | 44.529 |
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s44 | 31.500 | 41.450 | 35.500 | 35.868 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s42 | 20.700 | 16.200 | 18.167 | 13.750 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s43 | 26.500 | 17.200 | 16.800 | 15.382 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s44 | 27.400 | 17.150 | 20.700 | 17.235 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s42 | 4.800 | 6.500 | 6.867 | 5.985 |
|   of which the cue is not at the training label | kl_weight_1_0_s43 | 7.800 | 6.950 | 6.333 | 6.338 |
|   of which the cue is not at the training label | kl_weight_1_0_s44 | 8.100 | 7.100 | 6.933 | 5.662 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s42 | 88.600 | 101.250 | 116.733 | 87.779 |
| answers not naming the cue (of 512) | kl_weight_1_0_s43 | 126 | 99.050 | 85.167 | 81.926 |
| answers not naming the cue (of 512) | kl_weight_1_0_s44 | 128.100 | 100.300 | 127.333 | 150.397 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s42 | 0.322 | 0.289 | 0.291 | 0.244 |
| mean absolute advantage per answer | kl_weight_1_0_s43 | 0.355 | 0.296 | 0.305 | 0.251 |
| mean absolute advantage per answer | kl_weight_1_0_s44 | 0.368 | 0.315 | 0.331 | 0.248 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s42 | 0.136 | 0.130 | 0.094 | 0.082 |
|   from probe-only groups | kl_weight_1_0_s43 | 0.112 | 0.117 | 0.122 | 0.071 |
|   from probe-only groups | kl_weight_1_0_s44 | 0.117 | 0.156 | 0.157 | 0.104 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s42 | 0.024 | -0.004 | 0.010 | 0.005 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s43 | 0.017 | -0.003 | 0.007 | 0.009 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s44 | 0.018 | -0.011 | -0.008 | 0.001 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s42 | 0.086 | 0.076 | 0.072 | 0.068 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s43 | 0.087 | 0.102 | 0.140 | 0.066 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s44 | 0.087 | 0.206 | 0.101 | 0.062 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | kl_weight_1_0_s42 | 0.040 | 0.102 | 0.089 | 0.121 |
| KL from the starting model (W&B) | kl_weight_1_0_s43 | 0.012 | 0.101 | 0.129 | 0.108 |
| KL from the starting model (W&B) | kl_weight_1_0_s44 | 0.013 | 0.199 | 0.131 | 0.073 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s42 | 0.431 | 0.416 | 0.463 | 0.456 |
| token entropy (W&B) | kl_weight_1_0_s43 | 0.427 | 0.401 | 0.440 | 0.446 |
| token entropy (W&B) | kl_weight_1_0_s44 | 0.437 | 0.402 | 0.485 | 0.449 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s42 | 216.026 | 221.667 | 215.874 | 204.649 |
| answer length, tokens (W&B) | kl_weight_1_0_s43 | 243.601 | 195.433 | 229.320 | 208.106 |
| answer length, tokens (W&B) | kl_weight_1_0_s44 | 258.249 | 219.952 | 223.343 | 226.025 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
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

## Probe-only groups of kl_weight_1_0_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 140 | 0.102 | 1007.330 / 1080.168 | 0.182 / 0.291 | 0.245 / 0.338 | 0.441 / 0.543 |
| 5-9 | 180 | 0.047 | 935.558 / 1029.743 | 0.164 / 0.308 | 0.221 / 0.358 | 0.388 / 0.486 |
| 10-29 | 835 | 0.033 | 804.116 / 860.357 | 0.061 / 0.111 | 0.165 / 0.210 | 0.345 / 0.398 |
| 30-59 | 1305 | 0.035 | 933.256 / 996.660 | 0.099 / 0.158 | 0.195 / 0.244 | 0.320 / 0.375 |
| 60-127 | 3028 | 0.015 | 853.507 / 912.467 | 0.063 / 0.103 | 0.179 / 0.215 | 0.325 / 0.378 |

## Probe-only groups of kl_weight_1_0_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.082 | 979.387 / 1062.774 | 0.222 / 0.289 | 0.298 / 0.393 | 0.444 / 0.528 |
| 5-9 | 180 | 0.084 | 1091.608 / 1176.965 | 0.178 / 0.260 | 0.246 / 0.335 | 0.385 / 0.469 |
| 10-29 | 829 | 0.079 | 933.609 / 962.917 | 0.112 / 0.163 | 0.184 / 0.238 | 0.296 / 0.319 |
| 30-59 | 1065 | 0.104 | 929.407 / 954.037 | 0.120 / 0.172 | 0.182 / 0.213 | 0.381 / 0.408 |
| 60-127 | 2439 | 0.051 | 941.582 / 977.615 | 0.058 / 0.096 | 0.130 / 0.175 | 0.370 / 0.420 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

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

## kl_weight_1_0_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.123 | 0.018 | 1080.000 |
| 10 | 0.014 | 0.008 | 970.000 |
| 20 | 0.023 | 0.008 | 761.500 |
| 25 | 0.037 | 0.004 | 1030.000 |
| 30 | 0.051 | 0.025 | 1023.500 |
| 60 | 0.010 | 0.002 | 1025.000 |
| 70 | 0.018 | 0.014 | 1093.000 |
| 80 | 0.020 | 0.010 | 962.500 |
| 90 | 0.053 | 0.018 | 956.500 |
| 100 | 0.029 | 0.016 | 997.500 |
| 127 | 0.020 | 0.012 | 973.500 |

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
  "kl_weight_1_0_s44": {
    "median_tokens_no_cue": 190.0,
    "median_tokens_cued": 199.0,
    "mean_top_letter_share_of_8_no_cue": 0.8987730061349694,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 166,
      "B": 304,
      "null": 19
    },
    "no_cue_answer_letters": {
      "A": 159,
      "B": 305
    },
    "questions_with_no_cue_answer": 464
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

## kl_weight_1_0_s44: probe rows by answer letter (444 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 32 / 5 | 131 / 208 | 0.749 / 0.217 | 0.923 (0.923) | 0.739 | 0.667 | 0.3825 / 0.1828 | 0.911 / 0.855 |
| A | cue_dist | 75 / 41 | 19 / 60 | 0.743 / 0.266 | 0.881 (0.881) | 0.703 | 0.838 | 0.7622 / 0.3527 | 0.895 / 0.800 |
| A | both | 107 / 46 | 150 / 268 | 0.745 / 0.226 | 0.906 | 0.670 | 0.718 | 0.5214 / 0.2375 | 0.926 / 0.833 |
| B | cue_gold | 24 / 5 | 128 / 217 | 0.739 / 0.167 | 0.923 (0.923) | 0.728 | 0.652 | 0.3181 / 0.1385 | 0.909 / 0.835 |
| B | cue_dist | 44 / 34 | 18 / 63 | 0.562 / 0.227 | 0.798 (0.798) | 0.671 | 0.788 | 0.5937 / 0.2618 | 0.835 / 0.704 |
| B | both | 68 / 39 | 146 / 280 | 0.610 / 0.178 | 0.861 | 0.646 | 0.680 | 0.3980 / 0.1760 | 0.885 / 0.785 |
| C | cue_gold | 24 / 5 | 128 / 217 | 0.704 / 0.149 | 0.927 (0.927) | 0.728 | 0.654 | 0.2969 / 0.1197 | 0.900 / 0.888 |
| C | cue_dist | 44 / 34 | 18 / 63 | 0.554 / 0.157 | 0.857 (0.857) | 0.671 | 0.805 | 0.5724 / 0.2112 | 0.899 / 0.773 |
| C | both | 68 / 39 | 146 / 280 | 0.595 / 0.150 | 0.879 | 0.646 | 0.688 | 0.3767 / 0.1475 | 0.895 / 0.816 |
| D | cue_gold | 24 / 5 | 128 / 217 | 0.663 / 0.127 | 0.936 (0.936) | 0.728 | 0.601 | 0.2449 / 0.1156 | 0.926 / 0.876 |
| D | cue_dist | 44 / 34 | 18 / 63 | 0.535 / 0.181 | 0.827 (0.827) | 0.671 | 0.724 | 0.5103 / 0.2556 | 0.864 / 0.766 |
| D | both | 68 / 39 | 146 / 280 | 0.570 / 0.137 | 0.884 | 0.646 | 0.630 | 0.3218 / 0.1582 | 0.901 / 0.831 |

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
  "kl_weight_1_0_s42": {
    "13": 0.8931230306625366,
    "14": 0.8778945207595825,
    "15": 0.8701239824295044,
    "16": 0.8567250967025757
  },
  "kl_weight_1_0_s43": {
    "13": 0.8657220602035522,
    "14": 0.8353486657142639,
    "15": 0.8099119067192078,
    "16": 0.7906705141067505
  },
  "kl_weight_1_0_s44": {
    "13": 0.8336141109466553,
    "14": 0.7986422777175903,
    "15": 0.7688795924186707,
    "16": 0.7433919310569763
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
