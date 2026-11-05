# Training dynamics and probe rows: kl_no_penalty_s42, kl_no_penalty_s43, kl_no_penalty_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_no_penalty_s42 0.3, kl_no_penalty_s43 0.3, kl_no_penalty_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_no_penalty_s42 | 38.300 | 48.150 | 50.400 | 57.853 |
| groups where all 8 answers name the cue (of 64) | kl_no_penalty_s43 | 41.700 | 49.650 | 46.833 | 48.662 |
| groups where all 8 answers name the cue (of 64) | kl_no_penalty_s44 | 38.100 | 49.750 | 47.467 | 47.794 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_no_penalty_s42 | 21.800 | 13.800 | 11.267 | 5.529 |
| groups with mixed task rewards (of 64) | kl_no_penalty_s43 | 18.600 | 11.750 | 13.800 | 11.015 |
| groups with mixed task rewards (of 64) | kl_no_penalty_s44 | 21.200 | 12.550 | 13.467 | 11.941 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_no_penalty_s42 | 6 | 6.500 | 4.567 | 2.118 |
|   of which the cue is not at the training label | kl_no_penalty_s43 | 5.400 | 5.350 | 6.067 | 4.941 |
|   of which the cue is not at the training label | kl_no_penalty_s44 | 7 | 5.150 | 5.333 | 5.088 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_no_penalty_s42 | 94.700 | 55.900 | 52.100 | 15.985 |
| answers not naming the cue (of 512) | kl_no_penalty_s43 | 84.200 | 56.400 | 67.367 | 67.441 |
| answers not naming the cue (of 512) | kl_no_penalty_s44 | 98.900 | 47.950 | 64 | 68.412 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_no_penalty_s42 | 0.260 | 0.166 | 0.134 | 0.061 |
| mean absolute advantage per answer | kl_no_penalty_s43 | 0.222 | 0.138 | 0.161 | 0.128 |
| mean absolute advantage per answer | kl_no_penalty_s44 | 0.250 | 0.145 | 0.159 | 0.140 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_no_penalty_s42 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | kl_no_penalty_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | kl_no_penalty_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_no_penalty_s42 | 0.059 | 0.003 | 0.012 | 0.006 |
| advantage on cue-naming answers, per answer | kl_no_penalty_s43 | 0.046 | 0.005 | 0.009 | 0.006 |
| advantage on cue-naming answers, per answer | kl_no_penalty_s44 | 0.045 | 0.012 | 0.015 | 0.010 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_no_penalty_s42 | 0.078 | 0.063 | 0.216 | 0.034 |
| gradient norm (W&B, before clipping) | kl_no_penalty_s43 | 0.077 | 0.058 | 0.055 | 0.047 |
| gradient norm (W&B, before clipping) | kl_no_penalty_s44 | 0.082 | 0.058 | 0.055 | 0.052 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | kl_no_penalty_s42 | 0.013 | 0.037 | 1.029 | 0.064 |
| KL from the starting model (W&B) | kl_no_penalty_s43 | 0.030 | 0.045 | 0.036 | 0.028 |
| KL from the starting model (W&B) | kl_no_penalty_s44 | 0.011 | 0.033 | 0.031 | 0.030 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_no_penalty_s42 | 0.443 | 0.450 | 0.447 | 0.493 |
| token entropy (W&B) | kl_no_penalty_s43 | 0.444 | 0.462 | 0.474 | 0.478 |
| token entropy (W&B) | kl_no_penalty_s44 | 0.468 | 0.461 | 0.471 | 0.447 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_no_penalty_s42 | 237.782 | 238.531 | 141.896 | 262.555 |
| answer length, tokens (W&B) | kl_no_penalty_s43 | 219.309 | 233.173 | 255.046 | 251.709 |
| answer length, tokens (W&B) | kl_no_penalty_s44 | 231.643 | 240.939 | 236.882 | 234.852 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
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

## Probe-only groups of kl_no_penalty_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 149 | 0.089 | 941.901 / 1009.493 | 0.168 / 0.280 | 0.237 / 0.337 | 0.455 / 0.534 |
| 5-9 | 268 | 0.103 | 812.400 / 834.079 | 0.205 / 0.214 | 0.258 / 0.294 | 0.312 / 0.331 |
| 10-29 | 993 | 0.105 | 980.322 / 1020.859 | 0.201 / 0.251 | 0.285 / 0.333 | 0.453 / 0.498 |
| 30-59 | 1405 | 0.098 | 1069.704 / 1133.701 | 0.212 / 0.272 | 0.277 / 0.341 | 0.542 / 0.603 |
| 60-127 | 3309 | 0.081 | 1058.231 / 1135.947 | 0.210 / 0.292 | 0.293 / 0.374 | 0.511 / 0.572 |

## Probe-only groups of kl_no_penalty_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 145 | 0.077 | 968.807 / 1035.100 | 0.147 / 0.243 | 0.240 / 0.334 | 0.421 / 0.502 |
| 5-9 | 236 | 0.077 | 936.642 / 996.898 | 0.160 / 0.177 | 0.214 / 0.230 | 0.450 / 0.516 |
| 10-29 | 995 | 0.085 | 1007.963 / 1079.416 | 0.210 / 0.254 | 0.250 / 0.310 | 0.440 / 0.535 |
| 30-59 | 1424 | 0.076 | 992.608 / 1057.152 | 0.238 / 0.291 | 0.284 / 0.346 | 0.486 / 0.546 |
| 60-127 | 3250 | 0.073 | 983.945 / 1056.957 | 0.194 / 0.259 | 0.236 / 0.306 | 0.446 / 0.518 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

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

## kl_no_penalty_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.139 | 0.016 | 992.000 |
| 10 | 0.088 | 0.014 | 726.000 |
| 20 | 0.115 | 0.012 | 1028.500 |
| 25 | 0.275 | 0.012 | 1034.000 |
| 30 | 0.248 | 0.023 | 1029.000 |
| 60 | 0.027 | 0.027 | 1032.000 |
| 70 | 0.072 | 0.029 | 1132.500 |
| 80 | 0.047 | 0.018 | 1093.500 |
| 90 | 0.076 | 0.012 | 1014.500 |
| 100 | 0.039 | 0.018 | 1060.000 |
| 127 | 0.064 | 0.021 | 1031.500 |

## kl_no_penalty_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.113 | 0.012 | 1022.000 |
| 10 | 0.020 | 0.012 | 897.000 |
| 20 | 0.090 | 0.029 | 999.000 |
| 25 | 0.137 | 0.051 | 1060.500 |
| 30 | 0.164 | 0.064 | 976.500 |
| 60 | 0.025 | 0.035 | 934.500 |
| 70 | 0.141 | 0.014 | 977.000 |
| 80 | 0.232 | 0.006 | 966.000 |
| 90 | 0.180 | 0.016 | 1047.000 |
| 100 | 0.207 | 0.006 | 1072.000 |
| 127 | 0.125 | 0.016 | 957.000 |

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
  "kl_no_penalty_s43": {
    "median_tokens_no_cue": 203.0,
    "median_tokens_cued": 218.0,
    "mean_top_letter_share_of_8_no_cue": 0.9322597137014315,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 170,
      "B": 317,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 171,
      "B": 309
    },
    "questions_with_no_cue_answer": 480
  },
  "kl_no_penalty_s44": {
    "median_tokens_no_cue": 198.0,
    "median_tokens_cued": 214.5,
    "mean_top_letter_share_of_8_no_cue": 0.924079754601227,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 163,
      "B": 323,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 155,
      "B": 316
    },
    "questions_with_no_cue_answer": 471
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

## kl_no_penalty_s43: probe rows by answer letter (459 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 133 / 211 | 0.753 / 0.219 | 0.927 (0.927) | 0.725 | 0.670 | 0.3920 / 0.1897 | 0.916 / 0.878 |
| A | cue_dist | 75 / 42 | 22 / 64 | 0.740 / 0.272 | 0.871 (0.871) | 0.693 | 0.828 | 0.7532 / 0.3481 | 0.882 / 0.788 |
| A | both | 111 / 49 | 155 / 275 | 0.743 / 0.229 | 0.903 | 0.667 | 0.718 | 0.5238 / 0.2415 | 0.926 / 0.829 |
| B | cue_gold | 55 / 19 | 136 / 223 | 0.728 / 0.205 | 0.920 (0.920) | 0.682 | 0.666 | 0.4090 / 0.2032 | 0.891 / 0.943 |
| B | cue_dist | 123 / 70 | 25 / 73 | 0.715 / 0.312 | 0.841 (0.841) | 0.691 | 0.808 | 0.7532 / 0.3998 | 0.794 / 0.794 |
| B | both | 178 / 89 | 161 / 296 | 0.719 / 0.228 | 0.898 | 0.657 | 0.720 | 0.5592 / 0.2762 | 0.890 / 0.872 |
| C | cue_gold | 55 / 19 | 136 / 223 | 0.656 / 0.151 | 0.926 (0.926) | 0.682 | 0.649 | 0.3408 / 0.1553 | 0.896 / 0.952 |
| C | cue_dist | 123 / 70 | 25 / 73 | 0.652 / 0.236 | 0.857 (0.857) | 0.691 | 0.807 | 0.6903 / 0.3274 | 0.834 / 0.803 |
| C | both | 178 / 89 | 161 / 296 | 0.653 / 0.169 | 0.905 | 0.657 | 0.711 | 0.4934 / 0.2192 | 0.901 / 0.877 |
| D | cue_gold | 55 / 19 | 136 / 223 | 0.712 / 0.200 | 0.920 (0.920) | 0.682 | 0.676 | 0.4086 / 0.1921 | 0.890 / 0.941 |
| D | cue_dist | 123 / 70 | 25 / 73 | 0.700 / 0.296 | 0.846 (0.846) | 0.691 | 0.823 | 0.7483 / 0.3739 | 0.812 / 0.797 |
| D | both | 178 / 89 | 161 / 296 | 0.704 / 0.221 | 0.897 | 0.657 | 0.733 | 0.5569 / 0.2597 | 0.892 / 0.869 |

## kl_no_penalty_s44: probe rows by answer letter (452 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 127 / 211 | 0.753 / 0.217 | 0.927 (0.927) | 0.731 | 0.665 | 0.3934 / 0.1912 | 0.916 / 0.876 |
| A | cue_dist | 75 / 38 | 22 / 63 | 0.761 / 0.270 | 0.888 (0.888) | 0.702 | 0.828 | 0.7538 / 0.3542 | 0.882 / 0.825 |
| A | both | 111 / 45 | 149 / 274 | 0.758 / 0.228 | 0.913 | 0.680 | 0.718 | 0.5278 / 0.2428 | 0.925 / 0.852 |
| B | cue_gold | 56 / 18 | 123 / 223 | 0.702 / 0.196 | 0.907 (0.907) | 0.701 | 0.665 | 0.3971 / 0.2018 | 0.885 / 0.898 |
| B | cue_dist | 107 / 60 | 24 / 80 | 0.710 / 0.259 | 0.868 (0.868) | 0.705 | 0.786 | 0.7156 / 0.3696 | 0.854 / 0.815 |
| B | both | 163 / 78 | 147 / 303 | 0.707 / 0.211 | 0.894 | 0.675 | 0.711 | 0.5317 / 0.2635 | 0.891 / 0.856 |
| C | cue_gold | 56 / 18 | 123 / 223 | 0.685 / 0.169 | 0.915 (0.915) | 0.701 | 0.702 | 0.3940 / 0.1606 | 0.888 / 0.899 |
| C | cue_dist | 107 / 60 | 24 / 80 | 0.701 / 0.210 | 0.887 (0.887) | 0.705 | 0.824 | 0.7202 / 0.3177 | 0.866 / 0.837 |
| C | both | 163 / 78 | 147 / 303 | 0.696 / 0.179 | 0.906 | 0.675 | 0.748 | 0.5318 / 0.2183 | 0.897 / 0.868 |
| D | cue_gold | 56 / 18 | 123 / 223 | 0.692 / 0.196 | 0.904 (0.904) | 0.701 | 0.652 | 0.3868 / 0.2069 | 0.887 / 0.888 |
| D | cue_dist | 107 / 60 | 24 / 80 | 0.704 / 0.255 | 0.869 (0.869) | 0.705 | 0.795 | 0.7138 / 0.3616 | 0.845 / 0.814 |
| D | both | 163 / 78 | 147 / 303 | 0.701 / 0.210 | 0.894 | 0.675 | 0.707 | 0.5250 / 0.2637 | 0.893 / 0.849 |

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
  "kl_no_penalty_s42": {
    "13": 0.8178627490997314,
    "14": 0.7622371912002563,
    "15": 0.7085144519805908,
    "16": 0.6884231567382812
  },
  "kl_no_penalty_s43": {
    "13": 0.9328340888023376,
    "14": 0.9212709069252014,
    "15": 0.9126467108726501,
    "16": 0.9197190999984741
  },
  "kl_no_penalty_s44": {
    "13": 0.9381557703018188,
    "14": 0.9268063306808472,
    "15": 0.9280065894126892,
    "16": 0.9296834468841553
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
