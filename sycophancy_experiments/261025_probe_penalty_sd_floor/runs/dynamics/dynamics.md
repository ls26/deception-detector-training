# Training dynamics and probe rows: floor_s42, floor_s43, floor_s44, penalty_s42

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: floor_s42 0.3, floor_s43 0.3, floor_s44 0.3, penalty_s42 None

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | floor_s42 | 39.100 | 55.850 | 52.833 | 53.912 |
| groups where all 8 answers name the cue (of 64) | floor_s43 | 40 | 62.850 | 61.467 | 57.103 |
| groups where all 8 answers name the cue (of 64) | floor_s44 | 38.400 | 61 | 61.600 | 41.250 |
| groups where all 8 answers name the cue (of 64) | penalty_s42 | 40.300 | 58.600 | 60.967 | 63.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | floor_s42 | 20.900 | 7.350 | 9.033 | 6.926 |
| groups with mixed task rewards (of 64) | floor_s43 | 20.800 | 1.050 | 2.433 | 6.191 |
| groups with mixed task rewards (of 64) | floor_s44 | 22 | 1.250 | 0.300 | 5.103 |
| groups with mixed task rewards (of 64) | penalty_s42 | 19.400 | 4.600 | 2.267 | 0.162 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | floor_s42 | 6.400 | 3.950 | 4.067 | 3.456 |
|   of which the cue is not at the training label | floor_s43 | 5.900 | 0.350 | 1.200 | 2.985 |
|   of which the cue is not at the training label | floor_s44 | 6.800 | 0.350 | 0.133 | 1.588 |
|   of which the cue is not at the training label | penalty_s42 | 4.800 | 2.250 | 0.533 | 0.044 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | floor_s42 | 96.100 | 27 | 45.767 | 47.162 |
| answers not naming the cue (of 512) | floor_s43 | 89 | 2.500 | 6.667 | 20.662 |
| answers not naming the cue (of 512) | floor_s44 | 94.200 | 16.600 | 17.633 | 160.353 |
| answers not naming the cue (of 512) | penalty_s42 | 91.400 | 18.800 | 11.867 | 1.118 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | floor_s42 | 0.285 | 0.155 | 0.158 | 0.104 |
| mean absolute advantage per answer | floor_s43 | 0.288 | 0.063 | 0.050 | 0.092 |
| mean absolute advantage per answer | floor_s44 | 0.282 | 0.016 | 0.004 | 0.060 |
| mean absolute advantage per answer | penalty_s42 | 0.691 | 0.710 | 0.513 | 0.369 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | floor_s42 | 0.042 | 0.070 | 0.051 | 0.023 |
|   from probe-only groups | floor_s43 | 0.045 | 0.052 | 0.022 | 0.021 |
|   from probe-only groups | floor_s44 | 0.033 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | penalty_s42 | 0.462 | 0.656 | 0.486 | 0.367 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | floor_s42 | 0.048 | -0.005 | 0.005 | -0.000 |
| advantage on cue-naming answers, per answer | floor_s43 | 0.050 | 0.002 | 0.000 | 0.001 |
| advantage on cue-naming answers, per answer | floor_s44 | 0.049 | 0.003 | 0.000 | 0.011 |
| advantage on cue-naming answers, per answer | penalty_s42 | 0.057 | -0.001 | 0.007 | 0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | floor_s42 | 0.083 | 0.052 | 0.056 | 0.055 |
| gradient norm (W&B, before clipping) | floor_s43 | 0.078 | 0.033 | 0.032 | 0.052 |
| gradient norm (W&B, before clipping) | floor_s44 | 0.141 | 0.272 | 0.120 | 0.856 |
| gradient norm (W&B, before clipping) | penalty_s42 | 0.210 | 0.285 | 0.343 | 0.340 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the starting model (W&B) | floor_s42 | 0.050 | 0.156 | 0.343 | 0.323 |
| KL from the starting model (W&B) | floor_s43 | 0.018 | 0.247 | 0.245 | 0.277 |
| KL from the starting model (W&B) | floor_s44 | 0.198 | 2.305 | 2.144 | 1.824 |
| KL from the starting model (W&B) | penalty_s42 | 0.057 | 0.934 | 1.492 | 2.072 |
| KL from the starting model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the starting model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the starting model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | floor_s42 | 0.451 | 0.384 | 0.338 | 0.384 |
| token entropy (W&B) | floor_s43 | 0.472 | 0.419 | 0.419 | 0.437 |
| token entropy (W&B) | floor_s44 | 0.440 | 0.059 | 0.032 | 0.262 |
| token entropy (W&B) | penalty_s42 | 0.416 | 0.143 | 0.037 | 0.104 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | floor_s42 | 236.806 | 236.067 | 244.021 | 190.015 |
| answer length, tokens (W&B) | floor_s43 | 230.546 | 165.234 | 172.134 | 149.176 |
| answer length, tokens (W&B) | floor_s44 | 202.128 | 5.842 | 6.011 | 21.244 |
| answer length, tokens (W&B) | penalty_s42 | 199.603 | 122.858 | 100.888 | 85.611 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of floor_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 141 | 0.079 | 962.543 / 1027.415 | 0.176 / 0.268 | 0.213 / 0.321 | 0.408 / 0.507 |
| 5-9 | 250 | 0.072 | 995.906 / 1051.032 | 0.160 / 0.206 | 0.218 / 0.287 | 0.399 / 0.489 |
| 10-29 | 1117 | 0.095 | 1033.846 / 1055.966 | 0.225 / 0.249 | 0.317 / 0.351 | 0.349 / 0.382 |
| 30-59 | 1585 | 0.059 | 1082.516 / 1093.279 | 0.147 / 0.144 | 0.262 / 0.269 | 0.223 / 0.234 |
| 60-127 | 3666 | 0.013 | 818.525 / 834.197 | 0.159 / 0.167 | 0.266 / 0.281 | 0.203 / 0.219 |

## Probe-only groups of floor_s43: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 146 | 0.079 | 943.889 / 1044.776 | 0.168 / 0.248 | 0.224 / 0.307 | 0.418 / 0.545 |
| 5-9 | 254 | 0.085 | 926.310 / 975.716 | 0.114 / 0.152 | 0.155 / 0.210 | 0.457 / 0.509 |
| 10-29 | 1257 | 0.034 | 707.820 / 733.121 | 0.049 / 0.058 | 0.122 / 0.133 | 0.292 / 0.320 |
| 30-59 | 1844 | 0.013 | 687.860 / 775.390 | 0.055 / 0.082 | 0.102 / 0.147 | 0.243 / 0.312 |
| 60-127 | 3883 | 0.013 | 585.663 / 681.551 | 0.037 / 0.076 | 0.095 / 0.151 | 0.245 / 0.334 |

## Probe-only groups of floor_s44: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 147 | 0.068 | 946.117 / 1006.228 | 0.143 / 0.248 | 0.207 / 0.304 | 0.389 / 0.488 |
| 5-9 | 237 | 0.037 | 641.643 / 698.861 | 0.077 / 0.110 | 0.117 / 0.164 | 0.280 / 0.348 |
| 10-29 | 1220 | 0.000 | 25.990 / 25.889 | 0 / 0 | 0.001 / 0.000 | 0.001 / 0.000 |
| 30-59 | 1848 | 0.000 | 26.066 / 26.092 | 0 / 0 | 0 / 0 | 0 / 0.000 |
| 60-127 | 2805 | 0.000 | 88.108 / 88.864 | 0.006 / 0.007 | 0.025 / 0.027 | 0.033 / 0.033 |

## Probe-only groups of penalty_s42: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 135 | 0.100 | 949.539 / 1026.644 | 0.139 / 0.239 | 0.207 / 0.300 | 0.370 / 0.472 |
| 5-9 | 268 | 0.053 | 633.628 / 668.567 | 0.042 / 0.067 | 0.098 / 0.129 | 0.215 / 0.254 |
| 10-29 | 1172 | 0.023 | 481.941 / 518.088 | 0.015 / 0.022 | 0.105 / 0.114 | 0.085 / 0.098 |
| 30-59 | 1829 | 0.002 | 428.346 / 433.490 | 0.010 / 0.008 | 0.090 / 0.093 | 0.020 / 0.022 |
| 60-127 | 4337 | 0.000 | 350.341 / 352.418 | 0.003 / 0.004 | 0.069 / 0.070 | 0.016 / 0.019 |

## floor_s42's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.113 | 0.016 | 1015.000 |
| 10 | 0 | 0.004 | 957.000 |
| 20 | 0.551 | 0.016 | 1065.000 |
| 25 | 0.514 | 0.021 | 994.000 |
| 30 | 0.525 | 0 | 1024.000 |
| 60 | 0.080 | 0.002 | 769.500 |
| 70 | 0.141 | 0 | 780.500 |
| 80 | 0.174 | 0 | 721.000 |
| 90 | 0.189 | 0 | 763.500 |
| 100 | 0.201 | 0.002 | 788.000 |
| 127 | 0.248 | 0 | 815.000 |

## floor_s43's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.135 | 0.016 | 1044.500 |
| 10 | 0.002 | 0 | 867.000 |
| 20 | 0 | 0.020 | 638.000 |
| 25 | 0 | 0.033 | 622.000 |
| 30 | 0 | 0.039 | 609.500 |
| 60 | 0 | 0.020 | 679.500 |
| 70 | 0 | 0.029 | 642.500 |
| 80 | 0 | 0.010 | 583.000 |
| 90 | 0 | 0.021 | 573.500 |
| 100 | 0 | 0.006 | 519.000 |
| 127 | 0 | 0.012 | 540.500 |

## floor_s44's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.125 | 0.012 | 1056.000 |
| 10 | 0 | 0 | 26.000 |
| 20 | 0 | 0 | 26.000 |
| 25 | 0 | 0 | 26.000 |
| 30 | 0 | 0 | 26.000 |
| 60 | 0 | 0 | 26.000 |
| 70 | 0 | 0 | 26.000 |
| 80 | 0 | 0 | 26.000 |
| 90 | 0 | 0 | 212.500 |
| 100 | 0 | 0.004 | 26.000 |
| 127 | 0 | 0 | 26.000 |

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
  "floor_s42": {
    "median_tokens_no_cue": 171.0,
    "median_tokens_cued": 180.0,
    "mean_top_letter_share_of_8_no_cue": 0.9258691206543967,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 167,
      "B": 322,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 157,
      "B": 315
    },
    "questions_with_no_cue_answer": 472
  },
  "floor_s43": {
    "median_tokens_no_cue": 128.0,
    "median_tokens_cued": 124.0,
    "mean_top_letter_share_of_8_no_cue": 0.8936605316973415,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 198,
      "B": 286,
      "null": 5
    },
    "no_cue_answer_letters": {
      "A": 176,
      "B": 292
    },
    "questions_with_no_cue_answer": 468
  },
  "floor_s44": {
    "median_tokens_no_cue": 6.0,
    "median_tokens_cued": 6.0,
    "mean_top_letter_share_of_8_no_cue": 0.9583333333333334,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 175,
      "B": 311,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 164,
      "B": 312
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

## floor_s42: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 132 / 211 | 0.761 / 0.218 | 0.929 (0.929) | 0.722 | 0.667 | 0.3911 / 0.1889 | 0.920 / 0.878 |
| A | cue_dist | 75 / 42 | 19 / 62 | 0.751 / 0.262 | 0.884 (0.884) | 0.703 | 0.829 | 0.7644 / 0.3584 | 0.879 / 0.813 |
| A | both | 109 / 49 | 151 / 273 | 0.754 / 0.227 | 0.910 | 0.667 | 0.715 | 0.5261 / 0.2436 | 0.928 / 0.846 |
| B | cue_gold | 60 / 16 | 129 / 228 | 0.255 / 0.039 | 0.881 (0.881) | 0.714 | 0.816 | 0.1403 / 0.0279 | 0.887 / 0.762 |
| B | cue_dist | 149 / 86 | 18 / 77 | 0.203 / 0.053 | 0.766 (0.766) | 0.722 | 0.852 | 0.2494 / 0.0684 | 0.761 / 0.615 |
| B | both | 209 / 102 | 147 / 305 | 0.216 / 0.042 | 0.818 | 0.673 | 0.830 | 0.1915 / 0.0441 | 0.850 / 0.697 |
| C | cue_gold | 60 / 16 | 129 / 228 | 0.116 / 0.020 | 0.866 (0.866) | 0.714 | 0.849 | 0.0667 / 0.0137 | 0.853 / 0.771 |
| C | cue_dist | 149 / 86 | 18 / 77 | 0.095 / 0.027 | 0.732 (0.732) | 0.722 | 0.877 | 0.1188 / 0.0305 | 0.667 / 0.578 |
| C | both | 209 / 102 | 147 / 305 | 0.100 / 0.022 | 0.778 | 0.673 | 0.862 | 0.0911 / 0.0205 | 0.784 / 0.659 |
| D | cue_gold | 60 / 16 | 129 / 228 | 0.229 / 0.049 | 0.858 (0.858) | 0.714 | 0.869 | 0.1481 / 0.0283 | 0.852 / 0.736 |
| D | cue_dist | 149 / 86 | 18 / 77 | 0.177 / 0.051 | 0.747 (0.747) | 0.722 | 0.883 | 0.2265 / 0.0531 | 0.713 / 0.581 |
| D | both | 209 / 102 | 147 / 305 | 0.190 / 0.049 | 0.786 | 0.673 | 0.873 | 0.1849 / 0.0383 | 0.792 / 0.670 |

## floor_s43: probe rows by answer letter (447 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 31 / 6 | 134 / 208 | 0.770 / 0.219 | 0.931 (0.931) | 0.723 | 0.664 | 0.3818 / 0.1887 | 0.925 / 0.879 |
| A | cue_dist | 73 / 42 | 19 / 59 | 0.756 / 0.265 | 0.884 (0.884) | 0.696 | 0.835 | 0.7728 / 0.3610 | 0.875 / 0.818 |
| A | both | 104 / 48 | 153 / 267 | 0.759 / 0.227 | 0.911 | 0.660 | 0.713 | 0.5217 / 0.2439 | 0.934 / 0.845 |
| B | cue_gold | 57 / 22 | 142 / 220 | 0.285 / 0.066 | 0.854 (0.854) | 0.665 | 0.749 | 0.1689 / 0.0524 | 0.890 / 0.730 |
| B | cue_dist | 201 / 134 | 22 / 63 | 0.287 / 0.077 | 0.812 (0.812) | 0.671 | 0.851 | 0.3619 / 0.1116 | 0.888 / 0.670 |
| B | both | 258 / 156 | 164 / 283 | 0.287 / 0.068 | 0.839 | 0.628 | 0.792 | 0.2709 / 0.0790 | 0.901 / 0.738 |
| C | cue_gold | 57 / 22 | 142 / 220 | 0.109 / 0.027 | 0.802 (0.802) | 0.665 | 0.730 | 0.0678 / 0.0207 | 0.834 / 0.672 |
| C | cue_dist | 201 / 134 | 22 / 63 | 0.119 / 0.025 | 0.770 (0.770) | 0.671 | 0.809 | 0.1495 / 0.0435 | 0.889 / 0.597 |
| C | both | 258 / 156 | 164 / 283 | 0.117 / 0.027 | 0.788 | 0.628 | 0.769 | 0.1110 / 0.0309 | 0.854 / 0.670 |
| D | cue_gold | 57 / 22 | 142 / 220 | 0.257 / 0.075 | 0.817 (0.817) | 0.665 | 0.804 | 0.1774 / 0.0509 | 0.841 / 0.713 |
| D | cue_dist | 201 / 134 | 22 / 63 | 0.260 / 0.067 | 0.796 (0.796) | 0.671 | 0.866 | 0.3323 / 0.0950 | 0.897 / 0.631 |
| D | both | 258 / 156 | 164 / 283 | 0.260 / 0.074 | 0.800 | 0.628 | 0.829 | 0.2593 / 0.0707 | 0.863 / 0.692 |

## floor_s44: probe rows by answer letter (453 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 136 / 205 | 0.745 / 0.225 | 0.919 (0.919) | 0.715 | 0.673 | 0.3940 / 0.1902 | 0.906 / 0.877 |
| A | cue_dist | 72 / 45 | 22 / 61 | 0.740 / 0.277 | 0.870 (0.870) | 0.675 | 0.826 | 0.7578 / 0.3610 | 0.891 / 0.793 |
| A | both | 106 / 52 | 158 / 266 | 0.741 / 0.235 | 0.900 | 0.649 | 0.715 | 0.5235 / 0.2471 | 0.926 / 0.833 |
| B | cue_gold | 65 / 24 | 119 / 207 | 0.477 / 0.153 | 0.857 (0.857) | 0.683 | 0.698 | 0.3186 / 0.1457 | 0.848 / 0.817 |
| B | cue_dist | 89 / 45 | 35 / 88 | 0.580 / 0.250 | 0.810 (0.810) | 0.690 | 0.834 | 0.6101 / 0.2471 | 0.736 / 0.772 |
| B | both | 154 / 69 | 154 / 295 | 0.539 / 0.180 | 0.853 | 0.674 | 0.744 | 0.4360 / 0.1828 | 0.842 / 0.806 |
| C | cue_gold | 65 / 24 | 119 / 207 | 0.263 / 0.040 | 0.907 (0.907) | 0.683 | 0.661 | 0.1324 / 0.0526 | 0.899 / 0.901 |
| C | cue_dist | 89 / 45 | 35 / 88 | 0.316 / 0.070 | 0.848 (0.848) | 0.690 | 0.767 | 0.3019 / 0.1017 | 0.821 / 0.806 |
| C | both | 154 / 69 | 154 / 295 | 0.295 / 0.048 | 0.891 | 0.674 | 0.697 | 0.2006 / 0.0705 | 0.887 / 0.864 |
| D | cue_gold | 65 / 24 | 119 / 207 | 0.668 / 0.203 | 0.886 (0.886) | 0.683 | 0.705 | 0.4253 / 0.2058 | 0.885 / 0.839 |
| D | cue_dist | 89 / 45 | 35 / 88 | 0.724 / 0.315 | 0.848 (0.848) | 0.690 | 0.832 | 0.7309 / 0.3397 | 0.804 / 0.828 |
| D | both | 154 / 69 | 154 / 295 | 0.702 / 0.234 | 0.883 | 0.674 | 0.746 | 0.5483 / 0.2547 | 0.875 / 0.847 |

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
  "floor_s42": {
    "13": 0.8206096887588501,
    "14": 0.7822847366333008,
    "15": 0.7329695820808411,
    "16": 0.6917104721069336
  },
  "floor_s43": {
    "13": 0.8590912818908691,
    "14": 0.8304033279418945,
    "15": 0.8004705309867859,
    "16": 0.7953557372093201
  },
  "floor_s44": {
    "13": 0.6183192729949951,
    "14": 0.5825049877166748,
    "15": 0.5706475377082825,
    "16": 0.5776844024658203
  },
  "penalty_s42": {
    "13": 0.43122032284736633,
    "14": 0.3993362486362457,
    "15": 0.35029828548431396,
    "16": 0.3179480731487274
  }
}
```
