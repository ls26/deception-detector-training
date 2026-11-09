# Training dynamics and probe rows: kl_weight_1_0_s42, s42_round2, s42_round3, s42_round4, s42_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s42 0.3, s42_round2 0.3, s42_round3 0.3, s42_round4 0.3, s42_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s42 | 39.200 | 41.850 | 39.133 | 44.912 |
| groups where all 8 answers name the cue (of 64) | s42_round2 | 44.100 | 41.500 | 44.300 | 43.662 |
| groups where all 8 answers name the cue (of 64) | s42_round3 | 44.800 | 42.350 | 43 | 45.441 |
| groups where all 8 answers name the cue (of 64) | s42_round4 | 46.500 | 43.600 | 43.633 | 45.471 |
| groups where all 8 answers name the cue (of 64) | s42_round5 | 46.900 | 43.700 | 44.133 | 45.118 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s42 | 20.700 | 16.200 | 18.167 | 13.750 |
| groups with mixed task rewards (of 64) | s42_round2 | 12.900 | 12.900 | 10.300 | 9.368 |
| groups with mixed task rewards (of 64) | s42_round3 | 7.900 | 7.450 | 7.833 | 3.515 |
| groups with mixed task rewards (of 64) | s42_round4 | 2.200 | 3.600 | 3.700 | 2.706 |
| groups with mixed task rewards (of 64) | s42_round5 | 1.700 | 2.800 | 2.100 | 1.265 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s42 | 4.800 | 6.500 | 6.867 | 5.985 |
|   of which the cue is not at the training label | s42_round2 | 5.500 | 5 | 4.067 | 3.647 |
|   of which the cue is not at the training label | s42_round3 | 3.100 | 3.100 | 3.200 | 1.574 |
|   of which the cue is not at the training label | s42_round4 | 0.900 | 1.650 | 1.433 | 1.426 |
|   of which the cue is not at the training label | s42_round5 | 0.900 | 1.400 | 0.833 | 0.338 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s42 | 88.600 | 101.250 | 116.733 | 87.779 |
| answers not naming the cue (of 512) | s42_round2 | 99.200 | 122.850 | 112.900 | 119.368 |
| answers not naming the cue (of 512) | s42_round3 | 118.700 | 139.550 | 133.900 | 133.632 |
| answers not naming the cue (of 512) | s42_round4 | 129.200 | 145.900 | 148.100 | 138.265 |
| answers not naming the cue (of 512) | s42_round5 | 129.500 | 154.050 | 150 | 145.956 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s42 | 0.322 | 0.289 | 0.291 | 0.244 |
| mean absolute advantage per answer | s42_round2 | 0.261 | 0.246 | 0.200 | 0.214 |
| mean absolute advantage per answer | s42_round3 | 0.193 | 0.142 | 0.145 | 0.089 |
| mean absolute advantage per answer | s42_round4 | 0.131 | 0.118 | 0.123 | 0.125 |
| mean absolute advantage per answer | s42_round5 | 0.125 | 0.121 | 0.114 | 0.108 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s42 | 0.136 | 0.130 | 0.094 | 0.082 |
|   from probe-only groups | s42_round2 | 0.146 | 0.120 | 0.095 | 0.118 |
|   from probe-only groups | s42_round3 | 0.123 | 0.072 | 0.070 | 0.054 |
|   from probe-only groups | s42_round4 | 0.114 | 0.087 | 0.092 | 0.101 |
|   from probe-only groups | s42_round5 | 0.109 | 0.098 | 0.097 | 0.100 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s42 | 0.024 | -0.004 | 0.010 | 0.005 |
| advantage on cue-naming answers, per answer | s42_round2 | -0.012 | -0.001 | 0.001 | 0.001 |
| advantage on cue-naming answers, per answer | s42_round3 | -0.003 | -0.003 | -0.002 | -0.000 |
| advantage on cue-naming answers, per answer | s42_round4 | -0.003 | -0.004 | -0.000 | -0.004 |
| advantage on cue-naming answers, per answer | s42_round5 | -0.003 | -0.003 | -0.002 | 0.000 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s42 | 0.086 | 0.076 | 0.072 | 0.068 |
| gradient norm (W&B, before clipping) | s42_round2 | 0.063 | 0.060 | 0.059 | 0.072 |
| gradient norm (W&B, before clipping) | s42_round3 | 0.052 | 0.051 | 0.043 | 0.031 |
| gradient norm (W&B, before clipping) | s42_round4 | 0.033 | 0.038 | 0.034 | 0.032 |
| gradient norm (W&B, before clipping) | s42_round5 | 0.029 | 0.035 | 0.032 | 0.026 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s42 | 0.040 | 0.102 | 0.089 | 0.121 |
| KL from the base model (W&B) | s42_round2 | 0.112 | 0.102 | 0.133 | 0.156 |
| KL from the base model (W&B) | s42_round3 | 0.114 | 0.078 | 0.072 | 0.062 |
| KL from the base model (W&B) | s42_round4 | 0.079 | 0.092 | 0.095 | 0.099 |
| KL from the base model (W&B) | s42_round5 | 0.095 | 0.099 | 0.097 | 0.097 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s42 | 0.431 | 0.416 | 0.463 | 0.456 |
| token entropy (W&B) | s42_round2 | 0.466 | 0.468 | 0.451 | 0.491 |
| token entropy (W&B) | s42_round3 | 0.504 | 0.455 | 0.459 | 0.483 |
| token entropy (W&B) | s42_round4 | 0.469 | 0.466 | 0.468 | 0.471 |
| token entropy (W&B) | s42_round5 | 0.448 | 0.481 | 0.448 | 0.459 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s42 | 216.026 | 221.667 | 215.874 | 204.649 |
| answer length, tokens (W&B) | s42_round2 | 196.047 | 202.987 | 184.784 | 185.752 |
| answer length, tokens (W&B) | s42_round3 | 182.392 | 205.329 | 200.720 | 202.557 |
| answer length, tokens (W&B) | s42_round4 | 183.121 | 186.171 | 189.214 | 195.072 |
| answer length, tokens (W&B) | s42_round5 | 179.786 | 189.961 | 182.796 | 177.388 |
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

## Probe-only groups of s42_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 221 | 0.075 | 807.264 / 887 | 0.060 / 0.094 | 0.191 / 0.215 | 0.322 / 0.376 |
| 5-9 | 220 | 0.071 | 820.760 / 886.565 | 0.072 / 0.076 | 0.175 / 0.178 | 0.333 / 0.406 |
| 10-29 | 830 | 0.050 | 845.526 / 917.360 | 0.094 / 0.125 | 0.149 / 0.176 | 0.344 / 0.400 |
| 30-59 | 1329 | 0.033 | 757.354 / 848.630 | 0.071 / 0.123 | 0.131 / 0.172 | 0.363 / 0.403 |
| 60-127 | 2969 | 0.049 | 785.053 / 834.021 | 0.071 / 0.124 | 0.128 / 0.173 | 0.348 / 0.388 |

## Probe-only groups of s42_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 227 | 0.078 | 790.910 / 831.080 | 0.085 / 0.102 | 0.164 / 0.187 | 0.359 / 0.394 |
| 5-9 | 221 | 0.039 | 768.240 / 801.368 | 0.067 / 0.124 | 0.094 / 0.174 | 0.314 / 0.378 |
| 10-29 | 847 | 0.025 | 860.176 / 932.269 | 0.089 / 0.132 | 0.137 / 0.192 | 0.341 / 0.426 |
| 30-59 | 1290 | 0.022 | 838.692 / 899.015 | 0.085 / 0.143 | 0.140 / 0.194 | 0.331 / 0.400 |
| 60-127 | 3090 | 0.014 | 856.187 / 926.635 | 0.096 / 0.144 | 0.150 / 0.210 | 0.367 / 0.439 |

## Probe-only groups of s42_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 231 | 0.063 | 815.643 / 886.404 | 0.054 / 0.124 | 0.131 / 0.221 | 0.357 / 0.436 |
| 5-9 | 234 | 0.046 | 714.897 / 767.373 | 0.025 / 0.042 | 0.060 / 0.106 | 0.229 / 0.310 |
| 10-29 | 872 | 0.041 | 779.504 / 836.258 | 0.015 / 0.035 | 0.061 / 0.091 | 0.231 / 0.292 |
| 30-59 | 1309 | 0.044 | 793.947 / 853.344 | 0.031 / 0.061 | 0.087 / 0.131 | 0.290 / 0.362 |
| 60-127 | 3092 | 0.046 | 815.507 / 884.210 | 0.036 / 0.075 | 0.100 / 0.155 | 0.293 / 0.363 |

## Probe-only groups of s42_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 231 | 0.065 | 768.605 / 835.402 | 0.021 / 0.062 | 0.093 / 0.146 | 0.234 / 0.312 |
| 5-9 | 238 | 0.043 | 739.495 / 790.734 | 0.037 / 0.038 | 0.066 / 0.092 | 0.245 / 0.317 |
| 10-29 | 874 | 0.049 | 774.128 / 849.831 | 0.021 / 0.047 | 0.059 / 0.099 | 0.223 / 0.288 |
| 30-59 | 1324 | 0.048 | 741.069 / 810.906 | 0.020 / 0.044 | 0.070 / 0.109 | 0.222 / 0.289 |
| 60-127 | 3068 | 0.049 | 725.883 / 782.160 | 0.019 / 0.033 | 0.058 / 0.087 | 0.200 / 0.255 |

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

## s42_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.035 | 0.008 | 842.500 |
| 10 | 0.062 | 0.004 | 939.500 |
| 20 | 0.012 | 0.010 | 852.000 |
| 25 | 0.010 | 0.023 | 823.500 |
| 30 | 0.031 | 0.014 | 851.500 |
| 60 | 0.070 | 0.008 | 756.500 |
| 70 | 0.104 | 0.016 | 825.500 |
| 80 | 0.070 | 0.012 | 776.500 |
| 90 | 0.072 | 0.010 | 816.000 |
| 100 | 0.051 | 0.002 | 779.000 |
| 127 | 0.059 | 0.006 | 828.500 |

## s42_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.064 | 0.008 | 800.000 |
| 10 | 0.080 | 0 | 869.000 |
| 20 | 0.111 | 0 | 819.500 |
| 25 | 0.111 | 0.016 | 779.000 |
| 30 | 0.213 | 0.004 | 857.500 |
| 60 | 0.012 | 0.010 | 879.500 |
| 70 | 0.031 | 0.002 | 873.000 |
| 80 | 0.062 | 0.004 | 904.000 |
| 90 | 0.055 | 0.025 | 907.000 |
| 100 | 0.053 | 0.002 | 847.500 |
| 127 | 0.043 | 0.006 | 781.500 |

## s42_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.051 | 0.018 | 868.000 |
| 10 | 0.008 | 0.006 | 667.500 |
| 20 | 0.045 | 0 | 847.000 |
| 25 | 0.066 | 0.002 | 878.500 |
| 30 | 0.049 | 0.002 | 732.000 |
| 60 | 0.029 | 0.014 | 771.500 |
| 70 | 0.062 | 0.010 | 843.000 |
| 80 | 0.055 | 0.008 | 923.500 |
| 90 | 0.027 | 0.018 | 833.500 |
| 100 | 0.041 | 0.004 | 844.000 |
| 127 | 0.029 | 0.008 | 814.000 |

## s42_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.029 | 0.020 | 801.000 |
| 10 | 0.072 | 0.002 | 760.500 |
| 20 | 0.250 | 0.004 | 765.500 |
| 25 | 0.207 | 0.006 | 829.000 |
| 30 | 0.094 | 0.006 | 805.000 |
| 60 | 0.039 | 0.027 | 715.000 |
| 70 | 0.068 | 0.002 | 670.000 |
| 80 | 0.072 | 0.008 | 711.000 |
| 90 | 0.084 | 0.002 | 773.500 |
| 100 | 0.086 | 0.006 | 780.500 |
| 127 | 0.088 | 0.004 | 711.500 |

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
  "s42_round2": {
    "median_tokens_no_cue": 166.0,
    "median_tokens_cued": 173.0,
    "mean_top_letter_share_of_8_no_cue": 0.9381390593047034,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 185,
      "B": 300,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 178,
      "B": 292
    },
    "questions_with_no_cue_answer": 470
  },
  "s42_round3": {
    "median_tokens_no_cue": 174.0,
    "median_tokens_cued": 187.0,
    "mean_top_letter_share_of_8_no_cue": 0.9557770961145194,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 183,
      "B": 304,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 180,
      "B": 303
    },
    "questions_with_no_cue_answer": 483
  },
  "s42_round4": {
    "median_tokens_no_cue": 172.0,
    "median_tokens_cued": 178.5,
    "mean_top_letter_share_of_8_no_cue": 0.9575664621676891,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 186,
      "B": 303,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 182,
      "B": 300
    },
    "questions_with_no_cue_answer": 482
  },
  "s42_round5": {
    "median_tokens_no_cue": 161.0,
    "median_tokens_cued": 162.0,
    "mean_top_letter_share_of_8_no_cue": 0.9606339468302658,
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
      "A": 178,
      "B": 305
    },
    "questions_with_no_cue_answer": 483
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

## s42_round2: probe rows by answer letter (451 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 31 / 7 | 131 / 210 | 0.829 / 0.338 | 0.917 (0.917) | 0.716 | 0.694 | 0.5174 / 0.2906 | 0.909 / 0.852 |
| A | cue_dist | 75 / 41 | 22 / 60 | 0.817 / 0.411 | 0.871 (0.871) | 0.689 | 0.828 | 0.8212 / 0.4827 | 0.828 / 0.827 |
| A | both | 106 / 48 | 153 / 270 | 0.820 / 0.352 | 0.901 | 0.663 | 0.736 | 0.6311 / 0.3516 | 0.901 / 0.861 |
| B | cue_gold | 27 / 16 | 138 / 214 | 0.545 / 0.200 | 0.853 (0.853) | 0.618 | 0.609 | 0.2896 / 0.2010 | 0.883 / 0.798 |
| B | cue_dist | 60 / 29 | 29 / 64 | 0.491 / 0.275 | 0.747 (0.747) | 0.681 | 0.704 | 0.4694 / 0.2960 | 0.679 / 0.743 |
| B | both | 87 / 45 | 167 / 278 | 0.509 / 0.216 | 0.822 | 0.642 | 0.646 | 0.3526 / 0.2283 | 0.822 / 0.790 |
| C | cue_gold | 27 / 16 | 138 / 214 | 0.505 / 0.138 | 0.903 (0.903) | 0.618 | 0.524 | 0.2110 / 0.1547 | 0.921 / 0.858 |
| C | cue_dist | 60 / 29 | 29 / 64 | 0.433 / 0.184 | 0.809 (0.809) | 0.681 | 0.685 | 0.3818 / 0.2329 | 0.749 / 0.839 |
| C | both | 87 / 45 | 167 / 278 | 0.456 / 0.148 | 0.865 | 0.642 | 0.587 | 0.2709 / 0.1772 | 0.862 / 0.853 |
| D | cue_gold | 27 / 16 | 138 / 214 | 0.575 / 0.204 | 0.897 (0.897) | 0.618 | 0.658 | 0.3163 / 0.1924 | 0.913 / 0.864 |
| D | cue_dist | 60 / 29 | 29 / 64 | 0.516 / 0.254 | 0.807 (0.807) | 0.681 | 0.759 | 0.4933 / 0.2760 | 0.701 / 0.828 |
| D | both | 87 / 45 | 167 / 278 | 0.536 / 0.214 | 0.861 | 0.642 | 0.697 | 0.3783 / 0.2165 | 0.841 / 0.849 |

## s42_round3: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 137 / 211 | 0.686 / 0.285 | 0.869 (0.869) | 0.718 | 0.750 | 0.4643 / 0.2198 | 0.862 / 0.710 |
| A | cue_dist | 75 / 45 | 22 / 62 | 0.655 / 0.316 | 0.818 (0.818) | 0.682 | 0.852 | 0.7138 / 0.3349 | 0.825 / 0.712 |
| A | both | 109 / 52 | 159 / 273 | 0.662 / 0.291 | 0.843 | 0.654 | 0.779 | 0.5546 / 0.2577 | 0.868 / 0.750 |
| B | cue_gold | 21 / 6 | 137 / 220 | 0.566 / 0.177 | 0.889 (0.889) | 0.697 | 0.730 | 0.3087 / 0.1318 | 0.917 / 0.756 |
| B | cue_dist | 19 / 16 | 28 / 70 | 0.477 / 0.235 | 0.766 (0.766) | 0.629 | 0.811 | 0.4829 / 0.1985 | 0.650 / 0.760 |
| B | both | 40 / 22 | 165 / 290 | 0.516 / 0.190 | 0.840 | 0.641 | 0.740 | 0.3486 / 0.1501 | 0.846 / 0.774 |
| C | cue_gold | 21 / 6 | 137 / 220 | 0.452 / 0.078 | 0.916 (0.916) | 0.697 | 0.690 | 0.1769 / 0.0536 | 0.935 / 0.795 |
| C | cue_dist | 19 / 16 | 28 / 70 | 0.360 / 0.111 | 0.827 (0.827) | 0.629 | 0.782 | 0.3136 / 0.1019 | 0.729 / 0.830 |
| C | both | 40 / 22 | 165 / 290 | 0.400 / 0.085 | 0.884 | 0.641 | 0.702 | 0.2082 / 0.0669 | 0.887 / 0.839 |
| D | cue_gold | 21 / 6 | 137 / 220 | 0.606 / 0.173 | 0.886 (0.886) | 0.697 | 0.746 | 0.3212 / 0.1218 | 0.915 / 0.717 |
| D | cue_dist | 19 / 16 | 28 / 70 | 0.475 / 0.232 | 0.769 (0.769) | 0.629 | 0.817 | 0.4860 / 0.1918 | 0.684 / 0.770 |
| D | both | 40 / 22 | 165 / 290 | 0.532 / 0.186 | 0.844 | 0.641 | 0.752 | 0.3590 / 0.1411 | 0.852 / 0.785 |

## s42_round4: probe rows by answer letter (461 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 139 / 210 | 0.837 / 0.518 | 0.869 (0.869) | 0.728 | 0.724 | 0.6613 / 0.4626 | 0.854 / 0.694 |
| A | cue_dist | 76 / 46 | 20 / 63 | 0.807 / 0.545 | 0.821 (0.821) | 0.691 | 0.842 | 0.8412 / 0.5774 | 0.805 / 0.739 |
| A | both | 111 / 52 | 159 / 273 | 0.815 / 0.523 | 0.837 | 0.656 | 0.759 | 0.7253 / 0.5011 | 0.850 / 0.753 |
| B | cue_gold | 15 / 5 | 137 / 215 | 0.565 / 0.313 | 0.750 (0.750) | 0.680 | 0.664 | 0.4062 / 0.2715 | 0.816 / 0.455 |
| B | cue_dist | 34 / 19 | 28 / 71 | 0.611 / 0.407 | 0.715 (0.715) | 0.679 | 0.830 | 0.6686 / 0.3470 | 0.600 / 0.646 |
| B | both | 49 / 24 | 165 / 286 | 0.599 / 0.334 | 0.776 | 0.653 | 0.704 | 0.4822 / 0.2934 | 0.800 / 0.650 |
| C | cue_gold | 15 / 5 | 137 / 215 | 0.543 / 0.208 | 0.832 (0.832) | 0.680 | 0.680 | 0.3054 / 0.1712 | 0.877 / 0.653 |
| C | cue_dist | 34 / 19 | 28 / 71 | 0.538 / 0.287 | 0.762 (0.762) | 0.679 | 0.828 | 0.5726 / 0.2378 | 0.684 / 0.703 |
| C | both | 49 / 24 | 165 / 286 | 0.539 / 0.225 | 0.822 | 0.653 | 0.713 | 0.3828 / 0.1905 | 0.844 / 0.731 |
| D | cue_gold | 15 / 5 | 137 / 215 | 0.547 / 0.239 | 0.803 (0.803) | 0.680 | 0.696 | 0.3427 / 0.1960 | 0.860 / 0.564 |
| D | cue_dist | 34 / 19 | 28 / 71 | 0.540 / 0.308 | 0.737 (0.737) | 0.679 | 0.841 | 0.5924 / 0.2482 | 0.684 / 0.634 |
| D | both | 49 / 24 | 165 / 286 | 0.542 / 0.254 | 0.791 | 0.653 | 0.731 | 0.4151 / 0.2112 | 0.835 / 0.655 |

## s42_round5: probe rows by answer letter (461 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 138 / 212 | 0.883 / 0.633 | 0.852 (0.852) | 0.715 | 0.680 | 0.7339 / 0.5998 | 0.861 / 0.651 |
| A | cue_dist | 76 / 46 | 22 / 60 | 0.868 / 0.670 | 0.806 (0.806) | 0.677 | 0.803 | 0.8804 / 0.7030 | 0.795 / 0.723 |
| A | both | 109 / 53 | 160 / 272 | 0.872 / 0.640 | 0.828 | 0.651 | 0.720 | 0.7873 / 0.6334 | 0.849 / 0.732 |
| B | cue_gold | 12 / 3 | 133 / 215 | 0.787 / 0.456 | 0.872 (0.872) | 0.709 | 0.630 | 0.5354 / 0.4262 | 0.864 / 0.774 |
| B | cue_dist | 14 / 5 | 28 / 65 | 0.680 / 0.550 | 0.663 (0.663) | 0.718 | 0.778 | 0.7109 / 0.4886 | 0.602 / 0.406 |
| B | both | 26 / 8 | 161 / 280 | 0.727 / 0.476 | 0.787 | 0.700 | 0.656 | 0.5748 / 0.4414 | 0.793 / 0.597 |
| C | cue_gold | 12 / 3 | 133 / 215 | 0.661 / 0.273 | 0.891 (0.891) | 0.709 | 0.665 | 0.3664 / 0.2383 | 0.891 / 0.792 |
| C | cue_dist | 14 / 5 | 28 / 65 | 0.512 / 0.344 | 0.696 (0.696) | 0.718 | 0.804 | 0.5344 / 0.2750 | 0.559 / 0.548 |
| C | both | 26 / 8 | 161 / 280 | 0.578 / 0.288 | 0.813 | 0.700 | 0.690 | 0.4041 / 0.2472 | 0.798 / 0.675 |
| D | cue_gold | 12 / 3 | 133 / 215 | 0.724 / 0.308 | 0.905 (0.905) | 0.709 | 0.673 | 0.4087 / 0.2693 | 0.912 / 0.798 |
| D | cue_dist | 14 / 5 | 28 / 65 | 0.540 / 0.385 | 0.688 (0.688) | 0.718 | 0.802 | 0.5690 / 0.3172 | 0.584 / 0.511 |
| D | both | 26 / 8 | 161 / 280 | 0.621 / 0.324 | 0.818 | 0.700 | 0.694 | 0.4447 / 0.2809 | 0.813 / 0.661 |

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
  "s42_round2": {
    "train": {
      "biased": {
        "answers": 1512,
        "naming_a": 986
      },
      "unbiased": {
        "answers": 1512,
        "naming_a": 531
      }
    },
    "eval": {
      "biased": {
        "answers": 221,
        "naming_a": 144
      },
      "unbiased": {
        "answers": 454,
        "naming_a": 164
      }
    }
  },
  "s42_round3": {
    "train": {
      "biased": {
        "answers": 2092,
        "naming_a": 1325
      },
      "unbiased": {
        "answers": 2092,
        "naming_a": 748
      }
    },
    "eval": {
      "biased": {
        "answers": 132,
        "naming_a": 87
      },
      "unbiased": {
        "answers": 445,
        "naming_a": 167
      }
    }
  },
  "s42_round4": {
    "train": {
      "biased": {
        "answers": 2266,
        "naming_a": 1431
      },
      "unbiased": {
        "answers": 2266,
        "naming_a": 808
      }
    },
    "eval": {
      "biased": {
        "answers": 62,
        "naming_a": 40
      },
      "unbiased": {
        "answers": 455,
        "naming_a": 165
      }
    }
  },
  "s42_round5": {
    "train": {
      "biased": {
        "answers": 2374,
        "naming_a": 1487
      },
      "unbiased": {
        "answers": 2374,
        "naming_a": 853
      }
    },
    "eval": {
      "biased": {
        "answers": 73,
        "naming_a": 49
      },
      "unbiased": {
        "answers": 451,
        "naming_a": 165
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
  "s42_round2": {
    "13": 0.8227086067199707,
    "14": 0.8012235760688782,
    "15": 0.7706404328346252,
    "16": 0.7452505230903625
  },
  "s42_round3": {
    "13": 0.8835791945457458,
    "14": 0.8508777618408203,
    "15": 0.8183447122573853,
    "16": 0.7968559861183167
  },
  "s42_round4": {
    "13": 0.9273813962936401,
    "14": 0.9196779727935791,
    "15": 0.9148194789886475,
    "16": 0.913396954536438
  },
  "s42_round5": {
    "13": 0.9362862706184387,
    "14": 0.913969099521637,
    "15": 0.9042403697967529,
    "16": 0.902809202671051
  }
}
```
