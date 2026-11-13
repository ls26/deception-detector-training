# Training dynamics and probe rows: kl_weight_1_0_s42, s42_refit_round2, s42_refit_round3, s42_refit_round4, s42_refit_round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: kl_weight_1_0_s42 0.3, s42_refit_round2 0.3, s42_refit_round3 0.3, s42_refit_round4 0.3, s42_refit_round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | kl_weight_1_0_s42 | 39.200 | 41.850 | 39.133 | 44.912 |
| groups where all 8 answers name the cue (of 64) | s42_refit_round2 | 45.100 | 35.600 | 41.367 | 42.074 |
| groups where all 8 answers name the cue (of 64) | s42_refit_round3 | 42.900 | 45.450 | 44.467 | 44.382 |
| groups where all 8 answers name the cue (of 64) | s42_refit_round4 | 43.200 | 41.200 | 41.900 | 41.221 |
| groups where all 8 answers name the cue (of 64) | s42_refit_round5 | 41 | 38.500 | 37.900 | 40.471 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | kl_weight_1_0_s42 | 20.700 | 16.200 | 18.167 | 13.750 |
| groups with mixed task rewards (of 64) | s42_refit_round2 | 11.300 | 14.900 | 11.367 | 11.221 |
| groups with mixed task rewards (of 64) | s42_refit_round3 | 7.700 | 9.500 | 8.900 | 6.853 |
| groups with mixed task rewards (of 64) | s42_refit_round4 | 5.500 | 6.350 | 6.533 | 7.515 |
| groups with mixed task rewards (of 64) | s42_refit_round5 | 6.400 | 8.100 | 7.967 | 5.706 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | kl_weight_1_0_s42 | 4.800 | 6.500 | 6.867 | 5.985 |
|   of which the cue is not at the training label | s42_refit_round2 | 4 | 5.100 | 4.567 | 4.221 |
|   of which the cue is not at the training label | s42_refit_round3 | 2.500 | 4.150 | 3.567 | 2.721 |
|   of which the cue is not at the training label | s42_refit_round4 | 2 | 2.300 | 2.433 | 2.588 |
|   of which the cue is not at the training label | s42_refit_round5 | 1.900 | 3 | 2.600 | 1.779 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | kl_weight_1_0_s42 | 88.600 | 101.250 | 116.733 | 87.779 |
| answers not naming the cue (of 512) | s42_refit_round2 | 102.300 | 160.750 | 135.033 | 126.294 |
| answers not naming the cue (of 512) | s42_refit_round3 | 134 | 108.400 | 114.600 | 125.338 |
| answers not naming the cue (of 512) | s42_refit_round4 | 142.700 | 158.200 | 147.233 | 148.103 |
| answers not naming the cue (of 512) | s42_refit_round5 | 155.100 | 165.250 | 174 | 164.912 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | kl_weight_1_0_s42 | 0.322 | 0.289 | 0.291 | 0.244 |
| mean absolute advantage per answer | s42_refit_round2 | 0.238 | 0.228 | 0.210 | 0.212 |
| mean absolute advantage per answer | s42_refit_round3 | 0.161 | 0.154 | 0.153 | 0.122 |
| mean absolute advantage per answer | s42_refit_round4 | 0.114 | 0.112 | 0.118 | 0.136 |
| mean absolute advantage per answer | s42_refit_round5 | 0.136 | 0.138 | 0.139 | 0.121 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | kl_weight_1_0_s42 | 0.136 | 0.130 | 0.094 | 0.082 |
|   from probe-only groups | s42_refit_round2 | 0.148 | 0.109 | 0.097 | 0.105 |
|   from probe-only groups | s42_refit_round3 | 0.093 | 0.062 | 0.063 | 0.050 |
|   from probe-only groups | s42_refit_round4 | 0.074 | 0.058 | 0.061 | 0.074 |
|   from probe-only groups | s42_refit_round5 | 0.093 | 0.073 | 0.079 | 0.080 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | kl_weight_1_0_s42 | 0.024 | -0.004 | 0.010 | 0.005 |
| advantage on cue-naming answers, per answer | s42_refit_round2 | -0.007 | -0.006 | -0.002 | 0.001 |
| advantage on cue-naming answers, per answer | s42_refit_round3 | 0.003 | -0.003 | 0.001 | 0.003 |
| advantage on cue-naming answers, per answer | s42_refit_round4 | -0.004 | -0.001 | -0.000 | -0.001 |
| advantage on cue-naming answers, per answer | s42_refit_round5 | -0.001 | -0.002 | -0.001 | 0.000 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | kl_weight_1_0_s42 | 0.086 | 0.076 | 0.072 | 0.068 |
| gradient norm (W&B, before clipping) | s42_refit_round2 | 0.061 | 0.053 | 0.054 | 0.053 |
| gradient norm (W&B, before clipping) | s42_refit_round3 | 0.041 | 0.044 | 0.043 | 0.039 |
| gradient norm (W&B, before clipping) | s42_refit_round4 | 0.032 | 0.035 | 0.039 | 0.042 |
| gradient norm (W&B, before clipping) | s42_refit_round5 | 0.032 | 0.035 | 0.034 | 0.028 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | kl_weight_1_0_s42 | 0.040 | 0.102 | 0.089 | 0.121 |
| KL from the base model (W&B) | s42_refit_round2 | 0.116 | 0.073 | 0.107 | 0.120 |
| KL from the base model (W&B) | s42_refit_round3 | 0.095 | 0.090 | 0.091 | 0.092 |
| KL from the base model (W&B) | s42_refit_round4 | 0.073 | 0.068 | 0.073 | 0.068 |
| KL from the base model (W&B) | s42_refit_round5 | 0.071 | 0.080 | 0.084 | 0.076 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | kl_weight_1_0_s42 | 0.431 | 0.416 | 0.463 | 0.456 |
| token entropy (W&B) | s42_refit_round2 | 0.455 | 0.477 | 0.450 | 0.469 |
| token entropy (W&B) | s42_refit_round3 | 0.468 | 0.476 | 0.483 | 0.496 |
| token entropy (W&B) | s42_refit_round4 | 0.500 | 0.474 | 0.486 | 0.475 |
| token entropy (W&B) | s42_refit_round5 | 0.450 | 0.462 | 0.526 | 0.479 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | kl_weight_1_0_s42 | 216.026 | 221.667 | 215.874 | 204.649 |
| answer length, tokens (W&B) | s42_refit_round2 | 180.393 | 228.959 | 199.548 | 213.713 |
| answer length, tokens (W&B) | s42_refit_round3 | 210.877 | 212.551 | 214.765 | 207.171 |
| answer length, tokens (W&B) | s42_refit_round4 | 207.556 | 208.379 | 206.593 | 219.742 |
| answer length, tokens (W&B) | s42_refit_round5 | 213.315 | 222.791 | 218.081 | 225.440 |
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

## Probe-only groups of s42_refit_round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 220 | 0.081 | 823.415 / 893.653 | 0.078 / 0.142 | 0.174 / 0.217 | 0.327 / 0.367 |
| 5-9 | 231 | 0.065 | 704.471 / 735.948 | 0.042 / 0.066 | 0.083 / 0.108 | 0.268 / 0.311 |
| 10-29 | 712 | 0.064 | 956.686 / 1003.748 | 0.126 / 0.176 | 0.157 / 0.215 | 0.381 / 0.431 |
| 30-59 | 1241 | 0.037 | 836.347 / 897.879 | 0.062 / 0.105 | 0.142 / 0.166 | 0.347 / 0.387 |
| 60-127 | 2861 | 0.044 | 903.126 / 982.979 | 0.127 / 0.176 | 0.194 / 0.224 | 0.399 / 0.444 |

## Probe-only groups of s42_refit_round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 212 | 0.050 | 925.425 / 988.401 | 0.100 / 0.160 | 0.175 / 0.217 | 0.394 / 0.456 |
| 5-9 | 217 | 0.030 | 884.482 / 922.720 | 0.070 / 0.147 | 0.153 / 0.224 | 0.331 / 0.407 |
| 10-29 | 909 | 0.018 | 910.502 / 968.387 | 0.039 / 0.085 | 0.100 / 0.175 | 0.317 / 0.404 |
| 30-59 | 1334 | 0.016 | 924.640 / 973.740 | 0.031 / 0.087 | 0.085 / 0.154 | 0.317 / 0.380 |
| 60-127 | 3018 | 0.011 | 880.493 / 927.610 | 0.025 / 0.069 | 0.076 / 0.135 | 0.313 / 0.372 |

## Probe-only groups of s42_refit_round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 205 | 0.036 | 897.033 / 927.129 | 0.034 / 0.076 | 0.074 / 0.139 | 0.321 / 0.373 |
| 5-9 | 227 | 0.023 | 872.558 / 916.599 | 0.032 / 0.061 | 0.074 / 0.115 | 0.331 / 0.381 |
| 10-29 | 824 | 0.018 | 883.217 / 937.293 | 0.044 / 0.106 | 0.106 / 0.193 | 0.308 / 0.372 |
| 30-59 | 1257 | 0.021 | 878.062 / 935.545 | 0.033 / 0.067 | 0.099 / 0.160 | 0.294 / 0.336 |
| 60-127 | 2803 | 0.031 | 910.461 / 967.346 | 0.041 / 0.087 | 0.111 / 0.183 | 0.331 / 0.391 |

## Probe-only groups of s42_refit_round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 204 | 0.055 | 932.651 / 965.373 | 0.034 / 0.083 | 0.096 / 0.150 | 0.314 / 0.350 |
| 5-9 | 206 | 0.041 | 852.296 / 872.502 | 0.017 / 0.051 | 0.074 / 0.144 | 0.303 / 0.353 |
| 10-29 | 770 | 0.038 | 956.305 / 972.865 | 0.017 / 0.028 | 0.059 / 0.099 | 0.305 / 0.340 |
| 30-59 | 1137 | 0.045 | 934.779 / 950.959 | 0.019 / 0.045 | 0.058 / 0.102 | 0.307 / 0.345 |
| 60-127 | 2752 | 0.040 | 970.631 / 986.872 | 0.019 / 0.044 | 0.067 / 0.116 | 0.313 / 0.354 |

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

## s42_refit_round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.021 | 0.006 | 819.000 |
| 10 | 0.053 | 0 | 651.000 |
| 20 | 0.045 | 0.010 | 1095.500 |
| 25 | 0.021 | 0.021 | 1121.500 |
| 30 | 0.010 | 0.008 | 1016.000 |
| 60 | 0.055 | 0.006 | 774.000 |
| 70 | 0.033 | 0.008 | 866.500 |
| 80 | 0.027 | 0.037 | 927.500 |
| 90 | 0.025 | 0.002 | 880.000 |
| 100 | 0.064 | 0.010 | 882.500 |
| 127 | 0.041 | 0.010 | 908.500 |

## s42_refit_round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.051 | 0 | 954.500 |
| 10 | 0.570 | 0.002 | 827.000 |
| 20 | 0.494 | 0.002 | 972.500 |
| 25 | 0.225 | 0.006 | 867.000 |
| 30 | 0.143 | 0.002 | 835.500 |
| 60 | 0.031 | 0.020 | 873.500 |
| 70 | 0.043 | 0.008 | 829.500 |
| 80 | 0.020 | 0.004 | 881.000 |
| 90 | 0.031 | 0 | 858.000 |
| 100 | 0.025 | 0.004 | 890.000 |
| 127 | 0.035 | 0 | 869.500 |

## s42_refit_round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.010 | 0 | 813.000 |
| 10 | 0.051 | 0.010 | 813.000 |
| 20 | 0.320 | 0.002 | 923.500 |
| 25 | 0.438 | 0.018 | 957.500 |
| 30 | 0.168 | 0.002 | 934.000 |
| 60 | 0.033 | 0 | 887.000 |
| 70 | 0.041 | 0 | 876.500 |
| 80 | 0.037 | 0.002 | 899.500 |
| 90 | 0.043 | 0.018 | 904.000 |
| 100 | 0.041 | 0 | 913.500 |
| 127 | 0.059 | 0 | 852.500 |

## s42_refit_round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.053 | 0.002 | 920.000 |
| 10 | 0.031 | 0.012 | 835.000 |
| 20 | 0.008 | 0.004 | 900.000 |
| 25 | 0.033 | 0.010 | 954.500 |
| 30 | 0.062 | 0.023 | 996.500 |
| 60 | 0.072 | 0.008 | 1013.000 |
| 70 | 0.086 | 0.002 | 944.000 |
| 80 | 0.168 | 0.004 | 1023.000 |
| 90 | 0.111 | 0.002 | 941.000 |
| 100 | 0.105 | 0.006 | 896.000 |
| 127 | 0.074 | 0.006 | 916.000 |

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
  "s42_refit_round2": {
    "median_tokens_no_cue": 187.0,
    "median_tokens_cued": 193.0,
    "mean_top_letter_share_of_8_no_cue": 0.943762781186094,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 179,
      "B": 308,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 178,
      "B": 302
    },
    "questions_with_no_cue_answer": 480
  },
  "s42_refit_round3": {
    "median_tokens_no_cue": 187.0,
    "median_tokens_cued": 187.0,
    "mean_top_letter_share_of_8_no_cue": 0.9435071574642127,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 194,
      "B": 294,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 178,
      "B": 304
    },
    "questions_with_no_cue_answer": 482
  },
  "s42_refit_round4": {
    "median_tokens_no_cue": 188.0,
    "median_tokens_cued": 190.5,
    "mean_top_letter_share_of_8_no_cue": 0.9468302658486708,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 178,
      "B": 310,
      "null": 1
    },
    "no_cue_answer_letters": {
      "A": 168,
      "B": 307
    },
    "questions_with_no_cue_answer": 475
  },
  "s42_refit_round5": {
    "median_tokens_no_cue": 194.0,
    "median_tokens_cued": 204.0,
    "mean_top_letter_share_of_8_no_cue": 0.9460633946830266,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 174,
      "B": 313,
      "null": 2
    },
    "no_cue_answer_letters": {
      "A": 167,
      "B": 312
    },
    "questions_with_no_cue_answer": 479
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

## s42_refit_round2: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 137 / 210 | 0.818 / 0.347 | 0.905 (0.905) | 0.729 | 0.709 | 0.5313 / 0.2892 | 0.883 / 0.826 |
| A | cue_dist | 76 / 45 | 21 / 63 | 0.803 / 0.407 | 0.864 (0.864) | 0.689 | 0.842 | 0.8277 / 0.4725 | 0.818 / 0.817 |
| A | both | 111 / 51 | 158 / 273 | 0.807 / 0.358 | 0.889 | 0.659 | 0.745 | 0.6382 / 0.3503 | 0.888 / 0.844 |
| B | cue_gold | 42 / 17 | 136 / 219 | 0.583 / 0.220 | 0.893 (0.893) | 0.664 | 0.712 | 0.3687 / 0.1979 | 0.902 / 0.840 |
| B | cue_dist | 53 / 44 | 28 / 71 | 0.481 / 0.267 | 0.758 (0.758) | 0.632 | 0.790 | 0.5183 / 0.2703 | 0.755 / 0.701 |
| B | both | 95 / 61 | 164 / 290 | 0.520 / 0.230 | 0.832 | 0.624 | 0.727 | 0.4155 / 0.2216 | 0.856 / 0.772 |
| C | cue_gold | 42 / 17 | 136 / 219 | 0.570 / 0.156 | 0.920 (0.920) | 0.664 | 0.720 | 0.3149 / 0.1401 | 0.928 / 0.870 |
| C | cue_dist | 53 / 44 | 28 / 71 | 0.446 / 0.202 | 0.782 (0.782) | 0.632 | 0.838 | 0.4930 / 0.2026 | 0.794 / 0.714 |
| C | both | 95 / 61 | 164 / 290 | 0.493 / 0.166 | 0.857 | 0.624 | 0.747 | 0.3706 / 0.1606 | 0.896 / 0.787 |
| D | cue_gold | 42 / 17 | 136 / 219 | 0.613 / 0.213 | 0.914 (0.914) | 0.664 | 0.766 | 0.3929 / 0.1769 | 0.928 / 0.879 |
| D | cue_dist | 53 / 44 | 28 / 71 | 0.494 / 0.248 | 0.789 (0.789) | 0.632 | 0.869 | 0.5552 / 0.2390 | 0.806 / 0.717 |
| D | both | 95 / 61 | 164 / 290 | 0.539 / 0.221 | 0.851 | 0.624 | 0.789 | 0.4437 / 0.1972 | 0.895 / 0.790 |

## s42_refit_round3: probe rows by answer letter (460 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 136 / 211 | 0.586 / 0.206 | 0.881 (0.881) | 0.719 | 0.638 | 0.3246 / 0.1855 | 0.866 / 0.816 |
| A | cue_dist | 76 / 43 | 22 / 62 | 0.633 / 0.259 | 0.851 (0.851) | 0.688 | 0.761 | 0.6246 / 0.3414 | 0.802 / 0.827 |
| A | both | 110 / 50 | 158 / 273 | 0.621 / 0.217 | 0.882 | 0.660 | 0.679 | 0.4343 / 0.2361 | 0.874 / 0.850 |
| B | cue_gold | 37 / 14 | 138 / 223 | 0.296 / 0.094 | 0.855 (0.855) | 0.672 | 0.629 | 0.1540 / 0.0928 | 0.846 / 0.834 |
| B | cue_dist | 65 / 37 | 24 / 68 | 0.348 / 0.135 | 0.781 (0.781) | 0.688 | 0.762 | 0.3504 / 0.1594 | 0.770 / 0.715 |
| B | both | 102 / 51 | 162 / 291 | 0.331 / 0.102 | 0.833 | 0.655 | 0.671 | 0.2202 / 0.1132 | 0.850 / 0.775 |
| C | cue_gold | 37 / 14 | 138 / 223 | 0.239 / 0.038 | 0.916 (0.916) | 0.672 | 0.584 | 0.0946 / 0.0395 | 0.922 / 0.873 |
| C | cue_dist | 65 / 37 | 24 / 68 | 0.274 / 0.061 | 0.823 (0.823) | 0.688 | 0.770 | 0.2699 / 0.0908 | 0.826 / 0.755 |
| C | both | 102 / 51 | 162 / 291 | 0.262 / 0.043 | 0.884 | 0.655 | 0.645 | 0.1537 / 0.0552 | 0.905 / 0.824 |
| D | cue_gold | 37 / 14 | 138 / 223 | 0.467 / 0.117 | 0.902 (0.902) | 0.672 | 0.697 | 0.2438 / 0.0991 | 0.896 / 0.860 |
| D | cue_dist | 65 / 37 | 24 / 68 | 0.467 / 0.152 | 0.825 (0.825) | 0.688 | 0.847 | 0.4935 / 0.1686 | 0.787 / 0.776 |
| D | both | 102 / 51 | 162 / 291 | 0.467 / 0.124 | 0.872 | 0.655 | 0.741 | 0.3280 / 0.1204 | 0.876 / 0.820 |

## s42_refit_round4: probe rows by answer letter (454 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 6 | 135 / 212 | 0.795 / 0.430 | 0.866 (0.866) | 0.729 | 0.723 | 0.5893 / 0.3728 | 0.859 / 0.686 |
| A | cue_dist | 75 / 44 | 19 / 60 | 0.800 / 0.447 | 0.853 (0.853) | 0.695 | 0.835 | 0.8255 / 0.5089 | 0.780 / 0.817 |
| A | both | 108 / 50 | 154 / 272 | 0.799 / 0.433 | 0.866 | 0.661 | 0.756 | 0.6741 / 0.4167 | 0.868 / 0.806 |
| B | cue_gold | 26 / 8 | 134 / 225 | 0.698 / 0.307 | 0.887 (0.887) | 0.696 | 0.732 | 0.4702 / 0.2514 | 0.852 / 0.854 |
| B | cue_dist | 37 / 23 | 16 / 65 | 0.620 / 0.302 | 0.823 (0.823) | 0.710 | 0.881 | 0.6721 / 0.2956 | 0.865 / 0.700 |
| B | both | 63 / 31 | 150 / 290 | 0.648 / 0.306 | 0.841 | 0.665 | 0.763 | 0.5205 / 0.2635 | 0.857 / 0.752 |
| C | cue_gold | 26 / 8 | 134 / 225 | 0.637 / 0.169 | 0.921 (0.921) | 0.696 | 0.760 | 0.3507 / 0.1124 | 0.914 / 0.880 |
| C | cue_dist | 37 / 23 | 16 / 65 | 0.518 / 0.150 | 0.861 (0.861) | 0.710 | 0.924 | 0.5798 / 0.1421 | 0.924 / 0.755 |
| C | both | 63 / 31 | 150 / 290 | 0.561 / 0.166 | 0.875 | 0.665 | 0.793 | 0.4077 / 0.1206 | 0.906 / 0.796 |
| D | cue_gold | 26 / 8 | 134 / 225 | 0.616 / 0.193 | 0.904 (0.904) | 0.696 | 0.735 | 0.3505 / 0.1469 | 0.899 / 0.841 |
| D | cue_dist | 37 / 23 | 16 / 65 | 0.532 / 0.187 | 0.841 (0.841) | 0.710 | 0.895 | 0.5811 / 0.1850 | 0.904 / 0.709 |
| D | both | 63 / 31 | 150 / 290 | 0.562 / 0.192 | 0.861 | 0.665 | 0.767 | 0.4079 / 0.1574 | 0.897 / 0.760 |

## s42_refit_round5: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 6 | 133 / 212 | 0.850 / 0.550 | 0.860 (0.860) | 0.734 | 0.655 | 0.6556 / 0.5258 | 0.863 / 0.689 |
| A | cue_dist | 76 / 46 | 20 / 62 | 0.858 / 0.602 | 0.832 (0.832) | 0.690 | 0.789 | 0.8623 / 0.6596 | 0.772 / 0.788 |
| A | both | 111 / 52 | 153 / 274 | 0.856 / 0.560 | 0.859 | 0.661 | 0.697 | 0.7308 / 0.5701 | 0.868 / 0.795 |
| B | cue_gold | 24 / 10 | 131 / 221 | 0.712 / 0.376 | 0.858 (0.858) | 0.667 | 0.630 | 0.4744 / 0.3590 | 0.830 / 0.853 |
| B | cue_dist | 27 / 18 | 22 / 70 | 0.674 / 0.452 | 0.761 (0.761) | 0.680 | 0.773 | 0.6735 / 0.4416 | 0.726 / 0.671 |
| B | both | 51 / 28 | 153 / 291 | 0.690 / 0.391 | 0.825 | 0.650 | 0.656 | 0.5222 / 0.3818 | 0.826 / 0.770 |
| C | cue_gold | 24 / 10 | 131 / 221 | 0.595 / 0.228 | 0.881 (0.881) | 0.667 | 0.657 | 0.3360 / 0.2090 | 0.849 / 0.884 |
| C | cue_dist | 27 / 18 | 22 / 70 | 0.509 / 0.276 | 0.774 (0.774) | 0.680 | 0.803 | 0.5148 / 0.2621 | 0.796 / 0.664 |
| C | both | 51 / 28 | 153 / 291 | 0.546 / 0.238 | 0.837 | 0.650 | 0.682 | 0.3790 / 0.2236 | 0.844 / 0.777 |
| D | cue_gold | 24 / 10 | 131 / 221 | 0.606 / 0.236 | 0.875 (0.875) | 0.667 | 0.724 | 0.3770 / 0.1955 | 0.837 / 0.878 |
| D | cue_dist | 27 / 18 | 22 / 70 | 0.496 / 0.276 | 0.751 (0.751) | 0.680 | 0.837 | 0.5367 / 0.2437 | 0.756 / 0.631 |
| D | both | 51 / 28 | 153 / 291 | 0.543 / 0.244 | 0.821 | 0.650 | 0.743 | 0.4154 / 0.2088 | 0.822 / 0.761 |

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
  "s42_refit_round2": {
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
  "s42_refit_round3": {
    "train": {
      "biased": {
        "answers": 2104,
        "naming_a": 1375
      },
      "unbiased": {
        "answers": 2104,
        "naming_a": 710
      }
    },
    "eval": {
      "biased": {
        "answers": 156,
        "naming_a": 95
      },
      "unbiased": {
        "answers": 454,
        "naming_a": 164
      }
    }
  },
  "s42_refit_round4": {
    "train": {
      "biased": {
        "answers": 2668,
        "naming_a": 1787
      },
      "unbiased": {
        "answers": 2668,
        "naming_a": 872
      }
    },
    "eval": {
      "biased": {
        "answers": 153,
        "naming_a": 102
      },
      "unbiased": {
        "answers": 453,
        "naming_a": 162
      }
    }
  },
  "s42_refit_round5": {
    "train": {
      "biased": {
        "answers": 2982,
        "naming_a": 2006
      },
      "unbiased": {
        "answers": 2982,
        "naming_a": 954
      }
    },
    "eval": {
      "biased": {
        "answers": 94,
        "naming_a": 63
      },
      "unbiased": {
        "answers": 440,
        "naming_a": 150
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
  "s42_refit_round2": {
    "13": 0.9017161726951599,
    "14": 0.8807210326194763,
    "15": 0.8691755533218384,
    "16": 0.8372749090194702
  },
  "s42_refit_round3": {
    "13": 0.9114760756492615,
    "14": 0.8842827677726746,
    "15": 0.8654360175132751,
    "16": 0.8453565239906311
  },
  "s42_refit_round4": {
    "13": 0.9057942628860474,
    "14": 0.8982357978820801,
    "15": 0.8883262276649475,
    "16": 0.8907198309898376
  },
  "s42_refit_round5": {
    "13": 0.9116750359535217,
    "14": 0.9055212736129761,
    "15": 0.8956592082977295,
    "16": 0.8870508670806885
  }
}
```
