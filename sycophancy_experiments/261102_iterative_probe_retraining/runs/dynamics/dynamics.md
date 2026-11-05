# Training dynamics and probe rows: round1, round2, round3, round4, round5

Written by `dynamics.py`; definitions in its docstring.

Advantage SD floor per run: round1 0.3, round2 0.3, round3 0.3, round4 0.3, round5 0.3

## Training, means per step over each window

| | run | steps 0-9 | steps 10-29 | steps 30-59 | steps 60-127 |
|---|---|---|---|---|---|
| groups where all 8 answers name the cue (of 64) | round1 | 40 | 63.900 | 63.833 | 64 |
| groups where all 8 answers name the cue (of 64) | round2 | 63 | 63.800 | 62.533 | 61.441 |
| groups where all 8 answers name the cue (of 64) | round3 | 56.400 | 56.700 | 57.633 | 61.265 |
| groups where all 8 answers name the cue (of 64) | round4 | 61.500 | 40.850 | 41.500 | 47.559 |
| groups where all 8 answers name the cue (of 64) | round5 | 46.300 | 46.450 | 54.200 | 52.779 |
| groups where all 8 answers name the cue (of 64) | baseline_s42 | 41.700 | 61.200 | 53.567 | 52.632 |
| groups where all 8 answers name the cue (of 64) | baseline_s43 | 36.400 | 54.450 | 55.033 | 47.544 |
| groups where all 8 answers name the cue (of 64) | baseline_s44 | 38.600 | 59.900 | 56.067 | 56.162 |
| groups with mixed task rewards (of 64) | round1 | 20.800 | 0.100 | 0.100 | 0 |
| groups with mixed task rewards (of 64) | round2 | 0.800 | 0.100 | 0.600 | 0.397 |
| groups with mixed task rewards (of 64) | round3 | 2.900 | 1.200 | 0.867 | 0.500 |
| groups with mixed task rewards (of 64) | round4 | 1.300 | 8.200 | 3.500 | 2.221 |
| groups with mixed task rewards (of 64) | round5 | 1.700 | 1.300 | 0.767 | 0.515 |
| groups with mixed task rewards (of 64) | baseline_s42 | 19.900 | 2.750 | 9.900 | 10.368 |
| groups with mixed task rewards (of 64) | baseline_s43 | 23.600 | 8.700 | 6.400 | 12.603 |
| groups with mixed task rewards (of 64) | baseline_s44 | 22.200 | 3.550 | 5.733 | 5.221 |
|   of which the cue is not at the training label | round1 | 5.800 | 0.050 | 0 | 0 |
|   of which the cue is not at the training label | round2 | 0 | 0 | 0.233 | 0.176 |
|   of which the cue is not at the training label | round3 | 1 | 0.300 | 0.300 | 0.147 |
|   of which the cue is not at the training label | round4 | 0.800 | 3.050 | 1.333 | 1.088 |
|   of which the cue is not at the training label | round5 | 0.800 | 0.650 | 0.333 | 0.309 |
|   of which the cue is not at the training label | baseline_s42 | 5.200 | 1.450 | 4.400 | 4.882 |
|   of which the cue is not at the training label | baseline_s43 | 8.200 | 4.400 | 3 | 5.500 |
|   of which the cue is not at the training label | baseline_s44 | 6.600 | 1.700 | 2.667 | 2.794 |
| answers not naming the cue (of 512) | round1 | 88.800 | 0.250 | 0.600 | 0 |
| answers not naming the cue (of 512) | round2 | 3.800 | 0.200 | 9.500 | 18.279 |
| answers not naming the cue (of 512) | round3 | 49 | 53.600 | 46.733 | 19.515 |
| answers not naming the cue (of 512) | round4 | 13.800 | 151.850 | 165.733 | 122.721 |
| answers not naming the cue (of 512) | round5 | 136.400 | 136.150 | 75.733 | 87.529 |
| answers not naming the cue (of 512) | baseline_s42 | 78.100 | 4.800 | 28.567 | 33.750 |
| answers not naming the cue (of 512) | baseline_s43 | 102.600 | 30.400 | 34.733 | 70.265 |
| answers not naming the cue (of 512) | baseline_s44 | 93.100 | 12.900 | 35.800 | 36.191 |
| mean absolute advantage per answer | round1 | 0.279 | 0.003 | 0.006 | 0.003 |
| mean absolute advantage per answer | round2 | 0.012 | 0.001 | 0.007 | 0.005 |
| mean absolute advantage per answer | round3 | 0.028 | 0.012 | 0.008 | 0.007 |
| mean absolute advantage per answer | round4 | 0.019 | 0.095 | 0.039 | 0.026 |
| mean absolute advantage per answer | round5 | 0.024 | 0.017 | 0.011 | 0.007 |
| mean absolute advantage per answer | baseline_s42 | 0.236 | 0.030 | 0.115 | 0.120 |
| mean absolute advantage per answer | baseline_s43 | 0.272 | 0.100 | 0.073 | 0.150 |
| mean absolute advantage per answer | baseline_s44 | 0.259 | 0.042 | 0.067 | 0.060 |
|   from probe-only groups | round1 | 0.058 | 0.002 | 0.005 | 0.003 |
|   from probe-only groups | round2 | 0.002 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | round3 | 0.000 | 0.000 | 0.000 | 0.002 |
|   from probe-only groups | round4 | 0.005 | 0.009 | 0.003 | 0.003 |
|   from probe-only groups | round5 | 0.004 | 0.002 | 0.002 | 0.002 |
|   from probe-only groups | baseline_s42 | 0.001 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s43 | 0.000 | 0.000 | 0.000 | 0.000 |
|   from probe-only groups | baseline_s44 | 0.000 | 0.000 | 0.000 | 0.000 |
| advantage on cue-naming answers, per answer | round1 | 0.043 | 0.000 | 0.001 | -0.000 |
| advantage on cue-naming answers, per answer | round2 | 0.005 | 0.000 | 0.000 | -0.000 |
| advantage on cue-naming answers, per answer | round3 | 0.003 | 0.002 | 0.001 | 0.001 |
| advantage on cue-naming answers, per answer | round4 | -0.003 | 0.009 | 0.003 | -0.002 |
| advantage on cue-naming answers, per answer | round5 | -0.000 | -0.001 | 0.001 | -0.001 |
| advantage on cue-naming answers, per answer | baseline_s42 | 0.058 | -0.000 | 0.004 | 0.002 |
| advantage on cue-naming answers, per answer | baseline_s43 | 0.044 | -0.001 | 0.002 | 0.009 |
| advantage on cue-naming answers, per answer | baseline_s44 | 0.056 | 0.000 | 0.003 | -0.003 |
| gradient norm (W&B, before clipping) | round1 | 0.265 | 0.048 | 0.170 | 0.047 |
| gradient norm (W&B, before clipping) | round2 | 0.508 | 0.013 | 0.265 | 0.376 |
| gradient norm (W&B, before clipping) | round3 | 0.570 | 0.528 | 0.714 | 0.490 |
| gradient norm (W&B, before clipping) | round4 | 0.263 | 0.753 | 1.314 | 0.673 |
| gradient norm (W&B, before clipping) | round5 | 0.851 | 0.588 | 0.639 | 0.559 |
| gradient norm (W&B, before clipping) | baseline_s42 | 0.077 | 0.029 | 0.051 | 0.045 |
| gradient norm (W&B, before clipping) | baseline_s43 | 0.084 | 0.055 | 0.048 | 0.064 |
| gradient norm (W&B, before clipping) | baseline_s44 | 0.080 | 0.027 | 0.042 | 0.043 |
| KL from the base model (W&B) | round1 | 0.947 | 4.113 | 3.556 | 2.923 |
| KL from the base model (W&B) | round2 | 3.792 | 3.996 | 3.902 | 4.231 |
| KL from the base model (W&B) | round3 | 4.049 | 3.750 | 4.064 | 4.103 |
| KL from the base model (W&B) | round4 | 3.023 | 1.568 | 1.696 | 1.675 |
| KL from the base model (W&B) | round5 | 1.788 | 1.666 | 1.441 | 1.526 |
| KL from the base model (W&B) | baseline_s42 | 0.023 | 0.099 | 0.130 | 0.238 |
| KL from the base model (W&B) | baseline_s43 | 0.021 | 0.110 | 0.131 | 0.122 |
| KL from the base model (W&B) | baseline_s44 | 0.015 | 0.094 | 0.088 | 0.126 |
| token entropy (W&B) | round1 | 0.376 | 0.016 | 0.044 | 0.011 |
| token entropy (W&B) | round2 | 0.005 | 0.042 | 0.049 | 0.022 |
| token entropy (W&B) | round3 | 0.035 | 0.114 | 0.038 | 0.037 |
| token entropy (W&B) | round4 | 0.235 | 0.299 | 0.174 | 0.166 |
| token entropy (W&B) | round5 | 0.195 | 0.288 | 0.379 | 0.373 |
| token entropy (W&B) | baseline_s42 | 0.471 | 0.450 | 0.484 | 0.474 |
| token entropy (W&B) | baseline_s43 | 0.444 | 0.420 | 0.411 | 0.408 |
| token entropy (W&B) | baseline_s44 | 0.435 | 0.345 | 0.412 | 0.460 |
| answer length, tokens (W&B) | round1 | 163.908 | 5.044 | 5.814 | 6.629 |
| answer length, tokens (W&B) | round2 | 5.435 | 5.170 | 5.407 | 5.146 |
| answer length, tokens (W&B) | round3 | 5.063 | 5.672 | 5.090 | 5.053 |
| answer length, tokens (W&B) | round4 | 12.308 | 23.632 | 10.925 | 10.408 |
| answer length, tokens (W&B) | round5 | 10.090 | 20.846 | 34.135 | 27.281 |
| answer length, tokens (W&B) | baseline_s42 | 229.727 | 228.385 | 252.109 | 284.276 |
| answer length, tokens (W&B) | baseline_s43 | 241.155 | 227.671 | 214.266 | 226.480 |
| answer length, tokens (W&B) | baseline_s44 | 231.529 | 273.728 | 237.432 | 213.652 |

## Probe-only groups of round1: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 132 | 0.093 | 997.536 / 1071.797 | 0.172 / 0.280 | 0.265 / 0.358 | 0.473 / 0.578 |
| 5-9 | 268 | 0.008 | 274.366 / 306.437 | 0.040 / 0.057 | 0.048 / 0.063 | 0.109 / 0.142 |
| 10-29 | 1278 | 0.000 | 25.037 / 25.051 | 0 / 0 | 0 / 0 | 0 / 0 |
| 30-59 | 1915 | 0.000 | 25.910 / 25.860 | 0 / 0 | 0 / 0 | 0.005 / 0.006 |
| 60-127 | 4352 | 0.000 | 26.631 / 26.627 | 0 / 0 | 0 / 0 | 0 / 0 |

## Probe-only groups of round2: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 320 | 0.000 | 25.883 / 25.884 | 0 / 0 | 0 / 0 | 0 / 0 |
| 5-9 | 310 | 0.000 | 25.002 / 25.002 | 0 / 0 | 0 / 0 | 0 / 0 |
| 10-29 | 1276 | 0.000 | 25.175 / 25.166 | 0 / 0 | 0 / 0 | 0 / 0 |
| 30-59 | 1876 | 0.000 | 25.402 / 25.404 | 0 / 0 | 0 / 0 | 0 / 0 |
| 60-127 | 4178 | 0.000 | 25.151 / 25.150 | 0 / 0 | 0 / 0 | 0 / 0 |

## Probe-only groups of round3: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 310 | 0.000 | 25.006 / 25.014 | 0 / 0 | 0 / 0 | 0 / 0 |
| 5-9 | 254 | 0.000 | 25.062 / 25.086 | 0 / 0 | 0 / 0 | 0 / 0 |
| 10-29 | 1134 | 0.000 | 26.134 / 25.809 | 0 / 0 | 0 / 0.001 | 0.000 / 0.000 |
| 30-59 | 1729 | 0.000 | 25.115 / 25.172 | 0 / 0 | 0 / 0.000 | 0 / 0.000 |
| 60-127 | 4166 | 0.000 | 25.046 / 25.076 | 0 / 0 | 0 / 0.000 | 0 / 0 |

## Probe-only groups of round4: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 318 | 0.000 | 26.986 / 27.362 | 0 / 0 | 0 / 0 | 0.001 / 0.001 |
| 5-9 | 297 | 0.000 | 76.379 / 77.752 | 0.006 / 0.005 | 0.018 / 0.017 | 0.022 / 0.022 |
| 10-29 | 817 | 0.000 | 102.084 / 103.927 | 0.013 / 0.012 | 0.019 / 0.016 | 0.032 / 0.032 |
| 30-59 | 1245 | 0.000 | 47.095 / 46.700 | 0.001 / 0.001 | 0.008 / 0.005 | 0.009 / 0.007 |
| 60-127 | 3234 | 0.000 | 45.043 / 45.098 | 0.002 / 0.002 | 0.008 / 0.008 | 0.007 / 0.008 |

## Probe-only groups of round5: the 4 lowest-scoring answers against the 4 highest

| steps | groups | median SD of probe scores | chars | mentions the other letter | quotes the other option | hedge words |
|---|---|---|---|---|---|---|
| 0-4 | 243 | 0.000 | 45.870 / 49.458 | 0.002 / 0.003 | 0.009 / 0.011 | 0.011 / 0.014 |
| 5-9 | 220 | 0.000 | 38.589 / 38.436 | 0 / 0 | 0.007 / 0.002 | 0.002 / 0.005 |
| 10-29 | 929 | 0.000 | 86.696 / 86.123 | 0.003 / 0.002 | 0.021 / 0.022 | 0.023 / 0.022 |
| 30-59 | 1626 | 0.000 | 155.630 / 159.401 | 0.012 / 0.009 | 0.037 / 0.037 | 0.036 / 0.042 |
| 60-127 | 3589 | 0.000 | 116.753 / 119.559 | 0.009 / 0.010 | 0.022 / 0.023 | 0.029 / 0.031 |

## round1's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0.129 | 0.027 | 993.000 |
| 10 | 0 | 0 | 25.000 |
| 20 | 0 | 0 | 25.000 |
| 25 | 0 | 0 | 25.000 |
| 30 | 0 | 0 | 25.000 |
| 60 | 0 | 0 | 27.000 |
| 70 | 0 | 0 | 27.000 |
| 80 | 0 | 0 | 27.000 |
| 90 | 0 | 0 | 27.000 |
| 100 | 0 | 0 | 27.000 |
| 127 | 0 | 0 | 27.000 |

## round2's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0 | 27.000 |
| 10 | 0 | 0 | 25.000 |
| 20 | 0 | 0 | 25.000 |
| 25 | 0 | 0 | 25.000 |
| 30 | 0 | 0 | 25.000 |
| 60 | 0 | 0 | 26.000 |
| 70 | 0 | 0 | 25.000 |
| 80 | 0 | 0 | 25.000 |
| 90 | 0 | 0 | 25.000 |
| 100 | 0 | 0 | 25.000 |
| 127 | 0 | 0 | 25.000 |

## round3's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0 | 25.000 |
| 10 | 0 | 0 | 25.000 |
| 20 | 0 | 0 | 26.000 |
| 25 | 0 | 0 | 26.000 |
| 30 | 0 | 0 | 25.000 |
| 60 | 0 | 0 | 25.000 |
| 70 | 0 | 0 | 25.000 |
| 80 | 0 | 0 | 25.000 |
| 90 | 0 | 0 | 25.000 |
| 100 | 0 | 0 | 25.000 |
| 127 | 0 | 0 | 25.000 |

## round4's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0 | 25.000 |
| 10 | 0 | 0 | 28.000 |
| 20 | 0 | 0 | 28.000 |
| 25 | 0 | 0 | 28.000 |
| 30 | 0 | 0 | 28.000 |
| 60 | 0 | 0 | 28.000 |
| 70 | 0 | 0 | 28.000 |
| 80 | 0 | 0 | 28.000 |
| 90 | 0 | 0 | 28.000 |
| 100 | 0 | 0 | 28.000 |
| 127 | 0 | 0 | 28.000 |

## round5's training answers

| step | opens with "Let's break down" | 3 or more "!" | median chars |
|---|---|---|---|
| 0 | 0 | 0 | 28.000 |
| 10 | 0 | 0 | 26.000 |
| 20 | 0 | 0 | 28.000 |
| 25 | 0 | 0 | 31.500 |
| 30 | 0 | 0 | 48.000 |
| 60 | 0 | 0 | 28.000 |
| 70 | 0 | 0 | 26.000 |
| 80 | 0 | 0 | 28.000 |
| 90 | 0 | 0 | 28.000 |
| 100 | 0 | 0.008 | 28.000 |
| 127 | 0 | 0 | 28.000 |

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
  "round1": {
    "median_tokens_no_cue": 7.0,
    "median_tokens_cued": 7.0,
    "mean_top_letter_share_of_8_no_cue": 0.8346114519427403,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 232,
      "B": 257,
      "null": 0
    },
    "no_cue_answer_letters": {
      "A": 211,
      "B": 241
    },
    "questions_with_no_cue_answer": 452
  },
  "round2": {
    "median_tokens_no_cue": 5.0,
    "median_tokens_cued": 5.0,
    "mean_top_letter_share_of_8_no_cue": 0.956799591002045,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 121,
      "B": 364,
      "null": 4
    },
    "no_cue_answer_letters": {
      "A": 121,
      "B": 356
    },
    "questions_with_no_cue_answer": 477
  },
  "round3": {
    "median_tokens_no_cue": 5.0,
    "median_tokens_cued": 5.0,
    "mean_top_letter_share_of_8_no_cue": 0.971881390593047,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 310,
      "B": 167,
      "null": 12
    },
    "no_cue_answer_letters": {
      "A": 313,
      "B": 166
    },
    "questions_with_no_cue_answer": 479
  },
  "round4": {
    "median_tokens_no_cue": 10.0,
    "median_tokens_cued": 7.0,
    "mean_top_letter_share_of_8_no_cue": 0.9695807770961146,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 193,
      "B": 290,
      "null": 6
    },
    "no_cue_answer_letters": {
      "A": 196,
      "B": 283
    },
    "questions_with_no_cue_answer": 479
  },
  "round5": {
    "median_tokens_no_cue": 9.0,
    "median_tokens_cued": 7.0,
    "mean_top_letter_share_of_8_no_cue": 0.989519427402863,
    "template_share": {
      "none": 0,
      "cue_gold": 0,
      "cue_dist": 0
    },
    "no_cue_sample_0_letters": {
      "A": 172,
      "B": 314,
      "null": 3
    },
    "no_cue_answer_letters": {
      "A": 174,
      "B": 313
    },
    "questions_with_no_cue_answer": 487
  }
}
```

## round1: probe rows by answer letter (432 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 33 / 7 | 129 / 196 | 0.743 / 0.225 | 0.919 (0.919) | 0.714 | 0.666 | 0.3905 / 0.1945 | 0.907 / 0.872 |
| A | cue_dist | 72 / 43 | 21 / 60 | 0.735 / 0.263 | 0.876 (0.876) | 0.683 | 0.821 | 0.7465 / 0.3537 | 0.877 / 0.810 |
| A | both | 105 / 50 | 150 / 256 | 0.737 / 0.232 | 0.901 | 0.654 | 0.712 | 0.5204 / 0.2481 | 0.920 / 0.837 |
| B | cue_gold | 80 / 80 | 120 / 152 | 0.457 / 0.230 | 0.701 (0.701) | 0.529 | 0.683 | 0.4328 / 0.2120 | 0.751 / 0.645 |
| B | cue_dist | 152 / 120 | 80 / 80 | 0.727 / 0.535 | 0.664 (0.664) | 0.529 | 0.728 | 0.7856 / 0.5060 | 0.639 / 0.701 |
| B | both | 232 / 200 | 200 / 232 | 0.627 / 0.343 | 0.725 | 0.537 | 0.710 | 0.6223 / 0.3481 | 0.726 / 0.715 |
| C | cue_gold | 80 / 80 | 120 / 152 | 0.110 / 0.093 | 0.589 (0.589) | 0.529 | 0.892 | 0.1635 / 0.0437 | 0.511 / 0.649 |
| C | cue_dist | 152 / 120 | 80 / 80 | 0.120 / 0.132 | 0.497 (0.497) | 0.529 | 0.905 | 0.1883 / 0.0501 | 0.382 / 0.548 |
| C | both | 232 / 200 | 200 / 232 | 0.116 / 0.107 | 0.558 | 0.537 | 0.900 | 0.1768 / 0.0467 | 0.462 / 0.610 |
| D | cue_gold | 80 / 80 | 120 / 152 | 0.171 / 0.157 | 0.557 (0.557) | 0.529 | 0.801 | 0.2229 / 0.1105 | 0.518 / 0.570 |
| D | cue_dist | 152 / 120 | 80 / 80 | 0.208 / 0.199 | 0.514 (0.514) | 0.529 | 0.762 | 0.2580 / 0.1430 | 0.489 / 0.520 |
| D | both | 232 / 200 | 200 / 232 | 0.195 / 0.173 | 0.558 | 0.537 | 0.783 | 0.2417 / 0.1255 | 0.519 / 0.571 |

## round2: probe rows by answer letter (455 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 133 / 211 | 0.707 / 0.367 | 0.834 (0.834) | 0.721 | 0.859 | 0.5944 / 0.2568 | 0.780 / 0.762 |
| A | cue_dist | 76 / 43 | 22 / 63 | 0.649 / 0.392 | 0.760 (0.760) | 0.690 | 0.874 | 0.7348 / 0.3637 | 0.690 / 0.651 |
| A | both | 110 / 50 | 155 / 274 | 0.664 / 0.372 | 0.788 | 0.663 | 0.859 | 0.6463 / 0.2917 | 0.768 / 0.718 |
| B | cue_gold | 110 / 20 | 96 / 228 | 0.719 / 0.496 | 0.799 (0.799) | 0.775 | 0.679 | 0.6335 / 0.4987 | 0.811 / 0.702 |
| B | cue_dist | 199 / 96 | 20 / 111 | 0.764 / 0.583 | 0.769 (0.769) | 0.761 | 0.618 | 0.7484 / 0.6655 | 0.674 / 0.803 |
| B | both | 309 / 116 | 116 / 339 | 0.750 / 0.521 | 0.811 | 0.736 | 0.657 | 0.6927 / 0.5746 | 0.801 / 0.816 |
| C | cue_gold | 110 / 20 | 96 / 228 | 0.502 / 0.238 | 0.823 (0.823) | 0.775 | 0.755 | 0.4215 / 0.2244 | 0.803 / 0.701 |
| C | cue_dist | 199 / 96 | 20 / 111 | 0.536 / 0.283 | 0.807 (0.807) | 0.761 | 0.746 | 0.5571 / 0.3534 | 0.660 / 0.792 |
| C | both | 309 / 116 | 116 / 339 | 0.526 / 0.251 | 0.830 | 0.736 | 0.751 | 0.4914 / 0.2831 | 0.791 / 0.795 |
| D | cue_gold | 110 / 20 | 96 / 228 | 0.561 / 0.386 | 0.700 (0.700) | 0.775 | 0.438 | 0.4110 / 0.4564 | 0.845 / 0.656 |
| D | cue_dist | 199 / 96 | 20 / 111 | 0.662 / 0.491 | 0.701 (0.701) | 0.761 | 0.450 | 0.5899 / 0.6293 | 0.763 / 0.804 |
| D | both | 309 / 116 | 116 / 339 | 0.631 / 0.416 | 0.736 | 0.736 | 0.466 | 0.5032 / 0.5351 | 0.846 / 0.803 |

## round3: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 35 / 7 | 138 / 205 | 0.780 / 0.434 | 0.871 (0.871) | 0.716 | 0.646 | 0.5451 / 0.4118 | 0.895 / 0.689 |
| A | cue_dist | 74 / 46 | 22 / 62 | 0.766 / 0.504 | 0.794 (0.794) | 0.677 | 0.779 | 0.7809 / 0.5489 | 0.800 / 0.693 |
| A | both | 109 / 53 | 160 / 267 | 0.770 / 0.448 | 0.847 | 0.649 | 0.685 | 0.6293 / 0.4581 | 0.889 / 0.740 |
| B | cue_gold | 31 / 110 | 182 / 132 | 0.403 / 0.319 | 0.604 (0.604) | 0.320 | 0.769 | 0.4710 / 0.2342 | 0.773 / 0.758 |
| B | cue_dist | 132 / 181 | 111 / 31 | 0.599 / 0.542 | 0.559 (0.559) | 0.320 | 0.806 | 0.7138 / 0.4291 | 0.683 / 0.877 |
| B | both | 163 / 291 | 293 / 163 | 0.538 / 0.388 | 0.656 | 0.358 | 0.780 | 0.6004 / 0.3252 | 0.778 / 0.832 |
| C | cue_gold | 31 / 110 | 182 / 132 | 0.330 / 0.237 | 0.633 (0.633) | 0.320 | 0.747 | 0.3639 / 0.1794 | 0.766 / 0.795 |
| C | cue_dist | 132 / 181 | 111 / 31 | 0.447 / 0.335 | 0.625 (0.625) | 0.320 | 0.671 | 0.4795 / 0.3342 | 0.675 / 0.850 |
| C | both | 163 / 291 | 293 / 163 | 0.411 / 0.267 | 0.669 | 0.358 | 0.715 | 0.4255 / 0.2517 | 0.727 / 0.826 |
| D | cue_gold | 31 / 110 | 182 / 132 | 0.500 / 0.423 | 0.587 (0.587) | 0.320 | 0.864 | 0.6368 / 0.2798 | 0.848 / 0.826 |
| D | cue_dist | 132 / 181 | 111 / 31 | 0.663 / 0.600 | 0.577 (0.577) | 0.320 | 0.814 | 0.7836 / 0.4822 | 0.730 / 0.879 |
| D | both | 163 / 291 | 293 / 163 | 0.612 / 0.478 | 0.636 | 0.358 | 0.834 | 0.7150 / 0.3743 | 0.793 / 0.857 |

## round4: probe rows by answer letter (456 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 34 / 7 | 134 / 210 | 0.614 / 0.345 | 0.823 (0.823) | 0.720 | 0.623 | 0.4304 / 0.3299 | 0.838 / 0.625 |
| A | cue_dist | 74 / 44 | 22 / 61 | 0.621 / 0.402 | 0.764 (0.764) | 0.681 | 0.773 | 0.6492 / 0.4224 | 0.835 / 0.613 |
| A | both | 108 / 51 | 156 / 271 | 0.619 / 0.356 | 0.811 | 0.657 | 0.670 | 0.5099 / 0.3601 | 0.860 / 0.684 |
| B | cue_gold | 28 / 40 | 130 / 197 | 0.492 / 0.331 | 0.747 (0.747) | 0.507 | 0.728 | 0.4445 / 0.3019 | 0.766 / 0.758 |
| B | cue_dist | 29 / 67 | 47 / 76 | 0.571 / 0.441 | 0.687 (0.687) | 0.460 | 0.772 | 0.6162 / 0.4345 | 0.621 / 0.782 |
| B | both | 57 / 107 | 177 / 273 | 0.538 / 0.361 | 0.756 | 0.477 | 0.717 | 0.5002 / 0.3518 | 0.743 / 0.800 |
| C | cue_gold | 28 / 40 | 130 / 197 | 0.626 / 0.300 | 0.833 (0.833) | 0.507 | 0.704 | 0.4722 / 0.2791 | 0.788 / 0.897 |
| C | cue_dist | 29 / 67 | 47 / 76 | 0.669 / 0.420 | 0.784 (0.784) | 0.460 | 0.751 | 0.6747 / 0.4515 | 0.700 / 0.897 |
| C | both | 57 / 107 | 177 / 273 | 0.651 / 0.333 | 0.834 | 0.477 | 0.698 | 0.5379 / 0.3440 | 0.785 / 0.910 |
| D | cue_gold | 28 / 40 | 130 / 197 | 0.441 / 0.291 | 0.677 (0.677) | 0.507 | 0.660 | 0.3822 / 0.2737 | 0.782 / 0.626 |
| D | cue_dist | 29 / 67 | 47 / 76 | 0.484 / 0.350 | 0.669 (0.669) | 0.460 | 0.675 | 0.4971 / 0.3616 | 0.575 / 0.738 |
| D | both | 57 / 107 | 177 / 273 | 0.466 / 0.307 | 0.698 | 0.477 | 0.656 | 0.4195 / 0.3068 | 0.728 / 0.703 |

## round5: probe rows by answer letter (464 questions with a no-cue answer in both evals)

| row | cue | biased naming A / B | unbiased naming A / B | mean score, biased / unbiased | AUC (comparison.json) | AUC of the letter alone | AUC, A vs B answers | mean score, A / B answers | AUC within A / within B |
|---|---|---|---|---|---|---|---|---|---|
| A | cue_gold | 36 / 7 | 138 / 211 | 0.664 / 0.424 | 0.824 (0.824) | 0.721 | 0.732 | 0.5362 / 0.3824 | 0.843 / 0.515 |
| A | cue_dist | 76 / 45 | 22 / 64 | 0.638 / 0.446 | 0.763 (0.763) | 0.686 | 0.829 | 0.6828 / 0.4464 | 0.831 / 0.607 |
| A | both | 112 / 52 | 160 / 275 | 0.645 / 0.429 | 0.791 | 0.658 | 0.763 | 0.5890 / 0.4037 | 0.845 / 0.630 |
| B | cue_gold | 68 / 35 | 119 / 201 | 0.603 / 0.506 | 0.693 (0.693) | 0.644 | 0.595 | 0.5563 / 0.5079 | 0.736 / 0.616 |
| B | cue_dist | 81 / 56 | 44 / 95 | 0.629 / 0.572 | 0.599 (0.599) | 0.637 | 0.580 | 0.6226 / 0.5820 | 0.556 / 0.609 |
| B | both | 149 / 91 | 163 / 296 | 0.618 / 0.526 | 0.673 | 0.633 | 0.588 | 0.5829 / 0.5368 | 0.687 / 0.642 |
| C | cue_gold | 68 / 35 | 119 / 201 | 0.498 / 0.271 | 0.799 (0.799) | 0.644 | 0.594 | 0.3633 / 0.2966 | 0.807 / 0.777 |
| C | cue_dist | 81 / 56 | 44 / 95 | 0.483 / 0.349 | 0.708 (0.708) | 0.637 | 0.559 | 0.4359 / 0.3988 | 0.698 / 0.720 |
| C | both | 149 / 91 | 163 / 296 | 0.489 / 0.295 | 0.777 | 0.633 | 0.583 | 0.3924 / 0.3365 | 0.777 / 0.773 |
| D | cue_gold | 68 / 35 | 119 / 201 | 0.669 / 0.408 | 0.817 (0.817) | 0.644 | 0.641 | 0.5380 / 0.4191 | 0.842 / 0.750 |
| D | cue_dist | 81 / 56 | 44 / 95 | 0.658 / 0.493 | 0.724 (0.724) | 0.637 | 0.637 | 0.6277 / 0.5309 | 0.657 / 0.753 |
| D | both | 149 / 91 | 163 / 296 | 0.663 / 0.434 | 0.790 | 0.633 | 0.641 | 0.5740 / 0.4627 | 0.784 / 0.773 |

## Probe data: answers naming A

```
{
  "round1": {
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
  "round2": {
    "train": {
      "biased": {
        "answers": 1512,
        "naming_a": 928
      },
      "unbiased": {
        "answers": 1512,
        "naming_a": 601
      }
    },
    "eval": {
      "biased": {
        "answers": 432,
        "naming_a": 232
      },
      "unbiased": {
        "answers": 432,
        "naming_a": 200
      }
    }
  },
  "round3": {
    "train": {
      "biased": {
        "answers": 2268,
        "naming_a": 1524
      },
      "unbiased": {
        "answers": 2268,
        "naming_a": 766
      }
    },
    "eval": {
      "biased": {
        "answers": 425,
        "naming_a": 309
      },
      "unbiased": {
        "answers": 455,
        "naming_a": 116
      }
    }
  },
  "round4": {
    "train": {
      "biased": {
        "answers": 3024,
        "naming_a": 1787
      },
      "unbiased": {
        "answers": 3024,
        "naming_a": 1296
      }
    },
    "eval": {
      "biased": {
        "answers": 454,
        "naming_a": 163
      },
      "unbiased": {
        "answers": 456,
        "naming_a": 293
      }
    }
  },
  "round5": {
    "train": {
      "biased": {
        "answers": 3648,
        "naming_a": 2044
      },
      "unbiased": {
        "answers": 3648,
        "naming_a": 1536
      }
    },
    "eval": {
      "biased": {
        "answers": 164,
        "naming_a": 57
      },
      "unbiased": {
        "answers": 450,
        "naming_a": 177
      }
    }
  }
}
```

## Cosine of the initial and fresh probe weights

```
{
  "round1": {
    "13": 0.5438740849494934,
    "14": 0.4881299138069153,
    "15": 0.41527456045150757,
    "16": 0.3998945653438568
  },
  "round2": {
    "13": 0.4348925054073334,
    "14": 0.40327608585357666,
    "15": 0.3517371118068695,
    "16": 0.34192052483558655
  },
  "round3": {
    "13": 0.5335371494293213,
    "14": 0.5387215614318848,
    "15": 0.5078586339950562,
    "16": 0.5280196070671082
  },
  "round4": {
    "13": 0.4695541560649872,
    "14": 0.39645981788635254,
    "15": 0.36225607991218567,
    "16": 0.3405243754386902
  },
  "round5": {
    "13": 0.7196642756462097,
    "14": 0.7077804207801819,
    "15": 0.6830805540084839,
    "16": 0.6784794330596924
  }
}
```
