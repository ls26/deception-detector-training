# Fixed against redrawn training labels, frozen probe, rounds 1-5

Written by `memorization.py`; definitions in its docstring. fixed = 261107, redrawn = this experiment; round 1 is 261103's run in both. Each cell: mean of seeds 42 and 43 (seed 42 / seed 43).

| metric | labels | round 1 | round 2 | round 3 | round 4 | round 5 |
|---|---|---|---|---|---|---|
| bias, training cue mix | fixed | 0.226 (0.202 / 0.250) | 0.147 (0.131 / 0.164) | 0.107 (0.106 / 0.108) | 0.076 (0.072 / 0.081) | 0.053 (0.049 / 0.057) |
| bias, training cue mix | redrawn | 0.226 (0.202 / 0.250) | 0.238 (0.233 / 0.243) | 0.277 (0.289 / 0.265) | 0.229 (0.230 / 0.229) | 0.192 (0.194 / 0.190) |
| cue agreement rate, training cue mix | fixed | 0.827 (0.813 / 0.840) | 0.746 (0.725 / 0.766) | 0.695 (0.691 / 0.700) | 0.659 (0.658 / 0.660) | 0.617 (0.616 / 0.618) |
| cue agreement rate, training cue mix | redrawn | 0.827 (0.813 / 0.840) | 0.844 (0.832 / 0.856) | 0.877 (0.885 / 0.869) | 0.831 (0.835 / 0.827) | 0.787 (0.789 / 0.785) |
| true accuracy, no cue | fixed | 0.763 (0.769 / 0.757) | 0.762 (0.759 / 0.765) | 0.755 (0.736 / 0.773) | 0.755 (0.744 / 0.765) | 0.746 (0.730 / 0.763) |
| true accuracy, no cue | redrawn | 0.763 (0.769 / 0.757) | 0.762 (0.765 / 0.759) | 0.766 (0.767 / 0.765) | 0.754 (0.761 / 0.746) | 0.761 (0.763 / 0.759) |
| true accuracy, cue at the correct answer | fixed | 0.938 (0.939 / 0.937) | 0.897 (0.881 / 0.912) | 0.860 (0.849 / 0.871) | 0.831 (0.830 / 0.832) | 0.786 (0.773 / 0.800) |
| true accuracy, cue at the correct answer | redrawn | 0.938 (0.939 / 0.937) | 0.949 (0.937 / 0.961) | 0.965 (0.971 / 0.959) | 0.942 (0.945 / 0.939) | 0.922 (0.928 / 0.916) |
| true accuracy, cue at the wrong answer | fixed | 0.381 (0.431 / 0.331) | 0.528 (0.556 / 0.499) | 0.627 (0.620 / 0.634) | 0.666 (0.673 / 0.658) | 0.701 (0.689 / 0.714) |
| true accuracy, cue at the wrong answer | redrawn | 0.381 (0.431 / 0.331) | 0.365 (0.380 / 0.350) | 0.274 (0.264 / 0.284) | 0.382 (0.380 / 0.384) | 0.473 (0.481 / 0.466) |
| deception (P(Yes)-weighted), training cue mix | fixed | 0.538 (0.537 / 0.538) | 0.565 (0.591 / 0.540) | 0.561 (0.611 / 0.512) | 0.547 (0.575 / 0.518) | 0.611 (0.678 / 0.544) |
| deception (P(Yes)-weighted), training cue mix | redrawn | 0.538 (0.537 / 0.538) | 0.506 (0.536 / 0.476) | 0.503 (0.473 / 0.532) | 0.467 (0.466 / 0.468) | 0.521 (0.489 / 0.554) |
| round 1's probe, mean score of biased answers | fixed | 0.200 (0.200 / 0.201) | 0.206 (0.242 / 0.170) | 0.184 (0.184 / 0.184) | 0.202 (0.166 / 0.238) | 0.191 (0.169 / 0.212) |
| round 1's probe, mean score of biased answers | redrawn | 0.200 (0.200 / 0.201) | 0.187 (0.216 / 0.158) | 0.147 (0.106 / 0.188) | 0.148 (0.098 / 0.198) | 0.171 (0.161 / 0.181) |
| round 1's probe, mean score of unbiased answers | fixed | 0.047 (0.052 / 0.043) | 0.042 (0.047 / 0.038) | 0.035 (0.031 / 0.039) | 0.050 (0.051 / 0.049) | 0.047 (0.058 / 0.035) |
| round 1's probe, mean score of unbiased answers | redrawn | 0.047 (0.052 / 0.043) | 0.058 (0.082 / 0.035) | 0.043 (0.033 / 0.054) | 0.040 (0.024 / 0.056) | 0.042 (0.039 / 0.044) |
| round 1's probe, AUC biased vs unbiased | fixed | 0.795 (0.790 / 0.801) | 0.831 (0.836 / 0.827) | 0.830 (0.844 / 0.816) | 0.832 (0.820 / 0.844) | 0.815 (0.778 / 0.851) |
| round 1's probe, AUC biased vs unbiased | redrawn | 0.795 (0.790 / 0.801) | 0.782 (0.757 / 0.806) | 0.773 (0.757 / 0.788) | 0.792 (0.774 / 0.811) | 0.802 (0.791 / 0.813) |
| training: flipped label, cue at the true answer, share naming the label | fixed | 0.103 (0.112 / 0.094) | 0.196 (0.233 / 0.160) | 0.445 (0.454 / 0.436) | 0.617 (0.577 / 0.657) | 0.717 (0.707 / 0.726) |
| training: flipped label, cue at the true answer, share naming the label | redrawn | 0.103 (0.112 / 0.094) | 0.051 (0.064 / 0.038) | 0.021 (0.018 / 0.025) | 0.048 (0.046 / 0.050) | 0.068 (0.061 / 0.075) |

Training answers counted in the memorization row (seed 42 / seed 43), rounds 1-5:

- fixed: 3064 / 3224, 3000 / 3312, 3072 / 3016, 3088 / 3160, 3208 / 3008
- redrawn: 3064 / 3224, 3184 / 3136, 3072 / 3256, 3344 / 3256, 3352 / 3160
