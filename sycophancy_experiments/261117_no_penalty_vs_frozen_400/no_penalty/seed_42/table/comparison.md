# no penalty, seed 42: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

## New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | no penalty (s42) | 0.177 | 0.425 | 0.420 | 0.429 | 0.310 | 0.282 | 0.422 | 0.361 | 0.328 |
| bias, cue at the correct answer | no penalty (s42) | 0.089 | 0.193 | 0.187 | 0.194 | 0.163 | 0.160 | 0.200 | 0.201 | 0.192 |
| bias, cue at the wrong answer | no penalty (s42) | 0.265 | 0.658 | 0.653 | 0.664 | 0.457 | 0.404 | 0.645 | 0.521 | 0.463 |
| cue agreement rate, 50/50 mix | no penalty (s42) | 0.650 | 0.925 | 0.924 | 0.921 | 0.817 | 0.778 | 0.924 | 0.872 | 0.831 |
| cue agreement rate, cue at the correct answer | no penalty (s42) | 0.843 | 0.988 | 0.978 | 0.984 | 0.965 | 0.933 | 0.984 | 0.975 | 0.959 |
| cue agreement rate, cue at the wrong answer | no penalty (s42) | 0.458 | 0.863 | 0.871 | 0.859 | 0.669 | 0.624 | 0.865 | 0.769 | 0.703 |
| true accuracy, 50/50 mix | no penalty (s42) | 0.676 | 0.561 | 0.553 | 0.561 | 0.644 | 0.652 | 0.559 | 0.603 | 0.625 |
| true accuracy, cue at the correct answer | no penalty (s42) | 0.843 | 0.988 | 0.978 | 0.984 | 0.965 | 0.933 | 0.984 | 0.975 | 0.959 |
| true accuracy, cue at the wrong answer | no penalty (s42) | 0.509 | 0.135 | 0.129 | 0.139 | 0.323 | 0.372 | 0.135 | 0.231 | 0.290 |
| true accuracy, no cue | no penalty (s42) | 0.755 | 0.753 | 0.738 | 0.744 | 0.763 | 0.751 | 0.732 | 0.748 | 0.757 |
| initial probe, both cues pooled: mean score of biased answers | no penalty (s42) | 0.748 | 0.642 | 0.706 | 0.658 | 0.523 | 0.709 | 0.622 | 0.532 | 0.571 |
| initial probe, both cues pooled: mean score of unbiased answers | no penalty (s42) | 0.227 | 0.187 | 0.223 | 0.177 | 0.101 | 0.231 | 0.132 | 0.110 | 0.099 |
| initial probe, both cues pooled: AUC biased vs unbiased | no penalty (s42) | 0.907 | 0.886 | 0.896 | 0.895 | 0.890 | 0.878 | 0.903 | 0.886 | 0.909 |
