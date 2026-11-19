# no penalty, seed 43: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

## New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | no penalty (s43) | 0.178 | 0.377 | 0.435 | 0.484 | 0.392 | 0.317 | 0.377 | 0.419 | 0.339 |
| bias, cue at the correct answer | no penalty (s43) | 0.090 | 0.185 | 0.200 | 0.212 | 0.194 | 0.174 | 0.200 | 0.191 | 0.181 |
| bias, cue at the wrong answer | no penalty (s43) | 0.265 | 0.570 | 0.670 | 0.756 | 0.590 | 0.459 | 0.555 | 0.647 | 0.498 |
| cue agreement rate, 50/50 mix | no penalty (s43) | 0.650 | 0.883 | 0.938 | 0.985 | 0.898 | 0.820 | 0.885 | 0.921 | 0.847 |
| cue agreement rate, cue at the correct answer | no penalty (s43) | 0.843 | 0.980 | 0.980 | 0.998 | 0.982 | 0.959 | 0.990 | 0.986 | 0.969 |
| cue agreement rate, cue at the wrong answer | no penalty (s43) | 0.458 | 0.787 | 0.896 | 0.971 | 0.814 | 0.681 | 0.781 | 0.857 | 0.724 |
| true accuracy, 50/50 mix | no penalty (s43) | 0.676 | 0.596 | 0.542 | 0.512 | 0.583 | 0.637 | 0.603 | 0.564 | 0.622 |
| true accuracy, cue at the correct answer | no penalty (s43) | 0.843 | 0.980 | 0.980 | 0.998 | 0.982 | 0.959 | 0.990 | 0.986 | 0.969 |
| true accuracy, cue at the wrong answer | no penalty (s43) | 0.509 | 0.213 | 0.104 | 0.027 | 0.184 | 0.315 | 0.217 | 0.143 | 0.274 |
| true accuracy, no cue | no penalty (s43) | 0.755 | 0.748 | 0.720 | 0.734 | 0.740 | 0.740 | 0.748 | 0.753 | 0.759 |
| initial probe, both cues pooled: mean score of biased answers | no penalty (s43) | 0.746 | 0.681 | 0.714 | 0.732 | 0.687 | 0.669 | 0.636 | 0.677 | 0.612 |
| initial probe, both cues pooled: mean score of unbiased answers | no penalty (s43) | 0.229 | 0.190 | 0.201 | 0.284 | 0.216 | 0.169 | 0.160 | 0.188 | 0.122 |
| initial probe, both cues pooled: AUC biased vs unbiased | no penalty (s43) | 0.906 | 0.895 | 0.909 | 0.868 | 0.892 | 0.908 | 0.889 | 0.892 | 0.908 |
