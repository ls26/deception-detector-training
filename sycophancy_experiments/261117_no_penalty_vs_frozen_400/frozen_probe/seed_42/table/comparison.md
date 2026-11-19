# frozen probe, seed 42: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

## New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | frozen probe (s42) | 0.177 | 0.373 | 0.445 | 0.440 | 0.413 | 0.322 | 0.361 | 0.366 | 0.221 |
| bias, cue at the correct answer | frozen probe (s42) | 0.087 | 0.176 | 0.179 | 0.194 | 0.179 | 0.152 | 0.185 | 0.179 | 0.123 |
| bias, cue at the wrong answer | frozen probe (s42) | 0.266 | 0.571 | 0.711 | 0.686 | 0.647 | 0.492 | 0.537 | 0.554 | 0.318 |
| cue agreement rate, 50/50 mix | frozen probe (s42) | 0.650 | 0.876 | 0.946 | 0.942 | 0.916 | 0.822 | 0.864 | 0.877 | 0.721 |
| cue agreement rate, cue at the correct answer | frozen probe (s42) | 0.843 | 0.967 | 0.984 | 0.990 | 0.978 | 0.957 | 0.973 | 0.982 | 0.910 |
| cue agreement rate, cue at the wrong answer | frozen probe (s42) | 0.458 | 0.785 | 0.908 | 0.894 | 0.855 | 0.687 | 0.755 | 0.773 | 0.532 |
| true accuracy, 50/50 mix | frozen probe (s42) | 0.676 | 0.591 | 0.537 | 0.548 | 0.559 | 0.634 | 0.607 | 0.604 | 0.689 |
| true accuracy, cue at the correct answer | frozen probe (s42) | 0.843 | 0.967 | 0.984 | 0.990 | 0.978 | 0.957 | 0.973 | 0.982 | 0.910 |
| true accuracy, cue at the wrong answer | frozen probe (s42) | 0.509 | 0.215 | 0.090 | 0.106 | 0.141 | 0.311 | 0.241 | 0.227 | 0.468 |
| true accuracy, no cue | frozen probe (s42) | 0.755 | 0.759 | 0.757 | 0.757 | 0.744 | 0.775 | 0.753 | 0.744 | 0.759 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe (s42) | 0.747 | 0.169 | 0.147 | 0.122 | 0.169 | 0.149 | 0.147 | 0.123 | 0.148 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe (s42) | 0.228 | 0.048 | 0.046 | 0.040 | 0.047 | 0.044 | 0.044 | 0.027 | 0.050 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe (s42) | 0.906 | 0.782 | 0.762 | 0.748 | 0.775 | 0.769 | 0.797 | 0.797 | 0.751 |
