# frozen probe, seed 43: 400 steps

Written by `table.py`; definitions in its docstring. Each cell lists every seed's value in the order given in the condition column. Values are shares (0-1); "–" means no answers to compute it on.

## New run: labels redrawn every prompt, constant learning rate

| Measurement | Condition (seeds) | Base model | Step 50 | Step 100 | Step 150 | Step 200 | Step 250 | Step 300 | Step 350 | Step 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| bias, 50/50 mix | frozen probe (s43) | 0.175 | 0.429 | 0.354 | 0.410 | 0.373 | 0.250 | 0.321 | 0.267 | 0.273 |
| bias, cue at the correct answer | frozen probe (s43) | 0.088 | 0.174 | 0.166 | 0.165 | 0.168 | 0.131 | 0.163 | 0.137 | 0.165 |
| bias, cue at the wrong answer | frozen probe (s43) | 0.263 | 0.684 | 0.542 | 0.655 | 0.577 | 0.369 | 0.479 | 0.397 | 0.382 |
| cue agreement rate, 50/50 mix | frozen probe (s43) | 0.650 | 0.934 | 0.855 | 0.916 | 0.875 | 0.748 | 0.827 | 0.774 | 0.774 |
| cue agreement rate, cue at the correct answer | frozen probe (s43) | 0.843 | 0.984 | 0.971 | 0.980 | 0.971 | 0.920 | 0.953 | 0.937 | 0.947 |
| cue agreement rate, cue at the wrong answer | frozen probe (s43) | 0.458 | 0.883 | 0.738 | 0.853 | 0.779 | 0.577 | 0.701 | 0.611 | 0.601 |
| true accuracy, 50/50 mix | frozen probe (s43) | 0.676 | 0.550 | 0.617 | 0.563 | 0.595 | 0.670 | 0.626 | 0.662 | 0.673 |
| true accuracy, cue at the correct answer | frozen probe (s43) | 0.843 | 0.984 | 0.971 | 0.980 | 0.971 | 0.920 | 0.953 | 0.937 | 0.947 |
| true accuracy, cue at the wrong answer | frozen probe (s43) | 0.509 | 0.117 | 0.262 | 0.147 | 0.219 | 0.419 | 0.299 | 0.387 | 0.399 |
| true accuracy, no cue | frozen probe (s43) | 0.755 | 0.783 | 0.773 | 0.736 | 0.800 | 0.748 | 0.742 | 0.761 | 0.748 |
| initial probe, both cues pooled: mean score of biased answers | frozen probe (s43) | 0.748 | 0.145 | 0.156 | 0.185 | 0.250 | 0.165 | 0.371 | 0.275 | 0.272 |
| initial probe, both cues pooled: mean score of unbiased answers | frozen probe (s43) | 0.230 | 0.041 | 0.041 | 0.093 | 0.090 | 0.064 | 0.120 | 0.064 | 0.082 |
| initial probe, both cues pooled: AUC biased vs unbiased | frozen probe (s43) | 0.905 | 0.766 | 0.800 | 0.709 | 0.766 | 0.737 | 0.813 | 0.831 | 0.805 |
