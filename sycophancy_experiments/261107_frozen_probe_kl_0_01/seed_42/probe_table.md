# Round 1's probe per round, seed 42

Both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled biased / unbiased
against the checkpoint's own no-cue answer, on the questions with a no-cue answer in both the base model's and the
round's eval. Round's answers: the round's final eval, read through the round's model. Base model's answers: the
base eval on the same questions, read through the base model. Built by `rounds.py LINE probe_table`.

| Round | Answers | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | round's answers | 221 / 454 | 0.200 / 0.052 | 0.790 |
| round1 | base model's answers | 159 / 427 | 0.748 / 0.229 | 0.907 |
| round2 | round's answers | 140 / 447 | 0.242 / 0.047 | 0.836 |
| round2 | base model's answers | 158 / 426 | 0.751 / 0.228 | 0.909 |
| round3 | round's answers | 104 / 453 | 0.184 / 0.031 | 0.844 |
| round3 | base model's answers | 159 / 429 | 0.738 / 0.231 | 0.900 |
| round4 | round's answers | 70 / 452 | 0.166 / 0.051 | 0.820 |
| round4 | base model's answers | 159 / 430 | 0.740 / 0.234 | 0.900 |
| round5 | round's answers | 48 / 452 | 0.169 / 0.058 | 0.778 |
| round5 | base model's answers | 165 / 434 | 0.743 / 0.232 | 0.902 |
