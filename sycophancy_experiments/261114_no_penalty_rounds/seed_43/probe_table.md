# Round 1's probe per round, seed 43

Both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled biased / unbiased
against the checkpoint's own no-cue answer, on the questions with a no-cue answer in both the base model's and the
round's eval. Round's answers: the round's final eval, read through the round's model. Base model's answers: the
base eval on the same questions, read through the base model. Built by `rounds.py LINE probe_table`.

| Round | Answers | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | round's answers | 267 / 457 | 0.653 / 0.169 | 0.905 |
| round1 | base model's answers | 160 / 430 | 0.743 / 0.229 | 0.903 |
| round2 | round's answers | 188 / 447 | 0.576 / 0.112 | 0.910 |
| round2 | base model's answers | 157 / 422 | 0.748 / 0.225 | 0.906 |
| round3 | round's answers | 135 / 453 | 0.489 / 0.101 | 0.896 |
| round3 | base model's answers | 160 / 427 | 0.744 / 0.231 | 0.903 |
| round4 | round's answers | 95 / 450 | 0.461 / 0.112 | 0.862 |
| round4 | base model's answers | 159 / 428 | 0.749 / 0.229 | 0.906 |
| round5 | round's answers | 54 / 444 | 0.452 / 0.162 | 0.795 |
| round5 | base model's answers | 162 / 433 | 0.741 / 0.233 | 0.901 |
