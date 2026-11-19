# Round 1's probe per round, seed 42

Both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled biased / unbiased
against the checkpoint's own no-cue answer, on the questions with a no-cue answer in both the base model's and the
round's eval. Round's answers: the round's final eval, read through the round's model. Base model's answers: the
base eval on the same questions, read through the base model. Built by `rounds.py LINE probe_table`.

| Round | Answers | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | round's answers | 410 / 440 | 0.386 / 0.074 | 0.863 |
| round1 | base model's answers | 150 / 418 | 0.762 / 0.228 | 0.916 |
| round2 | round's answers | 188 / 455 | 0.622 / 0.126 | 0.903 |
| round2 | base model's answers | 162 / 430 | 0.750 / 0.230 | 0.908 |
| round3 | round's answers | 165 / 449 | 0.556 / 0.118 | 0.893 |
| round3 | base model's answers | 157 / 425 | 0.744 / 0.230 | 0.905 |
| round4 | round's answers | 73 / 439 | 0.399 / 0.085 | 0.852 |
| round4 | base model's answers | 162 / 430 | 0.744 / 0.231 | 0.903 |
| round5 | round's answers | 82 / 452 | 0.608 / 0.178 | 0.854 |
| round5 | base model's answers | 163 / 433 | 0.744 / 0.234 | 0.902 |
