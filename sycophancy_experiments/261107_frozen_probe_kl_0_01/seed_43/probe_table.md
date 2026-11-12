# Round 1's probe per round, seed 43

Both cue conditions pooled: the cued eval answers (sample 0) that name the cued option, labelled biased / unbiased
against the checkpoint's own no-cue answer, on the questions with a no-cue answer in both the base model's and the
round's eval. Round's answers: the round's final eval, read through the round's model. Base model's answers: the
base eval on the same questions, read through the base model. Built by `rounds.py LINE probe_table`.

| Round | Answers | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | round's answers | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round1 | base model's answers | 159 / 428 | 0.745 / 0.230 | 0.903 |
| round2 | round's answers | 181 / 451 | 0.170 / 0.038 | 0.827 |
| round2 | base model's answers | 159 / 425 | 0.746 / 0.231 | 0.903 |
| round3 | round's answers | 112 / 440 | 0.184 / 0.039 | 0.816 |
| round3 | base model's answers | 160 / 427 | 0.743 / 0.231 | 0.902 |
| round4 | round's answers | 85 / 443 | 0.238 / 0.049 | 0.844 |
| round4 | base model's answers | 159 / 427 | 0.747 / 0.232 | 0.902 |
| round5 | round's answers | 59 / 445 | 0.212 / 0.035 | 0.851 |
| round5 | base model's answers | 164 / 436 | 0.746 / 0.233 | 0.904 |
