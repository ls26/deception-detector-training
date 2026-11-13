# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 221 / 454 | 0.200 / 0.052 | 0.790 |
| round1 | Round R's probe; base model's answers; base model | 159 / 427 | 0.748 / 0.229 | 0.907 |
| round1 | First probe; round R's answers; round R's model | 221 / 454 | 0.200 / 0.052 | 0.790 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 221 / 454 | 0.673 / 0.352 | 0.840 |
| round2 | Round R's probe; round R's answers; round R's model | 156 / 454 | 0.493 / 0.166 | 0.857 |
| round2 | Round R's probe; base model's answers; base model | 162 / 431 | 0.807 / 0.358 | 0.889 |
| round2 | First probe; round R's answers; round R's model | 156 / 454 | 0.458 / 0.148 | 0.838 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 156 / 454 | 0.530 / 0.185 | 0.867 |
| round3 | Round R's probe; round R's answers; round R's model | 153 / 453 | 0.262 / 0.043 | 0.884 |
| round3 | Round R's probe; base model's answers; base model | 160 / 431 | 0.621 / 0.217 | 0.882 |
| round3 | First probe; round R's answers; round R's model | 153 / 453 | 0.327 / 0.038 | 0.889 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 153 / 453 | 0.635 / 0.194 | 0.889 |
| round4 | Round R's probe; round R's answers; round R's model | 94 / 440 | 0.561 / 0.166 | 0.875 |
| round4 | Round R's probe; base model's answers; base model | 158 / 426 | 0.799 / 0.433 | 0.866 |
| round4 | First probe; round R's answers; round R's model | 94 / 440 | 0.420 / 0.054 | 0.895 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 94 / 440 | 0.695 / 0.277 | 0.874 |
| round5 | Round R's probe; round R's answers; round R's model | 79 / 444 | 0.546 / 0.238 | 0.837 |
| round5 | Round R's probe; base model's answers; base model | 163 / 427 | 0.856 / 0.560 | 0.859 |
| round5 | First probe; round R's answers; round R's model | 79 / 444 | 0.222 / 0.033 | 0.839 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the
earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4208 |
| round3 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 4208 |
| round3 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 4208 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5336 |
| round4 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 5336 |
| round4 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 5336 |
| round4 | round3 | 282 / 600 / 2941 / 484 | 282 | 282 / 282 / 282 / 282 | none | 5336 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5964 |
| round5 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 5964 |
| round5 | round2 | 296 / 638 / 2941 / 521 | 296 | 296 / 296 / 296 / 296 | none | 5964 |
| round5 | round3 | 282 / 600 / 2941 / 484 | 282 | 282 / 282 / 282 / 282 | none | 5964 |
| round5 | round4 | 157 / 264 / 2956 / 492 | 157 | 157 / 157 / 157 / 157 | none | 5964 |
