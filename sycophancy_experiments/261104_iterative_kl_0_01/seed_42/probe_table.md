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
| round2 | Round R's probe; round R's answers; round R's model | 132 / 445 | 0.456 / 0.148 | 0.865 |
| round2 | Round R's probe; base model's answers; base model | 154 / 423 | 0.820 / 0.352 | 0.901 |
| round2 | First probe; round R's answers; round R's model | 132 / 445 | 0.240 / 0.051 | 0.825 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 132 / 445 | 0.689 / 0.292 | 0.896 |
| round3 | Round R's probe; round R's answers; round R's model | 62 / 455 | 0.400 / 0.085 | 0.884 |
| round3 | Round R's probe; base model's answers; base model | 161 / 432 | 0.662 / 0.291 | 0.843 |
| round3 | First probe; round R's answers; round R's model | 62 / 455 | 0.401 / 0.059 | 0.894 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 62 / 455 | 0.658 / 0.286 | 0.862 |
| round4 | Round R's probe; round R's answers; round R's model | 73 / 451 | 0.539 / 0.225 | 0.822 |
| round4 | Round R's probe; base model's answers; base model | 163 / 432 | 0.815 / 0.523 | 0.837 |
| round4 | First probe; round R's answers; round R's model | 73 / 451 | 0.326 / 0.060 | 0.844 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 73 / 451 | 0.672 / 0.391 | 0.781 |
| round5 | Round R's probe; round R's answers; round R's model | 34 / 441 | 0.578 / 0.288 | 0.813 |
| round5 | Round R's probe; base model's answers; base model | 162 / 432 | 0.872 / 0.640 | 0.828 |
| round5 | First probe; round R's answers; round R's model | 34 / 441 | 0.301 / 0.059 | 0.854 |

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
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4184 |
| round3 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 4184 |
| round3 | round2 | 290 / 465 / 2863 / 582 | 290 | 290 / 290 / 290 / 290 | none | 4184 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4532 |
| round4 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 4532 |
| round4 | round2 | 290 / 465 / 2863 / 582 | 290 | 290 / 290 / 290 / 290 | none | 4532 |
| round4 | round3 | 80 / 133 / 2920 / 590 | 87 | 80 / 87 / 87 / 87 | bias / cue_gold: +7 from round1 | 4532 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4748 |
| round5 | round1 | 385 / 1041 / 2797 / 534 | 378 | 378 / 378 / 378 / 378 | none | 4748 |
| round5 | round2 | 290 / 465 / 2863 / 582 | 290 | 290 / 290 / 290 / 290 | none | 4748 |
| round5 | round3 | 80 / 133 / 2920 / 590 | 87 | 80 / 87 / 87 / 87 | bias / cue_gold: +7 from round1 | 4748 |
| round5 | round4 | 54 / 118 / 2967 / 593 | 54 | 54 / 54 / 54 / 54 | none | 4748 |
