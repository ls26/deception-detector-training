# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 298 / 451 | 0.201 / 0.064 | 0.761 |
| round1 | Round R's probe; base model's answers; base model | 157 / 428 | 0.749 / 0.231 | 0.906 |
| round1 | First probe; round R's answers; round R's model | 298 / 451 | 0.201 / 0.064 | 0.761 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 298 / 451 | 0.637 / 0.344 | 0.832 |
| round2 | Round R's probe; round R's answers; round R's model | 167 / 446 | 0.311 / 0.105 | 0.821 |
| round2 | Round R's probe; base model's answers; base model | 156 / 419 | 0.786 / 0.325 | 0.908 |
| round2 | First probe; round R's answers; round R's model | 167 / 446 | 0.240 / 0.061 | 0.803 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 167 / 446 | 0.638 / 0.297 | 0.860 |
| round3 | Round R's probe; round R's answers; round R's model | 143 / 457 | 0.398 / 0.097 | 0.877 |
| round3 | Round R's probe; base model's answers; base model | 162 / 430 | 0.764 / 0.352 | 0.875 |
| round3 | First probe; round R's answers; round R's model | 143 / 457 | 0.343 / 0.057 | 0.892 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 143 / 457 | 0.580 / 0.179 | 0.885 |
| round4 | Round R's probe; round R's answers; round R's model | 108 / 448 | 0.497 / 0.100 | 0.887 |
| round4 | Round R's probe; base model's answers; base model | 157 / 425 | 0.791 / 0.395 | 0.875 |
| round4 | First probe; round R's answers; round R's model | 108 / 448 | 0.422 / 0.050 | 0.877 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 108 / 448 | 0.707 / 0.307 | 0.883 |
| round5 | Round R's probe; round R's answers; round R's model | 69 / 442 | 0.736 / 0.317 | 0.904 |
| round5 | Round R's probe; base model's answers; base model | 158 / 428 | 0.836 / 0.549 | 0.860 |
| round5 | First probe; round R's answers; round R's model | 69 / 442 | 0.565 / 0.091 | 0.915 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the
earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 449 / 1561 / 2716 / 523 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4228 |
| round3 | round1 | 449 / 1561 / 2716 / 523 | 378 | 378 / 378 / 378 / 378 | none | 4228 |
| round3 | round2 | 301 / 735 / 2840 / 467 | 301 | 301 / 301 / 301 / 301 | none | 4228 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5504 |
| round4 | round1 | 449 / 1561 / 2716 / 523 | 378 | 378 / 378 / 378 / 378 | none | 5504 |
| round4 | round2 | 301 / 735 / 2840 / 467 | 301 | 301 / 301 / 301 / 301 | none | 5504 |
| round4 | round3 | 319 / 642 / 2926 / 521 | 319 | 319 / 319 / 319 / 319 | none | 5504 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 6224 |
| round5 | round1 | 449 / 1561 / 2716 / 523 | 378 | 378 / 378 / 378 / 378 | none | 6224 |
| round5 | round2 | 301 / 735 / 2840 / 467 | 301 | 301 / 301 / 301 / 301 | none | 6224 |
| round5 | round3 | 319 / 642 / 2926 / 521 | 319 | 319 / 319 / 319 / 319 | none | 6224 |
| round5 | round4 | 180 / 353 / 2972 / 503 | 180 | 180 / 180 / 180 / 180 | none | 6224 |
