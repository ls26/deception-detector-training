# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 432 / 432 | 0.116 / 0.107 | 0.558 |
| round1 | Round R's probe; base model's answers; base model | 155 / 406 | 0.737 / 0.232 | 0.901 |
| round1 | First probe; round R's answers; round R's model | 432 / 432 | 0.116 / 0.107 | 0.558 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 432 / 432 | 0.543 / 0.372 | 0.736 |
| round2 | Round R's probe; round R's answers; round R's model | 425 / 455 | 0.526 / 0.251 | 0.830 |
| round2 | Round R's probe; base model's answers; base model | 160 / 429 | 0.664 / 0.372 | 0.788 |
| round2 | First probe; round R's answers; round R's model | 425 / 455 | 0.266 / 0.034 | 0.896 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 425 / 455 | 0.753 / 0.309 | 0.918 |
| round3 | Round R's probe; round R's answers; round R's model | 454 / 456 | 0.411 / 0.267 | 0.669 |
| round3 | Round R's probe; base model's answers; base model | 162 / 427 | 0.770 / 0.448 | 0.847 |
| round3 | First probe; round R's answers; round R's model | 454 / 456 | 0.176 / 0.116 | 0.540 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 454 / 456 | 0.682 / 0.427 | 0.791 |
| round4 | Round R's probe; round R's answers; round R's model | 164 / 450 | 0.651 / 0.333 | 0.834 |
| round4 | Round R's probe; base model's answers; base model | 159 / 427 | 0.619 / 0.356 | 0.811 |
| round4 | First probe; round R's answers; round R's model | 164 / 450 | 0.630 / 0.184 | 0.861 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 164 / 450 | 0.662 / 0.383 | 0.825 |
| round5 | Round R's probe; round R's answers; round R's model | 240 / 459 | 0.489 / 0.295 | 0.777 |
| round5 | Round R's probe; base model's answers; base model | 164 / 435 | 0.645 / 0.429 | 0.791 |
| round5 | First probe; round R's answers; round R's model | 240 / 459 | 0.434 / 0.077 | 0.858 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round3 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round3 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 4536 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round4 | round3 | 1197 / 2317 / 2328 / 1197 | 378 | 378 / 378 / 378 / 378 | none | 6048 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round1 | 972 / 1839 / 1839 / 972 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round2 | 951 / 2324 / 2544 / 972 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round3 | 1197 / 2317 / 2328 / 1197 | 378 | 378 / 378 / 378 / 378 | none | 7296 |
| round5 | round4 | 312 / 549 / 2809 / 757 | 312 | 312 / 312 / 312 / 312 | none | 7296 |
