# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 92 / 435 | 0.630 / 0.126 | 0.899 |
| round1 | Round R's probe; base model's answers; base model | 162 / 428 | 0.747 / 0.229 | 0.906 |
| round1 | First probe; round R's answers; round R's model | 92 / 435 | 0.630 / 0.126 | 0.899 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 92 / 435 | 0.704 / 0.211 | 0.903 |
| round2 | Round R's probe; round R's answers; round R's model | 202 / 443 | 0.142 / 0.038 | 0.801 |
| round2 | Round R's probe; base model's answers; base model | 150 / 419 | 0.762 / 0.326 | 0.889 |
| round2 | First probe; round R's answers; round R's model | 202 / 443 | 0.218 / 0.059 | 0.795 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 202 / 443 | 0.724 / 0.412 | 0.843 |
| round3 | Round R's probe; round R's answers; round R's model | 80 / 449 | 0.523 / 0.135 | 0.886 |
| round3 | Round R's probe; base model's answers; base model | 162 / 432 | 0.758 / 0.349 | 0.879 |
| round3 | First probe; round R's answers; round R's model | 80 / 449 | 0.494 / 0.063 | 0.903 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 80 / 449 | 0.648 / 0.214 | 0.894 |
| round4 | Round R's probe; round R's answers; round R's model | 78 / 444 | 0.605 / 0.210 | 0.880 |
| round4 | Round R's probe; base model's answers; base model | 157 / 424 | 0.740 / 0.396 | 0.867 |
| round4 | First probe; round R's answers; round R's model | 78 / 444 | 0.437 / 0.078 | 0.876 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 78 / 444 | 0.707 / 0.309 | 0.853 |
| round5 | Round R's probe; round R's answers; round R's model | 93 / 452 | 0.550 / 0.197 | 0.856 |
| round5 | Round R's probe; base model's answers; base model | 159 / 427 | 0.776 / 0.434 | 0.845 |
| round5 | First probe; round R's answers; round R's model | 93 / 452 | 0.383 / 0.053 | 0.879 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the
earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 2196 |
| round2 | round1 | 171 / 444 / 2719 / 527 | 171 | 171 / 171 / 171 / 171 | none | 2196 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3532 |
| round3 | round1 | 171 / 444 / 2719 / 527 | 171 | 171 / 171 / 171 / 171 | none | 3532 |
| round3 | round2 | 334 / 1049 / 2885 / 468 | 334 | 334 / 334 / 334 / 334 | none | 3532 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4344 |
| round4 | round1 | 171 / 444 / 2719 / 527 | 171 | 171 / 171 / 171 / 171 | none | 4344 |
| round4 | round2 | 334 / 1049 / 2885 / 468 | 334 | 334 / 334 / 334 / 334 | none | 4344 |
| round4 | round3 | 203 / 344 / 2898 / 587 | 203 | 203 / 203 / 203 / 203 | none | 4344 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4756 |
| round5 | round1 | 171 / 444 / 2719 / 527 | 171 | 171 / 171 / 171 / 171 | none | 4756 |
| round5 | round2 | 334 / 1049 / 2885 / 468 | 334 | 334 / 334 / 334 / 334 | none | 4756 |
| round5 | round3 | 203 / 344 / 2898 / 587 | 203 | 203 / 203 / 203 / 203 | none | 4756 |
| round5 | round4 | 103 / 201 / 3013 / 449 | 103 | 103 / 103 / 103 / 103 | none | 4756 |
