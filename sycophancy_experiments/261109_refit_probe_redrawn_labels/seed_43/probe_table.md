# Probe scores per round

Both cue conditions pooled. Answers except a probe's own test set: the cued eval answers (sample 0) that name the
cued option, labelled biased / unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue
answers), on the questions with a no-cue answer in both the base model's and the round's eval. A probe's own test
set: the trainer's saved test scores on its test file (round 1: 261010's 777 base-model eval answers; round R: round
R-1's final-eval answers in the same answer set; – if that file had one class). Built by `rounds.py table`.

| Round | Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|---|
| round1 | Round R's probe; its own test set; round R-1's model (round 1: base) | 292 / 485 | 0.722 / 0.255 | 0.873 |
| round1 | Round R's probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round1 | Round R's probe; base model's answers; base model | 159 / 428 | 0.745 / 0.230 | 0.903 |
| round1 | First probe; round R's answers; round R's model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| round2 | Round R's probe; its own test set; round R-1's model (round 1: base) | 274 / 451 | 0.646 / 0.288 | 0.857 |
| round2 | Round R's probe; round R's answers; round R's model | 127 / 445 | 0.473 / 0.163 | 0.861 |
| round2 | Round R's probe; base model's answers; base model | 159 / 428 | 0.696 / 0.260 | 0.887 |
| round2 | First probe; round R's answers; round R's model | 127 / 445 | 0.324 / 0.084 | 0.834 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 127 / 445 | 0.588 / 0.211 | 0.882 |
| round3 | Round R's probe; round R's answers; round R's model | 184 / 453 | 0.297 / 0.076 | 0.871 |
| round3 | Round R's probe; base model's answers; base model | 159 / 427 | 0.529 / 0.173 | 0.841 |
| round3 | First probe; round R's answers; round R's model | 184 / 453 | 0.483 / 0.145 | 0.859 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 184 / 453 | 0.634 / 0.256 | 0.874 |
| round4 | Round R's probe; round R's answers; round R's model | 120 / 451 | 0.416 / 0.127 | 0.861 |
| round4 | Round R's probe; base model's answers; base model | 160 / 427 | 0.663 / 0.312 | 0.849 |
| round4 | First probe; round R's answers; round R's model | 120 / 451 | 0.528 / 0.126 | 0.892 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 120 / 451 | 0.720 / 0.382 | 0.836 |
| round5 | Round R's probe; round R's answers; round R's model | 59 / 442 | 0.600 / 0.269 | 0.837 |
| round5 | Round R's probe; base model's answers; base model | 160 / 431 | 0.764 / 0.431 | 0.845 |
| round5 | First probe; round R's answers; round R's model | 59 / 442 | 0.440 / 0.082 | 0.880 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100; under 100 when the
earlier own-answer sources ran out of spare rows to fill the smallest group). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4084 |
| round3 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4084 |
| round3 | round2 | 265 / 477 / 2822 / 508 | 265 | 265 / 265 / 265 / 265 | none | 4084 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 5440 |
| round4 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 5440 |
| round4 | round2 | 265 / 477 / 2822 / 508 | 265 | 265 / 265 / 265 / 265 | none | 5440 |
| round4 | round3 | 339 / 735 / 2873 / 542 | 339 | 339 / 339 / 339 / 339 | none | 5440 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 6064 |
| round5 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 6064 |
| round5 | round2 | 265 / 477 / 2822 / 508 | 265 | 265 / 265 / 265 / 265 | none | 6064 |
| round5 | round3 | 339 / 735 / 2873 / 542 | 339 | 339 / 339 / 339 / 339 | none | 6064 |
| round5 | round4 | 156 / 365 / 2989 / 454 | 156 | 156 / 156 / 156 / 156 | none | 6064 |
