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
| round2 | Round R's probe; round R's answers; round R's model | 153 / 448 | 0.414 / 0.150 | 0.830 |
| round2 | Round R's probe; base model's answers; base model | 157 / 426 | 0.701 / 0.261 | 0.890 |
| round2 | First probe; round R's answers; round R's model | 153 / 448 | 0.281 / 0.096 | 0.777 |
| round3 | Round R's probe; its own test set; round R-1's model (round 1: base) | 153 / 448 | 0.561 / 0.266 | 0.836 |
| round3 | Round R's probe; round R's answers; round R's model | 85 / 454 | 0.372 / 0.107 | 0.866 |
| round3 | Round R's probe; base model's answers; base model | 163 / 431 | 0.669 / 0.305 | 0.855 |
| round3 | First probe; round R's answers; round R's model | 85 / 454 | 0.351 / 0.059 | 0.878 |
| round4 | Round R's probe; its own test set; round R-1's model (round 1: base) | 85 / 454 | 0.591 / 0.297 | 0.807 |
| round4 | Round R's probe; round R's answers; round R's model | 61 / 448 | 0.548 / 0.177 | 0.887 |
| round4 | Round R's probe; base model's answers; base model | 161 / 431 | 0.733 / 0.417 | 0.836 |
| round4 | First probe; round R's answers; round R's model | 61 / 448 | 0.440 / 0.065 | 0.911 |
| round5 | Round R's probe; its own test set; round R-1's model (round 1: base) | 61 / 448 | 0.661 / 0.265 | 0.879 |
| round5 | Round R's probe; round R's answers; round R's model | 59 / 445 | 0.397 / 0.179 | 0.738 |
| round5 | Round R's probe; base model's answers; base model | 160 / 429 | 0.723 / 0.430 | 0.806 |
| round5 | First probe; round R's answers; round R's model | 59 / 445 | 0.370 / 0.067 | 0.861 |

# Probe training sets per round

Pool: a source's rows per group (biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue
at wrong). Per group: the rows each group takes (the smallest group, capped at 378, floored at 100). Own rows: the
source's own rows per group; topped up: rows from an earlier own-answer source filling a group.

| Round | Source | Pool | Per group | Own rows | Topped up | Training rows (round) |
|---|---|---|---|---|---|---|
| round1 | original | 378 / 378 / 378 / 378 | 378 | 378 / 378 / 378 / 378 | none | 1512 |
| round2 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round2 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 3024 |
| round3 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4044 |
| round3 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4044 |
| round3 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4044 |
| round4 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4584 |
| round4 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4584 |
| round4 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4584 |
| round4 | round3 | 135 / 237 / 2923 / 614 | 135 | 135 / 135 / 135 / 135 | none | 4584 |
| round5 | original | 378 / 1037 / 2739 / 642 | 378 | 378 / 378 / 378 / 378 | none | 4984 |
| round5 | round1 | 453 / 1425 / 2727 / 563 | 378 | 378 / 378 / 378 / 378 | none | 4984 |
| round5 | round2 | 255 / 631 / 2840 / 551 | 255 | 255 / 255 / 255 / 255 | none | 4984 |
| round5 | round3 | 135 / 237 / 2923 / 614 | 135 | 135 / 135 / 135 / 135 | none | 4984 |
| round5 | round4 | 41 / 94 / 2936 / 642 | 100 | 41 / 94 / 100 / 100 | bias / cue_gold: +59 from round1; bias / cue_dist: +6 from round3 | 4984 |
