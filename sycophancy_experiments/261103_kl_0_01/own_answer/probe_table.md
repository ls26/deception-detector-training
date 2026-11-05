# Probe scores on each model's eval answers

Both cue conditions pooled. Answers: the cued eval answers (sample 0) that name the cued option, labelled biased /
unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue answers), on the questions with a
no-cue answer in both the base model's and the model's eval. Built by `own_probe.py table`.

## Weight 1.0, KL 0.01 s42 (455 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 159 / 427 | 0.748 / 0.229 | 0.907 |
| Initial probe; the model's answers; the model | 221 / 454 | 0.200 / 0.052 | 0.790 |
| Own-answer probe; the model's answers; the model | 221 / 454 | 0.646 / 0.285 | 0.853 |

## Weight 1.0, KL 0.01 s43 (457 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 159 / 428 | 0.745 / 0.230 | 0.903 |
| Initial probe; the model's answers; the model | 274 / 451 | 0.201 / 0.043 | 0.801 |
| Own-answer probe; the model's answers; the model | 274 / 451 | 0.749 / 0.356 | 0.877 |

## Weight 1.0, KL 0.01 s44 (444 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 153 / 418 | 0.745 / 0.226 | 0.906 |
| Initial probe; the model's answers; the model | 107 / 426 | 0.595 / 0.150 | 0.879 |
| Own-answer probe; the model's answers; the model | 107 / 426 | 0.776 / 0.209 | 0.904 |
