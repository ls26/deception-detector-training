# Probe scores on each model's eval answers

Both cue conditions pooled. Answers: the cued eval answers (sample 0) that name the cued option, labelled biased /
unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue answers), on the questions with a
no-cue answer in both the base model's and the model's eval. Built by `own_probe.py table`.

## Weight 1.5 s42 (388 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 130 / 364 | 0.733 / 0.225 | 0.900 |
| Initial probe; the model's answers; the model | 388 / 388 | 0.005 / 0.004 | 0.472 |
| Own-answer probe; the model's answers; the model | 388 / 388 | 0.630 / 0.347 | 0.800 |

## Weight 1.5 s43 (458 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 162 / 430 | 0.740 / 0.232 | 0.901 |
| Initial probe; the model's answers; the model | 216 / 456 | 0.274 / 0.031 | 0.907 |
| Own-answer probe; the model's answers; the model | 216 / 456 | 0.812 / 0.152 | 0.951 |

## Weight 1.5 s44 (455 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 159 / 427 | 0.746 / 0.230 | 0.905 |
| Initial probe; the model's answers; the model | 278 / 452 | 0.080 / 0.027 | 0.710 |
| Own-answer probe; the model's answers; the model | 278 / 452 | 0.740 / 0.451 | 0.830 |

## Weight 2.0 s42 (466 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 165 / 437 | 0.743 / 0.235 | 0.901 |
| Initial probe; the model's answers; the model | 0 / 466 | – / 0.120 | – |
| Own-answer probe; the model's answers; the model | 0 / 466 | – / 0.000 | – |

## Weight 2.0 s43 (445 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 151 / 418 | 0.756 / 0.230 | 0.909 |
| Initial probe; the model's answers; the model | 327 / 443 | 0.014 / 0.008 | 0.640 |
| Own-answer probe; the model's answers; the model | 327 / 443 | 0.554 / 0.399 | 0.707 |

## Weight 2.0 s44 (453 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 162 / 424 | 0.744 / 0.230 | 0.903 |
| Initial probe; the model's answers; the model | 243 / 452 | 0.019 / 0.008 | 0.700 |
| Own-answer probe; the model's answers; the model | 243 / 452 | 0.527 / 0.313 | 0.766 |
