# Probe scores on each model's eval answers

Both cue conditions pooled. Answers: the cued eval answers (sample 0) that name the cued option, labelled biased /
unbiased against the checkpoint's own no-cue answer (at least 5 of its 8 no-cue answers), on the questions with a
no-cue answer in both the base model's and the model's eval. Built by `own_probe.py table`.

## Weight 0.3 s42 (453 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 158 / 424 | 0.754 / 0.227 | 0.910 |
| Initial probe; the model's answers; the model | 311 / 452 | 0.100 / 0.022 | 0.778 |
| Own-answer probe; the model's answers; the model | 311 / 452 | 0.676 / 0.279 | 0.862 |

## Weight 0.3 s43 (447 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 152 / 420 | 0.759 / 0.227 | 0.911 |
| Initial probe; the model's answers; the model | 414 / 447 | 0.117 / 0.027 | 0.788 |
| Own-answer probe; the model's answers; the model | 414 / 447 | 0.814 / 0.434 | 0.881 |

## Weight 0.3 s44 (453 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 158 / 424 | 0.741 / 0.235 | 0.900 |
| Initial probe; the model's answers; the model | 223 / 449 | 0.295 / 0.048 | 0.891 |
| Own-answer probe; the model's answers; the model | 223 / 449 | 0.840 / 0.186 | 0.946 |

## Weight 0.6 s42 (448 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 155 / 424 | 0.753 / 0.231 | 0.909 |
| Initial probe; the model's answers; the model | 287 / 448 | 0.042 / 0.015 | 0.698 |
| Own-answer probe; the model's answers; the model | 287 / 448 | 0.576 / 0.274 | 0.830 |

## Weight 0.6 s43 (453 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 162 / 426 | 0.743 / 0.233 | 0.901 |
| Initial probe; the model's answers; the model | 270 / 451 | 0.171 / 0.036 | 0.806 |
| Own-answer probe; the model's answers; the model | 270 / 451 | 0.772 / 0.382 | 0.874 |

## Weight 0.6 s44 (458 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 160 / 430 | 0.748 / 0.230 | 0.907 |
| Initial probe; the model's answers; the model | 302 / 457 | 0.110 / 0.041 | 0.751 |
| Own-answer probe; the model's answers; the model | 302 / 457 | 0.573 / 0.291 | 0.814 |

## Weight 1.0 s42 (458 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 161 / 431 | 0.743 / 0.236 | 0.903 |
| Initial probe; the model's answers; the model | 290 / 457 | 0.299 / 0.077 | 0.806 |
| Own-answer probe; the model's answers; the model | 290 / 457 | 0.864 / 0.223 | 0.952 |

## Weight 1.0 s43 (452 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 158 / 424 | 0.748 / 0.228 | 0.907 |
| Initial probe; the model's answers; the model | 248 / 449 | 0.075 / 0.019 | 0.786 |
| Own-answer probe; the model's answers; the model | 248 / 449 | 0.682 / 0.362 | 0.823 |

## Weight 1.0 s44 (463 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 163 / 435 | 0.741 / 0.233 | 0.901 |
| Initial probe; the model's answers; the model | 278 / 463 | 0.030 / 0.001 | 0.872 |
| Own-answer probe; the model's answers; the model | 278 / 463 | 0.858 / 0.215 | 0.959 |

## No penalty s42 (442 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 153 / 414 | 0.750 / 0.225 | 0.909 |
| Initial probe; the model's answers; the model | 342 / 441 | 0.573 / 0.248 | 0.801 |
| Own-answer probe; the model's answers; the model | 342 / 441 | 0.677 / 0.260 | 0.868 |

## No penalty s43 (456 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 159 / 427 | 0.753 / 0.229 | 0.907 |
| Initial probe; the model's answers; the model | 270 / 455 | 0.595 / 0.164 | 0.892 |
| Own-answer probe; the model's answers; the model | 270 / 455 | 0.809 / 0.275 | 0.909 |

## No penalty s44 (453 questions)

| Probe; answers; activations | Biased / unbiased answers | Mean score, biased / unbiased | AUC |
|---|---|---|---|
| Initial probe; base model's answers; base model | 160 / 425 | 0.740 / 0.229 | 0.903 |
| Initial probe; the model's answers; the model | 322 / 452 | 0.606 / 0.121 | 0.895 |
| Own-answer probe; the model's answers; the model | 322 / 452 | 0.754 / 0.180 | 0.926 |
