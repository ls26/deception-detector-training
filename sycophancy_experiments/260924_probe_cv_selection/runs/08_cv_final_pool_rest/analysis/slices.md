# 08_cv_final_pool_rest: combined-score AUC by slice (seed 0 weights)

## bias_rm_answer

layers 13-16 (4), eval_filter=none, n_eval=1326 (321/1005), overall AUC (mean_sigmoid) 0.840

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.811 | 665 | 125 | 540 |
| cue_dist | 0.873 | 661 | 196 | 465 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.760 | 978 | 321 | 657 |
| against cue | nan | 348 | 0 | 348 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.825 | 241 | 56 | 185 |
| history | 0.901 | 214 | 33 | 181 |
| law | 0.779 | 285 | 80 | 205 |
| philosophy | 0.861 | 293 | 66 | 227 |
| physics | 0.836 | 293 | 86 | 207 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.904 | 783 | 147 | 636 |
| False | 0.734 | 543 | 174 | 369 |

## deception_rm_preyesno

layers 14-17 (4), eval_filter=none, n_eval=1326 (754/572), overall AUC (mean_sigmoid) 0.594

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.546 | 665 | 372 | 293 |
| cue_dist | 0.640 | 661 | 382 | 279 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.512 | 979 | 506 | 473 |
| against cue | 0.710 | 347 | 248 | 99 |

| by self-report | AUC | n | pos | neg |
|---|---|---|---|---|
| Yes | 0.696 | 836 | 634 | 202 |
| No | 0.405 | 490 | 120 | 370 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.558 | 241 | 132 | 109 |
| history | 0.655 | 214 | 129 | 85 |
| law | 0.587 | 285 | 163 | 122 |
| philosophy | 0.581 | 293 | 167 | 126 |
| physics | 0.601 | 293 | 163 | 130 |

| by bias | AUC | n | pos | neg |
|---|---|---|---|---|
| bias | 0.561 | 322 | 120 | 202 |
| no bias | 0.578 | 1004 | 634 | 370 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.622 | 783 | 455 | 328 |
| False | 0.554 | 543 | 299 | 244 |
