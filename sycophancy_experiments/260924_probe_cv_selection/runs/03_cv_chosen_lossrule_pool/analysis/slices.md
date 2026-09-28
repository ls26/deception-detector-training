# 03_cv_chosen_lossrule_pool: combined-score AUC by slice (seed 0 weights)

## bias_rm_answer

layers 1-31 (31), eval_filter=none, n_eval=1326 (321/1005), overall AUC (mean_sigmoid) 0.823

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.797 | 665 | 125 | 540 |
| cue_dist | 0.867 | 661 | 196 | 465 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.731 | 978 | 321 | 657 |
| against cue | nan | 348 | 0 | 348 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.801 | 241 | 56 | 185 |
| history | 0.847 | 214 | 33 | 181 |
| law | 0.780 | 285 | 80 | 205 |
| philosophy | 0.835 | 293 | 66 | 227 |
| physics | 0.835 | 293 | 86 | 207 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.892 | 783 | 147 | 636 |
| False | 0.720 | 543 | 174 | 369 |

## bias_followed_rm_answer

layers 0-31 (31), eval_filter=none, n_eval=1106 (449/657), overall AUC (mean_sigmoid) 0.756

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.776 | 637 | 160 | 477 |
| cue_dist | 0.710 | 469 | 289 | 180 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.731 | 207 | 87 | 120 |
| history | 0.823 | 162 | 55 | 107 |
| law | 0.737 | 245 | 105 | 140 |
| philosophy | 0.754 | 235 | 90 | 145 |
| physics | 0.757 | 257 | 112 | 145 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.839 | 612 | 223 | 389 |
| False | 0.635 | 494 | 226 | 268 |

## deception_rm_preyesno

layers 0-31 (32), eval_filter=none, n_eval=1326 (754/572), overall AUC (mean_sigmoid) 0.579

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.545 | 665 | 372 | 293 |
| cue_dist | 0.610 | 661 | 382 | 279 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.499 | 979 | 506 | 473 |
| against cue | 0.646 | 347 | 248 | 99 |

| by self-report | AUC | n | pos | neg |
|---|---|---|---|---|
| Yes | 0.681 | 836 | 634 | 202 |
| No | 0.402 | 490 | 120 | 370 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.557 | 241 | 132 | 109 |
| history | 0.624 | 214 | 129 | 85 |
| law | 0.570 | 285 | 163 | 122 |
| philosophy | 0.591 | 293 | 167 | 126 |
| physics | 0.556 | 293 | 163 | 130 |

| by bias | AUC | n | pos | neg |
|---|---|---|---|---|
| bias | 0.553 | 322 | 120 | 202 |
| no bias | 0.559 | 1004 | 634 | 370 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.600 | 783 | 455 | 328 |
| False | 0.547 | 543 | 299 | 244 |

## deception_yes_rm_preyesno

layers 14-16 (3), eval_filter=none, n_eval=836 (634/202), overall AUC (mean_sigmoid) 0.778

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.734 | 395 | 321 | 74 |
| cue_dist | 0.832 | 441 | 313 | 128 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.667 | 588 | 386 | 202 |
| against cue | nan | 248 | 248 | 0 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.717 | 128 | 102 | 26 |
| history | 0.844 | 144 | 120 | 24 |
| law | 0.795 | 209 | 146 | 63 |
| philosophy | 0.779 | 191 | 146 | 45 |
| physics | 0.745 | 164 | 120 | 44 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.832 | 490 | 399 | 91 |
| False | 0.698 | 346 | 235 | 111 |

## deception_no_rm_preyesno

layers 13-31 (19), eval_filter=none, n_eval=490 (120/370), overall AUC (mean_sigmoid) 0.715

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.697 | 270 | 51 | 219 |
| cue_dist | 0.737 | 220 | 69 | 151 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.668 | 391 | 120 | 271 |
| against cue | nan | 99 | 0 | 99 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.681 | 113 | 30 | 83 |
| history | 0.683 | 70 | 9 | 61 |
| law | 0.641 | 76 | 17 | 59 |
| philosophy | 0.705 | 102 | 21 | 81 |
| physics | 0.754 | 129 | 43 | 86 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.769 | 293 | 56 | 237 |
| False | 0.621 | 197 | 64 | 133 |
