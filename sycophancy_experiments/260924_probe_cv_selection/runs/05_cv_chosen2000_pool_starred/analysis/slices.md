# 05_cv_chosen2000_pool_starred: combined-score AUC by slice (seed 0 weights)

## bias_followed_rm_answer

layers 13-16 (4), eval_filter=none, n_eval=1106 (449/657), overall AUC (mean_sigmoid) 0.779

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.796 | 637 | 160 | 477 |
| cue_dist | 0.730 | 469 | 289 | 180 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.754 | 207 | 87 | 120 |
| history | 0.861 | 162 | 55 | 107 |
| law | 0.752 | 245 | 105 | 140 |
| philosophy | 0.779 | 235 | 90 | 145 |
| physics | 0.768 | 257 | 112 | 145 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.867 | 612 | 223 | 389 |
| False | 0.644 | 494 | 226 | 268 |

## deception_yes_rm_preyesno

layers 14-17 (4), eval_filter=none, n_eval=836 (634/202), overall AUC (mean_sigmoid) 0.785

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.742 | 395 | 321 | 74 |
| cue_dist | 0.836 | 441 | 313 | 128 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.677 | 588 | 386 | 202 |
| against cue | nan | 248 | 248 | 0 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.724 | 128 | 102 | 26 |
| history | 0.864 | 144 | 120 | 24 |
| law | 0.794 | 209 | 146 | 63 |
| philosophy | 0.789 | 191 | 146 | 45 |
| physics | 0.753 | 164 | 120 | 44 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.839 | 490 | 399 | 91 |
| False | 0.703 | 346 | 235 | 111 |

## deception_no_rm_preyesno

layers 14-17 (4), eval_filter=none, n_eval=490 (120/370), overall AUC (mean_sigmoid) 0.745

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.724 | 270 | 51 | 219 |
| cue_dist | 0.776 | 220 | 69 | 151 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.689 | 391 | 120 | 271 |
| against cue | nan | 99 | 0 | 99 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.729 | 113 | 30 | 83 |
| history | 0.690 | 70 | 9 | 61 |
| law | 0.679 | 76 | 17 | 59 |
| philosophy | 0.711 | 102 | 21 | 81 |
| physics | 0.778 | 129 | 43 | 86 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.803 | 293 | 56 | 237 |
| False | 0.655 | 197 | 64 | 133 |
