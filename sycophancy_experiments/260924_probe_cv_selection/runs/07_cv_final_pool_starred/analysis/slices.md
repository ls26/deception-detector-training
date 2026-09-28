# 07_cv_final_pool_starred: combined-score AUC by slice (seed 0 weights)

## bias_followed_rm_answer

layers 13-16 (4), eval_filter=none, n_eval=1106 (449/657), overall AUC (mean_sigmoid) 0.788

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.806 | 637 | 160 | 477 |
| cue_dist | 0.734 | 469 | 289 | 180 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.757 | 207 | 87 | 120 |
| history | 0.878 | 162 | 55 | 107 |
| law | 0.759 | 245 | 105 | 140 |
| philosophy | 0.797 | 235 | 90 | 145 |
| physics | 0.771 | 257 | 112 | 145 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.873 | 612 | 223 | 389 |
| False | 0.656 | 494 | 226 | 268 |

## deception_yes_rm_preyesno

layers 14-17 (4), eval_filter=none, n_eval=836 (634/202), overall AUC (mean_sigmoid) 0.793

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.749 | 395 | 321 | 74 |
| cue_dist | 0.842 | 441 | 313 | 128 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.687 | 588 | 386 | 202 |
| against cue | nan | 248 | 248 | 0 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.740 | 128 | 102 | 26 |
| history | 0.866 | 144 | 120 | 24 |
| law | 0.802 | 209 | 146 | 63 |
| philosophy | 0.800 | 191 | 146 | 45 |
| physics | 0.759 | 164 | 120 | 44 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.847 | 490 | 399 | 91 |
| False | 0.710 | 346 | 235 | 111 |

## deception_no_rm_preyesno

layers 15-18 (4), eval_filter=none, n_eval=490 (120/370), overall AUC (mean_sigmoid) 0.749

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.722 | 270 | 51 | 219 |
| cue_dist | 0.783 | 220 | 69 | 151 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.693 | 391 | 120 | 271 |
| against cue | nan | 99 | 0 | 99 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.737 | 113 | 30 | 83 |
| history | 0.709 | 70 | 9 | 61 |
| law | 0.676 | 76 | 17 | 59 |
| philosophy | 0.729 | 102 | 21 | 81 |
| physics | 0.782 | 129 | 43 | 86 |

| by none-unanimous | AUC | n | pos | neg |
|---|---|---|---|---|
| True | 0.799 | 293 | 56 | 237 |
| False | 0.663 | 197 | 64 | 133 |
