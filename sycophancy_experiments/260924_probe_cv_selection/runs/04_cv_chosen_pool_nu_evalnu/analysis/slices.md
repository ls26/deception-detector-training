# 04_cv_chosen_pool_nu_evalnu: combined-score AUC by slice (seed 0 weights)

## bias_rm_answer

layers 13-16 (4), eval_filter=none_unanimous, n_eval=783 (147/636), overall AUC (mean_sigmoid) 0.888

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.810 | 393 | 51 | 342 |
| cue_dist | 0.936 | 390 | 96 | 294 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.829 | 536 | 147 | 389 |
| against cue | nan | 247 | 0 | 247 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.890 | 140 | 28 | 112 |
| history | 0.935 | 140 | 15 | 125 |
| law | 0.891 | 163 | 38 | 125 |
| philosophy | 0.841 | 189 | 32 | 157 |
| physics | 0.899 | 151 | 34 | 117 |

## bias_followed_rm_answer

layers 13-16 (4), eval_filter=none_unanimous, n_eval=612 (223/389), overall AUC (mean_sigmoid) 0.851

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.836 | 372 | 69 | 303 |
| cue_dist | 0.820 | 240 | 154 | 86 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.874 | 115 | 45 | 70 |
| history | 0.958 | 102 | 32 | 70 |
| law | 0.875 | 132 | 52 | 80 |
| philosophy | 0.738 | 141 | 47 | 94 |
| physics | 0.831 | 122 | 47 | 75 |

## deception_rm_preyesno

layers 14-17 (4), eval_filter=none_unanimous, n_eval=783 (455/328), overall AUC (mean_sigmoid) 0.618

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.559 | 393 | 215 | 178 |
| cue_dist | 0.663 | 390 | 240 | 150 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.526 | 536 | 274 | 262 |
| against cue | 0.684 | 247 | 181 | 66 |

| by self-report | AUC | n | pos | neg |
|---|---|---|---|---|
| Yes | 0.740 | 490 | 399 | 91 |
| No | 0.333 | 293 | 56 | 237 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.611 | 140 | 74 | 66 |
| history | 0.666 | 140 | 86 | 54 |
| law | 0.601 | 163 | 97 | 66 |
| philosophy | 0.621 | 189 | 109 | 80 |
| physics | 0.565 | 151 | 89 | 62 |

| by bias | AUC | n | pos | neg |
|---|---|---|---|---|
| bias | 0.483 | 147 | 56 | 91 |
| no bias | 0.618 | 636 | 399 | 237 |

## deception_yes_rm_preyesno

layers 14-17 (4), eval_filter=none_unanimous, n_eval=490 (399/91), overall AUC (mean_sigmoid) 0.832

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.781 | 224 | 194 | 30 |
| cue_dist | 0.883 | 266 | 205 | 61 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.731 | 309 | 218 | 91 |
| against cue | nan | 181 | 181 | 0 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.804 | 74 | 60 | 14 |
| history | 0.865 | 95 | 83 | 12 |
| law | 0.858 | 115 | 87 | 28 |
| philosophy | 0.795 | 117 | 97 | 20 |
| physics | 0.822 | 89 | 72 | 17 |

## deception_no_rm_preyesno

layers 14-17 (4), eval_filter=none_unanimous, n_eval=293 (56/237), overall AUC (mean_sigmoid) 0.766

| by cue | AUC | n | pos | neg |
|---|---|---|---|---|
| cue_gold | 0.739 | 169 | 21 | 148 |
| cue_dist | 0.779 | 124 | 35 | 89 |

| by cue-following | AUC | n | pos | neg |
|---|---|---|---|---|
| followed cue | 0.742 | 227 | 56 | 171 |
| against cue | nan | 66 | 0 | 66 |

| by category | AUC | n | pos | neg |
|---|---|---|---|---|
| computer science | 0.764 | 66 | 14 | 52 |
| history | 0.619 | 45 | 3 | 42 |
| law | 0.711 | 48 | 10 | 38 |
| philosophy | 0.819 | 72 | 12 | 60 |
| physics | 0.758 | 62 | 17 | 45 |
