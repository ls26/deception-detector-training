# Reference arm: 260924 -> this run, one change at a time

spec_hash 10b44b32b4c2. Eval AUC of the mean sigmoid over the band, seed 0. Cells:

- **A**: 260924 labels, full size, 260924 recipe
- **B**: run labels, full size, 260924 recipe
- **D**: run labels, matched size, 260924 recipe
- **C**: run labels, matched size, CV recipe (the run)

| probe | A | B | D | C | A->B labels | B->D size | D->C recipe |
|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.840 | 0.840 | 0.827 | 0.827 | +0.000 | -0.013 | +0.000 |
| bias_followed_rm_answer | 0.788 | 0.788 | 0.768 | 0.768 | +0.000 | -0.020 | +0.000 |
| deception_rm_preyesno | 0.594 | 0.635 | 0.642 | 0.649 | +0.041 | +0.007 | +0.006 |
| deception_yes_rm_preyesno | 0.793 | 0.771 | 0.752 | 0.752 | -0.022 | -0.020 | -0.000 |
| deception_no_rm_preyesno | 0.749 | 0.799 | 0.753 | 0.706 | +0.050 | -0.046 | -0.048 |

Train rows per cell:

| probe | A | B | D | C |
|---|---|---|---|---|
| bias_rm_answer | 2908 | 2908 | 908 | 908 |
| bias_followed_rm_answer | 2908 | 2908 | 1252 | 1252 |
| deception_rm_preyesno | 2908 | 2908 | 976 | 976 |
| deception_yes_rm_preyesno | 1928 | 1992 | 680 | 680 |
| deception_no_rm_preyesno | 980 | 916 | 296 | 296 |
