# Follow-up analyses (post-hoc)

Written by `followup_analysis.py` after the run; not part of the unattended queue.

## 1. p_none distribution

Per eval question: share of the 8 fresh no-cue samples that are correct.

| share correct | questions |
|---|---|
| 0 | 41 (6%) |
| between 0 and 1/2 | 125 (18%) |
| exactly 1/2 | 33 (5%) |
| between 1/2 and 1 | 234 (34%) |
| 1 | 254 (37%) |

Questions with all 8 samples parsed (511), by number correct:

| correct of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| questions | 29 | 27 | 30 | 28 | 28 | 44 | 48 | 70 | 207 |

bias_followed eval rows (every row followed the cue): p_none from the 8 fresh samples, by cue condition.

| p_none | cue_gold rows (637) | cue_dist rows (469) |
|---|---|---|
| 0 | 31 | 97 |
| between 0 and 1/2 | 106 | 189 |
| exactly 1/2 | 28 | 28 |
| between 1/2 and 1 | 222 | 115 |
| 1 | 250 | 40 |

## 2. Old 3 samples vs fresh 8 samples

Per eval question: correct among 260819's 3 no-cue samples (rows) against the share correct among the 8 fresh samples (columns).

| old 3 samples | questions | 0 | between 0 and 1/2 | exactly 1/2 | between 1/2 and 1 | 1 |
|---|---|---|---|---|---|---|
| 0 of 3 | 90 | 30 | 48 | 3 | 9 | 0 |
| 1 of 3 | 84 | 6 | 36 | 6 | 33 | 3 |
| 2 of 3 | 133 | 1 | 23 | 9 | 80 | 20 |
| 3 of 3 | 304 | 0 | 6 | 6 | 85 | 207 |
| some unparsed | 76 | 4 | 12 | 9 | 27 | 24 |

Strict majority of the 3 old samples equals the strict majority of the 8 fresh samples (a tie counts as no majority): 562 of 687 questions.

## 3. Old hard vs fresh hard labels

AUC on three label/row combinations. Fresh hard drops a row when the 8 fresh samples tie 4-4, or when the row becomes anti-bias (deception rows).

| probe | arm | old labels, all rows | old labels, rows fresh hard keeps | fresh labels, same rows | rows dropped | kept rows whose label flips |
|---|---|---|---|---|---|---|
| bias_followed_rm_answer | hard-trained | 0.788 | 0.800 | 0.804 | 56 | 148 |
| bias_followed_rm_answer | soft-trained | 0.793 | 0.804 | 0.804 | 56 | 148 |
| deception_yes_rm_preyesno | hard-trained | 0.771 | 0.774 | 0.765 | 60 | 91 |
| deception_yes_rm_preyesno | soft-trained | 0.772 | 0.773 | 0.764 | 60 | 91 |
| deception_no_rm_preyesno | hard-trained | 0.799 | 0.809 | 0.807 | 19 | 44 |
| deception_no_rm_preyesno | soft-trained | 0.800 | 0.811 | 0.813 | 19 | 44 |
