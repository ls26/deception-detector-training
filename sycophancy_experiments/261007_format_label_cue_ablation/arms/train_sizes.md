# Train sizes

spec 10b44b32b4c2; arms ref_cot, nocot_sampled, nocot_logprob, bio_cot. Cells count the rows of step 1's train pool (one row per (question_id, condition) prompt after the class filter, POOL_SEED 53), i.e. the most step 1 can draw per cell. full_n_per_class = 2 x the arm's smallest cell; matched = min over arms. Rule: >= 200 matched, < 50 insufficient, otherwise unmatched.

## bias

| arm | class | cue_gold | cue_dist | full_n_per_class |
|---|---|---|---|---|
| ref_cot | bias | 244 | 497 | 488 |
| ref_cot | no bias | 1297 | 1044 | 488 |
| nocot_sampled | bias | 371 | 660 | 742 |
| nocot_sampled | no bias | 1218 | 930 | 742 |
| nocot_logprob | bias | 366 | 659 | 732 |
| nocot_logprob | no bias | 1208 | 912 | 732 |
| bio_cot | bias | 227 | 375 | 454 |
| bio_cot | no bias | 1316 | 1164 | 454 |

matched_n_per_class = 454, rule = **matched**.

`bias`: matched; step 1 draws 454 per class in every arm (227 per condition).

## bias_followed

| arm | class | cue_gold | cue_dist | full_n_per_class |
|---|---|---|---|---|
| ref_cot | bias | 328 | 700 | 656 |
| ref_cot | no bias | 1144 | 383 | 656 |
| nocot_sampled | bias | 396 | 733 | 792 |
| nocot_sampled | no bias | 1128 | 454 | 792 |
| nocot_logprob | bias | 366 | 659 | 732 |
| nocot_logprob | no bias | 1121 | 447 | 732 |
| bio_cot | bias | 313 | 632 | 626 |
| bio_cot | no bias | 1146 | 380 | 626 |

matched_n_per_class = 626, rule = **matched**.

`bias_followed`: matched; step 1 draws 626 per class in every arm (313 per condition).

## deception_yes

| arm | class | cue_gold | cue_dist | full_n_per_class |
|---|---|---|---|---|
| ref_cot | false admission | 751 | 809 | 340 |
| ref_cot | true admission | 170 | 356 | 340 |
| nocot_sampled | false admission | 925 | 774 | 606 |
| nocot_sampled | true admission | 303 | 518 | 606 |
| nocot_logprob | false admission | 914 | 761 | 602 |
| nocot_logprob | true admission | 301 | 513 | 602 |
| bio_cot | false admission | 1075 | 962 | 388 |
| bio_cot | true admission | 194 | 314 | 388 |

matched_n_per_class = 340, rule = **matched**.

`deception_yes`: matched; step 1 draws 340 per class in every arm (170 per condition).

## deception_no

| arm | class | cue_gold | cue_dist | full_n_per_class |
|---|---|---|---|---|
| ref_cot | false denial | 74 | 141 | 148 |
| ref_cot | true denial | 546 | 235 | 148 |
| nocot_sampled | false denial | 68 | 142 | 136 |
| nocot_sampled | true denial | 293 | 156 | 136 |
| nocot_logprob | false denial | 65 | 146 | 130 |
| nocot_logprob | true denial | 294 | 151 | 130 |
| bio_cot | false denial | 33 | 61 | 66 |
| bio_cot | true denial | 241 | 202 | 66 |

matched_n_per_class = 66, rule = **unmatched**.

`deception_no`: unmatched (matched size 66 < 200); step 1 draws each arm's full size (ref_cot 148, nocot_sampled 136, nocot_logprob 130, bio_cot 66) and the probe is marked unmatched.

`deception` (unstratified) is the union of the deception_yes and deception_no draws and takes no size of its own.
