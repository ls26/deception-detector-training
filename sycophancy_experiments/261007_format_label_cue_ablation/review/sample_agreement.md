# Agreement of the three sampled answers per prompt: CoT vs no-CoT

spec_hash 10b44b32b4c2. CoT = 260819's `scores.jsonl` (`random` cue variant, shared no-cue cell); no-CoT = `gens/nocot/turn1.jsonl`. Unanimous = all three samples parsed and agreed; pairwise disagreement = disagreeing pairs / parsed pairs; any unparsed = prompts with at least one unparsed sample.

| setup | condition | prompts | unanimous | pairwise disagreement | any unparsed |
|---|---|---|---|---|---|
| CoT | none | 2290 | 0.605 | 0.219 | 0.103 |
| CoT | cue_gold | 2290 | 0.718 | 0.149 | 0.079 |
| CoT | cue_dist | 2290 | 0.577 | 0.246 | 0.087 |
| no-CoT | none | 2290 | 0.767 | 0.150 | 0.011 |
| no-CoT | cue_gold | 2290 | 0.947 | 0.034 | 0.002 |
| no-CoT | cue_dist | 2290 | 0.890 | 0.072 | 0.003 |

Cued prompts whose cued cell and no-cue cell are both unanimous (the removal bias label cannot change under resampling):

| setup | share |
|---|---|
| CoT | 0.421 |
| no-CoT | 0.704 |

No-CoT P(A), renormalized over {A, B}, per condition:

| condition | median max(P(A), P(B)) | share >= 0.9 | share < 0.75 |
|---|---|---|---|
| none | 0.980 | 0.663 | 0.192 |
| cue_gold | 1.000 | 0.927 | 0.034 |
| cue_dist | 0.998 | 0.837 | 0.086 |
