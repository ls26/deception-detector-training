# Reproduction and extraction gates

spec_hash 10b44b32b4c2

## Reproduction gate: PASS

New step 3 on 260923's activations/pool, full train rows, seed 0, 260924's recipe per probe; eval AUC of the mean sigmoid over the band must be within 0.005 of 260924's number.

| probe | lr | steps | band | n_train | AUC | target | diff | pass | fit seconds | s / 1000 steps |
|---|---|---|---|---|---|---|---|---|---|---|
| bias_rm_answer | 0.01 | 4000 | 13-16 | 2908 | 0.8403 | 0.840 | +0.0003 | PASS | 187 | 47 |
| bias_followed_rm_answer | 0.01 | 4000 | 13-16 | 2908 | 0.7883 | 0.788 | +0.0003 | PASS | 186 | 47 |
| deception_rm_preyesno | 0.03 | 4000 | 14-17 | 2908 | 0.5944 | 0.590 | +0.0044 | PASS | 186 | 47 |
| deception_yes_rm_preyesno | 0.03 | 4000 | 14-17 | 1928 | 0.7934 | 0.793 | +0.0004 | PASS | 184 | 46 |
| deception_no_rm_preyesno | 0.03 | 4000 | 15-18 | 980 | 0.7490 | 0.749 | +0.0000 | PASS | 185 | 46 |

## Extraction gate: PASS

50 rows of 260923's pool eval probe sets through the new step 2; cosine with the cached activations at every layer must be >= 0.9999. lm_head on the layer-31 residual vs model(...).logits: max abs diff 0 (< 0.01).

| dataset | min cosine (all layers) | pass |
|---|---|---|
| bias_rm | 1.000000 | PASS |
| deception_rm | 1.000000 | PASS |
