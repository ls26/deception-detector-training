# Comparison

spec_hash 10b44b32b4c2. Eval AUC of the combined score (mean sigmoid over the CV-chosen band), one seed, per arm's final fit (runs/<arm>/). Δ vs ref_cot is computed on the eval questions shared by the two arms; the 95% CI is a question-level bootstrap (2000 draws, seed 20261007); * marks a CI that excludes 0. Verdict probes: bias_followed, deception_yes, deception_no.

## Main table

| probe | ref_cot AUC | nocot_sampled AUC (Δ vs ref, 95% CI) | nocot_logprob AUC (Δ vs ref, 95% CI) | bio_cot AUC (Δ vs ref, 95% CI) |
|---|---|---|---|---|
| bias | 0.827 | 0.874 (+0.050*, [+0.023, +0.077], n=665) | 0.901 (+0.073*, [+0.048, +0.099], n=656) | 0.845 (+0.018, [-0.007, +0.041], n=669) |
| bias_followed (verdict) | 0.768 | 0.852 (+0.082*, [+0.051, +0.116], n=662) | 0.880 (+0.117*, [+0.083, +0.151], n=652) | 0.766 (-0.003, [-0.032, +0.027], n=668) |
| deception | 0.649 | 0.711 (+0.066*, [+0.031, +0.102], n=665) | 0.696 (+0.044*, [+0.008, +0.081], n=656) | 0.637 (-0.012, [-0.050, +0.025], n=669) |
| deception_yes (verdict) | 0.752 | 0.819 (+0.062*, [+0.028, +0.097], n=512) | 0.809 (+0.049*, [+0.014, +0.083], n=509) | 0.755 (+0.008, [-0.033, +0.048], n=542) |
| deception_no (verdict) | 0.706 | 0.779 (+0.080, [-0.009, +0.176], n=139) | 0.762 (+0.055, [-0.039, +0.152], n=137) | 0.717 (-0.037, [-0.156, +0.077], n=118) |

## Checks

### Chosen recipe per arm and probe (lr / steps / band)

| probe | ref_cot | nocot_sampled | nocot_logprob | bio_cot |
|---|---|---|---|---|
| bias | 0.01 / 4000 / 13-16 | 0.01 / 4000 / 13-16 | 0.01 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| bias_followed | 0.01 / 4000 / 13-16 | 0.01 / 8000 / 24-27 | 0.03 / 2000 / 13-16 | 0.01 / 2000 / 13-16 |
| deception | 0.1 / 8000 / 17-20 | 0.1 / 8000 / 27-30 | 0.1 / 8000 / 18-21 | 0.01 / 2000 / 14-17 |
| deception_yes | 0.1 / 4000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 8000 / 18-21 | 0.03 / 2000 / 14-17 |
| deception_no | 0.1 / 8000 / 22-25 | 0.03 / 2000 / 15-18 | 0.1 / 8000 / 28-31 | 0.1 / 2000 / 11-14 |

### Train size per probe (n_train rows) and the size rule

| probe | rule | matched n/class | ref_cot | nocot_sampled | nocot_logprob | bio_cot |
|---|---|---|---|---|---|---|
| bias | matched | 454 | 908 | 908 | 908 | 908 |
| bias_followed | matched | 626 | 1252 | 1252 | 1252 | 1252 |
| deception | union of deception_yes + deception_no | — | 976 | 952 | 940 | 812 |
| deception_yes | matched | 340 | 680 | 680 | 680 | 680 |
| deception_no | unmatched | 66 | 296 | 272 | 260 | 132 |

### nocot_logprob on the confident slice (both top letters >= 0.75)

| probe | full eval AUC (n) | confident slice AUC (n) |
|---|---|---|
| bias | 0.901 (1333) | 0.936 (1025) |
| bias_followed | 0.880 (1099) | 0.920 (827) |
| deception | 0.696 (1333) | 0.715 (1025) |
| deception_yes | 0.809 (1072) | 0.841 (839) |
| deception_no | 0.762 (261) | 0.826 (186) |

### Footnote: 260924's full-size reference numbers

260924's final recipe on 260923's `pool` activations (train 2908 / 2908 / 2908 / 1928 / 980 rows, as review/parts/reproduction_*.json record; all 3 samples per prompt): bias 0.840, bias_followed 0.788, deception 0.590, deception_yes 0.793, deception_no 0.749. Those probes were trained at the full size with three rows per prompt, so they are not the same procedure as the arms above.
