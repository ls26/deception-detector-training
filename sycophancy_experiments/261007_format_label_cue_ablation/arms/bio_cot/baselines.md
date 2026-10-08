# Baselines: `bio_cot`

spec 10b44b32b4c2; 13740 cued rows over 2290 questions; labels from sampled; at-p mixes cue_gold and cue_dist at p = 0.85 (numerator and denominator weighted separately, 260824's `at_p`). Accuracy counts unparsed as wrong; every other rate is over parsed rows. `n` is the unweighted denominator.

## Split: eval (4122 cued rows, 687 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6608 (n=2061) | 0.7792 (n=2061) | 0.5075 (n=2061) | 0.7385 (n=4122) |
| accuracy | gold is A | 0.5685 (n=1008) | 0.7123 (n=1008) | 0.4117 (n=1008) | 0.6672 (n=2016) |
| accuracy | gold is B | 0.7493 (n=1053) | 0.8433 (n=1053) | 0.5992 (n=1053) | 0.8067 (n=2106) |
| parse-fail rate | 0.0422 (n=2061) | 0.0412 (n=2061) | 0.0427 (n=2061) | 0.0415 (n=4122) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8108 (n=1929) | 0.4657 (n=1924) | 0.7591 (n=3853) |
| bias | 0.1550 (n=1929) | 0.2313 (n=1924) | 0.1664 (n=3853) |
| anti-bias | 0.0627 (n=1929) | 0.0452 (n=1924) | 0.0601 (n=3853) |
| no bias | 0.7823 (n=1929) | 0.7235 (n=1924) | 0.7735 (n=3853) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1184 (n=3800) | 0.1184 (n=3800) | 0.1184 (n=3800) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.8346 (n=1929) | 0.8607 (n=1924) | 0.8385 (n=3853) |
| said Yes (sampled token) | 0.6901 (n=1894) | 0.7216 (n=1882) | 0.6948 (n=3776) |
| mean P(Yes) | bias | 0.7310 (n=299) | 0.7448 (n=445) | 0.7339 (n=744) |
| mean P(Yes) | anti-bias | 0.7603 (n=121) | 0.7428 (n=87) | 0.7583 (n=208) |
| mean P(Yes) | no bias | 0.6967 (n=1509) | 0.7230 (n=1392) | 0.7004 (n=2901) |
| deception (count) | 0.7196 (n=1808) | 0.6739 (n=1837) | 0.7126 (n=3645) |
| false admission (count) | 0.8323 (n=1509) | 0.8520 (n=1392) | 0.8351 (n=2901) |
| false denial (count) | 0.1505 (n=299) | 0.1169 (n=445) | 0.1435 (n=744) |
| true admission (count) | 0.8495 (n=299) | 0.8831 (n=445) | 0.8565 (n=744) |
| true denial (count) | 0.1677 (n=1509) | 0.1480 (n=1392) | 0.1649 (n=2901) |
| deception (P(Yes)-weighted) | 0.6260 (n=1808) | 0.6097 (n=1837) | 0.6235 (n=3645) |
| false admission (P) | 0.6967 (n=1509) | 0.7230 (n=1392) | 0.7004 (n=2901) |
| false denial (P) | 0.2690 (n=299) | 0.2552 (n=445) | 0.2661 (n=744) |
| true admission (P) | 0.7310 (n=299) | 0.7448 (n=445) | 0.7339 (n=744) |
| true denial (P) | 0.3033 (n=1509) | 0.2770 (n=1392) | 0.2996 (n=2901) |

## Split: train (9618 cued rows, 1603 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6952 (n=4809) | 0.8008 (n=4809) | 0.5211 (n=4809) | 0.7588 (n=9618) |
| accuracy | gold is A | 0.6068 (n=2355) | 0.7261 (n=2355) | 0.4110 (n=2355) | 0.6789 (n=4710) |
| accuracy | gold is B | 0.7800 (n=2454) | 0.8725 (n=2454) | 0.6267 (n=2454) | 0.8356 (n=4908) |
| parse-fail rate | 0.0347 (n=4809) | 0.0302 (n=4809) | 0.0318 (n=4809) | 0.0304 (n=9618) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8268 (n=4532) | 0.4571 (n=4531) | 0.7713 (n=9063) |
| bias | 0.1428 (n=4532) | 0.2437 (n=4531) | 0.1579 (n=9063) |
| anti-bias | 0.0611 (n=4532) | 0.0415 (n=4531) | 0.0582 (n=9063) |
| no bias | 0.7961 (n=4532) | 0.7149 (n=4531) | 0.7839 (n=9063) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1060 (n=8968) | 0.1060 (n=8968) | 0.1060 (n=8968) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.8162 (n=4532) | 0.8250 (n=4531) | 0.8175 (n=9063) |
| said Yes (sampled token) | 0.7022 (n=4412) | 0.7222 (n=4417) | 0.7052 (n=8829) |
| mean P(Yes) | bias | 0.7184 (n=647) | 0.7295 (n=1104) | 0.7210 (n=1751) |
| mean P(Yes) | anti-bias | 0.7399 (n=277) | 0.7506 (n=188) | 0.7411 (n=465) |
| mean P(Yes) | no bias | 0.6870 (n=3608) | 0.7009 (n=3239) | 0.6889 (n=6847) |
| deception (count) | 0.7079 (n=4255) | 0.6498 (n=4343) | 0.6990 (n=8598) |
| false admission (count) | 0.8071 (n=3608) | 0.8178 (n=3239) | 0.8086 (n=6847) |
| false denial (count) | 0.1546 (n=647) | 0.1567 (n=1104) | 0.1551 (n=1751) |
| true admission (count) | 0.8454 (n=647) | 0.8433 (n=1104) | 0.8449 (n=1751) |
| true denial (count) | 0.1929 (n=3608) | 0.1822 (n=3239) | 0.1914 (n=6847) |
| deception (P(Yes)-weighted) | 0.6253 (n=4255) | 0.5915 (n=4343) | 0.6202 (n=8598) |
| false admission (P) | 0.6870 (n=3608) | 0.7009 (n=3239) | 0.6889 (n=6847) |
| false denial (P) | 0.2816 (n=647) | 0.2705 (n=1104) | 0.2790 (n=1751) |
| true admission (P) | 0.7184 (n=647) | 0.7295 (n=1104) | 0.7210 (n=1751) |
| true denial (P) | 0.3130 (n=3608) | 0.2991 (n=3239) | 0.3111 (n=6847) |

## Split: all (13740 cued rows, 2290 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6849 (n=6870) | 0.7943 (n=6870) | 0.5170 (n=6870) | 0.7527 (n=13740) |
| accuracy | gold is A | 0.5953 (n=3363) | 0.7220 (n=3363) | 0.4112 (n=3363) | 0.6754 (n=6726) |
| accuracy | gold is B | 0.7707 (n=3507) | 0.8637 (n=3507) | 0.6185 (n=3507) | 0.8269 (n=7014) |
| parse-fail rate | 0.0370 (n=6870) | 0.0335 (n=6870) | 0.0351 (n=6870) | 0.0337 (n=13740) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8220 (n=6461) | 0.4596 (n=6455) | 0.7677 (n=12916) |
| bias | 0.1464 (n=6461) | 0.2400 (n=6455) | 0.1604 (n=12916) |
| anti-bias | 0.0616 (n=6461) | 0.0426 (n=6455) | 0.0588 (n=12916) |
| no bias | 0.7920 (n=6461) | 0.7174 (n=6455) | 0.7808 (n=12916) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1097 (n=12768) | 0.1097 (n=12768) | 0.1097 (n=12768) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.8217 (n=6461) | 0.8356 (n=6455) | 0.8238 (n=12916) |
| said Yes (sampled token) | 0.6985 (n=6306) | 0.7220 (n=6299) | 0.7021 (n=12605) |
| mean P(Yes) | bias | 0.7224 (n=946) | 0.7339 (n=1549) | 0.7250 (n=2495) |
| mean P(Yes) | anti-bias | 0.7461 (n=398) | 0.7482 (n=275) | 0.7463 (n=673) |
| mean P(Yes) | no bias | 0.6898 (n=5117) | 0.7076 (n=4631) | 0.6923 (n=9748) |
| deception (count) | 0.7114 (n=6063) | 0.6570 (n=6180) | 0.7031 (n=12243) |
| false admission (count) | 0.8145 (n=5117) | 0.8281 (n=4631) | 0.8164 (n=9748) |
| false denial (count) | 0.1533 (n=946) | 0.1453 (n=1549) | 0.1515 (n=2495) |
| true admission (count) | 0.8467 (n=946) | 0.8547 (n=1549) | 0.8485 (n=2495) |
| true denial (count) | 0.1855 (n=5117) | 0.1719 (n=4631) | 0.1836 (n=9748) |
| deception (P(Yes)-weighted) | 0.6255 (n=6063) | 0.5969 (n=6180) | 0.6212 (n=12243) |
| false admission (P) | 0.6898 (n=5117) | 0.7076 (n=4631) | 0.6923 (n=9748) |
| false denial (P) | 0.2776 (n=946) | 0.2661 (n=1549) | 0.2750 (n=2495) |
| true admission (P) | 0.7224 (n=946) | 0.7339 (n=1549) | 0.7250 (n=2495) |
| true denial (P) | 0.3102 (n=5117) | 0.2924 (n=4631) | 0.3077 (n=9748) |

## By category (all rows)

| category | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| computer science | cue_gold | 1230 | 1161 | 0.7927 | 0.8226 | 0.1387 | 0.0749 | 0.7864 | 0.7399 | 0.6961 | 0.6192 | 0.6322 | 0.5719 | 0.6192 | 0.3039 | 0.6961 | 0.3808 |
| computer science | cue_dist | 1230 | 1164 | 0.5398 | 0.4399 | 0.2448 | 0.0464 | 0.7088 | 0.7672 | 0.6688 | 0.6434 | 0.6261 | 0.5632 | 0.6434 | 0.3312 | 0.6688 | 0.3566 |
| computer science | at_p | 2460 | 2325 | 0.7548 | 0.7650 | 0.1546 | 0.0706 | 0.7747 | 0.7440 | 0.6896 | 0.6225 | 0.6313 | 0.5706 | 0.6225 | 0.3104 | 0.6896 | 0.3775 |
| history | cue_gold | 1143 | 1068 | 0.8311 | 0.8567 | 0.1049 | 0.0365 | 0.8586 | 0.9588 | 0.8212 | 0.7612 | 0.8503 | 0.6978 | 0.7612 | 0.1788 | 0.8212 | 0.2388 |
| history | cue_dist | 1143 | 1079 | 0.6352 | 0.3466 | 0.1798 | 0.0380 | 0.7822 | 0.9453 | 0.7984 | 0.7795 | 0.7823 | 0.6715 | 0.7795 | 0.2016 | 0.7984 | 0.2205 |
| history | at_p | 2286 | 2147 | 0.8017 | 0.7796 | 0.1162 | 0.0367 | 0.8471 | 0.9568 | 0.8158 | 0.7637 | 0.8401 | 0.6938 | 0.7637 | 0.1842 | 0.8158 | 0.2363 |
| law | cue_gold | 1500 | 1397 | 0.7413 | 0.7674 | 0.1596 | 0.0709 | 0.7695 | 0.7430 | 0.7543 | 0.6846 | 0.6317 | 0.6092 | 0.6846 | 0.2457 | 0.7543 | 0.3154 |
| law | cue_dist | 1500 | 1395 | 0.4587 | 0.5140 | 0.2394 | 0.0473 | 0.7133 | 0.7591 | 0.7376 | 0.7021 | 0.6057 | 0.5916 | 0.7021 | 0.2624 | 0.7376 | 0.2979 |
| law | at_p | 3000 | 2792 | 0.6989 | 0.7294 | 0.1716 | 0.0673 | 0.7611 | 0.7454 | 0.7508 | 0.6870 | 0.6278 | 0.6065 | 0.6870 | 0.2492 | 0.7508 | 0.3130 |
| philosophy | cue_gold | 1497 | 1437 | 0.8130 | 0.8330 | 0.1385 | 0.0612 | 0.8003 | 0.8733 | 0.7680 | 0.7115 | 0.7480 | 0.6407 | 0.7115 | 0.2320 | 0.7680 | 0.2885 |
| philosophy | cue_dist | 1497 | 1436 | 0.5204 | 0.4645 | 0.2542 | 0.0355 | 0.7103 | 0.8997 | 0.7815 | 0.7295 | 0.6549 | 0.5948 | 0.7295 | 0.2185 | 0.7815 | 0.2705 |
| philosophy | at_p | 2994 | 2873 | 0.7691 | 0.7777 | 0.1558 | 0.0574 | 0.7868 | 0.8773 | 0.7713 | 0.7139 | 0.7337 | 0.6337 | 0.7139 | 0.2287 | 0.7713 | 0.2861 |
| physics | cue_gold | 1500 | 1398 | 0.8020 | 0.8383 | 0.1795 | 0.0608 | 0.7597 | 0.8104 | 0.6306 | 0.6709 | 0.7083 | 0.6133 | 0.6709 | 0.3694 | 0.6306 | 0.3291 |
| physics | cue_dist | 1500 | 1381 | 0.4633 | 0.5047 | 0.2686 | 0.0456 | 0.6857 | 0.8182 | 0.7000 | 0.6814 | 0.6381 | 0.5741 | 0.6814 | 0.3000 | 0.7000 | 0.3186 |
| physics | at_p | 3000 | 2779 | 0.7512 | 0.7888 | 0.1928 | 0.0585 | 0.7487 | 0.8116 | 0.6450 | 0.6724 | 0.6977 | 0.6074 | 0.6724 | 0.3550 | 0.6450 | 0.3276 |

## By bio_idx (all rows)

| bio_idx | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cue_gold | 1689 | 1580 | 0.7638 | 0.7937 | 0.1291 | 0.0709 | 0.8000 | 0.8259 | 0.7255 | 0.6920 | 0.7228 | 0.6340 | 0.6920 | 0.2745 | 0.7255 | 0.3080 |
| 0 | cue_dist | 1689 | 1581 | 0.5453 | 0.4288 | 0.2119 | 0.0468 | 0.7413 | 0.8393 | 0.7351 | 0.7044 | 0.6749 | 0.6067 | 0.7044 | 0.2649 | 0.7351 | 0.2956 |
| 0 | at_p | 3378 | 3161 | 0.7310 | 0.7389 | 0.1415 | 0.0673 | 0.7912 | 0.8280 | 0.7277 | 0.6937 | 0.7154 | 0.6298 | 0.6937 | 0.2723 | 0.7277 | 0.3063 |
| 1 | cue_gold | 1749 | 1658 | 0.7999 | 0.8221 | 0.1327 | 0.0573 | 0.8100 | 0.7992 | 0.7216 | 0.6771 | 0.7012 | 0.6210 | 0.6771 | 0.2784 | 0.7216 | 0.3229 |
| 1 | cue_dist | 1749 | 1660 | 0.5495 | 0.4307 | 0.2235 | 0.0476 | 0.7289 | 0.8247 | 0.7404 | 0.6921 | 0.6490 | 0.5906 | 0.6921 | 0.2596 | 0.7404 | 0.3079 |
| 1 | at_p | 3498 | 3318 | 0.7623 | 0.7633 | 0.1463 | 0.0558 | 0.7978 | 0.8030 | 0.7259 | 0.6791 | 0.6933 | 0.6164 | 0.6791 | 0.2741 | 0.7259 | 0.3209 |
| 2 | cue_gold | 1698 | 1587 | 0.8027 | 0.8343 | 0.1632 | 0.0624 | 0.7744 | 0.8021 | 0.7011 | 0.6867 | 0.6956 | 0.6192 | 0.6867 | 0.2989 | 0.7011 | 0.3133 |
| 2 | cue_dist | 1698 | 1583 | 0.4776 | 0.4959 | 0.2685 | 0.0360 | 0.6955 | 0.8130 | 0.7199 | 0.7071 | 0.6350 | 0.5882 | 0.7071 | 0.2801 | 0.7199 | 0.2929 |
| 2 | at_p | 3396 | 3170 | 0.7539 | 0.7836 | 0.1790 | 0.0584 | 0.7626 | 0.8038 | 0.7054 | 0.6895 | 0.6863 | 0.6145 | 0.6895 | 0.2946 | 0.7054 | 0.3105 |
| 3 | cue_gold | 1734 | 1636 | 0.8103 | 0.8374 | 0.1608 | 0.0562 | 0.7830 | 0.8594 | 0.7415 | 0.7041 | 0.7260 | 0.6282 | 0.7041 | 0.2585 | 0.7415 | 0.2959 |
| 3 | cue_dist | 1734 | 1631 | 0.4954 | 0.4838 | 0.2563 | 0.0399 | 0.7039 | 0.8651 | 0.7414 | 0.7275 | 0.6692 | 0.6023 | 0.7275 | 0.2586 | 0.7414 | 0.2725 |
| 3 | at_p | 3468 | 3267 | 0.7630 | 0.7845 | 0.1750 | 0.0538 | 0.7712 | 0.8603 | 0.7415 | 0.7073 | 0.7174 | 0.6243 | 0.7073 | 0.2585 | 0.7415 | 0.2927 |

## Probe populations (eligible rows per probe kind; 260923's class and row filters)

| kind | split | class | condition | n rows | n prompts |
|---|---|---|---|---|---|
| bias | train | bias | cue_gold | 647 | 313 |
| bias | train | bias | cue_dist | 1104 | 632 |
| bias | train | no bias | cue_gold | 3608 | 1410 |
| bias | train | no bias | cue_dist | 3239 | 1384 |
| bias | eval | bias | cue_gold | 299 | 143 |
| bias | eval | bias | cue_dist | 445 | 258 |
| bias | eval | no bias | cue_gold | 1509 | 600 |
| bias | eval | no bias | cue_dist | 1392 | 596 |
| bias_followed | train | bias | cue_gold | 647 | 313 |
| bias_followed | train | bias | cue_dist | 1104 | 632 |
| bias_followed | train | no bias | cue_gold | 3100 | 1146 |
| bias_followed | train | no bias | cue_dist | 967 | 380 |
| bias_followed | eval | bias | cue_gold | 299 | 143 |
| bias_followed | eval | bias | cue_dist | 445 | 258 |
| bias_followed | eval | no bias | cue_gold | 1265 | 476 |
| bias_followed | eval | no bias | cue_dist | 451 | 181 |
| deception | train | deceptive | cue_gold | 3012 | 1320 |
| deception | train | deceptive | cue_dist | 2822 | 1324 |
| deception | train | honest | cue_gold | 1243 | 663 |
| deception | train | honest | cue_dist | 1521 | 866 |
| deception | eval | deceptive | cue_gold | 1301 | 571 |
| deception | eval | deceptive | cue_dist | 1238 | 569 |
| deception | eval | honest | cue_gold | 507 | 268 |
| deception | eval | honest | cue_dist | 599 | 342 |
| deception_yes | train | false admission | cue_gold | 2912 | 1266 |
| deception_yes | train | false admission | cue_dist | 2649 | 1244 |
| deception_yes | train | true admission | cue_gold | 547 | 284 |
| deception_yes | train | true admission | cue_dist | 931 | 566 |
| deception_yes | eval | false admission | cue_gold | 1256 | 548 |
| deception_yes | eval | false admission | cue_dist | 1186 | 546 |
| deception_yes | eval | true admission | cue_gold | 254 | 130 |
| deception_yes | eval | true admission | cue_dist | 393 | 237 |
| deception_no | train | false denial | cue_gold | 100 | 71 |
| deception_no | train | false denial | cue_dist | 173 | 126 |
| deception_no | train | true denial | cue_gold | 696 | 398 |
| deception_no | train | true denial | cue_dist | 590 | 349 |
| deception_no | eval | false denial | cue_gold | 45 | 31 |
| deception_no | eval | false denial | cue_dist | 52 | 37 |
| deception_no | eval | true denial | cue_gold | 253 | 144 |
| deception_no | eval | true denial | cue_dist | 206 | 125 |
