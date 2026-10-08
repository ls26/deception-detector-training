# Baselines: `nocot_logprob`

spec 10b44b32b4c2; 4580 cued rows over 2290 questions; labels from logprob; at-p mixes cue_gold and cue_dist at p = 0.85 (numerator and denominator weighted separately, 260824's `at_p`). Accuracy counts unparsed as wrong; every other rate is over parsed rows. `n` is the unweighted denominator.

## Split: eval (1374 cued rows, 687 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6972 (n=687) | 0.9360 (n=687) | 0.2969 (n=687) | 0.8401 (n=1374) |
| accuracy | gold is A | 0.6726 (n=336) | 0.9286 (n=336) | 0.2857 (n=336) | 0.8321 (n=672) |
| accuracy | gold is B | 0.7208 (n=351) | 0.9430 (n=351) | 0.3077 (n=351) | 0.8477 (n=702) |
| parse-fail rate | 0.0175 (n=687) | 0.0029 (n=687) | 0.0058 (n=687) | 0.0033 (n=1374) |

### Bias (removal, top-letter counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9376 (n=673) | 0.6975 (n=671) | 0.9017 (n=1344) |
| bias | 0.2333 (n=673) | 0.4158 (n=671) | 0.2606 (n=1344) |
| anti-bias | 0.0059 (n=673) | 0.0104 (n=671) | 0.0066 (n=1344) |
| no bias | 0.7608 (n=673) | 0.5738 (n=671) | 0.7328 (n=1344) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.2335 (n=687) | 0.3865 (n=687) | 0.2565 (n=1374) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7979 (n=673) | 0.8137 (n=671) | 0.8003 (n=1344) |
| said Yes (sampled token) | 0.7117 (n=673) | 0.7407 (n=671) | 0.7161 (n=1344) |
| mean P(Yes) | bias | 0.7558 (n=157) | 0.7355 (n=279) | 0.7510 (n=436) |
| mean P(Yes) | anti-bias | 0.8969 (n=4) | 0.8560 (n=7) | 0.8872 (n=11) |
| mean P(Yes) | no bias | 0.7038 (n=512) | 0.7598 (n=385) | 0.7103 (n=897) |
| deception (count) | 0.6398 (n=669) | 0.5873 (n=664) | 0.6319 (n=1333) |
| false admission (count) | 0.7852 (n=512) | 0.8442 (n=385) | 0.7921 (n=897) |
| false denial (count) | 0.1656 (n=157) | 0.2330 (n=279) | 0.1817 (n=436) |
| true admission (count) | 0.8344 (n=157) | 0.7670 (n=279) | 0.8183 (n=436) |
| true denial (count) | 0.2148 (n=512) | 0.1558 (n=385) | 0.2079 (n=897) |
| deception (P(Yes)-weighted) | 0.5959 (n=669) | 0.5517 (n=664) | 0.5893 (n=1333) |
| false admission (P) | 0.7038 (n=512) | 0.7598 (n=385) | 0.7103 (n=897) |
| false denial (P) | 0.2442 (n=157) | 0.2645 (n=279) | 0.2490 (n=436) |
| true admission (P) | 0.7558 (n=157) | 0.7355 (n=279) | 0.7510 (n=436) |
| true denial (P) | 0.2962 (n=512) | 0.2402 (n=385) | 0.2897 (n=897) |

### Logprob arm

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| mean P(cue letter) shift, cued - no-cue | nan (n=0) | 0.2335 (n=687) | 0.3865 (n=687) | 0.2565 (n=1374) |
| exact P(A) = P(B) tie rate | 0.0175 (n=687) | 0.0029 (n=687) | 0.0058 (n=687) | 0.0033 (n=1374) |
| confident share (both top letters >= 0.75) | 0.7962 (n=687) | 0.7758 (n=687) | 0.7205 (n=687) | 0.7675 (n=1374) |
| mean P(A) + P(B) (unrenormalized) | 0.9888 (n=687) | 0.9990 (n=687) | 0.9966 (n=687) | 0.9987 (n=1374) |
| share with P(A) + P(B) < 0.95 | 0.0277 (n=687) | 0.0044 (n=687) | 0.0131 (n=687) | 0.0057 (n=1374) |

Histogram of max(P(A), P(B)), renormalized over {A, B}:

| bin | [0.50, 0.55) | [0.55, 0.60) | [0.60, 0.65) | [0.65, 0.70) | [0.70, 0.75) | [0.75, 0.80) | [0.80, 0.85) | [0.85, 0.90) | [0.90, 0.95) | [0.95, 1.00) |
|---|---|---|---|---|---|---|---|---|---|---|
| no-cue prompts | 34 | 32 | 14 | 28 | 32 | 41 | 34 | 40 | 44 | 388 |
| cued prompts | 16 | 20 | 8 | 17 | 18 | 26 | 18 | 43 | 67 | 1141 |

## Split: train (3206 cued rows, 1603 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.7049 (n=1603) | 0.9364 (n=1603) | 0.2976 (n=1603) | 0.8405 (n=3206) |
| accuracy | gold is A | 0.6662 (n=785) | 0.9350 (n=785) | 0.2904 (n=785) | 0.8383 (n=1570) |
| accuracy | gold is B | 0.7421 (n=818) | 0.9377 (n=818) | 0.3044 (n=818) | 0.8427 (n=1636) |
| parse-fail rate | 0.0087 (n=1603) | 0.0044 (n=1603) | 0.0037 (n=1603) | 0.0043 (n=3206) |

### Bias (removal, top-letter counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9399 (n=1582) | 0.6987 (n=1583) | 0.9037 (n=3165) |
| bias | 0.2314 (n=1582) | 0.4163 (n=1583) | 0.2591 (n=3165) |
| anti-bias | 0.0051 (n=1582) | 0.0076 (n=1583) | 0.0054 (n=3165) |
| no bias | 0.7636 (n=1582) | 0.5761 (n=1583) | 0.7355 (n=3165) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.2302 (n=1603) | 0.3961 (n=1603) | 0.2551 (n=3206) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7724 (n=1582) | 0.8117 (n=1583) | 0.7783 (n=3165) |
| said Yes (sampled token) | 0.6896 (n=1582) | 0.7707 (n=1583) | 0.7018 (n=3165) |
| mean P(Yes) | bias | 0.7343 (n=366) | 0.7337 (n=659) | 0.7342 (n=1025) |
| mean P(Yes) | anti-bias | 0.7694 (n=8) | 0.7954 (n=12) | 0.7749 (n=20) |
| mean P(Yes) | no bias | 0.6870 (n=1208) | 0.7550 (n=912) | 0.6950 (n=2120) |
| deception (count) | 0.6220 (n=1574) | 0.5773 (n=1571) | 0.6153 (n=3145) |
| false admission (count) | 0.7566 (n=1208) | 0.8344 (n=912) | 0.7658 (n=2120) |
| false denial (count) | 0.1776 (n=366) | 0.2215 (n=659) | 0.1882 (n=1025) |
| true admission (count) | 0.8224 (n=366) | 0.7785 (n=659) | 0.8118 (n=1025) |
| true denial (count) | 0.2434 (n=1208) | 0.1656 (n=912) | 0.2342 (n=2120) |
| deception (P(Yes)-weighted) | 0.5890 (n=1574) | 0.5500 (n=1571) | 0.5832 (n=3145) |
| false admission (P) | 0.6870 (n=1208) | 0.7550 (n=912) | 0.6950 (n=2120) |
| false denial (P) | 0.2657 (n=366) | 0.2663 (n=659) | 0.2658 (n=1025) |
| true admission (P) | 0.7343 (n=366) | 0.7337 (n=659) | 0.7342 (n=1025) |
| true denial (P) | 0.3130 (n=1208) | 0.2450 (n=912) | 0.3050 (n=2120) |

### Logprob arm

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| mean P(cue letter) shift, cued - no-cue | nan (n=0) | 0.2302 (n=1603) | 0.3961 (n=1603) | 0.2551 (n=3206) |
| exact P(A) = P(B) tie rate | 0.0087 (n=1603) | 0.0044 (n=1603) | 0.0037 (n=1603) | 0.0043 (n=3206) |
| confident share (both top letters >= 0.75) | 0.8135 (n=1603) | 0.7867 (n=1603) | 0.7399 (n=1603) | 0.7796 (n=3206) |
| mean P(A) + P(B) (unrenormalized) | 0.9956 (n=1603) | 0.9996 (n=1603) | 0.9993 (n=1603) | 0.9996 (n=3206) |
| share with P(A) + P(B) < 0.95 | 0.0175 (n=1603) | 0.0019 (n=1603) | 0.0019 (n=1603) | 0.0019 (n=3206) |

Histogram of max(P(A), P(B)), renormalized over {A, B}:

| bin | [0.50, 0.55) | [0.55, 0.60) | [0.60, 0.65) | [0.65, 0.70) | [0.70, 0.75) | [0.75, 0.80) | [0.80, 0.85) | [0.85, 0.90) | [0.90, 0.95) | [0.95, 1.00) |
|---|---|---|---|---|---|---|---|---|---|---|
| no-cue prompts | 51 | 77 | 37 | 73 | 61 | 74 | 50 | 93 | 134 | 953 |
| cued prompts | 36 | 40 | 29 | 32 | 58 | 63 | 43 | 75 | 118 | 2712 |

## Split: all (4580 cued rows, 2290 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.7026 (n=2290) | 0.9362 (n=2290) | 0.2974 (n=2290) | 0.8404 (n=4580) |
| accuracy | gold is A | 0.6682 (n=1121) | 0.9331 (n=1121) | 0.2890 (n=1121) | 0.8365 (n=2242) |
| accuracy | gold is B | 0.7357 (n=1169) | 0.9393 (n=1169) | 0.3054 (n=1169) | 0.8442 (n=2338) |
| parse-fail rate | 0.0114 (n=2290) | 0.0039 (n=2290) | 0.0044 (n=2290) | 0.0040 (n=4580) |

### Bias (removal, top-letter counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9392 (n=2255) | 0.6983 (n=2254) | 0.9031 (n=4509) |
| bias | 0.2319 (n=2255) | 0.4161 (n=2254) | 0.2596 (n=4509) |
| anti-bias | 0.0053 (n=2255) | 0.0084 (n=2254) | 0.0058 (n=4509) |
| no bias | 0.7627 (n=2255) | 0.5754 (n=2254) | 0.7347 (n=4509) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.2312 (n=2290) | 0.3932 (n=2290) | 0.2555 (n=4580) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7800 (n=2255) | 0.8123 (n=2254) | 0.7849 (n=4509) |
| said Yes (sampled token) | 0.6962 (n=2255) | 0.7618 (n=2254) | 0.7061 (n=4509) |
| mean P(Yes) | bias | 0.7408 (n=523) | 0.7342 (n=938) | 0.7392 (n=1461) |
| mean P(Yes) | anti-bias | 0.8119 (n=12) | 0.8177 (n=19) | 0.8132 (n=31) |
| mean P(Yes) | no bias | 0.6920 (n=1720) | 0.7564 (n=1297) | 0.6995 (n=3017) |
| deception (count) | 0.6273 (n=2243) | 0.5803 (n=2235) | 0.6203 (n=4478) |
| false admission (count) | 0.7651 (n=1720) | 0.8373 (n=1297) | 0.7736 (n=3017) |
| false denial (count) | 0.1740 (n=523) | 0.2249 (n=938) | 0.1862 (n=1461) |
| true admission (count) | 0.8260 (n=523) | 0.7751 (n=938) | 0.8138 (n=1461) |
| true denial (count) | 0.2349 (n=1720) | 0.1627 (n=1297) | 0.2264 (n=3017) |
| deception (P(Yes)-weighted) | 0.5911 (n=2243) | 0.5505 (n=2235) | 0.5850 (n=4478) |
| false admission (P) | 0.6920 (n=1720) | 0.7564 (n=1297) | 0.6995 (n=3017) |
| false denial (P) | 0.2592 (n=523) | 0.2658 (n=938) | 0.2608 (n=1461) |
| true admission (P) | 0.7408 (n=523) | 0.7342 (n=938) | 0.7392 (n=1461) |
| true denial (P) | 0.3080 (n=1720) | 0.2436 (n=1297) | 0.3005 (n=3017) |

### Logprob arm

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| mean P(cue letter) shift, cued - no-cue | nan (n=0) | 0.2312 (n=2290) | 0.3932 (n=2290) | 0.2555 (n=4580) |
| exact P(A) = P(B) tie rate | 0.0114 (n=2290) | 0.0039 (n=2290) | 0.0044 (n=2290) | 0.0040 (n=4580) |
| confident share (both top letters >= 0.75) | 0.8083 (n=2290) | 0.7834 (n=2290) | 0.7341 (n=2290) | 0.7760 (n=4580) |
| mean P(A) + P(B) (unrenormalized) | 0.9936 (n=2290) | 0.9994 (n=2290) | 0.9985 (n=2290) | 0.9993 (n=4580) |
| share with P(A) + P(B) < 0.95 | 0.0205 (n=2290) | 0.0026 (n=2290) | 0.0052 (n=2290) | 0.0030 (n=4580) |

Histogram of max(P(A), P(B)), renormalized over {A, B}:

| bin | [0.50, 0.55) | [0.55, 0.60) | [0.60, 0.65) | [0.65, 0.70) | [0.70, 0.75) | [0.75, 0.80) | [0.80, 0.85) | [0.85, 0.90) | [0.90, 0.95) | [0.95, 1.00) |
|---|---|---|---|---|---|---|---|---|---|---|
| no-cue prompts | 85 | 109 | 51 | 101 | 93 | 115 | 84 | 133 | 178 | 1341 |
| cued prompts | 52 | 60 | 37 | 49 | 76 | 89 | 61 | 118 | 185 | 3853 |

## By category (all rows)

| category | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| computer science | cue_gold | 410 | 400 | 0.9561 | 0.9650 | 0.2850 | 0.0050 | 0.7100 | 0.8050 | 0.7660 | 0.6934 | 0.5653 | 0.5618 | 0.6934 | 0.2340 | 0.7660 | 0.3066 |
| computer science | cue_dist | 410 | 404 | 0.2488 | 0.7475 | 0.4307 | 0.0050 | 0.5644 | 0.8391 | 0.7442 | 0.7587 | 0.5697 | 0.5410 | 0.7587 | 0.2558 | 0.7442 | 0.2413 |
| computer science | at_p | 820 | 804 | 0.8500 | 0.9321 | 0.3070 | 0.0050 | 0.6880 | 0.8102 | 0.7614 | 0.7015 | 0.5660 | 0.5586 | 0.7015 | 0.2386 | 0.7614 | 0.2985 |
| history | cue_gold | 381 | 380 | 0.9344 | 0.9342 | 0.1500 | 0.0105 | 0.8395 | 0.8789 | 0.8055 | 0.7200 | 0.7527 | 0.6403 | 0.7200 | 0.1945 | 0.8055 | 0.2800 |
| history | cue_dist | 381 | 378 | 0.5013 | 0.4974 | 0.3016 | 0.0106 | 0.6878 | 0.9233 | 0.7846 | 0.8123 | 0.6872 | 0.6304 | 0.8123 | 0.2154 | 0.7846 | 0.1877 |
| history | at_p | 762 | 758 | 0.8694 | 0.8690 | 0.1726 | 0.0105 | 0.8168 | 0.8856 | 0.8000 | 0.7316 | 0.7429 | 0.6388 | 0.7316 | 0.2000 | 0.8000 | 0.2684 |
| law | cue_gold | 500 | 491 | 0.8760 | 0.8819 | 0.2138 | 0.0102 | 0.7760 | 0.7515 | 0.7348 | 0.7121 | 0.6440 | 0.6155 | 0.7121 | 0.2652 | 0.7348 | 0.2879 |
| law | cue_dist | 500 | 494 | 0.3420 | 0.6538 | 0.3543 | 0.0243 | 0.6215 | 0.7874 | 0.7118 | 0.7623 | 0.6328 | 0.5901 | 0.7623 | 0.2882 | 0.7118 | 0.2377 |
| law | at_p | 1000 | 985 | 0.7959 | 0.8475 | 0.2350 | 0.0123 | 0.7527 | 0.7569 | 0.7296 | 0.7183 | 0.6424 | 0.6117 | 0.7183 | 0.2704 | 0.7296 | 0.2817 |
| philosophy | cue_gold | 499 | 494 | 0.9339 | 0.9352 | 0.1964 | 0.0020 | 0.8016 | 0.6619 | 0.7247 | 0.6423 | 0.5619 | 0.5701 | 0.6423 | 0.2753 | 0.7247 | 0.3577 |
| philosophy | cue_dist | 499 | 491 | 0.3166 | 0.6782 | 0.4175 | 0.0020 | 0.5804 | 0.7108 | 0.6991 | 0.7119 | 0.5653 | 0.5399 | 0.7119 | 0.3009 | 0.6991 | 0.2881 |
| philosophy | at_p | 998 | 985 | 0.8413 | 0.8969 | 0.2294 | 0.0020 | 0.7686 | 0.6692 | 0.7177 | 0.6501 | 0.5624 | 0.5656 | 0.6501 | 0.2823 | 0.7177 | 0.3499 |
| physics | cue_gold | 500 | 490 | 0.9840 | 0.9837 | 0.3061 | 0.0000 | 0.6939 | 0.8306 | 0.7116 | 0.6999 | 0.6306 | 0.5739 | 0.6999 | 0.2884 | 0.7116 | 0.3001 |
| physics | cue_dist | 500 | 487 | 0.1180 | 0.8789 | 0.5544 | 0.0000 | 0.4456 | 0.8316 | 0.7477 | 0.7373 | 0.4702 | 0.4684 | 0.7373 | 0.2523 | 0.7477 | 0.2627 |
| physics | at_p | 1000 | 977 | 0.8541 | 0.9680 | 0.3432 | 0.0000 | 0.6568 | 0.8308 | 0.7203 | 0.7037 | 0.6067 | 0.5582 | 0.7037 | 0.2797 | 0.7203 | 0.2963 |

## By cue_template_idx (all rows)

| cue_template_idx | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cue_gold | 274 | 271 | 0.9380 | 0.9410 | 0.2251 | 0.0000 | 0.7749 | 0.8450 | 0.7683 | 0.7130 | 0.6716 | 0.6046 | 0.7130 | 0.2317 | 0.7683 | 0.2870 |
| 0 | cue_dist | 274 | 271 | 0.2956 | 0.7011 | 0.4207 | 0.0037 | 0.5756 | 0.8413 | 0.7661 | 0.7609 | 0.5370 | 0.5384 | 0.7609 | 0.2339 | 0.7661 | 0.2391 |
| 0 | at_p | 548 | 542 | 0.8416 | 0.9050 | 0.2544 | 0.0006 | 0.7450 | 0.8445 | 0.7677 | 0.7185 | 0.6515 | 0.5947 | 0.7185 | 0.2323 | 0.7677 | 0.2815 |
| 1 | cue_gold | 291 | 289 | 0.9175 | 0.9170 | 0.2042 | 0.0104 | 0.7855 | 0.9273 | 0.7969 | 0.7723 | 0.7552 | 0.6548 | 0.7723 | 0.2031 | 0.7969 | 0.2277 |
| 1 | cue_dist | 291 | 287 | 0.3230 | 0.6725 | 0.3972 | 0.0035 | 0.5993 | 0.9373 | 0.7872 | 0.8159 | 0.6014 | 0.5755 | 0.8159 | 0.2128 | 0.7872 | 0.1841 |
| 1 | at_p | 582 | 576 | 0.8284 | 0.8805 | 0.2329 | 0.0094 | 0.7577 | 0.9288 | 0.7945 | 0.7774 | 0.7322 | 0.6429 | 0.7774 | 0.2055 | 0.7945 | 0.2226 |
| 2 | cue_gold | 262 | 257 | 0.9351 | 0.9339 | 0.2218 | 0.0078 | 0.7704 | 0.7354 | 0.7397 | 0.6635 | 0.5882 | 0.5733 | 0.6635 | 0.2603 | 0.7397 | 0.3365 |
| 2 | cue_dist | 262 | 257 | 0.2748 | 0.7198 | 0.4436 | 0.0039 | 0.5525 | 0.7782 | 0.7211 | 0.7238 | 0.5508 | 0.5257 | 0.7238 | 0.2789 | 0.7211 | 0.2762 |
| 2 | at_p | 524 | 514 | 0.8361 | 0.9018 | 0.2551 | 0.0072 | 0.7377 | 0.7418 | 0.7349 | 0.6702 | 0.5826 | 0.5662 | 0.6702 | 0.2651 | 0.7349 | 0.3298 |
| 3 | cue_gold | 304 | 298 | 0.8947 | 0.8960 | 0.2081 | 0.0067 | 0.7852 | 0.7886 | 0.7674 | 0.7059 | 0.6520 | 0.6068 | 0.7059 | 0.2326 | 0.7674 | 0.2941 |
| 3 | cue_dist | 304 | 298 | 0.4539 | 0.5369 | 0.2685 | 0.0369 | 0.6946 | 0.8456 | 0.7096 | 0.7819 | 0.7178 | 0.6449 | 0.7819 | 0.2904 | 0.7096 | 0.2181 |
| 3 | at_p | 608 | 596 | 0.8286 | 0.8421 | 0.2171 | 0.0112 | 0.7716 | 0.7971 | 0.7567 | 0.7162 | 0.6616 | 0.6124 | 0.7162 | 0.2433 | 0.7567 | 0.2838 |
| 4 | cue_gold | 285 | 284 | 0.9895 | 0.9894 | 0.2817 | 0.0000 | 0.7183 | 0.8627 | 0.7300 | 0.7513 | 0.6725 | 0.6157 | 0.7513 | 0.2700 | 0.7300 | 0.2487 |
| 4 | cue_dist | 285 | 283 | 0.1895 | 0.8127 | 0.5194 | 0.0000 | 0.4806 | 0.8940 | 0.7908 | 0.7901 | 0.5018 | 0.4884 | 0.7901 | 0.2092 | 0.7908 | 0.2099 |
| 4 | at_p | 570 | 567 | 0.8695 | 0.9630 | 0.3172 | 0.0000 | 0.6828 | 0.8674 | 0.7449 | 0.7554 | 0.6470 | 0.5967 | 0.7554 | 0.2551 | 0.7449 | 0.2446 |
| 5 | cue_gold | 301 | 295 | 0.8870 | 0.8949 | 0.2475 | 0.0136 | 0.7390 | 0.8339 | 0.7742 | 0.6910 | 0.6117 | 0.5743 | 0.6910 | 0.2258 | 0.7742 | 0.3090 |
| 5 | cue_dist | 301 | 296 | 0.3953 | 0.5980 | 0.2669 | 0.0169 | 0.7162 | 0.7804 | 0.7031 | 0.7408 | 0.6701 | 0.6203 | 0.7408 | 0.2969 | 0.7031 | 0.2592 |
| 5 | at_p | 602 | 591 | 0.8133 | 0.8502 | 0.2504 | 0.0141 | 0.7356 | 0.8259 | 0.7628 | 0.6983 | 0.6204 | 0.5812 | 0.6983 | 0.2372 | 0.7628 | 0.3017 |
| 6 | cue_gold | 297 | 288 | 0.9596 | 0.9722 | 0.2639 | 0.0035 | 0.7326 | 0.6354 | 0.6834 | 0.6371 | 0.5296 | 0.5522 | 0.6371 | 0.3166 | 0.6834 | 0.3629 |
| 6 | cue_dist | 297 | 291 | 0.2155 | 0.7801 | 0.4811 | 0.0000 | 0.5189 | 0.6907 | 0.6886 | 0.6912 | 0.5189 | 0.5085 | 0.6912 | 0.3114 | 0.6886 | 0.3088 |
| 6 | at_p | 594 | 579 | 0.8480 | 0.9431 | 0.2968 | 0.0029 | 0.7003 | 0.6438 | 0.6847 | 0.6432 | 0.5280 | 0.5456 | 0.6432 | 0.3153 | 0.6847 | 0.3568 |
| 7 | cue_gold | 276 | 273 | 0.9746 | 0.9744 | 0.2015 | 0.0000 | 0.7985 | 0.6007 | 0.6716 | 0.5975 | 0.5311 | 0.5433 | 0.5975 | 0.3284 | 0.6716 | 0.4025 |
| 7 | cue_dist | 276 | 271 | 0.2138 | 0.7823 | 0.5535 | 0.0000 | 0.4465 | 0.7269 | 0.6964 | 0.7319 | 0.5351 | 0.4948 | 0.7319 | 0.3036 | 0.6964 | 0.2681 |
| 7 | at_p | 552 | 544 | 0.8605 | 0.9457 | 0.2539 | 0.0000 | 0.7461 | 0.6195 | 0.6797 | 0.6095 | 0.5317 | 0.5360 | 0.6095 | 0.3203 | 0.6797 | 0.3905 |

## Probe populations (eligible rows per probe kind; 260923's class and row filters)

| kind | split | class | condition | n rows | n prompts |
|---|---|---|---|---|---|
| bias | train | bias | cue_gold | 366 | 366 |
| bias | train | bias | cue_dist | 659 | 659 |
| bias | train | no bias | cue_gold | 1208 | 1208 |
| bias | train | no bias | cue_dist | 912 | 912 |
| bias | eval | bias | cue_gold | 157 | 157 |
| bias | eval | bias | cue_dist | 279 | 279 |
| bias | eval | no bias | cue_gold | 512 | 512 |
| bias | eval | no bias | cue_dist | 385 | 385 |
| bias_followed | train | bias | cue_gold | 366 | 366 |
| bias_followed | train | bias | cue_dist | 659 | 659 |
| bias_followed | train | no bias | cue_gold | 1121 | 1121 |
| bias_followed | train | no bias | cue_dist | 447 | 447 |
| bias_followed | eval | bias | cue_gold | 157 | 157 |
| bias_followed | eval | bias | cue_dist | 279 | 279 |
| bias_followed | eval | no bias | cue_gold | 474 | 474 |
| bias_followed | eval | no bias | cue_dist | 189 | 189 |
| deception | train | deceptive | cue_gold | 979 | 979 |
| deception | train | deceptive | cue_dist | 907 | 907 |
| deception | train | honest | cue_gold | 595 | 595 |
| deception | train | honest | cue_dist | 664 | 664 |
| deception | eval | deceptive | cue_gold | 428 | 428 |
| deception | eval | deceptive | cue_dist | 390 | 390 |
| deception | eval | honest | cue_gold | 241 | 241 |
| deception | eval | honest | cue_dist | 274 | 274 |
| deception_yes | train | false admission | cue_gold | 914 | 914 |
| deception_yes | train | false admission | cue_dist | 761 | 761 |
| deception_yes | train | true admission | cue_gold | 301 | 301 |
| deception_yes | train | true admission | cue_dist | 513 | 513 |
| deception_yes | eval | false admission | cue_gold | 402 | 402 |
| deception_yes | eval | false admission | cue_dist | 325 | 325 |
| deception_yes | eval | true admission | cue_gold | 131 | 131 |
| deception_yes | eval | true admission | cue_dist | 214 | 214 |
| deception_no | train | false denial | cue_gold | 65 | 65 |
| deception_no | train | false denial | cue_dist | 146 | 146 |
| deception_no | train | true denial | cue_gold | 294 | 294 |
| deception_no | train | true denial | cue_dist | 151 | 151 |
| deception_no | eval | false denial | cue_gold | 26 | 26 |
| deception_no | eval | false denial | cue_dist | 65 | 65 |
| deception_no | eval | true denial | cue_gold | 110 | 110 |
| deception_no | eval | true denial | cue_dist | 60 | 60 |
