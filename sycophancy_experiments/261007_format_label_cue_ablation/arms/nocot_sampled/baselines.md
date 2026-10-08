# Baselines: `nocot_sampled`

spec 10b44b32b4c2; 13740 cued rows over 2290 questions; labels from sampled; at-p mixes cue_gold and cue_dist at p = 0.85 (numerator and denominator weighted separately, 260824's `at_p`). Accuracy counts unparsed as wrong; every other rate is over parsed rows. `n` is the unweighted denominator.

## Split: eval (4122 cued rows, 687 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6929 (n=2061) | 0.9296 (n=2061) | 0.3086 (n=2061) | 0.8365 (n=4122) |
| accuracy | gold is A | 0.6478 (n=1008) | 0.9226 (n=1008) | 0.3095 (n=1008) | 0.8307 (n=2016) |
| accuracy | gold is B | 0.7360 (n=1053) | 0.9364 (n=1053) | 0.3077 (n=1053) | 0.8421 (n=2106) |
| parse-fail rate | 0.0102 (n=2061) | 0.0010 (n=2061) | 0.0029 (n=2061) | 0.0013 (n=4122) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9301 (n=2047) | 0.6903 (n=2044) | 0.8942 (n=4091) |
| bias | 0.2413 (n=2047) | 0.4031 (n=2044) | 0.2656 (n=4091) |
| anti-bias | 0.0103 (n=2047) | 0.0152 (n=2044) | 0.0110 (n=4091) |
| no bias | 0.7484 (n=2047) | 0.5817 (n=2044) | 0.7234 (n=4091) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.0814 (n=4064) | 0.0814 (n=4064) | 0.0814 (n=4064) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7973 (n=2047) | 0.8156 (n=2044) | 0.8000 (n=4091) |
| said Yes (sampled token) | 0.7157 (n=2047) | 0.7745 (n=2044) | 0.7245 (n=4091) |
| mean P(Yes) | bias | 0.7550 (n=494) | 0.7278 (n=824) | 0.7488 (n=1318) |
| mean P(Yes) | anti-bias | 0.8189 (n=21) | 0.7908 (n=31) | 0.8131 (n=52) |
| mean P(Yes) | no bias | 0.7049 (n=1532) | 0.7698 (n=1189) | 0.7127 (n=2721) |
| deception (count) | 0.6422 (n=2026) | 0.6080 (n=2013) | 0.6371 (n=4039) |
| false admission (count) | 0.7898 (n=1532) | 0.8570 (n=1189) | 0.7979 (n=2721) |
| false denial (count) | 0.1842 (n=494) | 0.2488 (n=824) | 0.1989 (n=1318) |
| true admission (count) | 0.8158 (n=494) | 0.7512 (n=824) | 0.8011 (n=1318) |
| true denial (count) | 0.2102 (n=1532) | 0.1430 (n=1189) | 0.2021 (n=2721) |
| deception (P(Yes)-weighted) | 0.5928 (n=2026) | 0.5661 (n=2013) | 0.5888 (n=4039) |
| false admission (P) | 0.7049 (n=1532) | 0.7698 (n=1189) | 0.7127 (n=2721) |
| false denial (P) | 0.2450 (n=494) | 0.2722 (n=824) | 0.2512 (n=1318) |
| true admission (P) | 0.7550 (n=494) | 0.7278 (n=824) | 0.7488 (n=1318) |
| true denial (P) | 0.2951 (n=1532) | 0.2302 (n=1189) | 0.2873 (n=2721) |

## Split: train (9618 cued rows, 1603 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6999 (n=4809) | 0.9308 (n=4809) | 0.3063 (n=4809) | 0.8371 (n=9618) |
| accuracy | gold is A | 0.6803 (n=2355) | 0.9265 (n=2355) | 0.2938 (n=2355) | 0.8316 (n=4710) |
| accuracy | gold is B | 0.7188 (n=2454) | 0.9348 (n=2454) | 0.3183 (n=2454) | 0.8423 (n=4908) |
| parse-fail rate | 0.0035 (n=4809) | 0.0008 (n=4809) | 0.0006 (n=4809) | 0.0008 (n=9618) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9313 (n=4787) | 0.6924 (n=4788) | 0.8954 (n=9575) |
| bias | 0.2275 (n=4787) | 0.4137 (n=4788) | 0.2554 (n=9575) |
| anti-bias | 0.0079 (n=4787) | 0.0100 (n=4788) | 0.0083 (n=9575) |
| no bias | 0.7646 (n=4787) | 0.5762 (n=4788) | 0.7363 (n=9575) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.0724 (n=9556) | 0.0724 (n=9556) | 0.0724 (n=9556) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7733 (n=4787) | 0.8131 (n=4788) | 0.7793 (n=9575) |
| said Yes (sampled token) | 0.7046 (n=4787) | 0.7356 (n=4788) | 0.7093 (n=9575) |
| mean P(Yes) | bias | 0.7364 (n=1089) | 0.7368 (n=1981) | 0.7365 (n=3070) |
| mean P(Yes) | anti-bias | 0.7344 (n=38) | 0.7983 (n=48) | 0.7460 (n=86) |
| mean P(Yes) | no bias | 0.6878 (n=3660) | 0.7524 (n=2759) | 0.6954 (n=6419) |
| deception (count) | 0.6288 (n=4749) | 0.5719 (n=4740) | 0.6203 (n=9489) |
| false admission (count) | 0.7604 (n=3660) | 0.8300 (n=2759) | 0.7686 (n=6419) |
| false denial (count) | 0.1864 (n=1089) | 0.2125 (n=1981) | 0.1928 (n=3070) |
| true admission (count) | 0.8136 (n=1089) | 0.7875 (n=1981) | 0.8072 (n=3070) |
| true denial (count) | 0.2396 (n=3660) | 0.1700 (n=2759) | 0.2314 (n=6419) |
| deception (P(Yes)-weighted) | 0.5905 (n=4749) | 0.5479 (n=4740) | 0.5841 (n=9489) |
| false admission (P) | 0.6878 (n=3660) | 0.7524 (n=2759) | 0.6954 (n=6419) |
| false denial (P) | 0.2636 (n=1089) | 0.2632 (n=1981) | 0.2635 (n=3070) |
| true admission (P) | 0.7364 (n=1089) | 0.7368 (n=1981) | 0.7365 (n=3070) |
| true denial (P) | 0.3122 (n=3660) | 0.2476 (n=2759) | 0.3046 (n=6419) |

## Split: all (13740 cued rows, 2290 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6978 (n=6870) | 0.9304 (n=6870) | 0.3070 (n=6870) | 0.8369 (n=13740) |
| accuracy | gold is A | 0.6705 (n=3363) | 0.9254 (n=3363) | 0.2985 (n=3363) | 0.8313 (n=6726) |
| accuracy | gold is B | 0.7240 (n=3507) | 0.9353 (n=3507) | 0.3151 (n=3507) | 0.8422 (n=7014) |
| parse-fail rate | 0.0055 (n=6870) | 0.0009 (n=6870) | 0.0013 (n=6870) | 0.0009 (n=13740) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.9309 (n=6834) | 0.6917 (n=6832) | 0.8951 (n=13666) |
| bias | 0.2316 (n=6834) | 0.4106 (n=6832) | 0.2585 (n=13666) |
| anti-bias | 0.0086 (n=6834) | 0.0116 (n=6832) | 0.0091 (n=13666) |
| no bias | 0.7597 (n=6834) | 0.5779 (n=6832) | 0.7325 (n=13666) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.0751 (n=13620) | 0.0751 (n=13620) | 0.0751 (n=13620) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.7805 (n=6834) | 0.8138 (n=6832) | 0.7855 (n=13666) |
| said Yes (sampled token) | 0.7079 (n=6834) | 0.7472 (n=6832) | 0.7138 (n=13666) |
| mean P(Yes) | bias | 0.7422 (n=1583) | 0.7341 (n=2805) | 0.7403 (n=4388) |
| mean P(Yes) | anti-bias | 0.7644 (n=59) | 0.7954 (n=79) | 0.7704 (n=138) |
| mean P(Yes) | no bias | 0.6928 (n=5192) | 0.7576 (n=3948) | 0.7005 (n=9140) |
| deception (count) | 0.6328 (n=6775) | 0.5827 (n=6753) | 0.6253 (n=13528) |
| false admission (count) | 0.7691 (n=5192) | 0.8381 (n=3948) | 0.7772 (n=9140) |
| false denial (count) | 0.1857 (n=1583) | 0.2232 (n=2805) | 0.1946 (n=4388) |
| true admission (count) | 0.8143 (n=1583) | 0.7768 (n=2805) | 0.8054 (n=4388) |
| true denial (count) | 0.2309 (n=5192) | 0.1619 (n=3948) | 0.2228 (n=9140) |
| deception (P(Yes)-weighted) | 0.5912 (n=6775) | 0.5534 (n=6753) | 0.5855 (n=13528) |
| false admission (P) | 0.6928 (n=5192) | 0.7576 (n=3948) | 0.7005 (n=9140) |
| false denial (P) | 0.2578 (n=1583) | 0.2659 (n=2805) | 0.2597 (n=4388) |
| true admission (P) | 0.7422 (n=1583) | 0.7341 (n=2805) | 0.7403 (n=4388) |
| true denial (P) | 0.3072 (n=5192) | 0.2424 (n=3948) | 0.2995 (n=9140) |

## By category (all rows)

| category | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| computer science | cue_gold | 1230 | 1230 | 0.9537 | 0.9537 | 0.2732 | 0.0024 | 0.7244 | 0.8041 | 0.7714 | 0.6958 | 0.5681 | 0.5679 | 0.6958 | 0.2286 | 0.7714 | 0.3042 |
| computer science | cue_dist | 1230 | 1230 | 0.2545 | 0.7455 | 0.4415 | 0.0130 | 0.5455 | 0.8415 | 0.7515 | 0.7529 | 0.5494 | 0.5273 | 0.7529 | 0.2485 | 0.7515 | 0.2471 |
| computer science | at_p | 2460 | 2460 | 0.8488 | 0.9224 | 0.2984 | 0.0040 | 0.6976 | 0.8097 | 0.7670 | 0.7025 | 0.5653 | 0.5619 | 0.7025 | 0.2330 | 0.7670 | 0.2975 |
| history | cue_gold | 1143 | 1133 | 0.9256 | 0.9285 | 0.1244 | 0.0106 | 0.8650 | 0.8782 | 0.8081 | 0.7231 | 0.7779 | 0.6562 | 0.7231 | 0.1919 | 0.8081 | 0.2769 |
| history | cue_dist | 1143 | 1135 | 0.4996 | 0.4996 | 0.3189 | 0.0062 | 0.6749 | 0.9260 | 0.7839 | 0.8143 | 0.6826 | 0.6223 | 0.8143 | 0.2161 | 0.7839 | 0.1857 |
| history | at_p | 2286 | 2268 | 0.8617 | 0.8641 | 0.1537 | 0.0099 | 0.8364 | 0.8854 | 0.8006 | 0.7341 | 0.7635 | 0.6511 | 0.7341 | 0.1994 | 0.8006 | 0.2659 |
| law | cue_gold | 1500 | 1500 | 0.8727 | 0.8727 | 0.2253 | 0.0207 | 0.7540 | 0.7533 | 0.7402 | 0.7114 | 0.6440 | 0.6075 | 0.7114 | 0.2598 | 0.7402 | 0.2886 |
| law | cue_dist | 1500 | 1500 | 0.3567 | 0.6433 | 0.3387 | 0.0273 | 0.6340 | 0.7880 | 0.7168 | 0.7603 | 0.6306 | 0.5942 | 0.7603 | 0.2832 | 0.7168 | 0.2397 |
| law | at_p | 3000 | 3000 | 0.7953 | 0.8383 | 0.2423 | 0.0217 | 0.7360 | 0.7585 | 0.7353 | 0.7177 | 0.6420 | 0.6055 | 0.7177 | 0.2647 | 0.7353 | 0.2823 |
| philosophy | cue_gold | 1497 | 1480 | 0.9292 | 0.9297 | 0.2142 | 0.0061 | 0.7797 | 0.6628 | 0.7351 | 0.6382 | 0.5506 | 0.5577 | 0.6382 | 0.2649 | 0.7351 | 0.3618 |
| philosophy | cue_dist | 1497 | 1476 | 0.3307 | 0.6646 | 0.3902 | 0.0061 | 0.6037 | 0.7168 | 0.6910 | 0.7215 | 0.5890 | 0.5596 | 0.7215 | 0.3090 | 0.6910 | 0.2785 |
| philosophy | at_p | 2994 | 2956 | 0.8394 | 0.8901 | 0.2405 | 0.0061 | 0.7534 | 0.6709 | 0.7244 | 0.6482 | 0.5564 | 0.5580 | 0.6482 | 0.2756 | 0.7244 | 0.3518 |
| physics | cue_gold | 1500 | 1491 | 0.9740 | 0.9738 | 0.3025 | 0.0027 | 0.6948 | 0.8310 | 0.7064 | 0.7023 | 0.6469 | 0.5783 | 0.7023 | 0.2936 | 0.7064 | 0.2977 |
| physics | cue_dist | 1500 | 1491 | 0.1300 | 0.8692 | 0.5473 | 0.0040 | 0.4487 | 0.8276 | 0.7418 | 0.7417 | 0.4808 | 0.4760 | 0.7417 | 0.2582 | 0.7418 | 0.2583 |
| physics | at_p | 3000 | 2982 | 0.8474 | 0.9581 | 0.3392 | 0.0029 | 0.6579 | 0.8305 | 0.7150 | 0.7063 | 0.6220 | 0.5630 | 0.7063 | 0.2850 | 0.7150 | 0.2937 |

## By cue_template_idx (all rows)

| cue_template_idx | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cue_gold | 822 | 819 | 0.9392 | 0.9389 | 0.2222 | 0.0049 | 0.7729 | 0.8449 | 0.7906 | 0.7096 | 0.6810 | 0.5979 | 0.7096 | 0.2094 | 0.7906 | 0.2904 |
| 0 | cue_dist | 822 | 816 | 0.2981 | 0.6998 | 0.4326 | 0.0110 | 0.5564 | 0.8444 | 0.7527 | 0.7740 | 0.5390 | 0.5436 | 0.7740 | 0.2473 | 0.7527 | 0.2260 |
| 0 | at_p | 1644 | 1635 | 0.8430 | 0.9032 | 0.2537 | 0.0058 | 0.7405 | 0.8448 | 0.7809 | 0.7168 | 0.6599 | 0.5898 | 0.7168 | 0.2191 | 0.7809 | 0.2832 |
| 1 | cue_gold | 873 | 870 | 0.9026 | 0.9023 | 0.1920 | 0.0138 | 0.7943 | 0.9276 | 0.7956 | 0.7749 | 0.7809 | 0.6639 | 0.7749 | 0.2044 | 0.7956 | 0.2251 |
| 1 | cue_dist | 873 | 870 | 0.3288 | 0.6701 | 0.4034 | 0.0092 | 0.5874 | 0.9414 | 0.7895 | 0.8174 | 0.5905 | 0.5703 | 0.8174 | 0.2105 | 0.7895 | 0.1826 |
| 1 | at_p | 1746 | 1740 | 0.8166 | 0.8675 | 0.2237 | 0.0131 | 0.7632 | 0.9297 | 0.7940 | 0.7798 | 0.7522 | 0.6498 | 0.7798 | 0.2060 | 0.7940 | 0.2202 |
| 2 | cue_gold | 786 | 784 | 0.9377 | 0.9401 | 0.2245 | 0.0064 | 0.7691 | 0.7360 | 0.7256 | 0.6670 | 0.6110 | 0.5783 | 0.6670 | 0.2744 | 0.7256 | 0.3330 |
| 2 | cue_dist | 786 | 786 | 0.3015 | 0.6985 | 0.4275 | 0.0076 | 0.5649 | 0.7761 | 0.7198 | 0.7238 | 0.5513 | 0.5327 | 0.7238 | 0.2802 | 0.7198 | 0.2762 |
| 2 | at_p | 1572 | 1570 | 0.8422 | 0.9037 | 0.2550 | 0.0066 | 0.7384 | 0.7420 | 0.7241 | 0.6735 | 0.6021 | 0.5715 | 0.6735 | 0.2759 | 0.7241 | 0.3265 |
| 3 | cue_gold | 912 | 909 | 0.8783 | 0.8779 | 0.1969 | 0.0187 | 0.7844 | 0.7921 | 0.7618 | 0.7106 | 0.6547 | 0.6158 | 0.7106 | 0.2382 | 0.7618 | 0.2894 |
| 3 | cue_dist | 912 | 909 | 0.4507 | 0.5479 | 0.2783 | 0.0308 | 0.6909 | 0.8603 | 0.7378 | 0.7809 | 0.7151 | 0.6320 | 0.7809 | 0.2622 | 0.7378 | 0.2191 |
| 3 | at_p | 1824 | 1818 | 0.8141 | 0.8284 | 0.2091 | 0.0205 | 0.7704 | 0.8023 | 0.7570 | 0.7201 | 0.6637 | 0.6182 | 0.7201 | 0.2430 | 0.7570 | 0.2799 |
| 4 | cue_gold | 855 | 846 | 0.9883 | 0.9882 | 0.2825 | 0.0000 | 0.7175 | 0.8617 | 0.7345 | 0.7499 | 0.6785 | 0.6130 | 0.7499 | 0.2655 | 0.7345 | 0.2501 |
| 4 | cue_dist | 855 | 846 | 0.1977 | 0.8038 | 0.5130 | 0.0035 | 0.4835 | 0.8972 | 0.7861 | 0.7964 | 0.5053 | 0.4965 | 0.7964 | 0.2139 | 0.7861 | 0.2036 |
| 4 | at_p | 1710 | 1692 | 0.8697 | 0.9605 | 0.3171 | 0.0005 | 0.6824 | 0.8670 | 0.7470 | 0.7548 | 0.6526 | 0.5956 | 0.7548 | 0.2530 | 0.7470 | 0.2452 |
| 5 | cue_gold | 903 | 902 | 0.8749 | 0.8758 | 0.2561 | 0.0188 | 0.7251 | 0.8337 | 0.7692 | 0.6937 | 0.6045 | 0.5728 | 0.6937 | 0.2308 | 0.7692 | 0.3063 |
| 5 | cue_dist | 903 | 902 | 0.4075 | 0.5920 | 0.2550 | 0.0255 | 0.7195 | 0.7738 | 0.6839 | 0.7422 | 0.6724 | 0.6307 | 0.7422 | 0.3161 | 0.6839 | 0.2578 |
| 5 | at_p | 1806 | 1804 | 0.8048 | 0.8333 | 0.2559 | 0.0198 | 0.7242 | 0.8247 | 0.7565 | 0.7009 | 0.6146 | 0.5815 | 0.7009 | 0.2435 | 0.7565 | 0.2991 |
| 6 | cue_gold | 891 | 878 | 0.9607 | 0.9613 | 0.2620 | 0.0046 | 0.7335 | 0.6367 | 0.6949 | 0.6354 | 0.5332 | 0.5485 | 0.6354 | 0.3051 | 0.6949 | 0.3646 |
| 6 | cue_dist | 891 | 879 | 0.2290 | 0.7679 | 0.4721 | 0.0011 | 0.5267 | 0.6883 | 0.6891 | 0.6899 | 0.5239 | 0.5108 | 0.6899 | 0.3109 | 0.6891 | 0.3101 |
| 6 | at_p | 1782 | 1757 | 0.8510 | 0.9322 | 0.2935 | 0.0040 | 0.7024 | 0.6444 | 0.6935 | 0.6415 | 0.5318 | 0.5428 | 0.6415 | 0.3065 | 0.6935 | 0.3585 |
| 7 | cue_gold | 828 | 826 | 0.9698 | 0.9722 | 0.2167 | 0.0000 | 0.7833 | 0.6029 | 0.6763 | 0.5961 | 0.5169 | 0.5371 | 0.5961 | 0.3237 | 0.6763 | 0.4039 |
| 7 | cue_dist | 828 | 824 | 0.2271 | 0.7718 | 0.5255 | 0.0012 | 0.4733 | 0.7257 | 0.7008 | 0.7264 | 0.5516 | 0.5016 | 0.7264 | 0.2992 | 0.7008 | 0.2736 |
| 7 | at_p | 1656 | 1650 | 0.8584 | 0.9422 | 0.2629 | 0.0002 | 0.7369 | 0.6213 | 0.6836 | 0.6086 | 0.5221 | 0.5318 | 0.6086 | 0.3164 | 0.6836 | 0.3914 |

## Probe populations (eligible rows per probe kind; 260923's class and row filters)

| kind | split | class | condition | n rows | n prompts |
|---|---|---|---|---|---|
| bias | train | bias | cue_gold | 1089 | 396 |
| bias | train | bias | cue_dist | 1981 | 733 |
| bias | train | no bias | cue_gold | 3660 | 1261 |
| bias | train | no bias | cue_dist | 2759 | 1005 |
| bias | eval | bias | cue_gold | 494 | 177 |
| bias | eval | bias | cue_dist | 824 | 308 |
| bias | eval | no bias | cue_gold | 1532 | 528 |
| bias | eval | no bias | cue_dist | 1189 | 434 |
| bias_followed | train | bias | cue_gold | 1089 | 396 |
| bias_followed | train | bias | cue_dist | 1981 | 733 |
| bias_followed | train | no bias | cue_gold | 3369 | 1128 |
| bias_followed | train | no bias | cue_dist | 1334 | 454 |
| bias_followed | eval | bias | cue_gold | 494 | 177 |
| bias_followed | eval | bias | cue_dist | 824 | 308 |
| bias_followed | eval | no bias | cue_gold | 1410 | 474 |
| bias_followed | eval | no bias | cue_dist | 587 | 201 |
| deception | train | deceptive | cue_gold | 2986 | 1032 |
| deception | train | deceptive | cue_dist | 2711 | 978 |
| deception | train | honest | cue_gold | 1763 | 621 |
| deception | train | honest | cue_dist | 2029 | 742 |
| deception | eval | deceptive | cue_gold | 1301 | 449 |
| deception | eval | deceptive | cue_dist | 1224 | 441 |
| deception | eval | honest | cue_gold | 725 | 254 |
| deception | eval | honest | cue_dist | 789 | 293 |
| deception_yes | train | false admission | cue_gold | 2783 | 965 |
| deception_yes | train | false admission | cue_dist | 2290 | 839 |
| deception_yes | train | true admission | cue_gold | 886 | 324 |
| deception_yes | train | true admission | cue_dist | 1560 | 576 |
| deception_yes | eval | false admission | cue_gold | 1210 | 420 |
| deception_yes | eval | false admission | cue_dist | 1019 | 373 |
| deception_yes | eval | true admission | cue_gold | 403 | 146 |
| deception_yes | eval | true admission | cue_dist | 619 | 232 |
| deception_no | train | false denial | cue_gold | 203 | 73 |
| deception_no | train | false denial | cue_dist | 421 | 158 |
| deception_no | train | true denial | cue_gold | 877 | 298 |
| deception_no | train | true denial | cue_dist | 469 | 167 |
| deception_no | eval | false denial | cue_gold | 91 | 31 |
| deception_no | eval | false denial | cue_dist | 205 | 76 |
| deception_no | eval | true denial | cue_gold | 322 | 108 |
| deception_no | eval | true denial | cue_dist | 170 | 61 |
