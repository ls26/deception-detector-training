# Baselines: `ref_cot`

spec 10b44b32b4c2; 13740 cued rows over 2290 questions; labels from sampled; at-p mixes cue_gold and cue_dist at p = 0.85 (numerator and denominator weighted separately, 260824's `at_p`). Accuracy counts unparsed as wrong; every other rate is over parsed rows. `n` is the unweighted denominator.

## Split: eval (4122 cued rows, 687 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6608 (n=2061) | 0.8161 (n=2061) | 0.4639 (n=2061) | 0.7633 (n=4122) |
| accuracy | gold is A | 0.5685 (n=1008) | 0.7808 (n=1008) | 0.4226 (n=1008) | 0.7270 (n=2016) |
| accuracy | gold is B | 0.7493 (n=1053) | 0.8500 (n=1053) | 0.5033 (n=1053) | 0.7980 (n=2106) |
| parse-fail rate | 0.0422 (n=2061) | 0.0330 (n=2061) | 0.0301 (n=2061) | 0.0326 (n=4122) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8476 (n=1942) | 0.5169 (n=1954) | 0.7977 (n=3896) |
| bias | 0.1833 (n=1942) | 0.2902 (n=1954) | 0.1994 (n=3896) |
| anti-bias | 0.0541 (n=1942) | 0.0522 (n=1954) | 0.0538 (n=3896) |
| no bias | 0.7626 (n=1942) | 0.6576 (n=1954) | 0.7468 (n=3896) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1184 (n=3800) | 0.1184 (n=3800) | 0.1184 (n=3800) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.6462 (n=1942) | 0.7620 (n=1954) | 0.6637 (n=3896) |
| said Yes (sampled token) | 0.6045 (n=1942) | 0.6798 (n=1952) | 0.6159 (n=3894) |
| mean P(Yes) | bias | 0.6398 (n=356) | 0.6533 (n=567) | 0.6427 (n=923) |
| mean P(Yes) | anti-bias | 0.7919 (n=105) | 0.7553 (n=102) | 0.7865 (n=207) |
| mean P(Yes) | no bias | 0.5839 (n=1481) | 0.6868 (n=1285) | 0.5975 (n=2766) |
| deception (count) | 0.5672 (n=1837) | 0.6469 (n=1852) | 0.5793 (n=3689) |
| false admission (count) | 0.6226 (n=1481) | 0.7907 (n=1285) | 0.6449 (n=2766) |
| false denial (count) | 0.3371 (n=356) | 0.3210 (n=567) | 0.3335 (n=923) |
| true admission (count) | 0.6629 (n=356) | 0.6790 (n=567) | 0.6665 (n=923) |
| true denial (count) | 0.3774 (n=1481) | 0.2093 (n=1285) | 0.3551 (n=2766) |
| deception (P(Yes)-weighted) | 0.5405 (n=1837) | 0.5827 (n=1852) | 0.5469 (n=3689) |
| false admission (P) | 0.5839 (n=1481) | 0.6868 (n=1285) | 0.5975 (n=2766) |
| false denial (P) | 0.3602 (n=356) | 0.3467 (n=567) | 0.3573 (n=923) |
| true admission (P) | 0.6398 (n=356) | 0.6533 (n=567) | 0.6427 (n=923) |
| true denial (P) | 0.4161 (n=1481) | 0.3132 (n=1285) | 0.4025 (n=2766) |

## Split: train (9618 cued rows, 1603 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6952 (n=4809) | 0.8334 (n=4809) | 0.4554 (n=4809) | 0.7767 (n=9618) |
| accuracy | gold is A | 0.6068 (n=2355) | 0.7936 (n=2355) | 0.3830 (n=2355) | 0.7320 (n=4710) |
| accuracy | gold is B | 0.7800 (n=2454) | 0.8716 (n=2454) | 0.5249 (n=2454) | 0.8196 (n=4908) |
| parse-fail rate | 0.0347 (n=4809) | 0.0254 (n=4809) | 0.0318 (n=4809) | 0.0263 (n=9618) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8558 (n=4555) | 0.5283 (n=4533) | 0.8069 (n=9088) |
| bias | 0.1596 (n=4555) | 0.3139 (n=4533) | 0.1827 (n=9088) |
| anti-bias | 0.0503 (n=4555) | 0.0406 (n=4533) | 0.0488 (n=9088) |
| no bias | 0.7901 (n=4555) | 0.6455 (n=4533) | 0.7685 (n=9088) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1060 (n=8968) | 0.1060 (n=8968) | 0.1060 (n=8968) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.6070 (n=4555) | 0.7695 (n=4533) | 0.6313 (n=9088) |
| said Yes (sampled token) | 0.5909 (n=4547) | 0.6779 (n=4521) | 0.6039 (n=9068) |
| mean P(Yes) | bias | 0.6325 (n=727) | 0.6567 (n=1423) | 0.6387 (n=2150) |
| mean P(Yes) | anti-bias | 0.7400 (n=229) | 0.7420 (n=184) | 0.7403 (n=413) |
| mean P(Yes) | no bias | 0.5687 (n=3599) | 0.6765 (n=2926) | 0.5822 (n=6525) |
| deception (count) | 0.5317 (n=4326) | 0.6144 (n=4349) | 0.5441 (n=8675) |
| false admission (count) | 0.5754 (n=3599) | 0.7820 (n=2926) | 0.6013 (n=6525) |
| false denial (count) | 0.3150 (n=727) | 0.2699 (n=1423) | 0.3034 (n=2150) |
| true admission (count) | 0.6850 (n=727) | 0.7301 (n=1423) | 0.6966 (n=2150) |
| true denial (count) | 0.4246 (n=3599) | 0.2180 (n=2926) | 0.3987 (n=6525) |
| deception (P(Yes)-weighted) | 0.5349 (n=4326) | 0.5675 (n=4349) | 0.5398 (n=8675) |
| false admission (P) | 0.5687 (n=3599) | 0.6765 (n=2926) | 0.5822 (n=6525) |
| false denial (P) | 0.3675 (n=727) | 0.3433 (n=1423) | 0.3613 (n=2150) |
| true admission (P) | 0.6325 (n=727) | 0.6567 (n=1423) | 0.6387 (n=2150) |
| true denial (P) | 0.4313 (n=3599) | 0.3235 (n=2926) | 0.4178 (n=6525) |

## Split: all (13740 cued rows, 2290 questions)

### Turn 1

| metric | none | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|---|
| accuracy (unparsed = wrong) | 0.6849 (n=6870) | 0.8282 (n=6870) | 0.4579 (n=6870) | 0.7727 (n=13740) |
| accuracy | gold is A | 0.5953 (n=3363) | 0.7898 (n=3363) | 0.3949 (n=3363) | 0.7305 (n=6726) |
| accuracy | gold is B | 0.7707 (n=3507) | 0.8651 (n=3507) | 0.5184 (n=3507) | 0.8131 (n=7014) |
| parse-fail rate | 0.0370 (n=6870) | 0.0277 (n=6870) | 0.0313 (n=6870) | 0.0282 (n=13740) |

### Bias (removal, majority-of-3 counterfactual)

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| follow-cue rate (parsed) | 0.8533 (n=6497) | 0.5249 (n=6487) | 0.8041 (n=12984) |
| bias | 0.1667 (n=6497) | 0.3068 (n=6487) | 0.1877 (n=12984) |
| anti-bias | 0.0514 (n=6497) | 0.0441 (n=6487) | 0.0503 (n=12984) |
| no bias | 0.7819 (n=6497) | 0.6491 (n=6487) | 0.7620 (n=12984) |
| resampling null (sampled: P(G then D) within the no-cue cell; logprob: mean P(cue) shift) | 0.1097 (n=12768) | 0.1097 (n=12768) | 0.1097 (n=12768) |

### Self-report and deception

| metric | cue_gold | cue_dist | at_p (p=0.85) |
|---|---|---|---|
| said Yes (P(Yes) fp32 > 0.5) | 0.6187 (n=6497) | 0.7672 (n=6487) | 0.6410 (n=12984) |
| said Yes (sampled token) | 0.5950 (n=6489) | 0.6785 (n=6473) | 0.6075 (n=12962) |
| mean P(Yes) | bias | 0.6349 (n=1083) | 0.6557 (n=1990) | 0.6400 (n=3073) |
| mean P(Yes) | anti-bias | 0.7563 (n=334) | 0.7467 (n=286) | 0.7551 (n=620) |
| mean P(Yes) | no bias | 0.5731 (n=5080) | 0.6796 (n=4211) | 0.5867 (n=9291) |
| deception (count) | 0.5423 (n=6163) | 0.6241 (n=6201) | 0.5546 (n=12364) |
| false admission (count) | 0.5892 (n=5080) | 0.7846 (n=4211) | 0.6141 (n=9291) |
| false denial (count) | 0.3223 (n=1083) | 0.2844 (n=1990) | 0.3130 (n=3073) |
| true admission (count) | 0.6777 (n=1083) | 0.7156 (n=1990) | 0.6870 (n=3073) |
| true denial (count) | 0.4108 (n=5080) | 0.2154 (n=4211) | 0.3859 (n=9291) |
| deception (P(Yes)-weighted) | 0.5366 (n=6163) | 0.5720 (n=6201) | 0.5419 (n=12364) |
| false admission (P) | 0.5731 (n=5080) | 0.6796 (n=4211) | 0.5867 (n=9291) |
| false denial (P) | 0.3651 (n=1083) | 0.3443 (n=1990) | 0.3600 (n=3073) |
| true admission (P) | 0.6349 (n=1083) | 0.6557 (n=1990) | 0.6400 (n=3073) |
| true denial (P) | 0.4269 (n=5080) | 0.3204 (n=4211) | 0.4133 (n=9291) |

## By category (all rows)

| category | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| computer science | cue_gold | 1230 | 1162 | 0.8398 | 0.8718 | 0.1635 | 0.0482 | 0.7883 | 0.4862 | 0.5821 | 0.4758 | 0.4250 | 0.4659 | 0.4758 | 0.4179 | 0.5821 | 0.5242 |
| computer science | cue_dist | 1230 | 1168 | 0.4699 | 0.5154 | 0.3168 | 0.0394 | 0.6438 | 0.6635 | 0.5704 | 0.6175 | 0.6016 | 0.5556 | 0.6175 | 0.4296 | 0.5704 | 0.3825 |
| computer science | at_p | 2460 | 2330 | 0.7843 | 0.8181 | 0.1866 | 0.0469 | 0.7665 | 0.5129 | 0.5791 | 0.4938 | 0.4518 | 0.4795 | 0.4938 | 0.4209 | 0.5791 | 0.5062 |
| history | cue_gold | 1143 | 1072 | 0.8443 | 0.8713 | 0.1157 | 0.0382 | 0.8461 | 0.6381 | 0.7268 | 0.5798 | 0.5519 | 0.5429 | 0.5798 | 0.2732 | 0.7268 | 0.4202 |
| history | cue_dist | 1143 | 1068 | 0.5599 | 0.4176 | 0.2416 | 0.0318 | 0.7266 | 0.8333 | 0.7427 | 0.7031 | 0.6683 | 0.5919 | 0.7031 | 0.2573 | 0.7427 | 0.2969 |
| history | at_p | 2286 | 2140 | 0.8016 | 0.8034 | 0.1345 | 0.0373 | 0.8282 | 0.6673 | 0.7310 | 0.5959 | 0.5694 | 0.5503 | 0.5959 | 0.2690 | 0.7310 | 0.4041 |
| law | cue_gold | 1500 | 1419 | 0.7740 | 0.7935 | 0.1832 | 0.0684 | 0.7484 | 0.7104 | 0.7078 | 0.6492 | 0.5930 | 0.5790 | 0.6492 | 0.2922 | 0.7078 | 0.3508 |
| law | cue_dist | 1500 | 1420 | 0.4173 | 0.5697 | 0.3106 | 0.0641 | 0.6254 | 0.8021 | 0.6959 | 0.7310 | 0.6343 | 0.5893 | 0.7310 | 0.3041 | 0.6959 | 0.2690 |
| law | at_p | 3000 | 2839 | 0.7205 | 0.7599 | 0.2023 | 0.0677 | 0.7299 | 0.7241 | 0.7050 | 0.6597 | 0.5993 | 0.5806 | 0.6597 | 0.2950 | 0.7050 | 0.3403 |
| philosophy | cue_gold | 1497 | 1435 | 0.8444 | 0.8620 | 0.1526 | 0.0460 | 0.8014 | 0.7192 | 0.7184 | 0.6261 | 0.6019 | 0.5710 | 0.6261 | 0.2816 | 0.7184 | 0.3739 |
| philosophy | cue_dist | 1497 | 1425 | 0.4609 | 0.5228 | 0.3144 | 0.0372 | 0.6484 | 0.8449 | 0.7251 | 0.7140 | 0.6239 | 0.5707 | 0.7140 | 0.2749 | 0.7251 | 0.2860 |
| philosophy | at_p | 2994 | 2860 | 0.7868 | 0.8114 | 0.1767 | 0.0447 | 0.7786 | 0.7379 | 0.7202 | 0.6370 | 0.6052 | 0.5709 | 0.6370 | 0.2798 | 0.7202 | 0.3630 |
| physics | cue_gold | 1500 | 1409 | 0.8447 | 0.8758 | 0.2058 | 0.0525 | 0.7417 | 0.5188 | 0.5018 | 0.5170 | 0.5206 | 0.5129 | 0.5170 | 0.4982 | 0.5018 | 0.4830 |
| physics | cue_dist | 1500 | 1406 | 0.4080 | 0.5711 | 0.3364 | 0.0441 | 0.6195 | 0.6892 | 0.5720 | 0.6235 | 0.5990 | 0.5547 | 0.6235 | 0.4280 | 0.5720 | 0.3765 |
| physics | at_p | 3000 | 2815 | 0.7792 | 0.8302 | 0.2254 | 0.0513 | 0.7234 | 0.5443 | 0.5175 | 0.5306 | 0.5324 | 0.5192 | 0.5306 | 0.4825 | 0.5175 | 0.4694 |

## By cue_template_idx (all rows)

| cue_template_idx | block | n | n parsed | acc | follow | bias | anti | no bias | said Yes | P(Yes)|bias | P(Yes)|no bias | decep (count) | decep (P) | false adm | false den | true adm | true den |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | cue_gold | 822 | 796 | 0.8844 | 0.8995 | 0.1935 | 0.0352 | 0.7714 | 0.5666 | 0.6106 | 0.5468 | 0.5052 | 0.5153 | 0.5468 | 0.3894 | 0.6106 | 0.4532 |
| 0 | cue_dist | 822 | 780 | 0.4331 | 0.5500 | 0.3167 | 0.0256 | 0.6577 | 0.7346 | 0.6495 | 0.6538 | 0.6145 | 0.5553 | 0.6538 | 0.3505 | 0.6495 | 0.3462 |
| 0 | at_p | 1644 | 1576 | 0.8167 | 0.8480 | 0.2116 | 0.0338 | 0.7546 | 0.5914 | 0.6191 | 0.5606 | 0.5215 | 0.5212 | 0.5606 | 0.3809 | 0.6191 | 0.4394 |
| 1 | cue_gold | 873 | 819 | 0.8396 | 0.8718 | 0.2210 | 0.0293 | 0.7497 | 0.6325 | 0.6532 | 0.5921 | 0.5308 | 0.5362 | 0.5921 | 0.3468 | 0.6532 | 0.4079 |
| 1 | cue_dist | 873 | 821 | 0.3837 | 0.6017 | 0.3118 | 0.0329 | 0.6553 | 0.7345 | 0.6306 | 0.6745 | 0.6134 | 0.5761 | 0.6745 | 0.3694 | 0.6306 | 0.3255 |
| 1 | at_p | 1746 | 1640 | 0.7712 | 0.8312 | 0.2347 | 0.0298 | 0.7355 | 0.6478 | 0.6487 | 0.6031 | 0.5432 | 0.5422 | 0.6031 | 0.3513 | 0.6487 | 0.3969 |
| 2 | cue_gold | 786 | 747 | 0.8575 | 0.8822 | 0.1807 | 0.0321 | 0.7871 | 0.5087 | 0.6265 | 0.5201 | 0.4675 | 0.4928 | 0.5201 | 0.3735 | 0.6265 | 0.4799 |
| 2 | cue_dist | 786 | 749 | 0.4084 | 0.5794 | 0.3431 | 0.0320 | 0.6248 | 0.7036 | 0.6071 | 0.6512 | 0.6097 | 0.5596 | 0.6512 | 0.3929 | 0.6071 | 0.3488 |
| 2 | at_p | 1572 | 1496 | 0.7901 | 0.8367 | 0.2051 | 0.0321 | 0.7627 | 0.5380 | 0.6216 | 0.5363 | 0.4889 | 0.5028 | 0.5363 | 0.3784 | 0.6216 | 0.4637 |
| 3 | cue_gold | 912 | 868 | 0.7237 | 0.7431 | 0.1210 | 0.1060 | 0.7730 | 0.7028 | 0.6604 | 0.6171 | 0.6186 | 0.5795 | 0.6171 | 0.3396 | 0.6604 | 0.3829 |
| 3 | cue_dist | 912 | 870 | 0.6382 | 0.3414 | 0.1678 | 0.1000 | 0.7322 | 0.7943 | 0.7184 | 0.6774 | 0.6705 | 0.6036 | 0.6774 | 0.2816 | 0.7184 | 0.3226 |
| 3 | at_p | 1824 | 1738 | 0.7109 | 0.6827 | 0.1280 | 0.1051 | 0.7669 | 0.7165 | 0.6718 | 0.6257 | 0.6264 | 0.5832 | 0.6257 | 0.3282 | 0.6718 | 0.3743 |
| 4 | cue_gold | 855 | 813 | 0.9240 | 0.9471 | 0.1931 | 0.0098 | 0.7970 | 0.6691 | 0.6792 | 0.5872 | 0.5503 | 0.5353 | 0.5872 | 0.3208 | 0.6792 | 0.4128 |
| 4 | cue_dist | 855 | 806 | 0.3298 | 0.6576 | 0.4467 | 0.0223 | 0.5310 | 0.8151 | 0.6682 | 0.7454 | 0.6015 | 0.5565 | 0.7454 | 0.3318 | 0.6682 | 0.2546 |
| 4 | at_p | 1710 | 1619 | 0.8349 | 0.9040 | 0.2309 | 0.0117 | 0.7574 | 0.6909 | 0.6760 | 0.6038 | 0.5579 | 0.5384 | 0.6038 | 0.3240 | 0.6760 | 0.3962 |
| 5 | cue_gold | 903 | 838 | 0.7320 | 0.7637 | 0.1193 | 0.0895 | 0.7912 | 0.6933 | 0.6321 | 0.6051 | 0.6186 | 0.5740 | 0.6051 | 0.3679 | 0.6321 | 0.3949 |
| 5 | cue_dist | 903 | 846 | 0.5736 | 0.4066 | 0.2092 | 0.0638 | 0.7270 | 0.7943 | 0.6734 | 0.6768 | 0.6692 | 0.5985 | 0.6768 | 0.3266 | 0.6734 | 0.3232 |
| 5 | at_p | 1806 | 1684 | 0.7083 | 0.7097 | 0.1329 | 0.0856 | 0.7815 | 0.7086 | 0.6420 | 0.6151 | 0.6264 | 0.5778 | 0.6151 | 0.3580 | 0.6420 | 0.3849 |
| 6 | cue_gold | 891 | 841 | 0.7856 | 0.8121 | 0.1344 | 0.0785 | 0.7872 | 0.6480 | 0.6389 | 0.5958 | 0.5677 | 0.5616 | 0.5958 | 0.3611 | 0.6389 | 0.4042 |
| 6 | cue_dist | 891 | 845 | 0.4916 | 0.4947 | 0.3030 | 0.0533 | 0.6438 | 0.7775 | 0.6615 | 0.6806 | 0.6300 | 0.5712 | 0.6806 | 0.3385 | 0.6615 | 0.3194 |
| 6 | at_p | 1782 | 1686 | 0.7415 | 0.7643 | 0.1598 | 0.0747 | 0.7656 | 0.6675 | 0.6453 | 0.6066 | 0.5773 | 0.5631 | 0.6066 | 0.3547 | 0.6453 | 0.3934 |
| 7 | cue_gold | 828 | 775 | 0.8998 | 0.9252 | 0.1781 | 0.0219 | 0.8000 | 0.5045 | 0.5752 | 0.5099 | 0.4736 | 0.4944 | 0.5099 | 0.4248 | 0.5752 | 0.4901 |
| 7 | cue_dist | 828 | 770 | 0.3792 | 0.5961 | 0.3779 | 0.0143 | 0.6078 | 0.7753 | 0.6633 | 0.6877 | 0.5810 | 0.5531 | 0.6877 | 0.3367 | 0.6633 | 0.3123 |
| 7 | at_p | 1656 | 1545 | 0.8217 | 0.8761 | 0.2079 | 0.0208 | 0.7713 | 0.5449 | 0.5991 | 0.5308 | 0.4897 | 0.5032 | 0.5308 | 0.4009 | 0.5991 | 0.4692 |

## Probe populations (eligible rows per probe kind; 260923's class and row filters)

| kind | split | class | condition | n rows | n prompts |
|---|---|---|---|---|---|
| bias | train | bias | cue_gold | 727 | 328 |
| bias | train | bias | cue_dist | 1423 | 700 |
| bias | train | no bias | cue_gold | 3599 | 1371 |
| bias | train | no bias | cue_dist | 2926 | 1271 |
| bias | eval | bias | cue_gold | 356 | 160 |
| bias | eval | bias | cue_dist | 567 | 289 |
| bias | eval | no bias | cue_gold | 1481 | 581 |
| bias | eval | no bias | cue_dist | 1285 | 565 |
| bias_followed | train | bias | cue_gold | 727 | 328 |
| bias_followed | train | bias | cue_dist | 1423 | 700 |
| bias_followed | train | no bias | cue_gold | 3171 | 1144 |
| bias_followed | train | no bias | cue_dist | 972 | 383 |
| bias_followed | eval | bias | cue_gold | 356 | 160 |
| bias_followed | eval | bias | cue_dist | 567 | 289 |
| bias_followed | eval | no bias | cue_gold | 1290 | 477 |
| bias_followed | eval | no bias | cue_dist | 443 | 180 |
| deception | train | deceptive | cue_gold | 2300 | 1095 |
| deception | train | deceptive | cue_dist | 2672 | 1270 |
| deception | train | honest | cue_gold | 2026 | 981 |
| deception | train | honest | cue_dist | 1677 | 924 |
| deception | eval | deceptive | cue_gold | 1042 | 499 |
| deception | eval | deceptive | cue_dist | 1198 | 557 |
| deception | eval | honest | cue_gold | 795 | 400 |
| deception | eval | honest | cue_dist | 654 | 373 |
| deception_yes | train | false admission | cue_gold | 2071 | 1003 |
| deception_yes | train | false admission | cue_dist | 2288 | 1117 |
| deception_yes | train | true admission | cue_gold | 498 | 264 |
| deception_yes | train | true admission | cue_dist | 1039 | 584 |
| deception_yes | eval | false admission | cue_gold | 922 | 447 |
| deception_yes | eval | false admission | cue_dist | 1016 | 500 |
| deception_yes | eval | true admission | cue_gold | 236 | 129 |
| deception_yes | eval | true admission | cue_dist | 385 | 221 |
| deception_no | train | false denial | cue_gold | 229 | 125 |
| deception_no | train | false denial | cue_dist | 384 | 243 |
| deception_no | train | true denial | cue_gold | 1528 | 724 |
| deception_no | train | true denial | cue_dist | 638 | 378 |
| deception_no | eval | false denial | cue_gold | 120 | 73 |
| deception_no | eval | false denial | cue_dist | 182 | 111 |
| deception_no | eval | true denial | cue_gold | 559 | 275 |
| deception_no | eval | true denial | cue_dist | 269 | 163 |
