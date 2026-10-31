# Training sets of the own-answer probes

Own pool: the model's cued answers on the 1217 training questions that finished, parsed, name the cued option, are
under 2048 tokens with the prompt, on questions with a usual answer (at least 6 of the model's 8 no-cue answers).
Per group: the size every group takes (the smallest own group, at least 200). Base added: base-model answers from
the original probe's pool filling a group with fewer own answers. The original probe: 378 per group, from a
base-model pool of 378 / 1037 / 2739 / 642. Built by `own_probe.py table`.

| Model | Questions with a usual answer | Own pool: biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue at wrong | Per group | Base added | Topped up | Trainer's training answers, biased / unbiased |
|---|---|---|---|---|---|---|
| Weight 0.3 s42 | 1124 | 604 / 1695 / 2689 / 678 | 604 | 0 / 0 / 0 / 0 | no | 1208 / 1208 |
| Weight 0.3 s43 | 1065 | 514 / 2334 / 2662 / 529 | 514 | 0 / 0 / 0 / 0 | no | 1028 / 1028 |
| Weight 0.3 s44 | 1177 | 591 / 977 / 2664 / 853 | 591 | 0 / 0 / 0 / 0 | no | 1182 / 1182 |
| Weight 0.6 s42 | 1094 | 515 / 1479 / 2690 / 591 | 515 | 0 / 0 / 0 / 0 | no | 1030 / 1030 |
| Weight 0.6 s43 | 1123 | 456 / 1464 / 2812 / 550 | 456 | 0 / 0 / 0 / 0 | no | 912 / 912 |
| Weight 0.6 s44 | 1121 | 558 / 1629 / 2729 / 630 | 558 | 0 / 0 / 0 / 0 | no | 1116 / 1116 |
| Weight 1.0 s42 | 1168 | 771 / 1258 / 2556 / 943 | 771 | 0 / 0 / 0 / 0 | no | 1542 / 1542 |
| Weight 1.0 s43 | 1139 | 481 / 1388 / 2785 / 625 | 481 | 0 / 0 / 0 / 0 | no | 962 / 962 |
| Weight 1.0 s44 | 1208 | 659 / 1321 / 2760 / 861 | 659 | 0 / 0 / 0 / 0 | no | 1318 / 1318 |
| No penalty s42 | 1066 | 543 / 1805 / 2592 / 601 | 543 | 0 / 0 / 0 / 0 | no | 1086 / 1086 |
| No penalty s43 | 1115 | 550 / 1380 / 2687 / 645 | 550 | 0 / 0 / 0 / 0 | no | 1100 / 1100 |
| No penalty s44 | 1108 | 567 / 1671 / 2689 / 627 | 567 | 0 / 0 / 0 / 0 | no | 1134 / 1134 |
