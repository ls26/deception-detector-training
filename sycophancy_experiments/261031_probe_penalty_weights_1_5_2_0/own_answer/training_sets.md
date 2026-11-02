# Training sets of the own-answer probes

Own pool: the model's cued answers on the 1217 training questions that finished, parsed, name the cued option, are
under 2048 tokens with the prompt, on questions with a usual answer (at least 6 of the model's 8 no-cue answers).
Per group: the size every group takes (the smallest own group, at least 200). Base added: base-model answers from
the original probe's pool filling a group with fewer own answers. The original probe: 378 per group, from a
base-model pool of 378 / 1037 / 2739 / 642. Built by `own_probe.py table`.

| Model | Questions with a usual answer | Own pool: biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue at wrong | Per group | Base added | Topped up | Trainer's training answers, biased / unbiased |
|---|---|---|---|---|---|---|
| Weight 1.5 s42 | 898 | 692 / 1999 / 2001 / 693 | 692 | 0 / 0 / 0 / 0 | no | 1384 / 1384 |
| Weight 1.5 s43 | 1174 | 496 / 884 / 2643 / 879 | 496 | 0 / 0 / 0 / 0 | no | 992 / 992 |
| Weight 1.5 s44 | 1132 | 598 / 1356 / 2666 / 727 | 598 | 0 / 0 / 0 / 0 | no | 1196 / 1196 |
| Weight 2.0 s42 | 1217 | 0 / 0 / 1953 / 1698 | 200 | 200 / 200 / 0 / 0 | yes | 400 / 400 |
| Weight 2.0 s43 | 1091 | 568 / 1681 / 2642 / 624 | 568 | 0 / 0 / 0 / 0 | no | 1136 / 1136 |
| Weight 2.0 s44 | 1142 | 456 / 1134 / 2796 / 625 | 456 | 0 / 0 / 0 / 0 | no | 912 / 912 |
