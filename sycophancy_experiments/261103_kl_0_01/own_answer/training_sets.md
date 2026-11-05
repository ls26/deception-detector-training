# Training sets of the own-answer probes

Own pool: the model's cued answers on the 1217 training questions that finished, parsed, name the cued option, are
under 2048 tokens with the prompt, on questions with a usual answer (at least 6 of the model's 8 no-cue answers).
Per group: the size every group takes (the smallest own group, at least 200). Base added: base-model answers from
the original probe's pool filling a group with fewer own answers. The original probe: 378 per group, from a
base-model pool of 378 / 1037 / 2739 / 642. Built by `own_probe.py table`.

| Model | Questions with a usual answer | Own pool: biased, cue at correct / biased, cue at wrong / unbiased, cue at correct / unbiased, cue at wrong | Per group | Base added | Topped up | Trainer's training answers, biased / unbiased |
|---|---|---|---|---|---|---|
| Weight 1.0, KL 0.01 s42 | 1115 | 380 / 1045 / 2792 / 538 | 380 | 0 / 0 / 0 / 0 | no | 760 / 760 |
| Weight 1.0, KL 0.01 s43 | 1106 | 443 / 1455 / 2752 / 549 | 443 | 0 / 0 / 0 / 0 | no | 886 / 886 |
| Weight 1.0, KL 0.01 s44 | 1104 | 244 / 420 / 2672 / 590 | 244 | 0 / 0 / 0 / 0 | no | 488 / 488 |
