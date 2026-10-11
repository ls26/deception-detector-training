# No-CoT logprob probe on its no-cue-confident eval rows, beside the CoT slice probes

No-CoT probe: 261007's `nocot_logprob` final fit, trained on all train questions; its saved eval scores, re-scored here with no retraining. No-cue confident: the question's no-cue prompt has a top letter with probability >= 0.75 (renormalized over A and B); 547 of 687 eval questions. AUC of the mean sigmoid over each probe's band; one row per prompt.

## No-CoT probe, three eval filters

| Probe | All eval rows: AUC (rows) | Cued and no-cue prompts confident (261007): AUC (rows) | No-cue prompt confident: AUC (rows, pos / neg, questions) |
|---|---|---|---|
| deception_yes_rm_preyesno | 0.809 (1072) | 0.841 (839) | 0.845 (888, 633 / 255, 492) |
| deception_no_rm_preyesno | 0.762 (261) | 0.826 (186) | 0.831 (199, 65 / 134, 144) |
| bias_followed_rm_answer | 0.880 (1099) | 0.920 (827) | 0.923 (863, 320 / 543, 546) |

## Each format on its own no-cue-confident eval rows

CoT rows: the 489 slice questions (at least 6 of 8 no-cue CoT samples agree), one CoT rollout per prompt, labels from that answer (`comparison.json`). No-CoT rows: the 547 questions whose no-cue top letter has probability >= 0.75, one-letter answers, labels from the top letters. The two formats are scored on different completions and different question sets; no Δ is computed between them. Main result of this experiment: the slice-trained CoT probe.

| Probe | CoT slice-trained probe AUC | CoT baseline probe AUC | CoT rows (pos / neg), questions | No-CoT probe AUC | No-CoT rows (pos / neg), questions |
|---|---|---|---|---|---|
| deception_yes_rm_preyesno | 0.836 | 0.825 | 664 (547 / 117), 410 | 0.845 | 888 (633 / 255), 492 |
| deception_no_rm_preyesno | 0.859 | 0.860 | 310 (64 / 246), 232 | 0.831 | 199 (65 / 134), 144 |
| bias_followed_rm_answer | 0.871 | 0.866 | 777 (292 / 485), 489 | 0.923 | 863 (320 / 543), 546 |
