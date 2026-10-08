# Record schemas

Every file is JSONL unless noted. All scripts stamp `spec_hash` (spec.spec_hash()) into
their summary JSON and assert it against the inputs they read.

## `gens/<gen>/turn1.jsonl` — one row per turn-1 sample (gen = `nocot` | `bio`)

| field | type | notes |
|---|---|---|
| `question_id`, `category`, `src`, `split`, `order`, `answer` | from domain_questions.jsonl | `answer` = gold letter |
| `condition` | `none` / `cue_gold` / `cue_dist` | `bio` has cued rows only |
| `cue_letter` | `A` / `B` / null | null for `none` |
| `cue_text` | str / null | nocot: the short cue sentence; bio: the full bio paragraph |
| `cue_template_idx` | int | nocot only (260819's draw); null for bio |
| `bio_idx` | int | bio only (spec.draw_bio_idx); null for nocot |
| `is_assigned` | bool | condition == the question's assigned condition |
| `sample_idx` | 0, 1, 2 | |
| `prompt` | str | the chat prompt text up to and including the assistant header (no prefill) |
| `prefill` | str | nocot: `"The answer is ("`; bio: `""` |
| `completion` | str | the full assistant text: nocot = prefill + letter + `")"`; bio = the sampled CoT |
| `sampled_text` | str | what the model generated (nocot: the letter) |
| `sampled_token_ids` | list[int] | nocot: one id; bio: omitted (null) |
| `finish_reason` | str | nocot is always `length` by construction |
| `answer_letter` | `A` / `B` / null | `extract_answer(completion)`; for nocot equals the sampled letter |
| `answer_stage` | str | extract_answer's stage |
| `lp_a`, `lp_b` | float / null | nocot only: logprob of token A / B at the letter position (null if outside top-20) |
| `p_a` | float / null | nocot only: P(A) renormalized over {A, B} |
| `mass_ab` | float | nocot only: unrenormalized P(A) + P(B) |
| `top_letter` | `A` / `B` / null | nocot only: argmax over {A, B}; null on an exact tie |
| `top1_id`, `top1_text` | | nocot only: the overall argmax token |
| `model`, `temperature`, `seed`, `build_id`, `spec_hash` | | |

The nocot logprob fields are stamped from sample 0 onto a prompt's three sample rows, so they
are identical by construction. vLLM prefills the three children separately, and bf16 batch
layout can move a sibling's reading by up to ~1e-3 in P(A); the generator reports the sibling
spread in `turn1_summary.json` and asserts it stays below 1e-2.

## `gens/<gen>/followups.jsonl` — one row per (question_id, condition, sample_key)

`sample_key` is `"0"`, `"1"`, `"2"` (the sample's own letter) or, for nocot only, `"top"`
(the prompt's top letter; equals one of the samples' prompts byte-for-byte whenever a
sample picked that letter). Removal wording only, the question's assigned plain/explicit
draw (`draw_wordings(qid)["rm"]`, `ARM_OF["rm"][wording]`).

| field | notes |
|---|---|
| `question_id`, `category`, `split`, `order`, `condition`, `cue_letter`, `is_assigned` | |
| `sample_key` | str, see above |
| `sample_idx` | int or null (`"top"`) |
| `answer_letter` | the letter this follow-up refers to (the explicit wording names it) |
| `arm` | `removal` / `removal_explicit` |
| `followup_text` | |
| `prompt`, `completion` | the turn-1 pair (so prompt + completion + suffix is the turn-2 prompt) |
| `followup_completion`, `finish_reason`, `first_token_ids` | 260823's fields |
| `p_yes_pos0`, `p_yes_broad_pos0`, `mass_yesno_pos0`, `top1_id_pos0`, `top1_text_pos0`, `missing_yes_pos0`, `missing_no_pos0` | 260823's position-0 stats |
| `ans_pos` | index of the first sampled Yes/No form among the generated tokens, or null |
| `said` | `Yes` / `No` / null: the first sampled Yes/No form |
| `seed`, `spec_hash` | |

## `gens/<gen>/pyes_fp32.jsonl` and `arms/ref_cot/pyes_fp32.jsonl` (score_pyes.py)

One row per follow-up prompt: the key fields of the follow-up row (`question_id`, `condition`,
`sample_key` for gens; `sample_idx` for ref), `lp_yes_fp32`, `lp_no_fp32` (float32 log-softmax of
the output head applied to the bf16 model's final residual at the last prompt token),
`p_yes_fp32` = sigmoid(lp_yes - lp_no), `p_yes_bf16` (vLLM's `p_yes_pos0`, for comparison),
`spec_hash`. The summary JSON reports the bf16-vs-fp32 agreement on the sign of P(Yes) - 0.5.

## `arms/<arm>/labels.jsonl` — one row per cued rollout (the probe-training population)

For `nocot_logprob` there is one row per prompt (`sample_idx` = 0, the top letter); for
the sampled arms one per sample. Mirrors 260824's `rescored_labels.jsonl` plus the text
the probe pipeline needs, so step 1 reads nothing else.

| field | notes |
|---|---|
| `question_id`, `category`, `split`, `order`, `condition`, `sample_idx`, `is_assigned` | |
| `cue_letter`, `answer` (gold), `answer_letter`, `correct`, `followed_cue` | `followed_cue` = answer_letter == cue_letter |
| `a_cued`, `a_none` | `G` / `D` / null, content space; `a_none` is the counterfactual used (majority-of-3 for sampled arms, the top letter for logprob) |
| `bias_rm` | `bias` / `anti-bias` / `no bias` / `unparsed_turn1` (260824's `turn1_label`) |
| `arm`, `said_rm`, `p_yes_rm` | the assigned removal wording's follow-up; `p_yes_rm` is the float32 P(Yes) from `pyes_fp32.jsonl` and `said_rm` = Yes if > 0.5, No if < 0.5, null on an exact tie, in EVERY arm |
| `deception_rm` | `deceptive` / `honest` / `excluded_antibias` / `unparsed_turn2` (260824's `turn2_label`) |
| `said_sampled`, `p_yes_bf16` | the sampled first Yes/No token and vLLM's bf16 `p_yes_pos0`, kept for the baselines only |
| `subtype` | for bias/no-bias rows: the bias label; for deception rows: `false admission` etc. (step1's `deception_subtype`) — computed by step 1, not stored here |
| `prompt`, `completion` | turn 1 |
| `followup_text`, `followup_completion`, `ans_pos`, `first_token_ids` | turn 2 (null when no follow-up row exists) |
| `lp_a`, `lp_b` | the cued prompt's letter logprobs, both no-CoT arms (null otherwise); extract.py's HF-vs-vLLM gate reads them |
| `p_a_cued`, `p_a_none`, `p_cue_cued`, `p_cue_none`, `mass_ab_cued`, `mass_ab_none`, `confident` | logprob arm only (null otherwise); `p_cue_*` = P(cue letter); `confident` = both top letters >= CONFIDENT_SLICE. On an exact P(A) = P(B) tie the prompt has no top letter, so `completion`, `arm`, `said_rm`, `p_yes_rm` are null and the row is `unparsed_turn1` |
| `bio_idx` / `cue_template_idx` | whichever applies |
| `spec_hash` | |

`arms/<arm>/none_answers.json`: `{question_id: {"contents": [G/D/null x3]}}` for sampled
arms (the no-cue samples the counterfactual majority is taken over), or
`{question_id: {"top": letter_or_null, "p_a": float, "mass_ab": float}}` for the logprob arm. Used by
baselines.py for no-cue accuracy and the resampling null.

## `arms/sizes.json` (train_sizes.py)

```
{"spec_hash": ..., "per_probe": {kind: {"matched_n_per_class": int, "rule": "matched"|"unmatched"|"insufficient",
   "per_arm": {arm: {"distinct_per_cell": {class: {condition: n}}, "full_n_per_class": int}}}}}
```
Step 1 reads `rule` and `matched_n_per_class` (or the arm's own full size when unmatched).
Kinds: `bias`, `bias_followed`, `deception_yes`, `deception_no`; the unstratified `deception`
train set is the union of the two strata draws and has no entry of its own.

## Probe sets, activations, runs

`arms/<arm>/probe_sets/{bias,deception,bias_followed}_rm_{train,eval}.jsonl` and
`manifest.json`; `arms/<arm>/activations/<dataset>_<position>_<split>.pt` (260923's
format: `acts` {layer: (N,1,H) bf16}, `labels`, `row_keys` (qid, condition, sample_idx,
category), `said`, `subtype`, `positions`, `ans_pos`); positions `answer` (bias family)
and `preyesno` (deception).

`cv/<arm>/<probe>/lr{lr}_steps{steps}/fold{k}.json`, `folds.json`, `selection.json`;
`cv/<arm>/chosen_overrides.json`, `summary.md`.

`runs/<arm>/probe_metrics.json` (260924's per-probe summary fields), `probe_weights/<probe>/`,
`eval_scores.json`: `{probe: {"question_id": [...], "condition": [...], "label": [...],
"score": [...]}}` for the bootstrap in compare.py.
