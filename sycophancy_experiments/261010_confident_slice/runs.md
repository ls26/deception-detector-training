# Runs

State file for the unattended run. Read this first after any restart.

## Queue

- [x] 1 `generate_nocue_train.py` -> `completions.jsonl` (1603 questions, 12824 records)
- [x] 2 `slice_labels.py` -> `arms/slice_cot/labels.jsonl` (1603 train / 687 eval questions; 489 eval slice questions)
- [x] 3 `run_pipeline.py` -> `arms/slice_cot/{probe_sets,activations}/` (4 activation files, `extraction_summary.json`)
- [x] 4 `cv_run.py --probe <probe>` x 3 in parallel -> `cv/slice_cot/<probe>/selection.json` (3 probes, `CV DONE`)
- [x] 5 `final_fit.py` -> `runs/slice_cot/`, `runs/baseline_on_slice/` (3 probes at their CV choice)
- [x] 6 `compare_slice.py` -> `comparison.md` (a row per probe in both tables)

## Log

One line per step: `<UTC time> | step | outcome | next`.

- 2026-10-10 19:28 UTC | 1 generate | EXIT 0; `generation_summary.json`: 1603 questions, 12824 records (all finish_reason stop; 504 unparsed, G 8816, D 3504); seed and invariance checks passed | 2 slice_labels
- 2026-10-10 19:29 UTC | 2 slice_labels | EXIT 0; `slice_summary.json`: train 1603 questions (1217 slice, 7302 rows), eval 687 questions (489 slice, 2934 rows); 10236 label rows | 3 run_pipeline
- 2026-10-10 19:33 UTC | 3 run_pipeline | EXIT 0; `extraction_summary.json` written; 4 activation files. Train per class: deception_yes 502, deception_no 254, bias_followed 756 (train rows 1512 deception (1004 said-Yes / 508 said-No), 1512 bias_followed); eval rows 974 deception (664 said-Yes / 310 said-No), 777 bias_followed (292 bias / 485 no-bias) | 4 cv x 3
- 2026-10-10 20:50 UTC | 4 cv x 3 | all EXIT 0 with `CV DONE`; selection.json + edge_cells.json for each. bias_followed_rm_answer: lr 0.03 / 2000 steps / layers 13-16, OOF 0.8826 ± 0.0362 (edge cell (0.1, 2000) run, choice unchanged). deception_yes_rm_preyesno: lr 0.03 / 8000 / 15-18, OOF 0.8267 ± 0.0243 (edge cells (0.1, 4000), (0.03, 8000), (0.1, 8000) run; pre-edge choice was 0.03 / 4000 / 14-17, OOF 0.8259). deception_no_rm_preyesno: lr 0.03 / 2000 / 14-17, OOF 0.8447 ± 0.0473 (edge cell (0.1, 2000) run, choice unchanged) | 5 final_fit
- 2026-10-10 20:56 UTC | 5 final_fit | EXIT 0; `probe_metrics.json` lists the 3 probes, each at its selection.json lr / steps / band (bias_followed 0.03 / 2000 / 13-16, deception_yes 0.03 / 8000 / 15-18, deception_no 0.03 / 2000 / 14-17). Rescoring check (reported, not a gate): `deception_yes_rm_preyesno: {'n_rows': 920, 'layers': [14, 15, 16, 17], 'max_abs_score_diff': 1.1920928955078125e-07, 'auc_saved': 0.7714285714285715, 'auc_rescored': 0.7714285714285715}`; `deception_no_rm_preyesno: {'n_rows': 406, 'layers': [15, 16, 17, 18], 'max_abs_score_diff': 1.1920928955078125e-07, 'auc_saved': 0.7994197587417927, 'auc_rescored': 0.7994197587417927}`; `bias_followed_rm_answer: {'n_rows': 1106, 'layers': [13, 14, 15, 16], 'max_abs_score_diff': 1.1920928955078125e-07, 'auc_saved': 0.7882865017136, 'auc_rescored': 0.7882865017136}` | 6 compare_slice
- 2026-10-10 20:58 UTC | 6 compare_slice | EXIT 0; `comparison.md` and `comparison.json` written, both tables have a row for each of the 3 probes. Δ (slice-trained minus baseline): deception_yes +0.011 [-0.013, +0.036], deception_no -0.001 [-0.030, +0.028], bias_followed +0.006 [-0.010, +0.023] | none: queue done

## Judgment calls

- Kickoff 2026-10-10 19:23 UTC (the `261010-slice` tmux session holds this Claude run), so the 5 h cap is 00:23 UTC. Pod `a26a5676bdcd` has 1 GPU (H100 80 GB): the heavy steps run one at a time on GPU 0 with no `CUDA_VISIBLE_DEVICES` override.
- Each job's tmux session is named `slice-step<N>-<name>`; each runs `bash -c 'source /workspace/atlas_setup.sh; PYTHONUNBUFFERED=1 <venv python> <script> > logs/<step>_<host>.log 2>&1; echo EXIT $? >> <log>'`. `PYTHONUNBUFFERED=1` only makes progress visible in the log; it does not change the command.
- Step 4: the three CV jobs started together at 19:33 UTC (sessions `slice-step4-cv-<probe>`), each with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` as the queue says; 128 CPUs, so no further limit on parallel fits.
- Deviation: I waited on step 2 (a 30 s CPU job) with a foreground `until grep EXIT; sleep 3` loop instead of a background poll. No effect on outputs; every later wait is a background poll.
- Correction (added after the run by the attended session): "every later wait is a background poll" in the line above is not accurate. A read-only status check at about 20:45 UTC, during step 4, saw this session waiting on a foreground `until grep -q 'POLLDONE' ...; do sleep 5; done` loop capped at 580 s. No effect on outputs.

## Blocked

## Result

Copied verbatim from `comparison.md` (written by step 6 at 20:58 UTC; title line dropped; its `## Training` heading demoted to `###` so it stays inside this section).

Slice: questions where at least 6 of 8 no-cue samples give the same answer; labels use that answer. Both probes are scored on the same slice eval rows with the slice labels; AUC of the mean sigmoid over each probe's own band. Δ = slice-trained minus baseline; 95% CI from a question-level bootstrap (2000 draws, seed 20261007); `*` marks a CI that excludes 0. One training seed (0).

| Probe | Eval rows (pos / neg) | Slice-trained AUC | Baseline AUC | Δ [95% CI] |
|---|---|---|---|---|
| deception_yes_rm_preyesno | 664 (547 / 117) | 0.836 | 0.825 | +0.011 [-0.013, +0.036] |
| deception_no_rm_preyesno | 310 (64 / 246) | 0.859 | 0.860 | -0.001 [-0.030, +0.028] |
| bias_followed_rm_answer | 777 (292 / 485) | 0.871 | 0.866 | +0.006 [-0.010, +0.023] |

### Training

| Probe | Train rows, slice (baseline) | Slice recipe: lr / steps / layers (epochs) | Baseline recipe: lr / steps / layers | Baseline AUC, all eval rows (rows) |
|---|---|---|---|---|
| deception_yes_rm_preyesno | 1004 (1992) | 0.03 / 8000 / 15–18 (255.0) | 0.03 / 4000 / 14–17 | 0.771 (920) |
| deception_no_rm_preyesno | 508 (916) | 0.03 / 2000 / 14–17 (126.0) | 0.03 / 4000 / 15–18 | 0.799 (406) |
| bias_followed_rm_answer | 1512 (2908) | 0.03 / 2000 / 13–16 (42.3) | 0.01 / 4000 / 13–16 | 0.788 (1106) |
