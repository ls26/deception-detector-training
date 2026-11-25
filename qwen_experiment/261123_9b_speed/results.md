# 261123–261124: why Qwen3.5-9B GRPO steps are slow, and what makes them faster

All runs: sycophancy, base Qwen3.5-9B, no probe penalty, seed 42, 64 questions x 8 answers = 512 answers per step,
on pod fdc2d68d6537 (one H100 80GB HBM3, Xeon Platinum 8462Y+, vLLM at 0.40 of the GPU) unless noted.

## Where the time goes (model only, `time_model_passes.py`, `model_timing/`)

- Each forward/backward pass has a fixed CPU cost: the CPU issues ~14,000-18,000 PyTorch operations per forward and
  ~61,000 per forward + backward, mostly in the 24 linear-attention layers (73% of a forward's CPU time) and the LoRA
  on every linear layer. At 1 answer per pass, one pass (probe-scoring forward + reference forward + policy forward
  and backward + 3 adapter switches) takes ~470 ms, the same at 150 or 1,000 tokens (`run_<host>.log`).
- With bigger passes of equal-length answers the GPU becomes the limit (`profile_sorted_<host>.log`; cuDNN attention
  off, cheap adapter switch; GPU computing = profiler's summed kernel time):

| Answers per pass (500 tokens each) | Pass | Time per pass ms | GPU computing ms | GPU computing share |
|---|---|---|---|---|
| 1 | probe scoring / reference / policy | 75 / 76 / 322 | 34 / 30 / 109 | 45% / 39% / 34% |
| 4 | probe scoring / reference / policy | 129 / 115 / 397 | 115 / 101 / 369 | 89% / 88% / 93% |
| 8 | probe scoring / reference / policy | 241 / 214 / 756 | 228 / 200 / 730 | 94% / 94% / 97% |

  Per answer, 8 equal-length answers per pass cost 151 ms and 4 cost 160 ms.
- cuDNN attention sets itself up again for each new sequence length and keeps that setup: a forward took 108 ms the
  first time through 300 lengths and 70 ms the second (`cudnn_cache_<host>.log`), so it mostly slows a run's first
  steps.

## Trainer runs, 3 steps each (H100)

`spd4_syc9b` ran 2026-11-23 with yesterday's code (cuDNN attention on, old adapter switch, 4,000-step probe fit, an
in-loop eval after step 3). `uns4_h100` and `srt4_h100` ran 2026-11-24 with cuDNN attention off, the cheap adapter
switch and a 200-step probe fit; they differ only in sorting. Training pass = the trainer's
`train_on_generation_batch` timer (probe scoring included); tokens = question + answer tokens of the step's 512 saved
answers; step time = time between consecutive "Step N timings" log lines (spd4_syc9b's step 3 minus its 44.6 s eval).

| Run | Answers per pass | Sorted | Generation s (steps 1/2/3) | Training pass s (1/2/3) | Of which probe scoring s | Padding share | Training-pass s per 100k tokens | Step time s (2/3) | Log-prob correlation (trainer vs vLLM) | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| `spd4_syc9b` (reference) | 4 | no | 45.8 / 43.3 / 43.8 | 161.0 / 134.2 / 125.9 | 32.0 / 30.2 / 24.5 | not logged | 53.3 / 47.3 / 42.1 | 178 / 169 | 0.9994 / 0.9993 / 0.9993 | 1024 each step |
| `uns4_h100` | 4 | no | 43.3 / 42.1 / 38.7 | 119.5 / 120.8 / 105.8 | 23.1 / 23.8 / 20.7 | 17.4% / 18.4% / 19.9% | 41.4 / 43.0 / 43.7 | 163 / 144 | 0.9994 / 0.9994 / 0.9994 | 1024 each step |
| `srt4_h100` | 4 | yes | 44.1 / 41.5 / 41.0 | 106.2 / 100.5 / 100.0 | 23.1 / 22.1 / 21.6 | 0.6% / 0.7% / 0.7% | 36.4 / 37.2 / 37.4 | 142 / 141 | 0.9995 / 0.9995 / 0.9994 | 1024 each step |

Peak GPU memory by `nvidia-smi` (vLLM included, PyTorch's cache included, so an upper bound): 80.9 / 81.0 / 81.1 GB of
81.6. Rough 400-step run time from the mean of steps 2-3: `spd4_syc9b` 19.3 h, `uns4_h100` 17.1 h, `srt4_h100`
15.7 h; the 1-answer-per-pass calibration run (261122) took 287 s per step, 32 h per 400 steps.

## Trainer runs, 3 steps each (H200)

Pod fa1ce76614ef: one H200 (143.8 GB), Intel Xeon Platinum 8558, 192 cores; load average 25.7 when the queue started
and 14.0 when it ended (`logs/queue_h200_sort_fa1ce76614ef.log`). Same settings as `uns4_h100` / `srt4_h100` (cuDNN
attention off, cheap adapter switch, 200-step probe fit, vLLM at 0.40 of the GPU), but 8 answers per pass (8 x 8 per
optimizer step). `srt8_h200` was the pod's first trainer, `uns8_h200` its second. Columns as in the H100 table.

| Run | Answers per pass | Sorted | Generation s (steps 1/2/3) | Training pass s (1/2/3) | Of which probe scoring s | Padding share | Training-pass s per 100k tokens | Step time s (2/3) | Log-prob correlation (trainer vs vLLM) | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| `uns8_h200` | 8 | no | 41.0 / 39.9 / 42.5 | 108.6 / 109.2 / 109.4 | 21.0 / 21.1 / 21.1 | 23.6% / 26.7% / 27.7% | 39.6 / 39.3 / 39.8 | 149 / 152 | 0.9994 / 0.9993 / 0.9994 | 1024 each step |
| `srt8_h200` | 8 | yes | 37.0 / 39.2 / 36.2 | 280.5 / 94.8 / 92.7 | 67.9 / 21.5 / 20.6 | 1.8% / 1.5% / 1.8% | 100.5 / 33.1 / 34.0 | 134 / 129 | 0.9995 / 0.9995 / 0.9995 | 1024 each step |

Peak GPU memory by `nvidia-smi` (an upper bound, as above): `uns8_h200` 137.3, `srt8_h200` 141.9 GB of 143.8. Rough
400-step run time from the mean of steps 2-3: `srt8_h200` 14.6 h, `uns8_h200` 16.7 h.

- `srt8_h200`'s step 1 training pass took 280.5 s (probe scoring 67.9 s); `uns8_h200`'s step 1, the pod's second
  trainer, took 108.6 s (21.0 s). The cause was not looked into.
- vLLM took 18.5 min to start the first time on this pod (20:36 to 20:54 UTC, compiling from scratch) and 12.5 min
  the second time. The trainer took 22.2 and 19.8 min from start to step 1.

Side by side, steps 2/3 (training-pass seconds per 100k tokens; step time in seconds; 400-step estimate):

| Comparison | Run | Training-pass s per 100k tokens | Step time s | 400 steps |
|---|---|---|---|---|
| Sorting on the H200 | `srt8_h200` / `uns8_h200` | 33.1 / 34.0 vs 39.3 / 39.8 | 134 / 129 vs 149 / 152 | 14.6 vs 16.7 h |
| Sorting on the H100 | `srt4_h100` / `uns4_h100` | 37.2 / 37.4 vs 43.0 / 43.7 | 142 / 141 vs 163 / 144 | 15.7 vs 17.1 h |
| Sorted, H200 at 8 per pass vs H100 at 4 | `srt8_h200` / `srt4_h100` | 33.1 / 34.0 vs 37.2 / 37.4 | 134 / 129 vs 142 / 141 | 14.6 vs 15.7 h |

The last row compares pods, not only GPUs: the two pods also differ in CPU and CPU load.
