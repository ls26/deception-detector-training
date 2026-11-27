# 261126 GPU tests: tables (written by results.py)

## Memory test (memory/memory.json)

| Setting | Peak reserved GiB | Peak allocated GiB | Free GiB after the pass |
|---|---|---|---|
| low_memory_passes on, 4 x (1024 + 1024) | 29.69 | 28.11 | 18.49 |
| low_memory_passes on, 8 x (1024 + 1024) | 37.50 | 34.66 | 10.41 |
| low_memory_passes off, 4 x (1024 + 1024) | out of memory | | |
| low_memory_passes off, 8 x (1024 + 1024) | out of memory | | |

Free GPU memory before the test started CUDA: 49.38 GiB; after: 48.40 GiB; model loaded: 17.66 GiB.

- numbers reference_logprobs: {'answer_tokens': 1524, 'max_abs_difference': 0.0, 'mean_abs_difference': 0.0, 'correlation': 1.0, 'identical': True}
- numbers policy_logprobs: {'answer_tokens': 1524, 'max_abs_difference': 0.0, 'mean_abs_difference': 0.0, 'correlation': 1.0, 'identical': True}
- numbers probe inputs bit for bit equal: True
- numbers reference_hidden inputs bit for bit equal: True

## 3-step runs

### lmp_on4 (W&B 4jgckzbt; EXIT 0; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 43.6 | 16.3 | 16.1 | 19.8 | 39.8 | 96.6 |  | 0.9994652196764946 | 0.006340055603381356 | 1024 |
| 2 | 42.4 | 14.8 | 15.8 | 20.0 | 38.1 | 93.2 | 136 | 0.999446811620146 | 0.0070207702293380025 | 1024 |
| 3 | 43.7 | 15.8 | 16.6 | 20.4 | 40.2 | 97.9 | 141 | 0.9994725324213505 | 0.008241388136712047 | 1024 |

Before step 1: 0.5 min; after the last step: 2.6 min; trainer subprocess 10.1 min; peak GPU memory by nvidia-smi 65.8 of 79.6 GiB.

### lmp_off4 (W&B zz7qdwgm; EXIT 0; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42.6 | 23.0 | 13.3 | 21.1 | 41.7 | 107.7 |  | 0.9994755391962826 | 0.006910026686999582 | 1024 |
| 2 | 41.0 | 22.0 | 12.6 | 19.3 | 38.0 | 99.5 | 140 | 0.9994252952747047 | 0.006322205039570594 | 1024 |
| 3 | 45.6 | 23.7 | 13.7 | 21.5 | 41.7 | 109.7 | 155 | 0.9994724122807384 | 0.006048333466336819 | 1024 |

Before step 1: 9.3 min; after the last step: 21.4 min; trainer subprocess 38.1 min; peak GPU memory by nvidia-smi 79.1 of 79.6 GiB.

### lmp_on8 (W&B lwts2w0j; EXIT 0; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 39.4 | 14.3 | 13.6 | 15.7 | 40.1 | 88.2 |  | 0.9994790647178888 | 0.01720438406915714 | 1024 |
| 2 | 44.7 | 15.4 | 15.3 | 18.1 | 36.0 | 89.6 | 134 | 0.9994762185961008 | 0.015536589966050562 | 1024 |
| 3 | 36.5 | 13.4 | 12.9 | 15.3 | 30.2 | 75.9 | 112 | 0.9994797110557556 | 0.017929030582429673 | 1024 |

Before step 1: 0.5 min; after the last step: 2.7 min; trainer subprocess 9.4 min; peak GPU memory by nvidia-smi 72.6 of 79.6 GiB.

### lmp_on4_main (W&B 6lkukfwv; EXIT 0; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42.5 | 15.7 | 15.8 | 19.8 | 38.9 | 94.7 |  | 0.9993998245336115 | 0.006527173752255866 | 1024 |
| 2 | 42.4 | 15.9 | 16.6 | 21.3 | 40.5 | 98.9 | 142 | 0.9994735419750214 | 0.007571667843251251 | 1024 |
| 3 | 37.0 | 13.4 | 14.3 | 18.4 | 34.8 | 84.6 | 121 | 0.999472585041076 | 0.007374226392493521 | 1024 |

Before step 1: 0.5 min; after the last step: 2.7 min; trainer subprocess 9.9 min; peak GPU memory by nvidia-smi 65.8 of 79.6 GiB.

### lmp_mbpp_on4 (W&B ae3enf03; EXIT 0; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31.4 | 18.4 | 12.7 | 17.5 | 32.5 | 232.5 |  | 0.9992785854265094 | 0.00836478252619599 | 1024 |
| 2 | 30.2 | 16.8 | 12.4 | 16.9 | 31.8 | 107.5 | 138 | 0.999273409601301 | 0.009807986035348049 | 1024 |
| 3 | 30.1 | 17.7 | 12.8 | 17.3 | 32.3 | 102.7 | 133 | 0.9992997632361948 | 0.010329599310641924 | 1024 |

Before step 1: 0.7 min; after the last step: 2.5 min; trainer subprocess 12.1 min; peak GPU memory by nvidia-smi 66.9 of 79.6 GiB.

### fzmbpp9b_s43 (261125, settings off; steps 1-3 of 300, H100 pod 7c244d9935a2) (W&B 6mn5758c; EXIT 1; out of memory: False)

| Step | Generation s | Rewards + probe scoring s | Reference pass s | Policy pass s | Backward s | Training pass s | Step-to-step s | Log-prob correlation | Padding share | Longest answer |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 34.6 | 34.4 | 11.1 | 17.4 | 94.9 | 311.9 |  | 0.9992690468207002 | 0.00929377653515584 | 1024 |
| 2 | 31.9 | 19.6 | 11.1 | 16.4 | 32.6 | 102.1 | 134 | 0.9993068776093423 | 0.00869422224108296 | 1024 |
| 3 | 34.2 | 19.6 | 11.5 | 17.9 | 33.2 | 103.0 | 137 | 0.9993045986630023 | 0.008162897377857026 | 1024 |

Before step 1: 5.2 min; after the last step: 9.7 min; trainer subprocess 653.5 min; peak GPU memory by nvidia-smi nan of 79.6 GiB.

## Probe penalty check (W&B rewards/detector against the run's saved answers)

| Run | Step | Answers in the step file | Penalty recomputed | W&B rewards/detector |
|---|---|---|---|---|
| lmp_on4 | 1 | 512 | -0.17344 | -0.17344 |
| lmp_on4 | 2 | 512 | -0.09429 | -0.09429 |
| lmp_on4 | 3 | 512 | -0.17102 | -0.17102 |
| lmp_off4 | 1 | 512 | -0.12963 | -0.12963 |
| lmp_off4 | 2 | 512 | -0.13427 | -0.13427 |
| lmp_off4 | 3 | 512 | -0.13725 | -0.13725 |
| lmp_on8 | 1 | 512 | -0.12924 | -0.12924 |
| lmp_on8 | 2 | 512 | -0.11098 | -0.11098 |
| lmp_on8 | 3 | 512 | -0.16576 | -0.16576 |
| lmp_on4_main | 1 | 512 | -0.14694 | -0.14694 |
| lmp_on4_main | 2 | 512 | -0.13857 | -0.13857 |
| lmp_on4_main | 3 | 512 | -0.13232 | -0.13232 |
| lmp_mbpp_on4 | 1 | 512 | -0.51420 | -0.51420 |
| lmp_mbpp_on4 | 2 | 512 | -0.48923 | -0.48923 |
| lmp_mbpp_on4 | 3 | 512 | -0.48969 | -0.48969 |

