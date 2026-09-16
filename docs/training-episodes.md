# Training episodes per SO-101 run

How many demonstration episodes each SO-101 post-training run learned from, and how many
times it went through them.

All runs read the same LeRobot dataset, `5hadytru/so101_bench_sim_6`: **2,228 episodes,
1,248,868 frames, 30 fps**, 742 distinct instructions. Each run filters it down with
`keep_tasks` / `max_episodes_per_task`, then holds out a seeded per-episode validation split.

Counts were measured on 2026-09-16 by instantiating `SO101LeRobotDataset` with each run's own
filters and split (`seed=0`), not estimated. A *window* is one training sample: 33 consecutive
frames (the current state plus a 32-step action chunk), so an episode of *n* frames gives
*n* − 32 windows.

## Summary

| Run | Model | Instructions | Train episodes | Held-out episodes | Train frames | Train windows | Iterations | Epochs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `action_policy_so101_sim6_1gpu` (full dataset) | Cosmos3-Nano | 742 | **2,206** | 22 | 1,236,834 | 1,166,242 | ~1,150 logged of 6,000 planned | ~0.03 |
| `action_policy_so101_focus5_1gpu` | Cosmos3-Nano | 5 | **87** | 10 | 31,544 | 28,760 | 4,000 (done) | 4.45 |
| `action_policy_so101_edge_focus5_multi_1gpu` | Cosmos3-Edge | 6 | **128** | 14 | 59,481 | 55,385 | 7,000 | 4.04 |
| `action_policy_so101_single_bin_from3750_1gpu` | Cosmos3-Nano | 34 | **379** | 20 | 142,442 | 130,314 | stopped after a few iterations | ~0 |

Epochs = iterations × global batch 32 ÷ train windows. Every run used global batch 32.

## Checkpoints and how much data each had seen

| Checkpoint | Train episodes | Epochs at checkpoint | Benchmark (`So101Bench-Bin-v0`, single object) |
| --- | ---: | ---: | --- |
| Nano focus5, iter 3750 | 87 | 4.17 | **17 / 542 (3.1%)**, plus 0 / 25 on 2026-09-16 |
| Nano focus5, iter 4000 | 87 | 4.45 | 0 / 20 |
| Edge focus5_multi, iter 1500 | 128 | 0.87 | 0 / 50 |
| Edge focus5_multi, iter 2500 | 128 | 1.44 | 0 / 100 |
| Edge focus5_multi, iter 6000 | 128 | 3.47 | not evaluated |
| Edge focus5_multi, iter 6500 | 128 | 3.76 | running (0 / 13 at the time of writing) |
| Edge focus5_multi, iter 7000 | 128 | 4.04 | not evaluated |

## Per-instruction breakdown

### Nano focus5: 87 train / 10 held out

| Instruction | Train | Held out | Total |
| --- | ---: | ---: | ---: |
| Place the green shoes in the plastic bin | 25 | 0 | 25 |
| Place the altoids container in the plastic bin | 17 | 1 | 18 |
| Place the cardboard box in the plastic bin | 17 | 1 | 18 |
| Place the cooking spoon in the plastic bin | 15 | 3 | 18 |
| Place the flower pot in the plastic bin | 13 | 5 | 18 |
| **Total** | **87** | **10** | **97** |

### Edge focus5_multi: 128 train / 14 held out

The multi-object instruction has 875 episodes in the dataset; `max_episodes_per_task=45` keeps
the 45 with the lowest episode index so it cannot swamp the five single-object instructions.

| Instruction | Train | Held out | Total |
| --- | ---: | ---: | ---: |
| Place each object in the plastic bin (4-object scenes, capped) | 43 | 2 | 45 |
| Place the green shoes in the plastic bin | 21 | 4 | 25 |
| Place the altoids container in the plastic bin | 17 | 1 | 18 |
| Place the cooking spoon in the plastic bin | 16 | 2 | 18 |
| Place the flower pot in the plastic bin | 16 | 2 | 18 |
| Place the cardboard box in the plastic bin | 15 | 3 | 18 |
| **Total** | **128** | **14** | **142** |

The two runs draw different held-out episodes, because the split is taken over each run's own
filtered episode list.

### Nano single_bin (stopped): 379 train / 20 held out

Every single-object "Place the X in the plastic bin" instruction in the dataset: 34 instructions,
399 episodes. The five focus5 instructions are 92 of the 379 training episodes (green shoes 25,
cooking spoon 18, altoids container 17, cardboard box 16, flower pot 16); the other 29 objects
have 6–16 each.

### Full dataset: 2,206 train / 22 held out

All 742 instructions. 89.5% of frames are "Place ... in the plastic bin" episodes and 10.5% are
"Move the X \<direction>" episodes; the median instruction has 3 episodes.

## Notes

- **Iteration counts for the full-dataset run** come from its surviving log
  (`outputs/train/logs/action_policy_so101_nano_1gpu_sft.log`, last logged iteration 1,150).
  Its checkpoints are no longer on disk.
- **Edge iterations 0–5500** ran on one RTX PRO 6000 (batch 16 × 2 accumulation steps);
  **5500–7000** were resumed on 2× RTX PRO 6000 on Brev (batch 16 per GPU). The data, split and
  global batch were identical.
- **Held-out episodes are not used by the benchmark.** The benchmark generates its own scenes
  from `tasks/focus5.jsonl` in Isaac Lab; it does not replay dataset episodes.

## Edge run: single-GPU phase vs Brev phase

The Edge run (`action_policy_so101_edge_focus5_multi_1gpu`) was trained in two phases. Both
used the same data, split and hyperparameters; the table was checked by diffing the
`config.yaml` each phase wrote. The Brev phase resumed from the saved `iter_000005500`
training state (model, optimizer, scheduler), so the learning-rate schedule continued where it
stopped.

### What differed

| | Single GPU (workstation) | Brev |
| --- | --- | --- |
| Iterations | 0 → 5,500 (reached 5,738; the steps after the last checkpoint were lost) | 5,500 → 7,000 |
| Hardware | 1× RTX PRO 6000 Blackwell, 96 GB | 2× RTX PRO 6000 Blackwell, 96 GB each (MassedCompute, $5.26/h) |
| Batch per GPU step (`max_samples_per_batch`) | 16 | 16 |
| GPUs (`data_parallel_shard_degree`) | 1 | 2 |
| Gradient accumulation (`grad_accum_iter`) | 2 | 1 |
| **Global batch** | **16 × 1 × 2 = 32** | **16 × 2 × 1 = 32** |
| Speed | ~21 s / iteration | ~9.5 s / iteration |
| Samples processed | 176,000 (3.18 epochs) | 48,000 (0.87 epoch) |
| Dates | 2026-09-11 → 2026-09-13 | 2026-09-16, 18:33 → ~22:31 IST |

Only machine-specific paths (dataset root, VAE path, output location) and the three parallelism
fields above differ between the two `config.yaml` files.

### Data (identical in both phases)

| Setting | Value |
| --- | --- |
| Dataset | `so101_bench_sim_6` (2,228 episodes, 30 fps) |
| `keep_tasks` | green shoes, cardboard box, altoids container, flower pot, cooking spoon (single object) + "Place each object in the plastic bin" |
| `max_episodes_per_task` | 45 |
| `val_ratio` / split seed | 0.10 / 0 |
| **Train episodes** | **128** (55,385 windows) |
| Held-out episodes | 14 |
| `chunk_length` / `fps` | 32 / 30.0 |
| `use_state` | true (row 0 = current state) |
| `action_normalization` | minmax (`so101_lerobot_stats.json`) |
| `resolution` | 480 (`concat_view`: wrist camera over overhead camera) |
| `mode` | wam |
| `cfg_dropout_rate` | 0.1 |
| `format_prompt_as_json` | false |
| `iterable_shuffle` / `episode_shuffle_seed` | true / 42 |
| DataLoader workers per rank | 8 |

### Model and optimisation (identical in both phases)

| Setting | Value |
| --- | --- |
| Base checkpoint | `nvidia/Cosmos3-Edge-Policy-DROID`, DROID action-head row (domain 8) copied into SO-101 (domain 22) |
| `keys_to_skip_loading` | `net_ema.`, `lora_` (action heads loaded, not re-initialised) |
| Precision | bfloat16 |
| LoRA | rank 64, alpha 128, on `q/k/v/o_proj_moe_gen` |
| Extra trainable | `action2llm`, `llm2action`, `action_modality_embed` |
| Optimizer | FusedAdam, betas (0.9, 0.99), eps 1e-8, weight decay 0.05 |
| `keys_to_select` | `lora_`, `action2llm`, `llm2action`, `action_modality_embed` |
| Learning rate | 1e-4, action-head multipliers 1.0 |
| Scheduler | LambdaLinear, 200 warm-up steps, linear decay to 0 over 7,000 iterations |
| Gradient clipping | 1.0 |
| Loss | `loss_scale` 10.0, `action_loss_weight` 10.0, action timesteps logit-normal |
| Flow-matching shift | 5 at 480p |
| EMA | disabled |
| Activation checkpointing | selective (keeps `fmha`) |
| Parallelism | FSDP |
| `max_iter` / `save_iter` / seed | 7,000 / 500 / 42 |
