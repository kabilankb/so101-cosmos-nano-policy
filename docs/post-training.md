# Post-training and fine-tuning

How the SO-101 policy is produced: staging the data and base weights, the LoRA recipe, the
dataset narrowing that made convergence possible on one GPU, launching, resuming, and
knowing when to stop.

Everything here runs in **cosmos-framework's** environment. The controller in this repo
wraps the launch step (`so101 train`) and everything downstream of it; the staging steps
are one-time and run directly.

- [Prerequisites](#prerequisites)
- [1. Stage the dataset and base checkpoint](#1-stage-the-dataset-and-base-checkpoint)
- [2. Validate the config](#2-validate-the-config)
- [3. The recipe](#3-the-recipe)
- [4. Launch](#4-launch)
- [5. Resume](#5-resume)
- [6. Watch it](#6-watch-it)
- [7. Stop at the right checkpoint](#7-stop-at-the-right-checkpoint)
- [What comes next](#what-comes-next)

---

## Prerequisites

```shell
cd /path/to/cosmos-framework
uv sync --all-extras --group=cu130-train
source .venv/bin/activate
```

Hugging Face access to the dataset (public) and to `nvidia/Cosmos3-Nano`. One 96 GB GPU is
enough for the single-GPU recipe below; the 8-GPU sibling recipe exists but is not what
this pipeline was verified on.

## 1. Stage the dataset and base checkpoint

```shell
uvx hf@latest download --repo-type dataset 5hadytru/so101_bench_sim_6 \
  --local-dir examples/data/so101_bench_sim_6

python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano --checkpoint-path Cosmos3-Nano
```

Start from the **generalist** `Cosmos3-Nano`, not `Cosmos3-Nano-Policy-DROID`. The
DROID-specialised checkpoint isn't in `convert_model_to_dcp`'s named registry, and the
Cosmos Policy method is a single fine-tuning stage with no architecture changes — so the
generalist base is the clean starting point for a new embodiment.

The launcher will convert the base and fetch the Wan2.2 VAE on first use if you skip this,
but it will not download the dataset; it hard-fails if `meta/info.json` is missing.

Three environment variables the launcher reads, all with defaults:

| | default |
| --- | --- |
| `DATASET_PATH` | `examples/data/so101_bench_sim_6` |
| `BASE_CHECKPOINT_PATH` | `examples/checkpoints/Cosmos3-Nano` |
| `WAN_VAE_PATH` | `examples/checkpoints/wan22_vae/Wan2.2_VAE.pth` |

`DATASET_PATH` is mirrored into `SO101_ROOT`, which is what the registered experiment
actually reads — the per-embodiment convention matching upstream's `DROID_ROOT` /
`LIBERO_ROOT`.

## 2. Validate the config

```shell
DATASET_PATH=examples/data/so101_bench_sim_6 \
BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Nano \
SO101_ROOT=examples/data/so101_bench_sim_6 \
PYTHONPATH=. python -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_so101_nano_focus5_1gpu.toml --dryrun
```

Resolves model, optimizer and dataloader wiring without training. `train.py` accepts only
`--sft-toml=<path>`; the experiment is selected by `[job].experiment` inside that TOML.

## 3. The recipe

### The dataset problem, and the narrowing

The full `so101_bench_sim_6` root is 2,228 episodes over 748 instructions. At global batch
32 that is **36,799 iterations per epoch** — about 20 days on one GPU. A 6,000-iteration run
covers **0.16 of one epoch**, and the median instruction is seen about three times.

Episode count was never the constraint. Dilution was:

| Task family | Episodes | Frames | Share |
| --- | ---: | ---: | ---: |
| `Place … in the plastic bin` | 1,878 | 1,117,903 | 89.5% |
| `Move the X <direction>` | 350 | 130,965 | 10.5% |

So the training set is narrowed to the five densest single-object bin-placement
instructions already present in the data — a `keep_tasks` filter applied *before* the
train/val split:

| Instruction | Episodes | Frames |
| --- | ---: | ---: |
| Place the green shoes in the plastic bin | 25 | 11,032 |
| Place the cooking spoon in the plastic bin | 18 | 7,095 |
| Place the cardboard box in the plastic bin | 18 | 6,256 |
| Place the flower pot in the plastic bin | 18 | 5,819 |
| Place the altoids container in the plastic bin | 18 | 4,930 |
| **Total** | **97** | **35,132** |

With `val_ratio = 0.10` that leaves **87 training episodes** and **28,760 valid windows** —
**899 iterations per epoch** instead of 36,799. `max_iter = 4000` is then 4.45 real epochs
in ~48 h, the first configuration that can actually converge on this hardware.

The 899 figure is verified by instantiating the loader, not estimated from frame counts.
Using the dataset's `meta/info.json` `total_frames` (1,248,868) instead would report ~2.7%
of an epoch while the run is at ~96%.

This lives in `action_policy_so101_nano_focus5`, a deepcopy of the base experiment with
`keep_tasks` and `val_ratio` overridden. The original experiment is untouched.

### What is trained

LoRA on the `moe_gen` attention projections, plus **full-rank** training of the action
heads:

```toml
lora_enabled         = true
lora_rank            = 16
lora_alpha           = 32
lora_target_modules  = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
lora_extra_trainable = "action2llm,llm2action,action_modality_embed"
```

Those three modules are the codebase's recognised action-head trio — the layers indexed by
`domain_id`. `checkpoint.keys_to_skip_loading` skips them (plus `action_pos_embed`) so they
**initialise fresh from random**: the public Cosmos3-Nano base has no SO-101 action heads.
That is 16.9M parameters learning an action representation from scratch on 87
demonstrations, and it is the single most important fact about this run's results.

They also get a 5× learning-rate multiplier, since they start from nothing while the
adapters start from a trained backbone:

```toml
[optimizer]
lr = 1.0e-04
keys_to_select = ["lora_", "action2llm", "llm2action", "action_modality_embed"]

[optimizer.lr_multipliers]
action2llm            = 5.0
llm2action            = 5.0
action_modality_embed = 5.0
```

### Everything else that matters

| Setting | Value | Why |
| --- | --- | --- |
| `max_samples_per_batch` × `grad_accum_iter` | 4 × 8 = **32** | global batch on one rank |
| `max_iter` | 4000 | 4.45 epochs of the focused split |
| `cycle_lengths` / `warm_up_steps` | 4000 / 200 | decay over the *real* horizon, not a nominal one |
| `save_iter` | 250 | 16 checkpoints × ~30.6 GB ≈ 490 GB |
| `precision` | bfloat16 | with FusedAdam fp32 master weights, eps 1e-8 |
| `ema.enabled` | false | EMA doubles resident model memory for no benefit here |
| `activation_checkpointing.mode` | full | 96 GB is tight; `fmha` ops are saved |
| `data_parallel_shard_degree` | 1 | single GPU, no FSDP sharding |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | fragmentation alone can fail a step at this budget |

Training and serving apply **identical** preprocessing: `So101LeRobotDataset` runs the same
`ActionTransformPipeline` class that `action_policy_server_robolab.py` uses at inference.
Row 0 of the action tensor is the current `observation.state` and rows 1..32 are future
joint-position targets, with video windowed to a matching 33 frames — so the layout the
server sends at inference is the layout the model trained on.

One documented discrepancy: `loss_scale` is **1.0** here, not DROID's 10.0. With
`action_loss_weight` at 10.0 the action loss dominates roughly 10:1. A `--dryrun` confirms
the resolved value. To match DROID's balance, pass it through `EXTRA_TAIL_OVERRIDES`.

## 4. Launch

Through this package:

```shell
so101 train              # runs the focus5 launcher
so101 train --dry-run    # print the command and environment, run nothing
so101 train --detach     # background job, tracked by `so101 jobs`
```

Or directly:

```shell
bash examples/launch_sft_action_policy_so101_nano_focus5_1gpu.sh
```

Which launcher `so101 train` invokes is the `launcher` setting in `config.py` — point it at
the 8-GPU or full-root recipe if that's what you want.

A smoke run first is worth the ten minutes; it measures peak memory and iteration time
without committing two days:

```shell
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 trainer.grad_accum_iter=1 \
  checkpoint.save_iter=100000 trainer.callbacks.device_monitor.every_n=1" \
  bash examples/launch_sft_action_policy_so101_nano_focus5_1gpu.sh
```

**Check the GPU is free first** — `so101 doctor`. A stale policy server holds ~32 GB and a
wedged Isaac Sim ~10 GB, and both outlive their parent. Killing two of them once took
iteration time from 43.5 s to 37.1 s.

Output lands in
`outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_focus5_1gpu`, with DCP
checkpoints under `checkpoints/iter_<N>/`.

## 5. Resume

Resuming is automatic and needs no flag: the checkpointer prefers
`checkpoints/latest_checkpoint.txt` over `checkpoint.load_path`, so relaunching the same
job name continues from saved model + optimizer + scheduler state rather than restarting.

`tools/resume_training.sh` in cosmos-framework wraps that for unattended recovery. It
refuses to start a second trainer, validates the checkpoint `latest_checkpoint.txt` points
at and rolls back to the previous complete one if a power cut truncated a write, and
exports the `LD_LIBRARY_PATH` cuBLAS fix. It is wired to a systemd user unit
(`cosmos-so101-train`, `Linger=yes`), verified recovering 5 s after boot.

## 6. Watch it

```shell
so101 web            # control page, :8800
so101 status         # one-shot
```

The trainer emits **two** line formats — `Iteration N: Hit counter: …` for the first 50
iterations and `N : iter_speed …` thereafter. Parsing only the first makes a live run look
stalled at iteration 550. `logparse.parse_training` handles both.

On GPU telemetry: `T.Limit` values from `nvidia-smi -q` are *margins*, not temperatures —
Blackwell reports degrees remaining before a limit engages, so `-2` means two degrees of
headroom. `SW Thermal Slowdown` / `SW Power Cap` name which limiter is currently costing
clock speed. On the reference box the card sits at ~92 °C and 594/600 W, running at 63–78%
of its 3090 MHz boost.

## 7. Stop at the right checkpoint

**Loss does not predict task success on this run.** Both directions:

- At iteration 1500 the curve looked plateaued and the working assumption was that more
  data, not more steps, was the lever. Wrong — the final LR anneal took loss from 0.193 to
  0.113 and produced the first checkpoint that ever completed the task.
- The *lowest*-loss checkpoint (4000, loss 0.113) scores **0/20**. Checkpoint 3750, at
  higher loss, scores **11/262**.

| Epochs | Mean loss | |
| --- | ---: | --- |
| 0.00 – 0.22 | 4.97 | action heads leaving random init |
| 0.22 – 0.44 | 0.38 | the bulk of the learning |
| 0.44 – 1.11 | 0.269 → 0.222 | plateau |
| 1.11 – 1.33 | 0.193 | local best |
| 1.33 – 1.78 | 0.202 → 0.223 | drift |
| → 4.45 | 0.113 | LR anneal in the final quarter |

So do not pick a checkpoint from the curve. Evaluate several and pick from
`so101 checkpoints`. The useful window on this run was narrow — 3500 and 4000 are both
0/20 while 3750 works — so consider a smaller `save_iter` through the region you expect to
matter.

Budget for it: **each evaluated checkpoint costs ~88 GB** across training (~29 GB), merged
(~29 GB) and exported (~30 GB) copies. `so101 clean` reclaims the merged intermediates once
an export exists; they regenerate in about a minute.

## What comes next

Merge the adapters, export to safetensors, serve, and evaluate — see the
[README](../README.md). In short:

```shell
so101 merge  --iter 3750
so101 export --iter 3750
so101 serve  --iter 3750
so101 warmup
so101 eval
```
