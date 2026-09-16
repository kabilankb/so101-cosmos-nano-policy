# SO-101 Action-Policy Post-Training on Cosmos3-Edge

Edge-tier counterpart of the SO-101 Nano pipeline in
[`so101_policy_finetune.md`](./so101_policy_finetune.md). Same robot, same
dataset, same single 96 GB GPU; different model tier and a different
checkpoint-loading policy.

Experiments: `action_policy_so101_edge`, `action_policy_so101_edge_focus5`,
`action_policy_so101_edge_focus5_multi`
(`cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_so101_edge.py`).
Launcher: `examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh`.
Recipe: `examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml`.

______________________________________________________________________

## Why Edge, honestly

Edge is the **smaller** tier, not the more accurate one. On the public RoboLab
suite the released DROID policies score **Edge 22.9% vs Nano 36.8%**. If you
read only the headline numbers, moving from Nano to Edge looks like a
downgrade, and on 64 nodes of GB200 it would be.

The reason it is the right move *on this box* is capacity per unit of VRAM:

| | Nano | Edge |
| --- | --- | --- |
| Reasoner backbone | Qwen3-VL-8B | Nemotron-2B-Dense-VL |
| Native resolution | 720 | 480 (matches SO-101 data) |
| What fit on 1x96 GB | rank-16 LoRA, microbatch 4 | rank-32 LoRA, microbatch 8 |
| Action heads at init | random | **trained** (from the DROID policy) |

The SO-101 focus5+multi mix is 128 training episodes. At that size the binding
constraint is not the pretrained ceiling, it is how much of the model can
actually be adapted and how good a starting point the action path has. Edge
wins on both on this hardware.

## The three changes that target the accuracy problem

### 1. Warm action heads (the big one)

`action_policy_so101_nano` initialises from base `nvidia/Cosmos3-Nano`, whose
action heads carry no manipulation signal. It therefore lists them in
`checkpoint.keys_to_skip_loading` and they start from **random init** — the
upstream DROID/LIBERO convention. The Nano run then spent its entire budget
relearning `action2llm` / `llm2action` from scratch on 128 episodes.

The Edge recipes initialise from **`nvidia/Cosmos3-Edge-Policy-DROID`**, which
is already a post-trained manipulation policy, and deliberately **keep** its
action heads:

```toml
[checkpoint]
keys_to_skip_loading = ["net_ema.", "lora_"]   # action heads NOT listed
```

**That alone is not enough, and this is the part that is easy to get wrong.**
`action2llm` / `llm2action` are `DomainAwareLinear`
(`model/generator/mot/domain_aware_linear.py`): their weights live in an
`nn.Embedding(num_embodiment_domains, out*in)`, so **every embodiment owns a
private row**. DROID is domain 8, SO-101 is domain 22.

Measured on the released checkpoint (diffusers names `action_proj_in` /
`action_proj_out`):

| tensor | row 8 (DROID) | row 22 (SO-101) | every other row |
| --- | --- | --- | --- |
| `action_proj_in.fc.weight` | 10.67 | **6.44** | 6.42–6.45 |
| `action_proj_in.bias.weight` | 1.56 | **0.000** | 0.000 |
| `action_proj_out.fc.weight` | 2.33 | **1.139** | 1.135–1.139 |
| `action_proj_out.bias.weight` | 0.087 | **0.000** | 0.000 |

Row 22 is still at `xavier_uniform_` init with an exactly-zero bias. So loading
the DROID policy hands SO-101's action head **nothing** the base checkpoint
would not have given it. The only part of the action path that transfers for
free is `action_modality_embed` — a single shared `[2048]` vector, *not* a
per-domain table.

`tools/transplant_domain.py` closes the gap by copying row 8 into row 22 before
DCP conversion:

```shell
python tools/transplant_domain.py \
    --src examples/checkpoints/Cosmos3-Edge-Policy-DROID \
    --dst examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
    --src-domain 8 --dst-domain 22
```

Both embodiments are single-arm absolute `joint_pos` policies padded into the
same 64-wide action space (DROID 7 joints + gripper, SO-101 5 joints +
gripper), so the token↔hidden mapping is largely shared structure even though
per-column joint semantics differ. Only shard 2 carries the per-domain tensors,
so everything else is hardlinked.

This gives three initialisations worth comparing:

| `BASE_CHECKPOINT_PATH` | SO-101 action head | LR multipliers |
| --- | --- | --- |
| `...-so101init-dcp` | **warm** (DROID row 8) | 1.0 (default) |
| `...-DROID-dcp` | cold (xavier) | set back to 5.0 |
| base Edge + `SO101_EDGE_COLD_HEADS=1` | cold, Nano convention | 5.0 |

Keeping the 5x multiplier on *warm* weights would destroy them within a few
hundred steps, which is why the shipped TOML uses a uniform 1.0.

### 2. Correct flow-matching shift

`omni_mot_model.py` selects the training `shift` by `model.config.resolution`:

```python
shift = dict(shift_config)[self.config.resolution]   # {"256": 3, "480": 5, "720": 10}
```

`NANO_MODEL_CONFIG` is `resolution="720"` and the SO-101 Nano recipe never
overrode it, while `SO101LeRobotDataset` emits `resolution="480"`. The Nano runs
therefore trained with **shift=10, the 720p schedule, on 480p data**.
`EDGE_MODEL_CONFIG` is natively `"480"`, so Edge gets shift=5 with no override.

This is worth back-porting to the Nano recipe independently of anything Edge —
see [Back-port to Nano](#back-port-to-nano).

### 3. More adapter rank

`lora_rank` 32 / `lora_alpha` 64, against Nano's 16/32, at microbatch 8 instead
of 4. Global batch stays 32 (8 x 4 accumulation) so iteration counts remain
comparable to the Nano run.

**LoRA targets are the four attention projections only**, same as Nano. This is
a constraint of the MoT naming, not a choice: the decoder layer suffixes only
its *direct* children with `_moe_gen` (`q/k/v/o_proj_moe_gen`, `mlp_moe_gen`,
`norm_moe_gen`). The gen MLP's own children are plain `gate_proj` / `up_proj` /
`down_proj`, identical in name to the frozen understanding tower's, and
`lora.py` matches by **exact child name anywhere in the tree**. So
`gate_proj_moe_gen` matches nothing (silent warning, zero modules injected) and
bare `gate_proj` would also adapt the und tower, which the recipe keeps frozen.

If rank-32 underfits, the next lever is adding `mlp_moe_gen` to
`lora_extra_trainable` — that trains the gen MLP at full rank and *is* correctly
gen-scoped, because `lora_extra_trainable` is a substring match on parameter
names. Re-measure memory with a smoke run first.

______________________________________________________________________

## Pipeline

### Step 1 — Stage the checkpoints

```shell
hf download nvidia/Cosmos3-Edge-Policy-DROID \
    --local-dir examples/checkpoints/Cosmos3-Edge-Policy-DROID \
    --exclude "assets/*" "images/*"

# Optional, for the cold-head A/B or base-model ablation:
hf download nvidia/Cosmos3-Edge \
    --local-dir examples/checkpoints/Cosmos3-Edge \
    --exclude "assets/*" "images/*"
```

Roughly 9 GB each. Both ship in diffusers layout (`transformer/`, `vae/`,
`vision_encoder/`).

Note: `edge_model_config.py` names `nvidia/Cosmos3-Edge-Reasoner` as its
backbone, and **that repo does not exist on the Hub**. It is not a problem —
`inference/common/checkpoints.py` registers `Cosmos3-Edge-Reasoner-590c1c0`
pointing at the public `nvidia/Cosmos3-Edge` repo, whose root safetensors index
carries the canonical Edge reasoner weights. There is no separate reasoner
download.

### Step 2 — Transplant the action-head row, then convert to DCP

Training loads DCP, not diffusers. Both steps run on CPU.

```shell
# Warm SO-101 action head (the default the launcher expects)
python tools/transplant_domain.py \
    --src examples/checkpoints/Cosmos3-Edge-Policy-DROID \
    --dst examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
    --src-domain 8 --dst-domain 22

python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
    -o examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init-dcp

# Cold control, for the A/B
python -m cosmos_framework.scripts.convert_model_to_dcp \
    --checkpoint-path examples/checkpoints/Cosmos3-Edge-Policy-DROID \
    -o examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp
```

Each DCP is ~6.3 GB.

### Step 3 — Smoke run (do this before any long horizon)

Ten iterations, no accumulation. This is what validates that the LoRA targets
resolve against the Nemotron backbone, that the action heads actually loaded,
and what peak memory and seconds-per-iteration really are:

```shell
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 trainer.grad_accum_iter=1 \
  checkpoint.save_iter=100000 trainer.callbacks.device_monitor.every_n=1" \
  bash examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh
```

Check in the log:

- `Injecting LoRA ... targets=[...]` followed by a non-zero replacement count.
  A `replaced 0 modules` warning or a `target modules not found` warning means
  the run will train nothing but the action heads.
- **Do NOT use starting loss to check the warm start.** It is tempting and it
  does not work on Edge: `loss_scale = 10.0` weights the vision flow-matching
  term, which dominates the scalar, and per-batch variance swamps the action
  contribution. The measured 10-iteration smoke ran 13.50, 13.46, 9.36, 12.60,
  11.14, 12.66, 10.65, 11.03, 14.04, 10.95 — a range far wider than any
  warm/cold gap.

  Check the weights directly instead. Read row 22 back out of the first saved
  checkpoint and compare it to row 8 and to an untouched row:

  ```python
  import torch, torch.distributed.checkpoint as dcp
  from torch.distributed.checkpoint import FileSystemReader
  p = "outputs/train/.../checkpoints/iter_000000010/model"
  md = FileSystemReader(p).read_metadata()
  keys = ["net.action2llm.fc.weight", "net.action2llm.bias.weight",
          "net.llm2action.fc.weight", "net.llm2action.bias.weight"]
  sd = {k: torch.empty(md.state_dict_metadata[k].size, dtype=torch.float32) for k in keys}
  dcp.load(sd, checkpoint_id=p)
  for k, v in sd.items():
      n = v.float().norm(dim=1)
      print(f"{k:30s} row8={n[8]:.4f} row22={n[22]:.4f} row0={n[0]:.4f}")
  ```

  Warm looks like this (row 22 == row 8, row 0 left behind)::

      net.action2llm.fc.weight       row8=10.6676 row22=10.6676 row0=6.4187
      net.action2llm.bias.weight     row8= 1.5603 row22= 1.5603 row0=0.0000
      net.llm2action.fc.weight       row8= 2.3256 row22= 2.3255 row0=1.1386

  If row 22 matches row 0 instead, the warm start did not take — check that
  `BASE_CHECKPOINT_PATH` is the `-so101init-dcp` conversion, that the action
  heads are absent from `keys_to_skip_loading`, and that `SO101_EDGE_COLD_HEADS`
  is unset.
- Peak memory from `device_monitor`. Drop `max_samples_per_batch` to 4 if it
  exceeds ~85 GB.

### Step 4 — Full run

```shell
bash examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh
```

`max_iter = 7000` = 4.05 epochs — the window where the Nano focus5 run produced
its only working checkpoint.

**Important:** the trainer auto-resumes from `latest_checkpoint.txt` in the job's
output directory. A smoke run under the same `job.name` will therefore be
resumed from rather than started over, at whatever LoRA rank the smoke used.
Either give smoke runs their own `job.name` (the sizing smoke above uses
`job.name=action_policy_so101_edge_sizing`) or clear
`outputs/train/cosmos3_action/action_sft_sim/<job.name>/` before the real launch.

## Measured on this box (1x RTX PRO 6000 Blackwell, 96 GB)

Everything below is measured, not estimated.

| | Nano focus5+multi | **Edge focus5+multi** |
| --- | --- | --- |
| LoRA rank / alpha | 16 / 32 | **64 / 128** |
| Microbatch x accum | 4 x 8 | **16 x 2** |
| Global batch | 32 | 32 |
| LoRA modules wrapped | — | 112 (28 layers x 4 proj) |
| Trainable params | — | 25,690,112 LoRA + 8,458,240 heads = **1.006%** |
| Frozen base params | — | 3,361,198,784 |
| Peak GPU memory | (fit in 96 GB) | **26.1 GB** (66.6 GB free) |
| Seconds / iter @ gb32 | ~40 s | **~20 s** |
| 7000 iters | ~78 h | **~39 h** |
| Checkpoint size | ~29 GB | **6.5 GB** |
| 14 checkpoints | ~406 GB | **~91 GB** |
| Dataset | 128 train eps / 55,385 windows | same |
| Iters / epoch | 1,731 | 1,731 |

Two things that follow from the memory number: the card is **nowhere near
memory-bound** (26 of 96 GB), and it *is* compute-bound — 100% util pinned
against the 600 W power cap, which is why switching `full` -> `selective`
activation checkpointing bought no throughput. If the policy underfits, spend
the idle 66 GB on trainable capacity (see `mlp_moe_gen` above), not on batch
size.

### Step 5 — Evaluate an epoch in the Isaac Lab sim

`tools/edge_eval_epoch.py` runs the whole closed loop for one checkpoint and
tears it down again: merge LoRA → export safetensors → start the policy
**server** → drive it with the Isaac Lab **client** → parse episode outcomes →
stop the server → write `result.json`.

The two halves need different interpreters — the server runs in the
cosmos-framework venv, the client under Isaac Sim's Kit python (the only
interpreter on this box with `omni` importable) — which is why this is an
orchestrator rather than a single process.

```shell
python3 tools/edge_eval_epoch.py --epoch 1          # nearest checkpoint to epoch 1
python3 tools/edge_eval_epoch.py --iter 3500 --gui  # exact iteration, viewer open
python3 tools/edge_eval_epoch.py --all --defer-until-training-done
```

Results land in `outputs/edge_eval/iter_XXXXXXXXX/` (`merge.log`, `export.log`,
`server.log`, `eval.log`, `result.json`) and appear automatically in the
dashboard's closed-loop eval panel.

Guards, all of which fire before anything touches the GPU:

- **Concurrency.** Refuses to start while the trainer is live unless you pass
  `--defer-until-training-done` (queues behind it) or `--allow-concurrent`
  (accepts the slowdown). Isaac Sim plus the policy server on the same card
  measurably slows training and raises temperature.
- **LoRA scale.** Asserts `lora_rank`/`lora_alpha` against the TOML before
  merging. Merging at the wrong scale produces a valid-looking but wrong policy
  that fails silently — there is no runtime error to catch it later.
- **Thermal.** `--max-gpu-temp 88` refuses to start on a hot card.
- **Server memory.** Always passes `--device-memory-bytes 50000000000`; without
  it the server sizes itself against the full 97 GB via NVML and collides with
  Isaac Sim.

Merge and export are idempotent and reused across runs; `--force` redoes them.

To drive the server by hand instead, the SO-101 flags are:

```shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path <model_export_dir> --port 8000 \
    --domain-name so101 --action-dim 6 --arm-joint-dim 5 \
    --action-space joint_pos --conditioning-fps 30 --no-flip-gripper \
    --action-normalization minmax \
    --normalizer-stats-path cosmos_framework/data/generator/action/normalizer_stats/so101_lerobot_stats.json \
    --no-guardrails --device-memory-bytes 50000000000
```

Every one of those domain flags fails *silently* if omitted — see
[`action_policy_robolab_server.md`](./action_policy_robolab_server.md#so-101-policy-checkpoint).

## Operating the run

| | |
| --- | --- |
| Telemetry dashboard | <http://<lan-host>:8810> — `cosmos-edge-dashboard.service`, **read-only**, safe on the LAN |
| Eval console | <http://127.0.0.1:8811> — `cosmos-edge-console.service`, **runs commands**, localhost only |
| Auto-resume | `cosmos-so101-edge-train.service` → `tools/resume_training_edge.sh` |
| Live log | `outputs/edge_setup/train_edge*.log` |
| Eval results | `outputs/edge_eval/iter_*/result.json` |
| Job logs | `outputs/edge_eval_jobs/*.log` |

### Eval console (port 8811)

`tools/edge_eval_console.py` is the point-and-click front end for the
server/client pair. It shows the Isaac Lab preflight (Kit python,
`openpi-client`, `omni`, `so101_bench`, task files, saved layouts, GPU headroom),
lists every checkpoint with its epoch number and past success rate, and launches:

| button | what it runs |
| --- | --- |
| Evaluate (headless) | `edge_eval_epoch.py --iter N` |
| Evaluate with viewer | same, `--gui` (Isaac Sim viewport) |
| Serve only | just the policy server, so you can drive `cosmos3_eval.py` by hand |
| Sweep all unevaluated | `edge_eval_epoch.py --all` |

The "run now" checkbox is the concurrency switch: unchecked it passes
`--defer-until-training-done` and the job waits for the trainer to exit;
checked it passes `--allow-concurrent`. A banner warns whenever training is
live or a preflight check is failing.

It binds **127.0.0.1 and refuses any other interface** without
`--unsafe-allow-remote`, because its buttons execute shell commands. The
telemetry dashboard on 8810 is the one that is safe to expose — it is read-only
and has no launch controls. Preflight is computed on a background thread with a
120 s TTL; probing Kit python costs seconds per call and the page polls every
4 s, so recomputing per request would stall the console.

To reach either page from another machine, prefer an SSH tunnel over binding
wider:

```shell
ssh -N -L 8811:127.0.0.1:8811 -L 8810:127.0.0.1:8810 <user>@<lan-host>
```

`resume_training_edge.sh` **blocks while a trainer is running** rather than
exiting. That difference matters: `tools/resume_training.sh` exits 0 in that
case, and under `Restart=always` systemd relaunches it every `RestartSec` until
`StartLimitBurst` trips and the unit dies permanently — which is exactly how
`cosmos-so101-train.service` failed on 2026-09-07 with `start-limit-hit` and
stayed dead for four days. Blocking keeps one long-lived supervised process, so
the unit is a real crash supervisor.

______________________________________________________________________

## Evaluation discipline

The open items from `so101_focus5_worklog.md` all still apply, and two matter
more now:

- **There is still no baseline.** Neither the un-finetuned model nor the Nano
  policy has a trustworthy number on this task. "Edge is better than Nano" is
  unanswerable until both are evaluated on the same held-out episodes. The Nano
  server relaunch command is saved at
  `outputs/edge_setup/nano_server_relaunch.cmd` for exactly this comparison.
- **Watch `best_lift`, not success rate.** Lift is continuous and moves before
  binary success does.
- Judge checkpoints on the 14 held-out episodes and on eval success, never on
  training loss — especially now, because warm heads make training loss start
  lower for reasons that have nothing to do with SO-101 performance.

## Back-port to Nano

The resolution finding is independent of Edge. It cannot be expressed in the
TOML: `ModelConfig` in `configs/toml_config/sft_config.py` is `extra="forbid"`
and has no `resolution` field, so `[model] resolution = "480"` raises a
`ValidationError`. Use a Hydra override instead:

```shell
EXTRA_TAIL_OVERRIDES='model.config.resolution="480"' \
  bash examples/launch_sft_action_policy_so101_nano_focus5_multi_1gpu.sh
```

Keep the quotes. The `shift` dict is keyed by the **strings** `"256"` / `"480"`
/ `"720"`; an unquoted `model.config.resolution=480` passes the int `480` and
`omni_mot_model.py` raises `Resolution '480' not found in shift dict`.

That alone changes the training shift from 10 to 5 and makes the Nano recipe
internally consistent with the 480p data its dataloader emits. It also changes
`VIDEO_RES_SIZE_INFO[config.resolution]`, so treat it as a fresh run, not a
resume of an existing one.
