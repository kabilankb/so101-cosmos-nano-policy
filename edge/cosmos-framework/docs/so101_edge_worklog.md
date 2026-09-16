# SO-101 Cosmos3-Edge post-training — worklog

Session of **2026-09-11**. Sibling of [`so101_1gpu_worklog.md`](./so101_1gpu_worklog.md)
and [`so101_focus5_worklog.md`](./so101_focus5_worklog.md), which cover the Nano runs.

Goal: post-train **`nvidia/Cosmos3-Edge-Policy-DROID`** into an SO-101 action policy,
after the Nano policy's accuracy proved inadequate. Reference pipeline documentation is
in [`so101_edge_posttrain.md`](./so101_edge_posttrain.md); this file records what was
done, what was found, and what is still open.

______________________________________________________________________

## State at 2026-09-11 22:23

| | |
| --- | --- |
| Training | `action_policy_so101_edge_focus5_multi_1gpu`, iter **528 / 7000**, loss ~1.97 |
| Rate | 21.0 s/iter alone; **29.9 s/iter** while an eval shares the GPU |
| Projected finish | ~2026-09-13 12:00, slipping ~1.5 h per concurrent eval |
| Checkpoints | `iter_000000500` saved (6.6 GB), merged, exported to `model_export_500` (7.3 GB) |
| Running eval | Nano 3750 @ `action_horizon 16` — the A/B described below |
| Auto-resume | `cosmos-so101-edge-train.service`, active, supervising |

______________________________________________________________________

## Findings

Ordered by how much they change what to do next, not by when they were found.

### 1. Action heads are per-embodiment, so "warm start" needed a transplant

`action2llm` / `llm2action` are `DomainAwareLinear`
(`model/generator/mot/domain_aware_linear.py`): weights live in an
`nn.Embedding(num_embodiment_domains, out*in)`, so **every embodiment owns a private
row**. DROID is domain 8; SO-101 is domain 22.

Measured on the released `nvidia/Cosmos3-Edge-Policy-DROID` (diffusers names
`action_proj_in` / `action_proj_out`):

| tensor | row 8 (DROID) | row 22 (SO-101) | every other row |
| --- | --- | --- | --- |
| `action_proj_in.fc.weight` | 10.6676 | **6.4360** | 6.42–6.45 |
| `action_proj_in.bias.weight` | 1.5603 | **0.0000** | 0.0000 |
| `action_proj_out.fc.weight` | 2.3256 | **1.1387** | 1.135–1.139 |
| `action_proj_out.bias.weight` | 0.0868 | **0.0000** | 0.0000 |

Row 22 ships at `xavier_uniform_` init with an exactly-zero bias. **Initialising from the
DROID policy therefore gives SO-101's action head nothing the base Edge checkpoint would
not have given it.** Only `action_modality_embed` — a single shared `[2048]` vector, *not*
a per-domain table — transfers for free.

`tools/transplant_domain.py` copies row 8 into row 22 before DCP conversion. Verified
after training by reading the saved checkpoint back: row 22 now reads 10.6676 (== row 8)
while untouched row 0 stays at 6.4187.

Both embodiments are single-arm absolute `joint_pos` policies padded into the same 64-wide
action space (DROID 7 joints + gripper, SO-101 5 + gripper), so the token↔hidden mapping is
largely shared structure. The backbone transfers regardless — it is the DROID-*post-trained*
world model, not base Edge.

### 2. The Nano recipe trained at the wrong flow-matching shift

`omni_mot_model.py:437` selects the training `shift` by `model.config.resolution`
(`{"256": 3, "480": 5, "720": 10}`). `NANO_MODEL_CONFIG` is `"720"` and the SO-101 Nano
recipe never overrode it, while `SO101LeRobotDataset` emits 480p. **The Nano runs trained
with shift=10, the 720p schedule, on 480p data.** `EDGE_MODEL_CONFIG` is natively `"480"`,
so Edge gets shift=5 for free.

To back-port to Nano, use a Hydra override — `ModelConfig` is `extra="forbid"` and has no
`resolution` key, so `[model] resolution = "480"` raises `ValidationError`:

```shell
EXTRA_TAIL_OVERRIDES='model.config.resolution="480"' bash examples/launch_...nano_focus5_multi_1gpu.sh
```

Keep the quotes: the shift dict is keyed by strings, and an unquoted `480` raises
`Resolution '480' not found in shift dict`.

### 3. Edge action-policy export was broken for every shipped recipe

`export_model.py:_build_edge_policy_metadata` resolved the action dataset at
`dataloader_train.dataloaders.action_data.dataloader.dataset` (+ `list_of_datasets`).
Every action recipe in this repo — `action_policy_droid_nano`,
`action_policy_libero_nano`, `action_policy_so101_*` — instead builds a
`PackingDataLoader` with `dataloader_train.dataloader.datasets.<name>.dataset`, so an Edge
action export always died with:

```
ValueError: Cosmos3 Edge export requires an action dataset config at
dataloader_train.dataloaders.action_data.dataloader.dataset.
```

This only bites on Edge: the metadata builder runs when the model is Edge **and**
`action_gen` is true, so Nano exports never reached it.

Fixed in two places:

- `_build_edge_policy_metadata` now resolves **both** layouts via a new
  `_action_dataset_configs()`, preferring the original.
- The resolver reads the embodiment from a config `embodiment_type` key or an
  `EMBODIMENT_TYPE` attribute on the dataset target. These factories are functions, not
  classes, so nothing exposed it; `EMBODIMENT_TYPE` is now attached to
  `get_action_so101_sft_dataset` (`"so101"`) and `get_action_droid_sft_dataset`
  (`"droid_lerobot"`), matching `domain_utils`.

Resulting `model_export_500/checkpoint.json`:

```json
{ "action_chunk_size": 32, "conditioning_fps": 30.0, "domain_name": "so101" }
```

`conditioning_fps: 30.0` is correct for SO-101 — the DROID policy we initialised from
carries `15.0`.

### 4. The eval client runs the action chunk fully open-loop

`so101_bench/source/so101_bench/utils/cosmos3.py:175`:

```python
if not self.action_queue:                    # refills only when EMPTY
    action_chunk = np.asarray(self.client.infer(request)["action"])   # [32, 6]
    horizon = min(action_chunk.shape[0], self.action_horizon)         # min(32, 32)
    self.action_queue.extend(action_chunk[:horizon])
pos_row = self.action_queue.popleft()
```

At `--action_horizon 32` the client queues all 32 steps and replans only when the queue
drains — **1.07 s of open-loop motion at 30 Hz**, with no correction from where the arm
actually went. NVIDIA's reference loop for this model family executes *half* the chunk and
re-asks, and the Cosmos 3 write-up is explicit that a state-conditioned policy's
self-correction comes from replanning off measured state.

**Every Nano evaluation that produced the 3% figure ran at horizon 32.** An A/B at
horizon 16 (`action_chunk=0.533s`, confirmed in the client's timing line) is in flight.

### 5. The Nano baseline is not 3% uniformly — three of five objects never succeed

`so101 checkpoints` over the Nano run: only `iter_000003750` (epoch 4.17) ever scored,
**13 / 437 ≈ 3.0%**. Every other checkpoint is a clean zero, including 4000.

Splitting the three 100-episode runs by object (`tasks/focus5.jsonl` is grouped):

| episodes | object | successes (3 runs) |
| --- | --- | --- |
| 1–20 | green shoes | **0 / 60** |
| 21–40 | cardboard box | **0 / 60** |
| 41–60 | altoids container | **0 / 60** |
| 61–80 | flower pot | 5 / 60 |
| 81–100 | cooking spoon | 4 / 60 |

All nine successes are flower pot or cooking spoon. A hard zero on three objects across
300 attempts — rather than low-but-nonzero everywhere — points at a systematic failure
(reaching the wrong place, or closing at the wrong moment) rather than undertraining, and
is the signature open-loop drift would produce: only forgiving geometry survives it.

### 6. Environment and API notes

- **`uvx` is not on `PATH` in non-interactive shells.** `so101 serve` shells out to
  `uvx hf download` to resolve the Wan2.2 VAE and dies with
  `FileNotFoundError: 'uvx'`. It works from an interactive shell and fails from
  systemd/cron/ssh-command. Fix by prepending `/home/<user>/.local/bin` in `base_env`.
- **`--device-memory-bytes` is a no-op on a single GPU** — it is only read by
  `_build_model_parallelism`, which ignores the value. Earlier docs implying it protects
  against Isaac Sim collisions are wrong.
- **LoRA cannot target the gen MLP by name.** The MoT layer suffixes only its *direct*
  children with `_moe_gen` (`q/k/v/o_proj_moe_gen`, `mlp_moe_gen`, `norm_moe_gen`). The gen
  MLP's own children are plain `gate_proj` / `up_proj` / `down_proj`, identical to the
  frozen understanding tower's, and `lora.py` matches by exact child name anywhere in the
  tree — so `gate_proj_moe_gen` injects **zero** modules (silent warning) and bare
  `gate_proj` would also adapt the und tower. To train the gen MLP, add `mlp_moe_gen` to
  `lora_extra_trainable` (substring match on parameter names, correctly gen-scoped).
- **The trainer auto-resumes from `latest_checkpoint.txt`.** A smoke run under the same
  `job.name` is resumed from, at whatever LoRA rank the smoke used. Give smoke runs their
  own `job.name` or clear the output directory.

______________________________________________________________________

## Measurements

Measured on 1× RTX PRO 6000 Blackwell (96 GB), not estimated.

| | Nano focus5+multi | **Edge focus5+multi** |
| --- | --- | --- |
| Backbone | Qwen3-VL-8B | Nemotron-2B-Dense-VL |
| Resolution / shift | 720 → shift 10 (on 480p data) | **480 → shift 5** |
| LoRA rank / alpha | 16 / 32 | **64 / 128** |
| LoRA modules | — | 112 (28 layers × 4 proj) |
| Trainable | — | 25,690,112 LoRA + 8,458,240 heads = **1.006%** |
| Frozen | — | 3,361,198,784 |
| Microbatch × accum | 4 × 8 | **16 × 2** (global batch 32 both) |
| Peak GPU memory | — | **26.1 GB / 96** |
| s/iter @ gb32 | ~40 | **~21** (29.9 under eval contention) |
| 7000 iters | ~78 h | **~39 h** |
| Checkpoint size | ~29 GB | **6.6 GB** |
| Action heads at init | random | **warm** (DROID row 8 transplanted) |
| Train episodes | 87 (focus5) | 128 (focus5 + capped multi) |
| Windows / iters-per-epoch | — | 55,385 / **1,731** |

Dataset note: `so101_bench_sim_6` holds **2,228 episodes / 1.25 M frames**; the run uses
**128** of them. `max_episodes_per_task=45` discards ~830 episodes of the multi-object
instruction, justified by wall-clock (uncapped ≈ 31 days for 4 epochs), not by learning.

______________________________________________________________________

## Files

### `cosmos-framework`

Created:

- `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_so101_edge.py`
  — three registered experiments (`_edge`, `_edge_focus5`, `_edge_focus5_multi`), derived
  from the Nano recipe so the data path cannot drift.
- `examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml`
- `examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh`
- `tools/transplant_domain.py` — copies a trained embodiment row into an untrained one.
- `tools/resume_training_edge.sh` — **blocks** while a trainer runs instead of exiting
  (see Open items).
- `tools/edge_train_dashboard.py` — read-only telemetry, port 8810.
- `docs/so101_edge_posttrain.md`, `docs/so101_edge_worklog.md` (this file).

Modified:

- `cosmos_framework/configs/base/config.py` — registers the Edge experiment module.
- `cosmos_framework/scripts/export_model.py` — Finding 3.
- `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` — Finding 3.
- `tools/train_monitor.py` — `--profile nano|edge` (unused now, but it hardcoded
  rank 16 / alpha 32, so pointing it at an Edge checkpoint would have merged at the wrong
  scale). Backup at `/tmp/train_monitor.py.bak`.

Superseded (files kept, services disabled): `tools/edge_eval_epoch.py`,
`tools/edge_eval_console.py` — written before finding `so101-cosmos`, which already does
this. Use that instead.

### `so101-cosmos`

- `so101-edge.toml` — Edge run config. **This is the entry point for Edge work.**
- `src/so101_cosmos/config.py` — added `train_log_glob` (default unchanged).
- `src/so101_cosmos/logparse.py` — `newest_training_log(framework, pattern=...)`.
- `src/so101_cosmos/web/server.py` — passes `cfg.train_log_glob`.

Without that key the Edge page reported the **Nano** run's iteration and loss while
claiming to describe Edge. 59/59 tests pass; Nano instance verified unchanged.

### Checkpoints

- `examples/checkpoints/Cosmos3-Edge/` (8.6 GB) — base, for the cold-head ablation.
- `examples/checkpoints/Cosmos3-Edge-Policy-DROID/` (8.6 GB) — as released.
- `examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init/` — row 8 → 22 transplanted.
- `examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init-dcp/` (6.3 GB) — **training
  init**.
- `examples/checkpoints/Cosmos3-Edge-Policy-DROID-dcp/` (6.3 GB) — cold control.

______________________________________________________________________

## Operating

| | |
| --- | --- |
| Edge control page (click-and-run) | <http://127.0.0.1:8802> — `so101-edge-web.service` |
| Edge page, LAN read-only | <http://<lan-host>:8812> — `so101-edge-web-lan.service` |
| Edge telemetry + ETA | <http://<lan-host>:8810> — `cosmos-edge-dashboard.service` |
| Nano control page | <http://127.0.0.1:8801> — unchanged |
| Auto-resume | `cosmos-so101-edge-train.service` |

```shell
cd /home/<user>/so101-cosmos
export PATH="/home/<user>/.local/bin:$PATH"          # uvx; see Finding 6
.venv/bin/so101 --config so101-edge.toml doctor
.venv/bin/so101 --config so101-edge.toml status
.venv/bin/so101 --config so101-edge.toml merge  --iter 500
.venv/bin/so101 --config so101-edge.toml export --iter 500
.venv/bin/so101 --config so101-edge.toml serve  --iter 500 --detach
SO101_CFG_ACTION_HORIZON=16 .venv/bin/so101 --config so101-edge.toml eval --iter 500 --detach
```

The buttons are refused on any non-loopback bind by design; tunnel rather than widening:

```shell
ssh -N -L 8802:127.0.0.1:8802 -L 8810:127.0.0.1:8810 <user>@<lan-host>
```

______________________________________________________________________

## Open items and risks

- **No baseline for the Edge model itself.** Neither un-finetuned Edge nor
  `Cosmos3-Edge-Policy-DROID` zero-shot has been measured on this twin. Without them the
  7000-iteration result is unattributable. Highest-value missing measurement.
- **The contract has never been validated end-to-end.** At 3% success the prior should be
  "something is mis-wired", not "needs more epochs" — one such bug (Finding 2) was already
  found, and the docs warn that every domain flag fails silently. The cheap test: feed a
  *training* episode's own observations to the policy server and compare predicted actions
  against the recorded ground truth. If it cannot reproduce actions on data it trained on,
  no amount of training fixes it.
- **128 of 2,228 episodes.** Compute-driven, not learning-driven. If the contract is clean
  and the policy still underperforms, more unique episodes beats more epochs over few.
- **1.006% trainable with ~66 GB of the card idle.** The card is compute/power-bound
  (100% util at the 600 W cap, ~92 °C, SM 2250 of 3090 MHz), not memory-bound. Adding
  capacity on 128 episodes risks overfitting; resolve the tension with data, not rank.
- **The 4.05-epoch horizon is weak evidence.** It comes from Nano's single non-zero
  checkpoint (13/437 at 3750, 0/20 at 4000) — one point surrounded by zeros with a wide
  confidence interval, not an established optimum.
- **Sim-only.** Training and evaluation are both `so101_bench_sim_6`. Says nothing about
  transfer to a physical SO-101.
- **Concurrent evaluation costs ~40% of training throughput.** Budget ~1.5 h of slipped
  finish time per eval; a full sweep should wait for training to end.

## Corrections to earlier assumptions

Recorded so they are not repeated.

- **`action_modality_embed` is not a per-domain table.** It is a single shared
  `[hidden_size]` vector. The per-embodiment parameters are the `DomainAwareLinear` rows in
  `action2llm` / `llm2action`.
- **Starting loss does not verify the warm start on Edge.** `loss_scale = 10.0` weights the
  vision flow-matching term, which dominates the scalar; the first ten smoke iterations
  ranged 9.36–14.04. Verify by reading action-head row norms out of a saved checkpoint
  instead.
- **`--device-memory-bytes` does not cap anything on a single GPU.**
- **Edge is "less accurate than Nano" only on NVIDIA's RoboLab DROID numbers**
  (22.9% vs 36.8%), a different benchmark and embodiment. Against this SO-101 twin the
  number to beat is **3%**.

______________________________________________________________________

## Results so far (updated 2026-09-13)

All runs: `So101Bench-Bin-v0`, `tasks/focus5.jsonl` (100 episodes),
`focus5_layouts_20260821_132325`, `action_horizon 32` unless noted.

| policy | epoch | result | note |
| --- | --- | --- | --- |
| Nano 3750, 3 historical runs | 4.17 | **13 / 437 = 3.0%** | the figure quoted before this session |
| Nano 3750, fresh replicate | 4.17 | **4 / 70 = 5.7%** | stopped early; successes at ep 18, 26, 61, 62 |
| Nano 3750, horizon **16** | 4.17 | **0 / 63** | the A/B; see below |
| Edge 1500 | 0.87 | 0 / 50 | stopped early to free the GPU |
| **Edge 2500** | **1.44** | **0 / 100** | complete; every episode `reason=time_out` |

### Horizon 16 is worse than 32 — hypothesis refuted

Run concurrently against the same server, the same episodes and the same layout
file, horizon 32 scored 4/70 while horizon 16 scored 0/63.

The reasoning that motivated the A/B was wrong. NVIDIA's reference loop executes
16 of 32 steps **at 15 Hz** — a 1.07 s replan interval. This dataset is 30 Hz, so
32 steps is *also* 1.07 s: horizon 32 already matches the reference cadence, and
horizon 16 replans twice as often as intended, querying the policy from mid-reach
states it never saw in training. The error was anchoring on the step count rather
than the wall-clock interval. **Keep horizon 32.**

### Two earlier claims corrected by the replicate

- **"Three objects never succeed" was an artifact of variance.** The replicate
  scored on green shoes (ep 18) and cardboard box (ep 26), both previously 0/60
  across three runs.
- **The Nano baseline is ~4-6%, not 3.0%.** Treat 3.0% as the low end of the
  range, not the number to beat.

### Reading the Edge zeros

Edge 2500 at 0/100 is **not** evidence against the approach: it is epoch 1.44
against Nano's 4.17. Nano's own trajectory was zero at every checkpoint through
epoch 3.89 and scored only at 4.17. Edge is tracking Nano's curve, not
underperforming it. The comparable Edge checkpoint is ~7000.

Every one of the 100 episodes ended `reason=time_out` — no failed grasps, no
wrong-object pickups. The arm simply does not complete an attempt inside 25 s,
which is the same failure signature the Nano checkpoints showed before 4.17.

Checkpoints worth an evaluation are 5500 (epoch 3.18), 6000 (3.47), 6500 (3.75)
and 7000 (4.05) — and they are far cheaper after training ends: a concurrent
evaluation costs ~5 h of training progress on top of its own ~11 h of wall-clock,
and it was concurrent evaluation that pushed the projected finish from ~12:00 to
~23:43 on 2026-09-13.

______________________________________________________________________

## The near/far split — the benchmark has an unsolved half

The single most useful finding of this session, and it is about the task rather
than the model.

Every success ever recorded on `So101Bench-Bin-v0` — 12 of them, across four
independent runs and all five objects — had the target object on the **same side
of the table as the bin**. The bin sits at `y = -0.0718`.

Grouping `focus5_layouts_20260821_132325` by the target's `y`:

| object position | episodes | wins | rate |
| --- | --- | --- | --- |
| `y < 0` (bin side) | 63 | **12** | **19.0%** |
| `y >= 0` (far side) | 37 | **0** | **0.0%** |

Confirmed directly: `tasks/focus5_farside.jsonl` (the 37 far-side episodes) run
against Nano `model_export_3750` returned **0/37**, every episode
`reason=time_out`, `failure_type=not_applicable`, `target_lift=0.00in`. Combined
with earlier runs the far-side record is **0 / 67**.

If the two classes were equivalent, 67 far-side attempts at the near-side rate of
19% would yield ~13 successes. Zero is not variance (p ≈ 1e-6). The split is real.

### Why this matters more than the headline percentage

- **"3-6% success" is the wrong summary.** The policy is at **19% on 63% of the
  benchmark and 0% on the other 37%**. Reporting a single blended number hides
  that a third of the task is unsolved rather than merely hard.
- **It is not a reach limit.** Distance from the robot base is effectively
  identical between the two classes (0.140 m vs 0.144 m). What differs is that the
  object must be carried *across* the workspace to the bin — the far-side mean
  object-to-bin distance is 0.340 m against 0.285 m near-side.
- **It is not visible in the failure telemetry.** Every far-side episode reports
  `time_out` with `failure_type=not_applicable` and `target_lift=0.00in` — and
  `target_lift` is unusable on the bin task regardless (see Results). The logs
  cannot distinguish "never approached" from "grasped but could not transport".
  Watching a GUI run is currently the only instrument for that distinction.

### Open question

Is the far-side gap a training-data coverage problem or a policy limitation?

Partial evidence against a blanket coverage gap: across the full
`so101_bench_sim_6` dataset, `shoulder_pan.pos` (base rotation, which sets the
reach direction) spans **-94.3 to +98.0, mean -4.0, std 36.3** — broad and
near-symmetric, so the arm does rotate both ways in training.

Two caveats keep this open:

- Those statistics cover all **2,228 episodes**; the run trains on **128**
  (focus5 + capped multi). The subset could be skewed where the parent is not.
- The dataset records joint states and video only, with **no object positions**,
  so training-set object placement cannot be measured directly. Base rotation is a
  proxy for where the arm went, not where objects were.

Resolving it means re-instantiating the loader to get the kept episode indices and
reading those parquet shards for the `shoulder_pan` distribution. That answer
decides between "train longer" and "collect different data", and is worth more
than another checkpoint evaluation.

### CORRECTION to the table above (2026-09-13)

The "19.0%" near-side figure above is **wrong** and must not be quoted. It is the
fraction of near-side *episodes* that succeeded at least once across four runs,
not a success rate. The per-attempt rates, counted over every logged episode:

| | attempts | successes | rate |
| --- | --- | --- | --- |
| Near side (`y < 0`) | 252 | 13 | **5.16%** |
| Far side (`y >= 0`) | 187 | 0 | **0.00%** |

The significance claim was also overstated: 187 far-side attempts at the
near-side rate predicts ~9.6 successes, so **p ≈ 6.5e-05**, not the 1e-6 quoted.
The split is still real; the effect is an order of magnitude less extreme.

**Only one scene has ever succeeded twice.** Success counts per episode across
all runs:

```
{18:1, 26:1, 61:1, 62:1, 63:1, 65:2, 78:1, 83:1, 84:1, 91:1, 94:1, 97:1}
```

Twelve near-one-off successes spread over twelve different episodes, with a
single repeat (episode 65, 2 of 4 runs), is the signature of a ~5% chance of
stumbling into success — not of reliable competence on a subset. It also explains
why replaying the twelve "winning" episodes returned 0/3 before being stopped:
those scenes were never reliably winnable.

**The accurate one-line summary of Nano 3750:** succeeds on ~5% of near-side
attempts, never on far-side attempts, and has demonstrated repeatable success on
exactly one scene out of a hundred. This, not "3%" or "4-6%" or "19% near-side",
is the number the Edge run should be compared against.
