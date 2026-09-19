# so101-cosmos

Post-training, serving and digital-twin evaluation for the **SO-101** Cosmos3 action
policies (**Cosmos3-Nano** and **Cosmos3-Edge**): one CLI and one control page over a
pipeline that spans two Python environments which cannot import each other.

**Released checkpoints**

| Model | Hugging Face | Best result so far |
| --- | --- | --- |
| Cosmos3-Nano | [kabilanKB/cosmos_nano_policy_so101](https://huggingface.co/kabilanKB/cosmos_nano_policy_so101) (iteration 3750) | 17 / 542 single-object (3.1%) |
| Cosmos3-Edge | [kabilanKB/cosmos_edge_policy_so101](https://huggingface.co/kabilanKB/cosmos_edge_policy_so101) (iterations 6000, 6500, 7000; single-object continuation 500–1750) | iteration 6500: **5 / 98 (5.1%)** single-object, see [Cosmos3-Edge](#cosmos3-edge); also served on a **Jetson Thor**, see [edge/thor-hil](edge/thor-hil/README.md) |

```
train ──► merge ──► export ──► serve ──────► warmup ──► evaluate
                                  │                        │
                            ┌─────┴──────┐          ┌──────┴───────┐
                            │   SERVER   │◄────────►│    CLIENT    │
                            │   Cosmos   │  :8000   │  Isaac Lab   │
                            └────────────┘          └──────────────┘
                       cosmos-framework venv     Isaac Sim Kit python
                          torch, py3.13            omni, isaaclab
```

The **server** holds the policy. The **client** is the simulation: it streams observations
and executes the action chunks that come back. They agree on things neither one checks —
image layout, action space, gripper convention, normalization, view description — which is
why every flag lives in one tested place here instead of in shell history.

## Install

```shell
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
so101 --help
```

Requires Python 3.11+ (for `tomllib`). Install it wherever you like — deliberately *not*
into either of the two environments it drives.

Those two environments are set up separately, once each:

| | |
| --- | --- |
| **[docs/setup-cosmos.md](docs/setup-cosmos.md)** | the server: `cosmos-framework`, its venv, HF access, the base checkpoint, and the `LD_LIBRARY_PATH` fix that stops every biased `addmm` from failing |
| **[docs/setup-isaaclab.md](docs/setup-isaaclab.md)** | the client: `so101_bench` under Isaac Sim's Kit python, USD assets, the shadowed-checkout trap, and the patches `cosmos3_eval.py` needs |
| **[docs/post-training.md](docs/post-training.md)** | producing the policy: dataset narrowing, the LoRA recipe, launching, resuming, choosing a checkpoint |

`so101 doctor` checks both environments once they are in place.

## Use

```shell
so101 doctor                 # preflight both environments
so101 status                 # what is up, what is servable
so101 checkpoints            # inventory with merge/export state and eval history

so101 train                  # launch or resume the SFT run (~48 h)
so101 merge  --iter 3750     # fold LoRA adapters into base weights
so101 export --iter 3750     # consolidated safetensors, ~30 GB

so101 serve  --iter 3750     # start the policy server        [server]
so101 warmup                 # one inference request          [client]
so101 eval                   # Isaac Lab rollout              [client]
so101 eval --gui             # …with the viewer

so101 web                    # control page on :8800 for server and client
```

Every stage takes `--dry-run` to print the exact command (with its environment) and run
nothing, and `--detach` to run it as a tracked background job:

```shell
so101 serve --iter 3750 --dry-run
so101 eval --detach && so101 jobs && so101 logs <job-id>
```

`so101 web` serves a local page with the same stages as buttons, a live checkpoint table,
GPU and disk tiles, and per-job log tails. It binds to `127.0.0.1` and **refuses to enable
launching on any other interface** — the buttons run shell commands. Use `--no-launch` for
a read-only dashboard.

## Post-training

The policy itself is produced by a LoRA fine-tune of `nvidia/Cosmos3-Nano` — adapters on
the `moe_gen` projections plus full-rank training of action heads that **initialise from
random**, because the public base has no SO-101 action heads. The training set is narrowed
to five dense single-object instructions (97 episodes, 35,132 frames), which takes one
epoch from 36,799 iterations to 899 and makes convergence possible on a single GPU.

**[docs/post-training.md](docs/post-training.md)** covers it end to end: staging the
dataset and base checkpoint, the config and why each setting is what it is, launching,
auto-resume after a power cut, and how to pick a checkpoint — which is not by lowest loss.

## Benchmark

Success rate of the first post-training run (`action_policy_so101_nano_focus5`, 4000
iterations = 4.45 epochs) in the `so101_bench` Isaac Lab twin (`So101Bench-Bin-v0`).
Single-object episodes get 25 s, 4-object episodes 90 s. Each cell is successes / episodes.

| Checkpoint | Epochs | 1-object | 4-object |
| ---: | ---: | ---: | ---: |
| 250 | 0.3 | 0 / 5 | 0 / 19 |
| 500 | 0.6 | 0 / 6 | – |
| 750 | 0.8 | 0 / 6 | 0 / 4 |
| 1000 | 1.1 | 0 / 1 | 0 / 2 |
| 1250 | 1.4 | 0 / 14 | – |
| 1500 | 1.7 | 0 / 32 | – |
| 1750 | 1.9 | 0 / 6 | – |
| 2000 | 2.2 | 0 / 20 | – |
| 3000 | 3.3 | 0 / 20 | – |
| 3500 | 3.9 | 0 / 20 | – |
| **3750** | **4.2** | **17 / 542 (3.1%)** | **0 / 48** |
| 4000 | 4.5 | 0 / 20 | – |

**Checkpoint 3750 is the only one that has ever succeeded.**

- **Single-object: 3.1%** (17 / 542). Aug 20–23 scored 11 / 295; Sep 9 (action horizon 8)
  2 / 100; Sep 11–12 (horizon 32) 4 / 70; Sep 11 (horizon 16) 0 / 63 before the eval process
  crashed. Horizon 32 is the setting to use: on the same server and scenes it scored 4 / 70
  where horizon 16 scored 0 / 63.
- **Near vs far side:** every success had the object on the bin's side of the table. On runs
  using the fixed layout file that side scores 13 / 252 (5.2%) and the far side 0 / 187.
- **4-object scenes: 0%** (0 / 48).
- **By object:** cooking spoon 7 / 73, flower pot 6 / 84, green shoes 3 / 162, cardboard box
  1 / 110, altoids container 0 / 113. Episodes without an object name in older logs are
  assigned by position in the task file, so these counts are approximate.
- **Failure mode:** almost all failures are time-outs in which the target never leaves the
  table (every failure that logs target lift reports 0.00 in against a 0.50 in threshold).
- **Re-run on 2026-09-16:** 0 / 13 on `focus5.jsonl` (stopped) and 0 / 12 on the 12 scenes
  that had ever succeeded (`focus5_wins`). Weights, server settings and code were unchanged;
  the result fits a ≈ 5% chance per attempt rather than reliably solved scenes.

Not counted: hand-picked subsets (`focus5_wins`, `focus5_farside` 0 / 37). Compare runs only at
the same action horizon. Figures are aggregated from `outputs/monitor_jobs/eval*.log`,
`outputs/so101_cosmos_jobs/evaluate*.log` and `outputs/so101_cosmos_jobs_edge/` in
`cosmos-framework`, as of 2026-09-16.

## Cosmos3-Edge

A second policy, post-trained from `nvidia/Cosmos3-Edge-Policy-DROID`. The DROID action-head
row (embodiment domain 8) is copied into SO-101's row (domain 22) before training, so the
action heads start from a trained manipulation mapping instead of random init.

| | |
| --- | --- |
| Data | 5 single-object bin instructions + "Place each object in the plastic bin" capped at 45 episodes: **128 train / 14 held out** |
| Method | LoRA rank 64 / alpha 128 + action heads, lr 1e-4, global batch 32, 7,000 iterations (4.04 epochs) |
| Iterations 0–5500 | 1× RTX PRO 6000, ~21 s / iteration |
| Iterations 5500–7000 | resumed on NVIDIA Brev, 2× RTX PRO 6000, 9.49 s / iteration, ≈ $27.70 (see [Training on NVIDIA Brev](#training-on-nvidia-brev)) |

**Benchmark** (`focus5.jsonl`, horizon 32):

| Checkpoint | Epochs | 1-object |
| ---: | ---: | --- |
| 1500 | 0.87 | 0 / 50 |
| 2500 | 1.44 | 0 / 100 |
| **6500** | **3.76** | **5 / 98 (5.1%)** (2 episodes skipped) |
| 7000 | 4.04 | 1 / 100 (1.0%) |

Checkpoint 6500 by object (GUI run on the workstation, 25 s per episode):

| Episodes | Object | Result | Nano 3750 on the same object |
| --- | --- | ---: | ---: |
| 1–20 | green shoes | **1 / 20** (14.73 s) | 3 / 162 |
| 21–40 | cardboard box | 0 / 20 | 1 / 110 |
| 41–60 | altoids container | **1 / 19** (12.43 s) | 0 / 113 |
| 61–80 | flower pot | **2 / 19** (17.93 s, 20.47 s) | 6 / 84 |
| 81–100 | cooking spoon | **1 / 20** (21.97 s) | 7 / 73 |

- **Best released checkpoint:** 5.1% against Nano 3750's 3.1% on the same benchmark; the altoids success
  is the first by any model (Nano 3750: 0 / 113).
- **Failures:** 92 time-outs with the target never lifted and one success that did not hold (the same
  passive failure mode as Nano).
- A single-object continuation from 6500 (`single_bin_from6500`, 34 instructions) reached 3 / 100 at
  iteration 1750 with a 60 s limit; 6500 remains the released best.

**On a Jetson Thor:** [edge/thor-hil](edge/thor-hil/README.md) serves the policy natively on a Jetson
Thor (~1.5–2 s per 32-step chunk with cuDNN attention and `torch.compile`) and closes the loop with
Isaac Lab 3.0 on an RTX PRO 5000 Blackwell laptop through a web UI that launches the Thor server,
switches checkpoints, picks the object and prompt, and runs with or without the Isaac Sim window.

Everything for this run is in **[edge/](edge/README.md)**:
- the training recipe and cosmos-framework tools;
- the Brev scripts and run record;
- the merge, export and upload scripts with the model card;
- the Jetson Thor hardware-in-the-loop setup and web UI (`edge/thor-hil/`);
- `edge/bench/edge_train_mix.jsonl`, a 75-episode benchmark matching the Edge training mix,
  including 4-object scenes.

Drive it with the same CLI:

```shell
so101 --config so101-edge.toml status
so101 --config so101-edge.toml serve --iter 7000
so101 --config so101-edge.toml eval --gui --iter 7000
```

The Edge merge must use `--lora-rank 64 --lora-alpha 128`; `so101-edge.toml` sets this.

Training data per run (episodes, samples, epochs) is measured in
**[docs/training-episodes.md](docs/training-episodes.md)**.

## Training on NVIDIA Brev

The last 1,500 Edge iterations (5500 → 7000) ran on an NVIDIA Brev cloud instance, paid from
a $60 credit, while the workstation GPU stayed free for evaluation. Scripts and the full run
record are in **[edge/brev/](edge/brev/README.md)**.

### Instance

| | |
| --- | --- |
| Machine | 2× RTX PRO 6000 Blackwell Server Edition (96 GB each), 30 CPUs, 283 GB RAM, 1.4 TB disk |
| Provider / price | MassedCompute via Brev, **$5.26 / hour**, **no stop/start** (billed until deleted) |
| Software | driver 580.126.09 (CUDA 13), Python 3.13, `torch 2.10.0+cu130` |
| Why this one | Same GPU model as the workstation, so no new CUDA or kernel risk. The 1× H100 ($3.00 / h) cannot run Isaac Sim, the 1× H200 ($5.40 / h) is only slightly faster, and the 8× H100 ($24.99 / h) could exhaust the credit. |

Edge training peaks at ~26–28 GB per GPU, so any 48 GB+ card fits.

### What ran

| Step | How | Time |
| --- | --- | --- |
| Push code + checkpoint | `edge/brev/push_to_brev.sh`: workstation → Brev over SSH with the Brev key forwarded from the laptop; checkpoint `iter_000005500` (6.6 GB) verified by md5 | ~45 min (home upload link) |
| Environment | `edge/brev/remote_setup.sh`: apt packages, `uv sync --group cu130-train`, dataset (18 GB) and Wan2.2 VAE (2.7 GB) downloaded from Hugging Face on the instance | **1 min 22 s** |
| Smoke test | `SMOKE=1 remote_train.sh`: 3 iterations; resumed at 5501 with model and optimizer state, loss 1.12–1.36 | ~3 min |
| Training | `remote_train.sh` in `tmux`: 16 per GPU × 2 GPUs × 1 accumulation = global batch 32, identical to the workstation phase | **3 h 58 min** at 9.49 s / iteration |
| Copy back | `edge/brev/sync_back.sh`: each new checkpoint (6000, 6500, 7000) copied into `checkpoints_brev/` and verified by md5 every 5 min | ~16–17 min per checkpoint |
| Delete | `brev delete` as soon as `iter_000007000` was verified | – |

Training finished with `Done (exit 0)` at iteration 7000, final loss ≈ 1.0. The loss continued
the single-GPU trend: over the same iterations 5501–5738 the Brev run averaged 1.28, against
1.35 before the workstation run stopped.

### Cost

| Phase | Time | Cost |
| --- | --- | ---: |
| Setup, checkpoint upload, smoke test | ~57 min | ≈ $5.00 |
| Training 5500 → 7000 | 3 h 58 min | ≈ $20.90 |
| Final copy-back and delete | ~20 min | ≈ $1.80 |
| **Total** | **≈ 5 h 16 min** | **≈ $27.70 of $60** |

For comparison, the same 1,500 iterations take ~8.75 h on the single workstation GPU.

### Re-running

```shell
# laptop: log in, create the instance in the Brev console, then
brev login && brev refresh
eval "$(ssh-agent -s)" && ssh-add ~/.brev/brev.pem

# workstation, reached with the key forwarded (ssh -A)
bash edge/brev/push_to_brev.sh <brev-user>@<ip> 5500

# on Brev
tmux new -d -s setup 'bash ~/remote_setup.sh 2>&1 | tee ~/setup.log'
SMOKE=1 bash ~/remote_train.sh
tmux new -d -s train 'bash ~/remote_train.sh 2>&1 | tee ~/train.log'

# workstation: copy checkpoints back as they are saved
bash edge/brev/sync_back.sh <brev-user>@<ip>

# laptop, once the last checkpoint is verified
brev delete <instance-name>
```

`remote_train.sh` keeps the global batch at 32 for any GPU count by adjusting gradient
accumulation.

### Lessons from the run

- **Download on the instance, upload only what you must.** The 6.6 GB checkpoint took ~45 min to
  upload from the workstation; the 18 GB dataset downloaded on Brev in under a minute.
- **The GPU-count override is `model.config.parallelism.data_parallel_shard_degree`**, not
  `model.parallelism…`; the wrong path fails at startup.
- **Use `ssh -n` inside scripts fed through `bash -s`**, or the inner `ssh` reads the rest of the
  script as its stdin.
- **Run one copy at a time and verify by md5.** Killing a local `ssh` does not stop the remote
  `rsync`: two concurrent copies plus a killed `--partial` copy left a truncated checkpoint once.
  Use `--append-verify` to resume.
- **No stop/start instances bill until deleted.** Copy checkpoints back as they are saved and
  delete as soon as the last one is verified.

## Package structure

```
src/so101_cosmos/
  config.py      Settings: every path and run parameter, from defaults / TOML / env
  pipeline.py    command construction for each stage — pure, returns (argv, cwd, env)
  proc.py        JobStore: launch detached, track, stop, tail; survives a controller restart
  inventory.py   checkpoints, merges, exports, eval history, disk headroom
  logparse.py    the two trainer line formats and the two episode line formats
  telemetry.py   GPU readings via nvidia-smi
  doctor.py      preflight checks for both environments
  cli.py         argparse front end
  web/
    server.py    stdlib HTTP control panel
    static/      the page it serves
tests/           command construction, log parsing, config resolution, inventory
docs/
  setup-cosmos.md         server environment: venv, HF access, base checkpoint, cuBLAS fix
  setup-isaaclab.md       client environment: Kit python, USD assets, eval-script patches
  post-training.md        the fine-tune recipe: dataset narrowing, LoRA, launch, resume
  training-episodes.md    measured episodes, samples and epochs for every run
  work-log-2026-09-16.md  review, benchmarks, releases, Edge training on Brev
edge/
  README.md               the Cosmos3-Edge run: data, method, results, layout
  cosmos-framework/       Edge recipe, row-transplant and eval tools, runbook, worklog
  brev/                   Brev setup/train/sync scripts and the 2026-09-16 run record
  huggingface/            merge -> export -> upload scripts and the model card
  bench/                  edge_train_mix.jsonl benchmark set
so101.example.toml        Nano settings template
so101-edge.toml           Edge settings
```

The split that matters is `pipeline.py` versus `proc.py`: **construction is pure and
tested, execution is separate.** Every flag in the serving contract fails *silently* — the
server starts happily and returns wrong actions — so `tests/test_pipeline.py` asserts on
the argv directly rather than hoping a rollout catches it.

## Why no dependencies

`[project.dependencies]` is empty on purpose. This package orchestrates subprocesses in two
foreign interpreters:

| Side | Interpreter | Has |
| --- | --- | --- |
| server | `cosmos-framework/.venv/bin/python` | torch, cosmos_framework |
| client | `IsaacLab/_isaac_sim/python.sh` | omni, isaacsim, isaaclab, so101_bench |

Importing torch or `omni` here would tie the controller to one of the two environments it
exists to keep apart, and neither can be installed alongside the other. So the controller
builds argv, sets three environment variables and calls `subprocess` — nothing more.

Two optional extras:

- `client` — `numpy`, `openpi-client`. Only to run the warmup probe from the controller's
  own interpreter; the Kit python already has both.
- `dev` — `pytest`, `ruff`.

## Configuration

Defaults match the workstation the pipeline was verified on: checkouts under your home
directory (`~/cosmos-framework`, `~/IsaacLab/so101_bench`, `~/IsaacLab/_isaac_sim/python.sh`)
and the Nano run. `train_log_glob` points status and the dashboard at the run's own training
log (`outputs/edge_setup/train_edge*.log` for Edge). Override with a TOML file or environment
variables:

```shell
so101 --config so101.toml status
SO101_CFG_POLICY_PORT=8001 so101 serve --iter 3750
```

Copy `so101.example.toml` to start. The env prefix is `SO101_CFG_`, not `SO101_`, because
**`SO101_ROOT` is read by cosmos-framework itself** to resolve checkpoint-metadata
transforms — keeping the namespaces apart means setting one can never silently change the
other.

## Things that fail silently

Encoded in `pipeline.py` and asserted in the tests. Each one produces a *running* system
that returns wrong answers:

| | |
| --- | --- |
| `--arm-joint-dim 5` | the server assumed DROID's 7-joint arm until it was generalised |
| `--no-flip-gripper` | DROID's `1.0 − x` flip on SO-101's `[0,100]` `.pos` scale sends `−86.3` |
| `--action-normalization` + `--normalizer-stats-path` | without **both**, every joint is commanded to ≈0 |
| `--conditioning-fps 30` | a wrong value shifts the time conditioning |
| `--view-description` | the server default describes three views; SO-101 sends two |
| export `--experiment`, not the run's `config.yaml` | that config still declares `lora_enabled = true` and demands adapter keys the merge folded away |
| `PYTHONPATH` on the client | `so101_bench` is pip-installed editable pointing at a different checkout |
| `LD_LIBRARY_PATH` first entry | mismatched cuBLAS/cuBLASLt makes every biased `addmm` raise `CUBLAS_STATUS_NOT_INITIALIZED` — on an idle GPU |
| merge `--lora-rank` / `--lora-alpha` | Nano is 16/32, Edge is 64/128; the wrong pair merges without error into a mis-scaled model |
| `SO101_ROOT` for export | export resolves the dataset config through it and fails without it |

`--device-memory-bytes` is deliberately **not** passed: it is a no-op on a single GPU,
reaching only `_build_model_parallelism`, which ignores the value.

## Operating notes

- Serve, warm up, *then* evaluate. The client connects once and dies on refusal, and the
  server's first request compiles for minutes.
- The server is single-threaded per request; a probe sent during a live rollout queues
  behind it and can time out.
- Check the GPU is free first (`so101 doctor`). A stale policy server holds ~32 GB and a
  wedged Isaac Sim ~10 GB, and both outlive their parent.
- One evaluated checkpoint costs ~88 GB across three formats. `so101 clean` deletes merged
  intermediates that already have an export; they regenerate in about a minute.
- Eval history is read from this package's job directory *and* from
  `outputs/monitor_jobs`, where `tools/train_monitor.py` recorded earlier runs, so nothing
  from before the package existed is lost. Legacy records are read-only.
- The newest checkpoint is not the best one. On the reference run, iteration 4000 has the
  lowest loss and scores 0/20, while 3750 scores 17/542 (see [Benchmark](#benchmark)).
  Pick from `so101 checkpoints`.
- Green shoes is two rigid bodies (left and right shoe). The benchmark tracks the first body
  only, so lifting the other shoe logs `target_lift=0.00in`.

## Tests

```shell
PYTHONPATH= .venv/bin/pytest   # no GPU, no Isaac Sim, no cosmos-framework needed
PYTHONPATH= .venv/bin/ruff check .
```

`PYTHONPATH=` is not optional on a box with ROS sourced: `/opt/ros/humble` puts its
python3.10 site-packages on the path, and pytest auto-loads the `launch_testing` plugin
from it, which fails on `import yaml` before any test runs. The same inherited profile is
why `LD_LIBRARY_PATH` has to be fixed for the GPU stages.

## Author

**Kabilan KB** — pipeline, dataset narrowing, single-GPU LoRA recipes for Cosmos3-Nano and
Cosmos3-Edge, the Brev training run, evaluation tooling and this package.

## Provenance

Command construction and log parsing are ported from `cosmos-framework`'s
`tools/train_monitor.py` and the runbook in `docs/so101_cosmos3_server_to_client.md`,
both verified against `cosmos-framework @ 5e67049` on 1× RTX PRO 6000 Blackwell.

## License

Not yet chosen — add one before distributing. `cosmos-framework` and `so101_bench` carry
their own terms and are not vendored here.
