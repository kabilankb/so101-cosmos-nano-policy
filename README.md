# so101-cosmos

Post-training, serving and digital-twin evaluation for the **SO-101** Cosmos3-Nano action
policy: one CLI and one control page over a pipeline that spans two Python environments
which cannot import each other.

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
  setup-cosmos.md    server environment: venv, HF access, base checkpoint, cuBLAS fix
  setup-isaaclab.md  client environment: Kit python, USD assets, eval-script patches
  post-training.md   the fine-tune recipe: dataset narrowing, LoRA, launch, resume
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

Defaults match the workstation the pipeline was verified on. Override with a TOML file or
environment variables:

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
  lowest loss and scores 0/20, while 3750 scores 11/262. Pick from `so101 checkpoints`.

## Tests

```shell
PYTHONPATH= .venv/bin/pytest   # no GPU, no Isaac Sim, no cosmos-framework needed
PYTHONPATH= .venv/bin/ruff check .
```

`PYTHONPATH=` is not optional on a box with ROS sourced: `/opt/ros/humble` puts its
python3.10 site-packages on the path, and pytest auto-loads the `launch_testing` plugin
from it, which fails on `import yaml` before any test runs. The same inherited profile is
why `LD_LIBRARY_PATH` has to be fixed for the GPU stages.

## Provenance

Command construction and log parsing are ported from `cosmos-framework`'s
`tools/train_monitor.py` and the runbook in `docs/so101_cosmos3_server_to_client.md`,
both verified against `cosmos-framework @ 5e67049` on 1× RTX PRO 6000 Blackwell.

## License

Not yet chosen — add one before distributing. `cosmos-framework` and `so101_bench` carry
their own terms and are not vendored here.
