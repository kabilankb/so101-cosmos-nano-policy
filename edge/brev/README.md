# Brev training bundle: SO-101 Cosmos3-Edge policy

Everything needed to run (or resume) the SO-101 Edge post-training on a fresh NVIDIA Brev
instance, plus the complete record of the first Brev run (2026-09-16).

A new instance does **not** need a copy of the old one. Setup on the first run took about two
minutes, because the Python environment, dataset and video decoder download from datacenters
far faster than they upload from the workstation. Only code, scripts and one checkpoint are
pushed from here.

## Contents

| Path | What |
| --- | --- |
| `push_to_brev.sh` | Workstation → Brev: code, scripts, run config, one checkpoint; verifies by md5 |
| `remote_setup.sh` | On Brev: apt packages, `uv`, `uv sync` (CUDA 13 if driver ≥ 580, else 12.8), dataset + VAE from Hugging Face, writes `~/edge_env.sh` |
| `remote_train.sh` | On Brev: resume the run at global batch 32 on any GPU count; `SMOKE=1` for a 3-iteration check |
| `sync_back.sh` | Workstation: copy each new checkpoint back into `checkpoints_brev/` and verify by md5, every 5 min |
| `run-2026-09-16/` | Record of the first run: `home/` (train, setup, smoke logs, env file, scripts as run), `run/` (config.yaml, config.pkl, launch info), `trainer_logs/`, `brev_machine_info.txt`, `brev_pip_freeze.txt` |

## First run (2026-09-16)

| | |
| --- | --- |
| Instance | `separate-gray-fly`, MassedCompute, 2× RTX PRO 6000 Blackwell Server (96 GB), 30 CPUs, 283 GB RAM, 1.4 TB disk, driver 580.126.09 |
| Price | $5.26 / hour, **no stop/start** (billed until deleted) |
| Environment | Python 3.13, `torch 2.10.0+cu130`, `transformers 4.57.6` (full list in `brev_pip_freeze.txt`) |
| Work | Resumed `action_policy_so101_edge_focus5_multi_1gpu` from `iter_000005500` to 7000 |
| Speed | 9.48 s / iteration (vs ~21 s on the single workstation GPU) |
| Timeline (IST) | deployed ~17:35 · setup done 17:45 · checkpoint upload done ~18:28 · training 18:33 → 22:31 |
| Cost | ≈ $28 of the $60 credit |
| Output | `iter_000006000`, `6500`, `7000`, copied back to `checkpoints_brev/` and verified; merged + exported to `model_export_brev_<iter>`; uploaded to `kabilanKB/cosmos_edge_policy_so101` |

## Re-running on a new instance

1. **Laptop:** `brev login`, create the instance in the Brev console (1× or 2× RTX PRO 6000,
   ≥ 200 GB disk), then `brev refresh` and note its direct IP from `~/.brev/ssh_config`
   (`Host <name>-host`).
2. **Laptop:** load the Brev key into an agent:
   `eval "$(ssh-agent -s)"; ssh-add ~/.brev/brev.pem`
3. **Workstation (via the laptop, key forwarded):**
   `ssh -A -p <port> <user>@<workstation> 'bash ~/brev_edge/push_to_brev.sh <brev-user>@<ip> <iteration>'`
4. **Brev:** `tmux new -d -s setup 'bash ~/remote_setup.sh 2>&1 | tee ~/setup.log'`
5. **Brev:** smoke test, then the real run:
   `SMOKE=1 bash ~/remote_train.sh` → `tmux new -d -s train 'bash ~/remote_train.sh 2>&1 | tee ~/train.log'`
6. **Workstation (via the laptop):**
   `ssh -A -p <port> <user>@<workstation> 'bash ~/brev_edge/sync_back.sh <brev-user>@<ip>'`
7. When the last checkpoint is copied back and verified: `brev delete <name>` from the laptop.

To train longer than 7000 iterations, raise `trainer.max_iter` and `scheduler.cycle_lengths`
together in `remote_train.sh`'s overrides; the schedule decays to zero at `cycle_lengths`.

## Lessons from the first run

- **Override path:** the GPU count is `model.config.parallelism.data_parallel_shard_degree`, not
  `model.parallelism...` (fixed in `remote_train.sh`).
- **`ssh -n`:** an `ssh` inside a script fed through `bash -s` reads stdin and silently swallows
  the rest of the script. Every direct `ssh` in these scripts uses `-n`.
- **One copy at a time:** killing the local side of an `ssh` does not stop the remote `rsync`.
  Two concurrent copies of the same checkpoint, and a killed `--partial` copy, left a truncated
  file. Use `--append-verify`, check `ps` before retrying, and always verify md5.
- **`pkill -f` over ssh** can match its own command line and kill the shell running it. Use
  `pgrep -f "[p]attern"`.
- **Upload is the slow part:** 6.6 GB took ~20 min from the workstation; the 18 GB dataset took
  ~1 min from Hugging Face on Brev. Download everything you can on the instance.
- **No stop/start instances bill until deleted.** Delete as soon as the last checkpoint is verified.
