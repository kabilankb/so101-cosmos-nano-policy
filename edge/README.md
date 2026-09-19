# Cosmos3-Edge SO-101 policy

The Cosmos3-Edge counterpart of the Nano pipeline on `main`: the training recipe, the NVIDIA Brev
run that finished it, the Hugging Face release, and a benchmark set matching its training data.

**Released checkpoints:**
[kabilanKB/cosmos_edge_policy_so101](https://huggingface.co/kabilanKB/cosmos_edge_policy_so101)
(`iter_6000/`, `iter_6500/`, `iter_7000/`, `single_bin_from6500/iter_*`).

## The run

| | |
| --- | --- |
| Experiment | `action_policy_so101_edge_focus5_multi` |
| Base model | `nvidia/Cosmos3-Edge-Policy-DROID`; the DROID action-head row (domain 8) is copied into SO-101's row (domain 22) with `cosmos-framework/tools/transplant_domain.py` before DCP conversion |
| Data | `so101_bench_sim_6`: 5 single-object bin instructions + "Place each object in the plastic bin" capped at 45 episodes. **128 train / 14 held out**, 55,385 samples |
| Method | LoRA rank 64 / alpha 128 on `q/k/v/o_proj_moe_gen` + action heads, lr 1e-4, global batch 32, linear decay over 7,000 iterations |
| Iterations 0–5500 | 1× RTX PRO 6000 (batch 16 × 2 accumulation), ~21 s / iteration |
| Iterations 5500–7000 | 2× RTX PRO 6000 on Brev (batch 16 per GPU), 9.48 s / iteration, ≈ $27.70 |
| Final | iteration 7000 (4.04 epochs), loss ≈ 1.0 |

## Benchmark so far

`so101_bench`, `So101Bench-Bin-v0`, `tasks/focus5.jsonl`, 25 s per episode, action horizon 32.

| Checkpoint | Epochs | Result |
| --- | ---: | --- |
| 1500 | 0.87 | 0 / 50 |
| 2500 | 1.44 | 0 / 100 |
| 6500 | 3.76 | **5 / 98 (5.1%)**: green shoes 1 / 20 (14.73 s), cardboard box 0 / 20, altoids container 1 / 19 (12.43 s, the first altoids success by any model), flower pot 2 / 19, cooking spoon 1 / 20 |
| 7000 | 4.04 | 1 / 100 (1.0%) |

**Jetson Thor:** the policy served on a Jetson Thor, with Isaac Lab on an RTX PRO 5000 Blackwell laptop
and a hardware-in-the-loop web UI, is in **[thor-hil/](thor-hil/README.md)**.

## Layout

| Path | What |
| --- | --- |
| `../so101-edge.toml` | `so101` CLI settings for the Edge run (`so101 --config so101-edge.toml status`) |
| `cosmos-framework/recipe/` | Experiment config (`action_policy_so101_edge.py`), single-GPU TOML and launch script. Copy into `cosmos_framework/configs/base/experiment/action/posttrain_config/`, `examples/toml/sft_config/` and `examples/` of a cosmos-framework checkout that has the SO-101 support. |
| `thor-hil/` | Jetson Thor policy server + Isaac Lab 3.0 client (RTX PRO 5000 Blackwell) + hardware-in-the-loop web UI |
| `cosmos-framework/tools/` | Action-head row transplant, training dashboard, eval console, per-epoch eval, queue/sweep/resume helpers |
| `cosmos-framework/docs/` | Edge post-training runbook and worklog |
| `brev/` | Scripts and README to resume or run training on a Brev instance, and to copy checkpoints back verified |
| `brev/run-2026-09-16/` | Record of the Brev run: training, setup and smoke logs, run config, machine info, exact package versions |
| `huggingface/` | Merge LoRA → export → upload scripts, and the model card as published |
| `bench/edge_train_mix.jsonl` | 75-episode benchmark matching the Edge training mix: 50 single-object (10 per trained object) + 25 four-object "Place each object in the plastic bin". Copy into `so101_bench/tasks/`. |

## Package changes on this branch

`Settings.train_log_glob` (default `outputs/train_resume_*.log`) lets the dashboard and status read
the Edge run's own training log, which lives under `outputs/edge_setup/`. Without it they report
the Nano run's iteration and loss for the Edge configuration.

## Notes

- Merging needs the Edge scale: `--lora-rank 64 --lora-alpha 128`. The Nano 16/32 produces a valid
  but wrongly scaled model without any error.
- Export needs `SO101_ROOT` set; `huggingface/merge_export.sh` handles it.
- Green shoes is two rigid bodies. The benchmark tracks the first one only, so lifting the other
  shoe logs `target_lift=0.00in`.
- Paths in the scripts are from the original workstation (`/home/<user>/...`); adjust them to your
  checkout.
