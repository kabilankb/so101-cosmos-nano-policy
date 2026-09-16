---
license: other
license_name: openmdw1.1-license
license_link: https://openmdw.ai/license/1-1/
base_model: nvidia/Cosmos3-Edge-Policy-DROID
library_name: cosmos
pipeline_tag: robotics
tags:
  - cosmos
  - cosmos3
  - cosmos3-edge
  - robotics
  - action-policy
  - so101
  - lerobot
  - isaac-lab
---

# cosmos_edge_policy_so101

A 6-DOF **SO-101** bin-placement action policy, post-trained from
[`nvidia/Cosmos3-Edge-Policy-DROID`](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID).
Each folder holds one training checkpoint with its LoRA adapters merged into the weights, exported
as consolidated safetensors.

| Folder | Iteration | Epoch (approx.) | Benchmark |
| --- | ---: | ---: | --- |
| `iter_6000/` | 6000 | 3.5 | not evaluated yet |
| `iter_6500/` | 6500 | 3.8 | evaluation in progress |
| `iter_7000/` | 7000 (final) | 4.0 | not evaluated yet |

**No full benchmark result yet.** Earlier checkpoints of the same run scored 0 / 50 (iteration
1500) and 0 / 100 (iteration 2500) on the benchmark below. An evaluation of iteration 6500 is in
progress and has recorded its first success. For comparison, the
[Cosmos3-Nano SO-101 policy](https://huggingface.co/kabilanKB/cosmos_nano_policy_so101) scores
3.1% on the same benchmark.

## Training

| Setting | Value |
| --- | --- |
| Base model | `nvidia/Cosmos3-Edge-Policy-DROID` |
| Action-head warm start | `action2llm` / `llm2action` keep a separate weight row per embodiment. The DROID row (domain 8) was copied into the SO-101 row (domain 22) before training, so SO-101 starts from DROID's trained action mapping instead of random init. |
| Experiment | `action_policy_so101_edge_focus5_multi` (cosmos-framework `5e67049` plus local SO-101 support) |
| Data | `so101_bench_sim_6`: 5 single-object "Place the X in the plastic bin" instructions (97 episodes) plus the multi-object "Place each object in the plastic bin", capped at 45 episodes. 142 episodes, 10% held out. |
| Action space | Absolute `joint_pos`, 6-D (5 arm joints plus gripper, LeRobot `.pos` units). Row 0 is the current state. Chunk of 32 steps at 30 fps. |
| Normalization | minmax against the calibration bounds: joints [-100, 100], gripper [0, 100] (`so101_lerobot_stats.json`) |
| Video | `concat_view`: front wrist camera stacked on top of the overhead camera, 480p |
| Method | LoRA rank 64 / alpha 128 on `q/k/v/o_proj_moe_gen`, plus full training of the action heads |
| Schedule | Global batch 32, learning rate 1e-4, 200 warm-up steps, linear decay over 7000 iterations |
| Hardware | Iterations 0–5500 on 1× RTX PRO 6000 (batch 16 × 2 accumulation steps); 5500–7000 resumed on 2× RTX PRO 6000 (batch 16 per GPU) |
| Embodiment domain id | 22 (`so101`) |

## Benchmark setup

`so101_bench` Isaac Lab digital twin, `So101Bench-Bin-v0`: 100 single-object episodes
(`tasks/focus5.jsonl`), 25 s per episode, 32 actions executed per inference call.

## Serving

Served with `cosmos_framework.scripts.action_policy_server_robolab`, an openpi websocket server.
Every SO-101 flag below is required, and a missing one fails silently: the server starts but
returns wrong actions.

```shell
huggingface-cli download kabilanKB/cosmos_edge_policy_so101 --include "iter_7000/*" "so101_lerobot_stats.json" \
    --local-dir cosmos_edge_policy_so101

python -m cosmos_framework.scripts.action_policy_server_robolab \
    --checkpoint-path cosmos_edge_policy_so101/iter_7000 \
    --port 8000 \
    --domain-name so101 \
    --action-dim 6 \
    --arm-joint-dim 5 \
    --action-space joint_pos \
    --conditioning-fps 30 \
    --no-flip-gripper \
    --action-normalization minmax \
    --normalizer-stats-path cosmos_edge_policy_so101/so101_lerobot_stats.json \
    --view-description 'The top half is from the front-facing wrist camera. The bottom half is from the fixed overhead camera.' \
    --no-guardrails
```

These flags depend on SO-101 support in the policy server (`--arm-joint-dim`, `--no-flip-gripper`,
`--action-normalization`, `--view-description`), which is not in upstream cosmos-framework `5e67049`.
The pipeline for training, merge, export, serving and evaluation is in
[kabilankb/so101-cosmos-nano-policy](https://github.com/kabilankb/so101-cosmos-nano-policy).

**Request format:** `prompt` (the instruction), `observation/image` (the concatenated view),
`observation/joint_position` (5 values), `observation/gripper_position` (1 value).

**Response:** `action`, 32 × 6 absolute joint targets in raw `.pos` units.

## Limitations

- Trained and evaluated only in simulation. It has not been tested on a physical SO-101.
- Covers five objects plus one multi-object instruction, with 18–45 demonstrations per instruction.
- Sampling is stochastic: the server draws a new seed for every request.

## License and attribution

This model is a derivative of **NVIDIA Cosmos3-Edge-Policy-DROID**, released under the
[OpenMDW License 1.1](https://openmdw.ai/license/1-1/), and is distributed under the same license.
The bundled vision encoder, processor and tokenizer files come from `nvidia/Cosmos3-Edge`.

Built with NVIDIA Cosmos. Post-trained by Kabilan KB.
