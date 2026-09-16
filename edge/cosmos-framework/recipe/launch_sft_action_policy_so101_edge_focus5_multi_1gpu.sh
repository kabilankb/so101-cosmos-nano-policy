#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Single-GPU (1 x 96GB) launch for action_policy_so101_edge_focus5_multi against
# examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml.
#
# Edge-tier sibling of launch_sft_action_policy_so101_nano_focus5_multi_1gpu.sh:
# same dataset mix, same global batch, but the Cosmos3-Edge (Nemotron-2B-Dense-VL)
# tier initialised from the Cosmos3-Edge-Policy-DROID checkpoint so the action
# heads start trained rather than random. See the TOML header for the rationale.
#
# BASE_CHECKPOINT_PATH must point at a DCP conversion, not at the raw HF
# download. Build the default (warm SO-101 action head) with:
#
#   python tools/transplant_domain.py \
#       --src examples/checkpoints/Cosmos3-Edge-Policy-DROID \
#       --dst examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
#       --src-domain 8 --dst-domain 22
#   python -m cosmos_framework.scripts.convert_model_to_dcp \
#       --checkpoint-path examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
#       -o examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init-dcp
#
# The transplant step is not optional for a warm start: action2llm/llm2action are
# per-embodiment (nn.Embedding rows), DROID trained row 8, and SO-101's row 22
# ships untrained. Converting Cosmos3-Edge-Policy-DROID directly yields the COLD
# control -- valid as an A/B, but set the action-head lr_multipliers back to 5.0
# in the TOML if you use it.
#
# Optional env vars:
#   DATASET_PATH          default: examples/data/so101_bench_sim_6
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init-dcp
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   NPROC_PER_NODE        default: 1
#   SO101_EDGE_COLD_HEADS set to 1 to fresh-init the action heads (A/B against
#                         the Nano convention, or when starting from base Edge)
#   EXTRA_TAIL_OVERRIDES  space-separated Hydra overrides
#
# Smoke run (10 iters, no grad accumulation -- measures peak memory + iter time,
# and is the run that validates the LoRA target names against the Nemotron
# backbone before committing to a multi-day horizon):
#   EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 trainer.grad_accum_iter=1 \
#     checkpoint.save_iter=100000 trainer.callbacks.device_monitor.every_n=1" \
#     bash examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh

TOML_FILE="examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml"
: "${DATASET_PATH:=examples/data/so101_bench_sim_6}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init-dcp}"
: "${NPROC_PER_NODE:=1}"
export NPROC_PER_NODE

# Same allocator note as the Nano launcher: on a single 96GB card, fragmentation
# alone can fail a step, and expandable segments keeps the reserved-but-unallocated
# tail reusable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/meta/info.json" ]] || { echo "ERROR: missing $DATASET_PATH/meta/info.json -- stage with: uvx hf@latest download --repo-type dataset 5hadytru/so101_bench_sim_6 --local-dir $DATASET_PATH" >&2; exit 1; }'

# The registered action_policy_so101_edge* experiments read SO101_ROOT (inherited
# from the Nano recipe's dataloader). Anchor + export it here, mirroring
# _sft_launcher_common.sh's anchoring, because that script exports DATASET_PATH
# only after we would need SO101_ROOT set.
_WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$DATASET_PATH" = /* ]] || DATASET_PATH="$_WORKDIR/$DATASET_PATH"
export SO101_ROOT="$DATASET_PATH"

# Surface the head-init mode in the log: which of the two it ran is the single
# most important thing to know when comparing checkpoints later.
if [[ "${SO101_EDGE_COLD_HEADS:-0}" == "1" ]]; then
    echo "[so101-edge] action heads: COLD (fresh init, Nano convention)"
else
    echo "[so101-edge] action heads: WARM (loaded from Cosmos3-Edge-Policy-DROID)"
fi

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
