# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_so101_edge`` — Cosmos3-Edge SO-101 action-policy SFT recipe.

Edge-tier sibling of ``action_policy_so101_nano``. Everything about the data
path is shared and imported from that module — same ``SO101LeRobotDataset``
(``joint_pos`` 6-D + ``use_state``, minmax-normalized against the SO-101 ``.pos``
calibration bounds, ``concat_view`` front-over-overhead video, chunk_length 32),
same ``PackingDataLoader`` / ``RankPartitionedDataLoader`` stack, same
``keep_tasks`` subsets. Only the model tier and the checkpoint-loading policy
differ.

Why Edge instead of Nano on this box
------------------------------------
Edge is the *smaller* tier, not the more accurate one: on the public RoboLab
suite the released DROID policies score Edge 22.9% vs Nano 36.8%. The reason to
prefer it here is not the released number, it is what fits. Nano's backbone is
Qwen3-VL-8B, which on one 96 GB GPU only trained as rank-16 LoRA on
``*_moe_gen`` at global batch 32. Edge's backbone is Nemotron-2B-Dense-VL —
roughly a quarter of the trainable surface — so the same GPU affords a much
larger fraction of the model actually being updated, a bigger batch, and more
epochs over the same 128 training episodes. On a dataset this small the binding
constraint is adaptation capacity per unit of VRAM, not pretrained ceiling.

Two Edge deltas fix latent mismatches in the Nano recipe
--------------------------------------------------------
* ``resolution``: ``EDGE_MODEL_CONFIG`` is natively ``"480"``, which matches the
  480x640-per-camera SO-101 data. ``NANO_MODEL_CONFIG`` is ``"720"`` and the
  SO-101 Nano recipe never overrode it, so training drew its flow-matching
  ``shift`` from the 720 entry (10) while the dataloader emitted 480p
  (``omni_mot_model.py`` picks ``shift[config.resolution]``). Edge gets the
  correct ``shift=5``.
* ``rectified_flow_training_config``: Edge ships ``loss_scale=10.0`` /
  ``image_loss_scale=None`` (vs Nano's 1.0/1.0), which is the "vision
  flow-matching weighted to match action loss" balance the Edge post-training
  recipe calls for.

Warm-started action heads
-------------------------
``action_policy_so101_nano`` initialises from the *base* ``nvidia/Cosmos3-Nano``,
whose action heads carry no SO-101 signal, so it lists them in
``keys_to_skip_loading`` and they start from random — the standard convention in
the upstream DROID/LIBERO recipes.

These recipes instead initialise from ``nvidia/Cosmos3-Edge-Policy-DROID``, a
checkpoint that has *already* been post-trained into a manipulation policy, and
deliberately keep its action heads out of ``keys_to_skip_loading``.

That alone is **not enough**, and the reason is worth stating precisely because
it is easy to get wrong. ``action2llm`` / ``llm2action`` are ``DomainAwareLinear``
layers (``mot/domain_aware_linear.py``): their weights live in an
``nn.Embedding(num_embodiment_domains, out*in)``, so every embodiment owns a
private row. DROID is domain 8; SO-101 is domain 22. Measured on the released
checkpoint (exported as ``action_proj_in`` / ``action_proj_out``)::

    action_proj_in.fc.weight    row 8 = 10.67    row 22 = 6.44   (== every other row)
    action_proj_in.bias.weight  row 8 =  1.56    row 22 = 0.000  (== every other row)

Row 22 is still at ``xavier_uniform_`` init with an exactly-zero bias. So simply
loading the DROID policy hands SO-101's action head nothing the base checkpoint
would not have given it. The only part of the action path that transfers for
free is ``action_modality_embed``, which is a single shared ``[hidden_size]``
vector, not a per-domain table.

``tools/transplant_domain.py`` closes that gap by copying row 8 into row 22
before DCP conversion, so SO-101 starts from DROID's trained manipulation
mapping. Both are single-arm absolute ``joint_pos`` policies padded into the same
64-wide action space (DROID 7 joints + gripper, SO-101 5 joints + gripper), so
the token<->hidden mapping is largely shared structure. Point
``BASE_CHECKPOINT_PATH`` at the transplanted conversion to get the warm start;
point it at the plain conversion for the un-transplanted control.

Set ``SO101_EDGE_COLD_HEADS=1`` to fall back to the fresh-init convention
entirely (and raise the action-head LR multipliers back to 5x, as in the Nano
recipe) for an A/B.

Usage (single GPU)::

    bash examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh

See docs/so101_edge_posttrain.md for the full pipeline (DCP conversion of the
Edge policy checkpoint, smoke run, export, serving and evaluation).
"""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_so101_nano import (
    SO101_FOCUS5_TASKS,
    SO101_MULTI_TASK,
    action_policy_so101_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG

cs = ConfigStore.instance()


# Base checkpoint is Cosmos3-Edge-Policy-DROID, which ships trained action heads.
# Keep them (see module docstring); skip only EMA and the LoRA adapters, which
# never exist in a base checkpoint.
_WARM_HEAD_SKIP_KEYS = [
    "net_ema.",
]

# Opt-out: fresh-init the action heads exactly like the Nano recipe, for an
# apples-to-apples comparison against the Nano baseline or when initialising
# from the *base* nvidia/Cosmos3-Edge instead of the DROID policy.
_COLD_HEAD_SKIP_KEYS = [
    "net_ema.",
    "action2llm",
    "llm2action",
    "action_modality_embed",
    "action_pos_embed",
]

_COLD_HEADS = os.environ.get("SO101_EDGE_COLD_HEADS", "0").lower() in ("1", "true", "yes")


def _make_edge_variant(name: str, *, keep_tasks=None, max_episodes_per_task=None, val_ratio=None):
    """Derive an Edge-tier variant from the shared SO-101 Nano recipe.

    Only the model tier and the checkpoint-loading policy change; the whole data
    path is inherited so the two tiers cannot drift apart.
    """
    cfg = copy.deepcopy(action_policy_so101_nano)
    cfg["job"]["name"] = name
    cfg["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["checkpoint"]["keys_to_skip_loading"] = list(
        _COLD_HEAD_SKIP_KEYS if _COLD_HEADS else _WARM_HEAD_SKIP_KEYS
    )

    dataset = cfg["dataloader_train"]["dataloader"]["datasets"]["so101"]["dataset"]
    # Re-point the interpolation at the Edge model config now that it was replaced.
    dataset["max_action_dim"] = "${model.config.max_action_dim}"
    dataset["tokenizer_config"] = "${model.config.vlm_config.tokenizer}"
    if keep_tasks is not None:
        dataset["keep_tasks"] = keep_tasks
    if max_episodes_per_task is not None:
        dataset["max_episodes_per_task"] = max_episodes_per_task
    if val_ratio is not None:
        dataset["val_ratio"] = val_ratio
    return cfg


# Full SO-101 root, all instructions.
action_policy_so101_edge = _make_edge_variant("action_policy_so101_edge")

# Five single-object bin-placement instructions (97 episodes / 35,132 frames).
action_policy_so101_edge_focus5 = _make_edge_variant(
    "action_policy_so101_edge_focus5",
    keep_tasks=SO101_FOCUS5_TASKS,
    val_ratio=0.10,
)

# focus5 plus the one multi-object instruction, capped so it cannot swamp them.
# This is the mix the Nano run converged on and the intended Edge launch target.
action_policy_so101_edge_focus5_multi = _make_edge_variant(
    "action_policy_so101_edge_focus5_multi",
    keep_tasks=SO101_FOCUS5_TASKS + (SO101_MULTI_TASK,),
    max_episodes_per_task=45,
    val_ratio=0.10,
)


for _item in [
    action_policy_so101_edge,
    action_policy_so101_edge_focus5,
    action_policy_so101_edge_focus5_multi,
]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
