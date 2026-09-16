#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transplant a trained embodiment-domain row into an untrained one.

Why this exists
---------------
``action2llm`` / ``llm2action`` (exported as ``action_proj_in`` /
``action_proj_out``) are ``DomainAwareLinear`` layers: their weights live in an
``nn.Embedding(num_domains, out*in)``, so **every embodiment gets its own private
row**. ``nvidia/Cosmos3-Edge-Policy-DROID`` was post-trained on DROID only, which
is domain 8. SO-101 is domain 22.

Measured on the released checkpoint:

    action_proj_in.fc.weight   row 8 = 10.67   row 22 = 6.44  (== every other row)
    action_proj_in.bias.weight row 8 =  1.56   row 22 = 0.000 (== every other row)

So initialising SO-101 training from that checkpoint gives the action head
*nothing* over the base model -- row 22 is still at ``xavier_uniform_`` init and
the bias is still exactly zero. The only part of the action path that transfers
for free is ``action_modality_embed``, which is a single shared 2048-vector.

This script copies row 8 into row 22 so the SO-101 action head starts from
DROID's trained manipulation mapping instead of from noise. Both embodiments are
single-arm absolute ``joint_pos`` policies padded into the same 64-wide action
space (DROID 7 joints + gripper, SO-101 5 joints + gripper), so the mapping
between action tokens and hidden space is largely shared structure even though
the per-column joint semantics differ.

Only shard 2 carries the per-domain tensors, so every other file is hardlinked
rather than copied.

Usage::

    python transplant_domain.py \
        --src examples/checkpoints/Cosmos3-Edge-Policy-DROID \
        --dst examples/checkpoints/Cosmos3-Edge-Policy-DROID-so101init \
        --src-domain 8 --dst-domain 22
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PER_DOMAIN_KEYS = (
    "action_proj_in.fc.weight",
    "action_proj_in.bias.weight",
    "action_proj_out.fc.weight",
    "action_proj_out.bias.weight",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--src-domain", type=int, default=8, help="trained domain (DROID)")
    ap.add_argument("--dst-domain", type=int, default=22, help="target domain (SO-101)")
    args = ap.parse_args()

    src_tf = args.src / "transformer"
    dst_tf = args.dst / "transformer"
    index_path = src_tf / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]

    target_shards = {weight_map[k] for k in PER_DOMAIN_KEYS}
    if len(target_shards) != 1:
        raise SystemExit(f"expected all per-domain tensors in one shard, got {target_shards}")
    target_shard = target_shards.pop()
    print(f"per-domain tensors live in: {target_shard}")

    # Mirror the whole checkpoint, hardlinking everything we are not rewriting.
    if args.dst.exists():
        raise SystemExit(f"destination already exists: {args.dst}")
    for dirpath, _dirnames, filenames in os.walk(args.src):
        rel = Path(dirpath).relative_to(args.src)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        (args.dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            s, d = Path(dirpath) / fn, args.dst / rel / fn
            if rel == Path("transformer") and fn == target_shard:
                continue  # rewritten below
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    # Rewrite the one shard that carries the per-domain rows.
    tensors: dict[str, torch.Tensor] = {}
    metadata = {}
    with safe_open(str(src_tf / target_shard), framework="pt") as f:
        metadata = f.metadata() or {}
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    for k in PER_DOMAIN_KEYS:
        t = tensors[k]
        before = t[args.dst_domain].float().norm().item()
        t[args.dst_domain] = t[args.src_domain].clone()
        after = t[args.dst_domain].float().norm().item()
        src_n = t[args.src_domain].float().norm().item()
        print(
            f"  {k:30s} row{args.dst_domain}: {before:8.4f} -> {after:8.4f} "
            f"(src row{args.src_domain} = {src_n:8.4f})"
        )

    save_file(tensors, str(dst_tf / target_shard), metadata=metadata)
    print(f"wrote {dst_tf / target_shard}")
    print("done")


if __name__ == "__main__":
    main()
