#!/usr/bin/env bash
# merge_export.sh <iteration>  -- merge LoRA then export one Brev-trained Edge checkpoint.
set -euo pipefail
IT=$(printf "%09d" "$1")
CF=$HOME/cosmos-framework
RUN=$CF/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu
SRC=$RUN/checkpoints_brev/iter_$IT
OUT=$RUN/model_export_brev_$1
cd "$CF"
export PATH=$CF/.venv/bin:$HOME/.local/bin:$PATH VIRTUAL_ENV=$CF/.venv
export LD_LIBRARY_PATH=$CF/.venv/lib/python3.13/site-packages/nvidia/cu13/lib
export SO101_ROOT=$CF/examples/data/so101_bench_sim_6
[[ -f $SRC/.complete ]] || { echo "not a verified sync: $SRC"; exit 1; }
echo "== merge $(date +%T)"
python -u -m cosmos_framework.scripts.merge_lora_dcp --input "$SRC" --output "${SRC}_merged" --lora-alpha 128 --lora-rank 64 --overwrite
[[ -f ${SRC}_merged/model/.metadata ]] || { echo "merge produced no .metadata"; exit 1; }
echo "== export $(date +%T)"
python -u -m cosmos_framework.scripts.export_model --checkpoint-path "${SRC}_merged" \
  --config-file cosmos_framework/configs/base/config.py --experiment action_policy_so101_edge_focus5_multi \
  --experiment-overrides model.config.diffusion_expert_config.load_weights_from_pretrained=False \
  model.config.vlm_config.pretrained_weights.enabled=False checkpoint.load_from_object_store.enabled=False model.config.ema.enabled=false \
  -o "$OUT"
ls "$OUT"; du -sh "$OUT"
grep -o '"domain_name": "[a-z0-9]*"' "$OUT/checkpoint.json"
echo "== MERGE_EXPORT DONE $1 $(date +%T)"
