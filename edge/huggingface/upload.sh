#!/bin/bash
set -e
export PATH=$HOME/.local/bin:$PATH
R=kabilanKB/cosmos_edge_policy_so101
IT=$1
RUN=$HOME/cosmos-framework/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu
echo "START $IT $(date -Is)"
uvx hf@latest upload $R $RUN/model_export_brev_$IT iter_$IT --repo-type model --commit-message "Add SO-101 Edge policy checkpoint iter $IT (merged + exported)"
uvx hf@latest upload $R $HOME/edge_hf/so101_lerobot_stats.json so101_lerobot_stats.json --commit-message "Add action normalizer stats"
uvx hf@latest upload $R $HOME/edge_hf/README.md README.md --commit-message "Update model card"
echo "UPLOAD DONE $IT $(date -Is)"
