#!/usr/bin/env bash
# Runs ON the Brev instance. Resumes action_policy_so101_edge_focus5_multi_1gpu
# from its latest checkpoint (iter_000005500) to max_iter 7000.
#
# Global batch is kept at 32, as on the single-GPU workstation run:
#   workstation: 16 per step x 1 GPU x grad_accum 2
#   here:        16 per step x NGPU  x grad_accum (32 / (16 * NGPU))
#
#   remote_train.sh           full resume
#   SMOKE=1 remote_train.sh   3 iterations, separate output root, no checkpoint
set -uo pipefail

source "$HOME/edge_env.sh"
cd "$HOME/cosmos-framework"

NGPU=$(nvidia-smi -L | wc -l)
ACCUM=$(( 32 / (16 * NGPU) ))
(( ACCUM >= 1 )) || ACCUM=1
echo "[train] GPUs=$NGPU grad_accum=$ACCUM global_batch=$(( 16 * NGPU * ACCUM ))"

export NPROC_PER_NODE=$NGPU
# Resume comes from latest_checkpoint.txt under the output root; the launcher only
# requires BASE_CHECKPOINT_PATH to exist.
export BASE_CHECKPOINT_PATH="$HOME/cosmos-framework/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu/checkpoints/iter_000005500"

OVERRIDES="model.config.parallelism.data_parallel_shard_degree=$NGPU trainer.grad_accum_iter=$ACCUM"
if [[ "${SMOKE:-0}" == "1" ]]; then
    export OUTPUT_ROOT="$HOME/smoke_out"
    SMOKE_RUN="$OUTPUT_ROOT/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu/checkpoints"
    rm -rf "$OUTPUT_ROOT"; mkdir -p "$SMOKE_RUN"
    ln -s "$BASE_CHECKPOINT_PATH" "$SMOKE_RUN/iter_000005500"
    echo iter_000005500 > "$SMOKE_RUN/latest_checkpoint.txt"
    OVERRIDES="$OVERRIDES trainer.max_iter=5503 checkpoint.save_iter=100000 trainer.logging_iter=1"
    export LOG_FILENAME=edge_resume_smoke.log
fi
export EXTRA_TAIL_OVERRIDES="$OVERRIDES"

exec bash examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh
