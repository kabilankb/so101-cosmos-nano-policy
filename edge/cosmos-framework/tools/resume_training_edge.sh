#!/usr/bin/env bash
# Resume the SO-101 Cosmos3-EDGE LoRA SFT run from its last checkpoint.
#
# Edge-tier sibling of tools/resume_training.sh. Targets
# action_policy_so101_edge_focus5_multi (128 train episodes, 1,731 iters/epoch,
# 7000 iters = 4.05 epochs) and the Edge launcher. The Nano runs are untouched.
#
# Safe to run at any time -- including automatically at boot after a power cut.
#
#   1. If a trainer is ALREADY running, this script WAITS for it rather than
#      exiting. That difference matters: tools/resume_training.sh exits 0
#      immediately in that case, and under `Restart=always` systemd then
#      restarts it every RestartSec until StartLimitBurst trips and the unit
#      dies permanently with 'start-limit-hit'. That is exactly how
#      cosmos-so101-train.service failed on 2026-09-07 and stayed dead for four
#      days. Blocking here keeps one long-lived supervised process instead.
#   2. Validates the checkpoint latest_checkpoint.txt points at, and rolls back
#      to the previous complete one if a power cut truncated it mid-write.
#   3. Exports the LD_LIBRARY_PATH cuBLASLt fix this box needs (venv cuBLAS 13.1
#      paired with system cuBLASLt 13.4 makes every biased addmm fail with
#      CUBLAS_STATUS_NOT_INITIALIZED).
#
# Training itself resumes automatically: the checkpointer prefers
# latest_checkpoint.txt over checkpoint.load_path, so relaunching the same job
# name continues from the saved model + optimizer + scheduler state. Checkpoints
# are every 500 iters (~2.8 h), so a crash costs at most that much.
#
#   bash tools/resume_training_edge.sh                # run it directly
#   systemctl --user start cosmos-so101-edge-train    # or via the paired unit
set -uo pipefail

FRAMEWORK="/home/<user>/cosmos-framework"
RUN_DIR="$FRAMEWORK/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
CKPT_DIR="$RUN_DIR/checkpoints"
LATEST="$CKPT_DIR/latest_checkpoint.txt"
LAUNCH="examples/launch_sft_action_policy_so101_edge_focus5_multi_1gpu.sh"
VENV_CU13="$FRAMEWORK/.venv/lib/python3.13/site-packages/nvidia/cu13/lib"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-30}"

log(){ echo "[resume-edge] $(date '+%F %T') $*"; }

cd "$FRAMEWORK" || { log "FATAL: $FRAMEWORK missing"; exit 1; }

# --- 1. supervise, don't duplicate ------------------------------------------
# Adopt an already-running trainer (e.g. the initial nohup launch) by waiting it
# out. When it exits -- cleanly or by crashing -- we fall through and relaunch,
# which is what makes this unit a real restart-on-crash supervisor.
if pgrep -f "cosmos_framework.scripts.train" >/dev/null 2>&1; then
    log "a trainer is already running (pid $(pgrep -f 'cosmos_framework.scripts.train' | head -1)) -- supervising it"
    while pgrep -f "cosmos_framework.scripts.train" >/dev/null 2>&1; do
        sleep "$WAIT_POLL_SECONDS"
    done
    log "the running trainer exited -- taking over"
    # If it exited because the whole 7000-iteration run finished, stop here
    # instead of relaunching a completed job in a loop.
    if [[ -f "$LATEST" ]]; then
        done_iter="$(tr -dc '0-9' < "$LATEST")"
        max_iter="$(grep -oP '^\s*max_iter\s*=\s*\K[0-9]+' \
            "$FRAMEWORK/examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml" | head -1)"
        if [[ -n "$done_iter" && -n "$max_iter" && "$((10#$done_iter))" -ge "$max_iter" ]]; then
            log "run already reached max_iter ($max_iter) -- nothing to resume"
            exit 0
        fi
    fi
fi

# --- 2. validate the checkpoint a power cut may have truncated ---------------
# A complete DCP checkpoint has all four state dirs and a non-empty model/.
checkpoint_complete(){
    local d="$1"
    [[ -d "$d/model" && -d "$d/optim" && -d "$d/scheduler" && -d "$d/trainer" ]] || return 1
    [[ -n "$(ls -A "$d/model" 2>/dev/null)" ]] || return 1
    return 0
}

if [[ -f "$LATEST" ]]; then
    name="$(tr -d '[:space:]' < "$LATEST")"
    target="$CKPT_DIR/$name"
    if [[ -n "$name" ]] && ! checkpoint_complete "$target"; then
        log "WARNING: $name is incomplete (likely interrupted mid-write)"
        prev=""
        while IFS= read -r d; do
            [[ "$(basename "$d")" == "$name" ]] && continue
            if checkpoint_complete "$d"; then prev="$(basename "$d")"; break; fi
        done < <(find "$CKPT_DIR" -maxdepth 1 -type d -name 'iter_*' ! -name '*_merged' | sort -r)

        if [[ -n "$prev" ]]; then
            [[ -d "$target" ]] && mv "$target" "${target}.incomplete.$(date +%s)" \
                && log "moved the truncated checkpoint aside (kept, not deleted)"
            echo "$prev" > "$LATEST"
            log "rolled latest_checkpoint.txt back to $prev"
        else
            log "no complete checkpoint found -- clearing latest_checkpoint.txt, run starts from base"
            rm -f "$LATEST"
        fi
    else
        log "resuming from $name"
    fi
else
    log "no latest_checkpoint.txt -- starting from the base checkpoint"
fi

# --- 3. launch ---------------------------------------------------------------
export LD_LIBRARY_PATH="$VENV_CU13:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PATH="$FRAMEWORK/.venv/bin:$PATH"
export VIRTUAL_ENV="$FRAMEWORK/.venv"
export HF_TOKEN="${HF_TOKEN:-$(cat /home/<user>/.cache/huggingface/token 2>/dev/null || true)}"

LOG="$FRAMEWORK/outputs/edge_setup/train_edge_resume_$(date +%Y%m%d_%H%M%S).log"
log "launching $LAUNCH -> $LOG"

# Keep the dashboard's canonical log path working across restarts: it follows
# outputs/edge_setup/train_edge.log, so point that at the newest run log.
ln -sfn "$LOG" "$FRAMEWORK/outputs/edge_setup/train_edge_current.log"

# exec so systemd tracks the real process.
exec bash "$LAUNCH" > "$LOG" 2>&1
