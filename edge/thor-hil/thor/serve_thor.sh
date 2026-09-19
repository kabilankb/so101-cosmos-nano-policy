#!/usr/bin/env bash
# Start the SO-101 Cosmos3-Edge policy server on the Jetson Thor (OpenPI WebSocket, port 8000).
#
#   bash serve_thor.sh              # foreground
#   bash serve_thor.sh --detach     # background; log in <dir>/logs/serve_<stamp>.log
#   bash serve_thor.sh --tmux       # tmux session "so101serve"; log in <dir>/logs/serve_latest.log
#                                   # (what the PC web UI's "Launch server" button runs over SSH)
#   bash serve_thor.sh --stop       # stop the tmux / detached server
#
# Health: curl localhost:8000/healthz  ->  OK     (from another machine: curl <thor-ip>:8000/healthz)
set -euo pipefail

DIR=${SO101_THOR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
[ -d "$DIR/cosmos-framework" ] || DIR=$HOME/so101-edge-thor
FW=$DIR/cosmos-framework
ITER=${SO101_ITER:-6500}
CKPT=${SO101_CHECKPOINT:-$DIR/checkpoints/iter_$ITER}
PORT=${SO101_PORT:-8000}

export PATH="$FW/.venv/bin:$HOME/.local/bin:$PATH"   # uvx resolves the cached VAE
export VIRTUAL_ENV=$FW/.venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# torch.compile: the venv's Triton ships a ptxas without sm_110a (Thor), so point Triton at
# JetPack's CUDA 13 ptxas. SO101_COMPILE=0 falls back to eager (NVIDIA's documented Thor
# setting, TORCHDYNAMO_DISABLE=1): same actions, ~1.5x slower per chunk (3.15 s vs 2.1 s).
if [ "${SO101_COMPILE:-1}" = "1" ] && [ -x /usr/local/cuda/bin/ptxas ]; then
  export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
else
  export TORCHDYNAMO_DISABLE=1
fi
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# The venv's own CUDA 13 libraries first, so a system CUDA cannot shadow libcublasLt.
CU13=$(ls -d "$FW"/.venv/lib/python3.*/site-packages/nvidia/cu13/lib 2>/dev/null | tail -1 || true)
[ -n "$CU13" ] && export LD_LIBRARY_PATH="$CU13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Every flag below is part of the SO-101 serving contract and fails silently if dropped:
# 5 arm joints (DROID has 7), gripper on [0, 100] (no flip), min-max normalization with the
# SO-101 stats (else joints are commanded to ~0), 30 fps conditioning, and the two-view layout
# the model was trained on (wrist on top, overhead below).
ARGS=(
  -u -m cosmos_framework.scripts.action_policy_server_robolab
  --checkpoint-path "$CKPT"
  --port "$PORT"
  --domain-name so101
  --action-dim 6
  --arm-joint-dim 5
  --action-space joint_pos
  --conditioning-fps 30
  --no-flip-gripper
  --action-normalization minmax
  --normalizer-stats-path cosmos_framework/data/generator/action/normalizer_stats/so101_lerobot_stats.json
  --view-description "The top half is from the front-facing wrist camera. The bottom half is from the fixed overhead camera."
  --no-guardrails
)

cd "$FW"
SESSION=${SO101_TMUX_SESSION:-so101serve}
if [ "${1:-}" = "--stop" ]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped tmux session $SESSION"
  pkill -f "[a]ction_policy_server_robolab.*--port $PORT" && echo "stopped server on port $PORT"
  exit 0
fi
[ -f "$CKPT/checkpoint.json" ] || { echo "no checkpoint at $CKPT (run setup_thor.sh or download it from the web UI)"; exit 1; }
if [ "${1:-}" = "--tmux" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then echo "already running in tmux session $SESSION"; exit 0; fi
  mkdir -p "$DIR/logs"; LOG=$DIR/logs/serve_$(date +%Y%m%d_%H%M%S).log
  ln -sfn "$LOG" "$DIR/logs/serve_latest.log"
  # tmux sessions take their environment from the tmux server, so pass the settings explicitly.
  tmux new-session -d -s "$SESSION" "SO101_THOR_DIR='$DIR' SO101_CHECKPOINT='$CKPT' SO101_PORT=$PORT SO101_COMPILE=${SO101_COMPILE:-1} bash '$DIR/serve_thor.sh' 2>&1 | tee '$LOG'"
  echo "policy server starting in tmux session $SESSION, log $LOG"
  exit 0
fi
if [ "${1:-}" = "--detach" ]; then
  mkdir -p "$DIR/logs"; LOG=$DIR/logs/serve_$(date +%Y%m%d_%H%M%S).log
  setsid nohup python "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  echo "policy server pid $! log $LOG"
  echo "ready when:  grep -q 'ready domain' $LOG   or   curl -s localhost:$PORT/healthz"
else
  exec python "${ARGS[@]}"
fi
