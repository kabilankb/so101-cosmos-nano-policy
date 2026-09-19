#!/usr/bin/env bash
# One-time setup of the SO-101 Cosmos3-Edge policy server on a Jetson Thor (JetPack 7, CUDA 13).
#
#   bash setup_thor.sh [install dir]          # default: ~/so101-edge-thor
#
# Needs: git, internet, ~20 GB free disk. No sudo. Installs uv into ~/.local/bin if missing.
# Result: <dir>/cosmos-framework (commit 5e67049 + SO-101 patch, uv venv with the cu130 group)
#         <dir>/checkpoints/iter_6500 (the released Edge 6500 export, ~7.3 GB)
set -euo pipefail

DIR=${1:-$HOME/so101-edge-thor}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FRAMEWORK_COMMIT=5e67049
HF_REPO=${SO101_HF_REPO:-kabilanKB/cosmos_edge_policy_so101}
ITER=${SO101_ITER:-6500}

log() { echo "[setup-thor] $*"; }
[ "$(uname -m)" = aarch64 ] || log "warning: this script is written for the aarch64 Jetson Thor"

mkdir -p "$DIR/logs"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

free_gb=$(df -BG --output=avail "$DIR" | tail -1 | tr -dc 0-9)
if [ ! -f "$DIR/checkpoints/iter_$ITER/checkpoint.json" ] && [ "$free_gb" -lt 20 ]; then
  log "warning: only ${free_gb} GB free; the venv, checkpoint and VAE need ~20 GB"
fi

# 1. cosmos-framework at the commit the policy was trained and exported with, plus SO-101 support
if [ ! -d "$DIR/cosmos-framework/.git" ]; then
  log "cloning cosmos-framework"
  git clone -q https://github.com/NVIDIA/cosmos-framework.git "$DIR/cosmos-framework"
fi
cd "$DIR/cosmos-framework"
git checkout -q "$FRAMEWORK_COMMIT"
if git apply --check "$HERE/so101-support.patch" 2>/dev/null; then
  git apply "$HERE/so101-support.patch"; log "applied SO-101 patch"
elif git apply --reverse --check "$HERE/so101-support.patch" 2>/dev/null; then
  log "SO-101 patch already applied"
else
  log "error: the SO-101 patch does not apply to $(git log --oneline -1)"; exit 1
fi

# 2. Python environment. The cu130 group has aarch64 builds of torch 2.10+cu130; flash-attn
#    is x86-only and is not needed for serving.
log "uv sync (10-30 min the first time)"
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
# cu130: aarch64 torch 2.10+cu130. policy-server: OpenPI's WebSocket server. train extra: modules
# the server imports (iopath, ...); no training runs here.
uv sync --group cu130 --group policy-server --extra train
# cuDNN >= 9.22 enables the framework's cuDNN attention backend on Thor (sm_110): attention is
# ~5x faster than the fallback and a chunk drops from ~4.3 s to ~2.1 s (with torch.compile).
# torch 2.10 ships 9.15; cuDNN 9 minor releases are ABI compatible. Re-run after any `uv sync`.
uv pip install --python .venv/bin/python "nvidia-cudnn-cu13>=9.22"

# 3. The policy checkpoint (public repo, no token needed)
if [ ! -f "$DIR/checkpoints/iter_$ITER/checkpoint.json" ]; then
  log "downloading $HF_REPO iter_$ITER"
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uvx hf@latest download "$HF_REPO" --include "iter_$ITER/*" \
    --local-dir "$DIR/checkpoints"
fi

# 4. Pre-fetch the Wan2.2 VAE the server loads at start (so the first start is not a download)
log "pre-fetching the Wan2.2 VAE"
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e >/dev/null

cp "$HERE/serve_thor.sh" "$DIR/serve_thor.sh" 2>/dev/null || true
log "done. Launch the server from the PC web UI, or here with:  bash $DIR/serve_thor.sh --tmux"
