#!/usr/bin/env bash
# Runs ON the Brev instance. Installs the environment and stages data for
# resuming action_policy_so101_edge_focus5_multi_1gpu from iter_000005500.
# The code and checkpoint are pushed separately by push_to_brev.sh.
set -euo pipefail

CF="$HOME/cosmos-framework"
log() { echo "[setup $(date +%H:%M:%S)] $*"; }

log "GPU / driver"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if (( DRIVER_MAJOR >= 580 )); then GROUP=cu130-train; LIBSUB=cu13; else GROUP=cu128-train; LIBSUB=""; fi
log "driver $DRIVER_MAJOR -> uv group $GROUP"

log "system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends curl ffmpeg git git-lfs libx11-dev rsync tmux >/dev/null

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd "$CF"
log "uv sync ($GROUP)"
uv sync --all-extras --group="$GROUP"

log "dataset so101_bench_sim_6 (~18 GB)"
mkdir -p examples/data examples/checkpoints/wan22_vae
uvx hf@latest download --repo-type dataset 5hadytru/so101_bench_sim_6 --local-dir examples/data/so101_bench_sim_6 >/dev/null

log "Wan2.2 VAE (~2.7 GB)"
uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
    --local-dir examples/checkpoints/wan22_vae >/dev/null

log "env file"
{
    echo "export PATH=\"$CF/.venv/bin:$HOME/.local/bin:\$PATH\""
    echo "export VIRTUAL_ENV=\"$CF/.venv\""
    if [[ -n "$LIBSUB" ]]; then
        echo "export LD_LIBRARY_PATH=\"$CF/.venv/lib/python3.13/site-packages/nvidia/$LIBSUB/lib\""
    else
        echo "export LD_LIBRARY_PATH="
    fi
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
} > "$HOME/edge_env.sh"

source "$HOME/edge_env.sh"
log "torch check"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
log "SETUP DONE"
