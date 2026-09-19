#!/usr/bin/env bash
# One-time setup of the SO-101 Isaac Lab client + web UI on the machine that runs the simulation.
#
#   bash setup_laptop.sh --python /path/to/isaaclab-env/bin/python [install dir]
#
# --python: the Python of an environment that already has Isaac Sim 6.0 + Isaac Lab 3.0
#           (e.g. ~/miniconda3/envs/env_isaaclab/bin/python). Nothing is installed into it.
# Install dir defaults to ~/so101-thor-client. Result:
#   so101_bench/   upstream 5hadytru/so101_bench @ 205d4f9 + so101_bench_isaaclab3.patch + USD assets
#   vendor_py/     openpi-client + pyzmq for the client (kept out of the Isaac Lab env)
#   webui/         the runner page (python webui/so101_thor_ui.py --thor <thor-ip>)
set -euo pipefail

PY=""; DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY=$2; shift 2;;
    -h|--help) sed -n 2,12p "$0"; exit 0;;
    *) DIR=$1; shift;;
  esac
done
DIR=${DIR:-$HOME/so101-thor-client}
PY=${PY:-$(command -v python || true)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH_COMMIT=205d4f9
log() { echo "[setup-laptop] $*"; }

[ -x "$PY" ] || { log "pass --python <isaaclab env python>"; exit 1; }
"$PY" - <<'PYEOF' || { echo "[setup-laptop] that python has no Isaac Lab 3 / Isaac Sim 6"; exit 1; }
import importlib.metadata as m
v = m.version("isaacsim"); print(f"[setup-laptop] isaacsim {v}, isaaclab {m.version('isaaclab')}")
assert int(v.split(".")[0]) >= 6, "Isaac Sim 6.x required (this client is ported to Isaac Lab 3.0)"
PYEOF

mkdir -p "$DIR"
# 1. so101_bench at the upstream commit, plus the SO-101 Cosmos client and the Isaac Lab 3 port
if [ ! -d "$DIR/so101_bench/.git" ]; then
  log "cloning 5hadytru/so101_bench"
  git clone -q https://github.com/5hadytru/so101_bench.git "$DIR/so101_bench"
fi
cd "$DIR/so101_bench"
git checkout -q "$BENCH_COMMIT"
if git apply --check "$HERE/so101_bench_isaaclab3.patch" 2>/dev/null; then
  git apply "$HERE/so101_bench_isaaclab3.patch"; log "applied so101_bench_isaaclab3.patch"
elif git apply --reverse --check "$HERE/so101_bench_isaaclab3.patch" 2>/dev/null; then
  log "patch already applied"
else
  log "error: so101_bench_isaaclab3.patch does not apply"; exit 1
fi

# 2. USD assets (~430 MB, gitignored upstream)
ASSETS=source/so101_bench/so101_bench/assets
if [ ! -f "$ASSETS/usd/room_scan.usdc" ]; then
  log "downloading USD assets from 5hadytru/so101_bench_assets"
  TMP=$(mktemp -d)
  "$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('5hadytru/so101_bench_assets', 'so101_bench_usd_assets.tar.gz', repo_type='dataset', local_dir='$TMP'))" >/dev/null
  tar -xzf "$TMP/so101_bench_usd_assets.tar.gz" -C "$ASSETS/"
  rm -rf "$TMP"
fi

# 3. Client-only packages, installed next to (not into) the Isaac Lab environment
log "installing openpi-client + pyzmq into $DIR/vendor_py"
"$PY" -m pip install -q --target "$DIR/vendor_py" --no-deps --upgrade openpi-client==0.1.2 pyzmq

# 4. Web UI
mkdir -p "$DIR/webui"
cp "$HERE/webui/so101_thor_ui.py" "$HERE/webui/index.html" "$DIR/webui/"
cat > "$DIR/run_ui.sh" <<RUN
#!/usr/bin/env bash
# Open http://127.0.0.1:8765/ after starting. --thor <ip> --thor-user <user> pre-fill the Thor fields.
exec "$PY" "$DIR/webui/so101_thor_ui.py" --bench "$DIR/so101_bench" "\$@"
RUN
chmod +x "$DIR/run_ui.sh"
log "done. Start the page with:  $DIR/run_ui.sh --thor <thor-ip> --thor-user <thor-user>   then open http://127.0.0.1:8765/"
