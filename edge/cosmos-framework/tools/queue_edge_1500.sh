#!/usr/bin/env bash
# Wait for the Nano horizon-16 A/B to finish, then hand the GPU over to the
# Cosmos3-Edge iter-1500 evaluation.
#
# Why queued rather than concurrent: the A/B holds ~32 GB (policy server) plus
# ~18 GB (Isaac Sim), training holds ~29 GB, and an Edge server + a second Isaac
# Sim needs roughly 40 GB more than the ~17 GB left. Running both OOMs.
#
# The Edge evaluation deliberately runs at action_horizon 32, NOT 16: every Nano
# number we are comparing against (13/437 = 3.0%) was measured at 32, and
# changing two variables at once would make the Edge result uninterpretable. A
# horizon-16 Edge run is a separate follow-up once the A/B verdict is known.
set -uo pipefail

FRAMEWORK=/home/<user>/cosmos-framework
SO101=/home/<user>/so101-cosmos
AB_LOG=$FRAMEWORK/outputs/so101_cosmos_jobs/evaluate_3750_20260911_215316.log
export PATH="/home/<user>/.local/bin:$PATH"   # uvx, needed to resolve the Wan2.2 VAE

log(){ echo "[queue] $(date '+%F %T') $*"; }

# --- 1. wait for the A/B client to exit -------------------------------------
log "waiting for the horizon-16 A/B to finish"
while pgrep -f "cosmos3_eval.py" >/dev/null 2>&1; do sleep 60; done
TOTAL=$(grep -c "success=" "$AB_LOG" 2>/dev/null || echo 0)
OK=$(grep -c "success=True" "$AB_LOG" 2>/dev/null || echo 0)
log "A/B finished: $OK/$TOTAL successes at action_horizon 16 (baseline was 4/100 at 32)"

# --- 2. release the Nano server ---------------------------------------------
# Matched by the checkpoint it serves, so a restarted server is still found.
NANO_PID=$(pgrep -f "action_policy_server_robolab.*model_export_3750" | head -1)
if [[ -n "${NANO_PID:-}" ]]; then
    log "stopping Nano policy server (pid $NANO_PID)"
    kill "$NANO_PID" 2>/dev/null
    for i in $(seq 1 30); do kill -0 "$NANO_PID" 2>/dev/null || break; sleep 5; done
    kill -0 "$NANO_PID" 2>/dev/null && { log "SIGKILL"; kill -9 "$NANO_PID"; sleep 5; }
fi

# Wait for the allocator to actually hand the memory back, not just for the
# process table to clear.
for i in $(seq 1 24); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    log "free VRAM: ${FREE} MiB"
    [[ "${FREE:-0}" -gt 40000 ]] && break
    sleep 10
done

# --- 3. serve the Edge 1500 export ------------------------------------------
log "starting Edge iter-1500 policy server"
cd "$SO101" || exit 1
.venv/bin/so101 --config so101-edge.toml serve --iter 1500 --detach 2>&1 | sed 's/^/[queue] /'

log "waiting for /healthz"
READY=0
for i in $(seq 1 90); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/healthz 2>/dev/null)" == "200" ]]; then
        READY=1; log "server healthy after $((i*20))s"; break
    fi
    sleep 20
done
[[ "$READY" == "1" ]] || { log "FATAL: server never became healthy -- see the serve job log"; exit 1; }

# --- 4. run the Isaac Lab client --------------------------------------------
log "starting Edge iter-1500 evaluation at action_horizon 32"
.venv/bin/so101 --config so101-edge.toml eval --iter 1500 --detach 2>&1 | sed 's/^/[queue] /'
log "handed off; watch with: so101 --config so101-edge.toml jobs"
