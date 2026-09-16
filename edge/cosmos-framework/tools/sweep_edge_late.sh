#!/usr/bin/env bash
# Evaluate the late Cosmos3-Edge checkpoints once training has finished.
#
# Sweeps 5500, 6000, 6500, 7000 -- epochs 3.18, 3.47, 3.75, 4.05 -- which is the
# window where the Nano run produced its only scoring checkpoint (4.17). Earlier
# checkpoints are not worth the GPU time: Nano was zero at every checkpoint
# through epoch 3.89, and Edge 1500 (0.87) and 2500 (1.44) both returned zero.
#
# For each checkpoint: merge LoRA -> export -> serve -> wait /healthz -> run the
# Isaac Lab client -> stop the server -> record. Sequential by construction, so
# only one policy server and one Isaac Sim ever hold the GPU.
#
# action_horizon stays at 32. Horizon 16 was tested and is worse (0/63 vs 4/70
# on identical episodes): 32 steps at 30 Hz is a 1.07 s replan interval, which
# matches NVIDIA's reference cadence.
#
#   nohup setsid tools/sweep_edge_late.sh > outputs/edge_setup/sweep.log 2>&1 &
set -uo pipefail

FRAMEWORK=/home/<user>/cosmos-framework
SO101=/home/<user>/so101-cosmos
CFG=$SO101/so101-edge.toml
CKPTS=(5500 6000 6500 7000)
SUMMARY=$FRAMEWORK/outputs/edge_setup/sweep_results.tsv
export PATH="/home/<user>/.local/bin:$PATH"   # uvx, needed to resolve the Wan2.2 VAE

log(){ echo "[sweep] $(date '+%F %T') $*"; }
cd "$SO101" || exit 1

# --- wait for training to finish --------------------------------------------
if pgrep -f "[c]osmos_framework.scripts.train" >/dev/null 2>&1; then
    log "training still running -- waiting for it to finish"
    while pgrep -f "[c]osmos_framework.scripts.train" >/dev/null 2>&1; do sleep 120; done
    log "training exited"
    sleep 60   # let the allocator settle
fi

printf 'iter\tepoch\tepisodes\tsuccesses\trate\n' > "$SUMMARY"

for IT in "${CKPTS[@]}"; do
    CKPT_DIR=$FRAMEWORK/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu/checkpoints/iter_$(printf '%09d' "$IT")
    if [[ ! -d "$CKPT_DIR/model" ]]; then
        log "iter $IT: no checkpoint on disk, skipping"
        continue
    fi
    EPOCH=$(python3 -c "print(f'{$IT/1731.4:.2f}')")
    log "=== iter $IT (epoch $EPOCH) ==="

    .venv/bin/so101 --config "$CFG" merge  --iter "$IT" >/dev/null 2>&1 || { log "merge failed";  continue; }
    .venv/bin/so101 --config "$CFG" export --iter "$IT" >/dev/null 2>&1 || { log "export failed"; continue; }
    log "iter $IT: merged + exported"

    .venv/bin/so101 --config "$CFG" serve --iter "$IT" --detach >/dev/null 2>&1
    READY=0
    for i in $(seq 1 90); do
        if [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://localhost:8000/healthz 2>/dev/null)" == "200" ]]; then
            READY=1; break
        fi
        sleep 20
    done
    if [[ "$READY" != "1" ]]; then
        log "iter $IT: server never became healthy, skipping"
        kill "$(pgrep -f '[a]ction_policy_server_robolab' | head -1)" 2>/dev/null
        sleep 20
        continue
    fi
    log "iter $IT: server healthy, starting evaluation"

    # Foreground: the sweep must not start the next checkpoint until this ends.
    .venv/bin/so101 --config "$CFG" eval --iter "$IT" >/dev/null 2>&1

    EVLOG=$(ls -t "$FRAMEWORK"/outputs/so101_cosmos_jobs_edge/evaluate_"$IT"_*.log 2>/dev/null | head -1)
    TOTAL=0; OK=0
    if [[ -n "$EVLOG" ]]; then
        TOTAL=$(grep -c "success=" "$EVLOG" 2>/dev/null || echo 0)
        OK=$(grep -c "success=True" "$EVLOG" 2>/dev/null || echo 0)
    fi
    RATE=$(python3 -c "print(f'{100*$OK/$TOTAL:.1f}%' if $TOTAL else 'n/a')")
    log "iter $IT (epoch $EPOCH): $OK/$TOTAL = $RATE"
    printf '%s\t%s\t%s\t%s\t%s\n' "$IT" "$EPOCH" "$TOTAL" "$OK" "$RATE" >> "$SUMMARY"

    kill "$(pgrep -f '[a]ction_policy_server_robolab' | head -1)" 2>/dev/null
    sleep 25
done

log "=== SWEEP COMPLETE ==="
cat "$SUMMARY"
