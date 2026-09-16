#!/usr/bin/env bash
# Runs ON the workstation. Every 5 minutes, copies complete new checkpoints from
# the Brev run back into the workstation's run directory, into a separate
# checkpoints_brev/ folder so the local latest_checkpoint.txt is untouched.
# Each copy is verified by md5 of every file before it is marked complete.
set -uo pipefail
HOST="${1:?usage: sync_back.sh <user@brev-host>}"
SSHO="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
RUN_REL="outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
DST="$HOME/cosmos-framework/$RUN_REL/checkpoints_brev"
mkdir -p "$DST"

while true; do
    latest=$($SSHO -n "$HOST" "cat cosmos-framework/$RUN_REL/checkpoints/latest_checkpoint.txt" 2>/dev/null | tr -d '[:space:]')
    for it in $($SSHO -n "$HOST" "ls cosmos-framework/$RUN_REL/checkpoints 2>/dev/null | grep -E '^iter_[0-9]{9}$'" 2>/dev/null); do
        [[ "$it" == iter_000005500 ]] && continue
        [[ -f "$DST/$it/.complete" ]] && continue
        # Only copy once the trainer has finished writing it (latest pointer at or past it).
        [[ -z "$latest" || "$latest" < "$it" ]] && continue
        echo "[sync $(date +%H:%M)] copying $it"
        if rsync -a --partial -e "$SSHO" "$HOST:cosmos-framework/$RUN_REL/checkpoints/$it" "$DST/" </dev/null; then
            L=$(cd "$DST" && find "$it" -type f ! -name .complete | sort | xargs md5sum)
            R=$($SSHO -n "$HOST" "cd cosmos-framework/$RUN_REL/checkpoints && find $it -type f | sort | xargs md5sum")
            if [[ "$L" == "$R" ]]; then
                touch "$DST/$it/.complete"
                echo "[sync $(date +%H:%M)] SYNCED $it ($(du -sh "$DST/$it" | cut -f1), $(echo "$L" | wc -l) files verified)"
            else
                echo "[sync $(date +%H:%M)] VERIFY FAILED $it, will retry"
            fi
        else
            echo "[sync $(date +%H:%M)] rsync failed for $it, will retry"
        fi
    done
    if $SSHO -n "$HOST" "grep -qE 'Done \\(exit' ~/train.log 2>/dev/null"; then
        echo "[sync $(date +%H:%M)] TRAINING PROCESS FINISHED on $HOST: $($SSHO -n "$HOST" "grep -E 'Done \\(exit' ~/train.log | tail -1")"
        n_pending=$($SSHO -n "$HOST" "ls cosmos-framework/$RUN_REL/checkpoints | grep -cE '^iter_[0-9]{9}$'")
        n_done=$(( $(ls -d "$DST"/iter_*/.complete 2>/dev/null | wc -l) + 1 ))
        if (( n_done >= n_pending )); then echo "[sync $(date +%H:%M)] ALL CHECKPOINTS SYNCED"; exit 0; fi
    fi
    sleep 300
done
