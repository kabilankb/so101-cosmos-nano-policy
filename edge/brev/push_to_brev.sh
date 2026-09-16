#!/usr/bin/env bash
# Runs ON the workstation. Pushes code, scripts, the run config and one checkpoint
# to a Brev instance, then verifies the checkpoint by md5.
#
#   push_to_brev.sh <user@brev-ip> [iteration]      default iteration: latest local
#
# Reach the instance with the Brev key forwarded from the laptop (the key never
# leaves the laptop):
#   laptop$ eval "$(ssh-agent -s)"; ssh-add ~/.brev/brev.pem
#   laptop$ ssh -A -p <port> <user>@<workstation> 'bash ~/brev_edge/push_to_brev.sh <brev-user>@<ip>'
set -uo pipefail
HOST="${1:?usage: push_to_brev.sh <user@brev-ip> [iteration]}"
CF="$HOME/cosmos-framework"
RUN_REL="outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
HERE="$(cd "$(dirname "$0")" && pwd)"
# -n on every direct ssh: without it, ssh reads stdin and swallows the rest of a
# script fed through `bash -s`.
SSHO="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

if [[ -n "${2:-}" ]]; then IT="iter_$(printf %09d "$2")"; else IT=$(tr -d '[:space:]' < "$CF/$RUN_REL/checkpoints/latest_checkpoint.txt"); fi
SRC_CKPT="$CF/$RUN_REL/checkpoints/$IT"
[[ -d "$SRC_CKPT" ]] || SRC_CKPT="$CF/$RUN_REL/checkpoints_brev/$IT"
[[ -d "$SRC_CKPT/model" ]] || { echo "checkpoint not found: $IT"; exit 1; }
echo "[push] checkpoint $IT from $SRC_CKPT"

echo "[push] code"
rsync -az --delete -e "$SSHO" \
    --exclude outputs --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude 'examples/checkpoints' --exclude 'examples/data' --exclude '*.pdf' \
    "$CF/" "$HOST:cosmos-framework/" </dev/null || exit 1

echo "[push] scripts"
rsync -az -e "$SSHO" "$HERE/remote_setup.sh" "$HERE/remote_train.sh" "$HOST:" </dev/null || exit 1

echo "[push] run config + checkpoint (~6.6 GB)"
$SSHO -n "$HOST" "mkdir -p cosmos-framework/$RUN_REL/checkpoints" || exit 1
rsync -a -e "$SSHO" "$CF/$RUN_REL/config.yaml" "$CF/$RUN_REL/config.pkl" "$CF/$RUN_REL/wandb_id.txt" \
    "$HOST:cosmos-framework/$RUN_REL/" </dev/null || exit 1
# --partial + --append-verify: an interrupted copy resumes instead of restarting.
# Never run two of these at once against the same checkpoint.
rsync -a --partial --append-verify -e "$SSHO" "$SRC_CKPT/" "$HOST:cosmos-framework/$RUN_REL/checkpoints/$IT/" </dev/null || exit 1

echo "[push] verify md5 of every checkpoint file"
L=$(cd "$SRC_CKPT" && find . -type f ! -name .complete | sort | xargs md5sum)
R=$($SSHO -n "$HOST" "cd cosmos-framework/$RUN_REL/checkpoints/$IT && find . -type f | sort | xargs md5sum")
[[ "$L" == "$R" ]] || { echo "VERIFY FAILED"; exit 1; }
$SSHO -n "$HOST" "echo $IT > cosmos-framework/$RUN_REL/checkpoints/latest_checkpoint.txt"
echo "[push] PUSH DONE: $IT verified, $(echo "$L" | wc -l) files"
