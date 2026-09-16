#!/bin/bash
set -uo pipefail
RUN=$HOME/cosmos-framework/outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu
echo "waiting for verified sync of iter_000007000 $(date +%T)"
until [ -f $RUN/checkpoints_brev/iter_000007000/.complete ]; do sleep 30; done
echo "SYNC VERIFIED $(date +%T)"
~/edge_hf/merge_export.sh 7000 > ~/edge_hf/merge_export_7000.log 2>&1 || { echo "MERGE/EXPORT FAILED"; grep -E "Traceback|Error" ~/edge_hf/merge_export_7000.log | tail -3; exit 1; }
grep -E "merged [0-9]+ adapter|domain_name" ~/edge_hf/merge_export_7000.log | cut -c1-120
diff <(ls $RUN/model_export_5500) <(ls $RUN/model_export_brev_7000) >/dev/null && echo "layout ok" || { echo "LAYOUT MISMATCH"; exit 1; }
~/edge_hf/upload.sh 7000 > ~/edge_hf/upload_7000.log 2>&1 || { echo "UPLOAD FAILED"; tail -5 ~/edge_hf/upload_7000.log; exit 1; }
python3 - <<PY
import json, os, urllib.request
RUN=os.path.expanduser("$RUN/model_export_brev_7000")
tree=json.load(urllib.request.urlopen("https://huggingface.co/api/models/kabilanKB/cosmos_edge_policy_so101/tree/main/iter_7000?recursive=true"))
hf={f["path"][len("iter_7000/"):]:f.get("size") for f in tree if f["type"]=="file"}
n=bad=0
for root,_,files in os.walk(RUN):
    for fn in files:
        p=os.path.join(root,fn); rel=os.path.relpath(p,RUN); n+=1
        if hf.get(rel)!=os.path.getsize(p): bad+=1; print("MISMATCH",rel)
info=json.load(urllib.request.urlopen("https://huggingface.co/api/models/kabilanKB/cosmos_edge_policy_so101"))
print(f"iter_7000 verified: {n} files, mismatches={bad} | repo folders: {sorted({s['rfilename'].split('/')[0] for s in info['siblings']})}")
PY
echo "HF PUSH 7000 DONE $(date +%T)"
