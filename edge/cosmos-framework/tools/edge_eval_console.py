#!/usr/bin/env python3
"""Web control panel for Cosmos3-Edge SO-101 post-training evaluation.

A launcher for the server/client pair that evaluates a trained epoch against the
so101_bench Isaac Lab digital twin:

  * Isaac Lab preflight -- Kit python, openpi-client, so101_bench, tasks, layouts
  * checkpoint list with epoch numbers, merge/export state, and past results
  * one-click launch of the full pipeline, or of any single stage
  * live job logs, and a policy-server health indicator
  * a guard that tells you when a launch would fight the live trainer for the GPU

The heavy lifting stays in tools/edge_eval_epoch.py; this page only launches it
and streams its logs, so the two cannot drift apart.

SECURITY: the buttons run shell commands, so this binds to 127.0.0.1 and refuses
any other interface unless you pass --unsafe-allow-remote. Do not expose it.
The read-only telemetry page (tools/edge_train_dashboard.py, port 8810) is the
one that is safe on the LAN.

    python3 tools/edge_eval_console.py                 # http://127.0.0.1:8811
    python3 tools/edge_eval_console.py --port 9100
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FRAMEWORK = Path("/home/<user>/cosmos-framework")
RUN_DIR = FRAMEWORK / "outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
CKPT_DIR = RUN_DIR / "checkpoints"
EVAL_DIR = FRAMEWORK / "outputs/edge_eval"
JOB_DIR = FRAMEWORK / "outputs/edge_eval_jobs"

VENV_PY = FRAMEWORK / ".venv/bin/python"
SO101_BENCH = Path("/home/<user>/IsaacLab/so101_bench")
KIT_PY = Path("/home/<user>/IsaacLab/_isaac_sim/python.sh")
EVAL_TOOL = FRAMEWORK / "tools/edge_eval_epoch.py"
STATS = FRAMEWORK / "cosmos_framework/data/generator/action/normalizer_stats/so101_lerobot_stats.json"

ITERS_PER_EPOCH = 55385 / 32
POLICY_PORT = 8000

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


# ------------------------------------------------------------------ helpers

def sh(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:
        return 1, str(exc)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def training_running() -> bool:
    return subprocess.run(["pgrep", "-f", "cosmos_framework.scripts.train"],
                          capture_output=True).returncode == 0


def server_health(port: int = POLICY_PORT) -> str:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=2) as r:
            return "ok" if r.status == 200 else f"http {r.status}"
    except Exception:
        return "down"


# ---------------------------------------------------------------- preflight

_pre_cache: dict = {"t": 0.0, "rows": []}
_PRE_TTL = 120.0


def preflight() -> list[dict]:
    """Cached preflight rows.

    The Kit-python import probes cost seconds apiece and the page polls every
    4 s, so recomputing per request would stack slow subprocesses until the
    console stopped responding. A background thread refreshes this; requests
    only ever read the cache.
    """
    return _pre_cache["rows"]


def preflight_refresher() -> None:
    while True:
        try:
            rows = _preflight_compute()
            _pre_cache["rows"] = rows
            _pre_cache["t"] = time.time()
        except Exception as exc:
            _pre_cache["rows"] = [{"name": "preflight", "ok": False,
                                   "detail": str(exc), "fix": ""}]
        time.sleep(_PRE_TTL)


def _preflight_compute() -> list[dict]:
    """Everything that must be true before a client run can work."""
    rows: list[dict] = []

    def add(name, ok, detail, fix=""):
        rows.append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix})

    add("Isaac Sim Kit python", KIT_PY.is_file(), str(KIT_PY),
        "Isaac Sim is not installed at the expected path")

    if KIT_PY.is_file():
        rc, out = sh([str(KIT_PY), "-c",
                      "import openpi_client,sys;print(getattr(openpi_client,'__version__','installed'))"], 180)
        add("openpi-client in Kit python", rc == 0, out.splitlines()[-1] if out else "missing",
            f"{KIT_PY} -m pip install openpi-client")
        rc2, out2 = sh([str(KIT_PY), "-c", "import omni;print('omni ok')"], 180)
        add("omni importable (Isaac Sim)", rc2 == 0,
            out2.splitlines()[-1] if out2 else "missing",
            "Use the Kit python, not a conda env — no conda env here has omni")

    add("so101_bench checkout", SO101_BENCH.is_dir(), str(SO101_BENCH))
    add("so101_bench source on PYTHONPATH",
        (SO101_BENCH / "source/so101_bench").is_dir(),
        str(SO101_BENCH / "source/so101_bench"),
        "PYTHONPATH is set automatically by the launcher")
    add("client script", (SO101_BENCH / "scripts/cosmos3_eval.py").is_file(),
        "scripts/cosmos3_eval.py",
        "untracked file — see docs/so101_bench_eval_launch.md for the required patches")

    tasks = sorted((SO101_BENCH / "tasks").glob("*.jsonl")) if (SO101_BENCH / "tasks").is_dir() else []
    add("episode task files", bool(tasks), f"{len(tasks)} found: " +
        ", ".join(t.name for t in tasks[:4]))
    lay = sorted((SO101_BENCH / "tasks/layouts").glob("focus5_layouts_*.jsonl")) \
        if (SO101_BENCH / "tasks/layouts").is_dir() else []
    add("saved focus5 layouts", bool(lay),
        lay[-1].name if lay else "none — layouts will be sampled (~22 s/episode)")

    add("eval orchestrator", EVAL_TOOL.is_file(), str(EVAL_TOOL.name))
    add("normalizer stats", STATS.is_file(), STATS.name)
    add("policy server port free or serving",
        server_health() in ("ok", "down"), f":{POLICY_PORT} {server_health()}")

    rc, out = sh(["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                  "--format=csv,noheader,nounits"])
    if rc == 0 and out:
        used, total, temp = (x.strip() for x in out.split(",")[:3])
        free = (float(total) - float(used)) / 1024
        add("GPU headroom for Isaac Sim", free > 12,
            f"{free:.1f} GiB free, {temp} C",
            "Isaac Sim needs roughly 10-12 GiB alongside the server")

    add("trainer idle", not training_running(),
        "training is RUNNING — a launch will compete for the GPU"
        if training_running() else "no trainer running",
        "Use 'queue behind training' so the run waits instead of competing")
    return rows


# -------------------------------------------------------------- checkpoints

def checkpoints() -> list[dict]:
    if not CKPT_DIR.is_dir():
        return []
    rows = []
    for d in sorted(CKPT_DIR.glob("iter_*")):
        if not d.is_dir() or d.name.endswith("_merged"):
            continue
        n = int(re.sub(r"\D", "", d.name) or 0)
        res = EVAL_DIR / d.name / "result.json"
        r = None
        if res.is_file():
            try:
                r = json.loads(res.read_text())
            except Exception:
                r = None
        rows.append({
            "iter": n, "name": d.name, "epoch": n / ITERS_PER_EPOCH,
            "merged": (CKPT_DIR / f"{d.name}_merged").is_dir(),
            "exported": (RUN_DIR / f"model_export_{n}" / "checkpoint.json").is_file(),
            "success_rate": (r or {}).get("success_rate"),
            "episodes": (r or {}).get("episodes"),
            "successes": (r or {}).get("successes"),
        })
    return rows


# -------------------------------------------------------------- job control

ACTIONS = {
    "pipeline":  "Merge → export → serve → evaluate (headless)",
    "pipeline_gui": "Merge → export → serve → evaluate (Isaac Sim viewer)",
    "serve":     "Start only the policy server (for driving the client by hand)",
    "sweep":     "Evaluate every checkpoint without a result yet",
}


def build_cmd(action: str, it: int | None, opts: dict) -> tuple[list[str], Path]:
    episodes = str(int(opts.get("episodes", 20)))
    concurrent = bool(opts.get("concurrent"))
    task = str(opts.get("task") or "So101Bench-Bin-v0")
    jsonl = str(opts.get("jsonl") or "tasks/focus5.jsonl")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", task + jsonl):
        raise ValueError("bad task/jsonl")

    if action == "serve":
        if it is None:
            raise ValueError("serve needs a checkpoint")
        exp = RUN_DIR / f"model_export_{it}"
        if not (exp / "checkpoint.json").is_file():
            raise ValueError(f"iter {it} has no export yet — run the pipeline first")
        return ([str(VENV_PY), "-u", "-m",
                 "cosmos_framework.scripts.action_policy_server_robolab",
                 "--checkpoint-path", str(exp), "--port", str(POLICY_PORT),
                 "--domain-name", "so101", "--action-dim", "6", "--arm-joint-dim", "5",
                 "--action-space", "joint_pos", "--conditioning-fps", "30",
                 "--no-flip-gripper", "--action-normalization", "minmax",
                 "--normalizer-stats-path", str(STATS), "--no-guardrails",
                 "--device-memory-bytes", "50000000000"], FRAMEWORK)

    cmd = ["python3", "-u", str(EVAL_TOOL), "--num-episodes", episodes,
           "--task", task, "--episodes-jsonl", jsonl]
    cmd += ["--all"] if action == "sweep" else ["--iter", str(int(it))]
    if action == "pipeline_gui":
        cmd.append("--gui")
    cmd.append("--allow-concurrent" if concurrent else "--defer-until-training-done")
    return (cmd, FRAMEWORK)


def launch(action: str, it: int | None, opts: dict) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    with _lock:
        for j in _jobs.values():
            if j["status"] == "running" and alive(j["pid"]):
                if action == "serve" and j["action"] == "serve":
                    raise RuntimeError("a policy server is already running")
                if action != "serve" and j["action"] != "serve":
                    raise RuntimeError(f"{j['action']} is already running")
        cmd, cwd = build_cmd(action, it, opts)
        JOB_DIR.mkdir(parents=True, exist_ok=True)
        jid = f"{action}_{it or 'all'}_{datetime.now():%Y%m%d_%H%M%S}"
        logf = JOB_DIR / f"{jid}.log"
        env = dict(os.environ)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env["SO101_ROOT"] = str(FRAMEWORK / "examples/data/so101_bench_sim_6")
        with logf.open("wb") as fh:
            p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        job = {"id": jid, "action": action, "iter": it, "pid": p.pid,
               "status": "running", "log": str(logf), "started": time.time(),
               "cmd": " ".join(shlex.quote(c) for c in cmd)}
        _jobs[jid] = job
        return job


def stop(jid: str) -> dict:
    with _lock:
        job = _jobs.get(jid)
        if not job:
            raise ValueError("no such job")
        if alive(job["pid"]):
            try:
                os.killpg(os.getpgid(job["pid"]), 15)
            except Exception:
                pass
        job["status"] = "stopped"
        return job


def refresh_jobs() -> list[dict]:
    with _lock:
        for j in _jobs.values():
            if j["status"] == "running" and not alive(j["pid"]):
                j["status"] = "finished"
        return sorted(_jobs.values(), key=lambda j: j["started"], reverse=True)


def tail(path: str, n: int = 400) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with p.open("rb") as f:
        if size > 300_000:
            f.seek(size - 300_000)
            f.readline()
        return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-n:])


# ------------------------------------------------------------------- server

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge SO-101 Eval Console</title><style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212a;--line:#272c37;--fg:#e6e9ef;
--dim:#939bab;--accent:#66d9a8;--accent2:#6aa9ff;--warn:#f0b866;--bad:#f0736a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:19px;margin:0 0 2px}.sub{color:var(--dim);font-size:12.5px;margin-bottom:20px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);margin:0 0 11px;font-weight:600}
.grid{display:grid;gap:14px;grid-template-columns:1fr 1fr}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
.ok{color:var(--accent)}.warn{color:var(--warn)}.bad{color:var(--bad)}.dimt{color:var(--dim)}
button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:7px;
padding:6px 11px;font-size:12.5px;cursor:pointer;font-weight:550}
button:hover:not(:disabled){border-color:var(--accent2);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:#1d3a2e;border-color:#2c5c47;color:#bff0d8}
button.danger{background:#3a1f1d;border-color:#5c2f2c;color:#f0c0bc}
select,input{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
border-radius:7px;padding:6px 9px;font-size:12.5px}
label{font-size:12.5px;color:var(--dim);display:inline-flex;align-items:center;gap:6px}
pre{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:12px;
overflow:auto;max-height:420px;font-size:11.5px;line-height:1.5;margin:0;
font-family:ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}
.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:99px;
font-size:11.5px;font-weight:600;border:1px solid var(--line);background:var(--panel2)}
.dot{width:7px;height:7px;border-radius:99px;background:var(--dim)}
.dot.on{background:var(--accent)}.dot.off{background:var(--bad)}.dot.warn{background:var(--warn)}
.note{color:var(--dim);font-size:12px;margin-top:10px}
.banner{border-radius:9px;padding:10px 13px;margin-bottom:14px;font-size:13px;border:1px solid}
.banner.warn{background:#2a2214;border-color:#5c4a24;color:#f0d8a8}
.banner.bad{background:#2a1614;border-color:#5c2f2c;color:#f0c0bc}
</style></head><body><div class="wrap">
<h1>Cosmos3-Edge &middot; SO-101 evaluation console</h1>
<div class="sub">Server + client launcher for the so101_bench Isaac Lab digital twin &middot;
  telemetry dashboard on <span class="mono">:8810</span></div>

<div id="banner"></div>

<div class="card"><h2>Launch an epoch</h2>
  <div class="row">
    <select id="ckpt"></select>
    <label>episodes <input id="episodes" type="number" value="20" min="1" max="100" style="width:72px"></label>
    <label><input id="concurrent" type="checkbox"> run now (compete with training)</label>
  </div>
  <div class="row" style="margin-top:11px">
    <button class="primary" onclick="go('pipeline')">Evaluate (headless)</button>
    <button onclick="go('pipeline_gui')">Evaluate with viewer</button>
    <button onclick="go('serve')">Serve only</button>
    <button onclick="go('sweep')">Sweep all unevaluated</button>
  </div>
  <div class="note">Unchecked, a launch queues behind the trainer and starts when it exits
    (<span class="mono">--defer-until-training-done</span>). Checked, it runs immediately and
    shares the GPU &mdash; which slows training and raises temperature.
    &ldquo;Serve only&rdquo; starts the policy server and leaves it up so you can drive
    <span class="mono">cosmos3_eval.py</span> by hand.</div></div>

<div class="grid">
  <div class="card"><h2>Isaac Lab preflight</h2><div id="pre"></div></div>
  <div class="card"><h2>Checkpoints</h2><div id="ck"></div></div>
</div>

<div class="card"><h2>Jobs</h2><div id="jobs"></div></div>
<div class="card"><h2>Log <span id="logname" class="mono dimt" style="text-transform:none;letter-spacing:0"></span></h2>
  <pre id="log">select a job…</pre></div>

<div class="card"><h2>Manual server / client commands</h2>
  <div class="note" style="margin-top:0">If you would rather run the two halves yourself.
    They need different interpreters: the server uses the cosmos-framework venv, the client
    uses Isaac Sim's Kit python (the only one here with <span class="mono">omni</span>).</div>
  <pre style="margin-top:10px" id="manual"></pre></div>

<script>
const $=id=>document.getElementById(id);
let sel=null, state=null;

function go(action){
  const it=$("ckpt").value;
  fetch("/api/launch",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({action, iter: it?parseInt(it):null,
      opts:{episodes:parseInt($("episodes").value)||20, concurrent:$("concurrent").checked}})})
   .then(r=>r.json()).then(j=>{ if(j.error) alert(j.error); else { sel=j.id; tick(); } });
}
function stopJob(id){ fetch("/api/stop",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify({id})}).then(()=>tick()); }
function pick(id){ sel=id; tick(); }

function fmtAgo(t){const s=Date.now()/1000-t;return s<60?Math.round(s)+"s":
  s<3600?Math.round(s/60)+"m":(s/3600).toFixed(1)+"h";}

async function tick(){
  let d; try{ d=await (await fetch("/api/state",{cache:"no-store"})).json(); }catch(e){ return; }
  state=d;

  const bad=d.preflight.filter(p=>!p.ok);
  const tr=d.training_running;
  let b="";
  if(tr) b+=`<div class="banner warn"><b>Training is running.</b> Launching now shares the GPU
    with it &mdash; slower training and higher temperature. Leave &ldquo;run now&rdquo; unchecked
    to queue instead.</div>`;
  if(bad.length) b+=`<div class="banner bad"><b>${bad.length} preflight check(s) failing.</b>
    ${bad.map(p=>p.name).join(", ")} &mdash; a client run will not work until these pass.</div>`;
  $("banner").innerHTML=b;

  $("pre").innerHTML='<table><tbody>'+d.preflight.map(p=>
    `<tr><td style="width:22px"><i class="dot ${p.ok?"on":"off"}"></i></td>
     <td>${p.name}<div class="note" style="margin:1px 0 0">${p.detail||""}${
       !p.ok&&p.fix?' &middot; <span class="warn">'+p.fix+'</span>':''}</div></td></tr>`).join("")
    +'</tbody></table>';

  const cks=d.checkpoints||[];
  const cur=$("ckpt").value;
  $("ckpt").innerHTML=cks.length?cks.slice().reverse().map(c=>
    `<option value="${c.iter}">iter ${c.iter} &middot; epoch ${c.epoch.toFixed(2)}${
      c.success_rate!=null?"  ("+(c.success_rate*100).toFixed(0)+"%)":""}</option>`).join("")
    :'<option value="">no checkpoint yet</option>';
  if(cur) $("ckpt").value=cur;

  $("ck").innerHTML=cks.length?
    '<table><thead><tr><th>iter</th><th class="num">epoch</th><th class="num">merged</th>'
    +'<th class="num">export</th><th class="num">success</th></tr></thead><tbody>'
    +cks.slice().reverse().map(c=>`<tr><td class="mono">${c.iter}</td>
      <td class="num">${c.epoch.toFixed(2)}</td>
      <td class="num ${c.merged?"ok":"dimt"}">${c.merged?"yes":"–"}</td>
      <td class="num ${c.exported?"ok":"dimt"}">${c.exported?"yes":"–"}</td>
      <td class="num ${c.success_rate==null?"dimt":c.success_rate>0?"ok":"bad"}">${
        c.success_rate==null?"–":c.successes+"/"+c.episodes+"  "+(c.success_rate*100).toFixed(0)+"%"}</td>
      </tr>`).join("")+"</tbody></table>"
    :'<div class="note" style="margin-top:0">No checkpoint saved yet — the first lands at iteration 500.</div>';

  const js=d.jobs||[];
  $("jobs").innerHTML=js.length?
    '<table><thead><tr><th>job</th><th>action</th><th class="num">age</th><th>status</th><th></th></tr></thead><tbody>'
    +js.map(j=>`<tr><td class="mono" style="font-size:11.5px">
      <a href="#" onclick="pick('${j.id}');return false" style="color:var(--accent2)">${j.id}</a></td>
      <td>${j.action}</td><td class="num">${fmtAgo(j.started)}</td>
      <td class="${j.status==="running"?"ok":j.status==="stopped"?"warn":"dimt"}">${j.status}</td>
      <td style="text-align:right">${j.status==="running"
        ?`<button class="danger" onclick="stopJob('${j.id}')">stop</button>`:""}</td></tr>`).join("")
    +"</tbody></table>"
    :'<div class="note" style="margin-top:0">No jobs launched in this session.</div>';

  $("manual").textContent=d.manual;

  if(!sel && js.length) sel=js[0].id;
  if(sel){
    $("logname").textContent=sel;
    try{ const t=await (await fetch("/api/log?job="+encodeURIComponent(sel),{cache:"no-store"})).text();
      const el=$("log"); const stick=el.scrollTop+el.clientHeight>=el.scrollHeight-40;
      el.textContent=t||"(empty)"; if(stick) el.scrollTop=el.scrollHeight; }catch(e){}
  }
}
tick(); setInterval(tick,4000);
</script></div></body></html>"""


def manual_text() -> str:
    return (
        "# 1. SERVER  (cosmos-framework venv)\n"
        f"cd {FRAMEWORK}\n"
        "source .venv/bin/activate\n"
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n"
        "SO101_ROOT=examples/data/so101_bench_sim_6 \\\n"
        "python -m cosmos_framework.scripts.action_policy_server_robolab \\\n"
        f"    --checkpoint-path {RUN_DIR}/model_export_<ITER> \\\n"
        f"    --port {POLICY_PORT} --domain-name so101 --action-dim 6 --arm-joint-dim 5 \\\n"
        "    --action-space joint_pos --conditioning-fps 30 --no-flip-gripper \\\n"
        "    --action-normalization minmax \\\n"
        f"    --normalizer-stats-path {STATS.relative_to(FRAMEWORK)} \\\n"
        "    --no-guardrails --device-memory-bytes 50000000000\n"
        f"curl http://localhost:{POLICY_PORT}/healthz     # -> OK\n\n"
        "# 2. CLIENT  (Isaac Sim Kit python, second terminal)\n"
        f"cd {SO101_BENCH}\n"
        f"PYTHONPATH={SO101_BENCH}/source/so101_bench \\\n"
        f"{KIT_PY} -u scripts/cosmos3_eval.py \\\n"
        "    --task So101Bench-Bin-v0 --episodes_jsonl tasks/focus5.jsonl \\\n"
        f"    --policy_host localhost --policy_port {POLICY_PORT} \\\n"
        "    --action_horizon 32 --num_episodes 20 --headless\n\n"
        "# 3. RESULT\n"
        "cat <output_dir>/So101Bench-Bin-v0/log_0_env0.json\n\n"
        "# Or let the orchestrator do all of it:\n"
        "python3 tools/edge_eval_epoch.py --iter <ITER> --defer-until-training-done\n"
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/state":
            data = {
                "preflight": preflight(),
                "checkpoints": checkpoints(),
                "jobs": refresh_jobs(),
                "training_running": training_running(),
                "server": server_health(),
                "manual": manual_text(),
            }
            self._send(200, json.dumps(data).encode(), "application/json")
        elif u.path == "/api/log":
            jid = (parse_qs(u.query).get("job") or [""])[0]
            with _lock:
                job = _jobs.get(jid)
            self._send(200, tail(job["log"]).encode() if job else b"", "text/plain; charset=utf-8")
        else:
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, b'{"error":"bad json"}', "application/json")
        try:
            if self.path == "/api/launch":
                job = launch(str(body.get("action", "")), body.get("iter"),
                             body.get("opts") or {})
            elif self.path == "/api/stop":
                job = stop(str(body.get("id", "")))
            else:
                return self._send(404, b'{"error":"no route"}', "application/json")
            self._send(200, json.dumps(job).encode(), "application/json")
        except Exception as exc:
            self._send(200, json.dumps({"error": str(exc)}).encode(), "application/json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--unsafe-allow-remote", action="store_true")
    args = ap.parse_args()

    if args.host not in ("127.0.0.1", "localhost") and not args.unsafe_allow_remote:
        raise SystemExit(
            f"Refusing to bind {args.host}: these buttons run shell commands.\n"
            "Pass --unsafe-allow-remote if you really mean it, or use the read-only\n"
            "telemetry dashboard (tools/edge_train_dashboard.py, :8810) on the LAN."
        )

    threading.Thread(target=preflight_refresher, daemon=True).start()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[console] http://{args.host}:{args.port}  (preflight warming up…)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
