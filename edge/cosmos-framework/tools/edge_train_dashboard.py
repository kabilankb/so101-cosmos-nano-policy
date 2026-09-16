#!/usr/bin/env python3
"""Live dashboard for the Cosmos3-Edge SO-101 post-training run.

Serves a read-only web page showing, for
``action_policy_so101_edge_focus5_multi_1gpu``:

  * progress, epochs, and a projected finish time from the trailing iteration rate
  * loss over time, with an EMA so the per-batch noise does not hide the trend
  * seconds-per-iteration, to spot thermal throttling or a slow dataloader
  * live GPU telemetry (memory, temperature, power, clocks, utilisation)
  * checkpoints on disk, and the warm-start verification for the action heads
  * whether the trainer and its systemd supervisor are actually alive

Standard library only. No pip installs, no CDN, works offline.

Unlike tools/train_monitor.py this is strictly READ-ONLY -- there are no
click-to-launch buttons and it never executes pipeline commands, so it is safe
to bind beyond localhost on a trusted LAN.

    python3 tools/edge_train_dashboard.py                      # 127.0.0.1:8810
    python3 tools/edge_train_dashboard.py --host 0.0.0.0       # reachable on the LAN
    python3 tools/edge_train_dashboard.py --port 9000 --max-iter 7000

Log parsing understands both line formats the trainer emits:
    [..] [RANK 0] Iteration 526: Hit counter: 26/50 | Loss: 0.1704 | Time: 35.58s
    [..] 763 : iter_speed 35.80 seconds per iteration | Loss: 0.2592
and merges every outputs/edge_setup/train_edge*.log, so a systemd restart that
opens a new log file does not blank the history.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRAMEWORK = Path("/home/<user>/cosmos-framework")
RUN_DIR = FRAMEWORK / "outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
CKPT_DIR = RUN_DIR / "checkpoints"
LOG_GLOB = "outputs/edge_setup/train_edge*.log"

# Measured from the dataset loader: 128 train episodes -> 55,385 windows.
WINDOWS_PER_EPOCH = 55385
GLOBAL_BATCH = 32
ITERS_PER_EPOCH = WINDOWS_PER_EPOCH / GLOBAL_BATCH  # 1731.4

# [09-11 19:09:08|INFO|...] [RANK 0] Iteration 1: Hit counter: 1/50 | Loss: 11.8465 | Time: 46.73s
RE_ITER = re.compile(
    r"\[(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\|.*?Iteration (\d+):.*?Loss: ([\d.eE+-]+) \| Time: ([\d.]+)s"
)
# [09-11 19:40:00|INFO|...] 763 : iter_speed 35.80 seconds per iteration | Loss: 0.2592
RE_SPEED = re.compile(
    r"\[(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\|.*?\b(\d+) : iter_speed ([\d.]+) seconds per iteration \| Loss: ([\d.eE+-]+)"
)

_state_lock = threading.Lock()
_state: dict = {"points": [], "updated": None}


def _ts(mo: str, da: str, hh: str, mm: str, ss: str) -> float:
    """Log lines carry no year; assume the current one."""
    now = datetime.now()
    try:
        dt = datetime(now.year, int(mo), int(da), int(hh), int(mm), int(ss))
    except ValueError:
        return 0.0
    # Guard a run that spans New Year.
    if dt - now > timedelta(days=1):
        dt = dt.replace(year=now.year - 1)
    return dt.timestamp()


def parse_logs() -> list[dict]:
    """Merge every run log into one iteration-keyed series."""
    merged: dict[int, dict] = {}
    for path in sorted(FRAMEWORK.glob(LOG_GLOB)):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for m in RE_ITER.finditer(text):
            mo, da, hh, mm, ss, it, loss, secs = m.groups()
            merged[int(it)] = {
                "iter": int(it),
                "t": _ts(mo, da, hh, mm, ss),
                "loss": float(loss),
                "secs": float(secs),
            }
        for m in RE_SPEED.finditer(text):
            mo, da, hh, mm, ss, it, secs, loss = m.groups()
            merged[int(it)] = {
                "iter": int(it),
                "t": _ts(mo, da, hh, mm, ss),
                "loss": float(loss),
                "secs": float(secs),
            }
    return [merged[k] for k in sorted(merged)]


def gpu_stats() -> dict:
    q = ("memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
         "utilization.gpu,clocks.sm,clocks.max.sm")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        v = [x.strip() for x in out.split(",")]
        return {
            "mem_used": float(v[0]), "mem_total": float(v[1]),
            "temp": float(v[2]), "power": float(v[3]), "power_limit": float(v[4]),
            "util": float(v[5]), "clock": float(v[6]), "clock_max": float(v[7]),
        }
    except Exception:
        return {}


def _du(path: Path) -> int:
    try:
        return int(subprocess.run(["du", "-sb", str(path)], capture_output=True,
                                  text=True, timeout=60).stdout.split()[0])
    except Exception:
        return 0


def checkpoints() -> list[dict]:
    if not CKPT_DIR.is_dir():
        return []
    rows = []
    for d in sorted(CKPT_DIR.glob("iter_*")):
        if not d.is_dir() or d.name.endswith("_merged"):
            continue
        complete = all((d / s).is_dir() for s in ("model", "optim", "scheduler", "trainer"))
        rows.append({
            "name": d.name,
            "iter": int(re.sub(r"\D", "", d.name) or 0),
            "bytes": _du(d),
            "complete": complete,
            "mtime": d.stat().st_mtime,
        })
    return rows


def eval_results() -> list[dict]:
    """Closed-loop sim results written by tools/edge_eval_epoch.py."""
    root = FRAMEWORK / "outputs/edge_eval"
    if not root.is_dir():
        return []
    rows = []
    for d in sorted(root.glob("iter_*")):
        f = d / "result.json"
        if f.is_file():
            try:
                r = json.loads(f.read_text())
            except Exception:
                continue
            rows.append({k: r.get(k) for k in
                         ("iter", "epoch", "episodes", "successes",
                          "success_rate", "best_lift_in", "task", "finished")})
        elif (d / "error.txt").is_file():
            rows.append({"iter": int(re.sub(r"\D", "", d.name) or 0), "error":
                         (d / "error.txt").read_text()[:200]})
    return sorted(rows, key=lambda r: r.get("iter") or 0)


def _pgrep(pattern: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def systemd_state() -> str:
    try:
        return subprocess.run(
            ["systemctl", "--user", "is-active", "cosmos-so101-edge-train.service"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build(max_iter: int) -> dict:
    pts = parse_logs()
    running = _pgrep("cosmos_framework.scripts.train")

    cur = pts[-1]["iter"] if pts else 0
    # Trailing rate: median of the last 50 iteration times, which ignores the
    # one-off compile cost on iteration 1 and any single stalled step.
    tail = sorted(p["secs"] for p in pts[-50:]) if pts else []
    rate = tail[len(tail) // 2] if tail else 0.0
    remaining = max(0, max_iter - cur)
    eta_s = remaining * rate if rate else 0
    finish = (datetime.now() + timedelta(seconds=eta_s)).isoformat() if eta_s else None

    started = pts[0]["t"] if pts else None
    elapsed = (time.time() - started) if started else 0

    return {
        "updated": datetime.now().isoformat(),
        "run": RUN_DIR.name,
        "running": running,
        "systemd": systemd_state(),
        "cur_iter": cur,
        "max_iter": max_iter,
        "pct": (cur / max_iter * 100) if max_iter else 0,
        "epoch": cur / ITERS_PER_EPOCH,
        "total_epochs": max_iter / ITERS_PER_EPOCH,
        "iters_per_epoch": ITERS_PER_EPOCH,
        "rate": rate,
        "eta_seconds": eta_s,
        "finish": finish,
        "elapsed": elapsed,
        "started": datetime.fromtimestamp(started).isoformat() if started else None,
        "points": pts,
        "gpu": gpu_stats(),
        "checkpoints": checkpoints(),
        "evals": eval_results(),
    }


def refresher(max_iter: int, interval: float):
    while True:
        try:
            data = build(max_iter)
            with _state_lock:
                _state.clear()
                _state.update(data)
        except Exception as exc:  # never let the poller die
            print(f"[dashboard] refresh error: {exc}")
        time.sleep(interval)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmos3-Edge SO-101 Training</title>
<style>
:root{
  --bg:#0f1115; --panel:#171a21; --panel2:#1d212a; --line:#272c37;
  --fg:#e6e9ef; --dim:#939bab; --accent:#66d9a8; --accent2:#6aa9ff;
  --warn:#f0b866; --bad:#f0736a;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:19px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:12.5px;margin-bottom:20px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.grid{display:grid;gap:14px}
.g4{grid-template-columns:repeat(4,1fr)}
.g2{grid-template-columns:repeat(2,1fr)}
@media(max-width:860px){.g4{grid-template-columns:repeat(2,1fr)}.g2{grid-template-columns:1fr}}
@media(max-width:460px){.g4{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;
  color:var(--dim);margin:0 0 10px;font-weight:600}
.big{font-size:26px;font-weight:650;letter-spacing:-.02em}
.unit{font-size:13px;color:var(--dim);font-weight:400}
.bar{height:8px;background:var(--panel2);border-radius:99px;overflow:hidden;margin-top:12px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));
  border-radius:99px;transition:width .6s ease}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 9px;border-radius:99px;
  font-size:11.5px;font-weight:600;border:1px solid var(--line);background:var(--panel2)}
.dot{width:7px;height:7px;border-radius:99px;background:var(--dim)}
.dot.on{background:var(--accent);box-shadow:0 0 0 3px rgba(102,217,168,.16)}
.dot.off{background:var(--bad);box-shadow:0 0 0 3px rgba(240,115,106,.16)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.07em}
td.num,th.num{text-align:right;font-family:ui-monospace,monospace}
.kv{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line)}
.kv:last-child{border-bottom:0}
.kv span:first-child{color:var(--dim)}
.kv span:last-child{font-family:ui-monospace,monospace}
svg{display:block;width:100%;height:auto;overflow:visible}
.ok{color:var(--accent)} .warn{color:var(--warn)} .bad{color:var(--bad)}
.note{color:var(--dim);font-size:12px;margin-top:9px;line-height:1.5}
</style></head><body><div class="wrap">
<h1>Cosmos3-Edge &middot; SO-101 action-policy post-training</h1>
<div class="sub mono" id="runname">&nbsp;</div>

<div class="grid g4" style="margin-bottom:14px">
  <div class="card"><h2>Progress</h2>
    <div class="big" id="pct">–</div>
    <div class="unit mono" id="iters">&nbsp;</div>
    <div class="bar"><i id="barfill" style="width:0%"></i></div>
    <div class="unit mono" style="margin-top:8px" id="epochs">&nbsp;</div></div>
  <div class="card"><h2>Finishes</h2>
    <div class="big" id="finish">–</div>
    <div class="unit mono" id="etarel">&nbsp;</div>
    <div class="unit mono" style="margin-top:8px" id="elapsed">&nbsp;</div></div>
  <div class="card"><h2>Rate</h2>
    <div class="big" id="rate">–</div>
    <div class="unit">seconds / iteration</div>
    <div class="unit mono" style="margin-top:8px" id="loss">&nbsp;</div></div>
  <div class="card"><h2>Status</h2>
    <div style="display:flex;flex-direction:column;gap:7px;margin-top:2px">
      <span class="pill"><i class="dot" id="d1"></i><span id="s1">trainer</span></span>
      <span class="pill"><i class="dot" id="d2"></i><span id="s2">supervisor</span></span>
    </div>
    <div class="note" id="updated">&nbsp;</div></div>
</div>

<div class="grid g2" style="margin-bottom:14px">
  <div class="card"><h2>Loss <span style="text-transform:none;letter-spacing:0">(raw + EMA)</span></h2>
    <div id="losschart"></div>
    <div class="note">Edge weights vision flow-matching at <span class="mono">loss_scale=10.0</span>,
      so the scalar is dominated by the video term and is noisy per batch. Judge checkpoints on
      held-out episodes and eval success, never on this curve.</div></div>
  <div class="card"><h2>Seconds per iteration</h2>
    <div id="ratechart"></div>
    <div class="note">A sustained rise means thermal/power throttling or a starved dataloader.
      Iteration 1 includes one-off <span class="mono">torch.compile</span> cost and is excluded
      from the rate estimate.</div></div>
</div>

<div class="grid g2">
  <div class="card"><h2>GPU</h2><div id="gpu"></div></div>
  <div class="card"><h2>Checkpoints on disk</h2><div id="ckpt"></div></div>
</div>

<div class="card" style="margin-top:14px"><h2>Closed-loop eval &mdash; so101_bench Isaac Lab sim</h2>
  <div id="evals"></div>
  <div class="note">Written by <span class="mono">tools/edge_eval_epoch.py</span>, which merges
    LoRA &rarr; exports &rarr; starts the policy server &rarr; drives it from the Isaac Lab sim
    &rarr; tears the server down. This is the number that decides whether post-training worked;
    the loss curve above is not.
    <br><span class="mono">python3 tools/edge_eval_epoch.py --all --defer-until-training-done</span>
  </div></div>

<div class="card" style="margin-top:14px"><h2>Warm-start verification</h2>
  <div class="note" style="margin-top:0">
    <span class="mono">action2llm</span>/<span class="mono">llm2action</span> are
    <span class="mono">DomainAwareLinear</span> &mdash; per-embodiment
    <span class="mono">nn.Embedding</span> rows. DROID trained row&nbsp;8; SO-101 is row&nbsp;22,
    which ships untrained. <span class="mono">tools/transplant_domain.py</span> copies 8&nbsp;&rarr;&nbsp;22,
    verified by reading the saved checkpoint back:</div>
  <table style="margin-top:10px"><thead><tr>
    <th>tensor</th><th class="num">row 8 (DROID)</th><th class="num">row 22 (SO-101)</th><th class="num">row 0 (untouched)</th>
  </tr></thead><tbody>
    <tr><td class="mono">action2llm.fc.weight</td><td class="num">10.6676</td><td class="num ok">10.6676</td><td class="num">6.4187</td></tr>
    <tr><td class="mono">action2llm.bias.weight</td><td class="num">1.5603</td><td class="num ok">1.5603</td><td class="num">0.0000</td></tr>
    <tr><td class="mono">llm2action.fc.weight</td><td class="num">2.3256</td><td class="num ok">2.3255</td><td class="num">1.1386</td></tr>
    <tr><td class="mono">llm2action.bias.weight</td><td class="num">0.0868</td><td class="num ok">0.0868</td><td class="num">0.0000</td></tr>
  </tbody></table></div>

<script>
const $=id=>document.getElementById(id);
const fmtDur=s=>{if(!s||s<0)return"–";const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),
  m=Math.floor(s%3600/60);return (d?d+"d ":"")+h+"h "+String(m).padStart(2,"0")+"m";};
const fmtBytes=b=>b>=1e9?(b/1e9).toFixed(1)+" GB":(b/1e6).toFixed(0)+" MB";

function chart(el,pts,key,color,ema){
  if(!pts.length){el.innerHTML='<div class="note">waiting for data…</div>';return;}
  const W=560,H=170,P=34;
  const xs=pts.map(p=>p.iter), ys=pts.map(p=>p[key]);
  const x0=Math.min(...xs),x1=Math.max(...xs);
  let y0=Math.min(...ys),y1=Math.max(...ys);
  if(y1-y0<1e-9){y1=y0+1;}
  const pad=(y1-y0)*0.12; y0-=pad; y1+=pad;
  const X=v=>P+(v-x0)/((x1-x0)||1)*(W-P-8);
  const Y=v=>H-22-(v-y0)/((y1-y0)||1)*(H-22-8);
  const d=pts.map((p,i)=>(i?"L":"M")+X(p.iter).toFixed(1)+" "+Y(p[key]).toFixed(1)).join(" ");
  let emaPath="";
  if(ema){let e=null;const a=2/(Math.min(60,pts.length)+1);
    emaPath=pts.map((p,i)=>{e=e===null?p[key]:a*p[key]+(1-a)*e;
      return (i?"L":"M")+X(p.iter).toFixed(1)+" "+Y(e).toFixed(1);}).join(" ");}
  const ticks=[y0+(y1-y0)*0.05,(y0+y1)/2,y1-(y1-y0)*0.05];
  el.innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${ticks.map(t=>`<line x1="${P}" x2="${W-8}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}"
        stroke="#272c37" stroke-width="1"/>
      <text x="${P-6}" y="${(Y(t)+3.5).toFixed(1)}" fill="#939bab" font-size="9.5"
        text-anchor="end" font-family="ui-monospace,monospace">${t.toFixed(t>100?0:2)}</text>`).join("")}
    <path d="${d}" fill="none" stroke="${color}" stroke-width="1.2" opacity="${ema?0.42:0.95}"
      stroke-linejoin="round"/>
    ${emaPath?`<path d="${emaPath}" fill="none" stroke="${color}" stroke-width="2.1"
      stroke-linejoin="round"/>`:""}
    <text x="${P}" y="${H-5}" fill="#939bab" font-size="9.5"
      font-family="ui-monospace,monospace">${x0}</text>
    <text x="${W-8}" y="${H-5}" fill="#939bab" font-size="9.5" text-anchor="end"
      font-family="ui-monospace,monospace">${x1}</text>
  </svg>`;
}

function kv(rows){return rows.map(([k,v,c])=>
  `<div class="kv"><span>${k}</span><span class="${c||""}">${v}</span></div>`).join("");}

async function tick(){
  let d; try{ d=await (await fetch("/api/data",{cache:"no-store"})).json(); }catch(e){ return; }
  $("runname").textContent=d.run;
  $("pct").innerHTML=d.pct.toFixed(1)+'<span class="unit"> %</span>';
  $("iters").textContent=d.cur_iter.toLocaleString()+" / "+d.max_iter.toLocaleString()+" iters";
  $("barfill").style.width=Math.min(100,d.pct)+"%";
  $("epochs").textContent=d.epoch.toFixed(2)+" / "+d.total_epochs.toFixed(2)+" epochs  ("
    +Math.round(d.iters_per_epoch).toLocaleString()+" it/epoch)";

  if(d.finish){const f=new Date(d.finish);
    $("finish").innerHTML=f.toLocaleDateString([], {month:"short",day:"numeric"})
      +' <span class="unit">'+f.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"})+'</span>';
    $("etarel").textContent=fmtDur(d.eta_seconds)+" remaining";}
  else {$("finish").textContent="–";$("etarel").textContent="";}
  $("elapsed").textContent=fmtDur(d.elapsed)+" elapsed";

  $("rate").innerHTML=d.rate?d.rate.toFixed(2):"–";
  const last=d.points.length?d.points[d.points.length-1]:null;
  $("loss").textContent=last?("loss "+last.loss.toFixed(4)+" @ iter "+last.iter):"";

  $("d1").className="dot "+(d.running?"on":"off");
  $("s1").textContent=d.running?"trainer running":"trainer NOT running";
  const sOK=d.systemd==="active";
  $("d2").className="dot "+(sOK?"on":"off");
  $("s2").textContent="supervisor "+d.systemd;
  $("updated").textContent="updated "+new Date(d.updated).toLocaleTimeString();

  const plot=d.points.filter(p=>p.iter>1);
  chart($("losschart"),plot,"loss","#6aa9ff",true);
  chart($("ratechart"),plot,"secs","#66d9a8",false);

  const g=d.gpu||{};
  $("gpu").innerHTML=g.mem_total?kv([
    ["memory",(g.mem_used/1024).toFixed(1)+" / "+(g.mem_total/1024).toFixed(0)+" GiB"],
    ["utilisation",g.util.toFixed(0)+" %"],
    ["temperature",g.temp.toFixed(0)+" °C",g.temp>=85?"bad":g.temp>=80?"warn":"ok"],
    ["power",g.power.toFixed(0)+" / "+g.power_limit.toFixed(0)+" W",
      g.power>=g.power_limit*0.98?"warn":""],
    ["SM clock",g.clock.toFixed(0)+" / "+g.clock_max.toFixed(0)+" MHz"],
  ]):'<div class="note">nvidia-smi unavailable</div>';

  const ev=d.evals||[];
  $("evals").innerHTML=ev.length?
    '<table><thead><tr><th>checkpoint</th><th class="num">epoch</th><th class="num">episodes</th>'
    +'<th class="num">success</th><th class="num">best lift</th><th>task</th></tr></thead><tbody>'
    +ev.slice().reverse().map(r=>{
      if(r.error) return `<tr><td class="mono">iter ${r.iter}</td>
        <td colspan="5" class="bad">failed: ${r.error.split("\n")[0]}</td></tr>`;
      const pc=r.success_rate==null?null:r.success_rate*100;
      const cls=pc==null?"":pc>=30?"ok":pc>0?"warn":"bad";
      return `<tr><td class="mono">iter ${r.iter}</td>
        <td class="num">${(r.epoch||0).toFixed(2)}</td>
        <td class="num">${r.successes}/${r.episodes}</td>
        <td class="num ${cls}">${pc==null?"n/a":pc.toFixed(1)+" %"}</td>
        <td class="num">${r.best_lift_in==null?"–":r.best_lift_in.toFixed(2)+" in"}</td>
        <td class="mono" style="font-size:11.5px">${r.task||""}</td></tr>`;}).join("")
    +"</tbody></table>"
    :'<div class="note" style="margin-top:0">no sim evaluation yet. The first checkpoint lands at '
    +'iteration 500; evaluate it with <span class="mono">tools/edge_eval_epoch.py --iter 500</span>.</div>';

  const c=d.checkpoints||[];
  $("ckpt").innerHTML=c.length?
    '<table><thead><tr><th>checkpoint</th><th class="num">size</th><th class="num">state</th></tr></thead><tbody>'
    +c.slice().reverse().map(r=>`<tr><td class="mono">${r.name}</td>
      <td class="num">${fmtBytes(r.bytes)}</td>
      <td class="num ${r.complete?"ok":"bad"}">${r.complete?"complete":"partial"}</td></tr>`).join("")
    +`</tbody></table><div class="note">${c.length} saved &middot; ${fmtBytes(c.reduce((a,b)=>a+b.bytes,0))} total &middot; saves every 500 iters</div>`
    :'<div class="note">no checkpoint saved yet — first save lands at iteration 500</div>';
}
tick(); setInterval(tick,5000);
</script></div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def do_GET(self):
        if self.path.startswith("/api/data"):
            with _state_lock:
                body = json.dumps(_state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--max-iter", type=int, default=7000)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    with _state_lock:
        _state.update(build(args.max_iter))
    threading.Thread(target=refresher, args=(args.max_iter, args.interval), daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard] http://{args.host}:{args.port}  (read-only)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
