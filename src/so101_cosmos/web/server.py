"""Local control page for the policy server and the simulation client.

Standard library only -- no pip installs, no CDN, works offline.

SECURITY: the launch endpoints run shell commands. The server binds to
127.0.0.1 by default and refuses to enable launching on any other interface.
Do not expose this port.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import inventory, telemetry
from ..config import Settings
from ..logparse import newest_training_log, parse_training
from ..proc import JobLaunchError, JobStore, state

STATIC = Path(__file__).parent / "static"


def payload(cfg: Settings, store: JobStore, sampler: telemetry.Sampler,
            enable_launch: bool) -> dict:
    exports = inventory.export_index(cfg)
    history = inventory.eval_history(cfg)

    rows = []
    for row in inventory.checkpoints(cfg):
        n = row["iteration"]
        h = history.get(n, {})
        rows.append({**row, "episodes": h.get("episodes", 0),
                     "successes": h.get("successes", 0), "rate": h.get("rate")})

    log = newest_training_log(cfg.framework)
    points = parse_training(log)[-400:] if log else []

    return {
        "launch_enabled": enable_launch,
        "run_dir": str(cfg.run_path),
        "policy_port": cfg.policy_port,
        "servable": sorted(exports),
        "latest": cfg.latest_iteration(),
        "checkpoints": rows,
        "state": state(cfg, store),
        "jobs": [
            {k: j[k] for k in ("id", "stage", "side", "iteration", "pid", "status",
                               "started", "ended")}
            for j in store.all()[:14]
        ],
        "gpu": sampler.latest,
        "gpu_apps": telemetry.compute_apps(),
        "disk": inventory.disk(cfg),
        "training": {
            "log": str(log) if log else None,
            "iter": points[-1]["iter"] if points else None,
            "loss": points[-1]["loss"] if points else None,
            "max_iter": cfg.max_iter,
            "iters_per_epoch": cfg.iters_per_epoch,
        },
    }


def make_handler(cfg: Settings, store: JobStore, sampler: telemetry.Sampler,
                 enable_launch: bool):
    index = (STATIC / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "so101-cosmos"

        def log_message(self, fmt, *a):  # noqa: A003 - quiet by default
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: object, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, index, "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._json(payload(cfg, store, sampler, enable_launch))
            elif self.path.startswith("/api/logs/"):
                job_id = self.path.rsplit("/", 1)[-1]
                self._json({"lines": store.tail(job_id, 60)})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return
            try:
                if self.path == "/api/launch":
                    job = store.launch(body["stage"], body.get("iteration"))
                    self._json({"job": job})
                elif self.path == "/api/stop":
                    self._json({"job": store.stop(body["job"], hard=bool(body.get("kill")))})
                else:
                    self._json({"error": "not found"}, 404)
            except (JobLaunchError, FileNotFoundError, ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, 400)

    return Handler


def serve_forever(cfg: Settings, port: int, host: str = "127.0.0.1",
                  enable_launch: bool = True) -> int:
    if enable_launch and host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"refusing to enable launching on {host}: the buttons run shell commands.\n"
            "Bind to 127.0.0.1, or pass --no-launch for a read-only dashboard.",
            file=sys.stderr,
        )
        return 2

    store = JobStore(cfg, enabled=enable_launch)
    sampler = telemetry.Sampler()
    sampler.start()

    httpd = ThreadingHTTPServer((host, port), make_handler(cfg, store, sampler, enable_launch))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    mode = "launch enabled" if enable_launch else "read-only"
    print(f"so101 control panel  http://{host}:{port}  ({mode})")
    print(f"  run       {cfg.run_path}")
    print(f"  servable  {', '.join(str(i) for i in sorted(inventory.export_index(cfg))) or 'none'}")
    print("  ctrl-c to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopping (launched jobs keep running)")
    finally:
        sampler.stop()
        httpd.shutdown()
    return 0
