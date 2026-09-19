"""Web UI for running the SO-101 Isaac Lab benchmark on this machine against a policy
server on a Jetson Thor (or any host running the Cosmos3-Edge policy server).

    python so101_thor_ui.py [--port 8765]        # then open http://127.0.0.1:8765/

The page takes the Thor address, checks the server (metadata + one timed inference),
then starts ``so101_bench/scripts/cosmos3_eval.py`` here. The Isaac Sim window opens on
this machine; the page shows every observation sent to Thor and every action chunk it
returns, per-episode results, and the client log.

Traffic path:  Isaac Lab client -> ws://127.0.0.1:<relay> (this process) -> ws://<thor>:<port>
Frames are relayed byte-for-byte; a copy of each request/response pair is decoded off the
relay path for the page, so the client and server see exactly what they would without it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent.parent / "vendor_py"  # openpi-client, pyzmq (see setup_laptop.sh)
if _VENDOR.is_dir():
    sys.path.insert(0, str(_VENDOR))

import numpy as np
import websockets
from openpi_client import msgpack_numpy
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # the client folder: so101_bench/, webui/, runs/
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
EPISODE_GAP_S = 15.0  # a pause this long between requests means the bench reset the scene
_EPISODE = re.compile(r"Episode (\d+)/(\d+): (?:success=(\w+), reason=(\w+), length=([\d.]+)s|skipped)")

#: Benchmark suites: task file and layout-file glob, both relative to so101_bench/.
SUITES = {
    "focus5": ("tasks/focus5.jsonl", "tasks/layouts/focus5_layouts_*.jsonl",
               "5 training objects, 20 fixed layouts each (100 episodes)"),
    "unseen": ("tasks/focus5_unseen.jsonl", "tasks/layouts/focus5_unseen_layouts_*.jsonl",
               "26 objects never seen in training, 2 layouts each (52 episodes)"),
    "wins": ("tasks/focus5_wins.jsonl", "tasks/layouts/focus5_layouts_*.jsonl",
             "12 layouts where a checkpoint has succeeded before (quick demo)"),
    "clutter": ("tasks/focus5_clutter.jsonl", "tasks/layouts/focus5_clutter_layouts_*.jsonl",
                "focus5 targets with 3 distractors on the table (100 episodes)"),
    "mix": ("tasks/edge_train_mix.jsonl", "tasks/layouts/edge_train_mix_layouts_*.jsonl",
            "training-distribution mix (75 episodes)"),
}


#: Objects the Edge 6500 policy was trained on (single-object bin instructions).
TRAINED_OBJECTS = ["green shoes", "cardboard box", "altoids container", "flower pot", "cooking spoon"]


def scene_objects(bench: Path) -> list[str]:
    """Every object with a USD asset in so101_bench, e.g. "cooking spoon"."""
    root = bench / "source/so101_bench/so101_bench/assets/usd/objects"
    names = {f.name.split(".")[0].replace("_", " ") for f in root.glob("*.usd*") if not f.name.endswith(".bak")}
    return sorted(names)


# ------------------------------------------------------------------ recording
class Recorder:
    """Decoded request/response pairs: the newest in memory, all of them on disk."""

    def __init__(self, keep: int = 2000) -> None:
        self.keep = keep
        self.lock = threading.Lock()
        self.reset(None)

    def reset(self, out: Path | None) -> None:
        with getattr(self, "lock", threading.Lock()):
            self.out = out
            self.records: deque[dict] = deque(maxlen=self.keep)
            self.count = 0
            self.episode = 0
            self.last_t: float | None = None
            self.last_prompt: str | None = None
            self.jsonl = None
            if out is not None:
                (out / "obs").mkdir(parents=True, exist_ok=True)
                self.jsonl = (out / "records.jsonl").open("a")

    def add(self, request: bytes, response: bytes, t_sent: float, t_recv: float) -> None:
        obs = msgpack_numpy.unpackb(request)
        reply = msgpack_numpy.unpackb(response)
        if not isinstance(obs, dict) or not isinstance(reply, dict) or "action" not in reply:
            return
        prompt = str(obs.get("prompt", ""))
        gap = None if self.last_t is None else t_sent - self.last_t
        if self.last_t is None or gap > EPISODE_GAP_S or prompt != self.last_prompt:
            self.episode += 1
        self.last_t, self.last_prompt = t_sent, prompt

        index = self.count
        self.count += 1
        image = np.asarray(obs.get("observation/image"))
        if image.ndim == 3 and self.out is not None:
            Image.fromarray(image.astype(np.uint8)).save(self.out / "obs" / f"{index:06d}.jpg", quality=85)
        joints = np.asarray(obs.get("observation/joint_position", np.zeros((1, 5)))).reshape(-1, 5)[-1]
        gripper = np.asarray(obs.get("observation/gripper_position", np.zeros((1, 1)))).reshape(-1)[-1]
        action = np.asarray(reply["action"], dtype=np.float64)
        timing = reply.get("server_timing", {}) or {}
        rec = {
            "i": index,
            "episode": self.episode,
            "wall": time.strftime("%H:%M:%S", time.localtime(t_sent)),
            "t": t_sent,
            "prompt": prompt.split(". This video")[0],
            "state": [round(float(v), 3) for v in [*joints, gripper]],
            "action": [[round(float(v), 3) for v in row] for row in action],
            "infer_ms": round(float(timing.get("infer_ms", 0.0)), 1),
            "roundtrip_ms": round((t_recv - t_sent) * 1000, 1),
            "since_prev_s": None if gap is None else round(gap, 2),
        }
        with self.lock:
            self.records.append(rec)
            if self.jsonl is not None:
                self.jsonl.write(json.dumps(rec) + "\n")
                self.jsonl.flush()

    def since(self, after: int, limit: int) -> list[dict]:
        with self.lock:
            return [r for r in self.records if r["i"] > after][:limit]


# ---------------------------------------------------------------------- relay
class Relay:
    """WebSocket relay to the policy server. The upstream is set when a run starts."""

    def __init__(self, recorder: Recorder, host: str, port: int) -> None:
        self.recorder = recorder
        self.host, self.port = host, port
        self.upstream: tuple[str, int] | None = None
        self.decode_q: asyncio.Queue | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    async def health(self, path, request_headers):
        # websockets<=12 (the version openpi-client runs with under Isaac Sim) legacy API
        if path == "/healthz":
            return HTTPStatus.OK, [], b"OK\n"
        return None

    async def handler(self, client, path: str | None = None) -> None:
        if self.upstream is None:
            await client.close(1011, "no policy server selected")
            return
        url = f"ws://{self.upstream[0]}:{self.upstream[1]}"
        async with websockets.connect(url, compression=None, max_size=None, open_timeout=30,
                                      ping_interval=None) as upstream:
            await client.send(await upstream.recv())  # server metadata
            try:
                while True:
                    request = await client.recv()
                    t_sent = time.time()
                    await upstream.send(request)
                    response = await upstream.recv()
                    t_recv = time.time()
                    await client.send(response)
                    if isinstance(request, bytes) and isinstance(response, bytes):
                        try:
                            self.decode_q.put_nowait((request, response, t_sent, t_recv))
                        except asyncio.QueueFull:
                            pass  # never slow the relay down for the page
            except websockets.ConnectionClosed:
                pass

    async def decoder(self) -> None:
        while True:
            item = await self.decode_q.get()
            try:
                await asyncio.to_thread(self.recorder.add, *item)
            except Exception as exc:  # a bad record must not stop the relay
                print(f"[ui] decode failed: {exc!r}", flush=True)

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.decode_q = asyncio.Queue(maxsize=64)
        asyncio.create_task(self.decoder())
        async with websockets.serve(self.handler, self.host, self.port, compression=None, max_size=None,
                                    ping_interval=None, process_request=self.health):
            await asyncio.Future()


def check_server(host: str, port: int, prompt: str) -> dict:
    """Connect to the policy server, read its metadata and time one synthetic inference."""
    from openpi_client import websocket_client_policy

    out: dict = {"host": host, "port": port}
    t0 = time.time()
    try:
        import socket
        with socket.create_connection((host, port), timeout=5):
            pass
    except OSError as exc:
        return {**out, "ok": False, "error": f"cannot reach {host}:{port} ({exc.strerror or exc})"}
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host, port)
        out["connect_ms"] = round((time.time() - t0) * 1000)
        meta = client.get_server_metadata() or {}
        out["metadata"] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}
        rng = np.random.default_rng(0)
        req = {
            "observation/image": rng.integers(0, 255, size=(960, 640, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((1, 5), dtype=np.float32),
            "observation/gripper_position": np.full((1, 1), 20.0, dtype=np.float32),
            "prompt": prompt,
        }
        times = []
        for _ in range(2):  # the first call after a server start compiles kernels
            t = time.time()
            reply = client.infer(req)
            times.append(round((time.time() - t) * 1000))
        a = np.asarray(reply["action"])
        timing = reply.get("server_timing", {}) or {}
        out.update(ok=True, action_shape=list(a.shape), action_min=round(float(a.min()), 2),
                   action_max=round(float(a.max()), 2), roundtrip_ms=times,
                   infer_ms=round(float(timing.get("infer_ms", 0.0))))
        # A healthy SO-101 chunk is in LeRobot .pos units (joints roughly +-100). Values
        # stuck inside [-1, 1] mean the server is missing its normalization flags.
        out["units_ok"] = bool(np.abs(a).max() > 1.5)
    except Exception as exc:
        out.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return out


#: Published checkpoints in the HF repo, with every result measured so far. "path" is the folder in
#: the repo and under <thor dir>/checkpoints/. All share the same SO-101 serving contract.
HF_REPO = os.environ.get("SO101_HF_REPO", "kabilanKB/cosmos_edge_policy_so101")
CHECKPOINTS = [
    {"path": "iter_6500", "run": "Edge focus5 multi", "iter": 6500, "best": True,
     "results": "focus5 5/98 (5.1%, 25 s) - best full run; unseen objects 2/43 (60 s, partial); "
                "Thor HIL wins 2/12 (60 s)"},
    {"path": "iter_7000", "run": "Edge focus5 multi", "iter": 7000, "results": "focus5 1/100 (1.0%, 25 s)"},
    {"path": "iter_6000", "run": "Edge focus5 multi", "iter": 6000, "results": "not evaluated"},
    {"path": "single_bin_from6500/iter_1750", "run": "single-bin from 6500", "iter": 1750,
     "results": "focus5 3/100 (3.0%, 60 s)"},
    {"path": "single_bin_from6500/iter_1500", "run": "single-bin from 6500", "iter": 1500, "results": "not evaluated"},
    {"path": "single_bin_from6500/iter_1000", "run": "single-bin from 6500", "iter": 1000,
     "results": "focus5 1/26 (60 s, partial)"},
    {"path": "single_bin_from6500/iter_500", "run": "single-bin from 6500", "iter": 500, "results": "not evaluated"},
]
CHECKPOINT_PATHS = {c["path"] for c in CHECKPOINTS}
CHECKPOINT_GB = 7.8  # download size of one export
DEFAULT_CHECKPOINT = "iter_6500"


# ------------------------------------------------------------- the Thor side
_SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]{1,253}$")
_SAFE_USER = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SAFE_DIR = re.compile(r"^[A-Za-z0-9._/~-]{1,256}$")


class ThorControl:
    """Start, watch and stop the policy server on the Thor over SSH (key-based, never prompts).

    Uses the scripts installed by ``setup_thor.sh``: ``<dir>/serve_thor.sh --tmux`` starts the server
    in the tmux session ``so101serve`` and links its log to ``<dir>/logs/serve_latest.log``.
    """

    def __init__(self, default_user: str, default_dir: str) -> None:
        self.default_user = default_user
        self.default_dir = default_dir

    def _target(self, cfg: dict) -> tuple[str, str, str, int]:
        host = str(cfg.get("host", "")).strip()
        user = str(cfg.get("user") or self.default_user).strip()
        tdir = str(cfg.get("dir") or self.default_dir).strip()
        port = int(cfg.get("port", 8000))
        if not _SAFE_HOST.match(host):
            raise ValueError("enter the Thor's IP address")
        if not _SAFE_USER.match(user):
            raise ValueError("enter the SSH user on the Thor")
        if not _SAFE_DIR.match(tdir):
            raise ValueError("invalid install folder")
        return host, user, tdir, port

    @staticmethod
    def _ssh(user: str, host: str, command: str, timeout: float = 20) -> tuple[int, str]:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
                f"{user}@{host}", command]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "ssh timed out"
        return r.returncode, (r.stdout + r.stderr).strip()

    @staticmethod
    def _healthy(host: str, port: int) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _ssh_error(self, user: str, host: str, out: str) -> str:
        if "Permission denied" in out or "publickey" in out:
            return (f"SSH key login to {user}@{host} failed. Add this PC's key to the Thor once "
                    f"(see the README, 'SSH key'), then try again.")
        if "Could not resolve" in out or "No route" in out or "timed out" in out or "refused" in out:
            return f"cannot reach {host} over SSH ({out.splitlines()[-1] if out else 'no answer'})"
        return out.splitlines()[-1] if out else "ssh failed"

    @staticmethod
    def _checkpoint(cfg: dict) -> str:
        ckpt = str(cfg.get("checkpoint") or DEFAULT_CHECKPOINT)
        if ckpt not in CHECKPOINT_PATHS:
            raise ValueError(f"unknown checkpoint {ckpt!r}")
        return ckpt

    def _loaded_checkpoint(self, user: str, host: str, tdir: str) -> str | None:
        """The checkpoint the running server loaded, from its log (None if unknown)."""
        code, out = self._ssh(user, host, f"grep -o \"checkpoint_path='[^']*'\" {tdir}/logs/serve_latest.log 2>/dev/null | tail -1")
        m = re.search(r"/checkpoints/(.+?)/?'", out or "")
        return m.group(1) if m else None

    def launch(self, cfg: dict) -> dict:
        host, user, tdir, port = self._target(cfg)
        ckpt = self._checkpoint(cfg)
        if self._healthy(host, port):
            loaded = self._loaded_checkpoint(user, host, tdir)
            if loaded == ckpt or (loaded is None and not cfg.get("restart_if_unknown")):
                return {"ok": True, "already_running": True, "checkpoint": loaded}
            # a different checkpoint is being served: stop it, then start the selected one
            self._ssh(user, host, f"SO101_PORT={port} bash {tdir}/serve_thor.sh --stop")
            for _ in range(20):
                if not self._healthy(host, port):
                    break
                time.sleep(1)
        code, out = self._ssh(user, host, f"test -f {tdir}/serve_thor.sh && echo HAS_SERVE")
        if code != 0:
            return {"ok": False, "error": self._ssh_error(user, host, out)}
        if "HAS_SERVE" not in out:
            return {"ok": False, "error": f"no {tdir}/serve_thor.sh on the Thor; run setup_thor.sh there first"}
        code, out = self._ssh(user, host, f"test -f {tdir}/checkpoints/{ckpt}/checkpoint.json && echo HAS_CKPT")
        if "HAS_CKPT" not in out:
            return {"ok": False, "error": f"{ckpt} is not on the Thor yet; download it first (Checkpoints panel)"}
        code, out = self._ssh(user, host, f"SO101_PORT={port} SO101_CHECKPOINT={tdir}/checkpoints/{ckpt} "
                                          f"bash {tdir}/serve_thor.sh --tmux", timeout=30)
        if code != 0:
            return {"ok": False, "error": out.splitlines()[-1] if out else "launch failed"}
        return {"ok": True, "checkpoint": ckpt, "message": out.splitlines()[-1] if out else "starting"}

    def checkpoints(self, cfg: dict) -> dict:
        """Which catalog checkpoints are on the Thor, free disk, and any download in progress."""
        host, user, tdir, port = self._target(cfg)
        code, out = self._ssh(
            user, host,
            f"cd {tdir}/checkpoints 2>/dev/null && find . -name checkpoint.json -printf '%h\\n' | sed 's|^\\./||' | "
            f"while read d; do echo \"CKPT $d $(du -sm \"$d\" | cut -f1)\"; done; "
            f"echo FREE $(df -BG --output=avail {tdir} | tail -1 | tr -dc 0-9); "
            f"tmux has-session -t so101dl 2>/dev/null && echo DL_RUNNING $(cat {tdir}/logs/download_target 2>/dev/null) "
            f"$(du -sm {tdir}/checkpoints/.cache 2>/dev/null | cut -f1); "
            f"tail -c 300 {tdir}/logs/download_latest.log 2>/dev/null | tr '\\r' '\\n' | tail -2 | sed 's/^/DL_LOG /'",
        )
        if code == 255:
            return {"ok": False, "error": self._ssh_error(user, host, out)}
        on_thor, free, dl, dl_log = {}, None, None, []
        for line in out.splitlines():
            parts = line.split()
            if parts[:1] == ["CKPT"] and len(parts) >= 3:
                on_thor[parts[1]] = int(parts[2])
            elif parts[:1] == ["FREE"] and len(parts) == 2:
                free = int(parts[1])
            elif parts[:1] == ["DL_RUNNING"]:
                dl = {"path": parts[1] if len(parts) > 1 else "?", "mb": int(parts[2]) if len(parts) > 2 else 0}
            elif parts[:1] == ["DL_LOG"]:
                dl_log.append(line[7:])
        loaded = self._loaded_checkpoint(user, host, tdir) if self._healthy(host, port) else None
        downloading = dl["path"] if dl else None  # checkpoint.json can land before the weights
        items = [{**c, "on_thor": c["path"] in on_thor and c["path"] != downloading,
                  "downloading": c["path"] == downloading, "size_mb": on_thor.get(c["path"]),
                  "loaded": c["path"] == loaded} for c in CHECKPOINTS]
        return {"ok": True, "checkpoints": items, "free_gb": free, "download": dl, "download_log": dl_log,
                "loaded": loaded, "need_gb": CHECKPOINT_GB}

    def download(self, cfg: dict) -> dict:
        host, user, tdir, port = self._target(cfg)
        ckpt = self._checkpoint(cfg)
        info = self.checkpoints(cfg)
        if not info.get("ok"):
            return info
        if info["download"]:
            return {"ok": False, "error": f"already downloading {info['download']['path']}"}
        if any(c["path"] == ckpt and c["on_thor"] for c in info["checkpoints"]):
            return {"ok": True, "message": f"{ckpt} is already on the Thor"}
        if info["free_gb"] is not None and info["free_gb"] < CHECKPOINT_GB + 1:
            return {"ok": False, "error": f"only {info['free_gb']} GB free on the Thor; a checkpoint needs "
                                          f"~{CHECKPOINT_GB:.0f} GB. Delete one first."}
        cmd = (f"mkdir -p {tdir}/logs {tdir}/checkpoints && echo {ckpt} > {tdir}/logs/download_target && "
               f"tmux new-session -d -s so101dl \"HF_HUB_DISABLE_IMPLICIT_TOKEN=1 $HOME/.local/bin/uvx hf@latest "
               f"download {HF_REPO} --include '{ckpt}/*' --local-dir {tdir}/checkpoints "
               f"> {tdir}/logs/download_latest.log 2>&1\"")
        code, out = self._ssh(user, host, cmd)
        if code != 0:
            return {"ok": False, "error": out.splitlines()[-1] if out else "download failed to start"}
        return {"ok": True, "message": f"downloading {ckpt} on the Thor"}

    def delete(self, cfg: dict) -> dict:
        host, user, tdir, port = self._target(cfg)
        ckpt = self._checkpoint(cfg)
        if self._healthy(host, port) and self._loaded_checkpoint(user, host, tdir) == ckpt:
            return {"ok": False, "error": f"{ckpt} is being served; stop the server first"}
        code, out = self._ssh(user, host, f"rm -rf {tdir}/checkpoints/{ckpt} && echo DELETED")
        if "DELETED" not in out:
            return {"ok": False, "error": out.splitlines()[-1] if out else "delete failed"}
        return {"ok": True, "message": f"deleted {ckpt} from the Thor"}

    def stop(self, cfg: dict) -> dict:
        host, user, tdir, port = self._target(cfg)
        code, out = self._ssh(user, host, f"SO101_PORT={port} bash {tdir}/serve_thor.sh --stop")
        if code != 0 and not out:
            return {"ok": False, "error": self._ssh_error(user, host, out)}
        return {"ok": True, "message": out or "stopped"}

    def status(self, cfg: dict) -> dict:
        host, user, tdir, port = self._target(cfg)
        healthy = self._healthy(host, port)
        code, out = self._ssh(
            user, host,
            f"tmux has-session -t so101serve 2>/dev/null && echo __SESSION__; "
            f"grep -q 'ready domain=' {tdir}/logs/serve_latest.log 2>/dev/null && echo __READY__; "
            f"grep -o \"checkpoint_path='[^']*'\" {tdir}/logs/serve_latest.log 2>/dev/null | tail -1 | sed 's/^/__CKPT__ /'; "
            f"tail -c 20000 {tdir}/logs/serve_latest.log 2>/dev/null | tr '\\r' '\\n' | "
            f"grep -vE 'OmniMoTModel: config|^\\s*$' | tail -n 40",
        )
        if code == 255:
            return {"ok": False, "healthy": healthy, "state": "unreachable", "error": self._ssh_error(user, host, out)}
        session = "__SESSION__" in out
        ready = "__READY__" in out
        m = re.search(r"__CKPT__ checkpoint_path='[^']*/checkpoints/(.+?)/?'", out)
        loaded = m.group(1) if m else None
        log_text = "\n".join(line for line in out.replace("__SESSION__", "").replace("__READY__", "").splitlines()
                              if not line.startswith("__CKPT__")).strip()
        if healthy:
            state = "ready"  # answering health checks (also when started by hand)
        elif session:
            # A busy server cannot answer /healthz during an inference, so trust the log once it
            # has said "ready".
            state = "ready" if ready else "starting"
        elif "Traceback" in log_text and not ready:
            state = "error"  # died while loading; the log shows why
        else:
            state = "stopped"
        return {"ok": True, "healthy": healthy, "session": session, "state": state, "log": log_text,
                "checkpoint": loaded if state in ("ready", "starting") else None}


# ------------------------------------------------------------------- the run
class Runner:
    """One Isaac Lab client process at a time."""

    def __init__(self, args: argparse.Namespace, relay: Relay, recorder: Recorder) -> None:
        self.args = args
        self.relay = relay
        self.recorder = recorder
        self.proc: subprocess.Popen | None = None
        self.run_dir: Path | None = None
        self.log_path: Path | None = None
        self.info: dict = {}
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def custom_tasks(self, cfg: dict, run_dir: Path) -> Path:
        """Write a task file for "pick this object with this prompt": one single-object bin episode
        per row. With no layout file, the benchmark places the object (and the bin) at a random
        valid pose per episode, reproducible for a given seed."""
        obj = str(cfg.get("object", "")).strip()
        if obj not in scene_objects(self.args.bench):
            raise ValueError(f"unknown object {obj!r}")
        prompt = " ".join(str(cfg.get("prompt") or "").split())
        if len(prompt) > 300:
            raise ValueError("the prompt is too long (300 characters max)")
        n = int(cfg.get("num_episodes") or 5)
        if not 1 <= n <= 100:
            raise ValueError("episodes must be between 1 and 100")
        path = run_dir / f"custom_{obj.replace(' ', '_')}.jsonl"
        group = "trained" if obj in TRAINED_OBJECTS else "not trained"
        # The benchmark validates each row's instruction against its task templates and scores the
        # episode as "object ends up in the bin", so the row keeps the standard wording; a different
        # prompt is sent to the policy with --lang_instruction (see command()).
        canonical = f"Place the {obj} in the plastic bin"
        self.info["policy_prompt"] = prompt if prompt and prompt != canonical else None
        with path.open("w") as f:
            for trial in range(n):
                f.write(json.dumps({"objects": [obj], "ood_key": group, "trial_id": trial, "n_objects": 1,
                                    "task_family": "bin", "instruction": canonical, "target": obj}) + "\n")
        return path

    def command(self, cfg: dict, run_dir: Path) -> tuple[list[str], dict]:
        bench = self.args.bench
        custom = cfg.get("mode") == "custom"
        if custom:
            tasks, layouts_glob = str(self.custom_tasks(cfg, run_dir)), None
        else:
            suite = cfg.get("suite", "focus5")
            if suite not in SUITES:
                raise ValueError(f"unknown suite {suite!r}")
            tasks, layouts_glob, _ = SUITES[suite]
        self.info["tasks_path"] = str(bench / tasks)
        argv = [
            str(self.args.python), "-u", "scripts/cosmos3_eval.py",
            "--task", "So101Bench-Bin-v0",
            "--episodes_jsonl", tasks,
            "--seed", str(int(cfg.get("seed") or 1984)),
            "--policy_host", "127.0.0.1",
            "--policy_port", str(self.args.relay_port),
            "--action_horizon", str(int(cfg.get("action_horizon", 32))),
            "--camera_snapshot_stdin", "false",
        ]
        layouts = sorted(bench.glob(layouts_glob), key=lambda p: p.stat().st_mtime) if layouts_glob else []
        if layouts:
            argv += ["--episode_layouts_jsonl", str(layouts[-1].relative_to(bench))]
        if cfg.get("num_episodes") and not custom:
            argv += ["--num_episodes", str(int(cfg["num_episodes"]))]
        if custom and self.info.get("policy_prompt"):
            argv += ["--lang_instruction", self.info["policy_prompt"]]
        # Isaac Lab 3 opens the Isaac Sim window only when the Kit visualizer is requested; without
        # --viz it runs headless even when --headless is not given.
        argv += ["--viz", "kit"] if cfg.get("gui", True) else ["--headless"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(bench / "source/so101_bench"), *map(str, self.args.pythonpath)])
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        if cfg.get("gui", True) and not env.get("DISPLAY"):
            # Started outside a desktop session (e.g. over SSH): use the local X display.
            sockets = sorted(Path("/tmp/.X11-unix").glob("X*"))
            env["DISPLAY"] = ":" + sockets[0].name[1:] if sockets else ":0"
        self.info["display"] = env.get("DISPLAY") if cfg.get("gui", True) else None
        return argv, env

    def start(self, cfg: dict) -> dict:
        with self.lock:
            if self.running():
                return {"ok": False, "error": "a run is already in progress"}
            host = str(cfg.get("host", "")).strip()
            port = int(cfg.get("port", 8000))
            if not host:
                return {"ok": False, "error": "enter the Thor address first"}
            stamp = time.strftime("%Y%m%d_%H%M%S")
            custom = cfg.get("mode") == "custom"
            label = ("custom_" + str(cfg.get("object", ""))) if custom else str(cfg.get("suite", "focus5"))
            label = re.sub(r"[^A-Za-z0-9_-]+", "_", label)[:60]
            run_dir = self.args.runs / f"{label}_{stamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            previous_info, self.info = self.info, {}
            try:
                argv, env = self.command(cfg, run_dir)
            except ValueError:
                shutil.rmtree(run_dir, ignore_errors=True)
                self.info = previous_info
                raise
            self.run_dir = run_dir
            self.log_path = self.run_dir / "eval.log"
            self.recorder.reset(self.run_dir)
            self.relay.upstream = (host, port)
            self.info = {**self.info, "host": host, "port": port,
                         "suite": label if custom else cfg.get("suite", "focus5"),
                         "checkpoint": str(cfg.get("checkpoint") or "") or None,
                         "prompt": (" ".join(str(cfg.get("prompt") or "").split()) or None) if custom else None,
                         "gui": bool(cfg.get("gui", True)), "started": time.strftime("%H:%M:%S"),
                         "started_t": time.time(), "run_dir": str(self.run_dir), "argv": argv}
            (self.run_dir / "run.json").write_text(json.dumps(self.info, indent=2))
            log = self.log_path.open("w")
            self.proc = subprocess.Popen(argv, cwd=self.args.bench, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                                         start_new_session=True, text=True)
            self.info["pid"] = self.proc.pid
            return {"ok": True, **self.info}

    def send(self, command: str) -> dict:
        """pause / resume / skip, through the eval script's terminal-control stdin."""
        if not self.running() or self.proc.stdin is None:
            return {"ok": False, "error": "no run in progress"}
        try:
            self.proc.stdin.write(command + "\n")
            self.proc.stdin.flush()
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def stop(self) -> dict:
        if not self.running():
            return {"ok": False, "error": "no run in progress"}
        try:
            os.killpg(self.proc.pid, signal.SIGINT)
            for _ in range(40):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.25)
            if self.proc.poll() is None:
                os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return {"ok": True}

    def status(self) -> dict:
        return {"running": self.running(), "exit_code": None if self.proc is None else self.proc.poll(),
                **{k: v for k, v in self.info.items() if k != "argv"}}

    def episodes(self) -> list[dict]:
        if self.log_path is None or not self.log_path.exists():
            return []
        tasks_file = Path(self.info.get("tasks_path") or self.args.bench / SUITES["focus5"][0])
        rows = [json.loads(line) for line in tasks_file.read_text().splitlines() if line.strip()]
        out = {}
        for m in _EPISODE.finditer(self.log_path.read_text(errors="ignore")):
            n = int(m.group(1))
            row = rows[n - 1] if 0 < n <= len(rows) else {}
            out[n] = {
                "n": n, "of": int(m.group(2)),
                "object": row.get("target") or ", ".join(row.get("objects", [])),
                "group": row.get("ood_key", ""),
                "result": "skipped" if m.group(3) is None else ("success" if m.group(3) == "True" else m.group(4)),
                "seconds": None if m.group(5) is None else float(m.group(5)),
            }
        return [out[k] for k in sorted(out)]

    def log_tail(self, lines: int) -> str:
        if self.log_path is None or not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            text = f.read().decode(errors="ignore")
        return "\n".join(text.splitlines()[-lines:])


# ----------------------------------------------------------------------- http
def make_http(args: argparse.Namespace, recorder: Recorder, runner: Runner, thor: ThorControl) -> ThreadingHTTPServer:
    page = HERE / "index.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj).encode(), "application/json", status)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self) -> None:
            path, _, query = self.path.partition("?")
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/":
                return self._send(page.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/config":
                return self._json({"default_host": args.thor, "default_port": args.thor_port,
                                   "default_user": args.thor_user, "default_dir": args.thor_dir,
                                   "suites": {k: v[2] for k, v in SUITES.items()},
                                   "objects": scene_objects(args.bench), "trained_objects": TRAINED_OBJECTS,
                                   "suite_sizes": {k: sum(1 for line in (args.bench / v[0]).read_text().splitlines() if line.strip())
                                                   for k, v in SUITES.items() if (args.bench / v[0]).exists()},
                                   "joints": JOINTS})
            if path == "/api/status":
                return self._json(runner.status())
            if path == "/api/records":
                after = int(params.get("after", -1))
                return self._json({"records": recorder.since(after, int(params.get("limit", 200))),
                                   "count": recorder.count})
            if path == "/api/episodes":
                return self._json(runner.episodes())
            if path == "/api/log":
                return self._send(runner.log_tail(int(params.get("lines", 80))).encode(), "text/plain; charset=utf-8")
            m = re.fullmatch(r"/obs/(\d+)\.jpg", path)
            if m and recorder.out is not None:
                f = recorder.out / "obs" / f"{int(m.group(1)):06d}.jpg"
                if f.exists():
                    return self._send(f.read_bytes(), "image/jpeg")
            self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:
            try:
                body = self._body()
                if self.path == "/api/check":
                    return self._json(check_server(str(body.get("host", "")).strip(), int(body.get("port", 8000)),
                                                   str(body.get("prompt") or "Place the cooking spoon in the plastic bin")))
                if self.path == "/api/thor/launch":
                    return self._json(thor.launch(body))
                if self.path == "/api/thor/stop":
                    return self._json(thor.stop(body))
                if self.path == "/api/thor/status":
                    return self._json(thor.status(body))
                if self.path == "/api/thor/checkpoints":
                    return self._json(thor.checkpoints(body))
                if self.path == "/api/thor/download":
                    return self._json(thor.download(body))
                if self.path == "/api/thor/delete":
                    return self._json(thor.delete(body))
                if self.path == "/api/start":
                    return self._json(runner.start(body))
                if self.path == "/api/stop":
                    return self._json(runner.stop())
                if self.path == "/api/control":
                    command = str(body.get("command", ""))
                    if command not in {"pause", "resume", "skip"}:
                        return self._json({"ok": False, "error": "unknown command"}, 400)
                    return self._json(runner.send(command))
            except ValueError as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:
                return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            self._send(b"not found", "text/plain", 404)

    return ThreadingHTTPServer((args.host, args.port), Handler)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="address the page listens on")
    p.add_argument("--port", type=int, default=8765, help="page port")
    p.add_argument("--relay-port", type=int, default=8001, help="local port the Isaac Lab client connects to")
    p.add_argument("--thor", default=os.environ.get("SO101_THOR_HOST", ""), help="pre-filled Thor address")
    p.add_argument("--thor-port", type=int, default=int(os.environ.get("SO101_THOR_PORT", 8000)))
    p.add_argument("--thor-user", default=os.environ.get("SO101_THOR_USER", ""), help="pre-filled SSH user on the Thor")
    p.add_argument("--thor-dir", default=os.environ.get("SO101_THOR_DIR", "~/so101-edge-thor"),
                   help="install folder of setup_thor.sh on the Thor")
    p.add_argument("--bench", type=Path, default=ROOT / "so101_bench", help="so101_bench checkout")
    p.add_argument("--python", type=Path, default=Path(sys.executable),
                   help="python with Isaac Sim 5.1 + Isaac Lab 2.3 (defaults to this interpreter)")
    p.add_argument("--runs", type=Path, default=ROOT / "runs", help="where run logs and frames are kept")
    p.add_argument("--pythonpath", type=Path, action="append", default=None,
                   help="extra PYTHONPATH entries for the client (default: <client>/vendor_py if present)")
    args = p.parse_args()
    args.bench = args.bench.resolve()
    if args.pythonpath is None:
        args.pythonpath = [ROOT / "vendor_py"] if (ROOT / "vendor_py").is_dir() else []
    args.runs.mkdir(parents=True, exist_ok=True)

    recorder = Recorder()
    relay = Relay(recorder, "127.0.0.1", args.relay_port)
    runner = Runner(args, relay, recorder)
    thor = ThorControl(args.thor_user, args.thor_dir)
    httpd = make_http(args, recorder, runner, thor)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[ui] open http://{args.host}:{args.port}/   (relay ws://127.0.0.1:{args.relay_port}, runs {args.runs})",
          flush=True)
    try:
        asyncio.run(relay.run())
    except KeyboardInterrupt:
        pass
    finally:
        if runner.running():
            runner.stop()


if __name__ == "__main__":
    main()
