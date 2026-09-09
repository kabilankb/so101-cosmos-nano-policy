"""Launching, tracking and stopping pipeline stages as detached processes.

Stages outlive the controller on purpose: a training run lasts ~48 h and an eval
~40 min, and neither should die because a dashboard was closed. Each launch gets
its own log file and a JSON sidecar, so state survives a restart of this process.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import Settings
from .pipeline import CLIENT_STAGES, SERVER_STAGES, build


def alive(pid: int | None) -> bool:
    """Is this pid still running? Signal 0 tests for existence without delivering."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    import socket

    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


class JobLaunchError(RuntimeError):
    """A stage could not be started."""


class JobStore:
    """Persistent record of launched stages, keyed by job id."""

    def __init__(self, cfg: Settings, enabled: bool = True) -> None:
        self.cfg = cfg
        self.enabled = enabled
        self.dir = cfg.jobs_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._restore()

    # ----------------------------------------------------------- persistence
    def _restore(self) -> None:
        for meta in sorted(self.dir.glob("*.meta.json")):
            try:
                job = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") == "running" and not alive(job.get("pid")):
                # The controller was not running when this exited, so the real
                # returncode is unrecoverable. Say so rather than guess.
                job["status"] = "unknown"
            self._jobs[job["id"]] = job

    def _save(self, job: dict) -> None:
        (self.dir / f"{job['id']}.meta.json").write_text(json.dumps(job, indent=1))

    # ---------------------------------------------------------------- access
    def all(self) -> list[dict]:
        self.refresh()
        return sorted(self._jobs.values(), key=lambda j: j["started"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def running(self, stage: str) -> dict | None:
        for job in self._jobs.values():
            if job["stage"] == stage and job["status"] == "running" and alive(job["pid"]):
                return job
        return None

    def refresh(self) -> None:
        """Reap finished processes and record their exit status."""
        with self._lock:
            for job in self._jobs.values():
                if job["status"] != "running" or alive(job["pid"]):
                    continue
                job["status"] = "done"
                job["ended"] = time.time()
                self._save(job)

    # --------------------------------------------------------------- control
    def launch(self, stage: str, iteration: int | None = None) -> dict:
        if not self.enabled:
            raise JobLaunchError("launching is disabled (--no-launch)")

        existing = self.running(stage)
        if existing:
            raise JobLaunchError(f"{stage} is already running (job {existing['id']})")

        argv, cwd, env = build(self.cfg, stage, iteration)

        job_id = f"{stage}_{iteration or 'na'}_{datetime.now():%Y%m%d_%H%M%S}"
        log = self.dir / f"{job_id}.log"
        with log.open("wb") as fh:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # survives the controller
            )

        job = {
            "id": job_id,
            "stage": stage,
            "side": "server" if stage in SERVER_STAGES else "client",
            "iteration": iteration,
            "pid": proc.pid,
            "status": "running",
            "log": str(log),
            "started": time.time(),
            "ended": None,
            "argv": argv,
            "cwd": str(cwd),
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        return job

    def stop(self, job_id: str, hard: bool = False) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobLaunchError(f"no such job: {job_id}")
        pid = job.get("pid")
        if alive(pid):
            # start_new_session put the stage in its own process group, so this
            # reaches the whole tree -- Isaac Sim spawns children that ignore a
            # signal sent to the leader alone.
            sig = signal.SIGKILL if hard else signal.SIGTERM
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(pid), sig)
        job["status"] = "stopped"
        job["ended"] = time.time()
        self._save(job)
        return job

    def tail(self, job_id: str, lines: int = 40) -> list[str]:
        job = self._jobs.get(job_id)
        if job is None:
            return []
        from .logparse import tail_text

        return tail_text(Path(job["log"]), 256 * 1024).splitlines()[-lines:]


def state(cfg: Settings, store: JobStore) -> dict:
    """A snapshot of what is up right now, for the CLI and the web UI."""
    server = store.running("serve")
    client = store.running("evaluate") or store.running("evaluate_gui")
    return {
        "server": {
            "job": server,
            "port": cfg.policy_port,
            "listening": port_open(cfg.policy_port),
            "stages": sorted(SERVER_STAGES),
        },
        "client": {
            "job": client,
            "stages": sorted(CLIENT_STAGES),
        },
        "training": store.running("train"),
    }
