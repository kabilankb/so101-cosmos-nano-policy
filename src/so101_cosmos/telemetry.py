"""GPU telemetry, read from ``nvidia-smi``. No pynvml dependency.

Two readings are easy to misread on this card:

* ``T.Limit`` values in ``nvidia-smi -q`` are *margins*, not temperatures --
  Blackwell reports degrees remaining before a limit engages, so ``-2`` means two
  degrees of headroom and ``0`` means you are at the limit now.
* ``SW Thermal Slowdown`` / ``SW Power Cap`` name which limiter is currently
  costing you clock speed, which is why `sample` reports the clock ratio.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque

_QUERY = "temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm,memory.used,memory.total"


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _num(token: str) -> float | None:
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        return None


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def sample() -> dict:
    """One telemetry reading. Empty dict when nvidia-smi is unavailable."""
    out = _run(["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"])
    row = out.strip().splitlines()
    if not row:
        return {}
    parts = row[0].split(",")
    if len(parts) < 7:
        return {}
    temp, power, power_limit, clock, clock_max, mem_used, mem_total = (
        _num(p) for p in parts[:7]
    )
    return {
        "t": time.time(),
        "temp": temp,
        "power": power,
        "power_limit": power_limit,
        "clock": clock,
        "clock_max": clock_max,
        "clock_ratio": (clock / clock_max) if clock and clock_max else None,
        "mem_used": mem_used,
        "mem_total": mem_total,
    }


def compute_apps() -> list[dict]:
    """Processes currently holding GPU memory.

    Worth checking before launching anything: a stale policy server holds ~32 GB
    and a wedged Isaac Sim ~10 GB, and both outlive their parent. On the verified
    box, killing two of them took iteration time from 43.5 s to 37.1 s.
    """
    out = _run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory,name", "--format=csv,noheader,nounits"]
    )
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            apps.append({"pid": int(parts[0]), "mib": _num(parts[1]), "name": parts[2]})
    return apps


class Sampler(threading.Thread):
    """Background poller keeping a bounded history of readings."""

    def __init__(self, interval: float = 5.0, keep: int = 2880) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.history: deque[dict] = deque(maxlen=keep)
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            reading = sample()
            if reading:
                self.history.append(reading)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()

    @property
    def latest(self) -> dict:
        return self.history[-1] if self.history else {}
