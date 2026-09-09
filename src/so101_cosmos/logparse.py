"""Parsers for the two log formats this pipeline produces.

Both parsers read only the tail of a file, so they stay cheap on a multi-GB
training log and can be called on every dashboard poll.
"""

from __future__ import annotations

import re
from pathlib import Path

# The trainer emits two different shapes for the same information: the first 50
# iterations report a hit counter, everything after reports iteration speed.
# Grepping only the first makes a live run look stalled at iteration 550.
_RE_EARLY = re.compile(r"Iteration (\d+):.*?Loss: ([\d.]+)(?:.*?Time: ([\d.]+)s)?")
_RE_SPEED = re.compile(
    r"\]\s*(\d+)\s*:\s*iter_speed\s+([\d.]+)\s+seconds per iteration.*?Loss: ([\d.]+)"
)

# The benchmark emits different fields per task family. Move episodes carry lift
# telemetry; bin episodes (the focus5 eval) carry none of it, because lift is a
# move-task metric. Parse the common prefix first, then pick off whatever tail
# fields are present -- so a bin episode yields lift=None, not a false 0.00.
_RE_EPISODE = re.compile(
    r"Episode (\d+)/(\d+): success=(True|False), reason=(\S+?), length=([\d.]+)s(?P<tail>.*)"
)
_RE_FAILTYPE = re.compile(r"(?:failure_type|live_failure_reason)=(\S+?)(?:,|$)")
_RE_LIFT = re.compile(r"target_lift=([\d.]+)in")
_RE_DISTRACTOR = re.compile(r"max_distractor_lift=([\d.]+)in")

_ERROR = re.compile(r"Traceback|CUBLAS|out of memory|CUDA error|RuntimeError|Killed", re.I)

TAIL_BYTES = 4 * 1024 * 1024


def tail_text(path: Path, nbytes: int = TAIL_BYTES) -> str:
    """Last ``nbytes`` of a file, starting at a line boundary."""
    if not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > nbytes:
            fh.seek(size - nbytes)
            fh.readline()
        return fh.read().decode("utf-8", errors="replace")


def parse_training(path: Path) -> list[dict]:
    """Return ``[{iter, loss, sec}]`` from a training log, oldest first."""
    points: dict[int, dict] = {}
    for line in tail_text(path).splitlines():
        if "Loss:" not in line:
            continue
        if m := _RE_SPEED.search(line):
            points[int(m.group(1))] = {
                "iter": int(m.group(1)),
                "loss": float(m.group(3)),
                "sec": float(m.group(2)),
            }
        elif m := _RE_EARLY.search(line):
            points[int(m.group(1))] = {
                "iter": int(m.group(1)),
                "loss": float(m.group(2)),
                "sec": float(m.group(3)) if m.group(3) else None,
            }
    return [points[k] for k in sorted(points)]


def parse_episodes(path: Path) -> list[dict]:
    """Return one dict per rollout episode found in an eval log."""
    out: dict[int, dict] = {}
    for line in tail_text(path).splitlines():
        m = _RE_EPISODE.search(line)
        if not m:
            continue
        tail = m.group("tail") or ""
        ft = _RE_FAILTYPE.search(tail)
        lift = _RE_LIFT.search(tail)
        dis = _RE_DISTRACTOR.search(tail)
        n = int(m.group(1))
        out[n] = {
            "n": n,
            "total": int(m.group(2)),
            "success": m.group(3) == "True",
            "reason": m.group(4).rstrip(","),
            "length": float(m.group(5)),
            "failure_type": ft.group(1) if ft else None,
            "lift": float(lift.group(1)) if lift else None,
            "distractor": float(dis.group(1)) if dis else None,
        }
    return [out[k] for k in sorted(out)]


def summarize(episodes: list[dict]) -> dict:
    """Success counts for a set of parsed episodes."""
    done = len(episodes)
    ok = sum(1 for e in episodes if e["success"])
    return {
        "episodes": done,
        "successes": ok,
        "rate": (ok / done) if done else None,
        "total": episodes[-1]["total"] if episodes else None,
    }


def has_error(path: Path, lines: int = 400) -> str | None:
    """First error-looking line near the end of a log, if any."""
    for line in tail_text(path).splitlines()[-lines:]:
        if _ERROR.search(line):
            return line.strip()[:300]
    return None


def newest_training_log(framework: Path) -> Path | None:
    """Most recently modified ``train_resume_*.log`` under ``outputs/``."""
    hits = sorted(
        framework.glob("outputs/train_resume_*.log"), key=lambda p: p.stat().st_mtime
    )
    return hits[-1] if hits else None
