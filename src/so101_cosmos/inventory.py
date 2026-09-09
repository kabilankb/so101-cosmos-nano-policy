"""What exists on disk: checkpoints, their merges, their exports, and the space it costs."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from pathlib import Path

from .config import Settings

_ITER = re.compile(r"^iter_(\d{9})$")


def export_index(cfg: Settings) -> dict[int, Path]:
    """Map iteration -> export directory, read from each export's provenance.

    The directory name is a convention; ``checkpoint.json`` is the record. Read
    the file rather than trusting the name -- an export from ``iter_000000250``
    sitting in ``model_export_500`` is exactly the mistake this catches.
    """
    found: dict[int, Path] = {}
    for path in sorted(cfg.run_path.glob("model_export*")):
        marker = path / "checkpoint.json"
        if not marker.exists():
            continue
        try:
            blob = marker.read_text(errors="ignore")
        except OSError:
            continue
        if m := re.search(r"iter_(\d+)", blob):
            found[int(m.group(1))] = path
    return found


def checkpoints(cfg: Settings) -> list[dict]:
    """Every training checkpoint with its merge/export state, oldest first."""
    exports = export_index(cfg)
    rows = []
    for path in sorted(cfg.ckpt_dir.glob("iter_*")):
        m = _ITER.match(path.name)
        if not m:
            continue  # skips iter_*_merged, which is an intermediate, not a checkpoint
        n = int(m.group(1))
        exported = exports.get(n)
        rows.append(
            {
                "iteration": n,
                "name": path.name,
                "epochs": round(n / cfg.iters_per_epoch, 2) if cfg.iters_per_epoch else None,
                "merged": (path.parent / f"{path.name}_merged").exists(),
                "export": str(exported) if exported else None,
                "servable": exported is not None,
            }
        )
    return rows


def merged_intermediates(cfg: Settings) -> list[dict]:
    """``_merged`` dirs whose checkpoint already has an export -- safe to delete.

    A merged directory is a pure intermediate: it regenerates from the training
    checkpoint in about a minute, and costs ~29 GB while it sits there.
    """
    exports = export_index(cfg)
    out = []
    for path in sorted(cfg.ckpt_dir.glob("iter_*_merged")):
        if m := re.search(r"iter_(\d+)_merged", path.name):
            n = int(m.group(1))
            if n in exports:
                out.append({"iteration": n, "path": str(path), "bytes": dir_bytes(path)})
    return out


def dir_bytes(path: Path) -> int:
    """Recursive size in bytes. Symlinks are not followed."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            with contextlib.suppress(OSError):
                total += entry.stat().st_size
    return total


def disk(cfg: Settings) -> dict:
    """Free space where the run lives, and what one more checkpoint would cost.

    One evaluated checkpoint costs roughly 88 GB: ~29 GB training + ~29 GB merged
    + ~30 GB export.
    """
    target = cfg.run_path if cfg.run_path.exists() else cfg.framework
    usage = shutil.disk_usage(target)
    per_checkpoint = 88 * 1024**3
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent_used": round(100 * usage.used / usage.total, 1) if usage.total else None,
        "per_checkpoint_estimate": per_checkpoint,
        "checkpoints_remaining": usage.free // per_checkpoint,
        "reclaimable": sum(m["bytes"] for m in merged_intermediates(cfg)),
    }


def _legacy_eval_jobs(cfg: Settings) -> list[dict]:
    """Eval jobs recorded by `tools/train_monitor.py`, normalised to our schema.

    That tool keyed the stage as ``action`` and the checkpoint as a padded
    ``iter_000003750`` string. Its records are read-only here: the history is
    worth keeping, but nothing writes back into its directory.
    """
    jobs = []
    for directory in cfg.legacy_jobs_paths:
        for meta in sorted(directory.glob("*.meta.json")):
            try:
                raw = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            action = raw.get("action", "")
            if not action.startswith("eval"):
                continue
            name = raw.get("checkpoint") or ""
            if not name.startswith("iter_"):
                continue
            try:
                iteration = int(name.removeprefix("iter_"))
            except ValueError:
                continue
            jobs.append({"stage": "evaluate", "iteration": iteration, "log": raw.get("log", "")})
    return jobs


def eval_history(cfg: Settings) -> dict[int, dict]:
    """Aggregate every eval run by the checkpoint it was run against.

    Reads this package's own jobs plus any legacy directories, so history from
    before the package existed is not lost.
    """
    from .logparse import parse_episodes, summarize
    from .proc import JobStore

    records = [
        {"stage": j.get("stage", ""), "iteration": j.get("iteration"), "log": j.get("log", "")}
        for j in JobStore(cfg, enabled=False).all()
    ] + _legacy_eval_jobs(cfg)

    by_iter: dict[int, list[dict]] = {}
    for rec in records:
        if not rec["stage"].startswith("evaluate") or rec["iteration"] is None:
            continue
        if not rec["log"]:
            continue
        episodes = parse_episodes(Path(rec["log"]))
        if episodes:
            by_iter.setdefault(int(rec["iteration"]), []).extend(episodes)
    return {n: summarize(eps) for n, eps in by_iter.items()}


def human_bytes(n: float | None) -> str:
    if n is None:
        return "—"
    for unit in ("B", "K", "M", "G", "T"):
        if abs(n) < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit in ("B", "K") else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"
