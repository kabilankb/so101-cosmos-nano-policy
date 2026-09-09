"""Preflight checks for the two environments this pipeline straddles.

Every check here corresponds to a failure that was actually hit, and most of them
fail *silently* at runtime -- a wrong interpreter, a shadowed checkout or a
missing stats file produces a running system that returns wrong answers rather
than an error. Run ``so101 doctor`` before a long job.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import inventory, telemetry
from .config import Settings


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""

    @property
    def mark(self) -> str:
        return "ok  " if self.ok else "FAIL"


def _exists(name: str, path: Path, fix: str = "") -> Check:
    return Check(name, path.exists(), str(path), fix)


def run(cfg: Settings, quick: bool = False) -> list[Check]:
    """All checks. ``quick`` skips the ones that start a foreign interpreter."""
    checks: list[Check] = [
        _exists("cosmos-framework checkout", cfg.framework),
        _exists("framework venv python", cfg.venv_python,
                "create the uv venv in cosmos-framework"),
        _exists("training run directory", cfg.run_path),
        _exists("normalizer stats", cfg.stats_path,
                "without this the server returns normalized actions verbatim"),
        _exists("so101_bench checkout", cfg.bench),
        _exists("Isaac Sim Kit python", cfg.kit_python,
                "no conda env on the verified box has `omni` importable"),
    ]

    cuda = cfg.cuda_lib
    checks.append(
        Check(
            "venv cuBLAS libs",
            cuda is not None,
            str(cuda) if cuda else "not found under .venv/lib/python3.*/site-packages/nvidia/cu13",
            "LD_LIBRARY_PATH must start with this or every biased addmm raises "
            "CUBLAS_STATUS_NOT_INITIALIZED",
        )
    )

    exports = inventory.export_index(cfg)
    checks.append(
        Check(
            "exported checkpoints",
            bool(exports),
            f"{len(exports)} servable: {', '.join(str(i) for i in sorted(exports))}"
            if exports
            else "none -- run merge then export",
            "an export is required to serve; the server has no LoRA support",
        )
    )

    layouts = cfg.newest_layouts()
    checks.append(
        Check(
            "saved eval layouts",
            layouts is not None,
            str(layouts) if layouts else "none; layouts will be sampled fresh (~22 s/episode)",
            "must be a focus5_layouts_* file -- older test_layouts_* carry retired trial ids",
        )
    )

    episodes = cfg.bench / cfg.episodes_jsonl
    checks.append(_exists("eval episode set", episodes))

    disk = inventory.disk(cfg)
    checks.append(
        Check(
            "disk headroom",
            disk["free"] > 88 * 1024**3,
            f"{inventory.human_bytes(disk['free'])} free "
            f"({disk['percent_used']}% used), "
            f"{disk['checkpoints_remaining']} more checkpoint(s) fit",
            f"{inventory.human_bytes(disk['reclaimable'])} reclaimable via `so101 clean`",
        )
    )

    if telemetry.available():
        apps = telemetry.compute_apps()
        heavy = [a for a in apps if (a.get("mib") or 0) > 1024]
        checks.append(
            Check(
                "GPU is free",
                not heavy,
                ", ".join(f"pid {a['pid']} {a['mib']:.0f} MiB {a['name']}" for a in heavy)
                or "no process holding more than 1 GiB",
                "a stale policy server holds ~32 GB and a wedged Isaac Sim ~10 GB",
            )
        )

    if not quick:
        checks.append(_bench_shadowing(cfg))

    return checks


def _bench_shadowing(cfg: Settings) -> Check:
    """Is `so101_bench` pip-installed editable pointing somewhere else?

    On the verified box it resolves to a GR00T-era checkout with no
    ``utils/cosmos3.py`` and no ``scripts/cosmos3_eval.py``. The ``PYTHONPATH``
    prefix in `pipeline.evaluate` shadows it without disturbing that project.
    """
    if not cfg.kit_python.exists():
        return Check("so101_bench install", False, "Kit python missing; cannot check")
    try:
        out = subprocess.run(
            [str(cfg.kit_python), "-m", "pip", "show", "so101_bench"],
            capture_output=True, text=True, timeout=180, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("so101_bench install", False, f"could not run pip: {exc}")

    editable = ""
    for line in out.splitlines():
        if line.lower().startswith("editable project location"):
            editable = line.split(":", 1)[1].strip()

    if not editable:
        return Check("so101_bench install", False, "not installed, or not editable",
                     "the PYTHONPATH prefix used by `so101 eval` covers this anyway")

    expected = str(cfg.bench / "source/so101_bench")
    return Check(
        "so101_bench install",
        editable == expected,
        editable,
        "" if editable == expected
        else f"points elsewhere; `so101 eval` shadows it with PYTHONPATH={expected}",
    )


def report(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"  [{c.mark}] {c.name.ljust(width)}  {c.detail}")
        if c.fix and not c.ok:
            lines.append(f"         {' ' * width}  -> {c.fix}")
    failed = sum(1 for c in checks if not c.ok)
    lines.append("")
    lines.append(f"  {len(checks) - failed}/{len(checks)} checks passed")
    return "\n".join(lines)
