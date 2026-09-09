"""Resolved paths and run parameters for the SO-101 Cosmos pipeline.

Every path the pipeline touches is declared here once. Defaults match the
workstation the pipeline was verified on; override any field with an environment
variable (``SO101_CFG_<FIELD>``) or a TOML file passed as ``--config`` or named
by ``SO101_CFG_FILE``.

The env prefix is ``SO101_CFG_``, not ``SO101_``, because ``SO101_ROOT`` is a
variable *cosmos-framework itself* reads to resolve checkpoint-metadata
transforms. Keeping the namespaces apart means setting one can never silently
change the other.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from pathlib import Path

ENV_PREFIX = "SO101_CFG_"
ENV_CONFIG_FILE = "SO101_CFG_FILE"


@dataclasses.dataclass(frozen=True, slots=True)
class Settings:
    """Where everything lives, and the run parameters that must not drift."""

    # -- checkouts and interpreters ------------------------------------------
    framework: Path = Path("/home/zeux/cosmos-framework")
    bench: Path = Path("/home/zeux/IsaacLab/so101_bench")
    kit_python: Path = Path("/home/zeux/IsaacLab/_isaac_sim/python.sh")

    # -- the training run, relative to `framework` ---------------------------
    run_dir: str = "outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_focus5_1gpu"
    job_dir: str = "outputs/so101_cosmos_jobs"
    # Job directories written by an earlier tool, read for eval history only.
    # Colon-separated, relative to `framework`. Never written to.
    legacy_job_dirs: str = "outputs/monitor_jobs"
    dataset: str = "examples/data/so101_bench_sim_6"
    stats: str = "cosmos_framework/data/generator/action/normalizer_stats/so101_lerobot_stats.json"
    launcher: str = "examples/launch_sft_action_policy_so101_nano_focus5_1gpu.sh"

    # -- model / serving contract --------------------------------------------
    # Changing any of these without retraining silently breaks the policy: the
    # server keeps answering, with wrong actions. See `pipeline.serve`.
    experiment: str = "action_policy_so101_nano_focus5"
    domain: str = "so101"
    action_dim: int = 6
    arm_joint_dim: int = 5
    action_space: str = "joint_pos"
    conditioning_fps: int = 30
    action_normalization: str = "minmax"
    view_description: str = (
        "The top half is from the front-facing wrist camera. "
        "The bottom half is from the fixed overhead camera."
    )
    lora_alpha: int = 32
    lora_rank: int = 16

    # -- evaluation ----------------------------------------------------------
    task: str = "So101Bench-Bin-v0"
    episodes_jsonl: str = "tasks/focus5.jsonl"
    layouts_glob: str = "tasks/layouts/focus5_layouts_*.jsonl"
    action_horizon: int = 32

    # -- training bookkeeping ------------------------------------------------
    max_iter: int = 4000
    save_iter: int = 250
    global_batch: int = 32
    iters_per_epoch: int = 899  # 28,760 valid windows / global batch 32

    # -- ports ---------------------------------------------------------------
    policy_port: int = 8000
    web_port: int = 8800

    # ------------------------------------------------------------------ paths
    @property
    def run_path(self) -> Path:
        return self.framework / self.run_dir

    @property
    def ckpt_dir(self) -> Path:
        return self.run_path / "checkpoints"

    @property
    def jobs_path(self) -> Path:
        return self.framework / self.job_dir

    @property
    def legacy_jobs_paths(self) -> list[Path]:
        return [self.framework / d for d in self.legacy_job_dirs.split(':') if d]

    @property
    def stats_path(self) -> Path:
        return self.framework / self.stats

    @property
    def venv_python(self) -> Path:
        return self.framework / ".venv/bin/python"

    @property
    def cuda_lib(self) -> Path | None:
        """The venv's own cu13 lib dir, found by glob so a python upgrade can't stale it.

        This must be prepended to ``LD_LIBRARY_PATH``. An inherited ROS profile
        puts ``/usr/local/cuda/lib64`` first, which loads ``libcublasLt`` from
        system CUDA while ``libcublas`` comes from the venv; the mismatched pair
        cannot initialise a handle and every biased ``addmm`` fails with
        ``CUBLAS_STATUS_NOT_INITIALIZED``.
        """
        hits = sorted(self.framework.glob(".venv/lib/python3.*/site-packages/nvidia/cu13/lib"))
        return hits[-1] if hits else None

    def export_path(self, iteration: int) -> Path:
        return self.run_path / f"model_export_{iteration}"

    def checkpoint_path(self, iteration: int) -> Path:
        return self.ckpt_dir / f"iter_{iteration:09d}"

    def latest_iteration(self) -> int | None:
        """Whatever ``latest_checkpoint.txt`` points at, as an int.

        Note this is the *newest* checkpoint, which is not the *best* one --
        iter 4000 has the lowest loss and scores 0/20.
        """
        marker = self.ckpt_dir / "latest_checkpoint.txt"
        if not marker.exists():
            return None
        try:
            return int(marker.read_text().strip().removeprefix("iter_"))
        except ValueError:
            return None

    def newest_layouts(self) -> Path | None:
        """Most recent saved layout file, or None to sample fresh ones.

        Must be a ``focus5_layouts_*`` file: the older ``test_layouts_*`` carry
        trial ids for the retired Move trials only, and ``cosmos3_eval.py``
        hard-errors on any requested trial id missing from the layout file.
        """
        hits = sorted(self.bench.glob(self.layouts_glob))
        return hits[-1] if hits else None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, config_file: Path | str | None = None) -> Settings:
        """Defaults, then the TOML file, then the environment."""
        values: dict[str, object] = {}

        path = config_file or os.environ.get(ENV_CONFIG_FILE)
        if path:
            path = Path(path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"config file not found: {path}")
            with path.open("rb") as fh:
                values.update(tomllib.load(fh).get("so101", {}))

        fields = {f.name: f for f in dataclasses.fields(cls)}
        for name in fields:
            env = os.environ.get(ENV_PREFIX + name.upper())
            if env is not None:
                values[name] = env

        unknown = set(values) - set(fields)
        if unknown:
            raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")

        coerced = {n: _coerce(fields[n].type, v) for n, v in values.items()}
        return cls(**coerced)  # type: ignore[arg-type]


def _coerce(declared: object, value: object) -> object:
    """Cast a TOML or env value to the field's declared type."""
    text = str(declared)
    if value is None:
        return value
    if "Path" in text:
        return Path(str(value)).expanduser()
    if "int" in text:
        return int(value)
    return str(value) if "str" in text else value
