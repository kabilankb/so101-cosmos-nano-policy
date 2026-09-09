"""Command construction for every stage of the pipeline.

Each builder returns ``(argv, cwd, env)`` and runs nothing. Keeping construction
pure is what makes the flags testable: the whole serving contract is a set of
arguments that fail *silently* when wrong -- the server starts happily and
returns wrong actions -- so `tests/test_pipeline.py` asserts on them directly.

Stage order::

    train -> merge -> export -> serve -> warmup -> evaluate
                                  |
                        (server)  |  (client)
                     cosmos-framework <-> so101_bench / Isaac Lab
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from .config import Settings

Command = tuple[list[str], Path, dict[str, str]]

STAGES = ("train", "merge", "export", "serve", "warmup", "evaluate", "evaluate_gui")

#: Stages that run in cosmos-framework's venv and hold the GPU as the *server*.
SERVER_STAGES = frozenset({"train", "merge", "export", "serve"})
#: Stages that run under the Kit python and drive the simulation as the *client*.
CLIENT_STAGES = frozenset({"warmup", "evaluate", "evaluate_gui"})


def base_env(cfg: Settings) -> dict[str, str]:
    """Environment shared by every stage that touches the GPU.

    Three variables, each fixing a distinct failure:

    ``LD_LIBRARY_PATH``
        Prepends the venv's own cu13 libs so cuBLAS and cuBLASLt come from the
        same install. Without it every biased ``addmm`` raises
        ``CUBLAS_STATUS_NOT_INITIALIZED`` -- on an idle GPU, in eager mode. It is
        not memory pressure, and once it fires the CUDA context is poisoned.
    ``PYTORCH_CUDA_ALLOC_CONF``
        Reduces allocator fragmentation while training and simulation share one card.
    ``SO101_ROOT``
        Silences the checkpoint-metadata transform lookup, which otherwise drops
        the server back to the default ``ActionTransformPipeline``.
    """
    env = dict(os.environ)
    cuda = cfg.cuda_lib
    if cuda is not None:
        env["LD_LIBRARY_PATH"] = f"{cuda}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["SO101_ROOT"] = str(cfg.framework / cfg.dataset)
    return env


def train(cfg: Settings, extra: Sequence[str] = ()) -> Command:
    """Launch (or auto-resume) the SFT run.

    Auto-resume reads ``checkpoints/latest_checkpoint.txt`` and takes priority
    over ``checkpoint.load_path``, so relaunching the same job name continues
    rather than restarts.
    """
    return (["bash", str(cfg.framework / cfg.launcher), *extra], cfg.framework, base_env(cfg))


def merge(cfg: Settings, iteration: int) -> Command:
    """Fold LoRA adapters into the base weights.

    Neither ``export_model`` nor the policy server understands LoRA.
    ``LoraInjectedLinear`` keeps the base weight at its original key and computes
    ``y = Wx + (alpha/rank)*BAx``, so folding is exact. The scale is *not*
    recorded in the checkpoint -- passing the wrong alpha/rank would mis-scale
    silently, so the script raises on a rank mismatch instead.
    """
    src = cfg.checkpoint_path(iteration)
    argv = [
        str(cfg.venv_python), "-u", "-m", "cosmos_framework.scripts.merge_lora_dcp",
        "--input", str(src),
        "--output", f"{src}_merged",
        "--lora-alpha", str(cfg.lora_alpha),
        "--lora-rank", str(cfg.lora_rank),
    ]
    return (argv, cfg.framework, base_env(cfg))


def export(cfg: Settings, iteration: int) -> Command:
    """Export a merged checkpoint to consolidated safetensors.

    Exports against the *registered experiment*, never the run's own
    ``config.yaml``: that config still declares ``lora_enabled = true`` and
    demands the adapter keys the merge just folded away. The four overrides are
    what the last working export recorded in its own ``checkpoint.json``.
    """
    src = f"{cfg.checkpoint_path(iteration)}_merged"
    argv = [
        str(cfg.venv_python), "-u", "-m", "cosmos_framework.scripts.export_model",
        "--checkpoint-path", src,
        "--config-file", "cosmos_framework/configs/base/config.py",
        "--experiment", cfg.experiment,
        "--experiment-overrides",
        "model.config.diffusion_expert_config.load_weights_from_pretrained=False",
        "model.config.vlm_config.pretrained_weights.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "model.config.ema.enabled=false",
        "-o", str(cfg.export_path(iteration)),
    ]
    return (argv, cfg.framework, base_env(cfg))


def serve(cfg: Settings, iteration: int) -> Command:
    """Start the Cosmos policy server against an exported checkpoint.

    Every flag below fails silently if omitted or wrong:

    ``--arm-joint-dim 5``
        the server assumed DROID's 7-joint arm until this was generalised.
    ``--conditioning-fps 30``
        matches the dataset; a wrong value shifts the time conditioning.
    ``--no-flip-gripper``
        the default ``1.0 - x`` flip is DROID's ``[0,1]`` convention. SO-101
        trains on raw LeRobot ``.pos`` ``[0,100]``, so the flip sends ``-86.3``.
    ``--action-normalization`` + ``--normalizer-stats-path``
        without *both*, normalized outputs (~``[-1,1]``) are returned verbatim
        and every joint is commanded to about zero.
    ``--view-description``
        the text conditioning the model trained against. The server default is
        DROID's wording, which describes three views; SO-101 sends two.

    ``--device-memory-bytes`` is deliberately absent: it is a no-op on a single
    GPU, passed only to ``_build_model_parallelism``, which ignores the value.
    """
    exported = cfg.export_path(iteration)
    if not exported.exists():
        raise FileNotFoundError(
            f"no export for iteration {iteration} at {exported} -- run merge, then export"
        )
    argv = [
        str(cfg.venv_python), "-u", "-m",
        "cosmos_framework.scripts.action_policy_server_robolab",
        "--checkpoint-path", str(exported),
        "--port", str(cfg.policy_port),
        "--domain-name", cfg.domain,
        "--action-dim", str(cfg.action_dim),
        "--arm-joint-dim", str(cfg.arm_joint_dim),
        "--action-space", cfg.action_space,
        "--conditioning-fps", str(cfg.conditioning_fps),
        "--no-flip-gripper",
        "--action-normalization", cfg.action_normalization,
        "--normalizer-stats-path", cfg.stats,
        "--view-description", cfg.view_description,
        "--no-guardrails",
    ]
    return (argv, cfg.framework, base_env(cfg))


WARMUP_SOURCE = '''
import numpy as np
from openpi_client import websocket_client_policy

rng = np.random.default_rng(0)
req = {
    # wrist (480,640,3) stacked over overhead (480,640,3); server resizes to 540x640
    "observation/image": rng.integers(0, 255, size=(960, 640, 3), dtype=np.uint8),
    "observation/joint_position": np.zeros((1, %(arm)d), dtype=np.float32),
    "observation/gripper_position": np.full((1, 1), 20.0, dtype=np.float32),
    "prompt": "%(prompt)s",
}
c = websocket_client_policy.WebsocketClientPolicy("%(host)s", %(port)d)
a = np.asarray(c.infer(req)["action"])
print("warmup ok", a.shape, "min %%.2f max %%.2f" %% (a.min(), a.max()), flush=True)
'''


def warmup(cfg: Settings, host: str = "localhost", prompt: str | None = None) -> Command:
    """One synthetic inference request -- the whole wire contract in a single call.

    This is exactly the payload ``Cosmos3RemotePolicy._build_request`` produces,
    so a success proves image layout, normalization and denormalization are all
    correct. A healthy chunk lands in ``.pos`` units, roughly ``[-76, 80]`` per
    dim; anything inside ``[-1, 1]`` means the normalization flags did not take.

    Run it *before* the client. The server is single-threaded per request, so a
    probe sent during a live rollout queues behind it and can time out. The first
    request also triggers a ``torch.compile`` that takes minutes -- getting that
    out of the way while the GPU is still empty is the point.
    """
    src = WARMUP_SOURCE % {
        "arm": cfg.arm_joint_dim,
        "prompt": prompt or "Place the cooking spoon in the plastic bin",
        "host": host,
        "port": cfg.policy_port,
    }
    return ([str(cfg.kit_python), "-u", "-c", src], cfg.framework, base_env(cfg))


def evaluate(cfg: Settings, gui: bool = False, layouts: Path | None = None,
             episodes: str | None = None) -> Command:
    """Run the Isaac Lab client against a live policy server.

    ``PYTHONPATH`` is mandatory: ``so101_bench`` is pip-installed editable
    pointing at a different checkout, and the prefix shadows it without
    disturbing that project. The Kit python is likewise mandatory -- no conda env
    on the verified box has ``omni`` importable, and ``isaaclab`` alone imports
    in one of them, which makes it look viable until Isaac Sim fails to boot.
    """
    env = base_env(cfg)
    env["PYTHONPATH"] = str(cfg.bench / "source/so101_bench")
    # setdefault is not enough: an SSH session commonly carries DISPLAY as an
    # empty string, which is present-but-useless and would leave Isaac Sim with
    # no display to open a window on.
    if gui and not env.get("DISPLAY"):
        env["DISPLAY"] = ":0"

    argv = [
        str(cfg.kit_python), "-u", "scripts/cosmos3_eval.py",
        "--task", cfg.task,
        "--episodes_jsonl", episodes or cfg.episodes_jsonl,
        "--policy_host", "localhost",
        "--policy_port", str(cfg.policy_port),
        "--action_horizon", str(cfg.action_horizon),
    ]

    chosen = layouts if layouts is not None else cfg.newest_layouts()
    if chosen is not None:
        rel = chosen.relative_to(cfg.bench) if chosen.is_absolute() else chosen
        argv += ["--episode_layouts_jsonl", str(rel)]
    if not gui:
        argv.append("--headless")
    return (argv, cfg.bench, env)


def build(cfg: Settings, stage: str, iteration: int | None = None, **kw: object) -> Command:
    """Dispatch by stage name, for the web UI and the CLI alike."""
    if stage == "train":
        return train(cfg)
    if stage == "warmup":
        return warmup(cfg)
    if stage in ("evaluate", "evaluate_gui"):
        # `iteration` is bookkeeping only here: the client talks to whatever
        # checkpoint the server already loaded. Recording it is what lets
        # eval history attribute episodes to a checkpoint.
        return evaluate(cfg, gui=stage.endswith("_gui"))
    if stage in ("merge", "export", "serve"):
        if iteration is None:
            raise ValueError(f"stage {stage!r} needs an iteration")
        return {"merge": merge, "export": export, "serve": serve}[stage](cfg, iteration)
    raise ValueError(f"unknown stage {stage!r} (known: {', '.join(STAGES)})")
