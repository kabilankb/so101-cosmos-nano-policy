#!/usr/bin/env python3
"""Evaluate a Cosmos3-Edge SO-101 checkpoint in the so101_bench Isaac Lab sim.

Runs the whole closed loop for one checkpoint ("one epoch") and leaves nothing
behind: merge LoRA -> export safetensors -> start the policy SERVER -> warm it
up -> drive it with the Isaac Lab CLIENT -> parse episode outcomes -> tear the
server down -> write result.json.

    server: cosmos_framework.scripts.action_policy_server_robolab  (this venv)
    client: so101_bench/scripts/cosmos3_eval.py                    (Kit python)

The two halves must run under different interpreters -- the server needs the
cosmos-framework venv, the client needs Isaac Sim's Kit python, which is the
only interpreter on this box with `omni` importable. That is why this is an
orchestrator rather than one process.

Examples::

    # nearest saved checkpoint to the end of epoch 1
    python3 tools/edge_eval_epoch.py --epoch 1

    # a specific iteration, with the Isaac Sim viewer open
    python3 tools/edge_eval_epoch.py --iter 3500 --gui

    # every saved checkpoint that has no result yet, 20 episodes each
    python3 tools/edge_eval_epoch.py --all --num-episodes 20

    # queue behind the live training run instead of fighting it for the GPU
    python3 tools/edge_eval_epoch.py --all --defer-until-training-done

SHARING THE GPU WITH TRAINING
-----------------------------
Memory is not the problem: training peaks at ~26 GB of 96 GB, and the server is
capped with --device-memory-bytes so it does not size itself as if it owned the
card. Heat and compute are the problem. The training run already sits at ~92 C
with sw_thermal_slowdown active and SM clocks pulled from 3090 to ~2250 MHz.
Adding Isaac Sim on top slows training further and pushes thermals harder, so
--defer-until-training-done is the safe default posture for a long sweep;
--allow-concurrent is the explicit opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

FRAMEWORK = Path("/home/<user>/cosmos-framework")
RUN_DIR = FRAMEWORK / "outputs/train/cosmos3_action/action_sft_sim/action_policy_so101_edge_focus5_multi_1gpu"
CKPT_DIR = RUN_DIR / "checkpoints"
EVAL_DIR = FRAMEWORK / "outputs/edge_eval"

VENV_PY = FRAMEWORK / ".venv/bin/python"
VENV_CU13 = FRAMEWORK / ".venv/lib/python3.13/site-packages/nvidia/cu13/lib"
SO101_BENCH = Path("/home/<user>/IsaacLab/so101_bench")
KIT_PY = Path("/home/<user>/IsaacLab/_isaac_sim/python.sh")

DATASET = "examples/data/so101_bench_sim_6"
STATS = "cosmos_framework/data/generator/action/normalizer_stats/so101_lerobot_stats.json"
VIEW_DESC = ("The top half is from the front-facing wrist camera. "
             "The bottom half is from the fixed overhead camera.")
EXPERIMENT = "action_policy_so101_edge_focus5_multi"

# Must match [model] in action_policy_so101_edge_focus5_multi_1gpu.toml. Merging
# with the wrong scale silently produces a valid-looking but wrong policy, so
# these are asserted against the TOML at startup rather than trusted.
LORA_RANK, LORA_ALPHA = "64", "128"
TOML = FRAMEWORK / "examples/toml/sft_config/action_policy_so101_edge_focus5_multi_1gpu.toml"

ITERS_PER_EPOCH = 55385 / 32          # 1731.4, measured from the loader
POLICY_PORT = 8000
SERVER_MEM_BYTES = "50000000000"      # leave room for Isaac Sim + training

_RE_EPISODE = re.compile(
    r"Episode (\d+)/(\d+): success=(True|False), reason=(\S+?), length=([\d.]+)s(?P<tail>.*)"
)
_RE_FAILTYPE = re.compile(r"(?:failure_type|live_failure_reason)=(\S+?)(?:,|$)")
_RE_LIFT = re.compile(r"target_lift=([\d.]+)in")


def log(msg: str) -> None:
    print(f"[eval] {datetime.now():%H:%M:%S} {msg}", flush=True)


def gpu_env() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{VENV_CU13}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["SO101_ROOT"] = str(FRAMEWORK / DATASET)
    return env


def verify_lora_scale() -> None:
    """Refuse to merge at a scale the run was not trained at."""
    if not TOML.is_file():
        log(f"WARNING: {TOML.name} missing; trusting rank={LORA_RANK} alpha={LORA_ALPHA}")
        return
    text = TOML.read_text()
    rank = re.search(r"^\s*lora_rank\s*=\s*(\d+)", text, re.M)
    alpha = re.search(r"^\s*lora_alpha\s*=\s*(\d+)", text, re.M)
    if rank and alpha and (rank.group(1), alpha.group(1)) != (LORA_RANK, LORA_ALPHA):
        raise SystemExit(
            f"LoRA scale mismatch: TOML says rank={rank.group(1)} alpha={alpha.group(1)}, "
            f"this script merges at rank={LORA_RANK} alpha={LORA_ALPHA}. Fix one of them."
        )


def training_running() -> bool:
    return subprocess.run(["pgrep", "-f", "cosmos_framework.scripts.train"],
                          capture_output=True).returncode == 0


def gpu_temp() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


# --------------------------------------------------------------- checkpoints

def saved_checkpoints() -> list[int]:
    if not CKPT_DIR.is_dir():
        return []
    out = []
    for d in sorted(CKPT_DIR.glob("iter_*")):
        if d.is_dir() and not d.name.endswith("_merged") and (d / "model").is_dir():
            out.append(int(re.sub(r"\D", "", d.name)))
    return sorted(out)


def iter_name(n: int) -> str:
    return f"iter_{n:09d}"


def resolve_target(args) -> list[int]:
    have = saved_checkpoints()
    if not have:
        raise SystemExit(f"no saved checkpoints in {CKPT_DIR} yet (first save lands at iter 500)")
    if args.iter is not None:
        if args.iter not in have:
            raise SystemExit(f"iteration {args.iter} not saved. Available: {have}")
        return [args.iter]
    if args.epoch is not None:
        want = args.epoch * ITERS_PER_EPOCH
        best = min(have, key=lambda n: abs(n - want))
        log(f"epoch {args.epoch} -> iter {want:.0f} -> nearest saved checkpoint {best}")
        return [best]
    if args.latest:
        return [have[-1]]
    if args.all:
        todo = [n for n in have if not (EVAL_DIR / iter_name(n) / "result.json").is_file()]
        if not todo:
            log("every saved checkpoint already has a result.json — nothing to do")
        return todo
    return [have[-1]]


# ------------------------------------------------------------------- stages

def run(cmd: list[str], cwd: Path, env: dict, logfile: Path, what: str) -> None:
    log(f"{what}: {' '.join(str(c) for c in cmd[:7])} …")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("wb") as fh:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env,
                            stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    if rc != 0:
        raise RuntimeError(f"{what} failed (rc={rc}); see {logfile}")
    log(f"{what}: ok")


def stage_merge(n: int, out: Path, force: bool) -> Path:
    src = CKPT_DIR / iter_name(n)
    dst = CKPT_DIR / f"{iter_name(n)}_merged"
    if dst.is_dir() and not force:
        log(f"merge: reusing {dst.name}")
        return dst
    if dst.is_dir():
        shutil.rmtree(dst)
    run([VENV_PY, "-u", "-m", "cosmos_framework.scripts.merge_lora_dcp",
         "--input", src, "--output", dst,
         "--lora-alpha", LORA_ALPHA, "--lora-rank", LORA_RANK],
        FRAMEWORK, gpu_env(), out / "merge.log", "merge")
    return dst


def stage_export(n: int, out: Path, force: bool) -> Path:
    dst = RUN_DIR / f"model_export_{n}"
    if (dst / "checkpoint.json").is_file() and not force:
        log(f"export: reusing {dst.name}")
        return dst
    if dst.is_dir():
        shutil.rmtree(dst)
    run([VENV_PY, "-u", "-m", "cosmos_framework.scripts.export_model",
         "--checkpoint-path", CKPT_DIR / f"{iter_name(n)}_merged",
         "--config-file", "cosmos_framework/configs/base/config.py",
         "--experiment", EXPERIMENT,
         "--experiment-overrides",
         "model.config.diffusion_expert_config.load_weights_from_pretrained=False",
         "model.config.vlm_config.pretrained_weights.enabled=False",
         "checkpoint.load_from_object_store.enabled=False",
         "model.config.ema.enabled=false",
         "-o", dst],
        FRAMEWORK, gpu_env(), out / "export.log", "export")
    return dst


def wait_healthz(port: int, proc: subprocess.Popen, timeout: float) -> None:
    """Poll /healthz, failing fast if the server process dies first."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"policy server exited early (rc={proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"policy server not healthy after {timeout:.0f}s")


def start_server(export: Path, out: Path, port: int) -> subprocess.Popen:
    logfile = out / "server.log"
    cmd = [VENV_PY, "-u", "-m", "cosmos_framework.scripts.action_policy_server_robolab",
           "--checkpoint-path", export, "--port", str(port),
           "--domain-name", "so101", "--action-dim", "6", "--arm-joint-dim", "5",
           "--action-space", "joint_pos", "--conditioning-fps", "30",
           "--no-flip-gripper", "--action-normalization", "minmax",
           "--normalizer-stats-path", STATS,
           "--view-description", VIEW_DESC, "--no-guardrails",
           "--device-memory-bytes", SERVER_MEM_BYTES]
    log(f"server: starting on :{port} from {export.name}")
    fh = logfile.open("wb")
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(FRAMEWORK), env=gpu_env(),
                            stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    wait_healthz(port, proc, timeout=1800)
    log("server: healthz OK")
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    log("server: stopping")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    log("server: stopped")


def stage_client(out: Path, args) -> str:
    """Drive the server from the Isaac Lab sim. Returns the client's stdout."""
    env = gpu_env()
    env["PYTHONPATH"] = str(SO101_BENCH / "source/so101_bench")
    if args.gui:
        env["DISPLAY"] = env.get("DISPLAY", ":0")
    cmd = [KIT_PY, "-u", "scripts/cosmos3_eval.py",
           "--task", args.task,
           "--episodes_jsonl", args.episodes_jsonl,
           "--policy_host", "localhost", "--policy_port", str(args.port),
           "--action_horizon", "32",
           "--num_episodes", str(args.num_episodes)]
    # Replaying saved layouts skips ~22 s of placement sampling per episode.
    # Must match the episode file: cosmos3_eval.py hard-errors on any requested
    # trial_id missing from the layout file.
    stem = Path(args.episodes_jsonl).stem
    layouts = sorted((SO101_BENCH / "tasks/layouts").glob(f"{stem}_layouts_*.jsonl"))
    if layouts and not args.fresh_layouts:
        cmd += ["--episode_layouts_jsonl", str(layouts[-1].relative_to(SO101_BENCH))]
        log(f"client: replaying layouts {layouts[-1].name}")
    if not args.gui:
        cmd.append("--headless")

    logfile = out / "eval.log"
    log(f"client: {args.num_episodes} episodes on {args.task}")
    with logfile.open("wb") as fh:
        rc = subprocess.run([str(c) for c in cmd], cwd=str(SO101_BENCH), env=env,
                            stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL).returncode
    text = logfile.read_text(errors="replace")
    if rc != 0:
        log(f"client: rc={rc} (parsing whatever episodes completed)")
    return text


def parse_episodes(text: str) -> list[dict]:
    eps = []
    for m in _RE_EPISODE.finditer(text):
        idx, total, ok, reason, length = m.group(1, 2, 3, 4, 5)
        tail = m.group("tail") or ""
        ft = _RE_FAILTYPE.search(tail)
        lift = _RE_LIFT.search(tail)
        eps.append({
            "episode": int(idx), "of": int(total),
            "success": ok == "True", "reason": reason,
            "length_s": float(length),
            "failure_type": ft.group(1) if ft else None,
            # Bin tasks carry no lift telemetry; None, not a misleading 0.0.
            "lift_in": float(lift.group(1)) if lift else None,
        })
    return eps


def evaluate(n: int, args) -> dict:
    out = EVAL_DIR / iter_name(n)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    server = None
    try:
        stage_merge(n, out, args.force)
        export = stage_export(n, out, args.force)
        server = start_server(export, out, args.port)
        text = stage_client(out, args)
    finally:
        stop_server(server)

    eps = parse_episodes(text)
    ok = sum(1 for e in eps if e["success"])
    lifts = [e["lift_in"] for e in eps if e["lift_in"] is not None]
    result = {
        "iter": n,
        "iter_name": iter_name(n),
        "epoch": n / ITERS_PER_EPOCH,
        "task": args.task,
        "episodes_jsonl": args.episodes_jsonl,
        "requested_episodes": args.num_episodes,
        "episodes": len(eps),
        "successes": ok,
        "success_rate": (ok / len(eps)) if eps else None,
        "best_lift_in": max(lifts) if lifts else None,
        "detail": eps,
        "seconds": round(time.time() - started, 1),
        "finished": datetime.now().isoformat(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    rate = f"{result['success_rate']*100:.1f}%" if eps else "n/a"
    log(f"iter {n} (epoch {result['epoch']:.2f}): {ok}/{len(eps)} = {rate}  -> {out/'result.json'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--epoch", type=float, help="evaluate the checkpoint nearest this epoch")
    sel.add_argument("--iter", type=int, help="evaluate this exact iteration")
    sel.add_argument("--latest", action="store_true", help="evaluate the newest checkpoint")
    sel.add_argument("--all", action="store_true", help="evaluate every checkpoint lacking a result")
    ap.add_argument("--task", default="So101Bench-Bin-v0")
    ap.add_argument("--episodes-jsonl", dest="episodes_jsonl", default="tasks/focus5.jsonl")
    ap.add_argument("--num-episodes", dest="num_episodes", type=int, default=20)
    ap.add_argument("--port", type=int, default=POLICY_PORT)
    ap.add_argument("--gui", action="store_true", help="show the Isaac Sim viewer")
    ap.add_argument("--fresh-layouts", action="store_true",
                    help="re-sample placements instead of replaying saved layouts")
    ap.add_argument("--force", action="store_true", help="redo merge/export even if present")
    ap.add_argument("--defer-until-training-done", action="store_true",
                    help="wait for the trainer to exit before touching the GPU")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="run even though training is live (slows it, raises temps)")
    ap.add_argument("--max-gpu-temp", type=float, default=None,
                    help="refuse to start if the GPU is hotter than this (C)")
    args = ap.parse_args()

    for p, what in ((KIT_PY, "Isaac Sim Kit python"), (SO101_BENCH, "so101_bench checkout"),
                    (VENV_PY, "cosmos-framework venv")):
        if not p.exists():
            raise SystemExit(f"missing {what}: {p}")
    verify_lora_scale()

    if args.defer_until_training_done:
        while training_running():
            log("training is running — deferring (checking again in 5 min)")
            time.sleep(300)
    elif training_running() and not args.allow_concurrent:
        raise SystemExit(
            "Training is running. Isaac Sim + the policy server will compete with it for the\n"
            "GPU and push thermals (it is already throttling at ~92 C). Re-run with\n"
            "  --defer-until-training-done   to queue behind it, or\n"
            "  --allow-concurrent            to accept the slowdown."
        )

    if args.max_gpu_temp is not None:
        t = gpu_temp()
        if t is not None and t > args.max_gpu_temp:
            raise SystemExit(f"GPU at {t:.0f} C, above --max-gpu-temp {args.max_gpu_temp:.0f} C")

    targets = resolve_target(args)
    if not targets:
        return
    log(f"evaluating {len(targets)} checkpoint(s): {targets}")

    results = []
    for n in targets:
        try:
            results.append(evaluate(n, args))
        except Exception as exc:
            log(f"iter {n} FAILED: {exc}")
            (EVAL_DIR / iter_name(n)).mkdir(parents=True, exist_ok=True)
            (EVAL_DIR / iter_name(n) / "error.txt").write_text(str(exc))

    if results:
        print()
        print(f"{'iter':>8}  {'epoch':>6}  {'episodes':>9}  {'success':>8}")
        for r in results:
            rate = f"{r['success_rate']*100:.1f}%" if r["success_rate"] is not None else "n/a"
            print(f"{r['iter']:>8}  {r['epoch']:>6.2f}  "
                  f"{r['successes']:>3}/{r['episodes']:<5}  {rate:>8}")
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
