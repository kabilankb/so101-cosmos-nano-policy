"""``so101`` -- one command for training, serving, testing and inference.

    so101 doctor                 # preflight both environments
    so101 status                 # what is up, what is servable
    so101 checkpoints            # inventory with merge/export state

    so101 train                  # or resume; ~48 h
    so101 merge  --iter 3750     # fold LoRA adapters
    so101 export --iter 3750     # consolidated safetensors

    so101 serve  --iter 3750     # the Cosmos policy server  (server)
    so101 warmup                 # one inference request     (client)
    so101 eval                   # Isaac Lab rollout         (client)

    so101 web                    # control page for server and client

Add ``--dry-run`` to any stage to print the exact command instead of running it.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from . import __version__, doctor, inventory, pipeline, proc
from .config import Settings


def _run_stage(cfg: Settings, args: argparse.Namespace, stage: str,
               iteration: int | None = None) -> int:
    try:
        argv, cwd, env = pipeline.build(cfg, stage, iteration)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"# cwd: {cwd}")
        for key in ("LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF", "SO101_ROOT", "PYTHONPATH"):
            if key in env:
                print(f"{key}={shlex.quote(env[key])} \\")
        print(shlex.join(argv))
        return 0

    if args.detach:
        store = proc.JobStore(cfg)
        try:
            job = store.launch(stage, iteration)
        except proc.JobLaunchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{job['id']}  pid {job['pid']}\n  log: {job['log']}")
        return 0

    print(f"# {stage} in {cwd}", file=sys.stderr)
    return subprocess.run(argv, cwd=str(cwd), env=env, check=False).returncode


def _cmd_status(cfg: Settings, args: argparse.Namespace) -> int:
    store = proc.JobStore(cfg, enabled=False)
    st = proc.state(cfg, store)
    exports = inventory.export_index(cfg)
    latest = cfg.latest_iteration()

    print(f"run          {cfg.run_path}")
    print(f"latest ckpt  iter_{latest:09d}" if latest else "latest ckpt  none")
    print(f"servable     {', '.join(str(i) for i in sorted(exports)) or 'none'}")

    server = st["server"]
    listening = "listening" if server["listening"] else "not listening"
    job = server["job"]
    print(f"server       port {server['port']} {listening}"
          + (f"  (job {job['id']}, pid {job['pid']})" if job else ""))

    client = st["client"]
    print("client       " + (f"{client['job']['stage']} running (pid {client['job']['pid']})"
                             if client["job"] else "idle"))
    if st["training"]:
        print(f"training     running (pid {st['training']['pid']})")

    d = inventory.disk(cfg)
    print(f"disk         {inventory.human_bytes(d['free'])} free, "
          f"{d['percent_used']}% used, {d['checkpoints_remaining']} more checkpoint(s) fit")

    running = [j for j in store.all() if j["status"] == "running"]
    for j in running:
        print(f"job          {j['id']}  pid {j['pid']}")
    return 0


def _cmd_checkpoints(cfg: Settings, args: argparse.Namespace) -> int:
    rows = inventory.checkpoints(cfg)
    if not rows:
        print(f"no checkpoints under {cfg.ckpt_dir}", file=sys.stderr)
        return 1
    history = inventory.eval_history(cfg)
    print(f"{'iter':>6}  {'epochs':>6}  {'merged':>6}  {'export':>7}  {'episodes':>8}  {'ok':>3}")
    for r in rows:
        h = history.get(r["iteration"], {})
        print(
            f"{r['iteration']:>6}  {r['epochs'] or 0:>6.2f}  "
            f"{'yes' if r['merged'] else '-':>6}  "
            f"{'ready' if r['servable'] else '-':>7}  "
            f"{h.get('episodes', 0):>8}  {h.get('successes', 0):>3}"
        )
    return 0


def _cmd_doctor(cfg: Settings, args: argparse.Namespace) -> int:
    checks = doctor.run(cfg, quick=args.quick)
    print(doctor.report(checks))
    return 0 if all(c.ok for c in checks) else 1


def _cmd_jobs(cfg: Settings, args: argparse.Namespace) -> int:
    store = proc.JobStore(cfg, enabled=False)
    for job in store.all()[: args.limit]:
        print(f"{job['status']:>8}  {job['id']:<44}  pid {job['pid']}")
    return 0


def _cmd_logs(cfg: Settings, args: argparse.Namespace) -> int:
    store = proc.JobStore(cfg, enabled=False)
    if not store.get(args.job):
        print(f"no such job: {args.job}", file=sys.stderr)
        return 1
    print("\n".join(store.tail(args.job, args.lines)))
    return 0


def _cmd_stop(cfg: Settings, args: argparse.Namespace) -> int:
    store = proc.JobStore(cfg)
    target = args.job
    if target in pipeline.STAGES:
        job = store.running(target)
        if job is None:
            print(f"no running {target} job", file=sys.stderr)
            return 1
        target = job["id"]
    try:
        job = store.stop(target, hard=args.kill)
    except proc.JobLaunchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"stopped {job['id']}")
    return 0


def _cmd_clean(cfg: Settings, args: argparse.Namespace) -> int:
    import shutil

    targets = inventory.merged_intermediates(cfg)
    if not targets:
        print("nothing to reclaim")
        return 0
    total = sum(t["bytes"] for t in targets)
    for t in targets:
        print(f"{'would remove' if args.dry_run else 'removing'}  "
              f"{t['path']}  ({inventory.human_bytes(t['bytes'])})")
        if not args.dry_run:
            shutil.rmtree(t["path"], ignore_errors=True)
    print(f"{inventory.human_bytes(total)} {'reclaimable' if args.dry_run else 'reclaimed'}")
    return 0


def _cmd_web(cfg: Settings, args: argparse.Namespace) -> int:
    from .web.server import serve_forever

    return serve_forever(cfg, port=args.port, host=args.host, enable_launch=not args.no_launch)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="so101",
        description="SO-101 Cosmos3-Nano post-training, serving and digital-twin evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1] if __doc__ else None,
    )
    p.add_argument("--version", action="version", version=f"so101-cosmos {__version__}")
    p.add_argument("--config", type=Path, metavar="TOML", help="settings file")
    sub = p.add_subparsers(dest="command", required=True)

    def stage_parser(name: str, help_: str, needs_iter: bool = False) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        if needs_iter:
            sp.add_argument("--iter", type=int, required=True, metavar="N",
                            help="training iteration, e.g. 3750")
        sp.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
        sp.add_argument("--detach", action="store_true", help="run in the background as a job")
        return sp

    stage_parser("train", "launch or resume the SFT run")
    stage_parser("merge", "fold LoRA adapters into base weights", needs_iter=True)
    stage_parser("export", "export a merged checkpoint to safetensors", needs_iter=True)
    stage_parser("serve", "start the Cosmos policy server", needs_iter=True)
    stage_parser("warmup", "send one inference request to the server")
    ev = stage_parser("eval", "run the Isaac Lab client against the server")
    ev.add_argument("--gui", action="store_true", help="show the viewer (needs a display)")
    ev.add_argument("--iter", type=int, metavar="N",
                    help="checkpoint the server is serving; recorded so the run counts "
                         "toward that checkpoint in `so101 checkpoints`")

    sp = sub.add_parser("status", help="what is running and what is servable")
    sp.set_defaults(func=_cmd_status)

    sp = sub.add_parser("checkpoints", help="checkpoint inventory with eval history")
    sp.set_defaults(func=_cmd_checkpoints)

    sp = sub.add_parser("doctor", help="preflight both environments")
    sp.add_argument("--quick", action="store_true", help="skip checks that boot a foreign python")
    sp.set_defaults(func=_cmd_doctor)

    sp = sub.add_parser("jobs", help="list launched jobs")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=_cmd_jobs)

    sp = sub.add_parser("logs", help="tail a job's log")
    sp.add_argument("job")
    sp.add_argument("-n", "--lines", type=int, default=40)
    sp.set_defaults(func=_cmd_logs)

    sp = sub.add_parser("stop", help="stop a job by id or stage name")
    sp.add_argument("job")
    sp.add_argument("--kill", action="store_true", help="SIGKILL instead of SIGTERM")
    sp.set_defaults(func=_cmd_stop)

    sp = sub.add_parser("clean", help="delete merged intermediates that are already exported")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=_cmd_clean)

    sp = sub.add_parser("web", help="serve the server/client control page")
    sp.add_argument("--port", type=int, default=None)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--no-launch", action="store_true", help="read-only dashboard")
    sp.set_defaults(func=_cmd_web)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = Settings.load(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "web" and args.port is None:
        args.port = cfg.web_port

    if func := getattr(args, "func", None):
        return func(cfg, args)

    stage = {"eval": "evaluate_gui" if getattr(args, "gui", False) else "evaluate"}.get(
        args.command, args.command
    )
    return _run_stage(cfg, args, stage, getattr(args, "iter", None))


if __name__ == "__main__":
    raise SystemExit(main())
