# Setting up the Cosmos side (the server)

The environment that trains the policy and serves it: `cosmos-framework`, its venv, the
base checkpoint, and the Hugging Face access it needs. This is the **server** half of the
pipeline — the client half is [setup-isaaclab.md](setup-isaaclab.md).

Nothing in this repo is installed into this environment. The controller calls into it by
absolute path.

- [Requirements](#requirements)
- [Install](#install)
- [The LD_LIBRARY_PATH problem](#the-ld_library_path-problem)
- [Hugging Face access](#hugging-face-access)
- [Downloading checkpoints](#downloading-checkpoints)
- [Convert the base checkpoint](#convert-the-base-checkpoint)
- [Docker](#docker)
- [Unified-memory GPUs](#unified-memory-gpus)
- [Verify](#verify)

---

## Requirements

An NVIDIA GPU with enough memory for the recipe you intend to run. The single-GPU LoRA
recipe was verified on 1× RTX PRO 6000 Blackwell (96 GB); the policy server alone settles
at ~32 GB, and training and serving can share one card if you are careful about stale
processes.

CUDA 13.0 is recommended; 12.8 is supported through a different dependency group.

## Install

```shell
cd /path/to/cosmos-framework

# CUDA 13.0 (recommended); for CUDA 12.8 use --group=cu128-train
uv sync --all-extras --group=cu130-train
source .venv/bin/activate
```

`uv sync` prints a warning about an unknown `[tool.uv.audit]` field in `pyproject.toml`.
It is ignored and does not block the sync.

## The `LD_LIBRARY_PATH` problem

Upstream's setup instructions end with `export LD_LIBRARY_PATH=` — clearing it entirely.
That is not decoration, and skipping it produces one of the most confusing failures in this
whole pipeline:

```
RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling
  `cublasLtMatmulAlgoGetHeuristic(...)`
```

**Cause:** an inherited profile — ROS and gazebo are the usual culprits — puts
`/usr/local/cuda/lib64` on the path ahead of the venv. `libcublasLt` then loads from system
CUDA while `libcublas` loads from the venv. The mismatched pair cannot initialise a handle,
so **every biased `addmm` fails while plain `mm` still works**.

It is *not* memory pressure. It reproduces with 65 GB free, no simulator running, in eager
mode without `torch.compile`. Once it fires the CUDA context is poisoned and every later
call fails the same way — restart the process.

Minimal reproduction on an idle GPU:

```shell
python - <<'EOF'
import torch
a = torch.randn(4096, 192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(4096, 192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
print("mm   ", (a @ w.t()).shape)                            # passes
print("addmm", torch.nn.functional.linear(a, w, b).shape)    # fails when mismatched
EOF
```

Confirm which libraries actually loaded:

```shell
python - <<'EOF'
import torch, os
torch.zeros(1, device="cuda")
print({l.split()[-1] for l in open(f"/proc/{os.getpid()}/maps") if "cublas" in l.lower()})
EOF
```

**Two fixes, both valid.** Clearing the variable (upstream's advice) removes system CUDA
from the search path. This pipeline instead *prepends* the venv's own cu13 libs, which is
equivalent for the purpose and survives a shell that needs the rest of its path for other
tools:

```shell
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

`pipeline.base_env()` does this for every stage it launches, finding the directory by glob
so a Python upgrade cannot stale it. `so101 doctor` checks it resolves.

## Hugging Face access

Set `HF_TOKEN` for an account with access to `nvidia/Cosmos3-Nano`.

One repo is **gated** and will crash the server if you are not approved for it:

```
Error: Access denied. This repository requires approval.  (nvidia/Cosmos-Guardrail1)
```

Guardrails are on by default and loading them downloads that gated checkpoint. Either
request access and wait for approval, or pass `--no-guardrails`, which is what this
pipeline does for local evaluation. Turn them back on before anything resembling a
deployment.

## Downloading checkpoints

**A silent download is not a hang.** Checkpoint resolution shells out to:

```
uvx hf@1.16.4 download --format=json <repo> --repo-type model --revision main --include '*'
```

`--format=json` suppresses the progress bar so the command can emit clean JSON at the end.
For a ~33 GB repo that means **no terminal output at all** for the whole download. Observed
throughput ranged 2.4–10.3 MB/s, so treat any ETA as rough.

To confirm it is progressing, measure the cache growing:

```shell
du -sb ~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano
```

Run it twice a fixed interval apart.

**Cache location gotcha:** `HF_HOME` takes precedence over a separately mounted cache
directory. Setting `HF_HOME=/workspace/.cache/huggingface` while also mounting
`$HOME/.cache/huggingface` elsewhere means the mounted cache is ignored — in one observed
case a 2.82 GB VAE that was already fully cached got re-downloaded because of exactly this.
Either drop the `HF_HOME` override or point it at the mounted path.

## Convert the base checkpoint

```shell
python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Nano --checkpoint-path Cosmos3-Nano
```

Start from the generalist `Cosmos3-Nano`, not `Cosmos3-Nano-Policy-DROID` — the
DROID-specialised checkpoint is not in `convert_model_to_dcp`'s named registry, and the
method is a single fine-tuning stage with no architecture changes, so the generalist base
is the right starting point for a new embodiment. See
[post-training.md](post-training.md) for what happens next.

## Docker

An alternative to the venv:

```shell
image_tag=$(docker build -q .)
docker run -it --runtime=nvidia --ipc=host --rm \
  -e HF_TOKEN="$HF_TOKEN" \
  -v .:/workspace \
  "$image_tag"
```

`--ipc=host` matters: parallel `torchrun` consumes a lot of shared memory. If your security
policy forbids it, raise `--shm-size` instead. If Docker reports `unknown or invalid runtime
name: nvidia`, configure the container toolkit:

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Unified-memory GPUs

On GB10 / DGX Spark-class Grace Blackwell parts there is no discrete VRAM framebuffer, so
the NVML call used to size parallelism always fails:

```
pynvml.NVMLError_NotSupported: Not Supported
```

This is the **one** situation where `--device-memory-bytes` is worth passing — give it total
unified system memory in bytes from `free -b`. On a normal discrete GPU the flag is a no-op:
it reaches only `_build_model_parallelism`, which ignores the value, so this pipeline does
not pass it.

## Verify

```shell
so101 doctor
```

Checks the checkout, the venv python, the cu13 libs, the run directory, the normalizer
stats, exported checkpoints and disk headroom — plus the client-side items in
[setup-isaaclab.md](setup-isaaclab.md).
