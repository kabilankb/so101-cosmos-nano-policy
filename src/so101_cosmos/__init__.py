"""SO-101 Cosmos3-Nano post-training, serving and digital-twin evaluation.

The pipeline spans two environments that cannot import each other:

* the **server** side runs in ``cosmos-framework``'s uv venv (torch, py3.13) and
  holds the policy;
* the **client** side runs under Isaac Sim's Kit python (omni, isaaclab) and
  drives the simulation.

This package imports neither. It builds commands, launches them as detached
processes, parses their logs, and serves a control page. See `pipeline` for the
stage definitions and `config.Settings` for every path it depends on.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings

__all__ = ["Settings", "__version__"]
