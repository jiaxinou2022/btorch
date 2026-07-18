"""Best-effort provenance capture, attached to every record at ingest time.

Everything here is best-effort: missing tools (no GPU, no git) degrade to nulls
rather than failing. ``hardware_id`` is the stable key results are compared within.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from typing import Any


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"commit": commit, "branch": branch, "dirty": bool(status) if status is not None else None}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "gpu": None,
        "arch": None,
        "sm": None,
        "ram_gb": None,
        "interconnect": None,
    }
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if smi:
        info["gpu"] = smi.splitlines()[0].split(",")[0].strip()
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = info["gpu"] or props.name
            info["sm"] = props.major * 10 + props.minor
            info["ram_gb"] = round(props.total_memory / 1e9, 1)
            info["arch"] = _arch_name(props.major)
    except Exception:
        pass
    return info


def _arch_name(major: int) -> str | None:
    return {7: "volta_turing", 8: "ampere_ada", 9: "hopper", 10: "blackwell"}.get(major)


def software_info() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "cuda": None, "driver": None, "frameworks": {}}
    for name, mod in (("torch", "torch"), ("jax", "jax"), ("cupy", "cupy")):
        try:
            info["frameworks"][name] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch  # noqa: PLC0415

        info["cuda"] = torch.version.cuda
    except Exception:
        pass
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv:
        info["driver"] = drv.splitlines()[0].strip()
    return info


def hardware_id(hardware: dict[str, Any]) -> str:
    # GPU identity when present; fall back to CPU/platform so CPU-only hosts don't collide.
    keys = ("gpu", "arch", "sm") if hardware.get("gpu") else ("cpu", "platform")
    key = json.dumps({k: hardware.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


@lru_cache(maxsize=1)
def collect() -> dict[str, Any]:
    hw = hardware_info()
    return {
        "hardware": hw,
        "hardware_id": hardware_id(hw),
        "software": software_info(),
        "git": git_info(),
    }
