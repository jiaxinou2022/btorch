from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def configure_conda_cuda_toolchain() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    cuda_target = Path(conda_prefix) / "targets" / "x86_64-linux"
    cuda_header = cuda_target / "include" / "cuda_runtime.h"
    if cuda_header.is_file() and "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
        os.environ["CUDA_HOME"] = str(cuda_target)

    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    wanted = [str(Path(conda_prefix) / "bin"), str(Path(conda_prefix) / "nvvm" / "bin")]
    for entry in reversed(wanted):
        if Path(entry).is_dir():
            path_parts = [part for part in path_parts if part != entry]
            path_parts.insert(0, entry)
    os.environ["PATH"] = os.pathsep.join(path_parts)


def patch_brainevent_conda_nvcc(brainevent_module: Any) -> None:
    if brainevent_module is None:
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_root = Path(conda_prefix)
    env_nvcc = conda_root / "bin" / "nvcc"
    target_include = conda_root / "targets" / "x86_64-linux" / "include"
    if not env_nvcc.is_file() or not (target_include / "cuda_runtime.h").is_file():
        return

    import brainevent._op.kernix_toolchain as kt

    original_select_nvcc = kt._select_nvcc

    def _select_nvcc_conda_split():
        nvcc, includes, probes = original_select_nvcc()
        if nvcc == str(env_nvcc) and str(target_include) not in includes:
            return nvcc, [str(target_include)], probes
        if nvcc == str(conda_root / "targets" / "x86_64-linux" / "bin" / "nvcc"):
            return str(env_nvcc), [str(target_include)], probes
        return nvcc, includes, probes

    kt._select_nvcc = _select_nvcc_conda_split


def resolve_torch_device(requested: str) -> str:
    import torch

    device = requested
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"
    return device
