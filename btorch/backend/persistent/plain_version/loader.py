"""JIT loader for the plain persistent SNN CUDA extension."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import which

from torch.utils.cpp_extension import load as load_extension


_loaded_config: tuple[bool, bool] | None = None


def _host_compiler() -> str | None:
    return which("g++-12") or which("g++")


@contextmanager
def _temporary_cxx(cxx: str | None):
    if cxx is None:
        yield
        return
    old = os.environ.get("CXX")
    os.environ["CXX"] = cxx
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = old


@contextmanager
def _default_cuda_arch():
    old = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if old is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _detect_cuda_arch_list()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old


def _detect_cuda_arch_list() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            caps = {
                f"{major}.{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(i)
                    for i in range(torch.cuda.device_count())
                )
            }
            if caps:
                return ";".join(sorted(caps))
    except Exception:
        pass
    return "8.9"


@lru_cache(maxsize=4)
def load(
    *, enable_block_stats: bool = False, enable_block_hash: bool = False
):
    """Build and load the plain persistent SNN CUDA extension."""

    global _loaded_config
    requested_config = (enable_block_stats, enable_block_hash)
    if (
        _loaded_config is not None
        and _loaded_config != requested_config
    ):
        raise RuntimeError(
            "The persistent SNN extension cannot switch debug/experimental "
            "compile flags "
            "inside one process; start a fresh process for the other mode."
        )

    if which("ninja") is None:
        raise RuntimeError(
            "Ninja is required to load the persistent SNN CUDA extension."
        )
    cxx = _host_compiler()
    if cxx is None:
        raise RuntimeError(
            "A C++ compiler is required to load the persistent SNN CUDA extension."
        )
    cuda_cflags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "--target-directory=..",
    ]
    cflags = ["-O3"]
    suffixes = []
    if enable_block_stats:
        suffixes.append("stats")
    if enable_block_hash:
        suffixes.append("hash")
    extension_suffix = f"_{'_'.join(suffixes)}" if suffixes else ""
    if enable_block_stats:
        cflags.append("-DENABLE_BLOCK_STATS")
        cuda_cflags.append("-DENABLE_BLOCK_STATS")
    if enable_block_hash:
        cflags.append("-DENABLE_BLOCK_HASH")
        cuda_cflags.append("-DENABLE_BLOCK_HASH")
    if cxx.endswith("g++-12"):
        cuda_cflags.append(f"-ccbin={cxx}")
    here = Path(__file__).resolve().parent
    build_directory = Path(
        f"/tmp/btorch_extensions/btorch_persistent_snn_plain{extension_suffix}"
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    with _temporary_cxx(cxx), _default_cuda_arch():
        extension = load_extension(
            name=f"btorch_persistent_snn_plain{extension_suffix}",
            sources=[
                str(here / "persistent_snn.cpp"),
                str(here / "persistent_snn_plain_kernel.cu"),
                str(here / "persistent_snn_binned_kernel.cu"),
                str(here / "persistent_snn_spike_block_kernel.cu"),
            ],
            build_directory=str(build_directory),
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            is_python_module=False,
            verbose=False,
        )
    _loaded_config = requested_config
    return extension
