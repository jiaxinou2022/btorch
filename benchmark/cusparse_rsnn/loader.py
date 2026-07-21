"""JIT loader for the direct cuSPARSE RSNN benchmark extension."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import which

from torch.utils.cpp_extension import load as load_extension


def _host_compiler() -> str | None:
    return which("g++-12") or which("g++")


@contextmanager
def _temporary_cxx(cxx: str | None):
    if cxx is None:
        yield
        return
    previous = os.environ.get("CXX")
    os.environ["CXX"] = cxx
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CXX", None)
        else:
            os.environ["CXX"] = previous


@contextmanager
def _default_cuda_arch():
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if previous is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = _detect_cuda_arch_list()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous


def _detect_cuda_arch_list() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            capabilities = {
                f"{major}.{minor}"
                for major, minor in (
                    torch.cuda.get_device_capability(index)
                    for index in range(torch.cuda.device_count())
                )
            }
            if capabilities:
                return ";".join(sorted(capabilities))
    except Exception:
        pass
    return "8.9"


@lru_cache(maxsize=1)
def load():
    """Build and load the direct cuSPARSE benchmark extension."""

    if which("ninja") is None:
        raise RuntimeError("Ninja is required to build the cuSPARSE baseline.")
    compiler = _host_compiler()
    if compiler is None:
        raise RuntimeError("A C++ compiler is required for the cuSPARSE baseline.")

    cuda_flags = ["-O3", "--expt-relaxed-constexpr", "--target-directory=.."]
    if compiler.endswith("g++-12"):
        cuda_flags.append(f"-ccbin={compiler}")
    source_root = Path(__file__).resolve().parent
    build_directory = Path("/tmp/btorch_extensions/btorch_cusparse_rsnn")
    build_directory.mkdir(parents=True, exist_ok=True)
    with _temporary_cxx(compiler), _default_cuda_arch():
        return load_extension(
            name="btorch_cusparse_rsnn",
            sources=[
                str(source_root / "cusparse_rsnn.cpp"),
                str(source_root / "cusparse_rsnn_kernels.cu"),
            ],
            build_directory=str(build_directory),
            extra_cflags=["-O3"],
            extra_cuda_cflags=cuda_flags,
            extra_ldflags=["-lcusparse"],
            is_python_module=True,
            verbose=False,
        )
