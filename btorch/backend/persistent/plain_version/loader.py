"""JIT loader for the plain persistent SNN CUDA extension."""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import which

from torch.utils.cpp_extension import load as load_extension


_loaded_config: tuple[
    bool, bool, int, int, int, int, int, int, int, int
] | None = None


def _block_v4_config() -> tuple[int, int, int]:
    block_budget = int(os.environ.get("BTORCH_BLOCK_EDGE_BUDGET", "0"))
    long_segment = int(os.environ.get("BTORCH_LONG_SEGMENT_SIZE", "1024"))
    tile_reduce = int(os.environ.get("BTORCH_TILE_REDUCE_MODE", "0"))
    if block_budget not in (0, 128, 256, 512, 1024):
        raise ValueError("BTORCH_BLOCK_EDGE_BUDGET has an unsupported value.")
    if long_segment not in (128, 256, 512, 1024, 2048):
        raise ValueError("BTORCH_LONG_SEGMENT_SIZE has an unsupported value.")
    if tile_reduce not in (0, 1, 2):
        raise ValueError("BTORCH_TILE_REDUCE_MODE has an unsupported value.")
    return block_budget, long_segment, tile_reduce


def _block_hash_config() -> tuple[int, int, int, int, int]:
    aggregation = int(os.environ.get("BTORCH_BLOCK_HASH_AGGREGATION", "512"))
    capacity = int(os.environ.get("BTORCH_BLOCK_HASH_CAPACITY", "512"))
    max_probe = int(os.environ.get("BTORCH_BLOCK_HASH_MAX_PROBE", "4"))
    min_edges = int(os.environ.get("BTORCH_BLOCK_HASH_MIN_EDGES", "256"))
    used_slots = int(os.environ.get("BTORCH_BLOCK_HASH_USED_SLOTS", "1"))
    if aggregation not in (128, 256, 512):
        raise ValueError("BTORCH_BLOCK_HASH_AGGREGATION is unsupported.")
    if capacity not in (128, 256, 512):
        raise ValueError("BTORCH_BLOCK_HASH_CAPACITY is unsupported.")
    if max_probe not in (4, 8, 16):
        raise ValueError("BTORCH_BLOCK_HASH_MAX_PROBE is unsupported.")
    if min_edges not in (0, 64, 128, 192, 256):
        raise ValueError("BTORCH_BLOCK_HASH_MIN_EDGES is unsupported.")
    if used_slots not in (0, 1):
        raise ValueError("BTORCH_BLOCK_HASH_USED_SLOTS is unsupported.")
    return aggregation, capacity, max_probe, min_edges, used_slots


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
    block_budget, long_segment, tile_reduce = _block_v4_config()
    (
        hash_aggregation,
        hash_capacity,
        hash_max_probe,
        hash_min_edges,
        hash_used_slots,
    ) = _block_hash_config()
    requested_config = (
        enable_block_stats,
        enable_block_hash,
        block_budget,
        long_segment,
        tile_reduce,
        hash_aggregation,
        hash_capacity,
        hash_max_probe,
        hash_min_edges,
        hash_used_slots,
    )
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
    suffixes.extend(
        [
            f"b{block_budget}",
            f"s{long_segment}",
            f"r{tile_reduce}",
        ]
    )
    if enable_block_hash:
        suffixes.extend(
            [
                f"ha{hash_aggregation}",
                f"hc{hash_capacity}",
                f"hp{hash_max_probe}",
                f"hm{hash_min_edges}",
                f"hu{hash_used_slots}",
            ]
        )
    extension_suffix = f"_{'_'.join(suffixes)}"
    cuda_cflags.extend(
        [
            f"-DBTORCH_BLOCK_EDGE_BUDGET={block_budget}",
            f"-DBTORCH_LONG_SEGMENT_SIZE={long_segment}",
            f"-DBTORCH_TILE_REDUCE_MODE={tile_reduce}",
            f"-DBTORCH_BLOCK_HASH_AGGREGATION={hash_aggregation}",
            f"-DBTORCH_BLOCK_HASH_CAPACITY={hash_capacity}",
            f"-DBTORCH_BLOCK_HASH_MAX_PROBE={hash_max_probe}",
            f"-DBTORCH_BLOCK_HASH_MIN_EDGES={hash_min_edges}",
            f"-DBTORCH_BLOCK_HASH_USED_SLOTS={hash_used_slots}",
        ]
    )
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
