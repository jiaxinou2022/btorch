"""In-tree FlashSparse benchmark leaf — the tensor-core weighted SpMM run directly under
the harness (the same kernel is also exposed out-of-tree via the provider registry).

Skips cleanly unless FS_SpMM/FS_Block are built (scripts/build_flashsparse.sh) and a CUDA
device is present. FlashSparse is self-timed: we record its kernel-only ms, not wall-clock.
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.metrics import flops_spmm

from . import spmm as fs

pytest.importorskip("torch")


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
def test_spmm(benchmark, spmv_case, torch_device, bench_cfg, batch_size, precision):
    if not torch_device.startswith("cuda"):
        pytest.skip("FlashSparse requires CUDA")
    if not fs.supports("csr", precision):
        pytest.skip(f"FlashSparse does not support {precision}")
    try:
        fs.load_mods()
    except OSError as e:
        pytest.skip(str(e))

    x_np = np.random.default_rng(0).random((spmv_case.n, batch_size), dtype=np.float32)
    try:
        run = fs.make_fn(spmv_case.matrix, batch_size=batch_size, precision=precision, x_np=x_np)
    except NotImplementedError as e:
        pytest.skip(str(e))  # e.g. N=1 (tensor core needs N%8==0)

    out, kernel_ms = run()
    got = out.detach().to("cpu", dtype=out.dtype).float().numpy()
    status = "ok" if prec.validate_spmm(got, spmv_case.matrix, x_np, precision) else "incorrect"

    run_pedantic(
        benchmark,
        lambda: run()[0],
        warmup=bench_cfg.warmup_spmv,
        rounds=bench_cfg.rounds_spmv,
        extra_info=base_extra_info(
            framework="torch",  # invoked via torch (FS_SpMM is a torch CUDA extension)
            provider="flashsparse",
            target="spmm",
            variant="csr",
            case=spmv_case,
            device="cuda",
            problem={"N": batch_size},
            knobs={"precision": precision},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"self_timed_ms": kernel_ms},  # ingest prefers this over wall-clock
            status=status,
        ),
    )
