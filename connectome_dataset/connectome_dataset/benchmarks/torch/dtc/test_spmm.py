"""In-tree DTC-SpMM benchmark leaf (tensor-core weighted SpMM). Skips unless the DTCSpMM
extension is built (scripts/build_dtc.sh) and CUDA is present. Self-timed (kernel ms)."""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.metrics import flops_spmm

from . import spmm as dtc

pytest.importorskip("torch")


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
def test_spmm(benchmark, spmv_case, torch_device, bench_cfg, batch_size, precision):
    if not torch_device.startswith("cuda"):
        pytest.skip("DTC-SpMM requires CUDA")
    if not dtc.supports("csr", precision):
        pytest.skip(f"DTC-SpMM does not support {precision}")
    try:
        dtc.load_mod()
    except OSError as e:
        pytest.skip(str(e))

    x_np = np.random.default_rng(0).random((spmv_case.n, batch_size), dtype=np.float32)
    try:
        run = dtc.make_fn(spmv_case.matrix, batch_size=batch_size, precision=precision, x_np=x_np)
    except NotImplementedError as e:
        pytest.skip(str(e))

    out, kernel_ms = run()
    got = out.detach().to("cpu", dtype=out.dtype).float().numpy()
    status = "ok" if prec.validate_spmm(got, spmv_case.matrix, x_np, precision) else "incorrect"

    run_pedantic(
        benchmark,
        lambda: run()[0],
        warmup=0,
        rounds=1,  # kernel averages internally over EXE_TIME iterations
        extra_info=base_extra_info(
            framework="torch",  # DTC-SpMM is a torch CUDA extension
            provider="dtc_spmm",
            target="spmm",
            variant="csr",
            case=spmv_case,
            device="cuda",
            problem={"N": batch_size},
            knobs={"precision": precision},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"self_timed_ms": kernel_ms},
            status=status,
        ),
    )
