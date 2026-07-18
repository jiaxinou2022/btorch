"""SpMV benchmarks — torch-sparse (PyG), via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/torch/pytorch_sparse/ -v -s
"""
from __future__ import annotations

import pytest
import torch

from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmm

pytest.importorskip("torch_sparse", reason="torch_sparse not installed")

from .spmv import make_fn  # noqa: E402  — imports torch_sparse; must follow importorskip


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
def test_spmv(benchmark, spmv_case, torch_device, bench_cfg, batch_size):
    fn = make_fn(spmv_case.matrix, device=torch_device, batch_size=batch_size)
    sync = torch.cuda.synchronize if torch_device.startswith("cuda") else None

    def run():
        result = fn()
        if sync is not None:
            sync()
        return result

    compile_ms = measure_compile_ms(run)

    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_spmv,
        rounds=bench_cfg.rounds_spmv,
        extra_info=base_extra_info(
            framework="torch",
            provider="pytorch_sparse",
            target="spmm",
            variant="default",
            case=spmv_case,
            device=torch_device,
            problem={"N": batch_size},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"compile_ms": compile_ms},
        ),
    )
