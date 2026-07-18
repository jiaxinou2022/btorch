"""SpMV benchmarks — PyTorch native sparse, via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/torch/native_sparse/ -v -s
    pytest connectome_dataset/benchmarks/torch/native_sparse/test_spmv.py -k csr
    pytest ... --precisions fp32,fp16,bf16      # precision knob sweep
"""
from __future__ import annotations

import numpy as np
import torch
import pytest

from connectome_dataset.benchmarks import precision as prec
from connectome_dataset.benchmarks.config import SPMV_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, measure_compile_ms, run_pedantic
from connectome_dataset.metrics import flops_spmm

from .spmv import VARIANTS, make_fn, supports


@pytest.mark.parametrize("batch_size", SPMV_DEFAULTS.batch_sizes)
@pytest.mark.parametrize("variant", VARIANTS)
def test_spmv(benchmark, spmv_case, torch_device, bench_cfg, variant, batch_size, precision):
    if not supports(variant, precision):
        pytest.skip(f"{variant} does not support {precision}")

    x_np = np.random.default_rng(0).random((spmv_case.n, batch_size), dtype=np.float32)
    fn = make_fn(
        spmv_case.matrix, variant=variant, device=torch_device,
        batch_size=batch_size, precision=precision, x_np=x_np,
    )
    sync = torch.cuda.synchronize if torch_device.startswith("cuda") else None

    try:
        compile_ms = measure_compile_ms(fn, sync)
        # correctness contract: validate against the fp32 SciPy oracle before ranking.
        out = fn()
    except NotImplementedError as e:  # e.g. bf16 sparse mm unsupported on the CPU/MKL backend
        pytest.skip(f"{variant}/{precision} unsupported on {torch_device}: {e}")
    if sync is not None:
        sync()
    got = out.detach().to("cpu", torch.float32).numpy()
    status = "ok" if prec.validate_spmm(got, spmv_case.matrix, x_np, precision) else "incorrect"

    def run():
        result = fn()
        if sync is not None:
            sync()
        return result

    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_spmv,
        rounds=bench_cfg.rounds_spmv,
        extra_info=base_extra_info(
            framework="torch",
            provider="native_sparse",
            target="spmm",
            variant=variant,
            case=spmv_case,
            device=torch_device,
            problem={"N": batch_size},
            knobs={"precision": precision},
            total_flops=flops_spmm(spmv_case.nnz, batch_size),
            timings={"compile_ms": compile_ms},
            status=status,
        ),
    )
