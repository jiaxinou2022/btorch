"""RSNN benchmarks — btorch, via pytest-benchmark.

Run:
    pytest connectome_dataset/benchmarks/torch/btorch/ -v -s
    pytest connectome_dataset/benchmarks/torch/btorch/test_rsnn.py -k sparse
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest

from btorch.models import environ, functional

from connectome_dataset.benchmarks.config import RSNN_DEFAULTS
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.metrics import flops_rsnn_step

from .rsnn import DenseRSNN, SparseRSNN


@pytest.fixture(scope="module", autouse=True)
def _btorch_env():
    environ.set(dt=1.0)


def _reset(net: torch.nn.Module, batch_size: int, device: str) -> None:
    functional.reset_net(net, batch_size=batch_size, device=device, dtype=torch.float32)


def _make_run(net, inputs, *, pass_name, batch_size, device, num_output):
    """Build the timed closure for a forward or forward+backward pass."""
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    if pass_name == "fwd":
        net.eval()

        def run():
            _reset(net, batch_size, device)
            with torch.no_grad():
                out = net(inputs)
            sync()
            return out
    else:
        net.train()
        targets = torch.zeros(batch_size, num_output, device=device)
        opt = torch.optim.SGD(net.parameters(), lr=1e-3)

        def run():
            _reset(net, batch_size, device)
            opt.zero_grad(set_to_none=True)
            out = net(inputs)
            F.mse_loss(out, targets).backward()
            sync()
            return out

    return run


def _bench(benchmark, bench_cfg, *, net, case, pass_name, timesteps, batch_size, num_input,
           num_output, total_flops, device, kind):
    inputs = torch.randn(timesteps, batch_size, num_input, device=device)
    functional.init_net_state(net, batch_size=batch_size, device=device, dtype=torch.float32)
    run = _make_run(net, inputs, pass_name=pass_name, batch_size=batch_size, device=device,
                    num_output=num_output)
    run_pedantic(
        benchmark,
        run,
        warmup=bench_cfg.warmup_rsnn,
        rounds=bench_cfg.rounds_rsnn,
        extra_info=base_extra_info(
            # variant = connectivity kernel (consistent axis with the SpMV suite);
            # the fwd vs fwd+bwd pass is a cell-defining workload axis, so it lives
            # in `problem`, not `variant` (fwd+bwd is inherently more work).
            framework="torch", provider="btorch", target="rsnn",
            variant="dense" if kind == "dense" else "native",
            case=case, device=device,
            problem={"timesteps": timesteps, "batch_size": batch_size, "pass": pass_name},
            total_flops=total_flops, extra={"dt_ms": 1.0},
        ),
    )


@pytest.mark.parametrize("n", RSNN_DEFAULTS.dense_sizes)
@pytest.mark.parametrize("pass_name", ["fwd", "fwd+bwd"])
def test_rsnn_dense(benchmark, torch_device, bench_cfg, n, pass_name):
    if bench_cfg.mode not in ("dense", "all"):
        pytest.skip("--mode excludes dense")
    timesteps = bench_cfg.timesteps or RSNN_DEFAULTS.timesteps
    batch_size = bench_cfg.batch_size or RSNN_DEFAULTS.batch_size
    net = DenseRSNN(RSNN_DEFAULTS.num_input, n, RSNN_DEFAULTS.num_output,
                    device=torch_device, dtype=torch.float32)
    # Framework-namespaced id: fully-dense n*n workload, distinct from jax's sparse case.
    case = SimpleNamespace(case_id=f"syn_btorch_dense_{n}", graph_id=f"syn_btorch_{n}",
                           n=n, nnz=n * n, density=1.0, matrix=None, metadata={})
    _bench(benchmark, bench_cfg, net=net, case=case, pass_name=pass_name, timesteps=timesteps,
           batch_size=batch_size, num_input=RSNN_DEFAULTS.num_input,
           num_output=RSNN_DEFAULTS.num_output,
           total_flops=flops_rsnn_step(n * n, n) * timesteps * batch_size,
           device=torch_device, kind="dense")


@pytest.mark.parametrize("pass_name", ["fwd", "fwd+bwd"])
def test_rsnn_sparse(benchmark, rsnn_case, torch_device, bench_cfg, pass_name):
    if bench_cfg.mode not in ("sparse", "all"):
        pytest.skip("--mode excludes sparse")
    net = SparseRSNN(rsnn_case.matrix, rsnn_case.num_input, rsnn_case.num_output,
                     device=torch_device, dtype=torch.float32)
    total_flops = flops_rsnn_step(rsnn_case.nnz, rsnn_case.n) * rsnn_case.timesteps * rsnn_case.batch_size
    _bench(benchmark, bench_cfg, net=net, case=rsnn_case, pass_name=pass_name,
           timesteps=rsnn_case.timesteps, batch_size=rsnn_case.batch_size,
           num_input=rsnn_case.num_input, num_output=rsnn_case.num_output,
           total_flops=total_flops, device=torch_device, kind="sparse")
