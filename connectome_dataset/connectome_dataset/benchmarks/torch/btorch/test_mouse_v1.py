"""btorch Billeh-GLIF mouse-V1 models — RSNN benchmark (per-cell-type GLIF3).

Two *distinct* mouse-V1 models, each a heterogeneous GLIF3 population with its own per-cell-type
parameter fits, recorded on the RSNN track as ``torch.btorch.rsnn.{mouse_v1_guozhang, mouse_column_v1}``:

- ``mouse_v1_guozhang`` — the Billeh V1 core (~111 cell types); size is the radial-core subsample
  ``n_neurons`` (as upstream load_sparse.py).
- ``mouse_column_v1``   — Guozhang's generated VISp column (21 cell types, Allen VISp GLIF fits);
  size is ``--replicate`` block-diagonal tiling, keeping its generated wiring.

Run:  pytest connectome_dataset/benchmarks/torch/btorch/test_mouse_v1.py --device cuda -v -s
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from connectome_dataset.benchmarks.config import MOUSE_V1_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.graph_loader import (
    MICE_COLUMN_V1_ROOT, MICE_V1_GUOZHANG_ROOT, load_mice_column_v1_glif, load_mice_v1_glif,
)
from connectome_dataset.metrics import flops_spmspv


def _load(variant: str):
    if variant == "mouse_v1_guozhang":
        W, neurons = load_mice_v1_glif(n_neurons=D.n_neurons, seed=D.seed,
                                       connected_selection=D.connected_selection)
        return W, neurons, "mice_v1_guozhang", "billeh_glif_core"
    W, neurons = load_mice_column_v1_glif(replicate=D.column_replicate)
    return W, neurons, "mice_column_v1", "guozhang_visp_column"


_PARAMS_PRESENT = {
    "mouse_v1_guozhang": MICE_V1_GUOZHANG_ROOT / "mice_v1_guozhang_neurons.npz",
    "mouse_column_v1": MICE_COLUMN_V1_ROOT / "mice_column_v1_neurons.npz",
}


@pytest.mark.parametrize("variant", ["mouse_v1_guozhang", "mouse_column_v1"])
def test_mouse_v1_ground_state(benchmark, bench_cfg, variant):
    device = bench_cfg.device or "cpu"
    if not device.startswith("cuda"):
        pytest.skip("mouse-V1 GLIF simulation requires a CUDA device")
    if not _PARAMS_PRESENT[variant].exists():
        pytest.skip(f"{variant} GLIF params not staged (see scripts/build_mice_*_glif.py)")

    import torch

    from .mouse_v1 import MouseV1GLIF, MouseV1Params

    W, neurons, graph_id, model_name = _load(variant)
    model = MouseV1GLIF(
        W, neurons,
        params=MouseV1Params(dt=D.dt_ms, bg_rate=D.bg_rate_hz, bg_weight=D.bg_weight_pa, seed=D.seed),
        device=device,
    )

    def run():
        rates = model.simulate(timesteps=D.timesteps, warmup=D.warmup, seed=D.seed)
        torch.cuda.synchronize()
        return rates

    rates = run().cpu().numpy()
    mean_hz = float(rates.mean())
    active_ratio = float((rates > 0).mean())
    ok = 0.0 < mean_hz < 100.0 and active_ratio > 0.0
    if bench_cfg.strict_correctness:
        assert ok, f"{variant} ground state implausible: mean {mean_hz:.1f} Hz, active {active_ratio:.2f}"

    n = W.shape[0]
    case = SimpleNamespace(
        case_id=graph_id, graph_id=graph_id,
        n=n, nnz=int(W.nnz), density=W.nnz / (n * n), matrix=None, metadata={},
    )
    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="torch", provider="btorch", target="rsnn", variant=variant,
            case=case, device=device,
            problem={"timesteps": D.timesteps, "batch_size": 1, "pass": "fwd"},
            knobs={"precision": "fp32", "n_neurons": n},
            total_flops=flops_spmspv(int(W.nnz)) * D.timesteps,
            status="ok" if ok else "incorrect",
            extra={
                "model": model_name, "n_cell_types": int(np.unique(neurons["node_type_id"]).size),
                "mean_rate_hz": mean_hz, "active_ratio": active_ratio,
                "dt_ms": D.dt_ms, "n_neurons": n,
            },
        ),
    )
