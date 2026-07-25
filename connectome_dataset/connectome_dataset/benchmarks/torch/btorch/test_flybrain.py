"""btorch whole-brain *Drosophila* LIF model (Shiu et al. 2024) — RSNN benchmark.

Simulates the entire adult-fly connectome (``flywire_783``, 138,639 neurons) in
btorch and records it on the RSNN track as ``torch.btorch.rsnn.flybrain``. **All**
sensory drive types (sugar / bitter / Ir94e / water gustatory receptor neurons +
Johnston's-Organ auditory/mechanosensory neurons — 230 input neurons) are activated
simultaneously at 150 Hz; the timed work is the full multi-step network simulation
(recurrent sparse delivery + LIF).

The port is cross-validated against the upstream brian2 model (per-neuron rate
correlation r ≈ 0.999 on a small network); see connectome_dataset/benchmarks/torch/
btorch/flybrain.py. brian2 is only a reference — it is not a runtime dependency.

Run:  pytest connectome_dataset/benchmarks/torch/btorch/test_flybrain.py \
        --device cuda -v -s
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from connectome_dataset.benchmarks.config import FLYBRAIN_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.graph_loader import FLYWIRE_783_ROOT, all_flywire_drive_ids, load_flywire_783
from connectome_dataset.metrics import flops_spmspv


def test_flybrain_all_drives(benchmark, bench_cfg):
    device = bench_cfg.device or "cpu"
    if not device.startswith("cuda"):
        pytest.skip("FlyBrain simulation requires a CUDA device")
    if not (FLYWIRE_783_ROOT / "Connectivity_783.parquet").exists():
        pytest.skip("flywire_783 data not staged (see datasets/flywire_783/README.md)")

    import torch

    from .flybrain import FlyBrainModel, FlyBrainParams

    W, flyids = load_flywire_783(return_ids=True)
    known = set(int(x) for x in flyids)
    present = [f for f in all_flywire_drive_ids() if f in known]  # all sensory drives at once
    model = FlyBrainModel(W, flyids, params=FlyBrainParams(dt=D.dt_ms, r_poi=D.r_poi_hz), device=device)

    def run():
        rates = model.simulate(model._to_idx(present), n_run=D.n_run, t_run=D.t_run_ms, seed=D.seed)
        torch.cuda.synchronize()
        return rates

    rates = run().mean(0).cpu().numpy()
    active = rates > 0
    activated_hz = float(rates[model._to_idx(present)].mean())
    # sanity: the driven neurons fire near the Poisson activation rate.
    ok = abs(activated_hz - D.r_poi_hz) < 30.0
    if bench_cfg.strict_correctness:
        assert ok, f"activated neurons fired at {activated_hz:.1f} Hz (expected ~150)"

    n = W.shape[0]
    events_per_step = int(np.diff(W.tocsr().indptr)[model._to_idx(present)].sum())
    case = SimpleNamespace(
        case_id="flywire_783", graph_id="flywire_783",
        n=n, nnz=int(W.nnz), density=W.nnz / (n * n), matrix=None, metadata={},
    )
    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="torch", provider="btorch", target="rsnn", variant="flybrain",
            case=case, device=device,
            problem={"timesteps": int(D.t_run_ms / D.dt_ms), "batch_size": D.n_run, "pass": "fwd"},
            knobs={"precision": "fp32"},
            total_flops=flops_spmspv(events_per_step),
            status="ok" if ok else "incorrect",
            extra={
                "model": "shiu_lif_flywire",
                "n_activated": len(present),
                "activated_rate_hz": activated_hz,
                "active_ratio": float(active.mean()),
                "population_rate_hz": float(rates.mean()),
                "dt_ms": D.dt_ms, "t_run_ms": D.t_run_ms,
            },
        ),
    )
