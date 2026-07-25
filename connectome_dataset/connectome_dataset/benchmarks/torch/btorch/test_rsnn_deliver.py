"""btorch connectome-RSNN spike-delivery benchmark — the recurrent operator (``spikes @ W``,
the beNNch *deliver* phase) with three backends, recorded as the ``variant`` axis:

    torch.btorch.rsnn.native     torch sparse SpMV (btorch's own SparseConn)
    torch.btorch.rsnn.vdha       VDHA SpMSpV      (one sparse spike vector)
    torch.btorch.rsnn.mh_spgemm  MH-SpGEMM SpGEMM (batch of sparse spike vectors)

Spikes are driven at the network's operating firing rate over ``timesteps`` steps; each
backend self-times the delivery and its result is validated against a SciPy reference. VDHA
(SpMSpV) is compared to native at batch 1; MH-SpGEMM (SpGEMM) at batch 32 (its B-tiling needs
≥ BLOCK_SIZE columns), with native at 32 as its baseline in the same cell.

Run:  pytest connectome_dataset/benchmarks/torch/btorch/test_rsnn_deliver.py \
        --device cuda --matrix-set connectome -v -s
"""
from __future__ import annotations

import numpy as np
import pytest

from connectome_dataset.benchmarks.config import RSNN_DELIVERY_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.metrics import flops_spmspv

from . import delivery as dv

# (variant, batch): native is the btorch baseline at each batch it is compared against.
_VARIANTS = [("native", 1), ("vdha", 1), ("native", 32), ("mh_spgemm", 32)]


def _spikes(n: int, batch: int) -> np.ndarray:
    p = D.firing_rate_hz * D.dt_ms * 1e-3
    rng = np.random.default_rng(D.seed)
    shape = (n,) if batch == 1 else (batch, n)
    spk = rng.random(shape) < p
    flat = spk.reshape(batch, n) if batch > 1 else spk.reshape(1, n)
    for b in range(flat.shape[0]):  # no silent sample (empty operand is a kernel edge case)
        if not flat[b].any():
            flat[b, rng.integers(n)] = True
    return spk


@pytest.mark.parametrize("variant_batch", _VARIANTS, ids=lambda vb: f"{vb[0]}-b{vb[1]}")
def test_rsnn_deliver(benchmark, spmv_case, bench_cfg, variant_batch):
    variant, batch = variant_batch
    device = bench_cfg.device or "cpu"
    if not device.startswith("cuda"):
        pytest.skip("RSNN delivery benchmarks require a CUDA device")

    W = dv.weighted_connectivity(spmv_case.matrix)
    n = W.shape[0]
    timesteps = D.timesteps
    spikes = _spikes(n, batch)

    try:
        if variant == "native":
            ms, got = dv.native_deliver(W, spikes, timesteps, device)
        elif variant == "vdha":
            ms, got = dv.vdha_deliver(W, spikes, timesteps)
        else:
            ms, got = dv.mh_deliver(W, spikes, timesteps)
    except (OSError, RuntimeError, ImportError) as e:
        pytest.skip(f"{variant} delivery unavailable: {e}")

    ref = dv.reference(W, spikes)
    ok = np.allclose(got.reshape(ref.shape), ref, rtol=1e-3, atol=1e-3)
    if bench_cfg.strict_correctness:
        assert ok, f"btorch {variant} delivery failed the spikes @ W reference"

    out_deg = np.diff(W.tocsr().indptr)  # synapses fired when a presynaptic neuron spikes
    spk2d = spikes.reshape(1, n) if spikes.ndim == 1 else spikes
    events_per_step = int(sum(out_deg[np.nonzero(spk2d[b])[0]].sum() for b in range(spk2d.shape[0])))
    n_spikes = int(spk2d.sum()) * timesteps
    total_ms = ms * timesteps

    def run():
        return got

    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="torch", provider="btorch", target="rsnn", variant=variant,
            case=spmv_case, device=device,
            problem={"timesteps": timesteps, "batch_size": batch, "pass": "fwd"},
            knobs={"precision": "fp32"},
            total_flops=flops_spmspv(events_per_step),
            timings={"self_timed_ms": total_ms},
            status="ok" if ok else "incorrect",
            extra={
                "n_spikes": n_spikes, "n_syn_events": events_per_step * timesteps,
                "dt_ms": D.dt_ms, "firing_rate_hz": D.firing_rate_hz, "deliver_ms_per_step": ms,
            },
        ),
    )
