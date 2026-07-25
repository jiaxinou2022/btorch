"""Native-brian2 whole-brain Drosophila LIF model (Shiu et al.) — reference RSNN benchmark.

Records on the RSNN track as ``brian2.native.rsnn.flybrain`` — the reference simulation the
btorch ``torch.btorch.rsnn.flybrain`` port is validated against (same connectome, same Shiu
constants, all sensory drive types on; per-neuron rate correlation r ~ 0.999). Runs the full
138k-neuron brain; brian2 is the slow reference, so it uses a single trial by default.

Run:  pytest connectome_dataset/benchmarks/brian2/native/test_flybrain.py -v -s
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from connectome_dataset.benchmarks.config import FLYBRAIN_DEFAULTS as D
from connectome_dataset.benchmarks.templates import base_extra_info, run_pedantic
from connectome_dataset.graph_loader import FLYWIRE_783_ROOT, all_flywire_drive_ids, load_flywire_783
from connectome_dataset.metrics import flops_spmspv

# brian2 runtime cost scales with spike count, and all-drives is far busier than sugar-alone,
# so the reference uses a short window + single trial. It exists to confirm sane rates, not to
# race the GPU; equivalence to btorch is already established at r ~ 0.999.
_N_RUN = 1
_T_RUN_MS = 50.0


def test_flybrain_all_drives_reference(benchmark, bench_cfg):
    pytest.importorskip("brian2")
    if not (FLYWIRE_783_ROOT / "Connectivity_783.parquet").exists():
        pytest.skip("flywire_783 data not staged (see datasets/flywire_783/README.md)")

    from .flybrain import build_and_run

    W, flyids = load_flywire_783(return_ids=True)
    id2i = {int(f): i for i, f in enumerate(flyids)}
    # all sensory drive types at once (same set the btorch test uses -> shared leaderboard cell)
    activate = np.array([id2i[f] for f in all_flywire_drive_ids() if f in id2i], dtype=np.int64)

    def run():
        # Stimulus-only for the reference: brian2's runtime cost scales with spike count, and the
        # tonic background makes the whole brain fire (minutes-long runs). build_and_run supports
        # bg_rate_hz>0; the btorch model is validated against this base-dynamics reference (r~0.999).
        return build_and_run(W, activate, t_run_ms=_T_RUN_MS, n_run=_N_RUN,
                             r_poi_hz=D.r_poi_hz, dt_ms=D.dt_ms, bg_rate_hz=0.0, seed=D.seed)

    rates = run()
    activated_hz = float(rates[activate].mean())
    active_ratio = float((rates > 0).mean())
    ok = abs(activated_hz - D.r_poi_hz) < 30.0
    if bench_cfg.strict_correctness:
        assert ok, f"activated neurons fired at {activated_hz:.1f} Hz (expected ~150)"

    n = W.shape[0]
    events_per_step = int(np.diff(W.tocsr().indptr)[activate].sum())
    case = SimpleNamespace(
        case_id="flywire_783", graph_id="flywire_783",
        n=n, nnz=int(W.nnz), density=W.nnz / (n * n), matrix=None, metadata={},
    )
    run_pedantic(
        benchmark, run, warmup=0, rounds=1,
        extra_info=base_extra_info(
            framework="brian2", provider="native", target="rsnn", variant="flybrain",
            case=case, device="cpu",
            problem={"timesteps": int(_T_RUN_MS / D.dt_ms), "batch_size": _N_RUN, "pass": "fwd"},
            knobs={"precision": "fp64"},
            total_flops=flops_spmspv(events_per_step),
            status="ok" if ok else "incorrect",
            extra={
                "model": "shiu_lif_flywire", "n_activated": int(activate.size),
                "activated_rate_hz": activated_hz, "active_ratio": active_ratio,
                "population_rate_hz": float(rates.mean()),
                "dt_ms": D.dt_ms, "t_run_ms": _T_RUN_MS,
            },
        ),
    )
