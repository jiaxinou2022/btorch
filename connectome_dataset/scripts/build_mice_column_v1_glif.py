"""Stage per-cell-type GLIF parameters for the mice_column_v1 model.

``mice_column_v1`` (from ``~/src/mice_unnamed_torch_dev``) is Guozhang's generated mouse-V1
column with per-cell-type Allen GLIF fits under ``glif_models_VISp/`` — a *different* model from
``mice_v1_guozhang`` (the Billeh V1 core). Each of the column's cell types maps to one or more
Allen GLIF model JSONs; we extract a representative point-GLIF parameter set per cell type
(averaged over the type's example cells) and stage the per-neuron table so the btorch
``MouseV1GLIF`` can build a heterogeneous GLIF3 population on the column.

Allen GLIF JSON is SI; converted to btorch units (mV, pF, ms, pA). Threshold ``th_inf`` is
relative to ``El``; reset is to ``El``; ``tau = R_input * C``; ``t_ref = spike_cut_length * dt``;
``k = 1 / asc_tau`` (1/ms); ``asc_amps`` in pA.

Usage (one-time; needs the mice_unnamed_torch_dev checkout for glif_models_VISp):
    micromamba run -n ml-py312 python scripts/build_mice_column_v1_glif.py \
        --glif-dir /home/fanqixuan/src/mice_unnamed_torch_dev/glif_models_VISp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_GLIF = Path("/home/fanqixuan/src/mice_unnamed_torch_dev/glif_models_VISp")
DEFAULT_ROOT = _REPO / "data" / "external" / "mice_column_v1"


def glif_from_json(d: dict) -> dict:
    """Allen GLIF model JSON (SI) -> btorch GLIF3 params (mV, pF, ms, pA)."""
    el, th = d["El_reference"], d["th_inf"]           # volts (th_inf relative to El)
    asc_tau = np.asarray(d["asc_tau_array"], dtype=np.float64)   # s
    return dict(
        E_L=el * 1e3, V_th=(el + th) * 1e3, V_reset=el * 1e3,    # mV
        C_m=d["C"] * 1e12,                                        # pF
        tau=d["R_input"] * d["C"] * 1e3,                          # ms  (R*C)
        t_ref=d["spike_cut_length"] * d["dt"] * 1e3,             # ms
        k=(1.0 / asc_tau) * 1e-3,                                 # 1/ms
        asc_amps=np.asarray(d["asc_amp_array"], dtype=np.float64) * 1e12,  # pA
    )


def cell_type_params(glif_dir: Path, cell_type: str) -> dict | None:
    """Average GLIF params over every Allen model for a cell type (dir + its subtypes)."""
    dirs = [p for p in glif_dir.iterdir() if p.is_dir() and p.name.split("_")[0] == cell_type.split("_")[0]
            and (p.name == cell_type or p.name.startswith(cell_type + "_"))]
    files = [f for p in dirs for f in p.glob("*.json")]
    if not files:
        return None
    fits = [glif_from_json(json.load(open(f))) for f in files]
    scalar = {k: float(np.mean([fit[k] for fit in fits])) for k in ("E_L", "V_th", "V_reset", "C_m", "tau", "t_ref")}
    scalar["k"] = np.mean([fit["k"] for fit in fits], axis=0).astype(np.float32)
    scalar["asc_amps"] = np.mean([fit["asc_amps"] for fit in fits], axis=0).astype(np.float32)
    scalar["n_models"] = len(files)
    return scalar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glif-dir", type=Path, default=DEFAULT_GLIF, help="glif_models_VISp directory")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="mice_column_v1 dataset dir")
    args = ap.parse_args()

    neurons = pd.read_parquet(args.root / "mice_neurons_processed.parquet", columns=["simple_id", "type", "EI"])
    neurons = neurons.sort_values("simple_id")
    types = list(neurons["type"].unique())

    # Fallback for a type with no Allen dir: the mean over same-E/I types that do resolve.
    resolved = {t: cell_type_params(args.glif_dir, t) for t in types}
    ei_of = dict(zip(neurons["type"], neurons["EI"]))
    for pool_ei in ("E", "I"):
        pool = [p for t, p in resolved.items() if p is not None and ei_of.get(t) == pool_ei]
        if not pool:
            continue
        fallback = {k: float(np.mean([p[k] for p in pool])) for k in ("E_L", "V_th", "V_reset", "C_m", "tau", "t_ref")}
        fallback["k"] = np.mean([p["k"] for p in pool], axis=0).astype(np.float32)
        fallback["asc_amps"] = np.mean([p["asc_amps"] for p in pool], axis=0).astype(np.float32)
        for t in types:
            if resolved[t] is None and ei_of.get(t) == pool_ei:
                print(f"  [fallback] {t}: no Allen GLIF dir -> {pool_ei}-type mean")
                resolved[t] = fallback

    type_list = types
    type_idx = {t: i for i, t in enumerate(type_list)}
    node_type_id = neurons["type"].map(type_idx).to_numpy(np.int64)
    table = {k: np.array([resolved[t][k] for t in type_list], np.float32)
             for k in ("E_L", "V_th", "V_reset", "C_m", "tau", "t_ref")}
    table["k"] = np.stack([resolved[t]["k"] for t in type_list])
    table["asc_amps"] = np.stack([resolved[t]["asc_amps"] for t in type_list])
    table["node_type_id"] = node_type_id
    table["type_names"] = np.array(type_list)

    out = args.root / "mice_column_v1_neurons.npz"
    np.savez_compressed(out, **table)
    print(f"staged {len(type_list)} cell types over {node_type_id.size:,} neurons -> {out}")
    print(f"  tau range {table['tau'].min():.1f}-{table['tau'].max():.1f} ms, "
          f"V_th [{table['V_th'].min():.0f},{table['V_th'].max():.0f}] mV")


if __name__ == "__main__":
    main()
