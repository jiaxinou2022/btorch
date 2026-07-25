"""Stage the macaque multi-area model's mesoscale description (Schmidt et al. 2018).

Runs the INM-6 ``multi-area-model`` once to extract its population-level connectivity
— the *mesoscale* ``(N, K, W)`` that fully specifies the network at any scale — and
saves it as a small ``.npz``. The neuron-level network (millions of neurons, up to
~24 billion synapses at full scale) is **never** built here; it is instantiated on
demand at a configurable scale by ``connectome_dataset.cortical_network``.

Mesoscale arrays (P = 254 populations = 32 areas x 8 layers/types, ``<area>_<pop>``):
  * ``N``      (P,)      full-scale neurons per population
  * ``K_int``  (P, P)    mean in-degree, target population <- source population
  * ``K_ext``  (P,)      mean external (background) in-degree per target population
  * ``W_int``  (P, P)    mean synaptic weight (pA; <0 inhibitory), target <- source
  * ``W_ext``  (P,)      external synaptic weight (pA)
  * ``labels`` (P,)      "<area>_<pop>" identifier per population

Usage (one-time; needs the multi-area-model checkout + its deps):
    micromamba run -n ml-py312 python scripts/build_multiarea_mesoscale.py \
        --src /home/fanqixuan/src/multi-area-model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path("/home/fanqixuan/src/multi-area-model")
DEFAULT_OUT = _REPO / "data" / "external" / "multiarea_mesoscale"


def extract_mesoscale(src: Path):
    """Import the multi-area model and pull out its population-level (N, K, W)."""
    import sys

    src = src.resolve()
    if not (src / "config.py").exists():
        # The model refuses to import without config.py; write a minimal one.
        (src / "config.py").write_text(
            f"base_path = {str(src)!r}\ndata_path = {str(src / '.mam_data')!r}\n"
            "jobscript_template = ''\nsubmit_cmd = None\n"
        )
        (src / ".mam_data").mkdir(exist_ok=True)
    sys.path.insert(0, str(src))
    from multiarea_model import MultiAreaModel

    m = MultiAreaModel({"connection_params": {"g": -11.0}}, theory=False, simulation=False)
    N = np.asarray(m.N_vec, dtype=np.float64)
    K = np.asarray(m.K_matrix, dtype=np.float64)      # (P, P+1): last col = external
    W = np.asarray(m.W_matrix, dtype=np.float64)      # (P, P+1): last col = external
    labels = [f"{area}_{pop}" for area in m.area_list for pop in m.structure[area]]
    neuron = {k: float(m.params["neuron_params"]["single_neuron_dict"][k])
              for k in ("C_m", "tau_m", "tau_syn_ex", "tau_syn_in", "E_L", "V_th", "V_reset", "t_ref")}
    delay = {k: float(v) for k, v in m.params["delay_params"].items()
             if isinstance(v, (int, float))}
    bg_rate = float(m.params["input_params"]["rate_ext"])
    return dict(N=N, K_int=K[:, :-1], K_ext=K[:, -1], W_int=W[:, :-1], W_ext=W[:, -1],
                labels=labels, neuron=neuron, delay=delay, bg_rate=bg_rate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="multi-area-model checkout")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    args = ap.parse_args()

    data = extract_mesoscale(args.src)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "multiarea_mesoscale.npz",
        N=data["N"], K_int=data["K_int"], K_ext=data["K_ext"],
        W_int=data["W_int"], W_ext=data["W_ext"], labels=np.array(data["labels"]),
    )
    (args.out / "params.json").write_text(json.dumps(
        {"neuron": data["neuron"], "delay": data["delay"], "bg_rate": data["bg_rate"],
         "n_populations": len(data["labels"]), "n_full_scale": int(data["N"].sum())},
        indent=2))
    print(f"staged {len(data['labels'])} populations, "
          f"{int(data['N'].sum()):,} full-scale neurons -> {args.out}")


if __name__ == "__main__":
    main()
