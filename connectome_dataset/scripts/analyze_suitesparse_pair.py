#!/usr/bin/env python3
"""Analyse two SuiteSparse matrices (dimacs10_cs4, newman_as_22july06):
degree-segmented distributions, per-timestep RSNN spike recordings, and plots.

Usage:
    python scripts/analyze_suitesparse_pair.py [--no-download] [--timesteps 500] [--no-plot]
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

_REPO = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO / "results" / "suitesparse_analysis"
_SUITESPARSE_DIR = _REPO / "data" / "external" / "suitesparse"

TARGETS = [
    {"group": "DIMACS10", "name": "cs4", "label": "dimacs10_cs4"},
    {"group": "Newman", "name": "as-22july06", "label": "newman_as_22july06"},
]

RSNN_PARAMS = {
    "dt_ms": 0.1,
    "timesteps": 1000,
    "v_rest": -49.0,
    "v_th": -50.0,
    "v_reset": -60.0,
    "tau_m_ms": 20.0,
    "tau_ref_ms": 5.0,
    "tau_syn_ms": 5.0,
    "input_current": 0.15,   # mV-equivalent drive; purposefully low so recurrence dominates
}

_DOWNLOAD_AVAILABLE = True
try:
    from download_suitesparse import scrape_group, download_file  # type: ignore[import-untyped]
except ImportError:
    _DOWNLOAD_AVAILABLE = False


# ── download helpers ──────────────────────────────────────────────────────────────


def _ensure_downloaded(no_download: bool = False) -> dict[str, Path]:
    """Download the two matrices (or reuse cached tarballs) and return {label: path}."""
    if not _DOWNLOAD_AVAILABLE:
        raise RuntimeError(
            "download_suitesparse.py not importable; run from the repo root or install "
            "the package first."
        )
    _SUITESPARSE_DIR.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for t in TARGETS:
        dest = _SUITESPARSE_DIR / t["group"] / f"{t['name']}.tar.gz"
        if dest.exists():
            print(f"[skip] already downloaded: {dest}")
            out[t["label"]] = dest
            continue
        if no_download:
            raise FileNotFoundError(f"Matrix not cached and --no-download set: {dest}")

        print(f"[scrape] {t['group']} ...")
        matrices = scrape_group(t["group"])
        match = [m for m in matrices if m.name == t["name"]]
        if not match:
            raise RuntimeError(f"Matrix {t['group']}/{t['name']} not found on SuiteSparse")
        info = match[0]
        print(f"         → {info.name}  rows={info.rows:,}  cols={info.cols:,}  nnz={info.nnz:,}")
        download_file(info.url, dest)
        out[t["label"]] = dest
    return out


# ── loading ───────────────────────────────────────────────────────────────────────


def _load_csr(path: Path) -> sp.csr_matrix:
    """Load a SuiteSparse .tar.gz or .mtx into float32 CSR."""
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith(".mtx")), None)
            if member is None:
                raise FileNotFoundError(f"No .mtx in {path}")
            extracted = tf.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Cannot extract {member.name} from {path}")
            mat = mmread(extracted)
    else:
        mat = mmread(str(path))
    csr = sp.csr_matrix(mat, dtype=np.float32)
    if not csr.has_sorted_indices:
        csr.sort_indices()
    return csr


# ── degree distribution ───────────────────────────────────────────────────────────


def _gini(arr: np.ndarray) -> float:
    s = np.sort(arr.astype(np.float64))
    if s.size == 0 or np.all(s == 0):
        return 0.0
    idx = np.arange(1, s.size + 1)
    return float((2 * np.sum(idx * s) / (s.size * np.sum(s))) - (s.size + 1) / s.size)


def _auto_bins(values: np.ndarray) -> np.ndarray:
    """Return integer bin edges using heuristic auto-ranged bins."""
    vmax = int(values.max())
    if vmax <= 5:
        return np.arange(0, vmax + 2, dtype=int)
    if vmax <= 50:
        step = 5
        return np.arange(0, vmax + step, step=step, dtype=int)
    if vmax <= 200:
        step = 10
        return np.arange(0, vmax + step, step=step, dtype=int)
    edges = np.logspace(0, np.log10(max(vmax, 10)), num=20)
    edges = np.unique(np.round(edges).astype(int))
    edges = np.insert(edges, 0, 0)
    return edges


def compute_degree_stats(mat: sp.csr_matrix) -> dict:
    """Degree distribution with auto-binned histograms and tail metrics."""
    out_deg = np.diff(mat.indptr).astype(np.int32)
    in_deg = np.diff(mat.tocsc().indptr).astype(np.int32)
    n = mat.shape[0]

    def _build(deg: np.ndarray) -> dict:
        bins = _auto_bins(deg)
        counts, _ = np.histogram(deg, bins=bins)
        pcts = {f"p{p}": float(np.percentile(deg, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9)}

        hist_bins = []
        for i in range(len(counts)):
            lo, hi = bins[i], bins[i + 1]
            c = int(counts[i])
            if c == 0:
                continue
            hist_bins.append({
                "range": f"[{lo},{hi})" if hi > lo + 1 else f"[{lo}]" if lo == hi else f"[{lo},{hi})",
                "n_neurons": c,
                "fraction": round(c / n, 6),
            })

        sorted_deg = np.sort(deg)
        top_1_pct_n = max(1, int(n * 0.01))
        top_hub_nnz = int(sorted_deg[-top_1_pct_n:].sum())
        total_deg = int(sorted_deg.sum())
        n_leaf = int((deg == 0).sum())
        n_hub_10x = int((deg > 10 * deg.mean()).sum())
        n_hub_20x = int((deg > 20 * deg.mean()).sum())

        return {
            "bins": hist_bins,
            "percentiles": pcts,
            "min": int(deg.min()),
            "max": int(deg.max()),
            "mean": round(float(deg.mean()), 2),
            "std": round(float(deg.std()), 2),
            "gini": round(_gini(deg), 4),
            "n_leaf_neurons": n_leaf,
            "leaf_fraction": round(n_leaf / n, 6),
            "n_hubs_gt_10x_mean": n_hub_10x,
            "n_hubs_gt_20x_mean": n_hub_20x,
            "top_1pct_nnz_share": round(top_hub_nnz / total_deg, 4) if total_deg else 0.0,
        }

    return {
        "out_degree": _build(out_deg),
        "in_degree": _build(in_deg),
    }


# ── RSNN simulation (pure numpy LIF) ──────────────────────────────────────────────


class LifSim:
    """Pure-numpy LIF population with CSR connectivity (mirrors ConnectomeNet).

    Model: LIFRef with exponential synapse + CUBA output, clock-driven dt=0.1 ms.

    Synapse (Expon):   tau_s * dg/dt  = -g   →  g(t+dt) = g(t) * exp(-dt/tau_s) + Σ w_i
    CUBA output:        I_syn = g                            (current-based, in mV-equivalent)
    Membrane (LIF):     tau_m * dv/dt = -(v - V_rest) + I_ext + I_syn
                        → v(t+dt) = v(t) + dt/tau_m * (V_rest - v(t) + I_ext + I_syn)
    Refractory:         after spike, hold at V_reset for tau_ref_ms
    """

    def __init__(self, matrix: sp.csr_matrix, *, seed: int = 0):
        self.matrix = matrix.tocsr()
        self.matrix.sort_indices()
        self.n = matrix.shape[0]

        p = RSNN_PARAMS
        self.v_rest = p["v_rest"]
        self.v_th = p["v_th"]
        self.v_reset = p["v_reset"]
        self.tau_m = p["tau_m_ms"]
        self.tau_s = p["tau_syn_ms"]
        self.tau_ref = p["tau_ref_ms"]
        self.dt = p["dt_ms"]
        self.input_current = p["input_current"]
        self.timesteps = p["timesteps"]

        self.decay_v = float(np.exp(-self.dt / self.tau_m))
        self.decay_syn = float(np.exp(-self.dt / self.tau_s))
        self.ref_steps = max(1, int(self.tau_ref / self.dt))

        # Coupling: raw matrix entries are binary (1.0) for both targets.
        # Strong coupling so recurrence dominates over external drive.
        # Coupling=20 with degree~4 and 0.2% spike prob: g_syn_avg≈2.5 → comparable to I_ext=0.15.
        # Hub (degree 200): g_syn_peak≈500 → saturation at refractory-limited max rate.
        raw = self.matrix.data.astype(np.float64)
        self.coupling = raw * 20.0

        self.rng = np.random.default_rng(seed)
        self._precompute_csr_loops()
        self._reset()

    def _precompute_csr_loops(self):
        """Pre-build fast CSR multiply arrays (indices + data per row)."""
        indptr = self.matrix.indptr
        indices = self.matrix.indices
        data = self.coupling
        self._row_lengths = np.diff(indptr)
        self._row_start = indptr[:-1].copy()
        self._row_indices = indices.copy()
        self._row_data = data.copy()

    def _reset(self):
        n = self.n
        cr = self.rng.normal(-55.0, 2.0, n).astype(np.float64)
        cr[cr > self.v_th] = self.v_rest - 1.0
        self.v = cr
        self.g_syn = np.zeros(n, dtype=np.float64)
        self.ref_steps_left = np.zeros(n, dtype=np.int32)

    def step(self, t: int = 0) -> np.ndarray:
        """Advance one dt; return (n,) int spike vector."""
        v = self.v
        v_rest = self.v_rest
        v_th = self.v_th
        v_reset = self.v_reset
        dt = self.dt
        tau_m = self.tau_m
        I_ext = self.input_current

        # 1. Which neurons spike *now* (from previous step's voltage)
        spk = (v >= v_th)

        # 2. Event-driven SpMV: accumulate coupling from spiking pre-neurons
        syn_in = np.zeros(self.n, dtype=np.float64)
        if spk.any():
            spk_idx = np.where(spk)[0]
            for i in spk_idx:
                start = self._row_start[i]
                end = start + self._row_lengths[i]
                syn_in[self._row_indices[start:end]] += self._row_data[start:end]

        # 3. Exponential synapse: g(t+dt) = g(t) * e^{-dt/tau_s} + Σ w
        self.g_syn = self.g_syn * self.decay_syn + syn_in

        # 4. LIF membrane:  v += dt/tau_m * (V_rest - v + I_ext + g_syn)
        dv = (dt / tau_m) * (v_rest - v + I_ext + self.g_syn)
        v_new = v + dv

        # 5. Refractory period: clamp voltage at reset
        ref = self.ref_steps_left
        v_new[ref > 0] = v_reset
        self.ref_steps_left = np.maximum(ref - 1, 0)

        # 6. Threshold crossing → spike + reset + refractory
        new_spk = v_new >= v_th
        v_new[new_spk] = v_reset
        self.ref_steps_left[new_spk] = self.ref_steps
        self.v = v_new

        return new_spk.astype(np.int32)

    def run(self, timesteps: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Run full simulation, return (spike_counts_per_neuron, spike_counts_per_step)."""
        T = timesteps or self.timesteps
        per_neuron = np.zeros(self.n, dtype=np.float64)
        per_step = np.zeros(T, dtype=np.int32)
        g_syn_max = np.zeros(self.n, dtype=np.float64)
        g_syn_sum = np.zeros(self.n, dtype=np.float64)

        for t in range(T):
            spk = self.step(t)
            per_neuron += spk
            per_step[t] = int(spk.sum())
            np.maximum(g_syn_max, self.g_syn, out=g_syn_max)
            g_syn_sum += self.g_syn

        g_syn_avg = g_syn_sum / T
        n_active = (g_syn_max > 1e-6).sum()
        out_deg = np.diff(self.matrix.indptr)

        print(f"    g_syn peak:          mean={g_syn_max.mean():.2f}  max={g_syn_max.max():.1f}  active_neurons={n_active}/{self.n}")
        print(f"    g_syn avg:           mean={g_syn_avg.mean():.4f}  max={g_syn_avg.max():.2f}")
        top5 = np.argsort(out_deg)[-5:]
        print(f"    top-5 hub neurons:   deg={out_deg[top5]}  g_syn_peak={g_syn_max[top5]}  spikes={per_neuron[top5].astype(int)}")
        bottom5 = np.argsort(out_deg)[:5]
        print(f"    bottom-5 leaf neurons: deg={out_deg[bottom5]}  g_syn_peak={g_syn_max[bottom5]}  spikes={per_neuron[bottom5].astype(int)}")

        return per_neuron, per_step


def compute_spike_segmented(per_neuron_spikes: np.ndarray, mat: sp.csr_matrix, n_steps: int) -> dict:
    """Break spike counts down by out-degree bins."""
    out_deg = np.diff(mat.indptr).astype(np.int32)
    bins = _auto_bins(out_deg)
    dt_ms = RSNN_PARAMS["dt_ms"]
    T_s = n_steps * dt_ms * 1e-3

    result = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (out_deg >= lo) & (out_deg < hi)
        n_neurons = mask.sum()
        if n_neurons == 0:
            continue
        total_spikes = int(per_neuron_spikes[mask].sum())
        mean_hz = total_spikes / n_neurons / T_s if T_s > 0 else 0.0
        result.append({
            "degree_range": f"[{lo},{hi})" if hi > lo + 1 else f"[{lo}]",
            "n_neurons": int(n_neurons),
            "total_spikes": total_spikes,
            "mean_firing_rate_hz": round(mean_hz, 2),
            "spikes_per_neuron": round(total_spikes / n_neurons, 2) if n_neurons else 0.0,
        })

    return {
        "bio_time_s": round(T_s, 3),
        "n_timesteps": n_steps,
        "total_spikes": int(per_neuron_spikes.sum()),
        "mean_firing_rate_hz": round(float(per_neuron_spikes.mean()) / T_s, 2) if T_s > 0 else 0.0,
        "per_degree_bin": result,
    }


# ── plotting ──────────────────────────────────────────────────────────────────────


def _plot_degree(
    mat: sp.csr_matrix, label: str, out_dir: Path, degree_stats: dict
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_deg = np.diff(mat.indptr)
    in_deg = np.diff(mat.tocsc().indptr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, deg, title, stats_key in [
        (axes[0], out_deg, "Out-degree", "out_degree"),
        (axes[1], in_deg, "In-degree", "in_degree"),
    ]:
        gini_val = degree_stats[stats_key]["gini"]
        ax.hist(deg, bins=_auto_bins(deg), edgecolor="k", alpha=0.7)
        ax.set_xlabel("Degree")
        ax.set_ylabel("Number of neurons")
        ax.set_title(f"{label} — {title}  (Gini={gini_val})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / f"{label}_deg_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def _plot_spike_raster(
    per_neuron: np.ndarray,
    per_step: np.ndarray,
    out_deg: np.ndarray,
    label: str,
    out_dir: Path,
    max_neurons: int = 200,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (A) Spike count histogram per neuron (sorted by degree)
    ids = np.argsort(out_deg)
    axes[0, 0].bar(range(len(ids)), per_neuron[ids], width=1.0, alpha=0.7)
    axes[0, 0].set_xlabel("Neuron (sorted by out-degree ↑)")
    axes[0, 0].set_ylabel("Lifetime spike count")
    axes[0, 0].set_title(f"{label} — spikes per neuron (sorted by degree)")

    # (B) Per-timestep spike count
    steps = np.arange(len(per_step))
    axes[0, 1].plot(steps, per_step, linewidth=0.5)
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Spike count")
    axes[0, 1].set_title(f"{label} — spikes per timestep (total over {len(per_step)} steps)")

    # (C) Scatter: degree vs lifetime spikes (random sample)
    n_sample = min(len(out_deg), 5000)
    sample_idx = np.random.default_rng(42).choice(len(out_deg), n_sample, replace=False)
    axes[1, 0].scatter(out_deg[sample_idx], per_neuron[sample_idx], s=1, alpha=0.5)
    axes[1, 0].set_xlabel("Out-degree")
    axes[1, 0].set_ylabel("Lifetime spikes")
    axes[1, 0].set_title(f"{label} — degree vs spike count  (n={n_sample} sampled)")

    # (D) Mean firing rate per degree bin (binned)
    bins = _auto_bins(out_deg)
    bin_rates = []
    bin_centers = []
    T_s = len(per_step) * RSNN_PARAMS["dt_ms"] * 1e-3
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (out_deg >= lo) & (out_deg < hi)
        if mask.sum() < 5:
            continue
        rate = per_neuron[mask].mean() / T_s if T_s > 0 else 0
        bin_rates.append(rate)
        bin_centers.append((lo + hi) / 2)
    axes[1, 1].plot(bin_centers, bin_rates, "o-", markersize=4)
    axes[1, 1].set_xlabel("Out-degree bin center")
    axes[1, 1].set_ylabel("Mean firing rate (Hz)")
    axes[1, 1].set_title(f"{label} — firing rate by degree bin")

    fig.tight_layout()
    path = out_dir / f"{label}_spike_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


# ── main ──────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyse dimacs10_cs4 and newman_as_22july06")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    RSNN_PARAMS["timesteps"] = args.timesteps
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download ──
    print("=" * 60)
    print("PHASE 1: Download")
    print("=" * 60)
    downloads = _ensure_downloaded(no_download=args.no_download)
    print()

    # ── 2. Load ──
    print("=" * 60)
    print("PHASE 2: Load & matrix properties")
    print("=" * 60)
    all_results = {}

    for t in TARGETS:
        label = t["label"]
        path = downloads[label]
        print(f"\n--- {label} ---")
        mat = _load_csr(path)
        n, nnz = mat.shape[0], mat.nnz
        density = nnz / (n * n) if n else 0.0

        is_sym = (mat - mat.T).nnz == 0 if n == mat.shape[1] else False
        vals = mat.data
        val_min, val_max = float(vals.min()), float(vals.max())
        val_mean = float(vals.mean())
        val_types = len(np.unique(vals))

        print(f"  shape:     {mat.shape[0]} × {mat.shape[1]}")
        print(f"  nnz:       {nnz:,}")
        print(f"  density:   {density:.6f}  ({density*100:.4f}%)")
        print(f"  symmetric: {is_sym}")
        print(f"  values:    min={val_min:.4f}  max={val_max:.4f}  mean={val_mean:.4f}  unique_vals={val_types}")

        # ── 3. Degree distribution ──
        print(f"\n  Degree distribution:")
        deg_stats = compute_degree_stats(mat)
        od = deg_stats["out_degree"]
        id_ = deg_stats["in_degree"]
        print(f"    out-degree:  mean={od['mean']}  gini={od['gini']}  leaves={od['n_leaf_neurons']}  hubs>10x={od['n_hubs_gt_10x_mean']}")
        print(f"    in-degree:   mean={id_['mean']}  gini={id_['gini']}  leaves={id_['n_leaf_neurons']}  hubs>10x={id_['n_hubs_gt_10x_mean']}")
        print(f"    top-1% share of nnz: out={od['top_1pct_nnz_share']}  in={id_['top_1pct_nnz_share']}")
        print(f"    Out-degree bins (n={n}):")
        for b in od["bins"]:
            print(f"      {b['range']:>12s}: {b['n_neurons']:>6d} neurons ({b['fraction']*100:5.2f}%)")

        # ── 4. RSNN simulation ──
        print(f"\n  RSNN simulation (LIF, dt={RSNN_PARAMS['dt_ms']}ms, {args.timesteps} steps):")
        sim = LifSim(mat)
        t0 = time.perf_counter()
        per_neuron_spikes, per_step_spikes = sim.run(args.timesteps)
        sim_time = time.perf_counter() - t0
        print(f"    wall time: {sim_time:.1f}s")

        seg = compute_spike_segmented(per_neuron_spikes, mat, args.timesteps)
        print(f"    total spikes: {seg['total_spikes']:,}")
        print(f"    mean rate:    {seg['mean_firing_rate_hz']:.1f} Hz")
        print(f"    bio time:     {seg['bio_time_s']*1000:.1f} ms")
        print(f"    Per-degree-bin firing rates:")
        for b in seg["per_degree_bin"]:
            print(f"      degree {b['degree_range']:>12s}: {b['n_neurons']:>6d} neurons  "
                  f"{b['mean_firing_rate_hz']:>8.1f} Hz  ({b['spikes_per_neuron']:>6.1f} spikes/neuron)")

        # Early + steady-state comparison
        quarter = args.timesteps // 4
        early_rate = per_step_spikes[:quarter].mean() / (per_step_spikes.mean() + 1e-9)
        print(f"    early/mean step-spike ratio (first 25%): {early_rate:.2f}")

        # Store results
        all_results[label] = {
            "group": t["group"],
            "name": t["name"],
            "n": n,
            "nnz": nnz,
            "density": round(density, 8),
            "is_symmetric": is_sym,
            "values": {"min": val_min, "max": val_max, "mean": val_mean, "n_unique": val_types},
            "degree_distribution": deg_stats,
            "rsnn_simulation": {
                "params": dict(RSNN_PARAMS),
                "aggregate": seg,
                "per_timestep_spike_counts": per_step_spikes.tolist(),
                "wall_time_s": round(sim_time, 2),
            },
        }

        # ── Plots (if enabled) ──
        if not args.no_plot:
            _plot_degree(mat, label, _OUT_DIR, deg_stats)
            out_deg = np.diff(mat.indptr).astype(np.int32)
            _plot_spike_raster(per_neuron_spikes, per_step_spikes, out_deg, label, _OUT_DIR)

    # ── 5. Write outputs ──
    print("\n" + "=" * 60)
    print("PHASE 3: Output")
    print("=" * 60)

    json_path = _OUT_DIR / "suitesparse_analysis.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=int) + "\n")
    print(f"[json]  {json_path}  ({json_path.stat().st_size:,} bytes)")

    txt_path = _OUT_DIR / "suitesparse_analysis.txt"
    lines = []
    for label, r in all_results.items():
        lines.append(f"=== {label} ({r['group']}/{r['name']}) ===")
        lines.append(f"  n={r['n']:,}  nnz={r['nnz']:,}  density={r['density']:.6f}")
        od = r["degree_distribution"]["out_degree"]
        lines.append(f"  out-degree: mean={od['mean']}  gini={od['gini']}  leaves={od['n_leaf_neurons']}  hubs>10x={od['n_hubs_gt_10x_mean']}  top1pct_share={od['top_1pct_nnz_share']}")
        rs = r["rsnn_simulation"]["aggregate"]
        lines.append(f"  RSNN: {rs['n_timesteps']} steps  {rs['bio_time_s']*1000:.0f}ms bio  {rs['total_spikes']:,} spikes  {rs['mean_firing_rate_hz']:.1f}Hz")
        for b in rs["per_degree_bin"]:
            lines.append(f"    deg{b['degree_range']:>12s}: {b['n_neurons']:>6d} neurons  {b['mean_firing_rate_hz']:>8.1f} Hz")
        lines.append("")
    txt_path.write_text("\n".join(lines))
    print(f"[txt]   {txt_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
