"""Unit tests for the history layer — ingest normalization, leaderboard cell/ranking."""
from __future__ import annotations

import json

from connectome_dataset.benchmarks.history import ingest, leaderboard, store


def _rec(**kw):
    """A minimal already-normalized record for leaderboard tests."""
    base = {
        "framework": "torch", "provider": "native_sparse", "target": "spmm", "variant": "csr",
        "case_id": "g_spmm", "n": 100, "nnz": 500, "problem": {"N": 8}, "knobs": {},
        "hardware_id": "hw1", "status": "ok", "metrics": {},
    }
    base.update(kw)
    return base


# ── canonicalization ────────────────────────────────────────────────────────
def test_canonical_target_spmv_to_spmm():
    assert ingest._canonical_target("spmv") == "spmm"
    assert ingest._canonical_target("spgemm") == "spgemm"
    assert ingest._canonical_target("rsnn") == "rsnn"


def test_canonical_problem_unifies_producers():
    # Python {batch_size,N}, C++ {batch_size}, gbench {bs,N} all reduce to {N}
    assert ingest._canonical_problem({"batch_size": 8, "N": 8}, "spmm") == {"N": 8}
    assert ingest._canonical_problem({"batch_size": 8}, "spmm") == {"N": 8}
    assert ingest._canonical_problem({"bs": 8, "N": 8}, "spmm") == {"N": 8}
    # rsnn keeps timesteps/batch_size/pass; pass is a cell axis
    assert ingest._canonical_problem(
        {"timesteps": 50, "batch_size": 4, "pass": "fwd+bwd"}, "rsnn"
    ) == {"timesteps": 50, "batch_size": 4, "pass": "fwd+bwd"}


# ── leaderboard cell separation ──────────────────────────────────────────────
def test_cell_size_fallback_separates_unidentified_matrices():
    # No case_id (gbench counters) but different n -> different cells, not collapsed.
    a = _rec(case_id=None, n=256, metrics={"tflops": 1.0})
    b = _rec(case_id=None, n=1024, metrics={"tflops": 2.0}, provider="other")
    lb = leaderboard.build([a, b])
    assert len(lb["cells"]) == 2


def test_rsnn_pass_separates_cells():
    fwd = _rec(target="rsnn", provider="btorch", variant="native", case_id="g_rsnn",
               problem={"timesteps": 50, "batch_size": 4, "pass": "fwd"},
               metrics={"real_time_factor": 2.0})
    bwd = _rec(target="rsnn", provider="btorch", variant="native", case_id="g_rsnn",
               problem={"timesteps": 50, "batch_size": 4, "pass": "fwd+bwd"},
               metrics={"real_time_factor": 6.0})
    lb = leaderboard.build([fwd, bwd])
    assert len(lb["cells"]) == 2  # fwd and fwd+bwd never ranked against each other


# ── leaderboard ranking ──────────────────────────────────────────────────────
def test_ranking_by_tflops_and_speedups():
    fast = _rec(provider="a", variant="fast", metrics={"tflops": 4.0, "compute_ms_median": 1.0})
    slow = _rec(provider="b", variant="slow", metrics={"tflops": 1.0, "compute_ms_median": 4.0})
    cell = leaderboard.build([fast, slow])["cells"][0]
    assert cell["primary_metric"] == "tflops"
    assert cell["ranking"][0]["provider"] == "a"       # higher tflops ranks first
    assert cell["ranking"][0]["speedup_vs_best"] == 1.0
    assert abs(cell["ranking"][1]["speedup_vs_best"] - 0.25) < 1e-9


def test_mixed_metric_falls_back_to_common_metric():
    # One record has tflops, the other only median -> rank BOTH on median (comparable),
    # never mix tflops (higher-better) with median (lower-better).
    with_t = _rec(provider="a", metrics={"tflops": 4.0, "compute_ms_median": 1.0})
    no_t = _rec(provider="b", metrics={"compute_ms_median": 2.0})
    cell = leaderboard.build([with_t, no_t])["cells"][0]
    assert cell["primary_metric"] == "compute_ms_median"
    assert cell["ranking"][0]["provider"] == "a"       # 1.0 ms beats 2.0 ms
    assert abs(cell["ranking"][1]["speedup_vs_best"] - 0.5) < 1e-9


def test_geomean_aggregate_keys_and_baseline():
    base = _rec(provider="cusparse", metrics={"tflops": 2.0})
    ours = _rec(provider="mine", metrics={"tflops": 4.0})
    agg = {a["name"]: a for a in leaderboard.build([base, ours])["aggregate"]}
    mine = agg["torch.mine.spmm.csr"]
    assert mine["geomean_vs_best"] == 1.0            # fastest in cell
    assert abs(mine["geomean_vs_baseline"] - 2.0) < 1e-9  # 4.0 / 2.0 vs cusparse


def test_status_incorrect_excluded():
    ok = _rec(provider="a", metrics={"tflops": 1.0})
    bad = _rec(provider="b", status="incorrect", metrics={"tflops": 99.0})
    cell = leaderboard.build([ok, bad])["cells"][0]
    assert [r["provider"] for r in cell["ranking"]] == ["a"]


# ── store round-trip ─────────────────────────────────────────────────────────
def test_store_append_and_load(tmp_path):
    path = tmp_path / "history.jsonl"
    store.append([_rec(provider="a"), _rec(provider="b")], path)
    store.append([_rec(provider="c")], path)
    loaded = store.load(path)
    assert [r["provider"] for r in loaded] == ["a", "b", "c"]


# ── precision knob / cell alignment ─────────────────────────────────────────
def test_precision_fp32_shares_cell_reduced_splits():
    # fp32 is the implicit default: untagged and precision=fp32 land in one cell;
    # fp16 splits into its own so it never ranks against fp32.
    untagged = leaderboard._cell_key(_rec(knobs={}))
    fp32 = leaderboard._cell_key(_rec(knobs={"precision": "fp32"}))
    fp16 = leaderboard._cell_key(_rec(knobs={"precision": "fp16"}))
    assert untagged == fp32
    assert fp16 != fp32


def test_ingest_cpp_dtype_maps_to_precision_knob():
    assert ingest._precision_knob("float16") == {"precision": "fp16"}
    assert ingest._precision_knob("bfloat16") == {"precision": "bf16"}
    assert ingest._precision_knob("float32") == {}  # fp32 is the default, no knob


# ── C++ JSONL ingest ─────────────────────────────────────────────────────────
def test_ingest_connectome_jsonl_normalizes(tmp_path):
    rec = {"target": "spmv", "alg": "reference.eigen_csr", "variant": "eigen_csr",
           "case_id": "tiny_spmv", "graph_id": "tiny", "n": 4, "nnz": 5, "density": 0.3,
           "batch_size": 8, "median_ms": 0.5, "gflops_s": 2.0, "status": "ok"}
    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    out = ingest.ingest_connectome_jsonl(p, prov={"hardware": {}, "hardware_id": "x",
                                                  "software": {}, "git": {}})[0]
    assert out["framework"] == "cpp" and out["provider"] == "reference"
    assert out["target"] == "spmm"          # spmv normalized to spmm
    assert out["problem"] == {"N": 8}       # batch_size -> N
    assert abs(out["metrics"]["tflops"] - 0.002) < 1e-9
