"""Shared pytest fixtures and CLI options for all benchmark suites."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from connectome_dataset.benchmarks import corpora, synthetic
from connectome_dataset.benchmarks.cases import load_rsnn_case, load_spmv_case
from connectome_dataset.benchmarks.config import BENNCH_DEFAULTS, RSNN_DEFAULTS, SPMV_DEFAULTS


def _csv(value: str | None) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()] if value else []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--warmup", type=int, default=None, help="Warmup rounds (default: per-target from config)")
    parser.addoption("--rounds", type=int, default=None, help="Benchmark rounds (default: per-target from config)")
    parser.addoption("--graph", default="mice_column_v1", help="Dataset graph ID for sparse cases")
    parser.addoption("--replicate", type=int, default=1)
    parser.addoption("--batch-size", type=int, default=None, dest="batch_size")
    parser.addoption("--timesteps", type=int, default=None)
    parser.addoption("--device", default=None, help="Device override (cuda, cpu, ...)")
    parser.addoption("--mode", default="all", choices=["dense", "sparse", "all"])
    parser.addoption(
        "--synthetic", action="store_true",
        help="Run the synthetic controlled sweep (banded/ER/power-law/RMAT) instead of the catalog graph",
    )
    parser.addoption("--synthetic-size", type=int, default=4096, dest="synthetic_size")
    parser.addoption("--synthetic-degree", type=float, default=16.0, dest="synthetic_degree")
    parser.addoption(
        "--matrix-set", default=None, dest="matrix_set",
        help="Sweep a named real-connectome corpus (e.g. 'connectome') instead of a single --graph",
    )
    parser.addoption(
        "--precisions", default="fp32", dest="precisions",
        help="Comma-separated precision knobs to sweep (e.g. fp32,fp16,bf16). Default: fp32",
    )
    parser.addoption(
        "--rsnn-model", default=None, dest="rsnn_model",
        help="Comma-separated beNNch RSNN models to sweep (coba|cuba|connectome). Default: config",
    )
    parser.addoption(
        "--rsnn-scale", default=None, dest="rsnn_scale",
        help="Comma-separated network scales (strong scaling) for coba/cuba. Default: config",
    )
    parser.addoption(
        "--rsnn-timesteps", type=int, default=None, dest="rsnn_timesteps",
        help="Override the beNNch propagation length (number of dt steps).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    cfg = metafunc.config
    # Expand the case fixture over the selected corpus so every benchmark leaf runs
    # once per case. Precedence: --synthetic > --matrix-set > single --graph.
    if "spmv_case" in metafunc.fixturenames:
        if cfg.getoption("--synthetic"):
            specs = synthetic.default_sweep(cfg.getoption("synthetic_size"), cfg.getoption("synthetic_degree"))
            params, ids = [("synthetic", s) for s in specs], [s.name for s in specs]
        else:
            matrix_set = cfg.getoption("matrix_set")
            graphs = corpora.resolve_matrix_set(matrix_set) if matrix_set else [cfg.getoption("--graph")]
            params, ids = [("catalog", g) for g in graphs], list(graphs)
        metafunc.parametrize("spmv_case", params, ids=ids, indirect=True)
    # The precision knob is a plain value fixture (not indirect) that any leaf can request.
    if "precision" in metafunc.fixturenames:
        precisions = [p.strip() for p in cfg.getoption("precisions").split(",") if p.strip()]
        metafunc.parametrize("precision", precisions)
    # The beNNch RSNN track sweeps (model, scale); connectome ignores scale.
    if "rsnn_bennch" in metafunc.fixturenames:
        models = _csv(cfg.getoption("rsnn_model")) or list(BENNCH_DEFAULTS.models)
        scales = [float(s) for s in _csv(cfg.getoption("rsnn_scale"))] or list(BENNCH_DEFAULTS.scales)
        params, ids = [], []
        for model in models:
            for scale in (scales if model != "connectome" else [1.0]):
                params.append((model, scale))
                ids.append(f"{model}-s{scale:g}" if model != "connectome" else "connectome")
        metafunc.parametrize("rsnn_bennch", params, ids=ids, indirect=True)


@pytest.fixture(scope="session")
def bench_cfg(request: pytest.FixtureRequest) -> SimpleNamespace:
    opt = request.config.getoption
    warmup = opt("--warmup")
    rounds = opt("--rounds")
    return SimpleNamespace(
        warmup_rsnn=warmup if warmup is not None else RSNN_DEFAULTS.warmup,
        warmup_spmv=warmup if warmup is not None else SPMV_DEFAULTS.warmup,
        rounds_rsnn=rounds if rounds is not None else RSNN_DEFAULTS.rep,
        rounds_spmv=rounds if rounds is not None else SPMV_DEFAULTS.rep,
        graph=opt("--graph"),
        replicate=opt("--replicate"),
        batch_size=opt("--batch-size"),
        timesteps=opt("--timesteps"),
        device=opt("--device"),
        mode=opt("--mode"),
    )


@pytest.fixture(scope="module")
def spmv_case(request, bench_cfg):
    # Indirectly parametrized by pytest_generate_tests: ("catalog", graph_id) or ("synthetic", spec).
    kind, ref = getattr(request, "param", ("catalog", bench_cfg.graph))
    if kind == "synthetic":
        return synthetic.spmv_case(ref, batch_sizes=list(SPMV_DEFAULTS.batch_sizes))
    return load_spmv_case(
        ref, replicate=bench_cfg.replicate,
        batch_sizes=list(SPMV_DEFAULTS.batch_sizes),
    )


@pytest.fixture(scope="module")
def rsnn_case(bench_cfg):
    return load_rsnn_case(
        bench_cfg.graph, replicate=bench_cfg.replicate,
        timesteps=bench_cfg.timesteps or RSNN_DEFAULTS.timesteps,
        batch_size=bench_cfg.batch_size or RSNN_DEFAULTS.batch_size,
    )


@pytest.fixture
def rsnn_bennch(request):
    """(model, scale) for one beNNch RSNN workload, from pytest_generate_tests."""
    return getattr(request, "param", (BENNCH_DEFAULTS.models[0], BENNCH_DEFAULTS.scales[0]))


@pytest.fixture
def rsnn_timesteps(request):
    override = request.config.getoption("rsnn_timesteps")
    return override if override is not None else BENNCH_DEFAULTS.timesteps
